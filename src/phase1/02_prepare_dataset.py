#!/usr/bin/env python3
"""
02_prepare_dataset.py — Phase 1, Step 2
=========================================
Reads csv_manifest.json (produced by 01_verify_csvs.py) and builds
clean, balanced train / test datasets without loading everything into RAM.

Pipeline
--------
1. Leakage-resistant splitting:
   - Attack classes with >= 2 source files: stratified file-level split per
     class so entire source captures are held out for test while keeping every
     multi-file attack family in both splits.
   - Attack classes with 1 source file: split at the file level across those
     single-file classes (so unseen attack types also appear in test).
   - Single-file Normal class (BenignTraffic3.pcap.csv): cleaned and
     deduplicated first, then split into contiguous blocks (first 1-test_ratio
     for TRAIN, trailing test_ratio for TEST) to preserve temporal separation
     and ensure Normal samples exist in both splits.
2. Per-chunk cleaning for every source file:
   a. Replace ±inf → NaN
   b. Drop rows with any NaN
   c. O(1)-per-row hash-based duplicate removal across the split
   d. Attach binary_label column (0 = Normal, 1 = Attack)
3. Incremental write to single-stream gzip-compressed CSVs.
4. Class-imbalance handling:
   - If attack:normal ratio in train set exceeds IMBALANCE_THRESHOLD,
     the attack class is randomly undersampled during write (TRAIN only).
   - Balanced class weights + pos_weight are computed from actual TRAIN counts
     and saved for use during model training.

Outputs (data/processed/phase1/)
---------------------------------
  train.csv.gz            — cleaned, labelled training rows
  test.csv.gz             — cleaned, labelled test rows
  split_info.json         — which source files/blocks went to train vs test
  class_weights.json      — {normal, attack, pos_weight} for training
  preparation_report.json — full run statistics

Usage
-----
    python src/phase1/02_prepare_dataset.py [--test-ratio 0.2] [--chunk 100000]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
MANIFEST_PATH   = Path("data/processed/phase1/csv_manifest.json")
OUT_DIR         = Path("data/processed/phase1")
TRAIN_PATH      = OUT_DIR / "train.csv.gz"
TEST_PATH       = OUT_DIR / "test.csv.gz"
SPLIT_INFO_PATH = OUT_DIR / "split_info.json"
WEIGHTS_PATH    = OUT_DIR / "class_weights.json"
REPORT_PATH     = OUT_DIR / "preparation_report.json"

LABEL_COL           = "binary_label"
IMBALANCE_THRESHOLD = 10.0   # undersample attack class in TRAIN if ratio exceeds this
SEED                = 42


# ── Splitting helpers ──────────────────────────────────────────────────────────

def split_source_files(
    files: list[dict],
    test_ratio: float,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Partition source files into train-only, test-only, and contiguous-block-split
    lists while avoiding data leakage.

    - Multi-file classes (>= 2 files): stratified whole-file split per class.
    - Single-file attack classes: whole-file split across the pool of 1-file classes.
    - Single-file benign class: placed in `block_split_files` so the single benign
      capture is split contiguously (head → train, tail → test) after dedup.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        by_class[f["class_name"]].append(f)

    train_files: list[dict] = []
    test_files: list[dict] = []
    block_split_files: list[dict] = []
    single_attack_files: list[dict] = []

    for class_name in sorted(by_class):
        group = list(by_class[class_name])
        rng.shuffle(group)
        label = group[0]["label"]

        if len(group) == 1:
            if label == 0:
                block_split_files.append(group[0])
            else:
                single_attack_files.append(group[0])
        else:
            n_test = max(1, round(len(group) * test_ratio))
            n_test = min(n_test, len(group) - 1)  # keep at least 1 file in train
            test_files.extend(group[:n_test])
            train_files.extend(group[n_test:])

    if single_attack_files:
        rng.shuffle(single_attack_files)
        n_test = max(1, round(len(single_attack_files) * test_ratio)) if len(single_attack_files) > 1 else 0
        test_files.extend(single_attack_files[:n_test])
        train_files.extend(single_attack_files[n_test:])

    return train_files, test_files, block_split_files


# ── Chunk cleaning & deduplication ─────────────────────────────────────────────

def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN in numeric columns and drop rows with any NaN."""
    numeric_cols = chunk.select_dtypes(include=np.number).columns
    if len(numeric_cols):
        chunk = chunk.copy()
        chunk[numeric_cols] = chunk[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
    return chunk.dropna()


def dedup_chunk(chunk: pd.DataFrame, seen_hashes: set[int]) -> pd.DataFrame:
    """
    Drop rows whose 64-bit feature hash already exists in `seen_hashes`.
    Uses O(1) Python set lookups per row (avoids pandas Series.isin O(N*M) rebuilds).
    """
    hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
    mask_new = np.fromiter(
        (h not in seen_hashes for h in hashes),
        dtype=bool,
        count=len(hashes),
    )
    if mask_new.any():
        seen_hashes.update(hashes[mask_new].tolist())
    return chunk.iloc[mask_new]


# ── Streaming writers ──────────────────────────────────────────────────────────

def write_chunk_to_gz(
    chunk: pd.DataFrame,
    gz_handle,
    write_header: bool,
) -> bool:
    """Write a DataFrame chunk to an open gzip text handle; returns next write_header flag."""
    chunk.to_csv(gz_handle, index=False, header=write_header, lineterminator="\n")
    return False


def process_block_split_file(
    fmeta: dict,
    test_ratio: float,
    chunksize: int,
    train_gz,
    test_gz,
    train_write_header: bool,
    test_write_header: bool,
) -> dict:
    """
    Clean and deduplicate a single-file class, then split contiguously into
    TRAIN (head) and TEST (tail) to preserve temporal separation and avoid
    row-level overlap.
    """
    csv_path = Path(fmeta["path"])
    label = fmeta["label"]
    seen_hashes: set[int] = set()
    cleaned_parts: list[pd.DataFrame] = []

    rows_read = 0
    rows_clean = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        rows_read += len(chunk)
        chunk = clean_chunk(chunk)
        if chunk.empty:
            continue
        rows_clean += len(chunk)
        chunk = dedup_chunk(chunk, seen_hashes)
        if not chunk.empty:
            cleaned_parts.append(chunk)

    if cleaned_parts:
        df = pd.concat(cleaned_parts, ignore_index=True)
    else:
        df = pd.DataFrame()

    rows_dedup = len(df)
    df[LABEL_COL] = label

    split_idx = int(rows_dedup * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    if not train_df.empty:
        train_write_header = write_chunk_to_gz(train_df, train_gz, train_write_header)
    if not test_df.empty:
        test_write_header = write_chunk_to_gz(test_df, test_gz, test_write_header)

    return {
        "file": str(csv_path),
        "label": label,
        "rows_read": rows_read,
        "rows_clean": rows_clean,
        "rows_dedup": rows_dedup,
        "train_written": len(train_df),
        "test_written": len(test_df),
        "train_write_header": train_write_header,
        "test_write_header": test_write_header,
    }


def stream_and_write(
    csv_path: Path,
    label: int,
    gz_handle,
    chunksize: int,
    seen_hashes: set[int],
    write_header: bool,
    undersample_keep: float | None,
    rng: random.Random,
) -> tuple[int, int, int, int, bool]:
    """
    Read one CSV in chunks, clean, deduplicate, label, optionally undersample,
    and write to an already-open gzip file handle.
    """
    rows_read = 0
    rows_after_clean = 0
    rows_after_dedup = 0
    rows_written = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        rows_read += len(chunk)

        chunk = clean_chunk(chunk)
        if chunk.empty:
            continue
        rows_after_clean += len(chunk)

        chunk = dedup_chunk(chunk, seen_hashes)
        if chunk.empty:
            continue
        rows_after_dedup += len(chunk)

        chunk = chunk.copy()
        chunk[LABEL_COL] = label

        if undersample_keep is not None and label == 1:
            n_keep = int(round(len(chunk) * undersample_keep))
            if n_keep <= 0:
                continue
            if n_keep < len(chunk):
                chunk = chunk.sample(
                    n=n_keep, random_state=rng.randint(0, 2**31 - 1)
                )

        write_header = write_chunk_to_gz(chunk, gz_handle, write_header)
        rows_written += len(chunk)

    return rows_read, rows_after_clean, rows_after_dedup, rows_written, write_header


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    test_ratio: float = 0.20,
    chunksize: int = 100_000,
    imbalance_threshold: float = IMBALANCE_THRESHOLD,
) -> None:
    if not MANIFEST_PATH.exists():
        print(
            f"[ERROR] Manifest not found: {MANIFEST_PATH}\n"
            "Run src/phase1/01_verify_csvs.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())

    ok_files = [f for f in manifest["files"] if f["status"] == "OK"]
    skipped = len(manifest["files"]) - len(ok_files)

    print(f"Loaded manifest: {len(ok_files)} OK files  ({skipped} skipped)")

    # ── 1. Split source files ──────────────────────────────────────────────────
    train_files, test_files, block_split_files = split_source_files(ok_files, test_ratio)

    print(f"\nSplit strategy  (test_ratio={test_ratio:.0%}, seed={SEED})")
    print(f"  Attack whole-file split : train={len(train_files)} files, test={len(test_files)} files")
    print(f"  Normal block split      : {len(block_split_files)} single capture file(s) split head/tail")

    split_info = {
        "test_ratio": test_ratio,
        "seed": SEED,
        "train_files": [f["path"] for f in train_files],
        "test_files": [f["path"] for f in test_files],
        "block_split_files": [
            {
                "path": f["path"],
                "class_name": f["class_name"],
                "reason": "Only 1 source file exists for this class; contiguous head/tail split used to avoid zero normal samples in train while preserving temporal separation.",
            }
            for f in block_split_files
        ],
    }
    SPLIT_INFO_PATH.write_text(json.dumps(split_info, indent=2))
    print(f"\nSplit info saved → {SPLIT_INFO_PATH}")

    # ── 2. Estimate TRAIN undersampling ratio ──────────────────────────────────
    est_train_normal = round(
        sum(f["row_count"] for f in block_split_files if f["label"] == 0) * (1.0 - test_ratio)
        + sum(f["row_count"] for f in train_files if f["label"] == 0)
    )
    # Estimate deduplicated attack rows (~55% survival after cross-file dedup)
    est_train_attack_raw = sum(f["row_count"] for f in train_files if f["label"] == 1)
    est_train_attack_dedup = int(est_train_attack_raw * 0.55)
    est_ratio = (
        est_train_attack_dedup / est_train_normal
        if est_train_normal > 0 else float("inf")
    )

    undersample_keep: float | None = None
    if est_train_normal > 0 and est_ratio > imbalance_threshold:
        undersample_keep = (est_train_normal * imbalance_threshold) / est_train_attack_dedup
        undersample_keep = min(1.0, max(0.001, undersample_keep))
        print(
            f"\n⚠  Estimated post-dedup attack:normal ratio = {est_ratio:.1f}:1  "
            f"(threshold={imbalance_threshold:.0f}:1)\n"
            f"   Attack class will be undersampled to "
            f"≈{undersample_keep:.1%} of deduplicated rows in TRAIN only."
        )
    else:
        print(
            f"\nEstimated attack:normal ratio = {est_ratio:.2f}:1  "
            f"— within threshold, using class weights only."
        )

    rng = random.Random(SEED)
    t0_global = time.time()

    for path in (TRAIN_PATH, TEST_PATH):
        if path.exists():
            path.unlink()

    all_stats: dict[str, dict] = {
        "train": {
            "output_path": str(TRAIN_PATH),
            "rows_read": 0,
            "rows_after_clean": 0,
            "rows_after_dedup": 0,
            "rows_written": 0,
            "normal_rows": 0,
            "attack_rows": 0,
        },
        "test": {
            "output_path": str(TEST_PATH),
            "rows_read": 0,
            "rows_after_clean": 0,
            "rows_after_dedup": 0,
            "rows_written": 0,
            "normal_rows": 0,
            "attack_rows": 0,
        },
    }

    with (
        gzip.open(TRAIN_PATH, "wt", encoding="utf-8", compresslevel=6) as train_gz,
        gzip.open(TEST_PATH, "wt", encoding="utf-8", compresslevel=6) as test_gz,
    ):
        train_write_header = True
        test_write_header = True

        # ── 3a. Contiguous block split for single-file benign class ────────────
        if block_split_files:
            print(f"\n{'─' * 80}")
            print("Processing block-split file(s) (Normal head → TRAIN, tail → TEST)")
            print(f"{'─' * 80}")
            for fmeta in block_split_files:
                res = process_block_split_file(
                    fmeta=fmeta,
                    test_ratio=test_ratio,
                    chunksize=chunksize,
                    train_gz=train_gz,
                    test_gz=test_gz,
                    train_write_header=train_write_header,
                    test_write_header=test_write_header,
                )
                train_write_header = res["train_write_header"]
                test_write_header = res["test_write_header"]

                # Attribute read/clean/dedup counts proportionally for reporting
                tr_w = res["train_written"]
                te_w = res["test_written"]
                all_stats["train"]["rows_read"] += round(res["rows_read"] * (1.0 - test_ratio))
                all_stats["train"]["rows_after_clean"] += round(res["rows_clean"] * (1.0 - test_ratio))
                all_stats["train"]["rows_after_dedup"] += tr_w
                all_stats["train"]["rows_written"] += tr_w
                all_stats["train"]["normal_rows"] += tr_w

                all_stats["test"]["rows_read"] += res["rows_read"] - round(res["rows_read"] * (1.0 - test_ratio))
                all_stats["test"]["rows_after_clean"] += res["rows_clean"] - round(res["rows_clean"] * (1.0 - test_ratio))
                all_stats["test"]["rows_after_dedup"] += te_w
                all_stats["test"]["rows_written"] += te_w
                all_stats["test"]["normal_rows"] += te_w

                print(
                    f"  {Path(res['file']).name}: "
                    f"read={res['rows_read']:,}, clean={res['rows_clean']:,}, "
                    f"dedup={res['rows_dedup']:,}  →  train={tr_w:,}, test={te_w:,}"
                )

        # ── 3b. Stream whole attack files into TRAIN and TEST ──────────────────
        for split_name, file_list, gz_handle, write_header in [
            ("TRAIN", train_files, train_gz, train_write_header),
            ("TEST",  test_files,  test_gz,  test_write_header),
        ]:
            key = split_name.lower()
            print(f"\n{'─' * 80}")
            print(f"Streaming {split_name} whole files  →  {all_stats[key]['output_path']}")
            print(f"{'─' * 80}")
            print(
                f"  {'#':>4}  {'File':<48}  {'Class':>6}  "
                f"{'Read':>9}  {'Clean':>9}  {'Dedup':>9}  {'Written':>9}"
            )
            print(f"  {'─' * 76}")

            seen_hashes: set[int] = set()
            t0_split = time.time()

            for i, fmeta in enumerate(file_list, 1):
                csv_path = Path(fmeta["path"])
                label = fmeta["label"]
                us_keep = undersample_keep if split_name == "TRAIN" else None

                read, clean, dedup, written, write_header = stream_and_write(
                    csv_path=csv_path,
                    label=label,
                    gz_handle=gz_handle,
                    chunksize=chunksize,
                    seen_hashes=seen_hashes,
                    write_header=write_header,
                    undersample_keep=us_keep,
                    rng=rng,
                )

                all_stats[key]["rows_read"] += read
                all_stats[key]["rows_after_clean"] += clean
                all_stats[key]["rows_after_dedup"] += dedup
                all_stats[key]["rows_written"] += written
                if label == 0:
                    all_stats[key]["normal_rows"] += written
                else:
                    all_stats[key]["attack_rows"] += written

                print(
                    f"  {i:>4}  {csv_path.name:<48}  "
                    f"{'Normal' if label == 0 else 'Attack':>6}  "
                    f"{read:>9,}  {clean:>9,}  {dedup:>9,}  {written:>9,}"
                )

            all_stats[key]["elapsed_seconds"] = round(time.time() - t0_split, 1)

    # Finalize file sizes & ratios
    for key, out_path in [("train", TRAIN_PATH), ("test", TEST_PATH)]:
        s = all_stats[key]
        s["output_size_mb"] = round(out_path.stat().st_size / 1_048_576, 2) if out_path.exists() else 0.0
        n0, n1 = s["normal_rows"], s["attack_rows"]
        ratio = n1 / n0 if n0 > 0 else float("inf")
        s["attack_normal_ratio"] = round(ratio, 2)

        print(f"\n  {key.upper()} SUMMARY:")
        print(f"    Rows read:    {s['rows_read']:>12,}")
        print(f"    After clean:  {s['rows_after_clean']:>12,}  (dropped {s['rows_read'] - s['rows_after_clean']:,} NaN/Inf)")
        print(f"    After dedup:  {s['rows_after_dedup']:>12,}  (dropped {s['rows_after_clean'] - s['rows_after_dedup']:,} duplicates)")
        print(f"    Written:      {s['rows_written']:>12,}")
        print(f"    Normal (0):   {n0:>12,}")
        print(f"    Attack (1):   {n1:>12,}")
        print(f"    Ratio (A:N):  {ratio:>11.2f}:1")
        print(f"    Size:         {s['output_size_mb']} MB")

    # ── 4. Class weights ───────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("Computing balanced class weights from TRAIN label counts…")

    n_normal = all_stats["train"]["normal_rows"]
    n_attack = all_stats["train"]["attack_rows"]
    n_total = n_normal + n_attack

    w_normal = n_total / (2.0 * n_normal) if n_normal > 0 else 1.0
    w_attack = n_total / (2.0 * n_attack) if n_attack > 0 else 1.0
    pos_weight = n_normal / n_attack if n_attack > 0 else 1.0

    class_weights = {
        "normal": round(w_normal, 6),
        "attack": round(w_attack, 6),
        "pos_weight": round(pos_weight, 6),
        "note": (
            "Use 'pos_weight' in nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])). "
            "Or pass 'normal' and 'attack' as sample weights to sklearn estimators."
        ),
    }

    WEIGHTS_PATH.write_text(json.dumps(class_weights, indent=2))
    print(f"  weight(Normal=0): {class_weights['normal']:.4f}")
    print(f"  weight(Attack=1): {class_weights['attack']:.4f}")
    print(f"  pos_weight (for BCEWithLogitsLoss): {class_weights['pos_weight']:.4f}")
    print(f"  Saved → {WEIGHTS_PATH}")

    # ── 5. Full report ─────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - t0_global, 1)
    report = {
        "config": {
            "test_ratio": test_ratio,
            "chunksize": chunksize,
            "seed": SEED,
            "imbalance_threshold": imbalance_threshold,
            "undersample_keep": round(undersample_keep, 6) if undersample_keep is not None else None,
        },
        "splits": all_stats,
        "class_weights": class_weights,
        "total_elapsed_seconds": total_elapsed,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n{'═' * 80}")
    print(f"  Phase 1 preparation complete in {total_elapsed:.1f}s")
    print(f"  Train  → {TRAIN_PATH}  ({all_stats['train']['rows_written']:,} rows, {all_stats['train']['output_size_mb']} MB)")
    print(f"  Test   → {TEST_PATH}   ({all_stats['test']['rows_written']:,} rows, {all_stats['test']['output_size_mb']} MB)")
    print(f"  Report → {REPORT_PATH}")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 — real dataset preparation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.20,
        help="Fraction of source files (per class) / block to hold out for the test set.",
    )
    parser.add_argument(
        "--chunk", type=int, default=100_000,
        help="Number of rows per pandas read_csv chunk (tunes memory vs speed).",
    )
    parser.add_argument(
        "--max-attack-ratio", type=float, default=IMBALANCE_THRESHOLD,
        help="Maximum attack:normal ratio in TRAIN before undersampling attack rows.",
    )
    args = parser.parse_args()
    main(
        test_ratio=args.test_ratio,
        chunksize=args.chunk,
        imbalance_threshold=args.max_attack_ratio,
    )
