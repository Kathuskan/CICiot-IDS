#!/usr/bin/env python3
"""
01_verify_csvs.py — Phase 1, Step 1
=====================================
Scans every CSV under data/raw/, validates structure and data quality,
and writes a machine-readable manifest to data/processed/phase1/csv_manifest.json.

For each file it records:
  - Relative path, class name, binary label
  - File size (MB), row count, column count
  - Column names (validated against a reference set)
  - Missing-value count, infinite-value count
  - Status: OK | COLUMN_MISMATCH | EMPTY | ERROR

Run from the project root:
    python src/phase1/01_verify_csvs.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
OUT_DIR       = Path("data/processed/phase1")
MANIFEST_PATH = OUT_DIR / "csv_manifest.json"
CHUNKSIZE     = 100_000

# Folder names that map to the NORMAL (benign) class — everything else is ATTACK
BENIGN_CLASSES: frozenset[str] = frozenset({"Benign_Final"})


def binary_label(class_name: str) -> int:
    """Return 0 for benign traffic, 1 for any attack class."""
    return 0 if class_name in BENIGN_CLASSES else 1


# ── Per-file verification ──────────────────────────────────────────────────────

def verify_file(
    csv_path: Path,
    reference_columns: list[str] | None,
    chunksize: int = CHUNKSIZE,
) -> dict:
    """
    Stream through one CSV file and collect quality statistics.

    Parameters
    ----------
    csv_path          : Path to the CSV file.
    reference_columns : Column list from the first valid file; used to detect
                        schema drift in subsequent files.
    chunksize         : Rows per pandas chunk (keeps memory flat).

    Returns
    -------
    A dict with all metadata and quality stats for this file.
    """
    class_name = csv_path.parent.name
    record: dict = {
        "path"        : str(csv_path.relative_to(Path("."))),
        "class_name"  : class_name,
        "label"       : binary_label(class_name),
        "size_mb"     : round(csv_path.stat().st_size / 1_048_576, 3),
        "row_count"   : 0,
        "col_count"   : 0,
        "columns"     : [],
        "missing"     : 0,
        "infinite"    : 0,
        "status"      : "OK",
        "status_detail": "",
    }

    try:
        first_chunk = True
        missing_total = 0
        inf_total     = 0
        rows_total    = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            if first_chunk:
                cols = chunk.columns.tolist()
                record["col_count"] = len(cols)
                record["columns"]   = cols
                first_chunk = False

                # Schema validation against the reference
                if reference_columns is not None and cols != reference_columns:
                    record["status"] = "COLUMN_MISMATCH"
                    extra   = set(cols) - set(reference_columns)
                    missing = set(reference_columns) - set(cols)
                    record["status_detail"] = (
                        f"extra={sorted(extra)}, missing={sorted(missing)}"
                    )
                    return record

            rows_total    += len(chunk)
            missing_total += int(chunk.isna().sum().sum())

            numeric = chunk.select_dtypes(include=np.number)
            if not numeric.empty:
                inf_total += int(np.isinf(numeric.to_numpy()).sum())

        record["row_count"] = rows_total
        record["missing"]   = missing_total
        record["infinite"]  = inf_total

        if rows_total == 0:
            record["status"] = "EMPTY"
            record["status_detail"] = "File exists but contains no data rows."

    except Exception as exc:
        record["status"] = "ERROR"
        record["status_detail"] = str(exc)

    return record


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found under {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files under {RAW_DIR}/")
    print(f"{'─' * 90}")
    print(f"  {'#':>4}  {'File':<55}  {'Rows':>10}  {'MB':>6}  {'NaN':>8}  {'Inf':>8}  Status")
    print(f"{'─' * 90}")

    reference_cols: list[str] | None = None
    manifest: list[dict] = []
    t0 = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        rel = str(csv_path.relative_to(RAW_DIR))
        record = verify_file(csv_path, reference_cols)

        # Establish column reference from the first good file
        if reference_cols is None and record["status"] == "OK":
            reference_cols = record["columns"]

        manifest.append(record)

        # Pretty-print progress line
        status_sym = "✓" if record["status"] == "OK" else "✗"
        print(
            f"  {idx:>4}  {rel:<55}  "
            f"{record['row_count']:>10,}  "
            f"{record['size_mb']:>5.1f}  "
            f"{record['missing']:>8,}  "
            f"{record['infinite']:>8,}  "
            f"{status_sym} {record['status']}"
        )

    elapsed = time.time() - t0

    # ── Aggregate summary ──────────────────────────────────────────────────────
    ok_files  = [r for r in manifest if r["status"] == "OK"]
    bad_files = [r for r in manifest if r["status"] != "OK"]

    normal_rows = sum(r["row_count"] for r in manifest if r["label"] == 0)
    attack_rows = sum(r["row_count"] for r in manifest if r["label"] == 1)
    total_rows  = normal_rows + attack_rows

    total_missing = sum(r["missing"]  for r in manifest)
    total_inf     = sum(r["infinite"] for r in manifest)

    imbalance = attack_rows / normal_rows if normal_rows > 0 else float("inf")

    # Class breakdown table
    class_stats: dict[str, dict] = {}
    for r in manifest:
        cn = r["class_name"]
        if cn not in class_stats:
            class_stats[cn] = {
                "label": r["label"], "files": 0, "rows": 0
            }
        class_stats[cn]["files"] += 1
        class_stats[cn]["rows"]  += r["row_count"]

    print(f"\n{'═' * 90}")
    print(f"  VERIFICATION SUMMARY  ({elapsed:.1f}s)")
    print(f"{'═' * 90}")
    print(f"  Total files   : {len(manifest):,}  (OK: {len(ok_files)}, Issues: {len(bad_files)})")
    print(f"  Total rows    : {total_rows:,}")
    print(f"    Normal (0)  : {normal_rows:,}")
    print(f"    Attack (1)  : {attack_rows:,}")
    print(f"  Imbalance     : {imbalance:.1f}:1  (attack:normal)")
    print(f"  Missing vals  : {total_missing:,}")
    print(f"  Infinite vals : {total_inf:,}")

    if bad_files:
        print(f"\n  {'─' * 70}")
        print(f"  FILES WITH ISSUES ({len(bad_files)}):")
        for r in bad_files:
            print(f"    {r['path']}  →  {r['status']}  {r['status_detail']}")

    print(f"\n  {'─' * 70}")
    print(f"  {'Class':<35} {'Label':>5} {'Files':>5} {'Rows':>12}")
    print(f"  {'─' * 70}")
    for cn, s in sorted(class_stats.items(), key=lambda x: (-x[1]["label"], -x[1]["rows"])):
        print(f"  {cn:<35} {s['label']:>5} {s['files']:>5} {s['rows']:>12,}")
    print(f"{'═' * 90}\n")

    # ── Write manifest ─────────────────────────────────────────────────────────
    manifest_data = {
        "reference_columns": reference_cols,
        "summary": {
            "total_files"    : len(manifest),
            "ok_files"       : len(ok_files),
            "bad_files"      : len(bad_files),
            "total_rows"     : total_rows,
            "normal_rows"    : normal_rows,
            "attack_rows"    : attack_rows,
            "imbalance_ratio": round(imbalance, 2),
            "total_missing"  : total_missing,
            "total_infinite" : total_inf,
        },
        "class_stats": class_stats,
        "files": manifest,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest_data, indent=2))
    print(f"Manifest written → {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
