#!/usr/bin/env python3
"""
evaluate.py — Phase 2
=====================
Loads the saved Phase 2 PyTorch checkpoint and `preprocessor.json`, streams
through the held-out `data/processed/phase1/test.csv.gz` in chunks, and computes:
  - Precision, Recall, F1-score (Attack class, Normal class, Macro, Weighted)
  - ROC-AUC and PR-AUC (Average Precision)
  - Confusion matrix (TN, FP, FN, TP)

Also supports single-flow / sample inference sanity check.

Usage
-----
    python src/phase2/evaluate.py
    python src/phase2/evaluate.py --threshold 0.5 --chunk 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CICIoTNet, FlowPreprocessor, select_device  # noqa: E402

TEST_CSV    = Path("data/processed/phase1/test.csv.gz")
CKPT_PATH   = Path("models/phase2/ciciot_ids_model.pt")
PREP_PATH   = Path("models/phase2/preprocessor.json")
RESULTS_DIR = Path("results/phase2")
REPORT_PATH = RESULTS_DIR / "evaluation_report.json"
LABEL_COL   = "binary_label"


def load_artifacts(
    ckpt_path: Path = CKPT_PATH,
    prep_path: Path = PREP_PATH,
    device: torch.device | None = None,
) -> tuple[CICIoTNet, FlowPreprocessor, dict, torch.device]:
    """Load saved PyTorch model weights and preprocessing parameters."""
    if not ckpt_path.exists():
        print(f"[ERROR] Model checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not prep_path.exists():
        print(f"[ERROR] Preprocessor file not found: {prep_path}", file=sys.stderr)
        sys.exit(1)

    if device is None:
        device = select_device("auto")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = CICIoTNet(
        in_features=cfg["in_features"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preprocessor = FlowPreprocessor.load(prep_path)
    return model, preprocessor, ckpt, device


def main(
    chunksize: int = 200_000,
    batch_size: int = 16_384,
    threshold: float | None = None,
    device_name: str = "auto",
) -> None:
    if not TEST_CSV.exists():
        print(f"[ERROR] Test dataset not found: {TEST_CSV}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    model, preprocessor, ckpt, device = load_artifacts(device=device)
    eff_threshold = preprocessor.threshold if threshold is None else threshold

    print(f"Loaded checkpoint : {CKPT_PATH} (best epoch {ckpt['training_config']['best_epoch']})")
    print(f"Loaded preprocessor: {PREP_PATH} ({len(preprocessor.feature_columns)} features)")
    print(f"Evaluating on     : {TEST_CSV}  (device={device}, threshold={eff_threshold})")

    t0 = time.time()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for chunk_idx, chunk in enumerate(pd.read_csv(TEST_CSV, chunksize=chunksize), 1):
            y_chunk = chunk[LABEL_COL].to_numpy(dtype=np.int64)
            X_chunk = preprocessor.transform(chunk)

            probs_chunk = np.empty(len(X_chunk), dtype=np.float32)
            for start in range(0, len(X_chunk), batch_size):
                end = min(start + batch_size, len(X_chunk))
                xb = torch.from_numpy(X_chunk[start:end]).to(device)
                probs_chunk[start:end] = torch.sigmoid(model(xb)).squeeze(1).cpu().numpy()

            all_targets.append(y_chunk)
            all_probs.append(probs_chunk)
            if chunk_idx % 5 == 0:
                print(f"  Processed {chunk_idx * chunksize:,} test rows…")

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= eff_threshold).astype(np.int64)
    elapsed = time.time() - t0

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc       = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    f1_macro  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weight = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    roc_auc   = float(roc_auc_score(y_true, y_prob))
    pr_auc    = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    normal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    cls_rep = classification_report(
        y_true,
        y_pred,
        target_names=["Normal (0)", "Attack (1)"],
        digits=4,
        zero_division=0,
    )

    print(f"\n{'═' * 72}")
    print(f"  TEST SET EVALUATION  ({len(y_true):,} flows in {elapsed:.1f}s)")
    print(f"{'═' * 72}")
    print(f"  Accuracy          : {acc:.4%}")
    print(f"  Precision (Attack): {precision:.4f}")
    print(f"  Recall    (Attack): {recall:.4f}")
    print(f"  F1-score  (Attack): {f1:.4f}")
    print(f"  Recall    (Normal): {normal_recall:.4f}")
    print(f"  Macro F1          : {f1_macro:.4f}")
    print(f"  Weighted F1       : {f1_weight:.4f}")
    print(f"  ROC-AUC           : {roc_auc:.5f}")
    print(f"  PR-AUC            : {pr_auc:.5f}")
    print(f"\n  Confusion Matrix:")
    print(f"                   Pred Normal    Pred Attack")
    print(f"    Actual Normal  {tn:>11,}    {fp:>11,}")
    print(f"    Actual Attack  {fn:>11,}    {tp:>11,}")
    print(f"\n{cls_rep}")

    report = {
        "test_rows": int(len(y_true)),
        "normal_rows": int((y_true == 0).sum()),
        "attack_rows": int((y_true == 1).sum()),
        "threshold": eff_threshold,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": {
            "accuracy": round(acc, 6),
            "precision_attack": round(precision, 6),
            "recall_attack": round(recall, 6),
            "f1_attack": round(f1, 6),
            "recall_normal": round(normal_recall, 6),
            "f1_macro": round(f1_macro, 6),
            "f1_weighted": round(f1_weight, 6),
            "roc_auc": round(roc_auc, 6),
            "pr_auc": round(pr_auc, 6),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "checkpoint": str(CKPT_PATH),
        "preprocessor": str(PREP_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 — Evaluate saved CICIoTNet on held-out test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunk", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    main(
        chunksize=args.chunk,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device_name=args.device,
    )
