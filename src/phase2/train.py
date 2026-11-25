#!/usr/bin/env python3
"""
train.py — Phase 2
==================
Trains `CICIoTNet` on Phase 1's processed `train.csv.gz` using a stratified
train/validation split (never touching `test.csv.gz`).

Key features
------------
- Fits `FlowPreprocessor` (log1p + z-score standardization) strictly on the
  training split to prevent validation/test leakage.
- Uses `pos_weight` from `class_weights.json` in `BCEWithLogitsLoss`.
- Evaluates Precision, Recall, F1-score, and ROC-AUC on the validation split
  every epoch, using cosine LR scheduling and early stopping on Val ROC-AUC.
- Saves the best model checkpoint (`models/phase2/ciciot_ids_model.pt`),
  preprocessing parameters (`models/phase2/preprocessor.json`), and
  training history (`models/phase2/training_history.json`).

Usage
-----
    python src/phase2/train.py
    python src/phase2/train.py --epochs 20 --batch-size 4096 --lr 1e-3
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
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running directly via `python src/phase2/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CICIoTNet, FlowPreprocessor, select_device  # noqa: E402

# ── Default Paths & Constants ─────────────────────────────────────────────────
TRAIN_CSV     = Path("data/processed/phase1/train.csv.gz")
WEIGHTS_JSON  = Path("data/processed/phase1/class_weights.json")
MODEL_DIR     = Path("models/phase2")
CKPT_PATH     = MODEL_DIR / "ciciot_ids_model.pt"
PREP_PATH     = MODEL_DIR / "preprocessor.json"
HISTORY_PATH  = MODEL_DIR / "training_history.json"
LABEL_COL     = "binary_label"
SEED          = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics for Attack=1 and macro averages."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and return (avg_loss, y_true, y_prob)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            targets_list.append(yb.squeeze(1).numpy().copy())
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
            probs_list.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())

    return (
        total_loss / max(1, total_samples),
        np.concatenate(targets_list).astype(np.int64),
        np.concatenate(probs_list).astype(np.float32),
    )


def main(
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.15,
    patience: int = 5,
    device_name: str = "auto",
) -> None:
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        print(f"[ERROR] Missing {TRAIN_CSV}. Run Phase 1 first.", file=sys.stderr)
        sys.exit(1)

    device = select_device(device_name)
    print(f"Using device: {device}")

    # ── 1. Load Phase 1 training dataset ──────────────────────────────────────
    t0 = time.time()
    print(f"Loading {TRAIN_CSV}…")
    df = pd.read_csv(TRAIN_CSV)
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    y_all = df[LABEL_COL].to_numpy(dtype=np.int64)

    print(
        f"Loaded {len(df):,} rows x {len(feature_cols)} features "
        f"(Normal={int((y_all == 0).sum()):,}, Attack={int((y_all == 1).sum()):,}) "
        f"in {time.time() - t0:.1f}s"
    )

    # ── 2. Stratified train / validation split ────────────────────────────────
    idx_all = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=val_ratio,
        stratify=y_all,
        random_state=SEED,
    )

    X_train_df = df.iloc[train_idx][feature_cols]
    X_val_df   = df.iloc[val_idx][feature_cols]
    y_train    = y_all[train_idx].astype(np.float32)
    y_val      = y_all[val_idx].astype(np.float32)
    del df  # free raw DataFrame RAM

    # ── 3. Fit preprocessor strictly on train split ───────────────────────────
    preprocessor = FlowPreprocessor().fit(X_train_df)
    X_train = preprocessor.transform(X_train_df)
    X_val   = preprocessor.transform(X_val_df)
    del X_train_df, X_val_df

    print(f"Train split: {len(X_train):,} rows | Val split: {len(X_val):,} rows")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    # ── 4. Model, weighted loss, optimizer, scheduler ─────────────────────────
    hidden_dims = (256, 128, 64)
    dropout = 0.2
    model = CICIoTNet(
        in_features=len(feature_cols),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    pos_weight_val = 1.0
    if WEIGHTS_JSON.exists():
        weights_data = json.loads(WEIGHTS_JSON.read_text())
        pos_weight_val = float(weights_data.get("pos_weight", 1.0))

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: CICIoTNet({len(feature_cols)} -> {hidden_dims} -> 1), trainable params={n_params:,}")
    print(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight_val:.4f})\n")

    # ── 5. Training loop with early stopping on Val ROC-AUC ───────────────────
    best_auc = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    epochs_no_improve = 0
    history: list[dict] = []

    header = (
        f"{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>8}  "
        f"{'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'F1-Mac':>7}  {'ROC-AUC':>8}  {'LR':>9}  {'Time':>5}"
    )
    print(header)
    print("─" * len(header))

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        val_loss, y_true_val, y_prob_val = evaluate_loader(model, val_loader, loss_fn, device)
        val_metrics = compute_metrics(y_true_val, y_prob_val, threshold=0.5)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        ep_time = time.time() - t_ep
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(current_lr, 8),
            **{k: round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(row)

        improved = val_metrics["roc_auc"] > best_auc + 1e-5
        marker = " ★" if improved else ""
        print(
            f"{epoch:>5}  {train_loss:>9.5f}  {val_loss:>8.5f}  "
            f"{val_metrics['precision']:>7.4f}  {val_metrics['recall']:>7.4f}  "
            f"{val_metrics['f1']:>7.4f}  {val_metrics['f1_macro']:>7.4f}  "
            f"{val_metrics['roc_auc']:>8.5f}  {current_lr:>9.2e}  {ep_time:>4.1f}s{marker}"
        )

        if improved:
            best_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_no_improve = 0

            # Save best model checkpoint + preprocessing parameters
            preprocessor.save(PREP_PATH)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "in_features": len(feature_cols),
                        "hidden_dims": list(hidden_dims),
                        "dropout": dropout,
                    },
                    "training_config": {
                        "epochs": epochs,
                        "best_epoch": best_epoch,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "pos_weight": pos_weight_val,
                        "val_ratio": val_ratio,
                        "seed": SEED,
                    },
                    "best_val_metrics": best_val_metrics,
                },
                CKPT_PATH,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

    HISTORY_PATH.write_text(json.dumps(history, indent=2))

    print(f"\n{'═' * 72}")
    print(f"  Training complete — best epoch {best_epoch}")
    print(f"  Val Precision : {best_val_metrics['precision']:.4f}")
    print(f"  Val Recall    : {best_val_metrics['recall']:.4f}")
    print(f"  Val F1        : {best_val_metrics['f1']:.4f}  (Macro F1: {best_val_metrics['f1_macro']:.4f})")
    print(f"  Val ROC-AUC   : {best_val_metrics['roc_auc']:.5f}")
    print(f"  Val PR-AUC    : {best_val_metrics['pr_auc']:.5f}")
    print(f"  Model saved   → {CKPT_PATH}")
    print(f"  Preprocessor  → {PREP_PATH}")
    print(f"  History       → {HISTORY_PATH}")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 — Train CICIoTNet on Phase 1 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        patience=args.patience,
        device_name=args.device,
    )
