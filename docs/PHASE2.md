# Phase 2 — Deep Learning Intrusion Detection

> **Goal:** Build, train, and evaluate a real-data PyTorch neural network on
> Phase 1's processed CICIoT2023 splits using Precision, Recall, F1-score, and
> ROC-AUC, then persist both the trained weights and deterministic preprocessing
> parameters for inference.

---

## Table of Contents

- [Overview](#overview)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Verified Results](#verified-results)
  - [Validation Split (165,445 flows)](#validation-split-165445-flows)
  - [Held-Out Test Set (3,214,278 flows)](#held-out-test-set-3214278-flows)
- [Saved Artifacts](#saved-artifacts)
- [Quick Start](#quick-start)
- [Programmatic Inference](#programmatic-inference)

---

## Overview

Phase 2 transitions from the synthetic 4-feature demo (`ids_demo.py`) to the
full **39-feature** real CICIoT2023 pipeline:

```
data/processed/phase1/train.csv.gz  (1,102,962 rows)
        │
        ├── 85% stratified train (937,517) ──► fit FlowPreprocessor + train CICIoTNet
        └── 15% stratified val   (165,445) ──► early stopping & best checkpoint (Val ROC-AUC)
                                                      │
                                                      ▼
                                  models/phase2/ciciot_ids_model.pt
                                  models/phase2/preprocessor.json
                                                      │
                                                      ▼
data/processed/phase1/test.csv.gz   (3,214,278 rows) ─┴─► evaluate.py ──► results/phase2/evaluation_report.json
```

All code lives in `src/phase2/`:

| File | Purpose |
|---|---|
| `src/phase2/model.py` | `FlowPreprocessor` (log1p + z-score) and `CICIoTNet` neural network |
| `src/phase2/train.py` | Trains `CICIoTNet` on Apple Silicon MPS / CUDA / CPU with weighted loss & early stopping |
| `src/phase2/evaluate.py` | Streams `test.csv.gz` in chunks and computes Precision, Recall, F1, ROC-AUC, PR-AUC, and confusion matrix |

---

## Preprocessing Pipeline

Network flow features span several orders of magnitude (`Rate` up to `7.34e6`,
`Variance` up to `9.09e7`, `IAT` down to `1.9e-7`). Every feature in the
dataset is non-negative (`min >= 0`).

`FlowPreprocessor` applies two deterministic steps, **fit strictly on the
85% training split**:

1. **`log1p` compression** on 16 skewed continuous/count columns (`np.log1p(np.clip(x, 0, None))`):
   `Header_Length`, `Time_To_Live`, `Rate`, `ack_count`, `syn_count`,
   `fin_count`, `rst_count`, `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`,
   `IAT`, `Number`, `Variance`.
2. **Z-score standardization**: `(x - mean) / std` across all 39 features
   (guarding `std == 0 -> 1.0`).

All fitted arrays (`feature_columns`, `log1p_columns`, `mean`, `std`,
`threshold`) are saved as human-readable JSON in `models/phase2/preprocessor.json`.

---

## Model Architecture

`CICIoTNet` is a regularized tabular MLP (`52,353` trainable parameters):

```
Input (39 features)
       │
  Linear(39 → 256) → BatchNorm1d(256) → SiLU → Dropout(0.20)
       │
  Linear(256 → 128) → BatchNorm1d(128) → SiLU → Dropout(0.20)
       │
  Linear(128 → 64) → BatchNorm1d(64)  → SiLU → Dropout(0.10)
       │
  Linear(64 → 1)     [raw logit]
       │
  BCEWithLogitsLoss(pos_weight=0.1039)  ←── training
       │
  Sigmoid(logit) ≥ 0.50                 ←── inference
```

---

## Training Configuration

| Setting | Value |
|---|---|
| Train / Val split | 85% / 15% stratified (`seed=42`) |
| Device | Apple Silicon `mps` (auto-selects `mps` / `cuda` / `cpu`) |
| Optimizer | `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) |
| LR Scheduler | `CosineAnnealingLR` (`T_max=20`, `eta_min=1e-5`) |
| Gradient clipping | `max_norm=2.0` |
| Loss | `BCEWithLogitsLoss(pos_weight=0.103929)` from Phase 1 `class_weights.json` |
| Batch size | `4,096` (train), `8,192` (val), `16,384` (test eval) |
| Early stopping | Patience `5` epochs monitoring Validation ROC-AUC |
| Best epoch | **18** / 20 (~5.7 s/epoch) |

---

## Verified Results

### Validation Split (165,445 flows)

| Metric | Value (Epoch 18) |
|---|---|
| Precision (Attack) | **0.9997** |
| Recall (Attack) | **0.9760** |
| F1-score (Attack) | **0.9877** |
| Macro F1 | **0.9415** |
| ROC-AUC | **0.99589** |
| PR-AUC | **0.99959** |

### Held-Out Test Set (3,214,278 flows)

Evaluated in **4.9 s** on unseen source capture files (`25,960` Normal, `3,188,318` Attack):

| Metric | Value |
|---|---|
| Accuracy | **96.0678%** |
| Precision (Attack) | **1.0000** |
| Recall (Attack) | **0.9604** |
| F1-score (Attack) | **0.9798** |
| Recall (Normal) | **0.9976** |
| Macro F1 | **0.6352** |
| Weighted F1 | **0.9742** |
| ROC-AUC | **0.99509** |
| PR-AUC | **0.99996** |

**Confusion Matrix (`threshold = 0.5`):**

|  | Pred Normal (`0`) | Pred Attack (`1`) |
|---|---|---|
| **Actual Normal (`0`)** | 25,897 | 63 |
| **Actual Attack (`1`)** | 126,329 | 3,061,989 |

> [!NOTE]
> Because `pos_weight=0.1039` balances class penalties during training, the
> default `0.5` threshold achieves **99.76% Normal recall** (only 63 false
> alarms out of 25,960 normal flows) alongside **96.04% Attack recall** and
> **0.99509 ROC-AUC** on completely unseen captures.

---

## Saved Artifacts

| Path | Size | Contents |
|---|---|---|
| `models/phase2/ciciot_ids_model.pt` | ~216 KB | `model_state_dict`, `model_config`, `training_config`, `best_val_metrics` |
| `models/phase2/preprocessor.json` | ~2.8 KB | `feature_columns`, `log1p_columns`, `mean`, `std`, `threshold` |
| `models/phase2/training_history.json` | ~4.8 KB | Per-epoch loss, LR, and validation metrics |
| `results/phase2/evaluation_report.json` | ~0.6 KB | Full held-out test metrics and confusion matrix |

---

## Quick Start

```bash
source .venv/bin/activate

# 1. Train the neural network (saves best checkpoint & preprocessor)
python src/phase2/train.py

# 2. Evaluate on the held-out Phase 1 test set
python src/phase2/evaluate.py
```

---

## Programmatic Inference

```python
import pandas as pd
import torch
from src.phase2.evaluate import load_artifacts

model, preprocessor, ckpt, device = load_artifacts()

# DataFrame with the 39 raw CICIoT2023 flow columns
sample_df = pd.read_csv("data/processed/phase1/test.csv.gz", nrows=5).drop(columns=["binary_label"])
X = torch.from_numpy(preprocessor.transform(sample_df)).to(device)

with torch.no_grad():
    probs = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()

for i, p in enumerate(probs):
    label = "Attack" if p >= preprocessor.threshold else "Normal"
    print(f"Flow {i}: {label} (p={p:.4f})")
```
