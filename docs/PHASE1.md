# Phase 1 — Real Dataset Preparation

> **Goal:** Turn 215 raw pcap-derived CSVs (3.8 GB, 21.4 M rows) into clean,
> labelled, leak-minimized `train.csv.gz` and `test.csv.gz` files that fit
> comfortably in Mac memory for downstream training and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Data Layout](#data-layout)
- [Binary Label Mapping](#binary-label-mapping)
- [Scripts](#scripts)
  - [Step 1 — Verify CSVs](#step-1--verify-csvs-01_verify_csvspy)
  - [Step 2 — Prepare Dataset](#step-2--prepare-dataset-02_prepare_datasetpy)
- [Design Decisions](#design-decisions)
  - [Leakage-Resistant Splitting](#leakage-resistant-splitting)
  - [Streaming Cleaning & O(1) Deduplication](#streaming-cleaning--o1-deduplication)
  - [Class Imbalance](#class-imbalance)
- [Verified Run Results](#verified-run-results)
- [Outputs](#outputs)
- [Quick Start](#quick-start)
- [Troubleshooting](#troubleshooting)

---

## Overview

The CICIoT2023 dataset ships as 215 CSV files spread across 33 non-empty class
directories (32 attack classes + 1 benign class; `VulnerabilityScan/` is empty).
Phase 1 handles verification, schema checks, binary labelling, memory-safe
streaming cleaning, cross-file deduplication, and leakage-resistant splitting:

```
data/raw/  (215 CSVs, 3.8 GB, 21,447,054 rows)
        │
        ▼  Step 1 — Verify (01_verify_csvs.py)
csv_manifest.json
        │
        ▼  Step 2 — Prepare (02_prepare_dataset.py)
train.csv.gz  +  test.csv.gz  +  class_weights.json  +  split_info.json
```

---

## Data Layout

```
data/raw/
├── Benign_Final/
│   └── BenignTraffic3.pcap.csv          ← label 0  (Normal, 1 file, 129,832 rows)
├── Backdoor_Malware/
│   └── Backdoor_Malware.pcap.csv        ← label 1  (Attack)
├── DDoS-ICMP_Flood/
│   └── … (27 files, 7,200,501 rows)     ← label 1
├── Mirai-greeth_flood/                  ← 29 files
├── Mirai-greip_flood/                   ← 22 files
├── Mirai-udpplain/                      ← 25 files
│   … (33 non-empty class directories total)
└── README_CSV.pdf
```

Each CSV contains **39 flow-level numeric feature columns** — no label column is
stored inside the CSVs; class identity is derived from the parent directory name.

| Stat | Value |
|---|---|
| Total CSV files | 215 (all valid) |
| Total raw size | 3.8 GB |
| Total raw rows | 21,447,054 |
| Feature columns | 39 |
| Missing values (`NaN`) | 923 (across all raw CSVs) |
| Infinite values (`±inf`) | 588 (across all raw CSVs) |
| Empty directories | `VulnerabilityScan/` (0 CSVs) |

---

## Binary Label Mapping

| Directory name | Class | Binary label (`binary_label`) |
|---|---|---|
| `Benign_Final` | Normal traffic | **0** |
| Any other directory | Attack traffic | **1** |

Defined in `src/phase1/01_verify_csvs.py`:

```python
BENIGN_CLASSES: frozenset[str] = frozenset({"Benign_Final"})

def binary_label(class_name: str) -> int:
    return 0 if class_name in BENIGN_CLASSES else 1
```

---

## Scripts

### Step 1 — Verify CSVs (`01_verify_csvs.py`)

**What it does**

1. Recursively discovers every `*.csv` under `data/raw/`.
2. Establishes a reference 39-column schema from the first valid file.
3. Streams each file in 100k-row chunks, collecting row counts, `NaN` counts,
   `±inf` counts, and verifying columns match the reference schema.
4. Flags issues (`COLUMN_MISMATCH`, `EMPTY`, `ERROR`) if encountered.
5. Writes `data/processed/phase1/csv_manifest.json`.

```bash
python src/phase1/01_verify_csvs.py
```

---

### Step 2 — Prepare Dataset (`02_prepare_dataset.py`)

**What it does**

1. Loads `csv_manifest.json` and keeps only `OK` files.
2. Applies a leakage-resistant split across source files (see below).
3. Streams each file in chunks into single-stream gzip CSVs (`train.csv.gz` and
   `test.csv.gz`), keeping RAM usage bounded.
4. Replaces `±inf → NaN`, drops rows with any `NaN`, and removes exact duplicate
   feature vectors across the split.
5. Undersamples the majority Attack class in `TRAIN` only when the post-dedup
   ratio exceeds `--max-attack-ratio` (default `10:1`).
6. Computes balanced class weights and `pos_weight` from actual `TRAIN` counts
   and writes `class_weights.json`, `split_info.json`, and `preparation_report.json`.

```bash
# Default: 80/20 split, 100k-row chunks, max 10:1 train attack:normal ratio
python src/phase1/02_prepare_dataset.py

# Custom settings
python src/phase1/02_prepare_dataset.py --test-ratio 0.15 --chunk 50000 --max-attack-ratio 5.0
```

| Flag | Default | Description |
|---|---|---|
| `--test-ratio` | `0.20` | Fraction of source files per class (and trailing block for single-file Normal) held out for test |
| `--chunk` | `100000` | Rows per pandas `read_csv` chunk |
| `--max-attack-ratio` | `10.0` | Maximum Attack:Normal ratio in `TRAIN` before undersampling Attack rows |

---

## Design Decisions

### Leakage-Resistant Splitting

> [!IMPORTANT]
> Whole source files are held out for test wherever multiple captures exist,
> rather than randomly shuffling rows across captures.

| Class group | Strategy | Result |
|---|---|---|
| **Multi-file attack classes** (17 classes, `>= 2` files each) | Stratified whole-file split per class (`seed=42`) | Every multi-file attack family appears in both `TRAIN` and `TEST` via completely separate capture files |
| **Single-file attack classes** (15 classes, 1 file each) | Whole-file split across the 1-file pool (12 train, 3 test) | Tests generalization to unseen attack types (`Backdoor_Malware`, `DDoS-HTTP_Flood`, `DNS_Spoofing` held out in test) |
| **Single-file benign class** (`Benign_Final/BenignTraffic3.pcap.csv`) | Cleaned & deduplicated first, then split into contiguous blocks: leading 80% → `TRAIN`, trailing 20% → `TEST` | Guarantees Normal (`0`) samples in both splits with 0 duplicates across splits and minimal temporal leakage |

All exact file assignments are recorded in `data/processed/phase1/split_info.json`.

---

### Streaming Cleaning & O(1) Deduplication

Each chunk goes through:

1. **Infinite → NaN → Drop:** `np.inf` and `-np.inf` in numeric columns are
   converted to `NaN`, then any row with `NaN` is dropped (`dropna()`).
2. **Cross-file deduplication:** `pd.util.hash_pandas_object(chunk, index=False)`
   hashes every row's 39 feature values into 64-bit integers and filters against
   a `seen_hashes` set shared across the split using $O(1)$ per-row set lookups.
3. **Single-stream gzip writing:** `train.csv.gz` and `test.csv.gz` are kept
   open as single `gzip.open(..., "wt")` streams for the entire run, finishing all
   215 files (21.4 M rows) in ~88.5 seconds on Mac.

---

### Class Imbalance

Raw CICIoT2023 has a **164.2:1** Attack:Normal imbalance (`21,317,222` Attack vs
`129,832` Normal). Two complementary safeguards are used:

1. **Train-only random undersampling:** Caps `TRAIN` Attack rows at
   `--max-attack-ratio` (default `10:1`, actual run `9.62:1`). `TEST` is **never**
   undersampled so evaluation reflects the real-world skew.
2. **Balanced class weights (`class_weights.json`):** Computed from the final
   `TRAIN` label counts:

```python
import json, torch
from torch import nn

weights = json.load(open("data/processed/phase1/class_weights.json"))
loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([weights["pos_weight"]])
)
```

---

## Verified Run Results

Observed results from running both Phase 1 scripts on the full 215-file dataset:

| Metric | TRAIN | TEST |
|---|---|---|
| Source files | 168 attack files + 80% Normal block | 46 attack files + 20% Normal block |
| Rows read | 16,511,378 | 4,935,676 |
| Rows after cleaning (`NaN`/`Inf` removed) | 16,510,912 | 4,935,551 |
| Rows after deduplication | 8,784,474 | 3,214,278 |
| Rows written | **1,102,962** | **3,214,278** |
| Normal (`0`) rows | 103,838 | 25,960 |
| Attack (`1`) rows | 999,124 | 3,188,318 |
| Attack:Normal ratio | **9.62:1** | **122.82:1** |
| Compressed file size | **42.74 MB** | **113.6 MB** |

Computed `class_weights.json`:
- `normal`: `5.310975`
- `attack`: `0.551965`
- `pos_weight`: `0.103929`

---

## Outputs

All outputs are written to `data/processed/phase1/` (ignored by git):

| File | Description |
|---|---|
| `csv_manifest.json` | Catalog of all 215 raw CSVs with quality stats and labels |
| `train.csv.gz` | Cleaned, deduplicated, balanced training split (40 cols: 39 features + `binary_label`) |
| `test.csv.gz` | Cleaned, deduplicated test split (natural class distribution) |
| `split_info.json` | Exact list of source files per split and block-split metadata |
| `class_weights.json` | Balanced weights + `pos_weight` for `BCEWithLogitsLoss` |
| `preparation_report.json` | Complete statistics and timing for the preparation run |

---

## Quick Start

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Verify all raw CSVs (produces csv_manifest.json)
python src/phase1/01_verify_csvs.py

# 3. Build clean, leak-minimized train / test datasets
python src/phase1/02_prepare_dataset.py
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Manifest not found` | Step 2 run before Step 1 | Run `python src/phase1/01_verify_csvs.py` first |
| `No CSV files found` | Wrong working directory | Run scripts from the project root |
| `MemoryError` | Chunk size too large for host RAM | Re-run with `--chunk 25000` |
| `COLUMN_MISMATCH` files | Schema drift in a CSV | Check flagged file in `csv_manifest.json`; excluded automatically |
