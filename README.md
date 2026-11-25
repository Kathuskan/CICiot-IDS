# CICIoT-IDS — Intrusion Detection System for IoT Traffic

A PyTorch-based binary intrusion detection system (IDS) trained on the **CICIoT2023** dataset. It classifies network flows as **Normal** or **Suspicious/Attack** using a fully-connected neural network, and includes utility scripts to inspect and validate the raw CSV data.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Setup](#setup)
- [Phases](#phases)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Roadmap](#roadmap)

---

## Overview

This project demonstrates a complete pipeline for network intrusion detection on IoT traffic:

1. **Data exploration** — inspect raw pcap-derived CSVs for shape, missing values, infinite values, and label distribution.
2. **Model training** — a lightweight 2-layer feedforward neural network trained end-to-end with PyTorch.
3. **Inference** — classify a single new network flow as *Normal* or *Suspicious*.

The current `ids_demo.py` uses a **synthetically generated** dataset (matching the statistical profile of real IoT traffic) so you can run a full training/evaluation loop without downloading the full CICIoT2023 dataset first.

---

## Project Structure

```
ciciot-ids/
├── data/
│   └── raw/                     # CICIoT2023 CSV files (one sub-folder per class)
│       ├── Backdoor_Malware/
│       │   └── Backdoor_Malware.pcap.csv
│       ├── Benign_Final/
│       │   └── BenignTraffic3.pcap.csv
│       ├── DDoS-HTTP_Flood/
│       │   └── DDoS-HTTP_Flood-.pcap.csv
│       ├── ... (34 attack/benign categories total)
│       └── README_CSV.pdf       # Official dataset column documentation
├── docs/
│   └── PHASE1.md                # Phase 1 documentation & verified dataset report
├── src/
│   ├── check_csv.py             # Quick data-quality check for a single CSV
│   ├── inspect_dataset.py       # Full dataset inspection (label distribution, dtypes, NaN/Inf)
│   └── phase1/
│       ├── 01_verify_csvs.py    # Phase 1 step 1: verify all 215 CSVs & schema
│       └── 02_prepare_dataset.py# Phase 1 step 2: clean, deduplicate, split & balance
├── ids_demo.py                  # End-to-end demo: synthetic data → train → evaluate → infer
├── .venv/                       # Python virtual environment (Python 3.9)
└── README.md
```

---

## Dataset

The raw data follows the **CICIoT2023** benchmark, which captures 34 distinct traffic categories:

| Category | Examples |
|---|---|
| **Benign** | BenignTraffic |
| **DDoS** | ACK Fragmentation, HTTP Flood, ICMP Flood, ICMP Fragmentation, PSH-ACK Flood, RST-FIN Flood, SYN Flood, SlowLoris, SynonymousIP Flood, TCP Flood, UDP Flood, UDP Fragmentation |
| **DoS** | HTTP Flood, SYN Flood, TCP Flood, UDP Flood |
| **Recon** | Host Discovery, OS Scan, Ping Sweep, Port Scan |
| **Web attacks** | Browser Hijacking, Command Injection, SQL Injection, Uploading Attack, XSS |
| **Malware** | Backdoor, Mirai (greeth_flood, greip_flood, udpplain) |
| **Other** | DNS Spoofing, Dictionary Brute Force, MITM ARP Spoofing, Vulnerability Scan |

Each CSV contains **39 flow-level features** extracted from pcap captures:

| Group | Features |
|---|---|
| Header metadata | `Header_Length`, `Protocol Type`, `Time_To_Live` |
| Traffic rate | `Rate`, `IAT`, `Number` |
| TCP flags | `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number` |
| Flag counts | `ack_count`, `syn_count`, `fin_count`, `rst_count` |
| Application protocols | `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC` |
| Network protocols | `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IGMP`, `IPv`, `LLC` |
| Payload statistics | `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `Variance` |

> [!NOTE]
> The dataset is not bundled in this repository due to size. Download it from the [CICIoT2023 official page](https://www.unb.ca/cic/datasets/iotdataset-2023.html) and place each CSV under its corresponding `data/raw/<ClassName>/` folder.

---

## Features

- **Synthetic demo mode** — run the full pipeline immediately with no external data dependency (`ids_demo.py`)
- **Chunked CSV inspection** — `check_csv.py` and `inspect_dataset.py` process arbitrarily large CSV files in 100 k-row chunks without exhausting memory
- **Standard ML pipeline** — train/test split (80/20), z-score feature standardisation, binary cross-entropy loss
- **Single-flow inference** — classify any new network flow with a single forward pass

---

## Phases

The project is built as a series of explicit phases, each with its own documentation:

| Phase | Description | Docs |
|---|---|---|
| **Phase 1** | Real dataset preparation — verify CSVs, label mapping, clean & split | [docs/PHASE1.md](docs/PHASE1.md) |

---

## Setup

**Requirements:** Python 3.9 and the packages below (all already installed in `.venv`).

| Package | Version |
|---|---|
| `torch` | 2.8.0 |
| `numpy` | 2.0.2 |
| `pandas` | 2.3.3 |
| `scikit-learn` | 1.6.1 |
| `networkx` | 3.2.1 |
| `sympy` | 1.14.0 |

**Activate the virtual environment:**

```bash
# From the project root
source .venv/bin/activate
```

**Or create a fresh environment and install dependencies:**

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scikit-learn
```

---

## Usage

### 1. Run the end-to-end demo (synthetic data)

No dataset required. Generates synthetic IoT flow data, trains the model, and evaluates it:

```bash
python ids_demo.py
```

Expected output:
```
Training feature shape: torch.Size([1600, 4])
Testing feature shape:  torch.Size([400, 4])
Epoch  20/100, Loss: 0.6821
...
Epoch 100/100, Loss: 0.5934
Test accuracy: ~75%
Suspicious-traffic score: 0.812
Prediction: Suspicious
```

---

### 2. Inspect a specific CSV file (data quality check)

Hardcoded to `data/raw/Backdoor_Malware/Backdoor_Malware.pcap.csv`. Reports row count, column types, missing values, infinite values, and numeric feature ranges:

```bash
python src/check_csv.py
```

---

### 3. Inspect any CSV with label distribution

Pass any raw CSV path. Auto-detects the label column and prints full data quality stats plus the class distribution:

```bash
python src/inspect_dataset.py data/raw/Backdoor_Malware/Backdoor_Malware.pcap.csv
python src/inspect_dataset.py data/raw/Benign_Final/BenignTraffic3.pcap.csv
```

---

## Model Architecture

The demo model is a minimal feedforward network:

```
Input (4 features)
       │
  Linear(4 → 16)
       │
     ReLU
       │
  Linear(16 → 1)
       │
  BCEWithLogitsLoss  ←── training
       │
   Sigmoid ≥ 0.5    ←── inference threshold
```

**Training details:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.01 |
| Epochs | 100 |
| Loss | Binary Cross-Entropy with Logits |
| Train/Test split | 80 / 20 |

**Demo features (synthetic):**

| Feature | Normal traffic | Attack traffic |
|---|---|---|
| `duration` | log-normal(µ=0.7, σ=0.7) | log-normal(µ=−0.1, σ=0.8) |
| `packet_count` | Poisson(λ=25) | Poisson(λ=65) |
| `bytes_per_second` | log-normal(µ=7.5, σ=0.8) | log-normal(µ=8.5, σ=0.9) |
| `failed_connection_rate` | Beta(1, 12) | Beta(3, 5) |

---

## Results

On the synthetic demo dataset (seed 42, n=2 000):

| Metric | Value |
|---|---|
| Test accuracy | ~75% |
| Train/Test rows | 1 600 / 400 |

> [!TIP]
> Training on the full CICIoT2023 dataset (215 CSV files, millions of rows) with all 39 features and a deeper network is expected to yield significantly higher accuracy.

---

## Roadmap

- [ ] Load and merge the full CICIoT2023 CSVs into a single training pipeline
- [ ] Multi-class classification across all 34 attack categories
- [ ] Feature selection / importance analysis
- [ ] Deeper network architectures (BatchNorm, Dropout, residual blocks)
- [ ] Evaluation metrics: F1, precision, recall, ROC-AUC, confusion matrix
- [ ] Model checkpointing and serialisation (`torch.save`)
- [ ] CLI interface for inference on live pcap/CSV input
