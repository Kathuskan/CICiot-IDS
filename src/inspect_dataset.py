
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 1. Get the CSV path from the command line
if len(sys.argv) < 2:
    print(
        "Usage: python src/inspect_dataset.py "
        "data/raw/Backdoor_Malware.pcap.csv"
    )
    raise SystemExit(1)

csv_path = Path(sys.argv[1])

if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# 2. Inspect a small sample first
sample = pd.read_csv(csv_path, nrows=5, low_memory=False)

print("\n=== DATASET OVERVIEW ===")
print("File:", csv_path.name)
print("Columns:", len(sample.columns))
print("\nColumn names:")
for column in sample.columns:
    print("-", repr(column))

print("\n=== FIRST FIVE ROWS ===")
print(sample.to_string(index=False))

# 3. Find a likely target-label column
label_candidates = {
    "label", "class", "target", "attack", "attack_type"
}

label_column = next(
    (
        col for col in sample.columns
        if col.strip().lower() in label_candidates
    ),
    None
)

if label_column is None:
    print(
        "\nNo familiar label column was detected. "
        "Inspect the column names above before proceeding."
    )
    raise SystemExit(0)

print("\nDetected label column:", repr(label_column))

# 4. Scan the CSV in chunks to limit memory use
chunksize = 100_000
total_rows = 0
missing_counts = pd.Series(0, index=sample.columns, dtype="int64")
infinite_counts = pd.Series(0, index=sample.columns, dtype="int64")
label_counts = {}

for chunk in pd.read_csv(
    csv_path,
    chunksize=chunksize,
    low_memory=False
):
    total_rows += len(chunk)

    missing_counts = missing_counts.add(
        chunk.isna().sum(), fill_value=0
    )

    # Check numeric columns for positive/negative infinity
    numeric = chunk.select_dtypes(include=[np.number])
    if not numeric.empty:
        inf_counts = pd.Series(
            np.isinf(numeric.to_numpy()).sum(axis=0),
            index=numeric.columns
        )
        infinite_counts = infinite_counts.add(
            inf_counts, fill_value=0
        )

    counts = chunk[label_column].value_counts(dropna=False)
    for label, count in counts.items():
        label_counts[label] = (
            label_counts.get(label, 0) + int(count)
        )

print("\n=== DATASET SUMMARY ===")
print("Total rows scanned:", f"{total_rows:,}")
print("Total columns:", len(sample.columns))

print("\n=== COLUMN DATA TYPES (sample) ===")
print(sample.dtypes.to_string())

print("\n=== MISSING VALUES ===")
missing_counts = missing_counts.astype(int)
print(missing_counts[missing_counts > 0].sort_values(
    ascending=False
).to_string() or "No missing values found.")

print("\n=== INFINITE VALUES ===")
infinite_counts = infinite_counts.fillna(0).astype(int)
print(infinite_counts[infinite_counts > 0].sort_values(
    ascending=False
).to_string() or "No infinite values found in numeric columns.")

print("\n=== LABEL DISTRIBUTION ===")
for label, count in sorted(
    label_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{label!r}: {count:,}")

print("\nInspection complete.")
