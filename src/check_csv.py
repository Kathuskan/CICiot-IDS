
from pathlib import Path
import sys

import numpy as np
import pandas as pd

path = Path("data/raw/Backdoor_Malware.pcap.csv")

# Read the file in chunks to reduce memory usage
total_rows = 0
missing = None
infinite = None
dtypes = None
minimum = None
maximum = None

for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
    total_rows += len(chunk)

    if missing is None:
        missing = chunk.isna().sum()
        numeric = chunk.select_dtypes(include=np.number)
        infinite = pd.Series(0, index=numeric.columns, dtype="int64")
        dtypes = chunk.dtypes.astype(str)
        minimum = numeric.min()
        maximum = numeric.max()
    else:
        missing = missing.add(chunk.isna().sum(), fill_value=0)
        numeric = chunk.select_dtypes(include=np.number)
        infinite = infinite.add(
            pd.Series(
                np.isinf(numeric.to_numpy()).sum(axis=0),
                index=numeric.columns,
            ),
            fill_value=0,
        )
        minimum = pd.concat([minimum, numeric.min()], axis=1).min(axis=1)
        maximum = pd.concat([maximum, numeric.max()], axis=1).max(axis=1)

print("File:", path.name)
print("Rows:", f"{total_rows:,}")
print("Columns:", len(dtypes))
print("\nData types:")
print(dtypes.to_string())

print("\nMissing values by column:")
print(missing[missing > 0].sort_values(ascending=False).to_string()
      or "None")

print("\nInfinite values by column:")
infinite = infinite.fillna(0).astype(int)
print(infinite[infinite > 0].to_string() or "None")

print("\nNumeric feature ranges:")
print(pd.DataFrame({"min": minimum, "max": maximum}).to_string())

print("\nColumn names containing 'label', 'class', or 'attack':")
matches = [
    c for c in dtypes.index
    if any(word in c.lower() for word in ("label", "class", "attack"))
]
print(matches if matches else "None found")
