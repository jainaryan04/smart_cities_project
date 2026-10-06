import os
import pandas as pd
import numpy as np

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export.csv")

print(f"Checking file: {csv_path}")
print(f"File size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")

# Read first few lines directly
with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
    header_line = f.readline().strip()
    sample_lines = [f.readline().strip() for _ in range(5)]

print("\n--- Header ---")
print(header_line)

print("\n--- First 3 Sample Rows ---")
for i, line in enumerate(sample_lines[:3]):
    print(f"Row {i+1}: {line}")

# Inspect with pandas (read in chunks or full to get exact stats)
print("\nReading dataset info via pandas...")
chunksize = 1_000_000
total_rows = 0
cols = None
null_counts = None
locations = set()
v_classes = set()
total_vehicles = set()

for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
    total_rows += len(chunk)
    if cols is None:
        cols = chunk.columns.tolist()
        null_counts = chunk.isnull().sum()
    else:
        null_counts += chunk.isnull().sum()
    
    if 'Location' in chunk.columns:
        locations.update(chunk['Location'].dropna().unique())
    if 'v_Class' in chunk.columns:
        v_classes.update(chunk['v_Class'].dropna().unique())
    if 'Vehicle_ID' in chunk.columns:
        # Note: Vehicle IDs are unique per location/dataset section
        if 'Location' in chunk.columns:
            for loc, grp in chunk.groupby('Location'):
                total_vehicles.update([(loc, vid) for vid in grp['Vehicle_ID'].unique()])
        else:
            total_vehicles.update(chunk['Vehicle_ID'].unique())

print(f"\nTotal rows: {total_rows:,}")
print(f"Columns ({len(cols)}): {cols}")
print(f"\nLocations present: {locations}")
print(f"Vehicle Classes present: {v_classes}")
print(f"Total Unique (Location, Vehicle_ID) combinations: {len(total_vehicles):,}")

print("\nMissing values per column:")
for col, cnt in null_counts.items():
    pct = (cnt / total_rows) * 100
    print(f"  {col}: {cnt:,} ({pct:.2f}%)")

# Detailed column summary
sample_df = pd.read_csv(csv_path, nrows=1000)
print("\nData Types:")
print(sample_df.dtypes)
print("\nStatistical Summary (Sample 1000 rows):")
print(sample_df[['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Lane_ID']].describe() if all(c in sample_df.columns for c in ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Lane_ID']) else sample_df.describe())
