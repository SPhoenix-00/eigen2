"""Quick script to check column name at index 8"""
import pandas as pd
from pathlib import Path

# Load the pickle file
pkl_path = Path(__file__).parent / "Eigen2_Master_PY_OUTPUT.pkl"

if pkl_path.exists():
    print(f"Loading pickle file: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    print(f"\nTotal columns in pkl: {df.shape[1]}")
    print(f"Total rows in pkl: {df.shape[0]}")

    print(f"\n{'='*60}")
    print(f"Column at index 8 (INVESTABLE_START_COL):")
    print(f"{'='*60}")
    print(f"  Index: 8")
    print(f"  Name: '{df.columns[8]}'")

    print(f"\n{'='*60}")
    print(f"First 10 column names:")
    print(f"{'='*60}")
    for i in range(min(10, len(df.columns))):
        marker = " <-- INVESTABLE_START_COL" if i == 8 else ""
        print(f"  {i:3d}: {df.columns[i]}{marker}")

    print(f"\n{'='*60}")
    print(f"Columns around index 8:")
    print(f"{'='*60}")
    for i in range(max(0, 8-2), min(len(df.columns), 8+5)):
        marker = " <-- INVESTABLE_START_COL" if i == 8 else ""
        print(f"  {i:3d}: {df.columns[i]}{marker}")

    print(f"\n{'='*60}")
    print(f"Last few columns we'll load (113-116):")
    print(f"{'='*60}")
    for i in range(113, min(len(df.columns), 117)):
        marker = " <-- INVESTABLE_END_COL" if i == 115 else ""
        marker2 = " <-- Last column loaded (column 116)" if i == 116 else ""
        print(f"  {i:3d}: {df.columns[i]}{marker}{marker2}")

else:
    print(f"ERROR: Pickle file not found at {pkl_path}")
