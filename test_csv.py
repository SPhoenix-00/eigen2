"""Quick test to verify CSV can be read"""
import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / "Eigen2_Master(GFIN)_03_training.csv"

print(f"CSV Path: {csv_path}")
print(f"File exists: {csv_path.exists()}")

if csv_path.exists():
    print("\nReading CSV...")
    df = pd.read_csv(csv_path, nrows=5)  # Read just first 5 rows
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nFirst few columns:")
    print(df.columns[:10].tolist())
    print(f"\nFirst row, first cell (should be a date):")
    print(df.iloc[0, 0])
    print(f"\nFirst row, column K (11th column, should be stock data):")
    print(df.iloc[0, 10])
else:
    print("ERROR: CSV file not found!")