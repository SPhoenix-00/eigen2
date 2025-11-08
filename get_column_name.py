#!/usr/bin/env python3
"""Quick script to get column name at index 8 from pkl file"""

import pandas as pd
from pathlib import Path

# Load the pickle file
pkl_path = Path(__file__).parent / "Eigen2_Master_PY_OUTPUT.pkl"
df = pd.read_pickle(pkl_path)

# Get column at index 8 (INVESTABLE_START_COL as per config.py)
col_name = df.columns[8]
print(f"Column at index 8 (INVESTABLE_START_COL): {col_name}")
