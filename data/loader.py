"""
Data loader for Project Eigen 2
Handles CSV parsing, feature extraction, and windowing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import ast
from tqdm import tqdm

from utils.config import Config


class StockDataLoader:
    """
    Loads and preprocesses stock market data from CSV.
    Handles nan values, extracts 5 selected features from 9-element arrays, and creates sliding windows.
    Selected features: close, RSI, MACD_signal, TRIX, diff20DMA
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            csv_path: Path to CSV file. If None, uses Config.DATA_PATH
        """
        self.csv_path = csv_path or Config.DATA_PATH
        self.df = None
        self.dates = None
        self.data_array = None  # Shape: [num_days, num_columns, 9_features]
        self.train_indices = None
        self.val_indices = None
        
    def load_csv(self) -> pd.DataFrame:
        """Load CSV file into pandas DataFrame."""
        print(f"Loading CSV from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        
        # Drop rows where date column is NaN (blank rows at end)
        date_col_name = self.df.columns[Config.DATE_COLUMN]
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[date_col_name])
        dropped_rows = initial_rows - len(self.df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} blank rows")
        
        print(f"Loaded data shape: {self.df.shape}")
        print(f"Columns: {self.df.shape[1]}, Rows (days): {self.df.shape[0]}")
        
        # Extract dates (column A, index 0)
        self.dates = self.df.iloc[:, Config.DATE_COLUMN].values
        print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
        
        return self.df
    
    def parse_cell_data(self, cell_value) -> np.ndarray:
        """
        Parse a single cell containing string-formatted array and extract selected features.

        Args:
            cell_value: String like "[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]" or nan

        Returns:
            numpy array of shape (5,) with selected features: [close, RSI, MACD_signal, TRIX, diff20DMA]
            Original indices: [1, 4, 6, 7, 8] from the 9-element array
        """
        # Handle nan/None values
        if pd.isna(cell_value):
            return np.full(Config.FEATURES_PER_CELL, np.nan, dtype=np.float32)

        # Handle empty strings
        if isinstance(cell_value, str) and cell_value.strip() == '':
            return np.full(Config.FEATURES_PER_CELL, np.nan, dtype=np.float32)

        try:
            # Parse string as list
            data = ast.literal_eval(cell_value)

            # Convert to numpy array
            arr = np.array(data, dtype=np.float32)

            # Select only the 5 features we want: close, RSI, MACD_signal, TRIX, diff20DMA
            # Original 9-element array: [open, close, high, low, RSI, MACD, MACD_signal, TRIX, diff20DMA]
            # Indices we want:           [  0,     1,    2,   3,   4,    5,           6,     7,        8]
            # Select indices:            [       1,               4,                   6,     7,        8]

            if len(arr) >= 9:
                # Extract the 5 selected features
                selected_features = arr[[1, 4, 6, 7, 8]]
                return selected_features.astype(np.float32)
            else:
                # If array is too short, return nans
                print(f"Warning: Expected at least 9 features, got {len(arr)}")
                return np.full(Config.FEATURES_PER_CELL, np.nan, dtype=np.float32)

        except (ValueError, SyntaxError, IndexError) as e:
            # If parsing fails, return nans
            return np.full(Config.FEATURES_PER_CELL, np.nan, dtype=np.float32)
    
    def extract_features(self) -> np.ndarray:
        """
        Extract all features from DataFrame into structured array.

        Returns:
            numpy array of shape [num_days, num_columns-1, 5]
            (excluding date column, only 5 selected features)
        """
        if self.df is None:
            raise ValueError("Must call load_csv() first")
        
        num_days = len(self.df)
        # Exclude date column (column A)
        num_columns = self.df.shape[1] - 1
        
        print(f"\nExtracting features from {num_days} days × {num_columns} columns...")
        
        # Initialize array
        self.data_array = np.zeros((num_days, num_columns, Config.FEATURES_PER_CELL), 
                                   dtype=np.float32)
        
        # Process each column (skip date column)
        for col_idx in tqdm(range(1, self.df.shape[1]), desc="Processing columns"):
            for day_idx in range(num_days):
                cell_value = self.df.iloc[day_idx, col_idx]
                # Store in data_array (col_idx-1 because we skip date column)
                self.data_array[day_idx, col_idx - 1, :] = self.parse_cell_data(cell_value)
        
        # Report statistics
        total_values = self.data_array.size
        nan_count = np.isnan(self.data_array).sum()
        nan_percentage = (nan_count / total_values) * 100
        
        print(f"\nData extraction complete:")
        print(f"  Shape: {self.data_array.shape}")
        print(f"  Total values: {total_values:,}")
        print(f"  NaN values: {nan_count:,} ({nan_percentage:.2f}%)")
        print(f"  Valid values: {total_values - nan_count:,} ({100-nan_percentage:.2f}%)")
        
        # Check investable stocks specifically
        investable_data = self.data_array[:, Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL+1, :]
        inv_nan_pct = (np.isnan(investable_data).sum() / investable_data.size) * 100
        print(f"  NaN in investable stocks (columns {Config.INVESTABLE_START_COL}-{Config.INVESTABLE_END_COL}): {inv_nan_pct:.2f}%")
        
        return self.data_array
    
    def create_train_val_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into train and holdout sets (time-series aware).

        Walk-forward validation strategy:
        - Last HOLDOUT_DAYS rows are reserved as "secret" holdout data (never used during training)
        - All other rows are available for training and walk-forward validation
        - Validation uses random slices from training data at each generation

        Returns:
            Tuple of (train_indices, holdout_indices)
        """
        if self.data_array is None:
            raise ValueError("Must call extract_features() first")

        num_days = len(self.data_array)

        # Reserve last HOLDOUT_DAYS for final validation only
        holdout_start_idx = num_days - Config.HOLDOUT_DAYS

        self.train_indices = np.arange(0, holdout_start_idx)
        self.val_indices = np.arange(holdout_start_idx, num_days)  # Renamed to holdout_indices conceptually

        print(f"\nData Split (Walk-Forward Validation):")
        print(f"  Total days: {num_days}")
        print(f"  Training/Validation days: {len(self.train_indices)} ({self.dates[0]} to {self.dates[holdout_start_idx-1]})")
        print(f"  Holdout days (secret): {len(self.val_indices)} ({self.dates[holdout_start_idx]} to {self.dates[-1]})")
        print(f"  Note: Validation uses random slices from training data each generation")

        return self.train_indices, self.val_indices
    
    def get_window(self, end_idx: int) -> Optional[np.ndarray]:
        """
        Get a context window ending at end_idx.
        
        Args:
            end_idx: Index of the current day (window ends here)
            
        Returns:
            Window of shape [context_window_days, num_columns, 9_features]
            or None if not enough history
        """
        if self.data_array is None:
            raise ValueError("Must call extract_features() first")
        
        start_idx = end_idx - Config.CONTEXT_WINDOW_DAYS + 1
        
        # Check if we have enough history
        if start_idx < 0:
            return None
        
        window = self.data_array[start_idx:end_idx + 1, :, :]
        
        # Validate shape
        assert window.shape[0] == Config.CONTEXT_WINDOW_DAYS, \
            f"Window has {window.shape[0]} days, expected {Config.CONTEXT_WINDOW_DAYS}"
        
        return window
    
    def compute_normalization_stats(self) -> dict:
        """
        Compute mean and std for normalization from training data only.
        Handles nan values by computing stats only on valid data.
        
        Returns:
            Dictionary with 'mean' and 'std' arrays of shape [num_columns, 9_features]
        """
        if self.data_array is None or self.train_indices is None:
            raise ValueError("Must call extract_features() and create_train_val_split() first")
        
        print("\nComputing normalization statistics from training data...")
        
        # Get training data
        train_data = self.data_array[self.train_indices, :, :]
        
        # Compute mean and std per feature, ignoring nans
        mean = np.nanmean(train_data, axis=0)  # Shape: [num_columns, 5]
        std = np.nanstd(train_data, axis=0)    # Shape: [num_columns, 5]
        
        # Handle edge case: if std is 0 (constant feature), set to 1 to avoid division by zero
        std = np.where(std == 0, 1.0, std)
        std = np.where(np.isnan(std), 1.0, std)  # If all values were nan, set std to 1
        
        # Replace nan means with 0
        mean = np.where(np.isnan(mean), 0.0, mean)
        
        print(f"  Mean shape: {mean.shape}")
        print(f"  Std shape: {std.shape}")
        print(f"  Sample mean (column 0, feature 0): {mean[0, 0]:.4f}")
        print(f"  Sample std (column 0, feature 0): {std[0, 0]:.4f}")
        
        return {
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32)
        }
    
    def normalize_window(self, window: np.ndarray, stats: dict) -> np.ndarray:
        """
        Normalize a window using provided statistics.
        NaN values remain as NaN after normalization.
        
        Args:
            window: Array of shape [context_window_days, num_columns, 9_features]
            stats: Dictionary with 'mean' and 'std'
            
        Returns:
            Normalized window (same shape)
        """
        mean = stats['mean']
        std = stats['std']
        
        # Normalize: (x - mean) / std
        # NaN values will remain NaN
        normalized = (window - mean) / std
        
        return normalized.astype(np.float32)
    
    def load_and_prepare(self) -> Tuple[np.ndarray, dict]:
        """
        Convenience method to run full data loading pipeline.
        
        Returns:
            Tuple of (data_array, normalization_stats)
        """
        print("="*60)
        print("Data Loading Pipeline")
        print("="*60)
        
        # Load CSV
        self.load_csv()
        
        # Extract features
        self.extract_features()
        
        # Create train/val split
        self.create_train_val_split()
        
        # Compute normalization stats
        stats = self.compute_normalization_stats()
        
        print("\n" + "="*60)
        print("Data loading complete!")
        print("="*60)
        
        return self.data_array, stats


# Standalone test
if __name__ == "__main__":
    print("Testing StockDataLoader...\n")
    
    loader = StockDataLoader()
    data_array, stats = loader.load_and_prepare()
    
    print("\n--- Testing Window Extraction ---")
    # Test getting a window from middle of training data
    test_idx = len(loader.train_indices) // 2 + Config.CONTEXT_WINDOW_DAYS
    window = loader.get_window(test_idx)
    
    if window is not None:
        print(f"Window shape: {window.shape}")
        print(f"Window date range: {loader.dates[test_idx - Config.CONTEXT_WINDOW_DAYS + 1]} to {loader.dates[test_idx]}")
        
        # Test normalization
        normalized_window = loader.normalize_window(window, stats)
        print(f"Normalized window shape: {normalized_window.shape}")
        print(f"Normalized window mean (should be ~0): {np.nanmean(normalized_window):.4f}")
        print(f"Normalized window std (should be ~1): {np.nanstd(normalized_window):.4f}")
        
        # Check investable stocks specifically
        investable_slice = normalized_window[:, Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL+1, :]
        print(f"\nInvestable stocks slice shape: {investable_slice.shape}")
        print(f"NaN percentage in investable stocks: {(np.isnan(investable_slice).sum() / investable_slice.size) * 100:.2f}%")
    else:
        print("Not enough history for window at this index")
    
    print("\n✓ Data loader test complete!")