"""
Validation slice generation for Project Eigen 2.

Extracted from ERLTrainer — generates walk-forward validation slices
and gauntlet stress-test slices from training/validation data ranges.
"""

import numpy as np
from typing import List, Tuple

from utils.config import Config


def generate_validation_slices(
    val_start_idx: int,
    val_end_idx: int,
) -> List[Tuple[int, int, int]]:
    """
    Generate 10 validation slices from validation set.

    Divides validation period into 4 equal quarters, then samples:
    - 4 slices from within each quarter
    - 3 straddling slices between quarters
    - 3 random slices from anywhere in the validation period

    Args:
        val_start_idx: First valid day index for validation trading start
        val_end_idx: Last valid day index for validation data

    Returns:
        List of 10 tuples: (start_idx, end_idx, trading_end_idx)
    """
    min_start = val_start_idx
    max_start = val_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

    if max_start < min_start:
        raise ValueError(f"Not enough validation data: need {Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS} days")

    total_range = max_start - min_start + 1
    segment_size = total_range // 4

    slices = []

    # 1. Sample one slice from each quarter (4 slices)
    for segment_idx in range(4):
        segment_start = min_start + (segment_idx * segment_size)
        segment_end = max_start + 1 if segment_idx == 3 else min_start + ((segment_idx + 1) * segment_size)

        start_idx = np.random.randint(segment_start, segment_end)
        end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

        slices.append((start_idx, end_idx, trading_end_idx))

    # 2. Sample straddling slices between quarters (3 slices)
    for straddle_idx in range(3):
        straddle_start = min_start + (segment_size // 2) + (straddle_idx * segment_size)
        straddle_end = min_start + (segment_size // 2) + ((straddle_idx + 1) * segment_size)
        straddle_end = min(straddle_end, max_start + 1)

        if straddle_end > straddle_start:
            start_idx = np.random.randint(straddle_start, straddle_end)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

            slices.append((start_idx, end_idx, trading_end_idx))

    # 3. Sample 3 completely random slices (3 slices)
    for _ in range(3):
        start_idx = np.random.randint(min_start, max_start + 1)
        end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

        slices.append((start_idx, end_idx, trading_end_idx))

    return slices


def generate_gauntlet_slices(
    train_start_idx: int,
    train_end_idx: int,
    val_start_idx: int,
    val_end_idx: int,
) -> List[Tuple[int, int, int]]:
    """
    Generate 20+ rigorous validation slices for Gauntlet stress test.

    Samples slices from BOTH training and validation data to ensure the agent
    performs robustly across all market regimes.

    Args:
        train_start_idx: First day of training data
        train_end_idx: Last day of training data
        val_start_idx: First day of validation data
        val_end_idx: Last day of validation data

    Returns:
        List of 20 tuples: (start_idx, end_idx, trading_end_idx)
    """
    slices = []

    # 1. Sample 10 slices from training data
    min_start_train = train_start_idx
    max_start_train = train_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

    if max_start_train >= min_start_train:
        train_range = max_start_train - min_start_train + 1
        train_segment_size = max(1, train_range // 10)

        for i in range(10):
            segment_start = min_start_train + (i * train_segment_size)
            segment_end = min(max_start_train + 1, segment_start + train_segment_size)

            if segment_end > segment_start:
                start_idx = np.random.randint(segment_start, segment_end)
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

                slices.append((start_idx, end_idx, trading_end_idx))

    # 2. Sample 10 slices from validation data
    min_start_val = val_start_idx
    max_start_val = val_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

    if max_start_val >= min_start_val:
        val_range = max_start_val - min_start_val + 1
        val_segment_size = max(1, val_range // 10)

        for i in range(10):
            segment_start = min_start_val + (i * val_segment_size)
            segment_end = min(max_start_val + 1, segment_start + val_segment_size)

            if segment_end > segment_start:
                start_idx = np.random.randint(segment_start, segment_end)
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

                slices.append((start_idx, end_idx, trading_end_idx))

    return slices
