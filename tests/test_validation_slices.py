"""
Unit tests for training/validation_slices.py — slice generation functions.
"""

import numpy as np
import pytest
from unittest.mock import patch

from training.validation_slices import (
    generate_validation_slices,
    generate_gauntlet_slices,
)


class TestGenerateValidationSlices:
    def test_produces_10_slices(self):
        np.random.seed(42)
        slices = generate_validation_slices(val_start_idx=1000, val_end_idx=2000)
        assert len(slices) == 10

    def test_slice_format(self):
        np.random.seed(42)
        slices = generate_validation_slices(val_start_idx=1000, val_end_idx=2000)
        for start, end, trading_end in slices:
            assert isinstance(start, (int, np.integer))
            assert isinstance(end, (int, np.integer))
            assert isinstance(trading_end, (int, np.integer))
            assert trading_end > start
            assert end > trading_end

    def test_slices_within_bounds(self):
        np.random.seed(42)
        slices = generate_validation_slices(val_start_idx=1000, val_end_idx=2000)
        for start, end, trading_end in slices:
            assert start >= 1000

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Not enough validation data"):
            generate_validation_slices(val_start_idx=1000, val_end_idx=1010)

    def test_deterministic_with_seed(self):
        np.random.seed(42)
        slices1 = generate_validation_slices(val_start_idx=1000, val_end_idx=2000)
        np.random.seed(42)
        slices2 = generate_validation_slices(val_start_idx=1000, val_end_idx=2000)
        assert slices1 == slices2


class TestGenerateGauntletSlices:
    def test_produces_20_slices(self):
        np.random.seed(42)
        slices = generate_gauntlet_slices(
            train_start_idx=0, train_end_idx=3000,
            val_start_idx=3000, val_end_idx=4000,
        )
        assert len(slices) == 20

    def test_slice_format(self):
        np.random.seed(42)
        slices = generate_gauntlet_slices(
            train_start_idx=0, train_end_idx=3000,
            val_start_idx=3000, val_end_idx=4000,
        )
        for start, end, trading_end in slices:
            assert isinstance(start, (int, np.integer))
            assert end > trading_end
            assert trading_end > start

    def test_spans_both_train_and_val(self):
        np.random.seed(42)
        slices = generate_gauntlet_slices(
            train_start_idx=0, train_end_idx=3000,
            val_start_idx=3000, val_end_idx=4000,
        )
        train_slices = [s for s in slices if s[0] < 3000]
        val_slices = [s for s in slices if s[0] >= 3000]
        assert len(train_slices) > 0
        assert len(val_slices) > 0
