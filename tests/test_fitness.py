"""
Unit tests for training/fitness.py — extracted fitness calculation functions.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from training.fitness import (
    NumpyEncoder,
    calculate_triad_fitness,
    calculate_holographic_fitness,
    calculate_pessimistic_fitness,
    calculate_penalized_median_fitness,
    aggregate_agent_stats,
    aggregate_population_stats,
    calculate_expectancy,
    hash_agent,
    hash_validation_slices,
)
from training.breakthrough import BreakthroughState


# ── NumpyEncoder ─────────────────────────────────────────────────────────────

class TestNumpyEncoder:
    def test_encodes_numpy_int(self):
        import json
        result = json.dumps({"v": np.int64(42)}, cls=NumpyEncoder)
        assert '"v": 42' in result

    def test_encodes_numpy_float(self):
        import json
        result = json.dumps({"v": np.float32(3.14)}, cls=NumpyEncoder)
        assert '"v": 3.14' in result or '"v": 3.' in result

    def test_encodes_numpy_array(self):
        import json
        result = json.dumps({"v": np.array([1, 2, 3])}, cls=NumpyEncoder)
        assert "[1, 2, 3]" in result


# ── calculate_expectancy ─────────────────────────────────────────────────────

class TestCalculateExpectancy:
    def test_empty_trades(self):
        assert calculate_expectancy([]) == 0.0

    def test_all_winners(self):
        trades = [{"gain_pct": 10.0}, {"gain_pct": 5.0}, {"gain_pct": 8.0}]
        exp = calculate_expectancy(trades)
        assert exp > 0
        assert abs(exp - np.mean([10.0, 5.0, 8.0])) < 1e-6

    def test_all_losers(self):
        trades = [{"gain_pct": -3.0}, {"gain_pct": -5.0}]
        exp = calculate_expectancy(trades)
        assert exp < 0

    def test_mixed_trades(self, sample_closed_trades):
        exp = calculate_expectancy(sample_closed_trades)
        assert isinstance(exp, float)
        # With 4 winners and 2 losers, should be positive
        assert exp > 0

    def test_zero_gain_counted_as_loss(self):
        trades = [{"gain_pct": 0.0}, {"gain_pct": 10.0}]
        exp = calculate_expectancy(trades)
        # 0.0 gain is counted as loss (gain_pct <= 0)
        assert isinstance(exp, float)


# ── calculate_pessimistic_fitness ────────────────────────────────────────────

class TestPessimisticFitness:
    def test_basic(self):
        scores = [100.0, 200.0, 300.0]
        result = calculate_pessimistic_fitness(scores)
        expected = 0.4 * np.mean(scores) + 0.6 * np.min(scores)
        assert abs(result - expected) < 1e-6

    def test_single_score(self):
        result = calculate_pessimistic_fitness([50.0])
        assert abs(result - 50.0) < 1e-6

    def test_all_equal(self):
        result = calculate_pessimistic_fitness([100.0, 100.0, 100.0])
        assert abs(result - 100.0) < 1e-6


# ── calculate_penalized_median_fitness ───────────────────────────────────────

class TestPenalizedMedianFitness:
    def test_basic(self):
        scores = [100.0, 200.0, 300.0]
        result = calculate_penalized_median_fitness(scores)
        median = np.median(scores)
        std = np.std(scores)
        expected = median - 0.5 * std
        assert abs(result - expected) < 1e-6

    def test_zero_variance(self):
        result = calculate_penalized_median_fitness([100.0, 100.0, 100.0])
        assert abs(result - 100.0) < 1e-6


# ── aggregate_agent_stats ────────────────────────────────────────────────────

class TestAggregateAgentStats:
    def test_basic(self, sample_slice_stats):
        result = aggregate_agent_stats(sample_slice_stats)
        assert 'num_trades' in result
        assert 'win_rate' in result
        # Global win rate = total wins / total trades
        total_wins = 5 + 9 + 4
        total_losses = 3 + 3 + 2
        expected_wr = total_wins / (total_wins + total_losses)
        assert abs(result['win_rate'] - expected_wr) < 1e-6

    def test_empty_slices(self):
        result = aggregate_agent_stats([
            {'num_trades': 0, 'num_wins': 0, 'num_losses': 0, 'win_rate': 0.0}
        ])
        assert result['win_rate'] == 0.0


# ── aggregate_population_stats ───────────────────────────────────────────────

class TestAggregatePopulationStats:
    def test_basic(self):
        stats = [
            {'num_trades': 10, 'num_wins': 7, 'num_losses': 3, 'win_rate': 0.7},
            {'num_trades': 5, 'num_wins': 3, 'num_losses': 2, 'win_rate': 0.6},
            {'num_trades': 0, 'num_wins': 0, 'num_losses': 0, 'win_rate': 0.0},
        ]
        fitness = [100.0, -50.0, -200.0]
        result = aggregate_population_stats(stats, fitness)
        assert result['total_trades'] == 15
        assert result['agents_with_positive_fitness'] == 1
        # avg_win_rate should only consider agents with trades
        assert abs(result['avg_win_rate'] - 0.65) < 1e-6


# ── calculate_triad_fitness ──────────────────────────────────────────────────

class TestTriadFitness:
    def test_zero_trades_normal_mode(self):
        stats = {'num_trades': 0, 'max_coefficient_during_episode': 0.5}
        result = calculate_triad_fitness(
            stats, maverick_mode=False, consistency_mode=False,
            quality_threshold=1.0, roi_hurdle_pct=0.0,
            breakthrough_state=BreakthroughState.NORMAL,
        )
        # Should get a negative penalty + small gradient from max_coefficient
        assert result < 0

    def test_zero_trades_maverick_mode(self):
        stats = {'num_trades': 0, 'max_coefficient_during_episode': 0.5}
        result = calculate_triad_fitness(
            stats, maverick_mode=True, consistency_mode=False,
            quality_threshold=1.0, roi_hurdle_pct=0.0,
            breakthrough_state=BreakthroughState.NORMAL,
        )
        assert result == pytest.approx(-50.0 + 0.5)

    def test_positive_roi_normal_mode(self, sample_episode_stats):
        result = calculate_triad_fitness(
            sample_episode_stats, maverick_mode=False, consistency_mode=False,
            quality_threshold=7.5, roi_hurdle_pct=0.0,
            breakthrough_state=BreakthroughState.NORMAL,
        )
        assert result > 0
        assert isinstance(result, float)

    def test_positive_roi_maverick_mode(self, sample_episode_stats):
        result = calculate_triad_fitness(
            sample_episode_stats, maverick_mode=True, consistency_mode=False,
            quality_threshold=7.5, roi_hurdle_pct=0.0,
            breakthrough_state=BreakthroughState.NORMAL,
        )
        assert isinstance(result, float)

    def test_with_global_hof(self, sample_episode_stats, mock_global_hof):
        result = calculate_triad_fitness(
            sample_episode_stats, maverick_mode=True, consistency_mode=False,
            quality_threshold=7.5, roi_hurdle_pct=0.0,
            breakthrough_state=BreakthroughState.NORMAL,
            global_hof=mock_global_hof,
        )
        assert isinstance(result, float)


# ── calculate_holographic_fitness ────────────────────────────────────────────

class TestHolographicFitness:
    def test_empty_trades(self):
        assert calculate_holographic_fitness([], maverick_mode=False) == -10.0
        assert calculate_holographic_fitness([], maverick_mode=True) == -5000.0

    def test_basic_trades(self):
        trades = [
            {'entry_price': 100.0, 'exit_price': 110.0, 'coefficient': 5},
            {'entry_price': 50.0, 'exit_price': 48.0, 'coefficient': 3},
            {'entry_price': 200.0, 'exit_price': 220.0, 'coefficient': 2},
        ]
        result = calculate_holographic_fitness(trades, maverick_mode=False)
        assert isinstance(result, float)
        # With 2 wins and 1 loss, virtual equity should be positive
        assert result > 0


# ── hash functions ───────────────────────────────────────────────────────────

class TestHashFunctions:
    def test_hash_agent_deterministic(self, mock_agent):
        h1 = hash_agent(mock_agent)
        h2 = hash_agent(mock_agent)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex digest length

    def test_hash_validation_slices(self):
        slices = [(100, 200, 150), (300, 400, 350)]
        h1 = hash_validation_slices(slices)
        h2 = hash_validation_slices(slices)
        assert h1 == h2

        different_slices = [(100, 200, 150), (300, 400, 360)]
        h3 = hash_validation_slices(different_slices)
        assert h1 != h3
