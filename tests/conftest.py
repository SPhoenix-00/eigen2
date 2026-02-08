"""
Shared pytest fixtures for eigen2 test suite.
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def mock_agent():
    """Create a lightweight DDPGAgent on CPU for testing."""
    from models.ddpg_agent import DDPGAgent
    agent = DDPGAgent(agent_id=0)
    return agent


@pytest.fixture
def sample_closed_trades():
    """Sample closed trades for fitness/expectancy testing."""
    return [
        {'gain_pct': 12.5, 'entry_price': 100.0, 'exit_price': 112.5, 'coefficient': 5},
        {'gain_pct': -3.2, 'entry_price': 50.0, 'exit_price': 48.4, 'coefficient': 3},
        {'gain_pct': 8.0, 'entry_price': 200.0, 'exit_price': 216.0, 'coefficient': 2},
        {'gain_pct': -1.5, 'entry_price': 75.0, 'exit_price': 73.875, 'coefficient': 4},
        {'gain_pct': 15.0, 'entry_price': 120.0, 'exit_price': 138.0, 'coefficient': 6},
        {'gain_pct': 5.0, 'entry_price': 90.0, 'exit_price': 94.5, 'coefficient': 3},
    ]


@pytest.fixture
def sample_episode_stats():
    """Sample episode stats dict mimicking env.get_episode_summary() output."""
    return {
        'num_trades': 10,
        'num_wins': 7,
        'num_losses': 3,
        'win_rate': 0.7,
        'raw_pnl': 500.0,
        'total_investment': 5000.0,
        'peak_capital_employed': 3000.0,
        'market_return_pct': 5.0,
        'max_coefficient_during_episode': 8.0,
        'closed_trades': [
            {'gain_pct': 12.5},
            {'gain_pct': -3.2},
            {'gain_pct': 8.0},
            {'gain_pct': -1.5},
            {'gain_pct': 15.0},
            {'gain_pct': 5.0},
            {'gain_pct': 10.0},
            {'gain_pct': -2.0},
            {'gain_pct': 7.5},
            {'gain_pct': 3.0},
        ],
        'zero_trades_penalty': 0.0,
        'avg_reward_per_trade': 50.0,
        'steps': 155,
    }


@pytest.fixture
def sample_slice_stats():
    """Sample per-slice episode stats for aggregation testing."""
    return [
        {'num_trades': 8, 'num_wins': 5, 'num_losses': 3, 'win_rate': 0.625,
         'avg_reward_per_trade': 40.0},
        {'num_trades': 12, 'num_wins': 9, 'num_losses': 3, 'win_rate': 0.75,
         'avg_reward_per_trade': 60.0},
        {'num_trades': 6, 'num_wins': 4, 'num_losses': 2, 'win_rate': 0.667,
         'avg_reward_per_trade': 35.0},
    ]


@pytest.fixture
def mock_global_hof():
    """Mock GlobalHallOfFame with standard thresholds."""
    hof = MagicMock()
    hof.enabled = True
    hof.entry_threshold = 100.0
    hof.gauntlet_p25 = 200.0
    hof.gauntlet_median = 300.0
    hof.roi_threshold = 5.0
    hof.roi_p25 = 10.0
    hof.roi_median = 15.0
    hof.expectancy_threshold = 1.0
    hof.expectancy_p25 = 2.0
    hof.expectancy_median = 3.0
    return hof
