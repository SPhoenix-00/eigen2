"""
Committee utilities: constants, data helpers, and DRY helper functions.

This module contains all pure utility functions, constants, data loading helpers,
and shared helper functions that eliminate duplication across the committee package.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import gc

# Project Imports
from utils.config import Config
from data.loader import StockDataLoader
from models.ddpg_agent import DDPGAgent

# --- Constants ---
GLOBAL50_BASE_DIR = Path("global50")
COMMITTEE_DIR = Path("committee_results")  # Legacy local-only path
ROSTER_FILENAME = "committee_roster.json"
CORRELATION_FILENAME = "committee_correlation.png"


# --- Pure Utility Functions ---

def sanitize_date_for_filename(date_value) -> str:
    """
    Sanitize a date value for use in filenames.

    Converts dates to a safe filename format by:
    - Converting to string if needed
    - Replacing slashes with dashes
    - Replacing spaces with underscores
    - Removing any other invalid filename characters

    Args:
        date_value: Date value (string, datetime, Timestamp, etc.)

    Returns:
        Sanitized date string safe for filenames
    """
    # Convert to string representation
    if isinstance(date_value, str):
        date_str = date_value
    elif isinstance(date_value, (pd.Timestamp, datetime)):
        # Format as DD-MM-YY for consistency
        date_str = date_value.strftime('%d-%m-%y')
    else:
        # Try to convert to string
        date_str = str(date_value)

    # Replace slashes with dashes (common date format issue)
    date_str = date_str.replace('/', '-')
    # Replace spaces with underscores
    date_str = date_str.replace(' ', '_')
    # Remove any colons (from datetime strings)
    date_str = date_str.replace(':', '-')

    return date_str


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, numpy type, or primitive)

    Returns:
        Object with all numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def calculate_max_drawdown(cumulative_returns):
    """Calculates Maximum Drawdown from a cumulative return series."""
    if len(cumulative_returns) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_returns)
    peak = np.where(peak == 0, 1e-9, peak)
    drawdown = (cumulative_returns - peak) / peak
    return float(np.min(drawdown))


def format_quality_ratio(value) -> str:
    """
    Format a quality ratio value for display, handling inf/nan.

    Args:
        value: Quality ratio (float, possibly inf or nan)

    Returns:
        Formatted string
    """
    if np.isinf(value):
        return "inf"
    elif np.isnan(value):
        return "nan"
    else:
        return f"{value:.3f}"


# --- Date Parsing ---

def parse_date_input(date_str: str) -> datetime:
    """
    Parse user input date string in DD-MM-YY format.

    Args:
        date_str: Date string like '29-07-22'

    Returns:
        datetime object

    Raises:
        ValueError if format is invalid
    """
    try:
        return datetime.strptime(date_str, "%d-%m-%y")
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Expected DD-MM-YY (e.g., 29-07-22)")


def parse_date_flexible(date_value) -> datetime:
    """
    Parse date value in various formats (handles strings, datetime objects, Timestamps, etc.).

    Supports multiple date string formats:
    - M/D/YYYY (e.g., '10/7/2022')
    - M/D/YY (e.g., '10/7/22')
    - DD-MM-YY (e.g., '07-10-22')
    - YYYY-MM-DD (e.g., '2022-10-07')
    - DD/MM/YY (e.g., '07/10/22')
    - DD/MM/YYYY (e.g., '07/10/2022')

    Args:
        date_value: Date value (string, datetime, Timestamp, etc.)

    Returns:
        datetime object

    Raises:
        ValueError if date cannot be parsed
    """
    # If already a datetime object, return it
    if isinstance(date_value, datetime):
        return date_value

    # If pandas Timestamp, convert to datetime
    if isinstance(date_value, pd.Timestamp):
        return date_value.to_pydatetime()

    # Convert to string for parsing
    date_str = str(date_value).strip()

    # Remove time component if present
    if ' ' in date_str:
        date_str = date_str.split()[0]

    # Try various date formats
    formats = [
        "%m/%d/%Y",      # M/D/YYYY (e.g., '10/7/2022')
        "%m/%d/%y",      # M/D/YY (e.g., '10/7/22')
        "%d-%m-%y",      # DD-MM-YY (e.g., '07-10-22')
        "%Y-%m-%d",      # YYYY-MM-DD (e.g., '2022-10-07')
        "%d/%m/%y",      # DD/MM/YY (e.g., '07/10/22')
        "%d/%m/%Y",      # DD/MM/YYYY (e.g., '07/10/2022')
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse date: {date_value} (tried formats: {formats})")


def find_date_index(loader, target_date: datetime) -> int:
    """
    Find the index in loader.dates that matches or is closest to target_date.

    Args:
        loader: StockDataLoader with dates array
        target_date: Target datetime to find

    Returns:
        Index of matching or nearest date

    Raises:
        ValueError if date is out of range
    """
    # Convert loader dates to datetime objects for comparison
    loader_dates = []
    for d in loader.dates:
        if isinstance(d, str):
            loader_dates.append(datetime.strptime(d, "%d-%m-%y"))
        else:
            # numpy datetime64 or pandas Timestamp
            loader_dates.append(pd.Timestamp(d).to_pydatetime())

    # Check bounds
    if target_date < loader_dates[0]:
        raise ValueError(f"Date {target_date.strftime('%d-%m-%y')} is before dataset start ({loader_dates[0].strftime('%d-%m-%y')})")
    if target_date > loader_dates[-1]:
        raise ValueError(f"Date {target_date.strftime('%d-%m-%y')} is after dataset end ({loader_dates[-1].strftime('%d-%m-%y')})")

    # Find exact match or nearest
    for i, d in enumerate(loader_dates):
        if d >= target_date:
            return i

    return len(loader_dates) - 1


# --- Data Loading & Verification ---

def verify_data_split(loader):
    """
    Verifies THREE-TIER split: Training → Validation → Holdout

    Returns: (is_valid, error_message, holdout_info_dict)
    """
    total_days = len(loader.data_array_full)

    min_required = (Config.CONTEXT_WINDOW_DAYS + Config.VALIDATION_DAYS +
                    Config.COMMITTEE_HOLDOUT_DAYS + Config.MIN_HOLDING_PERIOD)
    if total_days < min_required:
        return False, f"Dataset too small: {total_days} days (need {min_required})", None

    expected_holdout_start = total_days - Config.COMMITTEE_HOLDOUT_DAYS
    holdout_end = total_days - 1

    if not hasattr(loader, 'train_end_idx') or loader.train_end_idx is None:
        return False, "Loader missing train_end_idx! Must call create_train_val_split() first.", None

    train_end = loader.train_end_idx
    train_start = 0
    train_size = train_end - train_start + 1

    if not hasattr(loader, 'val_end_idx') or loader.val_end_idx is None:
        return False, "Loader missing val_end_idx! Must call create_train_val_split() first.", None

    val_start = loader.val_start_idx
    val_end = loader.val_end_idx
    val_size = val_end - val_start + 1

    if val_end >= expected_holdout_start:
        return False, f"VALIDATION OVERLAPS WITH HOLDOUT! val_end={val_end}, holdout_start={expected_holdout_start}", None

    holdout_info = {
        'holdout_start': expected_holdout_start,
        'holdout_end': holdout_end,
        'train_start': train_start,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
    }

    print("✅ THREE-TIER DATA SPLIT VERIFICATION PASSED")
    print(f"\n  Total Days:     {total_days:,}")
    print(f"  Training:       {train_size:,} days (indices {train_start:,} to {train_end:,})")
    print(f"  Validation:     {val_size:,} days (indices {val_start:,} to {val_end:,})")
    print(f"  Holdout:        {Config.COMMITTEE_HOLDOUT_DAYS:,} days (indices {expected_holdout_start:,} to {holdout_end:,})")

    return True, None, holdout_info


def load_normalization_stats():
    """Calculates normalization stats deterministically."""
    print("Loading data to calculate normalization stats...")
    loader = StockDataLoader()
    _, stats = loader.load_and_prepare()
    return loader, stats


def load_agent_actor_only(filepath, agent_id):
    """Loads only the Actor network to save VRAM."""
    try:
        agent = DDPGAgent(agent_id=agent_id)
        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.actor.eval()
        del agent.critic
        del agent.critic_target
        del agent.actor_target
        return agent
    except Exception as e:
        print(f"⚠ Error loading {filepath}: {e}")
        return None


def get_holdout_data(loader, stats, holdout_info):
    """
    Get holdout data tensor for inference.

    Returns:
        (inputs_tensor, valid_indices)
    """
    holdout_start = holdout_info['holdout_start']
    holdout_end = holdout_info['holdout_end']

    input_data = loader.data_array  # [Days, Stocks, 5_feats]

    # Valid inference points (need room for MIN_HOLDING_PERIOD forward)
    valid_indices = list(range(holdout_start, holdout_end - Config.MIN_HOLDING_PERIOD + 1))

    inputs = []
    for i in valid_indices:
        window = input_data[i - Config.CONTEXT_WINDOW_DAYS : i]
        normalized = (window - stats['mean']) / stats['std']
        inputs.append(normalized)

    inputs = np.array(inputs)
    inputs_tensor = torch.FloatTensor(inputs).to(Config.DEVICE)

    return inputs_tensor, valid_indices


def get_validation_data(loader, stats, holdout_info):
    """
    Get VALIDATION data tensor for correlation calculation during Draft.

    IMPORTANT: This uses validation data (NOT holdout) to prevent data leakage.
    The holdout period must remain unseen until final validation.

    Returns:
        (inputs_tensor, valid_indices)
    """
    val_start = holdout_info['val_start']
    val_end = holdout_info['val_end']

    input_data = loader.data_array  # [Days, Stocks, 5_feats]

    # Valid inference points within validation period
    # Need context window before and room for MIN_HOLDING_PERIOD forward
    valid_indices = list(range(
        max(val_start, Config.CONTEXT_WINDOW_DAYS),
        val_end - Config.MIN_HOLDING_PERIOD + 1
    ))

    inputs = []
    for i in valid_indices:
        window = input_data[i - Config.CONTEXT_WINDOW_DAYS : i]
        normalized = (window - stats['mean']) / stats['std']
        inputs.append(normalized)

    inputs = np.array(inputs)
    inputs_tensor = torch.FloatTensor(inputs).to(Config.DEVICE)

    return inputs_tensor, valid_indices


# --- Global50 Loading ---

def load_global50_candidates(context_window_days: int) -> list:
    """
    Load agents from Global50, sorted by gauntlet_score descending.

    Args:
        context_window_days: Context window to load (e.g., 151)

    Returns:
        List of entry dicts with agent metadata
    """
    json_path = GLOBAL50_BASE_DIR / f"cw{context_window_days}" / "global50.json"

    if not json_path.exists():
        print(f"❌ Global50 ledger not found: {json_path}")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    entries = data.get('entries', [])

    # Sort by gauntlet_score descending
    entries.sort(key=lambda e: e.get('gauntlet_score', 0), reverse=True)

    print(f"✓ Loaded {len(entries)} agents from Global50 (cw{context_window_days})")

    return entries


def get_agent_filepath(entry: dict, context_window_days: int) -> Path:
    """Get the filepath for an agent's weights."""
    filename = f"{entry['run_name']}_{entry['agent_id']}.pth"
    return GLOBAL50_BASE_DIR / f"cw{context_window_days}" / "agents" / filename


# --- Expectancy ---

def calculate_expectancy(closed_trades):
    """
    Calculate Expectancy metric for trading performance.

    Expectancy = (Win Rate × Avg Win %) − (Loss Rate × Avg Loss %)

    Args:
        closed_trades: List of closed trade dictionaries with 'gain_pct' field

    Returns:
        Expectancy value (float)
    """
    if not closed_trades:
        return 0.0

    # Separate wins and losses
    wins = [t['gain_pct'] for t in closed_trades if t['gain_pct'] > 0]
    losses = [abs(t['gain_pct']) for t in closed_trades if t['gain_pct'] <= 0]

    if not wins and not losses:
        return 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    win_rate = len(wins) / len(closed_trades)
    loss_rate = 1.0 - win_rate

    # Expectancy = (Probability of Win * Reward) - (Probability of Loss * Risk)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    return expectancy


# --- DRY Helpers (eliminate duplicated patterns) ---

def generate_validation_slices(holdout_info, episode_length, num_val_slices=3, num_holdout_slices=2):
    """
    Generate deterministic, evenly-spaced validation slices.

    Replaces the duplicated slice-index calculation that appeared in
    run_validation, run_validation_sweep, and sweep functions.

    Args:
        holdout_info: Holdout period info dict
        episode_length: Length of each episode in days
        num_val_slices: Number of validation slices (default: 3)
        num_holdout_slices: Number of holdout slices (default: 2)

    Returns:
        List of dicts with keys: index, start, end, type
    """
    val_start_idx = holdout_info['val_start']
    val_end_idx = holdout_info['val_end']
    holdout_start_idx = holdout_info['holdout_start']
    holdout_end_idx = holdout_info['holdout_end']

    val_days = val_end_idx - val_start_idx + 1
    holdout_days = holdout_end_idx - holdout_start_idx + 1

    val_usable_range = val_days - episode_length
    val_step = val_usable_range // (num_val_slices - 1) if num_val_slices > 1 else 0

    holdout_usable_range = holdout_days - episode_length
    holdout_step = holdout_usable_range // (num_holdout_slices - 1) if num_holdout_slices > 1 else 0

    slices = []
    num_slices = num_val_slices + num_holdout_slices

    for s in range(num_slices):
        if s < num_val_slices:
            slice_start = val_start_idx + (s * val_step)
            slice_end = slice_start + episode_length
            slice_type = 'validation'
        else:
            holdout_slice_idx = s - num_val_slices
            slice_start = holdout_start_idx + (holdout_slice_idx * holdout_step)
            slice_end = slice_start + episode_length
            slice_type = 'holdout'

        slices.append({
            'index': s,
            'start': slice_start,
            'end': slice_end,  # Exclusive end
            'type': slice_type,
        })

    return slices


def aggregate_slice_metrics(committee_slices, all_closed_trades=None):
    """
    Aggregate slice metrics into committee-level summary.

    Replaces the 4× duplicated aggregation logic in run_validation,
    run_validation_sweep, and agent-level aggregation.

    Args:
        committee_slices: List of slice result dicts
        all_closed_trades: Optional list of all closed trades for expectancy calculation

    Returns:
        Dict with mean_fitness, mean_win_rate, mean_quality_ratio,
        mean_expectancy, mean_roi, total_trades
    """
    mean_fitness = float(np.mean([s['fitness'] for s in committee_slices]))

    total_wins = sum(s.get('num_wins', 0) for s in committee_slices)
    total_losses = sum(s.get('num_losses', 0) for s in committee_slices)
    total_trades = total_wins + total_losses

    mean_win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0
    mean_quality_ratio = (total_wins / total_losses) if total_losses > 0 else float('inf')

    if all_closed_trades is not None:
        mean_expectancy = float(calculate_expectancy(all_closed_trades))
    else:
        mean_expectancy = float(np.mean([s.get('expectancy', 0.0) for s in committee_slices]))

    total_raw_pnl = sum(s.get('raw_pnl', 0.0) for s in committee_slices)
    total_peak_capital = sum(s.get('peak_capital_employed', 0.0) for s in committee_slices)
    mean_roi = (total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0

    return {
        'mean_fitness': mean_fitness,
        'mean_win_rate': mean_win_rate,
        'mean_quality_ratio': mean_quality_ratio,
        'mean_expectancy': mean_expectancy,
        'mean_roi': mean_roi,
        'total_trades': int(total_trades),
    }


def aggregate_consensus_stats(all_consensus_stats):
    """
    Aggregate consensus stats from multiple slices.

    Replaces the 2× duplicated consensus aggregation logic.

    Args:
        all_consensus_stats: List of consensus stat dicts from individual slices

    Returns:
        Dict with aggregated consensus metrics
    """
    if all(isinstance(cs, dict) and 'avg_consensus_votes' in cs for cs in all_consensus_stats):
        return {
            'avg_unanimity_pct': float(np.mean([cs['unanimity_pct'] for cs in all_consensus_stats])),
            'avg_min_consensus_pct': float(np.mean([cs['min_consensus_pct'] for cs in all_consensus_stats])),
            'avg_consensus_votes': float(np.mean([cs['avg_consensus_votes'] for cs in all_consensus_stats])),
            'total_trades_by_quorum': int(sum(cs['trades_by_quorum'] for cs in all_consensus_stats)),
            'total_trades_by_conviction': int(sum(cs['trades_by_conviction'] for cs in all_consensus_stats)),
            'total_trades_vetoed': int(sum(cs['trades_vetoed'] for cs in all_consensus_stats)),
        }
    else:
        return {'note': 'Consensus tracking was not enabled or no data available'}


def build_metrics_result(summary, consensus_stats=None, extra_fields=None):
    """
    Build standardized metrics result dict from a TradingEnvironment episode summary.

    Replaces the 3× duplicated result dict construction in evaluate_agent_on_slice,
    evaluate_committee_on_slice, and simulate_committee_continuous.

    Args:
        summary: Episode summary from TradingEnvironment.get_episode_summary()
        consensus_stats: Optional consensus stats dict
        extra_fields: Optional dict of additional fields to include

    Returns:
        Standardized metrics dict
    """
    result = {
        'num_trades': summary['num_trades'],
        'num_wins': summary['num_wins'],
        'num_losses': summary['num_losses'],
        'win_rate': summary['win_rate'],
        'quality_ratio': (summary['num_wins'] / summary['num_losses']) if summary['num_losses'] > 0 else float('inf'),
        'expectancy': summary['avg_reward_per_trade'],
        'roi': summary['roi'],
        'fitness': summary['total_reward'],
        'raw_pnl': summary.get('raw_pnl', 0.0),
        'peak_capital_employed': summary.get('peak_capital_employed', 0.0),
        'closed_trades': summary.get('closed_trades', []),
    }
    if consensus_stats is not None:
        result['consensus_stats'] = consensus_stats
    if extra_fields:
        result.update(extra_fields)
    return result


def build_member_data(entry, conviction_threshold_vector):
    """
    Build standardized member dict for roster.

    Replaces the 2× duplicated member dict construction in run_draft and run_swap_agent.

    Args:
        entry: Global50 entry dict
        conviction_threshold_vector: Numpy array or list of conviction thresholds

    Returns:
        Member dict suitable for roster storage
    """
    return {
        'filename': f"{entry['run_name']}_{entry['agent_id']}.pth",
        'agent_id': entry['agent_id'],
        'run_name': entry['run_name'],
        'gauntlet_score': entry['gauntlet_score'],
        'roi': entry.get('roi', 0.0),
        'expectancy': entry.get('expectancy', 0.0),
        'quality_ratio': entry.get('quality_ratio', 0.0),
        'win_ratio': entry.get('win_ratio', 0.0),
        'is_maverick': entry.get('is_maverick', False),
        'stats': {
            'conviction_threshold_vector': (
                conviction_threshold_vector.tolist()
                if hasattr(conviction_threshold_vector, 'tolist')
                else conviction_threshold_vector
            )
        }
    }


def enrich_closed_trades(closed_trades, loader):
    """
    Add stock_name and exit_date to closed trades.

    Replaces the 2× duplicated trade enrichment in evaluate_committee_on_slice
    and simulate_committee_continuous.

    Args:
        closed_trades: List of closed trade dicts
        loader: StockDataLoader with column_names

    Returns:
        The same list, mutated with stock_name and exit_date fields
    """
    for trade in closed_trades:
        stock_id = trade.get('stock_id')
        if stock_id is not None and loader.column_names:
            actual_col_idx = Config.INVESTABLE_START_COL + stock_id
            if actual_col_idx < len(loader.column_names):
                trade['stock_name'] = loader.column_names[actual_col_idx]
            else:
                trade['stock_name'] = f"UNKNOWN_{stock_id}"
        else:
            trade['stock_name'] = f"UNKNOWN_{stock_id}"
        trade['exit_date'] = trade.get('day', '')
    return closed_trades

