"""
Fitness calculation functions for Project Eigen 2.

Pure/near-pure scoring functions extracted from ERLTrainer.
These compute fitness scores, expectancy, aggregation, and hashing
without requiring trainer state — all dependencies are passed as parameters.
"""

import json
import math
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def calculate_triad_fitness(
    stats: Dict,
    maverick_mode: bool,
    consistency_mode: bool,
    quality_threshold: float,
    roi_hurdle_pct: float,
    breakthrough_state,
    global_hof=None,
) -> float:
    """
    Triad 3.0: Maverick Selectivity & Gradient Update.

    Args:
        stats: Episode statistics dict from environment
        maverick_mode: Whether maverick scoring is active
        consistency_mode: Whether consistency mode is active
        quality_threshold: Minimum gain_pct to count as a quality trade
        roi_hurdle_pct: ROI hurdle EMA (unused in current logic but kept for API)
        breakthrough_state: BreakthroughState enum value
        global_hof: Optional GlobalHallOfFame for proximity gradient

    Returns:
        Fitness score (float)
    """
    from utils.config import Config
    from training.breakthrough import BreakthroughState

    total_trades = stats.get('num_trades', 0)

    # 1. Handle Inactivity
    if total_trades == 0:
        if maverick_mode:
            return -50.0 + stats.get('max_coefficient_during_episode', 0)

        # In consistency mode during stabilization/gauntlet, use soft penalty
        if consistency_mode and breakthrough_state in (BreakthroughState.STABILIZATION, BreakthroughState.GAUNTLET):
            penalty = Config.ZERO_TRADES_PENALTY_GAUNTLET
        elif consistency_mode:
            penalty = Config.ZERO_TRADES_PENALTY_CONSISTENCY
        else:
            penalty = Config.ZERO_TRADES_PENALTY_NORMAL
        return -penalty + stats.get('max_coefficient_during_episode', 0)

    # 2. Calculate Core Metrics
    win_rate = stats.get('win_rate', 0.0)

    # Calculate Quality Ratio (QR)
    closed_trades = stats.get('closed_trades', [])
    if closed_trades:
        q_thresh = quality_threshold if quality_threshold is not None else 1.0
        quality_count = sum(1 for t in closed_trades if t.get('gain_pct', 0) >= q_thresh)
        qr = quality_count / total_trades
    else:
        qr = 0.0

    raw_pnl = stats.get('raw_pnl', 0.0)
    total_inv = stats.get('total_investment', 0.0)

    # Calculate Expectancy
    if closed_trades:
        wins = [t['gain_pct'] for t in closed_trades if t['gain_pct'] > 0]
        losses = [abs(t['gain_pct']) for t in closed_trades if t['gain_pct'] <= 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        wr_calc = len(wins) / len(closed_trades)
        lr_calc = 1.0 - wr_calc
        expectancy = (wr_calc * avg_win) - (lr_calc * avg_loss)
    else:
        expectancy = 0.0

    # --- MAVERICK MODE: Triad 3.0 scoring with Gradient ---
    if maverick_mode:
        peak_capital = stats.get('peak_capital_employed', total_inv)
        if peak_capital <= 0:
            peak_capital = total_inv if total_inv > 0 else 1.0

        roi_pct = (raw_pnl / peak_capital * 100) if peak_capital > 0 else 0.0

        # A. ROI Score (Power Law)
        if roi_pct >= 0:
            roi_score = (roi_pct ** 1.1)
        else:
            roi_score = -(abs(roi_pct) ** 1.1)

        # B. CLAMPED Volume Scalar (20k Limit)
        raw_vol_scalar = math.log10(peak_capital + 10)
        volume_scalar = min(raw_vol_scalar, 4.3)

        # C. Expectancy (Precision Reward)
        if expectancy >= 0:
            exp_score = (expectancy ** 1.1)
        else:
            exp_score = -(abs(expectancy) ** 1.1)

        # D. Base Fitness
        if roi_score > 0:
            fitness = roi_score * volume_scalar * (1.0 + (exp_score * 0.1))
            fitness *= (win_rate ** 2)
            if win_rate < 0.60:
                wr_deficit = win_rate - 0.60
                penalty = wr_deficit * 1000.0
                fitness += penalty
        else:
            fitness = roi_score * volume_scalar + (exp_score * volume_scalar)

        # E. FOMO Penalty (Dampened)
        market_return = stats.get('market_return_pct', 0.0)
        alpha_gap = market_return - roi_pct
        fomo_penalty = max(0.0, alpha_gap) * 5.0
        fitness -= fomo_penalty

        # F. Global 50 Proximity Gradient
        if global_hof is not None and global_hof.enabled and global_hof.entry_threshold > -999:
            t_score = [global_hof.entry_threshold, global_hof.gauntlet_p25, global_hof.gauntlet_median]
            t_roi = [global_hof.roi_threshold, global_hof.roi_p25, global_hof.roi_median]
            t_exp = [global_hof.expectancy_threshold, global_hof.expectancy_p25, global_hof.expectancy_median]

            curr_score = fitness
            curr_roi = roi_pct
            curr_exp = expectancy

            def calc_gap(target, current):
                return max(0.0, target - current)

            gap_score = (calc_gap(t_score[0], curr_score) * 3.0 + calc_gap(t_score[1], curr_score) * 1.5 + calc_gap(t_score[2], curr_score))
            gap_roi = (calc_gap(t_roi[0], curr_roi) * 3.0 + calc_gap(t_roi[1], curr_roi) * 1.5 + calc_gap(t_roi[2], curr_roi))
            gap_exp = (calc_gap(t_exp[0], curr_exp) * 3.0 + calc_gap(t_exp[1], curr_exp) * 1.5 + calc_gap(t_exp[2], curr_exp))

            proximity_penalty = (gap_score * 0.5) + (gap_roi * 1.0) + (gap_exp * 1.0)
            fitness -= proximity_penalty

        return float(fitness)

    # --- NORMAL/CONSISTENCY MODE: Standard Triad scoring ---
    roi_pct = (raw_pnl / total_inv * 100) if total_inv > 0 else 0.0

    if roi_pct >= 0:
        roi_score = (roi_pct ** 1.1)
    else:
        roi_score = -(abs(roi_pct) ** 1.1)

    volume_scalar = math.log10(abs(raw_pnl) + 10)

    if roi_score > 0:
        base_score = roi_score * volume_scalar
        consistency_bonus = (1.0 + (win_rate ** 2))
        conviction_bonus = (1.0 + qr)
        fitness = base_score * consistency_bonus * conviction_bonus * 10.0
    else:
        fitness = roi_score * volume_scalar * 10.0

    return float(fitness)


def calculate_holographic_fitness(
    all_slices_trades: List[Dict],
    maverick_mode: bool,
    global_hof=None,
) -> float:
    """
    HOLOGRAPHIC SCORING:
    Stitches trades from all slices into a single 'Virtual Equity Curve'.

    Args:
        all_slices_trades: List of trade dicts from all validation slices
        maverick_mode: Whether maverick scoring is active
        global_hof: Optional GlobalHallOfFame for proximity gradient

    Returns:
        Holographic fitness score (float)
    """
    from datetime import datetime

    # 1. Safety Checks
    if not all_slices_trades:
        return -5000.0 if maverick_mode else -10.0

    total_trades = len(all_slices_trades)

    # 2. Sort trades chronologically
    try:
        def get_sort_key(t):
            d_str = t.get('entry_date') or t.get('day') or t.get('exit_date')
            if isinstance(d_str, str) and d_str:
                try:
                    return datetime.strptime(d_str, '%Y-%m-%d')
                except ValueError:
                    pass
            return datetime.min
        all_slices_trades.sort(key=get_sort_key)
    except Exception:
        pass

    # 3. Calculate Virtual Equity Curve & Stats
    virtual_equity = 1.0
    peak_equity = 1.0
    max_drawdown = 0.0

    wins = 0
    winning_pnl = []
    losing_pnl = []

    for trade in all_slices_trades:
        shares = int(trade.get('coefficient', 0))
        if shares == 0:
            continue

        entry_price = trade.get('entry_price', 0.0)
        exit_price = trade.get('exit_price', 0.0)

        if entry_price > 0:
            raw_roi = (exit_price - entry_price) / entry_price
            virtual_equity *= (1.0 + (raw_roi * 0.1))

            if virtual_equity > peak_equity:
                peak_equity = virtual_equity

            current_dd = (peak_equity - virtual_equity) / peak_equity
            if current_dd > max_drawdown:
                max_drawdown = current_dd

            if raw_roi > 0:
                wins += 1
                winning_pnl.append(raw_roi)
            else:
                losing_pnl.append(abs(raw_roi))

    # 4. Calculate Final Metrics
    holographic_roi = (virtual_equity - 1.0) * 100.0
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    avg_win = np.mean(winning_pnl) * 100.0 if winning_pnl else 0.0
    avg_loss = np.mean(losing_pnl) * 100.0 if losing_pnl else 0.0
    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

    # 5. Maverick Scoring (Triad 3.0)
    if maverick_mode:
        if holographic_roi >= 0:
            roi_score = (holographic_roi ** 1.1)
        else:
            roi_score = -(abs(holographic_roi) ** 1.1)

        if expectancy >= 0:
            exp_score = (expectancy ** 1.1)
        else:
            exp_score = -(abs(expectancy) ** 1.1)

        total_raw_pnl = sum(
            (t.get('exit_price', 0) - t.get('entry_price', 0)) * int(t.get('coefficient', 0))
            for t in all_slices_trades
        )
        volume_proxy = math.log10(abs(total_raw_pnl) + 10)
        volume_scalar = min(volume_proxy, 4.3)

        if roi_score > 0:
            fitness = roi_score * volume_scalar * (1.0 + (exp_score * 0.1))
            fitness *= (win_rate ** 2)
            if win_rate < 0.60:
                wr_deficit = win_rate - 0.60
                penalty = wr_deficit * 1000.0
                fitness += penalty
        else:
            fitness = roi_score * volume_scalar + (exp_score * volume_scalar)

        if max_drawdown > 0.05:
            penalty_factor = max(0.1, 1.0 - (max_drawdown - 0.05) * 10.0)
            fitness *= penalty_factor

        if global_hof is not None and global_hof.enabled and global_hof.entry_threshold > -999:
            t_score = [global_hof.entry_threshold, global_hof.gauntlet_p25, global_hof.gauntlet_median]
            curr_score = fitness

            def calc_gap(target, current):
                return max(0.0, target - current)

            gap_score = (calc_gap(t_score[0], curr_score) * 3.0 + calc_gap(t_score[1], curr_score) * 1.5 + calc_gap(t_score[2], curr_score))
            fitness -= gap_score * 0.5

        return float(fitness)

    return float(holographic_roi)


def calculate_pessimistic_fitness(slice_fitness_scores: List[float]) -> float:
    """
    Calculate fitness using pessimistic aggregator (0.4*mean + 0.6*min).

    Args:
        slice_fitness_scores: List of fitness scores from multiple evaluation slices

    Returns:
        Pessimistically aggregated fitness score
    """
    mean_score = np.mean(slice_fitness_scores)
    min_score = np.min(slice_fitness_scores)
    return (0.4 * mean_score) + (0.6 * min_score)


def calculate_penalized_median_fitness(slice_fitness_scores: List[float]) -> float:
    """
    Calculate fitness using Penalized Median scoring: Median - (0.5 * StdDev).

    Args:
        slice_fitness_scores: List of fitness scores from multiple evaluation slices

    Returns:
        Penalized median fitness score
    """
    scores_np = np.array(slice_fitness_scores)
    median_score = float(np.median(scores_np))
    std_score = float(np.std(scores_np))
    return median_score - (0.5 * std_score)


def aggregate_agent_stats(slice_episode_stats: List[Dict]) -> Dict:
    """
    Aggregate episode statistics across all training slices for a single agent.

    Args:
        slice_episode_stats: List of episode info dicts from multiple slices

    Returns:
        Aggregated stats dict
    """
    agent_total_wins = sum(s['num_wins'] for s in slice_episode_stats)
    agent_total_losses = sum(s['num_losses'] for s in slice_episode_stats)
    agent_total_trades = agent_total_wins + agent_total_losses
    agent_win_rate = agent_total_wins / agent_total_trades if agent_total_trades > 0 else 0.0

    return {
        'num_trades': int(np.mean([s['num_trades'] for s in slice_episode_stats])),
        'num_wins': int(np.mean([s['num_wins'] for s in slice_episode_stats])),
        'num_losses': int(np.mean([s['num_losses'] for s in slice_episode_stats])),
        'win_rate': agent_win_rate,
    }


def aggregate_population_stats(all_episode_stats: List[Dict], fitness_scores: List[float]) -> Dict:
    """
    Aggregate statistics across all agents in the population.

    Args:
        all_episode_stats: List of per-agent aggregated stats dicts
        fitness_scores: List of fitness scores for all agents

    Returns:
        Population-level aggregate stats dict
    """
    agents_with_trades = [s for s in all_episode_stats if s['num_trades'] > 0]
    avg_win_rate = (
        float(sum(s['win_rate'] for s in agents_with_trades) / len(agents_with_trades))
        if agents_with_trades else 0.0
    )

    return {
        'total_trades': int(sum(s['num_trades'] for s in all_episode_stats)),
        'avg_trades_per_agent': float(sum(s['num_trades'] for s in all_episode_stats) / len(all_episode_stats)),
        'total_wins': int(sum(s['num_wins'] for s in all_episode_stats)),
        'total_losses': int(sum(s['num_losses'] for s in all_episode_stats)),
        'avg_win_rate': avg_win_rate,
        'agents_with_positive_fitness': int(sum(1 for f in fitness_scores if f > 0)),
    }


def calculate_expectancy(closed_trades: list) -> float:
    """
    Calculate Expectancy metric for trading performance.

    Expectancy = (Win Rate x Avg Win %) - (Loss Rate x Avg Loss %)

    Args:
        closed_trades: List of closed trade dicts with 'gain_pct' field

    Returns:
        Expectancy value (float)
    """
    if not closed_trades:
        return 0.0

    wins = [t['gain_pct'] for t in closed_trades if t['gain_pct'] > 0]
    losses = [abs(t['gain_pct']) for t in closed_trades if t['gain_pct'] <= 0]

    if not wins and not losses:
        return 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    win_rate = len(wins) / len(closed_trades)
    loss_rate = 1.0 - win_rate

    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    return expectancy


def hash_agent(agent) -> str:
    """
    Create hash of agent's weights for caching.

    Args:
        agent: DDPGAgent to hash

    Returns:
        MD5 hash string
    """
    actor_weights = agent.actor.state_dict()
    first_layer = list(actor_weights.values())[0].cpu().numpy()
    return hashlib.md5(first_layer.tobytes()).hexdigest()


def hash_validation_slices(slices: List[Tuple[int, int, int]]) -> str:
    """
    Hash validation slices configuration.

    Args:
        slices: List of (start_idx, end_idx, trading_end_idx) tuples

    Returns:
        MD5 hash string
    """
    return hashlib.md5(str(slices).encode()).hexdigest()
