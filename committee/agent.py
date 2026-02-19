"""
Committee Agent: inference engine and conviction threshold calculation.

CommitteeAgent acts as a drop-in replacement for a single agent, implementing
the same interface (predict action from observation) but using committee
consensus logic internally.

Consensus rules:
  1. Quorum: >= COMMITTEE_QUORUM agents with coeff >= COEFFICIENT_THRESHOLD
  2. Conviction bypass: any single agent with coeff > its own conviction threshold
  3. Trade coefficient: max(supporters' coefficients), floored at COEFFICIENT_THRESHOLD
  4. Supporter = any agent with coeff >= COEFFICIENT_THRESHOLD or with conviction
"""

import copy
import numpy as np
import torch
import gc
from tqdm import tqdm

from utils.config import Config
from committee.utils import (
    get_agent_filepath, load_agent_actor_only, get_validation_data,
)


class CommitteeAgent:
    """
    Committee-as-Agent: Aggregates multiple agents' decisions into a single action vector.

    This class acts as a drop-in replacement for a single agent, implementing the same
    interface (predict action from observation) but using committee consensus logic internally.
    """

    def __init__(self, members: list, context_window_days: int, track_consensus: bool = False):
        """
        Initialize committee agent.

        Args:
            members: List of member dicts with agent metadata and stats
            context_window_days: Context window for loading agent files
            track_consensus: If True, track consensus statistics per step
        """
        self.members = members
        self.context_window_days = context_window_days
        self.loaded_agents = []
        self.track_consensus = track_consensus

        # Track which stocks were opened by which mechanism (persists across steps)
        self.quorum_opened = set()
        self.conviction_opened = set()

        if self.track_consensus:
            self.consensus_history = {
                'total_steps': 0,
                'total_trades': 0,
                'unanimity_count': 0,
                'trades_by_quorum': 0,
                'trades_by_conviction_only': 0,
                'avg_votes_per_trade': [],
            }
            self.signaled_stocks = set()

        print(f"Loading {len(members)} committee members...")
        for member in tqdm(members, desc="Loading agents"):
            filepath = get_agent_filepath(member, context_window_days)
            agent = load_agent_actor_only(filepath, member['agent_id'])
            if agent is None:
                raise ValueError(f"Failed to load agent: {filepath}")
            self.loaded_agents.append(agent)

        print(f"Committee agent ready with {len(self.loaded_agents)} members")

    def predict_action(self, observation: np.ndarray, held_stock_ids: list = None) -> np.ndarray:
        """
        Generate committee consensus action from observation.

        Args:
            observation: Normalized observation [context_window, num_stocks, features]
            held_stock_ids: Optional list of stock IDs already held (prevents re-signaling)

        Returns:
            action: [num_stocks, 2] array with [coefficient, sale_target] per stock
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(Config.DEVICE)

        all_actions = []
        with torch.no_grad():
            for agent in self.loaded_agents:
                action = agent.actor(obs_tensor).cpu().numpy()[0]  # [num_stocks, 2]
                all_actions.append(action)

        all_actions = np.array(all_actions)  # [num_members, num_stocks, 2]

        all_coeffs = all_actions[:, :, 0]       # [num_members, num_stocks]
        all_sale_targets = all_actions[:, :, 1]  # [num_members, num_stocks]

        final_coeffs = self._apply_consensus(all_coeffs, held_stock_ids)
        final_sale_targets = np.mean(all_sale_targets, axis=0)

        action = np.stack([final_coeffs, final_sale_targets], axis=1)  # [num_stocks, 2]
        return action

    def _apply_consensus(self, all_coeffs: np.ndarray, held_stock_ids: list = None) -> np.ndarray:
        """
        Apply committee consensus logic in two phases.

        Phase 1 — Quorum (evaluated first, independently):
          >= COMMITTEE_QUORUM agents with coeff >= COEFFICIENT_THRESHOLD.
          Only blocked by stocks already held from prior quorum trades, so quorum
          results are identical regardless of conviction percentile.

        Phase 2 — Conviction (additive, never interferes with quorum):
          Any single agent with coeff > its own scalar conviction threshold.
          Blocked by ALL held stocks (quorum + conviction) and by stocks already
          approved by quorum in this step.

        Trade coefficient = max of supporters' coefficients, floored at COEFFICIENT_THRESHOLD.

        Args:
            all_coeffs: [num_members, num_stocks] coefficient predictions
            held_stock_ids: List of stock IDs to exclude from signaling

        Returns:
            final_coeffs: [num_stocks] consensus coefficients
        """
        num_members, num_stocks = all_coeffs.shape
        held_set = set(held_stock_ids) if held_stock_ids else set()

        # Prune closed positions from tracking
        self.quorum_opened = self.quorum_opened & held_set
        self.conviction_opened = self.conviction_opened & held_set

        # === Phase 1: Quorum ===
        # Quorum is only blocked by quorum-opened positions, NOT conviction-opened ones.
        # This ensures quorum trades are identical regardless of conviction percentile.
        votes = all_coeffs >= Config.COEFFICIENT_THRESHOLD
        vote_counts = np.sum(votes, axis=0)
        quorum_pass = vote_counts >= Config.COMMITTEE_QUORUM

        quorum_held_mask = np.zeros(num_stocks, dtype=bool)
        for sid in self.quorum_opened:
            quorum_held_mask[sid] = True
        quorum_approved = quorum_pass & (~quorum_held_mask)

        # === Phase 2: Conviction (only for stocks NOT approved by quorum) ===
        # Conviction is blocked by ALL held stocks (quorum + conviction).
        conviction_thresholds = np.array([
            m['stats']['conviction_threshold'] for m in self.members
        ])  # [num_members]
        agent_convictions = all_coeffs > conviction_thresholds[:, np.newaxis]  # [num_members, num_stocks]
        conviction_any = np.any(agent_convictions, axis=0)  # [num_stocks]

        all_held_mask = np.zeros(num_stocks, dtype=bool)
        for sid in held_set:
            all_held_mask[sid] = True
        conviction_approved = conviction_any & (~all_held_mask) & (~quorum_approved)

        # === Combined result ===
        should_trade = quorum_approved | conviction_approved

        # Update position tracking
        for stock_id in range(num_stocks):
            if quorum_approved[stock_id]:
                self.quorum_opened.add(stock_id)
            elif conviction_approved[stock_id]:
                self.conviction_opened.add(stock_id)

        # Consensus tracking
        if self.track_consensus:
            self.consensus_history['total_steps'] += 1

            stocks_to_count = should_trade.copy()
            for stock_id in range(num_stocks):
                if should_trade[stock_id] and stock_id in self.signaled_stocks:
                    stocks_to_count[stock_id] = False
                elif should_trade[stock_id]:
                    self.signaled_stocks.add(stock_id)

            trades_approved = np.sum(stocks_to_count)
            self.consensus_history['total_trades'] += int(trades_approved)

            unanimity = np.sum(vote_counts[stocks_to_count] == num_members)
            self.consensus_history['unanimity_count'] += int(unanimity)

            is_quorum_trade = quorum_approved & stocks_to_count
            is_conviction_only_trade = conviction_approved & stocks_to_count

            self.consensus_history['trades_by_quorum'] += int(np.sum(is_quorum_trade))
            self.consensus_history['trades_by_conviction_only'] += int(np.sum(is_conviction_only_trade))

            if trades_approved > 0:
                avg_votes = np.mean(vote_counts[stocks_to_count])
                self.consensus_history['avg_votes_per_trade'].append(float(avg_votes))

        # Signal aggregation: max of supporters' coefficients, floored at threshold
        supporters = votes | agent_convictions
        masked_coeffs = np.where(supporters, all_coeffs, -np.inf)
        max_coeffs = np.max(masked_coeffs, axis=0)
        final_coeffs = np.where(should_trade, np.maximum(max_coeffs, Config.COEFFICIENT_THRESHOLD), 0.0).astype(np.float32)

        return final_coeffs

    def get_consensus_summary(self) -> dict:
        """
        Get summary of consensus statistics from tracked history.

        Returns:
            Dict with consensus metrics (empty if tracking disabled)
        """
        if not self.track_consensus:
            return {'note': 'Consensus tracking was not enabled'}

        history = self.consensus_history
        total_steps = history['total_steps']
        total_trades = history['total_trades']

        if total_steps == 0:
            return {'note': 'No steps recorded'}

        return {
            'total_steps': total_steps,
            'total_trades': total_trades,
            'avg_trades_per_step': total_trades / total_steps if total_steps > 0 else 0,
            'unanimity_pct': 100.0 * history['unanimity_count'] / total_trades if total_trades > 0 else 0,
            'trades_by_quorum': history['trades_by_quorum'],
            'trades_by_conviction_only': history['trades_by_conviction_only'],
            'avg_consensus_votes': float(np.mean(history['avg_votes_per_trade'])) if history['avg_votes_per_trade'] else 0,
        }

    def reset_episode(self):
        """Reset episode-level tracking (call at start of each new episode)."""
        self.quorum_opened = set()
        self.conviction_opened = set()
        if self.track_consensus:
            self.signaled_stocks = set()

    def cleanup(self):
        """Release GPU memory from loaded agents."""
        for agent in self.loaded_agents:
            del agent
        self.loaded_agents.clear()
        torch.cuda.empty_cache()


# --- Conviction Threshold Calculation ---

def calculate_conviction_threshold(agent_coeff_history_2d: np.ndarray,
                                   percentile: float = 95,
                                   is_maverick: bool = False) -> float:
    """
    Calculate a single scalar conviction threshold for an agent.

    Takes the Nth percentile of coefficients that resulted in actual trades
    (coeff >= COEFFICIENT_THRESHOLD) across all stocks and all days.
    Conviction means "this agent is unusually confident relative to its own
    trading history."

    Maverick agents use MAVERICK_CONVICTION_PERCENTILE_FLOOR as their
    percentile, independent of the general conviction percentile.

    Args:
        agent_coeff_history_2d: Numpy array [Days, Stocks] for a single agent
        percentile: Percentile to use for conviction threshold (default: 95)
        is_maverick: If True, uses MAVERICK_CONVICTION_PERCENTILE_FLOOR instead of percentile

    Returns:
        Scalar conviction threshold (float)
    """
    effective_percentile = Config.MAVERICK_CONVICTION_PERCENTILE_FLOOR if is_maverick else percentile

    traded_coeffs = agent_coeff_history_2d[agent_coeff_history_2d >= Config.COEFFICIENT_THRESHOLD]

    if len(traded_coeffs) >= 20:
        return float(np.percentile(traded_coeffs, effective_percentile))
    else:
        return 100.0


# Backward compatibility alias
calculate_agent_stats_vectorized = calculate_conviction_threshold


def recalculate_conviction_thresholds(members: list, loader, stats, holdout_info,
                                       context_window_days: int, percentile: float,
                                       val_tensor_cpu=None) -> list:
    """
    Recalculate conviction thresholds for all committee members at a given percentile.

    This creates a deep copy of members with updated conviction_threshold values,
    avoiding pollution of the original roster data.

    Args:
        members: Original list of member dicts from roster
        loader: StockDataLoader instance
        stats: Normalization stats dict
        holdout_info: Holdout period info dict
        context_window_days: Context window size
        percentile: Percentile to use for conviction threshold (e.g., 90, 95, 99, 99.9)
        val_tensor_cpu: Optional pre-loaded validation tensor on CPU (for sweeps to avoid reloading)

    Returns:
        New list of member dicts with recalculated conviction thresholds
    """
    new_members = copy.deepcopy(members)

    if val_tensor_cpu is None:
        val_tensor, valid_indices = get_validation_data(loader, stats, holdout_info)
        val_tensor_cpu = val_tensor.cpu()
        del val_tensor
        torch.cuda.empty_cache()
        should_cleanup_tensor = True
    else:
        should_cleanup_tensor = False

    for member_idx, member in enumerate(tqdm(new_members, desc=f"Recalculating P{percentile} thresholds")):
        filepath = get_agent_filepath(member, context_window_days)
        agent = load_agent_actor_only(filepath, 0)

        if agent is not None:
            try:
                with torch.no_grad():
                    batch_size = 32
                    num_samples = val_tensor_cpu.shape[0]
                    all_coefs = []

                    for start_idx in range(0, num_samples, batch_size):
                        end_idx = min(start_idx + batch_size, num_samples)
                        batch = val_tensor_cpu[start_idx:end_idx].to(Config.DEVICE)
                        batch_actions = agent.actor(batch).cpu().numpy()
                        all_coefs.append(batch_actions[:, :, 0])
                        del batch

                    agent_coeffs_2d = np.concatenate(all_coefs, axis=0)

                conviction_threshold = calculate_conviction_threshold(
                    agent_coeffs_2d, percentile=percentile,
                    is_maverick=member.get('is_maverick', False),
                )
                member['stats']['conviction_threshold'] = float(conviction_threshold)

            except Exception as e:
                print(f"  Warning: Error processing member {member.get('agent_id', member_idx)}: {e}")
                import traceback
                traceback.print_exc()

            try:
                if agent is not None:
                    agent.actor.cpu()
                    del agent
            except:
                pass

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if (member_idx + 1) % 3 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

    if should_cleanup_tensor:
        del val_tensor_cpu
        torch.cuda.empty_cache()
    gc.collect()

    mav_members = [m for m in new_members if m.get('is_maverick', False)]
    non_mav_members = [m for m in new_members if not m.get('is_maverick', False)]
    eff_mav_pct = Config.MAVERICK_CONVICTION_PERCENTILE_FLOOR if mav_members else percentile
    if mav_members:
        mav_thresholds = [m['stats']['conviction_threshold'] for m in mav_members]
        print(f"  Maverick thresholds (effective P{eff_mav_pct}): "
              f"min={min(mav_thresholds):.4f}, max={max(mav_thresholds):.4f}, "
              f"mean={sum(mav_thresholds)/len(mav_thresholds):.4f}  ({len(mav_members)} agents)")
    if non_mav_members:
        non_mav_thresholds = [m['stats']['conviction_threshold'] for m in non_mav_members]
        print(f"  Regular thresholds  (P{percentile}): "
              f"min={min(non_mav_thresholds):.4f}, max={max(non_mav_thresholds):.4f}, "
              f"mean={sum(non_mav_thresholds)/len(non_mav_thresholds):.4f}  ({len(non_mav_members)} agents)")

    return new_members

