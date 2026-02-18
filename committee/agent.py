"""
Committee Agent: inference engine and conviction threshold calculation.

CommitteeAgent acts as a drop-in replacement for a single agent, implementing
the same interface (predict action from observation) but using committee
consensus logic internally.
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

        # Consensus tracking
        if self.track_consensus:
            self.consensus_history = {
                'unanimity_count': 0,
                'min_consensus_count': 0,
                'total_steps': 0,
                'total_trades': 0,
                'trades_by_quorum': 0,
                'trades_by_conviction': 0,
                'trades_by_maverick': 0,
                'trades_vetoed': 0,
                'avg_votes_per_trade': [],
            }
            # Track stocks already signaled this episode to prevent double-counting
            self.signaled_stocks = set()

        # Load all member agents (actor only)
        print(f"Loading {len(members)} committee members...")
        for member in tqdm(members, desc="Loading agents"):
            filepath = get_agent_filepath(member, context_window_days)
            agent = load_agent_actor_only(filepath, member['agent_id'])
            if agent is None:
                raise ValueError(f"Failed to load agent: {filepath}")
            self.loaded_agents.append(agent)

        print(f"✓ Committee agent ready with {len(self.loaded_agents)} members")

    def predict_action(self, observation: np.ndarray, held_stock_ids: list = None) -> np.ndarray:
        """
        Generate committee consensus action from observation.

        Args:
            observation: Normalized observation [context_window, num_stocks, features]
            held_stock_ids: Optional list of stock IDs already held (prevents re-signaling)

        Returns:
            action: [num_stocks, 2] array with [coefficient, sale_target] per stock
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(Config.DEVICE)

        # Collect predictions from all members
        all_actions = []

        with torch.no_grad():
            for agent in self.loaded_agents:
                action = agent.actor(obs_tensor).cpu().numpy()[0]  # [num_stocks, 2]
                all_actions.append(action)

        # Stack: [num_members, num_stocks, 2]
        all_actions = np.array(all_actions)

        # Extract coefficients and sale targets
        all_coeffs = all_actions[:, :, 0]  # [num_members, num_stocks]
        all_sale_targets = all_actions[:, :, 1]  # [num_members, num_stocks]

        # Apply committee consensus logic
        final_coeffs = self._apply_consensus(all_coeffs, held_stock_ids)

        # Average sale targets for stocks that passed consensus
        # For stocks with coeff=0 (not approved), sale target doesn't matter
        final_sale_targets = np.mean(all_sale_targets, axis=0)

        # Combine into action
        action = np.stack([final_coeffs, final_sale_targets], axis=1)  # [num_stocks, 2]

        return action

    def _apply_consensus(self, all_coeffs: np.ndarray, held_stock_ids: list = None) -> np.ndarray:
        """
        Apply committee consensus logic with maverick validation.

        Maverick agents can never trigger trades alone — they always need validation
        from non-maverick committee members. The validation threshold depends on
        whether the maverick has conviction:

        Maverick WITH conviction:  1+ non-maverick with coeff >= threshold
        Maverick WITHOUT conviction: 1 non-maverick with conviction, OR
                                     2+ non-mavericks with coeff >= threshold

        Non-maverick agents trade autonomously via standard quorum or conviction.

        Args:
            all_coeffs: [num_members, num_stocks] coefficient predictions
            held_stock_ids: List of stock IDs to exclude from signaling

        Returns:
            final_coeffs: [num_stocks] consensus coefficients
        """
        num_members, num_stocks = all_coeffs.shape

        maverick_mask = np.array([m.get('is_maverick', False) for m in self.members])  # [num_members]
        non_maverick_mask = ~maverick_mask

        # Standard votes: coeff >= global threshold
        votes = all_coeffs >= Config.COEFFICIENT_THRESHOLD  # [num_members, num_stocks]

        # Per-agent conviction thresholds (stock-specific)
        conviction_thresholds = np.array([
            m['stats']['conviction_threshold_vector'] for m in self.members
        ])  # [num_members, num_stocks]
        agent_convictions = all_coeffs > conviction_thresholds  # [num_members, num_stocks]

        # === PATH A: Non-maverick autonomous trading ===
        non_mav_votes = votes & non_maverick_mask[:, np.newaxis]
        non_mav_vote_counts = np.sum(non_mav_votes, axis=0)  # [num_stocks]
        is_non_mav_quorum = non_mav_vote_counts >= Config.COMMITTEE_QUORUM

        non_mav_convictions = agent_convictions & non_maverick_mask[:, np.newaxis]
        is_non_mav_conviction = np.any(non_mav_convictions, axis=0)  # [num_stocks]

        # === PATH B: Maverick-initiated trades (require non-maverick validation) ===
        mav_votes = votes & maverick_mask[:, np.newaxis]
        maverick_signals = np.any(mav_votes, axis=0)  # [num_stocks]

        mav_convictions = agent_convictions & maverick_mask[:, np.newaxis]
        maverick_has_conviction = np.any(mav_convictions, axis=0)  # [num_stocks]

        # Maverick WITH conviction: 1+ non-maverick supporters validate
        mav_conviction_validated = maverick_has_conviction & (non_mav_vote_counts >= 1)

        # Maverick WITHOUT conviction: stricter validation required
        mav_standard_validated = (
            maverick_signals & ~maverick_has_conviction & (
                is_non_mav_conviction | (non_mav_vote_counts >= 2)
            )
        )

        is_maverick_validated = mav_conviction_validated | mav_standard_validated

        # === Veto check ===
        is_silent = all_coeffs < Config.COMMITTEE_VETO_THRESHOLD
        silent_counts = np.sum(is_silent, axis=0)
        is_vetoed = silent_counts >= Config.COMMITTEE_VETO_COUNT

        # === Final decision ===
        should_trade = (is_non_mav_quorum | is_non_mav_conviction | is_maverick_validated) & (~is_vetoed)

        if held_stock_ids:
            held_mask = np.zeros(num_stocks, dtype=bool)
            held_mask[held_stock_ids] = True
            should_trade = should_trade & (~held_mask)

        # Track consensus stats
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

            all_vote_counts = np.sum(votes, axis=0)
            unanimity = np.sum(all_vote_counts[stocks_to_count] == num_members)
            self.consensus_history['unanimity_count'] += int(unanimity)

            min_consensus = np.sum(non_mav_vote_counts[stocks_to_count] == Config.COMMITTEE_QUORUM)
            self.consensus_history['min_consensus_count'] += int(min_consensus)

            # Trades by mechanism (mutually exclusive, priority: quorum > conviction > maverick)
            is_quorum_trade = is_non_mav_quorum & stocks_to_count
            is_conviction_trade = (~is_non_mav_quorum) & is_non_mav_conviction & stocks_to_count
            is_maverick_trade = (~is_non_mav_quorum) & (~is_non_mav_conviction) & is_maverick_validated & stocks_to_count

            self.consensus_history['trades_by_quorum'] += int(np.sum(is_quorum_trade))
            self.consensus_history['trades_by_conviction'] += int(np.sum(is_conviction_trade))
            self.consensus_history['trades_by_maverick'] += int(np.sum(is_maverick_trade))

            potential_trades = (is_non_mav_quorum | is_non_mav_conviction | is_maverick_validated)
            if held_stock_ids:
                potential_trades = potential_trades & (~held_mask)
            vetoed_trades = potential_trades & is_vetoed
            self.consensus_history['trades_vetoed'] += int(np.sum(vetoed_trades))

            if trades_approved > 0:
                avg_votes = np.mean(all_vote_counts[stocks_to_count])
                self.consensus_history['avg_votes_per_trade'].append(float(avg_votes))

        # Signal Aggregation
        final_coeffs = np.zeros(num_stocks, dtype=np.float32)

        for stock_idx in range(num_stocks):
            if should_trade[stock_idx]:
                if is_non_mav_quorum[stock_idx]:
                    supporting_agents_mask = non_mav_votes[:, stock_idx]
                elif is_non_mav_conviction[stock_idx]:
                    supporting_agents_mask = non_mav_votes[:, stock_idx] | non_mav_convictions[:, stock_idx]
                else:
                    supporting_agents_mask = votes[:, stock_idx] | agent_convictions[:, stock_idx]

                if np.any(supporting_agents_mask):
                    supporting_coeffs = all_coeffs[supporting_agents_mask, stock_idx]
                    final_coeffs[stock_idx] = np.mean(supporting_coeffs)
                else:
                    final_coeffs[stock_idx] = Config.COEFFICIENT_THRESHOLD

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
            'min_consensus_pct': 100.0 * history['min_consensus_count'] / total_trades if total_trades > 0 else 0,
            'trades_by_quorum': history['trades_by_quorum'],
            'trades_by_conviction': history['trades_by_conviction'],
            'trades_by_maverick': history['trades_by_maverick'],
            'trades_vetoed': history['trades_vetoed'],
            'avg_consensus_votes': float(np.mean(history['avg_votes_per_trade'])) if history['avg_votes_per_trade'] else 0,
        }

    def reset_episode(self):
        """Reset episode-level tracking (call at start of each new episode)."""
        if self.track_consensus:
            self.signaled_stocks = set()

    def cleanup(self):
        """Release GPU memory from loaded agents."""
        for agent in self.loaded_agents:
            del agent
        self.loaded_agents.clear()
        torch.cuda.empty_cache()


# --- Conviction Threshold Calculation ---

def calculate_agent_stats_vectorized(agent_coeff_history_2d: np.ndarray,
                                      percentile: float = 95) -> np.ndarray:
    """
    Calculates the Nth percentile conviction threshold for each stock.

    Uses CONVICTION_COEFFICIENT_FLOOR (not COEFFICIENT_THRESHOLD) so that
    conviction thresholds can fall below the global trade threshold. This allows
    agents with sub-threshold but personally-high coefficients to trigger conviction
    without having a standard vote, making the conviction percentile parameter
    meaningful at all quorum levels.

    Args:
        agent_coeff_history_2d: Numpy array [Days, Stocks] for a single agent
        percentile: Percentile to use for conviction threshold (default: 95)

    Returns:
        threshold_vector: Numpy array [Stocks] of Nth percentile conviction thresholds
    """
    days, num_stocks = agent_coeff_history_2d.shape
    threshold_vector = np.zeros(num_stocks, dtype=np.float32)

    floor = Config.CONVICTION_COEFFICIENT_FLOOR

    valid_mask = agent_coeff_history_2d >= floor
    all_active = agent_coeff_history_2d[valid_mask]

    if len(all_active) > 0:
        global_threshold = np.percentile(all_active, percentile)
    else:
        global_threshold = 100.0

    for i in range(num_stocks):
        stock_coeffs = agent_coeff_history_2d[:, i]
        active_coeffs = stock_coeffs[stock_coeffs >= floor]

        if len(active_coeffs) >= 20:
            min_points_for_high_percentile = max(20, int(100 / (100 - percentile)) if percentile >= 99.0 else 20)
            if len(active_coeffs) >= min_points_for_high_percentile:
                threshold_vector[i] = np.percentile(active_coeffs, percentile)
            else:
                threshold_vector[i] = np.max(active_coeffs) if len(active_coeffs) > 0 else global_threshold
        else:
            threshold_vector[i] = global_threshold

    return threshold_vector


def recalculate_conviction_thresholds(members: list, loader, stats, holdout_info,
                                       context_window_days: int, percentile: float,
                                       val_tensor_cpu=None) -> list:
    """
    Recalculate conviction thresholds for all committee members at a given percentile.

    This creates a deep copy of members with updated conviction_threshold_vector values,
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
    # Deep copy to avoid modifying original
    new_members = copy.deepcopy(members)

    # Get validation data tensor (reuse if provided, otherwise load fresh)
    if val_tensor_cpu is None:
        val_tensor, valid_indices = get_validation_data(loader, stats, holdout_info)
        # Move to CPU immediately to avoid GPU memory fragmentation
        val_tensor_cpu = val_tensor.cpu()
        del val_tensor
        torch.cuda.empty_cache()
        should_cleanup_tensor = True
    else:
        # Reusing provided tensor - don't clean it up
        should_cleanup_tensor = False

    num_stocks = Config.NUM_INVESTABLE_STOCKS

    for member_idx, member in enumerate(tqdm(new_members, desc=f"Recalculating P{percentile} thresholds")):
        filepath = get_agent_filepath(member, context_window_days)
        agent = load_agent_actor_only(filepath, 0)

        if agent is not None:
            try:
                # Generate coefficients using same method as calculate_coefficient_correlations
                with torch.no_grad():
                    batch_size = 32
                    num_samples = val_tensor_cpu.shape[0]
                    all_coefs = []

                    for start_idx in range(0, num_samples, batch_size):
                        end_idx = min(start_idx + batch_size, num_samples)
                        # Move batch to GPU only when needed
                        batch = val_tensor_cpu[start_idx:end_idx].to(Config.DEVICE)
                        batch_actions = agent.actor(batch).cpu().numpy()
                        # Extract coefficients (first output dimension)
                        all_coefs.append(batch_actions[:, :, 0])
                        # Delete batch (but DON'T call empty_cache here - it's too expensive!)
                        del batch

                    # Concatenate to [Days, Stocks]
                    agent_coeffs_2d = np.concatenate(all_coefs, axis=0)

                # Recalculate with new percentile
                conviction_threshold_vector = calculate_agent_stats_vectorized(
                    agent_coeffs_2d, percentile=percentile
                )
                member['stats']['conviction_threshold_vector'] = conviction_threshold_vector.tolist()

            except Exception as e:
                print(f"  ⚠ Error processing member {member.get('agent_id', member_idx)}: {e}")
                import traceback
                traceback.print_exc()

            # Aggressive cleanup - move actor to CPU first to ensure GPU memory is released
            try:
                if agent is not None:
                    agent.actor.cpu()
                    del agent
            except:
                pass

            # Only call empty_cache once per agent, not per batch!
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Additional cleanup every few agents to prevent memory buildup
            if (member_idx + 1) % 3 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

    # Clean up val_tensor_cpu only if we created it (not if it was provided)
    if should_cleanup_tensor:
        del val_tensor_cpu
        torch.cuda.empty_cache()
    gc.collect()

    return new_members

