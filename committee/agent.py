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
        Apply committee consensus logic to coefficient predictions.

        Args:
            all_coeffs: [num_members, num_stocks] coefficient predictions
            held_stock_ids: List of stock IDs to exclude from signaling

        Returns:
            final_coeffs: [num_stocks] consensus coefficients
        """
        num_members, num_stocks = all_coeffs.shape

        # 1. Standard Voting (Quorum) - coeff >= global threshold
        votes = all_coeffs >= Config.COEFFICIENT_THRESHOLD
        vote_counts = np.sum(votes, axis=0)  # [num_stocks]
        is_quorum = vote_counts >= Config.COMMITTEE_QUORUM

        # 2. Conviction Check (Stock-Specific Override)
        # Conviction = agent exceeds their own P99 threshold for that stock
        conviction_thresholds = np.array([
            m['stats']['conviction_threshold_vector'] for m in self.members
        ])  # [num_members, num_stocks]

        agent_convictions = all_coeffs > conviction_thresholds
        is_conviction_raw = np.any(agent_convictions, axis=0)  # [num_stocks]

        # SOCIAL PROOF: Conviction counts as a vote for the social proof check.
        # An agent with conviction is "voting" even if below global threshold.
        # This allows a specialist (coeff=0.9, p99=0.8) to count toward social proof.
        effective_votes = votes | agent_convictions  # [num_members, num_stocks]
        effective_vote_counts = np.sum(effective_votes, axis=0)  # [num_stocks]

        # A conviction trade requires at least 2 total supporters (votes OR convictions)
        # This prevents a single hallucinating agent from triggering trades alone.
        is_conviction = is_conviction_raw & (effective_vote_counts >= 2)

        # 3. Veto Check (Fixed number of silent members)
        is_silent = all_coeffs < Config.COMMITTEE_VETO_THRESHOLD
        silent_counts = np.sum(is_silent, axis=0)
        is_vetoed = silent_counts >= Config.COMMITTEE_VETO_COUNT

        # 4. Final Decision Logic
        # PASS if: (Quorum OR Conviction) AND (NOT Veto)
        should_trade = (is_quorum | is_conviction) & (~is_vetoed)

        # FILTER: Mask out stocks we already hold to prevent "Ghost Signals" in stats
        if held_stock_ids:
            held_mask = np.zeros(num_stocks, dtype=bool)
            held_mask[held_stock_ids] = True
            # If we hold it, we don't "trade" it (prevents stats inflation)
            should_trade = should_trade & (~held_mask)

        # Track consensus stats if enabled
        if self.track_consensus:
            self.consensus_history['total_steps'] += 1

            # Filter to only count stocks we haven't signaled before (prevents double-counting)
            stocks_to_count = should_trade.copy()
            for stock_id in range(num_stocks):
                if should_trade[stock_id] and stock_id in self.signaled_stocks:
                    stocks_to_count[stock_id] = False  # Already counted this stock
                elif should_trade[stock_id]:
                    self.signaled_stocks.add(stock_id)  # Mark as signaled

            # Count trades approved by each mechanism (only new signals)
            trades_approved = np.sum(stocks_to_count)
            self.consensus_history['total_trades'] += int(trades_approved)

            # Unanimity: all members voted for the trade
            unanimity = np.sum(vote_counts[stocks_to_count] == num_members)
            self.consensus_history['unanimity_count'] += int(unanimity)

            # Min consensus: exactly quorum votes
            min_consensus = np.sum(vote_counts[stocks_to_count] == Config.COMMITTEE_QUORUM)
            self.consensus_history['min_consensus_count'] += int(min_consensus)

            # Trades by Quorum vs Conviction (MUTUALLY EXCLUSIVE)
            # For trades that actually happened, determine which mechanism approved them
            # If quorum was met, credit quorum (even if conviction also triggered)
            # Only credit conviction for trades where quorum was NOT met (conviction "rescues")
            is_quorum_trade = is_quorum & stocks_to_count
            is_conviction_only_trade = (~is_quorum) & is_conviction & stocks_to_count

            # Verify mathematical correctness: these should sum to total trades
            self.consensus_history['trades_by_quorum'] += int(np.sum(is_quorum_trade))
            self.consensus_history['trades_by_conviction'] += int(np.sum(is_conviction_only_trade))

            # Trades vetoed (Only count vetoes on signals that otherwise would have passed)
            # This filters out "vetoes" on stocks nobody wanted anyway
            potential_trades = (is_quorum | is_conviction)
            if held_stock_ids:
                potential_trades = potential_trades & (~held_mask)

            vetoed_trades = potential_trades & is_vetoed
            self.consensus_history['trades_vetoed'] += int(np.sum(vetoed_trades))

            # Average votes per trade
            if trades_approved > 0:
                avg_votes = np.mean(vote_counts[stocks_to_count])
                self.consensus_history['avg_votes_per_trade'].append(float(avg_votes))

        # 5. Signal Aggregation
        # For approved trades, average coefficients from ALL supporting agents
        # (both standard voters AND conviction agents)
        final_coeffs = np.zeros(num_stocks, dtype=np.float32)

        for stock_idx in range(num_stocks):
            if should_trade[stock_idx]:
                # Get all agents who support this trade (vote OR conviction)
                # This ensures the conviction specialist is included in the average
                supporting_agents_mask = effective_votes[:, stock_idx]  # Boolean mask

                if np.any(supporting_agents_mask):
                    # Average coefficients from all supporting agents
                    supporting_coeffs = all_coeffs[supporting_agents_mask, stock_idx]
                    final_coeffs[stock_idx] = np.mean(supporting_coeffs)
                else:
                    # Fallback to threshold (shouldn't happen, but safety)
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
                                      percentile: int = 95) -> np.ndarray:
    """
    Calculates the Nth percentile conviction threshold for each stock.

    CRITICAL: Only considers coefficients that would actually trigger a trade
    (>= Config.COEFFICIENT_THRESHOLD). Including non-trading noise (< 1.0) drags
    the percentile down, allowing sub-threshold signals to masquerade as 'high conviction'.

    Args:
        agent_coeff_history_2d: Numpy array [Days, Stocks] for a single agent
        percentile: Percentile to use for conviction threshold (default: 95)

    Returns:
        threshold_vector: Numpy array [Stocks] of Nth percentile conviction thresholds
    """
    days, num_stocks = agent_coeff_history_2d.shape
    threshold_vector = np.zeros(num_stocks, dtype=np.float32)

    # Filter: Only look at coefficients that are actual trades
    # We use Config.COEFFICIENT_THRESHOLD (1.0) as the floor
    valid_trades_mask = agent_coeff_history_2d >= Config.COEFFICIENT_THRESHOLD
    all_active = agent_coeff_history_2d[valid_trades_mask]

    # Calculate Global percentile fallback
    # If the agent has NEVER traded (or very rarely), we set a high fallback
    # so it cannot easily trigger conviction on noise.
    if len(all_active) > 0:
        global_threshold = np.percentile(all_active, percentile)
    else:
        # Agent is a ghost (no trades > 1.0). Set threshold to infinity to disable conviction.
        global_threshold = 100.0

    for i in range(num_stocks):
        # Extract history for this specific stock
        stock_coeffs = agent_coeff_history_2d[:, i]

        # Only calculate percentile based on actual trades for this stock
        active_coeffs = stock_coeffs[stock_coeffs >= Config.COEFFICIENT_THRESHOLD]

        if len(active_coeffs) >= 20:
            # Sufficient history for this stock
            # For very high percentiles (>= 99.0), ensure we have enough data points
            # to avoid edge cases in percentile calculation
            min_points_for_high_percentile = max(20, int(100 / (100 - percentile)) if percentile >= 99.0 else 20)
            if len(active_coeffs) >= min_points_for_high_percentile:
                threshold_vector[i] = np.percentile(active_coeffs, percentile)
            else:
                # Not enough points for reliable high percentile, use max value instead
                threshold_vector[i] = np.max(active_coeffs) if len(active_coeffs) > 0 else global_threshold
        else:
            # Insufficient history, fallback to global threshold
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

