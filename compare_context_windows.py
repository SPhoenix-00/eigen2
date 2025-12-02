"""
Context Window Comparison Script for Global 50 Agents

Tests existing G50 agents (trained on 504-day windows) with both:
- Original 504-day context window
- Reduced 151-day context window

Measures performance degradation to answer: "Can we use 504-day agents with 151-day context?"

Usage:
    python compare_context_windows.py
    python compare_context_windows.py --agent-path workspace/global50/agents/agent_001.pth
    python compare_context_windows.py --num-slices 10
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from utils.config import Config
from training.erl_trainer import ERLTrainer


@dataclass
class ContextWindowResult:
    """Results from evaluating with a specific context window."""
    context_days: int
    avg_score: float
    std_score: float
    avg_roi: float
    std_roi: float
    win_rate: float
    num_trades: int
    avg_gain_pct: float
    max_drawdown: float
    slice_scores: List[float]
    slice_rois: List[float]


class ContextWindowComparator:
    """Compare agent performance across different context window sizes."""

    def __init__(self, num_slices: int = 20, use_gauntlet: bool = False):
        """
        Initialize comparator.

        Args:
            num_slices: Number of validation slices to test (default 20 for thorough testing)
            use_gauntlet: If True, use exact gauntlet validation (10 training + 10 validation slices)
        """
        self.num_slices = num_slices
        self.use_gauntlet = use_gauntlet

        print("\n" + "="*80)
        print("Context Window Comparison Tool")
        print("="*80)

        # Load data
        print("\n1. Loading market data...")
        self.data_loader = StockDataLoader()
        data_array, stats = self.data_loader.load_and_prepare()
        self.data_array = self.data_loader.data_array_full
        self.data_array_reduced = data_array
        self.dates = self.data_loader.dates
        self.normalization_stats = stats

        print(f"   Loaded {len(self.data_array)} days of data")
        print(f"   Validation range: {self.data_loader.val_start_idx} - {self.data_loader.val_end_idx}")

    def create_environment_with_context(self, context_window_days: int) -> TradingEnvironment:
        """
        Create a TradingEnvironment with a specific context window size.

        NOTE: Does NOT restore Config.CONTEXT_WINDOW_DAYS - caller must handle this!

        Args:
            context_window_days: Context window size to use

        Returns:
            TradingEnvironment configured with specified context window
        """
        # Config.CONTEXT_WINDOW_DAYS should already be set by caller
        # Create environment (will use the current CONTEXT_WINDOW_DAYS)
        env = TradingEnvironment(
            data_array=self.data_array_reduced,
            dates=self.dates,
            normalization_stats=self.normalization_stats,
            start_idx=context_window_days,  # Minimum index with enough history
            end_idx=len(self.data_array),
            data_array_full=self.data_array,
            is_training=False  # Evaluation mode - no observation noise
        )

        return env

    def generate_test_slices(self, context_window_days: int) -> List[Tuple[int, int, int]]:
        """
        Generate validation slices for testing.

        Uses same logic as ERLTrainer.generate_gauntlet_slices() but respects
        the custom context window size.

        Args:
            context_window_days: Minimum context needed before slice can start

        Returns:
            List of (start_idx, end_idx, trading_end_idx) tuples
        """
        slices = []

        # Use validation data for consistent comparison
        val_start = self.data_loader.val_start_idx
        val_end = self.data_loader.val_end_idx

        # Each slice is TRADING_PERIOD_DAYS + SETTLEMENT_PERIOD_DAYS
        episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

        # Ensure we have enough data for context window
        min_start = max(context_window_days, val_start)
        max_start = val_end - episode_length

        if max_start <= min_start:
            print(f"   WARNING: Not enough validation data for {context_window_days}-day context window")
            print(f"   Using minimal slices from available data")
            # Use whatever we can get
            max_start = min_start + 1

        # Generate evenly-spaced slices
        if max_start > min_start:
            slice_starts = np.linspace(min_start, max_start, self.num_slices, dtype=int)
        else:
            slice_starts = [min_start]

        for start_idx in slice_starts:
            end_idx = start_idx + episode_length
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

            # Ensure we don't exceed data bounds
            if end_idx <= len(self.data_array):
                slices.append((int(start_idx), int(end_idx), int(trading_end_idx)))

        return slices

    def generate_gauntlet_slices(self) -> List[Tuple[int, int, int]]:
        """
        Generate gauntlet validation slices using the ORIGINAL ERLTrainer method.

        This delegates to ERLTrainer.generate_gauntlet_slices() to ensure
        100% identical logic to actual G50 evaluation.

        Returns:
            List of (start_idx, end_idx, trading_end_idx) tuples
        """
        # Import here to avoid circular dependency
        from training.erl_trainer import ERLTrainer

        # Use a dummy trainer just to call the static-ish method
        # We need to create a minimal trainer instance with required attributes
        dummy_trainer = ERLTrainer.__new__(ERLTrainer)
        dummy_trainer.data_loader = self.data_loader

        # Set training and validation ranges (same as ERLTrainer.__init__)
        dummy_trainer.train_start_idx = Config.CONTEXT_WINDOW_DAYS
        dummy_trainer.train_end_idx = len(self.data_loader.train_indices)
        dummy_trainer.val_start_idx = len(self.data_loader.train_indices)
        dummy_trainer.val_end_idx = dummy_trainer.val_start_idx + len(self.data_loader.val_indices)

        # Call the original method
        slices = dummy_trainer.generate_gauntlet_slices()

        print(f"   Gauntlet: {len(slices)} slices total (using ERLTrainer.generate_gauntlet_slices)")
        return slices

    def evaluate_agent_with_context(self, agent: DDPGAgent, context_window_days: int) -> ContextWindowResult:
        """
        Evaluate an agent using a specific context window size.

        Args:
            agent: DDPG agent to evaluate
            context_window_days: Context window size (151 or 504)

        Returns:
            ContextWindowResult with aggregated metrics
        """
        print(f"\n   Testing with {context_window_days}-day context window...")

        # Save and temporarily modify config
        original_context = Config.CONTEXT_WINDOW_DAYS
        print(f"   Setting Config.CONTEXT_WINDOW_DAYS: {original_context} → {context_window_days}")
        Config.CONTEXT_WINDOW_DAYS = context_window_days

        # Create environment with modified context window
        env = self.create_environment_with_context(context_window_days)
        print(f"   Environment observation space: {env.observation_space.shape}")

        # Generate test slices (gauntlet or regular)
        if self.use_gauntlet:
            slices = self.generate_gauntlet_slices()
        else:
            slices = self.generate_test_slices(context_window_days)
            print(f"   Generated {len(slices)} validation slices")

        if len(slices) == 0:
            print(f"   ERROR: No valid slices for {context_window_days}-day window")
            Config.CONTEXT_WINDOW_DAYS = original_context
            return None

        # Run evaluation on each slice
        slice_scores = []
        slice_rois = []
        total_trades = 0
        total_wins = 0
        all_gain_pcts = []
        max_drawdown = 0.0

        for slice_idx, (start_idx, end_idx, trading_end_idx) in enumerate(slices):
            # Reset environment for this slice
            obs, info = env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

            # Debug: verify observation shape matches context window
            if slice_idx == 0:
                expected_shape = (context_window_days, obs.shape[1], obs.shape[2])
                actual_shape = obs.shape
                print(f"   First slice observation shape: {actual_shape} (expected: {expected_shape})")
                if actual_shape[0] != context_window_days:
                    print(f"   WARNING: Context window mismatch! Expected {context_window_days}, got {actual_shape[0]}")

            done = False
            episode_reward = 0.0

            # Run episode
            while not done:
                # Get action from agent (deterministic evaluation)
                action = agent.select_action(obs, add_noise=False)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            # Get full episode summary (contains all metrics)
            episode_summary = env.get_episode_summary()

            # Calculate fitness (same logic as ERLTrainer)
            fitness = episode_reward  # Start with cumulative reward

            # Apply zero trades penalty
            if episode_summary['num_trades'] == 0:
                fitness -= episode_summary['zero_trades_penalty']

            # Apply win rate bonus if enough trades
            if episode_summary['num_trades'] >= Config.WIN_RATE_BONUS_MIN_TRADES:
                win_rate_pct = episode_summary['win_rate'] * 100.0
                if win_rate_pct > Config.WIN_RATE_BONUS_THRESHOLD:
                    bonus = (win_rate_pct - Config.WIN_RATE_BONUS_THRESHOLD) ** 2
                    fitness += bonus

            # Extract metrics
            roi = episode_summary.get('roi', 0.0)  # ROI already in percentage
            num_trades = episode_summary.get('num_trades', 0)
            num_wins = episode_summary.get('num_wins', 0)
            closed_trades = episode_summary.get('closed_trades', [])

            slice_scores.append(fitness)
            slice_rois.append(roi)

            # Aggregate trade statistics
            total_trades += num_trades
            total_wins += num_wins

            for trade in closed_trades:
                all_gain_pcts.append(trade.get('gain_pct', 0.0))

            # Track max drawdown (most negative ROI)
            if roi < max_drawdown:
                max_drawdown = roi

        # Calculate aggregate metrics (convert to Python float for JSON serialization)
        avg_score = float(np.mean(slice_scores))
        std_score = float(np.std(slice_scores))
        avg_roi = float(np.mean(slice_rois))
        std_roi = float(np.std(slice_rois))
        win_rate = float((total_wins / total_trades * 100) if total_trades > 0 else 0.0)
        avg_gain_pct = float(np.mean(all_gain_pcts)) if len(all_gain_pcts) > 0 else 0.0

        # Restore config
        print(f"   Restoring Config.CONTEXT_WINDOW_DAYS: {context_window_days} → {original_context}")
        Config.CONTEXT_WINDOW_DAYS = original_context

        result = ContextWindowResult(
            context_days=context_window_days,
            avg_score=avg_score,
            std_score=std_score,
            avg_roi=avg_roi,
            std_roi=std_roi,
            win_rate=win_rate,
            num_trades=total_trades,
            avg_gain_pct=avg_gain_pct,
            max_drawdown=max_drawdown,
            slice_scores=[float(x) for x in slice_scores],  # Convert to Python float
            slice_rois=[float(x) for x in slice_rois]  # Convert to Python float
        )

        return result

    def compare_agent(self, agent_path: str) -> Dict:
        """
        Compare agent performance on 504-day vs 151-day context windows.

        Args:
            agent_path: Path to agent checkpoint (.pth file)

        Returns:
            Dictionary with comparison results
        """
        print(f"\n2. Loading agent: {agent_path}")
        agent = DDPGAgent()
        agent.load(agent_path)
        agent.actor.eval()
        agent.critic.eval()

        print("\n3. Running evaluations...")

        # Test with 504-day context (original training window)
        print("\n   [1/2] Evaluating with 504-day context window (TRAINED)")
        result_504 = self.evaluate_agent_with_context(agent, context_window_days=504)

        # Test with 151-day context (reduced window)
        print("\n   [2/2] Evaluating with 151-day context window (OUT-OF-DISTRIBUTION)")
        result_151 = self.evaluate_agent_with_context(agent, context_window_days=151)

        # Calculate performance degradation
        if result_504 and result_151:
            score_delta = result_151.avg_score - result_504.avg_score
            score_delta_pct = (score_delta / result_504.avg_score * 100) if result_504.avg_score != 0 else 0

            roi_delta = result_151.avg_roi - result_504.avg_roi
            roi_delta_pct = (roi_delta / result_504.avg_roi * 100) if result_504.avg_roi != 0 else 0

            comparison = {
                'agent_path': agent_path,
                'timestamp': datetime.now().isoformat(),
                'num_slices': self.num_slices,
                'result_504d': asdict(result_504),
                'result_151d': asdict(result_151),
                'degradation': {
                    'score_delta': score_delta,
                    'score_delta_pct': score_delta_pct,
                    'roi_delta': roi_delta,
                    'roi_delta_pct': roi_delta_pct,
                    'win_rate_delta': result_151.win_rate - result_504.win_rate,
                    'trade_count_delta': result_151.num_trades - result_504.num_trades
                }
            }

            return comparison
        else:
            return None

    def print_comparison(self, comparison: Dict):
        """Pretty-print comparison results."""
        if not comparison:
            print("\nERROR: No comparison results to display")
            return

        print("\n" + "="*80)
        print("CONTEXT WINDOW COMPARISON RESULTS")
        print("="*80)

        r504 = comparison['result_504d']
        r151 = comparison['result_151d']
        deg = comparison['degradation']

        print(f"\nAgent: {Path(comparison['agent_path']).name}")
        print(f"Slices Tested: {comparison['num_slices']}")
        if self.use_gauntlet:
            print(f"Validation Mode: ⚔️ GAUNTLET (10 training + 10 validation slices)")
        else:
            print(f"Validation Mode: Standard (validation data only)")

        print("\n" + "-"*80)
        print(f"{'Metric':<30} {'504-day':<20} {'151-day':<20} {'Delta':<10}")
        print("-"*80)

        print(f"{'Avg Fitness Score':<30} {r504['avg_score']:>18.2f}  {r151['avg_score']:>18.2f}  {deg['score_delta']:>8.2f} ({deg['score_delta_pct']:+.1f}%)")
        print(f"{'Std Fitness Score':<30} {r504['std_score']:>18.2f}  {r151['std_score']:>18.2f}")
        print(f"{'Avg ROI %':<30} {r504['avg_roi']:>18.2f}  {r151['avg_roi']:>18.2f}  {deg['roi_delta']:>8.2f} ({deg['roi_delta_pct']:+.1f}%)")
        print(f"{'Std ROI %':<30} {r504['std_roi']:>18.2f}  {r151['std_roi']:>18.2f}")
        print(f"{'Win Rate %':<30} {r504['win_rate']:>18.2f}  {r151['win_rate']:>18.2f}  {deg['win_rate_delta']:>8.2f}")
        print(f"{'Total Trades':<30} {r504['num_trades']:>18}  {r151['num_trades']:>18}  {deg['trade_count_delta']:>8}")
        print(f"{'Avg Gain % (per trade)':<30} {r504['avg_gain_pct']:>18.2f}  {r151['avg_gain_pct']:>18.2f}")
        print(f"{'Max Drawdown %':<30} {r504['max_drawdown']:>18.2f}  {r151['max_drawdown']:>18.2f}")

        print("\n" + "-"*80)
        print("\nINTERPRETATION:")

        if deg['score_delta_pct'] < -10:
            print("   SEVERE DEGRADATION: >10% fitness loss with reduced context")
            print("   → Agent heavily relies on 504-day patterns")
            print("   → NOT recommended to use with 151-day windows")
        elif deg['score_delta_pct'] < -5:
            print("   MODERATE DEGRADATION: 5-10% fitness loss")
            print("   → Agent performance notably worse with less context")
            print("   → Use with caution if 151-day context is required")
        elif deg['score_delta_pct'] < 0:
            print("   MINOR DEGRADATION: <5% fitness loss")
            print("   → Agent relatively robust to context window reduction")
            print("   → May be usable with 151-day context (verify on holdout)")
        else:
            print("   NO DEGRADATION: Agent performs equally or better with less context")
            print("   → Unexpected result - agent may be overfit to noise in 504-day window")
            print("   → Or 151-day patterns are more predictive for this validation period")

        print("="*80)

    def save_comparison(self, comparison: Dict, output_path: str):
        """Save comparison results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare agent performance on different context windows")
    parser.add_argument('--agent-path', type=str, help='Path to specific agent checkpoint to test')
    parser.add_argument('--num-slices', type=int, default=20, help='Number of validation slices to test')
    parser.add_argument('--output', type=str, default='context_window_comparison.json', help='Output JSON file path')
    parser.add_argument('--test-all-g50', action='store_true', help='Test all Global 50 agents')
    parser.add_argument('--gauntlet', action='store_true', help='Use exact gauntlet validation (20 slices: 10 training + 10 validation)')

    args = parser.parse_args()

    # Initialize comparator
    comparator = ContextWindowComparator(num_slices=args.num_slices, use_gauntlet=args.gauntlet)

    if args.gauntlet:
        print("\n⚔️  GAUNTLET MODE: Using exact G50 validation methodology")
        print("   - 10 slices from training data (generalization test)")
        print("   - 10 slices from validation data (held-out test)")
        print("   - Same slice generation as ERLTrainer.generate_gauntlet_slices()")

    if args.test_all_g50:
        # Test all G50 agents
        # Try both possible paths (workspace/global50 for Windows, global50 for Docker)
        g50_dir = Path("global50/agents") if Path("global50/agents").exists() else Path("workspace/global50/agents")
        if not g50_dir.exists():
            print(f"\nERROR: Global 50 directory not found. Tried:")
            print("  - global50/agents")
            print("  - workspace/global50/agents")
            sys.exit(1)

        agent_files = sorted(g50_dir.glob("*.pth"))
        if len(agent_files) == 0:
            print(f"\nERROR: No agents found in {g50_dir}")
            sys.exit(1)

        print(f"\nFound {len(agent_files)} Global 50 agents to test")

        all_results = []
        for agent_file in agent_files:
            comparison = comparator.compare_agent(str(agent_file))
            if comparison:
                comparator.print_comparison(comparison)
                all_results.append(comparison)

        # Save all results
        output_file = args.output.replace('.json', '_all_agents.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to: {output_file}")

    elif args.agent_path:
        # Test specific agent
        if not Path(args.agent_path).exists():
            print(f"\nERROR: Agent not found: {args.agent_path}")
            sys.exit(1)

        comparison = comparator.compare_agent(args.agent_path)
        if comparison:
            comparator.print_comparison(comparison)
            comparator.save_comparison(comparison, args.output)

    else:
        # Default: test best G50 agent
        # Try both possible paths (workspace/global50 for Windows, global50 for Docker)
        g50_dir = Path("global50/agents") if Path("global50/agents").exists() else Path("workspace/global50/agents")
        if not g50_dir.exists():
            print(f"\nERROR: Global 50 directory not found. Tried:")
            print("  - global50/agents")
            print("  - workspace/global50/agents")
            print("Please specify an agent with --agent-path")
            sys.exit(1)

        # Load global50.json to find best agent
        metadata_file = Path("global50/global50.json") if Path("global50/global50.json").exists() else Path("workspace/global50/global50.json")
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if 'agents' in metadata and len(metadata['agents']) > 0:
                # Find agent with highest gauntlet score
                best_agent = max(metadata['agents'], key=lambda x: x.get('gauntlet_score', float('-inf')))
                agent_path = g50_dir / best_agent['agent_filename']

                print(f"\nTesting best G50 agent (score: {best_agent.get('gauntlet_score', 0):.2f})")

                comparison = comparator.compare_agent(str(agent_path))
                if comparison:
                    comparator.print_comparison(comparison)
                    comparator.save_comparison(comparison, args.output)
            else:
                print("\nERROR: No agents found in global50.json")
                sys.exit(1)
        else:
            print(f"\nERROR: Metadata file not found: {metadata_file}")
            print("Please specify an agent with --agent-path")
            sys.exit(1)


if __name__ == "__main__":
    main()
