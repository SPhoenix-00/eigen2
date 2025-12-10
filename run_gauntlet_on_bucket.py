"""
Run Gauntlet Validation on a Global 50 Bucket

This script loads all agents from a specific Global 50 context window bucket
and runs the full gauntlet validation on each agent, producing the same
detailed output as the main.py training script.

Usage:
    python run_gauntlet_on_bucket.py --cw151              # Test agents from cw151 bucket
    python run_gauntlet_on_bucket.py --cw504              # Test agents from cw504 bucket
    python run_gauntlet_on_bucket.py --cw151 --top 5      # Only test top 5 agents
    python run_gauntlet_on_bucket.py --cw151 --agent <name>  # Test specific agent

Parameters used match main.py exactly:
    - ZERO_TRADES_PENALTY_GAUNTLET = 10.0 (not 500, for soft penalty during gauntlet)
    - Consistency mode enabled (same as Global 50 evaluation)
    - 20 slices (10 training + 10 validation)
    - Aggregator: 0.67*mean + 0.33*min (excluding top slice)
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
from datetime import datetime

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from utils.config import Config
from utils.display import visualize_gauntlet_slices
from training.erl_trainer import ERLTrainer


class GauntletRunner:
    """Runs gauntlet validation on agents from a Global 50 bucket."""

    def __init__(self, context_window: int):
        """
        Initialize the gauntlet runner.

        Args:
            context_window: Context window in days (e.g., 151, 504)
        """
        self.context_window = context_window
        self.context_window_id = f"cw{context_window}"

        print("\n" + "="*70)
        print(f"Gauntlet Runner - Context Window: {context_window} days")
        print("="*70)

        # Validate context window matches config
        if context_window != Config.CONTEXT_WINDOW_DAYS:
            print(f"\n WARNING: Requested context window ({context_window}) differs from config ({Config.CONTEXT_WINDOW_DAYS})")
            print(f"  The gauntlet will use the data loader's current configuration.")
            print(f"  For accurate testing, ensure Config.CONTEXT_WINDOW_DAYS matches your bucket.\n")

        # Initialize data loader
        print("\n1. Loading market data...")
        self.data_loader = StockDataLoader()
        data_array, stats = self.data_loader.load_and_prepare()
        self.normalization_stats = stats
        print(f"   Loaded {len(self.data_loader.data_array_full)} days of data")

        # Get validation indices
        self.val_start_idx = self.data_loader.val_start_idx
        self.val_end_idx = self.data_loader.val_end_idx
        self.train_start_idx = Config.CONTEXT_WINDOW_DAYS
        self.train_end_idx = self.val_start_idx
        print(f"   Training range: {self.train_start_idx} - {self.train_end_idx}")
        print(f"   Validation range: {self.val_start_idx} - {self.val_end_idx}")

        # Create evaluation environment
        print("\n2. Setting up evaluation environment...")
        full_end_idx = len(self.data_loader.data_array_full)
        self.eval_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            start_idx=Config.CONTEXT_WINDOW_DAYS,
            end_idx=full_end_idx,
            trading_end_idx=Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS,
            data_array_full=self.data_loader.data_array_full,
            consistency_mode=True  # Global 50 ALWAYS uses consistency mode
        )

        # Bind ERLTrainer methods for gauntlet
        self._bind_trainer_methods()

        # Locate bucket directory
        self.bucket_dir = self._find_bucket_dir()
        if self.bucket_dir:
            print(f"\n3. Located bucket: {self.bucket_dir}")
        else:
            print(f"\n ERROR: Could not find global50/{self.context_window_id}/ directory")
            print(f"  Tried: global50/{self.context_window_id}/, workspace/global50/{self.context_window_id}/")
            sys.exit(1)

        print("\n" + "="*70)

    def _find_bucket_dir(self) -> Optional[Path]:
        """Find the global50 bucket directory for the context window."""
        candidates = [
            Path("global50") / self.context_window_id,
            Path("workspace/global50") / self.context_window_id,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _bind_trainer_methods(self):
        """Bind ERLTrainer methods for use in gauntlet validation."""
        # Create a minimal helper class to hold state needed by ERLTrainer methods
        class GauntletHelper:
            pass

        self.helper = GauntletHelper()
        self.helper.data_loader = self.data_loader
        self.helper.val_start_idx = self.val_start_idx
        self.helper.val_end_idx = self.val_end_idx
        self.helper.train_start_idx = self.train_start_idx
        self.helper.train_end_idx = self.train_end_idx
        self.helper.eval_env = self.eval_env
        self.helper.replay_buffer = None  # Not needed for validation

        # Bind methods
        self.helper.generate_gauntlet_slices = ERLTrainer.generate_gauntlet_slices.__get__(
            self.helper, GauntletHelper
        )
        self.helper.run_episode_batched = ERLTrainer.run_episode_batched.__get__(
            self.helper, GauntletHelper
        )
        self.helper.calculate_expectancy = ERLTrainer.calculate_expectancy.__get__(
            self.helper, GauntletHelper
        )

    def discover_agents(self) -> List[Tuple[Path, dict]]:
        """
        Discover all agents in the bucket with their metadata.

        Returns:
            List of (agent_path, metadata) tuples, sorted by gauntlet_score descending
        """
        agents = []

        # Look for agents directory
        agents_dir = self.bucket_dir / "agents"
        if not agents_dir.exists():
            print(f"   No 'agents' subdirectory found in {self.bucket_dir}")
            # Try looking directly in bucket dir
            agents_dir = self.bucket_dir

        # Find all .pth files
        agent_files = list(agents_dir.glob("*.pth"))
        print(f"   Found {len(agent_files)} agent files")

        # Try to load global50.json for metadata
        json_path = self.bucket_dir / "global50.json"
        metadata_map = {}

        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    entries = data.get('entries', [])
                    for entry in entries:
                        # Build filename from entry
                        run_name = entry.get('run_name', 'unknown')
                        agent_id = entry.get('agent_id', 0)
                        filename = f"{run_name}_{agent_id}.pth"
                        metadata_map[filename] = entry
                    print(f"   Loaded metadata for {len(entries)} agents from global50.json")
            except Exception as e:
                print(f"   Warning: Could not load global50.json: {e}")

        # Build list with metadata
        for agent_path in agent_files:
            metadata = metadata_map.get(agent_path.name, {})
            if not metadata:
                # Create minimal metadata from filename
                metadata = {
                    'agent_name': agent_path.stem,
                    'gauntlet_score': 0.0
                }
            agents.append((agent_path, metadata))

        # Sort by gauntlet_score descending
        agents.sort(key=lambda x: x[1].get('gauntlet_score', 0.0), reverse=True)

        return agents

    def run_gauntlet(self, agent: DDPGAgent, agent_name: str) -> dict:
        """
        Run gauntlet validation on an agent.

        Uses the exact same parameters as main.py:
        - ZERO_TRADES_PENALTY_GAUNTLET = 10.0 (soft penalty)
        - 20 slices (10 training + 10 validation)
        - Aggregator: 0.67*mean + 0.33*min (excluding top slice)

        Args:
            agent: Agent to evaluate
            agent_name: Name for logging

        Returns:
            Dictionary with gauntlet results
        """
        print(f"\n{'='*70}")
        print(f" GAUNTLET VALIDATION - {agent_name}")
        print(f"{'='*70}")

        # Generate gauntlet slices (10 training + 10 validation)
        gauntlet_slices = self.helper.generate_gauntlet_slices()
        num_training_slices = sum(1 for s in gauntlet_slices if s[0] < self.val_start_idx)
        num_val_slices = len(gauntlet_slices) - num_training_slices
        print(f"Testing on {len(gauntlet_slices)} slices ({num_training_slices} training + {num_val_slices} validation)")

        # Set agent to eval mode
        agent.actor.eval()
        agent.critic.eval()

        # Enable gauntlet mode for soft zero-trades penalty (10.0, not 500)
        self.eval_env.set_gauntlet_mode(True)

        slice_results = []
        all_closed_trades = []

        # Evaluate each slice
        for i, (start_idx, end_idx, _) in enumerate(gauntlet_slices):
            # Use ERLTrainer's optimized batched inference
            fitness, episode_info = self.helper.run_episode_batched(
                agent=agent,
                env=self.eval_env,
                start_idx=start_idx,
                end_idx=end_idx,
                training=False,
                batch_size=16
            )

            # Apply zero-trades gradient (matches ERLTrainer logic)
            if episode_info['num_trades'] == 0:
                max_coeff = episode_info.get('max_coefficient_during_episode', 0.0)
                fitness = fitness + max_coeff

            slice_results.append({
                'fitness': fitness,
                'win_rate': episode_info['win_rate'],
                'num_trades': episode_info['num_trades'],
                'num_wins': episode_info['num_wins'],
                'num_losses': episode_info['num_losses'],
                'avg_reward_per_trade': episode_info['avg_reward_per_trade'],
                'raw_pnl': episode_info.get('raw_pnl', 0.0),
                'total_investment': episode_info.get('total_investment', 0.0)
            })

            # Collect closed trades for expectancy calculation
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{len(gauntlet_slices)} slices...")

        # Extract fitness scores
        fitness_scores = [result['fitness'] for result in slice_results]

        # Exclude top slice from gauntlet score calculation (prevents lucky outliers from inflating score)
        # But keep max_score for display purposes
        max_score = float(np.max(fitness_scores))
        scores_without_top = sorted(fitness_scores)[:-1]  # Remove the highest score

        # Use same aggregator as ERLTrainer: 0.67*mean + 0.33*min (excluding top slice)
        mean_score = float(np.mean(scores_without_top))
        min_score = float(np.min(scores_without_top))
        gauntlet_score = float((0.67 * mean_score) + (0.33 * min_score))

        # Aggregate metrics
        total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
        total_investment = sum([r['total_investment'] for r in slice_results])
        roi = float((total_raw_pnl / total_investment * 100) if total_investment > 0 else 0.0)

        total_wins = sum([r['num_wins'] for r in slice_results])
        total_losses = sum([r['num_losses'] for r in slice_results])
        total_trades = int(total_wins + total_losses)
        win_rate = float((total_wins / total_trades * 100) if total_trades > 0 else 0.0)

        # Calculate expectancy using ERLTrainer's method
        expectancy = float(self.helper.calculate_expectancy(all_closed_trades))

        # Calculate quality_count (trades with gain >= Config.ROI_QUALITY_THRESHOLD)
        quality_threshold = Config.ROI_QUALITY_THRESHOLD
        if all_closed_trades:
            quality_count = sum(1 for t in all_closed_trades if t.get('gain_pct', 0) >= quality_threshold)
        else:
            quality_count = 0

        # Calculate ratios
        quality_ratio = float(quality_count / total_trades) if total_trades > 0 else 0.0
        win_ratio = float(total_wins / total_trades) if total_trades > 0 else 0.0

        # Display detailed visualization (same as main.py)
        visualize_gauntlet_slices(fitness_scores, mean_score, min_score, max_score, gauntlet_score)

        # Print trading metrics summary (same format as ERLTrainer.run_gauntlet_validation)
        print(f"\n{'='*70}")
        print(f"{'GAUNTLET TRADING METRICS':^70}")
        print(f"{'='*70}")
        print(f"  ROI:               {roi:>12.2f}%")
        print(f"  Win Rate:          {win_rate:>11.1f}%")
        print(f"  Total Trades:      {total_trades:>12}")
        print(f"  Quality Trades:    {quality_count:>12}  (>= {quality_threshold}% gain)")
        print(f"  Quality Ratio:     {quality_ratio:>12.3f}")
        print(f"  Expectancy:        {expectancy:>11.2f}%")
        print(f"{'='*70}")

        # Reset gauntlet mode
        self.eval_env.set_gauntlet_mode(False)

        return {
            'agent_name': agent_name,
            'gauntlet_score': gauntlet_score,
            'mean_fitness': mean_score,
            'min_fitness': min_score,
            'max_fitness': max_score,
            'roi': roi,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'quality_count': quality_count,
            'quality_ratio': quality_ratio,
            'win_ratio': win_ratio,
            'expectancy': expectancy,
            'num_slices': len(gauntlet_slices),
            'fitness_all_slices': fitness_scores
        }

    def run(self, top_n: Optional[int] = None, agent_name: Optional[str] = None):
        """
        Run gauntlet on agents from the bucket.

        Args:
            top_n: Only test top N agents (by stored gauntlet_score)
            agent_name: Only test specific agent by name
        """
        # Discover agents
        print("\n" + "="*70)
        print("Discovering Agents")
        print("="*70)
        agents = self.discover_agents()

        if not agents:
            print("\nNo agents found in bucket. Exiting.")
            return

        # Filter agents if requested
        if agent_name:
            agents = [(p, m) for p, m in agents if agent_name in p.stem]
            if not agents:
                print(f"\nNo agent matching '{agent_name}' found. Exiting.")
                return
            print(f"\nFiltered to {len(agents)} agent(s) matching '{agent_name}'")

        if top_n:
            agents = agents[:top_n]
            print(f"\nTesting top {len(agents)} agents")

        # Display agent list
        print(f"\n{'='*70}")
        print(f"{'AGENTS TO TEST':^70}")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Agent Name':<40} {'Stored Score':>12}")
        print("-" * 70)
        for i, (path, meta) in enumerate(agents, 1):
            score = meta.get('gauntlet_score', 0.0)
            print(f"{i:<4} {path.stem:<40} {score:>12.2f}")
        print("="*70)

        # Run gauntlet on each agent
        results = []
        for i, (agent_path, metadata) in enumerate(agents, 1):
            print(f"\n[{i}/{len(agents)}] Loading {agent_path.name}...")

            try:
                # Load agent
                agent = DDPGAgent(agent_id=0)
                agent.load(str(agent_path))

                # Run gauntlet
                result = self.run_gauntlet(agent, agent_path.stem)

                # Compare with stored score
                stored_score = metadata.get('gauntlet_score', 0.0)
                score_diff = result['gauntlet_score'] - stored_score
                print(f"\n  Stored Score:  {stored_score:>12.2f}")
                print(f"  New Score:     {result['gauntlet_score']:>12.2f}  ({score_diff:+.2f})")

                results.append(result)

            except Exception as e:
                print(f"\n   ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Print final summary
        self._print_summary(results)

    def _print_summary(self, results: List[dict]):
        """Print summary of all gauntlet results."""
        if not results:
            return

        print("\n" + "="*70)
        print(f"{'GAUNTLET SUMMARY':^70}")
        print("="*70)

        print(f"\n{'Agent Name':<30} {'Score':>10} {'ROI':>10} {'Win%':>8} {'Trades':>8}")
        print("-" * 70)

        # Sort by gauntlet_score descending
        results_sorted = sorted(results, key=lambda x: x['gauntlet_score'], reverse=True)

        for r in results_sorted:
            print(f"{r['agent_name'][:30]:<30} {r['gauntlet_score']:>10.2f} {r['roi']:>9.2f}% {r['win_rate']:>7.1f}% {r['total_trades']:>8}")

        print("-" * 70)

        # Aggregate statistics
        scores = [r['gauntlet_score'] for r in results]
        rois = [r['roi'] for r in results]

        print(f"\n{'AGGREGATE STATISTICS':^70}")
        print("-" * 70)
        print(f"  Agents Tested:     {len(results):>12}")
        print(f"  Mean Score:        {np.mean(scores):>12.2f}")
        print(f"  Median Score:      {np.median(scores):>12.2f}")
        print(f"  Score StdDev:      {np.std(scores):>12.2f}")
        print(f"  Best Score:        {np.max(scores):>12.2f}")
        print(f"  Worst Score:       {np.min(scores):>12.2f}")
        print(f"  Mean ROI:          {np.mean(rois):>11.2f}%")
        print(f"  Median ROI:        {np.median(rois):>11.2f}%")

        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run Gauntlet validation on a Global 50 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_gauntlet_on_bucket.py --cw151              # Test all agents in cw151
    python run_gauntlet_on_bucket.py --cw504              # Test all agents in cw504
    python run_gauntlet_on_bucket.py --cw151 --top 5      # Only test top 5 agents
    python run_gauntlet_on_bucket.py --cw151 --agent azure-thunder  # Test specific agent

Note: Uses ZERO_TRADES_PENALTY_GAUNTLET = 10.0 (same as main.py during gauntlet)
        """
    )

    # Context window selection (mutually exclusive)
    cw_group = parser.add_mutually_exclusive_group(required=True)
    cw_group.add_argument(
        '--cw151',
        action='store_true',
        help='Test agents from the cw151 (151-day context window) bucket'
    )
    cw_group.add_argument(
        '--cw504',
        action='store_true',
        help='Test agents from the cw504 (504-day context window) bucket'
    )
    cw_group.add_argument(
        '--cw',
        type=int,
        metavar='DAYS',
        help='Test agents from a custom context window bucket (e.g., --cw 252)'
    )

    # Filtering options
    parser.add_argument(
        '--top',
        type=int,
        metavar='N',
        help='Only test the top N agents (by stored gauntlet score)'
    )
    parser.add_argument(
        '--agent',
        type=str,
        metavar='NAME',
        help='Only test agent(s) whose name contains NAME'
    )

    args = parser.parse_args()

    # Determine context window
    if args.cw151:
        context_window = 151
    elif args.cw504:
        context_window = 504
    else:
        context_window = args.cw

    print("\n" + "="*70)
    print("Run Gauntlet on Global 50 Bucket")
    print("="*70)
    print(f"\nContext Window: {context_window} days")
    print(f"ZERO_TRADES_PENALTY_GAUNTLET: {Config.ZERO_TRADES_PENALTY_GAUNTLET}")
    print(f"Consistency Mode: Enabled (always for Global 50)")

    # Initialize and run
    runner = GauntletRunner(context_window)
    runner.run(top_n=args.top, agent_name=args.agent)

    print("\nGauntlet validation complete!")


if __name__ == "__main__":
    main()
