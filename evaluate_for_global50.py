"""
Evaluate existing agents for Global 50 promotion.

This script loads agents from a specified folder, runs them through gauntlet
validation, and promotes qualifying agents to the Global Hall of Fame.

Usage:
    python evaluate_for_global50.py --agent-dir <path> [--run-name <name>]

Example:
    python evaluate_for_global50.py --agent-dir checkpoints/azure-thunder-123/hall_of_fame
    python evaluate_for_global50.py --agent-dir workspace/elite_agents --run-name batch-eval-001
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from erl.global_hof import GlobalHallOfFame, LeagueRules
from utils.config import Config
from utils.cloud_sync import get_cloud_sync_from_env


class AgentEvaluator:
    """Evaluates agents and promotes them to Global 50."""

    def __init__(self, run_name: str = "batch-evaluation"):
        """
        Initialize evaluator.

        Args:
            run_name: Name to use for this evaluation batch
        """
        self.run_name = run_name

        print("\n" + "="*70)
        print("Global 50 Agent Evaluator")
        print("="*70)

        # Initialize data loader
        print("\n1. Loading market data...")
        self.data_loader = StockDataLoader()
        data_array, stats = self.data_loader.load_and_prepare()
        self.normalization_stats = stats  # Store for environment creation
        print(f"   Loaded {len(self.data_loader.data_array_full)} days of data")

        # Get validation indices (for gauntlet)
        self.val_start_idx = self.data_loader.val_start_idx
        self.val_end_idx = self.data_loader.val_end_idx
        print(f"   Validation range: {self.val_start_idx} - {self.val_end_idx}")

        # Initialize cloud sync
        print("\n2. Initializing cloud storage...")
        self.cloud_sync = get_cloud_sync_from_env()

        # Initialize Global HoF
        print("\n3. Initializing Global Hall of Fame...")
        league_rules = LeagueRules(context_window_days=Config.CONTEXT_WINDOW_DAYS)
        checkpoint_dir = Path("workspace") / "global50_evaluation"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_hof = GlobalHallOfFame(
            cloud_sync=self.cloud_sync,
            run_name=self.run_name,
            league_rules=league_rules,
            checkpoint_dir=checkpoint_dir,
            disable_global50=False
        )

        print("\n" + "="*70)

    def discover_agents(self, agent_dir: Path) -> List[Path]:
        """
        Discover all agent .pth files in directory.

        Args:
            agent_dir: Directory to search for agents

        Returns:
            List of agent file paths
        """
        print(f"\nDiscovering agents in: {agent_dir}")

        if not agent_dir.exists():
            print(f"   ERROR: Directory does not exist: {agent_dir}")
            return []

        # Find all .pth files
        agent_files = list(agent_dir.glob("*.pth"))

        print(f"   Found {len(agent_files)} agent files")

        if len(agent_files) == 0:
            print("   No .pth files found in directory")
            print("   Make sure the directory contains agent checkpoint files")

        return agent_files

    def generate_gauntlet_slices(self) -> List[Tuple[int, int, int]]:
        """
        Generate 20+ rigorous validation slices for Gauntlet stress test.
        EXACTLY mirrors ERLTrainer.generate_gauntlet_slices()

        Samples slices from BOTH training and validation data to ensure the agent
        performs robustly across all market regimes, not just validation period.

        Strategy:
        - 10 slices from training data (different market conditions)
        - 10 slices from validation data (out-of-sample)

        Returns:
            List of tuples: (start_idx, end_idx, trading_end_idx)
        """
        slices = []
        train_start_idx = 0  # Training starts at beginning
        train_end_idx = self.val_start_idx  # Training ends where validation starts

        # 1. Sample 10 slices from training data
        min_start_train = train_start_idx
        max_start_train = train_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start_train >= min_start_train:
            # Divide training range into 10 segments
            train_range = max_start_train - min_start_train + 1
            train_segment_size = max(1, train_range // 10)

            for i in range(10):
                segment_start = min_start_train + (i * train_segment_size)
                segment_end = min(max_start_train + 1, segment_start + train_segment_size)

                if segment_end > segment_start:
                    start_idx = np.random.randint(segment_start, segment_end)
                    end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

                    slices.append((start_idx, end_idx, trading_end_idx))

        # 2. Sample 10 slices from validation data
        min_start_val = self.val_start_idx
        max_start_val = self.val_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start_val >= min_start_val:
            # Divide validation range into 10 segments
            val_range = max_start_val - min_start_val + 1
            val_segment_size = max(1, val_range // 10)

            for i in range(10):
                segment_start = min_start_val + (i * val_segment_size)
                segment_end = min(max_start_val + 1, segment_start + val_segment_size)

                if segment_end > segment_start:
                    start_idx = np.random.randint(segment_start, segment_end)
                    end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

                    slices.append((start_idx, end_idx, trading_end_idx))

        return slices

    def run_gauntlet(self, agent: DDPGAgent, agent_name: str) -> Tuple[float, dict]:
        """
        Run gauntlet validation on an agent - EXACTLY mirroring ERLTrainer's logic.

        This must match the trainer's gauntlet validation precisely to ensure
        Global 50 scores are accurate and consistent.

        Args:
            agent: Agent to evaluate
            agent_name: Name for logging

        Returns:
            Tuple of (gauntlet_score, detailed_metrics)
        """
        # Generate gauntlet slices (10 training + 10 validation, EXACTLY as ERLTrainer does)
        gauntlet_slices = self.generate_gauntlet_slices()
        print(f"   Running Gauntlet ({len(gauntlet_slices)} slices: {sum(1 for s in gauntlet_slices if s[0] < self.val_start_idx)} training + {sum(1 for s in gauntlet_slices if s[0] >= self.val_start_idx)} validation)...")

        # Create persistent eval environment (mirrors ERLTrainer)
        # Note: Environment needs enough data for context window (504 days)
        full_end_idx = len(self.data_loader.data_array_full)
        eval_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            start_idx=Config.CONTEXT_WINDOW_DAYS,  # Minimum for context window
            end_idx=full_end_idx,
            trading_end_idx=Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS,
            data_array_full=self.data_loader.data_array_full,
            consistency_mode=False  # Gauntlet always uses normal mode
        )

        slice_results = []
        all_closed_trades = []

        # Evaluate each slice
        agent.actor.eval()
        agent.critic.eval()

        for i, (start_idx, end_idx, trading_end_idx) in enumerate(gauntlet_slices):
            # Run episode (mirrors run_episode_batched behavior but simpler)
            fitness, episode_info = eval_env.run_episode(
                agent=agent,
                start_idx=start_idx,
                end_idx=end_idx,
                trading_end_idx=trading_end_idx,
                training=False
            )

            # Apply zero-trades gradient (CRITICAL - matches trainer logic)
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

            # Collect closed trades for expectancy
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        # Extract fitness scores
        fitness_scores = [result['fitness'] for result in slice_results]

        # CRITICAL: Exact aggregator as trainer (0.75*mean + 0.25*min)
        mean_score = np.mean(fitness_scores)
        min_score = np.min(fitness_scores)
        max_score = np.max(fitness_scores)
        gauntlet_score = (0.75 * mean_score) + (0.25 * min_score)

        # Aggregate metrics (exact calculations as trainer)
        total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
        total_investment = sum([r['total_investment'] for r in slice_results])
        roi = (total_raw_pnl / total_investment * 100) if total_investment > 0 else 0.0

        total_wins = sum([r['num_wins'] for r in slice_results])
        total_losses = sum([r['num_losses'] for r in slice_results])
        total_trades = total_wins + total_losses
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        # Calculate expectancy (mirrors trainer's calculate_expectancy)
        if all_closed_trades:
            avg_gain = np.mean([t['reward_pct'] for t in all_closed_trades if t['reward_pct'] > 0]) if any(t['reward_pct'] > 0 for t in all_closed_trades) else 0.0
            avg_loss = np.mean([abs(t['reward_pct']) for t in all_closed_trades if t['reward_pct'] < 0]) if any(t['reward_pct'] < 0 for t in all_closed_trades) else 0.0
            win_rate_decimal = win_rate / 100.0
            expectancy = (win_rate_decimal * avg_gain) - ((1 - win_rate_decimal) * avg_loss)
        else:
            expectancy = 0.0

        detailed_metrics = {
            'gauntlet_score': gauntlet_score,
            'mean_fitness': mean_score,
            'min_fitness': min_score,
            'max_fitness': max_score,
            'roi': roi,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'num_slices': num_slices,
            'fitness_all_slices': fitness_scores
        }

        return gauntlet_score, detailed_metrics

    def evaluate_agent(self, agent_path: Path, generation: int = 0) -> dict:
        """
        Evaluate a single agent for Global 50.

        Args:
            agent_path: Path to agent .pth file
            generation: Generation number (default 0 for external agents)

        Returns:
            Dictionary with evaluation results
        """
        agent_name = agent_path.stem

        print(f"\n{'='*70}")
        print(f"Evaluating: {agent_name}")
        print(f"{'='*70}")

        result = {
            'agent_name': agent_name,
            'agent_path': str(agent_path),
            'success': False,
            'promoted': False,
            'error': None
        }

        try:
            # Load agent
            print(f"Loading agent...")
            agent = DDPGAgent(agent_id=0)
            agent.load(str(agent_path))
            print(f"   Agent loaded successfully")

            # Run gauntlet
            print(f"Running gauntlet ({Config.GAUNTLET_NUM_SLICES} slices)...")
            gauntlet_score, metrics = self.run_gauntlet(agent, agent_name)

            print(f"\nGauntlet Results:")
            print(f"   Gauntlet Score:    {gauntlet_score:>10.2f}")
            print(f"   Mean Fitness:      {metrics['mean_fitness']:>10.2f}")
            print(f"   Min Fitness:       {metrics['min_fitness']:>10.2f}")
            print(f"   ROI:               {metrics['roi']:>10.2f}%")
            print(f"   Total Trades:      {metrics['total_trades']:>10}")
            print(f"   Win Rate:          {metrics['win_rate']:>10.1f}%")
            print(f"   Expectancy:        {metrics['expectancy']:>10.2f}")

            result.update(metrics)
            result['success'] = True

            # Check if qualifies for Global 50
            if self.global_hof.should_promote(gauntlet_score):
                print(f"\n   Agent QUALIFIES for Global 50!")
                print(f"   Threshold: {self.global_hof.entry_threshold:.2f}")
                print(f"   Attempting promotion...")

                # Attempt promotion
                promoted = self.global_hof.check_and_promote(
                    agent=agent,
                    gauntlet_score=gauntlet_score,
                    generation=generation,
                    roi=metrics['roi'],
                    expectancy=metrics['expectancy'],
                    quality_count=metrics['total_trades'],
                    total_trades=metrics['total_trades']
                )

                result['promoted'] = promoted

                if promoted:
                    print(f"   SUCCESS: Agent promoted to Global 50!")
                else:
                    print(f"   WARNING: Promotion failed (concurrent update?)")
            else:
                print(f"\n   Agent does not qualify for Global 50")
                print(f"   Score: {gauntlet_score:.2f} < Threshold: {self.global_hof.entry_threshold:.2f}")

        except Exception as e:
            print(f"\n   ERROR: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)

        return result

    def evaluate_batch(self, agent_dir: Path) -> List[dict]:
        """
        Evaluate all agents in a directory.

        Args:
            agent_dir: Directory containing agent .pth files

        Returns:
            List of evaluation results
        """
        # Discover agents
        agent_files = self.discover_agents(agent_dir)

        if len(agent_files) == 0:
            return []

        # Evaluate each agent
        results = []

        print(f"\n{'='*70}")
        print(f"Evaluating {len(agent_files)} agents")
        print(f"{'='*70}")

        for i, agent_path in enumerate(agent_files, 1):
            print(f"\n[{i}/{len(agent_files)}] Processing {agent_path.name}")

            result = self.evaluate_agent(agent_path, generation=i)
            results.append(result)

        return results

    def print_summary(self, results: List[dict]):
        """
        Print summary of evaluation results.

        Args:
            results: List of evaluation results
        """
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)

        total = len(results)
        successful = sum(1 for r in results if r['success'])
        promoted = sum(1 for r in results if r['promoted'])
        failed = total - successful

        print(f"\nTotal Agents:      {total:>10}")
        print(f"Evaluated:         {successful:>10}")
        print(f"Promoted:          {promoted:>10}")
        print(f"Failed:            {failed:>10}")

        if promoted > 0:
            print(f"\nPromoted Agents:")
            print("-" * 70)
            for r in results:
                if r['promoted']:
                    print(f"  {r['agent_name']:.<50} {r['gauntlet_score']:>10.2f}")

        if self.global_hof.enabled:
            stats = self.global_hof.get_stats()
            print(f"\nGlobal 50 Status:")
            print(f"  Size:              {stats['size']:>10} / 50")
            print(f"  Entry Threshold:   {stats['entry_threshold']:>10.2f}")
            print(f"  Best Score:        {stats['best_score']:>10.2f}")
            print(f"  Mean Score:        {stats['mean_score']:>10.2f}")

        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agents for Global 50 promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup (initialize Global 50 structure)
  python evaluate_for_global50.py --init

  # Evaluate agents from Hall of Fame directory
  python evaluate_for_global50.py --agent-dir checkpoints/azure-thunder-123/hall_of_fame

  # Evaluate with custom run name
  python evaluate_for_global50.py --agent-dir workspace/elite_agents --run-name backfill-2025

  # Evaluate specific run's champions
  python evaluate_for_global50.py --agent-dir checkpoints/crimson-wave-456/hall_of_fame
        """
    )

    parser.add_argument(
        '--agent-dir',
        type=str,
        required=False,
        help='Directory containing agent .pth files to evaluate'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default='batch-evaluation',
        help='Run name to use for this evaluation batch (default: batch-evaluation)'
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize Global 50 structure (first-time setup). Creates empty global50.json and validates cloud sync.'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = AgentEvaluator(run_name=args.run_name)

    # Handle --init mode (first-time setup)
    if args.init:
        print("\n" + "="*70)
        print("INITIALIZATION MODE - First-Time Setup")
        print("="*70)

        if evaluator.global_hof.enabled:
            print("\n✓ Global 50 initialized successfully!")
            print(f"  Cloud Provider: {evaluator.cloud_sync.provider}")
            print(f"  Bucket: {evaluator.cloud_sync.bucket_name}")
            print(f"  Project: {evaluator.cloud_sync.project_name}")

            stats = evaluator.global_hof.get_stats()
            print(f"\n  Current Size: {stats['size']}/50")
            print(f"  Entry Threshold: {stats['entry_threshold']}")

            print("\nGlobal 50 structure created:")
            print(f"  Local:  {evaluator.global_hof.local_dir}")
            print(f"  Cloud:  gs://{evaluator.cloud_sync.bucket_name}/{evaluator.cloud_sync.project_name}/global50/")

            print("\n✓ Setup complete! You can now run evaluations.")
            print("\nNext step:")
            print(f"  python evaluate_for_global50.py --agent-dir <path>")
        else:
            print("\n✗ Initialization FAILED")
            print(f"  Cloud Provider: {evaluator.cloud_sync.provider}")
            print("\nPlease check your environment variables:")
            print("  - CLOUD_PROVIDER=gcs")
            print("  - CLOUD_BUCKET=<your-bucket>")
            print("  - GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials>")

        print("\n" + "="*70)
        return

    # Require agent_dir for evaluation mode
    if not args.agent_dir:
        print("Error: --agent-dir is required (or use --init for first-time setup)")
        parser.print_help()
        return

    # Convert to Path
    agent_dir = Path(args.agent_dir)

    # Evaluate agents
    results = evaluator.evaluate_batch(agent_dir)

    # Print summary
    if results:
        evaluator.print_summary(results)
    else:
        print("\nNo agents to evaluate. Exiting.")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
