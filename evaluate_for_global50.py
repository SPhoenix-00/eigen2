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
from tqdm import tqdm

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
        self.data_loader = StockDataLoader(Config.DATA_PATH)
        print(f"   Loaded {len(self.data_loader.data)} days of data")

        # Get validation indices (for gauntlet)
        self.val_start_idx = self.data_loader.train_end_idx
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

        if not self.global_hof.enabled:
            print("\n   WARNING: Global 50 is DISABLED")
            print("   Reasons could be:")
            print("   - Cloud sync is set to 'local' (set CLOUD_PROVIDER=gcs)")
            print("   - League rules don't match existing Global 50")
            print("\n   Evaluation will run, but agents won't be promoted.")
        else:
            print(f"   Global 50 Status: ENABLED")
            print(f"   Current Size: {len(self.global_hof.entries)}/50")
            print(f"   Entry Threshold: {self.global_hof.entry_threshold:.2f}")

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

    def run_gauntlet(self, agent: DDPGAgent, agent_name: str) -> Tuple[float, dict]:
        """
        Run gauntlet validation on an agent.

        Args:
            agent: Agent to evaluate
            agent_name: Name for logging

        Returns:
            Tuple of (gauntlet_score, detailed_metrics)
        """
        num_slices = Config.GAUNTLET_NUM_SLICES
        slice_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

        # Calculate available validation window
        val_window_size = self.val_end_idx - self.val_start_idx
        max_slices = val_window_size // slice_length

        if max_slices < num_slices:
            print(f"   WARNING: Only {max_slices} slices possible, requested {num_slices}")
            num_slices = max_slices

        if num_slices == 0:
            print(f"   ERROR: Validation window too small for gauntlet")
            return 0.0, {}

        # Generate uniformly distributed slice start points
        slice_starts = np.linspace(
            self.val_start_idx,
            self.val_end_idx - slice_length,
            num=num_slices,
            dtype=int
        )

        fitness_scores = []
        roi_values = []
        total_trades_list = []
        win_counts = []

        # Evaluate each slice
        agent.actor.eval()
        agent.critic.eval()

        for i, start_idx in enumerate(slice_starts):
            end_idx = start_idx + slice_length

            # Create environment for this slice
            env = TradingEnvironment(
                data=self.data_loader.data,
                context_window_days=Config.CONTEXT_WINDOW_DAYS,
                min_holding_period=Config.MIN_HOLDING_PERIOD,
                max_holding_period=Config.MAX_HOLDING_PERIOD,
                trading_period_days=Config.TRADING_PERIOD_DAYS,
                settlement_period_days=Config.SETTLEMENT_PERIOD_DAYS,
                loss_penalty_multiplier=1.0,  # Normal mode for gauntlet
                consistency_mode=False
            )

            # Run episode
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
            _, episode_info = env.run_episode(
                agent=agent,
                start_idx=start_idx,
                end_idx=end_idx,
                trading_end_idx=trading_end_idx,
                training=False
            )

            fitness_scores.append(episode_info['fitness'])
            roi_values.append(episode_info['roi'])
            total_trades_list.append(episode_info['total_trades'])
            win_counts.append(episode_info['total_wins'])

        # Compute gauntlet score (pessimistic aggregator)
        mean_score = np.mean(fitness_scores)
        min_score = np.min(fitness_scores)
        gauntlet_score = (0.75 * mean_score) + (0.25 * min_score)

        # Aggregate metrics
        mean_roi = np.mean(roi_values)
        total_trades = sum(total_trades_list)
        total_wins = sum(win_counts)
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        # Calculate expectancy
        if total_trades > 0:
            expectancy = mean_roi * (win_rate / 100.0)
        else:
            expectancy = 0.0

        detailed_metrics = {
            'gauntlet_score': gauntlet_score,
            'mean_fitness': mean_score,
            'min_fitness': min_score,
            'roi': mean_roi,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'num_slices': num_slices
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
        required=True,
        help='Directory containing agent .pth files to evaluate'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default='batch-evaluation',
        help='Run name to use for this evaluation batch (default: batch-evaluation)'
    )

    args = parser.parse_args()

    # Convert to Path
    agent_dir = Path(args.agent_dir)

    # Initialize evaluator
    evaluator = AgentEvaluator(run_name=args.run_name)

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
