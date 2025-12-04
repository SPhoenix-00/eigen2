"""
Evaluation script for best agent and all Hall of Fame agents from last training run.

This script:
1. Loads the best agent from the last run (using last_run.json or specified run name)
2. Loads all agents from the Hall of Fame
3. Evaluates each agent on 10 validation slices (as done in training: 4 from quarters + 3 straddling + 3 random)
4. Outputs summary comparison of all agents (no detailed trades file)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import argparse
import sys

from data.loader import StockDataLoader
from models.ddpg_agent import DDPGAgent
from environment.trading_env import TradingEnvironment
from utils.config import Config
from utils.cloud_sync import get_cloud_sync_from_env
from erl.hall_of_fame import HallOfFame, HallOfFameEntry


class HallOfFameEvaluator:
    """Evaluates best agent and all Hall of Fame agents with summary reports."""

    def __init__(self, run_name: str = None):
        """
        Initialize evaluator.

        Args:
            run_name: Name of the run to evaluate. If None, uses last_run.json
        """
        self.run_name = run_name or self._get_last_run_name()
        if not self.run_name:
            raise ValueError("No run name specified and last_run.json not found")

        print(f"\n{'='*80}")
        print(f"Evaluating Hall of Fame from Run: {self.run_name}")
        print(f"{'='*80}\n")

        # Initialize cloud sync
        self.cloud_sync = get_cloud_sync_from_env()

        # Load data
        print("Loading data...")
        self.data_loader = StockDataLoader()
        self.data_loader.load_and_prepare()

        # Extract data components
        self.data_array = self.data_loader.data_array
        self.data_array_full = self.data_loader.data_array_full
        self.dates = self.data_loader.dates
        self.norm_stats = self.data_loader.normalization_stats
        self.train_start = 0
        self.train_end = self.data_loader.train_end_idx
        self.val_start = self.data_loader.val_start_idx
        self.val_end = self.data_loader.val_end_idx  # Excludes holdout period

        # Get stock column names
        self._load_stock_names()

        print(f"Data loaded: {len(self.data_array)} days")
        print(f"  Training: days 0-{self.train_end}")
        print(f"  Validation: days {self.val_start}-{self.val_end}")

        # Set up checkpoint directory
        self.checkpoint_dir = Path("checkpoints") / self.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Download from cloud if not present locally
        best_agent_path = self.checkpoint_dir / "best_agent.pth"
        if not best_agent_path.exists():
            print("  Downloading from cloud...")
            self.cloud_sync.download_checkpoints(str(self.checkpoint_dir))

        # Load agents
        print(f"\nLoading agents from run: {self.run_name}")
        self.agents = self._load_all_agents()

        # Storage for results
        self.agent_results = {}  # agent_name -> list of slice summaries

    def _get_last_run_name(self) -> str:
        """Get run name from last_run.json."""
        last_run_path = Path("last_run.json")
        if not last_run_path.exists():
            return None

        with open(last_run_path, 'r') as f:
            data = json.load(f)
            return data.get('run_name')

    def _load_stock_names(self):
        """Load stock ticker names from the data."""
        df = pd.read_pickle(Config.DATA_PATH)
        all_columns = df.columns.tolist()
        self.stock_names = all_columns[Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL + 1]
        self.non_investable_names = all_columns[:Config.INVESTABLE_START_COL]
        print(f"Loaded {len(self.stock_names)} investable stock tickers")

    def _load_all_agents(self) -> Dict[str, DDPGAgent]:
        """Load best agent and all Hall of Fame agents."""
        agents = {}

        # Load best agent
        best_agent_path = self.checkpoint_dir / "best_agent.pth"
        if best_agent_path.exists():
            agent = DDPGAgent(agent_id='best')
            agent.load(str(best_agent_path))
            agents['Best Agent'] = agent
            print(f"  Loaded Best Agent from {best_agent_path}")
        else:
            print(f"  WARNING: Best agent not found at {best_agent_path}")

        # Load Hall of Fame agents
        hof_dir = self.checkpoint_dir / "hall_of_fame"
        hof_metadata_path = hof_dir / "hall_of_fame.json"

        if hof_metadata_path.exists():
            with open(hof_metadata_path, 'r') as f:
                metadata = json.load(f)

            entries = [HallOfFameEntry.from_dict(e) for e in metadata['entries']]

            # Sort by validation score (best first)
            entries_sorted = sorted(entries, key=lambda e: e.validation_score, reverse=True)

            for rank, entry in enumerate(entries_sorted, 1):
                agent_path = hof_dir / f"hof_agent_{entry.agent_id}.pth"
                if agent_path.exists():
                    agent = DDPGAgent(agent_id=entry.agent_id)
                    agent.load(str(agent_path))
                    agent_name = f"HoF #{rank} (Gen {entry.generation}, Val: {entry.validation_score:.2f})"
                    agents[agent_name] = agent
                    print(f"  Loaded {agent_name}")
                else:
                    print(f"  WARNING: HoF agent {entry.agent_id} not found at {agent_path}")

            print(f"Loaded {len(entries_sorted)} Hall of Fame agents")
        else:
            print(f"  No Hall of Fame found at {hof_metadata_path}")

        print(f"\nTotal agents to evaluate: {len(agents)}")
        return agents

    def generate_validation_slices(self) -> List[Tuple[int, int, int]]:
        """Generate 10 validation slices from validation set (4 from quarters + 3 straddling + 3 random)."""
        # Set seed for reproducibility
        np.random.seed(42)

        min_start = self.val_start
        max_start = self.val_end - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start < min_start:
            raise ValueError(f"Not enough validation data")

        # Divide the validation range into 4 equal segments
        total_range = max_start - min_start + 1
        segment_size = total_range // 4

        slices = []

        # 1. Sample one slice from each quarter (4 slices)
        for segment_idx in range(4):
            # Calculate segment boundaries
            segment_start = min_start + (segment_idx * segment_size)
            # For the last segment, extend to max_start to avoid rounding issues
            segment_end = max_start + 1 if segment_idx == 3 else min_start + ((segment_idx + 1) * segment_size)

            # Sample one random start index from this segment
            start_idx = np.random.randint(segment_start, segment_end)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
            slices.append((start_idx, end_idx, trading_end_idx))

        # 2. Sample straddling slices between quarters (3 slices)
        for straddle_idx in range(3):
            # Define straddling region: from halfway through quarter N to halfway through quarter N+1
            straddle_start = min_start + (segment_size // 2) + (straddle_idx * segment_size)
            straddle_end = min_start + (segment_size // 2) + ((straddle_idx + 1) * segment_size)

            # Ensure we don't exceed max_start
            straddle_end = min(straddle_end, max_start + 1)

            # Sample one random start index from this straddling region
            if straddle_end > straddle_start:
                start_idx = np.random.randint(straddle_start, straddle_end)
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
                slices.append((start_idx, end_idx, trading_end_idx))

        # 3. Sample 3 completely random slices from entire validation range (3 slices)
        for _ in range(3):
            start_idx = np.random.randint(min_start, max_start + 1)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
            slices.append((start_idx, end_idx, trading_end_idx))

        return slices


    def run_episode(self, agent: DDPGAgent, start_idx: int, end_idx: int,
                   trading_end_idx: int) -> Tuple[float, Dict]:
        """
        Run one evaluation episode.

        Args:
            agent: DDPGAgent to evaluate
            start_idx: Start index for trading
            end_idx: End index (includes settlement period)
            trading_end_idx: Last day new positions can be opened

        Returns:
            Tuple of (fitness, summary_stats)
        """
        env = TradingEnvironment(
            data_array=self.data_array,
            dates=self.dates,
            normalization_stats=self.norm_stats,
            start_idx=start_idx,
            end_idx=end_idx,
            trading_end_idx=trading_end_idx,
            data_array_full=self.data_array_full,
            is_training=False
        )

        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        summary = env.get_episode_summary()

        fitness = summary['total_reward']
        if summary['num_trades'] == 0:
            fitness -= summary['zero_trades_penalty']

        return fitness, summary

    def evaluate_all_agents(self):
        """Run evaluation on all agents across all slices."""
        print(f"\n{'='*80}")
        print("EVALUATION PROCESS")
        print(f"{'='*80}\n")

        # Generate all slices
        val_slices = self.generate_validation_slices()

        all_slices = [
            ("Val_1", val_slices[0]),
            ("Val_2", val_slices[1]),
            ("Val_3", val_slices[2]),
            ("Val_4", val_slices[3]),
            ("Val_5", val_slices[4]),
            ("Val_6", val_slices[5]),
            ("Val_7", val_slices[6]),
            ("Val_8", val_slices[7]),
            ("Val_9", val_slices[8]),
            ("Val_10", val_slices[9]),
        ]

        # Evaluate each agent
        for agent_name, agent in self.agents.items():
            print(f"\nEvaluating: {agent_name}")
            print("-" * 60)

            self.agent_results[agent_name] = []

            for slice_name, (start, end, trading_end) in all_slices:
                fitness, summary = self.run_episode(agent, start, end, trading_end)

                result = {
                    'slice_name': slice_name,
                    'start_date': self.dates[start],
                    'end_date': self.dates[end-1],
                    'fitness': fitness,
                    **summary
                }
                self.agent_results[agent_name].append(result)

                print(f"  {slice_name}: Fitness={fitness:.2f}, "
                      f"P&L=${summary.get('raw_pnl', 0.0):.2f}, "
                      f"Trades={summary['num_trades']}, "
                      f"Win Rate={summary['win_rate']*100:.1f}%, "
                      f"ROI={summary.get('roi', 0.0):.2f}%")

    def export_results(self):
        """Export summary results to text and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)

        text_file = output_dir / f"hof_evaluation_{self.run_name}_{timestamp}.txt"
        summary_csv = output_dir / f"hof_summary_{self.run_name}_{timestamp}.csv"

        self._export_text_report(text_file)
        self._export_summary_csv(summary_csv)

        print(f"\n{'='*80}")
        print("RESULTS EXPORTED")
        print(f"{'='*80}")
        print(f"Detailed Report: {text_file}")
        print(f"Summary CSV: {summary_csv}")
        print(f"{'='*80}\n")

    def _export_text_report(self, filepath: Path):
        """Export detailed text report."""
        with open(filepath, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"HALL OF FAME EVALUATION REPORT\n")
            f.write(f"Run: {self.run_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Agents Evaluated: {len(self.agents)}\n")
            f.write("="*80 + "\n\n")

            # Summary table by agent
            f.write("SUMMARY BY AGENT\n")
            f.write("-"*80 + "\n\n")

            # Calculate averages for ranking
            agent_averages = []
            for agent_name, results in self.agent_results.items():
                avg_fitness = np.mean([r['fitness'] for r in results])
                total_trades = sum(r['num_trades'] for r in results)
                total_wins = sum(r['num_wins'] for r in results)
                overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
                total_raw_pnl = sum(r.get('raw_pnl', 0.0) for r in results)
                total_investment = sum(r.get('total_investment', 0.0) for r in results)
                overall_roi = (total_raw_pnl / total_investment * 100) if total_investment > 0 else 0.0

                agent_averages.append({
                    'name': agent_name,
                    'avg_fitness': avg_fitness,
                    'total_trades': total_trades,
                    'win_rate': overall_win_rate,
                    'total_pnl': total_raw_pnl,
                    'overall_roi': overall_roi
                })

            # Sort by average fitness
            agent_averages.sort(key=lambda x: x['avg_fitness'], reverse=True)

            # Print ranking
            f.write("OVERALL RANKING (by Average Fitness)\n")
            f.write("-"*80 + "\n")
            for rank, agent in enumerate(agent_averages, 1):
                f.write(f"{rank}. {agent['name']}\n")
                f.write(f"   Avg Fitness: {agent['avg_fitness']:.2f}\n")
                f.write(f"   Total Trades: {agent['total_trades']}, Win Rate: {agent['win_rate']*100:.1f}%\n")
                f.write(f"   Total P&L: ${agent['total_pnl']:.2f}, Overall ROI: {agent['overall_roi']:.2f}%\n")
                f.write("\n")

            # Detailed results for each agent
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS BY AGENT\n")
            f.write("="*80 + "\n\n")

            for agent_name, results in self.agent_results.items():
                f.write(f"\n{agent_name}\n")
                f.write("-"*60 + "\n")

                for result in results:
                    f.write(f"  {result['slice_name']}:\n")
                    f.write(f"    Period: {result['start_date']} to {result['end_date']}\n")
                    f.write(f"    Fitness: {result['fitness']:.2f}\n")
                    f.write(f"    Raw P&L: ${result.get('raw_pnl', 0.0):.2f}, ROI: {result.get('roi', 0.0):.2f}%\n")
                    f.write(f"    Trades: {result['num_trades']}, Wins: {result['num_wins']}, Losses: {result['num_losses']}\n")
                    f.write(f"    Win Rate: {result['win_rate']*100:.1f}%\n")
                    f.write(f"    Avg Reward/Trade: {result['avg_reward_per_trade']:.2f}\n")
                    f.write("\n")

            # Comparison table
            f.write("\n" + "="*80 + "\n")
            f.write("COMPARISON TABLE (Fitness by Slice)\n")
            f.write("="*80 + "\n\n")

            # Header
            slice_names = ['Val_1', 'Val_2', 'Val_3', 'Val_4', 'Val_5', 'Val_6', 'Val_7', 'Val_8', 'Val_9', 'Val_10', 'Average']
            header = f"{'Agent':<50} " + " ".join([f"{s:>12}" for s in slice_names])
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Data rows
            for agent_name, results in self.agent_results.items():
                fitness_values = [r['fitness'] for r in results]
                avg_fitness = np.mean(fitness_values)

                row = f"{agent_name:<50} "
                row += " ".join([f"{f:>12.2f}" for f in fitness_values])
                row += f" {avg_fitness:>12.2f}"
                f.write(row + "\n")

        print(f"Text report saved to {filepath}")

    def _export_summary_csv(self, filepath: Path):
        """Export summary statistics to CSV."""
        rows = []

        for agent_name, results in self.agent_results.items():
            for result in results:
                row = {
                    'agent_name': agent_name,
                    'slice_name': result['slice_name'],
                    'start_date': result['start_date'],
                    'end_date': result['end_date'],
                    'fitness': result['fitness'],
                    'num_trades': result['num_trades'],
                    'num_wins': result['num_wins'],
                    'num_losses': result['num_losses'],
                    'win_rate': result['win_rate'],
                    'total_reward': result['total_reward'],
                    'avg_reward_per_trade': result['avg_reward_per_trade'],
                    'raw_pnl': result.get('raw_pnl', 0.0),
                    'roi': result.get('roi', 0.0),
                    'total_investment': result.get('total_investment', 0.0),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Summary CSV saved to {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate best agent and Hall of Fame agents from a training run"
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name of the run to evaluate. If not specified, uses last_run.json'
    )

    args = parser.parse_args()

    try:
        evaluator = HallOfFameEvaluator(run_name=args.run_name)
        evaluator.evaluate_all_agents()
        evaluator.export_results()

        print("\n" + "="*80)
        print("HALL OF FAME EVALUATION COMPLETE!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
