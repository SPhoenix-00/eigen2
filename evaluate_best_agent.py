"""
Evaluation script for best agent from last training run.

This script:
1. Loads the best agent from the last run (using last_run.json or specified run name)
2. Evaluates on 3 validation slices (as done in training)
3. Evaluates on first 125 days of holdout period
4. Outputs detailed trade information including:
   - Stock purchased, entry/exit dates, prices
   - Whether it was an active sell or automatic liquidation
5. Exports results to text file and CSV
6. Provides summary of fitness, number of trades, and win rate
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


class AgentEvaluator:
    """Evaluates best agent and generates detailed trade reports."""

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
        print(f"Evaluating Best Agent from Run: {self.run_name}")
        print(f"{'='*80}\n")

        # Initialize cloud sync
        self.cloud_sync = get_cloud_sync_from_env()

        # Load data
        print("Loading data...")
        self.data_loader = StockDataLoader()
        self.data_loader.load_and_prepare()

        # Extract data components
        self.data_array = self.data_loader.data_array
        self.dates = self.data_loader.dates
        self.norm_stats = self.data_loader.normalization_stats
        self.train_start = 0
        self.train_end = self.data_loader.train_end_idx
        self.interim_val_start = self.data_loader.interim_val_start_idx
        self.interim_val_end = self.data_loader.interim_val_end_idx
        self.holdout_start = self.data_loader.val_start_idx
        self.holdout_end = len(self.data_array)

        # Get stock column names
        self._load_stock_names()

        print(f"✓ Data loaded: {len(self.data_array)} days")
        print(f"  Training: days 0-{self.train_end}")
        print(f"  Interim Validation: days {self.interim_val_start}-{self.interim_val_end}")
        print(f"  Holdout: days {self.holdout_start}-{self.holdout_end}")

        # Load best agent
        print(f"\nLoading best agent from run: {self.run_name}")
        self.agent = self._load_best_agent()

        # Storage for results
        self.all_trades = []
        self.slice_summaries = []

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
        # Load the pickle to get column names
        df = pd.read_pickle(Config.DATA_PATH)
        all_columns = df.columns.tolist()

        # Extract investable stock names (columns 10-117)
        self.stock_names = all_columns[Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL + 1]

        # Also store non-investable column names for reference
        self.non_investable_names = all_columns[:Config.INVESTABLE_START_COL]

        print(f"✓ Loaded {len(self.stock_names)} investable stock tickers")

    def _load_best_agent(self) -> DDPGAgent:
        """Download and load best agent from GCP."""
        # Set up local checkpoint directory
        checkpoint_dir = Path("checkpoints") / self.run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Download from cloud if not present locally
        best_agent_path = checkpoint_dir / "best_agent.pth"
        if not best_agent_path.exists():
            print("  Downloading from cloud...")
            self.cloud_sync.download_checkpoints(str(checkpoint_dir))

        if not best_agent_path.exists():
            raise FileNotFoundError(f"Best agent not found at {best_agent_path}")

        # Load agent
        agent = DDPGAgent(agent_id='best')
        agent.load(str(best_agent_path))
        print(f"✓ Best agent loaded from {best_agent_path}")

        return agent

    def generate_validation_slices(self) -> List[Tuple[int, int, int]]:
        """
        Generate 3 random validation slices from interim validation set.

        Returns:
            List of 3 tuples: (start_idx, end_idx, trading_end_idx)
        """
        # Set seed for reproducibility
        np.random.seed(42)

        min_start = self.interim_val_start
        max_start = self.interim_val_end - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start < min_start:
            raise ValueError(f"Not enough interim validation data")

        slices = []
        for i in range(3):
            start_idx = np.random.randint(min_start, max_start + 1)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
            slices.append((start_idx, end_idx, trading_end_idx))

        return slices

    def get_holdout_slice(self) -> Tuple[int, int, int]:
        """
        Get the first 125 days of holdout period.

        Returns:
            Tuple: (start_idx, end_idx, trading_end_idx)
        """
        start_idx = self.holdout_start
        end_idx = min(start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS,
                      self.holdout_end)
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

        return (start_idx, end_idx, trading_end_idx)

    def run_episode(self, start_idx: int, end_idx: int, trading_end_idx: int,
                   slice_name: str) -> Tuple[float, Dict, List[Dict]]:
        """
        Run one evaluation episode and collect trade details.

        Args:
            start_idx: Start index for trading
            end_idx: End index (includes settlement period)
            trading_end_idx: Last day new positions can be opened
            slice_name: Name of this slice for reporting

        Returns:
            Tuple of (fitness, summary_stats, trade_list)
        """
        # Create environment
        env = TradingEnvironment(
            data_array=self.data_array,
            dates=self.dates,
            normalization_stats=self.norm_stats,
            start_idx=start_idx,
            end_idx=end_idx,
            trading_end_idx=trading_end_idx
        )

        # Reset environment
        obs, _ = env.reset()
        done = False

        # Run episode
        while not done:
            # Agent selects action (no exploration noise)
            action = self.agent.select_action(obs, add_noise=False)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Get episode summary
        summary = env.get_episode_summary()

        # Calculate fitness (same as in training)
        fitness = summary['total_reward']
        fitness -= summary['inaction_penalty_applied']
        if summary['num_trades'] == 0:
            fitness -= summary['zero_trades_penalty']

        # Process trades from episode_actions
        trades = self._extract_trades(env.episode_actions, slice_name)

        return fitness, summary, trades

    def _extract_trades(self, episode_actions: List[Dict], slice_name: str) -> List[Dict]:
        """
        Extract trade information from episode actions.

        Args:
            episode_actions: List of action dictionaries from environment
            slice_name: Name of the evaluation slice

        Returns:
            List of trade dictionaries with detailed information
        """
        trades = []
        open_positions = {}

        for action in episode_actions:
            if action['action'] == 'open':
                # Record opening
                stock_id = action['stock_id']
                open_positions[stock_id] = {
                    'slice': slice_name,
                    'stock_id': stock_id,
                    'stock_ticker': self.stock_names[stock_id],
                    'entry_date': action['day'],
                    'entry_price': action['entry_price'],
                    'coefficient': action['coefficient'],
                    'sale_target_pct': action['sale_target_pct'],
                    'sale_target_price': action['sale_target_price'],
                }

            elif action['action'] == 'close':
                # Complete the trade record
                stock_id = action['stock_id']
                if stock_id in open_positions:
                    trade = open_positions.pop(stock_id)
                    trade.update({
                        'exit_date': action['day'],
                        'exit_price': action['exit_price'],
                        'days_held': action['days_held'],
                        'gain_pct': action['gain_pct'],
                        'reward': action['reward'],
                        'exit_reason': action['reason'],
                        'is_win': action['gain_pct'] >= 0,
                        'is_active_sell': action['reason'] == 'target_hit',
                        'is_auto_liquidation': action['reason'] in ['max_holding_period', 'stock_delisted']
                    })
                    trades.append(trade)

        return trades

    def evaluate_all_slices(self):
        """Run evaluation on all validation slices and holdout."""
        print(f"\n{'='*80}")
        print("EVALUATION PROCESS")
        print(f"{'='*80}\n")

        # Generate validation slices
        val_slices = self.generate_validation_slices()

        # Evaluate on validation slices
        print("Evaluating on 3 Validation Slices...")
        print("-" * 80)

        for i, (start, end, trading_end) in enumerate(val_slices, 1):
            slice_name = f"Validation_Slice_{i}"
            print(f"\n{slice_name}:")
            print(f"  Period: {self.dates[start]} to {self.dates[end-1]}")
            print(f"  Trading days: {trading_end - start}, Settlement days: {end - trading_end}")

            fitness, summary, trades = self.run_episode(start, end, trading_end, slice_name)

            # Store results
            self.all_trades.extend(trades)
            self.slice_summaries.append({
                'slice_name': slice_name,
                'start_date': self.dates[start],
                'end_date': self.dates[end-1],
                'fitness': fitness,
                **summary
            })

            print(f"  Fitness: {fitness:.2f}")
            print(f"  Trades: {summary['num_trades']}, Win Rate: {summary['win_rate']*100:.1f}%")

        # Evaluate on holdout
        print(f"\n{'='*80}")
        print("Evaluating on Holdout Period (First 125 Days)...")
        print("-" * 80)

        start, end, trading_end = self.get_holdout_slice()
        slice_name = "Holdout"
        print(f"\n{slice_name}:")
        print(f"  Period: {self.dates[start]} to {self.dates[end-1]}")
        print(f"  Trading days: {trading_end - start}, Settlement days: {end - trading_end}")

        fitness, summary, trades = self.run_episode(start, end, trading_end, slice_name)

        # Store results
        self.all_trades.extend(trades)
        self.slice_summaries.append({
            'slice_name': slice_name,
            'start_date': self.dates[start],
            'end_date': self.dates[end-1],
            'fitness': fitness,
            **summary
        })

        print(f"  Fitness: {fitness:.2f}")
        print(f"  Trades: {summary['num_trades']}, Win Rate: {summary['win_rate']*100:.1f}%")

    def export_results(self):
        """Export results to text and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)

        # Generate file names
        text_file = output_dir / f"evaluation_{self.run_name}_{timestamp}.txt"
        csv_file = output_dir / f"trades_{self.run_name}_{timestamp}.csv"
        summary_csv = output_dir / f"summary_{self.run_name}_{timestamp}.csv"

        # Export detailed text report
        self._export_text_report(text_file)

        # Export trades to CSV
        self._export_trades_csv(csv_file)

        # Export summary to CSV
        self._export_summary_csv(summary_csv)

        print(f"\n{'='*80}")
        print("RESULTS EXPORTED")
        print(f"{'='*80}")
        print(f"Detailed Report: {text_file}")
        print(f"Trades CSV: {csv_file}")
        print(f"Summary CSV: {summary_csv}")
        print(f"{'='*80}\n")

    def _export_text_report(self, filepath: Path):
        """Export detailed text report."""
        with open(filepath, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"AGENT EVALUATION REPORT\n")
            f.write(f"Run: {self.run_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Summary by slice
            f.write("SUMMARY BY EVALUATION SLICE\n")
            f.write("-"*80 + "\n\n")

            for summary in self.slice_summaries:
                f.write(f"{summary['slice_name']}:\n")
                f.write(f"  Period: {summary['start_date']} to {summary['end_date']}\n")
                f.write(f"  Fitness: {summary['fitness']:.2f}\n")
                f.write(f"  Total Trades: {summary['num_trades']}\n")
                f.write(f"  Wins: {summary['num_wins']}, Losses: {summary['num_losses']}\n")
                f.write(f"  Win Rate: {summary['win_rate']*100:.1f}%\n")
                f.write(f"  Total Reward: {summary['total_reward']:.2f}\n")
                f.write(f"  Avg Reward per Trade: {summary['avg_reward_per_trade']:.2f}\n")
                f.write(f"  Days with Positions: {summary['days_with_positions']}\n")
                f.write(f"  Days without Positions: {summary['days_without_positions']}\n")
                f.write(f"  Inaction Penalty: {summary['inaction_penalty_applied']:.2f}\n")
                f.write("\n")

            # Overall statistics
            f.write("="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n\n")

            total_trades = sum(s['num_trades'] for s in self.slice_summaries)
            total_wins = sum(s['num_wins'] for s in self.slice_summaries)
            total_losses = sum(s['num_losses'] for s in self.slice_summaries)
            avg_fitness = np.mean([s['fitness'] for s in self.slice_summaries])
            overall_win_rate = total_wins / total_trades if total_trades > 0 else 0

            f.write(f"Total Trades: {total_trades}\n")
            f.write(f"Total Wins: {total_wins}\n")
            f.write(f"Total Losses: {total_losses}\n")
            f.write(f"Overall Win Rate: {overall_win_rate*100:.1f}%\n")
            f.write(f"Average Fitness: {avg_fitness:.2f}\n")
            fitness_values = [f"{s['fitness']:.2f}" for s in self.slice_summaries]
            f.write(f"Fitness by Slice: {fitness_values}\n")
            f.write("\n")

            # Detailed trade log
            f.write("="*80 + "\n")
            f.write("DETAILED TRADE LOG\n")
            f.write("-"*80 + "\n\n")

            for i, trade in enumerate(self.all_trades, 1):
                f.write(f"Trade #{i} ({trade['slice']}):\n")
                f.write(f"  Stock: {trade['stock_ticker']} (ID: {trade['stock_id']})\n")
                f.write(f"  Entry: {trade['entry_date']} @ ${trade['entry_price']:.2f}\n")
                f.write(f"  Exit:  {trade['exit_date']} @ ${trade['exit_price']:.2f}\n")
                f.write(f"  Days Held: {trade['days_held']}\n")
                f.write(f"  Gain/Loss: {trade['gain_pct']:.2f}%\n")
                f.write(f"  Reward: {trade['reward']:.2f}\n")
                f.write(f"  Coefficient: {trade['coefficient']:.2f}\n")
                f.write(f"  Target: {trade['sale_target_pct']:.1f}% (${trade['sale_target_price']:.2f})\n")
                f.write(f"  Exit Type: {'Active Sell (Target Hit)' if trade['is_active_sell'] else 'Auto Liquidation (' + trade['exit_reason'] + ')'}\n")
                f.write(f"  Result: {'WIN' if trade['is_win'] else 'LOSS'}\n")
                f.write("\n")

        print(f"✓ Text report saved to {filepath}")

    def _export_trades_csv(self, filepath: Path):
        """Export trades to CSV."""
        if not self.all_trades:
            print("! No trades to export")
            return

        df = pd.DataFrame(self.all_trades)

        # Reorder columns for better readability
        column_order = [
            'slice', 'stock_ticker', 'stock_id',
            'entry_date', 'entry_price',
            'exit_date', 'exit_price',
            'days_held', 'gain_pct', 'reward',
            'coefficient', 'sale_target_pct', 'sale_target_price',
            'exit_reason', 'is_active_sell', 'is_auto_liquidation', 'is_win'
        ]

        df = df[column_order]
        df.to_csv(filepath, index=False)
        print(f"✓ Trades CSV saved to {filepath}")

    def _export_summary_csv(self, filepath: Path):
        """Export summary statistics to CSV."""
        df = pd.DataFrame(self.slice_summaries)
        df.to_csv(filepath, index=False)
        print(f"✓ Summary CSV saved to {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate best agent from a training run"
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name of the run to evaluate (e.g., "azure-thunder-123"). If not specified, uses last_run.json'
    )

    args = parser.parse_args()

    try:
        # Create evaluator
        evaluator = AgentEvaluator(run_name=args.run_name)

        # Run evaluation
        evaluator.evaluate_all_slices()

        # Export results
        evaluator.export_results()

        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
