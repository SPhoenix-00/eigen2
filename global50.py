"""
Evaluate existing agents for Global 50 promotion.

This script loads agents from a specified folder, runs them through gauntlet
validation, and promotes qualifying agents to the Global Hall of Fame.

The script maintains a mirrored directory structure between local (global50/)
and GCP cloud storage. Use --mirror to check sync status across ALL context
windows (cw151, cw504, etc.) and automatically download any missing agent files.

EFFICIENCY GATING:
    Gauntlet scores are efficiency-adjusted to prevent "volume swindling" where
    agents inflate scores via massive capital usage with low ROI efficiency.
    
    Formula: Adjusted Score = Raw Score + abs(Raw Score) * (10 * (ROI% - BaselineROI%))
    
    This symmetric formula:
    - Adds a bonus/penalty proportional to the magnitude of the raw score
    - Bonus when ROI > baseline, penalty when ROI < baseline
    - Works symmetrically for both positive and negative raw scores
    - Example: Raw=-100, ROI=10%, Baseline=8% → -100 + 100*10*(10-8)/100 = -100 + 20 = -80
    - Example: Raw=100, ROI=6%, Baseline=8% → 100 + 100*10*(6-8)/100 = 100 - 20 = 80
    
    Baseline: Config.EFFICIENCY_BASELINE_ROI (default: 8.0%)
    This ensures only efficient agents (high ROI per dollar) get high scores.

MAVERICK MODE:
    Maverick agents [M] are trained with aggressive reward functions:
    - Triad 2.0-inspired fitness (ROI^1.5 * log10(Peak Capital + 10), boosted by (1+WR^4))
    - 1.2x loss multiplier (per trade rewards)
    - 0.3% hurdle rate (50% of normal)
    - No forced exit penalty
    - FOMO penalty in fitness (fitness only, not per trade)
    Mavericks are tracked separately and can break committee inaction but must still
    meet efficiency standards to enter Global 50.

Usage:
    python global50.py --init                              # Initialize Global 50
    python global50.py --mirror                            # Check sync status
    python global50.py --eval                              # Re-evaluate all agents
    python global50.py --trim                              # Interactive trim (prompts for thresholds)
    python global50.py --archive-fill                      # Fill Global 50 from archive
    python global50.py --cleanup                           # Archive orphan agents
    python global50.py --cleanup-dry-run                   # Report orphans (no changes)
    python global50.py --reactivate                        # Reactivate agents between archive and long-term-archive
    python global50.py --stats                             # Display comprehensive statistics
    python global50.py --stats-all                         # Display all agents in global50, archive, and long-term-archive
    python global50.py --mark-maverick <run_name>          # Mark all agents with run_name as Mavericks
    python global50.py --unmark-maverick <run_name>        # Unmark all agents with run_name as Mavericks
    python global50.py --agent-dir <path> [--run-name <name>] [--maverick]

Options:
    --init              First-time setup. Creates empty global50.json and validates cloud sync.
    --mirror            Check mirror status between local and GCP. Downloads missing agent files.
    --eval              Re-evaluate all agents with efficiency-adjusted gauntlet scoring.
    --trim              Interactive trim mode: prompts for gauntlet, ROI, expectancy, and trades thresholds.
    --archive-fill      Fill Global 50 from archive. Evaluates archived agents and promotes qualifying ones.
    --cleanup           Find and archive orphan agents (files in agents/ not in global50.json).
    --cleanup-dry-run   Like --cleanup but only reports orphans without archiving them.
    --reactivate        Reactivate agents: move qualifying agents from long-term-archive to archive,
                        and move non-qualifying agents from archive to long-term-archive.
    --stats             Display comprehensive statistics about the current Global 50 agents.
    --stats-all         Display detailed agent lists for all agents in global50, archive, and long-term-archive.
    --mark-maverick     Mark all agents with the given run_name as Mavericks in global50, archive, and long-term-archive.
    --unmark-maverick   Unmark all agents with the given run_name as Mavericks in global50, archive, and long-term-archive.
    --cw DAYS           Context window size in days (e.g., --cw 504 for cw504).
    --agent-dir PATH    Directory containing agent .pth files to evaluate.
    --run-name NAME     Run name for evaluation batch (default: batch-evaluation).
    --maverick          Flag agents from --agent-dir as Maverick. For --archive-fill, Maverick status is read from archive metadata.

Examples:
    python global50.py --eval                              # Update all metrics (with efficiency gating)
    python global50.py --trim                              # Interactive trim
    python global50.py --cleanup-dry-run                   # Preview orphan cleanup
    python global50.py --cleanup                           # Archive orphan agents
    python global50.py --stats                             # Display statistics
    python global50.py --stats --cw 504                     # Display statistics for cw504
    python global50.py --stats-all                         # Display all agents in all locations
    python global50.py --stats-all --cw 504                 # Display all agents for cw504
    python global50.py --mark-maverick twilight-haze-313    # Mark all agents with run_name as Mavericks
    python global50.py --unmark-maverick twilight-haze-313  # Unmark all agents with run_name as Mavericks
    python global50.py --agent-dir checkpoints/run-123/hall_of_fame
    python global50.py --agent-dir workspace/elite_agents --run-name batch-eval-001
    python global50.py --agent-dir checkpoints/maverick-run-456/hall_of_fame --maverick
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
from datetime import datetime
import json
import shutil

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from erl.global_hof import GlobalHallOfFame, LeagueRules, GlobalHoFEntry
from utils.config import Config
from utils.cloud_sync import get_cloud_sync_from_env
from training.erl_trainer import ERLTrainer


class TeeLogger:
    """Logger that writes to both file and stdout."""

    def __init__(self, filepath: Path):
        """
        Initialize TeeLogger.

        Args:
            filepath: Path to the log file
        """
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8', buffering=1)  # Line buffered

    def write(self, message):
        """Write message to both terminal and file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to disk

    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        """Check if the terminal is a TTY (needed by wandb and other libraries)."""
        return self.terminal.isatty()

    def fileno(self):
        """Return the file descriptor (needed by some libraries)."""
        return self.terminal.fileno()

    def close(self):
        """Close the log file."""
        self.log_file.close()

    def __del__(self):
        """Ensure file is closed on deletion."""
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()


class AgentEvaluator:
    """Evaluates agents and promotes them to Global 50."""

    # Minimum ROI threshold - agents below this are excluded (not real performers)
    MIN_ROI_THRESHOLD = 0.01  # 0.01%

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

        # Use global50 directory (mirrored with GCP)
        checkpoint_dir = Path("global50")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_hof = GlobalHallOfFame(
            cloud_sync=self.cloud_sync,
            run_name=self.run_name,
            league_rules=league_rules,
            checkpoint_dir=checkpoint_dir,
            disable_global50=False
        )

        # Log mirroring setup
        if self.global_hof.enabled:
            print(f"   Local directory: {self.global_hof.local_dir}")
            print(f"   Cloud mirror: gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")
            print(f"   Context Window: {league_rules.context_window_days} days")
            print(f"   Mirroring: ENABLED")

        # Initialize a minimal trainer helper for accessing gauntlet methods
        # We'll use ERLTrainer's methods directly to avoid code duplication
        print("\n4. Setting up gauntlet validation...")
        self._setup_gauntlet_helper()

        print("\n" + "="*70)

    def _setup_gauntlet_helper(self):
        """
        Create a minimal helper object to access ERLTrainer's gauntlet methods.

        This avoids duplicating the gauntlet logic while keeping initialization lightweight.
        We only set up the necessary attributes for gauntlet validation, without the
        heavy initialization (wandb, population, replay buffer, etc.).
        """
        # Create a minimal object that has the necessary attributes for gauntlet methods
        class GauntletHelper:
            def __init__(self, data_loader, val_start_idx, val_end_idx, normalization_stats):
                self.data_loader = data_loader
                self.val_start_idx = val_start_idx
                self.val_end_idx = val_end_idx
                self.normalization_stats = normalization_stats

                # Training indices (used by generate_gauntlet_slices)
                self.train_start_idx = Config.CONTEXT_WINDOW_DAYS
                self.train_end_idx = val_start_idx

                # Create persistent evaluation environment (reused across slices)
                # Note: consistency_mode is set per-evaluation based on is_maverick flag
                # Maverick agents are incompatible with consistency mode
                full_end_idx = len(data_loader.data_array_full)
                self.eval_env = TradingEnvironment(
                    data_array=data_loader.data_array,
                    dates=data_loader.dates,
                    normalization_stats=normalization_stats,
                    start_idx=Config.CONTEXT_WINDOW_DAYS,
                    end_idx=full_end_idx,
                    trading_end_idx=Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS,
                    data_array_full=data_loader.data_array_full,
                    consistency_mode=False  # Will be set per-evaluation based on is_maverick
                )

                # Replay buffer is not needed for gauntlet (training=False)
                self.replay_buffer = None

        # Create helper and borrow methods from ERLTrainer
        self.gauntlet_helper = GauntletHelper(
            self.data_loader,
            self.val_start_idx,
            self.val_end_idx,
            self.normalization_stats
        )

        # Bind ERLTrainer methods to our helper object
        self.gauntlet_helper.generate_gauntlet_slices = ERLTrainer.generate_gauntlet_slices.__get__(
            self.gauntlet_helper, GauntletHelper
        )
        self.gauntlet_helper.run_episode_batched = ERLTrainer.run_episode_batched.__get__(
            self.gauntlet_helper, GauntletHelper
        )
        self.gauntlet_helper.calculate_expectancy = ERLTrainer.calculate_expectancy.__get__(
            self.gauntlet_helper, GauntletHelper
        )

    def discover_agents(self, agent_dir: Path) -> List[Path]:
        """
        Discover all agent .pth files in directory (searches recursively).

        Args:
            agent_dir: Directory to search for agents

        Returns:
            List of agent file paths
        """
        print(f"\nDiscovering agents in: {agent_dir} (recursive search)")

        if not agent_dir.exists():
            print(f"   ERROR: Directory does not exist: {agent_dir}")
            return []

        # Find all .pth files recursively using **/*.pth
        agent_files = list(agent_dir.glob("**/*.pth"))

        # Show which subdirectories contain agents
        if len(agent_files) > 0:
            subdirs = set(f.parent.relative_to(agent_dir) for f in agent_files)
            print(f"   Found {len(agent_files)} agent files across {len(subdirs)} subdirectories")
            if len(subdirs) <= 10:  # Show subdirs if not too many
                for subdir in sorted(subdirs):
                    subdir_files = [f for f in agent_files if f.parent.relative_to(agent_dir) == subdir]
                    print(f"      {subdir}: {len(subdir_files)} agents")
        else:
            print("   No .pth files found in directory or subdirectories")
            print("   Make sure the directory contains agent checkpoint files")

        return agent_files

    def generate_gauntlet_slices(self) -> List[Tuple[int, int, int]]:
        """
        Generate 20+ rigorous validation slices for Gauntlet stress test.
        Delegates to ERLTrainer.generate_gauntlet_slices() to avoid code duplication.

        Returns:
            List of tuples: (start_idx, end_idx, trading_end_idx)
        """
        return self.gauntlet_helper.generate_gauntlet_slices()

    def run_gauntlet(self, agent: DDPGAgent, agent_name: str, is_maverick: bool = False) -> Tuple[float, dict]:
        """
        Run gauntlet validation on an agent using ERLTrainer's methods.

        This delegates to ERLTrainer's run_episode_batched and calculate_expectancy
        to ensure Global 50 scores are perfectly consistent with trainer validation.

        Args:
            agent: Agent to evaluate
            agent_name: Name for logging
            is_maverick: If True, disables consistency mode (maverick agents are incompatible with consistency mode)

        Returns:
            Tuple of (gauntlet_score, detailed_metrics)
        """
        # Generate gauntlet slices using ERLTrainer's method
        gauntlet_slices = self.generate_gauntlet_slices()
        print(f"   Running Gauntlet ({len(gauntlet_slices)} slices: {sum(1 for s in gauntlet_slices if s[0] < self.val_start_idx)} training + {sum(1 for s in gauntlet_slices if s[0] >= self.val_start_idx)} validation)...")

        slice_results = []
        all_closed_trades = []

        # Set agent to eval mode
        agent.actor.eval()
        agent.critic.eval()

        # Set consistency mode: disabled for maverick agents (incompatible)
        # Maverick agents use aggressive reward functions that conflict with consistency training
        self.gauntlet_helper.eval_env.set_consistency_mode(not is_maverick)
        
        # Enable gauntlet mode for soft zero-trades penalty (tactical no-trade is acceptable)
        self.gauntlet_helper.eval_env.set_gauntlet_mode(True)

        # Evaluate each slice using ERLTrainer's run_episode_batched
        for i, (start_idx, end_idx, _) in enumerate(gauntlet_slices):
            # Use ERLTrainer's optimized batched inference
            # Note: training=False so replay_buffer is not used
            fitness, episode_info = self.gauntlet_helper.run_episode_batched(
                agent=agent,
                env=self.gauntlet_helper.eval_env,
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
                'total_investment': episode_info.get('total_investment', 0.0),
                'peak_capital_employed': episode_info.get('peak_capital_employed', 0.0)
            })

            # Collect closed trades for expectancy calculation
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        # Extract fitness scores
        fitness_scores = [result['fitness'] for result in slice_results]

        # Calculate statistics for Penalized Median scoring
        max_score = float(np.max(fitness_scores))
        min_score = float(np.min(fitness_scores))
        mean_score = float(np.mean(fitness_scores))
        median_score = float(np.median(fitness_scores))
        std_score = float(np.std(fitness_scores))

        # Calculate Coefficient of Variation (CV) - measures "noise" relative to "signal"
        # CV = StdDev / |Mean| (use absolute value to handle negative means)
        # Lower CV = more stable/consistent agent
        if abs(mean_score) > 1e-10:
            cv = std_score / abs(mean_score)
        else:
            cv = float('inf')  # Undefined CV when mean is ~0

        # NEW SCORING FORMULA: Penalized Median
        # This rewards agents that reliably perform well while penalizing volatility
        # gauntlet_score = Median - (0.5 * StdDev)
        raw_gauntlet_score = float(median_score - (0.5 * std_score))

        # Penalty for agents that consistently refuse to trade
        # Score between -8.99 and -10 indicates no trades across all slices
        if -10.0 <= raw_gauntlet_score <= -9.00:
            raw_gauntlet_score = -2000.0

        # Aggregate metrics (same calculations as ERLTrainer)
        total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
        total_investment = sum([r['total_investment'] for r in slice_results])  # Legacy cumulative
        # Pooled ROI: sum of peak capitals (slices are parallel/independent scenarios)
        # This answers: "For every dollar of max drawdown capacity across all scenarios, how much profit?"
        total_peak_capital = sum([r['peak_capital_employed'] for r in slice_results])
        roi = float((total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0)

        # --- ANTI-SWINDLE: EFFICIENCY-ADJUSTED GAUNTLET SCORE ---
        #
        # The raw Gauntlet score (Median - 0.5*StdDev) can be gamed by volume.
        # We normalize by ROI to ensure only efficient agents get high scores.
        #
        # Baseline: Config.EFFICIENCY_BASELINE_ROI (default 8.0%)
        #
        # Formula: Adjusted Score = Raw Score + abs(Raw Score) * (10 * (ROI% - BaselineROI%))
        #
        # This symmetric formula:
        # - Adds a bonus/penalty proportional to the magnitude of the raw score
        # - Bonus when ROI > baseline, penalty when ROI < baseline
        # - Works symmetrically for both positive and negative raw scores
        # - Example: Raw=-100, ROI=10%, Baseline=8% → -100 + 100*10*(10-8)/100 = -100 + 20 = -80
        # - Example: Raw=100, ROI=6%, Baseline=8% → 100 + 100*10*(6-8)/100 = 100 - 20 = 80
        #
        efficiency_ratio = roi / Config.EFFICIENCY_BASELINE_ROI
        roi_diff_percentage_points = roi - Config.EFFICIENCY_BASELINE_ROI
        # Formula: 10 * (ROI% - BaselineROI%) where both are percentages
        # Convert percentage points to decimal: (ROI% - BaselineROI%) / 100
        adjustment = abs(raw_gauntlet_score) * (10.0 * roi_diff_percentage_points / 100.0)
        gauntlet_score = raw_gauntlet_score + adjustment

        total_wins = sum([r['num_wins'] for r in slice_results])
        total_losses = sum([r['num_losses'] for r in slice_results])
        total_trades = int(total_wins + total_losses)
        win_rate = float((total_wins / total_trades * 100) if total_trades > 0 else 0.0)

        # Use ERLTrainer's calculate_expectancy method
        expectancy = float(self.gauntlet_helper.calculate_expectancy(all_closed_trades))

        # Calculate quality_count (trades with gain >= Config.ROI_QUALITY_THRESHOLD)
        # This is the threshold used for confidence factor in ROI adjustment
        quality_threshold = Config.ROI_QUALITY_THRESHOLD  # Default: 7.5% gain
        if all_closed_trades:
            quality_count = sum(1 for t in all_closed_trades if t.get('gain_pct', 0) >= quality_threshold)
        else:
            quality_count = 0

        # Calculate ratios
        quality_ratio = float(quality_count / total_trades) if total_trades > 0 else 0.0
        win_ratio = float(total_wins / total_trades) if total_trades > 0 else 0.0

        # Reset gauntlet mode after validation
        self.gauntlet_helper.eval_env.set_gauntlet_mode(False)
        # Reset consistency mode (will be set again for next evaluation)
        self.gauntlet_helper.eval_env.set_consistency_mode(False)

        detailed_metrics = {
            'gauntlet_score': gauntlet_score,
            'raw_gauntlet_score': raw_gauntlet_score,  # Pre-efficiency adjustment
            'efficiency_ratio': efficiency_ratio,  # ROI / baseline
            'mean_fitness': mean_score,
            'median_fitness': median_score,
            'std_fitness': std_score,
            'min_fitness': min_score,
            'max_fitness': max_score,
            'cv': cv,  # Coefficient of Variation - lower is more stable
            'roi': roi,
            'total_trades': total_trades,
            'quality_count': quality_count,
            'quality_ratio': quality_ratio,
            'win_rate': win_rate,
            'win_ratio': win_ratio,
            'expectancy': expectancy,
            'num_slices': len(gauntlet_slices),
            'fitness_all_slices': [float(score) for score in fitness_scores]
        }

        return gauntlet_score, detailed_metrics

    def evaluate_agent(self, agent_path: Path, generation: int = 0, is_maverick: bool = False) -> dict:
        """
        Evaluate a single agent for Global 50.

        Args:
            agent_path: Path to agent .pth file
            generation: Generation number (default 0 for external agents)
            is_maverick: If True, flag this agent as Maverick (aggressive reward function)

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
            'qualified': False,  # Whether agent qualified for promotion (even if not promoted)
            'error': None,
            'agent': None,  # Store agent object if qualified but not promoted (for archiving)
            'metrics': None  # Store metrics if qualified but not promoted
        }

        try:
            # Load agent
            print(f"Loading agent...")
            agent = DDPGAgent(agent_id=0)
            agent.load(str(agent_path))
            print(f"   Agent loaded successfully")

            # Run gauntlet
            print(f"Running gauntlet ({Config.GAUNTLET_NUM_SLICES} slices)...")
            gauntlet_score, metrics = self.run_gauntlet(agent, agent_name, is_maverick=is_maverick)

            print(f"\nGauntlet Results:")
            raw_score = metrics.get('raw_gauntlet_score', gauntlet_score)
            efficiency_ratio = metrics.get('efficiency_ratio', 1.0)
            print(f"   Raw Score:         {raw_score:>10.2f}  (Median - 0.5*StdDev)")
            print(f"   Efficiency Ratio:  {efficiency_ratio:>10.3f}  (ROI {metrics['roi']:.2f}% / {Config.EFFICIENCY_BASELINE_ROI:.1f}%)")
            print(f"   Final Score:       {gauntlet_score:>10.2f}  (Efficiency-Adjusted)")
            print(f"   Median Fitness:    {metrics['median_fitness']:>10.2f}")
            print(f"   Std Fitness:       {metrics['std_fitness']:>10.2f}")
            print(f"   CV (Stability):    {metrics['cv']:>10.3f}")
            print(f"   ROI:               {metrics['roi']:>10.2f}%")
            print(f"   Total Trades:      {metrics['total_trades']:>10}")
            print(f"   Quality Ratio:     {metrics['quality_ratio']:>10.3f}")
            print(f"   Win Ratio:         {metrics['win_ratio']:>10.3f}")
            print(f"   Expectancy:        {metrics['expectancy']:>10.2f}")

            result.update(metrics)
            result['success'] = True

            # Check if qualifies for Global 50 (now includes CV as 4th criterion)
            if self.global_hof.should_promote(gauntlet_score, metrics['roi'], metrics['expectancy'], metrics['cv'], is_maverick=is_maverick):
                result['qualified'] = True
                print(f"\n   Agent QUALIFIES for Global 50!")
                print(f"   Gauntlet Threshold: {self.global_hof.entry_threshold:.2f}")
                print(f"   ROI Threshold: {self.global_hof.roi_threshold:.2f}%")
                print(f"   Expectancy Threshold: {self.global_hof.expectancy_threshold:.4f}")
                print(f"   CV Threshold: {self.global_hof.cv_threshold:.3f} (max allowed)")
                print(f"   Attempting promotion...")

                # Attempt promotion
                promoted, rank = self.global_hof.check_and_promote(
                    agent=agent,
                    gauntlet_score=gauntlet_score,
                    generation=generation,
                    roi=metrics['roi'],
                    expectancy=metrics['expectancy'],
                    cv=metrics['cv'],
                    quality_ratio=metrics['quality_ratio'],
                    win_ratio=metrics['win_ratio'],
                    total_trades=metrics['total_trades'],
                    is_maverick=is_maverick
                )

                result['promoted'] = promoted
                result['rank'] = rank

                # Note: Detailed promotion status is already printed by check_and_promote()
                # Don't print redundant success/failure messages here
                if not promoted:
                    # Store agent and metrics for potential archiving (if maverick cap reached)
                    if is_maverick:
                        result['agent'] = agent
                        result['metrics'] = {
                            'gauntlet_score': gauntlet_score,
                            'generation': generation,
                            'roi': metrics['roi'],
                            'expectancy': metrics['expectancy'],
                            'cv': metrics['cv'],
                            'quality_ratio': metrics['quality_ratio'],
                            'win_ratio': metrics['win_ratio'],
                            'total_trades': metrics['total_trades']
                        }
            else:
                print(f"\n   Agent does not qualify for Global 50")
                # Check individual criteria for detailed feedback
                passes_gauntlet_min = gauntlet_score > self.global_hof.entry_threshold
                passes_roi_min = metrics['roi'] > self.global_hof.roi_threshold
                passes_expectancy_min = metrics['expectancy'] > self.global_hof.expectancy_threshold
                passes_cv_min = metrics['cv'] < self.global_hof.cv_threshold  # CV: lower is better
                beats_gauntlet_p25 = gauntlet_score > self.global_hof.gauntlet_p25
                beats_roi_p25 = metrics['roi'] > self.global_hof.roi_p25
                beats_expectancy_p25 = metrics['expectancy'] > self.global_hof.expectancy_p25
                
                # Count all 3 metrics (gauntlet, ROI, expectancy), need 2 of 3
                metrics_for_p25 = [beats_gauntlet_p25, beats_roi_p25, beats_expectancy_p25]
                count_above_p25 = sum(metrics_for_p25)

                # All 4 minimums required (gauntlet, ROI, expectancy, CV)
                min_required = 4
                count_above_min = sum([passes_gauntlet_min, passes_roi_min, passes_expectancy_min, passes_cv_min])
                print(f"   Minimums ({count_above_min}/{min_required}, {min_required} required): Gauntlet {'✓' if passes_gauntlet_min else '✗'} | ROI {'✓' if passes_roi_min else '✗'} | Expectancy {'✓' if passes_expectancy_min else '✗'} | CV {'✓' if passes_cv_min else '✗'}")
                print(f"   P25 ({count_above_p25}/2, 2 required): Gauntlet {'✓' if beats_gauntlet_p25 else '✗'} | ROI {'✓' if beats_roi_p25 else '✗'} | Expectancy {'✓' if beats_expectancy_p25 else '✗'}")

        except Exception as e:
            print(f"\n   ERROR: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)

        return result

    def evaluate_batch(self, agent_dir: Path, is_maverick: bool = False) -> List[dict]:
        """
        Evaluate all agents in a directory.

        When using --maverick flag:
        - Enforces MAVERICK_CAP (5) limit on promotions
        - Archives any qualifying maverick agents that can't be promoted due to cap

        Args:
            agent_dir: Directory containing agent .pth files
            is_maverick: If True, flag all agents as Maverick (aggressive reward function)

        Returns:
            List of evaluation results
        """
        # Discover agents
        agent_files = self.discover_agents(agent_dir)

        if len(agent_files) == 0:
            return []

        # Load current Global 50 state (always, not just for mavericks)
        if self.global_hof.enabled:
            self.global_hof._download_global_ledger()
            self.global_hof._load_local_ledger()
            self.global_hof._update_entry_threshold()

        # For --agent-dir, we need to apply normal promotion guidelines even when not full.
        # When Global 50 is not full, _update_entry_threshold sets thresholds to -inf,
        # which would accept any agent. Instead, we compute thresholds from the existing
        # population to maintain quality standards.
        #
        # We define a helper function to recompute thresholds after each promotion,
        # since check_and_promote internally calls _update_entry_threshold which resets to -inf.
        def recompute_thresholds_from_population():
            """Recompute thresholds from existing population (used during --agent-dir)."""
            if len(self.global_hof.entries) == 0:
                return  # No entries, can't compute thresholds
            if len(self.global_hof.entries) >= self.global_hof.CAPACITY:
                return  # Full, normal thresholds apply

            # Define the comparison pool
            pool = self.global_hof.entries
            
            # MAVERICK FIX: Compare Mavericks only against Mavericks (Highlander within Mavericks)
            if is_maverick:
                maverick_pool = [e for e in pool if e.is_maverick]
                if len(maverick_pool) > 0:
                    pool = maverick_pool
                    print(f"   [Maverick Mode] Setting thresholds based on {len(pool)} existing Maverick(s)")
                else:
                    # If no Mavericks exist, we cannot enforce "Better than existing Mavericks".
                    # We skip recomputing thresholds (leaving them at -inf) to allow the first Mavericks to enter.
                    print(f"   [Maverick Mode] No existing Mavericks to set thresholds - Entry Open")
                    return
            else:
                # STANDARD AGENTS: Compare only against other standard agents
                non_maverick_pool = [e for e in pool if not e.is_maverick]
                if len(non_maverick_pool) > 0:
                    pool = non_maverick_pool
                    print(f"   [Standard Mode] Setting thresholds based on {len(pool)} existing Standard Agent(s)")
                else:
                    # Fallback if no standard agents exist (e.g. only mavericks in list)
                    # We accept all standard agents in this case
                    print(f"   [Standard Mode] No existing Standard Agents to set thresholds - Entry Open")
                    return

            gauntlet_scores = [e.gauntlet_score for e in pool]
            roi_values = [e.roi for e in pool]
            expectancy_values = [e.expectancy for e in pool]
            cv_values = [e.cv for e in pool]

            # Override the -inf/+inf thresholds with actual population statistics
            self.global_hof.entry_threshold = min(gauntlet_scores)
            self.global_hof.roi_threshold = min(roi_values)
            self.global_hof.expectancy_threshold = min(expectancy_values)
            # CV threshold = max CV in population (worst allowed volatility, lower is better)
            self.global_hof.cv_threshold = max(cv_values)

            self.global_hof.gauntlet_median = float(np.percentile(gauntlet_scores, 50))
            self.global_hof.roi_median = float(np.percentile(roi_values, 50))
            self.global_hof.expectancy_median = float(np.percentile(expectancy_values, 50))

            self.global_hof.gauntlet_p25 = float(np.percentile(gauntlet_scores, 25))
            self.global_hof.roi_p25 = float(np.percentile(roi_values, 25))
            self.global_hof.expectancy_p25 = float(np.percentile(expectancy_values, 25))
            
            # If we are in Maverick mode with existing Mavericks, we must also update the 
            # Maverick-specific thresholds in global_hof because the logic we just added to global_hof 
            # relies on them. However, since we are overriding the "standard" thresholds here 
            # to simulate a full population, we should ensure the Maverick thresholds align with this manual override
            # if we want analyze_promotion to work correctly for the manual check.
            
            # Actually, since we modified analyze_promotion to use Maverick thresholds when is_maverick=True,
            # we need to make sure those Maverick thresholds are set correctly here too.
            if is_maverick:
                self.global_hof.maverick_entry_threshold = self.global_hof.entry_threshold
                self.global_hof.maverick_roi_threshold = self.global_hof.roi_threshold
                self.global_hof.maverick_expectancy_threshold = self.global_hof.expectancy_threshold
                self.global_hof.maverick_cv_threshold = self.global_hof.cv_threshold
                
                self.global_hof.maverick_gauntlet_median = self.global_hof.gauntlet_median
                self.global_hof.maverick_roi_median = self.global_hof.roi_median
                self.global_hof.maverick_expectancy_median = self.global_hof.expectancy_median
                
                self.global_hof.maverick_gauntlet_p25 = self.global_hof.gauntlet_p25
                self.global_hof.maverick_roi_p25 = self.global_hof.roi_p25
                self.global_hof.maverick_expectancy_p25 = self.global_hof.expectancy_p25

        # Apply threshold recomputation if Global 50 is not full
        if self.global_hof.enabled:
            current_size = len(self.global_hof.entries)
            if current_size > 0 and current_size < self.global_hof.CAPACITY:
                recompute_thresholds_from_population()
                print(f"\n⚠ Global 50 not full ({current_size}/{self.global_hof.CAPACITY}) - applying normal promotion thresholds from existing population")

        # Check current maverick count if evaluating mavericks
        current_maverick_count = 0
        if is_maverick and self.global_hof.enabled:
            current_maverick_count = sum(1 for e in self.global_hof.entries if e.is_maverick)
            remaining_slots = self.global_hof.MAVERICK_CAP - current_maverick_count
            print(f"\nCurrent Maverick count in Global 50: {current_maverick_count}/{self.global_hof.MAVERICK_CAP}")
            if remaining_slots <= 0:
                print(f"⚠ WARNING: Maverick cap ({self.global_hof.MAVERICK_CAP}) already reached.")
                print(f"   Qualifying mavericks will be archived instead of promoted.")
            else:
                print(f"Remaining maverick slots: {remaining_slots}")

        # Evaluate each agent
        results = []
        mavericks_promoted_this_batch = 0

        print(f"\n{'='*70}")
        print(f"Evaluating {len(agent_files)} agents")
        if is_maverick:
            print(f"Maverick Mode: All agents will be flagged as Maverick [M]")
        print(f"{'='*70}")

        for i, agent_path in enumerate(agent_files, 1):
            # Reload ledger before each evaluation to get latest state
            if self.global_hof.enabled:
                self.global_hof._download_global_ledger()
                self.global_hof._load_local_ledger()
                # Recompute thresholds if not full (check_and_promote may have reset them to -inf)
                current_size = len(self.global_hof.entries)
                if current_size > 0 and current_size < self.global_hof.CAPACITY:
                    recompute_thresholds_from_population()

            # Check maverick cap before evaluating
            if is_maverick and self.global_hof.enabled:
                current_maverick_count = sum(1 for e in self.global_hof.entries if e.is_maverick)
                
                if current_maverick_count >= self.global_hof.MAVERICK_CAP:
                    print(f"\n[{i}/{len(agent_files)}] Processing {agent_path.name}")
                    print(f"⚠ Maverick cap ({self.global_hof.MAVERICK_CAP}) reached.")
                    print(f"   Evaluating agent to check if it qualifies (will archive if qualified)...")
                    # Continue to evaluate - we'll archive if it qualifies

            print(f"\n[{i}/{len(agent_files)}] Processing {agent_path.name}")

            result = self.evaluate_agent(agent_path, generation=i, is_maverick=is_maverick)
            results.append(result)
            
            # If agent was promoted, reload ledger and recompute thresholds
            if result.get('promoted', False) and self.global_hof.enabled:
                self.global_hof._download_global_ledger()
                self.global_hof._load_local_ledger()
                current_size = len(self.global_hof.entries)
                if current_size > 0 and current_size < self.global_hof.CAPACITY:
                    recompute_thresholds_from_population()
            
            # Track maverick promotions
            if is_maverick and result.get('promoted', False):
                mavericks_promoted_this_batch += 1
            
            # If maverick qualified but wasn't promoted (likely due to cap), archive it
            if is_maverick and result.get('qualified', False) and not result.get('promoted', False):
                # Check if cap is the reason (re-check current count)
                self.global_hof._download_global_ledger()
                self.global_hof._load_local_ledger()
                current_maverick_count = sum(1 for e in self.global_hof.entries if e.is_maverick)
                
                if current_maverick_count >= self.global_hof.MAVERICK_CAP and result.get('agent') and result.get('metrics'):
                    print(f"\n   Archiving qualifying maverick (cap reached)...")
                    archived = self.archive_qualifying_maverick(
                        agent_path=agent_path,
                        agent=result['agent'],
                        gauntlet_score=result['metrics']['gauntlet_score'],
                        generation=result['metrics']['generation'],
                        roi=result['metrics']['roi'],
                        expectancy=result['metrics']['expectancy'],
                        cv=result['metrics']['cv'],
                        quality_ratio=result['metrics']['quality_ratio'],
                        win_ratio=result['metrics']['win_ratio'],
                        total_trades=result['metrics']['total_trades'],
                        run_name=self.run_name
                    )
                    if archived:
                        result['archived'] = True
                        print(f"   ✓ Agent archived (qualified but maverick cap reached)")

        # Print summary
        if is_maverick:
            print(f"\n{'='*70}")
            print(f"Maverick Evaluation Summary")
            print(f"{'='*70}")
            print(f"  Promoted: {mavericks_promoted_this_batch} maverick(s)")
            archived_count = sum(1 for r in results if r.get('archived', False))
            if archived_count > 0:
                print(f"  Archived: {archived_count} qualifying maverick(s) (cap reached)")
            print(f"{'='*70}")

        # Final sync to ensure everything is mirrored to GCP
        if self.global_hof.enabled:
            print(f"\n{'='*70}")
            print(f"Syncing {self.global_hof.context_window_id}/ to GCP...")
            print(f"{'='*70}")
            self._sync_to_cloud()

        return results

    def _sync_to_cloud(self):
        """
        Sync the context-window-specific global50 directory to GCP.
        Ensures local and cloud are mirrored with verification.
        """
        if not self.global_hof.enabled:
            return

        failed_uploads = []

        # Sync global50.json with verification
        if self.global_hof.local_json_path.exists():
            print(f"   Uploading global50.json...")
            if not self.cloud_sync.upload_file_verified(
                str(self.global_hof.local_json_path),
                self.global_hof.cloud_json_path
            ):
                failed_uploads.append("global50.json")

        # Sync agents directory with verification
        agents_synced = 0
        if self.global_hof.local_agents_dir.exists():
            for agent_file in self.global_hof.local_agents_dir.glob("*.pth"):
                cloud_path = f"{self.global_hof.cloud_base}/agents/{agent_file.name}"
                if self.cloud_sync.upload_file_verified(str(agent_file), cloud_path):
                    agents_synced += 1
                else:
                    failed_uploads.append(agent_file.name)

        # Sync archive directory with verification
        archive_synced = 0
        if self.global_hof.local_archive_dir.exists():
            for archive_file in self.global_hof.local_archive_dir.glob("*"):
                cloud_path = f"{self.global_hof.cloud_base}/archive/{archive_file.name}"
                if self.cloud_sync.upload_file_verified(str(archive_file), cloud_path):
                    archive_synced += 1
                else:
                    failed_uploads.append(f"archive/{archive_file.name}")

        print(f"   Synced {agents_synced} agent files")
        print(f"   Synced {archive_synced} archive files")
        print(f"   Mirror: gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")

        if failed_uploads:
            print(f"   ⚠ WARNING: {len(failed_uploads)} file(s) failed to sync:")
            for f in failed_uploads[:5]:  # Show first 5
                print(f"      - {f}")
            if len(failed_uploads) > 5:
                print(f"      ... and {len(failed_uploads) - 5} more")
            print(f"   Status: INCOMPLETE - Run --mirror to verify")
        else:
            print(f"   Status: UP TO DATE")

    def move_to_exclude(self, entry: GlobalHoFEntry, reason: str = "low_roi") -> bool:
        """
        Move an agent to the exclude/ folder (for agents that don't meet minimum standards).

        This is different from archive - excluded agents are considered invalid/useless
        (e.g., ROI below minimum threshold), while archived agents are just not in the top 50.

        Args:
            entry: The GlobalHoFEntry to exclude
            reason: Reason for exclusion (for logging)

        Returns:
            True if successfully excluded, False otherwise
        """
        import shutil

        filename = entry.get_filename()
        scoresheet_filename = filename.replace('.pth', '.json')

        print(f"   Excluding: {filename} (reason: {reason})")

        # Local paths
        local_src_agents = self.global_hof.local_agents_dir / filename
        local_src_archive = self.global_hof.local_archive_dir / filename
        local_dst = self.global_hof.local_exclude_dir / filename
        local_scoresheet_dst = self.global_hof.local_exclude_dir / scoresheet_filename

        # Cloud paths
        cloud_agents_path = f"{self.global_hof.cloud_base}/agents/{filename}"
        cloud_archive_path = f"{self.global_hof.cloud_base}/archive/{filename}"
        cloud_exclude_path = f"{self.global_hof.cloud_exclude_prefix}{filename}"
        cloud_exclude_scoresheet = f"{self.global_hof.cloud_exclude_prefix}{scoresheet_filename}"

        try:
            # Find source file (could be in agents/ or archive/)
            if local_src_agents.exists():
                shutil.move(str(local_src_agents), str(local_dst))
                cloud_src_path = cloud_agents_path
            elif local_src_archive.exists():
                shutil.move(str(local_src_archive), str(local_dst))
                cloud_src_path = cloud_archive_path
            else:
                # Try to download from cloud
                if self.cloud_sync.download_file(cloud_agents_path, str(local_dst)):
                    cloud_src_path = cloud_agents_path
                elif self.cloud_sync.download_file(cloud_archive_path, str(local_dst)):
                    cloud_src_path = cloud_archive_path
                else:
                    print(f"   ⚠ Could not find {filename} in local or cloud storage")
                    return False

            # Save metadata scoresheet
            with open(local_scoresheet_dst, 'w') as f:
                data = entry.to_dict()
                data['excluded_reason'] = reason
                data['excluded_at'] = datetime.now().isoformat()
                json.dump(data, f, indent=2)

            # Upload to cloud exclude/ with verification
            if local_dst.exists():
                self.cloud_sync.upload_file_verified(str(local_dst), cloud_exclude_path)
            self.cloud_sync.upload_file_verified(str(local_scoresheet_dst), cloud_exclude_scoresheet)

            # Delete from cloud agents/ or archive/ after confirming exclude exists
            if self.cloud_sync.file_exists(cloud_exclude_path):
                self.cloud_sync.delete_file(cloud_src_path)
                # Also try to delete scoresheet from original location
                original_scoresheet = cloud_src_path.replace('.pth', '.json')
                self.cloud_sync.delete_file(original_scoresheet)
                print(f"   ✓ Excluded: {filename}")
                return True
            else:
                print(f"   ⚠ Failed to verify exclude upload for {filename}")
                return False

        except Exception as e:
            print(f"   ✗ Error excluding {filename}: {e}")
            return False

    def archive_qualifying_maverick(self, agent_path: Path, agent: DDPGAgent, gauntlet_score: float,
                                    generation: int, roi: float, expectancy: float, cv: float,
                                    quality_ratio: float, win_ratio: float, total_trades: int,
                                    run_name: str) -> bool:
        """
        Archive a maverick agent that qualifies for Global 50 but cannot be promoted due to maverick cap.

        Args:
            agent_path: Path to the agent .pth file
            agent: The DDPGAgent instance
            gauntlet_score: Gauntlet score
            generation: Generation number
            roi: ROI percentage
            expectancy: Expectancy metric
            cv: Coefficient of Variation
            quality_ratio: Quality ratio
            win_ratio: Win ratio
            total_trades: Total trades
            run_name: Run name for this agent

        Returns:
            True if successfully archived, False otherwise
        """
        import shutil

        filename = f"{run_name}_{agent.agent_id}.pth"
        scoresheet_filename = filename.replace('.pth', '.json')

        print(f"   Archiving qualifying maverick (cap reached): {filename}")

        # Create entry for archive metadata
        entry = GlobalHoFEntry(
            agent_id=agent.agent_id,
            run_name=run_name,
            gauntlet_score=gauntlet_score,
            generation=generation,
            roi=roi,
            expectancy=expectancy,
            cv=cv,
            quality_ratio=quality_ratio,
            win_ratio=win_ratio,
            total_trades=total_trades,
            is_maverick=True
        )

        # Local paths
        local_src = agent_path
        local_dst = self.global_hof.local_archive_dir / filename
        local_scoresheet_dst = self.global_hof.local_archive_dir / scoresheet_filename

        # Cloud paths
        cloud_archive_path = f"{self.global_hof.cloud_base}/archive/{filename}"
        cloud_archive_json = f"{self.global_hof.cloud_base}/archive/{scoresheet_filename}"

        try:
            # Ensure archive directory exists
            self.global_hof.local_archive_dir.mkdir(parents=True, exist_ok=True)

            # Copy agent file to archive (don't move, keep original)
            if local_src.exists():
                shutil.copy2(str(local_src), str(local_dst))
            else:
                print(f"   ⚠ Source file not found: {local_src}")
                return False

            # Save metadata scoresheet with archive reason
            entry_dict = entry.to_dict()
            entry_dict['archived_at'] = datetime.now().isoformat()
            entry_dict['archive_reason'] = 'maverick_cap_reached'
            entry_dict['qualified_for_promotion'] = True
            with open(local_scoresheet_dst, 'w') as f:
                json.dump(entry_dict, f, indent=2)

            # Upload to cloud archive/ with verification
            if self.global_hof.enabled:
                if local_dst.exists():
                    self.cloud_sync.upload_file_verified(str(local_dst), cloud_archive_path)
                self.cloud_sync.upload_file_verified(str(local_scoresheet_dst), cloud_archive_json)
                print(f"   ✓ Archived: {filename}")
                return True
            else:
                print(f"   ✓ Archived locally: {filename}")
                return True

        except Exception as e:
            print(f"   ✗ Error archiving {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def discover_cloud_context_windows(self) -> List[str]:
        """
        Discover all context window directories in cloud storage.

        Returns:
            List of context window IDs (e.g., ['cw151', 'cw504', ...])
        """
        context_windows = set()
        cloud_global50_prefix = f"{self.cloud_sync.project_name}/global50/"

        try:
            if self.cloud_sync.provider == "gcs":
                # List all blobs under global50/
                blobs = self.cloud_sync.bucket.list_blobs(prefix=cloud_global50_prefix, delimiter='/')
                # Get prefixes (subdirectories)
                for prefix in blobs.prefixes:
                    # prefix looks like "eigen2/global50/cw151/"
                    cw_id = prefix.rstrip('/').split('/')[-1]
                    if cw_id.startswith('cw'):
                        context_windows.add(cw_id)
            elif self.cloud_sync.provider == "s3":
                paginator = self.cloud_sync.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(
                    Bucket=self.cloud_sync.bucket_name,
                    Prefix=cloud_global50_prefix,
                    Delimiter='/'
                )
                for page in pages:
                    if 'CommonPrefixes' in page:
                        for prefix in page['CommonPrefixes']:
                            cw_id = prefix['Prefix'].rstrip('/').split('/')[-1]
                            if cw_id.startswith('cw'):
                                context_windows.add(cw_id)
            elif self.cloud_sync.provider == "azure":
                blob_list = self.cloud_sync.container_client.walk_blobs(
                    name_starts_with=cloud_global50_prefix,
                    delimiter='/'
                )
                for blob in blob_list:
                    if hasattr(blob, 'prefix'):
                        cw_id = blob.prefix.rstrip('/').split('/')[-1]
                        if cw_id.startswith('cw'):
                            context_windows.add(cw_id)
        except Exception as e:
            print(f"   ⚠ Error listing cloud context windows: {e}")

        return sorted(context_windows)

    def _sync_archive_directory(self, context_window_id: str, local_archive_dir: Path, cloud_archive_prefix: str) -> bool:
        """
        Sync archive directory between local and cloud for a context window.
        
        Args:
            context_window_id: Context window ID (e.g., 'cw151')
            local_archive_dir: Local archive directory path
            cloud_archive_prefix: Cloud archive prefix (e.g., 'eigen2/global50/cw151/archive/')
            
        Returns:
            True if synced successfully, False if errors occurred.
        """
        # Ensure local archive directory exists
        local_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # List local archive files
        local_files = {}
        if local_archive_dir.exists():
            for f in local_archive_dir.glob("*"):
                if f.is_file():
                    local_files[f.name] = f
        
        # List cloud archive files
        cloud_files = set()
        try:
            if self.cloud_sync.provider == "gcs":
                blobs = self.cloud_sync.bucket.list_blobs(prefix=cloud_archive_prefix)
                for blob in blobs:
                    filename = blob.name.split('/')[-1]
                    if filename:  # Skip empty names (directory markers)
                        cloud_files.add(filename)
            elif self.cloud_sync.provider == "s3":
                paginator = self.cloud_sync.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.cloud_sync.bucket_name, Prefix=cloud_archive_prefix)
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            filename = obj['Key'].split('/')[-1]
                            if filename:
                                cloud_files.add(filename)
            elif self.cloud_sync.provider == "azure":
                blob_list = self.cloud_sync.container_client.list_blobs(name_starts_with=cloud_archive_prefix)
                for blob in blob_list:
                    filename = blob.name.split('/')[-1]
                    if filename:
                        cloud_files.add(filename)
        except Exception as e:
            print(f"    ⚠ Error listing cloud archive: {e}")
            return False
        
        local_file_names = set(local_files.keys())
        
        # Files only in cloud - download them
        missing_local = cloud_files - local_file_names
        downloaded = 0
        failed_downloads = []
        
        for filename in missing_local:
            cloud_path = f"{cloud_archive_prefix}{filename}"
            local_path = local_archive_dir / filename
            try:
                if self.cloud_sync.download_file(cloud_path, str(local_path)):
                    downloaded += 1
                else:
                    failed_downloads.append(filename)
            except Exception as e:
                failed_downloads.append(filename)
        
        if downloaded > 0:
            print(f"    ✓ Downloaded {downloaded} archive file(s) from cloud")
        if failed_downloads:
            print(f"    ⚠ Failed to download {len(failed_downloads)} archive file(s)")
            return False
        
        # Files only in local - upload them
        missing_cloud = local_file_names - cloud_files
        uploaded = 0
        failed_uploads = []
        
        for filename in missing_cloud:
            local_path = local_files[filename]
            cloud_path = f"{cloud_archive_prefix}{filename}"
            try:
                if self.cloud_sync.upload_file_verified(str(local_path), cloud_path):
                    uploaded += 1
                else:
                    failed_uploads.append(filename)
            except Exception as e:
                failed_uploads.append(filename)
        
        if uploaded > 0:
            print(f"    ✓ Uploaded {uploaded} archive file(s) to cloud")
        if failed_uploads:
            print(f"    ⚠ Failed to upload {len(failed_uploads)} archive file(s)")
            return False
        
        # Files in both - verify they match (simple size check for now)
        common_files = local_file_names & cloud_files
        if common_files:
            # For now, we assume if files exist in both places they're synced
            # A more thorough check would compare checksums, but that's expensive
            pass
        
        return True

    def check_mirror_status_for_context_window(self, context_window_id: str) -> bool:
        """
        Check synchronization status for a specific context window.

        Args:
            context_window_id: Context window ID (e.g., 'cw151')

        Returns:
            True if in sync, False if mismatch detected.
        """
        import tempfile
        import os

        # Build paths for this context window
        local_dir = Path("global50") / context_window_id
        local_json_path = local_dir / "global50.json"
        local_agents_dir = local_dir / "agents"
        local_archive_dir = local_dir / "archive"
        cloud_base = f"{self.cloud_sync.project_name}/global50/{context_window_id}"
        cloud_json_path = f"{cloud_base}/global50.json"
        cloud_archive_prefix = f"{cloud_base}/archive/"

        print(f"\n  [{context_window_id}]")

        # Check if local file exists
        local_exists = local_json_path.exists()

        # Check if cloud file exists
        cloud_exists = False
        temp_cloud_path = None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            temp_cloud_path = tmp.name

        try:
            cloud_exists = self.cloud_sync.download_file(cloud_json_path, temp_cloud_path)
        except Exception:
            cloud_exists = False

        # Case 1: Neither exists (skip)
        if not local_exists and not cloud_exists:
            print(f"    ✓ Empty (no data)")
            if temp_cloud_path and os.path.exists(temp_cloud_path):
                os.unlink(temp_cloud_path)
            return True

        # Case 2: Only local exists
        if local_exists and not cloud_exists:
            print(f"    ⚠ Local exists but cloud is missing")
            if temp_cloud_path and os.path.exists(temp_cloud_path):
                os.unlink(temp_cloud_path)
            return False

        # Case 3: Only cloud exists - need user confirmation before downloading
        if not local_exists and cloud_exists:
            # Load cloud data to show entry count
            try:
                with open(temp_cloud_path, 'r') as f:
                    cloud_data = json.load(f)
                cloud_entries = len(cloud_data.get('entries', []))
            except Exception:
                cloud_entries = "unknown"

            print(f"    ⚠ Cloud exists ({cloud_entries} entries) but local is missing")
            print(f"")
            print(f"    This could mean:")
            print(f"      1. You're on a new machine and need to sync from cloud")
            print(f"      2. Local data was accidentally deleted")
            print(f"      3. You haven't run --init yet on this machine")
            print(f"")
            print(f"    Options:")
            print(f"      [d] Download from cloud (sync cloud → local)")
            print(f"      [i] Run --init first (recommended for new setups)")
            print(f"      [s] Skip this context window")
            print(f"")

            while True:
                choice = input("    Enter choice (d/i/s): ").strip().lower()
                if choice == 'd':
                    break
                elif choice == 'i':
                    print(f"    → Please run: python global50.py --init")
                    if temp_cloud_path and os.path.exists(temp_cloud_path):
                        os.unlink(temp_cloud_path)
                    return False
                elif choice == 's':
                    print(f"    → Skipping {context_window_id}")
                    if temp_cloud_path and os.path.exists(temp_cloud_path):
                        os.unlink(temp_cloud_path)
                    return True  # Not a failure, just skipped
                else:
                    print("    Invalid choice. Please enter d, i, or s.")

            # User chose to download
            print(f"    Downloading from cloud...")

            # Create local directory structure
            local_dir.mkdir(parents=True, exist_ok=True)
            local_agents_dir.mkdir(parents=True, exist_ok=True)
            local_archive_dir.mkdir(parents=True, exist_ok=True)

            # Move temp file to local json path
            import shutil
            shutil.move(temp_cloud_path, str(local_json_path))
            print(f"    ✓ Downloaded global50.json")

            # Load and download agents
            try:
                with open(local_json_path, 'r') as f:
                    data = json.load(f)

                entries = data.get('entries', [])
                downloaded = 0
                failed_downloads = []

                for entry in entries:
                    run_name = entry.get('run_name', 'unknown')
                    agent_id = entry.get('agent_id', 0)
                    filename = f"{run_name}_{agent_id}.pth"
                    local_agent_path = local_agents_dir / filename
                    cloud_agent_path = f"{cloud_base}/agents/{filename}"

                    if not local_agent_path.exists():
                        try:
                            success = self.cloud_sync.download_file(cloud_agent_path, str(local_agent_path))
                            if success:
                                downloaded += 1
                            else:
                                failed_downloads.append(filename)
                        except Exception:
                            failed_downloads.append(filename)

                if failed_downloads:
                    print(f"    ⚠ Downloaded {downloaded}/{len(entries)} agents")
                    print(f"    ⚠ {len(failed_downloads)} agents MISSING from cloud:")
                    for f in failed_downloads[:5]:
                        print(f"        - {f}")
                    if len(failed_downloads) > 5:
                        print(f"        ... and {len(failed_downloads) - 5} more")
                    print(f"")
                    print(f"    ⚠ Cloud data integrity issue: global50.json references agents")
                    print(f"      that don't exist in cloud storage.")
                    return False
                else:
                    print(f"    ✓ Downloaded {downloaded}/{len(entries)} agents")
            except Exception as e:
                print(f"    ⚠ Error downloading agents: {e}")
                return False

            # Sync archive directory
            print(f"    Syncing archive directory...")
            archive_synced = self._sync_archive_directory(context_window_id, local_archive_dir, cloud_archive_prefix)
            if not archive_synced:
                print(f"    ⚠ Archive sync had errors (continuing anyway)")

            return True

        # Case 4: Both exist - compare and sync
        try:
            with open(local_json_path, 'r') as f:
                local_data = json.load(f)
            with open(temp_cloud_path, 'r') as f:
                cloud_data = json.load(f)

            local_entries = len(local_data.get('entries', []))
            cloud_entries = len(cloud_data.get('entries', []))
            local_timestamp = local_data.get('last_updated', '')
            cloud_timestamp = cloud_data.get('last_updated', '')

            if local_data == cloud_data:
                print(f"    ✓ Synced ({local_entries} entries)")

                # Check for missing agent files
                entries = local_data.get('entries', [])
                missing_count = 0
                downloaded = 0
                failed_downloads = []

                for entry in entries:
                    run_name = entry.get('run_name', 'unknown')
                    agent_id = entry.get('agent_id', 0)
                    filename = f"{run_name}_{agent_id}.pth"
                    local_agent_path = local_agents_dir / filename

                    if not local_agent_path.exists():
                        missing_count += 1
                        cloud_agent_path = f"{cloud_base}/agents/{filename}"
                        try:
                            local_agents_dir.mkdir(parents=True, exist_ok=True)
                            success = self.cloud_sync.download_file(cloud_agent_path, str(local_agent_path))
                            if success:
                                downloaded += 1
                            else:
                                failed_downloads.append(filename)
                        except Exception:
                            failed_downloads.append(filename)

                if missing_count > 0:
                    if failed_downloads:
                        print(f"    ⚠ Downloaded {downloaded}/{missing_count} missing agents")
                        print(f"    ⚠ {len(failed_downloads)} agents MISSING from cloud storage:")
                        for f in failed_downloads[:5]:
                            print(f"        - {f}")
                        if len(failed_downloads) > 5:
                            print(f"        ... and {len(failed_downloads) - 5} more")
                        if temp_cloud_path and os.path.exists(temp_cloud_path):
                            os.unlink(temp_cloud_path)
                        return False  # Data integrity issue
                    else:
                        print(f"    ✓ Downloaded {downloaded}/{missing_count} missing agents")

                # Sync archive directory
                print(f"    Syncing archive directory...")
                archive_synced = self._sync_archive_directory(context_window_id, local_archive_dir, cloud_archive_prefix)
                if not archive_synced:
                    print(f"    ⚠ Archive sync had errors (continuing anyway)")

                if temp_cloud_path and os.path.exists(temp_cloud_path):
                    os.unlink(temp_cloud_path)
                return True
            else:
                # Mismatch - compare timestamps to determine which is newer
                print(f"    ⚠ Mismatch detected:")
                print(f"        Local:  {local_entries} entries (updated: {local_timestamp or 'unknown'})")
                print(f"        Cloud:  {cloud_entries} entries (updated: {cloud_timestamp or 'unknown'})")
                print(f"")

                # Determine which is newer
                local_newer = False
                cloud_newer = False
                if local_timestamp and cloud_timestamp:
                    local_newer = local_timestamp > cloud_timestamp
                    cloud_newer = cloud_timestamp > local_timestamp
                    if local_newer:
                        print(f"    → Local is NEWER (recommended: upload local to cloud)")
                    elif cloud_newer:
                        print(f"    → Cloud is NEWER (recommended: download cloud to local)")
                    else:
                        print(f"    → Timestamps are identical (manual resolution needed)")
                else:
                    print(f"    → Cannot determine which is newer (timestamps missing)")

                print(f"")
                print(f"    Options:")
                print(f"      [l] Use LOCAL version (upload to cloud)")
                print(f"      [c] Use CLOUD version (download to local)")
                print(f"      [s] Skip (leave as-is, report mismatch)")
                print(f"")

                while True:
                    choice = input("    Enter choice (l/c/s): ").strip().lower()
                    if choice == 'l':
                        # Upload local to cloud
                        print(f"    → Uploading local to cloud...")
                        if self.cloud_sync.upload_file_verified(str(local_json_path), cloud_json_path):
                            print(f"    ✓ Cloud updated to match local")
                            # Sync archive directory
                            print(f"    Syncing archive directory...")
                            archive_synced = self._sync_archive_directory(context_window_id, local_archive_dir, cloud_archive_prefix)
                            if not archive_synced:
                                print(f"    ⚠ Archive sync had errors (continuing anyway)")
                            if temp_cloud_path and os.path.exists(temp_cloud_path):
                                os.unlink(temp_cloud_path)
                            return True
                        else:
                            print(f"    ✗ Failed to upload to cloud")
                            if temp_cloud_path and os.path.exists(temp_cloud_path):
                                os.unlink(temp_cloud_path)
                            return False
                    elif choice == 'c':
                        # Download cloud to local
                        print(f"    → Downloading cloud to local...")
                        import shutil
                        shutil.copy(temp_cloud_path, str(local_json_path))
                        print(f"    ✓ Local updated to match cloud")
                        # Sync archive directory
                        print(f"    Syncing archive directory...")
                        archive_synced = self._sync_archive_directory(context_window_id, local_archive_dir, cloud_archive_prefix)
                        if not archive_synced:
                            print(f"    ⚠ Archive sync had errors (continuing anyway)")
                        if temp_cloud_path and os.path.exists(temp_cloud_path):
                            os.unlink(temp_cloud_path)
                        return True
                    elif choice == 's':
                        print(f"    → Skipping (mismatch remains)")
                        if temp_cloud_path and os.path.exists(temp_cloud_path):
                            os.unlink(temp_cloud_path)
                        return False
                    else:
                        print("    Invalid choice. Please enter l, c, or s.")

        except Exception as e:
            print(f"    ⚠ Error: {e}")
            if temp_cloud_path and os.path.exists(temp_cloud_path):
                os.unlink(temp_cloud_path)
            return False

    def check_mirror_status(self) -> bool:
        """
        Check synchronization status between local and GCP for ALL context windows.
        Verifies JSON metadata and downloads any missing agent files.
        Returns True if all in sync, False if any mismatch detected.
        """
        print(f"\n{'='*70}")
        print("Checking Mirror Status (All Context Windows)")
        print(f"{'='*70}")

        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            return True

        # Discover all context windows in cloud
        print("\nDiscovering context windows in cloud...")
        cloud_context_windows = self.discover_cloud_context_windows()

        # Also check local directories
        local_base = Path("global50")
        local_context_windows = set()
        if local_base.exists():
            for subdir in local_base.iterdir():
                if subdir.is_dir() and subdir.name.startswith('cw'):
                    local_context_windows.add(subdir.name)

        # Combine all context windows
        all_context_windows = sorted(set(cloud_context_windows) | local_context_windows)

        if not all_context_windows:
            print("✓ No context windows found (fresh setup)")
            return True

        print(f"Found {len(all_context_windows)} context window(s): {', '.join(all_context_windows)}")

        # Check each context window
        all_synced = True
        mismatched = []

        print("\nChecking synchronization status:")
        for cw_id in all_context_windows:
            synced = self.check_mirror_status_for_context_window(cw_id)
            if not synced:
                all_synced = False
                mismatched.append(cw_id)

        # Summary
        print(f"\n{'='*70}")
        if all_synced:
            print("✓ All context windows are fully synchronized!")
        else:
            print(f"⚠ {len(mismatched)} context window(s) have mismatches: {', '.join(mismatched)}")
            print("\nTo resolve mismatches, you may need to manually sync:")
            print("  - Option 1: Delete local global50/<cw_id>/ and re-run --mirror")
            print("  - Option 2: Use --eval to re-evaluate and sync")

        return all_synced

    def resolve_mirror_conflict(self):
        """
        Resolve mirror conflict by asking user which version to keep.
        """
        print(f"\n{'='*70}")
        print("Resolving Mirror Conflict")
        print(f"{'='*70}")

        print("\nWhich version should be kept?")
        print("  1. Local  - Upload local version to cloud (cloud will match local)")
        print("  2. Cloud  - Download cloud version to local (local will match cloud)")
        print("  3. Cancel - Exit without making changes")

        while True:
            choice = input("\nEnter choice (1/2/3): ").strip()

            if choice == '1':
                print("\n→ Syncing LOCAL to CLOUD...")
                self._sync_local_to_cloud()
                break
            elif choice == '2':
                print("\n→ Syncing CLOUD to LOCAL...")
                self._sync_cloud_to_local()
                break
            elif choice == '3':
                print("\n→ Cancelled. No changes made.")
                return
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def _sync_local_to_cloud(self):
        """
        Sync local directory to cloud (local supersedes cloud) with verification.
        """
        print("   Uploading local files to cloud...")

        failed_uploads = []

        # Upload global50.json with verification
        if self.global_hof.local_json_path.exists():
            if self.cloud_sync.upload_file_verified(
                str(self.global_hof.local_json_path),
                self.global_hof.cloud_json_path
            ):
                print(f"   ✓ Uploaded and verified global50.json")
            else:
                failed_uploads.append("global50.json")

        # Upload all agents with verification
        agents_uploaded = 0
        if self.global_hof.local_agents_dir.exists():
            for agent_file in self.global_hof.local_agents_dir.glob("*.pth"):
                cloud_path = f"{self.global_hof.cloud_base}/agents/{agent_file.name}"
                if self.cloud_sync.upload_file_verified(str(agent_file), cloud_path):
                    agents_uploaded += 1
                else:
                    failed_uploads.append(agent_file.name)

        # Upload all archive files with verification
        archive_uploaded = 0
        if self.global_hof.local_archive_dir.exists():
            for archive_file in self.global_hof.local_archive_dir.glob("*"):
                cloud_path = f"{self.global_hof.cloud_base}/archive/{archive_file.name}"
                if self.cloud_sync.upload_file_verified(str(archive_file), cloud_path):
                    archive_uploaded += 1
                else:
                    failed_uploads.append(f"archive/{archive_file.name}")

        print(f"   ✓ Uploaded {agents_uploaded} agent files")
        print(f"   ✓ Uploaded {archive_uploaded} archive files")

        if failed_uploads:
            print(f"\n⚠ WARNING: {len(failed_uploads)} file(s) failed to upload:")
            for f in failed_uploads[:5]:
                print(f"      - {f}")
            if len(failed_uploads) > 5:
                print(f"      ... and {len(failed_uploads) - 5} more")
            print(f"\n⚠ Cloud may not match local - run --mirror to verify")
        else:
            print(f"\n✓ Cloud now matches local")
        print(f"   Mirror: gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")

    def _sync_cloud_to_local(self):
        """
        Sync cloud directory to local (cloud supersedes local).
        """
        print("   Downloading cloud files to local...")

        # Download global50.json
        success = self.cloud_sync.download_file(
            self.global_hof.cloud_json_path,
            str(self.global_hof.local_json_path)
        )
        if success:
            print(f"   ✓ Downloaded global50.json")
            # Reload entries
            self.global_hof._load_local_ledger()
            self.global_hof._update_entry_threshold()

        # Download all agents from cloud
        # Note: This requires listing files in cloud, which depends on cloud provider
        # For now, we'll download agents mentioned in global50.json
        agents_downloaded = 0
        for entry in self.global_hof.entries:
            filename = entry.get_filename()
            cloud_path = f"{self.global_hof.cloud_base}/agents/{filename}"
            local_path = self.global_hof.local_agents_dir / filename

            try:
                success = self.cloud_sync.download_file(cloud_path, str(local_path))
                if success:
                    agents_downloaded += 1
                else:
                    print(f"   ⚠ Could not download {filename}: File not found in cloud")
            except Exception as e:
                print(f"   ⚠ Could not download {filename}: {e}")

        print(f"   ✓ Downloaded {agents_downloaded} agent files")
        print(f"\n✓ Local now matches cloud")
        print(f"   Local: {self.global_hof.local_dir}")

    def cleanup_orphan_agents(self, dry_run: bool = True) -> dict:
        """
        Find and archive orphan agents (files in agents/ not listed in global50.json).

        These orphans occur when agents are evicted from the top 50 but their .pth
        files weren't properly deleted from cloud storage.

        Args:
            dry_run: If True, only report orphans without archiving them

        Returns:
            Dictionary with cleanup statistics
        """
        print(f"\n{'='*70}")
        print("Orphan Agent Cleanup")
        print(f"{'='*70}")
        print(f"Mode: {'DRY RUN (report only)' if dry_run else 'CLEANUP (will archive orphans)'}")

        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            return {'success': False, 'error': 'not_enabled'}

        # Step 1: Get list of valid agents from global50.json
        print("\n1. Loading global50.json...")
        self.global_hof._download_global_ledger()
        self.global_hof._load_local_ledger()

        valid_filenames = set()
        for entry in self.global_hof.entries:
            valid_filenames.add(entry.get_filename())

        print(f"   Found {len(valid_filenames)} valid agents in global50.json")

        # Step 2: List all .pth files in cloud agents/ directory
        print("\n2. Listing agent files in cloud storage...")
        cloud_agents_prefix = f"{self.global_hof.cloud_base}/agents/"

        cloud_agent_files = set()
        try:
            if self.cloud_sync.provider == "gcs":
                blobs = self.cloud_sync.bucket.list_blobs(prefix=cloud_agents_prefix)
                for blob in blobs:
                    if blob.name.endswith('.pth'):
                        filename = blob.name.split('/')[-1]
                        cloud_agent_files.add(filename)
            elif self.cloud_sync.provider == "s3":
                paginator = self.cloud_sync.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.cloud_sync.bucket_name, Prefix=cloud_agents_prefix)
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.pth'):
                                filename = obj['Key'].split('/')[-1]
                                cloud_agent_files.add(filename)
            elif self.cloud_sync.provider == "azure":
                blob_list = self.cloud_sync.container_client.list_blobs(name_starts_with=cloud_agents_prefix)
                for blob in blob_list:
                    if blob.name.endswith('.pth'):
                        filename = blob.name.split('/')[-1]
                        cloud_agent_files.add(filename)
            else:
                print("   ⚠ Local provider - checking local files only")
                if self.global_hof.local_agents_dir.exists():
                    for f in self.global_hof.local_agents_dir.glob("*.pth"):
                        cloud_agent_files.add(f.name)
        except Exception as e:
            print(f"   ✗ Error listing cloud files: {e}")
            return {'success': False, 'error': str(e)}

        print(f"   Found {len(cloud_agent_files)} .pth files in agents/")

        # Step 3: Identify orphans
        orphan_files = cloud_agent_files - valid_filenames
        print(f"\n3. Identifying orphans...")
        print(f"   Valid agents (in JSON):     {len(valid_filenames)}")
        print(f"   Agent files (in storage):   {len(cloud_agent_files)}")
        print(f"   Orphan files:               {len(orphan_files)}")

        if not orphan_files:
            print("\n✓ No orphan agents found. Storage is clean!")
            return {
                'success': True,
                'valid_count': len(valid_filenames),
                'storage_count': len(cloud_agent_files),
                'orphan_count': 0,
                'archived_count': 0
            }

        # List orphans
        print(f"\n   Orphan files:")
        for filename in sorted(orphan_files):
            print(f"     - {filename}")

        if dry_run:
            print(f"\n{'='*70}")
            print("DRY RUN COMPLETE")
            print(f"{'='*70}")
            print(f"Found {len(orphan_files)} orphan agent(s) that would be archived.")
            print(f"\nTo actually archive these orphans, run:")
            print(f"  python global50.py --cleanup")
            return {
                'success': True,
                'valid_count': len(valid_filenames),
                'storage_count': len(cloud_agent_files),
                'orphan_count': len(orphan_files),
                'archived_count': 0,
                'dry_run': True
            }

        # Step 4: Archive orphans (move from agents/ to archive/)
        print(f"\n4. Archiving orphan agents...")
        archived_count = 0
        failed_count = 0

        for filename in sorted(orphan_files):
            cloud_src = f"{self.global_hof.cloud_base}/agents/{filename}"
            cloud_dst = f"{self.global_hof.cloud_base}/archive/{filename}"
            local_src = self.global_hof.local_agents_dir / filename
            local_dst = self.global_hof.local_archive_dir / filename

            try:
                # Download to local archive if not already there
                if not local_dst.exists():
                    if local_src.exists():
                        # Move locally
                        import shutil
                        shutil.move(str(local_src), str(local_dst))
                    else:
                        # Download from cloud to archive
                        self.cloud_sync.download_file(cloud_src, str(local_dst))

                # Upload to cloud archive with verification
                if local_dst.exists():
                    upload_success = self.cloud_sync.upload_file_verified(str(local_dst), cloud_dst)

                    if upload_success:
                        if self.cloud_sync.delete_file(cloud_src):
                            print(f"   ✓ Archived: {filename}")
                            archived_count += 1

                            # Also delete local source if it still exists
                            if local_src.exists():
                                local_src.unlink()
                        else:
                            print(f"   ⚠ Archived but failed to delete from agents/: {filename}")
                            failed_count += 1
                    else:
                        print(f"   ⚠ Archive upload failed, keeping in agents/: {filename}")
                        failed_count += 1
                else:
                    print(f"   ✗ Could not download {filename}")
                    failed_count += 1

            except Exception as e:
                print(f"   ✗ Error archiving {filename}: {e}")
                failed_count += 1

        print(f"\n{'='*70}")
        print("Cleanup Summary")
        print(f"{'='*70}")
        print(f"Orphan agents found:    {len(orphan_files)}")
        print(f"Successfully archived:  {archived_count}")
        if failed_count > 0:
            print(f"Failed to archive:      {failed_count}")
        print(f"{'='*70}")

        return {
            'success': True,
            'valid_count': len(valid_filenames),
            'storage_count': len(cloud_agent_files),
            'orphan_count': len(orphan_files),
            'archived_count': archived_count,
            'failed_count': failed_count,
            'dry_run': False
        }

    def reevaluate_global50(self):
        """
        Re-evaluate all agents in Global 50 with current evaluation logic.
        Updates all metrics (gauntlet_score, ROI, expectancy, etc.) for existing agents.
        """
        print(f"\n{'='*70}")
        print("Re-evaluating Global 50")
        print(f"{'='*70}")

        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            print("Cannot re-evaluate agents.")
            return

        # Load current Global 50 state
        print("\nLoading current Global 50 state...")
        self.global_hof._download_global_ledger()
        self.global_hof._load_local_ledger()

        if len(self.global_hof.entries) == 0:
            print("✓ Global 50 is empty. Nothing to re-evaluate.")
            return

        print(f"\nFound {len(self.global_hof.entries)} agents to re-evaluate")
        print(f"{'='*70}")

        # Ask for confirmation before starting
        print("\n⚠ This will:")
        print("  1. Download and load each agent")
        print("  2. Run full gauntlet evaluation for each agent")
        print("  3. Apply efficiency gating (ROI-adjusted scoring to prevent volume swindling)")
        print(f"  4. Update metrics in global50.json")
        print("  5. Sync changes to cloud")
        print(f"\nEstimated time: ~{len(self.global_hof.entries) * 2} minutes")
        print(f"\nNote: Gauntlet scores are now efficiency-adjusted (raw_score × ROI/{Config.EFFICIENCY_BASELINE_ROI:.1f}%)")
        print(f"      Maverick agents [M] are identified and tracked separately.")

        while True:
            confirmation = input("\nProceed with re-evaluation? (yes/no): ").strip().lower()
            if confirmation in ['yes', 'y']:
                print("\n→ Starting re-evaluation...")
                break
            elif confirmation in ['no', 'n']:
                print("\n→ Re-evaluation cancelled.")
                return
            else:
                print("Please enter 'yes' or 'no'.")

        # Re-evaluate each agent
        results = []
        updated_entries = []

        print(f"\n{'='*70}")
        print("Re-evaluating Agents")
        print(f"{'='*70}")

        for i, entry in enumerate(self.global_hof.entries, 1):
            maverick_tag = " [M]" if entry.is_maverick else ""
            print(f"\n[{i}/{len(self.global_hof.entries)}] {entry.run_name} (Agent {entry.agent_id}){maverick_tag}")
            print(f"  Current Score: {entry.gauntlet_score:.2f}")

            try:
                # Download agent if needed
                filename = entry.get_filename()
                local_agent_path = self.global_hof.local_agents_dir / filename

                if not local_agent_path.exists():
                    print(f"  Downloading agent...")
                    cloud_path = f"{self.global_hof.cloud_base}/agents/{filename}"
                    success = self.cloud_sync.download_file(cloud_path, str(local_agent_path))
                    if not success:
                        print(f"  ✗ Failed to download agent. Skipping.")
                        results.append({
                            'entry': entry,
                            'success': False,
                            'error': 'Failed to download'
                        })
                        continue

                # Load agent
                print(f"  Loading agent...")
                agent = DDPGAgent(agent_id=entry.agent_id)
                agent.load(str(local_agent_path))

                # Run gauntlet
                print(f"  Running gauntlet...")
                new_score, metrics = self.run_gauntlet(agent, f"{entry.run_name}_{entry.agent_id}", is_maverick=entry.is_maverick)

                # Create updated entry
                updated_entry = GlobalHoFEntry(
                    agent_id=entry.agent_id,
                    run_name=entry.run_name,
                    gauntlet_score=new_score,
                    generation=entry.generation,
                    roi=metrics['roi'],
                    expectancy=metrics['expectancy'],
                    cv=metrics['cv'],
                    quality_ratio=metrics['quality_ratio'],
                    win_ratio=metrics['win_ratio'],
                    total_trades=metrics['total_trades'],
                    is_maverick=entry.is_maverick
                )

                # Show results - simple old vs new comparison
                score_change = new_score - entry.gauntlet_score
                score_symbol = "↑" if score_change > 0 else "↓" if score_change < 0 else "="
                
                print(f"  Old Score: {entry.gauntlet_score:.2f} → New Score: {new_score:.2f} ({score_symbol} {abs(score_change):.2f})")
                print(f"  ROI: {metrics['roi']:.2f}% | Expectancy: {metrics['expectancy']:.2f} | CV: {metrics['cv']:.3f}")
                print(f"  Trades: {metrics['total_trades']} | Quality: {metrics['quality_ratio']:.3f} | Win: {metrics['win_ratio']:.3f}")

                results.append({
                    'entry': entry,
                    'updated_entry': updated_entry,
                    'old_score': entry.gauntlet_score,
                    'new_score': new_score,
                    'change': score_change,
                    'success': True,
                    'fingerprint': tuple(metrics['fitness_all_slices'])  # Behavioral fingerprint
                })
                updated_entries.append(updated_entry)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'entry': entry,
                    'success': False,
                    'error': str(e)
                })

        # Detect behavioral duplicates (agents with identical trading behavior)
        print(f"\n{'='*70}")
        print("Detecting Behavioral Duplicates")
        print(f"{'='*70}")

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        # Group agents by their behavioral fingerprint
        fingerprint_groups = {}
        for r in successful:
            fp = r['fingerprint']
            if fp not in fingerprint_groups:
                fingerprint_groups[fp] = []
            fingerprint_groups[fp].append(r)

        # Find duplicate groups (more than one agent with same fingerprint)
        duplicate_groups = {fp: group for fp, group in fingerprint_groups.items() if len(group) > 1}

        behavioral_duplicates = []
        if duplicate_groups:
            print(f"\n⚠ Found {len(duplicate_groups)} group(s) of behaviorally identical agents:")

            for i, (fp, group) in enumerate(duplicate_groups.items(), 1):
                # Sort by score descending, keep the best one
                group_sorted = sorted(group, key=lambda x: x['new_score'], reverse=True)
                keeper = group_sorted[0]
                duplicates_in_group = group_sorted[1:]

                print(f"\n  Group {i}: {len(group)} identical agents")
                print(f"    KEEPING: {keeper['entry'].run_name} (Agent {keeper['entry'].agent_id}) - Score: {keeper['new_score']:.2f}")
                for dup in duplicates_in_group:
                    print(f"    REMOVING: {dup['entry'].run_name} (Agent {dup['entry'].agent_id}) - Score: {dup['new_score']:.2f}")
                    behavioral_duplicates.append(dup)

            print(f"\n→ {len(behavioral_duplicates)} behavioral duplicate(s) will be removed")

            # Remove duplicates from successful list and updated_entries
            duplicate_keys = {(d['entry'].run_name, d['entry'].agent_id) for d in behavioral_duplicates}
            successful = [r for r in successful if (r['entry'].run_name, r['entry'].agent_id) not in duplicate_keys]
            updated_entries = [e for e in updated_entries if (e.run_name, e.agent_id) not in duplicate_keys]
        else:
            print("\n✓ No behavioral duplicates found - all agents have unique trading patterns")

        # Detect low-ROI agents that should be excluded
        print(f"\n{'='*70}")
        print(f"Detecting Low-ROI Agents (threshold: {self.MIN_ROI_THRESHOLD}%)")
        print(f"{'='*70}")

        low_roi_agents = []
        for r in successful:
            if r['updated_entry'].roi < self.MIN_ROI_THRESHOLD:
                low_roi_agents.append(r)

        if low_roi_agents:
            print(f"\n⚠ Found {len(low_roi_agents)} agent(s) with ROI below {self.MIN_ROI_THRESHOLD}%:")
            for r in low_roi_agents:
                print(f"    - {r['entry'].run_name} (Agent {r['entry'].agent_id}) - ROI: {r['updated_entry'].roi:.4f}%")

            print(f"\n→ These agents will be moved to exclude/ folder")

            # Remove low-ROI agents from successful list and updated_entries
            low_roi_keys = {(r['entry'].run_name, r['entry'].agent_id) for r in low_roi_agents}
            successful = [r for r in successful if (r['entry'].run_name, r['entry'].agent_id) not in low_roi_keys]
            updated_entries = [e for e in updated_entries if (e.run_name, e.agent_id) not in low_roi_keys]
        else:
            print(f"\n✓ All agents meet the minimum ROI threshold ({self.MIN_ROI_THRESHOLD}%)")

        # Show summary
        print(f"\n{'='*70}")
        print("Re-evaluation Summary")
        print(f"{'='*70}")

        print(f"\nTotal Agents:      {len(results)}")
        print(f"Successful:        {len(successful)}")
        print(f"Duplicates:        {len(behavioral_duplicates)}")
        print(f"Low ROI:           {len(low_roi_agents)}")
        print(f"Failed:            {len(failed)}")

        if len(successful) > 0:
            print(f"\nScore Changes:")
            print(f"{'='*70}")
            print(f"{'Run Name':<30} {'Old Score':<12} {'New Score':<12} {'Change':<12}")
            print(f"{'-'*70}")

            for r in sorted(successful, key=lambda x: x['change'], reverse=True):
                change_str = f"{r['change']:+.2f}"
                symbol = "↑" if r['change'] > 0 else "↓" if r['change'] < 0 else "="
                print(f"{r['entry'].run_name:<30} {r['old_score']:<12.2f} {r['new_score']:<12.2f} {symbol} {change_str:<10}")

            avg_change = sum(r['change'] for r in successful) / len(successful)

            # Calculate overall metrics from updated entries
            all_scores = [r['new_score'] for r in successful]
            all_rois = [r['updated_entry'].roi for r in successful]
            all_expectancies = [r['updated_entry'].expectancy for r in successful]
            all_cvs = [r['updated_entry'].cv for r in successful]
            all_trades = [r['updated_entry'].total_trades for r in successful]

            print(f"\n{'='*70}")
            print(f"Overall Metrics (after re-evaluation)")
            print(f"{'='*70}")
            print(f"  Gauntlet Score:  min={min(all_scores):.2f}  mean={sum(all_scores)/len(all_scores):.2f}  max={max(all_scores):.2f}")
            print(f"  ROI:             min={min(all_rois):.2f}%  mean={sum(all_rois)/len(all_rois):.2f}%  max={max(all_rois):.2f}%")
            print(f"  Expectancy:      min={min(all_expectancies):.4f}  mean={sum(all_expectancies)/len(all_expectancies):.4f}  max={max(all_expectancies):.4f}")
            print(f"  CV:              min={min(all_cvs):.4f}  mean={sum(all_cvs)/len(all_cvs):.4f}  max={max(all_cvs):.4f}")
            print(f"  Total Trades:    min={min(all_trades)}  mean={sum(all_trades)/len(all_trades):.1f}  max={max(all_trades)}")
            print(f"\nAverage Score Change: {avg_change:+.2f}")

        if len(failed) > 0:
            print(f"\nFailed Agents:")
            for r in failed:
                print(f"  - {r['entry'].run_name} (Agent {r['entry'].agent_id}): {r.get('error', 'Unknown error')}")

        # Ask for confirmation to save
        print(f"\n{'='*70}")
        print("⚠ Update Global 50 with new scores?")
        print(f"{'='*70}")

        while True:
            confirmation = input("\nSave updated scores? (yes/no): ").strip().lower()
            if confirmation in ['yes', 'y']:
                print("\n→ Updating Global 50...")
                break
            elif confirmation in ['no', 'n']:
                print("\n→ Changes discarded. Global 50 unchanged.")
                return
            else:
                print("Please enter 'yes' or 'no'.")

        # Create sets of agents to exclude/archive
        low_roi_keys = {(r['entry'].run_name, r['entry'].agent_id) for r in low_roi_agents}
        duplicate_keys = {(d['entry'].run_name, d['entry'].agent_id) for d in behavioral_duplicates}

        # Update entries (only successful ones, keep failed ones with old scores)
        # Create a map of updated entries by (run_name, agent_id)
        update_map = {
            (r['updated_entry'].run_name, r['updated_entry'].agent_id): r['updated_entry']
            for r in successful
        }

        # Update entries list - exclude low_roi and duplicate agents
        final_entries = []
        for entry in self.global_hof.entries:
            key = (entry.run_name, entry.agent_id)
            # Skip low ROI and duplicate agents
            if key in low_roi_keys or key in duplicate_keys:
                continue
            if key in update_map:
                final_entries.append(update_map[key])
            else:
                # Keep original entry (failed re-evaluation)
                final_entries.append(entry)

        # Sort by new scores
        final_entries.sort(key=lambda e: e.gauntlet_score, reverse=True)

        # Update global HoF
        self.global_hof.entries = final_entries
        self.global_hof._update_entry_threshold()

        # Save and upload with verification
        print("\nSaving updated global50.json...")
        self.global_hof._save_local_ledger()

        if self.global_hof.enabled:
            ledger_success = self.global_hof._upload_global_ledger_verified()
            if not ledger_success:
                print(f"⚠ WARNING: Ledger sync failed! Run --mirror to check consistency.")
        else:
            self.global_hof._upload_global_ledger()

        # Move low-ROI agents to exclude/ folder
        if low_roi_agents:
            print(f"\n{'='*70}")
            print(f"Moving {len(low_roi_agents)} low-ROI agents to exclude/ folder...")
            print(f"{'='*70}")
            excluded_count = 0
            for r in low_roi_agents:
                if self.move_to_exclude(r['updated_entry'], reason=f"ROI {r['updated_entry'].roi:.4f}% < {self.MIN_ROI_THRESHOLD}%"):
                    excluded_count += 1
            print(f"\n✓ Excluded {excluded_count}/{len(low_roi_agents)} low-ROI agents")

        # Archive behavioral duplicates
        if behavioral_duplicates:
            print(f"\n{'='*70}")
            print(f"Archiving {len(behavioral_duplicates)} behavioral duplicates...")
            print(f"{'='*70}")
            import shutil
            archived_count = 0
            for dup in behavioral_duplicates:
                entry = dup['entry']
                filename = entry.get_filename()
                local_src = self.global_hof.local_agents_dir / filename
                local_dst = self.global_hof.local_archive_dir / filename

                try:
                    if local_src.exists():
                        shutil.move(str(local_src), str(local_dst))

                    # Save metadata scoresheet
                    scoresheet_filename = filename.replace('.pth', '.json')
                    scoresheet_path = self.global_hof.local_archive_dir / scoresheet_filename
                    with open(scoresheet_path, 'w') as f:
                        data = dup['updated_entry'].to_dict()
                        data['archived_reason'] = 'behavioral_duplicate'
                        json.dump(data, f, indent=2)

                    # Cloud operations
                    cloud_archive_pth = f"{self.global_hof.cloud_base}/archive/{filename}"
                    cloud_archive_json = f"{self.global_hof.cloud_base}/archive/{scoresheet_filename}"
                    cloud_agents_pth = f"{self.global_hof.cloud_base}/agents/{filename}"

                    if local_dst.exists():
                        self.cloud_sync.upload_file_verified(str(local_dst), cloud_archive_pth)
                    self.cloud_sync.upload_file_verified(str(scoresheet_path), cloud_archive_json)

                    # Delete from cloud agents/
                    if self.cloud_sync.file_exists(cloud_archive_pth):
                        self.cloud_sync.delete_file(cloud_agents_pth)
                        print(f"   ✓ Archived: {filename}")
                        archived_count += 1
                except Exception as e:
                    print(f"   ⚠ Error archiving {filename}: {e}")

            print(f"\n✓ Archived {archived_count}/{len(behavioral_duplicates)} duplicates")

        print(f"\n{'='*70}")
        print("✓ Re-evaluation Complete!")
        print(f"{'='*70}")
        print(f"  Updated:        {len(successful)} agents")
        print(f"  Excluded:       {len(low_roi_agents)} agents (low ROI)")
        print(f"  Archived:       {len(behavioral_duplicates)} agents (duplicates)")
        print(f"  Failed:         {len(failed)} agents")
        print(f"  New threshold:  {self.global_hof.entry_threshold:.2f}")
        print(f"  Cloud mirror:   gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")
        print(f"{'='*70}")

        # Automatically clean up orphan agents in cloud storage
        # This archives any .pth files in agents/ that are no longer in global50.json
        print(f"\n{'='*70}")
        print("Cleaning up orphan agents in cloud storage...")
        print(f"{'='*70}")
        self.cleanup_orphan_agents(dry_run=False)

    def trim_agents(self):
        """
        Interactive trim mode: shows current thresholds and prompts for
        gauntlet, ROI, expectancy, CV, and total trades thresholds to trim agents.

        This is a manual trimming tool that allows custom thresholds.
        Note: Automatic promotion uses stricter criteria:
        - All 4 metrics must beat minimum thresholds (including CV)
        - 2 of 3 metrics must beat 25th percentile
        - 1 of 3 metrics must beat median
        """
        print(f"\n{'='*70}")
        print(f"Interactive Trim Mode")
        print(f"{'='*70}")

        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            print("Cannot trim agents.")
            return

        # Load current Global 50 state
        print("\nLoading current Global 50 state...")

        # Re-download to ensure we have the latest
        self.global_hof._download_global_ledger()
        self.global_hof._load_local_ledger()
        self.global_hof._update_entry_threshold()

        if len(self.global_hof.entries) == 0:
            print("✓ Global 50 is empty. Nothing to trim.")
            return

        # Show current state
        print(f"\n{'='*70}")
        print(f"Current Global 50 State")
        print(f"{'='*70}")
        print(f"  Population size: {len(self.global_hof.entries)}/{self.global_hof.CAPACITY}")
        
        # Count Mavericks
        maverick_count = sum(1 for e in self.global_hof.entries if e.is_maverick)
        if maverick_count > 0:
            print(f"  Mavericks: {maverick_count} (agents trained with aggressive reward functions)")
        
        print(f"\n  Note: Gauntlet scores are efficiency-adjusted (raw × ROI/{Config.EFFICIENCY_BASELINE_ROI:.1f}%)")

        if len(self.global_hof.entries) >= self.global_hof.CAPACITY:
            print(f"\n  Current Thresholds (population full):")
            print(f"    Minimums:  Gauntlet={self.global_hof.entry_threshold:.2f}, ROI={self.global_hof.roi_threshold:.2f}%, Expectancy={self.global_hof.expectancy_threshold:.4f}, CV={self.global_hof.cv_threshold:.3f}")
            print(f"    Medians:   Gauntlet={self.global_hof.gauntlet_median:.2f}, ROI={self.global_hof.roi_median:.2f}%, Expectancy={self.global_hof.expectancy_median:.4f}")
            print(f"    P25:       Gauntlet={self.global_hof.gauntlet_p25:.2f}, ROI={self.global_hof.roi_p25:.2f}%, Expectancy={self.global_hof.expectancy_p25:.4f}")
        else:
            # Show current minimums for reference
            min_gauntlet = min(e.gauntlet_score for e in self.global_hof.entries)
            min_roi = min(e.roi for e in self.global_hof.entries)
            min_expectancy = min(e.expectancy for e in self.global_hof.entries)
            max_cv = max(e.cv for e in self.global_hof.entries)
            print(f"\n  Current Minimums (population not full, thresholds are -inf/+inf):")
            print(f"    Gauntlet:    {min_gauntlet:.2f}")
            print(f"    ROI:         {min_roi:.2f}%")
            print(f"    Expectancy:  {min_expectancy:.4f}")
            print(f"    CV (max):    {max_cv:.3f} (lower is better)")

        # Show distribution
        print(f"\n  Score Ranges:")
        print(f"    Gauntlet:     {min(e.gauntlet_score for e in self.global_hof.entries):.2f} to {max(e.gauntlet_score for e in self.global_hof.entries):.2f}")
        print(f"    ROI:          {min(e.roi for e in self.global_hof.entries):.2f}% to {max(e.roi for e in self.global_hof.entries):.2f}%")
        print(f"    Expectancy:   {min(e.expectancy for e in self.global_hof.entries):.4f} to {max(e.expectancy for e in self.global_hof.entries):.4f}")
        print(f"    CV:           {min(e.cv for e in self.global_hof.entries):.3f} to {max(e.cv for e in self.global_hof.entries):.3f} (lower is better)")
        print(f"    Total Trades: {min(e.total_trades for e in self.global_hof.entries)} to {max(e.total_trades for e in self.global_hof.entries)}")

        # Prompt for thresholds
        print(f"\n{'='*70}")
        print(f"Enter Trim Thresholds")
        print(f"{'='*70}")
        print(f"Agents will be KEPT if: gauntlet >= threshold AND ROI >= threshold AND expectancy >= threshold AND CV <= threshold AND trades >= threshold")
        print(f"(Note: CV uses <= because lower is better. This is a manual trim. Automatic promotion requires additional p25/median criteria)")
        print(f"Press Enter to skip a threshold (use -inf for min thresholds, +inf for CV, or 0 for trades)")

        # Get gauntlet threshold
        while True:
            gauntlet_input = input("\n  Gauntlet threshold (or Enter to skip): ").strip()
            if gauntlet_input == "":
                gauntlet_threshold = float('-inf')
                break
            try:
                gauntlet_threshold = float(gauntlet_input)
                break
            except ValueError:
                print("  Invalid number. Please enter a numeric value.")

        # Get ROI threshold
        while True:
            roi_input = input("  ROI threshold % (or Enter to skip): ").strip()
            if roi_input == "":
                roi_threshold = float('-inf')
                break
            try:
                roi_threshold = float(roi_input)
                break
            except ValueError:
                print("  Invalid number. Please enter a numeric value.")

        # Get expectancy threshold
        while True:
            expectancy_input = input("  Expectancy threshold (or Enter to skip): ").strip()
            if expectancy_input == "":
                expectancy_threshold = float('-inf')
                break
            try:
                expectancy_threshold = float(expectancy_input)
                break
            except ValueError:
                print("  Invalid number. Please enter a numeric value.")

        # Get CV threshold (lower is better, so we use <= comparison)
        while True:
            cv_input = input("  CV threshold (max allowed, lower is better; Enter to skip): ").strip()
            if cv_input == "":
                cv_threshold = float('inf')
                break
            try:
                cv_threshold = float(cv_input)
                break
            except ValueError:
                print("  Invalid number. Please enter a numeric value.")

        # Get total trades threshold
        while True:
            trades_input = input("  Minimum total trades (or Enter to skip): ").strip()
            if trades_input == "":
                trades_threshold = 0
                break
            try:
                trades_threshold = int(trades_input)
                break
            except ValueError:
                print("  Invalid number. Please enter an integer value.")

        print(f"\n  Selected thresholds:")
        print(f"    Gauntlet:     {gauntlet_threshold if gauntlet_threshold != float('-inf') else 'skipped (-inf)'}")
        print(f"    ROI:          {roi_threshold if roi_threshold != float('-inf') else 'skipped (-inf)'}%")
        print(f"    Expectancy:   {expectancy_threshold if expectancy_threshold != float('-inf') else 'skipped (-inf)'}")
        print(f"    CV:           {cv_threshold if cv_threshold != float('inf') else 'skipped (+inf)'} (max allowed)")
        print(f"    Total Trades: {trades_threshold if trades_threshold > 0 else 'skipped (0)'}")

        # Determine which thresholds are active (not skipped)
        gauntlet_active = gauntlet_threshold != float('-inf')
        roi_active = roi_threshold != float('-inf')
        expectancy_active = expectancy_threshold != float('-inf')
        cv_active = cv_threshold != float('inf')
        trades_active = trades_threshold > 0

        # Identify agents to remove
        # Agent must pass ALL active thresholds to be kept
        agents_to_keep = []
        agents_to_remove = []

        for entry in self.global_hof.entries:
            passes_gauntlet = entry.gauntlet_score >= gauntlet_threshold
            passes_roi = entry.roi >= roi_threshold
            passes_expectancy = entry.expectancy >= expectancy_threshold
            passes_cv = entry.cv <= cv_threshold  # CV: lower is better, so use <=
            passes_trades = entry.total_trades >= trades_threshold

            # Check each active threshold
            failed = False
            if gauntlet_active and not passes_gauntlet:
                failed = True
            if roi_active and not passes_roi:
                failed = True
            if expectancy_active and not passes_expectancy:
                failed = True
            if cv_active and not passes_cv:
                failed = True
            if trades_active and not passes_trades:
                failed = True

            if failed:
                agents_to_remove.append(entry)
            else:
                agents_to_keep.append(entry)

        if len(agents_to_remove) == 0:
            print(f"\n✓ No agents would be removed with these thresholds.")
            print(f"  All {len(self.global_hof.entries)} agents meet the criteria.")
            return

        # Show what will be removed
        print(f"\n{'='*70}")
        print(f"⚠ WARNING: {len(agents_to_remove)} agents will be REMOVED from Global 50:")
        print(f"{'='*70}")
        print(f"Note: Gauntlet scores shown are efficiency-adjusted (raw × ROI/{Config.EFFICIENCY_BASELINE_ROI:.1f}%)")
        print(f"{'Gauntlet':<10} {'ROI %':<10} {'Expect':<10} {'CV':<8} {'Trades':<8} {'Run Name':<25} {'Agent':<8} {'M':<3} {'Reason'}")
        print(f"{'-'*115}")

        for entry in sorted(agents_to_remove, key=lambda e: e.gauntlet_score):
            # Determine why agent fails
            passes_gauntlet = entry.gauntlet_score >= gauntlet_threshold
            passes_roi = entry.roi >= roi_threshold
            passes_expectancy = entry.expectancy >= expectancy_threshold
            passes_cv = entry.cv <= cv_threshold
            passes_trades = entry.total_trades >= trades_threshold

            reasons = []
            if gauntlet_active and not passes_gauntlet:
                reasons.append("gauntlet")
            if roi_active and not passes_roi:
                reasons.append("ROI")
            if expectancy_active and not passes_expectancy:
                reasons.append("expectancy")
            if cv_active and not passes_cv:
                reasons.append("CV")
            if trades_active and not passes_trades:
                reasons.append("trades")
            reason_str = ", ".join(reasons) if reasons else "filter"

            maverick_flag = "Y" if entry.is_maverick else ""
            print(f"{entry.gauntlet_score:<10.2f} {entry.roi:<10.2f} {entry.expectancy:<10.4f} {entry.cv:<8.3f} {entry.total_trades:<8} {entry.run_name:<25} {entry.agent_id:<8} {maverick_flag:<3} {reason_str}")

        print(f"\n{len(agents_to_keep)} agents will remain in Global 50.")

        if len(agents_to_keep) > 0:
            print(f"\nRemaining score ranges:")
            print(f"  Gauntlet:     {min(e.gauntlet_score for e in agents_to_keep):.2f} to {max(e.gauntlet_score for e in agents_to_keep):.2f}")
            print(f"  ROI:          {min(e.roi for e in agents_to_keep):.2f}% to {max(e.roi for e in agents_to_keep):.2f}%")
            print(f"  Expectancy:   {min(e.expectancy for e in agents_to_keep):.4f} to {max(e.expectancy for e in agents_to_keep):.4f}")
            print(f"  CV:           {min(e.cv for e in agents_to_keep):.3f} to {max(e.cv for e in agents_to_keep):.3f} (lower is better)")
            print(f"  Total Trades: {min(e.total_trades for e in agents_to_keep)} to {max(e.total_trades for e in agents_to_keep)}")

        # Ask for confirmation
        print(f"\n{'='*70}")
        print("⚠ This action will:")
        print("  1. Remove these agents from global50.json")
        print("  2. Move their files to archive/ (locally and on cloud)")
        print("  3. Update both local and cloud storage")
        print(f"{'='*70}")

        while True:
            confirmation = input("\nProceed with trim? (yes/no): ").strip().lower()

            if confirmation in ['yes', 'y']:
                print("\n→ Proceeding with trim...")
                break
            elif confirmation in ['no', 'n']:
                print("\n→ Trim cancelled. No changes made.")
                return
            else:
                print("Please enter 'yes' or 'no'.")

        # Perform the trim
        import shutil

        print("\nArchiving removed agents...")

        # Move agents to archive
        for entry in agents_to_remove:
            filename = entry.get_filename()

            # Local: Move from agents/ to archive/
            local_src = self.global_hof.local_agents_dir / filename
            local_dst = self.global_hof.local_archive_dir / filename

            if local_src.exists():
                shutil.move(str(local_src), str(local_dst))
                print(f"   ✓ Archived locally: {filename}")
            else:
                # Download from cloud if not in local cache
                cloud_src = f"{self.global_hof.cloud_base}/agents/{filename}"
                try:
                    success = self.cloud_sync.download_file(cloud_src, str(local_dst))
                    if success:
                        print(f"   ✓ Downloaded and archived: {filename}")
                    else:
                        print(f"   ⚠ Could not download {filename}: File not found in cloud")
                except Exception as e:
                    print(f"   ⚠ Could not download {filename}: {e}")

            # Save metadata scoresheet
            scoresheet_filename = filename.replace('.pth', '.json')
            scoresheet_path = self.global_hof.local_archive_dir / scoresheet_filename
            with open(scoresheet_path, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)

            # Cloud: Upload to archive/ with verification
            cloud_archive_pth = f"{self.global_hof.cloud_base}/archive/{filename}"
            cloud_archive_json = f"{self.global_hof.cloud_base}/archive/{scoresheet_filename}"
            cloud_agents_pth = f"{self.global_hof.cloud_base}/agents/{filename}"

            if local_dst.exists():
                if self.global_hof.enabled:
                    self.cloud_sync.upload_file_verified(str(local_dst), cloud_archive_pth)
                else:
                    self.cloud_sync.upload_file(str(local_dst), cloud_archive_pth, background=False)
            if self.global_hof.enabled:
                self.cloud_sync.upload_file_verified(str(scoresheet_path), cloud_archive_json)
            else:
                self.cloud_sync.upload_file(str(scoresheet_path), cloud_archive_json, background=False)

            # Delete from cloud agents/ ONLY after confirming archive exists
            if self.cloud_sync.file_exists(cloud_archive_pth):
                if self.cloud_sync.delete_file(cloud_agents_pth):
                    print(f"   ✓ Deleted from cloud agents/: {filename}")
                else:
                    print(f"   ⚠ Failed to delete from cloud agents/: {filename}")

        # Update entries list
        self.global_hof.entries = agents_to_keep

        # Update threshold
        self.global_hof._update_entry_threshold()

        # Save and upload updated ledger with verification
        print("\nUpdating global50.json...")
        self.global_hof._save_local_ledger()

        if self.global_hof.enabled:
            ledger_success = self.global_hof._upload_global_ledger_verified()
            if not ledger_success:
                print(f"⚠ WARNING: Ledger sync failed! Run --mirror to check consistency.")
        else:
            self.global_hof._upload_global_ledger()

        print(f"\n{'='*70}")
        print("✓ Trim Complete!")
        print(f"{'='*70}")
        print(f"  Removed:        {len(agents_to_remove)} agents")
        print(f"  Remaining:      {len(agents_to_keep)} agents")

        # Show overall metrics for remaining agents
        if len(agents_to_keep) > 0:
            all_scores = [e.gauntlet_score for e in agents_to_keep]
            all_rois = [e.roi for e in agents_to_keep]
            all_expectancies = [e.expectancy for e in agents_to_keep]
            all_trades = [e.total_trades for e in agents_to_keep]

            print(f"\n  Overall Metrics (remaining agents):")
            print(f"    Gauntlet:     min={min(all_scores):.2f}  mean={sum(all_scores)/len(all_scores):.2f}  max={max(all_scores):.2f}")
            print(f"    ROI:          min={min(all_rois):.2f}%  mean={sum(all_rois)/len(all_rois):.2f}%  max={max(all_rois):.2f}%")
            print(f"    Expectancy:   min={min(all_expectancies):.4f}  mean={sum(all_expectancies)/len(all_expectancies):.4f}  max={max(all_expectancies):.4f}")
            print(f"    Total Trades: min={min(all_trades)}  mean={sum(all_trades)/len(all_trades):.1f}  max={max(all_trades)}")

        print(f"\n  New thresholds:")
        print(f"    Gauntlet:     {self.global_hof.entry_threshold:.2f}")
        print(f"    ROI:          {self.global_hof.roi_threshold:.2f}%")
        print(f"    Expectancy:   {self.global_hof.expectancy_threshold:.4f}")
        print(f"  Archived to:    {self.global_hof.local_archive_dir}")
        print(f"  Cloud mirror:   gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")
        print(f"{'='*70}")

    def archive_fill(self):
        """
        Fill Global 50 from archive.

        Downloads archived agents from cloud storage, evaluates them with the gauntlet,
        and promotes qualifying agents to fill empty slots in the Global 50.
        """
        print(f"\n{'='*70}")
        print("Archive Fill Mode")
        print(f"{'='*70}")

        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            print("Cannot fill from archive.")
            return

        # Load current Global 50 state
        print("\nLoading current Global 50 state...")
        self.global_hof._download_global_ledger()
        self.global_hof._load_local_ledger()
        self.global_hof._update_entry_threshold()

        current_size = len(self.global_hof.entries)

        print(f"\n  Current Global 50 size: {current_size}/{self.global_hof.CAPACITY}")

        # For archive-fill, we need to apply normal promotion guidelines even when not full.
        # When Global 50 is not full, _update_entry_threshold sets thresholds to -inf,
        # which would accept any agent. Instead, we compute thresholds from the existing
        # population to maintain quality standards.
        #
        # We define a helper function to recompute thresholds after each promotion,
        # since check_and_promote internally calls _update_entry_threshold which resets to -inf.
        def recompute_thresholds_from_population():
            """Recompute thresholds from existing population (used during archive-fill).
            
            Segregates mavericks and non-mavericks to maintain proper gating metrics.
            """
            if len(self.global_hof.entries) == 0:
                return  # No entries, can't compute thresholds
            if len(self.global_hof.entries) >= self.global_hof.CAPACITY:
                return  # Full, normal thresholds apply (handled by _update_entry_threshold)

            # Filter for non-maverick agents to ensure clean separation of pools
            non_mavericks = [e for e in self.global_hof.entries if not e.is_maverick]
            
            if not non_mavericks:
                # Fallback if no non-mavericks exist (should prevent crash if logic violated)
                non_mavericks = self.global_hof.entries

            # Compute non-maverick thresholds
            gauntlet_scores = [e.gauntlet_score for e in non_mavericks]
            roi_values = [e.roi for e in non_mavericks]
            expectancy_values = [e.expectancy for e in non_mavericks]
            cv_values = [e.cv for e in non_mavericks]

            # Override the -inf/+inf thresholds with actual population statistics
            self.global_hof.entry_threshold = min(gauntlet_scores)
            self.global_hof.roi_threshold = min(roi_values)
            self.global_hof.expectancy_threshold = min(expectancy_values)
            # CV threshold = max CV in population (worst allowed volatility, lower is better)
            self.global_hof.cv_threshold = max(cv_values)

            self.global_hof.gauntlet_median = float(np.percentile(gauntlet_scores, 50))
            self.global_hof.roi_median = float(np.percentile(roi_values, 50))
            self.global_hof.expectancy_median = float(np.percentile(expectancy_values, 50))

            self.global_hof.gauntlet_p25 = float(np.percentile(gauntlet_scores, 25))
            self.global_hof.roi_p25 = float(np.percentile(roi_values, 25))
            self.global_hof.expectancy_p25 = float(np.percentile(expectancy_values, 25))

            # Always calculate Maverick thresholds based on existing Mavericks (Highlander logic)
            # This applies even if the Global 50 is not full.
            mavericks = [e for e in self.global_hof.entries if e.is_maverick]
            self.global_hof.maverick_count = len(mavericks)

            if self.global_hof.maverick_count > 0:
                m_gauntlet_scores = [e.gauntlet_score for e in mavericks]
                m_roi_values = [e.roi for e in mavericks]
                m_expectancy_values = [e.expectancy for e in mavericks]
                m_cv_values = [e.cv for e in mavericks]

                self.global_hof.maverick_entry_threshold = min(m_gauntlet_scores)
                self.global_hof.maverick_roi_threshold = min(m_roi_values)
                self.global_hof.maverick_expectancy_threshold = min(m_expectancy_values)
                self.global_hof.maverick_cv_threshold = max(m_cv_values)

                self.global_hof.maverick_gauntlet_median = float(np.percentile(m_gauntlet_scores, 50))
                self.global_hof.maverick_roi_median = float(np.percentile(m_roi_values, 50))
                self.global_hof.maverick_expectancy_median = float(np.percentile(m_expectancy_values, 50))

                self.global_hof.maverick_gauntlet_p25 = float(np.percentile(m_gauntlet_scores, 25))
                self.global_hof.maverick_roi_p25 = float(np.percentile(m_roi_values, 25))
                self.global_hof.maverick_expectancy_p25 = float(np.percentile(m_expectancy_values, 25))
            else:
                # No Mavericks? Open entry for the first ones.
                self.global_hof.maverick_entry_threshold = float('-inf')
                self.global_hof.maverick_roi_threshold = float('-inf')
                self.global_hof.maverick_expectancy_threshold = float('-inf')
                self.global_hof.maverick_cv_threshold = float('inf')
                self.global_hof.maverick_gauntlet_median = float('-inf')
                self.global_hof.maverick_roi_median = float('-inf')
                self.global_hof.maverick_expectancy_median = float('-inf')
                self.global_hof.maverick_gauntlet_p25 = float('-inf')
                self.global_hof.maverick_roi_p25 = float('-inf')
                self.global_hof.maverick_expectancy_p25 = float('-inf')

        if current_size > 0 and current_size < self.global_hof.CAPACITY:
            recompute_thresholds_from_population()
            print(f"\n  ⚠ Global 50 not full - applying normal promotion thresholds from existing {current_size} agents")

        # Thresholds are managed by GlobalHoF - already updated above
        print(f"\n  Current Thresholds:")
        print(f"    Minimums:  Gauntlet={self.global_hof.entry_threshold:.2f}, ROI={self.global_hof.roi_threshold:.2f}%, Expectancy={self.global_hof.expectancy_threshold:.4f}, CV={self.global_hof.cv_threshold:.3f}")
        print(f"    Medians:   Gauntlet={self.global_hof.gauntlet_median:.2f}, ROI={self.global_hof.roi_median:.2f}%, Expectancy={self.global_hof.expectancy_median:.4f}")
        print(f"    P25:       Gauntlet={self.global_hof.gauntlet_p25:.2f}, ROI={self.global_hof.roi_p25:.2f}%, Expectancy={self.global_hof.expectancy_p25:.4f}")
        print(f"\n  Promotion Criteria:")
        print(f"    1. All 4 metrics must beat minimum thresholds (CV: lower is better)")
        print(f"    2. At least 2 of 3 metrics must beat 25th percentile")
        print(f"    3. At least 1 metric must beat median (50th percentile)")

        # Discover archived agents in cloud storage
        print(f"\n{'='*70}")
        print("Discovering Archived Agents")
        print(f"{'='*70}")

        archive_agents = self._discover_archive_agents()

        if not archive_agents:
            print("\n⚠ No archived agents found in cloud storage.")
            print(f"  Archive path: gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/archive/")
            return

        print(f"\nFound {len(archive_agents)} archived agents")

        # Filter out agents already in Global 50 (by run_name + agent_id)
        existing_keys = {(e.run_name, e.agent_id) for e in self.global_hof.entries}
        candidates = [a for a in archive_agents if (a['run_name'], a['agent_id']) not in existing_keys]
        already_in_g50 = len(archive_agents) - len(candidates)

        # Filter out agents with ROI below minimum threshold (using class constant)
        candidates_before_roi_filter = len(candidates)
        candidates = [a for a in candidates if a.get('roi', 0) >= self.MIN_ROI_THRESHOLD]
        excluded_low_roi = candidates_before_roi_filter - len(candidates)

        print(f"  Already in Global 50: {already_in_g50}")
        print(f"  Excluded (ROI < {self.MIN_ROI_THRESHOLD}%): {excluded_low_roi}")
        print(f"  Candidates for evaluation: {len(candidates)}")

        if not candidates:
            print("\n⚠ All archived agents are already in Global 50.")
            return

        # Sort candidates by their archived gauntlet score (highest first)
        # This prioritizes agents that were likely good performers
        candidates.sort(key=lambda a: a.get('gauntlet_score', 0), reverse=True)

        print(f"\n  Will evaluate all {len(candidates)} candidates (sorted by archived score)")

        # Show candidates
        print(f"\n{'='*70}")
        print("Candidates for Evaluation")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Archived Score':<15} {'ROI %':<10} {'Expectancy':<12} {'Trades':<8} {'Run Name':<30} {'M'}")
        print(f"{'-'*88}")

        for i, candidate in enumerate(candidates, 1):
            maverick_tag = " [M]" if candidate.get('is_maverick', False) else ""
            print(f"{i:<4} {candidate.get('gauntlet_score', 0):<15.2f} "
                  f"{candidate.get('roi', 0):<10.2f} "
                  f"{candidate.get('expectancy', 0):<12.4f} "
                  f"{candidate.get('total_trades', 0):<8} "
                  f"{candidate['run_name']:<30}{maverick_tag}")

        # Ask for confirmation
        print(f"\n{'='*70}")
        print("⚠ This will:")
        print(f"  1. Download {len(candidates)} archived agents from cloud")
        print(f"  2. Run full gauntlet evaluation for each agent")
        print(f"  3. Promote qualifying agents to Global 50 (moved from archive/ to agents/)")
        print(f"\nFilters applied:")
        print(f"  - Minimum archived ROI: {self.MIN_ROI_THRESHOLD}% (excluded {excluded_low_roi} agents)")
        print(f"  - Must beat all 4 minimum thresholds (Gauntlet, ROI, Expectancy, CV)")
        print(f"  - Must beat 2 of 3 metrics at P25 (Gauntlet, ROI, Expectancy)")
        print(f"  - Must beat 1 of 3 metrics at median (Gauntlet, ROI, Expectancy)")
        print(f"\nEstimated time: ~{len(candidates) * 2} minutes")

        while True:
            confirmation = input("\nProceed with archive fill? (yes/no): ").strip().lower()
            if confirmation in ['yes', 'y']:
                print("\n→ Starting archive fill...")
                break
            elif confirmation in ['no', 'n']:
                print("\n→ Archive fill cancelled.")
                return
            else:
                print("Please enter 'yes' or 'no'.")

        # Special handling: If no mavericks exist in Global 50, seed with best maverick from archive
        current_mavericks = [e for e in self.global_hof.entries if e.is_maverick]
        if len(current_mavericks) == 0:
            print(f"\n{'='*70}")
            print("SEEDING FIRST MAVERICK")
            print(f"{'='*70}")
            print("  No mavericks in Global 50. Selecting best maverick from archive")
            print("  based on ROI * Expectancy to establish maverick thresholds...")
            
            # Find all maverick candidates
            maverick_candidates = [c for c in candidates if c.get('is_maverick', False)]
            
            if maverick_candidates:
                # Sort by ROI * Expectancy (highest first)
                maverick_candidates.sort(
                    key=lambda a: a.get('roi', 0) * a.get('expectancy', 0), 
                    reverse=True
                )
                
                seed_candidate = maverick_candidates[0]
                roi_expectancy_score = seed_candidate.get('roi', 0) * seed_candidate.get('expectancy', 0)
                
                print(f"\n  Selected seed maverick:")
                print(f"    {seed_candidate['run_name']} (Agent {seed_candidate['agent_id']})")
                print(f"    ROI: {seed_candidate.get('roi', 0):.2f}%")
                print(f"    Expectancy: {seed_candidate.get('expectancy', 0):.4f}")
                print(f"    ROI × Expectancy: {roi_expectancy_score:.4f}")
                
                try:
                    # Download and load the seed maverick
                    filename = f"{seed_candidate['run_name']}_{seed_candidate['agent_id']}.pth"
                    local_agent_path = self.global_hof.local_archive_dir / filename
                    cloud_archive_path = f"{self.global_hof.cloud_base}/archive/{filename}"
                    
                    if not local_agent_path.exists():
                        print(f"  Downloading from archive...")
                        success = self.cloud_sync.download_file(cloud_archive_path, str(local_agent_path))
                        if not success:
                            print(f"  ✗ Failed to download seed maverick. Continuing with normal evaluation.")
                        else:
                            # Load and evaluate
                            print(f"  Loading agent...")
                            agent = DDPGAgent(agent_id=seed_candidate['agent_id'])
                            agent.load(str(local_agent_path))
                            
                            print(f"  Running gauntlet...")
                            new_score, metrics = self.run_gauntlet(
                                agent, 
                                f"{seed_candidate['run_name']}_{seed_candidate['agent_id']}", 
                                is_maverick=True
                            )
                            
                            print(f"  New Score: {new_score:.2f}")
                            print(f"  ROI: {metrics['roi']:.2f}% | Expectancy: {metrics['expectancy']:.4f} | CV: {metrics['cv']:.3f}")
                            
                            # Promote seed maverick (bypass thresholds by setting them to -inf)
                            self.global_hof.entry_threshold = float('-inf')
                            self.global_hof.roi_threshold = float('-inf')
                            self.global_hof.expectancy_threshold = float('-inf')
                            self.global_hof.cv_threshold = float('inf')
                            self.global_hof.gauntlet_median = float('-inf')
                            self.global_hof.roi_median = float('-inf')
                            self.global_hof.expectancy_median = float('-inf')
                            self.global_hof.gauntlet_p25 = float('-inf')
                            self.global_hof.roi_p25 = float('-inf')
                            self.global_hof.expectancy_p25 = float('-inf')
                            
                            promoted, rank = self.global_hof.check_and_promote(
                                agent=agent,
                                gauntlet_score=new_score,
                                generation=seed_candidate.get('generation', 0),
                                roi=metrics['roi'],
                                expectancy=metrics['expectancy'],
                                cv=metrics['cv'],
                                quality_ratio=metrics['quality_ratio'],
                                win_ratio=metrics['win_ratio'],
                                total_trades=metrics['total_trades'],
                                run_name=seed_candidate['run_name'],
                                suppress_threshold_output=True,
                                is_maverick=True
                            )
                            
                            if promoted:
                                print(f"  ✓ Seed maverick promoted to Global 50 (rank #{rank})")
                                
                                # Remove from archive
                                if local_agent_path.exists():
                                    local_agent_path.unlink()
                                scoresheet_path = self.global_hof.local_archive_dir / filename.replace('.pth', '.json')
                                if scoresheet_path.exists():
                                    scoresheet_path.unlink()
                                self.cloud_sync.delete_file(cloud_archive_path)
                                cloud_scoresheet_path = f"{self.global_hof.cloud_base}/archive/{filename.replace('.pth', '.json')}"
                                self.cloud_sync.delete_file(cloud_scoresheet_path)
                                
                                # Remove from candidates list so it doesn't get evaluated again
                                candidates = [c for c in candidates if (c['run_name'], c['agent_id']) != (seed_candidate['run_name'], seed_candidate['agent_id'])]
                                
                                # Recompute thresholds now that we have a maverick
                                recompute_thresholds_from_population()
                                
                                print(f"  ✓ Maverick thresholds established. Continuing with normal evaluation...")
                            else:
                                print(f"  ⚠ Failed to promote seed maverick (concurrent update?)")
                                
                except Exception as e:
                    print(f"  ✗ Error seeding maverick: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"  Continuing with normal evaluation...")
            else:
                print(f"  ⚠ No maverick candidates found in archive. Continuing with normal evaluation...")

        # Tiered promotion criteria (progressively relaxed):
        # Tier 0: Full criteria - minimums (4/4) + 2/3 P25 + 1/3 median
        # Tier 1: Remove median requirement - minimums (4/4) + 2/3 P25
        # Tier 2: Relax P25 to 1/3 - minimums (4/4) + 1/3 P25
        # Tier 3: Remove P25 requirement - minimums only (4/4)
        def should_promote_with_tier(gauntlet_score: float, roi: float, expectancy: float, cv: float, tier: int, is_maverick: bool = False) -> bool:
            """Check promotion with tiered criteria relaxation.
            
            Uses segregated thresholds for mavericks vs non-mavericks (Highlander rule).
            """
            # Select thresholds based on agent type (Maverick vs Standard)
            # Maverick Highlander Rule: Mavericks compete ONLY against other Mavericks
            # When no mavericks exist, maverick thresholds are set to -inf (open entry)
            if is_maverick:
                threshold_gauntlet = self.global_hof.maverick_entry_threshold
                threshold_roi = self.global_hof.maverick_roi_threshold
                threshold_expectancy = self.global_hof.maverick_expectancy_threshold
                threshold_cv = self.global_hof.maverick_cv_threshold
                
                p25_gauntlet = self.global_hof.maverick_gauntlet_p25
                p25_roi = self.global_hof.maverick_roi_p25
                p25_expectancy = self.global_hof.maverick_expectancy_p25
                
                median_gauntlet = self.global_hof.maverick_gauntlet_median
                median_roi = self.global_hof.maverick_roi_median
                median_expectancy = self.global_hof.maverick_expectancy_median
            else:
                # Standard agents compete against global thresholds
                threshold_gauntlet = self.global_hof.entry_threshold
                threshold_roi = self.global_hof.roi_threshold
                threshold_expectancy = self.global_hof.expectancy_threshold
                threshold_cv = self.global_hof.cv_threshold
                
                p25_gauntlet = self.global_hof.gauntlet_p25
                p25_roi = self.global_hof.roi_p25
                p25_expectancy = self.global_hof.expectancy_p25
                
                median_gauntlet = self.global_hof.gauntlet_median
                median_roi = self.global_hof.roi_median
                median_expectancy = self.global_hof.expectancy_median

            # Criterion 1: Must beat ALL 4 minimum thresholds
            if gauntlet_score <= threshold_gauntlet:
                return False
            if roi <= threshold_roi:
                return False
            if expectancy <= threshold_expectancy:
                return False
            # CV check: lower is better, so agent CV must be < threshold
            if cv >= threshold_cv:
                return False

            # Count P25 breaches (need 2 of 3)
            beats_gauntlet_p25 = gauntlet_score > p25_gauntlet
            beats_roi_p25 = roi > p25_roi
            beats_expectancy_p25 = expectancy > p25_expectancy
            # Need 2 of 3 (gauntlet, ROI, expectancy)
            metrics_to_check_p25 = [beats_gauntlet_p25, beats_roi_p25, beats_expectancy_p25]
            count_above_p25 = sum(metrics_to_check_p25)

            # Count median breaches (need 1 of 3)
            beats_gauntlet_median = gauntlet_score > median_gauntlet
            beats_roi_median = roi > median_roi
            beats_expectancy_median = expectancy > median_expectancy
            # Need 1 of 3 (gauntlet, ROI, expectancy)
            metrics_to_check_median = [beats_gauntlet_median, beats_roi_median, beats_expectancy_median]
            count_above_median = sum(metrics_to_check_median)

            if tier == 0:
                # Full criteria: 2/3 P25 + 1/3 median
                return count_above_p25 >= 2 and count_above_median >= 1
            elif tier == 1:
                # Remove median requirement: 2/3 P25 only
                return count_above_p25 >= 2
            elif tier == 2:
                # Relax P25 to 1/3: 1/3 P25 only
                return count_above_p25 >= 1
            else:
                # Tier 3+: minimums only (already passed above)
                return True

        def get_tier_description(tier: int) -> str:
            """Get human-readable description of tier criteria."""
            if tier == 0:
                return "Full criteria (minimums + 2/3 P25 + 1/3 median)"
            elif tier == 1:
                return "Relaxed (minimums + 2/3 P25, no median requirement)"
            elif tier == 2:
                return "Further relaxed (minimums + 1/3 P25)"
            else:
                return "Minimums only (no P25/median requirements)"

        # First pass: evaluate all candidates and cache results
        evaluated_candidates = []  # List of {candidate, agent, new_score, metrics, promoted}

        print(f"\n{'='*70}")
        print("Evaluating Archived Agents")
        print(f"{'='*70}")

        for i, candidate in enumerate(candidates, 1):
            maverick_tag = " [M]" if candidate.get('is_maverick', False) else ""
            print(f"\n[{i}/{len(candidates)}] {candidate['run_name']} (Agent {candidate['agent_id']}){maverick_tag}")
            print(f"  Archived Score: {candidate.get('gauntlet_score', 0):.2f}")

            try:
                # Download agent from archive
                filename = f"{candidate['run_name']}_{candidate['agent_id']}.pth"
                local_agent_path = self.global_hof.local_archive_dir / filename
                cloud_archive_path = f"{self.global_hof.cloud_base}/archive/{filename}"

                if not local_agent_path.exists():
                    print(f"  Downloading from archive...")
                    success = self.cloud_sync.download_file(cloud_archive_path, str(local_agent_path))
                    if not success:
                        print(f"  ✗ Failed to download agent. Skipping.")
                        continue

                # Load agent
                print(f"  Loading agent...")
                agent = DDPGAgent(agent_id=candidate['agent_id'])
                agent.load(str(local_agent_path))

                # Run gauntlet
                print(f"  Running gauntlet...")
                new_score, metrics = self.run_gauntlet(agent, f"{candidate['run_name']}_{candidate['agent_id']}", is_maverick=candidate.get('is_maverick', False))

                # Show results
                score_change = new_score - candidate.get('gauntlet_score', 0)
                score_symbol = "↑" if score_change > 0 else "↓" if score_change < 0 else "="
                print(f"  New Score: {new_score:.2f} ({score_symbol} {abs(score_change):.2f} from archived)")
                print(f"  ROI: {metrics['roi']:.2f}% | Expectancy: {metrics['expectancy']:.4f} | CV: {metrics['cv']:.3f}")
                print(f"  Trades: {metrics['total_trades']} | Quality: {metrics['quality_ratio']:.3f} | Win: {metrics['win_ratio']:.3f}")

                evaluated_candidates.append({
                    'candidate': candidate,
                    'agent': agent,
                    'new_score': new_score,
                    'metrics': metrics,
                    'promoted': False,
                    'local_agent_path': local_agent_path,
                    'cloud_archive_path': cloud_archive_path,
                    'filename': filename
                })

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

        # Multi-pass promotion with tiered criteria relaxation
        total_promoted = 0
        promotions_by_tier = {0: 0, 1: 0, 2: 0, 3: 0}

        for tier in range(4):  # Tiers 0-3
            # Check if we still have open slots
            current_size = len(self.global_hof.entries)
            if current_size >= self.global_hof.CAPACITY:
                print(f"\n{'='*70}")
                print(f"Global 50 is full. Stopping promotion passes.")
                print(f"{'='*70}")
                break

            open_slots = self.global_hof.CAPACITY - current_size

            print(f"\n{'='*70}")
            print(f"PROMOTION PASS - TIER {tier}")
            print(f"{'='*70}")
            print(f"  Criteria: {get_tier_description(tier)}")
            print(f"  Open slots: {open_slots}")
            print(f"  Remaining candidates: {sum(1 for ec in evaluated_candidates if not ec['promoted'])}")

            pass_promoted = 0

            # First, find all qualifying candidates at this tier
            qualifying_candidates = []
            for ec in evaluated_candidates:
                if ec['promoted']:
                    continue  # Already promoted in earlier tier

                candidate = ec['candidate']
                new_score = ec['new_score']
                metrics = ec['metrics']

                # Check if qualifies at this tier
                is_maverick = candidate.get('is_maverick', False)
                qualifies = should_promote_with_tier(new_score, metrics['roi'], metrics['expectancy'], metrics['cv'], tier, is_maverick=is_maverick)
                if qualifies:
                    qualifying_candidates.append(ec)

            # Sort qualifying candidates by new gauntlet score (highest first)
            qualifying_candidates.sort(key=lambda ec: ec['new_score'], reverse=True)

            print(f"  Qualifying candidates at Tier {tier}: {len(qualifying_candidates)}")

            # Promote qualifying candidates in order of score until slots are full
            for ec in qualifying_candidates:
                # Check if we still have slots
                if len(self.global_hof.entries) >= self.global_hof.CAPACITY:
                    break

                candidate = ec['candidate']
                new_score = ec['new_score']
                metrics = ec['metrics']
                agent = ec['agent']

                # Read is_maverick from archive metadata (if available)
                is_maverick = candidate.get('is_maverick', False)
                maverick_tag = " [M]" if is_maverick else ""
                
                print(f"\n  → {candidate['run_name']} (Agent {candidate['agent_id']}){maverick_tag}")
                print(f"    Score: {new_score:.2f} | ROI: {metrics['roi']:.2f}% | Expectancy: {metrics['expectancy']:.4f} | CV: {metrics['cv']:.3f} | Trades: {metrics['total_trades']}")

                # Attempt promotion - we bypass should_promote check since we did our own tiered check
                # Temporarily set all thresholds (minimums, medians, P25) to -inf/+inf to allow promotion
                # Note: The real qualification check was done via should_promote_with_tier() above
                self.global_hof.entry_threshold = float('-inf')
                self.global_hof.roi_threshold = float('-inf')
                self.global_hof.expectancy_threshold = float('-inf')
                self.global_hof.cv_threshold = float('inf')  # CV: lower is better, so +inf allows all
                self.global_hof.gauntlet_median = float('-inf')
                self.global_hof.roi_median = float('-inf')
                self.global_hof.expectancy_median = float('-inf')
                self.global_hof.gauntlet_p25 = float('-inf')
                self.global_hof.roi_p25 = float('-inf')
                self.global_hof.expectancy_p25 = float('-inf')
                
                promoted, rank = self.global_hof.check_and_promote(
                    agent=agent,
                    gauntlet_score=new_score,
                    generation=candidate.get('generation', 0),
                    roi=metrics['roi'],
                    expectancy=metrics['expectancy'],
                    cv=metrics['cv'],
                    quality_ratio=metrics['quality_ratio'],
                    win_ratio=metrics['win_ratio'],
                    total_trades=metrics['total_trades'],
                    run_name=candidate['run_name'],
                    suppress_threshold_output=True,  # Suppress -inf threshold during archive-fill
                    is_maverick=is_maverick
                )

                # Restore and recompute thresholds
                recompute_thresholds_from_population()

                if promoted:
                    ec['promoted'] = True
                    pass_promoted += 1
                    total_promoted += 1
                    promotions_by_tier[tier] += 1
                    print(f"    ✓ Promoted to Global 50 (Tier {tier})")

                    # Delete from archive
                    if ec['local_agent_path'].exists():
                        ec['local_agent_path'].unlink()
                    scoresheet_path = self.global_hof.local_archive_dir / ec['filename'].replace('.pth', '.json')
                    if scoresheet_path.exists():
                        scoresheet_path.unlink()
                    self.cloud_sync.delete_file(ec['cloud_archive_path'])
                    cloud_scoresheet_path = f"{self.global_hof.cloud_base}/archive/{ec['filename'].replace('.pth', '.json')}"
                    self.cloud_sync.delete_file(cloud_scoresheet_path)
                else:
                    print(f"    ⚠ Promotion failed (concurrent update?)")

            print(f"\n  Tier {tier} promoted: {pass_promoted} agents")

            # If no promotions at this tier and slots remain, move to next tier
            if pass_promoted == 0 and len(self.global_hof.entries) < self.global_hof.CAPACITY:
                print(f"  No qualifying candidates at Tier {tier}, relaxing criteria...")

        # Summary
        print(f"\n{'='*70}")
        print("Archive Fill Summary")
        print(f"{'='*70}")

        print(f"\n  Candidates evaluated: {len(evaluated_candidates)}")
        print(f"  Total promoted: {total_promoted}")
        print(f"\n  Promotions by tier:")
        for t, count in promotions_by_tier.items():
            if count > 0:
                print(f"    Tier {t} ({get_tier_description(t)}): {count}")

        # Reload and show final state
        self.global_hof._download_global_ledger()
        self.global_hof._load_local_ledger()
        self.global_hof._update_entry_threshold()

        final_size = len(self.global_hof.entries)
        print(f"\n  Final Global 50 size: {final_size}/{self.global_hof.CAPACITY}")

        if final_size < self.global_hof.CAPACITY:
            remaining_slots = self.global_hof.CAPACITY - final_size
            print(f"  Remaining empty slots: {remaining_slots}")
            print(f"\n  ⚠ Global 50 is not full. You may need to train more agents or")
            print(f"     lower quality thresholds to fill remaining slots.")

        if final_size > 0:
            all_scores = [e.gauntlet_score for e in self.global_hof.entries]
            all_rois = [e.roi for e in self.global_hof.entries]
            all_expectancies = [e.expectancy for e in self.global_hof.entries]
            all_cvs = [e.cv for e in self.global_hof.entries]
            all_trades = [e.total_trades for e in self.global_hof.entries]

            print(f"\n  Current Global 50 Metrics:")
            print(f"    Gauntlet:    min={min(all_scores):.2f}  mean={sum(all_scores)/len(all_scores):.2f}  max={max(all_scores):.2f}")
            print(f"    ROI:         min={min(all_rois):.2f}%  mean={sum(all_rois)/len(all_rois):.2f}%  max={max(all_rois):.2f}%")
            print(f"    Expectancy:  min={min(all_expectancies):.4f}  mean={sum(all_expectancies)/len(all_expectancies):.4f}  max={max(all_expectancies):.4f}")
            print(f"    CV:          min={min(all_cvs):.3f}  mean={sum(all_cvs)/len(all_cvs):.3f}  max={max(all_cvs):.3f}")
            print(f"    Total Trades: min={min(all_trades)}  mean={sum(all_trades)/len(all_trades):.1f}  max={max(all_trades)}")

            # Show updated thresholds
            print(f"\n  Updated Thresholds:")
            print(f"    Minimums:  Gauntlet={self.global_hof.entry_threshold:.2f}, ROI={self.global_hof.roi_threshold:.2f}%, Expectancy={self.global_hof.expectancy_threshold:.4f}, CV={self.global_hof.cv_threshold:.3f}")
            print(f"    Medians:   Gauntlet={self.global_hof.gauntlet_median:.2f}, ROI={self.global_hof.roi_median:.2f}%, Expectancy={self.global_hof.expectancy_median:.4f}")
            print(f"    P25:       Gauntlet={self.global_hof.gauntlet_p25:.2f}, ROI={self.global_hof.roi_p25:.2f}%, Expectancy={self.global_hof.expectancy_p25:.4f}")

        print(f"\n{'='*70}")

    def reactivate_agents(self):
        """
        Reactivate agents between archive and long-term-archive.
        
        - Moves qualifying agents from long-term-archive to archive (they can then be promoted)
        - Moves non-qualifying agents from archive to long-term-archive
        - All moves are verified before deletion from source location
        """
        print(f"\n{'='*70}")
        print("REACTIVATE MODE")
        print(f"{'='*70}")

        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            print("Cannot reactivate agents.")
            return

        # Load current Global 50 state
        print("\nLoading current Global 50 state...")
        self.global_hof._download_global_ledger()
        self.global_hof._load_local_ledger()
        self.global_hof._update_entry_threshold()

        print(f"\n  Current Global 50 size: {len(self.global_hof.entries)}/{self.global_hof.CAPACITY}")
        print(f"  Mavericks: {sum(1 for e in self.global_hof.entries if e.is_maverick)}/{self.global_hof.MAVERICK_CAP}")

        # Discover agents in long-term-archive
        print("\nDiscovering agents in long-term-archive...")
        long_term_agents = self._discover_long_term_archive_agents()
        print(f"  Found {len(long_term_agents)} agents in long-term-archive")

        # Discover agents in archive
        print("\nDiscovering agents in archive...")
        archive_agents = self._discover_archive_agents()
        print(f"  Found {len(archive_agents)} agents in archive")

        # Check which long-term-archive agents qualify
        print("\nChecking long-term-archive agents against current minimums...")
        qualifying_from_long_term = []
        sniper_mavericks_skipped = 0
        for agent_data in long_term_agents:
            # Skip agents missing required fields
            if 'gauntlet_score' not in agent_data or 'roi' not in agent_data:
                print(f"  ⚠ Skipping {agent_data.get('run_name', 'unknown')}_{agent_data.get('agent_id', 'unknown')}: missing required fields")
                continue
            
            is_maverick = agent_data.get('is_maverick', False)
            total_trades = agent_data.get('total_trades', 0)
            
            # Check anti-sniper rule for mavericks
            if is_maverick and not self._passes_anti_sniper_check(total_trades, is_maverick):
                sniper_mavericks_skipped += 1
                non_mavericks = [e for e in self.global_hof.entries if not e.is_maverick]
                max_non_maverick_trades = max(e.total_trades for e in non_mavericks) if non_mavericks else 0
                print(f"  ⚠ Skipping maverick {agent_data.get('run_name', 'unknown')}_{agent_data.get('agent_id', 'unknown')}: "
                      f"too few trades ({total_trades} < {max_non_maverick_trades}, Anti-Sniper Rule)")
                continue
            
            if self._meets_all_minimums(
                agent_data.get('gauntlet_score', 0.0),
                agent_data.get('roi', 0.0),
                agent_data.get('expectancy', 0.0),
                agent_data.get('cv', 100.0),
                is_maverick
            ):
                qualifying_from_long_term.append(agent_data)

        print(f"  {len(qualifying_from_long_term)} agents qualify for reactivation")
        if sniper_mavericks_skipped > 0:
            print(f"  {sniper_mavericks_skipped} maverick(s) skipped due to Anti-Sniper Rule")

        # Check which archive agents don't qualify
        print("\nChecking archive agents against current minimums...")
        non_qualifying_from_archive = []
        sniper_mavericks_moved = 0
        for agent_data in archive_agents:
            # Skip agents missing required fields
            if 'gauntlet_score' not in agent_data or 'roi' not in agent_data:
                print(f"  ⚠ Skipping {agent_data.get('run_name', 'unknown')}_{agent_data.get('agent_id', 'unknown')}: missing required fields")
                continue
            
            is_maverick = agent_data.get('is_maverick', False)
            total_trades = agent_data.get('total_trades', 0)
            
            # Check if maverick fails anti-sniper rule (should be moved to long-term-archive)
            if is_maverick and not self._passes_anti_sniper_check(total_trades, is_maverick):
                non_qualifying_from_archive.append(agent_data)
                sniper_mavericks_moved += 1
                non_mavericks = [e for e in self.global_hof.entries if not e.is_maverick]
                max_non_maverick_trades = max(e.total_trades for e in non_mavericks) if non_mavericks else 0
                print(f"  ⚠ Maverick {agent_data.get('run_name', 'unknown')}_{agent_data.get('agent_id', 'unknown')} "
                      f"fails Anti-Sniper Rule ({total_trades} < {max_non_maverick_trades} trades) - will move to long-term-archive")
                continue
            
            if not self._meets_all_minimums(
                agent_data.get('gauntlet_score', 0.0),
                agent_data.get('roi', 0.0),
                agent_data.get('expectancy', 0.0),
                agent_data.get('cv', 100.0),
                is_maverick
            ):
                non_qualifying_from_archive.append(agent_data)

        print(f"  {len(non_qualifying_from_archive)} agents should be moved to long-term-archive")
        if sniper_mavericks_moved > 0:
            print(f"  {sniper_mavericks_moved} maverick(s) moved due to Anti-Sniper Rule")

        # Move qualifying agents from long-term-archive to archive
        if qualifying_from_long_term:
            print(f"\n{'='*70}")
            print(f"Moving {len(qualifying_from_long_term)} qualifying agents from long-term-archive to archive...")
            print(f"{'='*70}")
            for agent_data in qualifying_from_long_term:
                self._move_agent_long_term_to_archive(agent_data)
        else:
            print("\nNo agents to move from long-term-archive to archive.")

        # Move non-qualifying agents from archive to long-term-archive
        if non_qualifying_from_archive:
            print(f"\n{'='*70}")
            print(f"Moving {len(non_qualifying_from_archive)} non-qualifying agents from archive to long-term-archive...")
            print(f"{'='*70}")
            for agent_data in non_qualifying_from_archive:
                self._move_agent_archive_to_long_term(agent_data)
        else:
            print("\nNo agents to move from archive to long-term-archive.")

        print(f"\n{'='*70}")
        print("✓ Reactivation Complete!")
        print(f"{'='*70}")
        print(f"  Reactivated:      {len(qualifying_from_long_term)} agents (long-term-archive → archive)")
        print(f"  Moved to long-term: {len(non_qualifying_from_archive)} agents (archive → long-term-archive)")

    def _passes_anti_sniper_check(self, total_trades: int, is_maverick: bool) -> bool:
        """
        Anti-sniper check for Mavericks: Prevent mavericks with fewer trades than
        the highest-trade non-maverick agent from being considered qualified.
        
        Args:
            total_trades: Agent's total number of trades
            is_maverick: Whether this is a maverick agent
            
        Returns:
            True if agent passes the anti-sniper check (or is not a maverick), False otherwise
        """
        # Only applies to mavericks
        if not is_maverick:
            return True
        
        # If no non-mavericks exist, allow the maverick (edge case)
        non_mavericks = [e for e in self.global_hof.entries if not e.is_maverick]
        if not non_mavericks:
            return True
        
        # Maverick must have at least as many trades as the highest-trade non-maverick
        max_non_maverick_trades = max(e.total_trades for e in non_mavericks)
        return total_trades >= max_non_maverick_trades

    def _meets_all_minimums(self, gauntlet_score: float, roi: float, expectancy: float, cv: float, is_maverick: bool) -> bool:
        """
        Check if an agent meets all 4 minimum thresholds (gauntlet, ROI, expectancy, CV).
        Uses segregated thresholds for mavericks vs non-mavericks.
        
        Args:
            gauntlet_score: Agent's gauntlet score
            roi: Agent's ROI percentage
            expectancy: Agent's expectancy metric
            cv: Agent's coefficient of variation (lower is better)
            is_maverick: Whether this is a maverick agent
            
        Returns:
            True if agent meets all 4 minimums, False otherwise
        """
        # Handle edge case: if Global 50 is empty, no agent can meet minimums
        if len(self.global_hof.entries) == 0:
            return False

        # Select thresholds based on agent type (Maverick vs Standard)
        # If maverick_count > 0, use maverick thresholds; otherwise use standard thresholds
        # (This matches the logic in analyze_promotion)
        if is_maverick and self.global_hof.maverick_count > 0:
            threshold_gauntlet = self.global_hof.maverick_entry_threshold
            threshold_roi = self.global_hof.maverick_roi_threshold
            threshold_expectancy = self.global_hof.maverick_expectancy_threshold
            threshold_cv = self.global_hof.maverick_cv_threshold
        else:
            threshold_gauntlet = self.global_hof.entry_threshold
            threshold_roi = self.global_hof.roi_threshold
            threshold_expectancy = self.global_hof.expectancy_threshold
            threshold_cv = self.global_hof.cv_threshold

        # Check all 4 minimums
        passes_gauntlet = gauntlet_score > threshold_gauntlet
        passes_roi = roi > threshold_roi
        passes_expectancy = expectancy > threshold_expectancy
        passes_cv = cv < threshold_cv  # CV: lower is better

        return passes_gauntlet and passes_roi and passes_expectancy and passes_cv

    def _discover_long_term_archive_agents(self) -> List[dict]:
        """
        Discover all agents in long-term-archive (both local and cloud).
        
        Returns:
            List of dicts with agent metadata (run_name, agent_id, gauntlet_score, etc.)
        """
        long_term_agents = []
        local_long_term_dir = self.global_hof.local_dir / "long-term-archive"
        cloud_long_term_prefix = f"{self.global_hof.cloud_base}/long-term-archive/"

        # Check local long-term-archive
        if local_long_term_dir.exists():
            print(f"  Checking local: {local_long_term_dir}")
            for json_file in local_long_term_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    long_term_agents.append(data)
                except Exception as e:
                    print(f"  ⚠ Could not parse {json_file}: {e}")

        # Check cloud long-term-archive
        try:
            if self.cloud_sync.provider == "gcs":
                blobs = self.cloud_sync.bucket.list_blobs(prefix=cloud_long_term_prefix)
                for blob in blobs:
                    if blob.name.endswith('.json'):
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                            temp_path = tmp.name
                        try:
                            blob.download_to_filename(temp_path)
                            with open(temp_path, 'r') as f:
                                data = json.load(f)
                            # Only add if not already in list (avoid duplicates)
                            key = (data.get('run_name'), data.get('agent_id'))
                            if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in long_term_agents):
                                long_term_agents.append(data)
                        except Exception as e:
                            print(f"  ⚠ Could not parse {blob.name}: {e}")
                        finally:
                            import os
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)

            elif self.cloud_sync.provider == "s3":
                paginator = self.cloud_sync.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.cloud_sync.bucket_name, Prefix=cloud_long_term_prefix)
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.json'):
                                import tempfile
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                    temp_path = tmp.name
                                try:
                                    self.cloud_sync.client.download_file(
                                        self.cloud_sync.bucket_name, obj['Key'], temp_path
                                    )
                                    with open(temp_path, 'r') as f:
                                        data = json.load(f)
                                    key = (data.get('run_name'), data.get('agent_id'))
                                    if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in long_term_agents):
                                        long_term_agents.append(data)
                                except Exception as e:
                                    print(f"  ⚠ Could not parse {obj['Key']}: {e}")
                                finally:
                                    import os
                                    if os.path.exists(temp_path):
                                        os.unlink(temp_path)

            elif self.cloud_sync.provider == "azure":
                blob_list = self.cloud_sync.container_client.list_blobs(name_starts_with=cloud_long_term_prefix)
                for blob in blob_list:
                    if blob.name.endswith('.json'):
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                            temp_path = tmp.name
                        try:
                            blob_client = self.cloud_sync.container_client.get_blob_client(blob.name)
                            with open(temp_path, 'wb') as f:
                                f.write(blob_client.download_blob().readall())
                            with open(temp_path, 'r') as f:
                                data = json.load(f)
                            key = (data.get('run_name'), data.get('agent_id'))
                            if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in long_term_agents):
                                long_term_agents.append(data)
                        except Exception as e:
                            print(f"  ⚠ Could not parse {blob.name}: {e}")
                        finally:
                            import os
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
        except Exception as e:
            print(f"  ⚠ Error listing cloud long-term-archive: {e}")

        return long_term_agents

    def _move_agent_long_term_to_archive(self, agent_data: dict):
        """
        Move an agent from long-term-archive to archive (both local and cloud).
        Verifies the move before deleting from source.
        
        Args:
            agent_data: Agent metadata dict
        """
        filename = f"{agent_data['run_name']}_{agent_data['agent_id']}.pth"
        scoresheet_filename = filename.replace('.pth', '.json')
        
        local_long_term_dir = self.global_hof.local_dir / "long-term-archive"
        local_archive_dir = self.global_hof.local_archive_dir
        
        # Ensure directories exist
        local_long_term_dir.mkdir(parents=True, exist_ok=True)
        local_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Local: Move from long-term-archive to archive
        local_src_pth = local_long_term_dir / filename
        local_dst_pth = local_archive_dir / filename
        local_src_json = local_long_term_dir / scoresheet_filename
        local_dst_json = local_archive_dir / scoresheet_filename

        # Download from cloud if not in local cache
        if not local_src_pth.exists():
            cloud_long_term_pth = f"{self.global_hof.cloud_base}/long-term-archive/{filename}"
            if self.cloud_sync.file_exists(cloud_long_term_pth):
                self.cloud_sync.download_file(cloud_long_term_pth, str(local_src_pth))
                print(f"   ✓ Downloaded {filename} from cloud long-term-archive")

        if not local_src_json.exists():
            cloud_long_term_json = f"{self.global_hof.cloud_base}/long-term-archive/{scoresheet_filename}"
            if self.cloud_sync.file_exists(cloud_long_term_json):
                self.cloud_sync.download_file(cloud_long_term_json, str(local_src_json))
                print(f"   ✓ Downloaded {scoresheet_filename} from cloud long-term-archive")

        # Move locally
        if local_src_pth.exists():
            shutil.move(str(local_src_pth), str(local_dst_pth))
            print(f"   ✓ Moved locally: {filename} (long-term-archive → archive)")
        else:
            print(f"   ⚠ {filename} not found locally, will download to archive")

        if local_src_json.exists():
            shutil.move(str(local_src_json), str(local_dst_json))
        else:
            # Create JSON from agent_data if missing
            with open(local_dst_json, 'w') as f:
                json.dump(agent_data, f, indent=2)

        # Cloud: Upload to archive/ with verification
        cloud_archive_pth = f"{self.global_hof.cloud_base}/archive/{filename}"
        cloud_archive_json = f"{self.global_hof.cloud_base}/archive/{scoresheet_filename}"
        cloud_long_term_pth = f"{self.global_hof.cloud_base}/long-term-archive/{filename}"
        cloud_long_term_json = f"{self.global_hof.cloud_base}/long-term-archive/{scoresheet_filename}"

        if local_dst_pth.exists():
            if self.global_hof.enabled:
                self.cloud_sync.upload_file_verified(str(local_dst_pth), cloud_archive_pth)
            else:
                self.cloud_sync.upload_file(str(local_dst_pth), cloud_archive_pth, background=False)
        
        if local_dst_json.exists():
            if self.global_hof.enabled:
                self.cloud_sync.upload_file_verified(str(local_dst_json), cloud_archive_json)
            else:
                self.cloud_sync.upload_file(str(local_dst_json), cloud_archive_json, background=False)

        # Delete from cloud long-term-archive ONLY after confirming archive exists
        if self.cloud_sync.file_exists(cloud_archive_pth):
            if self.cloud_sync.delete_file(cloud_long_term_pth):
                print(f"   ✓ Deleted from cloud long-term-archive: {filename}")
            else:
                print(f"   ⚠ Failed to delete from cloud long-term-archive: {filename}")
            
            if self.cloud_sync.file_exists(cloud_archive_json):
                if self.cloud_sync.delete_file(cloud_long_term_json):
                    print(f"   ✓ Deleted from cloud long-term-archive: {scoresheet_filename}")
        else:
            print(f"   ⚠ Archive upload failed, keeping in long-term-archive: {filename}")

        # Delete from local long-term-archive if still exists
        if local_src_pth.exists():
            local_src_pth.unlink()
        if local_src_json.exists():
            local_src_json.unlink()

    def _move_agent_archive_to_long_term(self, agent_data: dict):
        """
        Move an agent from archive to long-term-archive (both local and cloud).
        Verifies the move before deleting from source.
        
        Args:
            agent_data: Agent metadata dict
        """
        filename = f"{agent_data['run_name']}_{agent_data['agent_id']}.pth"
        scoresheet_filename = filename.replace('.pth', '.json')
        
        local_archive_dir = self.global_hof.local_archive_dir
        local_long_term_dir = self.global_hof.local_dir / "long-term-archive"
        
        # Ensure directories exist
        local_long_term_dir.mkdir(parents=True, exist_ok=True)
        
        # Local: Move from archive to long-term-archive
        local_src_pth = local_archive_dir / filename
        local_dst_pth = local_long_term_dir / filename
        local_src_json = local_archive_dir / scoresheet_filename
        local_dst_json = local_long_term_dir / scoresheet_filename

        # Download from cloud if not in local cache
        if not local_src_pth.exists():
            cloud_archive_pth = f"{self.global_hof.cloud_base}/archive/{filename}"
            if self.cloud_sync.file_exists(cloud_archive_pth):
                self.cloud_sync.download_file(cloud_archive_pth, str(local_src_pth))
                print(f"   ✓ Downloaded {filename} from cloud archive")

        if not local_src_json.exists():
            cloud_archive_json = f"{self.global_hof.cloud_base}/archive/{scoresheet_filename}"
            if self.cloud_sync.file_exists(cloud_archive_json):
                self.cloud_sync.download_file(cloud_archive_json, str(local_src_json))
                print(f"   ✓ Downloaded {scoresheet_filename} from cloud archive")

        # Move locally
        if local_src_pth.exists():
            shutil.move(str(local_src_pth), str(local_dst_pth))
            print(f"   ✓ Moved locally: {filename} (archive → long-term-archive)")
        else:
            print(f"   ⚠ {filename} not found locally, will download to long-term-archive")

        if local_src_json.exists():
            shutil.move(str(local_src_json), str(local_dst_json))
        else:
            # Create JSON from agent_data if missing
            with open(local_dst_json, 'w') as f:
                json.dump(agent_data, f, indent=2)

        # Cloud: Upload to long-term-archive/ with verification
        cloud_long_term_pth = f"{self.global_hof.cloud_base}/long-term-archive/{filename}"
        cloud_long_term_json = f"{self.global_hof.cloud_base}/long-term-archive/{scoresheet_filename}"
        cloud_archive_pth = f"{self.global_hof.cloud_base}/archive/{filename}"
        cloud_archive_json = f"{self.global_hof.cloud_base}/archive/{scoresheet_filename}"

        if local_dst_pth.exists():
            if self.global_hof.enabled:
                self.cloud_sync.upload_file_verified(str(local_dst_pth), cloud_long_term_pth)
            else:
                self.cloud_sync.upload_file(str(local_dst_pth), cloud_long_term_pth, background=False)
        
        if local_dst_json.exists():
            if self.global_hof.enabled:
                self.cloud_sync.upload_file_verified(str(local_dst_json), cloud_long_term_json)
            else:
                self.cloud_sync.upload_file(str(local_dst_json), cloud_long_term_json, background=False)

        # Delete from cloud archive ONLY after confirming long-term-archive exists
        if self.cloud_sync.file_exists(cloud_long_term_pth):
            if self.cloud_sync.delete_file(cloud_archive_pth):
                print(f"   ✓ Deleted from cloud archive: {filename}")
            else:
                print(f"   ⚠ Failed to delete from cloud archive: {filename}")
            
            if self.cloud_sync.file_exists(cloud_long_term_json):
                if self.cloud_sync.delete_file(cloud_archive_json):
                    print(f"   ✓ Deleted from cloud archive: {scoresheet_filename}")
        else:
            print(f"   ⚠ Long-term-archive upload failed, keeping in archive: {filename}")

        # Delete from local archive if still exists
        if local_src_pth.exists():
            local_src_pth.unlink()
        if local_src_json.exists():
            local_src_json.unlink()

    def _discover_archive_agents(self) -> List[dict]:
        """
        Discover all archived agents in cloud storage.

        Returns:
            List of dicts with agent metadata (run_name, agent_id, gauntlet_score, etc.)
        """
        archive_agents = []
        cloud_archive_prefix = f"{self.global_hof.cloud_base}/archive/"

        try:
            if self.cloud_sync.provider == "gcs":
                blobs = self.cloud_sync.bucket.list_blobs(prefix=cloud_archive_prefix)
                for blob in blobs:
                    if blob.name.endswith('.json'):
                        # Download and parse the JSON scoresheet
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                            temp_path = tmp.name
                        try:
                            blob.download_to_filename(temp_path)
                            with open(temp_path, 'r') as f:
                                data = json.load(f)
                            archive_agents.append(data)
                        except Exception as e:
                            print(f"  ⚠ Could not parse {blob.name}: {e}")
                        finally:
                            import os
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    elif blob.name.endswith('.pth'):
                        # Check if we have a corresponding JSON
                        json_name = blob.name.replace('.pth', '.json')
                        # We'll handle .pth files without .json below
                        pass

            elif self.cloud_sync.provider == "s3":
                paginator = self.cloud_sync.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.cloud_sync.bucket_name, Prefix=cloud_archive_prefix)
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.json'):
                                import tempfile
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                    temp_path = tmp.name
                                try:
                                    self.cloud_sync.client.download_file(
                                        self.cloud_sync.bucket_name, obj['Key'], temp_path
                                    )
                                    with open(temp_path, 'r') as f:
                                        data = json.load(f)
                                    archive_agents.append(data)
                                except Exception as e:
                                    print(f"  ⚠ Could not parse {obj['Key']}: {e}")
                                finally:
                                    import os
                                    if os.path.exists(temp_path):
                                        os.unlink(temp_path)

            elif self.cloud_sync.provider == "azure":
                blob_list = self.cloud_sync.container_client.list_blobs(name_starts_with=cloud_archive_prefix)
                for blob in blob_list:
                    if blob.name.endswith('.json'):
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                            temp_path = tmp.name
                        try:
                            blob_client = self.cloud_sync.container_client.get_blob_client(blob.name)
                            with open(temp_path, 'wb') as f:
                                f.write(blob_client.download_blob().readall())
                            with open(temp_path, 'r') as f:
                                data = json.load(f)
                            archive_agents.append(data)
                        except Exception as e:
                            print(f"  ⚠ Could not parse {blob.name}: {e}")
                        finally:
                            import os
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)

            else:
                # Local provider - check local archive directory
                print(f"  Checking local archive: {self.global_hof.local_archive_dir}")
                if self.global_hof.local_archive_dir.exists():
                    for json_file in self.global_hof.local_archive_dir.glob("*.json"):
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            archive_agents.append(data)
                        except Exception as e:
                            print(f"  ⚠ Could not parse {json_file}: {e}")

        except Exception as e:
            print(f"  ✗ Error listing archive: {e}")
            import traceback
            traceback.print_exc()

        # Handle .pth files without corresponding .json scoresheets
        # Extract run_name and agent_id from filename
        pth_files = set()
        json_keys = {(a['run_name'], a['agent_id']) for a in archive_agents}

        try:
            if self.cloud_sync.provider == "gcs":
                blobs = self.cloud_sync.bucket.list_blobs(prefix=cloud_archive_prefix)
                for blob in blobs:
                    if blob.name.endswith('.pth'):
                        filename = blob.name.split('/')[-1]
                        # Parse filename: [run_name]_[agent_id].pth
                        parts = filename.replace('.pth', '').rsplit('_', 1)
                        if len(parts) == 2:
                            run_name = parts[0]
                            try:
                                agent_id = int(parts[1])
                                if (run_name, agent_id) not in json_keys:
                                    # No JSON scoresheet - create minimal entry
                                    archive_agents.append({
                                        'run_name': run_name,
                                        'agent_id': agent_id,
                                        'gauntlet_score': 0.0,
                                        'generation': 0,
                                        'roi': 0.0,
                                        'expectancy': 0.0
                                    })
                            except ValueError:
                                pass
        except Exception:
            pass  # Already handled above

        return archive_agents

    def mark_maverick_by_run_name(self, run_name: str, is_maverick: bool = True):
        """
        Mark all agents with a given run_name as Mavericks (or unmark them).
        
        Updates:
        - global50.json entries
        - archive/ JSON files (local and cloud)
        - long-term-archive/ JSON files (local and cloud)
        
        Args:
            run_name: Run name to match (e.g., 'twilight-haze-313')
            is_maverick: True to mark as Maverick, False to unmark
        """
        print(f"\n{'='*70}")
        print(f"{'Marking' if is_maverick else 'Unmarking'} Maverick Status by Run Name")
        print(f"{'='*70}")
        print(f"Run Name: {run_name}")
        print(f"Context Window: {self.global_hof.context_window_id}")
        
        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            print("Cannot mark Mavericks in cloud storage.")
            return
        
        # Load current Global 50 state
        print("\n1. Loading Global 50 state...")
        self.global_hof._download_global_ledger()
        self.global_hof._load_local_ledger()
        
        # Count current mavericks in Global 50
        current_maverick_count = sum(1 for e in self.global_hof.entries if e.is_maverick)
        print(f"   Current Maverick count in Global 50: {current_maverick_count}/{self.global_hof.MAVERICK_CAP}")
        
        # Find matching entries in Global 50
        matched_g50 = [e for e in self.global_hof.entries if e.run_name == run_name]
        
        # Check maverick cap if setting flag
        if is_maverick:
            new_mavericks = sum(1 for e in matched_g50 if not e.is_maverick)
            if current_maverick_count + new_mavericks > self.global_hof.MAVERICK_CAP:
                print(f"\n   ⚠ WARNING: Would exceed Maverick cap of {self.global_hof.MAVERICK_CAP}")
                print(f"      Current: {current_maverick_count}, Adding: {new_mavericks}, Cap: {self.global_hof.MAVERICK_CAP}")
                print(f"      Note: This only checks Global 50. Archive entries will still be updated.")
                response = input(f"      Continue updating archive/long-term-archive anyway? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    return
        
        # Update Global 50 entries
        g50_changes = 0
        if matched_g50:
            print(f"\n2. Updating Global 50 entries ({len(matched_g50)} found)...")
            for entry in matched_g50:
                if entry.is_maverick != is_maverick:
                    entry.is_maverick = is_maverick
                    g50_changes += 1
                    action = "SET" if is_maverick else "UNSET"
                    print(f"   {action}: {entry.run_name}_{entry.agent_id}")
            
            if g50_changes > 0:
                # Save Global 50
                self.global_hof._save_local_ledger()
                self.global_hof._upload_global_ledger()
                print(f"   ✓ Updated {g50_changes} Global 50 entry/entries")
            else:
                print(f"   No changes needed in Global 50")
        else:
            print(f"\n2. No matching entries in Global 50")
        
        # Update archive JSON files
        print(f"\n3. Updating archive JSON files...")
        archive_changes = 0
        if self.global_hof.local_archive_dir.exists():
            for json_file in self.global_hof.local_archive_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get('run_name') == run_name:
                        if data.get('is_maverick', False) != is_maverick:
                            data['is_maverick'] = is_maverick
                            with open(json_file, 'w') as f:
                                json.dump(data, f, indent=2)
                            
                            # Upload to cloud
                            filename = json_file.name
                            cloud_path = f"{self.global_hof.cloud_base}/archive/{filename}"
                            if self.cloud_sync.upload_file_verified(str(json_file), cloud_path):
                                archive_changes += 1
                                action = "SET" if is_maverick else "UNSET"
                                print(f"   {action}: {filename}")
                except Exception as e:
                    print(f"   ⚠ Error updating {json_file.name}: {e}")
        
        # Also check cloud archive for files not in local cache
        cloud_archive_changes = 0
        cloud_archive_prefix = f"{self.global_hof.cloud_base}/archive/"
        try:
            if self.cloud_sync.provider == "gcs":
                blobs = list(self.cloud_sync.bucket.list_blobs(prefix=cloud_archive_prefix))
                for blob in blobs:
                    if blob.name.endswith('.json'):
                        filename = blob.name.split('/')[-1]
                        # Check if this file matches the run_name pattern
                        if run_name in filename:
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                temp_path = tmp.name
                            try:
                                blob.download_to_filename(temp_path)
                                with open(temp_path, 'r') as f:
                                    data = json.load(f)
                                
                                if data.get('run_name') == run_name and data.get('is_maverick', False) != is_maverick:
                                    data['is_maverick'] = is_maverick
                                    with open(temp_path, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    
                                    if self.cloud_sync.upload_file_verified(temp_path, blob.name):
                                        cloud_archive_changes += 1
                                        action = "SET" if is_maverick else "UNSET"
                                        print(f"   {action} (cloud): {filename}")
                            except Exception as e:
                                print(f"   ⚠ Error updating {blob.name}: {e}")
                            finally:
                                import os
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
            elif self.cloud_sync.provider == "s3":
                paginator = self.cloud_sync.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.cloud_sync.bucket_name, Prefix=cloud_archive_prefix)
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.json'):
                                filename = obj['Key'].split('/')[-1]
                                if run_name in filename:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                        temp_path = tmp.name
                                    try:
                                        self.cloud_sync.client.download_file(
                                            self.cloud_sync.bucket_name, obj['Key'], temp_path
                                        )
                                        with open(temp_path, 'r') as f:
                                            data = json.load(f)
                                        
                                        if data.get('run_name') == run_name and data.get('is_maverick', False) != is_maverick:
                                            data['is_maverick'] = is_maverick
                                            with open(temp_path, 'w') as f:
                                                json.dump(data, f, indent=2)
                                            
                                            self.cloud_sync.client.upload_file(
                                                temp_path, self.cloud_sync.bucket_name, obj['Key']
                                            )
                                            cloud_archive_changes += 1
                                            action = "SET" if is_maverick else "UNSET"
                                            print(f"   {action} (cloud): {filename}")
                                    except Exception as e:
                                        print(f"   ⚠ Error updating {obj['Key']}: {e}")
                                    finally:
                                        import os
                                        if os.path.exists(temp_path):
                                            os.unlink(temp_path)
            elif self.cloud_sync.provider == "azure":
                blob_list = self.cloud_sync.container_client.list_blobs(name_starts_with=cloud_archive_prefix)
                for blob in blob_list:
                    if blob.name.endswith('.json'):
                        filename = blob.name.split('/')[-1]
                        if run_name in filename:
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                temp_path = tmp.name
                            try:
                                blob_client = self.cloud_sync.container_client.get_blob_client(blob.name)
                                with open(temp_path, 'wb') as f:
                                    f.write(blob_client.download_blob().readall())
                                with open(temp_path, 'r') as f:
                                    data = json.load(f)
                                
                                if data.get('run_name') == run_name and data.get('is_maverick', False) != is_maverick:
                                    data['is_maverick'] = is_maverick
                                    with open(temp_path, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    
                                    with open(temp_path, 'rb') as f:
                                        blob_client.upload_blob(f, overwrite=True)
                                    cloud_archive_changes += 1
                                    action = "SET" if is_maverick else "UNSET"
                                    print(f"   {action} (cloud): {filename}")
                            except Exception as e:
                                print(f"   ⚠ Error updating {blob.name}: {e}")
                            finally:
                                import os
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
        except Exception as e:
            print(f"   ⚠ Error checking cloud archive: {e}")
        
        total_archive_changes = archive_changes + cloud_archive_changes
        if total_archive_changes > 0:
            print(f"   ✓ Updated {total_archive_changes} archive file(s)")
        else:
            print(f"   No matching archive files found")
        
        # Update long-term-archive JSON files
        print(f"\n4. Updating long-term-archive JSON files...")
        long_term_changes = 0
        local_long_term_dir = self.global_hof.local_dir / "long-term-archive"
        
        if local_long_term_dir.exists():
            for json_file in local_long_term_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get('run_name') == run_name:
                        if data.get('is_maverick', False) != is_maverick:
                            data['is_maverick'] = is_maverick
                            with open(json_file, 'w') as f:
                                json.dump(data, f, indent=2)
                            
                            # Upload to cloud
                            filename = json_file.name
                            cloud_path = f"{self.global_hof.cloud_base}/long-term-archive/{filename}"
                            if self.cloud_sync.upload_file_verified(str(json_file), cloud_path):
                                long_term_changes += 1
                                action = "SET" if is_maverick else "UNSET"
                                print(f"   {action}: {filename}")
                except Exception as e:
                    print(f"   ⚠ Error updating {json_file.name}: {e}")
        
        # Check cloud long-term-archive
        cloud_long_term_changes = 0
        cloud_long_term_prefix = f"{self.global_hof.cloud_base}/long-term-archive/"
        try:
            if self.cloud_sync.provider == "gcs":
                blobs = list(self.cloud_sync.bucket.list_blobs(prefix=cloud_long_term_prefix))
                for blob in blobs:
                    if blob.name.endswith('.json'):
                        filename = blob.name.split('/')[-1]
                        if run_name in filename:
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                temp_path = tmp.name
                            try:
                                blob.download_to_filename(temp_path)
                                with open(temp_path, 'r') as f:
                                    data = json.load(f)
                                
                                if data.get('run_name') == run_name and data.get('is_maverick', False) != is_maverick:
                                    data['is_maverick'] = is_maverick
                                    with open(temp_path, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    
                                    if self.cloud_sync.upload_file_verified(temp_path, blob.name):
                                        cloud_long_term_changes += 1
                                        action = "SET" if is_maverick else "UNSET"
                                        print(f"   {action} (cloud): {filename}")
                            except Exception as e:
                                print(f"   ⚠ Error updating {blob.name}: {e}")
                            finally:
                                import os
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
            elif self.cloud_sync.provider == "s3":
                paginator = self.cloud_sync.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.cloud_sync.bucket_name, Prefix=cloud_long_term_prefix)
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.json'):
                                filename = obj['Key'].split('/')[-1]
                                if run_name in filename:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                        temp_path = tmp.name
                                    try:
                                        self.cloud_sync.client.download_file(
                                            self.cloud_sync.bucket_name, obj['Key'], temp_path
                                        )
                                        with open(temp_path, 'r') as f:
                                            data = json.load(f)
                                        
                                        if data.get('run_name') == run_name and data.get('is_maverick', False) != is_maverick:
                                            data['is_maverick'] = is_maverick
                                            with open(temp_path, 'w') as f:
                                                json.dump(data, f, indent=2)
                                            
                                            self.cloud_sync.client.upload_file(
                                                temp_path, self.cloud_sync.bucket_name, obj['Key']
                                            )
                                            cloud_long_term_changes += 1
                                            action = "SET" if is_maverick else "UNSET"
                                            print(f"   {action} (cloud): {filename}")
                                    except Exception as e:
                                        print(f"   ⚠ Error updating {obj['Key']}: {e}")
                                    finally:
                                        import os
                                        if os.path.exists(temp_path):
                                            os.unlink(temp_path)
            elif self.cloud_sync.provider == "azure":
                blob_list = self.cloud_sync.container_client.list_blobs(name_starts_with=cloud_long_term_prefix)
                for blob in blob_list:
                    if blob.name.endswith('.json'):
                        filename = blob.name.split('/')[-1]
                        if run_name in filename:
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                temp_path = tmp.name
                            try:
                                blob_client = self.cloud_sync.container_client.get_blob_client(blob.name)
                                with open(temp_path, 'wb') as f:
                                    f.write(blob_client.download_blob().readall())
                                with open(temp_path, 'r') as f:
                                    data = json.load(f)
                                
                                if data.get('run_name') == run_name and data.get('is_maverick', False) != is_maverick:
                                    data['is_maverick'] = is_maverick
                                    with open(temp_path, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    
                                    with open(temp_path, 'rb') as f:
                                        blob_client.upload_blob(f, overwrite=True)
                                    cloud_long_term_changes += 1
                                    action = "SET" if is_maverick else "UNSET"
                                    print(f"   {action} (cloud): {filename}")
                            except Exception as e:
                                print(f"   ⚠ Error updating {blob.name}: {e}")
                            finally:
                                import os
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
        except Exception as e:
            print(f"   ⚠ Error checking cloud long-term-archive: {e}")
        
        total_long_term_changes = long_term_changes + cloud_long_term_changes
        if total_long_term_changes > 0:
            print(f"   ✓ Updated {total_long_term_changes} long-term-archive file(s)")
        else:
            print(f"   No matching long-term-archive files found")
        
        # Summary
        total_changes = g50_changes + total_archive_changes + total_long_term_changes
        print(f"\n{'='*70}")
        print(f"✓ Complete")
        print(f"  Global 50: {g50_changes} change(s)")
        print(f"  Archive: {total_archive_changes} change(s) ({archive_changes} local, {cloud_archive_changes} cloud)")
        print(f"  Long-term-archive: {total_long_term_changes} change(s) ({long_term_changes} local, {cloud_long_term_changes} cloud)")
        print(f"  Total: {total_changes} change(s)")
        
        if total_changes > 0:
            print(f"\n  Note: Run --mirror to verify all files are synced")
        print(f"{'='*70}")

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
            print(f"\nPromoted Agents (efficiency-adjusted gauntlet scores):")
            print("-" * 70)
            for r in results:
                if r['promoted']:
                    maverick_tag = " [M]" if r.get('is_maverick', False) else ""
                    trades = r.get('total_trades', r.get('metrics', {}).get('total_trades', 0))
                    print(f"  {r['agent_name']:.<50} {r['gauntlet_score']:>10.2f}  Trades: {trades:>6}{maverick_tag}")

        if self.global_hof.enabled:
            stats = self.global_hof.get_stats()
            print(f"\nGlobal 50 Status:")
            print(f"  Size:              {stats['size']:>10} / 50")
            print(f"  Entry Threshold:   {stats['entry_threshold']:>10.2f}")
            print(f"  Best Score:        {stats['best_score']:>10.2f}")
            print(f"  Mean Score:        {stats['mean_score']:>10.2f}")

        print("\n" + "="*70)


def print_global50_stats(context_window_days: int = None):
    """
    Load global50.json and display comprehensive statistics.
    
    Args:
        context_window_days: Context window size in days. If None, uses Config.CONTEXT_WINDOW_DAYS.
    """
    from utils.config import Config
    
    if context_window_days is None:
        context_window_days = Config.CONTEXT_WINDOW_DAYS
    
    context_window_id = f"cw{context_window_days}"
    json_path = Path("global50") / context_window_id / "global50.json"
    
    print("\n" + "="*70)
    print("GLOBAL 50 STATISTICS")
    print("="*70)
    print(f"Context Window: {context_window_days} days ({context_window_id})")
    
    if not json_path.exists():
        print(f"\n❌ Global50 ledger not found: {json_path}")
        print("   Run with --init to initialize, or --mirror to download from cloud.")
        print("="*70)
        return
    
    # Load JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n❌ Error loading global50.json: {e}")
        print("="*70)
        return
    
    entries = data.get('entries', [])
    league_rules = data.get('league_rules', {})
    capacity = data.get('capacity', 50)
    
    if not entries:
        print(f"\n⚠ No agents in Global 50 (empty ledger)")
        print("="*70)
        return
    
    # Extract numeric fields
    gauntlet_scores = [e.get('gauntlet_score', 0) for e in entries]
    rois = [e.get('roi', 0) for e in entries]
    expectancies = [e.get('expectancy', 0) for e in entries]
    cvs = [e.get('cv', 100.0) for e in entries]
    quality_ratios = [e.get('quality_ratio', 0) for e in entries]
    win_ratios = [e.get('win_ratio', 0) for e in entries]
    total_trades = [e.get('total_trades', 0) for e in entries]
    generations = [e.get('generation', 0) for e in entries]
    
    # Count mavericks
    mavericks = [e for e in entries if e.get('is_maverick', False)]
    num_mavericks = len(mavericks)
    
    # Helper function to calculate stats
    def calc_stats(values):
        if not values:
            return {
                'min': 0, 'max': 0, 'mean': 0, 'median': 0,
                'p25': 0, 'p75': 0, 'std': 0
            }
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'p25': sorted_vals[n // 4] if n > 0 else 0,
            'p75': sorted_vals[3 * n // 4] if n > 0 else 0,
            'std': np.std(values) if n > 1 else 0
        }
    
    # Calculate statistics for each metric
    gauntlet_stats = calc_stats(gauntlet_scores)
    roi_stats = calc_stats(rois)
    expectancy_stats = calc_stats(expectancies)
    cv_stats = calc_stats(cvs)
    quality_stats = calc_stats(quality_ratios)
    win_stats = calc_stats(win_ratios)
    trades_stats = calc_stats(total_trades)
    gen_stats = calc_stats(generations)
    
    # Print overview
    print(f"\n📊 OVERVIEW")
    print("-" * 70)
    print(f"  Total Agents:     {len(entries):>4} / {capacity}")
    print(f"  Mavericks [M]:     {num_mavericks:>4} ({100*num_mavericks/len(entries):.1f}%)")
    print(f"  Regular Agents:    {len(entries) - num_mavericks:>4} ({100*(len(entries)-num_mavericks)/len(entries):.1f}%)")
    
    # Print gauntlet score statistics
    print(f"\n🎯 GAUNTLET SCORE (Efficiency-Adjusted)")
    print("-" * 70)
    print(f"  Min:               {gauntlet_stats['min']:>10.2f}")
    print(f"  25th Percentile:   {gauntlet_stats['p25']:>10.2f}")
    print(f"  Median:            {gauntlet_stats['median']:>10.2f}")
    print(f"  75th Percentile:   {gauntlet_stats['p75']:>10.2f}")
    print(f"  Max:               {gauntlet_stats['max']:>10.2f}")
    print(f"  Mean:              {gauntlet_stats['mean']:>10.2f}")
    print(f"  Std Dev:           {gauntlet_stats['std']:>10.2f}")
    
    # Print ROI statistics
    print(f"\n💰 ROI (%)")
    print("-" * 70)
    print(f"  Min:               {roi_stats['min']:>10.2f}")
    print(f"  25th Percentile:   {roi_stats['p25']:>10.2f}")
    print(f"  Median:            {roi_stats['median']:>10.2f}")
    print(f"  75th Percentile:   {roi_stats['p75']:>10.2f}")
    print(f"  Max:               {roi_stats['max']:>10.2f}")
    print(f"  Mean:              {roi_stats['mean']:>10.2f}")
    print(f"  Std Dev:           {roi_stats['std']:>10.2f}")
    
    # Print expectancy statistics
    print(f"\n📈 EXPECTANCY")
    print("-" * 70)
    print(f"  Min:               {expectancy_stats['min']:>10.2f}")
    print(f"  25th Percentile:   {expectancy_stats['p25']:>10.2f}")
    print(f"  Median:            {expectancy_stats['median']:>10.2f}")
    print(f"  75th Percentile:   {expectancy_stats['p75']:>10.2f}")
    print(f"  Max:               {expectancy_stats['max']:>10.2f}")
    print(f"  Mean:              {expectancy_stats['mean']:>10.2f}")
    print(f"  Std Dev:           {expectancy_stats['std']:>10.2f}")
    
    # Print CV (Coefficient of Variation) statistics
    print(f"\n📉 COEFFICIENT OF VARIATION (CV) - Lower is Better")
    print("-" * 70)
    print(f"  Min:               {cv_stats['min']:>10.2f}")
    print(f"  25th Percentile:   {cv_stats['p25']:>10.2f}")
    print(f"  Median:            {cv_stats['median']:>10.2f}")
    print(f"  75th Percentile:   {cv_stats['p75']:>10.2f}")
    print(f"  Max:               {cv_stats['max']:>10.2f}")
    print(f"  Mean:              {cv_stats['mean']:>10.2f}")
    print(f"  Std Dev:           {cv_stats['std']:>10.2f}")
    
    # Print quality ratio statistics
    print(f"\n⭐ QUALITY RATIO")
    print("-" * 70)
    print(f"  Min:               {quality_stats['min']:>10.3f}")
    print(f"  25th Percentile:   {quality_stats['p25']:>10.3f}")
    print(f"  Median:            {quality_stats['median']:>10.3f}")
    print(f"  75th Percentile:   {quality_stats['p75']:>10.3f}")
    print(f"  Max:               {quality_stats['max']:>10.3f}")
    print(f"  Mean:              {quality_stats['mean']:>10.3f}")
    print(f"  Std Dev:           {quality_stats['std']:>10.3f}")
    
    # Print win ratio statistics
    print(f"\n🏆 WIN RATIO")
    print("-" * 70)
    print(f"  Min:               {win_stats['min']:>10.3f}")
    print(f"  25th Percentile:   {win_stats['p25']:>10.3f}")
    print(f"  Median:            {win_stats['median']:>10.3f}")
    print(f"  75th Percentile:   {win_stats['p75']:>10.3f}")
    print(f"  Max:               {win_stats['max']:>10.3f}")
    print(f"  Mean:              {win_stats['mean']:>10.3f}")
    print(f"  Std Dev:           {win_stats['std']:>10.3f}")
    
    # Print total trades statistics
    print(f"\n📊 TOTAL TRADES")
    print("-" * 70)
    print(f"  Min:               {trades_stats['min']:>10.0f}")
    print(f"  25th Percentile:   {trades_stats['p25']:>10.0f}")
    print(f"  Median:            {trades_stats['median']:>10.0f}")
    print(f"  75th Percentile:   {trades_stats['p75']:>10.0f}")
    print(f"  Max:               {trades_stats['max']:>10.0f}")
    print(f"  Mean:              {trades_stats['mean']:>10.1f}")
    print(f"  Std Dev:           {trades_stats['std']:>10.1f}")
    
    # Print generation statistics
    print(f"\n🧬 GENERATION")
    print("-" * 70)
    print(f"  Min:               {gen_stats['min']:>10.0f}")
    print(f"  25th Percentile:   {gen_stats['p25']:>10.0f}")
    print(f"  Median:            {gen_stats['median']:>10.0f}")
    print(f"  75th Percentile:   {gen_stats['p75']:>10.0f}")
    print(f"  Max:               {gen_stats['max']:>10.0f}")
    print(f"  Mean:              {gen_stats['mean']:>10.1f}")
    print(f"  Std Dev:           {gen_stats['std']:>10.1f}")
    
    # Print top agents
    print(f"\n🏅 TOP 5 AGENTS (by Gauntlet Score)")
    print("-" * 70)
    sorted_entries = sorted(entries, key=lambda e: e.get('gauntlet_score', 0), reverse=True)
    for i, entry in enumerate(sorted_entries[:5], 1):
        maverick_marker = "[M]" if entry.get('is_maverick', False) else "   "
        print(f"  {i}. {maverick_marker} {entry.get('run_name', 'unknown')}_{entry.get('agent_id', 0)}")
        print(f"     Score: {entry.get('gauntlet_score', 0):>8.2f} | "
              f"ROI: {entry.get('roi', 0):>6.2f}% | "
              f"Exp: {entry.get('expectancy', 0):>5.2f} | "
              f"Trades: {entry.get('total_trades', 0):>4}")
    
    # Print detailed agent list
    print(f"\n{'='*70}")
    print("All Agents in Global 50")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Score':<12} {'ROI %':<10} {'Expectancy':<12} {'CV':<8} {'Trades':<8} {'Run Name':<30} {'Agent ID':<10} {'M'}")
    print(f"{'-'*110}")
    
    sorted_entries = sorted(entries, key=lambda e: e.get('gauntlet_score', 0), reverse=True)
    for i, entry in enumerate(sorted_entries, 1):
        maverick_tag = " [M]" if entry.get('is_maverick', False) else ""
        print(f"{i:<4} {entry.get('gauntlet_score', 0):<12.2f} "
              f"{entry.get('roi', 0):<10.2f} "
              f"{entry.get('expectancy', 0):<12.4f} "
              f"{entry.get('cv', 100.0):<8.3f} "
              f"{entry.get('total_trades', 0):<8} "
              f"{entry.get('run_name', 'unknown'):<30} "
              f"{entry.get('agent_id', 0):<10}{maverick_tag}")
    
    # Print league rules info if available
    if league_rules:
        print(f"\n⚙️  LEAGUE RULES")
        print("-" * 70)
        cw_days = league_rules.get('context_window_days', context_window_days)
        print(f"  Context Window:    {cw_days} days")
        entry_threshold = data.get('entry_threshold', 'N/A')
        if entry_threshold != 'N/A':
            print(f"  Entry Threshold:   {entry_threshold:.2f}")
    
    print("\n" + "="*70)


def print_all_stats(context_window_days: int = None):
    """
    Display detailed statistics for all agents in global50, archive, and long-term-archive.
    
    Args:
        context_window_days: Context window size in days. If None, uses Config.CONTEXT_WINDOW_DAYS.
    """
    from utils.config import Config
    from utils.cloud_sync import get_cloud_sync_from_env
    
    if context_window_days is None:
        context_window_days = Config.CONTEXT_WINDOW_DAYS
    
    context_window_id = f"cw{context_window_days}"
    json_path = Path("global50") / context_window_id / "global50.json"
    
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICS (All Locations)")
    print("="*70)
    print(f"Context Window: {context_window_days} days ({context_window_id})")
    
    # Initialize cloud sync for discovering archive agents
    cloud_sync = get_cloud_sync_from_env()
    cloud_base = f"{cloud_sync.project_name}/global50/{context_window_id}"
    
    # Helper function to print agent table
    def print_agent_table(title: str, agents: list, source: str = ""):
        if not agents:
            print(f"\n{title}")
            print(f"{'='*70}")
            print(f"  No agents found in {source}")
            return
        
        print(f"\n{'='*70}")
        print(f"{title} ({len(agents)} agent(s))")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Score':<12} {'ROI %':<10} {'Expectancy':<12} {'CV':<8} {'Trades':<8} {'Run Name':<30} {'Agent ID':<10} {'M'}")
        print(f"{'-'*110}")
        
        # Sort by score (descending)
        sorted_agents = sorted(agents, key=lambda a: a.get('gauntlet_score', 0), reverse=True)
        for i, agent in enumerate(sorted_agents, 1):
            maverick_tag = " [M]" if agent.get('is_maverick', False) else ""
            print(f"{i:<4} {agent.get('gauntlet_score', 0):<12.2f} "
                  f"{agent.get('roi', 0):<10.2f} "
                  f"{agent.get('expectancy', 0):<12.4f} "
                  f"{agent.get('cv', 100.0):<8.3f} "
                  f"{agent.get('total_trades', 0):<8} "
                  f"{agent.get('run_name', 'unknown'):<30} "
                  f"{agent.get('agent_id', 0):<10}{maverick_tag}")
    
    # Load Global 50 agents
    global50_agents = []
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            global50_agents = data.get('entries', [])
        except Exception as e:
            print(f"\n⚠ Error loading global50.json: {e}")
    else:
        print(f"\n⚠ Global50 ledger not found: {json_path}")
    
    print_agent_table("Global 50 Agents", global50_agents, "Global 50")
    
    # Load archive agents
    archive_agents = []
    local_archive_dir = Path("global50") / context_window_id / "archive"
    cloud_archive_prefix = f"{cloud_base}/archive/"
    
    # Check local archive
    if local_archive_dir.exists():
        for json_file in local_archive_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                archive_agents.append(data)
            except Exception as e:
                pass  # Skip invalid files
    
    # Check cloud archive
    try:
        if cloud_sync.provider == "gcs":
            blobs = list(cloud_sync.bucket.list_blobs(prefix=cloud_archive_prefix))
            for blob in blobs:
                if blob.name.endswith('.json'):
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        temp_path = tmp.name
                    try:
                        blob.download_to_filename(temp_path)
                        with open(temp_path, 'r') as f:
                            data = json.load(f)
                        # Check if not already in list (avoid duplicates)
                        key = (data.get('run_name'), data.get('agent_id'))
                        if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in archive_agents):
                            archive_agents.append(data)
                    except Exception:
                        pass
                    finally:
                        import os
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
        elif cloud_sync.provider == "s3":
            paginator = cloud_sync.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=cloud_sync.bucket_name, Prefix=cloud_archive_prefix)
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.json'):
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                temp_path = tmp.name
                            try:
                                cloud_sync.client.download_file(
                                    cloud_sync.bucket_name, obj['Key'], temp_path
                                )
                                with open(temp_path, 'r') as f:
                                    data = json.load(f)
                                key = (data.get('run_name'), data.get('agent_id'))
                                if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in archive_agents):
                                    archive_agents.append(data)
                            except Exception:
                                pass
                            finally:
                                import os
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
        elif cloud_sync.provider == "azure":
            blob_list = cloud_sync.container_client.list_blobs(name_starts_with=cloud_archive_prefix)
            for blob in blob_list:
                if blob.name.endswith('.json'):
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        temp_path = tmp.name
                    try:
                        blob_client = cloud_sync.container_client.get_blob_client(blob.name)
                        with open(temp_path, 'wb') as f:
                            f.write(blob_client.download_blob().readall())
                        with open(temp_path, 'r') as f:
                            data = json.load(f)
                        key = (data.get('run_name'), data.get('agent_id'))
                        if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in archive_agents):
                            archive_agents.append(data)
                    except Exception:
                        pass
                    finally:
                        import os
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
    except Exception as e:
        print(f"\n⚠ Error checking cloud archive: {e}")
    
    print_agent_table("Archive Agents", archive_agents, "Archive")
    
    # Load long-term-archive agents
    long_term_agents = []
    local_long_term_dir = Path("global50") / context_window_id / "long-term-archive"
    cloud_long_term_prefix = f"{cloud_base}/long-term-archive/"
    
    # Check local long-term-archive
    if local_long_term_dir.exists():
        for json_file in local_long_term_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                long_term_agents.append(data)
            except Exception:
                pass
    
    # Check cloud long-term-archive
    try:
        if cloud_sync.provider == "gcs":
            blobs = list(cloud_sync.bucket.list_blobs(prefix=cloud_long_term_prefix))
            for blob in blobs:
                if blob.name.endswith('.json'):
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        temp_path = tmp.name
                    try:
                        blob.download_to_filename(temp_path)
                        with open(temp_path, 'r') as f:
                            data = json.load(f)
                        key = (data.get('run_name'), data.get('agent_id'))
                        if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in long_term_agents):
                            long_term_agents.append(data)
                    except Exception:
                        pass
                    finally:
                        import os
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
        elif cloud_sync.provider == "s3":
            paginator = cloud_sync.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=cloud_sync.bucket_name, Prefix=cloud_long_term_prefix)
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.json'):
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                                temp_path = tmp.name
                            try:
                                cloud_sync.client.download_file(
                                    cloud_sync.bucket_name, obj['Key'], temp_path
                                )
                                with open(temp_path, 'r') as f:
                                    data = json.load(f)
                                key = (data.get('run_name'), data.get('agent_id'))
                                if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in long_term_agents):
                                    long_term_agents.append(data)
                            except Exception:
                                pass
                            finally:
                                import os
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
        elif cloud_sync.provider == "azure":
            blob_list = cloud_sync.container_client.list_blobs(name_starts_with=cloud_long_term_prefix)
            for blob in blob_list:
                if blob.name.endswith('.json'):
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        temp_path = tmp.name
                    try:
                        blob_client = cloud_sync.container_client.get_blob_client(blob.name)
                        with open(temp_path, 'wb') as f:
                            f.write(blob_client.download_blob().readall())
                        with open(temp_path, 'r') as f:
                            data = json.load(f)
                        key = (data.get('run_name'), data.get('agent_id'))
                        if not any(a.get('run_name') == key[0] and a.get('agent_id') == key[1] for a in long_term_agents):
                            long_term_agents.append(data)
                    except Exception:
                        pass
                    finally:
                        import os
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
    except Exception as e:
        print(f"\n⚠ Error checking cloud long-term-archive: {e}")
    
    print_agent_table("Long-Term Archive Agents", long_term_agents, "Long-Term Archive")
    
    # Summary
    total_agents = len(global50_agents) + len(archive_agents) + len(long_term_agents)
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Global 50:          {len(global50_agents):>4} agent(s)")
    print(f"  Archive:            {len(archive_agents):>4} agent(s)")
    print(f"  Long-Term Archive:  {len(long_term_agents):>4} agent(s)")
    print(f"  Total:              {total_agents:>4} agent(s)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agents for Global 50 promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup (initialize Global 50 structure)
  python global50.py --init

  # Check mirror status between local and GCP
  python global50.py --mirror

  # Re-evaluate all agents with efficiency-adjusted gauntlet scoring (updates metrics)
  python global50.py --eval

  # Re-evaluate agents in a specific context window (e.g., cw504)
  python global50.py --eval --cw 504

  # Interactive trim - prompts for gauntlet (efficiency-adjusted), ROI, expectancy, and total trades thresholds
  # Maverick agents [M] are identified in the output
  python global50.py --trim

  # Evaluate agents from Hall of Fame directory
  python global50.py --agent-dir checkpoints/azure-thunder-123/hall_of_fame

  # Evaluate Maverick agents (trained with aggressive reward functions)
  python global50.py --agent-dir checkpoints/maverick-run-789/hall_of_fame --maverick

  # Evaluate with custom run name
  python global50.py --agent-dir workspace/elite_agents --run-name backfill-2025

  # Evaluate specific run's champions
  python global50.py --agent-dir checkpoints/crimson-wave-456/hall_of_fame

  # Fill Global 50 from archive (Maverick status preserved from archive metadata)
  python global50.py --archive-fill

  # Find orphan agents (dry run - report only)
  python global50.py --cleanup-dry-run

  # Archive orphan agents (move from agents/ to archive/)
  python global50.py --cleanup

  # Fill Global 50 from archive (after trimming)
  python global50.py --archive-fill

  # Fill Global 50 from archive for a specific context window
  python global50.py --archive-fill --cw 504

  # Reactivate agents between archive and long-term-archive
  python global50.py --reactivate

  # Reactivate agents for a specific context window
  python global50.py --reactivate --cw 504
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
        '--maverick',
        action='store_true',
        help='Flag agents from --agent-dir as Maverick (trained with aggressive reward functions). For --archive-fill, Maverick status is read from archive metadata.'
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize Global 50 structure (first-time setup). Creates empty global50.json and validates cloud sync.'
    )

    parser.add_argument(
        '--mirror',
        action='store_true',
        help='Check mirror status between local and GCP. Downloads any missing agent files. If JSON mismatch detected, resolve conflict interactively.'
    )

    parser.add_argument(
        '--trim',
        action='store_true',
        help='Interactive trim mode: shows current thresholds and prompts for gauntlet (efficiency-adjusted), ROI, expectancy, and total trades thresholds. Maverick agents [M] are identified.'
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        help='Re-evaluate all agents in Global 50 with efficiency-adjusted gauntlet scoring. Updates all metrics including raw_gauntlet_score and efficiency_ratio.'
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Find and archive orphan agents (files in agents/ not in global50.json). Moves orphans to archive/.'
    )

    parser.add_argument(
        '--cleanup-dry-run',
        action='store_true',
        help='Like --cleanup but only reports orphans without archiving them.'
    )

    parser.add_argument(
        '--cw',
        type=int,
        metavar='DAYS',
        help='Context window size in days (e.g., --cw 504 for cw504). Overrides Config.CONTEXT_WINDOW_DAYS for this script only.'
    )

    parser.add_argument(
        '--archive-fill',
        action='store_true',
        help='Fill Global 50 from archive. Downloads archived agents, evaluates them, and promotes qualifying ones to fill empty slots.'
    )

    parser.add_argument(
        '--reactivate',
        action='store_true',
        help='Reactivate agents: move qualifying agents from long-term-archive to archive, and move non-qualifying agents from archive to long-term-archive.'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Display comprehensive statistics about the current Global 50 agents.'
    )

    parser.add_argument(
        '--stats-all',
        action='store_true',
        help='Display detailed agent lists for all agents in global50, archive, and long-term-archive.'
    )

    parser.add_argument(
        '--mark-maverick',
        type=str,
        metavar='RUN_NAME',
        help='Mark all agents with the given run_name as Mavericks in global50, archive, and long-term-archive. Propagates through mirroring.'
    )

    parser.add_argument(
        '--unmark-maverick',
        type=str,
        metavar='RUN_NAME',
        help='Unmark all agents with the given run_name as Mavericks in global50, archive, and long-term-archive.'
    )

    args = parser.parse_args()

    # Setup logging when using --agent-dir --maverick
    tee_logger = None
    if args.agent_dir and args.maverick:
        # Create evaluation_results directory
        log_dir = Path("evaluation_results")
        log_dir.mkdir(exist_ok=True)

        # Create log file
        log_file = log_dir / "log.txt"

        # Redirect stdout to both terminal and log file
        tee_logger = TeeLogger(log_file)
        sys.stdout = tee_logger
        sys.stderr = tee_logger  # Also capture error messages

        print("="*60)
        print("Global 50 Agent Evaluation (Maverick Mode)")
        print(f"Logging to: {log_file}")
        print("="*60)

    try:
        # Override context window if specified
        context_window_days = None
        if args.cw:
            from utils.config import Config
            context_window_days = args.cw
            Config.CONTEXT_WINDOW_DAYS = args.cw
            print(f"Using context window: cw{args.cw}")

        # Handle --stats mode (doesn't need full evaluator setup)
        if args.stats:
            print_global50_stats(context_window_days=context_window_days)
            return

        # Handle --stats-all mode (doesn't need full evaluator setup)
        if args.stats_all:
            print_all_stats(context_window_days=context_window_days)
            return

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

                print("\nGlobal 50 structure created and mirrored:")
                print(f"  Local:  {evaluator.global_hof.local_dir}")
                print(f"  Cloud:  gs://{evaluator.cloud_sync.bucket_name}/{evaluator.global_hof.cloud_base}/")
                print(f"  Status: MIRRORED")

                print("\n  Subdirectories:")
                print(f"    - agents/  (active Global 50 agents)")
                print(f"    - archive/ (retired agents)")

                print("\n✓ Setup complete! You can now run evaluations.")
                print("\nNext step:")
                print(f"  python global50.py --agent-dir <path>")
            else:
                print("\n✗ Initialization FAILED")
                print(f"  Cloud Provider: {evaluator.cloud_sync.provider}")
                print("\nPlease check your environment variables:")
                print("  - CLOUD_PROVIDER=gcs")
                print("  - CLOUD_BUCKET=<your-bucket>")
                print("  - GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials>")

            print("\n" + "="*70)
            return

        # Handle --mirror mode (check sync status)
        if args.mirror:
            print("\n" + "="*70)
            print("MIRROR CHECK MODE (All Context Windows)")
            print("="*70)

            in_sync = evaluator.check_mirror_status()

            print("\n" + "="*70)
            return

        # Handle --trim mode (interactive multi-threshold trim)
        if args.trim:
            print("\n" + "="*70)
            print("TRIM MODE")
            print("="*70)

            evaluator.trim_agents()

            print("\n" + "="*70)
            return

        # Handle --eval mode (re-evaluate all agents)
        if args.eval:
            print("\n" + "="*70)
            print("RE-EVALUATION MODE")
            print("="*70)

            evaluator.reevaluate_global50()

            print("\n" + "="*70)
            return

        # Handle --cleanup and --cleanup-dry-run modes
        if args.cleanup or args.cleanup_dry_run:
            print("\n" + "="*70)
            print("CLEANUP MODE")
            print("="*70)

            dry_run = args.cleanup_dry_run  # --cleanup-dry-run = dry run, --cleanup = actually archive
            evaluator.cleanup_orphan_agents(dry_run=dry_run)

            print("\n" + "="*70)
            return

        # Handle --archive-fill mode (fill Global 50 from archive)
        if args.archive_fill:
            print("\n" + "="*70)
            print("ARCHIVE FILL MODE")
            print("="*70)

            evaluator.archive_fill()

            print("\n" + "="*70)
            return

        # Handle --reactivate mode (reactivate agents between archive and long-term-archive)
        if args.reactivate:
            print("\n" + "="*70)
            print("REACTIVATE MODE")
            print("="*70)

            evaluator.reactivate_agents()

            print("\n" + "="*70)
            return

        # Handle --mark-maverick mode
        if args.mark_maverick:
            evaluator.mark_maverick_by_run_name(args.mark_maverick, is_maverick=True)
            return

        # Handle --unmark-maverick mode
        if args.unmark_maverick:
            evaluator.mark_maverick_by_run_name(args.unmark_maverick, is_maverick=False)
            return

        # Require agent_dir for evaluation mode
        if not args.agent_dir:
            print("Error: --agent-dir is required (or use --init for first-time setup)")
            parser.print_help()
            return

        # Convert to Path
        agent_dir = Path(args.agent_dir)

        # Evaluate agents
        results = evaluator.evaluate_batch(agent_dir, is_maverick=args.maverick)

        # Print summary
        if results:
            evaluator.print_summary(results)
        else:
            print("\nNo agents to evaluate. Exiting.")

        print("\nEvaluation complete!")

    finally:
        # Restore original stdout/stderr and close log file if logging was enabled
        if tee_logger:
            sys.stdout = tee_logger.terminal
            sys.stderr = tee_logger.terminal
            log_file = tee_logger.log_file.name
            tee_logger.close()
            print(f"\n✓ Complete evaluation log saved to: {log_file}")


if __name__ == "__main__":
    main()
