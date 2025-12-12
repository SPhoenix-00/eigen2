"""
Evaluate existing agents for Global 50 promotion.

This script loads agents from a specified folder, runs them through gauntlet
validation, and promotes qualifying agents to the Global Hall of Fame.

The script maintains a mirrored directory structure between local (global50/)
and GCP cloud storage. Use --mirror to check sync status across ALL context
windows (cw151, cw504, etc.) and automatically download any missing agent files.

Usage:
    python evaluate_for_global50.py --init                              # Initialize Global 50
    python evaluate_for_global50.py --mirror                            # Check sync status
    python evaluate_for_global50.py --eval                              # Re-evaluate all agents
    python evaluate_for_global50.py --trim                              # Interactive trim (prompts for thresholds)
    python evaluate_for_global50.py --agent-dir <path> [--run-name <name>]

Example:
    python evaluate_for_global50.py --eval                              # Update all metrics
    python evaluate_for_global50.py --trim                              # Interactive trim with gauntlet/ROI/expectancy
    python evaluate_for_global50.py --agent-dir checkpoints/azure-thunder-123/hall_of_fame
    python evaluate_for_global50.py --agent-dir workspace/elite_agents --run-name batch-eval-001
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
from datetime import datetime
import json

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from erl.global_hof import GlobalHallOfFame, LeagueRules, GlobalHoFEntry
from utils.config import Config
from utils.cloud_sync import get_cloud_sync_from_env
from training.erl_trainer import ERLTrainer


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
                full_end_idx = len(data_loader.data_array_full)
                self.eval_env = TradingEnvironment(
                    data_array=data_loader.data_array,
                    dates=data_loader.dates,
                    normalization_stats=normalization_stats,
                    start_idx=Config.CONTEXT_WINDOW_DAYS,
                    end_idx=full_end_idx,
                    trading_end_idx=Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS,
                    data_array_full=data_loader.data_array_full,
                    consistency_mode=True  # Global 50 ALWAYS uses consistency mode for rigorous evaluation
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

    def run_gauntlet(self, agent: DDPGAgent, agent_name: str) -> Tuple[float, dict]:
        """
        Run gauntlet validation on an agent using ERLTrainer's methods.

        This delegates to ERLTrainer's run_episode_batched and calculate_expectancy
        to ensure Global 50 scores are perfectly consistent with trainer validation.

        Args:
            agent: Agent to evaluate
            agent_name: Name for logging

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

        # Exclude top slice from gauntlet score calculation (prevents lucky outliers from inflating score)
        # But keep max_score for display purposes
        max_score = float(np.max(fitness_scores))
        scores_without_top = sorted(fitness_scores)[:-1]  # Remove the highest score

        # Use same aggregator as ERLTrainer: 0.67*mean + 0.33*min (excluding top slice)
        mean_score = float(np.mean(scores_without_top))
        min_score = float(np.min(scores_without_top))
        gauntlet_score = float((0.67 * mean_score) + (0.33 * min_score))

        # Aggregate metrics (same calculations as ERLTrainer)
        total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
        total_investment = sum([r['total_investment'] for r in slice_results])  # Legacy cumulative
        # Pooled ROI: sum of peak capitals (slices are parallel/independent scenarios)
        # This answers: "For every dollar of max drawdown capacity across all scenarios, how much profit?"
        total_peak_capital = sum([r['peak_capital_employed'] for r in slice_results])
        roi = float((total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0)

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

        detailed_metrics = {
            'gauntlet_score': gauntlet_score,
            'mean_fitness': mean_score,
            'min_fitness': min_score,
            'max_fitness': max_score,
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
            print(f"   Quality Ratio:     {metrics['quality_ratio']:>10.3f}")
            print(f"   Win Ratio:         {metrics['win_ratio']:>10.3f}")
            print(f"   Expectancy:        {metrics['expectancy']:>10.2f}")

            result.update(metrics)
            result['success'] = True

            # Check if qualifies for Global 50
            if self.global_hof.should_promote(gauntlet_score, metrics['roi'], metrics['expectancy']):
                print(f"\n   Agent QUALIFIES for Global 50!")
                print(f"   Gauntlet Threshold: {self.global_hof.entry_threshold:.2f}")
                print(f"   ROI Threshold: {self.global_hof.roi_threshold:.2f}%")
                print(f"   Expectancy Threshold: {self.global_hof.expectancy_threshold:.4f}")
                print(f"   Attempting promotion...")

                # Attempt promotion
                promoted = self.global_hof.check_and_promote(
                    agent=agent,
                    gauntlet_score=gauntlet_score,
                    generation=generation,
                    roi=metrics['roi'],
                    expectancy=metrics['expectancy'],
                    quality_ratio=metrics['quality_ratio'],
                    win_ratio=metrics['win_ratio'],
                    total_trades=metrics['total_trades']
                )

                result['promoted'] = promoted

                if promoted:
                    print(f"   SUCCESS: Agent promoted to Global 50!")
                else:
                    print(f"   WARNING: Promotion failed (concurrent update?)")
            else:
                print(f"\n   Agent does not qualify for Global 50")
                print(f"   Gauntlet: {gauntlet_score:.2f} (threshold: {self.global_hof.entry_threshold:.2f})")
                print(f"   ROI: {metrics['roi']:.2f}% (threshold: {self.global_hof.roi_threshold:.2f}%)")
                print(f"   Expectancy: {metrics['expectancy']:.4f} (threshold: {self.global_hof.expectancy_threshold:.4f})")
                print(f"   Criteria: gauntlet > threshold AND (ROI > threshold OR expectancy > threshold)")

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
        Ensures local and cloud are mirrored.
        """
        if not self.global_hof.enabled:
            return

        # Sync global50.json
        if self.global_hof.local_json_path.exists():
            print(f"   Uploading global50.json...")
            self.cloud_sync.upload_file(
                str(self.global_hof.local_json_path),
                self.global_hof.cloud_json_path,
                background=False
            )

        # Sync agents directory
        agents_synced = 0
        if self.global_hof.local_agents_dir.exists():
            for agent_file in self.global_hof.local_agents_dir.glob("*.pth"):
                cloud_path = f"{self.global_hof.cloud_base}/agents/{agent_file.name}"
                self.cloud_sync.upload_file(
                    str(agent_file),
                    cloud_path,
                    background=False
                )
                agents_synced += 1

        # Sync archive directory
        archive_synced = 0
        if self.global_hof.local_archive_dir.exists():
            for archive_file in self.global_hof.local_archive_dir.glob("*"):
                cloud_path = f"{self.global_hof.cloud_base}/archive/{archive_file.name}"
                self.cloud_sync.upload_file(
                    str(archive_file),
                    cloud_path,
                    background=False
                )
                archive_synced += 1

        print(f"   Synced {agents_synced} agent files")
        print(f"   Synced {archive_synced} archive files")
        print(f"   Mirror: gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")
        print(f"   Status: UP TO DATE")

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
        cloud_base = f"{self.cloud_sync.project_name}/global50/{context_window_id}"
        cloud_json_path = f"{cloud_base}/global50.json"

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

        # Case 3: Only cloud exists - need to download
        if not local_exists and cloud_exists:
            print(f"    ⚠ Cloud exists but local is missing - downloading...")

            # Create local directory structure
            local_dir.mkdir(parents=True, exist_ok=True)
            local_agents_dir.mkdir(parents=True, exist_ok=True)

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
                        except Exception:
                            pass

                print(f"    ✓ Downloaded {downloaded}/{len(entries)} agents")
            except Exception as e:
                print(f"    ⚠ Error downloading agents: {e}")

            return True

        # Case 4: Both exist - compare and sync
        try:
            with open(local_json_path, 'r') as f:
                local_data = json.load(f)
            with open(temp_cloud_path, 'r') as f:
                cloud_data = json.load(f)

            local_entries = len(local_data.get('entries', []))
            cloud_entries = len(cloud_data.get('entries', []))

            if local_data == cloud_data:
                print(f"    ✓ Synced ({local_entries} entries)")

                # Check for missing agent files
                entries = local_data.get('entries', [])
                missing_count = 0
                downloaded = 0

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
                        except Exception:
                            pass

                if missing_count > 0:
                    print(f"    ✓ Downloaded {downloaded}/{missing_count} missing agents")

                if temp_cloud_path and os.path.exists(temp_cloud_path):
                    os.unlink(temp_cloud_path)
                return True
            else:
                print(f"    ⚠ Mismatch (local: {local_entries}, cloud: {cloud_entries})")
                if temp_cloud_path and os.path.exists(temp_cloud_path):
                    os.unlink(temp_cloud_path)
                return False

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
        Sync local directory to cloud (local supersedes cloud).
        """
        print("   Uploading local files to cloud...")

        # Upload global50.json
        if self.global_hof.local_json_path.exists():
            self.cloud_sync.upload_file(
                str(self.global_hof.local_json_path),
                self.global_hof.cloud_json_path,
                background=False
            )
            print(f"   ✓ Uploaded global50.json")

        # Upload all agents
        agents_uploaded = 0
        if self.global_hof.local_agents_dir.exists():
            for agent_file in self.global_hof.local_agents_dir.glob("*.pth"):
                cloud_path = f"{self.global_hof.cloud_base}/agents/{agent_file.name}"
                self.cloud_sync.upload_file(
                    str(agent_file),
                    cloud_path,
                    background=False
                )
                agents_uploaded += 1

        # Upload all archive files
        archive_uploaded = 0
        if self.global_hof.local_archive_dir.exists():
            for archive_file in self.global_hof.local_archive_dir.glob("*"):
                cloud_path = f"{self.global_hof.cloud_base}/archive/{archive_file.name}"
                self.cloud_sync.upload_file(
                    str(archive_file),
                    cloud_path,
                    background=False
                )
                archive_uploaded += 1

        print(f"   ✓ Uploaded {agents_uploaded} agent files")
        print(f"   ✓ Uploaded {archive_uploaded} archive files")
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
            print(f"  python evaluate_for_global50.py --cleanup")
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

                # Upload to cloud archive
                if local_dst.exists():
                    self.cloud_sync.upload_file(str(local_dst), cloud_dst, background=False)

                    # Verify archive exists before deleting from agents/
                    if self.cloud_sync.file_exists(cloud_dst):
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
        print(f"  3. Update metrics in global50.json")
        print("  4. Sync changes to cloud")
        print(f"\nEstimated time: ~{len(self.global_hof.entries) * 2} minutes")

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
            print(f"\n[{i}/{len(self.global_hof.entries)}] {entry.run_name} (Agent {entry.agent_id})")
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
                new_score, metrics = self.run_gauntlet(agent, f"{entry.run_name}_{entry.agent_id}")

                # Create updated entry
                updated_entry = GlobalHoFEntry(
                    agent_id=entry.agent_id,
                    run_name=entry.run_name,
                    gauntlet_score=new_score,
                    generation=entry.generation,
                    roi=metrics['roi'],
                    expectancy=metrics['expectancy'],
                    quality_ratio=metrics['quality_ratio'],
                    win_ratio=metrics['win_ratio'],
                    total_trades=metrics['total_trades']
                )

                # Show results
                score_change = new_score - entry.gauntlet_score
                score_symbol = "↑" if score_change > 0 else "↓" if score_change < 0 else "="
                print(f"  New Score: {new_score:.2f} ({score_symbol} {abs(score_change):.2f})")
                print(f"  ROI: {metrics['roi']:.2f}% | Expectancy: {metrics['expectancy']:.2f}")
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

        # Show summary
        print(f"\n{'='*70}")
        print("Re-evaluation Summary")
        print(f"{'='*70}")

        print(f"\nTotal Agents:      {len(results)}")
        print(f"Successful:        {len(successful)}")
        print(f"Duplicates:        {len(behavioral_duplicates)}")
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

        # Update entries (only successful ones, keep failed ones with old scores)
        # Create a map of updated entries by (run_name, agent_id)
        update_map = {
            (r['updated_entry'].run_name, r['updated_entry'].agent_id): r['updated_entry']
            for r in successful
        }

        # Update entries list
        final_entries = []
        for entry in self.global_hof.entries:
            key = (entry.run_name, entry.agent_id)
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

        # Save and upload
        print("\nSaving updated global50.json...")
        self.global_hof._save_local_ledger()
        self.global_hof._upload_global_ledger()

        print(f"\n{'='*70}")
        print("✓ Re-evaluation Complete!")
        print(f"{'='*70}")
        print(f"  Updated:        {len(successful)} agents")
        print(f"  Failed:         {len(failed)} agents")
        print(f"  New threshold:  {self.global_hof.entry_threshold:.2f}")
        print(f"  Cloud mirror:   gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")
        print(f"{'='*70}")

    def trim_agents(self):
        """
        Interactive trim mode: shows current thresholds and prompts for
        gauntlet, ROI, and expectancy thresholds to trim agents.

        Trim criteria matches promotion criteria:
        Agent is KEPT if: gauntlet >= threshold AND (ROI >= threshold OR expectancy >= threshold)
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

        if len(self.global_hof.entries) >= self.global_hof.CAPACITY:
            print(f"\n  Current Thresholds (population full):")
            print(f"    Gauntlet:    {self.global_hof.entry_threshold:.2f} (50th rank)")
            print(f"    ROI:         {self.global_hof.roi_threshold:.2f}% (minimum)")
            print(f"    Expectancy:  {self.global_hof.expectancy_threshold:.4f} (minimum)")
        else:
            # Show current minimums for reference
            min_gauntlet = min(e.gauntlet_score for e in self.global_hof.entries)
            min_roi = min(e.roi for e in self.global_hof.entries)
            min_expectancy = min(e.expectancy for e in self.global_hof.entries)
            print(f"\n  Current Minimums (population not full, thresholds are -inf):")
            print(f"    Gauntlet:    {min_gauntlet:.2f} (current min)")
            print(f"    ROI:         {min_roi:.2f}% (current min)")
            print(f"    Expectancy:  {min_expectancy:.4f} (current min)")

        # Show distribution
        print(f"\n  Score Ranges:")
        print(f"    Gauntlet:    {min(e.gauntlet_score for e in self.global_hof.entries):.2f} to {max(e.gauntlet_score for e in self.global_hof.entries):.2f}")
        print(f"    ROI:         {min(e.roi for e in self.global_hof.entries):.2f}% to {max(e.roi for e in self.global_hof.entries):.2f}%")
        print(f"    Expectancy:  {min(e.expectancy for e in self.global_hof.entries):.4f} to {max(e.expectancy for e in self.global_hof.entries):.4f}")

        # Prompt for thresholds
        print(f"\n{'='*70}")
        print(f"Enter Trim Thresholds")
        print(f"{'='*70}")
        print(f"Agents will be KEPT if: gauntlet >= threshold AND (ROI >= threshold OR expectancy >= threshold)")
        print(f"Press Enter to skip a threshold (use -inf)")

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

        print(f"\n  Selected thresholds:")
        print(f"    Gauntlet:    {gauntlet_threshold if gauntlet_threshold != float('-inf') else 'skipped (-inf)'}")
        print(f"    ROI:         {roi_threshold if roi_threshold != float('-inf') else 'skipped (-inf)'}%")
        print(f"    Expectancy:  {expectancy_threshold if expectancy_threshold != float('-inf') else 'skipped (-inf)'}")

        # Identify agents to remove using same logic as promotion
        # Keep if: gauntlet >= threshold AND (ROI >= threshold OR expectancy >= threshold)
        agents_to_keep = []
        agents_to_remove = []

        for entry in self.global_hof.entries:
            passes_gauntlet = entry.gauntlet_score >= gauntlet_threshold
            passes_roi = entry.roi >= roi_threshold
            passes_expectancy = entry.expectancy >= expectancy_threshold

            if passes_gauntlet and (passes_roi or passes_expectancy):
                agents_to_keep.append(entry)
            else:
                agents_to_remove.append(entry)

        if len(agents_to_remove) == 0:
            print(f"\n✓ No agents would be removed with these thresholds.")
            print(f"  All {len(self.global_hof.entries)} agents meet the criteria.")
            return

        # Show what will be removed
        print(f"\n{'='*70}")
        print(f"⚠ WARNING: {len(agents_to_remove)} agents will be REMOVED from Global 50:")
        print(f"{'='*70}")
        print(f"{'Gauntlet':<10} {'ROI %':<10} {'Expect':<10} {'Run Name':<25} {'Agent':<8} {'Reason'}")
        print(f"{'-'*90}")

        for entry in sorted(agents_to_remove, key=lambda e: e.gauntlet_score):
            # Determine why agent fails
            passes_gauntlet = entry.gauntlet_score >= gauntlet_threshold
            passes_roi = entry.roi >= roi_threshold
            passes_expectancy = entry.expectancy >= expectancy_threshold

            reasons = []
            if not passes_gauntlet:
                reasons.append("gauntlet")
            if not passes_roi and not passes_expectancy:
                reasons.append("ROI+expect")
            reason_str = ", ".join(reasons)

            print(f"{entry.gauntlet_score:<10.2f} {entry.roi:<10.2f} {entry.expectancy:<10.4f} {entry.run_name:<25} {entry.agent_id:<8} {reason_str}")

        print(f"\n{len(agents_to_keep)} agents will remain in Global 50.")

        if len(agents_to_keep) > 0:
            print(f"\nRemaining score ranges:")
            print(f"  Gauntlet:    {min(e.gauntlet_score for e in agents_to_keep):.2f} to {max(e.gauntlet_score for e in agents_to_keep):.2f}")
            print(f"  ROI:         {min(e.roi for e in agents_to_keep):.2f}% to {max(e.roi for e in agents_to_keep):.2f}%")
            print(f"  Expectancy:  {min(e.expectancy for e in agents_to_keep):.4f} to {max(e.expectancy for e in agents_to_keep):.4f}")

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

            # Cloud: Upload to archive/
            cloud_archive_pth = f"{self.global_hof.cloud_base}/archive/{filename}"
            cloud_archive_json = f"{self.global_hof.cloud_base}/archive/{scoresheet_filename}"

            if local_dst.exists():
                self.cloud_sync.upload_file(str(local_dst), cloud_archive_pth, background=False)
            self.cloud_sync.upload_file(str(scoresheet_path), cloud_archive_json, background=False)

        # Update entries list
        self.global_hof.entries = agents_to_keep

        # Update threshold
        self.global_hof._update_entry_threshold()

        # Save and upload updated ledger
        print("\nUpdating global50.json...")
        self.global_hof._save_local_ledger()
        self.global_hof._upload_global_ledger()

        print(f"\n{'='*70}")
        print("✓ Trim Complete!")
        print(f"{'='*70}")
        print(f"  Removed:        {len(agents_to_remove)} agents")
        print(f"  Remaining:      {len(agents_to_keep)} agents")
        print(f"  New thresholds:")
        print(f"    Gauntlet:     {self.global_hof.entry_threshold:.2f}")
        print(f"    ROI:          {self.global_hof.roi_threshold:.2f}%")
        print(f"    Expectancy:   {self.global_hof.expectancy_threshold:.4f}")
        print(f"  Archived to:    {self.global_hof.local_archive_dir}")
        print(f"  Cloud mirror:   gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_base}/")
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

  # Check mirror status between local and GCP
  python evaluate_for_global50.py --mirror

  # Re-evaluate all agents with current logic (updates metrics)
  python evaluate_for_global50.py --eval

  # Re-evaluate agents in a specific context window (e.g., cw504)
  python evaluate_for_global50.py --eval --cw 504

  # Interactive trim - prompts for gauntlet, ROI, and expectancy thresholds
  python evaluate_for_global50.py --trim

  # Evaluate agents from Hall of Fame directory
  python evaluate_for_global50.py --agent-dir checkpoints/azure-thunder-123/hall_of_fame

  # Evaluate with custom run name
  python evaluate_for_global50.py --agent-dir workspace/elite_agents --run-name backfill-2025

  # Evaluate specific run's champions
  python evaluate_for_global50.py --agent-dir checkpoints/crimson-wave-456/hall_of_fame

  # Find orphan agents (dry run - report only)
  python evaluate_for_global50.py --cleanup-dry-run

  # Archive orphan agents (move from agents/ to archive/)
  python evaluate_for_global50.py --cleanup
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

    parser.add_argument(
        '--mirror',
        action='store_true',
        help='Check mirror status between local and GCP. Downloads any missing agent files. If JSON mismatch detected, resolve conflict interactively.'
    )

    parser.add_argument(
        '--trim',
        action='store_true',
        help='Interactive trim mode: shows current thresholds and prompts for gauntlet, ROI, and expectancy thresholds'
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        help='Re-evaluate all agents in Global 50 with current evaluation logic. Updates all metrics.'
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

    args = parser.parse_args()

    # Override context window if specified
    if args.cw:
        from utils.config import Config
        Config.CONTEXT_WINDOW_DAYS = args.cw
        print(f"Using context window: cw{args.cw}")

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
