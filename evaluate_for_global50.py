"""
Evaluate existing agents for Global 50 promotion.

This script loads agents from a specified folder, runs them through gauntlet
validation, and promotes qualifying agents to the Global Hall of Fame.

The script maintains a mirrored directory structure between local (global50/)
and GCP cloud storage. Use --mirror to check sync status and download any missing
agent files. If JSON mismatch is detected, conflicts can be resolved interactively.

Usage:
    python evaluate_for_global50.py --init                              # Initialize Global 50
    python evaluate_for_global50.py --mirror                            # Check sync status
    python evaluate_for_global50.py --eval                              # Re-evaluate all agents
    python evaluate_for_global50.py --trim <threshold>                  # Remove agents below threshold
    python evaluate_for_global50.py --agent-dir <path> [--run-name <name>]

Example:
    python evaluate_for_global50.py --eval                              # Update all metrics
    python evaluate_for_global50.py --trim 0                            # Remove negative scores
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
            print(f"   Cloud mirror: gs://{self.cloud_sync.bucket_name}/{self.cloud_sync.project_name}/global50/")
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
                'total_investment': episode_info.get('total_investment', 0.0)
            })

            # Collect closed trades for expectancy calculation
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        # Extract fitness scores
        fitness_scores = [result['fitness'] for result in slice_results]

        # Use same aggregator as ERLTrainer: 0.75*mean + 0.25*min
        mean_score = float(np.mean(fitness_scores))
        min_score = float(np.min(fitness_scores))
        max_score = float(np.max(fitness_scores))
        gauntlet_score = float((0.75 * mean_score) + (0.25 * min_score))

        # Aggregate metrics (same calculations as ERLTrainer)
        total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
        total_investment = sum([r['total_investment'] for r in slice_results])
        roi = float((total_raw_pnl / total_investment * 100) if total_investment > 0 else 0.0)

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

        # Final sync to ensure everything is mirrored to GCP
        if self.global_hof.enabled:
            print(f"\n{'='*70}")
            print("Syncing global50/ to GCP...")
            print(f"{'='*70}")
            self._sync_to_cloud()

        return results

    def _sync_to_cloud(self):
        """
        Sync the entire global50/ directory to GCP.
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
        print(f"   Mirror: gs://{self.cloud_sync.bucket_name}/{self.cloud_sync.project_name}/global50/")
        print(f"   Status: UP TO DATE")

    def check_mirror_status(self) -> bool:
        """
        Check synchronization status between local and GCP.
        Verifies JSON metadata and downloads any missing agent files.
        Returns True if in sync, False if mismatch detected.
        """
        print(f"\n{'='*70}")
        print("Checking Mirror Status")
        print(f"{'='*70}")

        if not self.global_hof.enabled:
            print("⚠ Global 50 not enabled (local mode or disabled)")
            return True

        # Check if local file exists
        local_exists = self.global_hof.local_json_path.exists()

        # Check if cloud file exists by trying to download it to a temp location
        import tempfile
        import os

        cloud_exists = False
        temp_cloud_path = None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            temp_cloud_path = tmp.name

        try:
            cloud_exists = self.cloud_sync.download_file(
                self.global_hof.cloud_json_path,
                temp_cloud_path
            )
        except Exception:
            cloud_exists = False

        # Case 1: Neither exists
        if not local_exists and not cloud_exists:
            print("✓ No global50.json found locally or in cloud")
            print("  This appears to be a fresh setup")
            if temp_cloud_path and os.path.exists(temp_cloud_path):
                os.unlink(temp_cloud_path)
            return True

        # Case 2: Only local exists
        if local_exists and not cloud_exists:
            print("⚠ Mismatch detected:")
            print(f"  Local:  EXISTS at {self.global_hof.local_json_path}")
            print(f"  Cloud:  NOT FOUND")
            if temp_cloud_path and os.path.exists(temp_cloud_path):
                os.unlink(temp_cloud_path)
            return False

        # Case 3: Only cloud exists
        if not local_exists and cloud_exists:
            print("⚠ Mismatch detected:")
            print(f"  Local:  NOT FOUND")
            print(f"  Cloud:  EXISTS at gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_json_path}")
            if temp_cloud_path and os.path.exists(temp_cloud_path):
                os.unlink(temp_cloud_path)
            return False

        # Case 4: Both exist - compare content and timestamps
        try:
            # Load local file
            with open(self.global_hof.local_json_path, 'r') as f:
                local_data = json.load(f)

            # Load cloud file (from temp download)
            with open(temp_cloud_path, 'r') as f:
                cloud_data = json.load(f)

            # Get modification times
            local_mtime = os.path.getmtime(self.global_hof.local_json_path)
            cloud_mtime = os.path.getmtime(temp_cloud_path)

            local_time_str = datetime.fromtimestamp(local_mtime).strftime('%Y-%m-%d %H:%M:%S')
            cloud_time_str = datetime.fromtimestamp(cloud_mtime).strftime('%Y-%m-%d %H:%M:%S')

            # Compare content
            if local_data == cloud_data:
                print("✓ Local and cloud JSON are synchronized")
                print(f"  Local:  {len(local_data.get('entries', []))} entries, modified {local_time_str}")
                print(f"  Cloud:  {len(cloud_data.get('entries', []))} entries, modified {cloud_time_str}")

                # Check if agent files exist locally and verify cloud
                print("\nChecking agent files...")
                missing_local_agents = []
                present_local_agents = []
                entries = local_data.get('entries', [])

                for entry in entries:
                    # Construct filename from entry
                    run_name = entry.get('run_name', 'unknown')
                    agent_id = entry.get('agent_id', 0)
                    filename = f"{run_name}_{agent_id}.pth"

                    local_agent_path = self.global_hof.local_agents_dir / filename
                    cloud_path = f"{self.global_hof.cloud_base}/agents/{filename}"

                    if not local_agent_path.exists():
                        missing_local_agents.append((filename, cloud_path, str(local_agent_path)))
                    else:
                        present_local_agents.append((filename, str(local_agent_path), cloud_path))

                # Download missing local agents
                if missing_local_agents:
                    print(f"⚠ Found {len(missing_local_agents)} missing local agent files")
                    print(f"  Downloading from cloud...")

                    downloaded = 0
                    for filename, cloud_path, local_path in missing_local_agents:
                        try:
                            success = self.cloud_sync.download_file(cloud_path, local_path)
                            if success:
                                print(f"  ✓ Downloaded: {filename}")
                                downloaded += 1
                            else:
                                print(f"  ✗ Failed to download: {filename}")
                        except Exception as e:
                            print(f"  ✗ Error downloading {filename}: {e}")

                    print(f"  Downloaded {downloaded}/{len(missing_local_agents)} agent files")

                if not missing_local_agents:
                    print("✓ All agent files present locally")

                # Verify cloud has all agents (by checking if local→cloud download would fail)
                # This checks bidirectional sync without unnecessary uploads
                print(f"\n  Verifying cloud has all {len(present_local_agents)} agent files...")
                missing_in_cloud = []

                for filename, local_path, cloud_path in present_local_agents:
                    # Test if cloud has the file by attempting to download to a temp location
                    with tempfile.NamedTemporaryFile(delete=True) as tmp:
                        try:
                            exists = self.cloud_sync.download_file(cloud_path, tmp.name)
                            if not exists:
                                missing_in_cloud.append((filename, local_path, cloud_path))
                        except Exception:
                            missing_in_cloud.append((filename, local_path, cloud_path))

                # Upload missing agents to cloud
                if missing_in_cloud:
                    print(f"⚠ Found {len(missing_in_cloud)} agent files missing in cloud")
                    print(f"  Uploading to cloud...")

                    uploaded = 0
                    for filename, local_path, cloud_path in missing_in_cloud:
                        try:
                            self.cloud_sync.upload_file(
                                local_path,
                                cloud_path,
                                background=False
                            )
                            print(f"  ✓ Uploaded: {filename}")
                            uploaded += 1
                        except Exception as e:
                            print(f"  ✗ Error uploading {filename}: {e}")

                    print(f"  Uploaded {uploaded}/{len(missing_in_cloud)} agent files to cloud")
                else:
                    print("✓ All agent files present in cloud")

                print("\n✓ Local and cloud are fully synchronized")

                if temp_cloud_path and os.path.exists(temp_cloud_path):
                    os.unlink(temp_cloud_path)
                return True
            else:
                print("⚠ Mismatch detected:")
                print(f"\n  Local:  {self.global_hof.local_json_path}")
                print(f"          {len(local_data.get('entries', []))} entries")
                print(f"          Modified: {local_time_str}")

                print(f"\n  Cloud:  gs://{self.cloud_sync.bucket_name}/{self.global_hof.cloud_json_path}")
                print(f"          {len(cloud_data.get('entries', []))} entries")
                print(f"          Modified: {cloud_time_str}")

                # Store temp file path for potential sync
                self._temp_cloud_file = temp_cloud_path
                return False

        except Exception as e:
            print(f"⚠ Error comparing files: {e}")
            if temp_cloud_path and os.path.exists(temp_cloud_path):
                os.unlink(temp_cloud_path)
            return False

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
        print(f"   Mirror: gs://{self.cloud_sync.bucket_name}/{self.cloud_sync.project_name}/global50/")

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
                    'success': True
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

        # Show summary
        print(f"\n{'='*70}")
        print("Re-evaluation Summary")
        print(f"{'='*70}")

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"\nTotal Agents:   {len(results)}")
        print(f"Successful:     {len(successful)}")
        print(f"Failed:         {len(failed)}")

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
        print(f"  Cloud mirror:   gs://{self.cloud_sync.bucket_name}/{self.cloud_sync.project_name}/global50/")
        print(f"{'='*70}")

    def trim_agents(self, threshold: float):
        """
        Remove agents below a specified score threshold from Global 50.

        Args:
            threshold: Minimum gauntlet score to keep (agents below this will be removed)
        """
        print(f"\n{'='*70}")
        print(f"Trimming Global 50 - Threshold: {threshold:.2f}")
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

        if len(self.global_hof.entries) == 0:
            print("✓ Global 50 is empty. Nothing to trim.")
            return

        # Identify agents to remove
        agents_to_remove = [
            entry for entry in self.global_hof.entries
            if entry.gauntlet_score < threshold
        ]
        agents_to_keep = [
            entry for entry in self.global_hof.entries
            if entry.gauntlet_score >= threshold
        ]

        if len(agents_to_remove) == 0:
            print(f"\n✓ No agents below threshold {threshold:.2f}")
            print(f"  All {len(self.global_hof.entries)} agents meet the minimum score requirement.")
            return

        # Show what will be removed
        print(f"\n⚠ WARNING: {len(agents_to_remove)} agents will be REMOVED from Global 50:")
        print(f"{'='*70}")
        print(f"{'Score':<10} {'Run Name':<30} {'Agent ID':<10}")
        print(f"{'-'*70}")

        for entry in sorted(agents_to_remove, key=lambda e: e.gauntlet_score):
            print(f"{entry.gauntlet_score:<10.2f} {entry.run_name:<30} {entry.agent_id:<10}")

        print(f"\n{len(agents_to_keep)} agents will remain in Global 50.")

        if len(agents_to_keep) > 0:
            print(f"Score range after trim: {min(e.gauntlet_score for e in agents_to_keep):.2f} to {max(e.gauntlet_score for e in agents_to_keep):.2f}")

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
        print(f"  New threshold:  {self.global_hof.entry_threshold:.2f}")
        print(f"  Archived to:    {self.global_hof.local_archive_dir}")
        print(f"  Cloud mirror:   gs://{self.cloud_sync.bucket_name}/{self.cloud_sync.project_name}/global50/")
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

  # Remove agents with scores below 0
  python evaluate_for_global50.py --trim 0

  # Remove agents with scores below 5.0
  python evaluate_for_global50.py --trim 5.0

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

    parser.add_argument(
        '--mirror',
        action='store_true',
        help='Check mirror status between local and GCP. Downloads any missing agent files. If JSON mismatch detected, resolve conflict interactively.'
    )

    parser.add_argument(
        '--trim',
        type=float,
        metavar='THRESHOLD',
        help='Remove agents below specified score threshold (e.g., --trim 0 removes all agents with negative scores)'
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        help='Re-evaluate all agents in Global 50 with current evaluation logic. Updates all metrics.'
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

            print("\nGlobal 50 structure created and mirrored:")
            print(f"  Local:  {evaluator.global_hof.local_dir}")
            print(f"  Cloud:  gs://{evaluator.cloud_sync.bucket_name}/{evaluator.cloud_sync.project_name}/global50/")
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
        print("MIRROR CHECK MODE")
        print("="*70)

        in_sync = evaluator.check_mirror_status()

        if not in_sync:
            evaluator.resolve_mirror_conflict()

            # Verify sync after resolution
            print("\n" + "="*70)
            print("Verifying Sync Status")
            print("="*70)
            evaluator.check_mirror_status()

        print("\n" + "="*70)
        return

    # Handle --trim mode (remove agents below threshold)
    if args.trim is not None:
        print("\n" + "="*70)
        print("TRIM MODE")
        print("="*70)

        evaluator.trim_agents(threshold=args.trim)

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
