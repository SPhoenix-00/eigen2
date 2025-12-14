"""
Fix Global 50 - Rebuild from cloud agents.

This script downloads all agents from cloud storage (e.g., 77 agents in cw151/agents/),
evaluates each one 3 times, uses the average of the lowest 2 scores for selection,
selects the top 50, and rebuilds the global50.json.

After selection, it runs one final evaluation to get "pure" scores (not averaged).

Usage:
    python fix_global50.py                    # Fix cw151 (default)
    python fix_global50.py --context-window 504   # Fix cw504
    python fix_global50.py --dry-run          # Show what would happen without making changes
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict
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


class Global50Fixer:
    """Downloads and re-evaluates all agents from cloud to rebuild Global 50."""

    def __init__(self, context_window_days: int = 151):
        """
        Initialize fixer.

        Args:
            context_window_days: Context window to fix (e.g., 151, 504)
        """
        self.context_window_days = context_window_days
        self.context_window_id = f"cw{context_window_days}"

        print("\n" + "="*70)
        print("Global 50 Fixer")
        print("="*70)
        print(f"Context Window: {context_window_days} days ({self.context_window_id})")

        # Initialize data loader
        print("\n1. Loading market data...")
        self.data_loader = StockDataLoader()
        data_array, stats = self.data_loader.load_and_prepare()
        self.normalization_stats = stats
        print(f"   Loaded {len(self.data_loader.data_array_full)} days of data")

        # Get validation indices
        self.val_start_idx = self.data_loader.val_start_idx
        self.val_end_idx = self.data_loader.val_end_idx
        print(f"   Validation range: {self.val_start_idx} - {self.val_end_idx}")

        # Initialize cloud sync
        print("\n2. Initializing cloud storage...")
        self.cloud_sync = get_cloud_sync_from_env()
        print(f"   Provider: {self.cloud_sync.provider}")
        print(f"   Bucket: {self.cloud_sync.bucket_name}")

        # Paths
        self.cloud_base = f"{self.cloud_sync.project_name}/global50/{self.context_window_id}"
        self.cloud_agents_prefix = f"{self.cloud_base}/agents/"
        self.cloud_json_path = f"{self.cloud_base}/global50.json"
        self.cloud_archive_prefix = f"{self.cloud_base}/archive/"
        self.cloud_exclude_prefix = f"{self.cloud_base}/exclude/"

        # Minimum ROI threshold - agents below this are excluded (not real agents)
        self.MIN_ROI_THRESHOLD = 0.01

        # Local temp directory for downloads
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fix_global50_"))
        self.temp_agents_dir = self.temp_dir / "agents"
        self.temp_agents_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Temp directory: {self.temp_dir}")

        # Local global50 directory (will be updated at the end)
        self.local_dir = Path("global50") / self.context_window_id
        self.local_agents_dir = self.local_dir / "agents"
        self.local_archive_dir = self.local_dir / "archive"
        self.local_exclude_dir = self.local_dir / "exclude"
        self.local_json_path = self.local_dir / "global50.json"

        # Initialize gauntlet helper
        print("\n3. Setting up gauntlet validation...")
        self._setup_gauntlet_helper()

        print("\n" + "="*70)

    def _setup_gauntlet_helper(self):
        """Create a minimal helper object to access ERLTrainer's gauntlet methods."""
        class GauntletHelper:
            def __init__(self, data_loader, val_start_idx, val_end_idx, normalization_stats):
                self.data_loader = data_loader
                self.val_start_idx = val_start_idx
                self.val_end_idx = val_end_idx
                self.normalization_stats = normalization_stats
                self.train_start_idx = Config.CONTEXT_WINDOW_DAYS
                self.train_end_idx = val_start_idx

                full_end_idx = len(data_loader.data_array_full)
                self.eval_env = TradingEnvironment(
                    data_array=data_loader.data_array,
                    dates=data_loader.dates,
                    normalization_stats=normalization_stats,
                    start_idx=Config.CONTEXT_WINDOW_DAYS,
                    end_idx=full_end_idx,
                    trading_end_idx=Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS,
                    data_array_full=data_loader.data_array_full,
                    consistency_mode=True
                )
                self.replay_buffer = None

        self.gauntlet_helper = GauntletHelper(
            self.data_loader,
            self.val_start_idx,
            self.val_end_idx,
            self.normalization_stats
        )

        self.gauntlet_helper.generate_gauntlet_slices = ERLTrainer.generate_gauntlet_slices.__get__(
            self.gauntlet_helper, GauntletHelper
        )
        self.gauntlet_helper.run_episode_batched = ERLTrainer.run_episode_batched.__get__(
            self.gauntlet_helper, GauntletHelper
        )
        self.gauntlet_helper.calculate_expectancy = ERLTrainer.calculate_expectancy.__get__(
            self.gauntlet_helper, GauntletHelper
        )

    def get_agent_roi_from_scoresheet(self, filename: str, from_archive: bool = True) -> float:
        """
        Get ROI from agent's JSON scoresheet.
        Returns the ROI value or None if scoresheet doesn't exist or can't be parsed.
        """
        json_filename = filename.replace('.pth', '.json')
        if from_archive:
            cloud_path = f"{self.cloud_archive_prefix}{json_filename}"
        else:
            # For agents/, there's no individual scoresheet, check global50.json instead
            return None

        try:
            if self.cloud_sync.provider == "gcs":
                blob = self.cloud_sync.bucket.blob(cloud_path)
                if blob.exists():
                    content = blob.download_as_text()
                    data = json.loads(content)
                    return data.get('roi', None)
        except Exception:
            pass
        return None

    def get_roi_from_global50_json(self) -> Dict[str, float]:
        """
        Download global50.json and return a dict mapping filename to ROI.
        """
        roi_map = {}
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                temp_path = tmp.name

            if self.cloud_sync.download_file(self.cloud_json_path, temp_path):
                with open(temp_path, 'r') as f:
                    data = json.load(f)
                for entry in data.get('entries', []):
                    run_name = entry.get('run_name', '')
                    agent_id = entry.get('agent_id', 0)
                    roi = entry.get('roi', 0.0)
                    filename = f"{run_name}_{agent_id}.pth"
                    roi_map[filename] = roi

            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"   ⚠ Could not load global50.json for ROI lookup: {e}")
        return roi_map

    def list_cloud_agents(self, include_archive: bool = True) -> List[str]:
        """List all .pth files in cloud agents/ directory and optionally archive/."""
        agent_files = []
        archive_files = []

        print(f"\nListing agents in: gs://{self.cloud_sync.bucket_name}/{self.cloud_agents_prefix}")

        try:
            if self.cloud_sync.provider == "gcs":
                blobs = self.cloud_sync.bucket.list_blobs(prefix=self.cloud_agents_prefix)
                for blob in blobs:
                    if blob.name.endswith('.pth'):
                        filename = blob.name.split('/')[-1]
                        agent_files.append(filename)
            else:
                print(f"   ⚠ Provider {self.cloud_sync.provider} not fully supported")
                return []
        except Exception as e:
            print(f"   ✗ Error listing agents: {e}")
            return []

        print(f"   Found {len(agent_files)} agent files in agents/")

        # Also check archive
        if include_archive:
            print(f"\nListing archive in: gs://{self.cloud_sync.bucket_name}/{self.cloud_archive_prefix}")
            try:
                if self.cloud_sync.provider == "gcs":
                    blobs = self.cloud_sync.bucket.list_blobs(prefix=self.cloud_archive_prefix)
                    for blob in blobs:
                        if blob.name.endswith('.pth'):
                            filename = blob.name.split('/')[-1]
                            # Only add if not already in agents
                            if filename not in agent_files:
                                archive_files.append(filename)
            except Exception as e:
                print(f"   ✗ Error listing archive: {e}")

            print(f"   Found {len(archive_files)} additional agent files in archive/")

        all_files = agent_files + archive_files
        print(f"\n   Total: {len(all_files)} unique agent files")
        return sorted(all_files), agent_files, archive_files

    def download_agent(self, filename: str, from_archive: bool = False) -> Path:
        """Download a single agent from cloud (agents/ or archive/)."""
        if from_archive:
            cloud_path = f"{self.cloud_archive_prefix}{filename}"
        else:
            cloud_path = f"{self.cloud_agents_prefix}{filename}"
        local_path = self.temp_agents_dir / filename

        success = self.cloud_sync.download_file(cloud_path, str(local_path))
        if success:
            return local_path
        else:
            return None

    def run_gauntlet(self, agent: DDPGAgent, agent_name: str) -> Tuple[float, dict]:
        """Run gauntlet validation on an agent."""
        gauntlet_slices = self.gauntlet_helper.generate_gauntlet_slices()

        slice_results = []
        all_closed_trades = []

        agent.actor.eval()
        agent.critic.eval()
        self.gauntlet_helper.eval_env.set_gauntlet_mode(True)

        for i, (start_idx, end_idx, _) in enumerate(gauntlet_slices):
            fitness, episode_info = self.gauntlet_helper.run_episode_batched(
                agent=agent,
                env=self.gauntlet_helper.eval_env,
                start_idx=start_idx,
                end_idx=end_idx,
                training=False,
                batch_size=16
            )

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

            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        fitness_scores = [result['fitness'] for result in slice_results]
        max_score = float(np.max(fitness_scores))
        scores_without_top = sorted(fitness_scores)[:-1]

        mean_score = float(np.mean(scores_without_top))
        min_score = float(np.min(scores_without_top))
        gauntlet_score = float((0.67 * mean_score) + (0.33 * min_score))

        # Penalty for agents that consistently refuse to trade
        # Score between -8.99 and -10 indicates no trades across all slices
        if -10.0 <= gauntlet_score <= -8.99:
            gauntlet_score = -2000.0

        total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
        total_peak_capital = sum([r['peak_capital_employed'] for r in slice_results])
        roi = float((total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0)

        total_wins = sum([r['num_wins'] for r in slice_results])
        total_losses = sum([r['num_losses'] for r in slice_results])
        total_trades = int(total_wins + total_losses)

        expectancy = float(self.gauntlet_helper.calculate_expectancy(all_closed_trades))

        quality_threshold = Config.ROI_QUALITY_THRESHOLD
        if all_closed_trades:
            quality_count = sum(1 for t in all_closed_trades if t.get('gain_pct', 0) >= quality_threshold)
        else:
            quality_count = 0

        quality_ratio = float(quality_count / total_trades) if total_trades > 0 else 0.0
        win_ratio = float(total_wins / total_trades) if total_trades > 0 else 0.0

        self.gauntlet_helper.eval_env.set_gauntlet_mode(False)

        return gauntlet_score, {
            'gauntlet_score': gauntlet_score,
            'mean_fitness': mean_score,
            'min_fitness': min_score,
            'max_fitness': max_score,
            'roi': roi,
            'total_trades': total_trades,
            'quality_count': quality_count,
            'quality_ratio': quality_ratio,
            'win_ratio': win_ratio,
            'expectancy': expectancy,
            'fitness_all_slices': [float(score) for score in fitness_scores]
        }

    def evaluate_agent_multiple_times(self, agent_path: Path, num_evals: int = 3) -> Dict:
        """
        Evaluate an agent multiple times and return stats.

        Returns dict with:
            - scores: list of all scores
            - avg_lowest_2: average of lowest 2 scores (for selection)
            - metrics_list: list of all metrics from each eval
        """
        filename = agent_path.name
        # Parse run_name and agent_id from filename (format: run_name_agentid.pth)
        stem = agent_path.stem
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            run_name = parts[0]
            try:
                agent_id = int(parts[1])
            except ValueError:
                run_name = stem
                agent_id = 0
        else:
            run_name = stem
            agent_id = 0

        # Load agent
        try:
            agent = DDPGAgent(agent_id=agent_id)
            agent.load(str(agent_path))
        except Exception as e:
            return {
                'filename': filename,
                'run_name': run_name,
                'agent_id': agent_id,
                'success': False,
                'error': str(e)
            }

        # Run multiple evaluations
        scores = []
        metrics_list = []

        for i in range(num_evals):
            try:
                score, metrics = self.run_gauntlet(agent, stem)
                scores.append(score)
                metrics_list.append(metrics)
            except Exception as e:
                return {
                    'filename': filename,
                    'run_name': run_name,
                    'agent_id': agent_id,
                    'success': False,
                    'error': f"Eval {i+1} failed: {e}"
                }

        # Calculate average of lowest 2 scores
        sorted_scores = sorted(scores)
        avg_lowest_2 = float(np.mean(sorted_scores[:2]))

        # Use the metrics from the median score eval for other metrics
        median_idx = scores.index(sorted(scores)[len(scores)//2])
        median_metrics = metrics_list[median_idx]

        return {
            'filename': filename,
            'run_name': run_name,
            'agent_id': agent_id,
            'success': True,
            'scores': scores,
            'avg_lowest_2': avg_lowest_2,
            'metrics_list': metrics_list,
            'median_metrics': median_metrics
        }

    def fix(self, dry_run: bool = False):
        """
        Main fix routine:
        1. Discover agents and filter by ROI (from JSON scoresheets)
        2. Download valid agents
        3. Evaluate each 3 times (exclude any with ROI < threshold)
        4. Select top 50 by average of lowest 2 scores
        5. Ask for confirmation
        6. Archive non-selected agents
        7. Run final eval for pure scores
        8. Update global50.json
        """
        print("\n" + "="*70)
        print("PHASE 1: Discover agents and filter by ROI")
        print("="*70)

        # List all agents in cloud (both agents/ and archive/)
        all_files, agents_files, archive_files = self.list_cloud_agents(include_archive=True)
        if not all_files:
            print("✗ No agents found in cloud. Nothing to fix.")
            return

        # Create sets for quick lookup
        agents_set = set(agents_files)
        archive_set = set(archive_files)

        # Get ROI data from global50.json (for agents/) and scoresheets (for archive/)
        print(f"\nLoading ROI data to filter out low-quality agents (ROI < {self.MIN_ROI_THRESHOLD}%)...")
        roi_map = self.get_roi_from_global50_json()

        # For archive files, get ROI from their JSON scoresheets
        for filename in archive_files:
            if filename not in roi_map:
                roi = self.get_agent_roi_from_scoresheet(filename, from_archive=True)
                if roi is not None:
                    roi_map[filename] = roi

        # Filter out agents with ROI below threshold
        to_exclude = []
        valid_files = []
        for filename in all_files:
            roi = roi_map.get(filename)
            if roi is not None and roi < self.MIN_ROI_THRESHOLD:
                to_exclude.append((filename, roi, filename in archive_set))
            else:
                valid_files.append(filename)

        print(f"\n   Total agents found: {len(all_files)}")
        print(f"   Valid agents (ROI >= {self.MIN_ROI_THRESHOLD}% or unknown): {len(valid_files)}")
        print(f"   To exclude (ROI < {self.MIN_ROI_THRESHOLD}%): {len(to_exclude)}")

        # Show agents to exclude
        if to_exclude:
            print(f"\n   Agents to be moved to exclude/:")
            for filename, roi, from_archive in to_exclude:
                location = "archive" if from_archive else "agents"
                print(f"     - {filename} (ROI: {roi:.4f}%, from {location})")

        # Move excluded agents to exclude/ folder
        if to_exclude and not dry_run:
            print(f"\n   Moving {len(to_exclude)} agents to exclude/...")
            self.local_exclude_dir.mkdir(parents=True, exist_ok=True)

            for filename, roi, from_archive in to_exclude:
                if from_archive:
                    cloud_src = f"{self.cloud_archive_prefix}{filename}"
                else:
                    cloud_src = f"{self.cloud_agents_prefix}{filename}"
                cloud_dst = f"{self.cloud_exclude_prefix}{filename}"

                try:
                    # Download to local exclude
                    local_exclude_path = self.local_exclude_dir / filename
                    self.cloud_sync.download_file(cloud_src, str(local_exclude_path))

                    # Upload to cloud exclude
                    self.cloud_sync.upload_file_verified(str(local_exclude_path), cloud_dst)

                    # Delete from source
                    if self.cloud_sync.file_exists(cloud_dst):
                        self.cloud_sync.delete_file(cloud_src)
                        # Also delete JSON scoresheet if from archive
                        if from_archive:
                            json_src = cloud_src.replace('.pth', '.json')
                            json_dst = cloud_dst.replace('.pth', '.json')
                            if self.cloud_sync.file_exists(json_src):
                                # Download scoresheet
                                local_json = self.local_exclude_dir / filename.replace('.pth', '.json')
                                self.cloud_sync.download_file(json_src, str(local_json))
                                self.cloud_sync.upload_file_verified(str(local_json), json_dst)
                                self.cloud_sync.delete_file(json_src)
                        print(f"     ✓ {filename}")
                    else:
                        print(f"     ⚠ {filename} - upload not verified")
                except Exception as e:
                    print(f"     ✗ {filename} - {e}")

        print("\n" + "="*70)
        print("PHASE 2: Download valid agents")
        print("="*70)

        print(f"\nDownloading {len(valid_files)} agents...")
        downloaded_paths = []
        for i, filename in enumerate(valid_files, 1):
            from_archive = filename in archive_set
            location = "archive" if from_archive else "agents"
            print(f"  [{i}/{len(valid_files)}] {filename} ({location})...", end=" ")
            path = self.download_agent(filename, from_archive=from_archive)
            if path:
                downloaded_paths.append(path)
                print("✓")
            else:
                print("✗ FAILED")

        print(f"\n✓ Downloaded {len(downloaded_paths)}/{len(valid_files)} agents")

        if len(downloaded_paths) == 0:
            print("✗ No agents downloaded. Cannot proceed.")
            return

        print("\n" + "="*70)
        print("PHASE 3: Evaluate each agent 3 times")
        print("="*70)

        results = []
        excluded_after_eval = []  # Agents found to have ROI < threshold after evaluation

        for i, agent_path in enumerate(downloaded_paths, 1):
            print(f"\n[{i}/{len(downloaded_paths)}] Evaluating: {agent_path.name}")
            result = self.evaluate_agent_multiple_times(agent_path, num_evals=3)

            if result['success']:
                scores_str = ", ".join([f"{s:.2f}" for s in result['scores']])
                print(f"   Scores: [{scores_str}]")
                print(f"   Avg of lowest 2: {result['avg_lowest_2']:.2f}")
                print(f"   ROI: {result['median_metrics']['roi']:.2f}%")
                print(f"   Expectancy: {result['median_metrics']['expectancy']:.4f}")

                # Check if ROI is below threshold - if so, mark for exclusion
                if result['median_metrics']['roi'] < self.MIN_ROI_THRESHOLD:
                    print(f"   ⚠ ROI below {self.MIN_ROI_THRESHOLD}% - will be excluded")
                    excluded_after_eval.append(result)
                else:
                    results.append(result)
            else:
                print(f"   ✗ Error: {result.get('error', 'Unknown')}")
                results.append(result)

        # Filter successful results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"\n{'='*70}")
        print(f"PHASE 3 COMPLETE: {len(successful)} successful, {len(failed)} failed, {len(excluded_after_eval)} excluded (low ROI)")
        print(f"{'='*70}")

        # Move newly excluded agents to exclude/ folder
        if excluded_after_eval and not dry_run:
            print(f"\nMoving {len(excluded_after_eval)} low-ROI agents to exclude/...")
            self.local_exclude_dir.mkdir(parents=True, exist_ok=True)

            for r in excluded_after_eval:
                filename = r['filename']
                from_archive = filename in archive_set
                if from_archive:
                    cloud_src = f"{self.cloud_archive_prefix}{filename}"
                else:
                    cloud_src = f"{self.cloud_agents_prefix}{filename}"
                cloud_dst = f"{self.cloud_exclude_prefix}{filename}"

                try:
                    # Copy from temp to local exclude
                    src_path = self.temp_agents_dir / filename
                    local_exclude_path = self.local_exclude_dir / filename
                    if src_path.exists():
                        import shutil
                        shutil.copy(str(src_path), str(local_exclude_path))

                    # Upload to cloud exclude
                    self.cloud_sync.upload_file_verified(str(local_exclude_path), cloud_dst)

                    # Delete from source in cloud
                    if self.cloud_sync.file_exists(cloud_dst):
                        self.cloud_sync.delete_file(cloud_src)
                        # Also move JSON scoresheet if from archive
                        if from_archive:
                            json_src = cloud_src.replace('.pth', '.json')
                            json_dst = cloud_dst.replace('.pth', '.json')
                            if self.cloud_sync.file_exists(json_src):
                                local_json = self.local_exclude_dir / filename.replace('.pth', '.json')
                                self.cloud_sync.download_file(json_src, str(local_json))
                                self.cloud_sync.upload_file_verified(str(local_json), json_dst)
                                self.cloud_sync.delete_file(json_src)
                        print(f"  ✓ {filename} (ROI: {r['median_metrics']['roi']:.4f}%)")
                    else:
                        print(f"  ⚠ {filename} - upload not verified")
                except Exception as e:
                    print(f"  ✗ {filename} - {e}")

        if failed:
            print("\nFailed agents:")
            for r in failed:
                print(f"  ✗ {r['filename']}: {r.get('error', 'Unknown')}")

        print("\n" + "="*70)
        print("PHASE 4: Select top 50 by avg of lowest 2 scores")
        print("="*70)

        # Sort by avg_lowest_2 score (descending)
        successful.sort(key=lambda x: x['avg_lowest_2'], reverse=True)

        # Split into top 50 and remainder
        top_50 = successful[:50]
        to_archive = successful[50:]

        print(f"\nSelection Results:")
        print(f"  Top 50: {len(top_50)} agents")
        print(f"  To archive: {len(to_archive)} agents")

        if top_50:
            print(f"\n  Threshold (rank #50): {top_50[-1]['avg_lowest_2']:.2f}")
            print(f"  Best score: {top_50[0]['avg_lowest_2']:.2f}")

        # Show top 50
        print(f"\n{'='*70}")
        print("TOP 50 (will remain in Global 50):")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'Avg Low 2':<12} {'Scores':<30} {'Run Name':<30}")
        print("-"*70)
        for i, r in enumerate(top_50, 1):
            scores_str = ", ".join([f"{s:.2f}" for s in r['scores']])
            print(f"{i:<6} {r['avg_lowest_2']:<12.2f} [{scores_str}] {r['run_name'][:28]:<30}")

        # Show agents to archive
        if to_archive:
            print(f"\n{'='*70}")
            print("TO BE ARCHIVED (below threshold):")
            print(f"{'='*70}")
            print(f"{'Rank':<6} {'Avg Low 2':<12} {'Scores':<30} {'Run Name':<30}")
            print("-"*70)
            for i, r in enumerate(to_archive, 1):
                scores_str = ", ".join([f"{s:.2f}" for s in r['scores']])
                print(f"{50+i:<6} {r['avg_lowest_2']:<12.2f} [{scores_str}] {r['run_name'][:28]:<30}")

        print(f"\n{'='*70}")
        print("PHASE 5: Confirmation")
        print(f"{'='*70}")

        if dry_run:
            print("\n[DRY RUN] No changes will be made.")
            print(f"  Would keep: {len(top_50)} agents")
            print(f"  Would archive: {len(to_archive)} agents")
            return

        print(f"\nThis will:")
        print(f"  - Keep {len(top_50)} agents in Global 50")
        print(f"  - Archive {len(to_archive)} agents")
        print(f"  - Update global50.json locally and in cloud")
        print(f"  - Run final evaluation for pure scores")

        confirm = input("\nProceed? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("\n✗ Aborted by user.")
            return

        print("\n" + "="*70)
        print("PHASE 6: Archive non-selected agents")
        print("="*70)

        # Create local directories
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.local_agents_dir.mkdir(parents=True, exist_ok=True)
        self.local_archive_dir.mkdir(parents=True, exist_ok=True)

        # Archive agents that didn't make the cut
        for r in to_archive:
            filename = r['filename']
            from_archive = filename in archive_set
            location = "archive" if from_archive else "agents"
            print(f"  Archiving: {filename} (from {location})...", end=" ")

            # Determine source path in cloud
            if from_archive:
                cloud_src = f"{self.cloud_archive_prefix}{filename}"
            else:
                cloud_src = f"{self.cloud_agents_prefix}{filename}"
            cloud_dst = f"{self.cloud_archive_prefix}{filename}"

            try:
                # Copy from temp dir to local archive (we already have it downloaded)
                local_archive_path = self.local_archive_dir / filename
                src_path = self.temp_agents_dir / filename
                if src_path.exists():
                    import shutil
                    shutil.copy(str(src_path), str(local_archive_path))

                # Upload to cloud archive (if not already there)
                if not from_archive:
                    self.cloud_sync.upload_file_verified(str(local_archive_path), cloud_dst)

                    # Delete from cloud agents/
                    if self.cloud_sync.file_exists(cloud_dst):
                        self.cloud_sync.delete_file(cloud_src)
                        print("✓")
                    else:
                        print("⚠ Archive upload not verified")
                else:
                    # Already in archive, just print success
                    print("✓ (already in archive)")
            except Exception as e:
                print(f"✗ Error: {e}")

            # Save scoresheet
            scoresheet_filename = filename.replace('.pth', '.json')
            scoresheet_path = self.local_archive_dir / scoresheet_filename
            scoresheet_data = {
                'run_name': r['run_name'],
                'agent_id': r['agent_id'],
                'avg_lowest_2': r['avg_lowest_2'],
                'scores': r['scores'],
                'roi': r['median_metrics']['roi'],
                'expectancy': r['median_metrics']['expectancy'],
                'archived_at': datetime.utcnow().isoformat() + 'Z'
            }
            with open(scoresheet_path, 'w') as f:
                json.dump(scoresheet_data, f, indent=2)

        print("\n" + "="*70)
        print("PHASE 7: Run final evaluation for pure scores")
        print("="*70)

        # Copy selected agents to local agents dir and sync to cloud
        print("\nSyncing top 50 agents to local and cloud agents/...")
        for r in top_50:
            filename = r['filename']
            from_archive = filename in archive_set
            src = self.temp_agents_dir / filename
            dst = self.local_agents_dir / filename

            if src.exists():
                import shutil
                shutil.copy(str(src), str(dst))

                # If agent came from archive, upload to cloud agents/ and remove from archive
                if from_archive:
                    cloud_agents_path = f"{self.cloud_agents_prefix}{filename}"
                    cloud_archive_path = f"{self.cloud_archive_prefix}{filename}"

                    # Upload to agents/
                    self.cloud_sync.upload_file_verified(str(dst), cloud_agents_path)

                    # Delete from archive/
                    if self.cloud_sync.file_exists(cloud_agents_path):
                        self.cloud_sync.delete_file(cloud_archive_path)
                        # Also delete JSON scoresheet from archive
                        json_archive_path = cloud_archive_path.replace('.pth', '.json')
                        if self.cloud_sync.file_exists(json_archive_path):
                            self.cloud_sync.delete_file(json_archive_path)
                        print(f"  ✓ {filename} (moved from archive to agents)")
                else:
                    print(f"  ✓ {filename} (already in agents)")

        # Final evaluation (single run for pure scores)
        final_entries = []
        for i, r in enumerate(top_50, 1):
            print(f"\n[{i}/{len(top_50)}] Final eval: {r['filename']}")

            agent_path = self.local_agents_dir / r['filename']
            try:
                agent = DDPGAgent(agent_id=r['agent_id'])
                agent.load(str(agent_path))

                final_score, final_metrics = self.run_gauntlet(agent, r['run_name'])

                print(f"   Final Score: {final_score:.2f}")
                print(f"   ROI: {final_metrics['roi']:.2f}%")
                print(f"   Expectancy: {final_metrics['expectancy']:.4f}")

                entry = GlobalHoFEntry(
                    agent_id=r['agent_id'],
                    run_name=r['run_name'],
                    gauntlet_score=final_score,
                    generation=0,
                    roi=final_metrics['roi'],
                    expectancy=final_metrics['expectancy'],
                    quality_ratio=final_metrics['quality_ratio'],
                    win_ratio=final_metrics['win_ratio'],
                    total_trades=final_metrics['total_trades']
                )
                final_entries.append(entry)

            except Exception as e:
                print(f"   ✗ Error: {e}")
                # Use avg_lowest_2 as fallback
                entry = GlobalHoFEntry(
                    agent_id=r['agent_id'],
                    run_name=r['run_name'],
                    gauntlet_score=r['avg_lowest_2'],
                    generation=0,
                    roi=r['median_metrics']['roi'],
                    expectancy=r['median_metrics']['expectancy'],
                    quality_ratio=r['median_metrics']['quality_ratio'],
                    win_ratio=r['median_metrics']['win_ratio'],
                    total_trades=r['median_metrics']['total_trades']
                )
                final_entries.append(entry)

        # Sort by final score
        final_entries.sort(key=lambda e: e.gauntlet_score, reverse=True)

        print("\n" + "="*70)
        print("PHASE 8: Update global50.json")
        print("="*70)

        # Build new global50.json
        league_rules = LeagueRules(context_window_days=self.context_window_days)

        data = {
            'league_rules': league_rules.to_dict(),
            'entries': [e.to_dict() for e in final_entries],
            'capacity': 50,
            'version': '1.0',
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        }

        # Save locally
        with open(self.local_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved local: {self.local_json_path}")

        # Upload to cloud
        if self.cloud_sync.upload_file_verified(str(self.local_json_path), self.cloud_json_path):
            print(f"  ✓ Uploaded to cloud: {self.cloud_json_path}")
        else:
            print(f"  ⚠ Cloud upload may have failed")

        # Sync agent files to cloud
        print("\nSyncing agent files to cloud...")
        for agent_file in self.local_agents_dir.glob("*.pth"):
            cloud_path = f"{self.cloud_agents_prefix}{agent_file.name}"
            if self.cloud_sync.upload_file_verified(str(agent_file), cloud_path):
                print(f"  ✓ {agent_file.name}")
            else:
                print(f"  ⚠ {agent_file.name} - upload may have failed")

        print("\n" + "="*70)
        print("FIX COMPLETE")
        print("="*70)
        print(f"\nGlobal 50 rebuilt with {len(final_entries)} agents")
        print(f"  Best score:  {final_entries[0].gauntlet_score:.2f}")
        print(f"  Threshold:   {final_entries[-1].gauntlet_score:.2f}")
        print(f"  Archived:    {len(to_archive)} agents")
        print(f"\nLocal:  {self.local_dir}")
        print(f"Cloud:  gs://{self.cloud_sync.bucket_name}/{self.cloud_base}/")

        # Cleanup temp dir
        print(f"\nCleaning up temp directory...")
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("✓ Done")

    def cleanup(self):
        """
        Cleanup mode: Sync local state with cloud without re-evaluating.

        This assumes we just ran fix() and need to ensure:
        1. global50.json is consistent (entries match actual agent files)
        2. agents/ folder has all files for entries in global50.json
        3. archive/ has agents not in global50.json
        4. exclude/ has low-ROI agents
        5. No orphan files exist
        """
        print("\n" + "="*70)
        print("CLEANUP MODE")
        print("="*70)

        # Create local directories
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.local_agents_dir.mkdir(parents=True, exist_ok=True)
        self.local_archive_dir.mkdir(parents=True, exist_ok=True)
        self.local_exclude_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Download current global50.json
        print("\n1. Loading global50.json from cloud...")
        if not self.cloud_sync.download_file(self.cloud_json_path, str(self.local_json_path)):
            print("   ✗ Failed to download global50.json")
            return

        with open(self.local_json_path, 'r') as f:
            data = json.load(f)

        entries = data.get('entries', [])
        print(f"   Found {len(entries)} entries in global50.json")

        # Build expected filenames from entries
        expected_agents = {}
        for entry in entries:
            filename = f"{entry['run_name']}_{entry['agent_id']}.pth"
            expected_agents[filename] = entry

        # Step 2: List actual files in cloud
        print("\n2. Listing files in cloud storage...")
        all_files, agents_files, archive_files = self.list_cloud_agents(include_archive=True)

        agents_set = set(agents_files)
        archive_set = set(archive_files)

        # Step 3: Check for missing agent files
        print("\n3. Checking for missing agent files...")
        missing_agents = []
        for filename in expected_agents:
            if filename not in agents_set:
                missing_agents.append(filename)
                print(f"   ⚠ Missing in agents/: {filename}")

        if not missing_agents:
            print("   ✓ All expected agents present in agents/")

        # Step 4: Check for orphan files in agents/
        print("\n4. Checking for orphan files in agents/...")
        orphan_agents = []
        for filename in agents_files:
            if filename not in expected_agents:
                orphan_agents.append(filename)
                print(f"   ⚠ Orphan in agents/: {filename}")

        if not orphan_agents:
            print("   ✓ No orphan files in agents/")

        # Step 5: Try to recover missing agents from archive
        if missing_agents:
            print("\n5. Attempting to recover missing agents from archive...")
            for filename in missing_agents:
                if filename in archive_set:
                    print(f"   Found {filename} in archive, moving to agents/...")
                    cloud_src = f"{self.cloud_archive_prefix}{filename}"
                    cloud_dst = f"{self.cloud_agents_prefix}{filename}"
                    local_path = self.local_agents_dir / filename

                    # Download from archive
                    if self.cloud_sync.download_file(cloud_src, str(local_path)):
                        # Upload to agents
                        if self.cloud_sync.upload_file_verified(str(local_path), cloud_dst):
                            # Delete from archive
                            self.cloud_sync.delete_file(cloud_src)
                            # Delete JSON scoresheet from archive too
                            json_src = cloud_src.replace('.pth', '.json')
                            if self.cloud_sync.file_exists(json_src):
                                self.cloud_sync.delete_file(json_src)
                            print(f"   ✓ Recovered {filename}")
                        else:
                            print(f"   ✗ Failed to upload {filename}")
                    else:
                        print(f"   ✗ Failed to download {filename} from archive")
                else:
                    print(f"   ✗ {filename} not found in archive either - entry invalid!")
        else:
            print("\n5. No missing agents to recover")

        # Step 6: Move orphan agents to archive
        if orphan_agents:
            print("\n6. Moving orphan agents to archive...")
            for filename in orphan_agents:
                cloud_src = f"{self.cloud_agents_prefix}{filename}"
                cloud_dst = f"{self.cloud_archive_prefix}{filename}"
                local_path = self.local_archive_dir / filename

                # Download to local archive
                if self.cloud_sync.download_file(cloud_src, str(local_path)):
                    # Upload to cloud archive
                    if self.cloud_sync.upload_file_verified(str(local_path), cloud_dst):
                        # Delete from agents
                        self.cloud_sync.delete_file(cloud_src)
                        print(f"   ✓ Archived {filename}")

                        # Create scoresheet for archived agent
                        scoresheet_path = self.local_archive_dir / filename.replace('.pth', '.json')
                        scoresheet_data = {
                            'run_name': filename.rsplit('_', 1)[0],
                            'agent_id': int(filename.rsplit('_', 1)[1].replace('.pth', '')),
                            'archived_at': datetime.utcnow().isoformat() + 'Z',
                            'reason': 'orphan_cleanup'
                        }
                        with open(scoresheet_path, 'w') as f:
                            json.dump(scoresheet_data, f, indent=2)
                    else:
                        print(f"   ✗ Failed to archive {filename}")
                else:
                    print(f"   ✗ Failed to download {filename}")
        else:
            print("\n6. No orphan agents to archive")

        # Step 7: Remove invalid entries from global50.json
        print("\n7. Validating global50.json entries...")
        # Re-check which agents actually exist now
        valid_entries = []
        invalid_entries = []

        for entry in entries:
            filename = f"{entry['run_name']}_{entry['agent_id']}.pth"
            cloud_path = f"{self.cloud_agents_prefix}{filename}"
            if self.cloud_sync.file_exists(cloud_path):
                valid_entries.append(entry)
            else:
                invalid_entries.append(entry)
                print(f"   ⚠ Removing invalid entry: {filename}")

        if invalid_entries:
            print(f"\n   Removing {len(invalid_entries)} invalid entries...")
            data['entries'] = valid_entries
            data['last_updated'] = datetime.utcnow().isoformat() + 'Z'

            # Save locally
            with open(self.local_json_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Upload to cloud
            if self.cloud_sync.upload_file_verified(str(self.local_json_path), self.cloud_json_path):
                print(f"   ✓ Updated global50.json ({len(valid_entries)} entries)")
            else:
                print(f"   ✗ Failed to upload global50.json")
        else:
            print("   ✓ All entries are valid")

        # Step 8: Sync local directories with cloud
        print("\n8. Syncing local directories with cloud...")

        # Download all agents to local
        print("   Syncing agents/...")
        for filename in expected_agents:
            cloud_path = f"{self.cloud_agents_prefix}{filename}"
            local_path = self.local_agents_dir / filename
            if not local_path.exists():
                if self.cloud_sync.download_file(cloud_path, str(local_path)):
                    print(f"   ✓ Downloaded {filename}")

        # Summary
        print("\n" + "="*70)
        print("CLEANUP COMPLETE")
        print("="*70)
        print(f"\n  Global 50 entries: {len(valid_entries)}")
        print(f"  Invalid entries removed: {len(invalid_entries)}")
        print(f"  Missing agents recovered: {len([f for f in missing_agents if f in archive_set])}")
        print(f"  Orphan agents archived: {len(orphan_agents)}")
        print(f"\n  Local:  {self.local_dir}")
        print(f"  Cloud:  gs://{self.cloud_sync.bucket_name}/{self.cloud_base}/")


def main():
    parser = argparse.ArgumentParser(
        description="Fix Global 50 by re-evaluating all agents from cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_global50.py                    # Fix cw151 (default) - full re-evaluation
  python fix_global50.py --context-window 504   # Fix cw504
  python fix_global50.py --dry-run          # Preview without changes
  python fix_global50.py --cleanup          # Sync state without re-evaluating
        """
    )

    parser.add_argument(
        '--context-window',
        type=int,
        default=Config.CONTEXT_WINDOW_DAYS,
        help=f'Context window to fix (default: {Config.CONTEXT_WINDOW_DAYS})'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would happen without making changes'
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Sync local/cloud state without re-evaluating. Ensures global50.json matches actual files.'
    )

    args = parser.parse_args()

    fixer = Global50Fixer(context_window_days=args.context_window)

    if args.cleanup:
        fixer.cleanup()
    else:
        fixer.fix(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
