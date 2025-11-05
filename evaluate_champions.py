"""
Champion Agent Evaluation Script for Project Eigen 2

This script:
1. Downloads all best agents from GCP checkpoint runs
2. Validates each agent on:
   - The validation dataset
   - 5 random windows from the training dataset
3. Ranks agents by average validation performance
4. Saves top 3 agents to "Champion_Agents" folder in GCP with metadata
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.ddpg_agent import DDPGAgent
from environment.trading_env import TradingEnvironment
from data.data_loader import DataLoader
from utils.config import Config
from utils.cloud_sync import CloudSync, get_cloud_sync_from_env


class ChampionEvaluator:
    """Evaluates all champion agents from GCP and identifies the best."""

    def __init__(self):
        """Initialize the evaluator."""
        print("="*70)
        print("Champion Agent Evaluation System".center(70))
        print("="*70)

        # Initialize cloud sync from environment variables
        self.cloud_sync = get_cloud_sync_from_env()

        if self.cloud_sync.provider == "local":
            print("\n❌ ERROR: Cloud sync is set to 'local' mode.")
            print("Please set environment variables:")
            print("  CLOUD_PROVIDER=gcs")
            print("  CLOUD_BUCKET=your-bucket-name")
            print("  GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json")
            sys.exit(1)

        # Load data
        print("\n--- Loading Data ---")
        self.data_loader = DataLoader(str(Config.DATA_PATH))
        self.normalization_stats = self.data_loader.normalization_stats

        # Calculate data splits
        total_days = len(self.data_loader.dates)
        self.train_end_idx = int(total_days * Config.TRAIN_TEST_SPLIT)
        self.val_start_idx = self.train_end_idx
        self.val_end_idx = total_days

        print(f"Total days: {total_days}")
        print(f"Training days: 0 to {self.train_end_idx}")
        print(f"Validation days: {self.val_start_idx} to {self.val_end_idx}")

        # Create validation environment
        print("\n--- Creating Validation Environment ---")
        self.val_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            train_mode=False
        )

        # Create training evaluation environment
        self.train_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            train_mode=False
        )

        print("✓ Environments ready")

        # Local temporary directory for downloads
        self.temp_dir = Path("temp_champions")
        self.temp_dir.mkdir(exist_ok=True)

        # Results storage
        self.evaluation_results = []

    def list_all_runs(self) -> List[str]:
        """
        List all training run directories in GCP bucket.

        Returns:
            List of run names (directory names)
        """
        print("\n--- Discovering Training Runs in GCP ---")

        try:
            # List all directories in checkpoints folder
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(self.cloud_sync.bucket_name)

            # Get all blobs with prefix "checkpoints/"
            blobs = bucket.list_blobs(prefix=f"{self.cloud_sync.project_name}/checkpoints/")

            # Extract unique run names from paths
            run_names = set()
            for blob in blobs:
                # Path format: eigen2/checkpoints/{run_name}/...
                path_parts = blob.name.split('/')
                if len(path_parts) >= 3:
                    run_name = path_parts[2]
                    if run_name:  # Not empty
                        run_names.add(run_name)

            run_names = sorted(list(run_names))
            print(f"✓ Found {len(run_names)} training runs")
            for i, name in enumerate(run_names, 1):
                print(f"  {i}. {name}")

            return run_names

        except Exception as e:
            print(f"❌ Error listing runs: {e}")
            return []

    def download_best_agent(self, run_name: str) -> Tuple[DDPGAgent, Dict]:
        """
        Download and load best agent from a specific run.

        Args:
            run_name: Name of the training run

        Returns:
            Tuple of (agent, metadata) or (None, None) if failed
        """
        try:
            # Download best_agent.pth
            gcp_path = f"{self.cloud_sync.project_name}/checkpoints/{run_name}/best_agent.pth"
            local_path = self.temp_dir / f"{run_name}_best_agent.pth"

            self.cloud_sync.download_file(gcp_path, str(local_path))

            if not local_path.exists():
                print(f"  ❌ Failed to download: {run_name}")
                return None, None

            # Load agent
            agent = DDPGAgent(agent_id='eval')
            agent.load(str(local_path))

            # Try to load trainer state for metadata
            metadata = {'run_name': run_name, 'gcp_path': gcp_path}

            try:
                gcp_state_path = f"{self.cloud_sync.project_name}/checkpoints/{run_name}/trainer_state.json"
                local_state_path = self.temp_dir / f"{run_name}_trainer_state.json"

                self.cloud_sync.download_file(gcp_state_path, str(local_state_path))

                if local_state_path.exists():
                    with open(local_state_path, 'r') as f:
                        state = json.load(f)
                        metadata['generation'] = state.get('generation', 'unknown')
                        metadata['best_fitness'] = state.get('best_fitness', 'unknown')
                        metadata['best_validation_fitness'] = state.get('best_validation_fitness', 'unknown')
            except:
                pass  # Metadata is optional

            return agent, metadata

        except Exception as e:
            print(f"  ❌ Error loading {run_name}: {e}")
            return None, None

    def run_episode(self, agent: DDPGAgent, env: TradingEnvironment,
                   start_idx: int, end_idx: int) -> float:
        """
        Run a single episode and return fitness.

        Args:
            agent: Agent to evaluate
            env: Environment to use
            start_idx: Start index in data
            end_idx: End index in data

        Returns:
            Fitness score
        """
        state = env.reset(start_idx=start_idx, end_idx=end_idx)
        done = False

        while not done:
            # Get action from agent (no exploration noise)
            action = agent.select_action(state, noise_scale=0.0)

            # Take step
            next_state, reward, done, info = env.step(action)
            state = next_state

        # Return cumulative reward as fitness
        return info.get('cumulative_reward', 0.0)

    def evaluate_agent(self, agent: DDPGAgent, run_name: str) -> Dict:
        """
        Evaluate an agent on validation set + 5 random training windows.

        Args:
            agent: Agent to evaluate
            run_name: Name of the run (for logging)

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n  Evaluating {run_name}...")

        fitness_scores = []

        # 1. Validate on validation set
        val_start = self.val_start_idx
        val_end = min(
            self.val_start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS,
            self.val_end_idx
        )

        val_fitness = self.run_episode(agent, self.val_env, val_start, val_end)
        fitness_scores.append(val_fitness)
        print(f"    Validation set fitness: {val_fitness:.2f}")

        # 2. Validate on 5 random training windows
        print("    Random training windows:")
        for i in range(5):
            # Random window from training data
            window_size = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            max_start = self.train_end_idx - window_size

            if max_start <= 0:
                print(f"      Window {i+1}: Skipped (insufficient data)")
                continue

            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

            train_fitness = self.run_episode(agent, self.train_env, start_idx, end_idx)
            fitness_scores.append(train_fitness)
            print(f"      Window {i+1} (days {start_idx}-{end_idx}): {train_fitness:.2f}")

        # Calculate statistics
        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        min_fitness = np.min(fitness_scores)
        max_fitness = np.max(fitness_scores)

        print(f"    Average fitness: {avg_fitness:.2f} ± {std_fitness:.2f}")
        print(f"    Range: [{min_fitness:.2f}, {max_fitness:.2f}]")

        return {
            'run_name': run_name,
            'validation_fitness': val_fitness,
            'training_windows_fitness': fitness_scores[1:],  # Exclude validation
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'min_fitness': min_fitness,
            'max_fitness': max_fitness,
            'all_scores': fitness_scores
        }

    def save_champion_to_gcp(self, run_name: str, rank: int, results: Dict):
        """
        Save a champion agent to the Champion_Agents folder in GCP.

        Args:
            run_name: Original run name
            rank: Rank (1, 2, or 3)
            results: Evaluation results
        """
        print(f"\n  Saving Champion #{rank}: {run_name}")

        try:
            # Create champion directory name
            champion_name = f"rank_{rank}_{run_name}"

            # Copy agent file to new location
            source_path = f"{self.cloud_sync.project_name}/checkpoints/{run_name}/best_agent.pth"
            dest_path = f"{self.cloud_sync.project_name}/Champion_Agents/{champion_name}/best_agent.pth"

            # Download locally first (already done), then upload to new location
            local_agent_path = self.temp_dir / f"{run_name}_best_agent.pth"
            self.cloud_sync.upload_file(str(local_agent_path), dest_path)

            # Create and upload metadata
            metadata = {
                'rank': rank,
                'original_run_name': run_name,
                'original_gcp_path': source_path,
                'evaluation_results': results
            }

            metadata_path = self.temp_dir / f"{champion_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            dest_meta_path = f"{self.cloud_sync.project_name}/Champion_Agents/{champion_name}/metadata.json"
            self.cloud_sync.upload_file(str(metadata_path), dest_meta_path)

            print(f"    ✓ Saved to: {dest_path}")
            print(f"    ✓ Metadata: {dest_meta_path}")

        except Exception as e:
            print(f"    ❌ Error saving champion: {e}")

    def run_evaluation(self):
        """Main evaluation pipeline."""
        print("\n" + "="*70)
        print("Starting Champion Evaluation".center(70))
        print("="*70)

        # 1. List all runs
        run_names = self.list_all_runs()

        if not run_names:
            print("\n❌ No training runs found in GCP bucket.")
            return

        # 2. Download and evaluate each agent
        print("\n--- Downloading and Evaluating Agents ---")

        for run_name in tqdm(run_names, desc="Evaluating runs"):
            # Download agent
            agent, metadata = self.download_best_agent(run_name)

            if agent is None:
                continue

            # Evaluate agent
            results = self.evaluate_agent(agent, run_name)
            results['metadata'] = metadata

            self.evaluation_results.append(results)

            # Clean up agent to save memory
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 3. Rank agents
        print("\n" + "="*70)
        print("Ranking Results".center(70))
        print("="*70)

        # Sort by average fitness
        self.evaluation_results.sort(key=lambda x: x['avg_fitness'], reverse=True)

        # Print all results
        print("\nAll Agents (ranked by average fitness):")
        print("-"*70)
        for i, result in enumerate(self.evaluation_results, 1):
            print(f"\n{i}. {result['run_name']}")
            print(f"   Average Fitness: {result['avg_fitness']:.2f} ± {result['std_fitness']:.2f}")
            print(f"   Validation Fitness: {result['validation_fitness']:.2f}")
            print(f"   Original Metadata:")
            if 'metadata' in result and result['metadata']:
                for key, value in result['metadata'].items():
                    if key != 'gcp_path':  # Print path separately
                        print(f"     - {key}: {value}")

        # 4. Save top 3 to GCP
        print("\n" + "="*70)
        print("Saving Top 3 Champions to GCP".center(70))
        print("="*70)

        top_3 = self.evaluation_results[:3]

        for rank, result in enumerate(top_3, 1):
            print(f"\n{'='*70}")
            print(f"Champion #{rank}".center(70))
            print(f"{'='*70}")
            print(f"Run Name: {result['run_name']}")
            print(f"Original GCP Path: {result['metadata']['gcp_path']}")
            print(f"Average Fitness: {result['avg_fitness']:.2f} ± {result['std_fitness']:.2f}")
            print(f"Validation Fitness: {result['validation_fitness']:.2f}")
            print(f"Fitness Range: [{result['min_fitness']:.2f}, {result['max_fitness']:.2f}]")

            # Save to GCP
            self.save_champion_to_gcp(result['run_name'], rank, result)

        # 5. Create summary file
        print("\n--- Creating Summary Report ---")
        summary = {
            'evaluation_date': str(np.datetime64('now')),
            'total_runs_evaluated': len(self.evaluation_results),
            'top_3_champions': [
                {
                    'rank': i+1,
                    'run_name': result['run_name'],
                    'avg_fitness': float(result['avg_fitness']),
                    'validation_fitness': float(result['validation_fitness']),
                    'original_gcp_path': result['metadata']['gcp_path']
                }
                for i, result in enumerate(top_3)
            ]
        }

        summary_path = self.temp_dir / "champion_evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        gcp_summary_path = f"{self.cloud_sync.project_name}/Champion_Agents/evaluation_summary.json"
        self.cloud_sync.upload_file(str(summary_path), gcp_summary_path)

        print(f"✓ Summary saved to: {gcp_summary_path}")

        # Final summary
        print("\n" + "="*70)
        print("Evaluation Complete!".center(70))
        print("="*70)
        print(f"\nEvaluated: {len(self.evaluation_results)} agents")
        print(f"Top 3 champions saved to: gs://{self.cloud_sync.bucket_name}/{self.cloud_sync.project_name}/Champion_Agents/")
        print("\nTop 3 Champions:")
        for i, result in enumerate(top_3, 1):
            print(f"  {i}. {result['run_name']} - Avg Fitness: {result['avg_fitness']:.2f}")


def main():
    """Main entry point."""
    evaluator = ChampionEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
