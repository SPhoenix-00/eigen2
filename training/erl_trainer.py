"""
ERL Trainer for Project Eigen 2
Evolutionary Reinforcement Learning training loop
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import os
import json
import wandb
import gc
import matplotlib.pyplot as plt
import warnings
from gymnasium.vector import AsyncVectorEnv

# Suppress common library warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda.amp')

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from models.replay_buffer import ReplayBuffer, OnDiskReplayBuffer
from erl.genetic_ops import create_next_generation
from utils.config import Config
from utils.display import print_generation_summary, print_final_summary, plot_fitness_progress, ResourceTracker
from utils.cloud_sync import get_cloud_sync_from_env
from torch.utils.data import DataLoader
# from utils.memory_profiler import get_profiler, log_memory  # Memory profiling disabled


class ERLTrainer:
    """
    Evolutionary Reinforcement Learning Trainer.
    Manages population, training, and evolution.
    """

    def __init__(self, data_loader: StockDataLoader, resume_run_name: str = None):
        """
        Initialize ERL trainer.

        Args:
            data_loader: Loaded data with train/val splits
            resume_run_name: Optional wandb run name to resume from (e.g., "azure-thunder-123")
        """
        self.data_loader = data_loader
        self.resume_run_name = resume_run_name

        # Load stock names for trade reporting
        import pandas as pd
        df = pd.read_pickle(Config.DATA_PATH)
        all_columns = df.columns.tolist()
        self.stock_names = all_columns[Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL + 1]

        # Compute normalization stats ONCE and cache them
        print("Computing and caching normalization statistics...")
        self.normalization_stats = data_loader.compute_normalization_stats()
        
        # Initialize population
        print(f"Initializing population of {Config.POPULATION_SIZE} agents...")
        self.population = [DDPGAgent(agent_id=i) for i in range(Config.POPULATION_SIZE)]

        # Training range (excludes interim validation and holdout sets)
        self.train_start_idx = Config.CONTEXT_WINDOW_DAYS
        self.train_end_idx = len(data_loader.train_indices)

        # Interim validation range (for walk-forward validation during training)
        # This is separate from training data to ensure genuine out-of-sample validation
        self.interim_val_start_idx = len(data_loader.train_indices)
        self.interim_val_end_idx = self.interim_val_start_idx + len(data_loader.interim_val_indices)

        # Walk-forward validation slices (generated per generation)
        # Each generation uses 3 random validation slices from INTERIM validation set
        # Format: list of (start_idx, end_idx, trading_end_idx) tuples
        self.current_generation_val_slices = []
        
        # Logging
        self.writer = SummaryWriter(log_dir=str(Config.LOG_DIR))

        # Cloud sync (needed before wandb init for checkpoint downloading)
        self.cloud_sync = get_cloud_sync_from_env()

        # Initialize Weights & Biases and checkpoint directory
        # Note: entity defaults to your personal workspace (eigen2)
        if wandb.run is None:
            # If resuming from a specific run, try to load its wandb ID
            if self.resume_run_name:
                print(f"--- Resuming W&B run: {self.resume_run_name} ---")

                # Set checkpoint directory based on provided run name
                self.checkpoint_dir = Config.CHECKPOINT_DIR / self.resume_run_name
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Try to download checkpoint from cloud and extract wandb run ID
                if not self.checkpoint_dir.exists() or len(list(self.checkpoint_dir.glob('*'))) == 0:
                    print("! No local checkpoint found. Downloading from cloud...")
                    self.cloud_sync.download_checkpoints(str(self.checkpoint_dir))

                # Try to load trainer state to get wandb run ID
                wandb_run_id = None
                state_path = self.checkpoint_dir / "trainer_state.json"
                if state_path.exists():
                    try:
                        import json
                        with open(state_path, 'r') as f:
                            trainer_state = json.load(f)
                        wandb_run_id = trainer_state.get('wandb_run_id')
                        print(f"✓ Found wandb run ID: {wandb_run_id}")
                    except Exception as e:
                        print(f"⚠ Could not load wandb run ID from checkpoint: {e}")

                # Initialize wandb with the run ID for proper resume
                if wandb_run_id:
                    # CRITICAL: When resume="must", wandb does NOT allow config updates
                    # Config is already saved in the cloud, so don't pass it again
                    wandb.init(
                        project="eigen2-self",
                        id=wandb_run_id,
                        resume="must",  # Must resume this specific run
                    )
                else:
                    print("⚠ No wandb run ID found. Creating new run with same name...")
                    wandb.init(
                        project="eigen2-self",
                        name=self.resume_run_name,  # Try to use same name
                        resume="allow",
                        config={
                            "population_size": Config.POPULATION_SIZE,
                            "num_generations": Config.NUM_GENERATIONS,
                            "buffer_size": Config.BUFFER_SIZE,
                            "batch_size": Config.BATCH_SIZE,
                            "actor_lr": Config.ACTOR_LR,
                            "critic_lr": Config.CRITIC_LR,
                            "trading_period_days": Config.TRADING_PERIOD_DAYS,
                            "max_holding_period": Config.MAX_HOLDING_PERIOD,
                            "loss_penalty_multiplier": Config.LOSS_PENALTY_MULTIPLIER,
                            "num_stocks": Config.NUM_INVESTABLE_STOCKS,
                        }
                    )

                self.run_name = self.resume_run_name  # Use the provided name

                # Update last_run.json with the resumed run info
                self._write_last_run_file()
            else:
                # New training run
                print("--- Initializing new W&B run (main.py mode) ---")
                wandb.init(
                    project="eigen2-self",
                    #name=f"erl-{Config.NUM_GENERATIONS}gen",
                    config={
                        "population_size": Config.POPULATION_SIZE,
                        "num_generations": Config.NUM_GENERATIONS,
                        "buffer_size": Config.BUFFER_SIZE,
                        "batch_size": Config.BATCH_SIZE,
                        "actor_lr": Config.ACTOR_LR,
                        "critic_lr": Config.CRITIC_LR,
                        "trading_period_days": Config.TRADING_PERIOD_DAYS,
                        "max_holding_period": Config.MAX_HOLDING_PERIOD,
                        "loss_penalty_multiplier": Config.LOSS_PENALTY_MULTIPLIER,
                        "num_stocks": Config.NUM_INVESTABLE_STOCKS,
                    },
                    resume="allow"  # Allow resuming from checkpoints
                )

                # Create run-specific checkpoint directory using wandb run name
                self.run_name = wandb.run.name
                self.checkpoint_dir = Config.CHECKPOINT_DIR / self.run_name
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Write last_run.json for easy resume
                self._write_last_run_file()
        else:
            print("--- W&B run already active (sweep_runner.py mode) ---")
            # Create run-specific checkpoint directory using wandb run name
            self.run_name = wandb.run.name
            self.checkpoint_dir = Config.CHECKPOINT_DIR / self.run_name
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"W&B run: {wandb.run.name} (ID: {wandb.run.id})")

        # CRITICAL: Define metric step relationships for proper wandb resume
        # This tells wandb which metric to use as the x-axis for each metric group
        # Without this, resumed runs can have broken graphs
        wandb.define_metric("generation")
        wandb.define_metric("train/*", step_metric="generation")
        wandb.define_metric("fitness/*", step_metric="generation")
        wandb.define_metric("validation/*", step_metric="generation")
        wandb.define_metric("buffer/*", step_metric="generation")
        wandb.define_metric("training/*", step_metric="generation")
        wandb.define_metric("resources/*", step_metric="generation")
        wandb.define_metric("adaptive_mutation/*", step_metric="generation")
        wandb.define_metric("feature_importance/*", step_metric="generation")

        # Create replay buffer with storage INSIDE checkpoint directory
        # This ensures buffer files are synced to cloud along with checkpoints
        buffer_storage_path = str(self.checkpoint_dir / "buffer_storage")
        print(f"Buffer storage: {buffer_storage_path}")
        self.replay_buffer = OnDiskReplayBuffer(
            capacity=Config.BUFFER_SIZE,
            storage_path=buffer_storage_path
        )

        # Create DataLoader for asynchronous batch prefetching
        # Background workers prepare batches in parallel while GPU trains
        # This eliminates the GPU waiting for disk I/O
        self._create_dataloader()

        # Set unique random seed based on wandb run id
        run_id_hash = hash(wandb.run.id) % (2**32)
        self.seed = run_id_hash
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        import random
        random.seed(self.seed)
        print(f"Random seed: {self.seed}")

        self.start_generation = 0
        self.generation = 0
        self.best_fitness = float('-inf')  # Best training fitness (for logging)
        self.best_validation_fitness = float('-inf')  # Best validation fitness (for selection)
        self.best_agent = None

        # Statistics
        self.fitness_history = []
        self.generation_times = []
        self.validation_fitness_history = []  # Track validation fitness for plateau detection

        # Adaptive mutation parameters
        self.plateau_threshold = 0.02  # Consider plateau if improvement < 2% over window
        self.plateau_window = 3  # Number of generations to check for plateau
        self.base_mutation_rate = Config.MUTATION_RATE
        self.base_mutation_std = Config.MUTATION_STD
        self.current_mutation_rate = Config.MUTATION_RATE
        self.current_mutation_std = Config.MUTATION_STD
        self.mutation_boost_factor = 1.5  # Multiply by this when plateau detected
        self.max_mutation_rate = 0.8  # Cap mutation rate (doubled to allow plateau boost from 0.40 base)
        self.max_mutation_std = 0.1  # Cap mutation std (doubled to allow plateau boost from 0.05 base)
        self.plateau_detected = False

        # Track if we've saved buffer on first fill
        self.buffer_saved_on_first_fill = False

        # Feature importance tracking (cross-attention weights)
        self.feature_importance = torch.zeros(Config.TOTAL_COLUMNS, device=Config.DEVICE)  # Running average [num_columns]
        self.feature_importance_momentum = 0.99  # EMA momentum
        self.feature_importance_count = 0  # Track how many updates we've done

        # Feature importance history (per generation)
        from collections import deque
        self.feature_importance_history = deque(maxlen=10)  # Store last 10 generations

        # Column names for better logging
        self.column_names = data_loader.column_names if hasattr(data_loader, 'column_names') and data_loader.column_names else \
                           [f"col_{i}" for i in range(Config.TOTAL_COLUMNS)]

        # Initialize resource tracker
        self.resource_tracker = ResourceTracker(disk_path="/workspace")

        print(f"Training: days {self.train_start_idx}-{self.train_end_idx}, "
              f"Validation: days {self.interim_val_start_idx}-{self.interim_val_end_idx}, "
              f"Holdout: {Config.HOLDOUT_DAYS} days")
        print(f"Walk-forward: 3 random validation slices/generation")

        # Create persistent environment (reused across episodes to prevent memory leaks)
        print("Initializing environment...")
        self.eval_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            start_idx=self.train_start_idx,
            end_idx=self.train_end_idx,
            trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            data_array_full=self.data_loader.data_array_full
        )

        # Automatically load checkpoint if resuming
        if self.resume_run_name:
            print("\n" + "="*60)
            print("Loading checkpoint for resume...")
            print("="*60)
            self.load_checkpoint()

        # Memory profiling: Take baseline snapshot
        # print("\n🔍 Taking baseline memory snapshot...")
        # log_memory("Trainer initialized (baseline)", show_objects=True)

    def _write_last_run_file(self):
        """Write last_run.json to root directory for easy resume."""
        last_run_info = {
            'run_name': wandb.run.name,
            'run_id': wandb.run.id,  # W&B run ID (from URL)
            'project': 'eigen2-self',
            'timestamp': time.time()
        }

        last_run_file = Path("last_run.json")
        try:
            with open(last_run_file, 'w') as f:
                json.dump(last_run_info, f, indent=2)
            print(f"✓ Wrote run info to {last_run_file}")
        except Exception as e:
            print(f"⚠ Could not write last_run.json: {e}")

    def _create_dataloader(self):
        """
        Create or recreate DataLoader for the replay buffer.
        Called during __init__ and after loading checkpoints.
        """
        print(f"Creating DataLoader with {Config.NUM_DATALOADER_WORKERS} background workers...")
        self.replay_dataloader = DataLoader(
            self.replay_buffer,
            batch_size=None,  # Already batched by __iter__
            num_workers=Config.NUM_DATALOADER_WORKERS,
            pin_memory=True,  # Faster GPU transfer
            prefetch_factor=2,  # Each worker prefetches 2 batches ahead
            persistent_workers=True  # Keep workers alive between epochs
        )
        # Reset iterator when creating new DataLoader
        self.batch_iterator = None

    def _make_env(self, start_idx: int, end_idx: int, training_end_idx: int, is_training: bool):
        """
        Factory function to create a trading environment.
        Used for AsyncVectorEnv to create parallel environment instances.

        Args:
            start_idx: Starting day index
            end_idx: Ending day index
            training_end_idx: Last day new positions can be opened
            is_training: Whether environment is in training mode

        Returns:
            Callable that creates a TradingEnvironment
        """
        def _init():
            return TradingEnvironment(
                data_array=self.data_loader.data_array,
                dates=self.data_loader.dates,
                normalization_stats=self.normalization_stats,
                start_idx=start_idx,
                end_idx=end_idx,
                trading_end_idx=training_end_idx,
                data_array_full=self.data_loader.data_array_full,
                is_training=is_training
            )
        return _init

    def run_episode(self, agent: DDPGAgent, env: TradingEnvironment,
                   start_idx: int, end_idx: int,
                   training: bool = True) -> Tuple[float, Dict]:
        """
        Run one episode with an agent using a persistent environment.

        Args:
            agent: Agent to run
            env: Persistent TradingEnvironment to reuse (critical for memory efficiency)
            start_idx: Starting day index (first day of trading period)
            end_idx: Ending day index (includes settlement period)
            training: Whether this is training (adds to replay buffer)

        Returns:
            Tuple of (cumulative_reward, episode_info)
        """
        # Calculate trading end (when model stops opening new positions)
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

        # Set environment training mode (affects observation noise for regularization)
        env.set_training_mode(training)

        # CRITICAL FIX: Reset persistent environment with new indices
        # DO NOT create new TradingEnvironment here - reuse the passed env
        state, info = env.reset(
            start_idx=start_idx,
            end_idx=end_idx,
            trading_end_idx=trading_end_idx
        )
        cumulative_reward = 0.0
        steps = 0
        
        # Run episode
        while True:
            # Select action
            action = agent.select_action(state, add_noise=training)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition in replay buffer (if training)
            if training:
                self.replay_buffer.add(
                    state=state.astype(np.float32),
                    action=action.astype(np.float32),
                    reward=reward,
                    next_state=next_state.astype(np.float32),
                    done=float(terminated or truncated)
                )
            
            cumulative_reward += reward
            steps += 1
            
            # Move to next state
            state = next_state
            
            if terminated or truncated:
                break
        
        # Get episode summary
        episode_summary = env.get_episode_summary()
        episode_summary['steps'] = steps
        
        # CRITICAL: Use cumulative_reward from environment
        # This includes ALL penalties (inaction, losses, etc.)
        final_fitness = float(cumulative_reward)
        
        # Apply zero-trades penalty if no trades were made (silent - penalty speaks for itself)
        if episode_summary['num_trades'] == 0:
            final_fitness -= Config.ZERO_TRADES_PENALTY

        # NOTE: No need to delete env - we're reusing persistent environments now
        return final_fitness, episode_summary

    def run_vectorized_episodes(self, agent: DDPGAgent, episode_configs: List[Tuple[int, int, int]],
                               training: bool = True) -> List[Tuple[float, Dict]]:
        """
        Run multiple episodes in parallel using AsyncVectorEnv for CPU parallelization.

        Args:
            agent: Agent to run
            episode_configs: List of (start_idx, end_idx, trading_end_idx) tuples
            training: Whether this is training (adds to replay buffer)

        Returns:
            List of (cumulative_reward, episode_info) for each episode
        """
        num_envs = len(episode_configs)

        # Create environment makers for each configuration
        # Pass is_training flag to environment constructor
        env_fns = [self._make_env(start_idx, end_idx, trading_end_idx, training)
                   for start_idx, end_idx, trading_end_idx in episode_configs]

        # Create vectorized environment
        vec_env = AsyncVectorEnv(env_fns)

        # Reset all environments
        states, infos = vec_env.reset()

        # Initialize tracking
        cumulative_rewards = np.zeros(num_envs)
        steps = np.zeros(num_envs, dtype=int)
        dones = np.zeros(num_envs, dtype=bool)
        final_infos = [None] * num_envs  # Store final info for each env

        # Run episodes until all are done
        while not dones.all():
            # Get actions for all environments in parallel (batched inference on GPU)
            actions = agent.select_action(states, add_noise=training)

            # Step all environments in parallel
            next_states, rewards, terminateds, truncateds, infos = vec_env.step(actions)

            # Store transitions in replay buffer (if training)
            if training:
                for i in range(num_envs):
                    if not dones[i]:
                        self.replay_buffer.add(
                            state=states[i].astype(np.float32),
                            action=actions[i].astype(np.float32),
                            reward=rewards[i],
                            next_state=next_states[i].astype(np.float32),
                            done=float(terminateds[i] or truncateds[i])
                        )

            # Update tracking and capture final info when episode ends
            for i in range(num_envs):
                if not dones[i]:
                    cumulative_rewards[i] += rewards[i]
                    steps[i] += 1
                    # Check if this episode just finished
                    if terminateds[i] or truncateds[i]:
                        dones[i] = True
                        # Capture episode summary from info dict
                        # AsyncVectorEnv returns episode_summary as a dict where each value is an array
                        # We need to extract the value for this environment index
                        if isinstance(infos, dict) and 'episode_summary' in infos:
                            summary_dict = infos['episode_summary']
                            # Extract values for this environment index
                            final_infos[i] = {
                                k: v[i].item() if hasattr(v[i], 'item') else v[i]
                                for k, v in summary_dict.items()
                                if not k.startswith('_')  # Skip the boolean mask keys
                            }
                        else:
                            final_infos[i] = {}

            # Move to next state
            states = next_states

        # Collect results from all environments
        results = []
        for i in range(num_envs):
            # Get episode summary from captured final info
            episode_summary = final_infos[i] if final_infos[i] is not None else {}

            # Ensure it's a dict and has required keys
            if not isinstance(episode_summary, dict):
                episode_summary = {}

            # Add steps count if not already present
            if 'steps' not in episode_summary:
                episode_summary['steps'] = int(steps[i])

            # Ensure required keys exist with defaults
            episode_summary.setdefault('num_trades', 0)
            episode_summary.setdefault('num_wins', 0)
            episode_summary.setdefault('num_losses', 0)
            episode_summary.setdefault('win_rate', 0.0)
            episode_summary.setdefault('avg_reward_per_trade', 0.0)
            episode_summary.setdefault('max_coefficient_during_episode', 0.0)
            episode_summary.setdefault('closed_trades', [])

            # Calculate final fitness
            final_fitness = float(cumulative_rewards[i])

            # Apply zero-trades penalty if no trades were made
            if episode_summary['num_trades'] == 0:
                final_fitness -= Config.ZERO_TRADES_PENALTY

            results.append((final_fitness, episode_summary))

        # Clean up vectorized environment
        vec_env.close()
        del vec_env
        gc.collect()

        return results

    def generate_validation_slices(self) -> List[Tuple[int, int, int]]:
        """
        Generate 3 random validation slices for the current generation from INTERIM validation set.

        NEW: Divides interim validation period into 3 equal segments and samples one slice from each.
        This ensures diversity across different market conditions in the validation period.

        Each slice consists of:
        - CONTEXT_WINDOW_DAYS (504) of prior data (may come from training data for context)
        - TRADING_PERIOD_DAYS (125) where agent can trade (from interim validation set)
        - SETTLEMENT_PERIOD_DAYS (30) to close positions (from interim validation set)

        Returns:
            List of 3 tuples: (start_idx, end_idx, trading_end_idx)
        """
        # NOTE: start_idx is the first day of TRADING (not context)
        # The environment automatically looks back 504 days from start_idx for context
        # So we just need to ensure trading + settlement fit within interim validation set

        # Trading must start at or after interim_val_start_idx
        min_start = self.interim_val_start_idx

        # Trading + settlement must end before interim_val_end_idx
        max_start = self.interim_val_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start < min_start:
            raise ValueError(f"Not enough interim validation data: need {Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS} days")

        # Divide the validation range into 3 equal segments
        total_range = max_start - min_start + 1
        segment_size = total_range // 3

        slices = []
        for segment_idx in range(3):
            # Calculate segment boundaries
            segment_start = min_start + (segment_idx * segment_size)
            # For the last segment, extend to max_start to avoid rounding issues
            segment_end = max_start + 1 if segment_idx == 2 else min_start + ((segment_idx + 1) * segment_size)

            # Sample one random start index from this segment
            start_idx = np.random.randint(segment_start, segment_end)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

            slices.append((start_idx, end_idx, trading_end_idx))

        return slices

    def evaluate_population(self) -> Tuple[List[float], Dict]:
        """
        Evaluate all agents in population (fitness scores).

        NEW: Multi-slice evaluation for robust fitness signal
        - Each agent is evaluated on 5 different random training slices
        - Final fitness = average of the LOWEST 2 scores (conservative, robust estimate)
        - This prevents "lucky" agents from advancing and selects for consistency

        Returns:
            Tuple of (fitness_scores, aggregate_stats)
        """
        fitness_scores = []
        all_episode_stats = []

        print(f"\n--- Generation {self.generation + 1}: Evaluating Population ---")
        print(f"Multi-slice evaluation: 5 slices per agent, scoring = avg(lowest 2)")

        # Count elite vs exploratory agents for logging
        num_elites = sum(1 for a in self.population if a.is_elite)
        num_exploratory = len(self.population) - num_elites
        print(f"Replay buffer diversity: Only {num_exploratory}/{len(self.population)} exploratory agents contribute experiences")

        for agent in tqdm(self.population, desc="Evaluating agents"):
            # Prepare 5 random training slices for this agent
            episode_configs = []
            total_days_needed = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            max_start = self.train_end_idx - total_days_needed

            if max_start <= self.train_start_idx:
                raise ValueError(f"Not enough training data: need {total_days_needed} days")

            for _ in range(5):
                # Random start from training range only (excludes interim val and holdout)
                start_idx = np.random.randint(self.train_start_idx, max_start)
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
                episode_configs.append((start_idx, end_idx, trading_end_idx))

            # Run all 5 slices in parallel using AsyncVectorEnv (5 CPU cores simultaneously)
            # Only add to replay buffer if agent is exploratory (not elite) for diversity
            results = self.run_vectorized_episodes(
                agent=agent,
                episode_configs=episode_configs,
                training=(not agent.is_elite)
            )

            # Extract fitness scores and episode stats
            slice_fitness_scores = [fitness for fitness, _ in results]
            slice_episode_stats = [info for _, info in results]

            # Robust scoring: average of lowest 2 scores out of 5
            sorted_fitness = sorted(slice_fitness_scores)
            lowest_2_avg = np.mean(sorted_fitness[:2])

            fitness_scores.append(lowest_2_avg)

            # Aggregate episode stats across all 5 slices for this agent
            agent_aggregate_stats = {
                'num_trades': int(np.mean([s['num_trades'] for s in slice_episode_stats])),
                'num_wins': int(np.mean([s['num_wins'] for s in slice_episode_stats])),
                'num_losses': int(np.mean([s['num_losses'] for s in slice_episode_stats])),
                'win_rate': np.mean([s['win_rate'] for s in slice_episode_stats]),
            }
            all_episode_stats.append(agent_aggregate_stats)

        # Ensure fitness_scores are all plain floats
        fitness_scores = [float(f) for f in fitness_scores]

        # Aggregate statistics across all agents
        aggregate_stats = {
            'total_trades': int(sum(s['num_trades'] for s in all_episode_stats)),
            'avg_trades_per_agent': float(sum(s['num_trades'] for s in all_episode_stats) / len(all_episode_stats)),
            'total_wins': int(sum(s['num_wins'] for s in all_episode_stats)),
            'total_losses': int(sum(s['num_losses'] for s in all_episode_stats)),
            'avg_win_rate': float(sum(s['win_rate'] for s in all_episode_stats if s['num_trades'] > 0) / len([s for s in all_episode_stats if s['num_trades'] > 0])) if any(s['num_trades'] > 0 for s in all_episode_stats) else 0.0,
            'agents_with_positive_fitness': int(sum(1 for f in fitness_scores if f > 0)),
        }

        # CRITICAL FIX: Delete large all_episode_stats list and force aggressive GC
        # The all_episode_stats list is no longer needed after aggregation
        del all_episode_stats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (fitness_scores, aggregate_stats)
    
    def train_population(self):
        """Train all agents using shared replay buffer with gradient accumulation.

        Uses asynchronous DataLoader for batch prefetching:
        - Background CPU workers load and decompress batches from disk
        - Batches are queued in RAM before GPU needs them
        - GPU never waits for I/O - significant speedup
        """
        if not self.replay_buffer.is_ready():
            # Show correct threshold based on sweep vs regular training
            is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
            min_size = Config.MIN_BUFFER_SIZE_SWEEP if is_sweep else Config.MIN_BUFFER_SIZE
            print(f"Buffer not ready: {len(self.replay_buffer)} / {min_size}")
            return

        # Initialize batch iterator if not already created
        if self.batch_iterator is None:
            print("Starting DataLoader workers for async batch prefetching...")
            self.batch_iterator = iter(self.replay_dataloader)

        # Check if buffer is full for the first time and we haven't saved it yet
        # DISABLED: Buffer save causes RAM exhaustion during pickle serialization
        # buffer_is_full = (len(self.replay_buffer) == self.replay_buffer.capacity)
        # if buffer_is_full and not self.buffer_saved_on_first_fill:
        #     # Check if buffer exists on GCS
        #     buffer_path = Config.CHECKPOINT_DIR / "replay_buffer.pkl"
        #     buffer_exists_on_cloud = self.cloud_sync.file_exists_on_cloud("replay_buffer.pkl")
        #
        #     if not buffer_exists_on_cloud:
        #         print(f"\n--- Buffer Full for First Time ({len(self.replay_buffer)} transitions) ---")
        #         print("  No buffer found on GCS. Queueing initial buffer save+upload...")
        #         Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        #         # Save+upload in background (non-blocking)
        #         cloud_path = f"{self.cloud_sync.project_name}/checkpoints/replay_buffer.pkl"
        #         self.cloud_sync.save_and_upload_buffer(self.replay_buffer, str(buffer_path), cloud_path)
        #         print("  ✓ Initial buffer save+upload queued")
        #         self.buffer_saved_on_first_fill = True
        #     else:
        #         print("  Buffer exists on GCS. Skipping first-fill save.")
        #         self.buffer_saved_on_first_fill = True

        print(f"\n--- Training Population (Buffer: {len(self.replay_buffer)}) ---")
        
        # Train each agent
        for agent in tqdm(self.population, desc="Training agents"):
            actor_losses = []
            critic_losses = []

            # Multiple gradient steps per agent
            for step in range(Config.GRADIENT_STEPS_PER_GENERATION):
                # Gradient accumulation loop
                for accum_step in range(Config.GRADIENT_ACCUMULATION_STEPS):
                    # Get next batch from DataLoader (already prefetched by workers)
                    # This is FAST - batch is already in RAM, loaded asynchronously
                    batch_cpu = next(self.batch_iterator)

                    # Move batch to GPU (fast transfer thanks to pin_memory)
                    batch = {k: v.to(Config.DEVICE, non_blocking=True) for k, v in batch_cpu.items()}

                    # Update with gradient accumulation
                    is_last_accum = (accum_step == Config.GRADIENT_ACCUMULATION_STEPS - 1)
                    critic_loss, actor_loss = agent.update(batch, accumulate=not is_last_accum)

                    # Capture attention weights from actor (after forward pass in update)
                    # Only capture from first agent to avoid redundant logging
                    if agent.agent_id == 0:
                        attention_weights = agent.actor.get_attention_weights()
                        if attention_weights is not None:
                            self.update_feature_importance(attention_weights)

                    # Detach from computation graph to prevent memory leak
                    actor_losses.append(actor_loss.detach().cpu().item() if isinstance(actor_loss, torch.Tensor) else actor_loss)
                    critic_losses.append(critic_loss.detach().cpu().item() if isinstance(critic_loss, torch.Tensor) else critic_loss)

                    # Explicitly delete batch tensors to free GPU memory
                    del batch

            # Log agent stats
            if agent.agent_id == 0:  # Log first agent as representative
                self.writer.add_scalar('Train/Actor_Loss', np.mean(actor_losses), self.generation)
                self.writer.add_scalar('Train/Critic_Loss', np.mean(critic_losses), self.generation)

                # Log to wandb
                wandb.log({
                    "train/actor_loss": np.mean(actor_losses),
                    "train/critic_loss": np.mean(critic_losses),
                })

            # Explicitly clear loss lists to free memory
            del actor_losses
            del critic_losses

            # Clear GPU cache after each agent to prevent accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def evolve_population(self, fitness_scores: List[float]):
        """
        Evolve population using genetic algorithm.

        Args:
            fitness_scores: Fitness for each agent
        """
        print(f"\n--- Evolving Population (mutation: {self.current_mutation_rate:.3f}) ---")

        # CRITICAL FIX: Store old population reference before creating new one
        # This prevents memory leak from lingering agent references (~2-3GB per generation)
        old_population = self.population

        # Create next generation with adaptive mutation parameters
        # Elitism is handled by create_next_generation (keeps top ELITE_FRAC agents)
        self.population = create_next_generation(
            old_population,
            fitness_scores,
            mutation_rate=self.current_mutation_rate,
            mutation_std=self.current_mutation_std
        )

        # Explicitly delete old agents and force GC
        for agent in old_population:
            del agent
        del old_population
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def validate_agent(self, agent) -> Dict:
        """
        Validate agent using walk-forward validation on 3 random slices.

        Walk-forward validation strategy:
        - Runs agent on 3 validation slices (same slices for all agents in this generation)
        - Takes the lowest 2 fitness scores out of 3
        - Returns the average of those 2 scores as "true validation fitness"
        - This provides a conservative estimate that's robust to lucky runs

        Args:
            agent: The agent to validate

        Returns:
            Validation results with 'fitness' being the average of lowest 2 scores
        """
        if agent is None:
            return {}

        if not self.current_generation_val_slices:
            raise ValueError("No validation slices generated for this generation")

        # Run all 3 validation slices in parallel using AsyncVectorEnv (3 CPU cores simultaneously)
        results = self.run_vectorized_episodes(
            agent=agent,
            episode_configs=self.current_generation_val_slices,
            training=False
        )

        # Process results
        slice_results = []
        all_closed_trades = []  # Collect all closed trades from all slices

        for fitness, episode_info in results:
            # CRITICAL FIX: Create validation gradient for agents that don't trade
            # If agent made 0 trades, add max_coefficient bonus to reduce penalty
            # This creates a gradient that rewards agents who get "closer" to the threshold
            if episode_info['num_trades'] == 0:
                max_coeff = episode_info.get('max_coefficient_during_episode', 0.0)
                # Add max_coefficient as a bonus (reduces the harsh zero-trades penalty slightly)
                # Agent with max_coeff=0.9 gets better score than agent with max_coeff=0.2
                # The bonus is small relative to ZERO_TRADES_PENALTY, but creates gradient
                fitness = fitness + max_coeff

            slice_results.append({
                'fitness': fitness,
                'win_rate': episode_info['win_rate'],
                'num_trades': episode_info['num_trades'],
                'num_wins': episode_info['num_wins'],
                'num_losses': episode_info['num_losses'],
                'avg_reward_per_trade': episode_info['avg_reward_per_trade']
            })

            # Collect closed trades from this slice
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        # Extract fitness scores from all 3 slices
        fitness_scores = [result['fitness'] for result in slice_results]

        # Sort and take lowest 2 scores (conservative estimate)
        sorted_fitness = sorted(fitness_scores)
        lowest_2_avg = np.mean(sorted_fitness[:2])

        # Select one sample trade (first trade from all validation slices, if any)
        sample_trade = all_closed_trades[0] if all_closed_trades else None

        # Return aggregated results (average of lowest 2 fitness scores)
        # Also aggregate other metrics for logging
        return {
            'fitness': lowest_2_avg,
            'fitness_all_slices': fitness_scores,  # For debugging
            'win_rate': np.mean([r['win_rate'] for r in slice_results]),
            'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
            'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
            'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
            'avg_reward_per_trade': np.mean([r['avg_reward_per_trade'] for r in slice_results]),
            'sample_trade': sample_trade  # One sample trade for verification
        }

    def _run_evaluation(self):
        """
        Run evaluate_best_agent.py as a subprocess to generate detailed trade report.
        This always overwrites the previous evaluation results.
        """
        import subprocess

        try:
            # Run evaluation script with current run name
            result = subprocess.run(
                ['python', 'evaluate_best_agent.py', '--run-name', self.run_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print("  ✓ Evaluation complete - results saved to evaluation_summary.txt and evaluation_trades.csv")
            else:
                print(f"  ⚠ Evaluation script failed with code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")  # Print first 200 chars of error
        except subprocess.TimeoutExpired:
            print("  ⚠ Evaluation script timed out after 5 minutes")
        except Exception as e:
            print(f"  ⚠ Could not run evaluation: {e}")

    def validate_best_agent(self) -> Dict:
        """
        Validate best agent on validation set.

        Returns:
            Validation results
        """
        if self.best_agent is None:
            return {}

        val_results = self.validate_agent(self.best_agent)
        return val_results

    def check_and_adjust_mutation(self, current_val_fitness: float):
        """
        Check for fitness plateau and adaptively increase mutation parameters.
        Plateau is detected if validation fitness doesn't improve by plateau_threshold
        over the last plateau_window generations.
        """
        # Add current fitness to history
        self.validation_fitness_history.append(current_val_fitness)

        # Need at least plateau_window generations to detect plateau
        if len(self.validation_fitness_history) < self.plateau_window:
            return

        # Get fitness values from the plateau window
        recent_fitness = self.validation_fitness_history[-self.plateau_window:]

        # Calculate improvement: (max - min) / |min| to get relative improvement
        min_fitness = min(recent_fitness)
        max_fitness = max(recent_fitness)

        # Handle negative fitness values properly
        if min_fitness == 0:
            relative_improvement = float('inf') if max_fitness > 0 else 0
        else:
            relative_improvement = abs((max_fitness - min_fitness) / min_fitness)

        # Check if we're in a plateau (improvement below threshold)
        if relative_improvement < self.plateau_threshold:
            if not self.plateau_detected:
                # First time detecting plateau - boost mutation
                self.plateau_detected = True

                # Increase mutation parameters
                self.current_mutation_rate = min(
                    self.current_mutation_rate * self.mutation_boost_factor,
                    self.max_mutation_rate
                )
                self.current_mutation_std = min(
                    self.current_mutation_std * self.mutation_boost_factor,
                    self.max_mutation_std
                )

                print(f"\n{'='*60}")
                print(f"🔄 PLATEAU DETECTED - Adaptive Mutation Activated")
                print(f"{'='*60}")
                print(f"Validation fitness improvement over last {self.plateau_window} gens: {relative_improvement:.2%}")
                print(f"Threshold: {self.plateau_threshold:.2%}")
                print(f"\nBoosting mutation parameters:")
                print(f"  Mutation Rate: {self.base_mutation_rate:.3f} → {self.current_mutation_rate:.3f}")
                print(f"  Mutation STD:  {self.base_mutation_std:.4f} → {self.current_mutation_std:.4f}")
                print(f"{'='*60}\n")

                # Log to wandb
                wandb.log({
                    "adaptive_mutation/plateau_detected": 1,
                    "adaptive_mutation/mutation_rate": self.current_mutation_rate,
                    "adaptive_mutation/mutation_std": self.current_mutation_std,
                    "adaptive_mutation/relative_improvement": relative_improvement,
                })
            else:
                # Already in plateau - just log current status
                wandb.log({
                    "adaptive_mutation/plateau_detected": 1,
                    "adaptive_mutation/mutation_rate": self.current_mutation_rate,
                    "adaptive_mutation/mutation_std": self.current_mutation_std,
                    "adaptive_mutation/relative_improvement": relative_improvement,
                })
        else:
            # Improvement detected - reset to base if we were in plateau
            if self.plateau_detected:
                self.plateau_detected = False
                self.current_mutation_rate = self.base_mutation_rate
                self.current_mutation_std = self.base_mutation_std

                print(f"\n{'='*60}")
                print(f"✓ FITNESS IMPROVING - Resetting Mutation to Baseline")
                print(f"{'='*60}")
                print(f"Validation fitness improvement: {relative_improvement:.2%}")
                print(f"Mutation Rate: {self.current_mutation_rate:.3f}")
                print(f"Mutation STD:  {self.current_mutation_std:.4f}")
                print(f"{'='*60}\n")

                wandb.log({
                    "adaptive_mutation/plateau_detected": 0,
                    "adaptive_mutation/mutation_rate": self.current_mutation_rate,
                    "adaptive_mutation/mutation_std": self.current_mutation_std,
                    "adaptive_mutation/relative_improvement": relative_improvement,
                })
            else:
                # Not in plateau, just log status
                wandb.log({
                    "adaptive_mutation/plateau_detected": 0,
                    "adaptive_mutation/mutation_rate": self.current_mutation_rate,
                    "adaptive_mutation/mutation_std": self.current_mutation_std,
                    "adaptive_mutation/relative_improvement": relative_improvement,
                })

    def save_checkpoint(self):
        """Saves the entire training state to a checkpoint directory."""
        checkpoint_dir = self.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Saving Checkpoint (Gen {self.generation + 1}) ---")

        # 1. Save best agent
        if self.best_agent is not None:
            best_path = checkpoint_dir / "best_agent.pth"
            self.best_agent.save(str(best_path))
        
        # 2. Save population
        pop_dir = checkpoint_dir / "population"
        pop_dir.mkdir(exist_ok=True)
        for agent in self.population:
            agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
            agent.save(str(agent_path))
        
        # 3. Save the replay buffer every 5 generations (overwrites previous)
        if (self.generation + 1) % 5 == 0:
            buffer_path = checkpoint_dir / "replay_buffer.pkl"
            print(f"  Saving on-disk buffer metadata ({len(self.replay_buffer)} paths)...")
            self.replay_buffer.save(str(buffer_path))
            # The main cloud_sync (line 1175) will now pick up this small file.

        # 4. Save the trainer state
        trainer_state = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,  # Best training fitness
            'best_validation_fitness': self.best_validation_fitness,  # Best validation fitness
            'validation_fitness_history': self.validation_fitness_history,
            'current_mutation_rate': self.current_mutation_rate,
            'current_mutation_std': self.current_mutation_std,
            'plateau_detected': self.plateau_detected,
            'wandb_run_id': wandb.run.id,
            'wandb_run_name': wandb.run.name
        }
        state_path = checkpoint_dir / "trainer_state.json"
        with open(state_path, 'w') as f:
            json.dump(trainer_state, f, indent=4)

        # Sync to cloud storage in background (non-blocking)
        self.cloud_sync.sync_checkpoints(str(checkpoint_dir), background=True,
                                        exclude_patterns=["replay_buffer"])
        print(f"✓ Saved & syncing to cloud")

    def load_checkpoint(self):
        """Loads the entire training state from the checkpoint directory."""
        checkpoint_dir = self.checkpoint_dir
        print(f"\n--- Loading Checkpoint ---")

        # Try to download from cloud first
        if not checkpoint_dir.exists() or len(list(checkpoint_dir.glob('*'))) == 0:
            print("Downloading from cloud...")
            self.cloud_sync.download_checkpoints(str(checkpoint_dir))

        # 1. Check if directory exists
        if not checkpoint_dir.exists():
            print("! No checkpoint directory found. Starting new training.")
            return

        # 2. Load Population
        pop_dir = checkpoint_dir / "population"
        if pop_dir.exists():
            try:
                for i, agent in enumerate(self.population):
                    agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                    if agent_path.exists():
                        agent.load(str(agent_path))
                print(f"✓ Loaded {len(self.population)} agents")
            except Exception as e:
                print(f"❌ Error loading agents: {e}")

        # 3. Load Best Agent
        best_agent_path = checkpoint_dir / "best_agent.pth"
        if best_agent_path.exists():
            try:
                self.best_agent = DDPGAgent(agent_id='best')
                self.best_agent.load(str(best_agent_path))
                print("✓ Loaded best agent")
            except Exception as e:
                print(f"❌ Error loading best agent: {e}")

        # 4. Replay buffer: Load from checkpoint (both metadata and files)
        # Buffer storage is now INSIDE checkpoint dir, so it gets synced automatically
        buffer_path = checkpoint_dir / "replay_buffer.pkl"
        buffer_storage_path = str(checkpoint_dir / "buffer_storage")
        buffer_loaded = False

        if buffer_path.exists():
            try:
                # Load buffer with correct storage path (inside checkpoint dir)
                self.replay_buffer = OnDiskReplayBuffer.load(
                    str(buffer_path),
                    storage_path_override=buffer_storage_path
                )
                print(f"✓ Loaded on-disk replay buffer metadata ({len(self.replay_buffer)} transitions)")
                buffer_loaded = True
            except Exception as e:
                print(f"❌ Error loading buffer: {e}")
                print("  Creating new empty buffer...")
                self.replay_buffer = OnDiskReplayBuffer(
                    capacity=Config.BUFFER_SIZE,
                    storage_path=buffer_storage_path
                )
                buffer_loaded = True
        else:
            print("! No replay buffer checkpoint found.")
            print(f"  Buffer will be in: {buffer_storage_path}")
            # Buffer was already created in __init__ with correct path
            # No action needed

        # CRITICAL: Recreate DataLoader after loading buffer
        # The DataLoader created in __init__ points to the old empty buffer
        # We must recreate it to point to the loaded buffer
        if buffer_loaded:
            print("Recreating DataLoader for loaded buffer...")
            self._create_dataloader()
            print("✓ DataLoader recreated and ready")
        
        # 5. Load Trainer State
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    trainer_state = json.load(f)

                # Load basic training state
                self.start_generation = trainer_state.get('generation', 0) + 1
                self.best_fitness = trainer_state.get('best_fitness', float('-inf'))
                self.best_validation_fitness = trainer_state.get('best_validation_fitness', float('-inf'))

                # Load adaptive mutation state
                self.validation_fitness_history = trainer_state.get('validation_fitness_history', [])
                self.current_mutation_rate = trainer_state.get('current_mutation_rate', Config.MUTATION_RATE)
                self.current_mutation_std = trainer_state.get('current_mutation_std', Config.MUTATION_STD)
                self.plateau_detected = trainer_state.get('plateau_detected', False)

                print(f"✓ Resuming from Gen {self.start_generation} → Gen {self.start_generation + 1}")
                print(f"✓ Best validation fitness: {self.best_validation_fitness:.2f}")
            except Exception as e:
                print(f"❌ Error loading state: {e}")

    def update_feature_importance(self, attention_weights: torch.Tensor):
        """
        Update the running average of feature importance using exponential moving average.

        Args:
            attention_weights: [batch, num_columns] attention weights from the actor
        """
        if attention_weights is None:
            return

        # Average across batch dimension to get [num_columns]
        batch_mean = attention_weights.mean(dim=0)

        # Update running average with momentum
        if self.feature_importance_count == 0:
            # First update: initialize with current weights
            self.feature_importance = batch_mean.detach()
        else:
            # EMA update: avg = momentum * avg + (1 - momentum) * new
            momentum = self.feature_importance_momentum
            self.feature_importance = (momentum * self.feature_importance +
                                       (1 - momentum) * batch_mean.detach())

        self.feature_importance_count += 1

    def get_feature_importance(self) -> torch.Tensor:
        """
        Get the current feature importance vector.

        Returns:
            Feature importance [num_columns] - probabilities summing to ~1.0
        """
        return self.feature_importance.cpu()

    def analyze_persistent_low_importance_columns(self, current_importance: np.ndarray) -> dict:
        """
        Analyze columns that have been persistently below importance thresholds.

        Args:
            current_importance: Current feature importance [num_columns]

        Returns:
            Dictionary with analysis results
        """
        # Add current importance to history
        self.feature_importance_history.append(current_importance.copy())

        # Need enough history to analyze persistence
        history_len = len(self.feature_importance_history)

        # Thresholds (as fractions, e.g., 0.01 = 1%)
        thresholds = {
            '<1%': (0.01, 5),   # Less than 1% for past 5 generations
            '<0.1%': (0.001, 4), # Less than 0.1% for past 4 generations
            '<0.01%': (0.0001, 3) # Less than 0.01% for past 3 generations
        }

        results = {}

        for threshold_name, (threshold_val, required_gens) in thresholds.items():
            if history_len >= required_gens:
                # Get last N generations
                recent_history = list(self.feature_importance_history)[-required_gens:]

                # Find columns below threshold in ALL recent generations
                persistent_low = np.ones(Config.TOTAL_COLUMNS, dtype=bool)
                for gen_importance in recent_history:
                    persistent_low &= (gen_importance < threshold_val)

                # Get indices of persistent low-importance columns
                persistent_indices = np.where(persistent_low)[0]

                # Create list with both index and column name
                persistent_cols = []
                for idx in persistent_indices:
                    col_name = self.column_names[idx] if idx < len(self.column_names) else f"col_{idx}"
                    persistent_cols.append({
                        'index': int(idx),
                        'name': str(col_name),
                        'current_importance': float(current_importance[idx])
                    })

                results[threshold_name] = {
                    'count': len(persistent_cols),
                    'columns': persistent_cols
                }
            else:
                # Not enough history yet
                results[threshold_name] = {
                    'count': 0,
                    'columns': [],
                    'note': f'Need {required_gens} generations, currently at {history_len}'
                }

        # Count features below thresholds in current generation
        results['current_below_1pct'] = int(np.sum(current_importance < 0.01))
        results['current_below_0.1pct'] = int(np.sum(current_importance < 0.001))
        results['current_below_0.01pct'] = int(np.sum(current_importance < 0.0001))

        return results

    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting ERL Training")
        print("="*60)
        
        # Use start_generation for the loop
        for gen in range(self.start_generation, Config.NUM_GENERATIONS):
            self.generation = gen  # Keep this to track the *current* gen
            gen_start_time = time.time()

            # Reset peak memory stats for this generation
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            print(f"\n{'='*60}")
            print(f"Generation {gen + 1} / {Config.NUM_GENERATIONS}")
            print(f"Buffer: {len(self.replay_buffer)} / {self.replay_buffer.capacity} ({len(self.replay_buffer)/self.replay_buffer.capacity*100:.1f}%)")
            print(f"{'='*60}")

            # 1. Evaluate population (collect experiences)
            fitness_scores, pop_stats = self.evaluate_population()

            # Update resource tracker after evaluation
            self.resource_tracker.update()

            # 🔍 Memory tracking after evaluation
            # log_memory(f"Gen {gen+1}: After evaluate_population", show_objects=True)

            # Track statistics
            self.fitness_history.append(fitness_scores)
            mean_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            
            print(f"\nFitness statistics:")
            print(f"  Mean: {mean_fitness:.2f}")
            print(f"  Max: {max_fitness:.2f}")
            print(f"  Min: {min_fitness:.2f}")
            print(f"  Std: {np.std(fitness_scores):.2f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Fitness/Mean', mean_fitness, gen)
            self.writer.add_scalar('Fitness/Max', max_fitness, gen)
            self.writer.add_scalar('Fitness/Min', min_fitness, gen)
            self.writer.add_scalar('Fitness/Std', np.std(fitness_scores), gen)

            # Log to wandb
            wandb.log({
                "generation": gen,
                "fitness/mean": mean_fitness,
                "fitness/max": max_fitness,
                "fitness/min": min_fitness,
                "fitness/std": np.std(fitness_scores),
                "fitness/best_training_ever": self.best_fitness if hasattr(self, 'best_fitness') else max_fitness,
                "fitness/best_validation_ever": self.best_validation_fitness if hasattr(self, 'best_validation_fitness') else float('-inf'),
            })

            # Update best training fitness (for logging only)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness

            # Generate validation slices for this generation
            print(f"\n--- Walk-Forward Validation (Generation {gen + 1}) ---")
            self.current_generation_val_slices = self.generate_validation_slices()
            print(f"Generated 3 validation slices:")
            for i, (start, end, _) in enumerate(self.current_generation_val_slices, 1):
                start_date = self.data_loader.dates[start] if start < len(self.data_loader.dates) else "N/A"
                end_date = self.data_loader.dates[end-1] if end-1 < len(self.data_loader.dates) else "N/A"
                print(f"  Slice {i}: days {start}-{end}: {start_date} to {end_date}")

            best_val_fitness_this_gen = float('-inf')
            best_val_agent_idx = None
            validation_results = []

            for idx in tqdm(range(len(self.population)), desc="Validating agents"):
                val_results = self.validate_agent(self.population[idx])
                val_fitness = val_results['fitness']

                validation_results.append({
                    'idx': idx,
                    'training_fitness': fitness_scores[idx],
                    'validation_fitness': val_fitness,
                    'win_rate': val_results['win_rate'],
                    'num_trades': val_results['num_trades']
                })

                # Track best validation fitness in this generation
                if val_fitness > best_val_fitness_this_gen:
                    best_val_fitness_this_gen = val_fitness
                    best_val_agent_idx = idx

            # Print summary showing training vs validation rankings
            print(f"\n--- Validation Summary ---")
            validation_results.sort(key=lambda x: x['validation_fitness'], reverse=True)
            print("Top 5 by Validation Fitness:")
            for i, result in enumerate(validation_results[:5]):
                print(f"  {i+1}. Agent {result['idx']:2d}: Val={result['validation_fitness']:>8.2f}, Train={result['training_fitness']:>8.2f}, WR={result['win_rate']:.1%}")

            # Update best agent if we found a better one based on validation
            if best_val_agent_idx is not None and best_val_fitness_this_gen > self.best_validation_fitness:
                print(f"\n✓ New best! Val fitness: {best_val_fitness_this_gen:.2f} (prev: {self.best_validation_fitness:.2f})")
                if self.best_agent is not None:
                    del self.best_agent
                    gc.collect()
                self.best_validation_fitness = best_val_fitness_this_gen
                self.best_agent = self.population[best_val_agent_idx].clone()

                # Save checkpoint immediately to update best_agent.pth
                print("  Saving checkpoint with new best agent...")
                self.save_checkpoint()

                # Run full evaluation on best agent
                print("  Running full evaluation (evaluate_best_agent.py)...")
                self._run_evaluation()
            else:
                print(f"\n→ Best val fitness unchanged: {self.best_validation_fitness:.2f}")
            
            # 2. Train agents using replay buffer
            self.train_population()

            # Update resource tracker after training
            self.resource_tracker.update()

            # 🔍 Memory tracking after training
            # log_memory(f"Gen {gen+1}: After train_population", show_objects=True)

            # 2b. Log feature importance (computed during training)
            # Get feature importance for logging
            feature_importance = self.get_feature_importance().numpy()

            # Calculate summary statistics for feature importance
            top_k = 20  # Top 20 most important features
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_values = feature_importance[top_indices]

            # Calculate entropy of feature importance (higher = more distributed attention)
            # Clip to avoid log(0)
            fi_clipped = np.clip(feature_importance, 1e-10, 1.0)
            entropy = -np.sum(fi_clipped * np.log(fi_clipped))

            # Analyze persistent low-importance columns
            low_importance_analysis = self.analyze_persistent_low_importance_columns(feature_importance)

            # Log feature importance to wandb
            wandb.log({
                # Feature importance summary
                "feature_importance/entropy": entropy,
                "feature_importance/max": float(feature_importance.max()),
                "feature_importance/mean": float(feature_importance.mean()),
                "feature_importance/top_1": float(top_values[0]) if len(top_values) > 0 else 0.0,
                "feature_importance/top_5_sum": float(top_values[:5].sum()) if len(top_values) >= 5 else 0.0,
                "feature_importance/update_count": self.feature_importance_count,
                # Current generation thresholds
                "feature_importance/current_below_1pct": low_importance_analysis['current_below_1pct'],
                "feature_importance/current_below_0.1pct": low_importance_analysis['current_below_0.1pct'],
                "feature_importance/current_below_0.01pct": low_importance_analysis['current_below_0.01pct'],
                # Persistent low-importance counts
                "feature_importance/persistent_below_1pct_count": low_importance_analysis['<1%']['count'],
                "feature_importance/persistent_below_0.1pct_count": low_importance_analysis['<0.1%']['count'],
                "feature_importance/persistent_below_0.01pct_count": low_importance_analysis['<0.01%']['count'],
            })

            # Log full feature importance vector as histogram every 5 generations
            if (gen + 1) % 5 == 0:
                wandb.log({
                    "feature_importance/histogram": wandb.Histogram(feature_importance),
                    "feature_importance/top_20_columns": wandb.Table(
                        data=[[int(idx), float(val)] for idx, val in zip(top_indices, top_values)],
                        columns=["Column_Index", "Importance"]
                    )
                })

                # Log persistent low-importance columns as tables
                for threshold_name in ['<1%', '<0.1%', '<0.01%']:
                    threshold_data = low_importance_analysis[threshold_name]
                    if threshold_data['count'] > 0 and 'columns' in threshold_data:
                        table_data = [[col['index'], col['name'], col['current_importance']]
                                     for col in threshold_data['columns']]
                        wandb.log({
                            f"feature_importance/persistent_{threshold_name.replace('<', 'below_').replace('%', 'pct')}": wandb.Table(
                                data=table_data,
                                columns=["Column_Index", "Column_Name", "Current_Importance"]
                            )
                        })

            # 3. Evolve population
            self.evolve_population(fitness_scores)

            # Update resource tracker after evolution
            self.resource_tracker.update()

            # 🔍 Memory tracking after evolution
            # log_memory(f"Gen {gen+1}: After evolve_population", show_objects=True)
            
            # 4. Log validation metrics and check for plateau
            if (gen + 1) % Config.LOG_FREQUENCY == 0:
                # We already validated the best agent during selection, so use those results
                # Re-validate the current best_agent for logging
                if self.best_agent is not None:
                    val_results = self.validate_best_agent()
                    if val_results:
                        self.writer.add_scalar('Validation/Fitness', val_results['fitness'], gen)
                        self.writer.add_scalar('Validation/WinRate', val_results['win_rate'], gen)

                        # Log to wandb
                        wandb.log({
                            "validation/fitness": val_results['fitness'],
                            "validation/win_rate": val_results['win_rate'],
                            "validation/num_trades": val_results['num_trades'],
                            "validation/num_wins": val_results['num_wins'],
                            "validation/num_losses": val_results['num_losses'],
                            "validation/avg_reward_per_trade": val_results['avg_reward_per_trade'],
                        })

                        # Check for plateau and adjust mutation adaptively
                        # Use the best validation fitness for plateau detection
                        self.check_and_adjust_mutation(self.best_validation_fitness)

            # 5. Save checkpoint periodically
            if (gen + 1) % Config.SAVE_FREQUENCY == 0:
                self.save_checkpoint()
            
            # Generation time
            gen_time = time.time() - gen_start_time
            self.generation_times.append(gen_time)

            # Get final resource stats for this generation
            self.resource_tracker.update()
            resource_stats = self.resource_tracker.get_current_stats()

            # Print comprehensive generation summary with resource stats
            print_generation_summary(
                gen=gen,
                total_gens=Config.NUM_GENERATIONS,
                fitness_scores=fitness_scores,
                pop_stats=pop_stats,
                buffer_size=len(self.replay_buffer),
                best_fitness=self.best_validation_fitness,  # Use validation fitness for "best ever"
                gen_time=gen_time,
                avg_gen_time=np.mean(self.generation_times) if self.generation_times else 0,
                resource_stats=resource_stats
            )

            # Show progress plot every 5 generations
            if (gen + 1) % 5 == 0:
                plot_fitness_progress(self.fitness_history)
                # CRITICAL FIX: Close matplotlib figures to prevent memory leak (~100MB per plot)
                plt.close('all')

            # Buffer stats and generation time
            buffer_stats = self.replay_buffer.get_stats()
            self.writer.add_scalar('Buffer/Size', buffer_stats['size'], gen)
            self.writer.add_scalar('Buffer/Utilization', buffer_stats['utilization'], gen)

            # Log buffer stats, timing, and resource usage to wandb
            wandb.log({
                "buffer/size": buffer_stats['size'],
                "buffer/utilization": buffer_stats['utilization'],
                "buffer/capacity": buffer_stats['capacity'],
                "training/generation_time": gen_time,
                "training/avg_generation_time": np.mean(self.generation_times) if self.generation_times else 0,
                "resources/peak_vram_gb": resource_stats['peak_vram_gb'],
                "resources/peak_ram_gb": resource_stats['peak_ram_gb'],
                "resources/peak_disk_gb": resource_stats['peak_disk_gb'],
            })

            # Clear GPU cache and run garbage collection to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # 🔍 Print memory trend every generation
            # if (gen + 1) % 1 == 0:  # Every generation
            #     profiler = get_profiler()
            #     profiler.print_generation_trend()
            #     profiler.print_memory_growth(baseline_label="Trainer initialized (baseline)")

        # Final save
        self.save_checkpoint()

        # Sync logs to cloud in background
        self.cloud_sync.sync_logs(str(Config.LOG_DIR), background=True)

        # Wait for all uploads to complete before finishing
        print("\n" + "="*60)
        print("Training complete! Waiting for final uploads...")
        print("="*60)
        self.cloud_sync.wait_for_uploads()

        # Shutdown cloud sync
        self.cloud_sync.shutdown(wait=False)

        # Finish wandb run
        wandb.finish()

        # 🔍 Final comprehensive memory analysis
        # print("\n" + "="*70)
        # print("🔍 FINAL MEMORY ANALYSIS")
        # print("="*70)
        # from utils.memory_profiler import print_memory_summary
        # print_memory_summary()

        # Final summary
        print_final_summary(self)

        self.writer.close()


# Main execution
if __name__ == "__main__":
    print("Initializing Project Eigen 2 Training...\n")
    
    # Load data
    print("Loading data...")
    loader = StockDataLoader()
    data_array, stats = loader.load_and_prepare()
    
    # Create trainer
    trainer = ERLTrainer(loader)
    
    # Start training
    trainer.train()
    
    print("\n✓ Training complete!")