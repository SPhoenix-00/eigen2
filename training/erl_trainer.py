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
import sys
import json
import wandb
import gc
import math
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Suppress common library warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda.amp')

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from models.replay_buffer import ReplayBuffer, OnDiskReplayBuffer
from erl.genetic_ops import create_next_generation
from erl.hall_of_fame import HallOfFame
from utils.config import Config
from utils.display import print_generation_summary, print_final_summary, plot_fitness_progress, ResourceTracker
from utils.cloud_sync import get_cloud_sync_from_env
from utils.cleanup_orphans import cleanup_orphans
from torch.utils.data import DataLoader
# from utils.memory_profiler import get_profiler, log_memory  # Memory profiling disabled


# Global variable to store shared env_config in worker processes
_worker_env_config = None


def _init_worker(env_config):
    """
    Initializer function for worker processes.

    This is called once per worker when the ProcessPoolExecutor starts.
    Stores the env_config in a global variable so it doesn't need to be
    pickled with every task (significant performance improvement).

    Args:
        env_config: Environment configuration dict with data arrays
    """
    global _worker_env_config
    _worker_env_config = env_config


def _run_episode_worker(args):
    """
    Worker function for parallel episode execution.

    This function runs in a separate process, so it must:
    1. Reconstruct the agent from CPU state dicts
    2. Create its own environment instance (using global env_config)
    3. Run the episode independently
    4. Write transitions directly to disk (parallel I/O)

    Args:
        args: Tuple of (agent_state, start_idx, end_idx, training, seed, buffer_storage_path, file_id_start)
              Note: env_config is accessed from global _worker_env_config (set by initializer)

    Returns:
        Tuple of (fitness, episode_info, transition_file_paths)
    """
    global _worker_env_config
    agent_state, start_idx, end_idx, training, seed, buffer_storage_path, file_id_start = args
    env_config = _worker_env_config

    # Set worker-specific seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reconstruct agent from state dict (agents with CUDA tensors are not picklable)
    from models.ddpg_agent import DDPGAgent
    agent = DDPGAgent(agent_id=0)
    agent.actor.load_state_dict(agent_state['actor'])
    agent.critic.load_state_dict(agent_state['critic'])
    agent.actor.eval()
    agent.critic.eval()

    # Create environment for this worker
    from environment.trading_env import TradingEnvironment
    env = TradingEnvironment(**env_config)

    # Run episode
    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
    env.set_training_mode(training)
    state, _ = env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

    cumulative_reward = 0.0
    steps = 0
    transition_file_paths = []

    # Write transitions directly to disk during episode (parallel I/O)
    # Note: We save to buffer regardless of training flag (noise) - this enables "Teacher Forcing"
    # where Elites contribute high-quality positive-reward examples to help the Critic learn
    if buffer_storage_path:
        from pathlib import Path
        import pickle
        import gzip

        storage_path = Path(buffer_storage_path)
        file_id = file_id_start

        while True:
            # Select action
            action = agent.select_action(state, add_noise=training)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)

            # Write transition directly to disk (parallel I/O across all workers)
            transition = {
                'state': state.astype(np.float32),
                'action': action.astype(np.float32),
                'reward': reward,
                'next_state': next_state.astype(np.float32),
                'done': float(terminated or truncated)
            }

            file_path = storage_path / f"transition_{file_id}.pkl.gz"
            try:
                # Check if file already exists (can happen if resuming after crash mid-generation)
                if file_path.exists():
                    # Skip to next ID to avoid overwriting
                    file_id += 1
                    file_path = storage_path / f"transition_{file_id}.pkl.gz"

                with gzip.open(file_path, 'wb', compresslevel=1) as f:
                    pickle.dump(transition, f, protocol=pickle.HIGHEST_PROTOCOL)
                transition_file_paths.append(str(file_path))
                file_id += 1
            except Exception as e:
                print(f"⚠ Worker failed to write transition {file_path}: {e}")

            cumulative_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break
    else:
        # No training mode or no buffer - just run episode without saving transitions
        while True:
            action = agent.select_action(state, add_noise=training)
            next_state, reward, terminated, truncated, info = env.step(action)

            cumulative_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

    # Get episode summary
    episode_summary = env.get_episode_summary()
    episode_summary['steps'] = steps

    # Calculate final fitness
    final_fitness = float(cumulative_reward)
    # Apply zero trades penalty from episode summary (mode-specific)
    if episode_summary['num_trades'] == 0:
        final_fitness -= episode_summary['zero_trades_penalty']

    # Apply win rate bonus if enough trades and win rate above threshold
    if episode_summary['num_trades'] >= Config.WIN_RATE_BONUS_MIN_TRADES:
        win_rate_pct = episode_summary['win_rate'] * 100.0  # Convert to percentage
        if win_rate_pct > Config.WIN_RATE_BONUS_THRESHOLD:
            bonus = (win_rate_pct - Config.WIN_RATE_BONUS_THRESHOLD) ** 2
            final_fitness += bonus
            episode_summary['win_rate_bonus'] = bonus
        else:
            episode_summary['win_rate_bonus'] = 0.0
    else:
        episode_summary['win_rate_bonus'] = 0.0

    return final_fitness, episode_summary, transition_file_paths


class ERLTrainer:
    """
    Evolutionary Reinforcement Learning Trainer.
    Manages population, training, and evolution.
    """

    def __init__(self, data_loader: StockDataLoader, resume_run_name: str = None, enable_leverage: bool = False,
                 consistency_mode: bool = False, heroes_hof_dir: str = None, original_stdout=None, original_stderr=None):
        """
        Initialize ERL trainer.

        Args:
            data_loader: Loaded data with train/val splits
            resume_run_name: Optional wandb run name to resume from (e.g., "azure-thunder-123")
            enable_leverage: If True, enable leverage mode (replaces bottom 5 with top 5 HoF agents with 1.5x coefficients)
            consistency_mode: If True, evaluate with 5 episodes (sum) and loss magnification (see Config.CONSISTENCY_LOSS_MULTIPLIER)
            heroes_hof_dir: Path to Hall of Fame directory to load pre-trained agents from
            original_stdout: Original stdout before any redirection (for wandb console capture)
            original_stderr: Original stderr before any redirection (for wandb console capture)
        """
        self.data_loader = data_loader
        self.resume_run_name = resume_run_name
        self.enable_leverage = enable_leverage
        self.consistency_mode = consistency_mode
        self.heroes_hof_dir = heroes_hof_dir

        # Leverage mode tracking
        self.leverage_mode_active = False
        self.leverage_generations_remaining = 0
        self.leverage_total_generations = 5  # Run leverage mode for 5 generations

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

        # Training range (excludes validation set)
        self.train_start_idx = Config.CONTEXT_WINDOW_DAYS
        self.train_end_idx = len(data_loader.train_indices)

        # Validation range (for walk-forward validation during training)
        # This is separate from training data to ensure genuine out-of-sample validation
        self.val_start_idx = len(data_loader.train_indices)
        self.val_end_idx = self.val_start_idx + len(data_loader.val_indices)

        # Walk-forward validation slices (generated per generation)
        # Each generation uses 7 random validation slices from validation set (4 from quarters + 3 straddling)
        # Format: list of (start_idx, end_idx, trading_end_idx) tuples
        self.current_generation_val_slices = []
        
        # Logging
        self.writer = SummaryWriter(log_dir=str(Config.LOG_DIR))

        # Cloud sync (needed before wandb init for checkpoint downloading)
        self.cloud_sync = get_cloud_sync_from_env()

        # Initialize Weights & Biases and checkpoint directory
        # Note: entity defaults to your personal workspace (eigen2)

        # Temporarily restore original stdout/stderr for wandb initialization
        # This allows wandb to properly set up console output capture
        current_stdout = sys.stdout
        current_stderr = sys.stderr
        if original_stdout is not None:
            sys.stdout = original_stdout
        if original_stderr is not None:
            sys.stderr = original_stderr

        try:
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
                        wandb.init(
                            project="eigen2-self",
                            id=wandb_run_id,
                            resume="must",  # Must resume this specific run
                            config={
                                "population_size": Config.POPULATION_SIZE,
                                "num_generations": Config.NUM_GENERATIONS,
                                "buffer_size": Config.BUFFER_SIZE,
                                "batch_size": Config.BATCH_SIZE,
                                "actor_lr": Config.ACTOR_LR,
                                "critic_lr": Config.CRITIC_LR,
                                "trading_period_days": Config.TRADING_PERIOD_DAYS,
                                "max_holding_period": Config.MAX_HOLDING_PERIOD,
                                "loss_penalty_multiplier": Config.CONSISTENCY_LOSS_MULTIPLIER if self.consistency_mode else 1.0,
                                "consistency_mode": self.consistency_mode,
                                "num_stocks": Config.NUM_INVESTABLE_STOCKS,
                            },
                            settings=wandb.Settings(console="wrap")
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
                                "loss_penalty_multiplier": Config.CONSISTENCY_LOSS_MULTIPLIER if self.consistency_mode else 1.0,
                                "consistency_mode": self.consistency_mode,
                                "num_stocks": Config.NUM_INVESTABLE_STOCKS,
                            },
                            settings=wandb.Settings(console="wrap")
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
                            "loss_penalty_multiplier": Config.CONSISTENCY_LOSS_MULTIPLIER if self.consistency_mode else 1.0,
                            "consistency_mode": self.consistency_mode,
                            "num_stocks": Config.NUM_INVESTABLE_STOCKS,
                        },
                        resume="allow",  # Allow resuming from checkpoints
                        settings=wandb.Settings(console="wrap")
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
        finally:
            # Restore TeeLogger for stdout/stderr
            sys.stdout = current_stdout
            sys.stderr = current_stderr

        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"W&B run: {wandb.run.name} (ID: {wandb.run.id})")

        # Initialize Hall of Fame (now that checkpoint_dir is set)
        print("Initializing Hall of Fame (capacity: 10)...")
        self.hall_of_fame = HallOfFame(capacity=10, checkpoint_dir=self.checkpoint_dir)

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

        # ROI hurdle EMA - smoothed target for ROI-based scoring adjustment
        # Uses EMA with α=0.2 (converges to static target in ~5 iterations)
        self.roi_hurdle_ema = None  # Initialized from first HoF median

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

        # Validation caching - skip re-validation for unchanged elite agents
        self.validation_cache = {}  # Maps (agent_hash, slice_hash) → validation results
        self.val_slice_hash = None  # Hash of current validation slices

        # Initialize resource tracker
        self.resource_tracker = ResourceTracker(disk_path="/workspace")

        print(f"Training: days {self.train_start_idx}-{self.train_end_idx}, "
              f"Validation: days {self.val_start_idx}-{self.val_end_idx}")
        print(f"Walk-forward: 7 random validation slices/generation (4 from quarters + 3 straddling)")

        # Create persistent environment (reused across episodes to prevent memory leaks)
        print("Initializing environment...")
        if self.consistency_mode:
            print(f"  Consistency mode enabled: {Config.CONSISTENCY_LOSS_MULTIPLIER}x loss magnification")
        self.eval_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            start_idx=self.train_start_idx,
            end_idx=self.train_end_idx,
            trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            data_array_full=self.data_loader.data_array_full,
            consistency_mode=self.consistency_mode
        )

        # Load heroes from Hall of Fame if specified (must happen after env creation, before checkpoint load)
        if self.heroes_hof_dir:
            self.load_heroes_from_hof()

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

    def load_heroes_from_hof(self):
        """
        Load agents from a Hall of Fame directory, evaluate them, and select top 32.

        This is called when --heroes flag is provided to start training from
        a set of pre-trained agents instead of random initialization.
        """
        from pathlib import Path
        import glob

        hof_dir = Path(self.heroes_hof_dir)
        hof_subdir = hof_dir / "hall_of_fame"

        # Check if the directory exists
        if not hof_dir.exists():
            print(f"! Heroes HoF directory not found: {hof_dir}")
            print("  Continuing with random initialization.")
            return

        # Determine where agent files are
        if hof_subdir.exists():
            agent_dir = hof_subdir
        else:
            agent_dir = hof_dir

        # Load all agent files
        agent_files = sorted(glob.glob(str(agent_dir / "*.pth")))
        if not agent_files:
            print(f"! No agent files (.pth) found in: {agent_dir}")
            print("  Continuing with random initialization.")
            return

        print("\n" + "="*60)
        print("HEROES MODE: Loading pre-trained agents from Hall of Fame")
        print("="*60)
        print(f"Source: {agent_dir}")
        print(f"Found {len(agent_files)} agent files")

        # Load all agents
        loaded_agents = []
        for i, agent_file in enumerate(agent_files):
            try:
                agent = DDPGAgent(agent_id=i)
                agent.load(agent_file)
                agent.is_elite = False  # Will be re-determined after evaluation
                loaded_agents.append(agent)
            except Exception as e:
                print(f"  ! Failed to load {agent_file}: {e}")

        if not loaded_agents:
            print("! No agents could be loaded. Continuing with random initialization.")
            return

        print(f"Successfully loaded {len(loaded_agents)} agents")

        # Evaluate all loaded agents using current reward function (PARALLEL)
        print(f"\n--- Evaluating {len(loaded_agents)} heroes with current reward function (Parallel) ---")
        num_episodes = 5 if self.consistency_mode else 3
        if self.consistency_mode:
            print(f"  Using consistency mode: {num_episodes} episodes, 0.4*mean + 0.6*min, {Config.CONSISTENCY_LOSS_MULTIPLIER}x loss magnification")
        else:
            print(f"  Using standard mode: {num_episodes} episodes, 0.4*mean + 0.6*min")

        # Prepare environment config for parallel workers
        env_config = {
            'data_array': self.data_loader.data_array,
            'dates': self.data_loader.dates,
            'normalization_stats': self.normalization_stats,
            'start_idx': self.train_start_idx,
            'end_idx': self.train_end_idx,
            'trading_end_idx': self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            'data_array_full': self.data_loader.data_array_full,
            'consistency_mode': self.consistency_mode
        }

        # Prepare tasks for parallel evaluation
        tasks = []
        for agent_idx, agent in enumerate(loaded_agents):
            # Extract agent state (CPU tensors only)
            agent_state = {
                'actor': {k: v.cpu() for k, v in agent.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in agent.critic.state_dict().items()}
            }

            for slice_idx in range(num_episodes):
                # Calculate episode indices
                total_days_needed = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                max_start = self.train_end_idx - total_days_needed
                start_idx = np.random.randint(self.train_start_idx, max_start)
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

                # Create unique seed for this task
                task_seed = self.seed + agent_idx * 1000 + slice_idx

                # Note: env_config is passed via initializer, not in task tuple
                tasks.append((
                    agent_state,
                    start_idx,
                    end_idx,
                    False,  # training=False, don't add to buffer
                    task_seed,
                    None,  # No buffer storage
                    0  # file_id_start (unused)
                ))

        # Execute in parallel
        num_workers = min(mp.cpu_count() - 1, 8)
        print(f"  Using {num_workers} parallel workers for {len(tasks)} tasks")

        fitness_by_agent = [[] for _ in range(len(loaded_agents))]

        # Use initializer to pass env_config once per worker (not once per task)
        # This significantly reduces serialization overhead
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(env_config,)
        ) as executor:
            futures = {executor.submit(_run_episode_worker, task): idx for idx, task in enumerate(tasks)}

            for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating heroes"):
                task_idx = futures[future]
                agent_idx = task_idx // num_episodes

                try:
                    # raw_fitness is cumulative reward (good for RL, bad for Evolution)
                    raw_fitness, episode_info, _ = future.result()

                    # Calculate Structural Fitness for Evolution
                    triad_fitness = self.calculate_triad_fitness(episode_info)

                    fitness_by_agent[agent_idx].append(triad_fitness)
                except Exception as e:
                    print(f"\n  ! Worker failed for hero {agent_idx}: {e}")
                    fitness_by_agent[agent_idx].append(-10000.0)

        # Calculate final fitness for each hero
        hero_fitness = []
        for agent_idx, slice_fitness in enumerate(fitness_by_agent):
            if self.consistency_mode:
                final_fitness = sum(slice_fitness)
            else:
                sorted_fitness = sorted(slice_fitness)
                final_fitness = np.mean(sorted_fitness[:2])
            hero_fitness.append((final_fitness, loaded_agents[agent_idx]))

        # Sort by fitness (descending) and select top 32
        hero_fitness.sort(key=lambda x: x[0], reverse=True)
        num_to_select = min(Config.POPULATION_SIZE, len(hero_fitness))

        print(f"\n--- Selecting top {num_to_select} heroes ---")
        print("Top 5 heroes:")
        for i, (fitness, _) in enumerate(hero_fitness[:5]):
            print(f"  {i+1}. Fitness: {fitness:.2f}")

        # Replace population with selected heroes
        selected_heroes = []
        for i in range(num_to_select):
            fitness, agent = hero_fitness[i]
            agent.agent_id = i
            selected_heroes.append(agent)

        # Fill remaining slots with clones of top agents if needed
        while len(selected_heroes) < Config.POPULATION_SIZE:
            idx = len(selected_heroes) % num_to_select
            clone = hero_fitness[idx][1].clone()
            clone.agent_id = len(selected_heroes)
            selected_heroes.append(clone)

        # Replace population
        old_population = self.population
        self.population = selected_heroes

        # Clean up old population
        for agent in old_population:
            del agent
        gc.collect()

        print(f"\nPopulation replaced with {len(self.population)} heroes")

        # --- Validate heroes and populate Hall of Fame with ROI ---
        # This ensures median HoF ROI is properly set from the start
        print("\n--- Validating heroes for Hall of Fame (with ROI) ---")

        # Generate validation slices for hero evaluation
        self.current_generation_val_slices = self.generate_validation_slices()

        # Validate top heroes and add to HoF
        num_to_validate = min(10, len(self.population))  # Validate top 10 for HoF
        hero_validation_results = []

        for idx in tqdm(range(num_to_validate), desc="Validating heroes for HoF"):
            val_results = self.validate_agent_cached(self.population[idx])
            val_fitness = val_results['fitness']
            agent_roi = val_results.get('roi', 0.0)

            # Use validation fitness as combined score (no training penalty for initial heroes)
            combined_fitness = val_fitness

            hero_validation_results.append({
                'idx': idx,
                'combined_fitness': combined_fitness,
                'roi': agent_roi,
                'raw_pnl': val_results.get('raw_pnl', 0.0),
                'expectancy': val_results.get('expectancy', 0.0)
            })

        # Sort by combined fitness and add to HoF
        hero_validation_results.sort(key=lambda x: x['combined_fitness'], reverse=True)

        # Build candidates for HoF
        hof_candidates = []
        for result in hero_validation_results:
            agent_idx = result['idx']
            combined_score = result['combined_fitness']
            agent_roi = result['roi']
            agent_expectancy = result['expectancy']
            hof_candidates.append((self.population[agent_idx], combined_score, agent_idx, agent_roi, agent_expectancy))

        # Add heroes to Hall of Fame
        admission_results = self.hall_of_fame.update_from_generation(hof_candidates, generation=0)

        # Print HoF initialization summary
        admitted = [(idx, score, action) for idx, score, action in admission_results
                   if action == 'admitted' or action.startswith('replaced_')]
        if admitted:
            hof_stats = self.hall_of_fame.get_stats()
            print(f"\n⭐ Hall of Fame initialized with {len(admitted)} heroes:")
            for agent_idx, score, _ in admitted[:5]:  # Show top 5
                roi = next((r['roi'] for r in hero_validation_results if r['idx'] == agent_idx), 0.0)
                print(f"   + Agent {agent_idx}: Combined={score:.2f}, ROI={roi:.2f}%")
            if len(admitted) > 5:
                print(f"   ... and {len(admitted) - 5} more")
            print(f"   Median HoF ROI: {hof_stats['median_roi']:.2f}% (benchmark for ROI adjustment)")

        print(f"\nUsing heroes mode elite/offspring fractions:")
        print(f"  Elite: {Config.HEROES_ELITE_FRAC * 100:.1f}% ({int(Config.POPULATION_SIZE * Config.HEROES_ELITE_FRAC)} agents)")
        print(f"  Offspring: {Config.HEROES_OFFSPRING_FRAC * 100:.1f}% ({int(Config.POPULATION_SIZE * Config.HEROES_OFFSPRING_FRAC)} agents)")
        mutant_frac = 1.0 - Config.HEROES_ELITE_FRAC - Config.HEROES_OFFSPRING_FRAC
        print(f"  Mutants: {mutant_frac * 100:.1f}% ({int(Config.POPULATION_SIZE * mutant_frac)} agents)")
        print("="*60 + "\n")

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
            
            # Store transition in replay buffer (Teacher Forcing: all agents contribute)
            # Note: training flag controls noise (line 776), not buffer saving
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

        # Apply zero-trades penalty if no trades were made (mode-specific penalty from env)
        if episode_summary['num_trades'] == 0:
            final_fitness -= episode_summary['zero_trades_penalty']

        # Apply win rate bonus if enough trades and win rate above threshold
        if episode_summary['num_trades'] >= Config.WIN_RATE_BONUS_MIN_TRADES:
            win_rate_pct = episode_summary['win_rate'] * 100.0  # Convert to percentage
            if win_rate_pct > Config.WIN_RATE_BONUS_THRESHOLD:
                bonus = (win_rate_pct - Config.WIN_RATE_BONUS_THRESHOLD) ** 2
                final_fitness += bonus
                episode_summary['win_rate_bonus'] = bonus
            else:
                episode_summary['win_rate_bonus'] = 0.0
        else:
            episode_summary['win_rate_bonus'] = 0.0

        # NOTE: No need to delete env - we're reusing persistent environments now
        return final_fitness, episode_summary

    def run_episode_batched(self, agent: DDPGAgent, env: TradingEnvironment,
                           start_idx: int, end_idx: int,
                           training: bool = True, batch_size: int = 16) -> Tuple[float, Dict]:
        """
        Run one episode with batched inference for faster evaluation.

        Collects multiple states and processes them in a single forward pass through
        the actor network, providing 10-15% speedup for GPU inference.

        Args:
            agent: Agent to run
            env: Persistent TradingEnvironment to reuse
            start_idx: Starting day index
            end_idx: Ending day index
            training: Whether this is training (adds to replay buffer)
            batch_size: Number of states to batch together (default: 16)

        Returns:
            Tuple of (cumulative_reward, episode_info)
        """
        # Calculate trading end
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

        # Set environment training mode
        env.set_training_mode(training)

        # Reset environment
        state, info = env.reset(
            start_idx=start_idx,
            end_idx=end_idx,
            trading_end_idx=trading_end_idx
        )
        cumulative_reward = 0.0
        steps = 0

        # State buffer for batching
        state_buffer = []
        replay_buffer_data = []  # Store transitions for later addition to replay buffer

        while True:
            # Collect states for batching
            state_buffer.append(state.copy())

            # When buffer is full (or episode ending), process batch
            if len(state_buffer) >= batch_size:
                # ONE forward pass for entire batch (much faster on GPU)
                states_array = np.stack(state_buffer)  # [batch_size, context_days, num_cols, features]
                actions_batch = agent.select_actions_batch(states_array, add_noise=training)

                # Apply actions sequentially (environment is stateful)
                for i, action in enumerate(actions_batch):
                    next_state, reward, terminated, truncated, info = env.step(action)

                    # Store transition for replay buffer (if training)
                    if training:
                        replay_buffer_data.append({
                            'state': state_buffer[i].astype(np.float32),
                            'action': action.astype(np.float32),
                            'reward': reward,
                            'next_state': next_state.astype(np.float32),
                            'done': float(terminated or truncated)
                        })

                    cumulative_reward += reward
                    steps += 1
                    state = next_state

                    if terminated or truncated:
                        break

                # Clear buffer
                state_buffer = []

                if terminated or truncated:
                    break

        # Process any remaining states in buffer
        if len(state_buffer) > 0:
            states_array = np.stack(state_buffer)
            actions_batch = agent.select_actions_batch(states_array, add_noise=training)

            for i, action in enumerate(actions_batch):
                next_state, reward, terminated, truncated, info = env.step(action)

                if training:
                    replay_buffer_data.append({
                        'state': state_buffer[i].astype(np.float32),
                        'action': action.astype(np.float32),
                        'reward': reward,
                        'next_state': next_state.astype(np.float32),
                        'done': float(terminated or truncated)
                    })

                cumulative_reward += reward
                steps += 1
                state = next_state

                if terminated or truncated:
                    break

        # Add all transitions to replay buffer at once (if training)
        if training:
            for transition in replay_buffer_data:
                self.replay_buffer.add(
                    state=transition['state'],
                    action=transition['action'],
                    reward=transition['reward'],
                    next_state=transition['next_state'],
                    done=transition['done']
                )

        # Get episode summary
        episode_summary = env.get_episode_summary()
        episode_summary['steps'] = steps

        # Calculate final fitness
        final_fitness = float(cumulative_reward)

        # Apply zero-trades penalty if no trades were made (mode-specific penalty from env)
        if episode_summary['num_trades'] == 0:
            final_fitness -= episode_summary['zero_trades_penalty']

        return final_fitness, episode_summary

    def generate_validation_slices(self) -> List[Tuple[int, int, int]]:
        """
        Generate 7 random validation slices from validation set.

        Divides validation period into 4 equal quarters, then samples:
        - 4 slices from within each quarter
        - 3 straddling slices between quarters (Q1-Q2, Q2-Q3, Q3-Q4)

        This ensures comprehensive coverage with overlapping windows across different market conditions.

        Each slice consists of:
        - CONTEXT_WINDOW_DAYS (504) of prior data (may come from training data for context)
        - TRADING_PERIOD_DAYS (125) where agent can trade (from validation set)
        - SETTLEMENT_PERIOD_DAYS (30) to close positions (from validation set)

        Returns:
            List of 7 tuples: (start_idx, end_idx, trading_end_idx)
        """
        # NOTE: start_idx is the first day of TRADING (not context)
        # The environment automatically looks back 504 days from start_idx for context
        # So we just need to ensure trading + settlement fit within validation set

        # Trading must start at or after val_start_idx
        min_start = self.val_start_idx

        # Trading + settlement must end before val_end_idx
        max_start = self.val_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start < min_start:
            raise ValueError(f"Not enough validation data: need {Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS} days")

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

        return slices

    def _hash_agent(self, agent: DDPGAgent) -> str:
        """
        Create hash of agent's weights for caching.
        Uses first layer weights for efficiency while maintaining uniqueness.

        Args:
            agent: Agent to hash

        Returns:
            MD5 hash string
        """
        import hashlib
        actor_weights = agent.actor.state_dict()
        # Hash first layer weights only (sufficient for uniqueness)
        first_layer = list(actor_weights.values())[0].cpu().numpy()
        return hashlib.md5(first_layer.tobytes()).hexdigest()

    def _hash_validation_slices(self, slices: List[Tuple[int, int, int]]) -> str:
        """
        Hash validation slices configuration.

        Args:
            slices: List of (start_idx, end_idx, trading_end_idx) tuples

        Returns:
            MD5 hash string
        """
        import hashlib
        return hashlib.md5(str(slices).encode()).hexdigest()

    def calculate_triad_fitness(self, stats: Dict) -> float:
        """
        Triad 2.0: Stabilized Fitness Function
        Base = Signed ROI * Log(Volume)
        If Positive: Boosted by WR and QR
        If Negative: Penalized by Inconsistency
        """
        total_trades = stats.get('num_trades', 0)

        # 1. Handle Inactivity
        # Keep your existing gradient logic for zero trades
        if total_trades == 0:
            penalty = Config.ZERO_TRADES_PENALTY_CONSISTENCY if self.consistency_mode else Config.ZERO_TRADES_PENALTY_NORMAL
            return -penalty + stats.get('max_coefficient_during_episode', 0)

        # 2. Calculate Core Metrics
        win_rate = stats.get('win_rate', 0.0) # 0.0 to 1.0

        closed_trades = stats.get('closed_trades', [])
        quality_threshold = 0.2 # 0.2% gain
        if closed_trades:
            quality_count = sum(1 for t in closed_trades if t.get('gain_pct', 0) > quality_threshold)
            qr = quality_count / total_trades
        else:
            qr = 0.0

        raw_pnl = stats.get('raw_pnl', 0.0)
        total_inv = stats.get('total_investment', 0.0)
        roi_pct = (raw_pnl / total_inv * 100) if total_inv > 0 else 0.0

        # Clip ROI for stability (prevent one outlier from breaking scale)
        roi_score = np.clip(roi_pct, -25, 25)

        # 3. Volume Scalar (Log Magnitude)
        # Adding 10 ensures log is always > 1, providing a baseline score
        volume_scalar = math.log10(abs(raw_pnl) + 10)

        # 4. Calculate Fitness
        if roi_score > 0:
            # --- WINNING SCENARIO ---
            # Base Score: ROI * Volume
            base_score = roi_score * volume_scalar

            # Boosters: Reward High WR and High QR
            # We use (1 + x) so we don't punish a 50% WR by halving the score
            # WR^2 is kept as a bonus multiplier to reward the "Unicorn" 70%+ behavior
            consistency_bonus = (1.0 + (win_rate ** 2))
            conviction_bonus = (1.0 + qr)

            fitness = base_score * consistency_bonus * conviction_bonus * 10.0

        else:
            # --- LOSING SCENARIO ---
            # Pure Pain: ROI * Volume
            # We multiply by (2.0 - win_rate) to punish "consistent losers" less than "gamblers"
            # Actually, simpler is better: Just strict penalization of negative ROI.
            fitness = roi_score * volume_scalar * 10.0

        return float(fitness)

    def evaluate_population(self) -> Tuple[List[float], Dict]:
        """
        Evaluate all agents in population (fitness scores).

        NEW: Multi-slice evaluation for robust fitness signal
        - Each agent is evaluated on 3 different random training slices
        - Final fitness = average of the LOWEST 2 scores (conservative, robust estimate)
        - This prevents "lucky" agents from advancing and selects for consistency

        Returns:
            Tuple of (fitness_scores, aggregate_stats)
        """
        fitness_scores = []
        all_episode_stats = []

        # Use same number of episodes and scoring method for both modes
        # ALWAYS use pessimistic aggregator (0.4*mean + 0.6*min) to align with validation gatekeeper
        num_episodes = 5 if self.consistency_mode else 3
        scoring_method = "0.4*mean + 0.6*min (pessimistic, matches validation)"

        print(f"\n--- Generation {self.generation + 1}: Evaluating Population ---")
        print(f"Multi-slice evaluation: {num_episodes} slices per agent, scoring = {scoring_method}")

        # Count elite vs exploratory agents for logging
        num_elites = sum(1 for a in self.population if a.is_elite)
        num_exploratory = len(self.population) - num_elites
        print(f"Teacher Forcing enabled: {num_elites} elites (no noise) + {num_exploratory} exploratory (with noise) contribute to buffer")

        for agent_idx, agent in enumerate(tqdm(self.population, desc="Evaluating agents")):
            # Evaluate agent on multiple random training slices
            slice_fitness_scores = []
            slice_episode_stats = []

            for _ in range(num_episodes):
                # Calculate episode indices - SAMPLE FROM TRAINING DATA ONLY
                # Training episodes must not touch validation set
                # Need: context (504) + trading (125) + settlement (30) = 659 days total
                total_days_needed = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

                max_start = self.train_end_idx - total_days_needed
                if max_start <= self.train_start_idx:
                    raise ValueError(f"Not enough training data: need {total_days_needed} days")

                # Random start from training range only (excludes validation set)
                start_idx = np.random.randint(self.train_start_idx, max_start)
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

                # Run episode using persistent eval_env (CRITICAL FIX: prevents memory leak)
                # Only add to replay buffer if agent is exploratory (not elite) for diversity
                raw_fitness, episode_info = self.run_episode(
                    agent=agent,
                    env=self.eval_env,  # Reuse persistent environment instead of creating new ones
                    start_idx=start_idx,
                    end_idx=end_idx,
                    training=(not agent.is_elite)  # Only exploratory agents contribute to buffer
                )

                # Calculate Structural Fitness for Evolution
                triad_fitness = self.calculate_triad_fitness(episode_info)

                slice_fitness_scores.append(triad_fitness)
                slice_episode_stats.append(episode_info)

            # Calculate fitness using pessimistic aggregator (same as validation gatekeeper)
            # This aligns training incentives with validation requirements
            # 60% weight on worst slice, 40% weight on average
            mean_score = np.mean(slice_fitness_scores)
            min_score = np.min(slice_fitness_scores)
            final_fitness = (0.4 * mean_score) + (0.6 * min_score)

            fitness_scores.append(final_fitness)

            # Aggregate episode stats across all training slices for this agent
            # Calculate global win rate (total wins / total trades) not average of per-slice win rates
            agent_total_wins = sum([s['num_wins'] for s in slice_episode_stats])
            agent_total_losses = sum([s['num_losses'] for s in slice_episode_stats])
            agent_total_trades = agent_total_wins + agent_total_losses
            agent_win_rate = agent_total_wins / agent_total_trades if agent_total_trades > 0 else 0.0

            agent_aggregate_stats = {
                'num_trades': int(np.mean([s['num_trades'] for s in slice_episode_stats])),
                'num_wins': int(np.mean([s['num_wins'] for s in slice_episode_stats])),
                'num_losses': int(np.mean([s['num_losses'] for s in slice_episode_stats])),
                'win_rate': agent_win_rate,
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

    def evaluate_population_parallel(self) -> Tuple[List[float], Dict]:
        """
        Parallel version of evaluate_population using ProcessPoolExecutor.

        Evaluates all agents on multiple training slices in parallel across CPU cores.
        This provides significant speedup (4-8x) on multi-core systems while maintaining
        identical results to the sequential version.

        Returns:
            Tuple of (fitness_scores, aggregate_stats)
        """
        # Use same number of episodes and scoring method for both modes
        # ALWAYS use pessimistic aggregator (0.4*mean + 0.6*min) to align with validation gatekeeper
        num_episodes = 5 if self.consistency_mode else 3
        scoring_method = "0.4*mean + 0.6*min (pessimistic, matches validation)"

        print(f"\n--- Generation {self.generation + 1}: Evaluating Population (Parallel) ---")
        print(f"Multi-slice evaluation: {num_episodes} slices per agent, scoring = {scoring_method}")

        # Count elite vs exploratory agents for logging
        num_elites = sum(1 for a in self.population if a.is_elite)
        num_exploratory = len(self.population) - num_elites
        print(f"Teacher Forcing enabled: {num_elites} elites (no noise) + {num_exploratory} exploratory (with noise) contribute to buffer")

        # Prepare environment config (shared across all workers)
        env_config = {
            'data_array': self.data_loader.data_array,
            'dates': self.data_loader.dates,
            'normalization_stats': self.normalization_stats,
            'start_idx': self.train_start_idx,
            'end_idx': self.train_end_idx,
            'trading_end_idx': self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            'data_array_full': self.data_loader.data_array_full,
            'consistency_mode': self.consistency_mode
        }

        # Prepare all evaluation tasks with file ID allocation
        # Pre-allocate file IDs for each worker to avoid conflicts
        buffer_storage_path = str(self.replay_buffer.storage_path)
        file_id_counter = self.replay_buffer.total_added  # Start from current total

        tasks = []
        for agent_idx, agent in enumerate(self.population):
            # Extract agent state (CPU tensors only)
            agent_state = {
                'actor': {k: v.cpu() for k, v in agent.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in agent.critic.state_dict().items()}
            }

            for slice_idx in range(num_episodes):  # num_episodes slices per agent
                # Calculate random start indices
                total_days_needed = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                max_start = self.train_end_idx - total_days_needed
                start_idx = np.random.randint(self.train_start_idx, max_start)
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

                # Create unique seed for this task for reproducibility
                task_seed = self.seed + agent_idx * 1000 + slice_idx

                # Allocate file IDs for this episode (estimate ~150 transitions per episode)
                # Workers will write transitions with IDs starting from file_id_start
                file_id_start = file_id_counter
                file_id_counter += 200  # Reserve 200 IDs per episode (generous estimate)

                # Note: env_config is passed via initializer, not in task tuple
                tasks.append((
                    agent_state,
                    start_idx,
                    end_idx,
                    not agent.is_elite,  # training flag (controls noise, not buffer saving)
                    task_seed,
                    buffer_storage_path,  # All agents write to buffer (Teacher Forcing)
                    file_id_start
                ))

        # Execute in parallel
        num_workers = min(mp.cpu_count() - 1, 8)  # Leave 1 core free, cap at 8
        print(f"Using {num_workers} parallel workers")

        fitness_by_agent = [[] for _ in range(len(self.population))]

        # CRITICAL: Use 'spawn' method for CUDA compatibility on macOS/Linux
        # Fork method doesn't work with CUDA after initialization
        all_transition_file_paths = []  # Collect all file paths written by workers

        # Use initializer to pass env_config once per worker (not once per task)
        # This significantly reduces serialization overhead, especially in consistency mode
        # where we have 160 tasks (32 agents x 5 episodes) vs 96 in normal mode
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(env_config,)
        ) as executor:
            # Submit all tasks
            futures = {executor.submit(_run_episode_worker, task): idx for idx, task in enumerate(tasks)}

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating (parallel)"):
                task_idx = futures[future]
                agent_idx = task_idx // num_episodes  # Each agent has num_episodes slices

                try:
                    # raw_fitness is the sum of rewards from env (good for RL, bad for Evolution)
                    raw_fitness, episode_info, transition_file_paths = future.result()

                    # Calculate Structural Fitness for Evolution
                    triad_fitness = self.calculate_triad_fitness(episode_info)

                    # Store triad_fitness instead of raw_fitness
                    fitness_by_agent[agent_idx].append((triad_fitness, episode_info))

                    # Collect transition file paths from exploratory agents
                    if transition_file_paths:
                        all_transition_file_paths.extend(transition_file_paths)

                except Exception as e:
                    print(f"\n⚠ Worker failed for agent {agent_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use penalty fitness for failed episodes
                    fitness_by_agent[agent_idx].append((-10000.0, {
                        'num_trades': 0, 'num_wins': 0, 'num_losses': 0, 'win_rate': 0.0
                    }))

        # Aggregate results (same logic as sequential version)
        fitness_scores = []
        all_episode_stats = []

        for agent_slices in fitness_by_agent:
            slice_fitness = [f for f, _ in agent_slices]
            slice_stats = [info for _, info in agent_slices]

            # Calculate fitness using pessimistic aggregator (same as validation gatekeeper)
            # This aligns training incentives with validation requirements
            mean_score = np.mean(slice_fitness)
            min_score = np.min(slice_fitness)
            final_fitness = (0.4 * mean_score) + (0.6 * min_score)
            fitness_scores.append(final_fitness)

            # Aggregate stats - calculate global win rate (not average of per-slice win rates)
            agent_total_wins = sum([s['num_wins'] for s in slice_stats])
            agent_total_losses = sum([s['num_losses'] for s in slice_stats])
            agent_total_trades = agent_total_wins + agent_total_losses
            agent_win_rate = agent_total_wins / agent_total_trades if agent_total_trades > 0 else 0.0

            agent_stats = {
                'num_trades': int(np.mean([s['num_trades'] for s in slice_stats])),
                'num_wins': int(np.mean([s['num_wins'] for s in slice_stats])),
                'num_losses': int(np.mean([s['num_losses'] for s in slice_stats])),
                'win_rate': agent_win_rate,
            }
            all_episode_stats.append(agent_stats)

        # Ensure fitness_scores are all plain floats
        fitness_scores = [float(f) for f in fitness_scores]

        # Add transition file paths to replay buffer (transitions already written to disk by workers!)
        print(f"\n--- Adding {len(all_transition_file_paths)} transitions to replay buffer ---")
        if all_transition_file_paths:
            # Transitions were written to disk during parallel evaluation - just add paths to buffer
            print(f"  Transitions already written to disk by workers (parallel I/O)")
            for file_path in all_transition_file_paths:
                self.replay_buffer.buffer.append(file_path)

            # Update total_added counter
            self.replay_buffer.total_added = file_id_counter

            # Handle buffer overflow - remove oldest files if we exceeded capacity
            if len(self.replay_buffer.buffer) > self.replay_buffer.capacity:
                num_to_remove = len(self.replay_buffer.buffer) - self.replay_buffer.capacity
                print(f"  Buffer overflow: removing {num_to_remove} oldest transitions")

                import os
                for _ in range(num_to_remove):
                    old_path = self.replay_buffer.buffer.popleft()
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass

            print(f"  ✓ Buffer updated: {len(self.replay_buffer)} transitions")
        else:
            print("  No transitions collected this generation")

        # Aggregate statistics across all agents
        aggregate_stats = {
            'total_trades': int(sum(s['num_trades'] for s in all_episode_stats)),
            'avg_trades_per_agent': float(sum(s['num_trades'] for s in all_episode_stats) / len(all_episode_stats)),
            'total_wins': int(sum(s['num_wins'] for s in all_episode_stats)),
            'total_losses': int(sum(s['num_losses'] for s in all_episode_stats)),
            'avg_win_rate': float(sum(s['win_rate'] for s in all_episode_stats if s['num_trades'] > 0) / len([s for s in all_episode_stats if s['num_trades'] > 0])) if any(s['num_trades'] > 0 for s in all_episode_stats) else 0.0,
            'agents_with_positive_fitness': int(sum(1 for f in fitness_scores if f > 0)),
        }

        # Clean up
        del all_episode_stats
        del all_transition_file_paths
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
                }, step=self.generation)

            # Explicitly clear loss lists to free memory
            del actor_losses
            del critic_losses

            # Clear GPU cache after each agent to prevent accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def evolve_population(self, fitness_scores: List[float], validation_scores: List[float] = None):
        """
        Evolve population using genetic algorithm.

        Args:
            fitness_scores: Training fitness for each agent (used for tournament selection)
            validation_scores: Validation fitness for elite selection (if None, uses fitness_scores)

        Note:
            - Training fitness is used by DDPG for gradient updates (done in train_population)
            - Validation fitness is used by GA for selecting elites (prevents overfitting)
            - Tournament selection uses training fitness to maintain exploration diversity
        """
        print(f"\n--- Evolving Population (mutation: {self.current_mutation_rate:.3f}) ---")

        # CRITICAL FIX: Store old population reference before creating new one
        # This prevents memory leak from lingering agent references (~2-3GB per generation)
        old_population = self.population

        # Use validation scores for elite selection if provided, otherwise fall back to training fitness
        elite_scores = validation_scores if validation_scores is not None else fitness_scores

        # Create next generation with adaptive mutation parameters
        # Elitism uses validation fitness for robustness and generalization
        # Tournament selection uses training fitness to maintain exploration
        self.population = create_next_generation(
            old_population,
            fitness_scores,
            elite_scores=elite_scores,
            mutation_rate=self.current_mutation_rate,
            mutation_std=self.current_mutation_std,
            heroes_mode=self.heroes_hof_dir is not None
        )

        # Explicitly delete old agents and force GC
        for agent in old_population:
            del agent
        del old_population
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def calculate_expectancy(self, closed_trades):
        """
        Calculate Expectancy metric for trading performance.

        Expectancy = (Win Rate × Avg Win %) − (Loss Rate × Avg Loss %)

        - If Expectancy > 0: The agent has a mathematical edge
        - If Expectancy trends up: The agent is becoming a sharper trader
        - If Expectancy is flat but PnL is up: The agent is just trading more (scaling), not getting smarter

        Args:
            closed_trades: List of closed trade dictionaries with 'gain_pct' field

        Returns:
            Expectancy value (float)
        """
        if not closed_trades:
            return 0.0

        # Separate wins and losses
        wins = [t['gain_pct'] for t in closed_trades if t['gain_pct'] > 0]
        losses = [abs(t['gain_pct']) for t in closed_trades if t['gain_pct'] <= 0]

        if not wins and not losses:
            return 0.0

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        win_rate = len(wins) / len(closed_trades)
        loss_rate = 1.0 - win_rate

        # Expectancy = (Probability of Win * Reward) - (Probability of Loss * Risk)
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return expectancy

    def validate_agent(self, agent) -> Dict:
        """
        Validate agent using walk-forward validation on 7 random slices.

        Walk-forward validation strategy:
        - Runs agent on 7 validation slices (same slices for all agents in this generation)
        - 4 slices from quarters + 3 straddling slices between quarters
        - Uses weighted aggregation: fitness = (0.4 * mean) + (0.6 * worst_case)
        - This rewards consistency and penalizes agents that fail in any market condition

        Args:
            agent: The agent to validate

        Returns:
            Validation results with 'fitness' emphasizing worst-case performance
        """
        if agent is None:
            return {}

        if not self.current_generation_val_slices:
            raise ValueError("No validation slices generated for this generation")

        # Run agent on all 7 validation slices
        slice_results = []
        all_closed_trades = []  # Collect all closed trades from all slices

        for start_idx, end_idx, _ in self.current_generation_val_slices:
            # Use batched inference for faster validation
            fitness, episode_info = self.run_episode_batched(
                agent=agent,
                env=self.eval_env,  # Reuse persistent environment
                start_idx=start_idx,
                end_idx=end_idx,
                training=False,
                batch_size=16
            )

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
                'avg_reward_per_trade': episode_info['avg_reward_per_trade'],
                'raw_pnl': episode_info.get('raw_pnl', 0.0),
                'total_investment': episode_info.get('total_investment', 0.0)
            })

            # Collect closed trades from this slice
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        # Extract fitness scores from all 7 slices
        fitness_scores = [result['fitness'] for result in slice_results]

        # Weighted aggregation: emphasize worst-case performance to reward consistency
        # 60% weight on worst slice, 40% weight on average
        # This forces agents to raise their "floor" rather than just their "ceiling"
        mean_score = np.mean(fitness_scores)
        min_score = np.min(fitness_scores)
        validation_fitness = (0.4 * mean_score) + (0.6 * min_score)

        # Select one sample trade (first trade from all validation slices, if any)
        sample_trade = all_closed_trades[0] if all_closed_trades else None

        # Return aggregated results (weighted combination emphasizing worst case)
        # Also aggregate other metrics for logging
        total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
        total_investment = sum([r['total_investment'] for r in slice_results])
        roi = (total_raw_pnl / total_investment * 100) if total_investment > 0 else 0.0

        # Calculate global win rate (total wins / total trades across all slices)
        # This ensures WR >= QR (quality rate) since all quality trades are winning trades
        total_wins = sum([r['num_wins'] for r in slice_results])
        total_losses = sum([r['num_losses'] for r in slice_results])
        total_trades = total_wins + total_losses
        global_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        # Calculate Expectancy metric from all closed trades
        expectancy = self.calculate_expectancy(all_closed_trades)

        return {
            'fitness': validation_fitness,
            'fitness_all_slices': fitness_scores,  # For debugging
            'fitness_mean': mean_score,  # Mean of all slices (for debugging)
            'fitness_min': min_score,  # Worst slice (for debugging)
            'win_rate': global_win_rate,  # Global WR (not average of per-slice WRs)
            'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
            'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
            'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
            'avg_reward_per_trade': np.mean([r['avg_reward_per_trade'] for r in slice_results]),
            'raw_pnl': total_raw_pnl,  # Total raw P&L across slices
            'total_investment': total_investment,  # Total investment across slices
            'roi': roi,  # ROI percentage
            'expectancy': expectancy,  # Expectancy metric: (Win Rate × Avg Win %) − (Loss Rate × Avg Loss %)
            'sample_trade': sample_trade,  # One sample trade for verification
            'closed_trades': all_closed_trades  # All closed trades for quality count calculation
        }

    def validate_agent_cached(self, agent: DDPGAgent) -> Dict:
        """
        Validate agent with caching - skip validation if agent weights unchanged.

        Elite agents often have unchanged weights across generations, so we can
        skip re-validation and use cached results for significant speedup (~20%).

        Args:
            agent: Agent to validate

        Returns:
            Validation results (from cache or fresh evaluation)
        """
        agent_hash = self._hash_agent(agent)
        slice_hash = self.val_slice_hash
        cache_key = f"{agent_hash}_{slice_hash}"

        if cache_key in self.validation_cache:
            # Cache hit - return cached results
            return self.validation_cache[cache_key]

        # Cache miss - run validation
        val_results = self.validate_agent(agent)
        self.validation_cache[cache_key] = val_results

        # Keep cache bounded (last 100 entries to prevent memory growth)
        if len(self.validation_cache) > 100:
            # Remove oldest entry (first key in dict)
            oldest_key = next(iter(self.validation_cache))
            del self.validation_cache[oldest_key]

        return val_results

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

            else:
                # Already in plateau - no additional logging needed
                pass
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
            'wandb_run_name': wandb.run.name,
            'leverage_mode_active': self.leverage_mode_active,
            'leverage_generations_remaining': self.leverage_generations_remaining,
            'roi_hurdle_ema': self.roi_hurdle_ema
        }
        state_path = checkpoint_dir / "trainer_state.json"
        with open(state_path, 'w') as f:
            json.dump(trainer_state, f, indent=4)

        # 5. Save Hall of Fame
        if self.hall_of_fame is not None and len(self.hall_of_fame) > 0:
            self.hall_of_fame.save()
            hof_stats = self.hall_of_fame.get_stats()
            print(f"  Saved Hall of Fame ({hof_stats['size']} champions)")

        # Sync to cloud storage in background (non-blocking)
        self.cloud_sync.sync_checkpoints(str(checkpoint_dir), background=True,
                                        exclude_patterns=["buffer_storage"])
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
        population_loaded = False
        if pop_dir.exists():
            try:
                for i, agent in enumerate(self.population):
                    agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                    if agent_path.exists():
                        agent.load(str(agent_path))
                print(f"✓ Loaded {len(self.population)} agents")
                population_loaded = True
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

                # Load leverage mode state
                self.leverage_mode_active = trainer_state.get('leverage_mode_active', False)
                self.leverage_generations_remaining = trainer_state.get('leverage_generations_remaining', 0)

                # Load ROI hurdle EMA (defaults to None for old checkpoints)
                self.roi_hurdle_ema = trainer_state.get('roi_hurdle_ema', None)

                print(f"✓ Resuming from Gen {self.start_generation} → Gen {self.start_generation + 1}")
                print(f"✓ Best validation fitness: {self.best_validation_fitness:.2f}")
                if self.roi_hurdle_ema is not None:
                    print(f"✓ ROI Hurdle EMA: {self.roi_hurdle_ema:.2f}%")

                # Restore leverage multiplier if leverage mode was active
                if self.leverage_mode_active and population_loaded:
                    print(f"✓ Leverage mode: {self.leverage_generations_remaining} generations remaining")
                    print("  Restoring leverage multiplier (1.5x) for loaded agents...")
                    # Note: We can't identify which specific agents had leverage, so we apply it to all
                    # This is safe because leverage mode affects the entire population during its active period
                    for agent in self.population:
                        agent.actor.leverage_multiplier = 1.5
                        agent.actor_target.leverage_multiplier = 1.5
                    print("  ✓ Leverage multiplier restored")
            except Exception as e:
                print(f"❌ Error loading state: {e}")

        # 6. Load Hall of Fame
        try:
            self.hall_of_fame.load()
            hof_stats = self.hall_of_fame.get_stats()
            if hof_stats['size'] > 0:
                print(f"✓ Loaded Hall of Fame: {hof_stats['size']} champions")
                print(f"  Best HoF score: {hof_stats['best_score']:.2f}, "
                      f"Range: {hof_stats['worst_score']:.2f}-{hof_stats['best_score']:.2f}")
            else:
                print("! No Hall of Fame found (starting fresh)")
        except Exception as e:
            print(f"⚠ Could not load Hall of Fame: {e}")

        # 7. Re-evaluate all loaded agents with current reward function
        if population_loaded:
            print("\n" + "="*60)
            print("Re-evaluating loaded agents with current reward function")
            print("="*60)

            # Generate validation slices for re-evaluation
            print("Generating validation slices for re-evaluation...")
            self.current_generation_val_slices = self.generate_validation_slices()
            self.val_slice_hash = self._hash_validation_slices(self.current_generation_val_slices)

            # Re-evaluate population on training data (using parallel evaluation for speed)
            print("\nRe-evaluating population on training data...")
            fitness_scores, _ = self.evaluate_population_parallel()

            # Re-evaluate population on validation data
            print("\nRe-evaluating population on validation data...")
            for idx, agent in enumerate(self.population):
                val_results = self.validate_agent(agent)
                # Note: Cache will be rebuilt naturally during first generation

            # Re-evaluate best agent if it exists
            if self.best_agent is not None:
                print("\nRe-evaluating best agent...")
                best_val_results = self.validate_agent(self.best_agent)
                self.best_validation_fitness = best_val_results['fitness']
                print(f"✓ Best agent validation fitness: {self.best_validation_fitness:.2f}")

            # Re-evaluate Hall of Fame agents by loading them from disk
            if len(self.hall_of_fame.entries) > 0:
                print(f"\nRe-evaluating {len(self.hall_of_fame.entries)} Hall of Fame agents...")
                print(f"HoF directory: {self.hall_of_fame.hof_dir}")

                success_count = 0
                for entry in self.hall_of_fame.entries:
                    # Load agent from HoF directory
                    agent_path = self.hall_of_fame.hof_dir / f"hof_agent_{entry.agent_id}.pth"
                    if agent_path.exists():
                        hof_agent = DDPGAgent(agent_id=entry.agent_id)
                        hof_agent.load(str(agent_path))

                        # Re-evaluate with current reward function
                        val_results = self.validate_agent(hof_agent)
                        entry.validation_score = val_results['fitness']
                        entry.roi = val_results.get('roi', 0.0)

                        # Clean up
                        del hof_agent
                        success_count += 1
                    else:
                        print(f"  ⚠ Warning: HoF agent {entry.agent_id} file not found at {agent_path}")

                if success_count > 0:
                    print(f"✓ Re-evaluated {success_count}/{len(self.hall_of_fame.entries)} Hall of Fame agents")
                else:
                    print(f"⚠ No Hall of Fame agent files found - skipping HoF re-evaluation")

            print("\n✓ All agents re-evaluated with current reward function")

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

        # Initialize leverage mode if requested
        if self.enable_leverage and not self.leverage_mode_active:
            hof_size = len(self.hall_of_fame)
            if hof_size >= 5:
                print("\n" + "="*60)
                print("🚀 LEVERAGE MODE ACTIVATED")
                print("="*60)
                print(f"Replacing bottom 5 agents with top 5 Hall of Fame champions")
                print(f"These agents will trade with 1.5x coefficients for {self.leverage_total_generations} generations")
                print("="*60)

                # Get top 5 champions from Hall of Fame
                top_5_champions = self.hall_of_fame.get_top_k(k=5)

                # Get indices of bottom 5 agents (we need to evaluate them first)
                # For now, we'll use a simple random sample since we haven't evaluated yet
                # In practice, this will replace the worst performers after first evaluation
                import random
                bottom_5_indices = random.sample(range(len(self.population)), 5)

                # Replace bottom 5 with leveraged HoF champions
                for i, (idx, champion) in enumerate(zip(bottom_5_indices, top_5_champions)):
                    # Clone the champion
                    leveraged_agent = champion.clone()
                    leveraged_agent.agent_id = idx
                    leveraged_agent.is_elite = False

                    # Set leverage multiplier on the actor
                    leveraged_agent.actor.leverage_multiplier = 1.5
                    leveraged_agent.actor_target.leverage_multiplier = 1.5

                    # Replace agent in population
                    old_agent = self.population[idx]
                    self.population[idx] = leveraged_agent
                    del old_agent

                    print(f"  Agent {idx}: Replaced with HoF champion (1.5x leverage)")

                # Activate leverage mode
                self.leverage_mode_active = True
                self.leverage_generations_remaining = self.leverage_total_generations

                print(f"\n✓ Leverage mode active for next {self.leverage_generations_remaining} generations")
                print("="*60 + "\n")
            else:
                print(f"\n⚠ Leverage mode requested but Hall of Fame only has {hof_size}/5 agents")
                print("  Leverage mode will not be activated. Continue training normally.\n")

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
            # Use parallel evaluation for significant speedup
            fitness_scores, pop_stats = self.evaluate_population_parallel()

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

            # Update best training fitness (for logging only)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness

            # Log comprehensive fitness metrics to wandb
            # Note: Training fitness is calculated from evaluate_population() on training data
            # Detailed metrics like ROI, win_rate, etc. are only available after validation
            wandb.log({
                "fitness/best_fitness": max_fitness,
                "fitness/mean_fitness": mean_fitness,
                "fitness/min_fitness": min_fitness,
                "fitness/std_fitness": np.std(fitness_scores),
                "fitness/best_ever": self.best_fitness,
            }, step=gen)

            # Generate validation slices for this generation
            print(f"\n--- Walk-Forward Validation (Generation {gen + 1}) ---")
            self.current_generation_val_slices = self.generate_validation_slices()
            self.val_slice_hash = self._hash_validation_slices(self.current_generation_val_slices)

            best_val_fitness_this_gen = float('-inf')
            best_val_agent_idx = None
            validation_results = []

            # Get median ROI from Hall of Fame for ROI-based scoring adjustment
            # Use EMA to smooth the hurdle and prevent feedback loops
            raw_median_hof_roi = self.hall_of_fame.get_median_roi()

            # Initialize EMA on first generation with HoF data, or update it
            if self.roi_hurdle_ema is None:
                self.roi_hurdle_ema = raw_median_hof_roi

            # Use the smoothed EMA as the benchmark for ROI adjustment
            median_hof_roi = self.roi_hurdle_ema

            # Determine quality threshold for confidence factor calculation
            # Quality trades are those that exceeded this profitability threshold
            if Config.ROI_USE_HOF_MEDIAN_AS_THRESHOLD and median_hof_roi > 0:
                quality_threshold = median_hof_roi
            else:
                quality_threshold = Config.ROI_QUALITY_THRESHOLD

            for idx in tqdm(range(len(self.population)), desc="Validating agents"):
                val_results = self.validate_agent_cached(self.population[idx])
                val_fitness = val_results['fitness']
                val_fitness_mean = val_results.get('fitness_mean', 0.0)
                val_fitness_min = val_results.get('fitness_min', 0.0)
                train_fitness = fitness_scores[idx]
                agent_roi = val_results.get('roi', 0.0)

                # Count quality trades (trades with gain_pct >= threshold)
                closed_trades = val_results.get('closed_trades', [])
                total_trades = len(closed_trades)  # Total trades across all slices
                quality_count = sum(1 for trade in closed_trades if trade.get('gain_pct', 0) >= quality_threshold)

                if self.consistency_mode:
                    # Consistency mode: Use the Pessimistic Score from validate_agent
                    # This is the CORRECT approach: trust the robustness score (0.4*mean + 0.6*min)
                    # already calculated in validate_agent, which heavily penalizes blow-up slices.
                    #
                    # OLD LOGIC (The Trap): Recalculating based on aggregated totals ignores
                    # the fact that an agent might have blown up in 3/5 slices, allowing
                    # "gambler agents" with high volume in 2 good slices to rank high despite
                    # terrible worst-case performance.
                    combined_fitness = val_fitness

                    # Store for logging (no ROI adjustment in consistency mode)
                    base_combined_fitness = combined_fitness
                    roi_adjustment = 0.0
                else:
                    # Standard mode: original fitness calculation
                    # Combined score: penalize agents with negative training fitness
                    # This prevents "lucky" agents that do well on validation but poorly on training
                    # Formula: combined = val_fitness + min(0, train_fitness)
                    base_combined_fitness = val_fitness + min(0.0, train_fitness)

                    # ROI-based scoring adjustment using Hall of Fame median as benchmark
                    # Formula: Score = Fitness + (|Fitness| × multiplier × (AgentROI − MedianROI) / 100)
                    # This rewards agents that outperform the HoF median ROI and penalizes those below

                    # Confidence factor: quality_count / target_count (capped at 1.0)
                    # This ensures agents only get full ROI bonus credit if they have enough quality trades
                    # Use different thresholds: consistency mode = 50, normal mode = 30
                    min_trades_threshold = Config.ROI_CONFIDENCE_MIN_TRADES_CONSISTENCY if self.consistency_mode else Config.ROI_CONFIDENCE_MIN_TRADES
                    confidence_factor = min(1.0, quality_count / min_trades_threshold)

                    roi_adjustment = abs(base_combined_fitness) * Config.ROI_ADJUSTMENT_MULTIPLIER * (agent_roi - median_hof_roi) / 100.0
                    roi_adjustment = roi_adjustment * confidence_factor  # Dampen based on quality trade count
                    combined_fitness = base_combined_fitness + roi_adjustment

                validation_results.append({
                    'idx': idx,
                    'training_fitness': train_fitness,
                    'validation_fitness': val_fitness,
                    'validation_fitness_mean': val_fitness_mean,
                    'validation_fitness_min': val_fitness_min,
                    'combined_fitness': combined_fitness,
                    'base_combined_fitness': base_combined_fitness,
                    'roi_adjustment': roi_adjustment,
                    'win_rate': val_results['win_rate'],
                    'num_trades': val_results['num_trades'],
                    'total_trades': total_trades,  # Total across all slices (for quality ratio)
                    'quality_count': quality_count,
                    'raw_pnl': val_results.get('raw_pnl', 0.0),
                    'roi': agent_roi,
                    'expectancy': val_results.get('expectancy', 0.0)
                })

                # Track best combined fitness in this generation
                if combined_fitness > best_val_fitness_this_gen:
                    best_val_fitness_this_gen = combined_fitness
                    best_val_agent_idx = idx

            # Extract combined scores for elite selection (sorted by agent index)
            validation_scores = [result['combined_fitness'] for result in sorted(validation_results, key=lambda x: x['idx'])]

            # Print summary showing training vs validation rankings
            print(f"\n--- Validation Summary ---")
            if self.consistency_mode:
                print("Consistency mode: WR^2 × QR × ROI × volume_scalar fitness function")
            else:
                print(f"ROI Hurdle EMA: {median_hof_roi:.2f}% (raw HoF median: {raw_median_hof_roi:.2f}%)")
                print(f"Quality threshold: {quality_threshold:.2f}% (min gain_pct for quality trades, need {Config.ROI_CONFIDENCE_MIN_TRADES} for full bonus)")
            validation_results.sort(key=lambda x: x['combined_fitness'], reverse=True)

            if self.consistency_mode:
                # Simplified output for consistency mode (no ROI adjustment)
                print("Top 5 by Combined Fitness - used for elite selection:")
                for i, result in enumerate(validation_results[:5]):
                    quality_ratio = result['quality_count'] / result['total_trades'] if result['total_trades'] > 0 else 0.0
                    print(f"  {i+1}. Agent {result['idx']:2d}: Combined={result['combined_fitness']:>8.2f}, Val=[mean:{result['validation_fitness_mean']:>6.2f}, min:{result['validation_fitness_min']:>6.2f}], ROI={result['roi']:>6.2f}%, QR={quality_ratio:.1%}, PnL=${result['raw_pnl']:>8.2f}, WR={result['win_rate']:.1%}")
            else:
                # Detailed output for standard mode (with ROI adjustment)
                print("Top 5 by Combined Fitness (with ROI adjustment) - used for elite selection:")
                for i, result in enumerate(validation_results[:5]):
                    roi_adj_sign = '+' if result['roi_adjustment'] >= 0 else ''
                    quality_ratio = result['quality_count'] / result['total_trades'] if result['total_trades'] > 0 else 0.0
                    print(f"  {i+1}. Agent {result['idx']:2d}: Combined={result['combined_fitness']:>8.2f} (base={result['base_combined_fitness']:>7.2f}, ROI adj={roi_adj_sign}{result['roi_adjustment']:>6.2f}), Val=[mean:{result['validation_fitness_mean']:>6.2f}, min:{result['validation_fitness_min']:>6.2f}], ROI={result['roi']:>6.2f}%, QR={quality_ratio:.1%}, PnL=${result['raw_pnl']:>8.2f}, WR={result['win_rate']:.1%}")

            # Update best agent if we found a better one based on combined fitness
            if best_val_agent_idx is not None and best_val_fitness_this_gen > self.best_validation_fitness:
                # Verify this is indeed the top agent in sorted results
                top_agent_idx = validation_results[0]['idx']
                if best_val_agent_idx != top_agent_idx:
                    print(f"\n⚠ WARNING: best_val_agent_idx ({best_val_agent_idx}) != top sorted agent ({top_agent_idx})")
                    print(f"   This indicates a bug in best agent selection!")

                print(f"\n✓ New best! Agent {best_val_agent_idx} with Combined fitness: {best_val_fitness_this_gen:.2f} (prev: {self.best_validation_fitness:.2f})")
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

            # --- Comprehensive Validation Logging ---
            # Calculate validation statistics across all agents
            validation_fitness_scores = [r['validation_fitness'] for r in validation_results]
            mean_validation_fitness = np.mean(validation_fitness_scores)

            # Get best agent's detailed metrics
            best_agent_result = validation_results[0]  # Already sorted by combined_fitness
            best_agent_win_rate = best_agent_result['win_rate']
            best_agent_num_trades = best_agent_result['num_trades']
            best_agent_roi = best_agent_result['roi']
            best_agent_expectancy = best_agent_result['expectancy']
            best_agent_quality_ratio = (best_agent_result['quality_count'] / best_agent_result['total_trades']
                                       if best_agent_result['total_trades'] > 0 else 0.0)

            # Get the best agent's per-slice fitness scores
            best_agent_idx = best_agent_result['idx']
            best_agent_val_results = self.validate_agent_cached(self.population[best_agent_idx])
            best_agent_slice_scores = best_agent_val_results.get('fitness_all_slices', [])

            # Log validation metrics to wandb
            validation_log = {
                "validation/best_fitness": best_val_fitness_this_gen,
                "validation/best_ever": self.best_validation_fitness,
                "validation/best_agent_roi": best_agent_roi,
                "validation/best_agent_num_trades": best_agent_num_trades,
                "validation/best_agent_win_rate": best_agent_win_rate,
                "validation/best_agent_quality_ratio": best_agent_quality_ratio,
                "validation/best_agent_expectancy": best_agent_expectancy,
            }

            wandb.log(validation_log, step=gen)

            # --- Hall of Fame Admission Logic ---
            # Build candidate list from all agents in this generation
            candidates = []
            for result in validation_results:
                agent_idx = result['idx']
                combined_score = result['combined_fitness']
                agent_roi = result['roi']
                agent_expectancy = result['expectancy']
                candidates.append((self.population[agent_idx], combined_score, agent_idx, agent_roi, agent_expectancy))

            # Use batch update with aggressive admission and cascading swaps
            admission_results = self.hall_of_fame.update_from_generation(candidates, gen)

            # Print admission results
            admitted = [(idx, score, action) for idx, score, action in admission_results
                       if action == 'admitted' or action.startswith('replaced_')]
            if admitted:
                hof_stats = self.hall_of_fame.get_stats()
                print(f"\n⭐ Hall of Fame updates ({len(admitted)} changes):")
                for agent_idx, score, action in admitted:
                    if action == 'admitted':
                        print(f"   + Agent {agent_idx} admitted (Combined: {score:.2f})")
                    elif action.startswith('replaced_'):
                        old_score = action.replace('replaced_', '')
                        print(f"   ↑ Agent {agent_idx} (Combined: {score:.2f}) replaced {old_score}")
                print(f"   HoF size: {hof_stats['size']}/{self.hall_of_fame.capacity}, "
                      f"Worst: {hof_stats['worst_score']:.2f}, Best: {hof_stats['best_score']:.2f}")

            # Update ROI hurdle EMA after HoF changes
            # EMA with α=0.2: converges to static target in ~5 iterations
            # This smooths the hurdle so it rises gradually as training progresses
            new_median_roi = self.hall_of_fame.get_median_roi()
            old_ema = self.roi_hurdle_ema
            self.roi_hurdle_ema = 0.2 * new_median_roi + 0.8 * self.roi_hurdle_ema
            if admitted:  # Only log if there were changes
                print(f"   ROI Hurdle EMA: {old_ema:.2f}% → {self.roi_hurdle_ema:.2f}% (new median: {new_median_roi:.2f}%)")

            # Log Hall of Fame metrics after admission check
            hof_stats = self.hall_of_fame.get_stats()
            self.writer.add_scalar('HallOfFame/Size', hof_stats['size'], gen)
            self.writer.add_scalar('HallOfFame/BestScore', hof_stats['best_score'], gen)
            self.writer.add_scalar('HallOfFame/WorstScore', hof_stats['worst_score'], gen)
            self.writer.add_scalar('HallOfFame/MeanScore', hof_stats['mean_score'], gen)
            self.writer.add_scalar('HallOfFame/MedianROI', hof_stats['median_roi'], gen)
            self.writer.add_scalar('HallOfFame/ROIHurdleEMA', self.roi_hurdle_ema, gen)

            # Calculate best ROI from Hall of Fame entries
            hof_roi_values = [entry.roi for entry in self.hall_of_fame.entries] if len(self.hall_of_fame.entries) > 0 else [0.0]
            hof_best_roi = float(max(hof_roi_values)) if hof_roi_values else 0.0

            # Calculate best Expectancy from Hall of Fame entries
            hof_expectancy_values = [entry.expectancy for entry in self.hall_of_fame.entries] if len(self.hall_of_fame.entries) > 0 else [0.0]
            hof_best_expectancy = float(max(hof_expectancy_values)) if hof_expectancy_values else 0.0

            # Calculate population ROI stats for this generation
            population_rois = [r['roi'] for r in validation_results]
            mean_population_roi = np.mean(population_rois) if population_rois else 0.0

            # Get best quality ratio from current generation's best agent
            # (HoF doesn't track quality_ratio, so we use current best as proxy)
            best_agent_result_for_hof = validation_results[0] if validation_results else None
            if best_agent_result_for_hof and best_agent_result_for_hof['total_trades'] > 0:
                hof_best_quality_ratio = best_agent_result_for_hof['quality_count'] / best_agent_result_for_hof['total_trades']
            else:
                hof_best_quality_ratio = 0.0

            # Log comprehensive Hall of Fame and adaptive mutation metrics to wandb
            wandb.log({
                "hall_of_fame/min_fitness": hof_stats['worst_score'],
                "hall_of_fame/max_fitness": hof_stats['best_score'],
                "hall_of_fame/mean_fitness": hof_stats['mean_score'],
                "hall_of_fame/median_roi": hof_stats['median_roi'],
                "hall_of_fame/best_roi": hof_best_roi,
                "hall_of_fame/best_quality_ratio": hof_best_quality_ratio,
                "hall_of_fame/best_expectancy": hof_best_expectancy,
                "mutation/rate": self.current_mutation_rate,
                "mutation/std": self.current_mutation_std,
                "mutation/plateau_detected": int(self.plateau_detected),
            }, step=gen)

            # 2. Train agents using replay buffer
            self.train_population()

            # Update resource tracker after training
            self.resource_tracker.update()

            # 🔍 Memory tracking after training
            # log_memory(f"Gen {gen+1}: After train_population", show_objects=True)

            # Update feature importance tracking (computed during training)
            feature_importance = self.get_feature_importance().numpy()
            self.analyze_persistent_low_importance_columns(feature_importance)

            # --- Hall of Fame Injection Logic ---
            # Inject champions from HoF to replace worst performers BEFORE evolution
            # This forces new mutants to beat historical best strategies
            if len(self.hall_of_fame) > 0:
                num_to_inject = min(3, len(self.population))  # Inject up to 3 agents

                # Find indices of worst performers (by training fitness)
                worst_indices = np.argsort(fitness_scores)[:num_to_inject]

                # Sample random champions from Hall of Fame
                hof_champions = self.hall_of_fame.sample_random(k=num_to_inject)

                if len(hof_champions) > 0:
                    print(f"\n🏆 Hall of Fame Injection: Replacing {len(hof_champions)} worst agents with champions")
                    for i, (worst_idx, champion) in enumerate(zip(worst_indices, hof_champions)):
                        # Clone the champion and assign it a new agent ID
                        champion_copy = champion.clone()
                        champion_copy.agent_id = worst_idx
                        champion_copy.is_elite = False  # Not an elite in current gen yet

                        # Replace worst agent with champion
                        old_fitness = fitness_scores[worst_idx]
                        old_agent = self.population[worst_idx]
                        self.population[worst_idx] = champion_copy
                        del old_agent  # Free memory

                        print(f"   Agent {worst_idx}: Fitness {old_fitness:.2f} → HoF Champion")

            # 3. Evolve population using validation fitness for elite selection
            # Note: We pass both training fitness and validation scores
            # - training fitness: used for tournament selection and DDPG gradient updates
            # - validation scores: used for elite selection (ensures robust generalization)
            self.evolve_population(fitness_scores, validation_scores)

            # Update resource tracker after evolution
            self.resource_tracker.update()

            # 🔍 Memory tracking after evolution
            # log_memory(f"Gen {gen+1}: After evolve_population", show_objects=True)

            # 4. Check for plateau and adjust mutation adaptively
            # Run this every generation to detect plateaus quickly
            self.check_and_adjust_mutation(self.best_validation_fitness)

            # Tensorboard logging (less frequent to reduce I/O)
            if (gen + 1) % Config.LOG_FREQUENCY == 0:
                if self.best_agent is not None:
                    val_results = self.validate_best_agent()
                    if val_results:
                        self.writer.add_scalar('Validation/Fitness', val_results['fitness'], gen)
                        self.writer.add_scalar('Validation/WinRate', val_results['win_rate'], gen)

            # 5. Save checkpoint periodically
            if (gen + 1) % Config.SAVE_FREQUENCY == 0:
                self.save_checkpoint()

            # 6. Cleanup orphaned buffer files every 5 generations
            if (gen + 1) % 5 == 0:
                print(f"\n--- Cleaning up orphaned buffer files (Generation {gen + 1}) ---")
                try:
                    cleanup_result = cleanup_orphans(
                        run_name=self.run_name,
                        dry_run=False,
                        verbose=False  # Keep output minimal during training
                    )
                    if cleanup_result['success'] and cleanup_result['orphaned_count'] > 0:
                        deleted = cleanup_result['deleted_count']
                        orphaned = cleanup_result['orphaned_count']
                        print(f"  ✓ Cleaned up {deleted}/{orphaned} orphaned buffer files")
                    elif cleanup_result['success']:
                        print(f"  ✓ No orphaned files found - buffer storage is clean")
                    else:
                        print(f"  ⚠ Cleanup failed: {cleanup_result.get('error', 'unknown error')}")
                except Exception as e:
                    print(f"  ⚠ Cleanup error: {e}")
                    print("  Continuing training...")

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

            # Log timing and buffer metrics to wandb
            wandb.log({
                "training/generation_time": gen_time,
                "buffer/size": buffer_stats['size'],
                "buffer/utilization": buffer_stats['utilization'],
            }, step=gen)

            # Clear GPU cache and run garbage collection to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Leverage mode tracking - decrement counter and deactivate after 5 generations
            if self.leverage_mode_active:
                self.leverage_generations_remaining -= 1
                print(f"\n📊 Leverage mode: {self.leverage_generations_remaining} generations remaining")

                if self.leverage_generations_remaining <= 0:
                    print("\n" + "="*60)
                    print("✓ LEVERAGE MODE COMPLETE")
                    print("="*60)
                    print("Resetting all agents to normal training (1.0x coefficients)")

                    # Reset leverage multiplier for all agents
                    for agent in self.population:
                        agent.actor.leverage_multiplier = 1.0
                        agent.actor_target.leverage_multiplier = 1.0

                    self.leverage_mode_active = False
                    print("✓ Resumed normal training")
                    print("="*60 + "\n")

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