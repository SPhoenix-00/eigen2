"""
ERL Trainer for Project Eigen 2
Evolutionary Reinforcement Learning training loop
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import json
import wandb
import gc
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import shared_memory
if sys.platform != 'win32':
    from multiprocessing import resource_tracker
from enum import Enum
from dataclasses import dataclass
from collections import OrderedDict

# Maximum number of agents to cache per worker to prevent memory leaks
# During evolution, agents mutate every generation, so old cache entries become stale
_WORKER_CACHE_MAX_SIZE = 50

# Suppress common library warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.cuda.amp')

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from models.replay_buffer import ReplayBuffer, OnDiskReplayBuffer
from erl.genetic_ops import create_next_generation, mutate
from erl.hall_of_fame import HallOfFame
from erl.global_hof import GlobalHallOfFame, LeagueRules
from utils.config import Config
from utils.display import print_generation_summary, print_final_summary, ResourceTracker
from utils.cloud_sync import get_cloud_sync_from_env
from utils.cleanup_orphans import cleanup_orphans
from torch.utils.data import DataLoader
# from utils.memory_profiler import get_profiler, log_memory  # Memory profiling disabled

# Local mode evaluator (CPU-optimized, no I/O during evaluation)
from training.local_evaluator import LocalEvaluator


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Global variables for worker processes
_worker_env_config = None  # Shared environment configuration
_worker_env = None  # Reusable environment instance (created once per worker)
_worker_agent_cache = None  # Cache of reconstructed agents {state_hash: agent}
_worker_shm_refs = []  # Keep references to shared memory objects to prevent cleanup


def _init_worker(env_config):
    """
    Initializer function for worker processes.

    This is called once per worker when the ProcessPoolExecutor starts.
    Stores the env_config in a global variable so it doesn't need to be
    pickled with every task (significant performance improvement).

    OPTIMIZATION: Uses shared memory for large numpy arrays to avoid serialization.
    The env_config contains shared memory names instead of actual arrays, and we
    reconstruct the arrays from shared memory here (zero-copy).

    Args:
        env_config: Environment configuration dict with shared memory refs for arrays
    """
    global _worker_env_config, _worker_env, _worker_agent_cache, _worker_shm_refs

    # Reconstruct arrays from shared memory if using shared memory mode
    if 'shm_data_array_name' in env_config:
        # Shared memory mode - reconstruct arrays from shared memory (zero-copy)
        shm_data = shared_memory.SharedMemory(name=env_config['shm_data_array_name'])
        shm_data_full = shared_memory.SharedMemory(name=env_config['shm_data_array_full_name'])

        # Keep references to prevent garbage collection
        _worker_shm_refs = [shm_data, shm_data_full]

        # Reconstruct numpy arrays from shared memory buffers
        data_array = np.ndarray(
            env_config['data_array_shape'],
            dtype=env_config['data_array_dtype'],
            buffer=shm_data.buf
        )
        data_array_full = np.ndarray(
            env_config['data_array_full_shape'],
            dtype=env_config['data_array_full_dtype'],
            buffer=shm_data_full.buf
        )

        # Build actual env_config for TradingEnvironment
        actual_env_config = {
            'data_array': data_array,
            'data_array_full': data_array_full,
            'dates': env_config['dates'],
            'normalization_stats': env_config['normalization_stats'],
            'start_idx': env_config['start_idx'],
            'end_idx': env_config['end_idx'],
            'trading_end_idx': env_config['trading_end_idx'],
            'is_training': env_config.get('is_training', True),
            'consistency_mode': env_config.get('consistency_mode', False),
            'gauntlet_mode': env_config.get('gauntlet_mode', False),
            'maverick_mode': env_config.get('maverick_mode', False)
        }
        _worker_env_config = actual_env_config
    else:
        # Legacy mode - arrays passed directly (fallback)
        _worker_env_config = env_config

    # Create ONE environment per worker that will be reused for all tasks
    # This avoids recreating the environment ~10 times per worker
    from environment.trading_env import TradingEnvironment
    _worker_env = TradingEnvironment(**_worker_env_config)

    # Cache for reconstructed agents to avoid rebuilding same agent multiple times
    # Key: hash of agent_state, Value: DDPGAgent instance
    # Uses OrderedDict for LRU eviction - most recently used items are moved to end
    _worker_agent_cache = OrderedDict()


def _cache_agent(state_hash, agent):
    """
    Add an agent to the LRU cache with size limit enforcement.

    Uses OrderedDict for LRU semantics - entries are ordered by insertion/access time.
    When cache exceeds _WORKER_CACHE_MAX_SIZE, the oldest entries are evicted.

    Args:
        state_hash: Hash key for the agent state
        agent: DDPGAgent instance to cache
    """
    global _worker_agent_cache

    # Evict oldest entries if cache is at capacity
    while len(_worker_agent_cache) >= _WORKER_CACHE_MAX_SIZE:
        # popitem(last=False) removes the oldest (first) entry
        _worker_agent_cache.popitem(last=False)

    _worker_agent_cache[state_hash] = agent


def _get_cached_agent(state_hash):
    """
    Retrieve an agent from cache, updating LRU order if found.

    Args:
        state_hash: Hash key to look up

    Returns:
        DDPGAgent if found, None otherwise
    """
    global _worker_agent_cache

    if state_hash in _worker_agent_cache:
        # Move to end to mark as recently used (LRU semantics)
        _worker_agent_cache.move_to_end(state_hash)
        return _worker_agent_cache[state_hash]
    return None


def _run_episode_worker(args):
    """
    Worker function for parallel episode execution.

    OPTIMIZED: Reuses environment and caches agents per worker to minimize overhead.

    This function runs in a separate process, so it must:
    1. Reconstruct the agent from CPU state dicts (cached for reuse)
    2. Reuse the worker's environment instance (created once in initializer)
    3. Run the episode independently
    4. Write transitions directly to disk (parallel I/O)

    Args:
        args: Tuple of (agent_state, start_idx, end_idx, training, seed, buffer_storage_path, file_id_start)
              Note: env_config, env, and agent_cache are accessed from globals (set by initializer)

    Returns:
        Tuple of (fitness, episode_info, transition_file_paths)
    """
    global _worker_env_config, _worker_env, _worker_agent_cache
    agent_state, start_idx, end_idx, training, seed, buffer_storage_path, file_id_start = args

    # Set worker-specific seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # OPTIMIZATION: Check agent cache first before reconstructing
    # Create a hashable key from agent state (using actor weights as proxy)
    import hashlib
    actor_bytes = str(agent_state['actor']).encode()
    state_hash = hashlib.md5(actor_bytes).hexdigest()

    # Use LRU cache with size limit to prevent memory leaks during evolution
    agent = _get_cached_agent(state_hash)
    if agent is None:
        # Reconstruct agent from state dict (agents with CUDA tensors are not picklable)
        from models.ddpg_agent import DDPGAgent
        agent = DDPGAgent(agent_id=0)
        agent.actor.load_state_dict(agent_state['actor'])
        agent.critic.load_state_dict(agent_state['critic'])
        agent.actor.eval()
        agent.critic.eval()
        _cache_agent(state_hash, agent)

    # OPTIMIZATION: Reuse worker's environment instead of creating new one
    env = _worker_env

    # Run episode
    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
    env.set_training_mode(training)
    state, _ = env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

    cumulative_reward = 0.0
    steps = 0
    transition_file_paths = []

    # Write transitions directly to disk during episode (parallel I/O)
    # Note: We save to buffer regardless of training flag (noise) - this enables "Elite Demonstration"
    # where Elites contribute high-quality off-policy examples to help the Critic learn
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


def _run_validation_worker(args):
    """
    Worker function for parallel validation execution.

    OPTIMIZED: Reuses environment and caches agents per worker to minimize overhead.

    Validates a single agent across all validation slices (typically 7).
    Similar to _run_episode_worker but focused on validation-only tasks.

    Args:
        args: Tuple of (agent_state, validation_slices, quality_threshold, seed, use_penalized_median)
              Note: env_config, env, and agent_cache are accessed from globals
              use_penalized_median: If True, use Penalized Median (median - 0.5*std) instead of
                                    pessimistic (0.4*mean + 0.6*min) aggregation

    Returns:
        Dict with validation results (fitness, metrics, etc.)
    """
    global _worker_env_config, _worker_env, _worker_agent_cache
    agent_state, validation_slices, quality_threshold, seed, use_penalized_median = args

    # Set worker-specific seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # OPTIMIZATION: Check agent cache first before reconstructing
    import hashlib
    actor_bytes = str(agent_state['actor']).encode()
    state_hash = hashlib.md5(actor_bytes).hexdigest()

    # Use LRU cache with size limit to prevent memory leaks during evolution
    agent = _get_cached_agent(state_hash)
    if agent is None:
        # Reconstruct agent from state dict
        from models.ddpg_agent import DDPGAgent
        agent = DDPGAgent(agent_id=0)
        agent.actor.load_state_dict(agent_state['actor'])
        agent.critic.load_state_dict(agent_state['critic'])
        agent.actor.eval()
        agent.critic.eval()
        _cache_agent(state_hash, agent)
    else:
        # Ensure cached agent is in eval mode (safety check)
        agent.actor.eval()
        agent.critic.eval()

    # OPTIMIZATION: Reuse worker's environment instead of creating new one
    env = _worker_env

    # Run agent on all validation slices
    slice_results = []
    all_closed_trades = []

    for start_idx, end_idx, _ in validation_slices:
        # Run episode using TIME-TRAVEL BATCHING (same optimization as local mode)
        # This pre-fetches all states and does batch inference, then replays actions
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
        num_trading_steps = Config.TRADING_PERIOD_DAYS
        
        env.set_training_mode(False)
        env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

        # Pre-fetch all states and batch inference (chunked for memory efficiency)
        trading_states = env.get_batch_observations(start_idx, num_trading_steps)
        with torch.no_grad():
            # Process in chunks to avoid OOM (similar to local mode)
            chunk_size = 16
            action_chunks = []
            for i in range(0, num_trading_steps, chunk_size):
                chunk = trading_states[i:i+chunk_size]
                action_chunks.append(agent.select_actions_batch(chunk, add_noise=False))
            trading_actions = np.concatenate(action_chunks, axis=0)

        cumulative_reward = 0.0
        steps = 0

        # Fast replay trading period
        for i in range(num_trading_steps):
            _, reward, terminated, truncated, _ = env.step(trading_actions[i])
            cumulative_reward += reward
            steps += 1
            if terminated or truncated:
                break

        # Settlement period
        if not (terminated or truncated):
            dummy_action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM), dtype=np.float32)
            while True:
                _, reward, terminated, truncated, _ = env.step(dummy_action)
                cumulative_reward += reward
                steps += 1
                if terminated or truncated:
                    break

        # Get episode summary
        episode_info = env.get_episode_summary()
        episode_info['steps'] = steps

        # Calculate fitness for this slice
        fitness = float(cumulative_reward)

        # Apply zero trades penalty
        if episode_info['num_trades'] == 0:
            fitness -= episode_info['zero_trades_penalty']

        # Apply win rate bonus if applicable
        if episode_info['num_trades'] >= Config.WIN_RATE_BONUS_MIN_TRADES:
            win_rate_pct = episode_info['win_rate'] * 100.0
            if win_rate_pct > Config.WIN_RATE_BONUS_THRESHOLD:
                bonus = (win_rate_pct - Config.WIN_RATE_BONUS_THRESHOLD) ** 2
                fitness += bonus
                episode_info['win_rate_bonus'] = bonus
            else:
                episode_info['win_rate_bonus'] = 0.0
        else:
            episode_info['win_rate_bonus'] = 0.0

        # Apply zero-trades gradient (for agents that don't trade)
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

        # Collect closed trades
        if 'closed_trades' in episode_info and episode_info['closed_trades']:
            all_closed_trades.extend(episode_info['closed_trades'])

    # Calculate aggregated validation fitness
    fitness_scores = [result['fitness'] for result in slice_results]
    mean_score = np.mean(fitness_scores)
    min_score = np.min(fitness_scores)

    if use_penalized_median:
        # Penalized Median: median - (0.5 * std)
        # Rewards consistent performance, penalizes volatility
        median_score = np.median(fitness_scores)
        std_score = np.std(fitness_scores)
        validation_fitness = float(median_score - (0.5 * std_score))
    else:
        # Pessimistic: 0.4*mean + 0.6*min
        validation_fitness = (0.4 * mean_score) + (0.6 * min_score)

    # Aggregate metrics
    total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
    total_investment = sum([r['total_investment'] for r in slice_results])  # Legacy cumulative
    # Pooled ROI: sum of peak capitals (slices are parallel/independent scenarios)
    # This answers: "For every dollar of max drawdown capacity across all scenarios, how much profit?"
    total_peak_capital = sum([r['peak_capital_employed'] for r in slice_results])
    roi = (total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0

    total_wins = sum([r['num_wins'] for r in slice_results])
    total_losses = sum([r['num_losses'] for r in slice_results])
    total_trades = total_wins + total_losses
    global_win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0

    # Calculate quality metrics
    quality_count = 0
    quality_roi_sum = 0.0
    if quality_threshold is not None and all_closed_trades:
        for trade in all_closed_trades:
            gain_pct = trade.get('gain_pct', 0.0)
            if gain_pct >= quality_threshold:
                quality_count += 1
                quality_roi_sum += gain_pct

    quality_roi = (quality_roi_sum / quality_count) if quality_count > 0 else 0.0

    sample_trade = all_closed_trades[0] if all_closed_trades else None

    # Calculate expectancy from all closed trades
    # Expectancy = (Win Rate × Avg Win %) − (Loss Rate × Avg Loss %)
    if not all_closed_trades:
        expectancy = 0.0
    else:
        wins = [t['gain_pct'] for t in all_closed_trades if t['gain_pct'] > 0]
        losses = [abs(t['gain_pct']) for t in all_closed_trades if t['gain_pct'] <= 0]

        if not wins and not losses:
            expectancy = 0.0
        else:
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            win_rate_calc = len(wins) / len(all_closed_trades)
            loss_rate = 1.0 - win_rate_calc
            expectancy = (win_rate_calc * avg_win) - (loss_rate * avg_loss)

    return {
        'fitness': validation_fitness,
        'fitness_all_slices': fitness_scores,  # For debugging
        'fitness_mean': mean_score,
        'fitness_min': min_score,
        'roi': roi,
        'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),  # Mean trades per slice
        'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
        'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
        'avg_reward_per_trade': np.mean([r['avg_reward_per_trade'] for r in slice_results]),
        'raw_pnl': total_raw_pnl,  # Total raw P&L across slices
        'total_investment': total_investment,  # Total investment across slices
        'total_trades': total_trades,  # Total across all slices
        'win_rate': global_win_rate,
        'expectancy': expectancy,  # Expectancy metric: (Win Rate × Avg Win %) − (Loss Rate × Avg Loss %)
        'quality_count': quality_count,
        'quality_roi': quality_roi,
        'sample_trade': sample_trade
    }


class BreakthroughState(Enum):
    """
    State machine for Gauntlet Mode breakthrough validation.

    Transitions:
        NORMAL → DETECTION (spike detected)
        DETECTION → STABILIZATION (candidate locked in)
        STABILIZATION → GAUNTLET (stabilization complete)
        GAUNTLET → CONFIRMED (passed stress test) → NORMAL (ratchet applied, search for next)
        GAUNTLET → REJECTED (failed stress test) → NORMAL (continue searching)
    """
    NORMAL = "normal"  # Normal training, monitoring for spikes
    DETECTION = "detection"  # Potential breakthrough detected
    STABILIZATION = "stabilization"  # Locked on candidate, allowing convergence
    GAUNTLET = "gauntlet"  # Running rigorous stress test
    CONFIRMED = "confirmed"  # Breakthrough confirmed, apply ratchet
    REJECTED = "rejected"  # Candidate failed Gauntlet, back to normal


@dataclass
class BreakthroughCandidate:
    """
    Represents a candidate agent (or agents) for breakthrough validation.

    Attributes:
        agent: The best candidate DDPGAgent (for single-agent compatibility)
        agents: List of all agents that breached the threshold (quorum)
        agent_idx: Index of best agent
        agent_indices: Indices of all agents that breached
        spike_score: Best validation score that triggered detection
        spike_scores: Scores of all agents that breached
        detection_generation: Generation when spike was detected
        stabilization_start_gen: Generation when stabilization started
        gauntlet_score: Score achieved in Gauntlet (None until tested)
    """
    agent: 'DDPGAgent'
    agents: list['DDPGAgent']
    agent_idx: int
    agent_indices: list[int]
    spike_score: float
    spike_scores: list[float]
    detection_generation: int
    stabilization_start_gen: Optional[int] = None
    gauntlet_score: Optional[float] = None


class ERLTrainer:
    """
    Evolutionary Reinforcement Learning Trainer.
    Manages population, training, and evolution.
    """

    def __init__(self, data_loader: StockDataLoader, resume_run_name: str = None, enable_leverage: bool = False,
                 consistency_mode: bool = False, heroes_hof_dir: str = None, single_agent_path: str = None,
                 buffer_storage_path: str = None, reset_limit: bool = False, original_stdout=None, original_stderr=None,
                 multi_mode: bool = False, multi_roster: dict = None, maverick_mode: bool = False,
                 local_mode: bool = False):
        """
        Initialize ERL trainer.

        Args:
            data_loader: Loaded data with train/val splits
            resume_run_name: Optional wandb run name to resume from (e.g., "azure-thunder-123")
            enable_leverage: If True, enable leverage mode (replaces bottom 5 with top 5 HoF agents with 1.5x coefficients)
            consistency_mode: If True, evaluate with 5 episodes (sum) and loss magnification (see Config.CONSISTENCY_LOSS_MULTIPLIER)
            heroes_hof_dir: Path to Hall of Fame directory to load pre-trained agents from
            single_agent_path: Path to a specific Global50 agent for single-agent focused refinement
            buffer_storage_path: Optional path to existing buffer_storage folder to reuse (e.g., "checkpoints/run-123/buffer_storage")
            reset_limit: If True, reset the fallback generation counter to current generation on resume
            original_stdout: Original stdout before any redirection (for wandb console capture)
            original_stderr: Original stderr before any redirection (for wandb console capture)
            multi_mode: If True, enable multi-agent committee training mode
            multi_roster: Committee roster dict with 9 members (required if multi_mode=True)
            maverick_mode: If True, enable Maverick training mode (aggressive reward functions)
            local_mode: If True, use sequential evaluation/validation and serialize disk writes
        """
        self.data_loader = data_loader
        self.resume_run_name = resume_run_name
        self.enable_leverage = enable_leverage
        self.consistency_mode = consistency_mode
        self.heroes_hof_dir = heroes_hof_dir
        self.single_agent_path = single_agent_path
        self.single_agent_mode = single_agent_path is not None
        self.external_buffer_storage_path = buffer_storage_path  # Optional path to reuse existing buffer
        self.reset_limit = reset_limit  # Reset fallback counter on resume
        self.maverick_mode = maverick_mode  # Maverick training mode (aggressive reward functions)
        self.local_mode = local_mode  # Local mode: sequential execution and serialized disk writes

        # Multi-agent committee mode (sequential training of each member)
        self.multi_mode = multi_mode
        self.multi_roster = multi_roster
        if multi_mode:
            self.num_committee_members = Config.COMMITTEE_SIZE  # 9
            self.member_breakthroughs = [0] * self.num_committee_members  # Breakthroughs per member
            self.member_baselines = [0.0] * self.num_committee_members  # Original scores
            self.turnovers_completed = 0
            self.current_member_idx = 0  # Which committee member we're currently training
            self.multi_generation_offset = 0  # Track total generations across all members
            self.member_training_start_gen = 0  # Track when current member started training for warmup enforcement
            # Stuck detection for multi-mode (post-warmup)
            self.multi_parent_agent = None  # Reference to the parent agent for parent mutant injection
            self.multi_gens_since_improvement = 0  # Generations since last improvement (post-warmup)
            self.multi_best_score_for_member = float('-inf')  # Best score seen for current member
            # Phase tracking for maverick/non-maverick separation
            self.multi_phase = 'non_maverick'  # 'non_maverick' or 'maverick'
            self.non_maverick_members = []  # List of member indices that are non-maverick
            self.maverick_members = []  # List of member indices that are maverick
            self.current_non_maverick_idx = 0  # Index into non_maverick_members list
            self.current_maverick_idx = 0  # Index into maverick_members list
            self.non_maverick_turnovers_per_agent = []  # Track turnovers per non-maverick agent
            self.maverick_turnovers_per_agent = []  # Track turnovers per maverick agent

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

        # Initialize population (multi-mode uses standard size, trains one member at a time)
        # Local mode uses smaller population (32 vs 96) for faster iteration
        pop_size = Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE
        print(f"Initializing population of {pop_size} agents{'  (local mode)' if self.local_mode else ''}...")
        self.population = [DDPGAgent(agent_id=i) for i in range(pop_size)]

        # Training range (excludes validation set)
        self.train_start_idx = Config.CONTEXT_WINDOW_DAYS
        self.train_end_idx = len(data_loader.train_indices)

        # Validation range (for walk-forward validation during training)
        # This is separate from training data to ensure genuine out-of-sample validation
        self.val_start_idx = len(data_loader.train_indices)
        self.val_end_idx = self.val_start_idx + len(data_loader.val_indices)

        # Walk-forward validation slices (generated per generation)
        # Each generation uses 10 validation slices from validation set (4 from quarters + 3 straddling + 3 random)
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
                                "population_size": Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE,
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
                                "population_size": Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE,
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

                    # Suppress W&B step order warnings when resuming
                    # (W&B's internal step counter may be ahead of our resume point)
                    import warnings
                    warnings.filterwarnings(
                        'ignore',
                        message='.*step.*less than the current step.*',
                        module='wandb.*'
                    )
                else:
                    # New training run
                    print("--- Initializing new W&B run (main.py mode) ---")
                    wandb.init(
                        project="eigen2-self",
                        #name=f"erl-{Config.NUM_GENERATIONS}gen",
                        config={
                            "population_size": Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE,
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

        # Initialize Global Hall of Fame (cross-run top 50 tracking)
        league_rules = LeagueRules(
            context_window_days=Config.CONTEXT_WINDOW_DAYS
        )
        self.global_hof = GlobalHallOfFame(
            cloud_sync=self.cloud_sync,
            run_name=wandb.run.name,
            league_rules=league_rules,
            checkpoint_dir=self.checkpoint_dir,
            disable_global50=False  # Set to True to disable Global 50 for debugging
        )

        # Create replay buffer with storage INSIDE checkpoint directory
        # This ensures buffer files are synced to cloud along with checkpoints
        # If external buffer path provided (via --buffer), use it to copy/load existing buffer
        if self.external_buffer_storage_path:
            # User provided an external buffer path - use it to initialize buffer
            from pathlib import Path
            external_path = Path(self.external_buffer_storage_path)

            if external_path.exists():
                print(f"Loading external buffer from: {external_path}")

                # New buffer will be stored in current run's checkpoint dir
                new_buffer_storage_path = str(self.checkpoint_dir / "buffer_storage")

                # Try to load buffer metadata from external path
                # Check if there's a replay_buffer.pkl file in parent directory
                metadata_path = external_path.parent / "replay_buffer.pkl"

                if metadata_path.exists():
                    print(f"  Found buffer metadata: {metadata_path}")
                    try:
                        # Load buffer metadata, but override storage path to point to external location
                        self.replay_buffer = OnDiskReplayBuffer.load(
                            str(metadata_path),
                            storage_path_override=str(external_path)
                        )
                        print(f"  ✓ Loaded {len(self.replay_buffer)} transitions from external buffer")

                        # Update the buffer's storage path to the new location for future writes
                        # This way, new transitions will be added to the current run's buffer_storage
                        self.replay_buffer.storage_path = Path(new_buffer_storage_path)
                        self.replay_buffer.storage_path.mkdir(parents=True, exist_ok=True)
                        print(f"  New transitions will be saved to: {new_buffer_storage_path}")

                        # Enable gradual migration: as old transitions are evicted, delete them from external source
                        self.replay_buffer.set_external_source_for_migration(str(external_path))
                    except Exception as e:
                        print(f"  ⚠ Error loading buffer metadata: {e}")
                        print(f"  Creating new buffer with storage in: {new_buffer_storage_path}")
                        self.replay_buffer = OnDiskReplayBuffer(
                            capacity=Config.BUFFER_SIZE,
                            storage_path=new_buffer_storage_path
                        )
                else:
                    print(f"  ⚠ No replay_buffer.pkl found at {metadata_path}")
                    print(f"  Creating new buffer (will still use external files if they exist)")
                    # Create buffer pointing to external storage
                    # This allows reusing the files even without metadata
                    self.replay_buffer = OnDiskReplayBuffer(
                        capacity=Config.BUFFER_SIZE,
                        storage_path=str(external_path)
                    )
                    # Update to new path for future writes
                    new_buffer_storage_path = str(self.checkpoint_dir / "buffer_storage")
                    self.replay_buffer.storage_path = Path(new_buffer_storage_path)
                    self.replay_buffer.storage_path.mkdir(parents=True, exist_ok=True)
                    print(f"  New transitions will be saved to: {new_buffer_storage_path}")

                    # Enable gradual migration: as old transitions are evicted, delete them from external source
                    self.replay_buffer.set_external_source_for_migration(str(external_path))
            else:
                print(f"⚠ External buffer path does not exist: {external_path}")
                print(f"  Creating new buffer instead")
                buffer_storage_path = str(self.checkpoint_dir / "buffer_storage")
                self.replay_buffer = OnDiskReplayBuffer(
                    capacity=Config.BUFFER_SIZE,
                    storage_path=buffer_storage_path
                )
        else:
            # No external buffer - create new buffer in checkpoint directory
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
        # Use hashlib instead of hash() for deterministic cross-process reproducibility
        # (Python's hash() is randomized per process due to PYTHONHASHSEED)
        import hashlib
        run_id_hash = int(hashlib.sha256(wandb.run.id.encode()).hexdigest(), 16) % (2**32)
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

        # Quality threshold for QR calculation (trades with gain_pct >= threshold)
        # Updated each generation based on HoF median ROI
        self.quality_threshold = Config.ROI_QUALITY_THRESHOLD  # Default: 7.5%

        # Statistics
        self.fitness_history = []
        self.generation_times = []
        self.validation_fitness_history = []  # Track validation fitness for plateau detection

        # Adaptive mutation parameters
        self.plateau_threshold = 0.02  # Consider plateau if improvement < 2% over window
        self.plateau_window = 3  # Number of generations to check for plateau
        # Use consistency-specific mutation rate if in consistency mode, otherwise use normal rate
        base_mutation_rate = Config.MUTATION_RATE_CONSISTENCY if self.consistency_mode else Config.MUTATION_RATE
        self.base_mutation_rate = base_mutation_rate
        self.base_mutation_std = Config.MUTATION_STD
        self.current_mutation_rate = base_mutation_rate
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

        # Gauntlet Mode - Breakthrough state machine
        self.gauntlet_mode_enabled = Config.GAUNTLET_MODE_ENABLED
        self.breakthrough_state = BreakthroughState.NORMAL
        self.breakthrough_candidate: Optional[BreakthroughCandidate] = None
        self.confirmed_baseline = 0.0  # Ratcheting baseline - only updated after Gauntlet confirmation
        self.confirmed_breakthroughs = 0  # Count of confirmed breakthroughs
        self.breakthrough_history = []  # Track breakthrough events with timestamps
        self.stabilization_generations_elapsed = 0  # Counter for stabilization phase
        self.initial_single_baseline = 0.0  # Original baseline for single-agent mode (for final summary)

        # Candidate queue and tracking (fixes Ghost Loop and Winner-Takes-All)
        # Only used in consistency mode with heroes, where pre-trained agents are deterministic
        self.use_candidate_queue = (self.consistency_mode and heroes_hof_dir is not None)
        self.candidate_queue = []  # Queue of all agents that breached threshold (sorted by fitness, descending)
        self.tested_candidate_indices = set()  # Set of agent indices that have been tested in Gauntlet
        self.pending_baseline_update = None  # Store baseline update until all candidates exhausted

        # Global50 injection pool (for heroes+consistency mode)
        # Instead of seeding HoF with Global50 agents (which can poison the HoF),
        # we inject mutated Global50 agents into the population until first breakthrough
        self.global50_injection_pool = []  # Stores loaded Global50 agents for injection
        self.global50_injection_count = 20  # Number of Global50 agents to inject per generation

        # Breakthrough threshold based on mode
        self.breakthrough_threshold = (Config.BREAKTHROUGH_THRESHOLD_CONSISTENCY
                                      if self.consistency_mode
                                      else Config.BREAKTHROUGH_THRESHOLD_NORMAL)

        # Breakthrough quorum based on mode
        self.breakthrough_quorum = (Config.BREAKTHROUGH_QUORUM_CONSISTENCY
                                   if self.consistency_mode
                                   else Config.BREAKTHROUGH_QUORUM_NORMAL)

        # Target breakthroughs based on mode
        self.target_breakthroughs = (Config.TARGET_BREAKTHROUGHS_CONSISTENCY
                                    if self.consistency_mode
                                    else Config.TARGET_BREAKTHROUGHS_NORMAL)

        # Hall of Fame turnover tracking (for consistency mode)
        self.hof_turnover_count = 0  # Number of complete HoF turnovers
        self.hof_current_median = None  # Current median ROI baseline (updated each turnover)
        self.target_hof_turnovers = Config.TARGET_HOF_TURNOVERS  # Target number of turnovers
        self.generation_at_last_turnover = 0  # Reset fallback counter on each turnover (consistency mode)

        # Maverick mode tracking (aggressive training for committee diversity)
        self._maverick_goal_achieved = False  # Set True when Maverick reaches target rank
        self._maverick_final_rank = -1  # Rank achieved when goal was met

        if self.gauntlet_mode_enabled:
            print(f"\n🎯 Gauntlet Mode ENABLED")
            print(f"  Breakthrough threshold: {self.breakthrough_threshold*100:.0f}% improvement")
            print(f"  Breakthrough quorum: {self.breakthrough_quorum} agents must breach simultaneously")
            print(f"  Stabilization: {Config.STABILIZATION_GENERATIONS} generations")
            print(f"  Gauntlet slices: {Config.GAUNTLET_NUM_SLICES} (vs 7 normal)")
            print(f"  Target breakthroughs: {self.target_breakthroughs}")
            print(f"  Mode: {'Consistency' if self.consistency_mode else 'Normal'}")

        if self.maverick_mode:
            print(f"\n🔥 Maverick Mode ENABLED")
            print(f"  Agent type: Maverick (aggressive reward functions)")
            print(f"  Maverick cap: {Config.MAVERICK_CAP} max in Global 50")
            print(f"  Target rank: #{Config.MAVERICK_TARGET_RANK} or higher")
            print(f"  Training continues until target rank achieved")

        # Initialize resource tracker
        self.resource_tracker = ResourceTracker(disk_path="/workspace")

        print(f"Training: days {self.train_start_idx}-{self.train_end_idx}, "
              f"Validation: days {self.val_start_idx}-{self.val_end_idx}")
        print(f"Walk-forward: 10 validation slices/generation (4 from quarters + 3 straddling + 3 random)")

        # Create persistent environment (reused across episodes to prevent memory leaks)
        print("Initializing environment...")
        if self.consistency_mode:
            print(f"  Consistency mode enabled: {Config.CONSISTENCY_LOSS_MULTIPLIER}x loss magnification")
            print(f"  Mutation rate: {self.base_mutation_rate} (consistency mode)")
        else:
            print(f"  Mutation rate: {self.base_mutation_rate} (normal mode)")
        self.eval_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            start_idx=self.train_start_idx,
            end_idx=self.train_end_idx,
            trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            data_array_full=self.data_loader.data_array_full,
            consistency_mode=self.consistency_mode,
            maverick_mode=self.maverick_mode
        )

        # Initialize shared memory for parallel worker data (eliminates serialization overhead)
        # This creates shared memory blocks for data arrays that workers can access directly
        if not self.local_mode:
            print("Initializing shared memory for parallel workers...")
            self._init_shared_memory()
        else:
            print("Local mode: Skipping shared memory (CPU-optimized evaluation)")
            self._shared_memory_names = {}  # Empty dict for compatibility

        # Initialize LocalEvaluator for CPU-optimized local mode
        # This must be initialized after eval_env but before checkpoint loading
        self.local_evaluator = None
        if self.local_mode:
            print("Initializing LocalEvaluator for CPU-optimized execution...")
            self.local_evaluator = LocalEvaluator(self)

        # Load heroes from Hall of Fame if specified (must happen after env creation, before checkpoint load)
        if self.heroes_hof_dir:
            self.load_heroes_from_hof()

        # Load single agent for focused refinement (mutually exclusive with heroes mode)
        if self.single_agent_mode:
            self.load_single_agent(self.single_agent_path)

        # Load multi-agent committee members (mutually exclusive with single/heroes mode)
        if self.multi_mode:
            self.load_multi_agents()

        # Automatically load checkpoint if resuming
        if self.resume_run_name:
            print("\n" + "="*60)
            print("Loading checkpoint for resume...")
            print("="*60)
            self.load_checkpoint()
            
            # Sync wandb step counter to match generation (0-indexed)
            # After loading checkpoint, start_generation is set to checkpoint_gen + 1 (next generation to run)
            # We need to set wandb's step to start_generation - 1 (last completed generation)
            # so that the next log with step=start_generation will be accepted
            if wandb.run is not None:
                # Set wandb step to the last completed generation (start_generation - 1)
                # This ensures the next log with step=start_generation will be accepted
                wandb.run.step = self.start_generation - 1

            # Refresh global50 entries after resume to get latest view from cloud
            print("\nRefreshing Global 50 entries...")
            self.global_hof.refresh()

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

    def _init_shared_memory(self):
        """
        Initialize shared memory blocks for data arrays used by parallel workers.

        This eliminates the serialization overhead when spawning workers by storing
        the large numpy arrays in shared memory. Workers can then access the data
        directly without copying (zero-copy access).

        Creates:
        - shm_data_array: Shared memory for reduced feature array (for observations)
        - shm_data_array_full: Shared memory for full feature array (for rewards)
        """
        # Store references to prevent garbage collection
        self._shm_blocks = []

        # Create shared memory for data_array (reduced features for model)
        data_array = self.data_loader.data_array
        self._shm_data_array = shared_memory.SharedMemory(
            create=True,
            size=data_array.nbytes
        )
        # Copy data into shared memory
        shm_data_view = np.ndarray(
            data_array.shape,
            dtype=data_array.dtype,
            buffer=self._shm_data_array.buf
        )
        shm_data_view[:] = data_array[:]
        self._shm_blocks.append(self._shm_data_array)
        # Register with resource tracker for crash-safe cleanup (prevents zombie segments)
        # Note: resource_tracker is POSIX-only, not available on Windows
        if sys.platform != 'win32':
            resource_tracker.register(self._shm_data_array.name, "shared_memory")

        # Create shared memory for data_array_full (full features for environment)
        data_array_full = self.data_loader.data_array_full
        self._shm_data_array_full = shared_memory.SharedMemory(
            create=True,
            size=data_array_full.nbytes
        )
        # Copy data into shared memory
        shm_full_view = np.ndarray(
            data_array_full.shape,
            dtype=data_array_full.dtype,
            buffer=self._shm_data_array_full.buf
        )
        shm_full_view[:] = data_array_full[:]
        self._shm_blocks.append(self._shm_data_array_full)
        # Register with resource tracker for crash-safe cleanup (prevents zombie segments)
        # Note: resource_tracker is POSIX-only, not available on Windows
        if sys.platform != 'win32':
            resource_tracker.register(self._shm_data_array_full.name, "shared_memory")

        # Store metadata for workers
        self._shm_metadata = {
            'shm_data_array_name': self._shm_data_array.name,
            'data_array_shape': data_array.shape,
            'data_array_dtype': str(data_array.dtype),
            'shm_data_array_full_name': self._shm_data_array_full.name,
            'data_array_full_shape': data_array_full.shape,
            'data_array_full_dtype': str(data_array_full.dtype),
        }

        data_mb = data_array.nbytes / (1024 * 1024)
        full_mb = data_array_full.nbytes / (1024 * 1024)
        print(f"  ✓ Shared memory initialized: {data_mb:.1f}MB + {full_mb:.1f}MB = {data_mb + full_mb:.1f}MB total")

    def _cleanup_shared_memory(self):
        """
        Clean up shared memory blocks when trainer is done.

        IMPORTANT: Must be called before trainer exits to avoid memory leaks.
        Shared memory persists beyond process lifetime if not explicitly freed.
        """
        if hasattr(self, '_shm_blocks'):
            for shm in self._shm_blocks:
                try:
                    shm.close()
                    shm.unlink()  # Remove the shared memory block
                except Exception as e:
                    print(f"⚠ Error cleaning up shared memory: {e}")
            self._shm_blocks = []
            print("✓ Shared memory cleaned up")

    def __del__(self):
        """Destructor - ensure shared memory is cleaned up even if train() doesn't complete."""
        try:
            self._cleanup_shared_memory()
        except Exception:
            pass  # Ignore errors during destruction

    def _get_shared_env_config(self, start_idx: int, end_idx: int, trading_end_idx: int,
                               is_training: bool = True, gauntlet_mode: bool = False) -> dict:
        """
        Build environment config dict using shared memory references.

        This config is passed to worker initializers and contains shared memory
        names instead of actual arrays, eliminating serialization overhead.

        Args:
            start_idx: Episode start index
            end_idx: Episode end index
            trading_end_idx: Last day to open new positions
            is_training: Whether environment is in training mode
            gauntlet_mode: Whether to use soft zero-trades penalty (for stabilization/gauntlet)

        Returns:
            Dict with shared memory references and other config
        """
        return {
            # Shared memory references (no serialization needed)
            **self._shm_metadata,
            # Small data that must be pickled (but fast)
            'dates': self.data_loader.dates,
            'normalization_stats': self.normalization_stats,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'trading_end_idx': trading_end_idx,
            'is_training': is_training,
            'consistency_mode': self.consistency_mode,
            'gauntlet_mode': gauntlet_mode,
            'maverick_mode': self.maverick_mode
        }

    def _create_dataloader(self):
        """
        Create or recreate DataLoader for the replay buffer.
        Called during __init__ and after loading checkpoints.
        """
        import gc

        # Explicitly shutdown old DataLoader workers before creating new one
        if hasattr(self, 'batch_iterator') and self.batch_iterator is not None:
            try:
                del self.batch_iterator  # Delete iterator first
            except:
                pass

        if hasattr(self, 'replay_dataloader') and self.replay_dataloader is not None:
            try:
                # Force shutdown of persistent workers
                if hasattr(self.replay_dataloader, '_iterator'):
                    if self.replay_dataloader._iterator is not None:
                        self.replay_dataloader._iterator._shutdown_workers()
                del self.replay_dataloader
            except:
                pass

        # Force garbage collection to ensure workers are cleaned up
        gc.collect()

        # Set batch size and workers based on mode
        # Local mode: FORCE num_workers=0 to prevent "Stale Buffer" issue on Windows
        # With workers > 0, worker processes get a COPY of self.buffer at iter() time.
        # They never see new files added in Generation 2+, causing infinite sleep/starvation.
        # With num_workers=0, main process runs __iter__ directly and always sees updates.
        if self.local_mode:
            self.replay_buffer.training_batch_size = Config.LOCAL_BATCH_SIZE
            num_workers = Config.LOCAL_NUM_DATALOADER_WORKERS  # Should be 0 for local mode
            print(f"Creating DataLoader with {num_workers} workers (batch_size={Config.LOCAL_BATCH_SIZE} for local mode)...")
        else:
            self.replay_buffer.training_batch_size = Config.BATCH_SIZE
            num_workers = Config.NUM_DATALOADER_WORKERS
            print(f"Creating DataLoader with {num_workers} background workers...")

        # num_workers=0 runs in main process - different options required
        if num_workers == 0:
            self.replay_dataloader = DataLoader(
                self.replay_buffer,
                batch_size=None,  # Already batched by __iter__
                num_workers=0,
                pin_memory=True  # Faster GPU transfer
            )
        else:
            self.replay_dataloader = DataLoader(
                self.replay_buffer,
                batch_size=None,  # Already batched by __iter__
                num_workers=num_workers,
                pin_memory=True,  # Faster GPU transfer
                prefetch_factor=2,  # Each worker prefetches 2 batches ahead
                persistent_workers=True  # Keep workers alive between epochs
            )
        # Reset iterator when creating new DataLoader
        self.batch_iterator = None

    def load_heroes_from_hof(self):
        """
        Load agents from Hall of Fame directory(ies), combining multiple sources if needed.

        Supports multiple directories separated by '|' to aggregate agents from:
        1. Current context window (e.g., cw151)
        2. Fallback context window (e.g., cw504)
        3. Legacy directories

        Fills any remaining population slots with random agents (not clones).
        """
        from pathlib import Path
        import glob

        # Support multiple directories separated by '|'
        hof_dirs = self.heroes_hof_dir.split('|') if '|' in self.heroes_hof_dir else [self.heroes_hof_dir]

        print("\n" + "="*60)
        print("HEROES MODE: Loading pre-trained agents from Hall of Fame")
        print("="*60)

        # Load agents from all specified directories
        all_agent_files = []
        for hof_dir_str in hof_dirs:
            hof_dir = Path(hof_dir_str.strip())
            hof_subdir = hof_dir / "hall_of_fame"

            # Check if the directory exists
            if not hof_dir.exists():
                print(f"  ⚠ Skipping {hof_dir}: directory not found")
                continue

            # Determine where agent files are (check both agents/ and hall_of_fame/)
            agent_subdirs = [
                hof_dir / "agents",         # Standard: global50/cw151/agents/
                hof_subdir,                 # Legacy: checkpoints/run/hall_of_fame/
                hof_dir                     # Root: global50/cw151/
            ]

            for agent_dir in agent_subdirs:
                if agent_dir.exists():
                    agent_files = sorted(glob.glob(str(agent_dir / "*.pth")))
                    if agent_files:
                        print(f"  ✓ {hof_dir}: found {len(agent_files)} agents in {agent_dir.name}/")
                        all_agent_files.extend(agent_files)
                        break  # Found agents in this directory, move to next hof_dir

        if not all_agent_files:
            print("! No agent files (.pth) found in any specified directory")
            print("  Continuing with random initialization.")
            return

        print(f"\nTotal agents found: {len(all_agent_files)}")

        # Load all agents
        loaded_agents = []
        for i, agent_file in enumerate(all_agent_files):
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

        # Prepare environment config using shared memory (eliminates serialization overhead)
        env_config = self._get_shared_env_config(
            start_idx=self.train_start_idx,
            end_idx=self.train_end_idx,
            trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            is_training=True
        )

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
        num_workers = min(mp.cpu_count() - 1, Config.EVAL_NUM_WORKERS)
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

        # Sort by fitness (descending) and select top agents
        # Respect local mode's smaller population size
        hero_fitness.sort(key=lambda x: x[0], reverse=True)
        pop_size = Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE
        num_to_select = min(pop_size, len(hero_fitness))

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

        # Fill remaining slots with random agents if needed (NOT clones)
        num_random_needed = pop_size - len(selected_heroes)
        if num_random_needed > 0:
            print(f"\n  Filling {num_random_needed} remaining slots with random agents")
            for i in range(num_random_needed):
                random_agent = DDPGAgent(agent_id=len(selected_heroes))
                selected_heroes.append(random_agent)

        # Replace population
        old_population = self.population
        self.population = selected_heroes

        # Clean up old population
        for agent in old_population:
            del agent
        gc.collect()

        print(f"\nPopulation replaced with {len(self.population)} heroes")

        # --- Validate heroes and populate Hall of Fame with ROI ---
        # Heroes are added directly to HoF (ensures median ROI is set from start)
        # HoF tracks top 10 agents ever seen; Global 50 is the "repository of excellence"

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
                'expectancy': val_results.get('expectancy', 0.0),
                'quality_count': val_results.get('quality_count', 0),
                'total_trades': val_results.get('total_trades', 0)
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
            train_fitness = 0.0  # Heroes don't have training fitness
            quality_count = result.get('quality_count', 0)
            total_trades = result.get('total_trades', 0)
            val_fitness = combined_score  # Use combined fitness as validation fitness
            base_combined_fitness = combined_score
            hof_candidates.append((self.population[agent_idx], combined_score, agent_idx, agent_roi, agent_expectancy,
                                 train_fitness, quality_count, total_trades, val_fitness, base_combined_fitness))

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
        print(f"  Elite: {Config.HEROES_ELITE_FRAC * 100:.1f}% ({int(pop_size * Config.HEROES_ELITE_FRAC)} agents)")
        print(f"  Offspring: {Config.HEROES_OFFSPRING_FRAC * 100:.1f}% ({int(pop_size * Config.HEROES_OFFSPRING_FRAC)} agents)")
        mutant_frac = 1.0 - Config.HEROES_ELITE_FRAC - Config.HEROES_OFFSPRING_FRAC
        print(f"  Mutants: {mutant_frac * 100:.1f}% ({int(pop_size * mutant_frac)} agents)")
        print("="*60 + "\n")

    def load_single_agent(self, agent_path: str, stored_gauntlet_score: float = None):
        """
        Load a single Global50 agent and initialize population for focused refinement.

        Population composition (from Config):
        - SINGLE_CLONE_FRAC: Pure clones of the agent (50%)
        - SINGLE_NORMAL_MUTATION_FRAC: Clones with normal mutation applied (25%)
        - SINGLE_PLATEAU_MUTATION_FRAC: Clones with plateau (1.5x) mutation applied (25%)

        Baseline is min(stored_score, evaluated_score) for safety.

        Args:
            agent_path: Path to the agent .pth file
            stored_gauntlet_score: Optional stored gauntlet score from global50.json
        """
        print(f"\n{'='*60}")
        print(f"🎯 SINGLE AGENT MODE - Loading and Evaluating")
        print(f"{'='*60}")

        # 1. Load the agent
        source_agent = DDPGAgent(agent_id=0)
        source_agent.load(agent_path)
        print(f"  Loaded: {Path(agent_path).name}")

        # 2. Evaluate to establish baseline using consistency-mode evaluation
        evaluated_score = self._evaluate_single_agent_for_baseline(source_agent)
        print(f"  Evaluated score: {evaluated_score:.2f}")

        if stored_gauntlet_score is not None:
            print(f"  Stored score: {stored_gauntlet_score:.2f}")
            baseline = min(stored_gauntlet_score, evaluated_score)
            print(f"  Baseline (min): {baseline:.2f}")
        else:
            baseline = evaluated_score
            print(f"  Baseline: {baseline:.2f}")

        # 3. Set confirmed baseline for breakthrough detection
        self.confirmed_baseline = baseline
        self.initial_single_baseline = baseline  # Store original for final summary

        # 4. Initialize population with clones + mutations
        # Respect local mode's smaller population size
        pop_size = Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE
        num_clones = int(pop_size * Config.SINGLE_CLONE_FRAC)
        num_normal_mutants = int(pop_size * Config.SINGLE_NORMAL_MUTATION_FRAC)
        num_plateau_mutants = pop_size - num_clones - num_normal_mutants

        print(f"\n  Population initialization:")
        print(f"    Pure clones: {num_clones} ({Config.SINGLE_CLONE_FRAC*100:.0f}%)")
        print(f"    Normal mutation: {num_normal_mutants} ({Config.SINGLE_NORMAL_MUTATION_FRAC*100:.0f}%)")
        print(f"    Plateau mutation: {num_plateau_mutants} ({Config.SINGLE_PLATEAU_MUTATION_FRAC*100:.0f}%)")

        new_population = []

        # Pure clones
        for i in range(num_clones):
            clone = source_agent.clone()
            clone.agent_id = i
            clone.is_elite = True  # Mark as elite initially
            new_population.append(clone)

        # Normal mutation clones
        base_mutation_rate = Config.MUTATION_RATE_CONSISTENCY
        base_mutation_std = Config.MUTATION_STD
        for i in range(num_normal_mutants):
            mutated_clone = mutate(source_agent, mutation_rate=base_mutation_rate, mutation_std=base_mutation_std)
            mutated_clone.agent_id = num_clones + i
            mutated_clone.is_elite = False
            new_population.append(mutated_clone)

        # Plateau mutation clones (1.5x mutation)
        plateau_mutation_rate = min(base_mutation_rate * 1.5, self.max_mutation_rate)
        plateau_mutation_std = min(base_mutation_std * 1.5, self.max_mutation_std)
        for i in range(num_plateau_mutants):
            mutated_clone = mutate(source_agent, mutation_rate=plateau_mutation_rate, mutation_std=plateau_mutation_std)
            mutated_clone.agent_id = num_clones + num_normal_mutants + i
            mutated_clone.is_elite = False
            new_population.append(mutated_clone)

        self.population = new_population
        print(f"  Population initialized: {len(self.population)} agents")

        # 5. Override target breakthroughs for single mode
        self.target_breakthroughs = Config.SINGLE_TARGET_BREAKTHROUGHS

        print(f"\n  Training configuration:")
        print(f"    Stabilization: {Config.SINGLE_STABILIZATION_GENERATIONS} generations")
        print(f"    Target breakthroughs: {Config.SINGLE_TARGET_BREAKTHROUGHS}")
        print(f"    Breakthrough threshold: 5%")
        print(f"    Fallback timeout: {Config.MAX_GENERATIONS_GAUNTLET} generations")
        print("="*60 + "\n")

        return baseline

    def load_multi_agents(self):
        """
        Initialize multi-agent sequential training mode.

        This evaluates all 9 committee members to establish baselines,
        then loads the first member for training. Members are trained
        sequentially - one breakthrough per member, rotating until
        all achieve a turnover.

        Uses same population composition as --single mode (96 agents).
        """
        from committee import get_agent_filepath

        print(f"\n{'='*60}")
        print(f"🎯 MULTI-AGENT MODE - Sequential Committee Training")
        print(f"{'='*60}")

        members = self.multi_roster['members']
        context_window = self.multi_roster['context_window_days']

        # Re-evaluate all members with ROI adjustment enabled to establish proper baselines
        # This is critical because stored gauntlet scores don't include ROI adjustment
        print(f"\nRe-evaluating {len(members)} committee members with ROI adjustment for baselines...")

        # Temporarily create a single-agent population for validation
        temp_population = []

        for member_idx, member in enumerate(members):
            agent_path = get_agent_filepath(member, context_window)

            if not agent_path.exists():
                raise FileNotFoundError(f"Committee member agent not found: {agent_path}")

            # Load member agent
            agent = DDPGAgent(agent_id=member_idx)
            agent.load(str(agent_path))
            temp_population.append(agent)

        # Save current population and replace with committee members
        saved_population = self.population
        self.population = temp_population

        # Generate validation slices before running validation
        self.current_generation_val_slices = self.generate_validation_slices()
        self.val_slice_hash = self._hash_validation_slices(self.current_generation_val_slices)

        # Run full validation to establish self-referential baselines
        # Note: validate_population_parallel returns raw validation metrics
        if self.local_mode:
            validation_results = self.local_evaluator.validate_population(quality_threshold=Config.ROI_QUALITY_THRESHOLD)
        else:
            validation_results = self.validate_population_parallel(quality_threshold=Config.ROI_QUALITY_THRESHOLD)

        # Initialize storage for each member's specific starting ROI
        # This allows each member to compete against THEMSELVES, not a global median
        self.member_starting_rois = [0.0] * self.num_committee_members

        # Post-process validation results to add combined_fitness
        # CRITICAL: Each member's baseline is SELF-REFERENTIAL (ROI adjustment = 0 initially)
        # This ensures every member starts on equal footing: needing to beat THEIR OWN previous best
        for member_idx, result in enumerate(validation_results):
            val_fitness = result['fitness']
            train_fitness = 0.0  # No training fitness for initial multi-agent eval
            agent_roi = result.get('roi', 0.0)
            quality_count = result.get('quality_count', 0)

            # 1. Capture the agent's specific starting ROI
            self.member_starting_rois[member_idx] = agent_roi

            # 2. Calculate Base Fitness
            # BUGFIX: Clamp val_fitness to 0 if agent isn't trading (val_fitness = -500)
            # Without this, a non-trading agent gets baseline of -500, making any improvement trivial
            clamped_val_fitness = max(0.0, val_fitness)
            base_combined_fitness = clamped_val_fitness + min(0.0, train_fitness)

            # 3. Calculate ROI adjustment using the SAME formula as training
            # This ensures baseline and training scores are comparable
            # Formula: |base_fitness| * multiplier * (agent_roi - hurdle_roi) / 100
            # For baseline, hurdle_roi = agent's own ROI, so adjustment = 0
            # But we compute it explicitly to match the training path
            hurdle_roi = agent_roi  # Self-referential: competing against own starting ROI
            min_trades_threshold = Config.ROI_CONFIDENCE_MIN_TRADES_CONSISTENCY
            confidence_factor = min(1.0, quality_count / min_trades_threshold) if min_trades_threshold > 0 else 1.0
            roi_adjustment = abs(base_combined_fitness) * Config.ROI_ADJUSTMENT_MULTIPLIER * (agent_roi - hurdle_roi) / 100.0
            roi_adjustment = roi_adjustment * confidence_factor  # = 0 since agent_roi == hurdle_roi

            # 4. Set the baseline
            combined_fitness = base_combined_fitness + roi_adjustment

            # Store in result
            result['combined_fitness'] = combined_fitness
            result['base_combined_fitness'] = base_combined_fitness
            result['roi_adjustment'] = roi_adjustment

            # Store the baseline
            self.member_baselines[member_idx] = combined_fitness

            member = members[member_idx]
            print(f"  Member {member_idx}: {member['run_name']}_{member['agent_id']} "
                  f"ROI={agent_roi:.2f}% -> Baseline={combined_fitness:.2f} (self-referential)")

        # Restore population (will be replaced again when loading first member)
        self.population = saved_population

        # Clean up temporary agents
        for agent in temp_population:
            del agent

        print(f"\n✓ All {len(members)} members evaluated")
        print(f"  Target turnovers: {Config.MULTI_TARGET_TURNOVERS}")
        print(f"  Breakthrough threshold: {Config.MULTI_BREAKTHROUGH_THRESHOLD * 100:.0f}%")
        
        # Separate members into non-maverick and maverick lists
        self.non_maverick_members = []
        self.maverick_members = []
        
        for member_idx, member in enumerate(members):
            is_maverick = member.get('is_maverick', False)
            if is_maverick:
                self.maverick_members.append(member_idx)
            else:
                self.non_maverick_members.append(member_idx)
        
        # Validate: at least one maverick in committee (user requirement)
        if len(self.maverick_members) == 0:
            raise ValueError(
                "ERROR: No maverick agents found in committee. "
                "At least one maverick agent is required in the committee."
            )
        
        # If all members are mavericks, skip non-maverick phase
        if len(self.non_maverick_members) == 0:
            print(f"\n⚠ All {len(members)} members are mavericks - skipping non-maverick phase")
            self.multi_phase = 'maverick'
            self.maverick_mode = True
            # Initialize maverick turnovers tracking
            self.maverick_turnovers_per_agent = [0] * len(self.maverick_members)
            self.current_maverick_idx = 0
            print(f"  Training sequence: 3 turnovers per maverick agent, sequentially")
        else:
            # Start with non-maverick phase
            self.multi_phase = 'non_maverick'
            self.maverick_mode = False
            # Initialize non-maverick turnovers tracking
            self.non_maverick_turnovers_per_agent = [0] * len(self.non_maverick_members)
            self.current_non_maverick_idx = 0
            print(f"  Non-maverick members: {len(self.non_maverick_members)}")
            print(f"  Maverick members: {len(self.maverick_members)}")
            print(f"  Training sequence: 3 turnovers per non-maverick agent, sequentially")
            print(f"  Then: Clear buffer/population and train maverick agents")
        
        # Now load the first member for training
        if self.multi_phase == 'maverick':
            first_member_idx = self.maverick_members[0]
        else:
            first_member_idx = self.non_maverick_members[0]
        self._load_multi_member(first_member_idx)

    def _load_multi_member(self, member_idx: int):
        """
        Load a specific committee member and initialize population for training.
        Similar to load_single_agent but for multi-mode rotation.

        Args:
            member_idx: Index of the committee member to load (0-8)
        """
        from committee import get_agent_filepath

        self.current_member_idx = member_idx
        member = self.multi_roster['members'][member_idx]
        context_window = self.multi_roster['context_window_days']
        
        # Ensure maverick_mode is set correctly based on phase
        if self.multi_phase == 'maverick':
            self.maverick_mode = True
            # Recreate environment with maverick_mode=True if not already set
            if not hasattr(self.eval_env, 'maverick_mode') or not self.eval_env.maverick_mode:
                print(f"  Recreating environment with maverick_mode=True...")
                self.eval_env = TradingEnvironment(
                    data_array=self.data_loader.data_array,
                    dates=self.data_loader.dates,
                    normalization_stats=self.normalization_stats,
                    start_idx=self.train_start_idx,
                    end_idx=self.train_end_idx,
                    trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
                    data_array_full=self.data_loader.data_array_full,
                    consistency_mode=self.consistency_mode,
                    maverick_mode=True
                )
                print(f"  ✓ Environment updated for maverick mode")
        else:
            self.maverick_mode = False

        # Record when this member started training for warmup enforcement
        self.member_training_start_gen = self.generation

        # Set the ROI hurdle to THIS member's starting ROI
        # This ensures the reward function scales based on improvement over THIS agent
        # Each member competes against themselves, not a global median
        target_roi = self.member_starting_rois[member_idx]
        self.roi_hurdle_ema = target_roi

        print(f"\n{'='*60}")
        print(f"🎯 MULTI-MODE: Loading Member {member_idx} for Training")
        print(f"{'='*60}")
        print(f"  Phase: {self.multi_phase}")
        print(f"  Member: {member['run_name']}_{member['agent_id']}")
        print(f"  Maverick mode: {self.maverick_mode}")
        print(f"  Member Starting ROI: {target_roi:.2f}%")
        print(f"  Training Hurdle set to: {self.roi_hurdle_ema:.2f}%")
        print(f"  Baseline: {self.member_baselines[member_idx]:.2f}")
        print(f"  Current breakthroughs: {self.member_breakthroughs[member_idx]}")
        if self.multi_phase == 'non_maverick':
            list_idx = self.non_maverick_members.index(member_idx) if member_idx in self.non_maverick_members else -1
            if list_idx >= 0:
                print(f"  Turnovers for this agent: {self.non_maverick_turnovers_per_agent[list_idx]}/{Config.MULTI_TARGET_TURNOVERS}")
        elif self.multi_phase == 'maverick':
            list_idx = self.maverick_members.index(member_idx) if member_idx in self.maverick_members else -1
            if list_idx >= 0:
                print(f"  Turnovers for this agent: {self.maverick_turnovers_per_agent[list_idx]}/{Config.MULTI_TARGET_TURNOVERS}")

        agent_path = get_agent_filepath(member, context_window)
        source_agent = DDPGAgent(agent_id=0)
        source_agent.load(str(agent_path))

        # FIX 2: Re-evaluate baseline if in Maverick phase (or if maverick_mode changed)
        # The stored baseline might be from a different reward function (Standard vs Maverick)
        # Maverick reward function uses completely different scale (e.g., -5000 penalties, different volume scalars)
        # Comparing Maverick scores against Standard baselines is mathematically invalid
        if self.maverick_mode:
            print(f"  Re-evaluating baseline for Maverick scoring...")
            # Use the helper to evaluate with current reward function (maverick_mode=True)
            # This ensures the baseline matches the CURRENT reward function's scale
            fresh_baseline = self._evaluate_single_agent_for_baseline(source_agent)
            self.member_baselines[member_idx] = fresh_baseline
            print(f"  Updated Baseline (Maverick Scale): {fresh_baseline:.2f}")
        # Note: For non-maverick phase, we keep the original baseline from initialization
        # which was calculated with the standard reward function

        # Store the parent agent for potential stuck recovery (parent mutant injection)
        self.multi_parent_agent = source_agent.clone()
        self.multi_parent_agent.agent_id = -1  # Mark as parent template

        # Reset stuck detection for this member
        self.multi_gens_since_improvement = 0
        self.multi_best_score_for_member = float('-inf')

        # Set confirmed baseline for breakthrough detection (used by existing logic)
        self.confirmed_baseline = self.member_baselines[member_idx]
        self.initial_single_baseline = self.member_baselines[member_idx]

        # Initialize population with clones + mutations (same as single mode)
        # Respect local mode's smaller population size
        pop_size = Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE
        num_clones = int(pop_size * Config.SINGLE_CLONE_FRAC)
        num_normal_mutants = int(pop_size * Config.SINGLE_NORMAL_MUTATION_FRAC)
        num_plateau_mutants = pop_size - num_clones - num_normal_mutants

        print(f"\n  Population initialization ({pop_size} agents):")
        print(f"    Pure clones: {num_clones} ({Config.SINGLE_CLONE_FRAC*100:.0f}%)")
        print(f"    Normal mutation: {num_normal_mutants} ({Config.SINGLE_NORMAL_MUTATION_FRAC*100:.0f}%)")
        print(f"    Plateau mutation: {num_plateau_mutants} ({Config.SINGLE_PLATEAU_MUTATION_FRAC*100:.0f}%)")

        new_population = []

        # Pure clones
        for i in range(num_clones):
            clone = source_agent.clone()
            clone.agent_id = i
            clone.is_elite = True
            new_population.append(clone)

        # Normal mutation clones
        base_mutation_rate = Config.MUTATION_RATE_CONSISTENCY
        base_mutation_std = Config.MUTATION_STD
        for i in range(num_normal_mutants):
            mutated_clone = mutate(source_agent, mutation_rate=base_mutation_rate, mutation_std=base_mutation_std)
            mutated_clone.agent_id = num_clones + i
            mutated_clone.is_elite = False
            new_population.append(mutated_clone)

        # Plateau mutation clones (1.5x mutation)
        plateau_mutation_rate = min(base_mutation_rate * 1.5, self.max_mutation_rate)
        plateau_mutation_std = min(base_mutation_std * 1.5, self.max_mutation_std)
        for i in range(num_plateau_mutants):
            mutated_clone = mutate(source_agent, mutation_rate=plateau_mutation_rate, mutation_std=plateau_mutation_std)
            mutated_clone.agent_id = num_clones + num_normal_mutants + i
            mutated_clone.is_elite = False
            new_population.append(mutated_clone)

        self.population = new_population

        # Reset metrics for fresh start with new member
        # This prevents a weaker member from never triggering "New Best" logs
        # if a stronger member ran previously
        self.best_fitness = float('-inf')
        self.best_validation_fitness = float('-inf')
        self.best_agent = None

        # Reset mutation parameters to base values
        self.current_mutation_rate = Config.MUTATION_RATE_CONSISTENCY
        self.current_mutation_std = Config.MUTATION_STD

        # Reset plateau detection state for adaptive mutation
        # Without this, plateau state from previous member would affect new member's mutation
        self.plateau_detected = False
        self.validation_fitness_history = []

        print(f"  Population initialized: {len(self.population)} agents")
        print(f"  Metrics reset for fresh training")
        print("="*60 + "\n")

    def _advance_to_next_multi_member(self) -> bool:
        """
        Advance to the next committee member based on phase.
        
        Non-maverick phase: Train each non-maverick agent for 3 turnovers sequentially.
        Maverick phase: Train each maverick agent for 3 turnovers sequentially.

        Returns:
            True if advanced to next member, False if all done or phase transition needed
        """
        if self.multi_phase == 'non_maverick':
            # Non-maverick phase: sequential 3 turnovers per agent
            if len(self.non_maverick_members) == 0:
                # No non-maverick members - transition to maverick phase
                return self._transition_to_maverick_phase()
            
            current_list_idx = self.current_non_maverick_idx
            current_member_idx = self.non_maverick_members[current_list_idx]
            # Check breakthroughs (turnovers) for current agent
            current_breakthroughs = self.member_breakthroughs[current_member_idx]
            
            # Check if current non-maverick agent has completed 3 turnovers (breakthroughs)
            if current_breakthroughs >= Config.MULTI_TARGET_TURNOVERS:
                # Move to next non-maverick agent
                self.current_non_maverick_idx += 1
                
                # Check if all non-maverick agents are done
                if self.current_non_maverick_idx >= len(self.non_maverick_members):
                    # All non-maverick agents complete - transition to maverick phase
                    print(f"\n{'='*60}")
                    print(f"✓ NON-MAVERICK PHASE COMPLETE")
                    print(f"  All {len(self.non_maverick_members)} non-maverick agents completed {Config.MULTI_TARGET_TURNOVERS} turnovers each")
                    print(f"{'='*60}")
                    return self._transition_to_maverick_phase()
                
                # Load next non-maverick agent
                next_member_idx = self.non_maverick_members[self.current_non_maverick_idx]
                self._load_multi_member(next_member_idx)
                return True
            else:
                # Current agent needs more turnovers - continue training same agent
                return True
        
        elif self.multi_phase == 'maverick':
            # Maverick phase: sequential 3 turnovers per agent
            if len(self.maverick_members) == 0:
                return False  # No maverick members (shouldn't happen due to validation)
            
            current_list_idx = self.current_maverick_idx
            current_member_idx = self.maverick_members[current_list_idx]
            # Check breakthroughs (turnovers) for current agent
            current_breakthroughs = self.member_breakthroughs[current_member_idx]
            
            # Check if current maverick agent has completed 3 turnovers (breakthroughs)
            if current_breakthroughs >= Config.MULTI_TARGET_TURNOVERS:
                # Move to next maverick agent
                self.current_maverick_idx += 1
                
                # Check if all maverick agents are done
                if self.current_maverick_idx >= len(self.maverick_members):
                    # All maverick agents complete - training done!
                    print(f"\n{'='*60}")
                    print(f"✓ MAVERICK PHASE COMPLETE")
                    print(f"  All {len(self.maverick_members)} maverick agents completed {Config.MULTI_TARGET_TURNOVERS} turnovers each")
                    print(f"{'='*60}")
                    return False
                
                # Load next maverick agent
                next_member_idx = self.maverick_members[self.current_maverick_idx]
                self._load_multi_member(next_member_idx)
                return True
            else:
                # Current agent needs more turnovers - continue training same agent
                return True
        
        return False  # Should not reach here
    
    def _transition_to_maverick_phase(self) -> bool:
        """
        Transition from non-maverick phase to maverick phase.
        Clears buffer, resets population, and enables maverick mode.
        
        Returns:
            True if transition successful and first maverick loaded, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"🔄 TRANSITIONING TO MAVERICK PHASE")
        print(f"{'='*60}")
        
        # Reset buffer and population
        self._reset_for_maverick_phase()
        
        # Load first maverick agent
        if len(self.maverick_members) > 0:
            first_maverick_idx = self.maverick_members[0]
            self.current_maverick_idx = 0
            self._load_multi_member(first_maverick_idx)
            return True
        
        return False
    
    def _reset_for_maverick_phase(self):
        """
        Reset buffer, population, and environment for maverick phase.
        Called when transitioning from non-maverick to maverick phase.
        """
        print(f"\n  Clearing replay buffer and resetting population...")
        
        # Clear replay buffer
        if hasattr(self.replay_buffer, 'clear'):
            self.replay_buffer.clear()
            
            # FIX 3: Physically delete buffer files to prevent "zombie" files
            # The clear() method clears the deque but doesn't delete physical files
            import shutil
            if hasattr(self.replay_buffer, 'storage_path'):
                storage_path = Path(self.replay_buffer.storage_path)
                if storage_path.exists():
                    print(f"  Physically deleting buffer files in {storage_path}...")
                    try:
                        # Delete entire directory and recreate to ensure clean slate
                        shutil.rmtree(storage_path)
                        storage_path.mkdir(parents=True, exist_ok=True)
                        print(f"  ✓ Buffer directory cleaned (deleted and recreated)")
                    except Exception as e:
                        print(f"  ⚠ Failed to clean buffer directory: {e}")
                    
                    # Reset file ID counter if buffer uses one
                    if hasattr(self.replay_buffer, 'file_id_counter'):
                        self.replay_buffer.file_id_counter = 0
                    if hasattr(self.replay_buffer, 'total_added'):
                        self.replay_buffer.total_added = 0
                    if hasattr(self.replay_buffer, 'total_transitions'):
                        self.replay_buffer.total_transitions = 0
            
            print(f"  ✓ Replay buffer cleared (memory + disk)")
        else:
            print(f"  ⚠ Replay buffer does not support clear() - manual cleanup may be needed")
        
        # Reset population - create fresh population
        pop_size = Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE
        print(f"  Creating fresh population of {pop_size} agents...")
        self.population = [DDPGAgent(agent_id=i) for i in range(pop_size)]
        print(f"  ✓ Population reset")
        
        # Set phase and enable maverick mode
        self.multi_phase = 'maverick'
        self.maverick_mode = True
        
        # Initialize maverick turnovers tracking
        self.maverick_turnovers_per_agent = [0] * len(self.maverick_members)
        self.current_maverick_idx = 0
        
        # Reset breakthrough tracking for maverick phase (keep member_breakthroughs for all members)
        # But reset member-specific tracking
        self.multi_gens_since_improvement = 0
        self.multi_best_score_for_member = float('-inf')
        self.multi_parent_agent = None
        
        # CRITICAL: Reset baselines for maverick phase
        # Mavericks use a different reward function (FOMO/ROI-First) with different fitness scale
        # Comparing maverick performance against non-maverick baselines would be incorrect
        self.confirmed_baseline = 0.0
        self.initial_single_baseline = 0.0
        print(f"  ✓ Baselines reset for maverick phase (different fitness scale)")
        
        # Recreate environment with maverick_mode=True
        print(f"  Recreating environment with maverick_mode=True...")
        self.eval_env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,
            start_idx=self.train_start_idx,
            end_idx=self.train_end_idx,
            trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            data_array_full=self.data_loader.data_array_full,
            consistency_mode=self.consistency_mode,
            maverick_mode=True  # Enable maverick mode
        )
        print(f"  ✓ Environment recreated with maverick_mode=True")
        
        # Update shared memory config for workers (if not in local mode)
        # CRITICAL: Recreate shared memory with maverick_mode=True to prevent global state leaks
        # Workers cache _worker_env_config globally, so we must recreate shared memory
        # to ensure they get the updated maverick_mode flag
        if not self.local_mode:
            print(f"  Recreating shared memory with maverick_mode=True for workers...")
            # FIX 1: Clean up OLD shared memory before creating new blocks
            # This prevents memory leaks (2-3GB of old shared memory staying locked)
            self._cleanup_shared_memory()
            # Reinitialize shared memory with updated maverick_mode
            self._init_shared_memory()
            print(f"  ✓ Shared memory recreated - workers will use maverick_mode=True on next ProcessPoolExecutor")
        
        print(f"  ✓ Reset complete - ready for maverick phase")

    def _evaluate_single_agent_for_baseline(self, agent: DDPGAgent) -> float:
        """
        Evaluate a single agent using consistency-mode evaluation.
        Uses 5 episodes with pessimistic (0.4*mean + 0.6*min) aggregation.

        Args:
            agent: The agent to evaluate

        Returns:
            Pessimistically aggregated fitness score
        """
        print(f"\n  Evaluating agent for baseline (5 episodes, pessimistic aggregation)...")

        num_episodes = 5

        # Prepare environment config using shared memory
        env_config = self._get_shared_env_config(
            start_idx=self.train_start_idx,
            end_idx=self.train_end_idx,
            trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            is_training=True
        )

        # Extract agent state (CPU tensors only)
        agent_state = {
            'actor': {k: v.cpu() for k, v in agent.actor.state_dict().items()},
            'critic': {k: v.cpu() for k, v in agent.critic.state_dict().items()}
        }

        # Prepare tasks for parallel evaluation
        tasks = []
        for slice_idx in range(num_episodes):
            # Calculate episode indices
            total_days_needed = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            max_start = self.train_end_idx - total_days_needed
            start_idx = np.random.randint(self.train_start_idx, max_start)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

            # Create unique seed for this task
            task_seed = self.seed + slice_idx * 1000

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
        num_workers = min(mp.cpu_count() - 1, Config.EVAL_NUM_WORKERS)

        slice_fitness_scores = []

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(env_config,)
        ) as executor:
            futures = {executor.submit(_run_episode_worker, task): idx for idx, task in enumerate(tasks)}

            for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating baseline"):
                try:
                    raw_fitness, episode_info, _ = future.result()
                    triad_fitness = self.calculate_triad_fitness(episode_info)
                    slice_fitness_scores.append(triad_fitness)
                except Exception as e:
                    print(f"\n  ! Worker failed: {e}")
                    slice_fitness_scores.append(-10000.0)

        # Calculate fitness using appropriate aggregator
        if self.multi_mode:
            # Penalized Median for multi-mode (consistent with Gauntlet/committee.py)
            final_fitness = self._calculate_penalized_median_fitness(slice_fitness_scores)
            aggregation_method = "Penalized Median"
        else:
            # Pessimistic for single/heroes mode
            final_fitness = self._calculate_pessimistic_fitness(slice_fitness_scores)
            aggregation_method = "Pessimistic"

        print(f"  Slice scores: {[f'{s:.2f}' for s in slice_fitness_scores]}")
        print(f"  {aggregation_method} aggregation: {final_fitness:.2f}")

        return final_fitness

    def run_episode(self, agent: DDPGAgent, env: TradingEnvironment,
                   start_idx: int, end_idx: int,
                   training: bool = True,
                   transition_collector: list = None) -> Tuple[float, Dict]:
        """
        Run one episode with an agent using a persistent environment.

        Args:
            agent: Agent to run
            env: Persistent TradingEnvironment to reuse (critical for memory efficiency)
            start_idx: Starting day index (first day of trading period)
            end_idx: Ending day index (includes settlement period)
            training: Whether this is training (adds to replay buffer)
            transition_collector: Optional list to collect transitions instead of writing to disk.
                                  When provided, transitions are appended here for batch writing later.
                                  Used in local mode to serialize disk I/O.

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

            # Store transition - either in collector for batch writing or directly to buffer
            transition = {
                'state': state.astype(np.float32),
                'action': action.astype(np.float32),
                'reward': reward,
                'next_state': next_state.astype(np.float32),
                'done': float(terminated or truncated)
            }

            if transition_collector is not None:
                # Collect for batch writing (local mode)
                transition_collector.append(transition)
            else:
                # Write directly to buffer (parallel mode writes in workers)
                self.replay_buffer.add(**transition)
            
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
        Generate 10 validation slices from validation set.

        Divides validation period into 4 equal quarters, then samples:
        - 4 slices from within each quarter
        - 3 straddling slices between quarters (Q1-Q2, Q2-Q3, Q3-Q4)
        - 3 random slices from anywhere in the validation period

        This ensures comprehensive coverage with overlapping windows across different market conditions.

        Each slice consists of:
        - CONTEXT_WINDOW_DAYS (151) of prior data (may come from training data for context)
        - TRADING_PERIOD_DAYS (125) where agent can trade (from validation set)
        - SETTLEMENT_PERIOD_DAYS (30) to close positions (from validation set)

        Returns:
            List of 10 tuples: (start_idx, end_idx, trading_end_idx)
        """
        # NOTE: start_idx is the first day of TRADING (not context)
        # The environment automatically looks back CONTEXT_WINDOW_DAYS from start_idx for context
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

        # 3. Sample 3 completely random slices from entire validation range (3 slices)
        for _ in range(3):
            start_idx = np.random.randint(min_start, max_start + 1)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

            slices.append((start_idx, end_idx, trading_end_idx))

        return slices

    def generate_gauntlet_slices(self) -> List[Tuple[int, int, int]]:
        """
        Generate 20+ rigorous validation slices for Gauntlet stress test.

        Samples slices from BOTH training and validation data to ensure the agent
        performs robustly across all market regimes, not just validation period.

        Strategy:
        - 10 slices from training data (different market conditions)
        - 10 slices from validation data (out-of-sample)

        This prevents agents from gaming the validation set and ensures genuine
        robustness across all available market data.

        Returns:
            List of 20 tuples: (start_idx, end_idx, trading_end_idx)
        """
        slices = []

        # 1. Sample 10 slices from training data
        min_start_train = self.train_start_idx
        max_start_train = self.train_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start_train >= min_start_train:
            # Divide training range into 10 segments
            train_range = max_start_train - min_start_train + 1
            train_segment_size = max(1, train_range // 10)

            for i in range(10):
                segment_start = min_start_train + (i * train_segment_size)
                segment_end = min(max_start_train + 1, segment_start + train_segment_size)

                if segment_end > segment_start:
                    start_idx = np.random.randint(segment_start, segment_end)
                    end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
                    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

                    slices.append((start_idx, end_idx, trading_end_idx))

        # 2. Sample 10 slices from validation data
        min_start_val = self.val_start_idx
        max_start_val = self.val_end_idx - (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS)

        if max_start_val >= min_start_val:
            # Divide validation range into 10 segments
            val_range = max_start_val - min_start_val + 1
            val_segment_size = max(1, val_range // 10)

            for i in range(10):
                segment_start = min_start_val + (i * val_segment_size)
                segment_end = min(max_start_val + 1, segment_start + val_segment_size)

                if segment_end > segment_start:
                    start_idx = np.random.randint(segment_start, segment_end)
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

        MAVERICK MODE: FOMO/ROI-First scoring with WR Gate.
        Formula: Base Score + Expectancy Bonus - FOMO Penalty - Fear Factor - WR Penalty
        - Base Score: (Raw PnL / Peak Capital) * log10(Peak Capital + 10) * WR^4
        - Expectancy Bonus: Expectancy * 20.0
        - FOMO Penalty: max(0, Market Return - Agent ROI) * 10.0
        - Fear Factor: 2.0 * (num_losses ^ 1.2)
        - WR Penalty: max(0, 0.60 - WR) * 100.0 (Forces WR > 60%)
        """
        total_trades = stats.get('num_trades', 0)

        # 1. Handle Inactivity
        if total_trades == 0:
            # MAVERICK MODE: Massive penalty for inaction - forces market engagement
            if self.maverick_mode:
                return -5000.0 + stats.get('max_coefficient_during_episode', 0)

            # In consistency mode during stabilization/gauntlet, use soft penalty
            # (tactical no-trade in a slice is acceptable, not a ghost indicator)
            if self.consistency_mode and self.breakthrough_state in (BreakthroughState.STABILIZATION, BreakthroughState.GAUNTLET):
                penalty = Config.ZERO_TRADES_PENALTY_GAUNTLET
            elif self.consistency_mode:
                penalty = Config.ZERO_TRADES_PENALTY_CONSISTENCY
            else:
                penalty = Config.ZERO_TRADES_PENALTY_NORMAL
            return -penalty + stats.get('max_coefficient_during_episode', 0)

        # 2. Calculate Core Metrics
        win_rate = stats.get('win_rate', 0.0) # 0.0 to 1.0
        num_losses = stats.get('num_losses', 0)

        # Calculate Quality Ratio (QR): ratio of trades beating the HoF median ROI threshold
        # This is dynamically updated each generation to reflect the rising bar of excellence
        closed_trades = stats.get('closed_trades', [])
        if closed_trades:
            quality_count = sum(1 for t in closed_trades if t.get('gain_pct', 0) >= self.quality_threshold)
            qr = quality_count / total_trades
        else:
            qr = 0.0

        raw_pnl = stats.get('raw_pnl', 0.0)
        total_inv = stats.get('total_investment', 0.0)
        roi_pct = (raw_pnl / total_inv * 100) if total_inv > 0 else 0.0

        # MAVERICK MODE: FOMO/ROI-First scoring
        if self.maverick_mode:
            # Get peak capital employed (more accurate than total_investment for efficiency)
            peak_capital = stats.get('peak_capital_employed', total_inv)
            if peak_capital <= 0:
                peak_capital = total_inv if total_inv > 0 else 1.0

            # Calculate Expectancy explicitly
            # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
            avg_win = 0.0
            avg_loss = 0.0
            if closed_trades:
                wins = [t.get('gain_pct', 0) for t in closed_trades if t.get('gain_pct', 0) >= 0]
                losses = [abs(t.get('gain_pct', 0)) for t in closed_trades if t.get('gain_pct', 0) < 0]
                if wins:
                    avg_win = sum(wins) / len(wins)
                if losses:
                    avg_loss = sum(losses) / len(losses)
            
            loss_rate = 1.0 - win_rate
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

            # 1. Base Score: ROI Efficiency + WR Boost
            # (Raw PnL / Peak Capital Employed) * log10(Peak Capital + 10) * win_rate^4
            roi_efficiency = (raw_pnl / peak_capital) * 100.0 if peak_capital > 0 else 0.0
            volume_scalar = math.log10(peak_capital + 10)
            
            # Massive boost for high WR (convex function)
            # WR 0.50 -> 0.0625 multiplier
            # WR 0.60 -> 0.1296 multiplier (2x boost)
            # WR 0.70 -> 0.2401 multiplier (4x boost)
            win_rate_multiplier = win_rate ** 4
            
            base_score = roi_efficiency * volume_scalar * win_rate_multiplier

            # 2. Expectancy Bonus
            # Add raw expectancy score (scaled to match ROI magnitude)
            # e.g., Expectancy 0.5% -> +10 points
            expectancy_score = expectancy * 20.0

            # 3. FOMO Penalty (Relative Performance)
            # Get market return from stats (Col 44 = S&P 500 proxy)
            market_return = stats.get('market_return_pct', 0.0)
            alpha_gap = market_return - roi_pct
            fomo_penalty = max(0.0, alpha_gap) * 10.0

            # 4. Fear Factor (Limited Ratchet)
            # Gentle non-linear drag on losses: 2.0 * (num_losses ^ 1.2)
            fear_factor = 2.0 * (num_losses ** 1.2) if num_losses > 0 else 0.0
            
            # 5. WR Gate / Penalty
            # Severe penalty for WR < 60% to force quality
            wr_penalty = 0.0
            if win_rate < 0.60:
                wr_penalty = (0.60 - win_rate) * 100.0  # e.g., 55% WR -> 5.0 penalty

            # Final Maverick Fitness
            fitness = base_score + expectancy_score - fomo_penalty - fear_factor - wr_penalty

            return float(fitness)

        # NORMAL/CONSISTENCY MODE: Standard Triad scoring
        # ROI Expansion Mode: Uncap ROI and apply super-linear scaling
        # Power law (** 1.1) makes high ROI disproportionately valuable
        # Sign preservation: apply power to absolute value, then restore sign
        if roi_pct >= 0:
            roi_score = (roi_pct ** 1.1)
        else:
            roi_score = -(abs(roi_pct) ** 1.1)

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

    def _calculate_pessimistic_fitness(self, slice_fitness_scores: List[float]) -> float:
        """
        Calculate fitness using pessimistic aggregator (0.4*mean + 0.6*min).

        This aligns training incentives with validation requirements by giving
        60% weight to worst slice performance and 40% to average performance.

        Args:
            slice_fitness_scores: List of fitness scores from multiple evaluation slices

        Returns:
            Pessimistically aggregated fitness score
        """
        mean_score = np.mean(slice_fitness_scores)
        min_score = np.min(slice_fitness_scores)
        return (0.4 * mean_score) + (0.6 * min_score)

    def _calculate_penalized_median_fitness(self, slice_fitness_scores: List[float]) -> float:
        """
        Calculate fitness using Penalized Median scoring: Median - (0.5 * StdDev).

        This rewards agents that reliably perform well while penalizing volatility.
        Used by Gauntlet validation and --multi mode to ensure consistent performance
        across diverse market conditions.

        Args:
            slice_fitness_scores: List of fitness scores from multiple evaluation slices

        Returns:
            Penalized median fitness score
        """
        scores_np = np.array(slice_fitness_scores)
        median_score = float(np.median(scores_np))
        std_score = float(np.std(scores_np))
        return median_score - (0.5 * std_score)

    def _aggregate_agent_stats(self, slice_episode_stats: List[Dict]) -> Dict:
        """
        Aggregate episode statistics across all training slices for a single agent.

        Calculates global win rate (total wins / total trades) rather than
        averaging per-slice win rates for more accurate representation.

        Args:
            slice_episode_stats: List of episode info dicts from multiple slices

        Returns:
            Aggregated stats dict with num_trades, num_wins, num_losses, win_rate
        """
        agent_total_wins = sum(s['num_wins'] for s in slice_episode_stats)
        agent_total_losses = sum(s['num_losses'] for s in slice_episode_stats)
        agent_total_trades = agent_total_wins + agent_total_losses
        agent_win_rate = agent_total_wins / agent_total_trades if agent_total_trades > 0 else 0.0

        return {
            'num_trades': int(np.mean([s['num_trades'] for s in slice_episode_stats])),
            'num_wins': int(np.mean([s['num_wins'] for s in slice_episode_stats])),
            'num_losses': int(np.mean([s['num_losses'] for s in slice_episode_stats])),
            'win_rate': agent_win_rate,
        }

    def _aggregate_population_stats(self, all_episode_stats: List[Dict], fitness_scores: List[float]) -> Dict:
        """
        Aggregate statistics across all agents in the population.

        Args:
            all_episode_stats: List of per-agent aggregated stats dicts
            fitness_scores: List of fitness scores for all agents

        Returns:
            Population-level aggregate stats dict
        """
        agents_with_trades = [s for s in all_episode_stats if s['num_trades'] > 0]
        avg_win_rate = (
            float(sum(s['win_rate'] for s in agents_with_trades) / len(agents_with_trades))
            if agents_with_trades else 0.0
        )

        return {
            'total_trades': int(sum(s['num_trades'] for s in all_episode_stats)),
            'avg_trades_per_agent': float(sum(s['num_trades'] for s in all_episode_stats) / len(all_episode_stats)),
            'total_wins': int(sum(s['num_wins'] for s in all_episode_stats)),
            'total_losses': int(sum(s['num_losses'] for s in all_episode_stats)),
            'avg_win_rate': avg_win_rate,
            'agents_with_positive_fitness': int(sum(1 for f in fitness_scores if f > 0)),
        }

    def evaluate_population(self) -> Tuple[List[float], Dict]:
        """
        Evaluate all agents in population (fitness scores).

        NEW: Multi-slice evaluation for robust fitness signal
        - Each agent is evaluated on 3 different random training slices
        - Final fitness = average of the LOWEST 2 scores (conservative, robust estimate)
        - This prevents "lucky" agents from advancing and selects for consistency

        In local mode, transitions are collected in memory and batch-written to disk
        at the end to avoid I/O thrashing from many small writes.

        Returns:
            Tuple of (fitness_scores, aggregate_stats)
        """
        fitness_scores = []
        all_episode_stats = []

        # Use same number of episodes and scoring method for both modes
        num_episodes = 5 if self.consistency_mode else 3

        # Use Penalized Median for multi-mode, pessimistic otherwise
        if self.multi_mode:
            scoring_method = "median - 0.5*std (Penalized Median, matches Gauntlet)"
        else:
            scoring_method = "0.4*mean + 0.6*min (pessimistic, matches validation)"

        print(f"\n--- Generation {self.generation + 1}: Evaluating Population ---")
        print(f"Multi-slice evaluation: {num_episodes} slices per agent, scoring = {scoring_method}")

        # Count elite vs exploratory agents for logging
        num_elites = sum(1 for a in self.population if a.is_elite)
        num_exploratory = len(self.population) - num_elites
        print(f"Elite Demonstration: {num_elites} elites (no noise) + {num_exploratory} exploratory (with noise) contribute to buffer")

        # In local mode, collect transitions for batch writing to avoid I/O thrashing
        transition_collector = [] if self.local_mode else None
        if self.local_mode:
            print(f"Local mode: Collecting transitions for batch write")

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
                    training=(not agent.is_elite),  # Only exploratory agents contribute to buffer
                    transition_collector=transition_collector  # Collect transitions in local mode
                )

                # Calculate Structural Fitness for Evolution
                triad_fitness = self.calculate_triad_fitness(episode_info)

                slice_fitness_scores.append(triad_fitness)
                slice_episode_stats.append(episode_info)

            # Calculate fitness using appropriate aggregator
            if self.multi_mode:
                final_fitness = self._calculate_penalized_median_fitness(slice_fitness_scores)
            else:
                final_fitness = self._calculate_pessimistic_fitness(slice_fitness_scores)
            fitness_scores.append(final_fitness)

            # Aggregate episode stats across all training slices for this agent
            all_episode_stats.append(self._aggregate_agent_stats(slice_episode_stats))

        # Ensure fitness_scores are all plain floats
        fitness_scores = [float(f) for f in fitness_scores]

        # Aggregate statistics across all agents
        aggregate_stats = self._aggregate_population_stats(all_episode_stats, fitness_scores)

        # In local mode, batch-write all collected transitions to disk
        if self.local_mode and transition_collector:
            print(f"\n--- Batch writing {len(transition_collector)} transitions to replay buffer ---")
            self.replay_buffer.add_batch(transition_collector)
            del transition_collector  # Free memory

        # CRITICAL FIX: Delete large all_episode_stats list and force aggressive GC
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
        num_episodes = 5 if self.consistency_mode else 3

        # Use Penalized Median for multi-mode, pessimistic otherwise
        if self.multi_mode:
            scoring_method = "median - 0.5*std (Penalized Median, matches Gauntlet)"
        else:
            scoring_method = "0.4*mean + 0.6*min (pessimistic, matches validation)"

        print(f"\n--- Generation {self.generation + 1}: Evaluating Population (Parallel) ---")
        print(f"Multi-slice evaluation: {num_episodes} slices per agent, scoring = {scoring_method}")

        # Count elite vs exploratory agents for logging
        num_elites = sum(1 for a in self.population if a.is_elite)
        num_exploratory = len(self.population) - num_elites
        print(f"Elite Demonstration: {num_elites} elites (no noise) + {num_exploratory} exploratory (with noise) contribute to buffer")

        # Prepare environment config using shared memory (eliminates serialization overhead)
        # Workers reconstruct arrays from shared memory names (zero-copy access)
        env_config = self._get_shared_env_config(
            start_idx=self.train_start_idx,
            end_idx=self.train_end_idx,
            trading_end_idx=self.train_start_idx + Config.TRADING_PERIOD_DAYS,
            is_training=True
        )

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
        # Use all available cores (leave 1 for OS), with configurable safety cap
        num_workers = min(mp.cpu_count() - 1, Config.EVAL_NUM_WORKERS)
        print(f"Using {num_workers} parallel workers (out of {mp.cpu_count()} vCPUs)")

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
            # Track at agent level (not slice level) for cleaner progress display
            completed_tasks = 0
            completed_agents = set()
            pbar = tqdm(total=len(self.population), desc="Evaluating agents")
            for future in as_completed(futures):
                task_idx = futures[future]
                agent_idx = task_idx // num_episodes  # Each agent has num_episodes slices

                try:
                    # raw_fitness is the sum of rewards from env (good for RL, bad for Evolution)
                    raw_fitness, episode_info, transition_file_paths = future.result(timeout=300)  # 5 min timeout

                    # Calculate Structural Fitness for Evolution
                    triad_fitness = self.calculate_triad_fitness(episode_info)

                    # Store triad_fitness instead of raw_fitness
                    fitness_by_agent[agent_idx].append((triad_fitness, episode_info))

                    # Collect transition file paths from exploratory agents
                    if transition_file_paths:
                        all_transition_file_paths.extend(transition_file_paths)

                    completed_tasks += 1

                except TimeoutError:
                    print(f"\n⚠ Worker TIMEOUT for agent {agent_idx} task {task_idx} (task {completed_tasks+1}/{len(tasks)})")
                    print(f"  This may indicate a deadlock or infinite loop in worker process")
                    # Use penalty fitness for failed episodes
                    fitness_by_agent[agent_idx].append((-10000.0, {
                        'num_trades': 0, 'num_wins': 0, 'num_losses': 0, 'win_rate': 0.0
                    }))
                    completed_tasks += 1
                except Exception as e:
                    print(f"\n⚠ Worker EXCEPTION for agent {agent_idx} task {task_idx} (task {completed_tasks+1}/{len(tasks)}): {e}")
                    import traceback
                    traceback.print_exc()
                    import sys
                    sys.stdout.flush()  # Force flush to ensure error is logged
                    # Use penalty fitness for failed episodes
                    fitness_by_agent[agent_idx].append((-10000.0, {
                        'num_trades': 0, 'num_wins': 0, 'num_losses': 0, 'win_rate': 0.0
                    }))
                    completed_tasks += 1

                # Update progress bar when an agent completes all its slices
                if len(fitness_by_agent[agent_idx]) == num_episodes and agent_idx not in completed_agents:
                    completed_agents.add(agent_idx)
                    pbar.update(1)

            pbar.close()

            # Aggregate results (same logic as sequential version)
            fitness_scores = []
            all_episode_stats = []

            for agent_slices in fitness_by_agent:
                slice_fitness = [f for f, _ in agent_slices]
                slice_stats = [info for _, info in agent_slices]

                # Calculate fitness using appropriate aggregator
                if self.multi_mode:
                    final_fitness = self._calculate_penalized_median_fitness(slice_fitness)
                else:
                    final_fitness = self._calculate_pessimistic_fitness(slice_fitness)
                fitness_scores.append(final_fitness)

                # Aggregate stats for this agent
                all_episode_stats.append(self._aggregate_agent_stats(slice_stats))

            # Ensure fitness_scores are all plain floats
            fitness_scores = [float(f) for f in fitness_scores]

            # Clean up large data structures before buffer operations
            del fitness_by_agent
            import gc
            gc.collect()

            # Add transition file paths to replay buffer (transitions already written to disk by workers!)
            print(f"\n--- Adding {len(all_transition_file_paths)} transitions to replay buffer ---")
            print(f"  Current buffer size: {len(self.replay_buffer)}/{self.replay_buffer.capacity}")
            if all_transition_file_paths:
                # Transitions were written to disk during parallel evaluation - just add paths to buffer
                print(f"  Transitions already written to disk by workers (parallel I/O)")

                # CRITICAL FIX: Reset if buffer IS full OR WILL overflow during this update
                # We must catch the specific generation where we cross the threshold (e.g. 967k -> 1.04M)
                # The previous fix checked if buffer was already full, but failed to account for
                # the generation where the buffer becomes full for the first time.
                current_size = len(self.replay_buffer.buffer)
                num_new = len(all_transition_file_paths)
                capacity = self.replay_buffer.capacity

                # Reset if we are about to drop files (overflow) OR if we are already at capacity
                should_reset_workers = (current_size + num_new > capacity) or (current_size >= capacity)

                # CRITICAL: Collect old paths that will be evicted so we can delete them from disk
                # When appending to a bounded deque, oldest entries are automatically removed,
                # but the FILES remain on disk unless we explicitly delete them
                num_to_evict = max(0, current_size + num_new - capacity)
                old_paths_to_delete = []
                if num_to_evict > 0:
                    old_paths_to_delete = [self.replay_buffer.buffer[i] for i in range(num_to_evict)]

                for file_path in all_transition_file_paths:
                    self.replay_buffer.buffer.append(file_path)

                # Delete evicted files from disk to prevent unbounded disk growth
                if old_paths_to_delete:
                    deleted_count = 0
                    external_deleted = 0
                    for old_path in old_paths_to_delete:
                        try:
                            # Check if from external source (migration cleanup)
                            if self.replay_buffer._is_from_external_source(old_path):
                                os.remove(old_path)
                                self.replay_buffer.migrated_count += 1
                                external_deleted += 1
                            else:
                                os.remove(old_path)
                            deleted_count += 1
                        except OSError:
                            pass  # File might already be gone
                    if external_deleted > 0:
                        print(f"  Migration cleanup: {external_deleted} external files deleted")
                    print(f"  Disk cleanup: deleted {deleted_count} evicted transition files")

                # Update total_added counter
                self.replay_buffer.total_added = file_id_counter
                
                # CRITICAL FIX: Update total_transitions counter
                # Each file contains exactly 1 transition (written by workers)
                num_new_transitions = len(all_transition_file_paths)
                num_evicted_transitions = len(old_paths_to_delete) if old_paths_to_delete else 0
                self.replay_buffer.total_transitions += num_new_transitions - num_evicted_transitions

                # Force DataLoader reset to ensure workers drop references to files we just pushed out
                if should_reset_workers:
                    print("  Buffer rotation detected: Resetting DataLoader workers to refresh file references...")
                    self._create_dataloader()
                    print("  ✓ DataLoader reset complete")

                print(f"  ✓ Buffer updated: {len(self.replay_buffer)} transitions")
            else:
                print("  No transitions collected this generation")

            # Aggregate statistics across all agents
            aggregate_stats = self._aggregate_population_stats(all_episode_stats, fitness_scores)

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

        # Debug: Check GPU memory at start of training
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [GPU] Memory at training start: {gpu_mem:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")

        # Use reduced gradient steps during stabilization phase or multi-mode for faster iteration
        # Normal: 32 steps × 192 batch = 6,144 samples (full exploration)
        # Stabilization/Multi: 10 steps × 192 batch = 1,920 samples (maintenance training)
        # Local: 8 steps for faster iteration on single GPU
        if self.local_mode:
            gradient_steps = Config.LOCAL_GRADIENT_STEPS_PER_GENERATION
            print(f"  [Local Mode: {gradient_steps} gradient steps (vs {Config.GRADIENT_STEPS_PER_GENERATION} distributed)]")
        elif self.breakthrough_state == BreakthroughState.STABILIZATION or self.multi_mode:
            gradient_steps = Config.GRADIENT_STEPS_PER_GENERATION_STABILIZATION
            mode_name = "Multi-Agent" if self.multi_mode else "Stabilization"
            print(f"  [{mode_name} Mode: {gradient_steps} gradient steps (vs {Config.GRADIENT_STEPS_PER_GENERATION} normal)]")
        else:
            gradient_steps = Config.GRADIENT_STEPS_PER_GENERATION

        # Train each agent
        # Local mode with GPU: Process agents in batches to manage GPU memory
        # Agents were moved to CPU after inference; we move them to GPU in batches
        # 32 agents / 8 per batch = 4 batches
        LOCAL_TRAINING_BATCH_SIZE = Config.LOCAL_TRAINING_AGENT_BATCH_SIZE

        # Local mode: Use larger batches to saturate GPU and minimize disk I/O
        # The LSTM processes batch_size × 117 sequences
        # Config.LOCAL_BATCH_SIZE (256) × 117 = 29,952 sequences - saturates GPU compute

        if self.local_mode and torch.cuda.is_available():
            # Batched GPU training for local mode
            # Uses move_to_device() to properly handle optimizer references and free GPU memory
            num_batches = (len(self.population) + LOCAL_TRAINING_BATCH_SIZE - 1) // LOCAL_TRAINING_BATCH_SIZE
            cpu_device = torch.device('cpu')

            # Local mode optimizations to reduce disk I/O (the bottleneck)
            # - Smaller population (32 vs 96)
            # - Fewer gradient steps (8 vs 32)
            # - Accumulation steps set via Config.LOCAL_GRADIENT_ACCUMULATION_STEPS
            local_accumulation_steps = Config.LOCAL_GRADIENT_ACCUMULATION_STEPS
            print(f"  [Local Mode] Training {num_batches} batches of {LOCAL_TRAINING_BATCH_SIZE} agents, {gradient_steps} steps/agent (batch_size={Config.LOCAL_BATCH_SIZE}, accum={local_accumulation_steps})")

            for batch_idx in range(num_batches):
                batch_start = batch_idx * LOCAL_TRAINING_BATCH_SIZE
                batch_end = min(batch_start + LOCAL_TRAINING_BATCH_SIZE, len(self.population))
                batch_agents = self.population[batch_start:batch_end]

                # Move batch of agents to GPU (with optimizer recreation)
                for agent in batch_agents:
                    agent.move_to_device(Config.DEVICE, recreate_optimizers=True)

                # Train agents in this batch using DataLoader iterator
                for agent in tqdm(batch_agents, desc=f"Training batch {batch_idx + 1}/{num_batches}"):
                    actor_losses = []
                    critic_losses = []

                    for step in range(gradient_steps):
                        for accum_step in range(local_accumulation_steps):
                            # Use DataLoader iterator (has async prefetching)
                            batch_cpu = next(self.batch_iterator)
                            batch = {k: v.to(Config.DEVICE, non_blocking=True) for k, v in batch_cpu.items()}

                            is_last_accum = (accum_step == local_accumulation_steps - 1)
                            critic_loss, actor_loss = agent.update(batch, accumulate=not is_last_accum)

                            if agent.agent_id == 0:
                                attention_weights = agent.actor.get_attention_weights()
                                if attention_weights is not None:
                                    self.update_feature_importance(attention_weights)

                            actor_losses.append(actor_loss.detach().cpu().item() if isinstance(actor_loss, torch.Tensor) else actor_loss)
                            critic_losses.append(critic_loss.detach().cpu().item() if isinstance(critic_loss, torch.Tensor) else critic_loss)
                            del batch

                    if agent.agent_id == 0:
                        self.writer.add_scalar('Train/Actor_Loss', np.mean(actor_losses), self.generation)
                        self.writer.add_scalar('Train/Critic_Loss', np.mean(critic_losses), self.generation)
                        wandb.log({
                            "train/actor_loss": np.mean(actor_losses),
                            "train/critic_loss": np.mean(critic_losses),
                        }, step=self.generation)

                    del actor_losses
                    del critic_losses

                # Move batch of agents back to CPU (with optimizer recreation to free GPU memory)
                for agent in batch_agents:
                    agent.move_to_device(cpu_device, recreate_optimizers=True)

                torch.cuda.empty_cache()

        else:
            # Original code path for distributed mode or CPU-only
            for agent in tqdm(self.population, desc="Training agents"):
                actor_losses = []
                critic_losses = []

                # Multiple gradient steps per agent
                for step in range(gradient_steps):
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

        # Clear GPU cache once after training all agents
        # Note: With expandable_segments=True, CUDA handles fragmentation efficiently
        # so we only need to clear once per generation, not after each agent
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

        # Determine if we should inject Global50 agents
        # Only inject until first breakthrough in heroes+consistency mode
        injection_pool = None
        injection_count = 0
        if (self.consistency_mode and self.heroes_hof_dir is not None and
            self.confirmed_breakthroughs == 0 and len(self.global50_injection_pool) > 0):
            injection_pool = self.global50_injection_pool
            injection_count = self.global50_injection_count
            print(f"  Global50 injection active (until first breakthrough): max {injection_count} agents")

        # Multi-mode stuck recovery: inject parent mutants after 5 generations without improvement
        # This helps the evolution "find its way back" if it drifted too far from the parent
        if self.multi_mode and self.multi_gens_since_improvement >= 5 and self.multi_parent_agent is not None:
            # Calculate how many mutant slots to use for parent mutants (half of total mutants)
            # Respect local mode's smaller population size
            pop_size = Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE
            num_elites = int(pop_size * Config.HEROES_ELITE_FRAC)
            num_offspring = int(pop_size * Config.HEROES_OFFSPRING_FRAC)
            num_mutants = pop_size - num_elites - num_offspring
            parent_injection_count = num_mutants // 2  # Half of mutants are parent-derived

            injection_pool = [self.multi_parent_agent]
            injection_count = parent_injection_count
            print(f"  🔄 STUCK RECOVERY: Injecting {parent_injection_count} parent mutants (stuck for {self.multi_gens_since_improvement} gens)")

        # Create next generation with adaptive mutation parameters
        # Elitism uses validation fitness for robustness and generalization
        # Tournament selection uses training fitness to maintain exploration
        # Note: Multi-mode uses standard evolution (sequential training, one member at a time)
        self.population = create_next_generation(
            old_population,
            fitness_scores,
            elite_scores=elite_scores,
            mutation_rate=self.current_mutation_rate,
            mutation_std=self.current_mutation_std,
            heroes_mode=self.heroes_hof_dir is not None or self.multi_mode,
            injection_pool=injection_pool,
            injection_count=injection_count
        )

        # Explicitly delete old agents and force GC
        for agent in old_population:
            del agent
        del old_population
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _check_multi_breakthrough(self, validation_scores):
        """
        Check for breakthrough on current committee member (sequential mode).

        A breakthrough occurs when the best agent exceeds the current baseline by 5%.
        After each breakthrough, the baseline is updated to the new score, so the next
        breakthrough requires 5% improvement over that new baseline (not compounding).

        Args:
            validation_scores: Validation fitness scores for population

        Returns:
            (best_agent, best_score) if breakthrough achieved, None otherwise
        """
        member_idx = self.current_member_idx
        baseline = self.member_baselines[member_idx]

        # Constant 5% improvement over current baseline
        # Note: baseline is updated after each breakthrough in _process_multi_breakthrough
        required_threshold = baseline * (1 + Config.MULTI_BREAKTHROUGH_THRESHOLD)

        # Find best agent
        best_idx = np.argmax(validation_scores)
        best_score = validation_scores[best_idx]

        if best_score >= required_threshold:
            return (self.population[best_idx], best_score)

        return None

    def _process_multi_improvement(self, improved_agent, score, val_result, is_breakthrough=False):
        """
        Process ANY improvement for current member: archive, save, update Global50 & committee.

        This is called whenever an agent beats the current baseline, regardless of whether
        it meets the 5% breakthrough threshold. This ensures we never lose progress.

        Args:
            improved_agent: The agent that achieved the improvement
            score: The BASE validation score (base_combined_fitness, WITHOUT ROI adjustment).
                   This is critical for fair comparison since baseline is also stored without
                   ROI adjustment. Using combined_fitness would cause score inflation when
                   the ROI hurdle changes.
            val_result: Full validation result dict with metrics
            is_breakthrough: If True, this improvement also qualifies as a breakthrough (5%+)
        """
        import shutil
        from committee import get_agent_filepath

        member_idx = self.current_member_idx
        member = self.multi_roster['members'][member_idx]
        context_window = self.multi_roster['context_window_days']

        improvement_pct = ((score / self.member_baselines[member_idx]) - 1) * 100 if self.member_baselines[member_idx] > 0 else 0

        if is_breakthrough:
            print(f"\n{'='*60}")
            print(f"🎉 BREAKTHROUGH for Member {member_idx}: {member['run_name']}_{member['agent_id']}")
            print(f"{'='*60}")
            print(f"  Breakthrough #{self.member_breakthroughs[member_idx] + 1} for this member")
        else:
            print(f"\n{'='*60}")
            print(f"📈 IMPROVEMENT for Member {member_idx}: {member['run_name']}_{member['agent_id']}")
            print(f"{'='*60}")

        print(f"  Baseline: {self.member_baselines[member_idx]:.2f} -> New: {score:.2f}")
        print(f"  Improvement: {improvement_pct:.1f}%")

        # Archive original member agent locally (cloud archiving is handled by check_and_promote)
        original_path = get_agent_filepath(member, context_window)

        # Archive locally (for local backup)
        if original_path.exists():
            archive_dir = original_path.parent / "archive"
            archive_dir.mkdir(exist_ok=True)
            suffix = f"_pre_multi_bt{self.member_breakthroughs[member_idx] + 1}" if is_breakthrough else f"_pre_multi_imp_gen{self.generation}"
            archive_path = archive_dir / f"{original_path.stem}{suffix}.pth"
            shutil.copy(original_path, archive_path)
            print(f"  Archived original to: {archive_path.name}")

        # NOTE: Cloud archiving (move from agents/ to archive/) is now handled by
        # check_and_promote via the replacing_entry parameter. This ensures atomic
        # operations where the parent entry is removed from the JSON and its file
        # is archived in a single transaction, preventing race conditions in --multi mode.

        # Save improved agent with new filename
        new_filename = f"{self.run_name}_{improved_agent.agent_id}.pth"
        new_path = original_path.parent / new_filename
        improved_agent.save(str(new_path))
        print(f"  Saved improved agent: {new_filename}")

        # Update Global50 ledger
        self._update_global50_for_multi_improvement(member_idx, new_filename, score, improved_agent, val_result)

        # CRITICAL: Update roster in memory so future loads use the improved agent
        # Must include ALL fields required by committee.py for consensus logic:
        # - filename, agent_id, run_name, gauntlet_score (identification)
        # - roi, expectancy, quality_ratio, win_ratio (metrics)
        # - stats.conviction_threshold_vector (95th percentile thresholds for conviction trades)
        old_run_name = member['run_name']
        old_agent_id = member['agent_id']

        # Extract metrics from validation result
        roi = val_result.get('roi', 0.0)
        expectancy = val_result.get('expectancy', 0.0)
        total_trades = val_result.get('total_trades', 0)
        quality_count = val_result.get('quality_count', 0)
        quality_ratio = (quality_count / total_trades) if total_trades > 0 else 0.0

        num_wins = val_result.get('num_wins', 0)
        num_losses = val_result.get('num_losses', 0)
        total_decisions = num_wins + num_losses
        win_ratio = (num_wins / total_decisions) if total_decisions > 0 else 0.0

        # Skip conviction threshold calculation during training - it's only needed for committee inference
        # The thresholds will be recalculated when the committee is loaded for production use
        # This saves ~10-30 seconds per improvement
        conviction_threshold_vector = member.get('stats', {}).get('conviction_threshold_vector', [0.0] * Config.NUM_INVESTABLE_STOCKS)

        self.multi_roster['members'][member_idx] = {
            'filename': new_filename,
            'run_name': self.run_name,
            'agent_id': improved_agent.agent_id,
            'gauntlet_score': score,
            'roi': roi,
            'expectancy': expectancy,
            'quality_ratio': quality_ratio,
            'win_ratio': win_ratio,
            'stats': {
                'conviction_threshold_vector': conviction_threshold_vector
            },
            # Preserve original identity for tracking lineage
            'original_run_name': member.get('original_run_name', old_run_name),
            'original_agent_id': member.get('original_agent_id', old_agent_id),
        }

        # Persist updated roster to disk so resume works correctly
        from committee import CommitteeManager
        manager = CommitteeManager(self.multi_roster['context_window_days'])
        manager.save_roster(self.multi_roster)
        print(f"  Updated roster: member {member_idx} now points to {new_filename} (synced to cloud)")

        # Update baseline for next improvement/breakthrough attempt
        self.member_baselines[member_idx] = score

        # Update member's starting ROI (self-referential hurdle)
        new_roi = val_result.get('roi', self.member_starting_rois[member_idx])
        self.member_starting_rois[member_idx] = new_roi
        self.roi_hurdle_ema = new_roi
        print(f"  Updated ROI hurdle: {self.member_starting_rois[member_idx]:.2f}%")

        # Update parent agent reference (for stuck recovery - now tracks latest improvement)
        self.multi_parent_agent = improved_agent.clone()
        self.multi_parent_agent.agent_id = -1

        # Reset stuck counter since we made progress
        self.multi_gens_since_improvement = 0
        self.multi_best_score_for_member = score

        if is_breakthrough:
            # Update breakthrough count for this member
            self.member_breakthroughs[member_idx] += 1
            
            # Track turnovers per agent based on phase
            # Note: In this context, "turnover" = "breakthrough", so 3 turnovers = 3 breakthroughs
            if self.multi_phase == 'non_maverick':
                # Find which non-maverick agent this is
                if member_idx in self.non_maverick_members:
                    list_idx = self.non_maverick_members.index(member_idx)
                    # Turnovers = breakthroughs (each breakthrough is a turnover)
                    breakthroughs_for_agent = self.member_breakthroughs[member_idx]
                    self.non_maverick_turnovers_per_agent[list_idx] = breakthroughs_for_agent
                    
                    if breakthroughs_for_agent >= Config.MULTI_TARGET_TURNOVERS:
                        print(f"\n  ✓ Non-Maverick Agent {list_idx} completed {Config.MULTI_TARGET_TURNOVERS} turnovers!")
            elif self.multi_phase == 'maverick':
                # Find which maverick agent this is
                if member_idx in self.maverick_members:
                    list_idx = self.maverick_members.index(member_idx)
                    breakthroughs_for_agent = self.member_breakthroughs[member_idx]
                    self.maverick_turnovers_per_agent[list_idx] = breakthroughs_for_agent
                    
                    if breakthroughs_for_agent >= Config.MULTI_TARGET_TURNOVERS:
                        print(f"\n  ✓ Maverick Agent {list_idx} completed {Config.MULTI_TARGET_TURNOVERS} turnovers!")

            # Print overall progress
            print(f"\n  Multi-Mode Progress:")
            print(f"    Phase: {self.multi_phase}")
            if self.multi_phase == 'non_maverick':
                print(f"    Current non-maverick agent: {self.current_non_maverick_idx}/{len(self.non_maverick_members)}")
                print(f"    Non-maverick turnovers: {self.non_maverick_turnovers_per_agent}")
            elif self.multi_phase == 'maverick':
                print(f"    Current maverick agent: {self.current_maverick_idx}/{len(self.maverick_members)}")
                print(f"    Maverick turnovers: {self.maverick_turnovers_per_agent}")
            print(f"    Breakthroughs: {self.member_breakthroughs}")

        print("="*60)

    def _process_multi_breakthrough(self, improved_agent, score, val_result):
        """
        Process a breakthrough (5%+ improvement) for current member.

        This is a wrapper around _process_multi_improvement that also handles
        breakthrough-specific logic (counting breakthroughs, advancing to next member).

        Args:
            improved_agent: The agent that achieved the breakthrough
            score: The validation score achieved (base_combined_fitness)
            val_result: Full validation result dict with metrics

        Returns:
            True if should advance to next member, False otherwise
        """
        # Process the improvement with breakthrough flag
        self._process_multi_improvement(improved_agent, score, val_result, is_breakthrough=True)

        return True  # Signal to advance to next member

    def _update_global50_for_multi_improvement(self, member_idx, new_filename, score, agent, val_result):
        """
        Update Global50 ledger with the improved agent.

        Args:
            member_idx: Index of the member being improved
            new_filename: Filename of the new agent
            score: Validation score achieved (base_combined_fitness)
            agent: The improved agent
            val_result: Full validation result dict with metrics
        """
        member = self.multi_roster['members'][member_idx]

        # Extract metrics from validation result
        roi = val_result.get('roi', 0.0)
        expectancy = val_result.get('expectancy', 0.0)
        cv = val_result.get('cv', 100.0)  # Default to high CV if not available
        total_trades = val_result.get('total_trades', 0)
        quality_count = val_result.get('quality_count', 0)
        quality_ratio = (quality_count / total_trades) if total_trades > 0 else 0.0

        num_wins = val_result.get('num_wins', 0)
        num_losses = val_result.get('num_losses', 0)
        total_decisions = num_wins + num_losses
        win_ratio = (num_wins / total_decisions) if total_decisions > 0 else 0.0

        # MULTI-MODE FIX: Bypass should_promote() gate by temporarily setting all thresholds to -inf/+inf
        # Committee members are already in Global50 - they're replacing themselves, not competing for entry.
        # The parent agent was archived from cloud storage but is still in the ledger until promotion completes.
        # Without this bypass, should_promote() uses thresholds that include the parent, causing rejection.
        original_thresholds = {
            'entry_threshold': self.global_hof.entry_threshold,
            'roi_threshold': self.global_hof.roi_threshold,
            'expectancy_threshold': self.global_hof.expectancy_threshold,
            'cv_threshold': self.global_hof.cv_threshold,
            'gauntlet_median': self.global_hof.gauntlet_median,
            'roi_median': self.global_hof.roi_median,
            'expectancy_median': self.global_hof.expectancy_median,
            'gauntlet_p25': self.global_hof.gauntlet_p25,
            'roi_p25': self.global_hof.roi_p25,
            'expectancy_p25': self.global_hof.expectancy_p25,
        }
        # Set thresholds to allow any agent through should_promote()
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

        # Upload to Global50 on every improvement
        # Pass the parent entry to be replaced so check_and_promote can:
        # 1. Remove it from the candidate pool before merging
        # 2. Archive its file as part of normal dropout handling
        # This prevents race conditions where the parent's entry persists in JSON
        # after its file has been archived by a concurrent run.
        parent_entry = (member['run_name'], member['agent_id'])
        promoted, rank = self.global_hof.check_and_promote(
            agent=agent,
            gauntlet_score=score,
            generation=self.generation,
            roi=roi,
            expectancy=expectancy,
            cv=cv,
            quality_ratio=quality_ratio,
            win_ratio=win_ratio,
            total_trades=total_trades,
            is_maverick=self.maverick_mode,
            replacing_entry=parent_entry
        )

        # Restore original thresholds (check_and_promote may have updated them via _update_entry_threshold)
        # We restore only if promotion failed; if it succeeded, thresholds are already recalculated correctly
        if not promoted:
            for key, value in original_thresholds.items():
                setattr(self.global_hof, key, value)

        if promoted:
            print(f"  ✓ Promoted to Global50: {new_filename} (Rank #{rank})")
            # Maverick stopping condition: Top 20 achieved
            if self.maverick_mode and rank <= Config.MAVERICK_TARGET_RANK:
                self._maverick_goal_achieved = True
                self._maverick_final_rank = rank
        else:
            print(f"  ℹ Not promoted to Global50 (score={score:.2f}, threshold={original_thresholds['entry_threshold']:.2f})")

    def _process_multi_turnover(self):
        """
        Process a turnover: all 9 members have achieved the same number of breakthroughs.
        Save milestone roster and persist updated roster to disk.
        """
        self.turnovers_completed = min(self.member_breakthroughs)

        print(f"\n{'='*60}")
        print(f"🎉 TURNOVER {self.turnovers_completed} COMPLETE!")
        print(f"   All {self.num_committee_members} committee members have achieved "
              f"{self.turnovers_completed} breakthrough(s)")
        print(f"{'='*60}")

        # Print breakthrough status for each member
        for member_idx in range(self.num_committee_members):
            member = self.multi_roster['members'][member_idx]
            bt = self.member_breakthroughs[member_idx]
            print(f"   Member {member_idx} ({member['run_name']}_{member['agent_id']}): "
                  f"{bt} breakthroughs")

        # Save milestone roster and persist updated roster
        from committee import CommitteeManager, convert_numpy_types
        import json
        context_window = self.multi_roster['context_window_days']
        manager = CommitteeManager(context_window)

        # FIX: Use local_roster_path (defined in committee.py) instead of roster_path
        roster_path = manager.local_roster_path
        archive_dir = roster_path.parent / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Save milestone archive (this marks the completion of turnover N)
        # Convert numpy types to native Python types for JSON serialization
        milestone_path = archive_dir / f"committee_roster_turnover_{self.turnovers_completed}_complete.json"
        with open(milestone_path, 'w') as f:
            json.dump(convert_numpy_types(self.multi_roster), f, indent=2)
        print(f"   Saved milestone roster: {milestone_path.name}")

        # Persist the updated roster as the active roster using standard save method
        manager.save_roster(self.multi_roster)
        print(f"   Updated active roster: {roster_path.name}")

        print(f"\n   Target turnovers: {Config.MULTI_TARGET_TURNOVERS}")
        print(f"   Progress: {self.turnovers_completed}/{Config.MULTI_TARGET_TURNOVERS}")

        if self.turnovers_completed >= Config.MULTI_TARGET_TURNOVERS:
            print(f"\n✓ MULTI-MODE TRAINING COMPLETE!")

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

    def calculate_conviction_threshold_vector(self, agent) -> list:
        """
        Calculate the 95th percentile conviction threshold vector for an agent.

        This runs the agent on validation data to collect coefficient predictions,
        then calculates the 95th percentile for each stock. This is used by the
        committee consensus logic to determine "conviction trades" - signals where
        an agent is exceptionally confident.

        Args:
            agent: The DDPGAgent to calculate thresholds for

        Returns:
            List of per-stock 95th percentile thresholds (length = NUM_INVESTABLE_STOCKS)
        """
        from committee import calculate_agent_stats_vectorized

        # Use validation slices to collect coefficient predictions
        # This gives us a representative sample of the agent's behavior
        if not self.current_generation_val_slices:
            # Generate validation slices if not available
            self.current_generation_val_slices = self.generate_validation_slices()

        # Collect coefficient predictions across multiple days
        all_coefficients = []

        agent.actor.eval()
        with torch.no_grad():
            for start_idx, end_idx, _ in self.current_generation_val_slices:
                # Iterate through each day in the slice
                for day_idx in range(start_idx, min(end_idx, start_idx + Config.TRADING_PERIOD_DAYS)):
                    # Get observation for this day
                    window_start = day_idx - Config.CONTEXT_WINDOW_DAYS
                    if window_start < 0:
                        continue

                    # Build observation
                    window = self.data_loader.data_array[window_start:day_idx]
                    normalized = (window - self.normalization_stats['mean']) / self.normalization_stats['std']
                    obs_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(Config.DEVICE)

                    # Get action from agent
                    action = agent.actor(obs_tensor).cpu().numpy()[0]  # [num_stocks, 2]

                    # Extract coefficients (first dimension)
                    coefficients = action[:, 0]  # [num_stocks]
                    all_coefficients.append(coefficients)

        if len(all_coefficients) == 0:
            # Fallback: return zeros (no conviction trades will trigger)
            print("  ⚠ No coefficient data collected for conviction threshold calculation")
            return [0.0] * Config.NUM_INVESTABLE_STOCKS

        # Stack into [Days, Stocks] array
        coeff_history = np.array(all_coefficients)  # [Days, Stocks]

        # Calculate 95th percentile for each stock
        p95_vector = calculate_agent_stats_vectorized(coeff_history)

        return p95_vector.tolist()

    def validate_agent(self, agent, quality_threshold: float = None) -> Dict:
        """
        Validate agent using walk-forward validation on 10 validation slices.

        Walk-forward validation strategy:
        - Runs agent on 10 validation slices (same slices for all agents in this generation)
        - 4 slices from quarters + 3 straddling slices + 3 random slices
        - Uses weighted aggregation: fitness = (0.4 * mean) + (0.6 * worst_case)
        - This rewards consistency and penalizes agents that fail in any market condition

        Args:
            agent: The agent to validate
            quality_threshold: Optional threshold for counting quality trades (trades with gain_pct >= threshold)

        Returns:
            Validation results with 'fitness' emphasizing worst-case performance
        """
        if agent is None:
            return {}

        if not self.current_generation_val_slices:
            raise ValueError("No validation slices generated for this generation")

        # Run agent on all 10 validation slices
        slice_results = []
        all_closed_trades = []  # Collect closed trades for metrics calculation only

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
                'total_investment': episode_info.get('total_investment', 0.0),
                'peak_capital_employed': episode_info.get('peak_capital_employed', 0.0)
            })

            # Collect closed trades from this slice
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        # Extract fitness scores from all 10 slices
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
        total_investment = sum([r['total_investment'] for r in slice_results])  # Legacy cumulative
        # Pooled ROI: sum of peak capitals (slices are parallel/independent scenarios)
        # This answers: "For every dollar of max drawdown capacity across all scenarios, how much profit?"
        total_peak_capital = sum([r['peak_capital_employed'] for r in slice_results])
        roi = (total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0

        # Calculate global win rate (total wins / total trades across all slices)
        # This ensures WR >= QR (quality rate) since all quality trades are winning trades
        total_wins = sum([r['num_wins'] for r in slice_results])
        total_losses = sum([r['num_losses'] for r in slice_results])
        total_trades = total_wins + total_losses
        global_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        # Calculate Expectancy metric from all closed trades
        expectancy = self.calculate_expectancy(all_closed_trades)

        # Calculate quality count if threshold provided (to avoid returning full closed_trades list)
        quality_count = 0
        if quality_threshold is not None:
            quality_count = sum(1 for trade in all_closed_trades if trade.get('gain_pct', 0) >= quality_threshold)

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
            'total_trades': total_trades,  # Total trades across all slices
            'quality_count': quality_count,  # Count of quality trades (if threshold provided)
        }

    def validate_agent_cached(self, agent: DDPGAgent, quality_threshold: float = None) -> Dict:
        """
        Validate agent with caching - skip validation if agent weights unchanged.

        Elite agents often have unchanged weights across generations, so we can
        skip re-validation and use cached results for significant speedup (~20%).

        Args:
            agent: Agent to validate
            quality_threshold: Optional threshold for counting quality trades

        Returns:
            Validation results (from cache or fresh evaluation)
        """
        agent_hash = self._hash_agent(agent)
        slice_hash = self.val_slice_hash
        # Include quality_threshold in cache key to handle different thresholds
        threshold_key = f"{quality_threshold:.4f}" if quality_threshold is not None else "none"
        cache_key = f"{agent_hash}_{slice_hash}_{threshold_key}"

        if cache_key in self.validation_cache:
            # Cache hit - return cached results
            return self.validation_cache[cache_key]

        # Cache miss - run validation
        val_results = self.validate_agent(agent, quality_threshold=quality_threshold)
        self.validation_cache[cache_key] = val_results

        # Keep cache bounded (last 100 entries to prevent memory growth)
        if len(self.validation_cache) > 100:
            # Remove oldest entry (first key in dict)
            oldest_key = next(iter(self.validation_cache))
            del self.validation_cache[oldest_key]

        return val_results

    def validate_population_parallel(self, quality_threshold: float = None) -> List[Dict]:
        """
        Validate entire population in parallel using ProcessPoolExecutor.

        Similar to evaluate_population_parallel but for validation phase.
        Checks cache first to skip validation for unchanged agents.

        Args:
            quality_threshold: Optional threshold for counting quality trades

        Returns:
            List of validation results for each agent in population
        """
        print(f"\n--- Walk-Forward Validation (Generation {self.generation + 1}) ---")

        # Prepare validation slices (shared across all agents)
        if not self.current_generation_val_slices:
            raise ValueError("No validation slices generated for this generation")

        validation_slices = self.current_generation_val_slices
        slice_hash = self.val_slice_hash
        threshold_key = f"{quality_threshold:.4f}" if quality_threshold is not None else "none"

        # Check cache and prepare tasks only for agents that need validation
        validation_results = [None] * len(self.population)  # Pre-allocate results list
        tasks = []
        agent_indices_to_validate = []

        for idx, agent in enumerate(self.population):
            agent_hash = self._hash_agent(agent)
            cache_key = f"{agent_hash}_{slice_hash}_{threshold_key}"

            if cache_key in self.validation_cache:
                # Cache hit - use cached results
                validation_results[idx] = self.validation_cache[cache_key]
            else:
                # Cache miss - need to validate this agent
                # Extract agent state (CPU tensors only to avoid CUDA sharing issues)
                agent_state = {
                    'actor': {k: v.cpu() for k, v in agent.actor.state_dict().items()},
                    'critic': {k: v.cpu() for k, v in agent.critic.state_dict().items()}
                }

                # Create unique seed for this validation task
                task_seed = self.seed + self.generation * 10000 + idx * 100

                # Use Penalized Median for multi-mode (consistent with Gauntlet/committee.py)
                use_penalized_median = self.multi_mode

                tasks.append((agent_state, validation_slices, quality_threshold, task_seed, use_penalized_median))
                agent_indices_to_validate.append((idx, agent_hash, cache_key))

        # If all agents were cached, return early
        if not tasks:
            print(f"✓ All {len(self.population)} agents validated (100% cache hit rate)")
            return validation_results

        # Prepare environment config using shared memory (eliminates serialization overhead)
        # Note: start_idx/end_idx are dummy values here, overridden per slice in worker reset()
        # Use gauntlet_mode during stabilization/gauntlet for soft zero-trades penalty
        use_gauntlet_mode = self.breakthrough_state in (BreakthroughState.STABILIZATION, BreakthroughState.GAUNTLET)
        env_config = self._get_shared_env_config(
            start_idx=self.val_start_idx,
            end_idx=self.val_end_idx,
            trading_end_idx=self.val_start_idx + Config.TRADING_PERIOD_DAYS,
            is_training=False,  # Validation mode: no noise
            gauntlet_mode=use_gauntlet_mode
        )

        # Execute validation in parallel
        num_workers = min(mp.cpu_count() - 1, Config.EVAL_NUM_WORKERS)
        cache_hits = len(self.population) - len(tasks)
        print(f"Validating {len(tasks)} agents ({cache_hits} cached, {len(tasks)} fresh)")
        print(f"Using {num_workers} parallel workers (out of {mp.cpu_count()} vCPUs)")

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(env_config,)
        ) as executor:
            # Submit all validation tasks
            future_to_idx = {
                executor.submit(_run_validation_worker, task): agent_info
                for task, agent_info in zip(tasks, agent_indices_to_validate)
            }

            # Collect results as they complete
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(tasks),
                desc="Validating (parallel)",
                disable=False
            ):
                idx, agent_hash, cache_key = future_to_idx[future]

                try:
                    val_results = future.result()

                    # Store results
                    validation_results[idx] = val_results

                    # Update cache
                    self.validation_cache[cache_key] = val_results

                    # Keep cache bounded
                    if len(self.validation_cache) > 100:
                        oldest_key = next(iter(self.validation_cache))
                        del self.validation_cache[oldest_key]

                except Exception as e:
                    print(f"\n⚠ Validation failed for agent {idx}: {e}")
                    # Return empty results for failed validation
                    validation_results[idx] = {
                        'fitness': -1000.0,
                        'fitness_mean': -1000.0,
                        'fitness_min': -1000.0,
                        'roi': 0.0,
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'quality_count': 0,
                        'quality_roi': 0.0,
                        'sample_trade': None
                    }

        return validation_results

    def run_gauntlet_validation(self, agent: DDPGAgent) -> Dict:
        """
        Run rigorous Gauntlet validation on candidate agent.

        The Gauntlet is a stress test using 20+ validation slices from both
        training and validation data to ensure the agent performs robustly
        across all available market regimes.

        Unlike normal validation (7 slices, validation-only), the Gauntlet:
        - Uses 20+ slices (10 from training, 10 from validation)
        - Tests across ALL market conditions, not just validation period
        - Uses Penalized Median scoring: Median - (0.5 * StdDev) to reward stability

        This prevents "Ghost Scores" where agents get lucky on specific
        validation slices but fail when tested more broadly.

        Args:
            agent: Candidate agent to test

        Returns:
            Dict with gauntlet_score (pessimistic aggregator) and detailed metrics
        """
        print(f"\n{'='*60}")
        print(f"🎯 GAUNTLET VALIDATION - Rigorous Stress Test")
        print(f"{'='*60}")

        # Generate Gauntlet slices (20 diverse slices from training + validation)
        gauntlet_slices = self.generate_gauntlet_slices()
        print(f"Testing on {len(gauntlet_slices)} slices (10 training + 10 validation)")

        # Enable gauntlet mode for soft zero-trades penalty (tactical no-trade is acceptable)
        self.eval_env.set_gauntlet_mode(True)

        # Run agent on all Gauntlet slices
        slice_results = []
        all_closed_trades = []  # Collect for expectancy calculation only

        for i, (start_idx, end_idx, _) in enumerate(gauntlet_slices):
            # Use batched inference for faster validation
            fitness, episode_info = self.run_episode_batched(
                agent=agent,
                env=self.eval_env,
                start_idx=start_idx,
                end_idx=end_idx,
                training=False,
                batch_size=16
            )

            # Apply zero-trades gradient (same as normal validation)
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

            # Collect closed trades for expectancy calculation only
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{len(gauntlet_slices)} slices...")

        # Extract fitness scores
        fitness_scores = [result['fitness'] for result in slice_results]

        # Calculate statistics for Penalized Median scoring
        scores_np = np.array(fitness_scores)
        max_score = float(np.max(scores_np))
        min_score = float(np.min(scores_np))
        mean_score = float(np.mean(scores_np))
        median_score = float(np.median(scores_np))
        std_score = float(np.std(scores_np))

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

        # Calculate aggregate metrics
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
        total_trades = total_wins + total_losses
        global_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        expectancy = self.calculate_expectancy(all_closed_trades)

        # Calculate quality_count (trades with gain >= quality threshold)
        # Use the current quality threshold (HoF median ROI or config default)
        quality_threshold = self.quality_threshold
        if all_closed_trades:
            quality_count = sum(1 for t in all_closed_trades if t.get('gain_pct', 0) >= quality_threshold)
        else:
            quality_count = 0

        # Display detailed visualization of slice scores
        from utils.display import visualize_gauntlet_slices
        visualize_gauntlet_slices(fitness_scores, mean_score, min_score, max_score, gauntlet_score)

        # Print trading metrics summary
        print(f"\n{'='*70}")
        print(f"{'GAUNTLET TRADING METRICS':^70}")
        print(f"{'='*70}")
        print(f"  ROI:               {roi:>12.2f}%")
        print(f"  Win Rate:          {global_win_rate:>11.1%}")
        print(f"  Total Trades:      {total_trades:>12}")
        print(f"  Quality Trades:    {quality_count:>12}")
        print(f"  Expectancy:        {expectancy:>11.2f}%")
        print(f"{'='*70}")

        # Reset gauntlet mode after validation
        self.eval_env.set_gauntlet_mode(False)

        return {
            'gauntlet_score': gauntlet_score,
            'raw_gauntlet_score': raw_gauntlet_score,  # Pre-efficiency adjustment
            'efficiency_ratio': efficiency_ratio,  # ROI / baseline
            'cv': cv,  # Coefficient of Variation - lower is more stable
            'median_fitness': median_score,
            'std_fitness': std_score,
            'mean_fitness': mean_score,
            'min_fitness': min_score,
            'max_fitness': max_score,
            'fitness_all_slices': fitness_scores,
            'roi': roi,
            'win_rate': global_win_rate,
            'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
            'total_trades': total_trades,
            'quality_count': quality_count,
            'expectancy': expectancy,
        }

    def stress_test_candidates(self, breaching_agents: list) -> list:
        """
        PHASE 2: STRESS TEST - Pre-stabilization filter to identify the strongest candidate.

        Instead of blindly selecting the top agent, run a quick "audition" on all candidates
        to filter out agents that got lucky on normal validation but fail under stress.

        The Logic:
        1. Run single validation episode on each candidate
        2. Re-rank candidates based on pessimistic fitness
        3. Select winner based on stress test performance

        This prevents wasting GPU time stabilizing weak candidates that won't survive the Gauntlet.

        Args:
            breaching_agents: List of candidate agents that passed initial threshold

        Returns:
            Ranked list of candidates based on stress test performance
        """
        if not Config.STRESS_TEST_ENABLED or not breaching_agents:
            return breaching_agents

        print(f"\n{'='*60}")
        print(f"⚡ STRESS TEST - Pre-Stabilization Audition")
        print(f"{'='*60}")
        print(f"  Testing {len(breaching_agents)} candidates with pessimistic fitness")
        print(f"  Metric: Pessimistic fitness (0.4*mean + 0.6*worst)")
        print(f"{'='*60}")

        # Run stress test on each candidate
        stress_results = []
        for i, agent_info in enumerate(breaching_agents, 1):
            agent_idx = agent_info['idx']
            agent = self.population[agent_idx]

            # Run single validation episode (use first validation slice for speed)
            if not self.current_generation_val_slices:
                print(f"  ⚠️  Warning: No validation slices available for stress test")
                return breaching_agents

            start_idx, end_idx, _ = self.current_generation_val_slices[0]

            # Run episode in validation mode
            fitness, episode_info = self.run_episode_batched(
                agent=agent,
                env=self.eval_env,
                start_idx=start_idx,
                end_idx=end_idx,
                training=False,
                batch_size=16
            )

            # Extract metrics
            num_trades = episode_info.get('num_trades', 0)
            win_rate = episode_info.get('win_rate', 0.0)

            stress_results.append({
                'idx': agent_idx,
                'original_fitness': agent_info['fitness'],
                'stress_fitness': fitness,
                'num_trades': num_trades,
                'win_rate': win_rate
            })

            print(f"  Agent {agent_idx:2d}: Fitness={fitness:>8.2f} | Trades={num_trades:>3d} | WinRate={win_rate*100:>5.1f}%")

        # Sort by stress test fitness (descending)
        stress_results.sort(key=lambda x: x['stress_fitness'], reverse=True)

        print(f"\n{'='*60}")
        print(f"✅ STRESS TEST COMPLETE")
        print(f"{'='*60}")
        winner = stress_results[0]
        print(f"  Winner: Agent {winner['idx']}")
        print(f"  Stress Fitness: {winner['stress_fitness']:.2f}")
        print(f"  Trades: {winner['num_trades']} | WinRate: {winner['win_rate']*100:.1f}%")
        print(f"{'='*60}")

        # Return re-ranked candidates (mapped back to original breaching_agents format)
        reranked = []
        for result in stress_results:
            # Find original agent_info
            original = next(a for a in breaching_agents if a['idx'] == result['idx'])
            reranked.append(original)

        return reranked

    def check_for_breakthrough(self, validation_results: list) -> bool:
        """
        Check if current generation has a potential breakthrough.

        A breakthrough is detected when sufficient agents (quorum) exceed the confirmed baseline
        by the breakthrough threshold (10% in normal mode, 5% in consistency mode).

        This implementation uses a queue-based system to prevent the "Ghost Loop" and
        "Winner-Takes-All" problems:
        - All qualifying agents are added to a candidate queue
        - Agents are tested one at a time through the Gauntlet
        - Failed agents are weeded out by natural selection and the next candidate is selected
        - Baseline only ratchets after ALL candidates have been exhausted

        Args:
            validation_results: List of validation results for all agents, sorted by combined_fitness

        Returns:
            True if breakthrough detected, False otherwise
        """
        if not self.gauntlet_mode_enabled:
            return False

        # WARMUP PERIOD: Prevent breakthrough detection until Generation > threshold
        # Let the population churn before declaring winners
        # Single-agent mode uses longer stabilization to let clones settle
        warmup_generations = (Config.SINGLE_STABILIZATION_GENERATIONS
                              if self.single_agent_mode
                              else Config.BREAKTHROUGH_WARMUP_GENERATIONS)
        if self.generation <= warmup_generations:
            return False

        if self.breakthrough_state != BreakthroughState.NORMAL:
            # Already processing a breakthrough
            return False

        # Check if we have candidates in the queue from a previous detection (heroes mode only)
        if self.use_candidate_queue and self.candidate_queue:
            # Try next candidate from queue
            return self._select_next_candidate_from_queue()

        # Find all agents that breach the threshold
        breaching_agents = []
        for result in validation_results:
            agent_idx = result['idx']
            val_fitness = result['validation_fitness']

            # Skip agents we've already tested (heroes mode only)
            if self.use_candidate_queue and agent_idx in self.tested_candidate_indices:
                continue

            # Calculate improvement over confirmed baseline
            if self.confirmed_baseline <= 0:
                # First breakthrough: any positive score qualifies
                improvement = 1.0 if val_fitness > 0 else 0.0
            else:
                improvement = (val_fitness - self.confirmed_baseline) / abs(self.confirmed_baseline)

            if improvement >= self.breakthrough_threshold:
                breaching_agents.append({
                    'idx': agent_idx,
                    'fitness': val_fitness,
                    'improvement': improvement
                })

        # PHASE 1: QUALITY CULL - Deduplication & Truncation
        # Remove weak candidates before wasting GPU time on them
        if breaching_agents:
            original_count = len(breaching_agents)

            # Step 1: Deduplicate - remove agents with identical fitness
            seen_fitness = {}
            deduplicated = []
            for agent_info in breaching_agents:
                fitness = agent_info['fitness']
                if fitness not in seen_fitness:
                    seen_fitness[fitness] = True
                    deduplicated.append(agent_info)

            duplicates_removed = original_count - len(deduplicated)

            # Step 2: Truncate - keep only top K candidates
            # Sort by fitness descending (highest first)
            deduplicated.sort(key=lambda x: x['fitness'], reverse=True)
            max_candidates = Config.MAX_CANDIDATES_FOR_STRESS_TEST
            truncated = deduplicated[:max_candidates]
            weak_removed = len(deduplicated) - len(truncated)

            if duplicates_removed > 0 or weak_removed > 0:
                print(f"\n{'='*60}")
                print(f"📊 ASSET SELECTION - QUALITY CULL")
                print(f"{'='*60}")
                print(f"  Original candidates: {original_count}")
                if duplicates_removed > 0:
                    print(f"  Duplicates removed: {duplicates_removed} (identical fitness)")
                if weak_removed > 0:
                    print(f"  Weak candidates culled: {weak_removed} (below top {max_candidates})")
                print(f"  Remaining candidates: {len(truncated)}")
                print(f"{'='*60}")

            breaching_agents = truncated

        # PHASE 2: STRESS TEST - Pre-Stabilization Filter
        # Run quick audition on remaining candidates to filter out weak agents
        breaching_agents = self.stress_test_candidates(breaching_agents)

        # If all candidates failed stress test, continue normal evolution
        if not breaching_agents:
            print(f"\n  Returning to normal evolution (no candidates passed stress test)\n")
            return False

        # Check if we have quorum
        if len(breaching_agents) >= self.breakthrough_quorum:
            print(f"\n{'='*60}")
            print(f"🔥 POTENTIAL BREAKTHROUGH DETECTED!")
            print(f"{'='*60}")
            print(f"  Quorum achieved: {len(breaching_agents)}/{self.breakthrough_quorum} agents breached threshold")
            print(f"  Confirmed Baseline: {self.confirmed_baseline:.2f}")
            print(f"  Threshold: {self.breakthrough_threshold*100:.0f}% improvement")

            if self.use_candidate_queue:
                # HEROES MODE: Queue-based selection to prevent Ghost Loop
                print(f"\n  All qualifying agents (will be tested sequentially):")
                for i, agent_info in enumerate(breaching_agents, 1):
                    already_tested = " [ALREADY TESTED]" if agent_info['idx'] in self.tested_candidate_indices else ""
                    print(f"    {i}. Agent {agent_info['idx']:2d}: {agent_info['fitness']:>8.2f} ({agent_info['improvement']*100:>5.1f}% improvement){already_tested}")

                # Populate candidate queue with ALL breaching agents (not just quorum)
                # This ensures we test all qualified heroes, not just the top few
                self.candidate_queue = breaching_agents.copy()
            else:
                # NORMAL MODE: Original simple selection (top agent only)
                print(f"\n  Breaching agents:")
                for i, agent_info in enumerate(breaching_agents[:self.breakthrough_quorum], 1):
                    print(f"    {i}. Agent {agent_info['idx']:2d}: {agent_info['fitness']:>8.2f} ({agent_info['improvement']*100:>5.1f}% improvement)")

                # Take the top quorum agents for the candidate
                top_breaching = breaching_agents[:self.breakthrough_quorum]

            print(f"{'='*60}")

            # Save snapshot before entering Gauntlet (for debugging/safety backup)
            # NOTE: No longer used for automatic restoration on failure (soft penalty approach)
            self.save_gauntlet_snapshot()

            if self.use_candidate_queue:
                # Heroes mode: Select first candidate from queue
                return self._select_next_candidate_from_queue()
            else:
                # Normal mode: Original logic (select best agent)
                best_agent_idx = top_breaching[0]['idx']
                best_fitness = top_breaching[0]['fitness']

                # Transition to DETECTION state
                self.breakthrough_state = BreakthroughState.DETECTION
                self.breakthrough_candidate = BreakthroughCandidate(
                    agent=self.population[best_agent_idx].clone(),
                    agents=[self.population[a['idx']].clone() for a in top_breaching],
                    agent_idx=best_agent_idx,
                    agent_indices=[a['idx'] for a in top_breaching],
                    spike_score=best_fitness,
                    spike_scores=[a['fitness'] for a in top_breaching],
                    detection_generation=self.generation
                )

                # Log breakthrough detection to wandb
                wandb.log({
                    'gauntlet/stabilization_phase': 0,
                }, step=self.generation)

                return True

        return False

    def _select_next_candidate_from_queue(self) -> bool:
        """
        Select the next untested candidate from the queue.

        This helper method handles the queue-based candidate selection to prevent
        the Ghost Loop problem. After a rejection, we move to the next candidate
        instead of re-selecting the same agent.

        Returns:
            True if a candidate was selected, False if queue is exhausted
        """
        while self.candidate_queue:
            # Pop the next candidate (highest fitness first)
            candidate_info = self.candidate_queue.pop(0)
            agent_idx = candidate_info['idx']

            # Check if we've already tested this agent
            if agent_idx in self.tested_candidate_indices:
                print(f"  ⏩ Skipping Agent {agent_idx} (already tested)")
                continue

            # Valid untested candidate found
            best_agent_idx = agent_idx
            best_fitness = candidate_info['fitness']

            print(f"\n{'='*60}")
            print(f"🎯 SELECTING CANDIDATE FROM QUEUE")
            print(f"{'='*60}")
            print(f"  Agent {best_agent_idx}: {best_fitness:.2f}")
            print(f"  Remaining in queue: {len(self.candidate_queue)}")
            print(f"  Already tested: {len(self.tested_candidate_indices)} agents")
            print(f"{'='*60}")

            # Mark this agent as tested
            self.tested_candidate_indices.add(best_agent_idx)

            # Transition to DETECTION state
            self.breakthrough_state = BreakthroughState.DETECTION
            self.breakthrough_candidate = BreakthroughCandidate(
                agent=self.population[best_agent_idx].clone(),
                agents=[self.population[best_agent_idx].clone()],  # Single agent for queue-based selection
                agent_idx=best_agent_idx,
                agent_indices=[best_agent_idx],
                spike_score=best_fitness,
                spike_scores=[best_fitness],
                detection_generation=self.generation
            )

            # Log queue-based breakthrough detection to wandb
            wandb.log({
                'gauntlet/stabilization_phase': 0,
            }, step=self.generation)

            return True

        # Queue exhausted - no more candidates to test
        print(f"\n{'='*60}")
        print(f"✅ CANDIDATE QUEUE EXHAUSTED")
        print(f"{'='*60}")
        print(f"  All {len(self.tested_candidate_indices)} qualifying agents have been tested")

        # Apply any pending baseline update (fixes Winner-Takes-All problem)
        if self.pending_baseline_update is not None:
            old_baseline = self.confirmed_baseline
            self.confirmed_baseline = self.pending_baseline_update
            self.confirmed_breakthroughs += 1
            print(f"\n  📈 APPLYING DEFERRED BASELINE RATCHET")
            print(f"     Old Baseline: {old_baseline:.2f}")
            print(f"     New Baseline: {self.confirmed_baseline:.2f}")
            print(f"     This was the highest score from all tested heroes")
            self.pending_baseline_update = None

        print(f"  Clearing tested agents set for next detection cycle")
        print(f"{'='*60}")

        # Clear the tested set for next breakthrough cycle
        self.tested_candidate_indices.clear()
        return False

    def process_gauntlet_state_machine(self, validation_results=None):
        """
        Process Gauntlet Mode state machine transitions.

        State transitions:
        - NORMAL: Monitor for breakthroughs
        - DETECTION → STABILIZATION: Lock candidate, begin stabilization
        - STABILIZATION: Count generations, transition to GAUNTLET when complete
        - GAUNTLET: Run stress test, transition to CONFIRMED or REJECTED
        - CONFIRMED: Apply ratchet, return to NORMAL
        - REJECTED: Return to NORMAL

        Args:
            validation_results: Current generation's validation results (sorted by combined_fitness)
        """
        if not self.gauntlet_mode_enabled:
            return

        if self.breakthrough_state == BreakthroughState.DETECTION:
            # Safety check: Ensure candidate exists
            if self.breakthrough_candidate is None:
                print(f"\n⚠️ WARNING: Breakthrough candidate is None at stabilization entry!")
                print(f"  Returning to NORMAL state.\n")
                self.breakthrough_state = BreakthroughState.NORMAL
                return

            # Transition to STABILIZATION
            print(f"\n{'='*60}")
            print(f"⏳ STABILIZATION PHASE STARTED")
            print(f"{'='*60}")
            print(f"  Locking training on candidate for {Config.STABILIZATION_GENERATIONS} generations")
            print(f"  Goal: Allow Actor/Critic networks to converge on new behavior")
            print(f"{'='*60}")

            # LOCK THE POPULATION: Replace all agents with mutants of the breakthrough candidate
            # This ensures the "Siege" actually happens - the candidate survives and the population
            # focuses on fine-tuning that specific strategy rather than evolving away from it
            print(f"\n  🔒 LOCKING POPULATION: Replacing all agents with mutants of Candidate (Agent {self.breakthrough_candidate.agent_idx})")

            # Store old population for cleanup
            old_population = self.population

            # Keep the candidate itself as the first agent (Elite)
            new_population = [self.breakthrough_candidate.agent.clone()]
            new_population[0].agent_id = 0
            new_population[0].is_elite = True

            # Use a tighter mutation rate for stabilization (fine-tuning, not exploration)
            # This creates small variations to find the optimal point in the strategy's neighborhood
            stabilization_mutation_rate = 0.05  # 5% of weights mutated (vs typical 10-20%)
            stabilization_mutation_std = self.current_mutation_std * 0.5  # Half the normal noise

            # Fill the rest of the population with mutants of the candidate
            # Respect local mode's smaller population size
            pop_size = Config.LOCAL_POPULATION_SIZE if self.local_mode else Config.POPULATION_SIZE
            for i in range(1, pop_size):
                mutant = mutate(
                    self.breakthrough_candidate.agent,
                    mutation_rate=stabilization_mutation_rate,
                    mutation_std=stabilization_mutation_std
                )
                mutant.agent_id = i
                mutant.is_elite = False
                new_population.append(mutant)

            # Replace the trainer's population
            self.population = new_population

            print(f"  ✓ Population locked: 1 elite + {pop_size - 1} mutants")
            print(f"  ✓ Stabilization mutation: rate={stabilization_mutation_rate}, std={stabilization_mutation_std:.4f}")

            # Cleanup old population to free memory
            for agent in old_population:
                del agent
            del old_population
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.breakthrough_state = BreakthroughState.STABILIZATION
            self.breakthrough_candidate.stabilization_start_gen = self.generation
            self.stabilization_generations_elapsed = 0

            # Log stabilization start to wandb
            wandb.log({
                'gauntlet/stabilization_phase': 0,
            }, step=self.generation)

        elif self.breakthrough_state == BreakthroughState.STABILIZATION:
            # Progressive Stabilization: Fail fast if candidate collapses
            self.stabilization_generations_elapsed += 1

            print(f"\n🔄 Stabilization: {self.stabilization_generations_elapsed}/{Config.STABILIZATION_GENERATIONS} generations")

            # PROGRESSIVE STABILIZATION CHECK: Abort if performance drops below baseline
            # (Disabled by default - ghost detection is too aggressive and doesn't give candidates a fair chance)
            if Config.STABILIZATION_GHOST_DETECTION and validation_results is not None and len(validation_results) > 0:
                best_result = validation_results[0]
                best_val_fitness = best_result['validation_fitness']

                # Check if current best is below baseline (ghost collapse detected)
                if best_val_fitness < self.confirmed_baseline:
                    print(f"\n{'='*60}")
                    print(f"❌ PROGRESSIVE STABILIZATION FAILED - GHOST DETECTED")
                    print(f"{'='*60}")
                    print(f"  Current Best Fitness: {best_val_fitness:.2f}")
                    print(f"  Confirmed Baseline: {self.confirmed_baseline:.2f}")
                    print(f"  Candidate collapsed below baseline - aborting stabilization")
                    print(f"  Returning to NORMAL state")
                    print(f"{'='*60}")

                    # Log ghost detection to wandb
                    wandb.log({
                        'gauntlet/stabilization_phase': 0,
                    }, step=self.generation)

                    # Clear the current candidate
                    self.breakthrough_candidate = None
                    self.stabilization_generations_elapsed = 0

                    # CRITICAL: Try next candidate from queue BEFORE evolution scrambles indices
                    if self.use_candidate_queue and self.candidate_queue:
                        print(f"\n  ⏩ Immediately switching to next candidate ({len(self.candidate_queue)} remaining)...")
                        next_candidate_found = self._select_next_candidate_from_queue()

                        if next_candidate_found:
                            print(f"  ✓ Next candidate loaded - will enter STABILIZATION on next state machine cycle")
                            return  # Exit early - stay in DETECTION state
                        else:
                            print(f"  ✗ Queue exhausted - returning to normal evolution")

                    # FIX: If we have a pending update from a previous candidate in this batch, apply it now!
                    if self.pending_baseline_update is not None:
                        old_baseline = self.confirmed_baseline
                        self.confirmed_baseline = self.pending_baseline_update
                        self.confirmed_breakthroughs += 1
                        print(f"\n  ⚠️ Candidate failed stabilization, but recovering pending breakthrough...")
                        print(f"     Old Baseline: {old_baseline:.2f}")
                        print(f"     New Baseline: {self.confirmed_baseline:.2f}")
                        print(f"  ✓ Breakthrough recovered! Count: {self.confirmed_breakthroughs}")
                        self.pending_baseline_update = None
                        self.tested_candidate_indices.clear()

                    # Only reach here if no next candidate was found
                    self.breakthrough_state = BreakthroughState.NORMAL
                    return

            # Log stabilization progress to wandb
            wandb.log({
                'gauntlet/stabilization_phase': self.stabilization_generations_elapsed,
            }, step=self.generation)

            if self.stabilization_generations_elapsed >= Config.STABILIZATION_GENERATIONS:
                # Stabilization complete - select current best agent for gauntlet
                print(f"\n{'='*60}")
                print(f"✅ STABILIZATION COMPLETE - Evaluating Current Best Agent")
                print(f"{'='*60}")

                # Get the current best agent from this generation's validation
                if validation_results is None or len(validation_results) == 0:
                    print(f"\n⚠️ WARNING: No validation results available at gauntlet entry!")
                    wandb.log({
                        'gauntlet/stabilization_phase': 0,
                    }, step=self.generation)

                    # Clear the current candidate
                    self.breakthrough_candidate = None
                    self.stabilization_generations_elapsed = 0

                    # CRITICAL: Try next candidate from queue BEFORE evolution scrambles indices
                    if self.use_candidate_queue and self.candidate_queue:
                        print(f"\n  ⏩ Immediately switching to next candidate ({len(self.candidate_queue)} remaining)...")
                        next_candidate_found = self._select_next_candidate_from_queue()

                        if next_candidate_found:
                            print(f"  ✓ Next candidate loaded - will enter STABILIZATION on next state machine cycle")
                            return  # Exit early - stay in DETECTION state
                        else:
                            print(f"  ✗ Queue exhausted - returning to normal evolution")

                    # FIX: If we have a pending update from a previous candidate in this batch, apply it now!
                    if self.pending_baseline_update is not None:
                        old_baseline = self.confirmed_baseline
                        self.confirmed_baseline = self.pending_baseline_update
                        self.confirmed_breakthroughs += 1
                        print(f"\n  ⚠️ No validation results, but recovering pending breakthrough...")
                        print(f"     Old Baseline: {old_baseline:.2f}")
                        print(f"     New Baseline: {self.confirmed_baseline:.2f}")
                        print(f"  ✓ Breakthrough recovered! Count: {self.confirmed_breakthroughs}")
                        self.pending_baseline_update = None
                        self.tested_candidate_indices.clear()

                    # Only reach here if no next candidate was found
                    self.breakthrough_state = BreakthroughState.NORMAL
                    return

                # Best agent is first in validation_results (already sorted by combined_fitness)
                best_result = validation_results[0]
                best_agent_idx = best_result['idx']
                best_val_fitness = best_result['validation_fitness']
                best_combined_fitness = best_result['combined_fitness']

                print(f"  Current Best Agent: Agent {best_agent_idx}")
                print(f"  Validation Fitness: {best_val_fitness:.2f}")
                print(f"  Combined Fitness: {best_combined_fitness:.2f}")
                print(f"  Confirmed Baseline: {self.confirmed_baseline:.2f}")

                # Check if the best agent clears the hurdle to enter the gauntlet
                # Calculate improvement over confirmed baseline
                if self.confirmed_baseline <= 0:
                    improvement = 1.0 if best_val_fitness > 0 else 0.0
                else:
                    improvement = (best_val_fitness - self.confirmed_baseline) / abs(self.confirmed_baseline)

                if improvement < self.breakthrough_threshold:
                    # Current best agent does not clear the hurdle
                    print(f"\n{'='*60}")
                    print(f"⚠️ CURRENT BEST AGENT FAILED TO CLEAR HURDLE")
                    print(f"{'='*60}")
                    print(f"  Agent {best_agent_idx} does not clear the hurdle")
                    print(f"  Required improvement: {self.breakthrough_threshold*100:.0f}%")
                    print(f"  Actual improvement: {improvement*100:.1f}%")
                    print(f"{'='*60}")

                    # FALLBACK: Check if the original agent can still enter the gauntlet
                    if self.breakthrough_candidate is not None:
                        original_agent_idx = self.breakthrough_candidate.agent_idx

                        # Only try fallback if original agent is different from current best
                        if original_agent_idx != best_agent_idx:
                            print(f"\n🎲 FALLBACK: Checking original breakthrough agent (Agent {original_agent_idx})...")

                            # Find the original agent in validation results
                            original_result = None
                            for result in validation_results:
                                if result['idx'] == original_agent_idx:
                                    original_result = result
                                    break

                            if original_result is not None:
                                original_val_fitness = original_result['validation_fitness']

                                # Calculate improvement for original agent
                                if self.confirmed_baseline <= 0:
                                    original_improvement = 1.0 if original_val_fitness > 0 else 0.0
                                else:
                                    original_improvement = (original_val_fitness - self.confirmed_baseline) / abs(self.confirmed_baseline)

                                print(f"  Original Agent {original_agent_idx} validation fitness: {original_val_fitness:.2f}")
                                print(f"  Original Agent improvement: {original_improvement*100:.1f}%")

                                if original_improvement >= self.breakthrough_threshold:
                                    # Original agent clears the hurdle! Give it a shot
                                    print(f"\n{'='*60}")
                                    print(f"🎯 FALLBACK SUCCESS!")
                                    print(f"{'='*60}")
                                    print(f"  Original Agent {original_agent_idx} clears the hurdle!")
                                    print(f"  Proceeding to Gauntlet with original agent...")
                                    print(f"{'='*60}")

                                    # Update breakthrough candidate with current version of original agent
                                    self.breakthrough_candidate = BreakthroughCandidate(
                                        agent=self.population[original_agent_idx].clone(),
                                        agents=[self.population[original_agent_idx].clone()],
                                        agent_idx=original_agent_idx,
                                        agent_indices=[original_agent_idx],
                                        spike_score=self.breakthrough_candidate.spike_score,
                                        spike_scores=[self.breakthrough_candidate.spike_score],
                                        detection_generation=self.breakthrough_candidate.detection_generation
                                    )

                                    # Continue to gauntlet (don't return)
                                    # Set a flag to skip the normal agent selection below
                                    fallback_used = True

                                else:
                                    print(f"  Original agent also fails to clear hurdle ({original_improvement*100:.1f}% < {self.breakthrough_threshold*100:.0f}%)")
                                    fallback_used = False
                            else:
                                print(f"  Could not find original agent in validation results")
                                fallback_used = False
                        else:
                            print(f"  Original agent is same as current best - no fallback needed")
                            fallback_used = False
                    else:
                        print(f"  No original breakthrough candidate available for fallback")
                        fallback_used = False

                    # If fallback was not used or failed, try next candidate or return to NORMAL
                    if not fallback_used:
                        print(f"\n{'='*60}")
                        print(f"❌ FAILED TO ENTER GAUNTLET")
                        print(f"{'='*60}")
                        print(f"  No agent clears the hurdle")
                        print(f"{'='*60}")

                        # Log failure to wandb
                        wandb.log({
                            'gauntlet/stabilization_phase': 0,
                        }, step=self.generation)

                        # Clear the current candidate
                        self.breakthrough_candidate = None
                        self.stabilization_generations_elapsed = 0

                        # CRITICAL: Try next candidate from queue BEFORE evolution scrambles indices
                        if self.use_candidate_queue and self.candidate_queue:
                            print(f"\n  ⏩ Immediately switching to next candidate ({len(self.candidate_queue)} remaining)...")
                            next_candidate_found = self._select_next_candidate_from_queue()

                            if next_candidate_found:
                                print(f"  ✓ Next candidate loaded - will enter STABILIZATION on next state machine cycle")
                                return  # Exit early - stay in DETECTION state
                            else:
                                print(f"  ✗ Queue exhausted - returning to normal evolution")

                        # FIX: If we have a pending update from a previous candidate in this batch, apply it now!
                        if self.pending_baseline_update is not None:
                            old_baseline = self.confirmed_baseline
                            self.confirmed_baseline = self.pending_baseline_update
                            self.confirmed_breakthroughs += 1
                            print(f"\n  ⚠️ Candidate failed hurdle, but recovering pending breakthrough...")
                            print(f"     Old Baseline: {old_baseline:.2f}")
                            print(f"     New Baseline: {self.confirmed_baseline:.2f}")
                            print(f"  ✓ Breakthrough recovered! Count: {self.confirmed_breakthroughs}")
                            self.pending_baseline_update = None
                            self.tested_candidate_indices.clear()

                        # Inject Global 50 agents to recover diversity after failed stabilization
                        if validation_results:
                            print(f"\n💉 INJECTING GLOBAL 50 AGENTS TO RECOVER DIVERSITY")
                            # Sort by agent index to ensure fitness matches population order
                            sorted_results = sorted(validation_results, key=lambda x: x['idx'])
                            training_fitness_scores = [r['training_fitness'] for r in sorted_results]
                            self._inject_global50_agents(training_fitness_scores)

                        # Only reach here if no next candidate was found
                        self.breakthrough_state = BreakthroughState.NORMAL
                        return
                else:
                    # Current best agent clears the hurdle
                    fallback_used = False

                # Only execute this block if we're proceeding to gauntlet normally (not via fallback)
                if not fallback_used:
                    # Agent clears the hurdle - proceed to gauntlet
                    print(f"  ✓ Agent clears hurdle ({improvement*100:.1f}% improvement)")
                    print(f"  Proceeding to Gauntlet validation...")
                    print(f"{'='*60}")

                    # Update breakthrough candidate with current best agent
                    # Keep original spike score for tracking, but use current agent
                    original_spike_score = self.breakthrough_candidate.spike_score if self.breakthrough_candidate else best_val_fitness
                    original_agent_idx = self.breakthrough_candidate.agent_idx if self.breakthrough_candidate else None

                    # If the current best agent is different from the original queued agent, mark it as tested
                    if self.use_candidate_queue and original_agent_idx is not None and best_agent_idx != original_agent_idx:
                        print(f"  Note: Current best agent (Agent {best_agent_idx}) differs from originally queued agent (Agent {original_agent_idx})")
                        print(f"  Marking Agent {best_agent_idx} as tested to prevent re-queuing")
                        self.tested_candidate_indices.add(best_agent_idx)

                    self.breakthrough_candidate = BreakthroughCandidate(
                        agent=self.population[best_agent_idx].clone(),
                        agents=[self.population[best_agent_idx].clone()],
                        agent_idx=best_agent_idx,
                        agent_indices=[best_agent_idx],
                        spike_score=original_spike_score,  # Keep original for comparison
                        spike_scores=[original_spike_score],
                        detection_generation=self.breakthrough_candidate.detection_generation if self.breakthrough_candidate else self.generation
                    )

                self.breakthrough_state = BreakthroughState.GAUNTLET

                # Log gauntlet test start to wandb
                wandb.log({
                    'gauntlet/stabilization_phase': 0,
                }, step=self.generation)

                # Run Gauntlet validation on the current best agent
                gauntlet_results = self.run_gauntlet_validation(self.breakthrough_candidate.agent)
                gauntlet_score = gauntlet_results['gauntlet_score']

                # Store Gauntlet score
                self.breakthrough_candidate.gauntlet_score = gauntlet_score

                # Check if agent passed the Gauntlet
                # FIRST BREAKTHROUGH: Accept any gauntlet score to establish initial baseline
                # This "gives away" the first breakthrough to provide a realistic starting point
                # Subsequent breakthroughs: gauntlet_score must exceed confirmed_baseline
                is_first_breakthrough = (self.confirmed_breakthroughs == 0)
                passed_gauntlet = is_first_breakthrough or (gauntlet_score > self.confirmed_baseline)

                if passed_gauntlet:
                    # CONFIRMED - Apply Ratchet (potentially deferred)
                    self.breakthrough_state = BreakthroughState.CONFIRMED

                    print(f"\n{'='*60}")
                    if is_first_breakthrough:
                        print(f"⭐ FIRST BREAKTHROUGH - BASELINE ESTABLISHED!")
                    else:
                        print(f"⭐ BREAKTHROUGH CONFIRMED!")
                    print(f"{'='*60}")
                    print(f"  Spike Score: {self.breakthrough_candidate.spike_score:.2f} (lucky)")
                    print(f"  Gauntlet Score: {gauntlet_score:.2f} (robust)")
                    print(f"  Previous Baseline: {self.confirmed_baseline:.2f}")
                    if is_first_breakthrough:
                        print(f"  NOTE: First breakthrough - accepting any gauntlet score to establish realistic baseline")

                    # Store old baseline before update
                    old_baseline = self.confirmed_baseline

                    # Check if we should defer the baseline ratchet (heroes mode only)
                    if self.use_candidate_queue and self.candidate_queue:
                        # HEROES MODE: Defer ratchet to allow other heroes to be tested
                        # This allows other heroes to be tested against the original baseline
                        if self.pending_baseline_update is None or gauntlet_score > self.pending_baseline_update:
                            self.pending_baseline_update = gauntlet_score
                        print(f"  Pending Baseline: {gauntlet_score:.2f} (deferred - {len(self.candidate_queue)} candidates remaining)")
                        print(f"  Current Baseline: {self.confirmed_baseline:.2f} (unchanged)")
                        print(f"  Reason: Testing remaining heroes against original baseline to avoid Winner-Takes-All")
                    else:
                        # NORMAL MODE or queue exhausted: Apply ratchet immediately
                        self.confirmed_baseline = gauntlet_score
                        self.confirmed_breakthroughs += 1
                        print(f"  New Baseline: {gauntlet_score:.2f} (applied immediately)")
                        if self.use_candidate_queue:
                            print(f"  Reason: Candidate queue exhausted")

                    print(f"{'='*60}")

                    # Record breakthrough event (only if baseline was actually applied)
                    # If deferred, we'll record it when the baseline is finally applied
                    if not self.candidate_queue:
                        breakthrough_event = {
                            'generation': self.generation,
                            'spike_score': self.breakthrough_candidate.spike_score,
                            'gauntlet_score': gauntlet_score,
                            'old_baseline': old_baseline,
                            'new_baseline': gauntlet_score,
                            'roi': gauntlet_results['roi'],
                            'win_rate': gauntlet_results['win_rate'],
                            'expectancy': gauntlet_results['expectancy']
                        }
                        self.breakthrough_history.append(breakthrough_event)

                    # GLOBAL 50: Attempt to promote agent to Global Hall of Fame
                    # All promotions use consistency-aligned gauntlet scores for fair comparison
                    if self.consistency_mode:
                        # Already in consistency mode - use existing gauntlet results
                        g50_gauntlet_score = gauntlet_score
                        g50_gauntlet_results = gauntlet_results
                    else:
                        # Normal mode: Check if agent might qualify, then run consistency-aligned re-gauntlet
                        g50_gauntlet_score = None
                        g50_gauntlet_results = None

                        if self.global_hof.enabled and self.global_hof.should_promote(
                            gauntlet_score,
                            gauntlet_results.get('roi', 0.0),
                            gauntlet_results.get('expectancy', 0.0),
                            gauntlet_results.get('cv', 100.0)
                        ):
                            print(f"\n   ⓘ Agent may qualify for Global 50 (passes all promotion criteria)")
                            print(f"   Running consistency-aligned re-gauntlet for fair Global 50 comparison...")

                            # Temporarily enable consistency mode on eval_env
                            self.eval_env.set_consistency_mode(True)
                            try:
                                g50_gauntlet_results = self.run_gauntlet_validation(self.breakthrough_candidate.agent)
                                g50_gauntlet_score = g50_gauntlet_results['gauntlet_score']
                                print(f"   Consistency-aligned gauntlet score: {g50_gauntlet_score:.2f}")
                            finally:
                                # Always reset consistency mode back to False
                                self.eval_env.set_consistency_mode(False)
                        else:
                            print(f"   ⓘ Agent gauntlet score: {gauntlet_score:.2f}")
                            if self.global_hof.enabled:
                                print(f"   Global 50 min threshold: {self.global_hof.entry_threshold:.2f} | p25: {self.global_hof.gauntlet_p25:.2f}")

                    # Attempt promotion if we have a consistency-aligned score
                    should_promote_result = False
                    if g50_gauntlet_score is not None:
                        should_promote_result = self.global_hof.should_promote(
                            g50_gauntlet_score,
                            g50_gauntlet_results.get('roi', 0.0),
                            g50_gauntlet_results.get('expectancy', 0.0),
                            g50_gauntlet_results.get('cv', 100.0)
                        )
                    
                    if g50_gauntlet_score is not None and should_promote_result:
                        agent_to_admit = self.breakthrough_candidate.agent
                        agent_roi = g50_gauntlet_results['roi']
                        agent_expectancy = g50_gauntlet_results['expectancy']
                        agent_cv = g50_gauntlet_results.get('cv', 100.0)
                        quality_count = g50_gauntlet_results.get('quality_count', 0)
                        total_trades = g50_gauntlet_results.get('total_trades', 0)
                        win_rate = g50_gauntlet_results.get('win_rate', 0.0)

                        # Calculate quality_ratio from quality_count and total_trades
                        quality_ratio = quality_count / total_trades if total_trades > 0 else 0.0

                        # Try to promote to Global 50
                        promoted, rank = self.global_hof.check_and_promote(
                            agent=agent_to_admit,
                            gauntlet_score=g50_gauntlet_score,
                            generation=self.generation,
                            roi=agent_roi,
                            expectancy=agent_expectancy,
                            cv=agent_cv,
                            quality_ratio=quality_ratio,
                            win_ratio=win_rate,
                            total_trades=total_trades,
                            is_maverick=self.maverick_mode
                        )

                        if promoted:
                            print(f"\n{'='*60}")
                            print(f"🎯 AGENT PROMOTED TO GLOBAL 50!")
                            print(f"{'='*60}")
                            print(f"  Rank: #{rank}")
                            print(f"  Gauntlet Score: {g50_gauntlet_score:.2f}")
                            if self.maverick_mode:
                                print(f"  Type: Maverick")
                                print(f"  Target Rank: <= #{Config.MAVERICK_TARGET_RANK}")
                            
                            # Maverick stopping condition: Top 20 achieved
                            if self.maverick_mode and rank <= Config.MAVERICK_TARGET_RANK:
                                self._maverick_goal_achieved = True
                                self._maverick_final_rank = rank
                                print(f"\n  ✅ MAVERICK GOAL ACHIEVED!")
                                print(f"  Training will stop at end of this generation")
                            elif self.maverick_mode:
                                print(f"\n  ⏳ Maverick goal not yet achieved (rank #{rank} > #{Config.MAVERICK_TARGET_RANK})")
                                print(f"  Training continues...")
                            print(f"{'='*60}\n")
                    elif g50_gauntlet_score is not None and self.maverick_mode:
                        # Agent passed gauntlet but didn't qualify for promotion
                        print(f"\n  ⓘ Agent did not qualify for Global 50 promotion")
                        
                        # Explicitly analyze and print promotion failure reasons
                        _, reasons = self.global_hof.analyze_promotion(
                            g50_gauntlet_score,
                            g50_gauntlet_results.get('roi', 0.0),
                            g50_gauntlet_results.get('expectancy', 0.0),
                            g50_gauntlet_results.get('cv', 100.0)
                        )
                        print("  ❌ Promotion Metrics Analysis:")
                        for r in reasons:
                            print(f"     {r}")
                            
                        print(f"  Training continues to find agent that reaches rank <= #{Config.MAVERICK_TARGET_RANK}")

                    # Log to wandb
                    wandb.log({
                        'gauntlet/confirmed_breakthroughs': self.confirmed_breakthroughs,
                        'gauntlet/confirmed_baseline': self.confirmed_baseline,
                        'gauntlet/stabilization_phase': 0,
                    }, step=self.generation)

                    # Return to NORMAL state
                    self.breakthrough_state = BreakthroughState.NORMAL
                    self.breakthrough_candidate = None
                    self.stabilization_generations_elapsed = 0

                else:
                    # REJECTED - Failed Gauntlet
                    self.breakthrough_state = BreakthroughState.REJECTED

                    print(f"\n{'='*60}")
                    print(f"❌ BREAKTHROUGH REJECTED")
                    print(f"{'='*60}")
                    print(f"  Spike Score: {self.breakthrough_candidate.spike_score:.2f} (Ghost Score!)")
                    print(f"  Gauntlet Score: {gauntlet_score:.2f} (Reality)")
                    print(f"  Confirmed Baseline: {self.confirmed_baseline:.2f}")
                    print(f"  Agent failed stress test - applying soft penalty")
                    print(f"  Population keeps evolutionary progress (no snapback)")
                    print(f"{'='*60}")

                    # Log to wandb
                    wandb.log({
                        'gauntlet/stabilization_phase': 0,
                    }, step=self.generation)

                    # SOFT PENALTY APPROACH: No population snapback
                    # The failed candidate will naturally be weeded out by selection pressure
                    # The rest of the population keeps their evolutionary progress
                    # This allows the model to continue forward instead of being trapped in the past

                    # Clear the current candidate
                    self.breakthrough_candidate = None
                    self.stabilization_generations_elapsed = 0

                    # CRITICAL FIX: Immediately select next candidate from queue BEFORE evolution runs
                    # If we wait until the next generation, evolve_population() will scramble the indices
                    # and the queue entries will point to wrong agents (Ghost Index Problem)
                    if self.use_candidate_queue and self.candidate_queue:
                        print(f"\n  ⏩ Immediately switching to next candidate ({len(self.candidate_queue)} remaining)...")
                        next_candidate_found = self._select_next_candidate_from_queue()

                        if next_candidate_found:
                            # Successfully transitioned to DETECTION state
                            # The next call to process_gauntlet_state_machine will move to STABILIZATION
                            print(f"  ✓ Next candidate loaded - will enter STABILIZATION on next state machine cycle")
                            return  # Exit early - don't reset to NORMAL
                        else:
                            print(f"  ✗ Queue exhausted - returning to normal evolution")
                    else:
                        if self.use_candidate_queue:
                            print(f"  No more candidates in queue - returning to normal evolution")
                        else:
                            print(f"  Returning to normal evolution")

                    # FIX: If we have a pending update from a previous candidate in this batch, apply it now!
                    # This fixes the "Last Candidate Trap" where the batch credit is lost if the last candidate fails
                    if self.pending_baseline_update is not None:
                        old_baseline = self.confirmed_baseline
                        self.confirmed_baseline = self.pending_baseline_update
                        self.confirmed_breakthroughs += 1
                        print(f"\n  ⚠️ Last candidate failed, but recovering pending breakthrough...")
                        print(f"     Old Baseline: {old_baseline:.2f}")
                        print(f"     New Baseline: {self.confirmed_baseline:.2f}")
                        print(f"  ✓ Breakthrough recovered! Count: {self.confirmed_breakthroughs}")
                        self.pending_baseline_update = None
                        self.tested_candidate_indices.clear()

                    # Inject Global 50 agents to recover diversity after failed Gauntlet
                    if validation_results:
                        print(f"\n💉 INJECTING GLOBAL 50 AGENTS TO RECOVER DIVERSITY")
                        # validation_results is sorted by combined_fitness, so we must resort by idx
                        # to match population order before extracting fitness
                        sorted_results = sorted(validation_results, key=lambda x: x['idx'])
                        training_fitness_scores = [r['training_fitness'] for r in sorted_results]
                        self._inject_global50_agents(training_fitness_scores)

                    # Only reach here if no next candidate was found
                    self.breakthrough_state = BreakthroughState.NORMAL

    def check_hof_turnover(self):
        """
        Check for Hall of Fame turnover in consistency mode.

        Unified turnover logic for all N turnovers:
        - HoF must be full (10 agents)
        - All agents must have ROI >= current median baseline
        - Initial median is 0 (since only gauntlet-passing agents enter HoF, all have score > 0)
        - Each turnover establishes new median: median₀=0 → median₁ → median₂ → ...

        This creates a ratcheting quality bar where the entire HoF must progressively improve.
        Training ends when target_hof_turnovers is reached.
        """
        if not self.consistency_mode:
            return

        # Check if HoF is full
        if len(self.hall_of_fame.entries) != self.hall_of_fame.capacity:
            return  # Wait until HoF is full

        # Initialize median to 0 for first check (all gauntlet agents have score > 0 by design)
        current_baseline = self.hof_current_median if self.hof_current_median is not None else 0.0

        # Check if all HoF agents have cleared the current baseline
        all_cleared_baseline = all(entry.roi >= current_baseline for entry in self.hall_of_fame.entries)

        if all_cleared_baseline:
            # Turnover complete!
            self.hof_turnover_count += 1
            previous_median = current_baseline
            self.hof_current_median = self.hall_of_fame.get_median_roi()
            self.generation_at_last_turnover = self.generation  # Reset fallback counter

            print(f"\n{'='*60}")
            print(f"🏆 HALL OF FAME TURNOVER {self.hof_turnover_count} COMPLETE!")
            print(f"{'='*60}")
            print(f"  HoF is full with {len(self.hall_of_fame.entries)} agents")
            print(f"  All agents cleared baseline: ROI >= {previous_median:.2f}%")
            print(f"  New median ROI: {self.hof_current_median:.2f}%")
            print(f"  Progress: {self.hof_turnover_count}/{self.target_hof_turnovers} turnovers")

            # Print top agents in current HoF
            sorted_entries = sorted(self.hall_of_fame.entries, key=lambda e: e.roi, reverse=True)
            print(f"\n  Top 5 agents in HoF:")
            for i, entry in enumerate(sorted_entries[:5], 1):
                print(f"    {i}. Agent {entry.agent_id}: ROI={entry.roi:.2f}%, Gen={entry.generation}")

            if self.hof_turnover_count >= self.target_hof_turnovers:
                print(f"\n  🎯 TRAINING OBJECTIVE ACHIEVED!")
                print(f"  Hall of Fame is now an Archive of the Proven")
            else:
                print(f"\n  Next goal: All HoF agents must have ROI >= {self.hof_current_median:.2f}%")
            print(f"{'='*60}")

            # Log to wandb
            wandb.log({
                'hof_turnover/turnover_complete': 1,
                'hof_turnover/previous_median': previous_median,
                'hof_turnover/current_median': self.hof_current_median,
                'hof_turnover/count': self.hof_turnover_count,
            }, step=self.generation)

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
                # Show stdout first (actual error messages usually go here)
                if result.stdout:
                    # Filter out tqdm progress bars, show last 500 chars
                    stdout_lines = [l for l in result.stdout.split('\n') if not l.strip().startswith('Processing')]
                    filtered_stdout = '\n'.join(stdout_lines[-10:])  # Last 10 lines
                    if filtered_stdout.strip():
                        print(f"  Output: {filtered_stdout}")
                # Show stderr (often contains traceback)
                if result.stderr:
                    # Filter out tqdm progress bars
                    stderr_lines = [l for l in result.stderr.split('\n') if not ('|' in l and '%' in l)]
                    filtered_stderr = '\n'.join(stderr_lines[-15:])  # Last 15 lines
                    if filtered_stderr.strip():
                        print(f"  Error:\n{filtered_stderr}")
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

    def _inject_global50_agents(self, fitness_scores: List[float]):
        """
        Inject mutated Global 50 agents to replace the worst performers in the population.
        Used during plateau detection AND after gauntlet failures to restore genetic diversity.

        Args:
            fitness_scores: List of fitness scores for the current population (used to find worst agents)
        """
        # Skip injection in multi-mode: we want to refine each committee member's
        # specific strategy, not dilute it with external Global50 agents
        if self.multi_mode:
            return

        # Only trigger if buffer is at least half full
        buffer_half_full = len(self.replay_buffer) >= (self.replay_buffer.capacity // 2)

        if not buffer_half_full:
            print(f"\n  ⚠ Global 50 injection skipped (buffer < 50% full: {len(self.replay_buffer)}/{self.replay_buffer.capacity})")
            return

        if fitness_scores is None or len(fitness_scores) == 0:
            return

        print(f"\n🧬 Global 50 Injection Protocol Activated")
        print(f"  Buffer status: {len(self.replay_buffer)}/{self.replay_buffer.capacity} (>50% full)")

        # Refresh global50 entries from cloud before injection to get latest view
        self.global_hof.refresh()

        # Inject up to 10 unique Global 50 agents (or as many as available)
        max_injections = 10

        # Count agents from current league + fallback leagues
        g50_agents_available = len(self.global_hof.entries) if hasattr(self.global_hof, 'entries') else 0
        if hasattr(self.global_hof, 'fallback_leagues'):
            for fallback in self.global_hof.fallback_leagues:
                g50_agents_available += fallback.get('entry_count', 0)

        num_to_request = min(max_injections, g50_agents_available, len(fitness_scores))

        if num_to_request > 0:
            # Get unique random agents (no duplicates)
            g50_agents = self.global_hof.get_random_agents(num_to_request)

            if len(g50_agents) > 0:
                print(f"  Injecting {len(g50_agents)} unique agents (requested: {num_to_request}, available: {g50_agents_available})")

                # Get indices of worst agents (sorted ascending by fitness)
                worst_indices = np.argsort(fitness_scores)[:len(g50_agents)]
                mutation_rate = Config.MUTATION_RATE_CONSISTENCY
                max_fitness_val = np.max(fitness_scores) if len(fitness_scores) > 0 else 1.0

                injected_count = 0
                replaced_slots = []
                for worst_idx, g50_agent in zip(worst_indices, g50_agents):
                    # Reset noise_scale to encourage exploration
                    # Injected agents from HoF may have decayed noise (converged to ~0.01)
                    # but need fresh exploration in the new population context
                    g50_agent.noise_scale = Config.NOISE_SCALE

                    # Apply mutation to the Global 50 agent
                    mutated_g50 = mutate(g50_agent, mutation_rate=mutation_rate, mutation_std=self.base_mutation_std)

                    # Replace worst agent
                    mutated_g50.agent_id = worst_idx
                    mutated_g50.is_elite = False
                    self.population[worst_idx] = mutated_g50

                    # Update fitness score to protect it from immediate culling
                    fitness_scores[worst_idx] = max_fitness_val

                    injected_count += 1
                    replaced_slots.append(worst_idx)
                    del g50_agent

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"  ✓ Injected {injected_count} agents into slots {replaced_slots} (mutation rate: {mutation_rate:.2f})")
            else:
                print(f"  ⚠ Global 50 injection skipped (failed to load agents)")
        else:
            print(f"  ⚠ Global 50 injection skipped (no agents available)")

    def check_and_adjust_mutation(self, current_val_fitness: float, fitness_scores: List[float] = None):
        """
        Check for fitness plateau and adaptively increase mutation parameters.
        Plateau is detected if validation fitness doesn't improve by plateau_threshold
        over the last plateau_window generations.

        When a plateau is detected and buffer is at least half full, inject a mutated
        Global 50 agent to replace the worst agent in the population.

        Args:
            current_val_fitness: Current validation fitness
            fitness_scores: Optional fitness scores of current population (for finding worst agent)
        """
        # GUARD: Disable plateau detection AND history recording during Stabilization/Gauntlet phases
        # During these phases, the population is locked to "1 elite + mutants of candidate" to force
        # convergence on a specific strategy. Recording these scores would pollute the history with
        # "frozen" population data, blinding the detector to long-term stagnation.
        # By skipping recording, Gen 50 compares to Gen 30 if Gens 31-49 were stabilization.
        if self.breakthrough_state != BreakthroughState.NORMAL:
            return

        # Add current fitness to history ONLY in NORMAL state
        # This ensures validation_fitness_history is a contiguous record of evolutionary progress
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

                # Global 50 Injection: Replace worst agents with mutated Global 50 agents
                if fitness_scores is not None and len(fitness_scores) > 0:
                    self._inject_global50_agents(fitness_scores)

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

            # Leverage mode state
            'leverage_mode_active': self.leverage_mode_active,
            'leverage_generations_remaining': self.leverage_generations_remaining,

            # ROI hurdle tracking
            'roi_hurdle_ema': self.roi_hurdle_ema,

            # Multi-Agent Mode State
            'multi_mode': self.multi_mode,
            'multi_state': {
                'current_member_idx': self.current_member_idx,
                'turnovers_completed': self.turnovers_completed,
                'member_breakthroughs': self.member_breakthroughs,
                'member_training_start_gen': self.member_training_start_gen,
                # We save these to ensure continuity, though load_multi_agents recalculates them
                'member_baselines': self.member_baselines,
                'member_starting_rois': getattr(self, 'member_starting_rois', []),
                # Stuck detection state
                'multi_gens_since_improvement': getattr(self, 'multi_gens_since_improvement', 0),
                'multi_best_score_for_member': getattr(self, 'multi_best_score_for_member', float('-inf')),
                # Phase tracking for maverick/non-maverick separation
                'multi_phase': getattr(self, 'multi_phase', 'non_maverick'),
                'non_maverick_members': getattr(self, 'non_maverick_members', []),
                'maverick_members': getattr(self, 'maverick_members', []),
                'current_non_maverick_idx': getattr(self, 'current_non_maverick_idx', 0),
                'current_maverick_idx': getattr(self, 'current_maverick_idx', 0),
                'non_maverick_turnovers_per_agent': getattr(self, 'non_maverick_turnovers_per_agent', []),
                'maverick_turnovers_per_agent': getattr(self, 'maverick_turnovers_per_agent', []),
            } if self.multi_mode else None,

            # Gauntlet Mode state
            'gauntlet_mode_enabled': self.gauntlet_mode_enabled,
            'breakthrough_state': self.breakthrough_state.value if self.gauntlet_mode_enabled else None,
            'confirmed_baseline': self.confirmed_baseline,
            'confirmed_breakthroughs': self.confirmed_breakthroughs,
            'breakthrough_history': self.breakthrough_history,
            'stabilization_generations_elapsed': self.stabilization_generations_elapsed,

            # Breakthrough candidate state (includes agent_idx, spike_score, detection_gen, etc.)
            'breakthrough_candidate': {
                'agent_idx': self.breakthrough_candidate.agent_idx,
                'spike_score': self.breakthrough_candidate.spike_score,
                'detection_generation': self.breakthrough_candidate.detection_generation,
                'stabilization_start_gen': self.breakthrough_candidate.stabilization_start_gen,
                'gauntlet_score': self.breakthrough_candidate.gauntlet_score,
            } if self.breakthrough_candidate is not None else None,

            # Heroes queue state (for consistency mode with heroes)
            'use_candidate_queue': self.use_candidate_queue,
            'candidate_queue': self.candidate_queue,  # List of {idx, fitness, agent_id} dicts
            'tested_candidate_indices': list(self.tested_candidate_indices),  # Convert set to list for JSON
            'pending_baseline_update': self.pending_baseline_update,

            # Hall of Fame turnover tracking (for consistency mode)
            'hof_turnover_count': self.hof_turnover_count,
            'hof_current_median': self.hof_current_median,
            'generation_at_last_turnover': self.generation_at_last_turnover,

            # Configuration mode flags (for proper restoration context)
            'consistency_mode': self.consistency_mode,
            'enable_leverage': self.enable_leverage,
            'breakthrough_threshold': self.breakthrough_threshold,
            'breakthrough_quorum': self.breakthrough_quorum,
            'target_breakthroughs': self.target_breakthroughs,
            'target_hof_turnovers': self.target_hof_turnovers,

            # Normalization stats (to avoid recomputing on resume)
            'normalization_stats': {
                'mean': self.normalization_stats['mean'].tolist(),
                'std': self.normalization_stats['std'].tolist()
            }
        }
        state_path = checkpoint_dir / "trainer_state.json"
        temp_path = checkpoint_dir / "trainer_state.tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(trainer_state, f, indent=4, cls=NumpyEncoder)
            # Atomic replace
            temp_path.replace(state_path)
        except Exception as e:
            print(f"⚠ Failed to save trainer state: {e}")

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

                # Resize buffer if config has changed since checkpoint was saved
                # The deque's maxlen is baked into the saved object, so we need to
                # explicitly create a new deque with the updated capacity
                if self.replay_buffer.capacity != Config.BUFFER_SIZE:
                    print(f"⚠️ Resizing buffer: {self.replay_buffer.capacity:,} -> {Config.BUFFER_SIZE:,}")
                    from collections import deque

                    # Create new deque with NEW capacity and copy old data
                    new_deque = deque(self.replay_buffer.buffer, maxlen=Config.BUFFER_SIZE)

                    # Update buffer object
                    self.replay_buffer.buffer = new_deque
                    self.replay_buffer.capacity = Config.BUFFER_SIZE
                    print(f"✓ Buffer resized successfully ({len(self.replay_buffer):,} transitions preserved)")

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
                checkpoint_gen = trainer_state.get('generation', 0)
                # Resume from next generation (0-indexed: if saved gen=27, resume at gen=28)
                self.start_generation = checkpoint_gen + 1
                self.best_fitness = trainer_state.get('best_fitness', float('-inf'))
                self.best_validation_fitness = trainer_state.get('best_validation_fitness', float('-inf'))

                # Load adaptive mutation state
                self.validation_fitness_history = trainer_state.get('validation_fitness_history', [])
                # Use consistency-aware fallback for mutation rate
                default_mutation_rate = Config.MUTATION_RATE_CONSISTENCY if self.consistency_mode else Config.MUTATION_RATE
                self.current_mutation_rate = trainer_state.get('current_mutation_rate', default_mutation_rate)
                self.current_mutation_std = trainer_state.get('current_mutation_std', Config.MUTATION_STD)
                self.plateau_detected = trainer_state.get('plateau_detected', False)

                # Load leverage mode state
                self.leverage_mode_active = trainer_state.get('leverage_mode_active', False)
                self.leverage_generations_remaining = trainer_state.get('leverage_generations_remaining', 0)

                # Restore Multi-Agent Mode State
                multi_state = trainer_state.get('multi_state')
                if self.multi_mode and multi_state:
                    self.current_member_idx = multi_state.get('current_member_idx', 0)
                    self.turnovers_completed = multi_state.get('turnovers_completed', 0)
                    self.member_breakthroughs = multi_state.get('member_breakthroughs', [0]*self.num_committee_members)
                    self.member_training_start_gen = multi_state.get('member_training_start_gen', 0)

                    # Restore baselines if available (critical for breakthrough calculation)
                    saved_baselines = multi_state.get('member_baselines')
                    if saved_baselines:
                        self.member_baselines = saved_baselines

                    saved_starting_rois = multi_state.get('member_starting_rois')
                    if saved_starting_rois:
                        self.member_starting_rois = saved_starting_rois

                    # Restore stuck detection state
                    self.multi_gens_since_improvement = multi_state.get('multi_gens_since_improvement', 0)
                    self.multi_best_score_for_member = multi_state.get('multi_best_score_for_member', float('-inf'))
                    
                    # Restore phase tracking state
                    self.multi_phase = multi_state.get('multi_phase', 'non_maverick')
                    self.non_maverick_members = multi_state.get('non_maverick_members', [])
                    self.maverick_members = multi_state.get('maverick_members', [])
                    self.current_non_maverick_idx = multi_state.get('current_non_maverick_idx', 0)
                    self.current_maverick_idx = multi_state.get('current_maverick_idx', 0)
                    self.non_maverick_turnovers_per_agent = multi_state.get('non_maverick_turnovers_per_agent', [])
                    self.maverick_turnovers_per_agent = multi_state.get('maverick_turnovers_per_agent', [])
                    
                    # Ensure maverick_mode is set correctly based on phase
                    if self.multi_phase == 'maverick':
                        self.maverick_mode = True
                    else:
                        self.maverick_mode = False

                    # Reload the parent agent for the current member (needed for stuck recovery)
                    # The population was restored from checkpoint, but multi_parent_agent is separate
                    from committee import get_agent_filepath
                    member = self.multi_roster['members'][self.current_member_idx]
                    context_window = self.multi_roster['context_window_days']
                    agent_path = get_agent_filepath(member, context_window)
                    if agent_path.exists():
                        self.multi_parent_agent = DDPGAgent(agent_id=-1)
                        self.multi_parent_agent.load(str(agent_path))

                    print(f"✓ Multi-Mode state restored:")
                    print(f"  Current Member: {self.current_member_idx}")
                    print(f"  Turnovers: {self.turnovers_completed}")
                    print(f"  Breakthroughs: {self.member_breakthroughs}")
                    print(f"  Gens since improvement: {self.multi_gens_since_improvement}")

                # Load ROI hurdle EMA (defaults to None for old checkpoints)
                self.roi_hurdle_ema = trainer_state.get('roi_hurdle_ema', None)

                # Load Gauntlet Mode state (backwards compatible with old checkpoints)
                if self.gauntlet_mode_enabled:
                    breakthrough_state_str = trainer_state.get('breakthrough_state', None)
                    if breakthrough_state_str:
                        self.breakthrough_state = BreakthroughState(breakthrough_state_str)
                    self.confirmed_baseline = trainer_state.get('confirmed_baseline', 0.0)
                    self.confirmed_breakthroughs = trainer_state.get('confirmed_breakthroughs', 0)
                    self.breakthrough_history = trainer_state.get('breakthrough_history', [])
                    self.stabilization_generations_elapsed = trainer_state.get('stabilization_generations_elapsed', 0)

                    # Load breakthrough candidate (may be None)
                    # NOTE: We can only restore metadata, not the actual agent objects
                    # The agent will need to be re-selected from the population after loading
                    candidate_data = trainer_state.get('breakthrough_candidate', None)
                    if candidate_data is not None:
                        agent_idx = candidate_data.get('agent_idx', 0)
                        # Reconstruct BreakthroughCandidate with agent from current population
                        self.breakthrough_candidate = BreakthroughCandidate(
                            agent=self.population[agent_idx].clone(),
                            agents=[self.population[agent_idx].clone()],
                            agent_idx=agent_idx,
                            agent_indices=[agent_idx],
                            spike_score=candidate_data.get('spike_score', 0.0),
                            spike_scores=[candidate_data.get('spike_score', 0.0)],
                            detection_generation=candidate_data.get('detection_generation', 0),
                            stabilization_start_gen=candidate_data.get('stabilization_start_gen'),
                            gauntlet_score=candidate_data.get('gauntlet_score')
                        )
                        print(f"  Breakthrough Candidate: Agent {agent_idx} "
                              f"(spike score: {self.breakthrough_candidate.spike_score:.2f})")
                    else:
                        self.breakthrough_candidate = None

                    # Load heroes queue state (for consistency mode with heroes)
                    self.use_candidate_queue = trainer_state.get('use_candidate_queue', False)
                    self.candidate_queue = trainer_state.get('candidate_queue', [])
                    # Convert list back to set
                    tested_indices_list = trainer_state.get('tested_candidate_indices', [])
                    self.tested_candidate_indices = set(tested_indices_list)
                    self.pending_baseline_update = trainer_state.get('pending_baseline_update', None)

                    print(f"✓ Gauntlet Mode state restored:")
                    print(f"  State: {self.breakthrough_state.value}")
                    print(f"  Confirmed Baseline: {self.confirmed_baseline:.2f}")
                    print(f"  Breakthroughs: {self.confirmed_breakthroughs}/{self.target_breakthroughs}")
                    if self.use_candidate_queue:
                        print(f"  Candidate Queue: {len(self.candidate_queue)} pending")
                        print(f"  Tested Candidates: {len(self.tested_candidate_indices)} agents")

                # Load normalization stats (backwards compatible - recompute if not in checkpoint)
                if 'normalization_stats' in trainer_state:
                    import numpy as np
                    norm_stats = trainer_state['normalization_stats']
                    self.normalization_stats = {
                        'mean': np.array(norm_stats['mean']),
                        'std': np.array(norm_stats['std'])
                    }
                    print("✓ Loaded normalization stats from checkpoint")
                else:
                    print("! Normalization stats not in checkpoint, will use the ones computed at init")

                # Load Hall of Fame turnover tracking (for consistency mode, backwards compatible)
                self.hof_turnover_count = trainer_state.get('hof_turnover_count', 0)
                self.hof_current_median = trainer_state.get('hof_current_median', None)
                self.generation_at_last_turnover = trainer_state.get('generation_at_last_turnover', 0)

                # Reset fallback counter if --reset-limit flag was used
                if self.reset_limit and self.consistency_mode:
                    old_value = self.generation_at_last_turnover
                    self.generation_at_last_turnover = self.start_generation
                    print(f"✓ Reset fallback counter: {old_value} → {self.start_generation}")
                    print(f"  Fresh runway of {Config.MAX_GENERATIONS_GAUNTLET} generations")

                if self.consistency_mode and (self.hof_turnover_count > 0 or self.hof_current_median is not None):
                    print(f"✓ HoF Turnover tracking restored:")
                    print(f"  Turnovers: {self.hof_turnover_count}/{self.target_hof_turnovers}")
                    if self.hof_current_median is not None:
                        print(f"  Current Median ROI: {self.hof_current_median:.2f}%")
                    print(f"  Generation at last turnover: {self.generation_at_last_turnover}")

                print(f"✓ Resuming from generation {self.start_generation} (0-indexed)")
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
        if population_loaded and not Config.SKIP_REEVALUATION_ON_RESUME:
            print("\n" + "="*60)
            print("Re-evaluating loaded agents with current reward function")
            print("="*60)

            # Generate validation slices for re-evaluation
            print("Generating validation slices for re-evaluation...")
            self.current_generation_val_slices = self.generate_validation_slices()
            self.val_slice_hash = self._hash_validation_slices(self.current_generation_val_slices)

            # Re-evaluate population on training data
            print("\nRe-evaluating population on training data...")
            if self.local_mode:
                fitness_scores, _ = self.local_evaluator.evaluate_population()
            else:
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
        elif population_loaded and Config.SKIP_REEVALUATION_ON_RESUME:
            print("\n⚡ Skipping re-evaluation on resume (SKIP_REEVALUATION_ON_RESUME=True)")
            print("   Set to False in config.py if reward function changed")

    def save_gauntlet_snapshot(self):
        """
        Save a snapshot of the trainer state before entering the Gauntlet.
        Currently used for debugging and safety backup purposes.

        NOTE: Snapshots are no longer used for automatic restoration on Gauntlet failure.
        The soft penalty approach allows the population to keep evolutionary progress.

        The snapshot includes:
        - Population (all agents)
        - Hall of Fame
        - Trainer state (generation, fitness, mutation rates, etc.)

        Note: Replay buffer is NOT saved in snapshot to save time/space.
        """
        try:
            snapshot_dir = self.checkpoint_dir / "gauntlet_snapshot"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📸 Saving Gauntlet Snapshot (Gen {self.generation})...")

            # 1. Save population
            pop_dir = snapshot_dir / "population"
            pop_dir.mkdir(exist_ok=True)
            for agent in self.population:
                agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                agent.save(str(agent_path))

            # 2. Save trainer state
            trainer_state = {
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'best_validation_fitness': self.best_validation_fitness,
                'validation_fitness_history': self.validation_fitness_history,
                'current_mutation_rate': self.current_mutation_rate,
                'current_mutation_std': self.current_mutation_std,
                'plateau_detected': self.plateau_detected,
                'leverage_mode_active': self.leverage_mode_active,
                'leverage_generations_remaining': self.leverage_generations_remaining,
                'roi_hurdle_ema': self.roi_hurdle_ema,
                # Gauntlet Mode state (save pre-detection state)
                'gauntlet_mode_enabled': self.gauntlet_mode_enabled,
                'breakthrough_state': BreakthroughState.NORMAL.value,  # Always restore to NORMAL
                'confirmed_baseline': self.confirmed_baseline,
                'confirmed_breakthroughs': self.confirmed_breakthroughs,
                'breakthrough_history': self.breakthrough_history,
                'stabilization_generations_elapsed': 0,  # Reset stabilization counter
                # Queue-based candidate tracking (save for debugging)
                'candidate_queue': self.candidate_queue.copy(),  # Preserve queue state
                'tested_candidate_indices': list(self.tested_candidate_indices),  # Convert set to list for JSON
                'pending_baseline_update': self.pending_baseline_update,
                # Hall of Fame turnover tracking (for consistency mode)
                'hof_turnover_count': self.hof_turnover_count,
                'hof_current_median': self.hof_current_median,
                'generation_at_last_turnover': self.generation_at_last_turnover,
            }
            state_path = snapshot_dir / "trainer_state.json"
            with open(state_path, 'w') as f:
                json.dump(trainer_state, f, indent=4, cls=NumpyEncoder)

            # 3. Save Hall of Fame (copy the current HoF directory)
            if self.hall_of_fame is not None and len(self.hall_of_fame) > 0:
                # Save HoF metadata
                hof_snapshot_path = snapshot_dir / "hall_of_fame.json"
                import shutil
                hof_source = self.hall_of_fame.hof_dir / "hall_of_fame.json"
                if hof_source.exists():
                    shutil.copy(hof_source, hof_snapshot_path)

                # Copy HoF agent files
                hof_agents_dir = snapshot_dir / "hof_agents"
                hof_agents_dir.mkdir(exist_ok=True)
                for entry in self.hall_of_fame.entries:
                    agent_path = self.hall_of_fame.hof_dir / f"hof_agent_{entry.agent_id}.pth"
                    if agent_path.exists():
                        dest_path = hof_agents_dir / f"hof_agent_{entry.agent_id}.pth"
                        shutil.copy(agent_path, dest_path)

            print(f"✓ Snapshot saved to {snapshot_dir}")

        except Exception as e:
            print(f"⚠ FAILED TO SAVE GAUNTLET SNAPSHOT: {e}")
            print(f"  Continuing training anyway...")

    def restore_gauntlet_snapshot(self):
        """
        Restore the trainer state from the Gauntlet snapshot.

        NOTE: This function is retained for manual debugging but is NO LONGER
        automatically called when a Gauntlet fails. The soft penalty approach
        allows the population to keep their evolutionary progress instead of
        resetting to the past.

        Restores:
        - Population (all agents)
        - Hall of Fame
        - Trainer state (generation, fitness, mutation rates, etc.)
        """
        snapshot_dir = self.checkpoint_dir / "gauntlet_snapshot"

        if not snapshot_dir.exists():
            print("⚠ No Gauntlet snapshot found - cannot restore")
            return

        print(f"\n🔄 RESTORING Gauntlet Snapshot (Manual Restore)...")
        print(f"{'='*60}")
        print(f"  NOTE: This is a manual/debug restore operation")
        print(f"  Reverting to pre-Gauntlet state")
        print(f"{'='*60}")

        # 1. Restore Population
        pop_dir = snapshot_dir / "population"
        if pop_dir.exists():
            try:
                for i, agent in enumerate(self.population):
                    agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                    if agent_path.exists():
                        agent.load(str(agent_path))
                print(f"✓ Restored {len(self.population)} agents to pre-Gauntlet state")
            except Exception as e:
                print(f"❌ Error restoring population: {e}")

        # 2. Restore Trainer State
        state_path = snapshot_dir / "trainer_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    trainer_state = json.load(f)

                # Restore all relevant state
                self.best_fitness = trainer_state.get('best_fitness', float('-inf'))
                self.best_validation_fitness = trainer_state.get('best_validation_fitness', float('-inf'))
                self.validation_fitness_history = trainer_state.get('validation_fitness_history', [])
                self.current_mutation_rate = trainer_state.get('current_mutation_rate', Config.MUTATION_RATE)
                self.current_mutation_std = trainer_state.get('current_mutation_std', Config.MUTATION_STD)
                self.plateau_detected = trainer_state.get('plateau_detected', False)
                self.leverage_mode_active = trainer_state.get('leverage_mode_active', False)
                self.leverage_generations_remaining = trainer_state.get('leverage_generations_remaining', 0)
                self.roi_hurdle_ema = trainer_state.get('roi_hurdle_ema', None)

                # Gauntlet state is always restored to NORMAL
                self.breakthrough_state = BreakthroughState.NORMAL
                self.breakthrough_candidate = None
                self.stabilization_generations_elapsed = 0

                # Restore queue-based candidate tracking (critical for Ghost Loop prevention)
                self.candidate_queue = trainer_state.get('candidate_queue', [])
                self.tested_candidate_indices = set(trainer_state.get('tested_candidate_indices', []))
                self.pending_baseline_update = trainer_state.get('pending_baseline_update', None)

                # Restore Hall of Fame turnover tracking (for consistency mode)
                self.hof_turnover_count = trainer_state.get('hof_turnover_count', 0)
                self.hof_current_median = trainer_state.get('hof_current_median', None)
                self.generation_at_last_turnover = trainer_state.get('generation_at_last_turnover', 0)

                print(f"✓ Restored trainer state")
                if self.candidate_queue:
                    print(f"  ✓ Restored candidate queue ({len(self.candidate_queue)} candidates)")
                    print(f"  ✓ Restored tested set ({len(self.tested_candidate_indices)} agents already tested)")
                if self.hof_turnover_count > 0:
                    print(f"  ✓ Restored HoF turnover tracking ({self.hof_turnover_count} turnovers)")
            except Exception as e:
                print(f"❌ Error restoring trainer state: {e}")

        # 3. Restore Hall of Fame
        hof_snapshot_path = snapshot_dir / "hall_of_fame.json"
        hof_agents_dir = snapshot_dir / "hof_agents"
        if hof_snapshot_path.exists() and hof_agents_dir.exists():
            try:
                import shutil
                # Restore HoF metadata
                hof_dest = self.hall_of_fame.hof_dir / "hall_of_fame.json"
                shutil.copy(hof_snapshot_path, hof_dest)

                # Restore HoF agent files
                for agent_file in hof_agents_dir.glob("hof_agent_*.pth"):
                    dest_path = self.hall_of_fame.hof_dir / agent_file.name
                    shutil.copy(agent_file, dest_path)

                # Reload Hall of Fame from restored files
                self.hall_of_fame.load()
                print(f"✓ Restored Hall of Fame ({len(self.hall_of_fame.entries)} champions)")
            except Exception as e:
                print(f"❌ Error restoring Hall of Fame: {e}")

        print(f"✓ Manual Restore Complete - Population restored to snapshot state")
        print(f"{'='*60}\n")

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

        # Print local mode notice if enabled
        if self.local_mode:
            print("\n" + "-"*60)
            print("LOCAL MODE ENABLED")
            print("-"*60)
            print("  - Sequential evaluation (no process spawning)")
            print("  - Sequential validation (no worker pool)")
            print("  - Batch disk writes (no I/O thrashing)")
            print("  - Optimized for local machines (Windows/WSL)")
            print("-"*60 + "\n")

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

                # Log leverage mode activation to wandb
                wandb.log({
                    'leverage/active': 1,
                    'leverage/generations_remaining': self.leverage_generations_remaining,
                    'leverage/multiplier': 1.5,
                }, step=self.start_generation)
            else:
                print(f"\n⚠ Leverage mode requested but Hall of Fame only has {hof_size}/5 agents")
                print("  Leverage mode will not be activated. Continue training normally.\n")

        # Use start_generation for the loop
        # Stopping condition: Gauntlet Mode uses breakthroughs, fallback to generation limit
        # In consistency mode, fallback resets with each turnover (checked inside loop)
        # Maverick mode: no generation limit - train until target rank achieved
        if self.consistency_mode and self.gauntlet_mode_enabled:
            # Use a high ceiling - actual stopping is controlled by turnover-based fallback inside loop
            max_generations = 10000
        elif self.maverick_mode:
            # Maverick mode: no timeout - train until target rank (default: 20) is achieved in Global 50
            max_generations = 10000
        else:
            max_generations = Config.MAX_GENERATIONS_GAUNTLET if self.gauntlet_mode_enabled else Config.NUM_GENERATIONS

        for gen in range(self.start_generation, max_generations):
            self.generation = gen  # Keep this to track the *current* gen (0-indexed)
            gen_start_time = time.time()

            # Check stopping conditions
            # Single-agent mode: Stop after 4 breakthroughs or fallback timeout
            # Consistency mode: Stop after target HoF turnovers achieved
            # Normal mode: Stop after target breakthroughs achieved

            # Single-agent mode success: 4 breakthroughs achieved
            if self.single_agent_mode and self.confirmed_breakthroughs >= Config.SINGLE_TARGET_BREAKTHROUGHS:
                print(f"\n{'='*60}")
                print(f"🎯 SINGLE AGENT MODE SUCCESS - {Config.SINGLE_TARGET_BREAKTHROUGHS} BREAKTHROUGHS ACHIEVED!")
                print(f"{'='*60}")
                print(f"  Original baseline: {self.initial_single_baseline:.2f}")
                print(f"  Final baseline: {self.confirmed_baseline:.2f}")
                if self.initial_single_baseline != 0:
                    total_improvement = (self.confirmed_baseline - self.initial_single_baseline) / abs(self.initial_single_baseline)
                    print(f"  Total improvement: {total_improvement:.1%}")
                print(f"  Generation: {gen + 1}")
                print(f"{'='*60}")
                break

            # Single-agent mode fallback timeout
            if self.single_agent_mode and gen >= Config.MAX_GENERATIONS_GAUNTLET:
                print(f"\n{'='*60}")
                print(f"⏱️ SINGLE AGENT MODE TIMEOUT - {Config.MAX_GENERATIONS_GAUNTLET} GENERATIONS")
                print(f"{'='*60}")
                print(f"  Breakthroughs achieved: {self.confirmed_breakthroughs}/{Config.SINGLE_TARGET_BREAKTHROUGHS}")
                print(f"  Original baseline: {self.initial_single_baseline:.2f}")
                print(f"  Current baseline: {self.confirmed_baseline:.2f}")
                if self.initial_single_baseline != 0:
                    total_improvement = (self.confirmed_baseline - self.initial_single_baseline) / abs(self.initial_single_baseline)
                    print(f"  Total improvement: {total_improvement:.1%}")
                print(f"  Generation: {gen + 1}")
                print(f"{'='*60}")
                break

            # Multi-agent mode success: target turnovers achieved
            if self.multi_mode and self.turnovers_completed >= Config.MULTI_TARGET_TURNOVERS:
                print(f"\n{'='*60}")
                print(f"🎯 MULTI-AGENT MODE SUCCESS - {Config.MULTI_TARGET_TURNOVERS} TURNOVERS ACHIEVED!")
                print(f"{'='*60}")
                print(f"  All {self.num_committee_members} committee members improved!")
                for member_idx in range(self.num_committee_members):
                    member = self.multi_roster['members'][member_idx]
                    bt = self.member_breakthroughs[member_idx]
                    print(f"    Member {member_idx} ({member['run_name']}_{member['agent_id']}): {bt} breakthroughs")
                print(f"  Generation: {gen + 1}")
                print(f"{'='*60}")
                break

            # Multi-agent mode: NO fallback timeout - train until target turnovers achieved
            # (fallback timeout disabled for --multi mode)

            # Maverick mode success: Target rank achieved in Global 50
            if self.maverick_mode and self._maverick_goal_achieved:
                print(f"\n{'='*60}")
                print(f"🎯 MAVERICK GOAL ACHIEVED!")
                print(f"{'='*60}")
                print(f"  Maverick Agent promoted to Rank #{self._maverick_final_rank} (Target: <= #{Config.MAVERICK_TARGET_RANK})")
                print(f"  Agent type: Maverick (aggressive reward functions)")
                print(f"  Confirmed baseline: {self.confirmed_baseline:.2f}")
                print(f"  Breakthroughs: {self.confirmed_breakthroughs}")
                print(f"  Generation: {gen + 1}")
                print(f"{'='*60}")
                break

            if self.consistency_mode and self.hof_turnover_count >= self.target_hof_turnovers:
                print(f"\n{'='*60}")
                print(f"🎯 CONSISTENCY MODE SUCCESS - {self.target_hof_turnovers} HoF TURNOVERS COMPLETE!")
                print(f"{'='*60}")
                print(f"  Total turnovers: {self.hof_turnover_count}")
                print(f"  Final median ROI: {self.hof_current_median:.2f}%")
                print(f"  Hall of Fame is now an Archive of the Proven")
                print(f"  Generation: {gen + 1}")
                print(f"{'='*60}")
                break

            # Consistency mode fallback: reset counter with each turnover
            # This gives more runway after each successful turnover instead of a hard global limit
            # Skip this fallback in multi_mode (multi_mode has no fallback - trains until target turnovers)
            if self.consistency_mode and self.gauntlet_mode_enabled and not self.multi_mode:
                generations_since_turnover = gen - self.generation_at_last_turnover
                if generations_since_turnover >= Config.MAX_GENERATIONS_GAUNTLET:
                    print(f"\n{'='*60}")
                    print(f"⏱️ CONSISTENCY MODE FALLBACK - NO TURNOVER IN {Config.MAX_GENERATIONS_GAUNTLET} GENERATIONS")
                    print(f"{'='*60}")
                    print(f"  Turnovers achieved: {self.hof_turnover_count}/{self.target_hof_turnovers}")
                    print(f"  Generations since last turnover: {generations_since_turnover}")
                    print(f"  Current median ROI: {self.hof_current_median:.2f}%" if self.hof_current_median else "  No median established yet")
                    print(f"  Generation: {gen + 1}")
                    print(f"{'='*60}")
                    break
            elif self.gauntlet_mode_enabled and not self.consistency_mode and not self.maverick_mode and self.confirmed_breakthroughs >= self.target_breakthroughs:
                print(f"\n{'='*60}")
                print(f"🎯 TARGET BREAKTHROUGHS ACHIEVED!")
                print(f"{'='*60}")
                print(f"  Confirmed Breakthroughs: {self.confirmed_breakthroughs}/{self.target_breakthroughs}")
                print(f"  Final Baseline: {self.confirmed_baseline:.2f}")
                print(f"  Generation: {gen + 1}")
                print(f"{'='*60}")
                break

            # Reset peak memory stats for this generation
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Print generation header with Gauntlet state
            print(f"\n{'='*60}")
            if self.gauntlet_mode_enabled:
                if self.consistency_mode:
                    generations_since_turnover = gen - self.generation_at_last_turnover
                    runway_remaining = Config.MAX_GENERATIONS_GAUNTLET - generations_since_turnover
                    turnover_status = f"Turnover: {self.hof_turnover_count}/{self.target_hof_turnovers}"
                    if self.hof_current_median is not None:
                        turnover_status += f" | Median: {self.hof_current_median:.2f}%"
                    turnover_status += f" | Runway: {runway_remaining}"
                    print(f"Generation {gen + 1} | {turnover_status} | State: {self.breakthrough_state.value}")
                else:
                    print(f"Generation {gen + 1} / {max_generations} | Breakthroughs: {self.confirmed_breakthroughs}/{self.target_breakthroughs} | State: {self.breakthrough_state.value}")
            else:
                print(f"Generation {gen + 1} / {max_generations}")
            print(f"Buffer: {len(self.replay_buffer)} / {self.replay_buffer.capacity} ({len(self.replay_buffer)/self.replay_buffer.capacity*100:.1f}%)")
            if self.gauntlet_mode_enabled:
                print(f"Confirmed Baseline: {self.confirmed_baseline:.2f}")
            print(f"{'='*60}")

            # 1. Evaluate population (collect experiences)
            # Use LocalEvaluator in local mode for CPU-optimized execution with in-memory transitions
            if self.local_mode:
                fitness_scores, pop_stats = self.local_evaluator.evaluate_population()
            else:
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
            fitness_log = {
                "fitness/best_fitness": max_fitness,
                "fitness/mean_fitness": mean_fitness,
                "fitness/min_fitness": min_fitness,
                "fitness/std_fitness": np.std(fitness_scores),
                "fitness/best_ever": self.best_fitness,
            }

            # Add gauntlet metrics
            if self.gauntlet_mode_enabled:
                fitness_log.update({
                    "gauntlet/confirmed_baseline": self.confirmed_baseline,
                    "gauntlet/confirmed_breakthroughs": self.confirmed_breakthroughs,
                    "gauntlet/stabilization_phase": self.stabilization_generations_elapsed if self.breakthrough_state == BreakthroughState.STABILIZATION else 0,
                })

            wandb.log(fitness_log, step=gen)

            # Generate validation slices for this generation
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

            # Update instance variable for use in fitness calculation
            self.quality_threshold = quality_threshold

            # Validate entire population
            # Use LocalEvaluator in local mode for CPU-optimized validation
            if self.local_mode:
                all_val_results = self.local_evaluator.validate_population(quality_threshold=quality_threshold)
            else:
                all_val_results = self.validate_population_parallel(quality_threshold=quality_threshold)

            # Process validation results and calculate combined fitness
            for idx, val_results in enumerate(all_val_results):
                val_fitness = val_results['fitness']
                val_fitness_mean = val_results.get('fitness_mean', 0.0)
                val_fitness_min = val_results.get('fitness_min', 0.0)
                train_fitness = fitness_scores[idx]
                agent_roi = val_results.get('roi', 0.0)

                # Get quality count and total trades from validation results
                total_trades = val_results.get('total_trades', 0)
                quality_count = val_results.get('quality_count', 0)

                if self.consistency_mode and not self.multi_mode:
                    # Consistency mode (non-multi): Use the Pessimistic Score from validate_agent
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
                    # Standard mode OR Multi-mode: ROI Expansion with HoF median benchmark
                    # Combined score: penalize agents with negative training fitness
                    # This prevents "lucky" agents that do well on validation but poorly on training
                    # Formula: combined = val_fitness + min(0, train_fitness)
                    base_combined_fitness = val_fitness + min(0.0, train_fitness)

                    # ROI-based scoring adjustment using Hall of Fame median as benchmark
                    # Formula: Score = Fitness + (|Fitness| × multiplier × (AgentROI − MedianROI) / 100)
                    # This rewards agents that outperform the HoF median ROI and penalizes those below
                    # MULTI-MODE: In multi-mode, median_hof_roi = member's starting ROI (self-referential baseline)
                    #             Each member competes against their own starting performance, not the global median

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
                    'num_trades': val_results['num_trades'],  # Mean trades per slice
                    'total_trades': total_trades,  # Total across all slices (for quality ratio)
                    'quality_count': quality_count,
                    'raw_pnl': val_results.get('raw_pnl', 0.0),
                    'roi': agent_roi,
                    'expectancy': val_results.get('expectancy', 0.0),
                    'fitness_all_slices': val_results.get('fitness_all_slices', [])  # Per-slice scores for logging
                })

                # Track best combined fitness in this generation
                if combined_fitness > best_val_fitness_this_gen:
                    best_val_fitness_this_gen = combined_fitness
                    best_val_agent_idx = idx

            # Extract combined scores for elite selection (sorted by agent index)
            validation_scores = [result['combined_fitness'] for result in sorted(validation_results, key=lambda x: x['idx'])]

            # MULTI-MODE: Also extract base_combined_fitness for baseline comparisons
            # This is critical because ROI adjustment varies with hurdle, making combined_fitness
            # incomparable across different hurdle values. Base fitness is consistent.
            if self.multi_mode:
                base_validation_scores = [result['base_combined_fitness'] for result in sorted(validation_results, key=lambda x: x['idx'])]

            # Print summary showing training vs validation rankings
            print(f"\n--- Validation Summary ---")
            if self.consistency_mode and not self.multi_mode:
                print("Consistency mode: WR^2 × QR × ROI × volume_scalar fitness function")
            else:
                print(f"ROI Hurdle EMA: {median_hof_roi:.2f}% (raw HoF median: {raw_median_hof_roi:.2f}%)")
                print(f"Quality threshold: {quality_threshold:.2f}% (min gain_pct for quality trades, need {Config.ROI_CONFIDENCE_MIN_TRADES} for full bonus)")
                if self.multi_mode:
                    print("🎯 MULTI-MODE: ROI Expansion enabled - agents must beat parent's ROI for bonus")
            validation_results.sort(key=lambda x: x['combined_fitness'], reverse=True)

            if self.consistency_mode and not self.multi_mode:
                # Simplified output for consistency mode (no ROI adjustment)
                print("Top 5 by Combined Fitness - used for elite selection:")
                for i, result in enumerate(validation_results[:5]):
                    quality_ratio = result['quality_count'] / result['total_trades'] if result['total_trades'] > 0 else 0.0
                    print(f"  {i+1}. Agent {result['idx']:2d}: Combined={result['combined_fitness']:>8.2f}, Val=[mean:{result['validation_fitness_mean']:>6.2f}, min:{result['validation_fitness_min']:>6.2f}], ROI={result['roi']:>6.2f}%, QR={quality_ratio:.1%}, PnL=${result['raw_pnl']:>8.2f}, WR={result['win_rate']:.1%}")
            else:
                # Detailed output for standard mode and multi-mode (with ROI adjustment)
                mode_label = "MULTI-MODE ROI Expansion" if self.multi_mode else "with ROI adjustment"
                print(f"Top 5 by Combined Fitness ({mode_label}) - used for elite selection:")
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

            # --- Gauntlet Mode Breakthrough Detection ---
            if self.gauntlet_mode_enabled and not self.multi_mode:
                # Check for breakthrough (only in NORMAL state)
                if self.breakthrough_state == BreakthroughState.NORMAL:
                    self.check_for_breakthrough(validation_results)

                # Process state machine transitions
                self.process_gauntlet_state_machine(validation_results)

            # --- Multi-Agent Mode Breakthrough Detection (Sequential) ---
            if self.multi_mode:
                # Enforce warmup period before allowing breakthroughs
                # Each member gets fresh warmup period starting when they were loaded
                generations_trained = self.generation - self.member_training_start_gen
                warmup_generations = Config.BREAKTHROUGH_WARMUP_GENERATIONS  # Default: 3

                if generations_trained < warmup_generations:
                    breakthrough_result = None
                    if generations_trained == warmup_generations - 1:
                        print(f"  ⏳ Member {self.current_member_idx} warmup completes next generation")
                else:
                    # Check for breakthrough on current committee member
                    # CRITICAL: Use base_validation_scores (no ROI adjustment) for fair comparison
                    # Baseline was set using base_combined_fitness, so we compare apples to apples
                    breakthrough_result = self._check_multi_breakthrough(base_validation_scores)

                if breakthrough_result is not None:
                    improved_agent, base_score = breakthrough_result
                    # Get full validation result for this agent to extract metrics
                    agent_idx = self.population.index(improved_agent)
                    agent_val_result = [r for r in validation_results if r['idx'] == agent_idx][0]
                    # Process the breakthrough and advance to next member
                    # Pass base_score (base_combined_fitness) as the score
                    self._process_multi_breakthrough(improved_agent, base_score, agent_val_result)
                    self._advance_to_next_multi_member()

                    # CRITICAL: Skip evolution for this generation
                    # The population was just replaced with fresh clones of the new member.
                    # The fitness_scores computed above belong to the OLD population.
                    # Calling evolve_population() would apply stale fitness to wrong agents.
                    # Instead, continue to next iteration for fresh evaluation.
                    print(f"  [Skipping evolution - new member loaded, will evaluate fresh next gen]")
                    continue

                # --- Multi-Mode Improvement & Stuck Detection (post-warmup only) ---
                # Check for ANY improvement over baseline (save immediately to Global50/committee)
                # Also track stuck generations for local optima detection
                if generations_trained >= warmup_generations:
                    # CRITICAL: Use base_validation_scores for baseline comparison
                    # This ensures consistency since baseline is stored as base_combined_fitness
                    best_idx = np.argmax(base_validation_scores)
                    best_base_score_this_gen = base_validation_scores[best_idx]
                    current_baseline = self.member_baselines[self.current_member_idx]

                    # Check for improvement over baseline (any amount, not just 5%)
                    if best_base_score_this_gen > current_baseline:
                        improved_agent = self.population[best_idx]
                        agent_val_result = [r for r in validation_results if r['idx'] == best_idx][0]

                        # Save improvement immediately (updates Global50, committee, baseline)
                        # Note: This is NOT a breakthrough (no advance to next member)
                        # Pass base_score (base_combined_fitness) as the score
                        self._process_multi_improvement(improved_agent, best_base_score_this_gen, agent_val_result, is_breakthrough=False)

                        # _process_multi_improvement already resets stuck counter and updates best score
                    else:
                        # No improvement over baseline - check if we at least improved over previous best
                        # Use base_score for consistency with baseline comparison
                        if best_base_score_this_gen > self.multi_best_score_for_member:
                            self.multi_best_score_for_member = best_base_score_this_gen
                            self.multi_gens_since_improvement = 0
                        else:
                            self.multi_gens_since_improvement += 1

                    # Local optima detection: 20 generations without improvement = move on
                    if self.multi_gens_since_improvement >= 20:
                        print(f"\n{'='*60}")
                        print(f"🔄 LOCAL OPTIMA DETECTED - Member {self.current_member_idx} stuck for 20 generations")
                        print(f"{'='*60}")
                        print(f"  Best score achieved: {self.multi_best_score_for_member:.2f}")
                        print(f"  Baseline required: {self.member_baselines[self.current_member_idx]:.2f}")
                        print(f"  Treating as completed - advancing to next member")
                        print(f"{'='*60}")

                        # Advance without recording a breakthrough (no baseline update)
                        self._advance_to_next_multi_member()
                        print(f"  [Skipping evolution - new member loaded after local optima]")
                        continue

                # Log multi-mode progress
                member = self.multi_roster['members'][self.current_member_idx]
                member_name = f"{member['run_name']}_{member['agent_id']}"
                if gen % 5 == 0:
                    current_baseline = self.member_baselines[self.current_member_idx]
                    current_roi_hurdle = self.member_starting_rois[self.current_member_idx]
                    print(f"\n  Multi-Mode Progress: Turnovers {self.turnovers_completed}/{Config.MULTI_TARGET_TURNOVERS}")
                    print(f"    Current member: {self.current_member_idx} ({member_name})")
                    print(f"    Baseline: {current_baseline:.2f}, ROI Hurdle: {current_roi_hurdle:.2f}%, Best: {max(validation_scores):.2f}")
                    print(f"    Breakthroughs: {self.member_breakthroughs}")

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

            # Get the best agent's per-slice fitness scores from already-validated results
            best_agent_slice_scores = best_agent_result.get('fitness_all_slices', [])

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

            # Add Gauntlet Mode metrics
            if self.gauntlet_mode_enabled and not self.multi_mode:
                validation_log.update({
                    "gauntlet/confirmed_baseline": self.confirmed_baseline,
                    "gauntlet/confirmed_breakthroughs": self.confirmed_breakthroughs,
                    "gauntlet/stabilization_phase": self.stabilization_generations_elapsed if self.breakthrough_state == BreakthroughState.STABILIZATION else 0,
                })

            # Add Multi-Mode metrics (sequential training of committee members)
            if self.multi_mode:
                # Log current member and overall progress
                validation_log.update({
                    "multi/current_member_idx": self.current_member_idx,
                    "multi/current_member_baseline": self.member_baselines[self.current_member_idx],
                    "multi/turnovers_completed": self.turnovers_completed,
                    "multi/min_breakthroughs": min(self.member_breakthroughs),
                    "multi/max_breakthroughs": max(self.member_breakthroughs),
                    "multi/total_breakthroughs": sum(self.member_breakthroughs),
                    "multi/avg_breakthroughs": np.mean(self.member_breakthroughs),
                })

            wandb.log(validation_log, step=gen)

            # --- Hall of Fame Admission Logic ---
            # In multi-mode, skip HoF updates to preserve member-specific ROI hurdles
            if not self.multi_mode:
                # Re-evaluate all existing HoF entries with current median (EMA-based erosion)
                # This ensures historical agents don't have unfair ROI advantages as median rises
                # Erosion uses EMA smoothing: α=0.33 means gradual adjustment over ~3 generations
                if len(self.hall_of_fame.entries) > 0:
                    erosion_alpha = getattr(Config, 'HOF_EROSION_ALPHA', 0.33)  # Default: α=0.33 ≈ 3 gen convergence
                    updated_count = self.hall_of_fame.recompute_all_scores(
                        current_median_roi=median_hof_roi,
                        roi_adjustment_multiplier=Config.ROI_ADJUSTMENT_MULTIPLIER,
                        min_trades_threshold=Config.ROI_CONFIDENCE_MIN_TRADES,
                        erosion_alpha=erosion_alpha,
                        consistency_mode=self.consistency_mode
                    )
                    if updated_count > 0:
                        print(f"\n🔄 Re-evaluated {updated_count}/{len(self.hall_of_fame.entries)} HoF agents with current median ({median_hof_roi:.2f}%) [EMA α={erosion_alpha:.2f}]")

                # Build candidate list from all agents in this generation
                candidates = []
                for result in validation_results:
                    agent_idx = result['idx']
                    combined_score = result['combined_fitness']
                    agent_roi = result['roi']
                    agent_expectancy = result['expectancy']
                    train_fitness = result['training_fitness']
                    quality_count = result['quality_count']
                    total_trades = result['total_trades']
                    val_fitness = result['validation_fitness']
                    base_combined_fitness = result['base_combined_fitness']
                    candidates.append((self.population[agent_idx], combined_score, agent_idx, agent_roi, agent_expectancy,
                                     train_fitness, quality_count, total_trades, val_fitness, base_combined_fitness))

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
            else:
                # Multi-mode: Skip Hall of Fame updates to preserve member-specific ROI hurdles
                if gen % 10 == 0:
                    print(f"\n  ℹ Multi-mode: Hall of Fame updates disabled (preserving member ROI hurdle: {self.roi_hurdle_ema:.2f}%)")

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
            log_data = {
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
            }


            # Add heroes queue tracking (for consistency mode with heroes)
            if self.use_candidate_queue:
                log_data.update({
                    "heroes/queue_size": len(self.candidate_queue),
                    "heroes/tested_count": len(self.tested_candidate_indices),
                })

            wandb.log(log_data, step=gen)

            # 2. Train agents using replay buffer
            # First, wait for background transfer to complete (if running in local mode)
            if self.local_mode:
                self.local_evaluator.wait_for_transfer()
                # Move agents to CPU before training (frees GPU memory)
                # Training will move agents to GPU in batches of 16
                self.local_evaluator.restore_agents_to_cpu()
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
                    # FIX 2: Calculate max fitness BEFORE injection to give champions a "safe" high score
                    # This ensures evolve_population keeps them instead of immediately culling
                    max_fitness = np.max(fitness_scores) if len(fitness_scores) > 0 else 1.0

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

                        # FIX 2: Update fitness_scores so evolve_population recognizes this as a high-value agent
                        # Without this, the evolution step sees the old low score and culls the champion
                        fitness_scores[worst_idx] = max_fitness

                        # Also update validation_scores if available (for elite selection)
                        if validation_scores is not None and len(validation_scores) > worst_idx:
                            max_val_fitness = np.max(validation_scores) if len(validation_scores) > 0 else max_fitness
                            validation_scores[worst_idx] = max_val_fitness

                        print(f"   Agent {worst_idx}: Fitness {old_fitness:.2f} → HoF Champion (Score: {max_fitness:.2f})")

            # 3. Check for plateau and adjust mutation adaptively (FIX 3: MOVED BEFORE evolve_population)
            # This must happen BEFORE evolution because:
            # - check_and_adjust_mutation identifies the worst agent by index and replaces it
            # - If we run it AFTER evolve_population, the indices are scrambled (crossover/mutation)
            # - Running it before ensures we replace the actual worst agent from THIS generation
            # Pass validation_scores to enable Global 50 injection (replaces worst agent)
            self.check_and_adjust_mutation(self.best_validation_fitness, fitness_scores=validation_scores)

            # 4. Evolve population using validation fitness for elite selection
            # Note: We pass both training fitness and validation scores
            # - training fitness: used for tournament selection and DDPG gradient updates
            # - validation scores: used for elite selection (ensures robust generalization)
            self.evolve_population(fitness_scores, validation_scores)

            # Update resource tracker after evolution
            self.resource_tracker.update()

            # 🔍 Memory tracking after evolution
            # log_memory(f"Gen {gen+1}: After evolve_population", show_objects=True)

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

            # Prepare gauntlet info for display
            gauntlet_info = None
            if self.gauntlet_mode_enabled:
                # Calculate stabilization progress if in stabilization state
                stab_progress = None
                if self.breakthrough_state == BreakthroughState.STABILIZATION:
                    stab_progress = (self.stabilization_generations_elapsed, Config.STABILIZATION_GENERATIONS)

                # Get Hall of Fame stats
                hof_stats = self.hall_of_fame.get_stats()

                # Get Global Hall of Fame stats
                global_hof_stats = self.global_hof.get_stats()

                gauntlet_info = {
                    'gauntlet_enabled': True,
                    'consistency_mode': self.consistency_mode,
                    'breakthrough_state': self.breakthrough_state.value,
                    'confirmed_baseline': self.confirmed_baseline,
                    'confirmed_breakthroughs': self.confirmed_breakthroughs,
                    'target_breakthroughs': self.target_breakthroughs,
                    'hof_turnover_count': self.hof_turnover_count,
                    'target_hof_turnovers': self.target_hof_turnovers,
                    'hof_current_median': self.hof_current_median,
                    'hof_size': hof_stats['size'],
                    'hof_capacity': self.hall_of_fame.capacity,
                    'stabilization_progress': stab_progress,
                    'breakthrough_history': self.breakthrough_history,
                    'global_hof_enabled': global_hof_stats['enabled'],
                    'global_hof_size': global_hof_stats['size'],
                    'global_hof_threshold': global_hof_stats['entry_threshold']
                }

                # Add queue size if using candidate queue (heroes mode)
                if self.use_candidate_queue:
                    gauntlet_info['queue_size'] = len(self.candidate_queue)

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
                resource_stats=resource_stats,
                gauntlet_info=gauntlet_info
            )

            # Show progress plot every 5 generations
            # if (gen + 1) % 5 == 0:
            #     plot_fitness_progress(self.fitness_history)
            #     # CRITICAL FIX: Close matplotlib figures to prevent memory leak (~100MB per plot)
            #     plt.close('all')

            # Buffer stats and generation time
            buffer_stats = self.replay_buffer.get_stats()
            self.writer.add_scalar('Buffer/Size', buffer_stats['size'], gen)
            self.writer.add_scalar('Buffer/Utilization', buffer_stats['utilization'], gen)

            # Log timing and buffer metrics to wandb
            wandb_log_data = {
                "training/generation_time": gen_time,
                "buffer/size": buffer_stats['size'],
                "buffer/utilization": buffer_stats['utilization'],
            }

            # Add Global HoF stats if enabled
            if self.global_hof.enabled:
                global_hof_stats = self.global_hof.get_stats()
                wandb_log_data.update({
                    'global_hof/size': global_hof_stats['size'],
                    'global_hof/entry_threshold': global_hof_stats['entry_threshold'],
                    'global_hof/best_score': global_hof_stats['best_score'],
                    'global_hof/mean_score': global_hof_stats['mean_score'],
                })

            wandb.log(wandb_log_data, step=gen)

            # Clear GPU cache and run garbage collection to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Leverage mode tracking - decrement counter and deactivate after 5 generations
            if self.leverage_mode_active:
                self.leverage_generations_remaining -= 1
                print(f"\n📊 Leverage mode: {self.leverage_generations_remaining} generations remaining")

                # Log leverage mode countdown to wandb
                wandb.log({
                    'leverage/countdown': self.leverage_generations_remaining,
                }, step=gen)

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

                    # Log leverage mode deactivation to wandb
                    wandb.log({
                        'leverage/active': 0,
                    }, step=gen)

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

        # Clean up shared memory (must be done before process exits)
        self._cleanup_shared_memory()

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