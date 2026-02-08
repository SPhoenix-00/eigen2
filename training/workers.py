"""
Worker process management and shared memory for Project Eigen 2.

Contains:
- Module-level worker functions for ProcessPoolExecutor (moved verbatim from erl_trainer.py)
- SharedMemoryManager class for zero-copy data sharing with worker processes
"""

import numpy as np
import torch
import sys
from collections import OrderedDict
from multiprocessing import shared_memory
if sys.platform != 'win32':
    from multiprocessing import resource_tracker

from utils.config import Config

# Maximum number of agents to cache per worker to prevent memory leaks
_WORKER_CACHE_MAX_SIZE = 50

# Global variables for worker processes
_worker_env_config = None  # Shared environment configuration
_worker_env = None  # Reusable environment instance (created once per worker)
_worker_agent_cache = None  # Cache of reconstructed agents {state_hash: agent}
_worker_shm_refs = []  # Keep references to shared memory objects to prevent cleanup


def _init_worker(env_config):
    """
    Initializer function for worker processes.

    Called once per worker when the ProcessPoolExecutor starts.
    Stores the env_config in a global variable so it doesn't need to be
    pickled with every task.

    OPTIMIZATION: Uses shared memory for large numpy arrays to avoid serialization.
    """
    global _worker_env_config, _worker_env, _worker_agent_cache, _worker_shm_refs

    # Reconstruct arrays from shared memory if using shared memory mode
    if 'shm_data_array_name' in env_config:
        shm_data = shared_memory.SharedMemory(name=env_config['shm_data_array_name'])
        shm_data_full = shared_memory.SharedMemory(name=env_config['shm_data_array_full_name'])

        _worker_shm_refs = [shm_data, shm_data_full]

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
        _worker_env_config = env_config

    from environment.trading_env import TradingEnvironment
    _worker_env = TradingEnvironment(**_worker_env_config)

    _worker_agent_cache = OrderedDict()


def _cache_agent(state_hash, agent):
    """Add an agent to the LRU cache with size limit enforcement."""
    global _worker_agent_cache

    while len(_worker_agent_cache) >= _WORKER_CACHE_MAX_SIZE:
        _worker_agent_cache.popitem(last=False)

    _worker_agent_cache[state_hash] = agent


def _get_cached_agent(state_hash):
    """Retrieve an agent from cache, updating LRU order if found."""
    global _worker_agent_cache

    if state_hash in _worker_agent_cache:
        _worker_agent_cache.move_to_end(state_hash)
        return _worker_agent_cache[state_hash]
    return None


def _run_episode_worker(args):
    """
    Worker function for parallel episode execution.

    Runs in a separate process, reconstructs agent from CPU state dicts,
    reuses worker environment, and writes transitions directly to disk.

    Args:
        args: Tuple of (agent_state, start_idx, end_idx, training, seed, buffer_storage_path, file_id_start)

    Returns:
        Tuple of (fitness, episode_info, transition_file_paths)
    """
    global _worker_env_config, _worker_env, _worker_agent_cache
    agent_state, start_idx, end_idx, training, seed, buffer_storage_path, file_id_start = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    import hashlib
    actor_bytes = str(agent_state['actor']).encode()
    state_hash = hashlib.md5(actor_bytes).hexdigest()

    agent = _get_cached_agent(state_hash)
    if agent is None:
        from models.ddpg_agent import DDPGAgent
        agent = DDPGAgent(agent_id=0)
        agent.actor.load_state_dict(agent_state['actor'])
        agent.critic.load_state_dict(agent_state['critic'])
        agent.actor.eval()
        agent.critic.eval()
        _cache_agent(state_hash, agent)

    env = _worker_env

    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
    env.set_training_mode(training)
    state, _ = env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

    cumulative_reward = 0.0
    steps = 0
    transition_file_paths = []

    if buffer_storage_path:
        from pathlib import Path
        import pickle
        import gzip

        storage_path = Path(buffer_storage_path)
        file_id = file_id_start

        while True:
            action = agent.select_action(state, add_noise=training)
            next_state, reward, terminated, truncated, info = env.step(action)

            transition = {
                'state': state.astype(np.float32),
                'action': action.astype(np.float32),
                'reward': reward,
                'next_state': next_state.astype(np.float32),
                'done': float(terminated or truncated)
            }

            file_path = storage_path / f"transition_{file_id}.pkl.gz"
            try:
                if file_path.exists():
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
        while True:
            action = agent.select_action(state, add_noise=training)
            next_state, reward, terminated, truncated, info = env.step(action)

            cumulative_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

    episode_summary = env.get_episode_summary()
    episode_summary['steps'] = steps

    final_fitness = float(cumulative_reward)
    if episode_summary['num_trades'] == 0:
        final_fitness -= episode_summary['zero_trades_penalty']

    if episode_summary['num_trades'] >= Config.WIN_RATE_BONUS_MIN_TRADES:
        win_rate_pct = episode_summary['win_rate'] * 100.0
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

    Validates a single agent across all validation slices.

    Args:
        args: Tuple of (agent_state, validation_slices, quality_threshold, seed, use_penalized_median)

    Returns:
        Dict with validation results
    """
    global _worker_env_config, _worker_env, _worker_agent_cache
    agent_state, validation_slices, quality_threshold, seed, use_penalized_median = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    import hashlib
    actor_bytes = str(agent_state['actor']).encode()
    state_hash = hashlib.md5(actor_bytes).hexdigest()

    agent = _get_cached_agent(state_hash)
    if agent is None:
        from models.ddpg_agent import DDPGAgent
        agent = DDPGAgent(agent_id=0)
        target_device = Config.DEVICE
        agent.move_to_device(target_device, recreate_optimizers=False)
        agent.actor.load_state_dict(agent_state['actor'])
        agent.critic.load_state_dict(agent_state['critic'])
        agent.actor.eval()
        agent.critic.eval()
        _cache_agent(state_hash, agent)
    else:
        agent.actor.eval()
        agent.critic.eval()
        target_device = Config.DEVICE
        if agent.device != target_device:
            agent.move_to_device(target_device, recreate_optimizers=False)

    env = _worker_env

    slice_results = []
    all_closed_trades = []

    for start_idx, end_idx, _ in validation_slices:
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
        num_trading_steps = Config.TRADING_PERIOD_DAYS

        env.set_training_mode(False)
        env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

        trading_states = env.get_batch_observations(start_idx, num_trading_steps)
        with torch.no_grad():
            chunk_size = 16
            action_chunks = []
            for i in range(0, num_trading_steps, chunk_size):
                chunk = trading_states[i:i + chunk_size]
                action_chunks.append(agent.select_actions_batch(chunk, add_noise=False))
            trading_actions = np.concatenate(action_chunks, axis=0)

        cumulative_reward = 0.0
        steps = 0

        for i in range(num_trading_steps):
            _, reward, terminated, truncated, _ = env.step(trading_actions[i])
            cumulative_reward += reward
            steps += 1
            if terminated or truncated:
                break

        if not (terminated or truncated):
            dummy_action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM), dtype=np.float32)
            while True:
                _, reward, terminated, truncated, _ = env.step(dummy_action)
                cumulative_reward += reward
                steps += 1
                if terminated or truncated:
                    break

        episode_info = env.get_episode_summary()
        episode_info['steps'] = steps

        fitness = float(cumulative_reward)
        if episode_info['num_trades'] == 0:
            fitness -= episode_info['zero_trades_penalty']

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
    mean_score = np.mean(fitness_scores)
    min_score = np.min(fitness_scores)

    if use_penalized_median:
        median_score = np.median(fitness_scores)
        std_score = np.std(fitness_scores)
        validation_fitness = float(median_score - (0.5 * std_score))
    else:
        validation_fitness = (0.4 * mean_score) + (0.6 * min_score)

    total_raw_pnl = sum([r['raw_pnl'] for r in slice_results])
    total_investment = sum([r['total_investment'] for r in slice_results])
    total_peak_capital = sum([r['peak_capital_employed'] for r in slice_results])
    roi = (total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0

    total_wins = sum([r['num_wins'] for r in slice_results])
    total_losses = sum([r['num_losses'] for r in slice_results])
    total_trades = total_wins + total_losses
    global_win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0

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
        'fitness_all_slices': fitness_scores,
        'fitness_mean': mean_score,
        'fitness_min': min_score,
        'roi': roi,
        'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
        'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
        'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
        'avg_reward_per_trade': np.mean([r['avg_reward_per_trade'] for r in slice_results]),
        'raw_pnl': total_raw_pnl,
        'total_investment': total_investment,
        'total_trades': total_trades,
        'win_rate': global_win_rate,
        'expectancy': expectancy,
        'quality_count': quality_count,
        'quality_roi': quality_roi,
        'sample_trade': sample_trade
    }


class SharedMemoryManager:
    """
    Manages shared memory blocks for zero-copy data sharing with worker processes.

    Encapsulates creation, access, and cleanup of shared memory used by
    ProcessPoolExecutor workers during parallel evaluation.
    """

    def __init__(self, data_loader, local_mode: bool = False):
        """
        Initialize shared memory from data loader arrays.

        Args:
            data_loader: StockDataLoader with data_array and data_array_full
            local_mode: If True, skip shared memory creation (not needed)
        """
        self.local_mode = local_mode
        self._shm_blocks = []
        self._shm_data_array = None
        self._shm_data_array_full = None
        self._shm_metadata = {}

        if not local_mode:
            self._create(data_loader)

    def _create(self, data_loader):
        """Create shared memory blocks from data loader arrays."""
        data_array = data_loader.data_array
        self._shm_data_array = shared_memory.SharedMemory(
            create=True, size=data_array.nbytes
        )
        shm_data_view = np.ndarray(
            data_array.shape, dtype=data_array.dtype, buffer=self._shm_data_array.buf
        )
        shm_data_view[:] = data_array[:]
        self._shm_blocks.append(self._shm_data_array)

        if sys.platform != 'win32':
            resource_tracker.register(self._shm_data_array.name, "shared_memory")

        data_array_full = data_loader.data_array_full
        self._shm_data_array_full = shared_memory.SharedMemory(
            create=True, size=data_array_full.nbytes
        )
        shm_full_view = np.ndarray(
            data_array_full.shape, dtype=data_array_full.dtype, buffer=self._shm_data_array_full.buf
        )
        shm_full_view[:] = data_array_full[:]
        self._shm_blocks.append(self._shm_data_array_full)

        if sys.platform != 'win32':
            resource_tracker.register(self._shm_data_array_full.name, "shared_memory")

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

    def cleanup(self):
        """Clean up shared memory blocks. Must be called before exit."""
        for shm in self._shm_blocks:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"⚠ Error cleaning up shared memory: {e}")
        self._shm_blocks = []
        if self._shm_metadata:
            print("✓ Shared memory cleaned up")
            self._shm_metadata = {}

    def get_env_config(self, data_loader, normalization_stats, start_idx, end_idx,
                       trading_end_idx, is_training=True, consistency_mode=False,
                       gauntlet_mode=False, maverick_mode=False) -> dict:
        """
        Build environment config dict using shared memory references or actual arrays.

        Args:
            data_loader: StockDataLoader instance
            normalization_stats: Normalization statistics dict
            start_idx: Episode start index
            end_idx: Episode end index
            trading_end_idx: Last day to open new positions
            is_training: Whether environment is in training mode
            consistency_mode: Whether consistency mode is active
            gauntlet_mode: Whether gauntlet soft penalty is active
            maverick_mode: Whether maverick mode is active

        Returns:
            Dict with shared memory references (or actual arrays in local mode)
        """
        if self.local_mode or not self._shm_metadata:
            return {
                'data_array': data_loader.data_array,
                'data_array_full': data_loader.data_array_full,
                'dates': data_loader.dates,
                'normalization_stats': normalization_stats,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'trading_end_idx': trading_end_idx,
                'is_training': is_training,
                'consistency_mode': consistency_mode,
                'gauntlet_mode': gauntlet_mode,
                'maverick_mode': maverick_mode,
            }
        else:
            return {
                **self._shm_metadata,
                'dates': data_loader.dates,
                'normalization_stats': normalization_stats,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'trading_end_idx': trading_end_idx,
                'is_training': is_training,
                'consistency_mode': consistency_mode,
                'gauntlet_mode': gauntlet_mode,
                'maverick_mode': maverick_mode,
            }

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass
