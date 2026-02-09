"""
CPU-Optimized Local Evaluator for Project Eigen 2

Clean-sheet implementation of evaluation and validation loops for --local mode.
Designed to eliminate I/O bottlenecks and minimize overhead through:

1. In-memory transition buffer (no disk I/O)
2. Batched agent inference
3. Pre-allocated numpy arrays
4. Minimal memory copying
5. Direct computation without caching overhead
6. Background disk I/O - transfer to replay buffer runs in parallel with validation

Usage:
    evaluator = LocalEvaluator(trainer)
    fitness_scores, stats = evaluator.evaluate_population()
    # Background transfer starts automatically - validation can proceed immediately
    val_results = evaluator.validate_population(quality_threshold)
    # Wait for transfer before DDPG training
    evaluator.wait_for_transfer()
"""

import numpy as np
import torch
import threading
import time
import sys
import platform
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

from utils.config import Config
from models.ddpg_agent import DDPGAgent
from training.fitness import calculate_expectancy as _calculate_expectancy


@dataclass
class TransitionBatch:
    """Pre-allocated transition storage for zero-copy collection."""
    states: np.ndarray       # [capacity, *state_shape]
    actions: np.ndarray      # [capacity, num_stocks, 2]
    rewards: np.ndarray      # [capacity]
    next_states: np.ndarray  # [capacity, *state_shape]
    dones: np.ndarray        # [capacity]
    count: int = 0

    @classmethod
    def create(cls, capacity: int, state_shape: Tuple[int, ...]) -> 'TransitionBatch':
        """Pre-allocate arrays for transition storage."""
        return cls(
            states=np.zeros((capacity, *state_shape), dtype=np.float32),
            actions=np.zeros((capacity, Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM), dtype=np.float32),
            rewards=np.zeros(capacity, dtype=np.float32),
            next_states=np.zeros((capacity, *state_shape), dtype=np.float32),
            dones=np.zeros(capacity, dtype=np.float32),
            count=0
        )

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: float):
        """Add a single transition (zero-copy into pre-allocated arrays)."""
        if self.count >= len(self.rewards):
            return  # Buffer full

        idx = self.count
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.count += 1

    def add_batch(self, states: np.ndarray, actions: np.ndarray,
                  rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> int:
        """Add a batch of transitions in one go (bulk slice copy). Much faster than per-step add."""
        n = len(rewards)
        if n == 0 or self.count + n > len(self.rewards):
            return 0
        end = self.count + n
        self.states[self.count:end] = states
        self.actions[self.count:end] = actions
        self.rewards[self.count:end] = rewards
        self.next_states[self.count:end] = next_states
        self.dones[self.count:end] = dones
        self.count = end
        return n

    def get_valid(self) -> Dict[str, np.ndarray]:
        """Return only the filled portion of the buffer."""
        n = self.count
        return {
            'states': self.states[:n],
            'actions': self.actions[:n],
            'rewards': self.rewards[:n],
            'next_states': self.next_states[:n],
            'dones': self.dones[:n]
        }

    def reset(self):
        """Reset counter without reallocating memory."""
        self.count = 0


class LocalEvaluator:
    """
    GPU-accelerated evaluation and validation for --local mode.

    Key optimizations:
    - Single environment instance reused for all episodes
    - In-memory transition buffer (no disk I/O during evaluation)
    - Batched inference on GPU for massive speedup
    - Time-travel batching: pre-fetch states, one batch inference
    - Minimal tensor creation overhead
    """

    def __init__(self, trainer):
        """
        Initialize with reference to ERLTrainer.

        Args:
            trainer: ERLTrainer instance (provides population, env, replay_buffer, etc.)
        """
        self.trainer = trainer

        # Cache frequently accessed trainer attributes
        self.population = trainer.population
        self.eval_env = trainer.eval_env
        self.replay_buffer = trainer.replay_buffer

        # Target device for inference - use GPU if available AND working
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Test if cuBLAS actually works (crashes silently on some systems)
            if self._test_cublas():
                self.target_device = torch.device('cuda:0')
                print(f"LocalEvaluator: Using GPU ({torch.cuda.get_device_name(0)})")
            else:
                print("\n" + "=" * 70)
                print("[!] CUDA cuBLAS TEST FAILED - Matrix multiplication is broken!")
                print("=" * 70)
                print("This is a known issue with some PyTorch/CUDA/driver combinations.")
                print("Possible fixes:")
                print("  1. Reinstall PyTorch with different CUDA version:")
                print("     pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")
                print("  2. Update or roll back NVIDIA driver")
                print("  3. Run with --force-cpu (very slow but functional)")
                print("=" * 70 + "\n")
                self.target_device = torch.device('cpu')
                print(f"LocalEvaluator: Falling back to CPU")
        else:
            # Check if PyTorch is CPU-only build and provide helpful error message
            torch_version = torch.__version__
            is_cpu_only = '+cpu' in torch_version or (not hasattr(torch.version, 'cuda') or torch.version.cuda is None)
            
            if is_cpu_only:
                print("\n" + "=" * 70)
                print("[!] CUDA NOT AVAILABLE - PyTorch CPU-only build detected!")
                print("=" * 70)
                print(f"Current PyTorch version: {torch_version}")
                print("\nPyTorch was installed without CUDA support (CPU-only build).")
                print("To enable GPU acceleration, reinstall PyTorch with CUDA support:")
                print("\nFor CUDA 11.8:")
                print("  pip uninstall torch torchvision torchaudio")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("\nFor CUDA 12.1:")
                print("  pip uninstall torch torchvision torchaudio")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                print("\nTo check your CUDA version, run: nvidia-smi")
                print("=" * 70 + "\n")
            
            self.target_device = torch.device('cpu')
            print(f"LocalEvaluator: Using CPU (CUDA not available)")

        # CPU optimization: Compile actor networks for faster inference (only when no GPU)
        # NOTE: Disable compilation on Windows - requires Visual Studio Build Tools (cl.exe)
        # Compilation errors occur during lazy compilation (first forward pass), not during torch.compile()
        self._compiled_actors = {}  # Cache of compiled actors by agent_id
        self._use_compiled = (self.target_device.type == 'cpu' and platform.system() != 'Windows')

        # Pre-allocate transition buffer with REDUCED capacity to avoid RAM exhaustion
        # Old: 50,000 × 151 days × 117 cols × 5 features × 4 bytes × 2 = ~35GB (causes swapping!)
        # New: 12,000 × same = ~8.5GB (fits comfortably in 32GB with headroom)
        # Buffer is flushed incrementally during evaluation when it hits 80% capacity
        state_shape = (Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        self.transition_buffer = TransitionBatch.create(12000, state_shape)

        # Background transfer thread for async disk I/O
        # Transfer runs in parallel with validation to hide disk latency
        self._transfer_thread: Optional[threading.Thread] = None
        self._transfer_start_time: float = 0.0
        self._pending_transitions: Optional[Dict[str, np.ndarray]] = None
        self._pending_count: int = 0

        # Pre-compute episode indices ranges
        self._init_episode_ranges()

    def _init_episode_ranges(self):
        """Pre-compute valid episode start/end ranges."""
        total_days_needed = (Config.CONTEXT_WINDOW_DAYS +
                            Config.TRADING_PERIOD_DAYS +
                            Config.SETTLEMENT_PERIOD_DAYS)

        self.train_start = self.trainer.train_start_idx
        self.train_end = self.trainer.train_end_idx
        self.max_start = self.train_end - total_days_needed

        if self.max_start <= self.train_start:
            raise ValueError(f"Not enough training data: need {total_days_needed} days")

    def _test_cublas(self) -> bool:
        """
        Test if cuBLAS (CUDA matrix multiplication) is working.

        Some PyTorch/CUDA/driver combinations have broken cuBLAS that causes
        silent crashes during torch.mm/matmul. This test catches that early
        so we can fall back to CPU gracefully.

        Returns:
            True if cuBLAS works, False if it crashes or errors
        """
        import torch
        import sys
        import subprocess

        # Run test in subprocess to catch segfaults
        test_code = '''
import torch
try:
    x = torch.randn(16, 16, device='cuda')
    torch.cuda.synchronize()
    y = torch.mm(x, x.T)
    torch.cuda.synchronize()
    print("OK")
except Exception as e:
    print(f"ERROR: {e}")
'''
        try:
            # Get python executable from current environment
            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, '-c', test_code],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and "OK" in result.stdout:
                return True
            else:
                print(f"  cuBLAS test failed: returncode={result.returncode}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print("  cuBLAS test timed out")
            return False
        except Exception as e:
            print(f"  cuBLAS test exception: {e}")
            return False

    def _same_device(self, d1: torch.device, d2: torch.device) -> bool:
        """Check if two devices are functionally the same (handles cuda vs cuda:0)."""
        if d1.type != d2.type:
            return False
        if d1.type == 'cpu':
            return True
        # Both CUDA - compare indices (None means 0)
        idx1 = d1.index if d1.index is not None else 0
        idx2 = d2.index if d2.index is not None else 0
        return idx1 == idx2

    def _ensure_agents_on_device(self):
        """
        Ensure all agents in population are on the target device (GPU if available).

        This fixes the device mismatch issue where agents might be on CPU after
        genetic operations or checkpoint loading, even when CUDA is available.

        NOTE: Uses move_to_device with recreate_optimizers=False since we don't
        train during inference. The optimizers will be recreated when we move
        back to CPU for training (in restore_agents_to_cpu).
        """
        moved_count = 0
        for agent in self.population:
            if not self._same_device(agent.device, self.target_device):
                # Move networks to target device
                # Don't recreate optimizers - we're only doing inference
                agent.move_to_device(self.target_device, recreate_optimizers=False)
                moved_count += 1

        if moved_count > 0:
            print(f"  [Device] Moved {moved_count} agents to {self.target_device}")

        # Clear compiled actors cache when population changes (new generation)
        self._compiled_actors = {}

    def _get_compiled_actor(self, agent: DDPGAgent):
        """
        Get a compiled version of the actor for faster CPU inference.

        torch.compile() provides 2-3x speedup on CPU by optimizing the computation graph.
        Compiled actors are cached per agent_id to avoid recompilation overhead.
        
        NOTE: On Windows, compilation requires Visual Studio Build Tools (cl.exe).
        Compilation is disabled on Windows to avoid this requirement.
        """
        if not self._use_compiled:
            return agent.actor

        agent_id = agent.agent_id
        if agent_id not in self._compiled_actors:
            try:
                # Compile with reduce-overhead mode for inference
                compiled = torch.compile(agent.actor, mode='reduce-overhead', fullgraph=False)
                self._compiled_actors[agent_id] = compiled
            except Exception:
                # Fallback to uncompiled if torch.compile fails
                self._compiled_actors[agent_id] = agent.actor

        return self._compiled_actors[agent_id]

    def evaluate_population(self) -> Tuple[List[float], Dict]:
        """
        Evaluate all agents in population with GPU-accelerated batch inference.

        Why evaluation is slower than validation (despite 5 vs 10 slices):
        - Evaluation collects transitions for the replay buffer (per-step copies + optional
          buffer flush wait), and computes triad_fitness per episode. Validation only
          runs episodes and aggregates scores (no transition collection).
        - So evaluation does more work per slice; validation runs more slices but each
          is lighter.

        Returns:
            Tuple of (fitness_scores, aggregate_stats)
        """
        # Refresh references (population changes after evolution)
        self.population = self.trainer.population
        self.eval_env = self.trainer.eval_env

        # Ensure all agents are on the target device (GPU if available)
        self._ensure_agents_on_device()

        num_episodes = 5 if self.trainer.consistency_mode else 3

        # Log evaluation info
        if self.trainer.multi2_mode:
            scoring_method = "median - 0.5*std (Penalized Median)"
        else:
            scoring_method = "0.4*mean + 0.6*min (pessimistic)"

        print(f"\n--- Generation {self.trainer.generation + 1}: Evaluating Population (Local/{self.target_device.type.upper()}) ---")
        print(f"Multi-slice evaluation: {num_episodes} slices per agent, scoring = {scoring_method}")

        num_elites = sum(1 for a in self.population if a.is_elite)
        print(f"Elite Demonstration: {num_elites} elites + {len(self.population) - num_elites} exploratory")

        # Reset transition buffer
        self.transition_buffer.reset()

        # Pre-generate all episode start indices for reproducibility
        np.random.seed(self.trainer.seed + self.trainer.generation * 1000)
        all_starts = np.random.randint(
            self.train_start,
            self.max_start,
            size=(len(self.population), num_episodes)
        )

        fitness_scores = []
        all_episode_stats = []

        # Process each agent
        device_label = self.target_device.type.upper()

        for agent_idx, agent in enumerate(tqdm(self.population, desc=f"Evaluating ({device_label})")):
            slice_fitness = []
            slice_stats = []

            for ep_idx in range(num_episodes):
                start_idx = int(all_starts[agent_idx, ep_idx])
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

                try:
                    # Run episode with optimized path
                    fitness, episode_info = self._run_episode_optimized(
                        agent=agent,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        collect_transitions=(not agent.is_elite)  # Only exploratory agents
                    )
                except Exception as e:
                    print(f"\n  [ERROR] Agent {agent_idx}, Episode {ep_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

                # Calculate structural fitness
                triad_fitness = self.trainer.calculate_triad_fitness(episode_info)
                slice_fitness.append(triad_fitness)
                slice_stats.append(episode_info)

            # Calculate final fitness
            if self.trainer.multi2_mode:
                final_fitness = self._penalized_median(slice_fitness)
            else:
                final_fitness = self._pessimistic(slice_fitness)

            fitness_scores.append(float(final_fitness))
            all_episode_stats.append(self._aggregate_stats(slice_stats))

            # Incremental flush: If buffer is > 80% full, flush to disk to prevent RAM exhaustion
            # This prevents the 35GB allocation that causes OS swapping (268s/it stall)
            buffer_capacity = self.transition_buffer.states.shape[0]
            if self.transition_buffer.count >= buffer_capacity * 0.8:
                self._start_background_transfer()
                # Wait for transfer to complete before continuing to ensure RAM is freed
                self.wait_for_transfer()

        # Final flush for any remaining transitions
        if self.transition_buffer.count > 0:
            print(f"\n--- Transferring final {self.transition_buffer.count} transitions to replay buffer ---")
            self._start_background_transfer()

        # Aggregate population stats
        aggregate_stats = self.trainer._aggregate_population_stats(all_episode_stats, fitness_scores)

        return fitness_scores, aggregate_stats

    def _run_episode_optimized(self, agent: DDPGAgent, start_idx: int, end_idx: int,
                                collect_transitions: bool = True) -> Tuple[float, Dict]:
        """
        Run episode using TIME-TRAVEL BATCHING for massive speedup.

        Key insight: Market data is immutable - your trades don't change historical prices.
        So we can pre-fetch ALL states for the episode, run ONE batch inference,
        then fast-replay the environment to calculate rewards.
        """
        env = self.eval_env
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
        num_trading_steps = Config.TRADING_PERIOD_DAYS

        # 1. Reset environment
        env.set_training_mode(collect_transitions)
        env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

        # 2. PRE-FETCH ALL TRADING STATES (zero-copy with stride tricks)
        trading_states = env.get_batch_observations(start_idx, num_trading_steps)

        # 3. BATCH INFERENCE (chunked for GPU memory)
        with torch.no_grad():
            if self._use_compiled:
                # CPU with torch.compile: Use compiled actor for faster inference
                compiled_actor = self._get_compiled_actor(agent)
                compiled_actor.eval()

                # Process full batch through compiled model
                states_tensor = torch.from_numpy(trading_states).float()
                
                try:
                    actions_tensor = compiled_actor(states_tensor)
                    trading_actions = actions_tensor.numpy()
                except Exception:
                    # Fallback to uncompiled actor if lazy compilation fails (e.g., missing compiler on Windows)
                    self._compiled_actors[agent.agent_id] = agent.actor
                    actions_tensor = agent.actor(states_tensor)
                    trading_actions = actions_tensor.numpy()

                # Add noise if collecting transitions
                if collect_transitions:
                    noise = np.random.normal(0, agent.noise_scale, trading_actions.shape)
                    trading_actions = trading_actions + noise
                    trading_actions[:, :, 0] = np.maximum(trading_actions[:, :, 0], 0)
                    trading_actions[:, :, 1] = np.clip(trading_actions[:, :, 1], Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET)

                # Safety clip for coefficients
                trading_actions[:, :, 0] = np.clip(trading_actions[:, :, 0], 0, 100)
            else:
                # GPU: Process in chunks to avoid OOM (LSTM is memory-hungry)
                chunk_size = 16  # ~4GB per chunk with 117 LSTM columns
                action_chunks = []
                for i in range(0, num_trading_steps, chunk_size):
                    chunk = trading_states[i:i+chunk_size]
                    action_chunks.append(agent.select_actions_batch(chunk, add_noise=collect_transitions))
                trading_actions = np.concatenate(action_chunks, axis=0)

        # 4. FAST REPLAY - Step environment to compute rewards
        # Uses fast_step() which skips _get_observation() and _get_info() since
        # observations are pre-fetched. This eliminates ~125 wasted [151,117,5]
        # array allocations per episode (plus noise generation in training mode).
        cumulative_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        # Pre-allocate for batched transition add (avoids 125× per-step buffer.add overhead)
        if collect_transitions:
            ep_rewards = np.zeros(num_trading_steps, dtype=np.float32)
            ep_dones = np.zeros(num_trading_steps, dtype=np.float32)

        # A. Trading Period (actions pre-computed)
        for i in range(num_trading_steps):
            reward, terminated, truncated = env.fast_step(trading_actions[i])

            if collect_transitions:
                ep_rewards[i] = reward
                ep_dones[i] = float(terminated or truncated)

            cumulative_reward += reward
            steps += 1
            if terminated or truncated:
                break

        # Batched add: one bulk copy per episode instead of per-step (much faster)
        if collect_transitions and steps > 0:
            batch_next = np.empty((steps,) + trading_states.shape[1:], dtype=np.float32)
            batch_next[:-1] = trading_states[1:steps]
            batch_next[-1] = env._get_observation()
            self.transition_buffer.add_batch(
                trading_states[:steps],
                trading_actions[:steps],
                ep_rewards[:steps],
                batch_next,
                ep_dones[:steps],
            )

        # B. Settlement Period
        if not (terminated or truncated):
            dummy_action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM), dtype=np.float32)
            while True:
                reward, terminated, truncated = env.fast_step(dummy_action)
                cumulative_reward += reward
                steps += 1
                if terminated or truncated:
                    break

        # Get episode summary
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

        return final_fitness, episode_summary

    def _transfer_to_buffer_sync(self, transitions: Dict[str, np.ndarray], count: int):
        """
        Actually transfer transitions to the on-disk replay buffer.

        This is the disk I/O work that runs in a background thread.
        Uses columnar format to avoid creating 50,000+ Python dict objects.
        """
        import gc

        if count == 0:
            return

        # Pass columnar data directly - no "unzipping" into list of dicts
        self.replay_buffer.add_batch_columnar(transitions, count)

        # Clear the pending data
        self._pending_transitions = None
        self._pending_count = 0

        # Force GC to free the large transition arrays immediately
        gc.collect()

    def _start_background_transfer(self):
        """
        Start background thread for disk I/O while validation runs.

        The transitions are copied to a holding area so the main buffer
        can be reset immediately. The background thread does the slow
        disk writes while GPU validation proceeds in parallel.
        """
        if self.transition_buffer.count == 0:
            return

        # Wait for any previous transfer to complete (shouldn't happen normally)
        self.wait_for_transfer()

        # Copy transitions to holding area (the numpy arrays will be written by background thread)
        # We need to copy because transition_buffer will be reset
        self._pending_transitions = {
            'states': self.transition_buffer.states[:self.transition_buffer.count].copy(),
            'actions': self.transition_buffer.actions[:self.transition_buffer.count].copy(),
            'rewards': self.transition_buffer.rewards[:self.transition_buffer.count].copy(),
            'next_states': self.transition_buffer.next_states[:self.transition_buffer.count].copy(),
            'dones': self.transition_buffer.dones[:self.transition_buffer.count].copy()
        }
        self._pending_count = self.transition_buffer.count

        # Reset buffer immediately so it's ready for next generation
        self.transition_buffer.reset()

        # Start background thread for disk I/O
        self._transfer_start_time = time.time()
        self._transfer_thread = threading.Thread(
            target=self._transfer_to_buffer_sync,
            args=(self._pending_transitions, self._pending_count),
            daemon=True
        )
        self._transfer_thread.start()

    def wait_for_transfer(self):
        """
        Wait for background transfer to complete.

        Call this before train_population() to ensure all transitions
        are in the replay buffer before DDPG gradient steps.
        """
        if self._transfer_thread is not None and self._transfer_thread.is_alive():
            self._transfer_thread.join()
        self._transfer_thread = None

    def restore_agents_to_cpu(self):
        """
        Move all agents back to CPU after GPU inference.

        Call this after evaluation/validation and before train_population()
        to free GPU memory. Training will move agents to GPU in batches of 16.

        This is critical for --local mode where all 96 agents are moved to GPU
        for fast batch inference, but training them all on GPU causes OOM.

        IMPORTANT: Uses move_to_device() which recreates optimizers. This is
        necessary to drop references to old GPU tensors and actually free memory.
        """
        import gc

        if self.target_device.type != 'cuda':
            print(f"  [Device] restore_agents_to_cpu: target_device is {self.target_device}, skipping")
            return  # No-op if not using GPU for inference

        # Refresh population reference
        self.population = self.trainer.population

        # Debug: Check GPU memory before restore
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
            print(f"  [Device] GPU memory before restore: {gpu_mem_before:.2f} GB")

        cpu_device = torch.device('cpu')
        moved = 0
        cuda_count = 0
        cpu_count = 0

        for agent in self.population:
            if agent.device.type == 'cuda':
                cuda_count += 1
                # Use move_to_device which properly handles optimizer references
                agent.move_to_device(cpu_device, recreate_optimizers=True)
                moved += 1
            else:
                cpu_count += 1

        # Force garbage collection to actually free the GPU tensors
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            print(f"  [Device] Moved {moved}/{len(self.population)} agents to CPU (were: {cuda_count} cuda, {cpu_count} cpu)")
            print(f"  [Device] GPU memory after restore: {gpu_mem_after:.2f} GB (freed {gpu_mem_before - gpu_mem_after:.2f} GB)")

    def validate_population(self, quality_threshold: float = None) -> List[Dict]:
        """
        Validate all agents with GPU-accelerated batch inference.

        Args:
            quality_threshold: Threshold for counting quality trades

        Returns:
            List of validation results for each agent
        """
        # Refresh references (population changes after evolution)
        self.population = self.trainer.population
        self.eval_env = self.trainer.eval_env

        # Ensure all agents are on the target device (GPU if available)
        self._ensure_agents_on_device()

        print(f"\n--- Walk-Forward Validation (Local/{self.target_device.type.upper()}) ---")

        if not self.trainer.current_generation_val_slices:
            raise ValueError("No validation slices for this generation")

        validation_slices = self.trainer.current_generation_val_slices
        use_penalized_median = self.trainer.multi2_mode

        validation_results = []

        device_label = self.target_device.type.upper()
        for agent in tqdm(self.population, desc=f"Validating ({device_label})"):
            val_result = self._validate_agent_optimized(
                agent=agent,
                slices=validation_slices,
                quality_threshold=quality_threshold,
                use_penalized_median=use_penalized_median
            )
            validation_results.append(val_result)

        return validation_results

    def _validate_agent_optimized(self, agent: DDPGAgent,
                                   slices: List[Tuple[int, int, str]],
                                   quality_threshold: float = None,
                                   use_penalized_median: bool = False) -> Dict:
        """
        Validate a single agent on all slices with minimal overhead.

        Args:
            agent: Agent to validate
            slices: List of (start_idx, end_idx, name) tuples
            quality_threshold: Threshold for quality trades
            use_penalized_median: Use penalized median scoring

        Returns:
            Validation results dictionary
        """
        env = self.eval_env
        slice_results = []
        all_closed_trades = []

        for start_idx, end_idx, _ in slices:
            # Run episode (no transitions collected for validation)
            fitness, episode_info = self._run_validation_episode(
                agent=agent,
                env=env,
                start_idx=start_idx,
                end_idx=end_idx
            )

            # Add max coefficient bonus for zero-trade agents
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

        # Aggregate slice fitness
        fitness_scores = [r['fitness'] for r in slice_results]

        if use_penalized_median:
            validation_fitness = self._penalized_median(fitness_scores)
        else:
            validation_fitness = self._pessimistic(fitness_scores)

        # Aggregate metrics
        total_raw_pnl = sum(r['raw_pnl'] for r in slice_results)
        total_peak_capital = sum(r['peak_capital_employed'] for r in slice_results)
        roi = (total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0

        total_wins = sum(r['num_wins'] for r in slice_results)
        total_losses = sum(r['num_losses'] for r in slice_results)
        total_trades = total_wins + total_losses
        global_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        expectancy = _calculate_expectancy(all_closed_trades)

        quality_count = 0
        if quality_threshold is not None:
            quality_count = sum(
                1 for trade in all_closed_trades
                if trade.get('gain_pct', 0) >= quality_threshold
            )

        sample_trade = all_closed_trades[0] if all_closed_trades else None

        return {
            'fitness': validation_fitness,
            'fitness_all_slices': fitness_scores,
            'fitness_mean': float(np.mean(fitness_scores)),
            'fitness_min': float(np.min(fitness_scores)),
            'win_rate': global_win_rate,
            'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
            'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
            'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
            'avg_reward_per_trade': float(np.mean([r['avg_reward_per_trade'] for r in slice_results])),
            'raw_pnl': total_raw_pnl,
            'total_investment': sum(r['total_investment'] for r in slice_results),
            'roi': roi,
            'expectancy': expectancy,
            'sample_trade': sample_trade,
            'total_trades': total_trades,
            'quality_count': quality_count,
        }

    def _run_validation_episode(self, agent: DDPGAgent, env,
                                 start_idx: int, end_idx: int) -> Tuple[float, Dict]:
        """
        Run validation episode using TIME-TRAVEL BATCHING.
        Same optimization as training - one batch inference for all trading steps.
        """
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
        num_trading_steps = Config.TRADING_PERIOD_DAYS

        env.set_training_mode(False)
        env.reset(start_idx=start_idx, end_idx=end_idx, trading_end_idx=trading_end_idx)

        # Pre-fetch all states and batch inference (chunked for GPU memory)
        trading_states = env.get_batch_observations(start_idx, num_trading_steps)
        with torch.no_grad():
            if self._use_compiled:
                # CPU with torch.compile: Use compiled actor
                compiled_actor = self._get_compiled_actor(agent)
                compiled_actor.eval()
                states_tensor = torch.from_numpy(trading_states).float()
                
                try:
                    actions_tensor = compiled_actor(states_tensor)
                    trading_actions = actions_tensor.numpy()
                except Exception as e:
                    # Fallback to uncompiled actor if lazy compilation fails
                    self._compiled_actors[agent.agent_id] = agent.actor
                    actions_tensor = agent.actor(states_tensor)
                    trading_actions = actions_tensor.numpy()
                
                # Safety clip for coefficients (no noise for validation)
                trading_actions[:, :, 0] = np.clip(trading_actions[:, :, 0], 0, 100)
            else:
                # GPU: Process in chunks to avoid OOM
                chunk_size = 16
                action_chunks = []
                for i in range(0, num_trading_steps, chunk_size):
                    chunk = trading_states[i:i+chunk_size]
                    action_chunks.append(agent.select_actions_batch(chunk, add_noise=False))
                trading_actions = np.concatenate(action_chunks, axis=0)

        cumulative_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        # Fast replay trading period — use fast_step() to skip wasted
        # _get_observation() and _get_info() (observations are pre-fetched)
        for i in range(num_trading_steps):
            reward, terminated, truncated = env.fast_step(trading_actions[i])
            cumulative_reward += reward
            steps += 1
            if terminated or truncated:
                break

        # Settlement period
        if not (terminated or truncated):
            dummy_action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM), dtype=np.float32)
            while True:
                reward, terminated, truncated = env.fast_step(dummy_action)
                cumulative_reward += reward
                steps += 1
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

        return final_fitness, episode_summary

    @staticmethod
    def _penalized_median(scores: List[float]) -> float:
        """Penalized median: median - 0.5 * std"""
        arr = np.array(scores)
        return float(np.median(arr) - 0.5 * np.std(arr))

    @staticmethod
    def _pessimistic(scores: List[float]) -> float:
        """Pessimistic aggregation: 0.4 * mean + 0.6 * min"""
        arr = np.array(scores)
        return float(0.4 * np.mean(arr) + 0.6 * np.min(arr))

    def _aggregate_stats(self, slice_stats: List[Dict]) -> Dict:
        """Aggregate episode stats across slices."""
        return self.trainer._aggregate_agent_stats(slice_stats)
