"""
Replay Buffer for Project Eigen 2
Shared experience replay for ERL population
"""

import numpy as np
import torch
from typing import Dict, Tuple
from collections import deque
import pickle
import gzip
import os
from torch.utils.data import IterableDataset
import time

from utils.config import Config
from pathlib import Path

class ReplayBuffer:
    """
    Circular replay buffer for storing and sampling experiences.
    Shared across all agents in the ERL population.
    """
    
    def __init__(self, capacity: int = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store. If None, uses Config.BUFFER_SIZE
        """
        self.capacity = capacity or Config.BUFFER_SIZE
        
        # Use deque for automatic circular buffer behavior
        self.buffer = deque(maxlen=self.capacity)
        
        # Statistics
        self.total_added = 0
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: State observation [context_days, num_columns, 9]
            action: Action taken [108, 2]
            reward: Reward received
            next_state: Next state observation [context_days, num_columns, 9]
            done: Whether episode ended
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.buffer.append(transition)
        self.total_added += 1
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with batched tensors on GPU
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        # Random sampling
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Extract transitions
        batch = [self.buffer[i] for i in indices]
        
        # Stack into tensors
        states = np.stack([t['state'] for t in batch])
        actions = np.stack([t['action'] for t in batch])
        rewards = np.array([t['reward'] for t in batch]).reshape(-1, 1)
        next_states = np.stack([t['next_state'] for t in batch])
        dones = np.array([t['done'] for t in batch]).reshape(-1, 1)
        
        # Convert to PyTorch tensors and move to GPU
        return {
            'states': torch.FloatTensor(states).to(Config.DEVICE),
            'actions': torch.FloatTensor(actions).to(Config.DEVICE),
            'rewards': torch.FloatTensor(rewards).to(Config.DEVICE),
            'next_states': torch.FloatTensor(next_states).to(Config.DEVICE),
            'dones': torch.FloatTensor(dones).to(Config.DEVICE)
        }
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training."""
        # Use lower threshold for W&B sweeps to enable DDPG with fewer generations
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        min_size = Config.MIN_BUFFER_SIZE_SWEEP if is_sweep else Config.MIN_BUFFER_SIZE
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added
        }
    
    def save(self, file_path: str, compress: bool = True):
        """
        Saves the buffer's core components to a file using pickle with optional compression.

        Args:
            file_path: Path to save buffer
            compress: If True, use gzip compression (reduces size by ~50-70%)
        """
        # Save the core components in a dictionary
        save_data = {
            'capacity': self.capacity,
            'buffer': self.buffer,  # This is the deque
        }

        buffer_size_mb = len(self.buffer) * 3.74  # Approximate MB
        print(f"  Saving replay buffer ({len(self.buffer)} transitions, ~{buffer_size_mb:.0f} MB) to {file_path}...")

        if compress:
            print(f"  Using gzip compression...")

        try:
            if compress:
                with gzip.open(file_path + '.gz', 'wb', compresslevel=6) as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Remove uncompressed file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                # Rename compressed file to original name
                os.rename(file_path + '.gz', file_path)

                # Report compression savings
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                compression_ratio = (1 - file_size_mb / buffer_size_mb) * 100
                print(f"  ✓ Buffer saved: {file_size_mb:.1f} MB ({compression_ratio:.0f}% compression)")
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  ✓ Buffer saved: {file_size_mb:.1f} MB (uncompressed)")

        except (MemoryError, OverflowError) as e:
            print(f"  ❌ ERROR saving buffer: {e}")
            print("  Buffer is too large to save directly. Saving failed.")

    @staticmethod
    def load(file_path: str) -> 'ReplayBuffer':
        """
        Loads a ReplayBuffer from a pickled data file (auto-detects compression).
        """
        print(f"  Loading replay buffer from {file_path}...")

        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")

        try:
            # Try to detect if file is gzip compressed by reading magic bytes
            with open(file_path, 'rb') as f:
                magic = f.read(2)
                is_gzipped = (magic == b'\x1f\x8b')

            if is_gzipped:
                print(f"  Detected gzip compression, decompressing...")
                with gzip.open(file_path, 'rb') as f:
                    save_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    save_data = pickle.load(f)

            # Create a new buffer with the loaded capacity
            new_buffer = ReplayBuffer(capacity=save_data['capacity'])

            # Set its internal data
            new_buffer.buffer = save_data['buffer']
            
            print(f"  ✓ Buffer loaded (size: {len(new_buffer.buffer)}).")
            return new_buffer
        except Exception as e:
            print(f"  ❌ ERROR loading buffer: {e}")
            print("  Could not load buffer. Creating a new empty one.")
            # Return a new, empty buffer as a fallback
            return ReplayBuffer(capacity=Config.REPLAY_BUFFER_SIZE)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    Samples transitions based on their TD-error (priority).
    More complex but can improve learning efficiency.
    """
    
    def __init__(self, capacity: int = None, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (compensates for bias)
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=self.capacity)
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add transition with maximum priority."""
        super().add(state, action, reward, next_state, done)
        # New transitions get max priority to ensure they're sampled at least once
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch based on priorities.
        
        Returns:
            Tuple of (batch_dict, indices, importance_weights)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Extract transitions
        batch = [self.buffer[i] for i in indices]
        
        # Stack into tensors
        states = np.stack([t['state'] for t in batch])
        actions = np.stack([t['action'] for t in batch])
        rewards = np.array([t['reward'] for t in batch]).reshape(-1, 1)
        next_states = np.stack([t['next_state'] for t in batch])
        dones = np.array([t['done'] for t in batch]).reshape(-1, 1)
        
        # Convert to PyTorch tensors
        batch_dict = {
            'states': torch.FloatTensor(states).to(Config.DEVICE),
            'actions': torch.FloatTensor(actions).to(Config.DEVICE),
            'rewards': torch.FloatTensor(rewards).to(Config.DEVICE),
            'next_states': torch.FloatTensor(next_states).to(Config.DEVICE),
            'dones': torch.FloatTensor(dones).to(Config.DEVICE)
        }
        
        return batch_dict, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Indices of sampled transitions
            td_errors: TD-errors (used as priorities)
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # Small epsilon to avoid zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


# Test replay buffer
if __name__ == "__main__":
    print("Testing Replay Buffer...\n")
    
    # Create buffer
    buffer = ReplayBuffer(capacity=1000)
    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Buffer size: {len(buffer)}")
    print(f"Is ready: {buffer.is_ready()}")
    
    # Add some dummy transitions
    print("\n--- Adding Transitions ---")
    num_transitions = 500
    for i in range(num_transitions):
        state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        action = np.random.randn(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM)
        reward = np.random.randn()
        next_state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        done = i % 50 == 49  # Every 50th transition is terminal

        buffer.add(state, action, reward, next_state, done)
    
    print(f"Added {num_transitions} transitions")
    print(f"Buffer size: {len(buffer)}")
    print(f"Is ready: {buffer.is_ready()}")
    
    stats = buffer.get_stats()
    print("\nBuffer stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test sampling
    print("\n--- Testing Sampling ---")
    batch_size = Config.BATCH_SIZE
    print(f"Sampling batch of size {batch_size}...")
    
    import time
    start = time.time()
    batch = buffer.sample(batch_size)
    elapsed = time.time() - start
    
    print(f"Sampling time: {elapsed:.3f} seconds")
    print(f"\nBatch contents:")
    for key, tensor in batch.items():
        print(f"  {key}: shape={tensor.shape}, device={tensor.device}, dtype={tensor.dtype}")
    
    # Test overflow (circular buffer behavior)
    print("\n--- Testing Circular Buffer ---")
    initial_size = len(buffer)
    for i in range(600):  # Add more than capacity
        state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        action = np.random.randn(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM)
        reward = 0.0
        next_state = state
        done = False
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Initial size: {initial_size}")
    print(f"After adding 600 more: {len(buffer)}")
    print(f"Should be capped at capacity: {buffer.capacity}")
    print(f"Total added (all time): {buffer.total_added}")
    
    # Test prioritized buffer (optional)
    print("\n--- Testing Prioritized Replay Buffer ---")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # Add transitions
    for i in range(100):
        state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        action = np.random.randn(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM)
        reward = np.random.randn()
        next_state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        done = False
        pri_buffer.add(state, action, reward, next_state, done)
    
    print(f"Prioritized buffer size: {len(pri_buffer)}")
    
    # Sample with priorities
    batch, indices, weights = pri_buffer.sample(batch_size)
    print(f"Sampled indices: {indices[:5]}...")
    print(f"Importance weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Update priorities
    dummy_td_errors = np.random.rand(batch_size)
    pri_buffer.update_priorities(indices, dummy_td_errors)
    print("Updated priorities based on TD-errors")
    
    print("\n✓ Replay buffer tests complete!")

class OnDiskReplayBuffer(IterableDataset):
    """
    Circular replay buffer that stores transitions on disk to save RAM.
    The in-memory buffer holds only file paths.
    Implements IterableDataset for asynchronous batch loading with DataLoader.
    """

    def __init__(self, capacity: int = None, storage_path: str = "buffer_storage"):
        """
        Initialize on-disk replay buffer.

        Args:
            capacity: Maximum number of transitions. Uses Config.BUFFER_SIZE if None.
            storage_path: Directory to store transition files.
        """
        super().__init__()
        self.capacity = capacity or Config.BUFFER_SIZE
        self.storage_path = Path(storage_path)

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Deque now stores file paths (strings)
        self.buffer = deque(maxlen=self.capacity)

        # Statistics
        self.total_added = 0

        print(f"OnDiskReplayBuffer initialized. Capacity: {self.capacity}, Storage: {self.storage_path}")

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """
        Save a transition to disk and add its path to the buffer.
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        # Get a unique file path for this transition
        file_id = self.total_added
        file_path = self.storage_path / f"transition_{file_id}.pkl.gz"

        # Check if buffer is full and we need to remove an old file
        old_path_to_remove = None
        if len(self.buffer) == self.capacity:
            # Get path of the oldest file (which deque will pop)
            old_path_to_remove = self.buffer[0]

        try:
            # Save the new transition file (low compression for speed)
            with gzip.open(file_path, 'wb', compresslevel=1) as f:
                pickle.dump(transition, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Add the new path to the buffer
            self.buffer.append(str(file_path))

            # If an old file was popped, delete it from disk
            if old_path_to_remove:
                try:
                    os.remove(old_path_to_remove)
                except OSError:
                    pass # File might already be gone

            self.total_added += 1

        except Exception as e:
            print(f"❌ ERROR saving on-disk transition {file_path}: {e}")

    def add_batch(self, transitions: list):
        """
        Efficiently add multiple transitions at once using parallel writes.
        Much faster than calling add() sequentially for large batches.

        Args:
            transitions: List of transition dicts with keys: state, action, reward, next_state, done
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading

        if not transitions:
            return

        num_transitions = len(transitions)
        print(f"  Adding {num_transitions} transitions to on-disk buffer...")

        # Thread-safe counter for file IDs
        lock = threading.Lock()

        def write_transition(transition):
            """Write a single transition to disk (runs in thread pool)"""
            with lock:
                file_id = self.total_added
                self.total_added += 1
                file_path = self.storage_path / f"transition_{file_id}.pkl.gz"

            try:
                # Write to disk (this is the slow I/O operation)
                with gzip.open(file_path, 'wb', compresslevel=1) as f:
                    pickle.dump(transition, f, protocol=pickle.HIGHEST_PROTOCOL)
                return str(file_path), True
            except Exception as e:
                print(f"❌ ERROR saving transition {file_path}: {e}")
                return None, False

        # Write all transitions in parallel using thread pool
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write_transition, transitions))

        # Add successful paths to buffer and track failed writes
        successful_paths = []
        for path, success in results:
            if success and path:
                successful_paths.append(path)

        # Add all paths to buffer (this is fast - just appending to deque)
        # Determine how many old files to remove
        num_to_remove = max(0, len(self.buffer) + len(successful_paths) - self.capacity)
        old_paths_to_remove = []

        if num_to_remove > 0:
            # Collect paths to remove before modifying buffer
            old_paths_to_remove = [self.buffer[i] for i in range(num_to_remove)]

        # Add new paths
        for path in successful_paths:
            self.buffer.append(path)

        # Delete old files from disk in parallel
        if old_paths_to_remove:
            def delete_file(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(delete_file, old_paths_to_remove)

        elapsed = time.time() - start_time
        rate = num_transitions / elapsed if elapsed > 0 else 0
        print(f"  ✓ Added {len(successful_paths)}/{num_transitions} transitions in {elapsed:.2f}s ({rate:.0f} trans/s)")
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch by loading transitions from disk.

        Note: When used with DataLoader, this runs in background worker processes.
        The tensors are created on CPU, then transferred to GPU after pin_memory.

        Returns:
            Dictionary of tensors, or None if sampling failed.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")

        # 1. Sample random indices from the deque
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # 2. Get the file paths for those indices
        batch_paths = [self.buffer[i] for i in indices]

        # 3. Load transitions from disk
        batch_transitions = []
        for path in batch_paths:
            try:
                with gzip.open(path, 'rb') as f:
                    batch_transitions.append(pickle.load(f))
            except Exception as e:
                # This can happen if a file is corrupted or mid-write
                # Don't recurse - let caller handle retry
                continue

        # Return None if we couldn't load any transitions
        if not batch_transitions:
            return None

        # 4. Stack into tensors
        try:
            states = np.stack([t['state'] for t in batch_transitions])
            actions = np.stack([t['action'] for t in batch_transitions])
            rewards = np.array([t['reward'] for t in batch_transitions]).reshape(-1, 1)
            next_states = np.stack([t['next_state'] for t in batch_transitions])
            dones = np.array([t['done'] for t in batch_transitions]).reshape(-1, 1)
        except Exception as e:
            # Data corrupted - return None and let caller retry
            print(f"❌ ERROR stacking transitions: {e}")
            return None

        # 5. Convert to PyTorch tensors (on CPU in worker processes)
        # DataLoader with pin_memory will automatically transfer to GPU efficiently
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'next_states': torch.FloatTensor(next_states),
            'dones': torch.FloatTensor(dones)
        }

    def __iter__(self):
        """
        Iterator for PyTorch DataLoader. Yields batches infinitely.
        This enables asynchronous prefetching - background workers prepare batches
        while GPU is busy training, eliminating I/O wait time.

        Implements robust retry logic to handle corrupted files without infinite recursion.
        """
        MAX_RETRIES = 10  # Max retries per batch before giving up

        while True:
            # Wait if buffer is not ready
            if not self.is_ready():
                time.sleep(0.1)
                continue

            # Try to sample a batch with retry logic
            batch = None
            retry_count = 0

            while batch is None and retry_count < MAX_RETRIES:
                try:
                    # Sample batch (this is the slow disk I/O operation)
                    # When used with DataLoader workers, this runs in parallel with GPU training
                    batch = self.sample(Config.BATCH_SIZE)

                    if batch is None:
                        # sample() returned None due to corrupted files
                        retry_count += 1
                        if retry_count < MAX_RETRIES:
                            # Wait a bit and try different samples
                            time.sleep(0.05)
                        else:
                            # Too many failures - skip this batch
                            print(f"⚠️ WARNING: Failed to load batch after {MAX_RETRIES} attempts. Skipping...")
                            break

                except Exception as e:
                    # Handle any unexpected errors
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        print(f"⚠️ Warning: Error sampling batch (attempt {retry_count}/{MAX_RETRIES}): {e}")
                        time.sleep(0.1)
                    else:
                        print(f"❌ ERROR: Failed to load batch after {MAX_RETRIES} attempts: {e}")
                        break

            # Only yield if we got a valid batch
            if batch is not None:
                yield batch
            else:
                # All retries failed - wait before trying again
                time.sleep(0.5)

    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self) -> bool:
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        min_size = Config.MIN_BUFFER_SIZE_SWEEP if is_sweep else Config.MIN_BUFFER_SIZE
        return len(self.buffer) >= min_size
    
    def clear(self):
        print("Clearing OnDiskReplayBuffer and deleting files...")
        for path in list(self.buffer): # Iterate copy
            try:
                os.remove(path)
            except OSError:
                pass
        self.buffer.clear()
        print("Buffer cleared.")
    
    def get_stats(self) -> dict:
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added
        }

    def save(self, file_path: str):
        """
        Saves the buffer's metadata (path deque, capacity, counter) to a file.
        This is very fast and uses minimal RAM.
        """
        print(f"  Saving on-disk buffer metadata to {file_path}...")
        save_data = {
            'capacity': self.capacity,
            'buffer': self.buffer,  # The deque of file paths
            'total_added': self.total_added,
            'storage_path': str(self.storage_path) # Save storage path for verification
        }
        try:
            with gzip.open(file_path, 'wb', compresslevel=1) as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  ✓ Buffer metadata saved ({len(self.buffer)} paths).")
        except Exception as e:
            print(f"  ❌ ERROR saving buffer metadata: {e}")

    @staticmethod
    def load(file_path: str, storage_path_override: str = None) -> 'OnDiskReplayBuffer':
        """
        Loads a ReplayBuffer's metadata from a file.
        Verifies that the storage_path still exists.
        """
        print(f"  Loading on-disk buffer metadata from {file_path}...")
        try:
            with gzip.open(file_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # Use override path if provided, else use saved path
            storage_path = storage_path_override or save_data.get('storage_path', 'buffer_storage')

            new_buffer = OnDiskReplayBuffer(
                capacity=save_data['capacity'], 
                storage_path=storage_path
            )
            new_buffer.buffer = save_data['buffer']
            new_buffer.total_added = save_data.get('total_added', len(new_buffer.buffer))
            
            print(f"  ✓ Buffer metadata loaded (size: {len(new_buffer.buffer)}).")
            
            # Verify that the files and storage path still exist
            if len(new_buffer.buffer) > 0:
                first_path_str = new_buffer.buffer[0]
                first_path = Path(first_path_str)
                
                # Check if path in metadata matches expected storage path
                if not first_path.is_relative_to(new_buffer.storage_path):
                     print(f"  ⚠️ WARNING: Path mismatch!")
                     print(f"  Buffer metadata path: {first_path.parent}")
                     print(f"  Trainer storage_path: {new_buffer.storage_path}")
                     print(f"  This may fail if files were moved. Make sure 'storage_path' is correct.")

                if not first_path.exists():
                    print(f"  ⚠️ WARNING: Transition file {first_path_str} not found on disk.")
                    print(f"  If you moved your project, ensure the '{storage_path}' folder was moved too.")
                else:
                    print("  ✓ Verified transition files seem to exist.")
            
            return new_buffer
        except Exception as e:
            print(f"  ❌ ERROR loading buffer metadata: {e}")
            print("  Could not load. Creating a new empty one.")
            return OnDiskReplayBuffer(capacity=Config.BUFFER_SIZE, storage_path=storage_path_override or 'buffer_storage')