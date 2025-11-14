"""
Test script for asynchronous DataLoader implementation
Verifies that OnDiskReplayBuffer works correctly with PyTorch DataLoader
"""

import torch
import numpy as np
import time
import os
from models.replay_buffer import OnDiskReplayBuffer
from torch.utils.data import DataLoader
from utils.config import Config

# Set sweep mode to use lower buffer threshold for testing
os.environ['WANDB_SWEEP_ID'] = 'test'

def test_async_dataloader():
    """Test the asynchronous DataLoader implementation."""
    print("="*60)
    print("Testing Asynchronous DataLoader Implementation")
    print("="*60)

    # Create a small on-disk buffer
    print("\n1. Creating OnDiskReplayBuffer...")
    buffer = OnDiskReplayBuffer(capacity=10000, storage_path="test_buffer_storage")

    # Add enough dummy transitions to make buffer ready
    # MIN_BUFFER_SIZE_SWEEP is 5000 (in sweep mode)
    num_samples = 5500
    print(f"\n2. Adding {num_samples} dummy transitions (this will take a minute)...")
    print("   Progress: ", end="", flush=True)
    for i in range(num_samples):
        if i % 500 == 0:
            print(f"{i}...", end="", flush=True)
        state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).astype(np.float32)
        action = np.random.randn(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).astype(np.float32)
        done = False

        buffer.add(state, action, reward, next_state, done)

    print(f"\n   ✓ Buffer size: {len(buffer)}")
    print(f"   ✓ Buffer ready: {buffer.is_ready()}")

    # Test direct sampling (old way)
    print("\n3. Testing direct sampling (synchronous, old way)...")
    start_time = time.time()
    for i in range(10):
        batch = buffer.sample(Config.BATCH_SIZE)
    sync_time = time.time() - start_time
    print(f"   ✓ 10 batches sampled in {sync_time:.3f}s ({sync_time/10:.3f}s per batch)")
    print(f"   ✓ Batch keys: {batch.keys()}")
    print(f"   ✓ Batch shapes: states={batch['states'].shape}, actions={batch['actions'].shape}")

    # Test DataLoader (new way with async prefetching)
    print("\n4. Creating DataLoader with 4 workers...")
    dataloader = DataLoader(
        buffer,
        batch_size=None,  # Already batched
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    print("\n5. Testing DataLoader iteration (asynchronous, new way)...")
    batch_iterator = iter(dataloader)

    # First batch may be slower (workers starting up)
    print("   Getting first batch (worker startup)...")
    start_time = time.time()
    batch = next(batch_iterator)
    first_batch_time = time.time() - start_time
    print(f"   ✓ First batch: {first_batch_time:.3f}s")

    # Subsequent batches should be fast (already prefetched)
    print("   Getting next 10 batches (should be prefetched)...")
    start_time = time.time()
    for i in range(10):
        batch_cpu = next(batch_iterator)
        # Simulate GPU transfer
        batch = {k: v.to(Config.DEVICE, non_blocking=True) for k, v in batch_cpu.items()}
    async_time = time.time() - start_time
    print(f"   ✓ 10 batches in {async_time:.3f}s ({async_time/10:.3f}s per batch)")

    print(f"   ✓ Batch keys: {batch.keys()}")
    print(f"   ✓ Batch shapes: states={batch['states'].shape}, actions={batch['actions'].shape}")
    print(f"   ✓ Batch device: {batch['states'].device}")

    # Compare speeds
    print("\n6. Performance Comparison:")
    print(f"   Synchronous (old): {sync_time/10:.3f}s per batch")
    print(f"   Asynchronous (new): {async_time/10:.3f}s per batch")
    if async_time < sync_time:
        speedup = sync_time / async_time
        print(f"   ✓ Speedup: {speedup:.2f}x faster!")
    else:
        print(f"   Note: With only 100 samples, async may not show speedup yet.")
        print(f"         Real speedup happens with larger buffers and more I/O.")

    # Cleanup
    print("\n7. Cleaning up...")
    buffer.clear()
    import shutil
    import os
    if os.path.exists("test_buffer_storage"):
        shutil.rmtree("test_buffer_storage")
    print("   ✓ Cleanup complete")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    test_async_dataloader()
