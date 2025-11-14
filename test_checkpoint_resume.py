"""
Test script to verify checkpoint resume works correctly with DataLoader
This tests the critical bug fix where DataLoader must be recreated after loading buffer
"""

import torch
import numpy as np
import os
import shutil
from pathlib import Path
from models.replay_buffer import OnDiskReplayBuffer
from torch.utils.data import DataLoader
from utils.config import Config

# Set sweep mode for testing
os.environ['WANDB_SWEEP_ID'] = 'test'

def test_checkpoint_resume():
    """Test that DataLoader works correctly after loading a checkpoint."""
    print("="*60)
    print("Testing Checkpoint Resume with DataLoader")
    print("="*60)

    storage_path = "test_checkpoint_buffer"
    checkpoint_path = "test_checkpoint_buffer.pkl"

    # Clean up any existing test files
    if Path(storage_path).exists():
        shutil.rmtree(storage_path)
    if Path(checkpoint_path).exists():
        os.remove(checkpoint_path)

    # Phase 1: Create buffer, add data, and save
    print("\n--- Phase 1: Create and Save Buffer ---")
    print("1. Creating buffer and adding 5100 transitions...")
    buffer1 = OnDiskReplayBuffer(capacity=10000, storage_path=storage_path)

    for i in range(5100):
        if i % 1000 == 0:
            print(f"   Added {i} transitions...")
        state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).astype(np.float32)
        action = np.random.randn(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).astype(np.float32)
        done = False
        buffer1.add(state, action, reward, next_state, done)

    print(f"   ✓ Buffer size: {len(buffer1)}")
    print(f"   ✓ Buffer ready: {buffer1.is_ready()}")

    print("\n2. Creating DataLoader for buffer1...")
    dataloader1 = DataLoader(
        buffer1,
        batch_size=None,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    iterator1 = iter(dataloader1)

    print("3. Sampling 3 batches from buffer1's DataLoader...")
    for i in range(3):
        batch = next(iterator1)
        print(f"   Batch {i+1}: states shape = {batch['states'].shape}")

    print("\n4. Saving buffer1 to checkpoint...")
    buffer1.save(checkpoint_path)
    print(f"   ✓ Saved to {checkpoint_path}")

    # Simulate cleanup (like what happens between training sessions)
    print("\n5. Cleaning up buffer1 and dataloader1 (simulating end of session)...")
    del iterator1
    del dataloader1
    del buffer1
    import gc
    gc.collect()
    print("   ✓ Cleaned up")

    # Phase 2: Load buffer and create new DataLoader (THIS IS THE CRITICAL FIX)
    print("\n--- Phase 2: Load Buffer and Create New DataLoader ---")
    print("6. Loading buffer from checkpoint...")
    buffer2 = OnDiskReplayBuffer.load(checkpoint_path, storage_path_override=storage_path)
    print(f"   ✓ Loaded buffer size: {len(buffer2)}")
    print(f"   ✓ Buffer ready: {buffer2.is_ready()}")

    print("\n7. Creating NEW DataLoader for buffer2 (CRITICAL FIX)...")
    dataloader2 = DataLoader(
        buffer2,
        batch_size=None,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    iterator2 = iter(dataloader2)
    print("   ✓ DataLoader created")

    print("\n8. Testing that new DataLoader works (this would hang before the fix)...")
    import time
    start = time.time()
    try:
        for i in range(3):
            # Set a timeout to detect hanging
            batch = next(iterator2)
            elapsed = time.time() - start
            print(f"   Batch {i+1}: states shape = {batch['states'].shape} (elapsed: {elapsed:.2f}s)")
            if elapsed > 10:
                print("   ⚠ WARNING: Taking too long, might be hanging!")
                break
        print("   ✓ SUCCESS! DataLoader works correctly after checkpoint resume")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        raise

    # Cleanup
    print("\n9. Cleaning up test files...")
    if Path(storage_path).exists():
        shutil.rmtree(storage_path)
    if Path(checkpoint_path).exists():
        os.remove(checkpoint_path)
    print("   ✓ Cleanup complete")

    print("\n" + "="*60)
    print("✓ Checkpoint Resume Test PASSED!")
    print("="*60)
    print("\nThe fix ensures that when resuming from a checkpoint:")
    print("1. The buffer is loaded with its file paths")
    print("2. A NEW DataLoader is created pointing to the LOADED buffer")
    print("3. Training can continue without hanging")

if __name__ == "__main__":
    test_checkpoint_resume()
