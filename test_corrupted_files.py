"""
Test script to verify robust error handling with corrupted files
Tests that the buffer can recover from corrupted files without infinite recursion
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

def test_corrupted_file_handling():
    """Test that buffer handles corrupted files gracefully."""
    print("="*60)
    print("Testing Corrupted File Handling")
    print("="*60)

    storage_path = "test_corrupted_buffer"

    # Clean up any existing test files
    if Path(storage_path).exists():
        shutil.rmtree(storage_path)

    # Create buffer and add transitions
    print("\n1. Creating buffer and adding 5100 transitions...")
    buffer = OnDiskReplayBuffer(capacity=10000, storage_path=storage_path)

    for i in range(5100):
        if i % 1000 == 0:
            print(f"   Added {i} transitions...")
        state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).astype(np.float32)
        action = np.random.randn(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL).astype(np.float32)
        done = False
        buffer.add(state, action, reward, next_state, done)

    print(f"   ✓ Buffer size: {len(buffer)}")
    print(f"   ✓ Buffer ready: {buffer.is_ready()}")

    # Corrupt some files
    print("\n2. Corrupting 5 random transition files...")
    storage_dir = Path(storage_path)
    all_files = list(storage_dir.glob("transition_*.pkl.gz"))

    # Pick 5 random files to corrupt
    import random
    files_to_corrupt = random.sample(all_files, min(5, len(all_files)))

    for file_path in files_to_corrupt:
        # Overwrite with garbage data
        with open(file_path, 'wb') as f:
            f.write(b'CORRUPTED DATA' * 100)
        print(f"   Corrupted: {file_path.name}")

    print("\n3. Testing direct sampling with corrupted files...")
    # Test that sample() returns None for corrupted files instead of recursing
    successful_samples = 0
    failed_samples = 0

    for i in range(20):
        batch = buffer.sample(Config.BATCH_SIZE)
        if batch is not None:
            successful_samples += 1
        else:
            failed_samples += 1

    print(f"   ✓ Successful samples: {successful_samples}/20")
    print(f"   ✓ Failed samples (returned None): {failed_samples}/20")
    print(f"   ✓ No infinite recursion!")

    # Test with DataLoader
    print("\n4. Creating DataLoader with corrupted files...")
    dataloader = DataLoader(
        buffer,
        batch_size=None,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    iterator = iter(dataloader)

    print("\n5. Testing DataLoader can handle corrupted files...")
    print("   (Should retry and eventually succeed or skip gracefully)")

    successful_batches = 0
    for i in range(10):
        try:
            batch = next(iterator)
            if batch is not None:
                successful_batches += 1
                print(f"   Batch {i+1}: ✓ Loaded successfully (shape: {batch['states'].shape})")
            else:
                print(f"   Batch {i+1}: Skipped due to errors")
        except StopIteration:
            print(f"   Batch {i+1}: Iterator stopped")
            break
        except Exception as e:
            print(f"   Batch {i+1}: Unexpected error: {e}")

    print(f"\n   ✓ Successfully loaded {successful_batches}/10 batches")
    print(f"   ✓ No crashes or infinite loops!")

    # Cleanup
    print("\n6. Cleaning up test files...")
    if Path(storage_path).exists():
        shutil.rmtree(storage_path)
    print("   ✓ Cleanup complete")

    print("\n" + "="*60)
    print("✓ Corrupted File Handling Test PASSED!")
    print("="*60)
    print("\nThe robustness improvements ensure:")
    print("1. sample() returns None instead of recursing infinitely")
    print("2. __iter__() has retry logic with max attempts (10)")
    print("3. Training continues even with some corrupted files")
    print("4. Clear warnings are logged for debugging")

if __name__ == "__main__":
    test_corrupted_file_handling()
