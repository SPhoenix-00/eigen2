"""
Test script for debugging cloud sync checkpoint upload hang.
"""
import os
import time
import tempfile
from pathlib import Path
from utils.cloud_sync import get_cloud_sync_from_env


def create_test_checkpoint():
    """Create a minimal test checkpoint directory."""
    test_dir = Path("test_checkpoint_sync")
    test_dir.mkdir(exist_ok=True)

    # Create some test files
    (test_dir / "test_file_1.txt").write_text("Test data 1")
    (test_dir / "test_file_2.txt").write_text("Test data 2")

    # Create a subdirectory
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "test_file_3.txt").write_text("Test data 3")

    return test_dir


def test_sync_with_timeout(timeout_seconds=30):
    """Test cloud sync with a timeout to detect hangs."""
    print("=" * 60)
    print("Testing Cloud Sync Upload")
    print("=" * 60)

    # Get cloud sync instance
    cloud_sync = get_cloud_sync_from_env()

    if cloud_sync is None:
        print("[X] Cloud sync not configured. Skipping test.")
        return

    print(f"[OK] Cloud sync initialized: {cloud_sync.provider}")

    # Create test checkpoint
    test_dir = create_test_checkpoint()
    print(f"[OK] Created test checkpoint at: {test_dir}")

    # Test background upload with monitoring
    print("\n" + "=" * 60)
    print("Starting background upload test...")
    print("=" * 60)

    start_time = time.time()

    try:
        # Call sync_checkpoints with background=True (this is what hangs)
        print("Calling sync_checkpoints(background=True)...")
        cloud_sync.sync_checkpoints(
            checkpoint_dir=str(test_dir),
            background=True,
            exclude_patterns=["replay_buffer"]
        )

        elapsed = time.time() - start_time
        print(f"[OK] sync_checkpoints returned after {elapsed:.2f}s")

        # Wait a bit to see if background upload completes
        print("\nWaiting for background upload to complete...")
        for i in range(timeout_seconds):
            time.sleep(1)
            print(f"  Waited {i+1}s...", end='\r')

            # Check if upload queue is empty (if accessible)
            if hasattr(cloud_sync, 'upload_queue'):
                queue_size = cloud_sync.upload_queue.qsize() if hasattr(cloud_sync.upload_queue, 'qsize') else "unknown"
                print(f"  Waited {i+1}s... Queue size: {queue_size}", end='\r')

        print(f"\n[OK] Test completed (waited {timeout_seconds}s)")

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n[INTERRUPTED] After {elapsed:.2f}s")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] After {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up test checkpoint...")
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("✓ Cleanup complete")


def test_sync_foreground():
    """Test foreground sync (non-background) for comparison."""
    print("\n" + "=" * 60)
    print("Testing FOREGROUND Upload (for comparison)")
    print("=" * 60)

    cloud_sync = get_cloud_sync_from_env()
    if cloud_sync is None:
        print("❌ Cloud sync not configured. Skipping test.")
        return

    test_dir = create_test_checkpoint()
    print(f"✓ Created test checkpoint at: {test_dir}")

    start_time = time.time()

    try:
        print("Calling sync_checkpoints(background=False)...")
        cloud_sync.sync_checkpoints(
            checkpoint_dir=str(test_dir),
            background=False,
            exclude_patterns=["replay_buffer"]
        )

        elapsed = time.time() - start_time
        print(f"✓ Foreground sync completed in {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    import sys

    print("Cloud Sync Upload Test")
    print("=" * 60)
    print("This script tests the checkpoint upload that may be hanging.")
    print("Press Ctrl+C to interrupt if it hangs.")
    print("=" * 60 + "\n")

    # Test background upload (the one that hangs)
    try:
        test_sync_with_timeout(timeout_seconds=30)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)

    # Optionally test foreground upload
    response = input("\nTest foreground upload too? (y/n): ")
    if response.lower() == 'y':
        test_sync_foreground()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
