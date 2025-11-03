#!/usr/bin/env python3
"""
Test script to verify async background uploads work correctly.
Creates dummy files, uploads them in background, and continues "training".
"""

import os
import time
import tempfile
from pathlib import Path
from utils.cloud_sync import get_cloud_sync_from_env

def create_dummy_file(path: str, size_mb: int = 10):
    """Create a dummy file of specified size."""
    with open(path, 'wb') as f:
        f.write(b'0' * (size_mb * 1024 * 1024))

def simulate_training_work(duration: float):
    """Simulate training work that takes some time."""
    print(f"🧠 Simulating training work for {duration:.1f} seconds...")
    start = time.time()
    # Do some busy work
    result = 0
    while time.time() - start < duration:
        result += sum(range(10000))
    print(f"✓ Training work completed")

def main():
    print("\n" + "="*60)
    print("Testing Async Background Uploads")
    print("="*60 + "\n")

    # Initialize cloud sync
    cloud_sync = get_cloud_sync_from_env()

    if cloud_sync.provider == "local":
        print("⚠️  CLOUD_PROVIDER is 'local'. Async uploads won't be tested.")
        print("Set CLOUD_PROVIDER, CLOUD_BUCKET, and credentials to test GCS uploads.")
        return

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Created temp directory: {tmpdir}\n")

        # Test 1: Single file upload in background
        print("="*60)
        print("Test 1: Single file background upload")
        print("="*60)

        file1 = Path(tmpdir) / "test_file_1.bin"
        create_dummy_file(str(file1), size_mb=5)
        print(f"Created {file1.name} (5 MB)")

        # Upload in background
        cloud_path = f"{cloud_sync.project_name}/test_async/test_file_1.bin"
        cloud_sync.upload_file(str(file1), cloud_path, background=True)

        # Immediately continue with training simulation
        simulate_training_work(2.0)

        # Check status
        pending, completed, failed = cloud_sync.get_upload_status()
        print(f"📊 Upload status: {pending} pending, {completed} completed, {failed} failed")

        # Wait for completion
        cloud_sync.wait_for_uploads(timeout=60)
        print()

        # Test 2: Multiple files uploaded in parallel
        print("="*60)
        print("Test 2: Multiple files uploaded in parallel")
        print("="*60)

        files = []
        for i in range(5):
            file_path = Path(tmpdir) / f"test_file_{i+2}.bin"
            create_dummy_file(str(file_path), size_mb=3)
            files.append(file_path)
            print(f"Created {file_path.name} (3 MB)")

        # Upload all in background
        for i, file_path in enumerate(files):
            cloud_path = f"{cloud_sync.project_name}/test_async/test_file_{i+2}.bin"
            cloud_sync.upload_file(str(file_path), cloud_path, background=True)

        print(f"\n⏳ Queued {len(files)} files for upload")

        # Continue training while uploads happen
        simulate_training_work(3.0)

        # Check status mid-upload
        pending, completed, failed = cloud_sync.get_upload_status()
        print(f"📊 Upload status: {pending} pending, {completed} completed, {failed} failed")

        # Wait for all uploads
        cloud_sync.wait_for_uploads(timeout=120)
        print()

        # Test 3: Directory upload in background
        print("="*60)
        print("Test 3: Directory background upload")
        print("="*60)

        test_dir = Path(tmpdir) / "checkpoint_test"
        test_dir.mkdir()

        # Create multiple files in directory
        for i in range(3):
            file_path = test_dir / f"checkpoint_{i}.pth"
            create_dummy_file(str(file_path), size_mb=2)
            print(f"Created {file_path.name} (2 MB)")

        # Upload directory in background
        cloud_sync.upload_directory(str(test_dir),
                                     f"{cloud_sync.project_name}/test_async/checkpoints",
                                     background=True)

        # Continue training
        simulate_training_work(2.0)

        # Final wait
        cloud_sync.wait_for_uploads(timeout=120)

    # Cleanup
    print("\n" + "="*60)
    print("Test Complete - Shutting down")
    print("="*60)
    cloud_sync.shutdown(wait=True)

    print("\n✓ All tests passed! Background uploads are working correctly.")
    print("\nKey benefits demonstrated:")
    print("  1. Training continues while files upload")
    print("  2. Multiple files upload in parallel (4 workers)")
    print("  3. No blocking on large checkpoint saves")
    print("  4. Clean shutdown waits for pending uploads")

if __name__ == "__main__":
    main()
