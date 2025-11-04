#!/usr/bin/env python3
"""
Script to delete old replay buffer from GCS
Run this on RunPod to clean up the stale buffer
"""

import os
import sys

def cleanup_gcs_buffer():
    """Delete old replay buffer from GCS."""
    print("=" * 60)
    print("Cleaning up old replay buffer from GCS")
    print("=" * 60)

    # Set credentials
    credentials_path = "/workspace/gcs-credentials.json"
    if os.path.exists(credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        print(f"✓ Using credentials: {credentials_path}")
    else:
        print(f"Warning: Credentials not found at {credentials_path}")

    try:
        from google.cloud import storage
    except ImportError:
        print("Error: google-cloud-storage not installed")
        print("Installing...")
        os.system("pip install google-cloud-storage")
        from google.cloud import storage

    # Initialize GCS client
    try:
        client = storage.Client()
        bucket_name = "eigen2-checkpoints-ase0"
        bucket = client.bucket(bucket_name)
        print(f"✓ Connected to GCS bucket: {bucket_name}")
    except Exception as e:
        print(f"Error connecting to GCS: {e}")
        return False

    # Files to delete
    files_to_delete = [
        "eigen2/checkpoints/replay_buffer.pkl",
        "eigen2/checkpoints/replay_buffer.pkl.gz",
    ]

    deleted = 0
    not_found = 0

    print("\nSearching for old buffer files...")

    for file_path in files_to_delete:
        blob = bucket.blob(file_path)

        if blob.exists():
            # Get file size
            blob.reload()
            size_mb = blob.size / (1024 * 1024)
            print(f"\n  Found: gs://{bucket_name}/{file_path}")
            print(f"  Size: {size_mb:.2f} MB")

            # Delete
            try:
                blob.delete()
                print(f"  ✓ Deleted successfully")
                deleted += 1
            except Exception as e:
                print(f"  ✗ Error deleting: {e}")
        else:
            print(f"\n  Not found: {file_path}")
            not_found += 1

    # Summary
    print("\n" + "=" * 60)
    print("Cleanup Summary:")
    print(f"  Files deleted: {deleted}")
    print(f"  Files not found: {not_found}")
    print("=" * 60)

    if deleted > 0:
        print("\n✓ Old replay buffers removed from GCS")
        print("  Future resumes will start with fresh buffers")
    else:
        print("\n✓ No old buffers found (already clean)")

    return True


if __name__ == "__main__":
    try:
        success = cleanup_gcs_buffer()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
