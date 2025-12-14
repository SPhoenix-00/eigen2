#!/usr/bin/env python3
"""
Upload specific agents to GCP for Global 50.

Usage:
    1. Place agent .pth files in global50/inbox/
    2. Run: python upload_agents_to_gcp.py

The script will:
    - Read agents from global50/inbox/
    - Upload them to the appropriate cloud location (eigen2/global50/<cw>/agents/)
    - Move successfully uploaded files to global50/inbox/uploaded/

Environment Variables (required):
    CLOUD_PROVIDER=gcs
    CLOUD_BUCKET=<your-bucket>
    GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials.json>
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.cloud_sync import get_cloud_sync_from_env


def main():
    # Configuration
    inbox_dir = Path("global50/inbox")
    uploaded_dir = inbox_dir / "uploaded"

    # Default context window - can be overridden with --cw flag
    context_window = 151

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Upload agents to GCP Global 50")
    parser.add_argument("--cw", type=int, default=151,
                        help="Context window in days (default: 151)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without actually uploading")
    parser.add_argument("files", nargs="*",
                        help="Specific files to upload (default: all .pth in inbox)")
    args = parser.parse_args()

    context_window = args.cw
    cloud_path_prefix = f"eigen2/global50/cw{context_window}/agents"

    print(f"\n{'='*60}")
    print("Upload Agents to GCP")
    print(f"{'='*60}")
    print(f"Context Window: {context_window} days")
    print(f"Cloud Path: {cloud_path_prefix}/")
    print(f"Inbox: {inbox_dir}")
    print()

    # Create directories if they don't exist
    inbox_dir.mkdir(parents=True, exist_ok=True)
    uploaded_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cloud sync
    cloud_sync = get_cloud_sync_from_env()

    if cloud_sync.provider == "local":
        print("ERROR: Cloud provider not configured!")
        print("Set environment variables:")
        print("  export CLOUD_PROVIDER=gcs")
        print("  export CLOUD_BUCKET=<your-bucket>")
        print("  export GOOGLE_APPLICATION_CREDENTIALS=<path-to-credentials.json>")
        sys.exit(1)

    # Find files to upload
    if args.files:
        # Specific files provided
        files_to_upload = []
        for f in args.files:
            path = Path(f)
            if path.exists():
                files_to_upload.append(path)
            elif (inbox_dir / f).exists():
                files_to_upload.append(inbox_dir / f)
            else:
                print(f"WARNING: File not found: {f}")
    else:
        # All .pth files in inbox
        files_to_upload = list(inbox_dir.glob("*.pth"))

    if not files_to_upload:
        print("No .pth files found to upload.")
        print(f"\nTo upload agents:")
        print(f"  1. Place .pth files in {inbox_dir}/")
        print(f"  2. Run: python upload_agents_to_gcp.py")
        print(f"\nOr specify files directly:")
        print(f"  python upload_agents_to_gcp.py /path/to/agent.pth")
        sys.exit(0)

    print(f"Found {len(files_to_upload)} file(s) to upload:")
    for f in files_to_upload:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print()

    if args.dry_run:
        print("DRY RUN - No files will be uploaded")
        print()
        for f in files_to_upload:
            cloud_path = f"{cloud_path_prefix}/{f.name}"
            print(f"Would upload: {f} -> gs://{cloud_sync.bucket_name}/{cloud_path}")
        sys.exit(0)

    # Upload each file
    successful = []
    failed = []

    for local_path in files_to_upload:
        filename = local_path.name
        cloud_path = f"{cloud_path_prefix}/{filename}"

        print(f"Uploading {filename}...")

        try:
            # Check if file already exists in cloud
            if cloud_sync.file_exists(cloud_path):
                print(f"  ⚠ Already exists in cloud, skipping: {filename}")
                # Move to uploaded anyway since it's already there
                dest = uploaded_dir / filename
                if local_path.parent != uploaded_dir:
                    shutil.move(str(local_path), str(dest))
                    print(f"  → Moved to {dest}")
                successful.append(filename)
                continue

            # Upload with verification
            success = cloud_sync.upload_file_verified(str(local_path), cloud_path)

            if success:
                print(f"  ✓ Uploaded and verified: {filename}")
                # Move to uploaded folder
                dest = uploaded_dir / filename
                if local_path.parent != uploaded_dir:
                    shutil.move(str(local_path), str(dest))
                    print(f"  → Moved to {dest}")
                successful.append(filename)
            else:
                print(f"  ✗ Upload verification failed: {filename}")
                failed.append(filename)

        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")
            failed.append(filename)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed uploads:")
        for f in failed:
            print(f"  - {f}")
        print(f"\nFailed files remain in {inbox_dir}/ for retry")

    if successful:
        print(f"\nSuccessfully uploaded files moved to {uploaded_dir}/")
        print(f"\nIMPORTANT: Uploaded agents are in cloud storage but NOT in global50.json!")
        print(f"To add them to the Global 50 leaderboard, you need to:")
        print(f"  1. Run gauntlet evaluation on these agents")
        print(f"  2. Use the promotion routine to add them to global50.json")

    print()


if __name__ == "__main__":
    main()
