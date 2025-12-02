"""
Quick script to download all Global 50 files from GCS.
Uses the existing cloud_sync infrastructure.
Supports context-window-specific leagues.
"""

from utils.cloud_sync import get_cloud_sync_from_env
from utils.config import Config
from pathlib import Path
import json
import argparse

def download_all_global50(context_window_days: int = None):
    """
    Download global50.json and all referenced agent files.

    Args:
        context_window_days: Context window in days (e.g., 151, 504).
                           If None, uses current Config.CONTEXT_WINDOW_DAYS.
    """

    print("Initializing cloud sync...")
    cloud_sync = get_cloud_sync_from_env()

    # Use current context window if not specified
    if context_window_days is None:
        context_window_days = Config.CONTEXT_WINDOW_DAYS
        print(f"Using current config context window: {context_window_days} days")
    else:
        print(f"Using specified context window: {context_window_days} days")

    # Context window identifier
    context_window_id = f"cw{context_window_days}"

    # Create local directory structure (context-window-specific)
    local_dir = Path("global50") / context_window_id
    local_agents_dir = local_dir / "agents"
    local_archive_dir = local_dir / "archive"

    local_dir.mkdir(parents=True, exist_ok=True)
    local_agents_dir.mkdir(exist_ok=True)
    local_archive_dir.mkdir(exist_ok=True)

    # Download global50.json for this context window
    print(f"\nDownloading global50.json for {context_window_days} days context window...")
    cloud_json_path = f"eigen2/global50/{context_window_id}/global50.json"
    local_json_path = local_dir / "global50.json"

    success = cloud_sync.download_file(cloud_json_path, str(local_json_path))

    if not success:
        print("✗ Failed to download global50.json")
        return

    print("✓ Downloaded global50.json")

    # Load the JSON to see what agents to download
    with open(local_json_path, 'r') as f:
        data = json.load(f)

    entries = data.get('entries', [])
    print(f"\nFound {len(entries)} agents in global50.json")

    if len(entries) == 0:
        print("No agents to download.")
        return

    # Download all agent files
    print(f"\nDownloading {len(entries)} agent files...")
    downloaded = 0
    failed = []

    for i, entry in enumerate(entries, 1):
        run_name = entry.get('run_name', 'unknown')
        agent_id = entry.get('agent_id', 0)
        filename = f"{run_name}_{agent_id}.pth"

        cloud_path = f"eigen2/global50/{context_window_id}/agents/{filename}"
        local_path = local_agents_dir / filename

        print(f"[{i}/{len(entries)}] Downloading {filename}...", end=" ")

        try:
            success = cloud_sync.download_file(cloud_path, str(local_path))
            if success:
                print("✓")
                downloaded += 1
            else:
                print("✗ Not found in cloud")
                failed.append(filename)
        except Exception as e:
            print(f"✗ Error: {e}")
            failed.append(filename)

    # Summary
    print(f"\n{'='*70}")
    print("Download Summary")
    print(f"{'='*70}")
    print(f"Total agents:     {len(entries)}")
    print(f"Downloaded:       {downloaded}")
    print(f"Failed:           {len(failed)}")

    if failed:
        print(f"\nFailed downloads:")
        for filename in failed:
            print(f"  - {filename}")
        print(f"\nThese agents don't exist in cloud storage.")
        print(f"You may need to re-evaluate them from checkpoint directories.")

    print(f"\n✓ Download complete!")
    print(f"Local directory: {local_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Global 50 agents from cloud storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using current config context window (151 days)
  python download_global50.py

  # Download from 504-day context window league
  python download_global50.py --context-window 504

  # Download from 252-day context window league
  python download_global50.py --context-window 252
        """
    )

    parser.add_argument(
        '--context-window',
        type=int,
        default=None,
        metavar='DAYS',
        help='Context window in days (e.g., 151, 504, 252). If not specified, uses current Config.CONTEXT_WINDOW_DAYS'
    )

    args = parser.parse_args()

    download_all_global50(context_window_days=args.context_window)
