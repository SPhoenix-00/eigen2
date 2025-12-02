"""
Migrate Global 50 agent filenames from old format (with scores) to new format (without scores).

Old format: 5.23_historical-backfill_12.pth
New format: historical-backfill_12.pth

This script:
1. Loads global50.json
2. Scans workspace/global50/agents/ for files with old naming
3. Renames them to new format
4. Updates cloud storage
"""

import json
import re
from pathlib import Path
from utils.cloud_sync import get_cloud_sync_from_env

def migrate_filenames():
    """Migrate agent filenames from old format to new format."""

    # Paths
    global50_dir = Path("global50")
    agents_dir = global50_dir / "agents"
    archive_dir = global50_dir / "archive"
    json_path = global50_dir / "global50.json"

    if not json_path.exists():
        print("❌ No global50.json found. Nothing to migrate.")
        return

    # Load global50.json
    with open(json_path, 'r') as f:
        data = json.load(f)

    entries = data.get('entries', [])
    print(f"\n{'='*70}")
    print(f"Global 50 Filename Migration")
    print(f"{'='*70}")
    print(f"\nFound {len(entries)} agents in global50.json")

    # Initialize cloud sync
    cloud_sync = get_cloud_sync_from_env()
    cloud_base = f"{cloud_sync.project_name}/global50"

    # Build mapping of old->new filenames from global50.json entries
    # Old format: {score}_{run_name}_{agent_id}.pth
    # New format: {run_name}_{agent_id}.pth
    rename_map = {}
    for entry in entries:
        old_filename = f"{entry['gauntlet_score']:.2f}_{entry['run_name']}_{entry['agent_id']}.pth"
        new_filename = f"{entry['run_name']}_{entry['agent_id']}.pth"
        rename_map[old_filename] = new_filename

    print(f"\nBuilt rename map for {len(rename_map)} agents")

    # Pattern to match old filenames: [score]_[run_name]_[agent_id].pth
    old_pattern = re.compile(r'^-?\d+\.\d+_(.+)_(\d+)\.pth$')

    renamed_count = 0

    # Migrate agents/ directory
    print(f"\n{'='*70}")
    print("Migrating agents/ directory")
    print(f"{'='*70}")

    if agents_dir.exists():
        for old_file in agents_dir.glob("*.pth"):
            match = old_pattern.match(old_file.name)
            if match:
                run_name = match.group(1)
                agent_id = match.group(2)
                new_filename = f"{run_name}_{agent_id}.pth"
                new_path = agents_dir / new_filename

                if old_file.name != new_filename:
                    print(f"\n  Renaming: {old_file.name}")
                    print(f"        →   {new_filename}")

                    # Rename locally
                    old_file.rename(new_path)

                    # Upload with new name to cloud
                    cloud_path = f"{cloud_base}/agents/{new_filename}"
                    cloud_sync.upload_file(str(new_path), cloud_path, background=False)

                    renamed_count += 1

    print(f"\n  ✓ Renamed {renamed_count} agent files")

    # Migrate archive/ directory
    print(f"\n{'='*70}")
    print("Migrating archive/ directory")
    print(f"{'='*70}")

    archived_count = 0
    if archive_dir.exists():
        for old_file in archive_dir.glob("*.pth"):
            match = old_pattern.match(old_file.name)
            if match:
                run_name = match.group(1)
                agent_id = match.group(2)
                new_filename = f"{run_name}_{agent_id}.pth"
                new_path = archive_dir / new_filename

                if old_file.name != new_filename:
                    print(f"\n  Renaming: {old_file.name}")
                    print(f"        →   {new_filename}")

                    # Rename locally
                    old_file.rename(new_path)

                    # Upload with new name to cloud
                    cloud_path = f"{cloud_base}/archive/{new_filename}"
                    cloud_sync.upload_file(str(new_path), cloud_path, background=False)

                    archived_count += 1

        # Also rename JSON scoresheets
        old_json_pattern = re.compile(r'^-?\d+\.\d+_(.+)_(\d+)\.json$')
        for old_file in archive_dir.glob("*.json"):
            match = old_json_pattern.match(old_file.name)
            if match:
                run_name = match.group(1)
                agent_id = match.group(2)
                new_filename = f"{run_name}_{agent_id}.json"
                new_path = archive_dir / new_filename

                if old_file.name != new_filename:
                    old_file.rename(new_path)

                    cloud_path = f"{cloud_base}/archive/{new_filename}"
                    cloud_sync.upload_file(str(new_path), cloud_path, background=False)

    print(f"\n  ✓ Renamed {archived_count} archived files")

    print(f"\n{'='*70}")
    print("✓ Migration Complete!")
    print(f"{'='*70}")
    print(f"  Agents renamed:   {renamed_count}")
    print(f"  Archives renamed: {archived_count}")
    print(f"  Cloud mirror:     gs://{cloud_sync.bucket_name}/{cloud_base}/")
    print(f"\nYou can now run: python evaluate_for_global50.py --eval")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    migrate_filenames()
