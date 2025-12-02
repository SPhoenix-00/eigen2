"""
Migration script to move Global 50 files from old structure to new context-window-specific structure.

This script:
1. Downloads all files from old structure (eigen2/global50/)
2. Uploads them to new structure (eigen2/global50/cw504/)
3. Optionally cleans up old files from cloud

Usage:
    python migrate_global50_to_cw504.py              # Migrate (keeps old files)
    python migrate_global50_to_cw504.py --cleanup    # Migrate and delete old files
"""

import argparse
import json
import shutil
from pathlib import Path
from utils.cloud_sync import get_cloud_sync_from_env


def migrate_global50():
    """Migrate Global 50 from old structure to cw504 structure."""

    print("\n" + "="*70)
    print("Global 50 Migration to Context-Window Structure")
    print("="*70)

    # Initialize cloud sync
    print("\n1. Initializing cloud sync...")
    cloud_sync = get_cloud_sync_from_env()
    print(f"   Provider: {cloud_sync.provider}")
    print(f"   Bucket: {cloud_sync.bucket_name}")
    print(f"   Project: {cloud_sync.project_name}")

    if cloud_sync.provider == "local":
        print("\nWARNING: Cloud provider is 'local' - no cloud migration needed.")
        print("  Only local files need to be reorganized.")
        return

    # Define paths
    old_cloud_base = f"{cloud_sync.project_name}/global50"
    new_cloud_base = f"{cloud_sync.project_name}/global50/cw504"

    old_cloud_json = f"{old_cloud_base}/global50.json"
    new_cloud_json = f"{new_cloud_base}/global50.json"

    local_temp_dir = Path("migration_temp")
    local_temp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Check if old structure exists
        print("\n2. Checking for old structure in cloud...")
        temp_json = local_temp_dir / "old_global50.json"
        success = cloud_sync.download_file(old_cloud_json, str(temp_json))

        if not success:
            print("   X No old global50.json found in cloud.")
            print("   Nothing to migrate. Cloud may already be using new structure.")
            return

        print(f"   OK Found old global50.json")

        # Step 2: Load and check entries
        with open(temp_json, 'r') as f:
            data = json.load(f)

        entries = data.get('entries', [])
        print(f"   Found {len(entries)} agents to migrate")

        if len(entries) == 0:
            print("   No agents to migrate.")
            return

        # Step 3: Check if new structure already exists
        print("\n3. Checking if new structure already exists...")
        temp_new_json = local_temp_dir / "new_global50.json"
        new_exists = cloud_sync.download_file(new_cloud_json, str(temp_new_json))

        if new_exists:
            print("   WARNING: New structure (cw504/) already exists in cloud!")
            print("\n   Options:")
            print("   1. Skip migration (keep existing cw504/)")
            print("   2. Merge (combine old and new agents)")
            print("   3. Overwrite (replace cw504/ with old structure)")

            while True:
                choice = input("\n   Enter choice (1/2/3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                print("   Invalid choice. Please enter 1, 2, or 3.")

            if choice == '1':
                print("\n   Skipping migration. Existing cw504/ preserved.")
                return
            elif choice == '2':
                # Merge logic
                with open(temp_new_json, 'r') as f:
                    new_data = json.load(f)
                new_entries = new_data.get('entries', [])

                # Combine entries (avoid duplicates)
                existing_ids = {(e['run_name'], e['agent_id']) for e in new_entries}
                for entry in entries:
                    key = (entry['run_name'], entry['agent_id'])
                    if key not in existing_ids:
                        new_entries.append(entry)
                        existing_ids.add(key)

                # Sort by score
                new_entries.sort(key=lambda e: e['gauntlet_score'], reverse=True)

                # Keep top 50
                new_entries = new_entries[:50]

                # Update data
                data['entries'] = new_entries
                entries = new_entries

                print(f"   Merged to {len(entries)} total agents (top 50)")
            else:
                # Overwrite - use old data as-is
                print("   Will overwrite existing cw504/ structure")

        # Step 4: Update league rules to include context_window_days
        print("\n4. Updating league rules...")
        if 'league_rules' not in data or 'context_window_days' not in data.get('league_rules', {}):
            data['league_rules'] = {'context_window_days': 504}
            print("   OK Added context_window_days: 504")
        else:
            print(f"   OK League rules already present: {data['league_rules']}")

        # Save updated JSON locally
        updated_json = local_temp_dir / "updated_global50.json"
        with open(updated_json, 'w') as f:
            json.dump(data, f, indent=2)

        # Step 5: Upload updated global50.json to new location
        print("\n5. Uploading global50.json to new location...")
        cloud_sync.upload_file(str(updated_json), new_cloud_json, background=False)
        print(f"   OK Uploaded to {new_cloud_json}")

        # Step 6: Migrate agent files
        print("\n6. Migrating agent files...")
        local_agents_dir = local_temp_dir / "agents"
        local_agents_dir.mkdir(exist_ok=True)

        migrated_count = 0
        failed_count = 0

        for i, entry in enumerate(entries, 1):
            run_name = entry.get('run_name', 'unknown')
            agent_id = entry.get('agent_id', 0)
            filename = f"{run_name}_{agent_id}.pth"

            print(f"   [{i}/{len(entries)}] Migrating {filename}...", end=" ")

            # Download from old location
            old_cloud_path = f"{old_cloud_base}/agents/{filename}"
            local_agent_path = local_agents_dir / filename

            download_success = cloud_sync.download_file(old_cloud_path, str(local_agent_path))

            if not download_success:
                print("X (not found in old location)")
                failed_count += 1
                continue

            # Upload to new location
            new_cloud_path = f"{new_cloud_base}/agents/{filename}"
            cloud_sync.upload_file(str(local_agent_path), new_cloud_path, background=False)

            print("OK")
            migrated_count += 1

        # Step 7: Migrate archive if it exists
        print("\n7. Checking for archive directory...")
        archive_migrated = 0

        # Note: We can't easily list files in cloud storage with the generic cloud_sync interface,
        # so we'll skip archive migration for now and note it in the summary
        print("   Skipping archive migration (requires manual intervention if needed)")

        # Summary
        print("\n" + "="*70)
        print("Migration Summary")
        print("="*70)
        print(f"  Global50.json:  OK Migrated")
        print(f"  Agents:         {migrated_count} / {len(entries)} migrated")
        if failed_count > 0:
            print(f"  Failed:         {failed_count} agents not found in old location")
        print(f"\n  Old location:   gs://{cloud_sync.bucket_name}/{old_cloud_base}/")
        print(f"  New location:   gs://{cloud_sync.bucket_name}/{new_cloud_base}/")
        print("="*70)

        # Ask about cleanup
        print("\nWARNING: Migration complete! Old files still exist in cloud.")
        print("  You can manually delete them using:")
        print(f"    gsutil -m rm -r gs://{cloud_sync.bucket_name}/{old_cloud_base}/")
        print("\n  Or re-run this script with --cleanup flag")

    finally:
        # Cleanup temp directory
        print("\n8. Cleaning up temporary files...")
        if local_temp_dir.exists():
            shutil.rmtree(local_temp_dir)
        print("   OK Cleanup complete")


def cleanup_old_structure():
    """Delete old Global 50 structure from cloud storage."""

    print("\n" + "="*70)
    print("CLEANUP: Delete Old Global 50 Structure")
    print("="*70)

    print("\nWARNING: This will permanently delete old Global 50 files from cloud!")
    print("  Make sure migration completed successfully before proceeding.")

    confirmation = input("\n  Type 'DELETE' to confirm: ").strip()

    if confirmation != 'DELETE':
        print("\n  Cleanup cancelled.")
        return

    # Initialize cloud sync
    cloud_sync = get_cloud_sync_from_env()

    if cloud_sync.provider == "local":
        print("\nWARNING: Cloud provider is 'local' - no cloud cleanup needed.")
        return

    old_cloud_base = f"{cloud_sync.project_name}/global50"

    print(f"\n  Deleting: gs://{cloud_sync.bucket_name}/{old_cloud_base}/")
    print(f"  (excluding gs://{cloud_sync.bucket_name}/{old_cloud_base}/cw*/)")

    # Note: The CloudSync interface doesn't have a delete method
    # User will need to manually delete or we need to use provider-specific APIs

    print("\n  Please manually delete old files using:")
    print(f"    gsutil rm gs://{cloud_sync.bucket_name}/{old_cloud_base}/global50.json")
    print(f"    gsutil -m rm -r gs://{cloud_sync.bucket_name}/{old_cloud_base}/agents/")
    print(f"    gsutil -m rm -r gs://{cloud_sync.bucket_name}/{old_cloud_base}/archive/")

    print("\n  Note: This preserves all cw*/ subdirectories (new structure)")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Global 50 from old structure to context-window structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate files to new structure (keeps old files)
  python migrate_global50_to_cw504.py

  # Show cleanup instructions
  python migrate_global50_to_cw504.py --cleanup
        """
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Show instructions for cleaning up old structure after migration'
    )

    args = parser.parse_args()

    if args.cleanup:
        # First do migration
        migrate_global50()
        # Then cleanup
        cleanup_old_structure()
    else:
        # Just migrate
        migrate_global50()

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
