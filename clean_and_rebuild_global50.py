"""
Clean Global 50 agent files and rebuild with correct naming.

This script:
1. Backs up global50.json
2. Deletes all .pth files from workspace/global50/agents/
3. Keeps the JSON metadata intact
4. You then run --eval to re-evaluate and save with correct filenames
"""

import json
import shutil
from pathlib import Path

def clean_global50():
    """Clean agent files from Global 50 directories."""

    # Paths
    global50_dir = Path("global50")
    agents_dir = global50_dir / "agents"
    json_path = global50_dir / "global50.json"
    backup_path = global50_dir / "global50.json.backup"

    if not json_path.exists():
        print("❌ No global50.json found. Nothing to clean.")
        return

    print(f"\n{'='*70}")
    print("Clean Global 50 Agent Files")
    print(f"{'='*70}")

    # Backup global50.json
    shutil.copy(json_path, backup_path)
    print(f"\n✓ Backed up global50.json to {backup_path}")

    # Count and delete agent files
    deleted_count = 0
    if agents_dir.exists():
        agent_files = list(agents_dir.glob("*.pth"))
        print(f"\nFound {len(agent_files)} agent files to delete")

        for agent_file in agent_files:
            print(f"  Deleting: {agent_file.name}")
            agent_file.unlink()
            deleted_count += 1

    print(f"\n{'='*70}")
    print("✓ Cleanup Complete!")
    print(f"{'='*70}")
    print(f"  Deleted:     {deleted_count} agent files")
    print(f"  Preserved:   global50.json (backed up)")
    print(f"  Backup:      {backup_path}")

    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Run: python evaluate_for_global50.py --eval")
    print("2. This will re-download agents from cloud with OLD names")
    print("3. Re-evaluate them and save with NEW names")
    print("4. Upload to cloud with NEW names")
    print("\nResult: Cloud will have both old and new files.")
    print("You'll need to manually delete old files from GCS console.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    clean_global50()
