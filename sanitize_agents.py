"""
Sanitize Global 50 Agent Files

This script validates and optionally removes corrupted agent files from
global50 directories. It attempts to load each .pth file and reports any
that fail validation.

Usage:
    # Dry run (report only, no deletions)
    python sanitize_agents.py

    # Actually delete corrupted files
    python sanitize_agents.py --delete

    # Specify custom directory
    python sanitize_agents.py --path workspace/global50
"""

import argparse
import torch
from pathlib import Path
from typing import List, Tuple, Dict


# Required keys for a valid agent checkpoint
REQUIRED_KEYS = [
    'agent_id',
    'actor_state_dict',
    'actor_target_state_dict',
    'critic_state_dict',
    'critic_target_state_dict',
    'actor_optimizer_state_dict',
    'critic_optimizer_state_dict',
    'noise_scale',
    'update_count'
]


def validate_agent_file(path: Path) -> Tuple[bool, str]:
    """
    Validate a single agent .pth file.

    Args:
        path: Path to the .pth file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to load the checkpoint
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Check it's a dictionary
        if not isinstance(checkpoint, dict):
            return False, f"Not a dictionary (got {type(checkpoint).__name__})"

        # Check for required keys
        missing_keys = [k for k in REQUIRED_KEYS if k not in checkpoint]
        if missing_keys:
            return False, f"Missing keys: {missing_keys}"

        # Check actor state dict has expected structure
        actor_state = checkpoint['actor_state_dict']
        if not isinstance(actor_state, dict) or len(actor_state) == 0:
            return False, "Empty or invalid actor_state_dict"

        # Check critic state dict has expected structure
        critic_state = checkpoint['critic_state_dict']
        if not isinstance(critic_state, dict) or len(critic_state) == 0:
            return False, "Empty or invalid critic_state_dict"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def find_agent_files(base_path: Path) -> List[Path]:
    """
    Find all .pth agent files in the given directory tree.

    Args:
        base_path: Root directory to search

    Returns:
        List of Path objects for all .pth files found
    """
    if not base_path.exists():
        return []

    return sorted(base_path.rglob("*.pth"))


def sanitize_agents(base_path: Path, delete: bool = False, verbose: bool = True) -> Dict:
    """
    Validate all agent files and optionally delete corrupted ones.

    Args:
        base_path: Root directory containing agent files
        delete: If True, delete corrupted files
        verbose: If True, print detailed progress

    Returns:
        Dictionary with sanitization statistics
    """
    print(f"\n{'='*70}")
    print("Agent File Sanitizer")
    print(f"{'='*70}")
    print(f"Base path: {base_path}")
    print(f"Mode: {'DELETE corrupted files' if delete else 'DRY RUN (report only)'}")
    print(f"{'='*70}\n")

    # Find all agent files
    agent_files = find_agent_files(base_path)

    if not agent_files:
        print("No .pth files found.")
        return {'total': 0, 'valid': 0, 'corrupted': 0, 'deleted': 0}

    print(f"Found {len(agent_files)} agent files to validate\n")

    valid_files = []
    corrupted_files = []
    deleted_count = 0

    for agent_file in agent_files:
        is_valid, message = validate_agent_file(agent_file)

        # Get relative path for cleaner output
        try:
            rel_path = agent_file.relative_to(base_path)
        except ValueError:
            rel_path = agent_file

        if is_valid:
            valid_files.append(agent_file)
            if verbose:
                print(f"  ✓ {rel_path}")
        else:
            corrupted_files.append((agent_file, message))
            print(f"  ✗ {rel_path}")
            print(f"    Error: {message}")

            if delete:
                try:
                    agent_file.unlink()
                    print(f"    → Deleted")
                    deleted_count += 1
                except Exception as e:
                    print(f"    → Failed to delete: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total files scanned: {len(agent_files)}")
    print(f"Valid files:         {len(valid_files)}")
    print(f"Corrupted files:     {len(corrupted_files)}")

    if delete:
        print(f"Files deleted:       {deleted_count}")
    elif corrupted_files:
        print(f"\nTo delete corrupted files, run with --delete flag")

    print(f"{'='*70}\n")

    # List corrupted files for easy reference
    if corrupted_files and not delete:
        print("Corrupted files:")
        for path, error in corrupted_files:
            print(f"  {path}")
        print()

    return {
        'total': len(agent_files),
        'valid': len(valid_files),
        'corrupted': len(corrupted_files),
        'deleted': deleted_count,
        'corrupted_files': [(str(p), e) for p, e in corrupted_files]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate and sanitize Global 50 agent files"
    )
    parser.add_argument(
        '--path',
        type=str,
        default='global50',
        help='Base path to scan for agent files (default: global50)'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Actually delete corrupted files (default: dry run)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show corrupted files and summary'
    )

    args = parser.parse_args()

    base_path = Path(args.path)

    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        print("\nTry one of these paths:")
        for candidate in ['global50', 'workspace/global50']:
            if Path(candidate).exists():
                print(f"  {candidate}")
        exit(1)

    result = sanitize_agents(
        base_path=base_path,
        delete=args.delete,
        verbose=not args.quiet
    )

    # Exit with error code if corrupted files found (and not deleted)
    if result['corrupted'] > 0 and not args.delete:
        exit(1)

    exit(0)


if __name__ == "__main__":
    main()
