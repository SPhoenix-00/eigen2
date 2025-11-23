"""
Cleanup Orphaned Replay Buffer Files

This utility finds and removes orphaned transition files from the replay buffer storage.
Orphaned files occur when training crashes after writing files but before updating metadata,
or when files are removed from the circular buffer but deletion failed.

Usage:
    python -m utils.cleanup_orphans --run-name <run_name>
    Or via main.py: python main.py --resume --cleanup
"""

import os
import gzip
import pickle
from pathlib import Path
from typing import Set, Tuple, Dict
import argparse

# Import for pickle unpacking - needed if metadata contains class instances
try:
    from models.replay_buffer import OnDiskReplayBuffer
except ImportError:
    pass  # If it's just a dict, we don't need it. If it's a class, pickle might need it.


def load_buffer_metadata(buffer_path: Path) -> Dict:
    """
    Load replay buffer metadata from the checkpoint file.

    Args:
        buffer_path: Path to the replay_buffer.pkl file

    Returns:
        Dictionary containing buffer metadata

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        Exception: If metadata cannot be loaded
    """
    if not buffer_path.exists():
        raise FileNotFoundError(f"Buffer metadata not found at: {buffer_path}")

    try:
        with gzip.open(buffer_path, 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except Exception as e:
        raise Exception(f"Failed to load buffer metadata: {e}")


def get_valid_files(metadata: Dict) -> Set[str]:
    """
    Extract set of valid FILENAMES (not full paths) from metadata.
    This prevents path mismatch issues between absolute/relative paths.

    Args:
        metadata: Buffer metadata dictionary

    Returns:
        Set of filenames (e.g., {'transition_0.pkl.gz', 'transition_1.pkl.gz', ...})
    """
    # The 'buffer' key contains a deque of file path strings
    buffer_deque = metadata.get('buffer', [])
    # Extract only the filename, ignore the directory path
    # This makes comparison robust to absolute vs relative path differences
    valid_filenames = {Path(file_path).name for file_path in buffer_deque}
    return valid_filenames


def get_actual_files(storage_dir: Path) -> Set[Path]:
    """
    Scan storage directory and get all actual transition files on disk.

    Args:
        storage_dir: Path to buffer_storage directory

    Returns:
        Set of Path objects for all transition files found
    """
    if not storage_dir.exists():
        return set()

    # Find all transition_*.pkl.gz files
    actual_files = set(storage_dir.glob("transition_*.pkl.gz"))
    return actual_files


def identify_orphans(valid_filenames: Set[str], actual_files: Set[Path]) -> Tuple[Set[Path], Set[str]]:
    """
    Compare valid filenames against actual files to find orphans and missing files.
    Uses filename-only comparison to avoid absolute/relative path mismatches.

    Args:
        valid_filenames: Set of filenames that SHOULD exist (from metadata)
        actual_files: Set of Path objects that actually exist on disk

    Returns:
        Tuple of (orphaned_files, missing_filenames)
        - orphaned_files: Path objects on disk but not in metadata
        - missing_filenames: Filenames in metadata but not on disk
    """
    orphaned = set()

    # Check every file on disk - if its name is not in valid set, it's orphaned
    for file_path in actual_files:
        if file_path.name not in valid_filenames:
            orphaned.add(file_path)

    # Calculate missing (inverse check) - filenames in metadata but not on disk
    actual_filenames = {f.name for f in actual_files}
    missing = valid_filenames - actual_filenames

    return orphaned, missing


def calculate_file_sizes(files: Set[Path]) -> int:
    """
    Calculate total size of files in bytes.

    Args:
        files: Set of file paths

    Returns:
        Total size in bytes
    """
    total_size = 0
    for file_path in files:
        try:
            if file_path.exists():
                total_size += file_path.stat().st_size
        except OSError:
            pass
    return total_size


def format_size(bytes_size: int) -> str:
    """
    Format bytes into human-readable size string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB", "250.3 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def delete_orphans(orphaned_files: Set[Path], dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Delete orphaned files from disk.

    Args:
        orphaned_files: Set of orphaned file paths to delete
        dry_run: If True, only simulate deletion without actually deleting

    Returns:
        Tuple of (deleted_count, failed_count, bytes_freed)
    """
    deleted_count = 0
    failed_count = 0
    bytes_freed = 0

    for file_path in orphaned_files:
        try:
            # Get size before deletion
            file_size = file_path.stat().st_size if file_path.exists() else 0

            if not dry_run:
                os.remove(file_path)

            deleted_count += 1
            bytes_freed += file_size
        except OSError as e:
            print(f"  ⚠ Failed to delete {file_path.name}: {e}")
            failed_count += 1

    return deleted_count, failed_count, bytes_freed


def cleanup_orphans(run_name: str, dry_run: bool = False, verbose: bool = True) -> Dict:
    """
    Main cleanup function to find and remove orphaned replay buffer files.

    Args:
        run_name: Name of the wandb run (checkpoint folder name)
        dry_run: If True, only report what would be deleted without deleting
        verbose: If True, print detailed progress information

    Returns:
        Dictionary with cleanup statistics

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist
    """
    # Setup paths
    checkpoint_dir = Path("checkpoints") / run_name
    buffer_metadata_path = checkpoint_dir / "replay_buffer.pkl"
    storage_dir = checkpoint_dir / "buffer_storage"

    # Validate checkpoint directory exists
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            f"Make sure the run name '{run_name}' is correct."
        )

    if verbose:
        print("\n" + "="*60)
        print("Replay Buffer Orphan Cleanup")
        print("="*60)
        print(f"Run name: {run_name}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Storage dir: {storage_dir}")
        if dry_run:
            print("Mode: DRY RUN (simulation only, no files will be deleted)")
        print("="*60 + "\n")

    # Step 1: Load metadata
    if verbose:
        print("Step 1: Loading buffer metadata...")

    try:
        metadata = load_buffer_metadata(buffer_metadata_path)
        if verbose:
            print(f"  ✓ Metadata loaded")
            print(f"  Buffer capacity: {metadata.get('capacity', 'unknown'):,}")
            print(f"  Total ever added: {metadata.get('total_added', 'unknown'):,}")
    except FileNotFoundError as e:
        if verbose:
            print(f"  ⚠ {e}")
            print("  No metadata file found. Cannot determine valid files.")
        return {
            'success': False,
            'error': 'metadata_not_found'
        }
    except Exception as e:
        if verbose:
            print(f"  ❌ Error loading metadata: {e}")
        return {
            'success': False,
            'error': str(e)
        }

    # Step 2: Get valid filenames from metadata
    if verbose:
        print("\nStep 2: Extracting valid file list from metadata...")

    valid_filenames = get_valid_files(metadata)
    if verbose:
        print(f"  ✓ Found {len(valid_filenames):,} valid files in metadata")

    # Step 3: Scan actual files on disk
    if verbose:
        print("\nStep 3: Scanning buffer storage directory...")

    actual_files = get_actual_files(storage_dir)
    if verbose:
        print(f"  ✓ Found {len(actual_files):,} actual files on disk")

    # Step 4: Identify orphans and missing files
    if verbose:
        print("\nStep 4: Comparing files (by filename only)...")

    orphaned_files, missing_filenames = identify_orphans(valid_filenames, actual_files)

    # Calculate sizes
    orphaned_size = calculate_file_sizes(orphaned_files)

    # Calculate valid (matched) files - files on disk whose names are in valid set
    actual_filenames = {f.name for f in actual_files}
    matched_filenames = valid_filenames & actual_filenames
    matched_files = {f for f in actual_files if f.name in matched_filenames}
    valid_size = calculate_file_sizes(matched_files)

    if verbose:
        print(f"  ✓ Orphaned files (on disk but not in metadata): {len(orphaned_files):,}")
        print(f"    Total size: {format_size(orphaned_size)}")

        if missing_filenames:
            print(f"  ⚠ Missing files (in metadata but not on disk): {len(missing_filenames):,}")
            if len(missing_filenames) <= 5:
                for missing_filename in list(missing_filenames)[:5]:
                    print(f"    - {missing_filename}")

        print(f"  ✓ Valid files (matched): {len(matched_files):,}")
        print(f"    Total size: {format_size(valid_size)}")

    # Step 5: Delete orphans (or simulate)
    if orphaned_files:
        if verbose:
            print(f"\nStep 5: {'Simulating' if dry_run else 'Deleting'} orphaned files...")

        deleted_count, failed_count, bytes_freed = delete_orphans(orphaned_files, dry_run=dry_run)

        if verbose:
            if dry_run:
                print(f"  ✓ Would delete {deleted_count:,} orphaned files")
                print(f"  ✓ Would free {format_size(bytes_freed)} of disk space")
            else:
                print(f"  ✓ Deleted {deleted_count:,} orphaned files")
                if failed_count > 0:
                    print(f"  ⚠ Failed to delete {failed_count:,} files")
                print(f"  ✓ Freed {format_size(bytes_freed)} of disk space")
    else:
        if verbose:
            print("\nStep 5: No orphaned files found!")
            print("  ✓ Buffer storage is clean")
        deleted_count = 0
        failed_count = 0
        bytes_freed = 0

    # Summary
    if verbose:
        print("\n" + "="*60)
        print("Cleanup Summary")
        print("="*60)
        print(f"Valid files: {len(valid_filenames):,}")
        print(f"Actual files on disk: {len(actual_files):,}")
        print(f"Orphaned files {'that would be' if dry_run else ''} deleted: {deleted_count:,}")
        if failed_count > 0:
            print(f"Failed deletions: {failed_count:,}")
        print(f"Disk space {'that would be' if dry_run else ''} freed: {format_size(bytes_freed)}")
        print("="*60 + "\n")

    return {
        'success': True,
        'valid_files_count': len(valid_filenames),
        'actual_files_count': len(actual_files),
        'orphaned_count': len(orphaned_files),
        'missing_count': len(missing_filenames),
        'deleted_count': deleted_count,
        'failed_count': failed_count,
        'bytes_freed': bytes_freed,
        'dry_run': dry_run
    }


def main():
    """Command-line interface for the cleanup utility."""
    parser = argparse.ArgumentParser(
        description="Clean up orphaned replay buffer files from crashed training runs."
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Name of the wandb run (checkpoint folder name)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate cleanup without actually deleting files'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (only show summary)'
    )

    args = parser.parse_args()

    try:
        result = cleanup_orphans(
            run_name=args.run_name,
            dry_run=args.dry_run,
            verbose=not args.quiet
        )

        if result['success']:
            exit(0)
        else:
            exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠ Cleanup interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n\n❌ Error during cleanup: {e}")
        raise


if __name__ == "__main__":
    main()
