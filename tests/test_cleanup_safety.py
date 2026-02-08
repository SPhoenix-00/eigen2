"""
Tests for cleanup_orphans safety logic.
Converted from root-level test_cleanup_safety.py to pytest format.
"""

from pathlib import Path
from utils.cleanup_orphans import identify_orphans


def test_path_mismatch_safety():
    """Filename-based comparison is safe even with path mismatches."""
    valid_filenames = {
        "transition_0.pkl.gz",
        "transition_1.pkl.gz",
        "transition_2.pkl.gz",
        "transition_100.pkl.gz",
        "transition_500.pkl.gz",
    }

    actual_files = {
        Path("checkpoints/run-123/buffer_storage/transition_0.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_1.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_2.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_100.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_500.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_999.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_1234.pkl.gz"),
    }

    orphaned_files, missing_filenames = identify_orphans(valid_filenames, actual_files)

    expected_orphans = {"transition_999.pkl.gz", "transition_1234.pkl.gz"}
    actual_orphan_names = {f.name for f in orphaned_files}
    assert actual_orphan_names == expected_orphans, "Only true orphans should be detected"


def test_all_paths_match():
    """No orphans when all paths match perfectly."""
    valid_filenames = {
        "transition_0.pkl.gz",
        "transition_1.pkl.gz",
        "transition_2.pkl.gz",
    }

    actual_files = {
        Path("buffer_storage/transition_0.pkl.gz"),
        Path("buffer_storage/transition_1.pkl.gz"),
        Path("buffer_storage/transition_2.pkl.gz"),
    }

    orphaned_files, missing_filenames = identify_orphans(valid_filenames, actual_files)
    assert len(orphaned_files) == 0
    assert len(missing_filenames) == 0


def test_missing_files_detection():
    """Correctly detects files referenced in metadata but missing from disk."""
    valid_filenames = {
        "transition_0.pkl.gz",
        "transition_1.pkl.gz",
        "transition_2.pkl.gz",
        "transition_100.pkl.gz",
        "transition_200.pkl.gz",
    }

    actual_files = {
        Path("buffer_storage/transition_0.pkl.gz"),
        Path("buffer_storage/transition_1.pkl.gz"),
        Path("buffer_storage/transition_2.pkl.gz"),
    }

    orphaned_files, missing_filenames = identify_orphans(valid_filenames, actual_files)

    expected_missing = {"transition_100.pkl.gz", "transition_200.pkl.gz"}
    assert missing_filenames == expected_missing
