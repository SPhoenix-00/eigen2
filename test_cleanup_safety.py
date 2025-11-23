"""
Test script to demonstrate the safety improvements in cleanup_orphans.py
Shows how filename-based comparison prevents catastrophic deletion from path mismatches.
"""

from pathlib import Path
from utils.cleanup_orphans import identify_orphans


def test_path_mismatch_safety():
    """
    Demonstrates that filename-based comparison is safe even with path mismatches.
    """
    print("="*70)
    print("TEST: Path Mismatch Safety")
    print("="*70)
    print("\nScenario: Metadata contains absolute paths, but we scan with relative paths")
    print("-"*70)

    # Simulate metadata with ABSOLUTE paths (like what might be saved)
    valid_filenames = {
        "transition_0.pkl.gz",
        "transition_1.pkl.gz",
        "transition_2.pkl.gz",
        "transition_100.pkl.gz",
        "transition_500.pkl.gz",
    }

    # Simulate actual files on disk with DIFFERENT paths (relative)
    actual_files = {
        Path("checkpoints/run-123/buffer_storage/transition_0.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_1.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_2.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_100.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_500.pkl.gz"),
        Path("checkpoints/run-123/buffer_storage/transition_999.pkl.gz"),  # Orphan!
        Path("checkpoints/run-123/buffer_storage/transition_1234.pkl.gz"),  # Orphan!
    }

    print(f"\nValid filenames in metadata: {len(valid_filenames)}")
    for fname in sorted(valid_filenames):
        print(f"  - {fname}")

    print(f"\nActual files on disk: {len(actual_files)}")
    for fpath in sorted(actual_files, key=lambda p: p.name):
        print(f"  - {fpath}")

    # Run the comparison
    orphaned_files, missing_filenames = identify_orphans(valid_filenames, actual_files)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n✓ Orphaned files detected: {len(orphaned_files)}")
    for fpath in sorted(orphaned_files, key=lambda p: p.name):
        print(f"  - {fpath.name} (would be DELETED)")

    print(f"\n✓ Missing files: {len(missing_filenames)}")
    if missing_filenames:
        for fname in sorted(missing_filenames):
            print(f"  - {fname}")
    else:
        print("  (none)")

    print(f"\n✓ Matched files (SAFE): {len(actual_files) - len(orphaned_files)}")
    for fpath in sorted(actual_files, key=lambda p: p.name):
        if fpath not in orphaned_files:
            print(f"  - {fpath.name} (PROTECTED)")

    # Verify safety
    print("\n" + "="*70)
    print("SAFETY VERIFICATION")
    print("="*70)

    expected_orphans = {"transition_999.pkl.gz", "transition_1234.pkl.gz"}
    actual_orphan_names = {f.name for f in orphaned_files}

    if actual_orphan_names == expected_orphans:
        print("\n✓ PASS: Only true orphans would be deleted!")
        print("✓ Valid files are PROTECTED from path mismatch!")
        success = True
    else:
        print("\n❌ FAIL: Wrong files would be deleted!")
        print(f"Expected orphans: {expected_orphans}")
        print(f"Detected orphans: {actual_orphan_names}")
        success = False

    return success


def test_all_paths_match():
    """
    Test case where all paths match perfectly.
    """
    print("\n\n" + "="*70)
    print("TEST: All Paths Match (Normal Case)")
    print("="*70)
    print("\nScenario: All files are valid, no orphans")
    print("-"*70)

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

    print(f"\nValid files: {len(valid_filenames)}")
    print(f"Actual files: {len(actual_files)}")
    print(f"Orphaned: {len(orphaned_files)}")
    print(f"Missing: {len(missing_filenames)}")

    if len(orphaned_files) == 0 and len(missing_filenames) == 0:
        print("\n✓ PASS: No false positives!")
        return True
    else:
        print("\n❌ FAIL: False positives detected!")
        return False


def test_missing_files():
    """
    Test case where metadata references files that don't exist on disk.
    """
    print("\n\n" + "="*70)
    print("TEST: Missing Files Detection")
    print("="*70)
    print("\nScenario: Metadata references files that were deleted externally")
    print("-"*70)

    valid_filenames = {
        "transition_0.pkl.gz",
        "transition_1.pkl.gz",
        "transition_2.pkl.gz",
        "transition_100.pkl.gz",  # Missing!
        "transition_200.pkl.gz",  # Missing!
    }

    actual_files = {
        Path("buffer_storage/transition_0.pkl.gz"),
        Path("buffer_storage/transition_1.pkl.gz"),
        Path("buffer_storage/transition_2.pkl.gz"),
    }

    orphaned_files, missing_filenames = identify_orphans(valid_filenames, actual_files)

    print(f"\nValid files in metadata: {len(valid_filenames)}")
    print(f"Actual files on disk: {len(actual_files)}")
    print(f"\nMissing files: {len(missing_filenames)}")
    for fname in sorted(missing_filenames):
        print(f"  ⚠ {fname}")

    expected_missing = {"transition_100.pkl.gz", "transition_200.pkl.gz"}
    if missing_filenames == expected_missing:
        print("\n✓ PASS: Correctly detected missing files!")
        return True
    else:
        print("\n❌ FAIL: Wrong missing files detected!")
        return False


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# CLEANUP ORPHANS - SAFETY TEST SUITE")
    print("#"*70)

    results = []

    # Run all tests
    results.append(("Path Mismatch Safety", test_path_mismatch_safety()))
    results.append(("All Paths Match", test_all_paths_match()))
    results.append(("Missing Files Detection", test_missing_files()))

    # Summary
    print("\n\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "#"*70)
    if all_passed:
        print("# ✓ ALL TESTS PASSED - CLEANUP IS SAFE TO USE")
    else:
        print("# ❌ SOME TESTS FAILED - DO NOT USE CLEANUP")
    print("#"*70 + "\n")
