# Cleanup Orphaned Replay Buffer Files

## Problem

When training crashes or is interrupted after writing replay buffer files but before the cleanup step completes, those files remain on disk as "orphans" or "zombies". The new instance of the trainer loads the metadata (which lists only valid files) and ignores these orphans. Over time, these invisible orphan files accumulate and consume disk space, potentially causing out-of-disk-space errors.

## Solution

The `cleanup_orphans.py` utility compares files physically on disk against the files the Trainer "knows about" (stored in metadata) and deletes the unknown orphaned files.

## Usage

### Option 1: Integrated with Resume (Recommended)

When resuming training, add the `--cleanup` flag to automatically clean up orphaned files before loading the checkpoint:

```bash
# Resume from last run with cleanup
python main.py --resume --cleanup

# Resume from specific run with cleanup
python main.py --resume-run azure-thunder-123 --cleanup

# Combine with other flags
python main.py --resume --cleanup --leverage
```

### Option 2: Standalone Cleanup

Run the cleanup utility independently without starting training:

```bash
# Clean up orphaned files for a specific run
python -m utils.cleanup_orphans --run-name azure-thunder-123

# Dry run (simulate without deleting)
python -m utils.cleanup_orphans --run-name azure-thunder-123 --dry-run

# Quiet mode (minimal output)
python -m utils.cleanup_orphans --run-name azure-thunder-123 --quiet
```

## How It Works

1. **Load Metadata**: Reads `checkpoints/{run_name}/replay_buffer.pkl` to get the list of valid transition files
2. **Scan Disk**: Scans `checkpoints/{run_name}/buffer_storage/` for all actual `transition_*.pkl.gz` files
3. **Compare**: Identifies files on disk that are not in the metadata (orphans)
4. **Delete**: Removes orphaned files and reports disk space freed
5. **Verify**: Reports any missing files (in metadata but not on disk)

## Example Output

```
============================================================
Replay Buffer Orphan Cleanup
============================================================
Run name: azure-thunder-123
Checkpoint dir: checkpoints/azure-thunder-123
Storage dir: checkpoints/azure-thunder-123/buffer_storage
============================================================

Step 1: Loading buffer metadata...
  ✓ Metadata loaded
  Buffer capacity: 1,000,000
  Total ever added: 5,234,567

Step 2: Extracting valid file list from metadata...
  ✓ Found 1,000,000 valid files in metadata

Step 3: Scanning buffer storage directory...
  ✓ Found 1,234,567 actual files on disk

Step 4: Comparing files...
  ✓ Orphaned files (on disk but not in metadata): 234,567
    Total size: 876.5 GB
  ✓ Valid files (matched): 1,000,000
    Total size: 3.7 TB

Step 5: Deleting orphaned files...
  ✓ Deleted 234,567 orphaned files
  ✓ Freed 876.5 GB of disk space

============================================================
Cleanup Summary
============================================================
Valid files: 1,000,000
Actual files on disk: 1,234,567
Orphaned files deleted: 234,567
Disk space freed: 876.5 GB
============================================================
```

## When to Use

- **Before resuming training** if you suspect disk space issues
- **After multiple crashes** during training
- **Periodically** during long training runs (every 50-100 generations)
- **When debugging** disk space problems

## Safety Features

1. **Dry Run Mode**: Test what would be deleted without actually deleting
2. **Metadata Validation**: Only deletes files not tracked in metadata
3. **Error Handling**: Continues even if some files fail to delete
4. **Detailed Reporting**: Shows exactly what files are orphaned and why
5. **Missing File Detection**: Warns if metadata references files that don't exist

## Technical Details

### File Structure

```
checkpoints/
└── {run_name}/
    ├── buffer_storage/              # Transition files
    │   ├── transition_0.pkl.gz
    │   ├── transition_1.pkl.gz
    │   ├── ...
    │   └── transition_N.pkl.gz
    └── replay_buffer.pkl            # Metadata (deque of valid paths)
```

### What Gets Deleted

A file is considered "orphaned" if:
- It exists in `buffer_storage/` directory
- It matches the pattern `transition_*.pkl.gz`
- Its path is **NOT** in the `buffer` deque stored in `replay_buffer.pkl`

### What Doesn't Get Deleted

- Files currently tracked in metadata
- Non-transition files (e.g., agent weights, trainer state)
- Files in other directories

## Common Scenarios

### Scenario 1: Training Crashed During Generation

```
Generation 45: Writing 200,000 transitions...
[CRASH] - Process killed due to OOM

Result: 200,000 transition files written but not added to metadata
Solution: Run cleanup before resuming
```

### Scenario 2: Circular Buffer Overflow Failed

```
Buffer at capacity, removing old file: transition_123.pkl.gz
[ERROR] OSError: Permission denied

Result: Old file not deleted, new file added to metadata
Solution: Cleanup will remove the old file
```

### Scenario 3: Parallel Worker Crash

```
Worker 15: Writing transitions to disk...
[CRASH] - Worker process terminated

Result: Partial writes, file IDs reserved but not all used
Solution: Cleanup removes incomplete/unused files
```

## Troubleshooting

### "Buffer metadata not found"

```bash
# Make sure the run name is correct
ls checkpoints/

# Check if the metadata file exists
ls checkpoints/{run_name}/replay_buffer.pkl
```

### "Checkpoint directory not found"

The run name might be incorrect. Check available runs:

```bash
ls checkpoints/
```

### Permission Errors

Some files may fail to delete due to permissions. The cleanup will report these but continue with other files.

## Integration with Cloud Sync

Note: The `buffer_storage/` directory is excluded from cloud sync (see `erl_trainer.py:1917`). This means:
- Orphaned files only affect local disk
- Cloud storage remains clean
- Cleanup only needs to run locally

## Performance

- **Metadata loading**: < 1 second
- **Directory scan**: ~1-5 seconds for 1M files
- **Deletion**: ~30-60 seconds for 100K files (depends on filesystem)

## Code Location

- **Utility**: `utils/cleanup_orphans.py`
- **Integration**: `main.py` (lines 187-207)
- **Replay Buffer**: `models/replay_buffer.py`
