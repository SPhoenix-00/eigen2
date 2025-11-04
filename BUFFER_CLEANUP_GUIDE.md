# Replay Buffer Cleanup Guide

## Why Clean Up the Old Buffer?

Since we disabled buffer saves to prevent RAM exhaustion, old buffer files on GCS are now stale and should be deleted to prevent them from being loaded on future resumes.

## What Gets Deleted

- `gs://eigen2-checkpoints-ase0/eigen2/checkpoints/replay_buffer.pkl`
- `gs://eigen2-checkpoints-ase0/eigen2/checkpoints/replay_buffer.pkl.gz`

These files contain experiences from generation 5 and earlier, which are no longer relevant since:
1. Buffer saves are now disabled
2. Loading old buffers causes agents to train on stale experiences
3. We want fresh buffers that fill from current agents

## How to Clean Up

### Option 1: Python Script (Recommended)

SSH into RunPod and run:

```bash
cd /workspace
python cleanup_old_buffer.py
```

**Output:**
```
============================================================
Cleaning up old replay buffer from GCS
============================================================
✓ Using credentials: /workspace/gcs-credentials.json
✓ Connected to GCS bucket: eigen2-checkpoints-ase0

Searching for old buffer files...

  Found: gs://eigen2-checkpoints-ase0/eigen2/checkpoints/replay_buffer.pkl
  Size: 12543.21 MB
  ✓ Deleted successfully

============================================================
Cleanup Summary:
  Files deleted: 1
  Files not found: 1
============================================================

✓ Old replay buffers removed from GCS
  Future resumes will start with fresh buffers
```

### Option 2: Bash Script

```bash
cd /workspace
chmod +x cleanup_old_buffer.sh
./cleanup_old_buffer.sh
```

### Option 3: Manual (Using gsutil)

```bash
# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# Check if buffer exists
gsutil ls gs://eigen2-checkpoints-ase0/eigen2/checkpoints/replay_buffer.pkl

# Delete it
gsutil rm gs://eigen2-checkpoints-ase0/eigen2/checkpoints/replay_buffer.pkl
gsutil rm gs://eigen2-checkpoints-ase0/eigen2/checkpoints/replay_buffer.pkl.gz
```

### Option 4: GCS Console (Web UI)

1. Go to: https://console.cloud.google.com/storage/browser/eigen2-checkpoints-ase0
2. Navigate to: `eigen2/checkpoints/`
3. Find `replay_buffer.pkl` and `replay_buffer.pkl.gz`
4. Select and delete them

## When to Run This

**Now:** Clean up the old buffer from generation 5

**Future:** You shouldn't need to run this again since buffer saves are disabled. But if you ever re-enable buffer saves and then disable them again, run this cleanup.

## What Happens After Cleanup

**Next Resume:**
- Code will look for `replay_buffer.pkl` on GCS
- Won't find it (deleted)
- Will start with empty buffer
- Buffer fills naturally from current generation agents
- Training continues normally with fresh, relevant experiences

**Current Run:**
- No impact - training is already running
- This cleanup is for future resumes only

## Verification

After running cleanup, verify deletion:

```bash
# Should return "not found" or similar
gsutil ls gs://eigen2-checkpoints-ase0/eigen2/checkpoints/replay_buffer.pkl
```

**Expected output:**
```
CommandException: One or more URLs matched no objects.
```

## Files Created

- [cleanup_old_buffer.py](cleanup_old_buffer.py) - Python cleanup script
- [cleanup_old_buffer.sh](cleanup_old_buffer.sh) - Bash cleanup script
- This guide

## Related Changes

- [training/erl_trainer.py](training/erl_trainer.py#L446-L460) - Updated to skip loading old buffers on resume
- [MEMORY_LEAK_FIX.md](MEMORY_LEAK_FIX.md) - Why buffer saves were disabled
- [BUFFER_MEMORY_FIX.md](BUFFER_MEMORY_FIX.md) - Original buffer save disable decision

## Summary

**Action:** Run `python cleanup_old_buffer.py` on RunPod to delete stale buffer from GCS

**Result:** Future resumes will start with fresh buffers instead of loading outdated generation 5 experiences

**Impact:** Better training quality on resumes, no stale experience distribution shift
