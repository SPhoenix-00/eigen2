# Deployment Instructions - Memory Debugging Version

## Summary of Changes

We've added comprehensive memory profiling to identify exactly what's consuming RAM. Additionally, upload messages have been reduced to one per batch instead of per-file spam.

### What's New

1. **Memory Profiling System** - Tracks RAM, GPU, and object counts at every critical point
2. **Automatic Snapshots** - After evaluate, train, evolve phases
3. **Trend Analysis** - Shows memory growth across generations
4. **Object Counting** - Identifies which types are accumulating
5. **Large Object Detection** - Finds memory hogs >= 50MB
6. **Cleaner Upload Messages** - One message per batch instead of 18 individual messages

## Files Modified

1. **[utils/memory_profiler.py](utils/memory_profiler.py)** - NEW: Complete memory profiling system
2. **[training/erl_trainer.py](training/erl_trainer.py)** - Added memory tracking at 6 key points
3. **[utils/cloud_sync.py](utils/cloud_sync.py)** - Reduced per-file prints to batch summaries
4. **[MEMORY_DEBUG_GUIDE.md](MEMORY_DEBUG_GUIDE.md)** - NEW: Complete guide to interpreting output

## Deployment Steps

### 1. Install psutil (Required)

```bash
# SSH into RunPod
pip install psutil
```

### 2. Pull Latest Changes

```bash
cd /workspace
git pull origin main
```

### 3. Verify Files

```bash
# Check memory profiler exists
ls -lh utils/memory_profiler.py
# Should show: ~15KB file

# Check trainer has memory tracking
grep "log_memory" training/erl_trainer.py | wc -l
# Should show: at least 5 lines
```

### 4. Resume Training

```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

python main.py --resume
```

## What You'll See

### Instead of This (Old - Spammy):

```
⏳ Queued for upload: checkpoints/best_agent.pth → eigen2/checkpoints/best_agent.pth
⏳ Queued for upload: checkpoints/trainer_state.json → eigen2/checkpoints/trainer_state.json
⏳ Queued for upload: checkpoints/population/agent_0.pth → eigen2/checkpoints/population/agent_0.pth
⏳ Queued for upload: checkpoints/population/agent_1.pth → eigen2/checkpoints/population/agent_1.pth
[... 14 more lines ...]
✓ Uploaded: checkpoints/best_agent.pth → eigen2/checkpoints/best_agent.pth
✓ Uploaded: checkpoints/trainer_state.json → eigen2/checkpoints/trainer_state.json
[... 14 more lines ...]
```

### You'll See This (New - Clean):

```
============================================================
Queueing checkpoints for background upload...
============================================================
⏳ 18 files queued for checkpoints
```

### Memory Profiling Output

After each phase, you'll see:

```
======================================================================
🔍 MEMORY SNAPSHOT: Gen 1: After evaluate_population
======================================================================
💾 Process Memory:
   RSS (actual RAM):      8.52 GB
   VMS (virtual):        18.34 GB
🎮 GPU Memory:
   Allocated:             2.34 GB
   Reserved:              3.00 GB
📊 Object Counts:
   TradingEnvironment:      2  <- MUST STAY AT 2!
   DDPGAgent:              16
   ReplayBuffer:            1
   Lists:               15678
   Dicts:               10234
   NumPy arrays:         2456
   PyTorch Tensors:       891
======================================================================
```

At end of each generation:

```
======================================================================
📊 MEMORY TREND ACROSS GENERATIONS
======================================================================
Label                          RAM (GB)       Change
------------------------------ ------------ ------------
Trainer initialized (baseline)         4.23     baseline
Gen 1: After evaluate_population       8.52    +4.29 GB
Gen 1: After train_population          9.12    +0.60 GB
Gen 1: After evolve_population         9.45    +0.33 GB
======================================================================

======================================================================
📈 MEMORY GROWTH ANALYSIS
======================================================================
Baseline: Trainer initialized (baseline)
Current:  Gen 1: After evolve_population

RAM Usage:
   Baseline:      4.23 GB
   Current:       9.45 GB
   Growth:       +5.22 GB (+123.4%)

Object Count Changes:
   TradingEnvironment :      2 →      2 (+0)  <- CRITICAL!
   DDPGAgent          :     16 →     17 (+1)  <- GOOD!
   ReplayBuffer       :      1 →      1 (+0)
   ndarray            :   1234 →   3456 (+2222)
   Tensor             :    567 →    891 (+324)
======================================================================
```

## What to Look For

### ✅ Healthy Pattern (No Leak)

```
Gen 1: After evaluate_population    8.52 GB
Gen 2: After evaluate_population   10.23 GB  (+1.71 GB)  <- Replay buffer
Gen 3: After evaluate_population   11.89 GB  (+1.66 GB)  <- Replay buffer
Gen 4: After evaluate_population   13.45 GB  (+1.56 GB)  <- Slowing
Gen 5: After evaluate_population   14.23 GB  (+0.78 GB)  <- Stable
Gen 6: After evaluate_population   14.89 GB  (+0.66 GB)
Gen 7: After evaluate_population   15.12 GB  (+0.23 GB)  <- Very stable

Object Count Changes:
   TradingEnvironment :  2 →  2 (+0)  <- PERFECT!
   DDPGAgent          : 16 → 17 (+1)  <- PERFECT!
```

### ❌ Leak Pattern (Problem!)

```
Gen 1: After evaluate_population    8.52 GB
Gen 2: After evaluate_population   15.34 GB  (+6.82 GB!) <- TOO MUCH!
Gen 3: After evaluate_population   22.67 GB  (+7.33 GB!) <- LEAK!

Object Count Changes:
   TradingEnvironment :  2 → 18 (+16)  <- PROBLEM!
```

## Monitoring

### Quick Check (Every 5 Minutes)

```bash
# In separate terminal
watch -n 10 'free -h | grep Mem | awk "{print \"RAM: \", \$3, \"/\", \$2}"'
```

Expected output:
```
RAM:  34.2G / 175.0G    # Should stay 30-40GB range
```

### Full Monitoring Script

Create `monitor.sh`:

```bash
#!/bin/bash
while true; do
    echo "===================="
    date
    free -h | grep Mem
    if nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    fi
    python -c "import gc; from environment.trading_env import TradingEnvironment; envs=[o for o in gc.get_objects() if isinstance(o, TradingEnvironment)]; print(f'Environments: {len(envs)}')" 2>/dev/null || echo "Can't count envs"
    sleep 60
done
```

Run it:
```bash
chmod +x monitor.sh
./monitor.sh | tee monitor.log &
```

## What to Report

If you see a leak, report this:

1. **Which phase** shows the jump:
   - "After evaluate_population" → Environment leak
   - "After train_population" → Training loop leak
   - "After evolve_population" → Population/agent leak

2. **RAM growth rate:**
   - "Gen X to Gen Y: +7 GB per generation"

3. **Object count changes:**
   - "TradingEnvironment: 2 → 18 (+16 each generation)"
   - "DDPGAgent: 16 → 32 (+16 each generation)"

4. **Full snapshot** from that phase (copy/paste the box)

5. **Generation number** when it started

Example report:
```
LEAK DETECTED:
- Phase: Gen 2: After evaluate_population
- Growth: +7.2 GB per generation (consistent)
- Object: TradingEnvironment count increasing: 2 → 18 → 34
- Started: Generation 1
- Pattern: Adds 16 TradingEnvironment per generation

Full snapshot:
[paste the snapshot here]
```

## Expected Behavior

### Generation 1-5 (Buffer Filling)

- RAM: +1-2 GB per generation
- TradingEnvironment: stays at 2
- DDPGAgent: stays at 16-17
- Growth: Normal (replay buffer)

### Generation 6-10 (Buffer Stabilizing)

- RAM: +0.5-1 GB per generation
- Object counts: stable
- Growth: Slowing down

### Generation 11-25 (Stable)

- RAM: +0-0.5 GB per generation
- Object counts: stable
- Growth: Minimal (just overhead)

### Total by Generation 25

- RAM: 30-40 GB (stable)
- TradingEnvironment: 2 (always)
- DDPGAgent: 17 (16 population + 1 best)
- ReplayBuffer: 1 (always)

## Troubleshooting

### "ModuleNotFoundError: No module named 'psutil'"

```bash
pip install psutil
```

### "Too much output, can't read logs"

Redirect to file:
```bash
python main.py --resume 2>&1 | tee training.log
```

Then tail just the memory snapshots:
```bash
grep -A 15 "MEMORY SNAPSHOT" training.log | tail -100
```

### "Want less verbose tracking"

Edit line 688 in `erl_trainer.py`:
```python
# Change this:
if (gen + 1) % 1 == 0:  # Every generation

# To this:
if (gen + 1) % 5 == 0:  # Every 5 generations
```

## Success Criteria

Training is working correctly if:

- ✅ RAM stays below 50 GB throughout
- ✅ TradingEnvironment count stays at 2
- ✅ DDPGAgent count stays at 16-17
- ✅ Memory growth decreases over time
- ✅ Completes all 25 generations

## Quick Start

```bash
# 1. Install dependency
pip install psutil

# 2. Pull code
cd /workspace && git pull origin main

# 3. Resume training
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
python main.py --resume

# 4. Monitor in separate terminal
watch -n 30 'free -h | grep Mem'
```

## Documentation

- **[MEMORY_DEBUG_GUIDE.md](MEMORY_DEBUG_GUIDE.md)** - Complete guide to interpreting memory output
- **[CRITICAL_FIX_ENVIRONMENT_REUSE.md](CRITICAL_FIX_ENVIRONMENT_REUSE.md)** - Details on environment reuse fix
- **[MEMORY_LEAK_FIXES_COMPREHENSIVE.md](MEMORY_LEAK_FIXES_COMPREHENSIVE.md)** - All previous memory fixes

---

**Status:** Ready for deployment
**Expected Result:** Clear visibility into memory usage, definitively identify any remaining leaks
**Impact:** Cleaner logs, comprehensive debugging data
