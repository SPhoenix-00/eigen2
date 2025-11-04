# Memory Debugging Guide

## Overview

We've added comprehensive memory profiling to identify exactly what's consuming RAM and what's not being cleaned up properly. This will help us pinpoint the real memory leak.

## What Was Added

### 1. Memory Profiling Utility ([utils/memory_profiler.py](utils/memory_profiler.py))

A complete memory profiling system that tracks:
- **Process Memory (RAM):** RSS (actual RAM used) and VMS (virtual memory)
- **GPU Memory:** Allocated and reserved CUDA memory
- **Object Counts:** Live instances of key types (TradingEnvironment, DDPGAgent, ReplayBuffer, lists, dicts, arrays, tensors)
- **Large Objects:** Top 20 objects consuming >= 10MB each
- **Memory Trends:** Growth across generations
- **Memory Snapshots:** Detailed point-in-time captures

### 2. Automatic Tracking Points

Memory is tracked automatically at these critical points:

1. **Baseline:** After trainer initialization (line 135)
2. **After Evaluation:** After `evaluate_population()` completes (line 563)
3. **After Training:** After `train_population()` completes (line 626)
4. **After Evolution:** After `evolve_population()` completes (line 632)
5. **End of Generation:** Trend and growth analysis (lines 688-691)
6. **End of Training:** Final comprehensive summary (lines 711-716)

---

## What You'll See During Training

### At Startup

```
🔍 Taking baseline memory snapshot...

======================================================================
🔍 MEMORY SNAPSHOT: Trainer initialized (baseline)
======================================================================
💾 Process Memory:
   RSS (actual RAM):      4.23 GB
   VMS (virtual):        12.45 GB
🎮 GPU Memory:
   Allocated:             0.15 GB
   Reserved:              0.50 GB
📊 Object Counts:
   TradingEnvironment:      2  <- Should be exactly 2 (eval_env + val_env)
   DDPGAgent:              16  <- Should be 16 (population)
   ReplayBuffer:            1  <- Should be 1
   Lists:               12345
   Dicts:                8901
   NumPy arrays:         1234
   PyTorch Tensors:       567
======================================================================
```

**What to look for:**
- ✅ **TradingEnvironment: 2** (eval_env + val_env)
- ✅ **DDPGAgent: 16** (initial population)
- ✅ **ReplayBuffer: 1**
- ✅ **RAM: 4-6 GB** is normal after initialization

---

### After Each evaluate_population()

```
======================================================================
🔍 MEMORY SNAPSHOT: Gen 1: After evaluate_population
======================================================================
💾 Process Memory:
   RSS (actual RAM):      8.52 GB  <- Should be ~6-8 GB after first eval
   VMS (virtual):        18.34 GB
🎮 GPU Memory:
   Allocated:             2.34 GB
   Reserved:              3.00 GB
📊 Object Counts:
   TradingEnvironment:      2  <- MUST stay at 2 (if not, we have a leak!)
   DDPGAgent:              16  <- Should stay at 16
   ReplayBuffer:            1
   Lists:               15678
   Dicts:               10234
   NumPy arrays:         2456
   PyTorch Tensors:       891
======================================================================
```

**What to look for:**
- ✅ **TradingEnvironment: 2** (still 2 - environments being reused correctly)
- ❌ **TradingEnvironment: 18** (LEAK! Creating new environments)
- ✅ **DDPGAgent: 16** (same count)
- ✅ **RAM growth: ~2-4 GB** (replay buffer filling with experiences)

---

### After Each train_population()

```
======================================================================
🔍 MEMORY SNAPSHOT: Gen 1: After train_population
======================================================================
💾 Process Memory:
   RSS (actual RAM):      9.12 GB  <- Small increase from DDPG training
   VMS (virtual):        19.23 GB
📊 Object Counts:
   TradingEnvironment:      2  <- Still 2
   DDPGAgent:              16  <- Still 16
   PyTorch Tensors:      1234  <- May increase during training
======================================================================
```

**What to look for:**
- ✅ **Minimal RAM increase** (~500MB - 1GB for gradients/optimizers)
- ❌ **Large RAM increase** (5GB+ would indicate leak in training loop)
- ✅ **Tensor count returns to baseline** after training completes

---

### After Each evolve_population()

```
======================================================================
🔍 MEMORY SNAPSHOT: Gen 1: After evolve_population
======================================================================
💾 Process Memory:
   RSS (actual RAM):      9.45 GB  <- Should be similar to after training
   VMS (virtual):        19.50 GB
📊 Object Counts:
   TradingEnvironment:      2  <- Still 2
   DDPGAgent:              17  <- Now 17 (16 population + 1 best_agent)
   ReplayBuffer:            1
======================================================================
```

**What to look for:**
- ✅ **DDPGAgent: 17** (16 population + 1 best_agent)
- ❌ **DDPGAgent: 30+** (LEAK! Old agents not being deleted)
- ✅ **RAM stable or small increase** (~200-500MB for new agent networks)

---

### End of Generation: Trend Analysis

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
Gen 2: After evaluate_population      10.23    +0.78 GB  <- Should be small
Gen 2: After train_population         10.89    +0.66 GB
Gen 2: After evolve_population        11.12    +0.23 GB
======================================================================
```

**What to look for:**
- ✅ **Small increases** (0.5-1.5 GB per generation = replay buffer growing)
- ❌ **Large increases** (5GB+ per generation = LEAK!)
- ✅ **Stabilizes** after buffer fills (gen 5-7)
- ❌ **Continues growing** linearly (indicates leak)

---

### End of Generation: Growth Analysis

```
======================================================================
📈 MEMORY GROWTH ANALYSIS
======================================================================
Baseline: Trainer initialized (baseline)
Current:  Gen 2: After evolve_population

RAM Usage:
   Baseline:      4.23 GB
   Current:      11.12 GB
   Growth:       +6.89 GB (+162.9%)

Object Count Changes:
   TradingEnvironment :      2 →      2 (+0)  <- GOOD!
   DDPGAgent          :     16 →     17 (+1)  <- GOOD!
   ReplayBuffer       :      1 →      1 (+0)  <- GOOD!
   ndarray            :   1234 →   3456 (+2222)  <- Expected (buffer)
   Tensor             :    567 →    891 (+324)    <- Expected (training)
======================================================================
```

**What to look for:**
- ✅ **TradingEnvironment: +0** (environments reused correctly)
- ❌ **TradingEnvironment: +30** (LEAK! Creating new ones each eval)
- ✅ **DDPGAgent: +0 or +1** (only best_agent added)
- ❌ **DDPGAgent: +16** (LEAK! Old population not deleted)
- ✅ **ndarray increasing** (replay buffer filling - expected)

---

### End of Training: Comprehensive Summary

```
======================================================================
🔍 FINAL MEMORY ANALYSIS
======================================================================

======================================================================
📊 MEMORY TREND ACROSS GENERATIONS
======================================================================
Label                          RAM (GB)       Change
------------------------------ ------------ ------------
Trainer initialized (baseline)         4.23     baseline
Gen 1: After evaluate_population       8.52    +4.29 GB
...
Gen 25: After evolve_population       32.45    +0.12 GB  <- Should be stable
======================================================================

======================================================================
📈 MEMORY GROWTH ANALYSIS
======================================================================
Baseline: Trainer initialized (baseline)
Current:  Gen 25: After evolve_population

RAM Usage:
   Baseline:      4.23 GB
   Current:      32.45 GB
   Growth:      +28.22 GB (+667.1%)

Object Count Changes:
   TradingEnvironment :      2 →      2 (+0)   <- PERFECT!
   DDPGAgent          :     16 →     17 (+1)   <- PERFECT!
   ReplayBuffer       :      1 →      1 (+0)   <- PERFECT!
   ndarray            :   1234 →  15678 (+14444) <- Expected
   Tensor             :    567 →   1234 (+667)   <- Expected
======================================================================

======================================================================
🐘 LARGE OBJECTS (>= 50 MB)
======================================================================
Type                           Size (MB)
------------------------------ ---------------
ndarray                             2345.67  <- Replay buffer data
Tensor                               456.23  <- Model parameters
dict                                  89.45  <- Episode stats
list                                  67.34  <- History lists
======================================================================
```

**What to look for:**
- ✅ **Total growth: 25-35 GB** (replay buffer + normal overhead)
- ❌ **Total growth: 100+ GB** (LEAK!)
- ✅ **TradingEnvironment: +0** (no leak)
- ✅ **DDPGAgent: +1** (only best_agent)
- ✅ **Large objects are expected** (replay buffer, model params)

---

## How to Interpret Results

### Scenario 1: Memory Leak in evaluate_population()

**Symptoms:**
```
Gen 1: After evaluate_population    8.52 GB
Gen 2: After evaluate_population   15.34 GB  (+6.82 GB!) <- Too much!
Gen 3: After evaluate_population   22.67 GB  (+7.33 GB!) <- LEAK!

Object Count Changes:
   TradingEnvironment :  2 → 18 (+16)  <- PROBLEM! Should be 2
```

**Diagnosis:** Environments not being reused, creating 16 new ones per generation

**Action:** Check that `evaluate_population()` passes `self.eval_env` to `run_episode()`

---

### Scenario 2: Memory Leak in train_population()

**Symptoms:**
```
Gen 1: After train_population       9.12 GB
Gen 1: After evaluate_population   10.89 GB  <- Small increase (OK)
Gen 2: After train_population      18.45 GB  (+8.56 GB!) <- LEAK!

Object Count Changes:
   Tensor : 567 → 5678 (+5111) <- Too many tensors!
```

**Diagnosis:** Tensors not being deleted after training, batch accumulation

**Action:** Check batch deletion, loss detachment, GC after training

---

### Scenario 3: Memory Leak in evolve_population()

**Symptoms:**
```
Gen 1: After evolve_population      9.45 GB
Gen 2: After evolve_population     20.12 GB  (+10.67 GB!) <- LEAK!

Object Count Changes:
   DDPGAgent : 17 → 33 (+16) <- Old population not deleted!
```

**Diagnosis:** Old population not being deleted before creating new one

**Action:** Check that `evolve_population()` deletes old_population and calls gc.collect()

---

### Scenario 4: Normal Behavior (No Leak)

**Symptoms:**
```
Gen 1: After evaluate_population    8.52 GB
Gen 2: After evaluate_population   10.23 GB  (+1.71 GB)  <- Replay buffer
Gen 3: After evaluate_population   11.89 GB  (+1.66 GB)  <- Replay buffer
Gen 4: After evaluate_population   13.45 GB  (+1.56 GB)  <- Replay buffer
Gen 5: After evaluate_population   14.23 GB  (+0.78 GB)  <- Slowing down
Gen 6: After evaluate_population   14.89 GB  (+0.66 GB)  <- Slowing down
Gen 7: After evaluate_population   15.12 GB  (+0.23 GB)  <- Stable!

Object Count Changes:
   TradingEnvironment :  2 →  2 (+0)  <- PERFECT!
   DDPGAgent          : 16 → 17 (+1)  <- PERFECT!
```

**Diagnosis:** Normal! Memory growth is just replay buffer filling

**Expected Timeline:**
- Gen 1-5: +1-2 GB per gen (buffer filling)
- Gen 6-10: +0.5-1 GB per gen (buffer slowing)
- Gen 11+: +0-0.5 GB per gen (buffer stable)
- Total: 25-35 GB by gen 25

---

## Dependencies

The memory profiler requires `psutil`. Install it:

```bash
# On RunPod
pip install psutil

# Or add to requirements.txt
echo "psutil>=5.9.0" >> requirements.txt
pip install -r requirements.txt
```

---

## Manual Debugging

If you want to check memory at a specific point during debugging:

```python
from utils.memory_profiler import log_memory, get_profiler

# Take a snapshot
log_memory("Debug checkpoint 1", show_objects=True)

# ... your code ...

log_memory("Debug checkpoint 2", show_objects=True)

# Show growth
profiler = get_profiler()
profiler.print_memory_growth(baseline_label="Debug checkpoint 1")
```

---

## What to Report

If you see a leak, report:

1. **Which phase** it occurs in (evaluate, train, or evolve)
2. **RAM growth rate** (GB per generation)
3. **Object count changes** (which type is increasing)
4. **Full memory snapshot** from that phase
5. **Generation number** when it started

Example report:
```
LEAK FOUND:
- Phase: evaluate_population
- Growth: +7 GB per generation
- TradingEnvironment count: 2 → 18 (increasing by 16 each gen)
- Started at generation 1
- Full snapshot attached below:
[paste snapshot here]
```

---

## Deployment

The memory profiling is already integrated. Just:

```bash
cd /workspace
git pull origin main
pip install psutil  # If not already installed
python main.py --resume
```

Watch the logs - you'll see detailed memory snapshots after each phase of every generation.

---

## Expected Output Format

You'll see lots of output. Here's what a healthy run looks like:

```
Generation 1 / 25
============================================================

--- Generation 0: Evaluating Population ---
Evaluating agents: 100%|████████| 16/16 [01:23<00:00,  5.21s/it]

======================================================================
🔍 MEMORY SNAPSHOT: Gen 1: After evaluate_population
======================================================================
💾 Process Memory:
   RSS (actual RAM):      8.52 GB  <- Note this number
   ...
   TradingEnvironment:      2  <- Should always be 2!
======================================================================

[... training happens ...]

======================================================================
🔍 MEMORY SNAPSHOT: Gen 1: After train_population
======================================================================
💾 Process Memory:
   RSS (actual RAM):      9.12 GB  <- Small increase (+0.6 GB) - OK!
======================================================================

[... evolution happens ...]

======================================================================
🔍 MEMORY SNAPSHOT: Gen 1: After evolve_population
======================================================================
💾 Process Memory:
   RSS (actual RAM):      9.45 GB  <- Small increase (+0.33 GB) - OK!
   ...
   DDPGAgent:              17  <- 16 population + 1 best_agent - OK!
======================================================================

======================================================================
📊 MEMORY TREND ACROSS GENERATIONS
======================================================================
[Shows all snapshots]
======================================================================

======================================================================
📈 MEMORY GROWTH ANALYSIS
======================================================================
RAM Usage:
   Baseline:      4.23 GB
   Current:       9.45 GB
   Growth:       +5.22 GB (+123.4%)  <- Mostly replay buffer

Object Count Changes:
   TradingEnvironment :      2 →      2 (+0)  <- PERFECT!
======================================================================

Generation 2 / 25
============================================================
[... repeats for each generation ...]
```

**Healthy pattern:** Small, decreasing growth each generation, object counts stable.

**Leak pattern:** Large, constant or increasing growth, object counts increasing.

---

## Summary

- ✅ Comprehensive memory tracking at all critical points
- ✅ Object count tracking to identify leaks
- ✅ Trend analysis to show growth patterns
- ✅ Large object detection to find memory hogs
- ✅ Automatic reporting at each generation
- ✅ Final comprehensive summary

This will definitively identify:
1. **Where** the leak is (which phase)
2. **What** is leaking (which object type)
3. **How much** (GB per generation)
4. **When** it started (which generation)

Deploy and run - the output will tell us exactly what's happening.
