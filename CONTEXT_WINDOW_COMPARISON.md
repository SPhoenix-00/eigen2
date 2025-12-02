# Context Window Comparison Guide

## Overview

This tool tests whether agents trained on **504-day context windows** can perform well with **151-day context windows**.

### The Question
"If we drop the context window from 504 to 151 days, can we still use our existing Global 50 agents?"

### The Answer
**Mathematically possible, but likely poor performance.** This tool measures the actual degradation.

---

## Quick Start

### Test your best Global 50 agent
```bash
python compare_context_windows.py
```

### Test a specific agent
```bash
python compare_context_windows.py --agent-path workspace/global50/agents/agent_001.pth
```

### Test all Global 50 agents
```bash
python compare_context_windows.py --test-all-g50
```

### Use fewer slices for faster testing
```bash
python compare_context_windows.py --num-slices 10
```

### Use exact gauntlet validation (same as G50)
```bash
python compare_context_windows.py --gauntlet
```

This uses the **exact same validation methodology** as your Global 50 evaluation:
- ⚔️ 10 slices from training data (tests generalization)
- ⚔️ 10 slices from validation data (tests on held-out data)
- Same logic as `ERLTrainer.generate_gauntlet_slices()`

---

## What It Does

### 1. **Loads Agent**
   - Uses existing G50 agent trained on 504-day windows
   - No retraining required

### 2. **Creates Two Evaluation Scenarios**
   - **Scenario A**: Agent sees 504 days of market history (TRAINED)
   - **Scenario B**: Agent sees 151 days of market history (OUT-OF-DISTRIBUTION)

### 3. **Runs Gauntlet Validation**
   - Tests agent on 20 validation slices (configurable)
   - Same test data for both scenarios
   - Deterministic evaluation (no exploration noise)

### 4. **Compares Performance**
   - Fitness score delta
   - ROI % delta
   - Win rate changes
   - Trade frequency changes

---

## Sample Output

```
================================================================================
CONTEXT WINDOW COMPARISON RESULTS
================================================================================

Agent: agent_gen045_elite_rank00.pth
Slices Tested: 20

--------------------------------------------------------------------------------
Metric                         504-day              151-day              Delta
--------------------------------------------------------------------------------
Avg Fitness Score                 1245.67              982.34         -263.33 (-21.1%)
Std Fitness Score                  156.23              201.45
Avg ROI %                           12.45                8.92           -3.53 (-28.4%)
Std ROI %                            3.21                4.67
Win Rate %                          68.50               62.30           -6.20
Total Trades                          847                 621            -226
Avg Gain % (per trade)               4.23                3.87
Max Drawdown %                       -8.45              -12.34
--------------------------------------------------------------------------------

INTERPRETATION:
   SEVERE DEGRADATION: >10% fitness loss with reduced context
   → Agent heavily relies on 504-day patterns
   → NOT recommended to use with 151-day windows
================================================================================
```

---

## Interpreting Results

### Score Delta %

| Delta | Interpretation | Recommendation |
|-------|---------------|----------------|
| **< -10%** | SEVERE DEGRADATION | ❌ Do NOT use with 151-day context |
| **-10% to -5%** | MODERATE DEGRADATION | ⚠️ Use with caution |
| **-5% to 0%** | MINOR DEGRADATION | ✅ May be usable (verify on holdout) |
| **> 0%** | NO DEGRADATION | ✅ Robust to context reduction |

### Why Degradation Happens

1. **Temporal Pattern Mismatch**
   - Agent learned 504-day patterns during training
   - 151 days is 70% shorter - different characteristics
   - LSTM hidden states evolve differently over shorter sequences

2. **Missing Long-Term Signals**
   - Patterns that require >151 days to emerge are unavailable
   - Agent may look for signals that don't exist in shorter window

3. **Feature Scale Differences**
   - Normalization stats computed from 504-day sequences
   - 151-day sequences may have different statistical properties

---

## Technical Details

### How It Works

1. **Temporary Config Override**
   ```python
   Config.CONTEXT_WINDOW_DAYS = 151  # Temporarily modified
   env = TradingEnvironment(...)     # Uses 151-day observations
   Config.CONTEXT_WINDOW_DAYS = 504  # Restored
   ```

2. **No Model Retraining**
   - Agent weights unchanged
   - Model architecture handles variable-length LSTM inputs
   - Performance degradation measured empirically

3. **Fair Comparison**
   - Same validation slices for both scenarios
   - Same deterministic evaluation (add_noise=False)
   - Same metrics calculation

### Validation Slices

- Generated from validation data range
- Each slice: 125 trading days + 30 settlement days = 155 days
- Evenly spaced across validation window
- Ensures enough history for context window:
  - 504-day slices: start_idx >= 504
  - 151-day slices: start_idx >= 151

---

## Use Cases

### 1. **Research Question**
   "Do our agents actually need 504 days of history, or would 151 suffice?"

### 2. **Computational Efficiency**
   - 151-day context = 70% less LSTM computation
   - Faster inference during evaluation
   - Smaller observation tensors

### 3. **Data Availability**
   - Some assets don't have 504 days of history
   - Can we apply agents to newer markets with less data?

### 4. **Overfitting Detection**
   - If agent performs BETTER with less context, it may be overfit
   - Could indicate memorization of noise in 504-day window

---

## Next Steps After Testing

### If SEVERE degradation (>10% loss):
1. **Train new agents specifically for 151-day context**
   - Set `CONTEXT_WINDOW_DAYS = 151` in config.py
   - Run full ERL training
   - Maintain separate G50 leagues for different context windows

### If MODERATE degradation (5-10% loss):
2. **Fine-tune existing agents on 151-day data**
   - Load 504-day agent weights
   - Continue training with 151-day context
   - Transfer learning approach

### If MINOR degradation (<5% loss):
3. **Test on holdout data to confirm**
   - Validation results may be optimistic
   - Verify performance on completely unseen data
   - Consider using 151-day context if efficiency gains are worth it

### If NO degradation (0% or positive):
4. **Investigate why**
   - May indicate overfitting to long-term noise
   - Could suggest 151-day patterns are more predictive
   - Consider retraining from scratch with 151 days

---

## Limitations

### What This Tool DOESN'T Test

1. **Training from Scratch with 151 Days**
   - This tests EXISTING 504-day agents on 151-day data
   - Agents trained from scratch on 151 days may perform better
   - They would learn 151-day specific patterns

2. **Different Architectures**
   - Current LSTM architecture may not be optimal for 151 days
   - Could explore different model designs for shorter contexts

3. **Statistical Significance**
   - Uses validation data (already seen during training)
   - True test requires holdout data
   - Results may vary on different time periods

---

## Files

- **compare_context_windows.py** - Main comparison script
- **CONTEXT_WINDOW_COMPARISON.md** - This guide
- **context_window_comparison.json** - Output results (auto-generated)

---

## Questions?

This experiment answers: **"Can we use 504-day agents with 151-day context?"**

To answer: **"Should we train new agents on 151-day context from scratch?"**
- You would need to run full ERL training with `CONTEXT_WINDOW_DAYS = 151`
- Compare those agents to your current 504-day agents
- That's a different experiment entirely!
