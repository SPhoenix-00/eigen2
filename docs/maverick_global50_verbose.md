# Maverick Mode: Verbose Global50 Promotion Logging

## Overview

When using `--maverick` mode, every breakthrough now explicitly attempts Global50 promotion with comprehensive verbose metrics showing which gates pass and which fail.

## Changes

### Behavior

- **Every breakthrough in Maverick mode** now explicitly attempts Global50 promotion using the existing gauntlet results
- Previously, Global50 promotion was only attempted if the agent passed an initial qualification check
- Now, all breakthroughs get a full Global50 evaluation attempt with detailed metrics using the same gauntlet that confirmed the breakthrough (no duplicate runs)

### Verbose Output

When a breakthrough occurs in Maverick mode, you'll see:

1. **Promotion Analysis Header**
   ```
   📊 MAVERICK MODE: Global50 Promotion Analysis
   ```

2. **Agent Metrics Display**
   - Gauntlet Score
   - ROI percentage
   - Expectancy
   - CV (Coefficient of Variation)
   - Total Trades
   - Win Rate

3. **Global50 Thresholds**
   - Entry thresholds for each metric
   - 25th percentile (p25) values
   - Median (50th percentile) values

4. **Gate-by-Gate Analysis**
   - **Gate 1**: Minimum Thresholds (all must pass)
     - Gauntlet Score > Entry Threshold
     - ROI > Entry Threshold
     - CV < Entry Threshold (lower is better)
     - Expectancy: SKIPPED for Mavericks
   - **Gate 2**: p25 Criterion (need 2/2 for Mavericks)
     - Gauntlet Score > p25
     - ROI > p25
     - Expectancy: SKIPPED for Mavericks
   - **Gate 3**: Median Criterion (need 1/2 for Mavericks)
     - Gauntlet Score > Median
     - ROI > Median
     - Expectancy: SKIPPED for Mavericks

5. **Final Result**
   - Clear indication if agent qualifies for Global50 promotion
   - If promoted: Shows rank and whether Maverick goal (rank ≤ 20) was achieved

## Example Output

```
============================================================
⭐ BREAKTHROUGH CONFIRMED!
============================================================
  Spike Score: 1300.00 (lucky)
  Gauntlet Score: 1248.30 (robust)
  Previous Baseline: 1200.00
  New Baseline: 1248.30 (applied immediately)
============================================================

============================================================
📊 MAVERICK MODE: Global50 Promotion Analysis
============================================================
  Agent Metrics:
    Gauntlet Score: 1248.30
    ROI: 12.45%
    Expectancy: 0.0234
    CV: 15.20 (lower is better)
    Total Trades: 342
    Win Rate: 58.50%

  Global50 Thresholds:
    Gauntlet Entry: 1200.00 | p25: 1250.00 | Median: 1300.00
    ROI Entry: 8.00% | p25: 10.00% | Median: 12.00%
    CV Entry: 20.00 (max) | (Mavericks skip expectancy requirement)

  Promotion Criteria Analysis:
    Gate 1: Minimum Thresholds (All must pass):
      - Gauntlet: 1248.30 > 1200.00 ✅
      - ROI: 12.45% > 8.00% ✅
      - Expectancy: SKIPPED (Maverick mode)
      - CV: 15.20 < 20.00 ✅ (lower is better)
    
    Gate 2: p25 Criterion (Need 2/2 to pass):
      - Gauntlet: 1248.30 > p25 (1250.00) ❌
      - ROI: 12.45% > p25 (10.00%) ✅
      - Expectancy: SKIPPED (Maverick mode)
      Result: 1/2 passed ❌
    
    Gate 3: Median Criterion (Need 1/2 to pass):
      - Gauntlet: 1248.30 > Median (1300.00) ❌
      - ROI: 12.45% > Median (12.00%) ✅
      - Expectancy: SKIPPED (Maverick mode)
      Result: 1/1 passed ✅

  ❌ FAILED: Agent does not qualify for Global50 promotion
============================================================
```

## Technical Details

### Files Modified

1. **`training/erl_trainer.py`**
   - Modified breakthrough confirmation logic to use existing gauntlet results for Global50 promotion (no duplicate runs)
   - Added comprehensive verbose metrics display section for Maverick mode
   - Enhanced promotion result display with detailed metrics
   - Simplified logic to reuse the gauntlet that confirmed the breakthrough

2. **`erl/global_hof.py`**
   - Enhanced `analyze_promotion()` method to always show detailed gate-by-gate analysis
   - Changed from returning simple "Passed all criteria" to showing all three gates with pass/fail indicators
   - Added clear visual indicators (✅/❌) for each metric check

### Maverick-Specific Behavior

- Mavericks skip expectancy requirements (as per existing Maverick mode design)
- All three gates are evaluated, but expectancy checks are marked as "SKIPPED"
- Gate 2 requires 2/2 metrics (Gauntlet, ROI) instead of 2/3
- Gate 3 requires 1/2 metrics (Gauntlet, ROI) instead of 1/3

## Usage

Simply use the `--maverick` flag as before:

```bash
python main.py --maverick
```

The verbose Global50 promotion logging is automatic for all breakthroughs in Maverick mode.

## Benefits

1. **Efficiency**: Uses existing breakthrough gauntlet results - no duplicate gauntlet runs
2. **Transparency**: Clear visibility into why an agent passes or fails Global50 promotion
3. **Debugging**: Easy to identify which specific gate or metric is preventing promotion
4. **Progress Tracking**: See how close agents are to meeting each threshold
5. **Training Guidance**: Understand what metrics need improvement for future training

