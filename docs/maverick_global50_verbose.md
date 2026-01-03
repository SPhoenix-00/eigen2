# Maverick Mode: Verbose Global50 Promotion Logging

## Overview

When using `--maverick` mode, every breakthrough now explicitly attempts Global50 promotion with comprehensive verbose metrics showing which gates pass and which fail.

## Changes

### Behavior

- **Every breakthrough in Maverick mode** now runs a consistency-aligned re-gauntlet for Global50 evaluation
- Previously, Global50 promotion was only attempted if the agent passed an initial qualification check
- Now, all breakthroughs get a full Global50 evaluation attempt with detailed metrics

### Verbose Output

When a breakthrough occurs in Maverick mode, you'll see:

1. **Promotion Attempt Header**
   ```
   🔥 MAVERICK MODE: Global50 Promotion Attempt
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
🔥 MAVERICK MODE: Global50 Promotion Attempt
============================================================
  Initial Gauntlet Score: 1250.50
  Running consistency-aligned re-gauntlet for fair Global 50 comparison...
   Consistency-aligned gauntlet score: 1248.30

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
   - Modified breakthrough confirmation logic to always run consistency-aligned re-gauntlet for Maverick mode
   - Added comprehensive verbose metrics display section
   - Enhanced promotion result display with detailed metrics

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

1. **Transparency**: Clear visibility into why an agent passes or fails Global50 promotion
2. **Debugging**: Easy to identify which specific gate or metric is preventing promotion
3. **Progress Tracking**: See how close agents are to meeting each threshold
4. **Training Guidance**: Understand what metrics need improvement for future training

