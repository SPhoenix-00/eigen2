# Triad 3.0: Maverick Stabilization Strategy

## Executive Summary

We identified a critical flaw in the "Maverick" agent behavior: **Volume Addiction**. High-frequency agents were "gaming" the reward system by placing massive bets to inflate their score, masking poor statistical expectancy (luck vs. skill). They were trading frequently but failing to converge on the robust win ratios seen in Consistency runs.

To fix this, we devised a three-part "Sniper" strategy to force selectivity without killing their aggressive nature.

---

## The Problem: "Lucky Whales"

### Old Logic
- **Rewarded:** `ROI * log(Capital)`
- **The Flaw:** An agent could have a coin-flip edge (0 expectancy) but get a massive score simply by using 100k capital.
- **The Result:** High variance, high volume, low reliability.

Agents were optimizing for volume rather than performance, leading to:
- Inflated scores from high capital usage
- Poor statistical edge (low expectancy)
- Unreliable performance that didn't generalize

---

## The Solution: Selectivity & Gradient

### 1. Clamp the Volume
- **Capped volume scalar at 4.3** (equivalent to ~20k capital)
- Forces agents to improve *performance* (ROI/Expectancy) to get a higher score
- Prevents "lucky whales" from gaming the system with massive capital

### 2. Reward Precision (Expectancy²)
- Introduced a **quadratic reward for Expectancy** (statistical edge)
- Scaled by 100 before squaring to prevent small decimals from vanishing
- **Examples:**
  - Mediocre edge (1%) = 1.0 score → 1.1x boost
  - High edge (5%) = 25.0 score → 3.5x boost

### 3. The "Proximity" Magnet (Gradient)
- Instead of a binary "Pass/Fail" for the Hall of Fame (Global 50), we added a **gradient**
- The agent now feels a "magnetic pull" (reduced penalty) the closer its metrics get to the Global 50 entry thresholds
- This guides the blind optimizer toward the "door" of the Hall of Fame

---

## Implementation Details

### Files Modified
- `training/erl_trainer.py` - `calculate_triad_fitness()` method
- `training/erl_trainer.py` - `calculate_holographic_fitness()` method (CRITICAL FIX)

### Critical Fix: Holographic Fitness Bypass

**Issue Identified:** When `maverick_mode` is enabled, the code calculates `triad_fitness` (with Triad 3.0 logic), but then **overwrites** it with `calculate_holographic_fitness()` during population selection. The old holographic method lacked the volume clamp, expectancy², and G50 gradient logic.

**Solution:** Updated `calculate_holographic_fitness()` to include all Triad 3.0 components:
- Expectancy calculation from stitched trades
- Expectancy² reward
- Volume clamp (using trade count as proxy)
- G50 proximity gradient

This ensures Mavericks actually feel the new incentives during the most important phase: population selection.

### Key Changes

#### A. ROI Score (Power Law)
```python
if roi_pct >= 0:
    roi_score = (roi_pct ** 1.5)
else:
    roi_score = -(abs(roi_pct) ** 1.5)
```

#### B. CLAMPED Volume Scalar (20k Limit)
```python
raw_vol_scalar = math.log10(peak_capital + 10)
volume_scalar = min(raw_vol_scalar, 4.3)  # Clamp to stop "lucky whales"
```

#### C. Expectancy² (Precision Reward)
```python
exp_scaled = expectancy * 100.0
if exp_scaled >= 0:
    exp_score = (exp_scaled ** 2)
else:
    exp_score = -(abs(exp_scaled) ** 2)
```

#### D. Base Fitness with Expectancy Boost
```python
if roi_score > 0:
    # Win Scenario: Boost by Expectancy
    fitness = roi_score * volume_scalar * (1.0 + (exp_score * 0.1))
else:
    # Loss Scenario: Expectancy failure amplifies pain
    fitness = roi_score * volume_scalar + (exp_score * volume_scalar)
```

#### E. FOMO Penalty (Dampened)
```python
market_return = stats.get('market_return_pct', 0.0)
alpha_gap = market_return - roi_pct
fomo_penalty = max(0.0, alpha_gap) * 5.0  # Reduced from 10.0
fitness -= fomo_penalty
```

#### F. Global 50 Proximity Gradient
```python
if hasattr(self, 'global_hof') and self.global_hof.enabled and self.global_hof.entry_threshold > -999:
    t_score = [self.global_hof.entry_threshold, self.global_hof.gauntlet_p25, self.global_hof.gauntlet_median]
    t_roi   = [self.global_hof.roi_threshold,   self.global_hof.roi_p25,      self.global_hof.roi_median]
    t_exp   = [self.global_hof.expectancy_threshold, self.global_hof.expectancy_p25, self.global_hof.expectancy_median]

    curr_score = fitness 
    curr_roi = roi_pct
    curr_exp = expectancy

    def calc_gap(target, current):
        return max(0.0, target - current)

    # Weights: Entry=3.0, p25=1.5, Median=1.0
    gap_score = (calc_gap(t_score[0], curr_score)*3.0 + calc_gap(t_score[1], curr_score)*1.5 + calc_gap(t_score[2], curr_score))
    gap_roi   = (calc_gap(t_roi[0], curr_roi)*3.0     + calc_gap(t_roi[1], curr_roi)*1.5     + calc_gap(t_roi[2], curr_roi))
    gap_exp   = (calc_gap(t_exp[0], curr_exp)*300.0   + calc_gap(t_exp[1], curr_exp)*150.0   + calc_gap(t_exp[2], curr_exp)*100.0)

    proximity_penalty = (gap_score * 0.5) + (gap_roi * 1.0) + (gap_exp * 1.0)
    fitness -= proximity_penalty
```

#### Holographic Mode Adaptations

In `calculate_holographic_fitness()`, the Triad 3.0 logic is adapted for the virtual equity curve:

- **Volume Proxy:** Uses `log10(total_trades + 10)` clamped at 4.3 (maps trade count to volume scalar)
- **Expectancy Calculation:** Computed from winning/losing PnL arrays across all stitched trades
- **G50 Gradient:** Focuses on score gap only (ROI/Expectancy gaps harder to map 1:1 in holographic mode)
- **Drawdown Penalty:** Steeper penalty (5.0x multiplier) to act as "Gauntlet Proxy"

### Gradient Descent Concept

The proximity gradient works like gradient descent in machine learning:
- Instead of a binary reward (pass/fail), agents receive a **continuous gradient signal**
- The closer they get to the thresholds, the smaller the penalty
- This creates a "magnetic pull" toward the optimal solution (Hall of Fame entry)
- Agents are guided rather than hoping to stumble upon the solution

**Reference:** [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)

---

## Expected Impact

### Before (Triad 2.0)
- Agents optimized for volume (high capital usage)
- Low statistical edge (poor expectancy)
- High variance, unreliable performance
- Difficulty converging on robust win ratios

### After (Triad 3.0)
- Agents forced to optimize for performance (ROI/Expectancy)
- Volume capped, preventing "lucky whale" strategies
- Gradient guidance toward Hall of Fame thresholds
- Better convergence on robust win ratios

---

## Compatibility

- **Normal/Consistency Mode:** Unchanged - standard Triad scoring preserved
- **Maverick Mode:** Updated to Triad 3.0 with volume clamping, expectancy² reward, and proximity gradient
- **Global 50:** Expectancy now included in gradient (no longer bypassed for Mavericks)

---

## Migration Notes

- Existing Maverick agents trained with Triad 2.0 will continue to work
- New training runs will automatically use Triad 3.0
- The Global 50 promotion logic now includes expectancy for all agents (Mavericks included)
- No configuration changes required - the update is automatic

---

## Related Documentation

- `MAVERICK_MODE_UPDATE.md` - Previous Maverick mode updates (Triad 2.0)
- `docs/maverick_promotion_logic.md` - Global 50 promotion details
- `training/erl_trainer.py` - Implementation location

---

## Technical Notes

### Expectancy Calculation
```python
# Expectancy = (Win% * AvgWin%) - (Loss% * AvgLoss%)
wins = [t['gain_pct'] for t in closed_trades if t['gain_pct'] > 0]
losses = [abs(t['gain_pct']) for t in closed_trades if t['gain_pct'] <= 0]

avg_win = np.mean(wins) if wins else 0.0
avg_loss = np.mean(losses) if losses else 0.0

wr_calc = len(wins) / len(closed_trades)
lr_calc = 1.0 - wr_calc
expectancy = (wr_calc * avg_win) - (lr_calc * avg_loss)
```

### Volume Clamping Rationale
- `log10(20,000) ≈ 4.3`
- This prevents agents from using excessive capital to inflate scores
- Forces focus on improving ROI and expectancy rather than just scaling up

### Gradient Weights
- **Entry Threshold:** 3.0x weight (most important - the "door")
- **P25:** 1.5x weight (good performance target)
- **Median:** 1.0x weight (baseline performance)

These weights ensure agents are most strongly pulled toward the entry threshold while still receiving guidance from percentile targets.

