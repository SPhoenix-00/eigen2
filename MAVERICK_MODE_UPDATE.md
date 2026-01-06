# Maverick Mode Update: Triad 2.0-Inspired Fitness

## Overview

Updated Maverick mode to use a Triad 2.0-inspired fitness function while maintaining aggressive training characteristics. This change simplifies the fitness calculation and aligns Maverick mode more closely with the standard Triad scoring approach.

## Changes Made

### 1. Per-Trade Reward Function (`environment/trading_env.py`)

**Loss Multiplier:**
- **Normal Mode**: 1.0x (no magnification)
- **Consistency Mode**: 1.5x (focus on reducing drawdowns)
- **Maverick Mode**: 1.2x (moderate loss sensitivity) ✨ **NEW**

The loss multiplier is now applied based on mode:
```python
if self.consistency_mode:
    loss_multiplier = Config.CONSISTENCY_LOSS_MULTIPLIER  # 1.5x
elif self.maverick_mode:
    loss_multiplier = Config.MAVERICK_LOSS_MULTIPLIER  # 1.2x
else:
    loss_multiplier = 1.0
```

### 2. Fitness Function (`training/erl_trainer.py`)

**Old Maverick Fitness (FOMO/ROI-First):**
- Base Score: `(Raw PnL / Peak Capital) * log10(Peak Capital + 10) * WR^4`
- Expectancy Bonus: `Expectancy * 20.0`
- FOMO Penalty: `max(0, Market Return - Agent ROI) * 10.0`
- Fear Factor: `2.0 * (num_losses ^ 1.2)`
- WR Penalty: `max(0, 0.60 - WR) * 100.0`

**New Maverick Fitness (Triad 2.0-Inspired):**
- Base Score: `ROI^1.5 * log10(Peak Capital + 10)`
- If Positive: Boosted by `(1 + WR^4)`
- If Negative: Penalized by ROI magnitude
- FOMO Penalty: `max(0, Market Return - Agent ROI) * 10.0` (fitness only, not per trade)

**Removed Components:**
- ❌ Expectancy Bonus
- ❌ Fear Factor
- ❌ WR Penalty (WR gate)

### 3. Configuration (`utils/config.py`)

Added new constant:
```python
MAVERICK_LOSS_MULTIPLIER = 1.2  # Magnify losses by 1.2x in maverick mode (per trade rewards)
```

### 4. Documentation Updates

**`main.py`:**
- Updated `--maverick` help text to reflect new behavior

**`global50.py`:**
- Updated Maverick mode documentation to describe new reward function characteristics

## Maverick Mode Characteristics (Updated)

| Feature | Value | Notes |
|---------|-------|-------|
| **Loss Multiplier** | 1.2x | Per-trade rewards only |
| **Hurdle Rate** | 0.3% | 50% of normal (0.6%) |
| **Forced Exit Penalty** | None | Encourages holding for bigger gains |
| **Zero Trades Penalty** | 5000.0 | Massive penalty to force market engagement |
| **Fitness Function** | Triad 2.0-inspired | `ROI^1.5 * log10(Peak Capital + 10)` |
| **FOMO Penalty** | Yes | Applied in fitness only, not per trade |

## Rationale

1. **Simplified Fitness**: Removed complex components (expectancy bonus, fear factor, WR penalty) in favor of a cleaner Triad 2.0-inspired approach
2. **Loss Sensitivity**: 1.2x loss multiplier provides moderate sensitivity without being as aggressive as consistency mode (1.5x)
3. **FOMO Awareness**: Maintained FOMO penalty to ensure Mavericks don't underperform the market
4. **Alignment**: Closer alignment with standard Triad scoring while maintaining aggressive characteristics

## Impact on Training

- **Per-Trade Rewards**: Losses are now magnified by 1.2x, providing stronger signal for loss avoidance
- **Fitness Calculation**: Simpler formula focuses on ROI efficiency and win rate, with FOMO penalty to ensure market-relative performance
- **Promotion Logic**: No changes required - Global 50 promotion logic is metric-based and independent of fitness function implementation

## Migration Notes

Existing Maverick agents trained with the old fitness function will continue to work, but new training runs will use the updated fitness function. The promotion logic in Global 50 is unaffected as it evaluates raw metrics (gauntlet score, ROI, expectancy, CV) rather than the fitness function itself.

