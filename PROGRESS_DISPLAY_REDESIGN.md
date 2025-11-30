# Progress Display Redesign

## Summary

The output system has been redesigned to be context-aware and provide meaningful progress information for both traditional fixed-generation training and the new gauntlet/consistency modes.

**New Information Displayed:**
- ✅ Hall of Fame size (X / 10)
- ✅ Candidate queue size (heroes mode only)
- ✅ Intelligent ETA based on breakthrough history
- ✅ Gauntlet state machine visualization with icons
- ✅ Stabilization progress bar
- ✅ Context-aware progress metrics (breakthroughs or turnovers, not just generations)

## Key Changes

### 1. Fixed Negative ETA Bug

**Problem:** When `Remaining Gens: -1`, the ETA showed negative values like `-25.5m (-0.4h)`

**Solution:** Added a check to display "Complete" when there are no remaining generations:
```
Remaining Gens:              -1
ETA:                  Complete  (0.0h)
```

### 2. Context-Aware Progress Reporting

The display now adapts based on training mode:

#### **Traditional Mode (Fixed Generations)**
Shows simple remaining generations and ETA:
```
  Remaining Gens:              39
  ETA:                      78.3m  (1.3h)
```

#### **Gauntlet Mode (Breakthrough-Based)**
Shows breakthrough progress with intelligent ETA estimation:
```
  🎯 BREAKTHROUGH PROGRESS
  Confirmed:                    2 / 4
  Needed:                       2
  Est. Gens Left:           40.0  (based on 2 breakthroughs)
  Est. ETA:                 80.3m  (1.3h)

  Fallback Limit:              69 generations remaining
  Max Time Left:           138.6m  (2.3h)
```

**ETA Calculation:**
- Uses breakthrough history to calculate average generations between breakthroughs
- Estimates remaining generations = avg_gens_per_breakthrough × breakthroughs_needed
- Shows "Unknown" if insufficient data (< 2 breakthroughs)
- Always shows fallback generation limit

#### **Consistency Mode (HoF Turnover-Based)**
Shows turnover progress:
```
  🎯 CONSISTENCY PROGRESS
  HoF Turnovers:                1 / 2
  Turnovers Needed:             1
  Est. ETA:              Unknown  (depends on performance)

  Fallback Limit:              59 generations remaining
  Max Time Left:           118.5m  (2.0h)
```

**ETA:** Shows "Unknown" because turnover timing is highly variable and depends on agent performance improvements.

### 3. Gauntlet State Machine Section

Added a new section showing gauntlet state when in gauntlet mode:

```
🎮 GAUNTLET STATE
----------------------------------------------------------------------
  Current State:     🟠 STABILIZATION
  Baseline:                  125.50
  Stab Progress:     [███░░] 3/5
  Breakthroughs:                2 / 4
```

**State Icons:**
- 🔵 NORMAL - Training normally
- 🟡 DETECTION - Breakthrough candidate detected
- 🟠 STABILIZATION - Locking training on candidate (shows progress bar)
- 🔴 GAUNTLET - Running rigorous 20-slice validation
- 🟢 CONFIRMED - Breakthrough confirmed (shows briefly)
- ⚫ REJECTED - Breakthrough rejected, snapback triggered (shows briefly)

**Consistency Mode Shows:**
- HoF Turnovers: X / Y
- HoF Median ROI: Z% (the ratcheting quality bar)

**Normal Gauntlet Mode Shows:**
- Breakthroughs: X / Y

### 4. Stabilization Progress Bar

When in STABILIZATION state, shows a visual progress bar:
```
Stab Progress:     [███░░] 3/5
```

This helps understand how close the candidate is to gauntlet validation.

### 5. Hall of Fame and Queue Tracking

**Hall of Fame Size:** Always shows current size / capacity
```
HoF Size:                     8 / 10
```

This helps you understand:
- How close you are to filling the HoF (required for turnovers in consistency mode)
- How many proven agents you've accumulated

**Candidate Queue Size (Heroes Mode Only):** Shows number of candidates waiting to be tested
```
Queue Size:                   3
```

In `--consistency --heroes` mode, when multiple agents breach the threshold, they're queued and tested one at a time to avoid the "Ghost Loop" problem. This shows how many candidates are still waiting to be tested in the gauntlet.

## Implementation Details

### New Function: `_print_progress_and_eta()`

Located in [utils/display.py:122-211](utils/display.py#L122-L211)

Handles all progress and ETA logic based on training mode. Takes:
- `gen`: Current generation
- `total_gens`: Fallback generation limit
- `avg_gen_time`: Average time per generation
- `gauntlet_info`: Dictionary with gauntlet state (optional)

### Updated Function: `print_generation_summary()`

Located in [utils/display.py:214-356](utils/display.py#L214-L356)

Now accepts optional `gauntlet_info` parameter and displays:
1. **🎮 GAUNTLET STATE** section (if in gauntlet mode)
2. Context-aware progress via `_print_progress_and_eta()`

### Training Loop Integration

Located in [training/erl_trainer.py:4058-4092](training/erl_trainer.py#L4058-L4092)

The training loop now:
1. Collects gauntlet state information
2. Calculates stabilization progress if applicable
3. Passes everything to `print_generation_summary()`

## Benefits

### 1. **Meaningful Progress Indicators**
- No more meaningless negative ETAs
- Progress based on actual goals (breakthroughs/turnovers) not arbitrary generation counts
- Shows what matters: "2/4 breakthroughs" not "30/100 generations"

### 2. **Intelligent ETA Estimation**
- For gauntlet mode: Uses historical breakthrough frequency to estimate remaining time
- Shows "Unknown" when appropriate rather than misleading numbers
- Always provides fallback generation limit for worst-case planning

### 3. **State Visibility**
- Can see exactly where you are in the gauntlet state machine
- Stabilization progress shows how close to gauntlet validation
- Visual indicators make state transitions obvious

### 4. **Mode-Specific Information**
- Consistency mode shows HoF turnover progress and median ROI
- Normal gauntlet shows breakthrough count
- Traditional mode shows simple generation count

## Example Outputs

### During NORMAL State (Gauntlet Mode)
```
🎮 GAUNTLET STATE
----------------------------------------------------------------------
  Current State:     🔵 NORMAL
  Baseline:                  100.00
  Breakthroughs:                1 / 4
  HoF Size:                     3 / 10

🎯 BREAKTHROUGH PROGRESS
  Confirmed:                    1 / 4
  Needed:                       3
  Est. ETA:              Unknown  (insufficient data)

  Fallback Limit:              85 generations remaining
  Max Time Left:           170.5m  (2.8h)
```

### During STABILIZATION State
```
🎮 GAUNTLET STATE
----------------------------------------------------------------------
  Current State:     🟠 STABILIZATION
  Baseline:                  100.00
  Stab Progress:     [███░░] 3/5
  Breakthroughs:                2 / 4
  HoF Size:                     5 / 10
```

### During GAUNTLET State
```
🎮 GAUNTLET STATE
----------------------------------------------------------------------
  Current State:     🔴 GAUNTLET
  Baseline:                  100.00
  Breakthroughs:                1 / 4
  HoF Size:                     3 / 10
```

### After Multiple Breakthroughs (With ETA)
```
🎮 GAUNTLET STATE
----------------------------------------------------------------------
  Current State:     🔵 NORMAL
  Baseline:                  175.50
  Breakthroughs:                3 / 4
  HoF Size:                     7 / 10

🎯 BREAKTHROUGH PROGRESS
  Confirmed:                    3 / 4
  Needed:                       1
  Est. Gens Left:           18.5  (based on 3 breakthroughs)
  Est. ETA:                 37.0m  (0.6h)

  Fallback Limit:              35 generations remaining
  Max Time Left:            70.0m  (1.2h)
```

### Consistency Mode with Heroes Queue
```
🎮 GAUNTLET STATE
----------------------------------------------------------------------
  Current State:     🔵 NORMAL
  Baseline:                  125.50
  HoF Size:                     8 / 10
  HoF Turnovers:                1 / 2
  HoF Median ROI:          15.75%
  Queue Size:                   3

🎯 CONSISTENCY PROGRESS
  HoF Turnovers:                1 / 2
  Turnovers Needed:             1
  Est. ETA:              Unknown  (depends on performance)

  Fallback Limit:              74 generations remaining
  Max Time Left:           148.4m  (2.5h)
```

## Testing

Run the test script to verify all display modes:
```bash
./venv/bin/python -c "from utils.display import _print_progress_and_eta; _print_progress_and_eta(...)"
```

See the test output above for examples of all four modes.
