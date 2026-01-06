# Maverick Promotion Logic: The "Highlander" Rule

## Overview

Maverick agents are subject to a hard cap of 5 agents in Global 50 (the "Highlander" rule). This prevents the committee from becoming too volatile while still allowing high-performing Mavericks to enter.

## The Maverick Cap

- **Maximum Mavericks**: 5 agents (`MAVERICK_CAP = 5`)
- **Purpose**: Prevent the Global 50 from becoming dominated by aggressive, volatile agents
- **Enforcement**: Applied during promotion in `check_and_promote()`

## Promotion Logic

When a Maverick qualifies for Global 50 promotion, the following logic applies:

### Case 1: Cap Not Reached (< 5 Mavericks)

If there are fewer than 5 Mavericks in Global 50:
- ✅ **Promoted normally** - No special handling needed
- The Maverick enters Global 50 if it meets all qualification criteria

### Case 2: Cap Reached (5/5 Mavericks)

If the cap is already full, the new Maverick can still be promoted through **displacement**:

1. **All candidates are merged and sorted by gauntlet score (descending)**
   - New Maverick + all existing Global 50 entries
   - Sorted from highest to lowest score

2. **Top-to-bottom processing with cap enforcement**
   - Normal agents: Always accepted (subject to Global 50 capacity of 50)
   - Mavericks: Accepted only if `maverick_count < MAVERICK_CAP` (5)

3. **Displacement occurs if:**
   - The new Maverick ranks high enough to be in the top 5 Mavericks
   - Lower-ranked existing Mavericks are displaced (moved to archive)
   - The new Maverick enters Global 50

4. **Archiving occurs if:**
   - The new Maverick qualifies but cannot displace any existing Maverick
   - It ranks 6th or lower among all Mavericks
   - The agent is archived with metadata saved for future reference

## Example Scenarios

### Scenario 1: Displacement Success

```
Current Global 50 Mavericks (5/5):
1. Maverick A: Score 1500
2. Maverick B: Score 1400
3. Maverick C: Score 1300
4. Maverick D: Score 1200
5. Maverick E: Score 1100

New Maverick: Score 1250

Result: ✅ PROMOTED
- New Maverick ranks 4th among Mavericks
- Maverick E (lowest at 1100) is displaced
- New Maverick enters at rank 4
```

### Scenario 2: Archiving (Cannot Displace)

```
Current Global 50 Mavericks (5/5):
1. Maverick A: Score 1500
2. Maverick B: Score 1400
3. Maverick C: Score 1300
4. Maverick D: Score 1200
5. Maverick E: Score 1100

New Maverick: Score 1050

Result: ❌ ARCHIVED
- New Maverick ranks 6th (below all existing Mavericks)
- Cannot displace any existing Maverick
- Qualifies but archived due to cap
```

## Implementation Details

### Code Location

The displacement logic is implemented in `erl/global_hof.py` in the `check_and_promote()` method:

```python
# Merge: Add new agent to candidate pool
candidates = self.entries + [new_entry]

# Sort: Rank by Gauntlet Score (descending)
candidates.sort(key=lambda e: e.gauntlet_score, reverse=True)

# Apply Maverick Cap Enforcement
for entry in candidates:
    if entry.is_maverick:
        if maverick_count < self.MAVERICK_CAP:
            final_list.append(entry)
            maverick_count += 1
        else:
            # Cap hit - this Maverick is rejected
            # Lower-ranked ones get dropped
            dropouts.append(entry)
```

### Archiving Logic

If a Maverick qualifies but cannot be promoted (due to cap), it is archived in `global50.py`:

```python
# If maverick qualified but wasn't promoted (likely due to cap), archive it
if is_maverick and result.get('qualified', False) and not result.get('promoted', False):
    if current_maverick_count >= self.global_hof.MAVERICK_CAP:
        # Archive the qualifying maverick
        archived = self.archive_qualifying_maverick(...)
```

## Key Points

1. **Displacement is automatic** - Higher-scoring Mavericks automatically displace lower-scoring ones
2. **No manual intervention needed** - The system handles cap enforcement transparently
3. **Archiving preserves qualified agents** - Agents that qualify but can't enter are archived with full metadata
4. **Score-based ranking** - Displacement is purely based on gauntlet score, ensuring the best Mavericks remain

## Related Documentation

- `MAVERICK_MODE_UPDATE.md` - Maverick training characteristics
- `docs/maverick_global50_verbose.md` - Verbose promotion logging
- `global50.py` - Main evaluation script with Maverick support

