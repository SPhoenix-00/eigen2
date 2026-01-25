# Buffer Size Analysis for Local Mode Training

## Current Situation

### Transitions Per Generation Calculation

**Local Mode Parameters:**
- `LOCAL_POPULATION_SIZE = 32` agents
- `ELITE_FRAC = 0.4` → ~13 elites, ~19 exploratory agents
- `num_episodes = 3` (normal mode) per agent
- `TRADING_PERIOD_DAYS = 125` steps per episode
- Only **exploratory agents** collect transitions (elites don't add noise)

**Transitions per generation:**
```
19 exploratory agents × 3 episodes × 125 steps = 7,125 transitions/generation
```

### Buffer Size Problem

**Current Configuration:**
- `BUFFER_SIZE = 1,500,000` transitions (shared for local and distributed)
- Transitions added per generation: ~7,500
- **Time to fill buffer: 1,500,000 / 7,500 = 200 generations**

**The Issue:**
- With only 7,500 transitions added per generation, it takes **200 generations** to fill the buffer
- This means very old, potentially **bad transitions from early training** stay in the buffer for an extremely long time
- Agents are training on outdated experiences from much weaker agents
- The buffer contains data from generation 1-200, but by generation 200, the agents are vastly improved

### Optimal Buffer Size

**Recommended: 120,000 transitions**
- **Generations of data: 120,000 / 7,500 = 16 generations**
- This keeps a rolling window of the **most recent 16 generations** of experience
- Old, bad transitions are naturally evicted as new ones are added
- Still provides sufficient diversity for stable learning
- Aligns with DDPG best practices (typically 10-20x batch size, or 10-20 generations of data)

## Memory Impact

### Per Transition Size
- State: (151, 117, 5) float32 = 151 × 117 × 5 × 4 bytes = 353,340 bytes ≈ **0.34 MB**
- Next State: Same = **0.34 MB**
- Action: (108, 2) float32 = 864 bytes ≈ **0.001 MB**
- Reward + Done: Negligible
- **Total per transition: ~0.7 MB uncompressed**

### Buffer Memory Usage
- **120,000 transitions:**
  - Uncompressed: 120,000 × 0.7 MB = **84 GB**
  - Compressed (60-70% reduction): **25-34 GB**
  - **Acceptable for 64GB RAM system** (leaves ~30GB for other operations)

- **1,500,000 transitions:**
  - Uncompressed: 1,500,000 × 0.7 MB = **1,050 GB** (1 TB!)
  - Compressed: **315-420 GB**
  - **Too large for local machine** (would cause swapping/OOM)

## Comparison: Local vs Distributed

| Parameter | Local Mode | Distributed Mode |
|-----------|------------|------------------|
| Population Size | 32 | 96 |
| Transitions/Generation | ~7,500 | ~22,500 (3× more) |
| Current Buffer Size | 1,500,000 | 1,500,000 |
| Generations to Fill | 200 | 67 |
| **Recommended Buffer** | **120,000** | **450,000** (3× local) |

## Recommendations

1. **Add `LOCAL_BUFFER_SIZE` parameter** to `config.py`
   - Set to 120,000 for local mode
   - Keep `BUFFER_SIZE = 1,500,000` for distributed mode
   - Use appropriate buffer size based on training mode

2. **Update buffer initialization** to use `LOCAL_BUFFER_SIZE` when in local mode

3. **Consider other parameter alignment:**
   - `LOCAL_GRADIENT_STEPS_PER_GENERATION = 8` (vs 32 distributed)
   - `LOCAL_BATCH_SIZE = 64` (vs 160 distributed)
   - These are already well-aligned for local mode

## Implementation

The buffer should be initialized with:
```python
if local_mode:
    buffer_capacity = Config.LOCAL_BUFFER_SIZE
else:
    buffer_capacity = Config.BUFFER_SIZE
```

This ensures local mode uses an appropriately sized buffer that:
- ✅ Keeps recent, relevant experience
- ✅ Evicts old, bad transitions quickly
- ✅ Fits in available RAM
- ✅ Aligns with training dynamics (16 generations of data)

