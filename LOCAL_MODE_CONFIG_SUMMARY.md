# Local Mode Configuration Summary

## Overview
All training parameters have been split into **Local Mode** (`--local` flag) and **Distributed Mode** (cloud/remote) versions to optimize for different resource constraints and training dynamics.

## Key Changes

### 1. Buffer Size Optimization
**Problem:** Local mode adds ~7,500 transitions/generation, but buffer was 1.5M (200 generations to fill), keeping very old bad transitions too long.

**Solution:**
- **Distributed:** `BUFFER_SIZE = 1,500,000` (for 96 agents, ~22,500 transitions/gen)
- **Local:** `LOCAL_BUFFER_SIZE = 120,000` (16 generations of data: 120k / 7.5k = 16 gens)
- **Rationale:** Keeps recent relevant experience while evicting old bad data quickly

### 2. Minimum Buffer Size
- **Distributed:** `MIN_BUFFER_SIZE = 23,200` (start training after ~1 generation)
- **Local:** `LOCAL_MIN_BUFFER_SIZE = 8,000` (start training after ~1 generation with 32 agents)
- **Sweep:** `MIN_BUFFER_SIZE_SWEEP = 5,000` (faster DDPG for hyperparameter sweeps)

### 3. Batch Size
- **Distributed:** `BATCH_SIZE = 160`
- **Local:** `LOCAL_BATCH_SIZE = 64` (reduced for VRAM constraints on RTX 4090)

### 4. Population Size
- **Distributed:** `POPULATION_SIZE = 96`
- **Local:** `LOCAL_POPULATION_SIZE = 32` (reduced for single GPU)

### 5. Gradient Steps Per Generation
- **Distributed Normal:** `GRADIENT_STEPS_PER_GENERATION = 32`
- **Distributed Stabilization:** `GRADIENT_STEPS_PER_GENERATION_STABILIZATION = 10`
- **Local Normal:** `LOCAL_GRADIENT_STEPS_PER_GENERATION = 32`
- **Local Stabilization:** `LOCAL_GRADIENT_STEPS_PER_GENERATION_STABILIZATION = 16`

### 6. DataLoader Workers
- **Distributed:** `NUM_DATALOADER_WORKERS = 6` (parallel batch loading)
- **Local:** `LOCAL_NUM_DATALOADER_WORKERS = 0` (main process only, avoids Windows multiprocessing overhead)

### 7. Gradient Accumulation
- **Distributed:** `GRADIENT_ACCUMULATION_STEPS = 1` (can be increased for larger effective batch)
- **Local:** `LOCAL_GRADIENT_ACCUMULATION_STEPS = 1` (no accumulation - each step = 1 disk read)

### 8. Training Agent Batch Size
- **Local Only:** `LOCAL_TRAINING_AGENT_BATCH_SIZE = 8` (train 8 agents at a time on GPU, 4 batches = 32 agents)

## Configuration Structure

The config is now organized with clear sections:

1. **Distributed Mode Parameters** - For cloud/remote training (96 agents, parallel processing)
2. **Local Mode Parameters** - For `--local` flag (32 agents, sequential, single GPU)
3. **Common Parameters** - Shared across both modes (environment, rewards, etc.)

## Implementation Details

### Buffer Initialization
All buffer creation now uses:
```python
buffer_capacity = Config.LOCAL_BUFFER_SIZE if self.local_mode else Config.BUFFER_SIZE
```

### Min Buffer Size Helper
Added `Config.get_min_buffer_size(local_mode, is_sweep)` method that returns the appropriate threshold:
- Checks for sweep mode first (highest priority)
- Then checks local mode
- Falls back to distributed mode

### Automatic Mode Detection
Buffer `is_ready()` methods automatically detect local mode by checking if `capacity == LOCAL_BUFFER_SIZE`.

## Resource Considerations

### Local Mode (RTX 4090, 24GB VRAM, 64GB RAM)
- **Buffer Memory:** 120k transitions × 0.7 MB = 84 GB uncompressed → ~25-34 GB compressed
- **Population:** 32 agents (vs 96 distributed)
- **Batch Size:** 64 (vs 160 distributed)
- **Gradient Steps:** 8 (vs 32 distributed)
- **Total VRAM Usage:** ~8-12 GB (comfortable for 24GB card)

### Distributed Mode (Cloud GPU)
- **Buffer Memory:** 1.5M transitions × 0.7 MB = 1,050 GB uncompressed → ~315-420 GB compressed
- **Population:** 96 agents
- **Batch Size:** 160
- **Gradient Steps:** 32
- **Parallel Workers:** 6 DataLoader workers + 48 evaluation workers

## Benefits

1. **Faster Training Start:** Local mode can start training after 8k transitions (vs 23k)
2. **Better Data Quality:** Buffer evicts old bad transitions after 16 generations (vs 200)
3. **Resource Efficient:** Smaller buffer fits in RAM, smaller batches fit in VRAM
4. **Aligned Parameters:** All local parameters are proportionally scaled for 32-agent population

## Migration Notes

- Existing checkpoints will automatically resize buffer if capacity changed
- Buffer capacity is checked on load and resized if needed
- Display now shows correct buffer capacity based on mode
- All buffer size checks use `Config.get_min_buffer_size()` helper

