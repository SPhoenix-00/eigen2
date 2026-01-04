# Batch Size, Chunking, and Gradient Accumulation Explained

## Current Configuration

### Actual Batch Size
- **`Config.BATCH_SIZE = 160`** - This is the actual batch size sampled from the replay buffer
- The DataLoader provides batches of **160 samples** per update step

### Chunking (Micro-batching for ROCm)
- **`chunk_size = 32`** - Physical processing chunk size for ROCm stability
- **Purpose**: Prevents ROCm attention kernel crashes by processing smaller chunks
- **How it works**: 
  - Batch of 160 is split into **5 chunks of 32 samples each**
  - Each chunk is processed separately through the network
  - Results are concatenated back together: `torch.cat([chunk1, chunk2, ..., chunk5], dim=0)`
  - **Mathematically identical** to processing 160 at once

### Gradient Accumulation
- **`Config.GRADIENT_ACCUMULATION_STEPS = 1`** - **No gradient accumulation**
- This means each update step processes exactly **1 batch of 160 samples**
- The optimizer steps after every batch (no accumulation across batches)

## Effective Batch Size

**The effective batch size is still 160.**

Here's why:
1. **DataLoader** samples 160 transitions from the replay buffer
2. **Chunking** splits this into 5×32 chunks for physical processing (ROCm stability)
3. **Loss computation** happens on the **full concatenated result** (all 160 samples)
4. **Gradients** are computed correctly across **all 160 samples** via Autograd
5. **Optimizer step** uses gradients from the full 160-sample batch

### Key Point: Chunking ≠ Batch Size Reduction

- **Chunking** is a **physical processing optimization** (prevents crashes)
- **Batch size** is a **mathematical hyperparameter** (affects gradient estimates)
- The chunking preserves the mathematical properties of batch size 160
- Autograd correctly backpropagates through `torch.cat()` operations

## Visual Flow

```
Replay Buffer
    ↓
DataLoader samples 160 transitions
    ↓
Batch: [160, 151, 117, 5]  ← Actual batch size = 160
    ↓
[ROCm Only] Split into chunks:
    Chunk 1: [32, 151, 117, 5]
    Chunk 2: [32, 151, 117, 5]
    Chunk 3: [32, 151, 117, 5]
    Chunk 4: [32, 151, 117, 5]
    Chunk 5: [32, 151, 117, 5]
    ↓
Process each chunk through network (separately)
    ↓
Concatenate: torch.cat([chunk1, chunk2, ..., chunk5], dim=0)
    ↓
Result: [160, ...]  ← Full batch preserved
    ↓
Compute loss on full 160-sample result
    ↓
Backward pass (gradients computed across all 160 samples)
    ↓
Optimizer step (using gradients from full 160-sample batch)
```

## Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Batch Size** | 160 | Mathematical hyperparameter (affects gradient estimates) |
| **Chunk Size** | 32 | Physical processing size (ROCm stability, prevents crashes) |
| **Gradient Accumulation** | 1 | No accumulation (each step = 1 batch) |
| **Effective Batch Size** | **160** | Still 160 (chunking is transparent to gradients) |

## Why Chunking Works

PyTorch's Autograd engine is smart enough to handle chunking:
- When you do `torch.cat([chunk1, chunk2, ...], dim=0)`, Autograd tracks the computation graph
- During backward pass, gradients flow back through the concatenation
- The final gradients are computed as if you processed the full batch at once
- **Result**: Mathematically identical to batch size 160, but physically processes 32 at a time

## Performance Impact

- **Slower than full batch**: Processing 5 chunks sequentially is slower than 1 full batch
- **But stable**: Prevents ROCm memory access faults
- **Trade-off**: Stability > Speed (for ROCm)

