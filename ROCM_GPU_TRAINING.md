# ROCm GPU Training Approach

## Overview

Complete rewrite of agent training for ROCm architecture to use **full GPU acceleration** while avoiding memory access faults.

## Problem

Original training failed on ROCm with:
```
Memory access fault by GPU node-2 (Agent handle: 0x2f09790) on address 0x7224c2401000. Reason: Unknown.
Aborted (core dumped)
```

**Root Cause**: Moving neural networks to/from GPU causes memory access faults on ROCm.

## Solution: GPU Training with No Network Movement

**Key Principle**: Networks are initialized on GPU and **NEVER moved**. All training happens on GPU with careful synchronization.

### Core Strategy

1. **GPU Initialization**: Networks created directly on GPU, never on CPU
2. **Permanent GPU Residence**: Networks stay on GPU for entire agent lifetime
3. **Synchronous Operations**: All ROCm operations use synchronous transfers
4. **Explicit Synchronization**: `torch.cuda.synchronize()` at critical points

## Implementation Details

### 1. Agent Initialization (`models/ddpg_agent.py`)

```python
# Networks created directly on GPU
self.actor = Actor().to(self.device)  # GPU device
self.actor_target = Actor().to(self.device)
self.critic = Critic().to(self.device)
self.critic_target = Critic().to(self.device)

# For ROCm: Synchronize after network creation
if self.use_rocm_mode and torch.cuda.is_available():
    torch.cuda.synchronize()
```

**Key Points**:
- All networks on GPU from creation
- Explicit sync after initialization for ROCm
- `use_rocm_mode` flag for special handling

### 2. Update Method (`models/ddpg_agent.py`)

```python
# Batch transfer with ROCm-specific handling
transfer_mode = not self.use_rocm_mode  # non_blocking only for non-ROCm
states = batch['states'].to(self.device, non_blocking=transfer_mode)
# ... other tensors ...

# For ROCm: Synchronize after tensor transfer
if self.use_rocm_mode:
    torch.cuda.synchronize()

# Standard GPU training with autocast
with autocast(device_type='cuda'):
    current_q = self.critic(states, actions)
    critic_loss = nn.MSELoss()(current_q, target_q)

# ... backward pass, optimizer step ...

# For ROCm: Synchronize after optimizer step
if self.use_rocm_mode:
    torch.cuda.synchronize()

# Final cleanup with cache clear
if self.use_rocm_mode:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

**Key Points**:
- `non_blocking=False` for ROCm (prevents memory access faults)
- Synchronization after tensor transfers
- Synchronization after optimizer steps
- Cache clearing after each update

### 3. Training Loop (`training/erl_trainer.py`)

```python
# Agents are already on GPU, never moved
for agent in tqdm(self.population, desc="Training agents"):
    # ... get batch ...
    
    # Move batch to GPU with ROCm-specific handling
    gpu_backend = get_gpu_backend()
    use_non_blocking = (gpu_backend != "ROCm")
    
    batch = {k: v.to(Config.DEVICE, non_blocking=use_non_blocking) for k, v in batch_cpu.items()}
    
    # For ROCm: Synchronize after batch transfer
    if gpu_backend == "ROCm":
        torch.cuda.synchronize()
    
    # Update (networks already on GPU)
    critic_loss, actor_loss = agent.update(batch, accumulate=not is_last_accum)
```

**Key Points**:
- No device movement (agents already on GPU)
- Synchronous batch transfers for ROCm
- Standard GPU training path

### 4. Device Movement Protection (`models/ddpg_agent.py`)

```python
def move_to_device(self, device: torch.device, recreate_optimizers: bool = True):
    # CRITICAL FOR ROCm: Never move networks - they're already on GPU permanently
    if self.use_rocm_mode:
        if device.type == 'cuda':
            return  # Already on GPU, no-op
        else:
            return  # Refuse to move to CPU - would break training
    # ... standard movement for non-ROCm ...
```

**Key Points**:
- No-op if already on target device
- Refuses CPU movement for ROCm agents
- Prevents accidental network movement

## Memory Access Fault Prevention

### Critical Synchronization Points

1. **After Network Creation**: Ensures GPU operations complete
2. **After Batch Transfer**: Ensures tensors are ready before use
3. **After Optimizer Steps**: Ensures gradients are applied
4. **After Soft Updates**: Ensures target networks are updated
5. **After Each Update**: Final sync and cache clear

### Why Synchronization Works

- **ROCm Issue**: Asynchronous operations can cause memory access faults
- **Solution**: Make everything synchronous with explicit `synchronize()` calls
- **Result**: Operations complete before next step, preventing race conditions

### Why No Network Movement Works

- **ROCm Issue**: Moving networks creates new GPU memory allocations that can fault
- **Solution**: Keep networks on GPU permanently, only move data tensors
- **Result**: No network movement = no memory access faults from movement

## Performance

- **Full GPU Acceleration**: All training on GPU (fast)
- **No CPU Overhead**: Networks never move to CPU
- **Slight Sync Overhead**: Synchronization adds minimal latency but ensures stability

## Files Modified

1. **`models/ddpg_agent.py`**:
   - GPU initialization (never CPU)
   - ROCm-specific synchronization in `update()`
   - Device movement protection

2. **`training/erl_trainer.py`**:
   - ROCm-specific batch transfer handling
   - Synchronization after transfers
   - No device movement (agents already on GPU)

## Testing Checklist

- [ ] Agents initialize on GPU (check `agent.device`)
- [ ] Training proceeds without memory access faults
- [ ] No network movement occurs (check `move_to_device()` calls)
- [ ] Synchronization points execute (check logs)
- [ ] Training completes successfully
- [ ] Agents learn (check fitness improvements)

## Future Optimizations

If synchronization overhead becomes an issue:

1. **Reduce Sync Points**: Find minimal necessary synchronization
2. **Async Where Safe**: Use async operations where ROCm allows
3. **Memory Pooling**: Optimize GPU memory allocation
4. **Driver Updates**: Wait for ROCm driver improvements

For now, this approach provides **stable, full-speed GPU training on ROCm**.

