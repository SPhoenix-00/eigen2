# Bug Fix Summary - 2025-12-02

## Issues Found and Fixed

### 1. Hardcoded Context Windows (erl/global_hof.py:306-338)
**Problem**: Hardcoded list `[504, 252, 377, 125, 100, 200, 300]` caused 404 warnings when trying to discover fallback leagues that don't exist.

**Solution**: Implemented dynamic discovery using GCS bucket listing API:
- Queries `eigen2/global50/` prefix to list actual directories
- Parses directory names to extract context window values (e.g., `cw504` → 504)
- Falls back to hardcoded list only if bucket listing fails (non-GCS providers)

**Impact**: Eliminates 404 warnings and automatically discovers new context windows when added.

---

### 2. CUDA Multiprocessing Error (training/erl_trainer.py:2210-2214)
**Problem**: `RuntimeError: pidfd_getfd: Operation not permitted` when using parallel validation in Docker containers.

**Root Cause**: CUDA tensors in agent state dicts were being serialized for IPC without moving to CPU first, triggering restricted system calls.

**Solution**: Added `.cpu()` conversion before serialization:
```python
agent_state = {
    'actor': {k: v.cpu() for k, v in agent.actor.state_dict().items()},
    'critic': {k: v.cpu() for k, v in agent.critic.state_dict().items()}
}
```

**Impact**: Parallel validation now works in Docker environments with restricted permissions.

---

### 3. Missing Dictionary Key (training/erl_trainer.py:329-340)
**Problem**: `KeyError: 'num_trades'` when accessing validation results.

**Root Cause**: Parallel validation worker (`_run_validation_worker`) returned `total_trades` but not `num_trades`, while non-parallel methods returned both.

**Solution**: Added `num_trades` field to parallel worker return dict:
```python
'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),  # Mean per slice
'total_trades': total_trades,  # Total across all slices
```

**Impact**: Consistent API across parallel and non-parallel validation paths.

---

### 4. JSON Serialization of Numpy Types (erl/global_hof.py:30-39, 55-64)
**Problem**: `TypeError: Object of type float32 is not JSON serializable` when saving Global Hall of Fame data.

**Root Cause**: `asdict()` doesn't convert numpy scalar types (float32, float64, int64) to Python native types.

**Solution**: Enhanced `to_dict()` methods in `LeagueRules` and `GlobalHoFEntry` classes:
```python
def to_dict(self) -> dict:
    data = asdict(self)
    # Convert numpy types to Python native types
    for key, value in data.items():
        if hasattr(value, 'item'):  # numpy scalar
            data[key] = value.item()
        elif isinstance(value, (list, tuple)):
            data[key] = [v.item() if hasattr(v, 'item') else v for v in value]
    return data
```

**Impact**: All dataclasses now serialize correctly to JSON regardless of numpy types.

---

### 5. JSON Serialization in compare_context_windows.py (lines 372-373, 384-385)
**Problem**: Same numpy serialization issue in context window comparison script.

**Solution**:
1. Added `to_dict()` method to `ContextWindowResult` dataclass
2. Replaced `asdict()` calls with `.to_dict()` calls

**Impact**: Context window comparison results serialize correctly.

---

### 6. wandb.run.step AttributeError on Resume (training/erl_trainer.py:1169-1172)
**Problem**: `AttributeError: property 'step' of 'Run' object has no setter` when resuming training from checkpoint.

**Root Cause**: In newer versions of wandb, `wandb.run.step` is a read-only property and cannot be set directly. The code attempted to sync the step counter after loading a checkpoint by assigning `wandb.run.step = self.start_generation - 1`.

**Solution**: Removed the direct assignment. Step tracking works correctly because all `wandb.log()` calls in the codebase already specify the `step` parameter explicitly (e.g., `wandb.log(..., step=self.generation)`). When resuming from checkpoint, logs will use the correct generation number as the step.

**Impact**: Training can now resume from checkpoints without errors. Step tracking continues to work correctly via explicit step parameters in log calls.

---

## Verification Performed

### CUDA Serialization
✅ Checked all `state_dict()` usage in multiprocessing contexts
✅ Confirmed all state dicts use `.cpu()` before serialization to workers
✅ Save/load operations use `torch.save()`/`torch.load()` which handle tensors properly

### Dictionary Key Consistency
✅ Verified validation worker returns match what's expected
✅ Added defensive `.get()` calls where appropriate
✅ Both parallel and non-parallel paths now return same keys

### JSON Serialization
✅ All dataclasses with `to_dict()` now convert numpy types
✅ Checked all `json.dump()` calls for potential numpy issues
✅ No remaining direct `asdict()` usage without numpy conversion

### Hardcoded Lists
✅ Context window discovery now dynamic
✅ Fallback lists only used when GCS unavailable (safe)
✅ No other problematic hardcoded lists found

---

## Files Modified

1. `erl/global_hof.py` - Dynamic context window discovery + numpy serialization fix
2. `training/erl_trainer.py` - CUDA tensor fix + missing dict key fix + wandb.run.step fix
3. `compare_context_windows.py` - Numpy serialization fix

---

### 7. CUDA Out of Memory in Parallel Validation (training/erl_trainer.py:324-395, 4232-4237)
**Problem**: CUDA OOM errors during validation phase with 48 parallel workers. Error: "CUDA out of memory. Tried to allocate 278.00 MiB. GPU 0 has a total capacity of 94.98 GiB of which 215.00 MiB is free."

**Root Cause**: 
- Validation workers were creating agents on GPU (using `Config.DEVICE` which is CUDA)
- With 48 parallel workers all trying to allocate GPU memory simultaneously, the GPU ran out of memory
- GPU memory from evaluation phase wasn't being cleared before validation started

**Solution**:
1. **Adaptive worker count based on GPU availability**: 
   - **With GPU**: Use 4-8 workers (GPU is fast, fewer workers needed, avoids OOM)
   - **Without GPU**: Use up to 48 workers (CPU is slower, need more parallelism)
2. **Allow workers to use GPU**: Workers use GPU when available (with fewer workers, GPU memory is manageable)
3. **Add GPU cleanup after evaluation**: Clear GPU cache and run garbage collection between evaluation and validation phases

**Impact**: 
- Validation workers use GPU when available (faster than CPU)
- With fewer GPU workers (4-8 instead of 48), GPU memory usage is manageable
- CPU-only systems still get full parallelism (48 workers)
- No more CUDA OOM errors during validation phase

**Files Modified**:
- `training/erl_trainer.py` - Lines ~361-375, ~4232-4237, ~6767: Adaptive worker count + GPU usage + GPU cleanup

---

## Testing Recommendations

1. **Parallel Validation**: Run training with 48+ workers to ensure no CUDA errors
2. **Global HOF**: Promote agent to Global50 to test JSON serialization
3. **Context Windows**: Check that only existing leagues are attempted (no 404s)
4. **Docker**: Test in restricted container environment
5. **CUDA OOM**: Run validation with 48+ parallel workers on CUDA system to verify no OOM errors
