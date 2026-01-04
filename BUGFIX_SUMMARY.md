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
2. `training/erl_trainer.py` - CUDA tensor fix + missing dict key fix + ROCm pin_memory fix
3. `compare_context_windows.py` - Numpy serialization fix

---

### 6. ROCm Memory Access Fault with DataLoader pin_memory (training/erl_trainer.py:1383-1410)
**Problem**: `Memory access fault by GPU node-2` error when using DataLoader with `pin_memory=True` and multiprocessing workers on ROCm.

**Root Cause**: ROCm's `pin_memory` implementation has compatibility issues with DataLoader multiprocessing workers. When workers try to pin memory, they access GPU memory without proper context, causing memory access faults.

**Solution**: Detect ROCm backend and disable `pin_memory` when `num_workers > 0`, but keep it enabled for `num_workers=0` (main process):
```python
from utils.device import get_gpu_backend
gpu_backend = get_gpu_backend()
use_pin_memory = True
if gpu_backend == "ROCm" and num_workers > 0:
    use_pin_memory = False
    print(f"  [ROCm] Disabling pin_memory for DataLoader workers (ROCm compatibility)")
```

**Impact**: Prevents memory access faults on ROCm while maintaining performance for CUDA and single-process scenarios.

---

## Testing Recommendations

1. **Parallel Validation**: Run training with 48+ workers to ensure no CUDA errors
2. **Global HOF**: Promote agent to Global50 to test JSON serialization
3. **Context Windows**: Check that only existing leagues are attempted (no 404s)
4. **Docker**: Test in restricted container environment
5. **ROCm DataLoader**: Verify training runs without memory access faults on AMD GPUs
