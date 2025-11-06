# Data Loader Fixes - Pickle Format Compatibility

## Issues Fixed

### 1. **Numpy Array Handling** (Critical Bug)
**Problem**: When loading pickle files, cell values are numpy arrays. Calling `pd.isna()` on arrays returns an array of booleans, causing "ValueError: ambiguous truth value" error.

**Fix**: Reordered type checks in `parse_cell_data()`:
- Check for `None` first
- Handle `np.ndarray` directly before calling `pd.isna()`
- Wrap `pd.isna()` in try-except for scalar values only

**Location**: `data/loader.py:75-163`

### 2. **Shape Mismatch** (Configuration Issue - NOT FIXED)
**Problem**: Pickle file has 698 columns, but `Config.TOTAL_COLUMNS` is 670. This mismatch will cause:
```
RuntimeError: shape '[669, 5, 504]' is invalid for input of size 1758960
```

**Root Cause**: The configuration doesn't match the actual data file.

**Solution Required**: Update `Config.TOTAL_COLUMNS` in `utils/config.py` to match the actual pickle file:
```python
TOTAL_COLUMNS = 698  # Changed from 670
```

**Why This Error Is Important**: This mismatch indicates a fundamental incompatibility between:
- The data file being used
- The configuration the agent was trained with

This error **should be thrown** to alert you that there's a mismatch. Either:
1. Update the config to match the new data file (698 columns), OR
2. Use the correct data file that matches the trained agent (670 columns)

**Note**: This fix was intentionally reverted - the error is important and should not be silently suppressed.

### 3. **Missing Attributes** (Compatibility Issue)
**Problem**: Evaluation script expected attributes that weren't being set:
- `normalization_stats`
- `train_end_idx`
- `interim_val_start_idx`
- `interim_val_end_idx`
- `val_start_idx`

**Fix**:
- Added attributes to `__init__()` with `None` defaults
- `load_and_prepare()` now stores normalization stats as attribute
- `create_train_val_split()` now stores split point indices

**Location**: `data/loader.py:37-43, 240-244, 361`

### 4. **Runtime Warnings** (Minor Issue)
**Problem**: "Mean of empty slice" warnings when columns have all NaN values.

**Fix**: Added warning suppression in `compute_normalization_stats()`:
- Warnings are expected for columns with no data
- Code handles NaN means/stds appropriately
- Suppressing prevents console clutter

**Location**: `data/loader.py:314-319`

## Changes Summary

### Modified Methods

1. **`__init__()`** - Added new attribute initialization
2. **`parse_cell_data()`** - Fixed numpy array handling
3. **`create_train_val_split()`** - Stores split indices as attributes
4. **`load_and_prepare()`** - Stores normalization stats as attribute
5. **`compute_normalization_stats()`** - Suppresses expected warnings

### Not Changed

- **`extract_features()`** - Reverted column limiting (intentionally allows shape mismatch error)

### Backward Compatibility

✅ **100% Backward Compatible** with existing code:
- `main.py` training pipeline unchanged
- `ERLTrainer` works without modifications
- All existing methods have same signatures
- Only additions, no breaking changes

### New Attributes

```python
# Set by __init__
self.normalization_stats = None
self.train_end_idx = None
self.interim_val_start_idx = None
self.interim_val_end_idx = None
self.val_start_idx = None

# Populated by load_and_prepare() and create_train_val_split()
```

## Testing

### Verified Compatibility

✅ Pickle file loading works
✅ Numpy array cells parsed correctly
⚠️ Shape mismatch error will occur (requires config update)
✅ No runtime warnings (NaN warnings suppressed)
✅ Training pipeline unchanged
⚠️ Evaluation script requires Config.TOTAL_COLUMNS update

### Before vs After

**Before**:
- ❌ Crashed on pickle numpy arrays
- ⚠️ Used all columns (698) - causes shape mismatch if config is 670
- ❌ Missing attributes for evaluation
- ⚠️ Noisy warnings

**After**:
- ✅ Handles numpy arrays correctly
- ⚠️ Uses all columns (698) - **will throw error if Config.TOTAL_COLUMNS != 698**
- ✅ All attributes available
- ✅ Clean output (warnings suppressed)

## Impact on Evaluation

**IMPORTANT**: Before running evaluation, you must update `Config.TOTAL_COLUMNS` to match your pickle file:

```python
# In utils/config.py, change:
TOTAL_COLUMNS = 670  # OLD - doesn't match pickle
# To:
TOTAL_COLUMNS = 698  # NEW - matches pickle file
```

After updating config, the evaluation script will work:
1. Loads data with matching column count (698)
2. Agents receive expected input shape
3. No shape mismatch in neural networks
4. Clean execution without warnings

## Impact on Training

**No impact** - training continues to work exactly as before:
1. ERLTrainer uses same data loader interface
2. Normalization stats computed the same way
3. Train/val split works identically
4. Column count matches agent expectations

## Column Count Mismatch - ACTION REQUIRED

- **Pickle file**: 698 columns total
- **Config.TOTAL_COLUMNS**: 670 (doesn't match!)
- **Problem**: Shape mismatch will cause evaluation to fail

**You must choose ONE of these solutions**:

### Option 1: Update Config (Recommended if pickle is correct)
```python
# In utils/config.py, line 16:
TOTAL_COLUMNS = 698  # Changed from 670
```

### Option 2: Use Different Pickle File
Find and use a pickle file with exactly 670 columns to match the existing config.

### Why Not Auto-Fix?
The shape mismatch error is important - it signals a fundamental incompatibility between your data and trained agents. Silently truncating columns could:
- Hide important data quality issues
- Break trained agents expecting specific columns
- Cause subtle bugs that are hard to debug

**The error forces you to make a conscious decision about which data to use.**

## Files Modified

1. `data/loader.py` - All fixes applied
2. `evaluate_best_agent.py` - Uses new attributes (already created)

## Files NOT Modified

- `main.py` - No changes needed
- `training/erl_trainer.py` - No changes needed
- `utils/config.py` - No changes needed
- `environment/trading_env.py` - No changes needed
- `models/*` - No changes needed

## Validation

### Step 1: Update Config First
```python
# In utils/config.py, line 16:
TOTAL_COLUMNS = 698  # Changed from 670
```

### Step 2: Test Data Loading
```bash
python -c "from data.loader import StockDataLoader; loader = StockDataLoader(); loader.load_and_prepare()"
```

Expected output:
- Data shape: (3888, 698, 5)
- No shape errors
- Clean execution

### Step 3: Test Evaluation
```bash
python evaluate_best_agent.py --run-name celestial-music-66
```

Expected output:
- Agent loads successfully
- No shape mismatch errors
- Evaluation completes
- Results exported to evaluation_results/

### If You DON'T Update Config

You will see:
```
RuntimeError: shape '[669, 5, 504]' is invalid for input of size 1758960
```

This is **intentional** - it alerts you to the mismatch.

## Summary

Data loader issues fixed:
1. ✅ Pickle format fully supported (numpy arrays handled)
2. ✅ Missing attributes added (evaluation script compatible)
3. ✅ Warnings suppressed (clean output)
4. ✅ No breaking changes (training unchanged)
5. ⚠️ **Shape mismatch intentionally NOT fixed** - requires config update

**ACTION REQUIRED**: Update `Config.TOTAL_COLUMNS` from 670 to 698 to match your pickle file.

The data loader now correctly handles pickle files with numpy arrays. The shape mismatch error will guide you to fix the config/data incompatibility rather than silently hiding it.
