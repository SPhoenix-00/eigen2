# Validation Logic Comparison: Normal vs --local Mode

## Summary
✅ **The validation logic is now similar between normal and --local modes**, with only minor differences in return dictionary fields.

## Core Validation Logic - IDENTICAL

### 1. Episode Execution
Both modes use **time-travel batching**:
- Pre-fetch all states for trading period
- Batch inference (chunked for memory efficiency)
- Fast replay of actions through environment
- Same settlement period handling

### 2. Fitness Calculation per Slice
Both apply the same penalties/bonuses in the same order:
1. ✅ **Zero trades penalty**: `fitness -= episode_info['zero_trades_penalty']`
2. ✅ **Win rate bonus**: `(win_rate_pct - threshold)^2` if win_rate > threshold and trades >= min
3. ✅ **Zero-trades gradient**: `fitness += max_coefficient_during_episode` (for agents that don't trade)

### 3. Fitness Aggregation
Both use the same aggregation methods:
- **Multi-mode**: `penalized_median = median - (0.5 * std)`
- **Normal mode**: `pessimistic = 0.4 * mean + 0.6 * min`

### 4. Metric Aggregation
Both calculate the same metrics:
- ✅ ROI: `(total_raw_pnl / total_peak_capital * 100)`
- ✅ Global win rate: `total_wins / total_trades`
- ✅ Total trades: `sum across all slices`
- ✅ Quality count: `count of trades with gain_pct >= threshold`
- ✅ Expectancy: Uses same `calculate_expectancy()` method

## Differences

### 1. Return Dictionary Fields

**Parallel Mode** (`_run_validation_worker`) returns:
```python
{
    'fitness': validation_fitness,
    'fitness_all_slices': fitness_scores,
    'fitness_mean': mean_score,
    'fitness_min': min_score,
    'roi': roi,
    'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
    'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
    'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
    'avg_reward_per_trade': np.mean([r['avg_reward_per_trade'] for r in slice_results]),
    'raw_pnl': total_raw_pnl,
    'total_investment': total_investment,
    'total_trades': total_trades,
    'win_rate': global_win_rate,
    'expectancy': expectancy,
    'quality_count': quality_count,
    'quality_roi': quality_roi,  # ⚠️ Only in parallel mode
    'sample_trade': sample_trade
}
```

**Local Mode** (`_validate_agent_optimized`) returns:
```python
{
    'fitness': validation_fitness,
    'fitness_all_slices': fitness_scores,
    'fitness_mean': float(np.mean(fitness_scores)),
    'fitness_min': float(np.min(fitness_scores)),
    'win_rate': global_win_rate,
    'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
    'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
    'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
    'avg_reward_per_trade': float(np.mean([r['avg_reward_per_trade'] for r in slice_results])),
    'raw_pnl': total_raw_pnl,
    'total_investment': sum(r['total_investment'] for r in slice_results),
    'roi': roi,
    'expectancy': expectancy,
    'sample_trade': sample_trade,
    'total_trades': total_trades,
    'quality_count': quality_count
    # ⚠️ Missing: 'quality_roi'
}
```

**Difference**: Parallel mode includes `quality_roi` (average ROI of quality trades), local mode doesn't.

### 2. Device Usage
- **Normal mode**: Validation workers use **CPU** (after recent fix to avoid GPU OOM)
- **Local mode**: Uses **GPU** (or CPU with torch.compile if GPU unavailable)

### 3. Execution Model
- **Normal mode**: Parallel workers (ProcessPoolExecutor)
- **Local mode**: Sequential (single process, single environment instance)

## Verification

Both modes:
- ✅ Use same validation slices (`self.trainer.current_generation_val_slices`)
- ✅ Use same `use_penalized_median` flag (`self.trainer.multi_mode`)
- ✅ Use same quality threshold
- ✅ Calculate fitness identically per slice
- ✅ Aggregate fitness identically
- ✅ Calculate ROI identically
- ✅ Calculate expectancy identically

## Recommendation

The validation logic is **functionally equivalent** between modes. The only difference is:
1. `quality_roi` field (not used downstream, only for logging)
2. Device/execution model (implementation detail, not logic difference)

**Conclusion**: ✅ Validation logic is similar enough for practical purposes. The core fitness calculation, aggregation, and metrics are identical.

