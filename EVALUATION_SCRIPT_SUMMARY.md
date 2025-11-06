# Agent Evaluation Script - Summary

## What Was Created

I've created a comprehensive evaluation script that analyzes your best trained agent's trading behavior. Here's what was delivered:

### 1. Main Evaluation Script
**File**: `evaluate_best_agent.py`

A complete Python script that:
- ✅ Loads the best agent from your last training run (or a specified run)
- ✅ Downloads the agent from GCP bucket automatically
- ✅ Evaluates on **3 validation slices** (same approach as during training)
- ✅ Evaluates on **first 125 days of holdout period** (completely unseen data)
- ✅ Tracks every single trade with comprehensive details
- ✅ Exports results to both **text** and **CSV** formats
- ✅ Provides summary statistics (fitness, number of trades, win rate)

### 2. Comprehensive Documentation
**File**: `EVALUATION_GUIDE.md`

Complete user guide including:
- How to run the script
- What happens during evaluation
- Detailed explanation of all output files
- Example outputs and what to expect
- Tips for analyzing results
- Troubleshooting section

### 3. Updated Quick Start Guide
**File**: `QUICK_START.md` (updated)

Added a new section on evaluating your best agent with quick reference commands.

---

## How to Use

### Basic Usage (Evaluate Last Run)
```bash
python evaluate_best_agent.py
```

This automatically uses the run name from `last_run.json`.

### Evaluate Specific Run
```bash
python evaluate_best_agent.py --run-name your-run-name-here
```

Replace `your-run-name-here` with the actual run name (e.g., "azure-thunder-123").

---

## What You Get

### Three Output Files (in `evaluation_results/` directory):

#### 1. Detailed Text Report (`evaluation_{run}_{timestamp}.txt`)
A comprehensive human-readable report with:
- **Summary by Slice**: Statistics for each validation slice and holdout
- **Overall Statistics**: Aggregate metrics across all evaluations
- **Detailed Trade Log**: Every trade with complete information

Example trade entry:
```
Trade #1 (Validation_Slice_1):
  Stock: VXF (ID: 10)
  Entry: 2015-03-15 @ $45.23
  Exit:  2015-04-10 @ $48.91
  Days Held: 18
  Gain/Loss: 8.14%
  Reward: 12.35
  Coefficient: 1.52
  Target: 15.0% ($52.01)
  Exit Type: Active Sell (Target Hit)
  Result: WIN
```

#### 2. Trades CSV (`trades_{run}_{timestamp}.csv`)
Spreadsheet with one row per trade, including:
- Stock ticker and ID
- Entry/exit dates and prices
- Days held
- Gain/loss percentage
- Reward earned
- Exit reason (target hit, max holding period, delisted)
- Whether it was an active sell or auto liquidation
- Win/loss flag

**Perfect for**: Excel analysis, filtering, pivot tables, statistics

#### 3. Summary CSV (`summary_{run}_{timestamp}.csv`)
One row per evaluation slice with:
- Fitness score
- Number of trades
- Win/loss breakdown
- Win rate percentage
- Average reward per trade
- Activity metrics (days with/without positions)
- Penalties applied

**Perfect for**: Comparing performance across different periods

---

## Trade Information Captured

For each trade, you'll see:

✅ **Stock**: Ticker symbol (e.g., "VXF", "DVY", "QQQ")
✅ **Entry Date**: When the position was opened
✅ **Entry Price**: Price paid
✅ **Exit Date**: When the position was closed
✅ **Exit Price**: Price received
✅ **Days Held**: How long the position was held
✅ **Gain/Loss %**: Percentage gain or loss
✅ **Reward**: Actual reward earned (factoring in coefficient and penalties)
✅ **Coefficient**: Agent's confidence/position size
✅ **Target**: Agent's target gain percentage
✅ **Exit Type**:
  - **Active Sell (target_hit)**: Agent sold at target price ✨
  - **Auto Liquidation (max_holding_period)**: Held for 20 days, force closed
  - **Auto Liquidation (stock_delisted)**: Stock data unavailable, force closed
✅ **Win/Loss**: Boolean flag for easy filtering

---

## Evaluation Methodology

### Validation Slices (3 slices)
Each slice consists of:
- **504 days of context** (2 years) - for agent to observe market
- **125 days of trading** - where agent can open new positions
- **20 days of settlement** - to close remaining positions

Slices are randomly sampled from the **interim validation set** (the same data used during training for walk-forward validation).

### Holdout Evaluation (1 slice)
- Takes the first **125 days** (+20 settlement) from the completely unseen **holdout set**
- This is the true test of generalization

### Fitness Calculation
```
fitness = total_reward - inaction_penalty - zero_trades_penalty
```

Where:
- `total_reward`: Sum of all trade rewards
- `inaction_penalty`: 5 points per day without an open position
- `zero_trades_penalty`: 10,000 points if no trades were made

---

## Example Workflow

1. **Run evaluation**:
   ```bash
   python evaluate_best_agent.py
   ```

2. **Check the text report** for overall performance:
   ```bash
   cat evaluation_results/evaluation_*.txt
   ```

3. **Analyze trades in Excel**:
   - Open `trades_*.csv` in Excel
   - Filter for winning trades: `is_win = TRUE`
   - Filter for active sells: `is_active_sell = TRUE`
   - Calculate average gain for winners vs losers
   - Identify best performing stocks

4. **Compare across slices**:
   - Open `summary_*.csv`
   - See which evaluation period performed best
   - Check consistency of win rate across slices

---

## Key Insights You Can Extract

### From the Trades CSV:

1. **Best performing stocks**:
   ```python
   import pandas as pd
   trades = pd.read_csv('trades_*.csv')
   by_stock = trades.groupby('stock_ticker')['gain_pct'].mean()
   print(by_stock.sort_values(ascending=False).head(10))
   ```

2. **Active sell vs auto liquidation performance**:
   ```python
   active_sells = trades[trades['is_active_sell'] == True]
   auto_liq = trades[trades['is_auto_liquidation'] == True]
   print(f"Active sells win rate: {active_sells['is_win'].mean()*100:.1f}%")
   print(f"Auto liquidation win rate: {auto_liq['is_win'].mean()*100:.1f}%")
   ```

3. **Average holding period for winners vs losers**:
   ```python
   winners = trades[trades['is_win'] == True]
   losers = trades[trades['is_win'] == False]
   print(f"Winners avg hold: {winners['days_held'].mean():.1f} days")
   print(f"Losers avg hold: {losers['days_held'].mean():.1f} days")
   ```

### From the Summary CSV:

- **Consistency**: Is the win rate similar across all slices?
- **Generalization**: How does holdout performance compare to validation?
- **Activity**: Is the agent making enough trades?

---

## What Makes This Script Powerful

1. **Automatic Agent Loading**: No manual downloading from GCP required
2. **Complete Trade History**: Every trade is logged with full context
3. **Multiple Output Formats**: Text for reading, CSV for analysis
4. **Validation & Holdout**: See both in-sample validation and out-of-sample holdout performance
5. **Exit Type Tracking**: Know if the agent is hitting targets or being force-liquidated
6. **Stock Ticker Names**: See actual stock symbols, not just IDs
7. **Ready for Excel**: CSV files can be directly opened and analyzed

---

## Next Steps

After running the evaluation:

1. **Analyze the results** to understand your agent's behavior
2. **Compare across runs** (run evaluation on multiple runs to find the best)
3. **Identify patterns** in winning vs losing trades
4. **Fine-tune training** based on insights (e.g., if targets are too ambitious, adjust rewards)
5. **Run champion evaluation** using `evaluate_champions.py` to compare all your runs

---

## Files Created

```
eigen2/
├── evaluate_best_agent.py          # Main evaluation script
├── EVALUATION_GUIDE.md              # Comprehensive usage guide
├── EVALUATION_SCRIPT_SUMMARY.md     # This file
└── QUICK_START.md                   # Updated with evaluation section
```

After running evaluation, you'll also have:
```
eigen2/
└── evaluation_results/
    ├── evaluation_{run}_{timestamp}.txt    # Detailed report
    ├── trades_{run}_{timestamp}.csv         # All trades
    └── summary_{run}_{timestamp}.csv        # Summary by slice
```

---

## Questions?

See the [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed documentation, examples, and troubleshooting.

Happy analyzing! 📊
