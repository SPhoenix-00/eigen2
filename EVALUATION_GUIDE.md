# Agent Evaluation Guide

This guide explains how to use the `evaluate_best_agent.py` script to analyze the trading behavior of your best trained agent.

## Overview

The evaluation script performs a comprehensive analysis of your best agent by:

1. **Loading the best agent** from your last training run (using `last_run.json`) or a specified run
2. **Evaluating on 3 validation slices** - randomly sampled from the interim validation set (same approach used during training)
3. **Evaluating on the first 125 days of holdout** - completely unseen data
4. **Tracking every trade** with detailed information:
   - Stock ticker and ID
   - Entry and exit dates
   - Entry and exit prices
   - Days held
   - Gain/loss percentage
   - Reward earned
   - Exit type (active sell vs auto liquidation)
5. **Generating comprehensive reports** in both text and CSV formats
6. **Providing summary statistics** including fitness, number of trades, and win rate

## Requirements

Before running the evaluation:

1. **Completed training run** - You need to have completed at least one training run
2. **GCP credentials** - Set up your Google Cloud Storage credentials (see `GCP_SETUP_GUIDE.md`)
3. **Environment variables** - Ensure the following are set:
   ```bash
   export CLOUD_PROVIDER=gcs
   export CLOUD_BUCKET=your-bucket-name
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-credentials.json
   ```

## Usage

### Option 1: Evaluate Last Run (Automatic)

If you have a `last_run.json` file from your most recent training run:

```bash
python evaluate_best_agent.py
```

This will automatically detect and evaluate the agent from your last run.

### Option 2: Evaluate Specific Run

To evaluate a specific run by name:

```bash
python evaluate_best_agent.py --run-name azure-thunder-123
```

Replace `azure-thunder-123` with your actual run name (you can find this in wandb or your checkpoint directories).

## What Happens During Evaluation

### 1. Agent Loading
The script will:
- Read the run name (from `last_run.json` or command line)
- Download the best agent from your GCP bucket (if not already cached locally)
- Load the trained agent weights

### 2. Data Loading
- Loads the full dataset (`Eigen2_Master_PY_OUTPUT.pkl`)
- Extracts stock ticker names (108 investable stocks)
- Sets up normalization statistics
- Identifies validation and holdout periods

### 3. Validation Slice Evaluation
Generates 3 random slices from the interim validation set, each consisting of:
- **504 days of context** (2 years) - for the agent to observe market conditions
- **125 days of trading** - where the agent can open new positions
- **20 days of settlement** - to close remaining positions

For each slice, the agent makes trading decisions and all trades are tracked.

### 4. Holdout Evaluation
Takes the first 125 days (+20 settlement) from the completely unseen holdout set and evaluates the agent's performance.

### 5. Report Generation
Creates three output files in the `evaluation_results/` directory:

## Output Files

### 1. Detailed Text Report
**File**: `evaluation_{run_name}_{timestamp}.txt`

Contains:
- **Summary by Slice**: Fitness, trades, win rate, penalties for each validation slice and holdout
- **Overall Statistics**: Aggregate metrics across all slices
- **Detailed Trade Log**: Every single trade with complete information:
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

### 2. Trades CSV
**File**: `trades_{run_name}_{timestamp}.csv`

A spreadsheet with one row per trade, including all fields:
- `slice`: Which evaluation slice (Validation_Slice_1/2/3 or Holdout)
- `stock_ticker`: Stock symbol (e.g., "VXF", "DVY")
- `stock_id`: Internal stock ID (0-107)
- `entry_date`, `entry_price`: When and at what price the position was opened
- `exit_date`, `exit_price`: When and at what price the position was closed
- `days_held`: How long the position was held
- `gain_pct`: Percentage gain or loss
- `reward`: Reward earned from this trade
- `coefficient`: Agent's confidence/position size
- `sale_target_pct`, `sale_target_price`: Target gain percentage and price
- `exit_reason`: Why the position closed (`target_hit`, `max_holding_period`, `stock_delisted`)
- `is_active_sell`: Boolean - did the agent sell at target?
- `is_auto_liquidation`: Boolean - was it forced liquidation?
- `is_win`: Boolean - was the trade profitable?

**Perfect for**: Excel analysis, filtering winning trades, calculating statistics

### 3. Summary CSV
**File**: `summary_{run_name}_{timestamp}.csv`

One row per evaluation slice with aggregate statistics:
- `slice_name`: Name of the slice
- `start_date`, `end_date`: Period covered
- `fitness`: Final fitness score
- `total_reward`: Sum of all rewards
- `num_trades`: Number of trades executed
- `num_wins`, `num_losses`: Win/loss breakdown
- `win_rate`: Percentage of winning trades
- `avg_reward_per_trade`: Average reward per trade
- `days_with_positions`, `days_without_positions`: Activity metrics
- `inaction_penalty_applied`: Penalty for days without positions
- `zero_trades_penalty`: Penalty if no trades were made

**Perfect for**: Comparing performance across different evaluation periods

## Example Output

```bash
$ python evaluate_best_agent.py

================================================================================
Evaluating Best Agent from Run: azure-thunder-123
================================================================================

Loading data...
✓ Data loaded: 4324 days
  Training: days 0-3822
  Interim Validation: days 3822-4074
  Holdout: days 4074-4324
✓ Loaded 108 investable stock tickers

Loading best agent from run: azure-thunder-123
  Downloading from cloud...
✓ Best agent loaded from checkpoints/azure-thunder-123/best_agent.pth

================================================================================
EVALUATION PROCESS
================================================================================

Evaluating on 3 Validation Slices...
--------------------------------------------------------------------------------

Validation_Slice_1:
  Period: 2023-01-15 to 2023-07-28
  Trading days: 125, Settlement days: 20
  Fitness: 45.32
  Trades: 23, Win Rate: 65.2%

Validation_Slice_2:
  Period: 2023-03-22 to 2023-09-15
  Trading days: 125, Settlement days: 20
  Fitness: 38.17
  Trades: 19, Win Rate: 57.9%

Validation_Slice_3:
  Period: 2023-05-10 to 2023-11-30
  Trading days: 125, Settlement days: 20
  Fitness: 52.89
  Trades: 26, Win Rate: 69.2%

================================================================================
Evaluating on Holdout Period (First 125 Days)...
--------------------------------------------------------------------------------

Holdout:
  Period: 2024-01-05 to 2024-07-20
  Trading days: 125, Settlement days: 20
  Fitness: 41.23
  Trades: 21, Win Rate: 61.9%

================================================================================
RESULTS EXPORTED
================================================================================
Detailed Report: evaluation_results/evaluation_azure-thunder-123_20250106_143022.txt
Trades CSV: evaluation_results/trades_azure-thunder-123_20250106_143022.csv
Summary CSV: evaluation_results/summary_azure-thunder-123_20250106_143022.csv
================================================================================

================================================================================
EVALUATION COMPLETE!
================================================================================
```

## Understanding the Results

### Fitness Score
The fitness score is calculated as:
```
fitness = total_reward - inaction_penalty - zero_trades_penalty
```

Where:
- `total_reward`: Sum of rewards from all closed positions
- `inaction_penalty`: 5 points per day without an open position
- `zero_trades_penalty`: 10,000 points if no trades were made

### Trade Exit Types

1. **Active Sell (target_hit)**: The stock hit the agent's target price and was sold for profit
2. **Auto Liquidation (max_holding_period)**: Position held for 20 days (max) and automatically closed
3. **Auto Liquidation (stock_delisted)**: Stock data became unavailable and position was force-closed

### Win Rate
Calculated as: `number_of_winning_trades / total_trades`

A trade is a "win" if `gain_pct >= 0`

## Tips for Analysis

### Using the CSV Files

1. **Filter winning trades**:
   ```python
   import pandas as pd
   trades = pd.read_csv('trades_run_timestamp.csv')
   winning_trades = trades[trades['is_win'] == True]
   print(f"Average winning trade: {winning_trades['gain_pct'].mean():.2f}%")
   ```

2. **Analyze exit types**:
   ```python
   active_sells = trades[trades['is_active_sell'] == True]
   print(f"Active sells: {len(active_sells)} ({len(active_sells)/len(trades)*100:.1f}%)")
   ```

3. **Best performing stocks**:
   ```python
   by_stock = trades.groupby('stock_ticker')['gain_pct'].mean().sort_values(ascending=False)
   print(by_stock.head(10))
   ```

### Interpreting Performance

- **High fitness**: Agent is profitable and active
- **High win rate** (>60%): Agent is selective and makes good entries
- **Many active sells**: Agent's targets are realistic and being hit
- **Few trades**: Agent may be too conservative (high inaction penalty)
- **Many auto liquidations**: Agent's targets may be too ambitious

## Troubleshooting

### "No run name specified and last_run.json not found"
- Either specify a run name with `--run-name` or ensure you have a `last_run.json` from a completed training run

### "Best agent not found"
- Check that your GCP credentials are set up correctly
- Verify the run name is correct
- Ensure the run completed successfully and saved a best agent

### "Not enough interim validation data"
- This shouldn't happen with the standard data, but indicates your data file may be too small

## Next Steps

After analyzing your agent's performance:

1. **Compare across runs**: Evaluate multiple runs to find the best performing agent
2. **Analyze trade patterns**: Look for common characteristics in winning vs losing trades
3. **Fine-tune hyperparameters**: Use insights from evaluation to adjust training parameters
4. **Run highlander evaluation**: Use `evaluate_champions.py` to compare best agents from all runs

## Related Scripts

- `main.py`: Main training script
- `evaluate_champions.py`: Compare best agents from all runs
- `download_metrics.py`: Download wandb metrics for analysis

## Questions?

See the main README.md or check the other documentation files:
- `QUICK_START.md`: Getting started with training
- `GCP_SETUP_GUIDE.md`: Setting up cloud storage
- `WANDB_INTEGRATION.md`: Setting up experiment tracking
