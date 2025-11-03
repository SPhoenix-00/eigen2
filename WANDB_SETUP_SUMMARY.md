# Weights & Biases Integration - Setup Summary

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `wandb>=0.16.0` (already added to requirements.txt)

### 2. Login to W&B

**One-time setup on your machine/RunPod:**

```bash
wandb login
```

Or set API key as environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

**Your W&B Account:**
- Username: `eigen2`
- Entity: `eigen2` (eigen2-self)
- API URL: https://api.wandb.ai
- Dashboard: https://wandb.ai/eigen2/eigen2-trading

### 3. Run Training

```bash
python main.py
```

W&B will automatically:
- Initialize tracking
- Log metrics every generation
- Create dashboard at https://wandb.ai/eigen2/eigen2-trading
- Display run URL in console

## What's Being Tracked

### Every Generation
- **Fitness:** mean, max, min, std, best_ever
- **Training:** actor_loss, critic_loss, generation_time
- **Buffer:** size, utilization, capacity

### Every 5 Generations (LOG_FREQUENCY)
- **Validation:** fitness, win_rate, total_trades, winning_trades

### Config (One-time)
All hyperparameters logged automatically:
- population_size, num_generations, buffer_size
- actor_lr, critic_lr, batch_size
- trading_period_days, max_holding_period
- num_stocks, loss_penalty_multiplier

## Files Changed

1. **[requirements.txt](requirements.txt)** - Added `wandb>=0.16.0`
2. **[training/erl_trainer.py](training/erl_trainer.py)** - Added wandb logging throughout

## Example Output

```
wandb: Currently logged in as: eigen2-self
wandb: Tracking run with wandb version 0.16.0
wandb: Syncing run erl-25gen
wandb: ⭐️ View project at https://wandb.ai/eigen2/eigen2-trading
wandb: 🚀 View run at https://wandb.ai/eigen2/eigen2-trading/runs/abc123

Generation 1/25
============================================================
Fitness Summary:
  Mean: 1050.32
  Max: 1320.45
...

wandb: Logged 12 metrics
```

## Verify It's Working

1. **Check console output** - Look for `wandb: Syncing run` message
2. **Open W&B dashboard** - Go to the URL shown in console
3. **View metrics** - Should see charts updating in real-time

## Disable W&B (Optional)

```bash
# Disable tracking
export WANDB_MODE=disabled
python main.py

# Or run in offline mode (log locally, sync later)
export WANDB_MODE=offline
python main.py
```

## Full Documentation

See [WANDB_INTEGRATION.md](WANDB_INTEGRATION.md) for:
- Complete metrics list
- Advanced features (sweeps, artifacts, comparison)
- Troubleshooting
- Best practices

## Next Steps

1. **Install wandb** on your RunPod instance
2. **Login** with your API key
3. **Start training** and watch metrics in real-time at https://wandb.ai/eigen2/eigen2-trading

That's it! W&B is now fully integrated and will track all your training runs automatically.
