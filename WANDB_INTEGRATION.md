# Weights & Biases Integration

## Overview

Weights & Biases (wandb) is now integrated into the ERL training pipeline, providing real-time tracking, visualization, and analysis of training metrics.

**W&B Dashboard:** https://wandb.ai/eigen2/eigen2-trading

## Features

### Tracked Metrics

#### Fitness Metrics (Every Generation)
- `fitness/mean` - Average fitness across population
- `fitness/max` - Best fitness in current generation
- `fitness/min` - Worst fitness in current generation
- `fitness/std` - Standard deviation of fitness
- `fitness/best_ever` - Best fitness achieved so far

#### Training Metrics (Every Generation)
- `train/actor_loss` - Actor network loss
- `train/critic_loss` - Critic network loss
- `training/generation_time` - Time taken for current generation
- `training/avg_generation_time` - Rolling average generation time

#### Validation Metrics (Every LOG_FREQUENCY Generations)
- `validation/fitness` - Fitness on validation set
- `validation/win_rate` - Win rate on validation trades
- `validation/total_trades` - Number of trades executed
- `validation/winning_trades` - Number of profitable trades

#### Buffer Metrics (Every Generation)
- `buffer/size` - Current number of transitions in buffer
- `buffer/utilization` - Percentage of buffer capacity used
- `buffer/capacity` - Maximum buffer capacity

### Hyperparameter Tracking

All key hyperparameters are automatically logged:
- Population size
- Number of generations
- Buffer size and batch size
- Learning rates (actor and critic)
- Trading environment parameters
- Number of stocks

## Setup

### 1. Install wandb

Already added to `requirements.txt`:
```bash
pip install wandb
```

### 2. Login to W&B

**On your local machine:**
```bash
wandb login
# Use your API key from: https://wandb.ai/authorize
```

**On RunPod/remote instance:**
```bash
# Option 1: Interactive login
wandb login

# Option 2: Use API key directly
export WANDB_API_KEY=your_api_key_here
```

### 3. Set W&B Entity (Optional)

The code is configured to use entity `eigen2`. If you need to change this:

Edit [training/erl_trainer.py](training/erl_trainer.py#L66):
```python
wandb.init(
    project="eigen2-trading",
    entity="eigen2",  # Change this to your W&B username or team
    ...
)
```

## Usage

### Start Training with W&B

W&B tracking starts automatically when you run training:

```bash
cd /workspace
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# wandb will initialize automatically
python main.py
```

**Expected output:**
```
wandb: Currently logged in as: eigen2-self
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in /workspace/wandb/run-20250103_120000-abc123
wandb: Run `wandb offline` to turn off syncing
wandb: Syncing run erl-25gen
wandb: ⭐️ View project at https://wandb.ai/eigen2/eigen2-trading
wandb: 🚀 View run at https://wandb.ai/eigen2/eigen2-trading/runs/abc123
```

### Resume Training with W&B

W&B is configured with `resume="allow"`, which means:
- If training crashes and you restart, wandb will continue the same run
- Run ID is based on the training session
- Metrics continue from where they left off

```bash
# Resume training from checkpoint
python main.py

# W&B will automatically resume the previous run
```

### Disable W&B (If Needed)

To train without W&B tracking:

```bash
# Option 1: Disable entirely
export WANDB_MODE=disabled
python main.py

# Option 2: Offline mode (logs locally, sync later)
export WANDB_MODE=offline
python main.py

# Later, sync offline runs:
wandb sync wandb/run-*
```

## W&B Dashboard

### Viewing Your Runs

1. Go to https://wandb.ai/eigen2/eigen2-trading
2. Click on your run (e.g., `erl-25gen`)
3. Explore metrics, charts, and system info

### Key Visualizations

**Fitness Over Time:**
- Chart: `fitness/mean`, `fitness/max`, `fitness/min`
- Shows population performance evolution

**Best Fitness Progression:**
- Chart: `fitness/best_ever`
- Shows if training is improving over time

**Training Losses:**
- Charts: `train/actor_loss`, `train/critic_loss`
- Monitor network learning progress

**Validation Performance:**
- Charts: `validation/fitness`, `validation/win_rate`
- Check for overfitting (train vs validation gap)

**Buffer Utilization:**
- Chart: `buffer/utilization`
- Shows when buffer fills up (reaches 1.0)

**Generation Time:**
- Chart: `training/generation_time`
- Monitor training speed

### Custom Charts

Create custom visualizations in W&B:
1. Click "Add visualization"
2. Select metrics to compare
3. Choose chart type (line, scatter, histogram, etc.)

## Advanced Features

### Comparing Runs

Compare multiple training runs side-by-side:
1. Select runs to compare (checkbox)
2. Click "Compare"
3. View metrics overlaid on same charts

**Use cases:**
- Compare different hyperparameters
- Evaluate impact of buffer size changes
- Compare training with/without buffer saves

### Hyperparameter Sweeps

W&B supports automated hyperparameter sweeps (not currently implemented):

```yaml
# sweep.yaml
program: main.py
method: bayes
metric:
  name: fitness/best_ever
  goal: maximize
parameters:
  actor_lr:
    values: [1e-4, 3e-4, 5e-4]
  critic_lr:
    values: [3e-4, 5e-4, 1e-3]
```

```bash
wandb sweep sweep.yaml
wandb agent eigen2/eigen2-trading/sweep_id
```

### Artifacts (For Future Use)

W&B supports saving model checkpoints as artifacts:

```python
# Save best agent as artifact
artifact = wandb.Artifact('best-agent', type='model')
artifact.add_file('checkpoints/best_agent.pth')
wandb.log_artifact(artifact)

# Load artifact
artifact = wandb.use_artifact('eigen2/eigen2-trading/best-agent:latest')
artifact_dir = artifact.download()
```

## Troubleshooting

### "wandb: ERROR Error uploading"

**Cause:** Network issues or API key problems

**Solution:**
```bash
# Check login status
wandb status

# Re-login
wandb login --relogin
```

### "wandb: WARNING Run is not syncing"

**Cause:** Offline mode or network disconnection

**Solution:**
```bash
# Check mode
echo $WANDB_MODE

# Ensure online mode
unset WANDB_MODE
```

### Large Run Logs

**Issue:** W&B logs can accumulate over long training runs

**Solution:**
```bash
# Clean old runs (keep only last 5)
wandb sync --clean

# Or manually delete old run folders
rm -rf wandb/run-old-id
```

### Multiple Runs with Same Name

**Issue:** Each training starts a new W&B run

**Solution:** This is intentional for tracking experiments. To group runs:
1. Use tags: `wandb.init(..., tags=["experiment-1"])`
2. Use groups: `wandb.init(..., group="hyperparameter-search")`

## Integration Details

### Files Modified

1. **[requirements.txt](requirements.txt#L6)**
   - Added `wandb>=0.16.0`

2. **[training/erl_trainer.py](training/erl_trainer.py)**
   - Line 15: Import wandb
   - Lines 64-82: Initialize wandb with config
   - Lines 500-508: Log fitness metrics
   - Lines 546-552: Log validation metrics
   - Lines 285-289: Log training losses
   - Lines 576-589: Log buffer and timing metrics
   - Line 607: Finish wandb run

### Metric Logging Frequency

- **Fitness, Buffer, Timing:** Every generation
- **Training Losses:** Every generation (agent 0 only)
- **Validation:** Every LOG_FREQUENCY generations (default: 5)

## Example W&B Output

**During Training:**
```
Generation 10/25
============================================================
Fitness Summary:
  Mean: 1250.45
  Max: 1580.32
  Min: 890.12
  Std: 180.45

wandb: Logged fitness/mean: 1250.45
wandb: Logged fitness/max: 1580.32
wandb: Logged buffer/size: 20000
wandb: Logged training/generation_time: 145.2s
```

**In W&B Dashboard:**
```
Run: erl-25gen
Status: Running
Runtime: 1h 23m
Step: 10

Metrics:
├─ fitness/mean: 1250.45
├─ fitness/max: 1580.32
├─ fitness/best_ever: 1650.20
├─ buffer/utilization: 0.40
└─ training/avg_generation_time: 142.8s
```

## Best Practices

1. **Tag your runs:** Add tags for experiments
   ```python
   wandb.init(..., tags=["baseline", "no-buffer-save"])
   ```

2. **Use descriptive names:** Change run name for clarity
   ```python
   wandb.init(..., name=f"erl-gen{Config.NUM_GENERATIONS}-bs{Config.BUFFER_SIZE}")
   ```

3. **Add notes:** Document what makes this run special
   ```python
   wandb.init(..., notes="Testing reduced buffer size for RAM optimization")
   ```

4. **Monitor during training:** Check W&B dashboard every few generations
   - Ensure metrics are logging
   - Watch for anomalies (sudden drops, NaN values)
   - Compare to previous runs

5. **Save key artifacts:** Log model checkpoints for important milestones
   ```python
   if best_fitness > previous_best:
       wandb.save('checkpoints/best_agent.pth')
   ```

## Summary

W&B integration provides:
- ✅ Real-time metric tracking (fitness, losses, validation)
- ✅ Automatic hyperparameter logging
- ✅ Run comparison and analysis
- ✅ Persistent training history
- ✅ Resume capability
- ✅ No code changes needed to enable/disable

All metrics are automatically logged alongside existing TensorBoard logging - you get the best of both worlds!
