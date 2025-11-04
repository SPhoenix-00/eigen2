# W&B Sweeps Guide - Hyperparameter Optimization

## Overview

This guide explains how to run **local hyperparameter sweeps** using Weights & Biases while your main training runs on RunPod.

**Benefits:**
- Automatically find optimal hyperparameters
- Run multiple experiments in parallel on your local machine
- Compare results in W&B dashboard
- Use Bayesian optimization for efficient search

## Quick Start

### 1. Prerequisites

Make sure you're logged into W&B:

```bash
wandb login
# Use your API key from: https://wandb.ai/authorize
```

### 2. Initialize Sweep

From your project directory:

```bash
# Create the sweep
wandb sweep sweep_config.yaml
```

**Output:**
```
wandb: Creating sweep from: sweep_config.yaml
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/eigen2/eigen2-self/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent eigen2/eigen2-self/abc123xyz
```

**Copy the sweep ID** (`abc123xyz`) - you'll need it to run agents.

### 3. Run Sweep Agents

Start one or more sweep agents to run experiments:

```bash
# Run a single agent
wandb agent eigen2/eigen2-self/SWEEP_ID

# Or run multiple agents in parallel (recommended)
# Terminal 1:
wandb agent eigen2/eigen2-self/SWEEP_ID

# Terminal 2:
wandb agent eigen2/eigen2-self/SWEEP_ID

# Terminal 3:
wandb agent eigen2/eigen2-self/SWEEP_ID
```

**Each agent will:**
1. Pull hyperparameters from the sweep controller
2. Run 10-generation training with those parameters
3. Report results back to W&B
4. Request next set of parameters
5. Repeat until you stop it (Ctrl+C)

### 4. Monitor Progress

Go to the sweep URL to see:
- Real-time results from all agents
- Parallel coordinates plot showing parameter importance
- Best performing runs
- Hyperparameter importance rankings

## Sweep Configuration

The sweep is configured in [sweep_config.yaml](sweep_config.yaml).

### Optimization Strategy

**Method:** Bayesian optimization (`bayes`)
- Most sample-efficient method
- Learns from previous runs to suggest better parameters
- Better than grid/random search for limited compute budget

**Metric to Maximize:** `fitness/best_ever`
- The best fitness achieved during the run
- Primary metric for evaluating trading agent performance

**Early Stopping:** Hyperband
- Automatically stops unpromising runs after 5 generations
- Saves compute time by not wasting resources on bad hyperparameters
- Aggressive settings (`eta: 2, s: 3`) for fast iteration

### Hyperparameters Being Optimized

#### Learning Rates (Most Important)
```yaml
actor_lr: [1e-5, 5e-4]  # Log-uniform distribution
critic_lr: [1e-4, 1e-3]  # Log-uniform distribution
```

#### DDPG Parameters
```yaml
gamma: [0.95, 0.97, 0.99, 0.995]  # Discount factor
tau: [0.001, 0.01]  # Soft update rate
```

#### Training Intensity
```yaml
gradient_steps_per_generation: [16, 32, 64, 96]
gradient_accumulation_steps: [4, 6, 8, 12]
batch_size: [2, 4, 8]
```

#### Exploration
```yaml
noise_scale: [0.05, 0.2]
noise_decay: [0.9995, 0.9997, 0.9999]
```

#### Genetic Algorithm
```yaml
mutation_rate: [0.05, 0.2]
mutation_std: [0.005, 0.05]
```

#### Trading Environment
```yaml
loss_penalty_multiplier: [5.0, 10.0, 15.0, 20.0]
inaction_penalty: [0.5, 1.0, 2.0, 3.0]
max_holding_period: [10, 15, 20, 25, 30]
```

#### Fixed Parameters (Not Tuned)
```yaml
num_generations: 10  # Shorter runs for sweep
population_size: 8   # Smaller population
buffer_size: 25000   # Smaller buffer for local
```

## Customizing the Sweep

### Add/Remove Parameters

Edit [sweep_config.yaml](sweep_config.yaml):

```yaml
parameters:
  # Add a new parameter
  weight_decay:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-4

  # Remove a parameter - just delete or comment out
  # noise_scale:
  #   distribution: uniform
  #   min: 0.05
  #   max: 0.2
```

Then update [sweep_runner.py](sweep_runner.py) to apply the new parameter:

```python
# In train_with_config function, add:
if hasattr(sweep_config, 'weight_decay'):
    Config.WEIGHT_DECAY = sweep_config.weight_decay
```

### Change Search Strategy

**Grid Search** (exhaustive, slow):
```yaml
method: grid
```

**Random Search** (simple, inefficient):
```yaml
method: random
```

**Bayesian** (recommended):
```yaml
method: bayes
```

### Adjust Run Length

For **faster iteration** (5 generations):
```yaml
parameters:
  num_generations:
    value: 5
```

For **more thorough evaluation** (25 generations):
```yaml
parameters:
  num_generations:
    value: 25
```

### Limit Number of Runs

Add to sweep config:
```yaml
# Stop after 50 runs
run_cap: 50
```

## Analyzing Results

### 1. View Sweep Dashboard

Go to: `https://wandb.ai/eigen2/eigen2-self/sweeps/SWEEP_ID`

**Key Visualizations:**

- **Parallel Coordinates Plot**: Shows how each parameter affects fitness
- **Parameter Importance**: Ranking of which parameters matter most
- **Sweep Table**: Sortable table of all runs

### 2. Find Best Run

In the sweep dashboard:
1. Sort by `fitness/best_ever` (descending)
2. Click on the best run
3. Note the hyperparameters

**Example best parameters:**
```
actor_lr: 2.3e-4
critic_lr: 5.8e-4
gamma: 0.99
gradient_steps_per_generation: 64
mutation_rate: 0.12
loss_penalty_multiplier: 15.0
```

### 3. Apply Best Parameters to RunPod

Once you find optimal hyperparameters:

1. **Update [utils/config.py](utils/config.py)** with best values:
   ```python
   ACTOR_LR = 2.3e-4  # From sweep
   CRITIC_LR = 5.8e-4  # From sweep
   GAMMA = 0.99
   # ... etc
   ```

2. **Push to GitHub:**
   ```bash
   git add utils/config.py
   git commit -m "Update hyperparameters from W&B sweep"
   git push origin main
   ```

3. **Pull on RunPod and restart training:**
   ```bash
   # SSH into RunPod
   cd /workspace
   git pull origin main
   python main.py
   ```

## Tips for Effective Sweeps

### 1. Start Broad, Then Narrow

**Phase 1: Coarse sweep (wide ranges)**
- Use broad parameter ranges
- Run 20-30 experiments
- Identify promising regions

**Phase 2: Fine-tuning (narrow ranges)**
- Focus on best parameter ranges from Phase 1
- Use tighter bounds
- Run 30-50 more experiments

### 2. Prioritize Important Parameters

**High Impact** (tune first):
- `actor_lr`, `critic_lr`
- `gradient_steps_per_generation`
- `loss_penalty_multiplier`

**Medium Impact** (tune second):
- `gamma`, `tau`
- `mutation_rate`, `mutation_std`
- `batch_size`

**Low Impact** (tune last or fix):
- `noise_decay`
- `inaction_penalty`

### 3. Use Multiple Parallel Agents

**Recommended:**
- 3-4 agents on a powerful local machine
- Each agent runs one experiment at a time
- Bayesian optimization benefits from parallel exploration

**Command:**
```bash
# Run 4 agents in parallel (4 separate terminals)
wandb agent eigen2/eigen2-self/SWEEP_ID  # Terminal 1
wandb agent eigen2/eigen2-self/SWEEP_ID  # Terminal 2
wandb agent eigen2/eigen2-self/SWEEP_ID  # Terminal 3
wandb agent eigen2/eigen2-self/SWEEP_ID  # Terminal 4
```

### 4. Monitor GPU/CPU Usage

**Check system resources:**
```bash
# GPU
nvidia-smi

# CPU/RAM
htop
```

**Adjust based on capacity:**
- If GPU is underutilized: increase `population_size` or `batch_size`
- If running out of RAM: decrease `buffer_size` or `population_size`

### 5. Stop Unpromising Sweeps Early

If after 10-15 runs all results are poor:
- Check for bugs in code
- Verify data is loading correctly
- Consider if parameter ranges are reasonable

## Troubleshooting

### "Sweep not found"

**Cause:** Incorrect sweep ID or entity

**Solution:**
```bash
# Check your sweeps
wandb sweep --list

# Use correct format
wandb agent eigen2/eigen2-self/SWEEP_ID
```

### "CUDA out of memory"

**Cause:** Local GPU can't handle the configuration

**Solution:** Reduce memory usage in sweep config:
```yaml
parameters:
  population_size:
    value: 4  # Smaller population

  batch_size:
    value: 2  # Smaller batch

  buffer_size:
    value: 10000  # Smaller buffer
```

### Runs failing immediately

**Check logs:**
```bash
# Runs will log errors to terminal
# Look for Python exceptions or data loading errors
```

**Common issues:**
- Data file path incorrect for local machine
- Missing dependencies
- Config validation failing

**Solution:** Test manually first:
```bash
python sweep_runner.py
```

### W&B sync issues

**Symptom:** "ERROR Error uploading"

**Solution:**
```bash
# Re-login
wandb login --relogin

# Or run in offline mode (sync later)
export WANDB_MODE=offline
wandb agent eigen2/eigen2-self/SWEEP_ID

# Later, sync results
wandb sync wandb/run-*
```

## Cost Optimization

### Estimated Time per Run

**10 generations with:**
- Population size: 8
- Buffer size: 25,000
- Local GPU (RTX 3080 or similar)

**Estimated time:** 20-40 minutes per run

### Recommended Sweep Budget

**Small sweep (explore):**
- 20 runs × 30 min = **10 hours**
- Use 2 parallel agents
- Actual time: **5 hours**

**Medium sweep (refine):**
- 50 runs × 30 min = **25 hours**
- Use 3 parallel agents
- Actual time: **8-9 hours**

**Large sweep (thorough):**
- 100 runs × 30 min = **50 hours**
- Use 4 parallel agents
- Actual time: **12-15 hours**

### Running Overnight

```bash
# Start sweep in tmux/screen to survive disconnections
tmux new -s sweep

# Run agents
wandb agent eigen2/eigen2-self/SWEEP_ID

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t sweep
```

## Advanced Features

### Custom Metrics

Track additional metrics in [sweep_runner.py](sweep_runner.py):

```python
# After training completes
wandb.log({
    "final/win_rate": final_win_rate,
    "final/avg_trade_duration": avg_duration,
    "final/sharpe_ratio": sharpe_ratio,
})
```

Then optimize for these in sweep config:
```yaml
metric:
  name: final/sharpe_ratio
  goal: maximize
```

### Multi-Objective Optimization

Optimize for multiple goals:
```yaml
metric:
  name: combined_score
  goal: maximize

# In sweep_runner.py:
combined_score = 0.7 * fitness + 0.3 * win_rate
wandb.log({"combined_score": combined_score})
```

## Summary

**To run a hyperparameter sweep:**

1. **Initialize sweep:**
   ```bash
   wandb sweep sweep_config.yaml
   ```

2. **Start agents (3-4 terminals):**
   ```bash
   wandb agent eigen2/eigen2-self/SWEEP_ID
   ```

3. **Monitor:** https://wandb.ai/eigen2/eigen2-self/sweeps/SWEEP_ID

4. **Apply best parameters** to [utils/config.py](utils/config.py)

5. **Deploy to RunPod** and run full 25-generation training

**Files:**
- [sweep_config.yaml](sweep_config.yaml) - Sweep configuration
- [sweep_runner.py](sweep_runner.py) - Training script for sweep
- This guide - [WANDB_SWEEPS_GUIDE.md](WANDB_SWEEPS_GUIDE.md)

Happy optimizing! 🎯
