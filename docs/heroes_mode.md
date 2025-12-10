# Heroes Mode: Training from Pre-trained Agents

This document describes how to use the `--heroes` flag to initialize training from a set of pre-trained agents (a "Hall of Fame").

## Overview

Heroes mode allows you to start a new training run using the best agents from a previous run as the initial population. This is useful when:

- You have a well-trained population and want to fine-tune it with a different reward function (e.g., consistency mode)
- You want to continue training with different hyperparameters
- You want to combine agents from multiple training runs

## Usage

```bash
python main.py --heroes /path/to/checkpoint_folder --consistency
```

### Command-line Arguments

- `--heroes HOF_DIR`: Path to a checkpoint directory containing agent files
- `--consistency`: (Optional) Enable consistency mode with loss magnification (1.5x by default, see Config.CONSISTENCY_LOSS_MULTIPLIER)

## Hall of Fame Folder Structure

The `--heroes` flag accepts a checkpoint directory path. The system will look for agent files in two possible locations:

### Option 1: Hall of Fame subdirectory (Recommended)

```
checkpoint_folder/
  hall_of_fame/
    hof_agent_0.pth
    hof_agent_1.pth
    hof_agent_2.pth
    ...
    hall_of_fame.json  (optional, metadata)
```

### Option 2: Root directory with agent files

```
checkpoint_folder/
  agent_0.pth
  agent_1.pth
  ...
  agent_31.pth
```

## Preparing a Heroes Folder

### From an existing training run

1. Locate your checkpoint directory (e.g., `checkpoints/honest_forest_117/`)
2. Use the `hall_of_fame/` subdirectory directly, or copy agent files to a new folder

```bash
# Example: Use existing checkpoint
python main.py --heroes checkpoints/honest_forest_117

# The system will automatically find agents in:
# checkpoints/honest_forest_117/hall_of_fame/hof_agent_*.pth
```

### Creating a custom heroes folder

1. Create a new directory
2. Copy `.pth` agent files into it
3. Files can have any name ending in `.pth`

```bash
# Create heroes folder
mkdir my_heroes
cp checkpoints/run_1/hall_of_fame/hof_agent_*.pth my_heroes/
cp checkpoints/run_2/hall_of_fame/hof_agent_*.pth my_heroes/

# Rename to avoid conflicts
cd my_heroes
for f in *.pth; do mv "$f" "hero_$f"; done

# Use in training
python main.py --heroes my_heroes --consistency
```

## What Happens During Loading

1. **Load**: All `.pth` files in the directory are loaded as agents
2. **Evaluate**: Each agent is evaluated using the current reward function:
   - Standard mode: 3 episodes, fitness = avg(lowest 2)
   - Consistency mode: 5 episodes, fitness = sum(all 5)
3. **Select**: Top agents by fitness are selected for the initial population (up to POPULATION_SIZE)
4. **Fill**: If fewer agents than POPULATION_SIZE, top performers are cloned to fill remaining slots

## Evolution Fractions in Heroes Mode

Heroes mode uses different elite/mutant ratios optimized for fine-tuning pre-trained agents:

| Parameter | Standard Mode | Heroes Mode |
|-----------|---------------|-------------|
| Elite %   | 40%           | 50%         |
| Offspring %| 40%          | 37.5%       |
| Mutant %  | 20%           | 12.5%       |

This preserves more of the proven performers while still allowing some genetic exploration.

## Example Workflow: Consistency Training

1. **Initial Training**: Train agents with standard reward function
   ```bash
   python main.py
   # Results in checkpoints/azure_thunder_42/
   ```

2. **Download Checkpoints**: Copy the Hall of Fame to your server
   ```bash
   # From GCS or local
   gsutil cp -r gs://your-bucket/checkpoints/azure_thunder_42 checkpoints/
   ```

3. **Consistency Fine-tuning**: Train for consistency
   ```bash
   python main.py --heroes checkpoints/azure_thunder_42 --consistency
   ```

## Notes

- The evaluation during loading uses the current environment settings (including consistency mode)
- Agents are evaluated on training data slices, not validation data
- The original agent IDs are discarded; new IDs are assigned by fitness rank
- If no valid agents can be loaded, training continues with random initialization

## Troubleshooting

### "No agent files found"

- Check that the directory contains `.pth` files
- Verify the path is correct and accessible

### "No agents could be loaded"

- Agent files may be corrupted or from an incompatible model version
- Check that the model architecture hasn't changed since the agents were saved

### Low fitness scores after loading

- This is expected if using a different reward function (e.g., consistency mode)
- The agents will adapt to the new objective during training
