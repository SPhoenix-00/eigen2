# Champion Agent Evaluation Script

## Overview

The `evaluate_champions.py` script evaluates all best agents from your training runs stored in GCP and identifies the top 3 performers.

## What It Does

1. **Downloads all best agents** from every training run in your GCP bucket
2. **Evaluates each agent** on:
   - The validation dataset (fixed window)
   - 5 random windows from the training dataset
3. **Ranks agents** by average fitness across all 6 evaluation windows
4. **Saves top 3 champions** to a new `Champion_Agents` folder in GCP with:
   - The agent file (`best_agent.pth`)
   - Metadata including evaluation results and original source
   - A summary report

## Prerequisites

1. **Environment Variables**: Set the following before running:
   ```bash
   export CLOUD_PROVIDER=gcs
   export CLOUD_BUCKET=your-gcp-bucket-name
   export CLOUD_PROJECT=eigen2  # Optional, defaults to 'eigen2'
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-credentials.json
   ```

2. **Dependencies**: Ensure you have all required packages:
   ```bash
   pip install google-cloud-storage torch numpy tqdm
   ```

## Usage

### Basic Usage

Simply run the script:

```bash
python evaluate_champions.py
```

### Windows PowerShell

```powershell
$env:CLOUD_PROVIDER="gcs"
$env:CLOUD_BUCKET="your-bucket-name"
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\credentials.json"
python evaluate_champions.py
```

### Linux/Mac

```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=your-bucket-name
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
python evaluate_champions.py
```

## Output

### Console Output

The script will print:

1. **Discovery Phase**: List of all training runs found
2. **Evaluation Progress**: Real-time evaluation of each agent
3. **Rankings**: Complete ranking of all agents
4. **Top 3 Summary**: Detailed information about the champions

### Example Output

```
======================================================================
                    Champion Agent Evaluation System
======================================================================

--- Discovering Training Runs in GCP ---
✓ Found 15 training runs
  1. absorbed-cloud-54
  2. breezy-puddle-62
  3. cosmic-dream-71
  ...

--- Downloading and Evaluating Agents ---
Evaluating runs: 100%|████████████████| 15/15

  Evaluating breezy-puddle-62...
    Validation set fitness: 36494.87
    Random training windows:
      Window 1 (days 250-395): 28345.12
      Window 2 (days 891-1036): 31256.78
      ...
    Average fitness: 32567.45 ± 3241.56

======================================================================
                               Ranking Results
======================================================================

All Agents (ranked by average fitness):

1. breezy-puddle-62
   Average Fitness: 32567.45 ± 3241.56
   Validation Fitness: 36494.87
   Original Metadata:
     - generation: 13
     - best_validation_fitness: 36494.87

======================================================================
                    Saving Top 3 Champions to GCP
======================================================================

======================================================================
                              Champion #1
======================================================================
Run Name: breezy-puddle-62
Original GCP Path: eigen2/checkpoints/breezy-puddle-62/best_agent.pth
Average Fitness: 32567.45 ± 3241.56
Validation Fitness: 36494.87
Fitness Range: [28345.12, 36494.87]

✓ Saved to: eigen2/Champion_Agents/rank_1_breezy-puddle-62/best_agent.pth
✓ Metadata: eigen2/Champion_Agents/rank_1_breezy-puddle-62/metadata.json
```

### GCP Output

The script creates the following structure in your GCP bucket:

```
gs://your-bucket/eigen2/Champion_Agents/
├── rank_1_breezy-puddle-62/
│   ├── best_agent.pth
│   └── metadata.json
├── rank_2_cosmic-dream-71/
│   ├── best_agent.pth
│   └── metadata.json
├── rank_3_stellar-wave-88/
│   ├── best_agent.pth
│   └── metadata.json
└── evaluation_summary.json
```

### Metadata Format

Each champion's `metadata.json` contains:

```json
{
    "rank": 1,
    "original_run_name": "breezy-puddle-62",
    "original_gcp_path": "eigen2/checkpoints/breezy-puddle-62/best_agent.pth",
    "evaluation_results": {
        "run_name": "breezy-puddle-62",
        "validation_fitness": 36494.87,
        "training_windows_fitness": [28345.12, 31256.78, 29876.54, 30567.23, 32145.67],
        "avg_fitness": 32567.45,
        "std_fitness": 3241.56,
        "min_fitness": 28345.12,
        "max_fitness": 36494.87
    }
}
```

## Evaluation Methodology

### Validation Windows

1. **Validation Set**: Standard validation window at the start of the validation period
2. **Random Training Windows**: 5 randomly selected windows from the training period
   - Each window includes context + trading + settlement periods
   - Windows are randomly placed to test generalization

### Ranking Criteria

Agents are ranked by **average fitness** across all 6 evaluation windows. This ensures:
- Generalization across different market conditions
- Robustness to market volatility
- Consistency of performance

## Troubleshooting

### "No training runs found in GCP bucket"

- Verify `CLOUD_BUCKET` is set correctly
- Check that your GCS credentials have read access
- Ensure training runs have been uploaded to GCP

### "Cloud sync is set to 'local' mode"

- Set `CLOUD_PROVIDER=gcs` environment variable
- Verify `CLOUD_BUCKET` is set
- Check `GOOGLE_APPLICATION_CREDENTIALS` path is valid

### Memory Issues

The script processes one agent at a time and clears GPU memory after each evaluation. If you still encounter memory issues:

1. Close other applications
2. Reduce batch processing by modifying the script
3. Run on a machine with more RAM/VRAM

## Notes

- The script creates a `temp_champions/` directory for temporary downloads
- Agents are evaluated without exploration noise (noise_scale=0.0)
- The validation environment is in `train_mode=False` for consistent evaluation
- All fitness scores are based on cumulative rewards

## After Evaluation

Once you have your top champions, you can:

1. Load them for further analysis
2. Use them as starting points for new training runs
3. Deploy them for live trading (with appropriate risk management)
4. Analyze their trading patterns and strategies

## Support

For issues or questions, please refer to the main project documentation or create an issue in the repository.
