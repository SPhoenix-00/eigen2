# Quick Start: RunPod GPU Training

## Prerequisites
- GCS Bucket: `eigen2-checkpoints-ase0`
- Credentials: `gcs-credentials.json`
- RunPod account with RTX 4090 instance
- **PyTorch:** Code uses current CUDA APIs (`total_memory`, `PYTORCH_ALLOC_CONF`); compatible with PyTorch 2.x.

---

## Quick Reference: Training Modes

| Mode | Command | Population | Use Case |
|------|---------|------------|----------|
| **Standard** | `python main.py` | 96 agents | Cloud/remote training (RunPod, AWS, GCP) |
| **Local** | `python main.py --local` | 32 agents | Windows, local dev, single-core systems |
| **Maverick** | `python main.py --maverick` | 96 agents | Aggressive agents to break committee inaction |
| **Local + Maverick** | `python main.py --local --maverick` | 32 agents | Local development of aggressive agents |

**Resume any mode:**
```bash
python main.py --resume              # Resume last run (uses same mode as original)
python main.py --local --resume      # Resume with local mode
python main.py --maverick --resume   # Resume maverick training
```

---

## Setup Commands

### 1. Create RunPod Instance
- Go to: https://www.runpod.io/console/gpu-cloud
- GPU: RTX 4090 (24GB+ VRAM)
- Template: PyTorch 2.1
- Disk: 250 GB

### 2. SSH and Install
```bash
ssh root@YOUR_RUNPOD_HOST -p YOUR_PORT

mkdir -p /workspace && cd /workspace
rm -rf * .??*
git clone https://github.com/SPhoenix-00/eigen2.git .
python3 -m pip install --no-cache-dir -r requirements.txt google-cloud-storage
```

### 3. Upload Credentials (from local machine)
```bash
scp -P YOUR_PORT gcs-credentials.json root@YOUR_HOST:/workspace/
```

### 4. Download Training Data

**On RunPod (bash):**
```bash
cd /workspace
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

python3 << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
blob = bucket.blob('eigen2/Eigen2_Master_PY_OUTPUT_060226.pkl')
blob.download_to_filename('/workspace/Eigen2_Master_PY_OUTPUT_060226.pkl')
print("✓ Training data downloaded")
EOF
```

**On Windows (PowerShell):**
```powershell
cd D:\GitHub\eigen2
$env:GOOGLE_APPLICATION_CREDENTIALS="D:\GitHub\eigen2\gcs-credentials.json"

python -c "from google.cloud import storage; client = storage.Client(); bucket = client.bucket('eigen2-checkpoints-ase0'); blob = bucket.blob('eigen2/Eigen2_Master_PY_OUTPUT_060226.pkl'); blob.download_to_filename('Eigen2_Master_PY_OUTPUT_060226.pkl'); print('✓ Training data downloaded')"
```

### 5. Login to W&B
```bash
wandb login
# Paste API key from: https://wandb.ai/authorize
# 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
# $env:WANDB_API_KEY="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"

```

### 6. Start Training

**On Linux/RunPod (bash):**
```bash
cd /workspace
tmux -u new -s training

export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

python main.py              # New training
python main.py --resume     # Resume from last run
```

**On Windows (PowerShell):**
```powershell
cd D:\GitHub\eigen2

$env:CLOUD_PROVIDER="gcs"
$env:CLOUD_BUCKET="eigen2-checkpoints-ase0"
$env:GOOGLE_APPLICATION_CREDENTIALS="D:\GitHub\eigen2\gcs-credentials.json"

python main.py              # New training
python main.py --resume     # Resume from last run
```

**Note:** Environment variables set with `$env:` in PowerShell are only valid for the current session. To make them persistent, use:
```powershell
[System.Environment]::SetEnvironmentVariable("CLOUD_PROVIDER", "gcs", "User")
[System.Environment]::SetEnvironmentVariable("CLOUD_BUCKET", "eigen2-checkpoints-ase0", "User")
[System.Environment]::SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "D:\GitHub\eigen2\gcs-credentials.json", "User")
```

---

## Training Modes and Flags

### Standard Training (Cloud/Remote)
Default mode optimized for distributed cloud environments (RunPod, AWS, etc.):

```bash
python main.py              # New training (96 agents, parallel processing)
python main.py --resume     # Resume from last run
```

**Characteristics:**
- Population size: 96 agents
- Parallel evaluation/validation workers
- Optimized for multi-core cloud instances
- Best for: RunPod, AWS, GCP instances with multiple CPU cores

---

### Local Mode (`--local`)

Optimized for local machines (Windows, Mac, or single-core systems):

```bash
python main.py --local              # New training with local optimizations
python main.py --local --resume     # Resume with local mode
```

**Use Cases:**
- Running on Windows (avoids multiprocessing issues)
- Local development/testing on personal machines
- When parallel workers cause slowdowns or crashes
- Single-core or limited CPU environments

**Optimizations:**
- Population size: 32 agents (reduced from 96)
- Sequential evaluation/validation (no process spawning)
- Eliminates IPC serialization overhead
- Serialized disk writes (prevents I/O thrashing on NVMe)
- GPU-accelerated batched inference (when GPU available)
- In-memory transition buffers (no disk I/O during evaluation)

**Performance:**
- Faster on Windows and single-core systems
- Lower memory footprint
- More stable on systems with multiprocessing issues
- Still uses GPU for inference when available

---

### Maverick Mode (`--maverick`)

Aggressive training mode for producing signal generators that break committee inaction:

```bash
python main.py --maverick              # New maverick training
python main.py --maverick --resume   # Resume maverick run
python main.py --maverick --local    # Maverick + local mode
```

**Use Cases:**
- Training aggressive agents to break committee deadlocks
- Producing high-conviction signal generators
- When committee shows excessive inaction (too few trades)
- Creating "disruptor" agents for committee diversity

**Training Characteristics:**
- **Reward Function**: FOMO/ROI-First (aggressive)
- **Hurdle Rate**: 50% of normal (lower barrier to entry)
- **Forced Exit Penalty**: Disabled (encourages holding for bigger gains)
- **Population**: Standard size (96 agents) unless combined with `--local` (32 agents)

**Constraints:**
- **Highlander Rule**: Maximum 5 mavericks allowed in Global 50
- **Auto-Stop**: Training stops when maverick reaches rank 20 or higher in Global 50
- **Purpose**: Designed to complement conservative committee members

**When to Use:**
- Committee evaluation shows low trade frequency
- Need more aggressive signal generation
- Want to diversify committee behavior
- Training specialized "disruptor" agents

---

### Combining Flags

Flags can be combined for specific use cases:

```bash
# Local development with maverick training
python main.py --local --maverick

# Resume local maverick training
python main.py --local --maverick --resume

# Standard training with other modes
python main.py --heroes auto          # Train with Global 50 heroes
python main.py --consistency          # Consistency-focused training
python main.py --multi roster.json    # Multi-agent committee training
```

**Common Combinations:**
- `--local --maverick`: Local development of aggressive agents
- `--local --heroes auto`: Local training with Global 50 initialization
- `--maverick --resume`: Continue aggressive training run

---

## tmux Commands

**Detach from session (training keeps running):**
```bash
Ctrl+B, then D
```

**Scroll in tmux (view past output):**
```bash
Ctrl+B, then [          # Enter scroll mode
Page Up/Down or arrows  # Scroll through output
q                       # Exit scroll mode
```

**Reattach to session:**
```bash
tmux attach -t training
```

**List sessions:**
```bash
tmux ls
```

**Kill session:**
```bash
tmux kill-session -t training
```

---

## Reconnect Workflow

```bash
# SSH back in
ssh root@YOUR_RUNPOD_HOST -p YOUR_PORT

# Reattach to training
tmux attach -t training

# Scroll through past output
Ctrl+B, then [
```

---

## Console Output and Verbosity

Training output defaults to a clean **dashboard mode** that shows only the metrics that matter. Full verbose detail is always captured in the log file (`evaluation_results/training_log_*.txt`).

### Verbosity Flags

| Flag | Level | Behavior |
|------|-------|----------|
| *(default)* | NORMAL | Clean per-generation dashboard with best agent profile, population health, HoF, loop timing, and trend deltas |
| `-v` / `--verbose` | VERBOSE | Full legacy output (all detail printed to console) |
| `-q` / `--quiet` | QUIET | Errors and warnings only (good for background/unattended runs) |

```bash
python main.py --local                 # Clean dashboard (default)
python main.py --local -v              # Full verbose output
python main.py --local -q              # Quiet mode (background runs)
```

### Per-Generation Dashboard

Each generation prints a structured dashboard (~20 lines) showing:

- **Best Agent**: Combined fitness, ROI, win rate (W/L), quality ratio, expectancy, PnL -- with deltas vs previous generation
- **Population Health**: Fitness distribution, positive agent count, mean ROI, mean win rate
- **Hall of Fame**: Size, best/worst scores, median ROI, hurdle EMA
- **Loop Performance**: Phase timing breakdown (eval/val/train/evolve), training bottleneck split (compute/data_load/gpu_transfer), GPU memory, ETA
- **Events**: Highlighted lines for breakthroughs, new bests, HoF changes, turnovers

### W&B Portal

Metrics are logged under clean namespaces with proper `define_metric()` axes:
- `fitness/` -- Population fitness distribution
- `best_agent/` -- Best agent's full trading profile
- `population/` -- Population health (positive count, mean ROI/WR)
- `hof/` -- Hall of Fame state + Global 50
- `gauntlet/` -- Gauntlet state machine (logged every gen, not sparse events)
- `train/` -- Actor/critic loss, mutation rate, buffer
- `perf/` -- Phase timing breakdown, training bottleneck split, GPU memory

Runs are tagged with mode labels (`consistency`, `local`, `multi`, `gauntlet`, `maverick`) and grouped for multi-agent runs.

---

## Monitor Training

```bash
# Check if running (without attaching)
ps aux | grep python
nvidia-smi

# View logs
tail -f /workspace/logs/training.log

# W&B Dashboard
https://wandb.ai/your-username/eigen2-self

# Download metrics to CSV
python download_metrics.py
```

---

## Evaluate Best Agent

After training completes, evaluate your best agent's trading behavior:

```bash
# Evaluate last run (uses last_run.json)
python evaluate_best_agent.py

# Or evaluate a specific run
python evaluate_best_agent.py --run-name azure-thunder-123
```

This will:
- Load the best agent from GCP
- Evaluate on 3 validation slices
- Evaluate on first 125 days of holdout period
- Track every trade with detailed information
- Export results to `evaluation_results/` directory:
  - **Text report**: Complete trade-by-trade analysis
  - **Trades CSV**: Spreadsheet of all trades for analysis
  - **Summary CSV**: Statistics by evaluation slice

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed usage and output format.

---

## Recent Fixes

**Population diversity (clone/mutate, 2026-02):** Fixed agents being identical (no variability) when initializing from a single seed. Cause: `DDPGAgent.clone()` could leave parameter tensors sharing storage with the source, so genetic mutation was not applied to independent copies. Fixes: (1) `clone()` now loads from an explicitly cloned state dict (`detach().clone().to(device)` per tensor) so the new agent's parameters never share storage. (2) `mutate()` in `erl/genetic_ops.py` uses in-place `param.data.add_(...)`, explicit device/dtype for RNG, and a runtime check that actor weights actually change; if not, it raises. Files: `models/ddpg_agent.py`, `erl/genetic_ops.py`.

---

## Troubleshooting

**GCS connection issues:**

**On Linux/RunPod (bash):**
```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
```

**On Windows (PowerShell):**
```powershell
$env:CLOUD_PROVIDER="gcs"
$env:CLOUD_BUCKET="eigen2-checkpoints-ase0"
$env:GOOGLE_APPLICATION_CREDENTIALS="D:\GitHub\eigen2\gcs-credentials.json"
```

**wandb login error:**
```bash
wandb login
# Get API key from: https://wandb.ai/authorize
```

**tmux not found:**
```bash
apt update && apt install -y tmux
```

**Training data missing:**
```bash
cd /workspace
python3 << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
blob = bucket.blob('eigen2/Eigen2_Master_PY_OUTPUT_060226.pkl')
blob.download_to_filename('/workspace/Eigen2_Master_PY_OUTPUT_060226.pkl')
EOF
```

**Multiprocessing errors or crashes on Windows:**
```bash
# Use --local flag to avoid multiprocessing issues
python main.py --local
```

**Training too slow on local machine:**
```bash
# --local mode uses sequential processing (faster on single-core systems)
python main.py --local
```

**Committee showing excessive inaction (too few trades):**
```bash
# Train maverick agents to break committee deadlocks
python main.py --maverick
```

**Out of memory errors:**
```bash
# Use --local to reduce population size (32 vs 96 agents)
python main.py --local
```
