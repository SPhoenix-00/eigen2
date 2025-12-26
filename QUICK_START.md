# Quick Start: RunPod GPU Training

## Prerequisites
- GCS Bucket: `eigen2-checkpoints-ase0`
- Credentials: `gcs-credentials.json`
- RunPod account with AMD MI300X instance (ROCm)

---

## Setup Commands

### 1. Create RunPod Instance
- Go to: https://www.runpod.io/console/gpu-cloud
- GPU: AMD MI300X (192GB VRAM) or similar ROCm GPU
- Template: `runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04`
- Disk: 250 GB

### 2. SSH and Install
```bash
ssh root@YOUR_RUNPOD_HOST -p YOUR_PORT

mkdir -p /workspace && cd /workspace
rm -rf * .??*
git clone -b eigen_rocm https://github.com/SPhoenix-00/eigen2.git .
pip install -r requirements.txt google-cloud-storage
```

> **Note:** PyTorch with ROCm support is pre-installed in the base image.
> The requirements.txt does NOT include torch to avoid overwriting it.

### 3. Upload Credentials (from local machine)
```bash
scp -P YOUR_PORT gcs-credentials.json root@YOUR_HOST:/workspace/
```

### 4. Download Training Data (on RunPod)
```bash
cd /workspace
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

python3 << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
blob = bucket.blob('eigen2/Eigen2_Master_PY_OUTPUT.pkl')
blob.download_to_filename('/workspace/Eigen2_Master_PY_OUTPUT.pkl')
print("✓ Training data downloaded")
EOF
```

### 5. Login to W&B
```bash
wandb login
# Paste API key from: https://wandb.ai/authorize
# 7deffe6a4942dc629a3327c7df7be882859d638c
```

### 6. Start Training in tmux
```bash
cd /workspace
tmux -u new -s training

export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

python main.py              # New training
python main.py --resume     # Resume from last run
```

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

## Monitor Training

```bash
# Check if running (without attaching)
ps aux | grep python
rocm-smi              # AMD GPU status (replaces nvidia-smi)

# View logs
tail -f /workspace/logs/training.log

# W&B Dashboard
https://wandb.ai/your-username/eigen2-self
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

## Troubleshooting

**GCS connection issues:**
```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
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
blob = bucket.blob('eigen2/Eigen2_Master_PY_OUTPUT.pkl')
blob.download_to_filename('/workspace/Eigen2_Master_PY_OUTPUT.pkl')
EOF
```
