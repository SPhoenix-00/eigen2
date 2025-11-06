# Quick Start: RunPod GPU Training

## Prerequisites
- GCS Bucket: `eigen2-checkpoints-ase0`
- Credentials: `gcs-credentials.json`
- RunPod account with RTX 4090 instance

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
```

### 6. Start Training in tmux
```bash
cd /workspace
tmux new -s training

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
