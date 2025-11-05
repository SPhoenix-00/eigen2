# Quick Start Guide: GPU Training with Cloud Storage

**Time: 5-10 minutes** | Supports: RunPod, Vast.ai, GCP

This guide covers deploying Eigen2 training on any GPU provider with automatic cloud backup.

---

## Table of Contents

- [Using tmux for Persistent Sessions](#using-tmux-for-persistent-sessions) ⭐ **Important!**
- [RunPod Setup](#runpod-setup) ← Recommended, simplest
- [Vast.ai Setup](#vastai-setup)
- [GCP Setup](#gcp-setup)
- [Common Issues](#troubleshooting)

---

## Using tmux for Persistent Sessions

### Why tmux?

**SSH connections to RunPod/cloud instances often disconnect after a few hours.** Using `tmux` allows your training to continue running even when your SSH connection drops. You can reconnect later and reattach to your session.

### Essential tmux Commands

```bash
# Start a new tmux session named "training"
tmux new -s training

# Detach from session (keeps it running in background)
# Press: Ctrl+B, then D

# List all tmux sessions
tmux ls

# Reattach to session after SSH reconnect
tmux attach -t training

# Kill a session (if needed)
tmux kill-session -t training
```

### Quick tmux Guide

1. **Start session**: `tmux new -s training`
2. **Run your training**: `python main.py`
3. **Detach safely**: Press `Ctrl+B`, then press `D`
4. **Close SSH**: You can now disconnect SSH - training keeps running!
5. **Reconnect later**: SSH back in, then `tmux attach -t training`

> 💡 **Pro tip**: You can have multiple tmux sessions. Use different names: `tmux new -s eval`, `tmux new -s debug`, etc.

---

## Prerequisites

✅ **GCS Bucket**: `eigen2-checkpoints-ase0` (already created)
✅ **Credentials**: `gcs-credentials.json` (from service account)
✅ **GitHub Repo**: https://github.com/SPhoenix-00/eigen2.git
✅ **Training Data**: In GCS at `eigen2/Eigen2_Master(GFIN)_03_training.csv`

---

## RunPod Setup

> ⚠️ **Important**: RunPod SSH sessions often disconnect after a few hours. **Always use tmux** (see instructions below) so your training continues even when disconnected!

### Step 1: Create RunPod Instance

1. Go to: **https://www.runpod.io/console/gpu-cloud**
2. Select GPU: **RTX 4090** (or 24GB+ VRAM)
3. Template: **PyTorch 2.1** (or similar CUDA 12.1+ image)
4. Disk: **250 GB minimum**
5. Click **"Deploy"**

### Step 2: Connect and Setup

**SSH into instance:**
```bash
ssh root@YOUR_RUNPOD_HOST -p YOUR_PORT
```

**Setup workspace:**
```bash
# Create and enter workspace
mkdir -p /workspace
cd /workspace

# Clean if needed
rm -rf * .??*

# Clone repository
git clone https://github.com/SPhoenix-00/eigen2.git .

# Install dependencies
python3 -m pip install --no-cache-dir -r requirements.txt google-cloud-storage
```

### Step 3: Upload Credentials and Download Training Data

**From your local machine (upload credentials):**
```bash
# Upload GCS credentials
scp -P YOUR_PORT gcs-credentials.json root@YOUR_HOST:/workspace/
```

**Back on RunPod instance:**
```bash
cd /workspace

# Verify google-cloud-storage is installed
python3 -c "import google.cloud.storage; print('✓ google-cloud-storage is installed')"

# Set up GCS authentication
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# Download training data from GCS bucket
python3 << 'EOF'
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
blob = bucket.blob('eigen2/Eigen2_Master(GFIN)_03_training.csv')

blob.download_to_filename('/workspace/Eigen2_Master(GFIN)_03_training.csv')
print("✓ Training data downloaded successfully")
EOF

# Verify download
ls -lh Eigen2_Master\(GFIN\)_03_training.csv
```

### Step 4: Login to Weights & Biases

```bash
# Login to wandb (one-time setup)
wandb login

# It will prompt: "wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)"
# Paste your API key from: https://wandb.ai/authorize
```

### Step 5: Set Environment Variables and Start Training with tmux

```bash
cd /workspace

# Start a tmux session (keeps training running if SSH disconnects)
tmux new -s training

# Set environment variables (REQUIRED!)
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# Verify environment is set
echo "Provider: $CLOUD_PROVIDER"
echo "Bucket: $CLOUD_BUCKET"
echo "Credentials: $GOOGLE_APPLICATION_CREDENTIALS"

# Start training
python main.py

# Or resume from checkpoint
python main.py --resume
```

**After training starts:**
- Press `Ctrl+B`, then `D` to detach from tmux (training continues in background)
- You can now safely disconnect SSH
- To reconnect later: `ssh` back in, then run `tmux attach -t training`

---

## Vast.ai Setup

> ⚠️ **Important**: SSH sessions often disconnect after a few hours. **Always use tmux** so your training continues even when disconnected!

### Step 1: Create Vast.ai Instance

Go to: **https://vast.ai/console/create/**

**Instance Configuration:**
- **GPU**: RTX 4090 (or 24GB+ VRAM)
- **Docker Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **Disk Space**: 250 GB

**Environment Variables:**
```bash
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=eigen2-checkpoints-ase0
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
```

**On-start Script:**
```bash
#!/bin/bash
cd /workspace

# Clone repository
git clone --depth 1 https://github.com/SPhoenix-00/eigen2.git .

# Wait for credentials
while [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; do
    echo "Waiting for gcs-credentials.json to be uploaded to /workspace/"
    sleep 10
done

# Install dependencies
python3 -m pip install --no-cache-dir -r requirements.txt google-cloud-storage

# Download training data
python3 << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
blob = bucket.blob('eigen2/Eigen2_Master(GFIN)_03_training.csv')
blob.download_to_filename('/workspace/Eigen2_Master(GFIN)_03_training.csv')
EOF

# Login to wandb (requires manual step - see below)
echo "IMPORTANT: SSH in and run 'wandb login' before starting training"
```

### Step 2: Upload Credentials and Login

```bash
# SSH into instance
ssh root@YOUR-VAST-IP

# Upload credentials (from local machine in separate terminal)
scp gcs-credentials.json root@YOUR-VAST-IP:/workspace/

# Login to wandb (on instance)
wandb login
```

### Step 3: Start Training with tmux

```bash
cd /workspace

# Start tmux session
tmux new -s training

# Start training
python main.py --resume
```

**Detach from tmux**: Press `Ctrl+B`, then `D`
**Reattach later**: `tmux attach -t training`

---

## GCP Setup

> 💡 **Tip**: While GCP connections are more stable, it's still recommended to use tmux for long training runs.

### Step 1: Create GCP VM

1. Go to: **https://console.cloud.google.com/compute**
2. Click **"CREATE INSTANCE"**
3. Configuration:
   - **Name**: eigen2-training
   - **Zone**: us-central1-a
   - **Machine type**: n1-standard-8
   - **GPU**: NVIDIA Tesla T4 or better
   - **Boot disk**: Ubuntu 20.04 LTS, 250 GB

### Step 2: SSH and Setup

```bash
# SSH into instance (via GCP console or gcloud)
gcloud compute ssh eigen2-training --zone=us-central1-a

# Install CUDA and PyTorch
sudo apt update
sudo apt install -y python3-pip git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone repository
cd ~
git clone https://github.com/SPhoenix-00/eigen2.git
cd eigen2

# Install dependencies
python3 -m pip install --no-cache-dir -r requirements.txt google-cloud-storage
```

### Step 3: Setup GCS Access and Download Data

```bash
# GCP VMs have automatic GCS access if using same project
# Set environment variables
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0

# Download training data
python3 << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
blob = bucket.blob('eigen2/Eigen2_Master(GFIN)_03_training.csv')
blob.download_to_filename('Eigen2_Master(GFIN)_03_training.csv')
EOF
```

### Step 4: Login to W&B and Train with tmux

```bash
# Login to wandb
wandb login

# Start tmux session
tmux new -s training

# Start training
python main.py
```

**Detach from tmux**: Press `Ctrl+B`, then `D`
**Reattach later**: `tmux attach -t training`

---

## Monitor Training

### Check Progress (from within tmux)

```bash
# If you're inside tmux session with running training:
# Press Ctrl+B, then [ to enter scroll mode
# Use arrow keys or Page Up/Down to scroll
# Press Q to exit scroll mode

# Watch logs (in a new window/pane)
tail -f logs/training.log

# Check current generation
cat checkpoints/<run-name>/trainer_state.json

# Check GPU usage
nvidia-smi

# View W&B dashboard
# Go to: https://wandb.ai/your-username/eigen2-self
```

### Check Training from Outside tmux

```bash
# SSH back into your instance
ssh root@YOUR_RUNPOD_HOST -p YOUR_PORT

# List tmux sessions
tmux ls

# Attach to your training session
tmux attach -t training

# You can also check status without attaching:
nvidia-smi  # Check if GPU is being used
ps aux | grep python  # Check if training is running
tail -f /workspace/logs/training.log  # Follow logs
```

### Verify GCS Sync

After first generation, check your bucket:
1. Go to: https://console.cloud.google.com/storage/browser/eigen2-checkpoints-ase0
2. Navigate to: `eigen2/checkpoints/<run-name>/`
3. You should see checkpoint files!

---

## Download Results

### Download Checkpoints and Logs

**After training completes:**

```bash
# Using gcloud CLI (install from: https://cloud.google.com/sdk/docs/install)
gcloud auth activate-service-account --key-file=gcs-credentials.json

# Download all checkpoints
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/checkpoints/ ./checkpoints/

# Download logs
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/logs/ ./logs/
```

### Download Metrics to CSV

**Download W&B metrics for analysis:**

```bash
# Download metrics from last run
python download_metrics.py

# Download metrics from specific run
python download_metrics.py breezy-puddle-62

# Download metrics from all runs in project
python download_metrics.py --all
```

**Output:**
- Creates `<run-name>_metrics.csv` in current directory
- Contains all logged metrics: fitness, win rate, trades, etc.
- Ready for analysis in Excel, pandas, or other tools

**Example CSV columns:**
- `generation` - Generation number
- `fitness/mean` - Mean population fitness
- `fitness/max` - Maximum fitness in generation
- `validation/fitness` - Validation performance
- `validation/win_rate` - Win rate on validation set
- And many more...

---

## Troubleshooting

### tmux Issues

**"session not found"**
```bash
# List all sessions to see what exists
tmux ls

# If no sessions, your training may have stopped
# Check if Python is still running:
ps aux | grep python
```

**"tmux: command not found"**
```bash
# Install tmux (RunPod usually has it pre-installed)
apt update && apt install -y tmux
```

**Accidentally closed tmux window**
```bash
# Don't panic! The session is still running
# Just reattach:
tmux attach -t training
```

**Need to kill stuck training**
```bash
# Attach to session
tmux attach -t training

# Press Ctrl+C to stop training

# Or kill the entire session:
tmux kill-session -t training
```

**Multiple sessions confusion**
```bash
# List all sessions with details
tmux ls

# Attach to specific session
tmux attach -t training

# Kill all sessions (careful!)
tmux kill-server
```

**Session keeps the old environment variables**
```bash
# Environment variables are set when you create the session
# If you need to change them:
# 1. Kill the old session: tmux kill-session -t training
# 2. Start new session: tmux new -s training
# 3. Set new variables: export CLOUD_PROVIDER=gcs (etc.)
# 4. Run training
```

### "Can't find training data"
```bash
cd /workspace  # or ~/eigen2 for GCP
ls -la Eigen2_Master*

# If missing, download again:
python3 << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
blob = bucket.blob('eigen2/Eigen2_Master(GFIN)_03_training.csv')
blob.download_to_filename('Eigen2_Master(GFIN)_03_training.csv')
EOF
```

### "GCS connection issues"
```bash
# Check environment variables
echo $GOOGLE_APPLICATION_CREDENTIALS
echo $CLOUD_BUCKET
echo $CLOUD_PROVIDER

# Verify credentials file
cat $GOOGLE_APPLICATION_CREDENTIALS | head

# Re-export if needed
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
```

### "wandb: ERROR api_key not configured"
```bash
# Login to wandb
wandb login

# Get your API key from: https://wandb.ai/authorize
# Paste it when prompted
```

### "CUDA out of memory"
- Requires GPU with 24GB+ VRAM
- Use RTX 4090, A100, or similar
- Check GPU memory: `nvidia-smi`

---

## What Gets Backed Up to GCS

**Every Generation:**
- `checkpoints/<run-name>/best_agent.pth` - Best performing agent
- `checkpoints/<run-name>/population/agent_*.pth` - All 16 agents
- `checkpoints/<run-name>/trainer_state.json` - Training progress

**Automatic Features:**
- ✅ Background uploads (training never blocks)
- ✅ Run-specific directories (parallel runs don't conflict)
- ✅ Automatic resume from cloud checkpoints
- ✅ 4 parallel upload threads

---

## Cost Estimates

| Provider | GPU | Cost/Hour | 20 Gen Training | Total |
|----------|-----|-----------|-----------------|-------|
| **RunPod** | RTX 4090 | $0.40-0.70 | ~10-15 hours | **$4-10** |
| **Vast.ai** | RTX 4090 | $0.30-0.60 | ~10-15 hours | **$3-9** |
| **GCP** | T4 | $0.35 | ~20 hours | **$7** |
| **GCP** | A100 | $2.93 | ~6 hours | **$18** |
| **GCS Storage** | - | $0.020/GB/month | - | **$0.01/month** |

---

## Key Features

✅ **Run-specific checkpoints** - Each wandb run gets its own directory
✅ **Unique random seeds** - Parallel runs have independent behavior
✅ **Automatic cloud sync** - Background uploads, no training interruption
✅ **Training data from GCS** - No need to upload large CSV files
✅ **W&B integration** - Track all metrics and experiments
✅ **Auto-resume** - Continues from last checkpoint on restart

---

## Quick Reference

**Essential tmux Commands:**
```bash
# Start new session
tmux new -s training

# Detach (training keeps running)
Ctrl+B, then D

# List sessions
tmux ls

# Reattach to session
tmux attach -t training

# Kill session
tmux kill-session -t training
```

**Essential Environment Variables:**
```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
```

**Complete Workflow:**
```bash
# 1. SSH into instance
ssh root@YOUR_RUNPOD_HOST -p YOUR_PORT

# 2. Start tmux session
cd /workspace
tmux new -s training

# 3. Set environment variables
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# 4. Start training
python main.py              # New training
python main.py --resume     # Resume from last run

# 5. Detach from tmux
# Press: Ctrl+B, then D

# 6. Disconnect SSH (training continues!)
exit

# 7. Later: Reconnect and reattach
ssh root@YOUR_RUNPOD_HOST -p YOUR_PORT
tmux attach -t training
```

**Resume from Specific Run:**
```bash
# Resume from a specific older run
python main.py --resume-run <run-name>
# Example: python main.py --resume-run azure-thunder-123

# Note: Run info is automatically saved to last_run.json for easy resume
# To see last run: cat last_run.json
```

**Monitor:**
```bash
# Check if training is running (without attaching to tmux)
ps aux | grep python
nvidia-smi

# View logs
tail -f logs/training.log

# Attach to tmux to see live output
tmux attach -t training

# W&B Dashboard
https://wandb.ai/your-username/eigen2-self

# Download metrics to CSV
python download_metrics.py
```

---

## Configuration

**Your Setup:**
- **GCS Bucket**: `eigen2-checkpoints-ase0`
- **GitHub**: https://github.com/SPhoenix-00/eigen2.git
- **W&B Project**: `eigen2-self`
- **Training Data**: `eigen2/Eigen2_Master(GFIN)_03_training.csv` (in GCS)

**Current Training Settings:**
- Generations: 20
- Population: 16 agents
- Buffer size: 14,000
- Checkpoints: Every generation

---

**Ready to train!** 🚀
