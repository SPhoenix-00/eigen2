# Quick Start: RunPod + GCS

**Time: 5 minutes**

---

## Step 1: Create RunPod Instance

1. Go to: **https://www.runpod.io/console/gpu-cloud**
2. Select GPU: **RTX 4090** (or 24GB+ VRAM)
3. Template: **PyTorch 2.1** (or similar CUDA 12.1+ image)
4. Disk: **50 GB**
5. Click **"Deploy"**

---

## Step 2: Connect and Setup

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

# Install dependencies (IMPORTANT: Do this BEFORE uploading files)
pip install --no-cache-dir -r requirements.txt google-cloud-storage
```

**⚠️ Note:** Leave this terminal open - you'll need to upload files next, then set environment variables.

---

## Step 3: Upload Required Files

**From your local machine:**
```bash
# Upload GCS credentials
scp -P YOUR_PORT gcs-credentials.json root@YOUR_HOST:/workspace/

# Upload training data
scp -P YOUR_PORT "Eigen2_Master(GFIN)_03_training.csv" root@YOUR_HOST:/workspace/
```

---

## Step 4: Set Environment Variables and Start Training

**Back on RunPod instance:**
```bash
cd /workspace

# Verify files exist
ls -la gcs-credentials.json
ls -la Eigen2_Master\(GFIN\)_03_training.csv

# Set environment variables (REQUIRED!)
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# Verify environment is set correctly
echo "Provider: $CLOUD_PROVIDER"
echo "Bucket: $CLOUD_BUCKET"
echo "Credentials: $GOOGLE_APPLICATION_CREDENTIALS"

# Start training
python main.py

# Or use the start script with resume
bash start_vast.sh --resume
```

**⚠️ Important:** The environment variables MUST be set in the same terminal session before running training!

---

## Monitor Training

```bash
# Watch logs
tail -f logs/training.log

# Check current generation
cat checkpoints/trainer_state.json

# Check GPU usage
nvidia-smi

# Verify GCS sync (after generation 5)
# Go to: https://console.cloud.google.com/storage/browser/eigen2-checkpoints-ase0
```

---

## Quick Troubleshooting

**Can't find training data:**
```bash
cd /workspace
ls -la Eigen2_Master*
```

**GCS connection issues:**
```bash
# Check if google-cloud-storage is installed
pip list | grep google-cloud-storage

# Check environment variables
echo $GOOGLE_APPLICATION_CREDENTIALS
echo $CLOUD_BUCKET
echo $CLOUD_PROVIDER

# Check credentials file
cat gcs-credentials.json | head

# If variables are not set, re-export them:
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
```

**Resume from checkpoint:**
```bash
# Training automatically resumes if checkpoints exist
python main.py
```

---

## Configuration

**Your Setup:**
- Bucket: `eigen2-checkpoints-ase0`
- Credentials: `gcs-credentials.json`
- GitHub: `https://github.com/SPhoenix-00/eigen2.git`

**Training Settings:**
- Generations: 25
- Checkpoints: Every generation
- Buffer saves: Every 5 generations (5, 10, 15, 20, 25)

---

## Download Results

**After training completes:**
```bash
# Install gcloud SDK locally: https://cloud.google.com/sdk/docs/install
gcloud auth activate-service-account --key-file=gcs-credentials.json

# Download checkpoints
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/checkpoints/ ./checkpoints/

# Download logs
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/logs/ ./logs/
```

---

**Done!** Training will sync to GCS automatically. 🚀
