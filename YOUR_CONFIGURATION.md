# Your Eigen2 Configuration Reference

Quick reference for your specific setup. Copy-paste values from here.

---

## Google Cloud Storage

**Bucket Name:**
```
eigen2-checkpoints-ase0
```

**Bucket URL:**
```
https://console.cloud.google.com/storage/browser/eigen2-checkpoints-ase0
```

**Credentials File:**
```
gcs-credentials.json
```

---

## Environment Variables for Vast.ai

Copy these exactly into your Vast.ai instance configuration:

```bash
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=eigen2-checkpoints-ase0
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
GITHUB_BRANCH=main
```

**⚠️ Replace `YOUR-USERNAME` with your actual GitHub username!**

---

## On-start Script for Vast.ai

Copy-paste this into Vast.ai "On-start Script" field:

```bash
#!/bin/bash
cd /workspace

# Configuration
export GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
export GITHUB_BRANCH=main
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# Clone repository
echo "Cloning repository from GitHub..."
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

# Wait for required files
echo "Waiting for credentials and training data..."
while [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ]; do
    echo "Please upload to /workspace/:"
    echo "  - gcs-credentials.json"
    echo "  - Eigen2_Master(GFIN)_03_training.csv"
    sleep 10
done

# Install dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt google-cloud-storage

# Start training
echo "Starting training with GCS checkpoint backup..."
bash start_vast.sh --resume
```

**⚠️ Replace `YOUR-USERNAME` with your actual GitHub username!**

---

## Files to Upload to Vast.ai

After instance starts, SSH in and upload these files:

```bash
# From your local machine:
scp gcs-credentials.json root@YOUR-VAST-IP:/workspace/
scp Eigen2_Master\(GFIN\)_03_training.csv root@YOUR-VAST-IP:/workspace/
```

---

## Download Checkpoints

After training completes, download results to your local machine:

```bash
# Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth activate-service-account --key-file=gcs-credentials.json

# Download checkpoints
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/checkpoints/ ./checkpoints/

# Download logs
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/logs/ ./logs/
```

---

## Verify Configuration

### Check Environment Variables (on Vast.ai instance)
```bash
echo $CLOUD_BUCKET
# Should output: eigen2-checkpoints-ase0

echo $GOOGLE_APPLICATION_CREDENTIALS
# Should output: /workspace/gcs-credentials.json

echo $CLOUD_PROVIDER
# Should output: gcs
```

### Check Files Exist
```bash
ls -la /workspace/gcs-credentials.json
ls -la /workspace/Eigen2_Master\(GFIN\)_03_training.csv
```

### Check GCS Connection
After training starts, look for:
```
✓ Connected to Google Cloud Storage bucket: eigen2-checkpoints-ase0
```

---

## Quick Links

**GCP Storage Console:**
```
https://console.cloud.google.com/storage/browser/eigen2-checkpoints-ase0
```

**GCP IAM Service Accounts:**
```
https://console.cloud.google.com/iam-admin/serviceaccounts
```

**Vast.ai Console:**
```
https://vast.ai/console/instances/
```

**Vast.ai Create Instance:**
```
https://vast.ai/console/create/
```

---

## Training Progress Commands

```bash
# SSH into instance
ssh root@YOUR-VAST-IP

# Watch training logs in real-time
tail -f logs/training.log

# Check current generation
cat checkpoints/trainer_state.json

# Example output:
# {"generation": 5, "best_fitness": -150.42}

# Check GPU usage
nvidia-smi

# Check if checkpoints are syncing
ls -lh checkpoints/
ls -lh checkpoints/population/
```

---

## Expected Training Timeline

| Generation | Time | Checkpoints Synced |
|------------|------|-------------------|
| 1-4 | ~2-4 hours | No (waiting for gen 5) |
| 5 | ~5-6 hours | ✅ First sync to GCS |
| 10 | ~10-12 hours | ✅ Second sync |
| 15 | ~15-18 hours | ✅ Third sync |
| 20 | ~20-24 hours | ✅ Final sync + logs |

---

## Cost Tracking

**Per training run:**
- GCS Storage: ~0.5 GB = $0.01/month
- Vast.ai GPU (RTX 4090): 20 hours × $0.40/hr = ~$8
- **Total: ~$8 per complete training run**

---

## Troubleshooting Quick Fixes

### Training not starting
```bash
# Check files exist
ls -la /workspace/gcs-credentials.json
ls -la /workspace/Eigen2_Master\(GFIN\)_03_training.csv

# If missing, upload them
```

### Cloud sync not working
```bash
# Verify credentials
cat /workspace/gcs-credentials.json | head

# Check environment
echo $CLOUD_BUCKET  # Should be: eigen2-checkpoints-ase0
echo $CLOUD_PROVIDER  # Should be: gcs
echo $GOOGLE_APPLICATION_CREDENTIALS  # Should be: /workspace/gcs-credentials.json
```

### Can't connect to GCS
```bash
# Test credentials manually
pip install google-cloud-storage
python3 << EOF
from google.cloud import storage
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/workspace/gcs-credentials.json'
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-ase0')
print(f"✓ Connected to bucket: {bucket.name}")
EOF
```

---

## Checklist

- [x] GCS bucket created: `eigen2-checkpoints-ase0`
- [ ] Service account created with "Storage Object Admin" role
- [ ] Credentials file downloaded: `gcs-credentials.json`
- [ ] Code pushed to GitHub
- [ ] GitHub repository URL updated in on-start script
- [ ] Training data CSV ready: `Eigen2_Master(GFIN)_03_training.csv`
- [ ] Vast.ai account created with payment method
- [ ] Environment variables configured (see above)
- [ ] On-start script copied (see above)
- [ ] Ready to deploy! 🚀

---

## Next Steps

1. ✅ Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Configured for GCS deployment"
   git push origin main
   ```

2. ✅ Create Vast.ai instance:
   - Go to: https://vast.ai/console/create/
   - Docker image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
   - GPU: RTX 4090 (24GB+ VRAM)
   - Disk: 50 GB
   - Environment variables: (see above)
   - On-start script: (see above)

3. ✅ Upload files to instance:
   ```bash
   scp gcs-credentials.json root@VAST-IP:/workspace/
   scp Eigen2_Master\(GFIN\)_03_training.csv root@VAST-IP:/workspace/
   ```

4. ✅ Training starts automatically!

5. ✅ Monitor progress:
   ```bash
   ssh root@VAST-IP
   tail -f logs/training.log
   ```

6. ✅ Download results when done:
   ```bash
   gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/checkpoints/ ./checkpoints/
   ```

---

**Questions?** See:
- [QUICK_START_GCP.md](QUICK_START_GCP.md) - Fast deployment guide
- [GCP_SETUP_GUIDE.md](GCP_SETUP_GUIDE.md) - Detailed GCP setup
- [QUICK_START_VAST.md](QUICK_START_VAST.md) - General Vast.ai guide
