# Quick Start: Vast.ai Deployment (GitHub Approach)

## 3-Minute Setup - No Docker Hub Required! 🎉

### Prerequisites

1. ✅ **GitHub repository** with your code pushed
2. ✅ **Vast.ai account** with payment method
3. ✅ **Cloud storage** (AWS S3, GCS, or Azure) - *optional but recommended*
4. ✅ **Training data CSV** file ready to upload

---

## Setup Steps

### Step 1: Push Code to GitHub

```bash
# Commit and push your code to GitHub
git add .
git commit -m "Ready for Vast.ai deployment"
git push origin main
```

That's it! No Docker image building required.

### Step 2: Create Vast.ai Instance

**GPU Requirements:**
- RTX 4090 or similar (24GB+ VRAM)
- 50GB+ disk space
- CUDA 12.1+

**Template Settings:**

| Field | Value |
|-------|-------|
| **Docker Image** | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| **Disk Space** | 50 GB |
| **On-start Script** | See below ⬇️ |

**On-start Script:**
```bash
cd /workspace
export GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
export GITHUB_BRANCH=main
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .
bash start_vast.sh --resume
```

**Environment Variables (example with AWS S3):**
```
CLOUD_PROVIDER=s3
CLOUD_BUCKET=my-eigen2-checkpoints
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
GITHUB_BRANCH=main
```

### Step 3: Upload Training Data

Once instance starts, SSH in and upload your data:

```bash
# From your local machine
scp Eigen2_Master\(GFIN\)_03_training.csv root@your-vast-ip:/workspace/
```

### Step 4: Monitor Training

```bash
# SSH into instance
ssh root@your-vast-ip

# Watch training progress
tail -f /workspace/logs/training.log

# Check current generation
cat /workspace/checkpoints/trainer_state.json
```

---

## Cloud Provider Configuration

### Option 1: AWS S3 (Recommended)

```bash
# Environment variables for Vast.ai
CLOUD_PROVIDER=s3
CLOUD_BUCKET=my-checkpoints
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

**Setup:**
1. Create S3 bucket: https://s3.console.aws.amazon.com
2. Create IAM user with S3 access
3. Copy access key and secret key

### Option 2: Google Cloud Storage

```bash
# Environment variables
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=my-checkpoints
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-creds.json
```

**Setup:**
1. Create GCS bucket: https://console.cloud.google.com/storage
2. Create service account with Storage Admin role
3. Download JSON credentials
4. Upload to instance: `scp gcs-creds.json root@vast-ip:/workspace/`

### Option 3: Azure Blob Storage

```bash
# Environment variables
CLOUD_PROVIDER=azure
CLOUD_BUCKET=my-container
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
```

**Setup:**
1. Create storage account: https://portal.azure.com
2. Create container
3. Copy connection string from "Access Keys"

### Option 4: Local Only (No Cloud)

```bash
CLOUD_PROVIDER=local
```

⚠️ **Warning:** Checkpoints will be lost when instance terminates!

---

## Complete On-start Script Examples

### With AWS S3
```bash
#!/bin/bash
cd /workspace
export GITHUB_REPO=https://github.com/SPhoenix-00/eigen2.git
export GITHUB_BRANCH=main
export CLOUD_PROVIDER=s3
export CLOUD_BUCKET=my-checkpoints

# Clone repository
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

# Wait for data file to be uploaded
while [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ]; do
    echo "Waiting for training data file..."
    sleep 10
done

# Start training
bash start_vast.sh --resume
```

### With GCS
```bash
#!/bin/bash
cd /workspace
export GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
export GITHUB_BRANCH=main
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=my-checkpoints
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-creds.json

# Clone repository
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

# Wait for files
while [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ] || [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; do
    echo "Waiting for training data and credentials..."
    sleep 10
done

# Start training
bash start_vast.sh --resume
```

### Local Only (No Cloud)
```bash
#!/bin/bash
cd /workspace
export GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
export GITHUB_BRANCH=main
export CLOUD_PROVIDER=local

# Clone repository
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

# Wait for data file
while [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ]; do
    echo "Waiting for training data file..."
    sleep 10
done

# Start training
bash start_vast.sh --resume
```

---

## Advantages of GitHub Approach

✅ **No Docker Hub account needed**
✅ **No custom image building**
✅ **Faster updates** - just `git push`
✅ **Version control built-in**
✅ **Easy collaboration**
✅ **Simpler workflow**

---

## Common Commands

### Check Training Progress
```bash
# View current generation
cat checkpoints/trainer_state.json

# Watch live logs
tail -f logs/training.log

# Check GPU usage
nvidia-smi
```

### Download Checkpoints

```bash
# AWS S3
aws s3 sync s3://my-checkpoints/eigen2/checkpoints/ ./checkpoints/

# GCS
gsutil -m rsync -r gs://my-checkpoints/eigen2/checkpoints/ ./checkpoints/

# Azure
az storage blob download-batch \
  --source my-container \
  --pattern "eigen2/checkpoints/*" \
  --destination ./checkpoints/
```

### Update Code and Resume Training

```bash
# Push changes to GitHub
git add .
git commit -m "Fix bug"
git push

# On Vast.ai instance
cd /workspace
git pull
bash start_vast.sh --resume
```

---

## Cost Estimate

| Item | Cost |
|------|------|
| RTX 4090 on Vast.ai | $0.30-0.60/hour |
| Full training (20 gen) | 10-20 hours |
| **Total compute cost** | **$3-12** |
| Cloud storage (S3/GCS/Azure) | ~$0.01/month |
| **Grand total** | **~$3-12 per run** |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Repository not found" | Check GitHub URL and ensure repo is public (or use SSH key) |
| "Permission denied" | Make repo public or add SSH key to Vast.ai instance |
| "No data file found" | Upload CSV with: `scp file.csv root@vast-ip:/workspace/` |
| "CUDA out of memory" | Use GPU with 24GB+ VRAM (RTX 4090 or better) |
| "Cloud sync failed" | Verify cloud credentials and bucket name |
| Git clone fails | Check internet connectivity on instance |

---

## Private Repository Access

If your repository is private:

**Option 1: Personal Access Token (Easiest)**
```bash
# In on-start script, use:
git clone https://YOUR_TOKEN@github.com/YOUR-USERNAME/eigen2.git .
```

**Option 2: SSH Key**
```bash
# Generate SSH key on Vast.ai instance
ssh-keygen -t ed25519 -f ~/.ssh/id_vast -N ""

# Copy public key and add to GitHub
cat ~/.ssh/id_vast.pub

# Clone with SSH
git clone git@github.com:YOUR-USERNAME/eigen2.git .
```

---

## Best Practices

1. ✅ **Always use cloud storage** for checkpoints (S3/GCS/Azure)
2. ✅ **Use `--resume`** to continue from checkpoints
3. ✅ **Monitor costs** - terminate instance when done
4. ✅ **Test locally first** before deploying to Vast.ai
5. ✅ **Keep GitHub repo updated** - push all changes
6. ❌ **Don't** commit credentials or data files to Git
7. ❌ **Don't** leave instances running idle

---

## Full Workflow Example

```bash
# 1. Push code to GitHub (local machine)
git add .
git commit -m "Ready for training"
git push origin main

# 2. Create Vast.ai instance with:
#    - Docker: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
#    - On-start script (see above)
#    - Environment variables for cloud storage

# 3. Upload data
scp Eigen2_Master\(GFIN\)_03_training.csv root@vast-ip:/workspace/

# 4. Training starts automatically!

# 5. Monitor progress
ssh root@vast-ip
tail -f logs/training.log

# 6. Download results when done
aws s3 sync s3://my-bucket/eigen2/checkpoints/ ./checkpoints/

# 7. Terminate instance to stop charges
```

---

## What Gets Synced to Cloud

**Every 5 Generations:**
- ✅ `best_agent.pth` - Best performing agent
- ✅ `population/agent_*.pth` - All population agents
- ✅ `trainer_state.json` - Training progress

**At End of Training:**
- ✅ Final checkpoints
- ✅ TensorBoard logs

---

## Need Help?

- **Vast.ai Issues**: https://vast.ai/faq or Discord
- **Cloud Storage**: Check provider documentation
- **GitHub**: https://docs.github.com
- **Full Guide**: See [VAST_DEPLOYMENT.md](VAST_DEPLOYMENT.md)

---

## Summary

**GitHub approach is simpler:**
- ❌ No Docker Hub account
- ❌ No custom image building
- ❌ No `docker push` commands
- ✅ Just `git push` and go!

**Total setup time: ~3 minutes** ⚡
