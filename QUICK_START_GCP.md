# Quick Start: Vast.ai + Google Cloud Storage

**Total time: 15 minutes** (10 min GCP setup + 5 min Vast.ai deployment)

---

## Part 1: Google Cloud Setup (One-time, ~10 minutes)

### Step 1: Create GCS Bucket

✅ **Already created:** `eigen2-checkpoints-ase0`

If you need to create another bucket:
1. Go to: **https://console.cloud.google.com/storage**
2. Click **"CREATE BUCKET"**
3. Name must be globally unique
4. Location: **Region** → `us-central1` (closest to Vast.ai)
5. Storage class: **Standard**
6. Click **"CREATE"**

### Step 2: Create Service Account

1. Go to: **https://console.cloud.google.com/iam-admin/serviceaccounts**
2. Click **"CREATE SERVICE ACCOUNT"**
3. Name: `eigen2-storage`
4. Click **"CREATE AND CONTINUE"**
5. Role: Search and select **"Storage Object Admin"**
6. Click **"CONTINUE"** → **"DONE"**

### Step 3: Download Credentials

1. Find `eigen2-storage@...` in the list
2. Click **⋮ (three dots)** → **"Manage keys"**
3. **"ADD KEY"** → **"Create new key"**
4. Format: **"JSON"**
5. Click **"CREATE"** (file downloads automatically)
6. Rename to: `gcs-credentials.json`
7. **Keep this file safe!**

✅ **You now have:** `gcs-credentials.json` in your Downloads

---

## Part 2: Vast.ai Deployment (~5 minutes)

### Step 1: Push Code to GitHub

```bash
cd eigen2
git add .
git commit -m "Ready for Vast.ai deployment"
git push origin main
```

### Step 2: Create Vast.ai Instance

Go to: **https://vast.ai/console/create/**

**Instance Configuration:**

| Setting | Value |
|---------|-------|
| **GPU** | RTX 4090 (or 24GB+ VRAM) |
| **Docker Image** | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| **Disk Space** | 50 GB |

**Environment Variables:**
```
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=eigen2-checkpoints-ase0
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
GITHUB_BRANCH=main
```

**On-start Script:**
```bash
#!/bin/bash
cd /workspace
git clone --depth 1 https://github.com/SPhoenix-00/eigen2.git .
bash start_vast.sh --resume
```

Click **"RENT"** and wait for instance to start (~1 minute)

### Step 3: Upload Files

**SSH into your instance:**
```bash
ssh root@YOUR-VAST-IP
```

**From your local machine, upload files:**
```bash
# Upload credentials
scp gcs-credentials.json root@YOUR-VAST-IP:/workspace/

# Upload training data
scp Eigen2_Master\(GFIN\)_03_training.csv root@YOUR-VAST-IP:/workspace/
```

### Step 4: Training Starts Automatically! 🎉

The on-start script will:
1. ✅ Clone code from GitHub
2. ✅ Wait for credentials and data
3. ✅ Install dependencies
4. ✅ Start training with GCS backup

---

## Monitoring Training

### Check Progress

```bash
# SSH into instance
ssh root@YOUR-VAST-IP

# Watch logs
tail -f logs/training.log

# Check current generation
cat checkpoints/trainer_state.json
```

### Verify GCS Sync

After 5 generations, check your bucket:
1. Go to: https://console.cloud.google.com/storage/browser
2. Click: `eigen2-checkpoints-ase0`
3. Navigate to: `eigen2/checkpoints/`
4. You should see checkpoint files!

---

## Complete On-start Script (Copy-Paste Ready)

Replace `YOUR-USERNAME` with your GitHub username:

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
echo "Cloning repository..."
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

# Wait for files
echo "Waiting for required files..."
while [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ]; do
    echo "  Upload gcs-credentials.json and training data CSV to /workspace/"
    sleep 10
done

echo "✓ Files found. Installing dependencies..."
pip install --no-cache-dir -r requirements.txt google-cloud-storage

echo "✓ Starting training..."
bash start_vast.sh --resume
```

---

## What Gets Backed Up to GCS

**Every 5 generations:**
- `best_agent.pth` - Best performing agent
- `population/agent_*.pth` - All agents (16 files)
- `trainer_state.json` - Training progress

**At end of training:**
- Final checkpoints
- TensorBoard logs

**Total size:** ~500 MB per training run
**Cost:** ~$0.01/month per run

---

## If Instance Gets Interrupted

No problem! Just:

1. Create new Vast.ai instance (same config)
2. Upload files again:
   ```bash
   scp gcs-credentials.json root@NEW-IP:/workspace/
   scp Eigen2_Master\(GFIN\)_03_training.csv root@NEW-IP:/workspace/
   ```
3. Training automatically downloads checkpoints from GCS
4. Training resumes from last generation

---

## Download Results

### After Training Completes

**Method 1: Using gsutil (Recommended)**
```bash
# Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
gcloud auth activate-service-account --key-file=gcs-credentials.json

# Download checkpoints
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/checkpoints/ ./checkpoints/

# Download logs
gsutil -m rsync -r gs://eigen2-checkpoints-ase0/eigen2/logs/ ./logs/
```

**Method 2: Web Interface**
1. Go to: https://console.cloud.google.com/storage/browser
2. Navigate to your bucket
3. Select files → Download

---

## Costs

| Item | Cost |
|------|------|
| **GCS Storage** | $0.020/GB/month |
| **Typical usage** | ~0.5 GB per run = **$0.01/month** |
| **Vast.ai RTX 4090** | $0.30-0.60/hour |
| **Full training** | 10-20 hours = **$3-12** |
| **Total** | **~$3-12 per training run** |

---

## Troubleshooting

### "Could not connect to GCS"
```bash
# Check credentials file exists
ls -la /workspace/gcs-credentials.json

# Verify environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS

# Check it's valid JSON
cat /workspace/gcs-credentials.json | head
```

### "Permission denied"
- Re-download credentials from GCP Console
- Verify service account has "Storage Object Admin" role
- Check bucket name matches exactly

### "Bucket not found"
```bash
# Verify bucket name
echo $CLOUD_BUCKET

# Should output: eigen2-checkpoints-ase0
# If wrong, update environment variable
```

---

## Checklist Before Starting

- [x] GCS bucket created: `eigen2-checkpoints-ase0`
- [ ] Service account created with "Storage Object Admin"
- [ ] Credentials downloaded: `gcs-credentials.json`
- [ ] Code pushed to GitHub
- [ ] Training data CSV ready
- [ ] Vast.ai account with payment method
- [ ] Environment variables configured (see above)

**Ready to go!** 🚀

---

## Full Example (Your Actual Configuration)

**GCS Configuration:**
```
Bucket Name: eigen2-checkpoints-ase0
Credentials: gcs-credentials.json
```

**GitHub:**
```
Repository: https://github.com/YOUR-USERNAME/eigen2.git
Branch: main
```

**Vast.ai Environment Variables:**
```
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=eigen2-checkpoints-ase0
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
GITHUB_BRANCH=main
```

**On-start Script:**
```bash
#!/bin/bash
cd /workspace
export GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
export GITHUB_BRANCH=main
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

while [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ]; do
    echo "Waiting for files..."
    sleep 10
done

pip install --no-cache-dir -r requirements.txt google-cloud-storage
bash start_vast.sh --resume
```

---

## Next Steps

1. ✅ Complete GCS setup (see [GCP_SETUP_GUIDE.md](GCP_SETUP_GUIDE.md) for details)
2. ✅ Push code to GitHub
3. ✅ Create Vast.ai instance with above configuration
4. ✅ Upload credentials and data
5. ✅ Monitor training progress
6. ✅ Download results when complete

**Need more details?** See [GCP_SETUP_GUIDE.md](GCP_SETUP_GUIDE.md) for full walkthrough.

**Questions?**
- GCP: https://console.cloud.google.com
- Vast.ai: https://vast.ai/faq
- GitHub: Your repository issues page
