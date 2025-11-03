# GCP Setup Guide for Eigen2 Cloud Storage

Complete step-by-step guide to set up Google Cloud Storage for checkpoint syncing.

---

## Overview

We'll configure Google Cloud Storage (GCS) to automatically backup your training checkpoints every 5 generations. If your Vast.ai instance is interrupted, you can resume training from the last checkpoint.

**What you'll set up:**
1. Google Cloud project
2. Storage bucket for checkpoints
3. Service account with credentials
4. Environment variables for Vast.ai

**Time required:** ~10 minutes
**Cost:** ~$0.02/month for storage (~1 GB of checkpoints)

---

## Prerequisites

- Google account (Gmail, etc.)
- Credit card for Google Cloud billing (required even for free tier)
- No previous GCP experience needed!

---

## Step 1: Create Google Cloud Project

### 1.1 Go to Google Cloud Console
👉 https://console.cloud.google.com

### 1.2 Create New Project
1. Click **"Select a project"** dropdown at the top
2. Click **"NEW PROJECT"**
3. Enter project details:
   - **Project name:** `eigen2-training` (or your preferred name)
   - **Organization:** Leave as "No organization" (unless you have one)
4. Click **"CREATE"**
5. Wait ~30 seconds for project to be created
6. **Select your new project** from the dropdown

✅ **Project created!** Note your project ID (shows below project name)

---

## Step 2: Enable Billing (Required)

### 2.1 Enable Billing for Your Project
1. Go to: https://console.cloud.google.com/billing
2. Click **"LINK A BILLING ACCOUNT"**
3. If you don't have one:
   - Click **"CREATE BILLING ACCOUNT"**
   - Enter credit card information
   - Google offers $300 free credits for new accounts!
4. Link the billing account to your `eigen2-training` project

**Note:** Storage costs ~$0.02/GB/month. Even with 10 training runs, you'll spend less than $0.50/month.

✅ **Billing enabled!**

---

## Step 3: Create Storage Bucket

### 3.1 Go to Cloud Storage
👉 https://console.cloud.google.com/storage/browser

Or: Navigate to **Storage** → **Cloud Storage** → **Buckets**

### 3.2 Create Bucket
1. Click **"CREATE BUCKET"** or **"CREATE"**
2. Enter bucket details:

**Step 1 - Name your bucket:**
- **Bucket name:** `eigen2-checkpoints-YOURNAME`
  - Must be globally unique
  - Example: `eigen2-checkpoints-john123`
  - Only lowercase letters, numbers, hyphens
  - **Write this down!** You'll need it later

**Step 2 - Choose where to store data:**
- **Location type:** Select **"Region"** (cheapest)
- **Region:** Choose closest to you or Vast.ai:
  - US: `us-central1` (Iowa) - recommended for Vast.ai
  - EU: `europe-west1` (Belgium)
  - Asia: `asia-east1` (Taiwan)

**Step 3 - Choose storage class:**
- **Default class:** **"Standard"** (best for frequently accessed data)

**Step 4 - Control access:**
- **Access control:** **"Uniform"** (recommended)
- **Uncheck** "Enforce public access prevention" (we'll use service account)

**Step 5 - Data protection:**
- Leave defaults (optional: enable versioning for backup history)

3. Click **"CREATE"**

✅ **Bucket created!** Write down: `eigen2-checkpoints-YOURNAME`

---

## Step 4: Create Service Account

Service accounts allow your training script to access GCS securely.

### 4.1 Go to IAM & Admin
👉 https://console.cloud.google.com/iam-admin/serviceaccounts

Or: Navigate to **IAM & Admin** → **Service Accounts**

### 4.2 Create Service Account
1. Click **"CREATE SERVICE ACCOUNT"**

**Step 1 - Service account details:**
- **Service account name:** `eigen2-storage`
- **Service account ID:** (auto-filled: `eigen2-storage@...`)
- **Description:** `Service account for Eigen2 checkpoint storage`
- Click **"CREATE AND CONTINUE"**

**Step 2 - Grant this service account access:**
- Click **"Select a role"** dropdown
- Search for: `Storage Object Admin`
- Select: **"Storage Object Admin"**
  - This allows read/write to your bucket
- Click **"CONTINUE"**

**Step 3 - Grant users access:**
- Leave blank (optional)
- Click **"DONE"**

✅ **Service account created!**

---

## Step 5: Create Service Account Key

This creates a JSON file with credentials.

### 5.1 Download Credentials
1. On the Service Accounts page, find `eigen2-storage@...`
2. Click the **three dots (⋮)** on the right → **"Manage keys"**
3. Click **"ADD KEY"** → **"Create new key"**
4. Select **"JSON"** format
5. Click **"CREATE"**

**A JSON file will download automatically:**
- Filename: `eigen2-training-xxxxx.json`
- **Keep this file safe!** It contains your credentials
- **Never commit to Git!** (already in .gitignore)

### 5.2 Rename the File (Optional)
For simplicity, rename to:
```bash
mv eigen2-training-xxxxx.json gcs-credentials.json
```

✅ **Credentials downloaded!** Location: Your Downloads folder

---

## Step 6: Test Your Setup (Optional but Recommended)

### 6.1 Install Google Cloud SDK (Optional)
Only needed for testing from your local machine.

**Windows:**
👉 https://cloud.google.com/sdk/docs/install#windows

**Mac/Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### 6.2 Test Upload/Download
```bash
# Authenticate with your service account
gcloud auth activate-service-account --key-file=gcs-credentials.json

# Test upload
echo "test" > test.txt
gsutil cp test.txt gs://eigen2-checkpoints-YOURNAME/test.txt

# Test download
gsutil cp gs://eigen2-checkpoints-YOURNAME/test.txt downloaded.txt
cat downloaded.txt  # Should show "test"

# Cleanup
rm test.txt downloaded.txt
gsutil rm gs://eigen2-checkpoints-YOURNAME/test.txt
```

If all commands succeed: ✅ **GCS setup is working!**

---

## Step 7: Configure for Vast.ai

### 7.1 Upload Credentials to Vast.ai Instance

After creating your Vast.ai instance, upload the credentials:

```bash
# From your local machine (where gcs-credentials.json is)
scp gcs-credentials.json root@YOUR-VAST-IP:/workspace/
```

**Alternative:** Use Vast.ai file upload in web interface:
1. SSH into instance
2. Click "Upload" button
3. Select `gcs-credentials.json`

### 7.2 Vast.ai Environment Variables

When creating your Vast.ai instance, set these environment variables:

```bash
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=eigen2-checkpoints-YOURNAME
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
GITHUB_BRANCH=main
```

Replace:
- `eigen2-checkpoints-YOURNAME` → Your actual bucket name
- `YOUR-USERNAME` → Your GitHub username

### 7.3 Complete Vast.ai On-start Script

```bash
#!/bin/bash
cd /workspace

# Set environment variables
export GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
export GITHUB_BRANCH=main
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen2-checkpoints-YOURNAME
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json

# Clone repository
echo "Cloning repository from GitHub..."
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

# Wait for credentials file to be uploaded
echo "Waiting for GCS credentials..."
while [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; do
    echo "  Please upload gcs-credentials.json to /workspace/"
    sleep 10
done
echo "✓ Credentials found"

# Wait for training data
echo "Waiting for training data..."
while [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ]; do
    echo "  Please upload training data CSV to /workspace/"
    sleep 10
done
echo "✓ Training data found"

# Install dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir google-cloud-storage

# Start training
echo "Starting training with GCS checkpoint backup..."
bash start_vast.sh --resume
```

---

## Step 8: Verify It's Working

### 8.1 Check Logs During Training

SSH into your Vast.ai instance:
```bash
ssh root@YOUR-VAST-IP
```

Look for these messages in the output:
```
✓ Connected to Google Cloud Storage bucket: eigen2-checkpoints-YOURNAME
```

After 5 generations, you should see:
```
============================================================
Syncing checkpoints to cloud storage...
============================================================
✓ Uploaded: checkpoints/best_agent.pth → eigen2/checkpoints/best_agent.pth
✓ Uploaded: checkpoints/trainer_state.json → eigen2/checkpoints/trainer_state.json
✓ Uploaded: checkpoints/population/agent_0.pth → eigen2/checkpoints/population/agent_0.pth
...
============================================================
```

### 8.2 Check GCS Bucket

1. Go to: https://console.cloud.google.com/storage/browser
2. Click your bucket: `eigen2-checkpoints-YOURNAME`
3. Navigate to: `eigen2/checkpoints/`
4. You should see:
   - `best_agent.pth`
   - `trainer_state.json`
   - `population/` folder with agent files

✅ **Checkpoints are syncing!**

---

## Common Issues & Solutions

### Issue: "Could not connect to GCS"

**Cause:** Credentials file not found or invalid

**Solution:**
1. Verify file exists: `ls -la /workspace/gcs-credentials.json`
2. Check environment variable: `echo $GOOGLE_APPLICATION_CREDENTIALS`
3. Verify file is valid JSON: `head /workspace/gcs-credentials.json`
4. Re-download credentials from GCP console

### Issue: "Permission denied" when uploading

**Cause:** Service account doesn't have access to bucket

**Solution:**
1. Go to: https://console.cloud.google.com/storage/browser
2. Click your bucket → **Permissions** tab
3. Click **"GRANT ACCESS"**
4. Add: `eigen2-storage@YOUR-PROJECT.iam.gserviceaccount.com`
5. Role: **"Storage Object Admin"**
6. Save

### Issue: "Bucket not found"

**Cause:** Bucket name typo or wrong project

**Solution:**
1. Verify bucket name: `gsutil ls` (if you have gcloud CLI)
2. Check `CLOUD_BUCKET` environment variable matches exactly
3. Ensure bucket is in the same project as service account

### Issue: "google-cloud-storage module not found"

**Cause:** Python package not installed

**Solution:**
```bash
pip install google-cloud-storage
```

### Issue: Credentials file uploaded but not found

**Cause:** File in wrong location

**Solution:**
```bash
# Check current location
find /workspace -name "gcs-credentials.json"

# Move to correct location
mv /path/to/gcs-credentials.json /workspace/gcs-credentials.json

# Verify
ls -la /workspace/gcs-credentials.json
```

---

## Security Best Practices

### ✅ Do:
- Keep `gcs-credentials.json` secure
- Use service accounts (not your personal credentials)
- Limit service account to "Storage Object Admin" role
- Use separate buckets for dev/prod

### ❌ Don't:
- Commit credentials to Git (already in .gitignore)
- Share credentials publicly
- Give service account more permissions than needed
- Use your personal GCP credentials

---

## Cost Management

### Storage Costs (Standard, us-central1)
- **Storage:** $0.020/GB/month
- **Operations:** Negligible (<$0.01/month for this use case)
- **Egress:** Free within same region, minimal otherwise

### Example Costs
| Scenario | Storage Used | Monthly Cost |
|----------|-------------|--------------|
| 1 training run | ~500 MB | $0.01 |
| 10 training runs | ~5 GB | $0.10 |
| 100 training runs | ~50 GB | $1.00 |

**Recommendation:** Delete old checkpoints you don't need:
```bash
# List all checkpoints
gsutil ls -r gs://eigen2-checkpoints-YOURNAME/

# Delete specific run
gsutil rm -r gs://eigen2-checkpoints-YOURNAME/eigen2-old-run/
```

### Set Budget Alerts

1. Go to: https://console.cloud.google.com/billing/budgets
2. Click **"CREATE BUDGET"**
3. Set budget: $5/month
4. Set alerts at: 50%, 90%, 100%
5. Get email when costs approach limit

---

## Downloading Checkpoints Locally

### Method 1: Using gsutil (Recommended)

```bash
# Download entire checkpoint directory
gsutil -m rsync -r gs://eigen2-checkpoints-YOURNAME/eigen2/checkpoints/ ./checkpoints/

# Download specific file
gsutil cp gs://eigen2-checkpoints-YOURNAME/eigen2/checkpoints/best_agent.pth ./best_agent.pth
```

### Method 2: Using Web Interface

1. Go to: https://console.cloud.google.com/storage/browser
2. Navigate to your bucket
3. Click files to download
4. Or select multiple → **Actions** → **Download**

### Method 3: Using Python Script

```python
from google.cloud import storage
import os

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs-credentials.json'

# Initialize client
client = storage.Client()
bucket = client.bucket('eigen2-checkpoints-YOURNAME')

# Download file
blob = bucket.blob('eigen2/checkpoints/best_agent.pth')
blob.download_to_filename('best_agent.pth')

print("Downloaded: best_agent.pth")
```

---

## Resume Training After Interruption

Your Vast.ai instance was terminated? No problem!

1. **Create new Vast.ai instance** (same config as before)
2. **Use same on-start script** (with `--resume` flag)
3. **Upload files:**
   ```bash
   scp gcs-credentials.json root@NEW-VAST-IP:/workspace/
   scp Eigen2_Master\(GFIN\)_03_training.csv root@NEW-VAST-IP:/workspace/
   ```
4. **Training automatically downloads checkpoints** from GCS
5. **Training resumes** from last generation

The script will show:
```
! No local checkpoint found. Attempting to download from cloud...
============================================================
Downloading checkpoints from cloud storage...
============================================================
✓ Downloaded: eigen2/checkpoints/best_agent.pth → checkpoints/best_agent.pth
...
✓ Resuming from Generation 6
```

---

## Summary Checklist

Before starting training on Vast.ai, verify:

- [ ] Google Cloud project created
- [ ] Billing enabled
- [ ] Storage bucket created: `eigen2-checkpoints-YOURNAME`
- [ ] Service account created with "Storage Object Admin" role
- [ ] Credentials downloaded: `gcs-credentials.json`
- [ ] Bucket name written down: ____________________
- [ ] Credentials file ready to upload to Vast.ai
- [ ] Environment variables configured for Vast.ai
- [ ] Tested upload/download (optional but recommended)

**You're ready to deploy!** 🚀

---

## Quick Reference

### Your GCP Configuration

Fill this out and keep for reference:

```
Project Name: ___________________________
Project ID: _____________________________
Bucket Name: ____________________________
Service Account: eigen2-storage@________.iam.gserviceaccount.com
Credentials File: gcs-credentials.json (location: ____________)
Region: _________________________________
```

### Environment Variables for Vast.ai

```bash
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=_________________________  # Your bucket name
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
GITHUB_REPO=https://github.com/___________/eigen2.git
GITHUB_BRANCH=main
```

---

## Next Steps

1. ✅ Complete this GCP setup
2. 📖 Follow [QUICK_START_VAST.md](QUICK_START_VAST.md) for Vast.ai deployment
3. 🚀 Start training with automatic checkpoint backup!

**Questions?**
- GCP Console: https://console.cloud.google.com
- GCP Documentation: https://cloud.google.com/storage/docs
- Vast.ai: https://vast.ai/faq
