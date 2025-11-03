# Vast.ai Deployment Guide for Eigen2

This guide covers deploying the Eigen2 training pipeline on Vast.ai GPU instances with automatic checkpoint syncing to cloud storage.

## Overview

The deployment setup includes:
- Docker containerization for consistent environments
- Automatic checkpoint backup to cloud storage (S3/GCS/Azure)
- Resume training from cloud checkpoints
- Automated startup scripts

## Prerequisites

1. **Vast.ai Account**: Sign up at [https://vast.ai](https://vast.ai)
2. **Cloud Storage** (optional but recommended):
   - AWS S3 bucket, OR
   - Google Cloud Storage bucket, OR
   - Azure Blob Storage container
3. **Training Data**: `Eigen2_Master(GFIN)_03_training.csv`

## Cloud Storage Setup

### Option 1: AWS S3

1. Create an S3 bucket (e.g., `my-eigen2-checkpoints`)
2. Create IAM credentials with S3 access
3. Note your:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - Bucket name

### Option 2: Google Cloud Storage

1. Create a GCS bucket (e.g., `my-eigen2-checkpoints`)
2. Create a service account with Storage Admin role
3. Download the JSON credentials file
4. Note your bucket name

### Option 3: Azure Blob Storage

1. Create a storage account and container
2. Get your connection string from Azure Portal
3. Note your container name

### Option 4: No Cloud Storage (Local Only)

Set `CLOUD_PROVIDER=local` to skip cloud syncing. Checkpoints will only be saved locally (will be lost when instance terminates).

## Deployment Steps

### Step 1: Prepare Your Docker Image

You have two options:

#### Option A: Use Docker Hub (Recommended)

1. Build and push the Docker image:
```bash
# On your local machine
docker build -t yourusername/eigen2:latest .
docker push yourusername/eigen2:latest
```

2. On Vast.ai, use: `yourusername/eigen2:latest`

#### Option B: Let Vast.ai Build It

Upload your code and use the Dockerfile directly on Vast.ai.

### Step 2: Find a GPU Instance

1. Go to [Vast.ai Client](https://vast.ai/console/create/)
2. Search for instances with:
   - GPU: RTX 4090 or similar (at least 24GB VRAM recommended)
   - Disk Space: At least 50GB
   - CUDA: 12.1 or higher

### Step 3: Create Instance

#### Template Configuration

**Docker Image:**
```
yourusername/eigen2:latest
```
OR
```
pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```
(if you'll upload code directly)

**On-start Script:**
```bash
cd /workspace
bash start_vast.sh --resume
```

**Environment Variables:**

For AWS S3:
```bash
CLOUD_PROVIDER=s3
CLOUD_BUCKET=my-eigen2-checkpoints
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

For Google Cloud Storage:
```bash
CLOUD_PROVIDER=gcs
CLOUD_BUCKET=my-eigen2-checkpoints
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
```

For Azure Blob Storage:
```bash
CLOUD_PROVIDER=azure
CLOUD_BUCKET=my-container-name
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

For Local Only:
```bash
CLOUD_PROVIDER=local
```

**Disk Space:**
- Minimum: 30GB
- Recommended: 50GB+

### Step 4: Upload Files

Once the instance starts, SSH into it and upload:

1. **Training data:**
```bash
# On your local machine
scp Eigen2_Master\(GFIN\)_03_training.csv root@your-vast-ip:/workspace/
```

2. **GCS credentials (if using GCS):**
```bash
scp gcs-credentials.json root@your-vast-ip:/workspace/
```

3. **Code (if not using Docker image):**
```bash
scp -r ./eigen2/* root@your-vast-ip:/workspace/
```

### Step 5: Start Training

#### First Time (New Training)

```bash
cd /workspace
bash start_vast.sh
```

#### Resume from Cloud Checkpoint

```bash
cd /workspace
bash start_vast.sh --resume
```

This will:
1. Download checkpoints from cloud storage
2. Resume training from the last saved generation

## Monitoring Training

### Option 1: SSH and View Logs

```bash
ssh root@your-vast-ip
cd /workspace
tail -f training.log
```

### Option 2: TensorBoard (Advanced)

1. On Vast.ai instance:
```bash
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

2. Forward port via SSH:
```bash
ssh -L 6006:localhost:6006 root@your-vast-ip
```

3. Open browser: `http://localhost:6006`

### Option 3: Check Cloud Storage

Checkpoints are automatically synced every 5 generations. You can download and inspect them:

```bash
# AWS S3
aws s3 ls s3://my-eigen2-checkpoints/eigen2/checkpoints/

# GCS
gsutil ls gs://my-eigen2-checkpoints/eigen2/checkpoints/

# Azure
az storage blob list --container-name my-container --prefix eigen2/checkpoints/
```

## Checkpoint Management

### Automatic Syncing

Checkpoints are automatically synced to cloud storage:
- Every 5 generations during training
- At the end of training
- Includes: best agent, population, trainer state

### Manual Checkpoint Download

To download checkpoints from cloud to local machine:

**AWS S3:**
```bash
aws s3 sync s3://my-eigen2-checkpoints/eigen2/checkpoints/ ./checkpoints/
```

**Google Cloud:**
```bash
gsutil -m rsync -r gs://my-eigen2-checkpoints/eigen2/checkpoints/ ./checkpoints/
```

**Azure:**
```bash
az storage blob download-batch --source my-container --destination ./checkpoints/ --pattern "eigen2/checkpoints/*"
```

### Resume Training on Different Instance

The beauty of cloud sync is you can:

1. Stop one Vast.ai instance
2. Start a new (potentially cheaper) instance
3. Checkpoints automatically download from cloud
4. Training continues from where it left off

Just use `--resume` flag and same cloud storage settings.

## Cost Optimization Tips

### 1. Use Interruptible Instances

Vast.ai has cheaper "interruptible" instances. With cloud sync, you can safely use these:
- Training saves checkpoints every 5 generations
- If interrupted, spin up new instance
- Training resumes automatically

### 2. Monitor and Stop

Training runs for 20 generations by default. Stop the instance when done:
```bash
# Check progress
cat checkpoints/trainer_state.json
```

### 3. Download Final Results

After training completes:
1. Download checkpoints from cloud
2. Terminate Vast.ai instance immediately
3. Avoid paying for idle time

## Troubleshooting

### Issue: "No checkpoint found"

**Solution:**
- Check cloud storage credentials
- Verify bucket name is correct
- Ensure `CLOUD_PROVIDER` matches your setup

### Issue: "CUDA out of memory"

**Solution:**
- Use instance with more GPU memory (24GB+ recommended)
- Reduce `BATCH_SIZE` in `utils/config.py`

### Issue: "Data file not found"

**Solution:**
- Upload `Eigen2_Master(GFIN)_03_training.csv` to `/workspace/`
- Check file name matches exactly (including parentheses)

### Issue: Cloud sync not working

**Solution:**
- Check environment variables are set correctly
- Verify credentials have correct permissions
- Test connection manually:
  ```bash
  # AWS
  aws s3 ls s3://my-bucket/

  # GCS
  gsutil ls gs://my-bucket/

  # Azure
  az storage container list
  ```

### Issue: Training stopped unexpectedly

**Solution:**
1. Check logs: `cat training.log` or view Vast.ai console logs
2. If instance was terminated, just restart with `--resume`
3. Checkpoints are safe in cloud storage

## Example: Complete Workflow

### First Time Setup

```bash
# 1. Build and push Docker image
docker build -t yourusername/eigen2:latest .
docker push yourusername/eigen2:latest

# 2. Create Vast.ai instance with environment variables:
#    CLOUD_PROVIDER=s3
#    CLOUD_BUCKET=my-checkpoints
#    AWS_ACCESS_KEY_ID=...
#    AWS_SECRET_ACCESS_KEY=...

# 3. SSH into instance
ssh root@your-vast-ip

# 4. Upload data
# (from local machine)
scp Eigen2_Master\(GFIN\)_03_training.csv root@your-vast-ip:/workspace/

# 5. Start training
cd /workspace
bash start_vast.sh
```

### Continue Training Later

```bash
# 1. Create new Vast.ai instance (same environment variables)

# 2. SSH into instance
ssh root@your-vast-ip

# 3. Upload data (if needed)
scp Eigen2_Master\(GFIN\)_03_training.csv root@your-vast-ip:/workspace/

# 4. Resume training (checkpoints auto-download from cloud)
cd /workspace
bash start_vast.sh --resume
```

### Download Final Results

```bash
# From local machine
aws s3 sync s3://my-checkpoints/eigen2/checkpoints/ ./checkpoints/
aws s3 sync s3://my-checkpoints/eigen2/logs/ ./logs/

# Or use the Python utilities
python -c "
from utils.cloud_sync import CloudSync
sync = CloudSync(provider='s3', bucket_name='my-checkpoints')
sync.download_checkpoints()
sync.download_directory('eigen2/logs', 'logs')
"
```

## File Structure After Deployment

```
/workspace/
├── main.py
├── requirements.txt
├── start_vast.sh
├── Eigen2_Master(GFIN)_03_training.csv
├── checkpoints/                    # Auto-synced to cloud
│   ├── best_agent.pth
│   ├── trainer_state.json
│   └── population/
│       ├── agent_0.pth
│       ├── agent_1.pth
│       └── ...
├── logs/                           # Auto-synced to cloud
│   └── events.out.tfevents.*
├── data/
├── models/
├── environment/
├── training/
├── erl/
└── utils/
```

## Environment Variable Reference

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `CLOUD_PROVIDER` | Cloud storage provider | No (default: local) | `s3`, `gcs`, `azure`, `local` |
| `CLOUD_BUCKET` | Bucket/container name | Yes (if not local) | `my-eigen2-checkpoints` |
| `CLOUD_PROJECT` | Project identifier | No (default: eigen2) | `eigen2` |
| `AWS_ACCESS_KEY_ID` | AWS access key | Yes (for S3) | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Yes (for S3) | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCS credentials path | Yes (for GCS) | `/workspace/gcs-creds.json` |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure connection string | Yes (for Azure) | `DefaultEndpointsProtocol=https;...` |

## Security Best Practices

1. **Never commit credentials** to Git
2. **Use environment variables** for all secrets
3. **Restrict cloud storage permissions** to minimum required
4. **Delete Vast.ai instances** when not in use
5. **Use separate buckets** for different projects
6. **Enable versioning** on cloud buckets (for checkpoint history)

## Support

For issues with:
- **Eigen2 code**: Check GitHub repository
- **Vast.ai platform**: [Vast.ai Discord](https://discord.gg/vast) or [Support](https://vast.ai/faq)
- **Cloud storage**: Refer to provider documentation (AWS/GCS/Azure)

## Next Steps

After successful deployment:
1. Monitor training progress via TensorBoard or logs
2. Download checkpoints periodically for backup
3. Evaluate best agent on test data after training completes
4. Consider hyperparameter tuning for improved performance
