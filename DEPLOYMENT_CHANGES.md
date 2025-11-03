# Deployment Changes for Vast.ai

This document summarizes all changes made to enable deployment on Vast.ai with cloud checkpoint syncing.

## Files Added

### 1. `Dockerfile`
- Base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- Installs all dependencies including cloud storage libraries
- Sets up working directory and environment variables
- Default command: `python main.py`

### 2. `.dockerignore`
- Excludes unnecessary files from Docker build context
- Prevents checkpoints, logs, and cache from being copied
- Reduces image size and build time

### 3. `utils/cloud_sync.py` ⭐ Core Feature
Cloud storage synchronization utility supporting:
- **AWS S3** via boto3
- **Google Cloud Storage** via google-cloud-storage
- **Azure Blob Storage** via azure-storage-blob
- **Local mode** (no cloud sync)

**Key Functions:**
- `upload_file()` / `download_file()` - Single file operations
- `upload_directory()` / `download_directory()` - Batch operations
- `sync_checkpoints()` - Checkpoint backup
- `sync_logs()` - TensorBoard logs backup
- `get_cloud_sync_from_env()` - Auto-configure from environment variables

### 4. `start_vast.sh`
Automated startup script for Vast.ai instances:
- Validates GPU availability (`nvidia-smi`)
- Checks for data file
- Installs cloud-specific dependencies
- Supports `--resume` flag for continuing training
- Configurable via command-line arguments

### 5. `VAST_DEPLOYMENT.md`
Comprehensive deployment guide covering:
- Cloud storage setup for all providers
- Step-by-step Vast.ai instance creation
- Environment variable configuration
- Monitoring and troubleshooting
- Cost optimization tips
- Security best practices

### 6. `QUICK_START_VAST.md`
Quick reference card:
- 5-minute setup guide
- Common commands
- Cloud provider examples
- Cost estimates
- Troubleshooting table

### 7. `.env.example`
Template for environment variables:
- Cloud provider configuration
- AWS/GCS/Azure credentials placeholders
- Optional training parameter overrides

### 8. Updated `README.md`
- Added Vast.ai deployment section
- Cloud storage support documentation
- Project structure overview
- Quick start commands

## Files Modified

### 1. `training/erl_trainer.py` ⭐ Core Integration

**Added imports:**
```python
from utils.cloud_sync import get_cloud_sync_from_env
```

**In `__init__()`:**
```python
# Cloud sync
self.cloud_sync = get_cloud_sync_from_env()
```

**In `save_checkpoint()`:**
```python
# Sync to cloud storage
self.cloud_sync.sync_checkpoints(str(checkpoint_dir))
```

**In `load_checkpoint()`:**
```python
# Try to download from cloud first
if not checkpoint_dir.exists() or len(list(checkpoint_dir.glob('*'))) == 0:
    print("! No local checkpoint found. Attempting to download from cloud...")
    self.cloud_sync.download_checkpoints(str(checkpoint_dir))
```

**In `train()` (end of training):**
```python
# Sync logs to cloud
self.cloud_sync.sync_logs(str(Config.LOG_DIR))
```

### 2. `.gitignore`
Added exclusions for:
- Environment files (`.env`)
- Credentials (`.pem`, `.key`, `*credentials*.json`)
- Cloud storage cache directories

## How It Works

### Training Flow with Cloud Sync

```
1. Instance starts
   ↓
2. start_vast.sh runs
   ↓
3. Checks for local checkpoints
   ↓
4. If none found → Downloads from cloud storage
   ↓
5. Training starts/resumes
   ↓
6. Every 5 generations:
   - Saves checkpoints locally
   - Uploads to cloud storage ⬆️
   ↓
7. Training completes
   ↓
8. Final checkpoint upload ⬆️
   ↓
9. Logs upload ⬆️
```

### Cloud Sync Behavior

| Scenario | Local Checkpoint | Cloud Storage | Action |
|----------|-----------------|---------------|--------|
| Fresh start | ❌ None | ❌ None | Start new training |
| Resume local | ✅ Exists | ❌ None | Resume from local |
| Resume cloud | ❌ None | ✅ Exists | Download → Resume |
| Resume both | ✅ Exists | ✅ Exists | Use local (faster) |

### Automatic Fallback

If cloud sync fails (missing credentials, network issues, etc.):
- **Warns** user but continues
- **Falls back** to local-only mode
- Training **never fails** due to cloud sync issues

## Environment Variables

### Required (for cloud sync)

```bash
CLOUD_PROVIDER=s3|gcs|azure|local
CLOUD_BUCKET=your-bucket-name
```

### Provider-Specific

**AWS S3:**
```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

**Google Cloud:**
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
```

**Azure:**
```bash
AZURE_STORAGE_CONNECTION_STRING=...
```

## Usage Examples

### Local Training (No Change)
```bash
python main.py --resume
```
Works exactly as before!

### Vast.ai with S3
```bash
# Set environment variables in Vast.ai template
CLOUD_PROVIDER=s3
CLOUD_BUCKET=my-checkpoints
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# On instance
bash start_vast.sh --resume
```

### Vast.ai without Cloud Sync
```bash
CLOUD_PROVIDER=local

# On instance
bash start_vast.sh --resume
```

## Migration Path

### From Local to Vast.ai

1. **Setup cloud storage** (S3/GCS/Azure)
2. **Upload existing checkpoints** (optional):
   ```python
   from utils.cloud_sync import CloudSync
   sync = CloudSync(provider='s3', bucket_name='my-bucket')
   sync.upload_directory('checkpoints', 'eigen2/checkpoints')
   ```
3. **Create Vast.ai instance** with environment variables
4. **Resume training** with `--resume` flag

### From Vast.ai Back to Local

1. **Download checkpoints**:
   ```bash
   aws s3 sync s3://my-bucket/eigen2/checkpoints/ ./checkpoints/
   ```
2. **Resume locally**:
   ```bash
   python main.py --resume
   ```

## Benefits

### 1. Fault Tolerance
- Instance terminated? Just restart and resume
- No progress lost

### 2. Cost Optimization
- Use cheaper interruptible instances
- Stop/start without losing work
- Pay only for compute time

### 3. Flexibility
- Switch between different GPU types
- Move training between providers
- Train locally or remotely with same code

### 4. Collaboration
- Share checkpoints via cloud storage
- Multiple team members can continue training
- Centralized backup

### 5. Monitoring
- Download checkpoints anytime
- Evaluate without stopping training
- Keep historical snapshots

## Security Considerations

### What's Protected
- ✅ Credentials never in code
- ✅ Environment variables only
- ✅ `.gitignore` excludes sensitive files
- ✅ Separate buckets per project recommended

### Best Practices
1. Use **IAM roles** on cloud instances (instead of keys)
2. Set **minimum permissions** on buckets
3. Enable **bucket versioning** for backup
4. Use **separate buckets** for dev/prod
5. **Rotate credentials** regularly
6. **Delete old instances** immediately after use

## Backward Compatibility

### Zero Breaking Changes
All existing code works without modification:
- `python main.py` still works locally
- No cloud setup required for local training
- Checkpoints still saved to `checkpoints/` directory
- Same CLI arguments (`--resume`)

### Optional Cloud Sync
Cloud syncing is **opt-in**:
- Default: `CLOUD_PROVIDER=local` (no cloud)
- Only syncs when explicitly configured
- Graceful fallback if credentials missing

## Testing

### Test Cloud Sync Locally
```python
from utils.cloud_sync import CloudSync

# Test S3 connection
sync = CloudSync(provider='s3', bucket_name='test-bucket')
sync.upload_file('test.txt', 'test/test.txt')
sync.download_file('test/test.txt', 'downloaded.txt')
```

### Test Vast.ai Script Locally
```bash
# Set test environment
export CLOUD_PROVIDER=local
bash start_vast.sh --resume
```

## Cost Estimates

### Cloud Storage
- **S3 Standard**: $0.023/GB/month
- **GCS Standard**: $0.020/GB/month
- **Azure Hot**: $0.018/GB/month

**Typical usage:**
- Checkpoints: ~500MB per training run
- Logs: ~100MB
- **Total: ~$0.01/month per run**

### Vast.ai GPU
- **RTX 4090**: $0.30-0.60/hour
- **Full training**: 10-20 hours
- **Total: $3-12 per run**

**Cloud storage is negligible compared to compute costs!**

## Support & Troubleshooting

### Common Issues

1. **"Module not found: boto3"**
   - Solution: `pip install boto3` (auto-installed in Docker)

2. **"Could not connect to S3"**
   - Check AWS credentials
   - Verify bucket name
   - Test: `aws s3 ls s3://bucket-name/`

3. **"Permission denied"**
   - Check IAM permissions
   - Ensure bucket exists
   - Verify credentials are correct

4. **"Checkpoints not downloading"**
   - Check `CLOUD_BUCKET` matches upload bucket
   - Verify files exist: `aws s3 ls s3://bucket/eigen2/checkpoints/`
   - Check bucket permissions

### Debug Mode
```python
# Test cloud sync
from utils.cloud_sync import get_cloud_sync_from_env
sync = get_cloud_sync_from_env()
print(f"Provider: {sync.provider}")
print(f"Bucket: {sync.bucket_name}")
```

## Future Enhancements

Potential improvements:
- [ ] Automatic checkpoint cleanup (keep last N only)
- [ ] Wandb integration for distributed monitoring
- [ ] Multi-instance distributed training
- [ ] Automatic hyperparameter tuning
- [ ] Checkpoint compression before upload
- [ ] Delta syncing (only changed files)

## Rollback

To remove cloud sync features:
1. Revert `training/erl_trainer.py` changes
2. Remove `utils/cloud_sync.py`
3. Remove Vast.ai related files

The core training code remains unchanged and fully functional.

## Conclusion

These changes enable:
- ✅ **Seamless** Vast.ai deployment
- ✅ **Automatic** checkpoint backup
- ✅ **Zero** code changes for local training
- ✅ **Flexible** cloud storage options
- ✅ **Fault-tolerant** distributed training

**Training can now run anywhere with full checkpoint persistence!**
