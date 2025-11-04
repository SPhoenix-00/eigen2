# GitHub Deployment Approach - Simplified!

## What Changed?

Instead of building and pushing custom Docker images to Docker Hub, we now use a **much simpler approach**:

1. ✅ Push code to GitHub
2. ✅ Vast.ai clones from GitHub
3. ✅ Training starts automatically

**No Docker Hub account needed!**

---

## Comparison: Docker Hub vs GitHub

| Aspect | Docker Hub Approach ❌ | GitHub Approach ✅ |
|--------|----------------------|-------------------|
| **Account Required** | Docker Hub account | Only GitHub (you already have) |
| **Build Step** | `docker build` + `docker push` | Just `git push` |
| **Update Code** | Rebuild + push image (~5-10 min) | `git push` (~10 sec) |
| **Image Size** | 5-10 GB | N/A (clones source only) |
| **Setup Time** | 15-30 minutes | 3 minutes |
| **Complexity** | Medium-High | Low |
| **Cost** | Free (but takes time) | Free |

---

## How It Works

### Traditional Docker Hub Approach (OLD):
```
Local Code
  ↓ docker build
Docker Image (5-10 GB)
  ↓ docker push
Docker Hub
  ↓ docker pull
Vast.ai Instance
  ↓ run
Training
```

### New GitHub Approach (SIMPLE):
```
Local Code
  ↓ git push
GitHub
  ↓ git clone (on-start script)
Vast.ai Instance
  ↓ run
Training
```

---

## Updated Workflow

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for training"
git push origin main
```

### Step 2: Create Vast.ai Instance

**Docker Image:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
(This is the official PyTorch image - no custom build needed!)

**On-start Script:**
```bash
#!/bin/bash
cd /workspace
export GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
export GITHUB_BRANCH=main

# Clone repository
git clone --depth 1 --branch $GITHUB_BRANCH $GITHUB_REPO .

# Wait for training data (upload via scp)
while [ ! -f "Eigen2_Master(GFIN)_03_training.csv" ]; do
    echo "Waiting for training data..."
    sleep 10
done

# Start training
bash start_vast.sh --resume
```

**Environment Variables:**
```
CLOUD_PROVIDER=s3
CLOUD_BUCKET=my-checkpoints
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
GITHUB_REPO=https://github.com/YOUR-USERNAME/eigen2.git
GITHUB_BRANCH=main
```

### Step 3: Upload Data
```bash
scp Eigen2_Master\(GFIN\)_03_training.csv root@vast-ip:/workspace/
```

### Step 4: Training Starts Automatically!

---

## Advantages

### 1. Faster Updates
```bash
# Make a code change
vim main.py

# Deploy instantly
git add .
git commit -m "Fix bug"
git push

# On Vast.ai (if already running)
git pull
bash start_vast.sh --resume
```

### 2. No Custom Images
- Use official PyTorch images
- Always up to date
- Smaller and faster

### 3. Version Control
- All code changes tracked
- Easy rollback: `git checkout v1.0`
- Branch testing: `export GITHUB_BRANCH=experimental`

### 4. Collaboration
- Team members can contribute
- Pull requests for code review
- Shared repository = shared deployment

### 5. Simpler Setup
- No Docker Hub login
- No image building
- No registry management

---

## Files Modified

### Updated Files:
1. **[start_vast.sh](start_vast.sh)** - Added GitHub cloning logic
2. **[QUICK_START.md](QUICK_START.md)** - Rewrote for GitHub approach
3. **[README.md](README.md)** - Updated deployment instructions

### Removed Files:
1. ~~`build_and_push.sh`~~ - No longer needed!

### Unchanged Files:
- **[Dockerfile](Dockerfile)** - Still available for local Docker use
- **[utils/cloud_sync.py](utils/cloud_sync.py)** - Cloud syncing unchanged
- All training code - works identically

---

## Private Repository Support

### Option 1: Personal Access Token (Easiest)

1. Create token: https://github.com/settings/tokens
2. Select scopes: `repo` (full control)
3. Use in on-start script:
```bash
git clone https://YOUR_TOKEN@github.com/YOUR-USERNAME/eigen2.git .
```

### Option 2: SSH Key

```bash
# On Vast.ai instance (via SSH)
ssh-keygen -t ed25519 -f ~/.ssh/id_vast -N ""
cat ~/.ssh/id_vast.pub  # Copy this

# Add to GitHub: https://github.com/settings/keys

# In on-start script:
git clone git@github.com:YOUR-USERNAME/eigen2.git .
```

### Option 3: Make Repository Public
Simplest option if you don't need privacy:
- GitHub Settings → Change visibility → Public

---

## Troubleshooting

### "Repository not found"
- Check repository URL is correct
- Ensure repo is public or using auth token
- Verify GITHUB_REPO environment variable

### "Permission denied (publickey)"
- Use HTTPS instead of SSH for public repos
- For private repos, use personal access token
- Or add SSH key to GitHub

### "Git not installed"
PyTorch images include git by default. If missing:
```bash
apt-get update && apt-get install -y git
```

### "Shallow clone failed"
If `--depth 1` fails (rare), use full clone:
```bash
git clone --branch $GITHUB_BRANCH $GITHUB_REPO .
```

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `GITHUB_REPO` | Repository URL | `https://github.com/user/eigen2.git` |
| `GITHUB_BRANCH` | Branch to clone | `main` or `dev` |
| `CLOUD_PROVIDER` | Cloud storage | `s3`, `gcs`, `azure`, `local` |
| `CLOUD_BUCKET` | Bucket name | `my-checkpoints` |
| `AWS_ACCESS_KEY_ID` | AWS key (if S3) | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret (if S3) | `...` |

---

## Best Practices

### 1. Use .gitignore
Already configured to exclude:
- ✅ Checkpoints (`checkpoints/`)
- ✅ Logs (`logs/`)
- ✅ Data files (`*.csv`)
- ✅ Credentials (`*.pem`, `.env`, `*credentials*.json`)

### 2. Separate Branches
```bash
# Development branch
git checkout -b dev
git push origin dev

# Production branch
git checkout main
git push origin main

# In Vast.ai, choose branch:
export GITHUB_BRANCH=dev  # or main
```

### 3. Tag Releases
```bash
git tag -a v1.0 -m "First release"
git push origin v1.0

# Use specific version:
export GITHUB_BRANCH=v1.0
```

### 4. Test Locally First
```bash
# Always test locally before deploying
python main.py

# Then deploy
git push origin main
```

---

## Migration Guide

### If You Already Have Docker Hub Setup:

**Option 1: Switch to GitHub (Recommended)**
- Simpler and faster
- Follow new QUICK_START.md

**Option 2: Keep Using Docker Hub**
- Old approach still works
- Dockerfile unchanged
- Use your custom image

**Option 3: Hybrid**
- Use GitHub for development
- Use Docker for production (cached dependencies)

---

## Performance Comparison

| Metric | Docker Hub | GitHub |
|--------|-----------|--------|
| Initial setup | 15-30 min | 3 min |
| Code update time | 5-10 min | 10 sec |
| Instance start time | ~2 min (pull image) | ~1 min (clone + pip install) |
| Storage used | 5-10 GB | ~100 MB |
| Dependency caching | Yes (in image) | No (pip install each time) |

**Note:** GitHub approach is slightly slower on first start (installs dependencies), but much faster for iterations.

---

## Advanced: Hybrid Approach

Best of both worlds - use GitHub for code, Docker for dependencies:

**Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Pre-install dependencies (cached)
RUN pip install boto3 google-cloud-storage azure-storage-blob

# Clone on startup
WORKDIR /workspace
CMD bash -c "git clone $GITHUB_REPO . && bash start_vast.sh --resume"
```

Build once, clone fresh code each time.

---

## Summary

**GitHub approach eliminates:**
- ❌ Docker Hub account requirement
- ❌ `docker build` step
- ❌ `docker push` step (slow for large images)
- ❌ Image registry management

**GitHub approach provides:**
- ✅ Instant code updates (`git push`)
- ✅ Version control integration
- ✅ Branch-based testing
- ✅ Simpler workflow
- ✅ Better collaboration

**Result: 3-minute setup instead of 30 minutes!** ⚡

---

## Questions?

- **GitHub Issues**: Use for code/deployment issues
- **Vast.ai**: https://vast.ai/faq
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Full Guide**: [VAST_DEPLOYMENT.md](VAST_DEPLOYMENT.md)
