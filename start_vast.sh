#!/bin/bash
# Vast.ai startup script for Eigen2 training

set -e  # Exit on error

echo "==================================="
echo "Eigen2 Vast.ai Training Startup"
echo "==================================="

# Environment variables (set these in Vast.ai template or override)
export CLOUD_PROVIDER="${CLOUD_PROVIDER:-local}"
export CLOUD_BUCKET="${CLOUD_BUCKET:-}"
export CLOUD_PROJECT="${CLOUD_PROJECT:-eigen2}"
export GITHUB_REPO="${GITHUB_REPO:-}"
export GITHUB_BRANCH="${GITHUB_BRANCH:-main}"

# Parse command line arguments
RESUME_TRAINING=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_TRAINING=true
            shift
            ;;
        --cloud-provider)
            export CLOUD_PROVIDER="$2"
            shift 2
            ;;
        --bucket)
            export CLOUD_BUCKET="$2"
            shift 2
            ;;
        --repo)
            export GITHUB_REPO="$2"
            shift 2
            ;;
        --branch)
            export GITHUB_BRANCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clone repository if not already present (GitHub approach)
if [ -n "$GITHUB_REPO" ] && [ ! -f "main.py" ]; then
    echo ""
    echo "Cloning repository from GitHub..."
    echo "  Repository: $GITHUB_REPO"
    echo "  Branch: $GITHUB_BRANCH"
    git clone --depth 1 --branch "$GITHUB_BRANCH" "$GITHUB_REPO" /workspace/eigen2
    cd /workspace/eigen2
    echo "✓ Repository cloned successfully"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Cloud Provider: $CLOUD_PROVIDER"
echo "  Cloud Bucket: ${CLOUD_BUCKET:-'(none - local only)'}"
echo "  Resume Training: $RESUME_TRAINING"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Verify data file exists
DATA_FILE="Eigen2_Master(GFIN)_03_training.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Please upload the data file to the working directory"
    exit 1
fi
echo "✓ Data file found: $DATA_FILE"

# Install any missing dependencies (in case they weren't in the image)
echo ""
echo "Ensuring dependencies are installed..."
pip install --no-cache-dir -r requirements.txt

# Install cloud storage dependencies based on provider
if [ "$CLOUD_PROVIDER" = "s3" ]; then
    echo "Installing AWS S3 dependencies..."
    pip install --no-cache-dir boto3
elif [ "$CLOUD_PROVIDER" = "gcs" ]; then
    echo "Installing Google Cloud Storage dependencies..."
    pip install --no-cache-dir google-cloud-storage
elif [ "$CLOUD_PROVIDER" = "azure" ]; then
    echo "Installing Azure Blob Storage dependencies..."
    pip install --no-cache-dir azure-storage-blob
fi

echo ""
echo "==================================="
echo "Starting Training"
echo "==================================="
echo ""

# Build training command
TRAIN_CMD="python main.py"
if [ "$RESUME_TRAINING" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --resume"
fi

# Run training
$TRAIN_CMD

echo ""
echo "==================================="
echo "Training Complete"
echo "==================================="
