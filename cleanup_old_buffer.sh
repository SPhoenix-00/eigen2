#!/bin/bash
# Script to delete old replay buffer from GCS
# Run this on RunPod to clean up the stale buffer

echo "========================================"
echo "Cleaning up old replay buffer from GCS"
echo "========================================"

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil not found. Installing..."
    pip install gsutil
fi

# Set GCS credentials if needed
if [ -f "/workspace/gcs-credentials.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json
fi

BUCKET="gs://eigen2-checkpoints-ase0"
BUFFER_PATH="${BUCKET}/eigen2/checkpoints/replay_buffer.pkl"

echo ""
echo "Checking for buffer file at: ${BUFFER_PATH}"

# Check if buffer exists
if gsutil ls "${BUFFER_PATH}" 2>/dev/null; then
    echo "Found old buffer file!"

    # Show file size
    echo ""
    echo "File details:"
    gsutil ls -lh "${BUFFER_PATH}"

    # Ask for confirmation (skip in non-interactive mode)
    if [ -t 0 ]; then
        read -p "Delete this file? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted. Buffer not deleted."
            exit 0
        fi
    fi

    # Delete the file
    echo ""
    echo "Deleting buffer..."
    gsutil rm "${BUFFER_PATH}"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully deleted old replay buffer from GCS"
    else
        echo "✗ Failed to delete buffer"
        exit 1
    fi
else
    echo "No buffer file found at ${BUFFER_PATH}"
    echo "(This is expected if it was already deleted or never saved)"
fi

# Also check for compressed version
BUFFER_GZ_PATH="${BUFFER_PATH}.gz"
if gsutil ls "${BUFFER_GZ_PATH}" 2>/dev/null; then
    echo ""
    echo "Found compressed buffer: ${BUFFER_GZ_PATH}"
    echo "Deleting..."
    gsutil rm "${BUFFER_GZ_PATH}"
    echo "✓ Deleted compressed buffer"
fi

echo ""
echo "========================================"
echo "Cleanup complete!"
echo "========================================"
