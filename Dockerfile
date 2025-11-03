FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for cloud storage
RUN pip install --no-cache-dir boto3 google-cloud-storage azure-storage-blob

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints logs data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command - can be overridden
CMD ["python", "main.py"]
