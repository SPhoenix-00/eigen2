#!/usr/bin/env python3
"""
Pre-flight validation script for ROCm deployment.

Validates all prerequisites before starting training:
- GPU availability and ROCm functionality
- Data file existence
- Environment variables
- GCS connectivity
- W&B login status
"""

import os
import sys
from pathlib import Path

def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_success(text: str):
    """Print a success message."""
    print(f"  ✓ {text}")

def print_error(text: str):
    """Print an error message."""
    print(f"  ❌ {text}")

def print_warning(text: str):
    """Print a warning message."""
    print(f"  ⚠ {text}")

def check_gpu():
    """Check GPU availability and ROCm functionality."""
    print_header("GPU Check")
    
    try:
        import torch
    except ImportError:
        print_error("PyTorch not installed")
        return False
    
    if not torch.cuda.is_available():
        print_warning("CUDA/ROCm not available - training will run on CPU (very slow)")
        return True  # Not a blocker, just a warning
    
    device_name = torch.cuda.get_device_name(0)
    print_success(f"GPU detected: {device_name}")
    
    # Detect backend
    from utils.device import get_gpu_backend, test_rocm_functionality
    backend = get_gpu_backend()
    print_success(f"Backend: {backend}")
    
    # Test ROCm functionality
    if backend == "ROCm":
        print("  Testing ROCm functionality...")
        if test_rocm_functionality():
            print_success("ROCm functionality test passed")
        else:
            print_error("ROCm functionality test failed - GPU operations may not work correctly")
            print_warning("Training may fail or produce incorrect results")
            return False
    elif backend == "CUDA":
        print_success("CUDA backend detected")
    else:
        print_warning("Unknown GPU backend")
    
    # Check GPU memory
    try:
        mem_info = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_success(f"GPU Memory: {mem_info:.1f} GB")
        if mem_info < 16:
            print_warning("GPU has less than 16GB VRAM - may run out of memory")
    except Exception as e:
        print_warning(f"Could not get GPU memory info: {e}")
    
    return True

def check_data_file():
    """Check if training data file exists."""
    print_header("Data File Check")
    
    try:
        from utils.config import Config
        from process_eigen_data import OUTPUT_FILE_PKL
        
        expected_path = Config.DATA_PATH
        expected_filename = OUTPUT_FILE_PKL
        
        if expected_path.exists():
            file_size = expected_path.stat().st_size / 1e9  # Size in GB
            print_success(f"Data file found: {expected_path}")
            print_success(f"  Filename: {expected_filename}")
            print_success(f"  Size: {file_size:.2f} GB")
            return True
        else:
            print_error(f"Data file not found: {expected_path}")
            print_error(f"  Expected filename: {expected_filename}")
            print_error(f"  Please download the training data file")
            print_error(f"  Location: {expected_path.parent}")
            return False
    except Exception as e:
        print_error(f"Error checking data file: {e}")
        return False

def check_environment_variables():
    """Check environment variables for cloud sync."""
    print_header("Environment Variables Check")
    
    cloud_provider = os.environ.get("CLOUD_PROVIDER", "").lower()
    
    if cloud_provider == "":
        print_warning("CLOUD_PROVIDER not set - using local storage only")
        print_warning("  Checkpoints will NOT be backed up to cloud")
        return True  # Not a blocker
    
    print_success(f"CLOUD_PROVIDER: {cloud_provider}")
    
    bucket_name = os.environ.get("CLOUD_BUCKET")
    if not bucket_name:
        print_error("CLOUD_BUCKET not set")
        return False
    print_success(f"CLOUD_BUCKET: {bucket_name}")
    
    if cloud_provider == "gcs":
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCS_CREDENTIALS")
        if not creds_path:
            print_error("GOOGLE_APPLICATION_CREDENTIALS not set")
            return False
        
        creds_file = Path(creds_path)
        if not creds_file.exists():
            print_error(f"GCS credentials file not found: {creds_path}")
            return False
        
        if not creds_file.is_file():
            print_error(f"GCS credentials path is not a file: {creds_path}")
            return False
        
        if not os.access(creds_file, os.R_OK):
            print_error(f"GCS credentials file is not readable: {creds_path}")
            return False
        
        print_success(f"GCS credentials: {creds_path}")
    
    return True

def check_gcs_connectivity():
    """Test GCS connectivity."""
    print_header("GCS Connectivity Check")
    
    cloud_provider = os.environ.get("CLOUD_PROVIDER", "").lower()
    if cloud_provider != "gcs":
        print_warning("CLOUD_PROVIDER is not 'gcs' - skipping GCS connectivity test")
        return True
    
    try:
        from utils.cloud_sync import get_cloud_sync_from_env
        
        cloud_sync = get_cloud_sync_from_env()
        print_success(f"Cloud sync initialized: {cloud_sync.provider}")
        print_success(f"Bucket: {cloud_sync.bucket_name}")
        
        # Try to list bucket (simple connectivity test)
        if cloud_sync.provider == "gcs":
            try:
                # Try to access bucket (this will fail if credentials are invalid)
                blobs = list(cloud_sync.bucket.list_blobs(max_results=1))
                print_success("GCS connectivity test passed")
                return True
            except Exception as e:
                print_error(f"GCS connectivity test failed: {e}")
                print_error("  Check your credentials file and bucket permissions")
                return False
        
        return True
    except ValueError as e:
        print_error(f"Cloud sync configuration error: {e}")
        return False
    except Exception as e:
        print_error(f"Error testing GCS connectivity: {e}")
        return False

def check_wandb():
    """Check W&B login status."""
    print_header("Weights & Biases Check")
    
    try:
        import wandb
    except ImportError:
        print_error("wandb not installed")
        print_error("  Install with: pip install wandb")
        return False
    
    # Check if wandb is logged in
    try:
        api = wandb.Api()
        # Try to access user info (will fail if not logged in)
        user = api.viewer()
        print_success(f"W&B logged in as: {user.get('username', 'unknown')}")
        return True
    except Exception as e:
        print_warning(f"W&B login check failed: {e}")
        print_warning("  Run 'wandb login' to authenticate")
        print_warning("  Training will still work but metrics won't be logged")
        return True  # Not a blocker, just a warning

def check_directories():
    """Check that required directories exist or can be created."""
    print_header("Directory Check")
    
    try:
        from utils.config import Config
        
        # Check checkpoint directory
        Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        print_success(f"Checkpoint directory: {Config.CHECKPOINT_DIR}")
        
        # Check log directory
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        print_success(f"Log directory: {Config.LOG_DIR}")
        
        # Check evaluation results directory
        eval_dir = Path("evaluation_results")
        eval_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Evaluation results directory: {eval_dir}")
        
        return True
    except Exception as e:
        print_error(f"Error checking directories: {e}")
        return False

def main():
    """Run all validation checks."""
    print("="*70)
    print("  ROCm Deployment Pre-Flight Validation")
    print("="*70)
    
    checks = [
        ("GPU", check_gpu),
        ("Data File", check_data_file),
        ("Environment Variables", check_environment_variables),
        ("GCS Connectivity", check_gcs_connectivity),
        ("W&B Login", check_wandb),
        ("Directories", check_directories),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Unexpected error in {name} check: {e}")
            results[name] = False
    
    # Summary
    print_header("Validation Summary")
    
    all_passed = True
    for name, passed in results.items():
        if passed:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✓ All checks passed! Ready for deployment.")
        print("="*70)
        return 0
    else:
        print("  ❌ Some checks failed. Please fix the issues above.")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())

