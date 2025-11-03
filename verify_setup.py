"""Verify project setup"""
import sys
from pathlib import Path

def verify_setup():
    print("Verifying Project Eigen 2 Setup...\n")
    
    issues = []
    
    # Check Python version
    py_version = sys.version.split()[0]
    print(f"✓ Python version: {py_version}")
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check imports
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  └─ CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  └─ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  └─ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            issues.append("CUDA not available - training will be slow")
    except ImportError as e:
        issues.append(f"PyTorch not installed: {e}")
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        issues.append("NumPy not installed")
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError:
        issues.append("Pandas not installed")
    
    try:
        import gymnasium as gym
        print(f"✓ Gymnasium: {gym.__version__}")
    except ImportError:
        issues.append("Gymnasium not installed")
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print(f"✓ TensorBoard available")
    except ImportError:
        issues.append("TensorBoard not installed")
    
    # Check directory structure
    print("\nDirectory structure:")
    required_dirs = ['data', 'environment', 'models', 'erl', 'training', 'utils', 'checkpoints', 'logs']
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/")
        else:
            issues.append(f"{dir_name}/ missing")
    
    # Check config
    print("\nConfiguration:")
    try:
        from utils.config import Config
        print(f"✓ Configuration loaded")
        print(f"  └─ Device: {Config.DEVICE}")
        print(f"  └─ Data path: {Config.DATA_PATH}")
        
        if not Path(Config.DATA_PATH).exists() and Config.DATA_PATH == "path/to/your/data.csv":
            issues.append("DATA_PATH not updated in utils/config.py")
    except ImportError as e:
        issues.append(f"Configuration not loaded: {e}")
    
    # Summary
    print("\n" + "="*60)
    if issues:
        print("⚠ Setup Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All checks passed! Ready to start development.")
    print("="*60)
    
    return len(issues) == 0

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)