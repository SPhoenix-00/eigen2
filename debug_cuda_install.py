"""
Debug script to diagnose CUDA installation issues
"""
import json
import os
import sys
import subprocess
from pathlib import Path

# Log path from debug mode configuration
LOG_PATH = Path(r"d:\GitHub\eigen2\.cursor\debug.log")

def log_entry(entry):
    """Write NDJSON log entry"""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

# #region agent log
log_entry({
    "sessionId": "debug-session",
    "runId": "initial",
    "hypothesisId": "A",
    "location": "debug_cuda_install.py:18",
    "message": "Starting CUDA diagnostic",
    "data": {"python_version": sys.version, "platform": sys.platform},
    "timestamp": int(__import__('time').time() * 1000)
})
# #endregion

# Hypothesis A: PyTorch torch package is CPU-only and wasn't upgraded
try:
    import torch
    torch_version = torch.__version__
    torch_cuda_available = torch.cuda.is_available()
    torch_cuda_version = getattr(torch.version, 'cuda', None)
    torch_cudnn_version = getattr(torch.backends.cudnn, 'version', lambda: None)() if torch.cuda.is_available() else None
    
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "A",
        "location": "debug_cuda_install.py:35",
        "message": "PyTorch torch package info",
        "data": {
            "torch_version": torch_version,
            "is_cpu_only": "+cpu" in torch_version,
            "cuda_available": torch_cuda_available,
            "cuda_version": torch_cuda_version,
            "cudnn_version": torch_cudnn_version,
            "has_cuda_attr": hasattr(torch.version, 'cuda')
        },
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    
    print(f"PyTorch version: {torch_version}")
    print(f"CUDA available: {torch_cuda_available}")
    print(f"CUDA version in torch: {torch_cuda_version}")
    print(f"cuDNN version: {torch_cudnn_version}")
    print(f"Is CPU-only build: {'+cpu' in torch_version}")
except Exception as e:
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "A",
        "location": "debug_cuda_install.py:55",
        "message": "Failed to import torch",
        "data": {"error": str(e)},
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print(f"Error importing torch: {e}")
    torch = None

# Hypothesis B: Version mismatch between torch and torchvision
try:
    import torchvision
    tv_version = torchvision.__version__
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "B",
        "location": "debug_cuda_install.py:68",
        "message": "torchvision version info",
        "data": {"torchvision_version": tv_version},
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print(f"torchvision version: {tv_version}")
except Exception as e:
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "B",
        "location": "debug_cuda_install.py:77",
        "message": "Failed to import torchvision",
        "data": {"error": str(e)},
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print(f"Error importing torchvision: {e}")

# Hypothesis C: Check pip installed packages
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "torch"],
        capture_output=True,
        text=True,
        timeout=10
    )
    pip_torch_info = result.stdout
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "C",
        "location": "debug_cuda_install.py:92",
        "message": "pip show torch output",
        "data": {"output": pip_torch_info, "returncode": result.returncode},
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print("\n=== pip show torch ===")
    print(pip_torch_info)
except Exception as e:
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "C",
        "location": "debug_cuda_install.py:102",
        "message": "Failed to run pip show torch",
        "data": {"error": str(e)},
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print(f"Error checking pip torch info: {e}")

# Hypothesis D: CUDA drivers/runtime check
try:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=10
    )
    nvidia_smi_output = result.stdout if result.returncode == 0 else result.stderr
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "D",
        "location": "debug_cuda_install.py:118",
        "message": "nvidia-smi output",
        "data": {"output": nvidia_smi_output, "returncode": result.returncode},
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print("\n=== nvidia-smi ===")
    print(nvidia_smi_output)
except Exception as e:
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "D",
        "location": "debug_cuda_install.py:128",
        "message": "nvidia-smi not available",
        "data": {"error": str(e)},
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print(f"nvidia-smi not available: {e}")

# Hypothesis E: Environment variables
env_vars = {
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    "CUDA_HOME": os.environ.get("CUDA_HOME"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    "PATH": os.environ.get("PATH", "")[:200]  # Truncate PATH
}
# #region agent log
log_entry({
    "sessionId": "debug-session",
    "runId": "initial",
    "hypothesisId": "E",
    "location": "debug_cuda_install.py:145",
    "message": "Environment variables",
    "data": env_vars,
    "timestamp": int(__import__('time').time() * 1000)
})
# #endregion
print("\n=== Environment Variables ===")
for key, value in env_vars.items():
    print(f"{key}: {value}")

# Final summary
if torch:
    is_cpu_only = "+cpu" in torch_version or (not hasattr(torch.version, 'cuda') or torch.version.cuda is None)
    # #region agent log
    log_entry({
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "SUMMARY",
        "location": "debug_cuda_install.py:158",
        "message": "Diagnostic summary",
        "data": {
            "torch_is_cpu_only": is_cpu_only,
            "torch_version": torch_version,
            "cuda_available": torch_cuda_available,
            "recommendation": "uninstall_and_reinstall" if is_cpu_only else "check_drivers"
        },
        "timestamp": int(__import__('time').time() * 1000)
    })
    # #endregion
    print("\n=== SUMMARY ===")
    if is_cpu_only:
        print("ISSUE FOUND: PyTorch is CPU-only build")
        print("SOLUTION: Uninstall and reinstall torch with CUDA support")
    else:
        print("PyTorch has CUDA support but CUDA is not available")
        print("Check CUDA drivers and runtime installation")



