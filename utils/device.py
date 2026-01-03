"""
Device abstraction layer for ROCm/CUDA/CPU compatibility.

This module provides a unified interface for GPU operations that works
transparently across CUDA (NVIDIA) and ROCm (AMD) backends. PyTorch's
torch.cuda.* API works on both backends, so this abstraction centralizes
all device-specific logic for easier maintenance.

Usage:
    from utils.device import get_device, get_device_type, setup_gpu_environment

    # Get device for model placement
    device = get_device()  # Returns torch.device("cuda") or torch.device("cpu")

    # Get device type string for autocast/GradScaler
    device_type = get_device_type()  # Returns "cuda" or "cpu"

    # Setup environment variables for optimal GPU performance
    setup_gpu_environment()
"""
import os
import torch


def get_device(verbose: bool = False) -> torch.device:
    """
    Determine the best available device for computation.

    PyTorch's CUDA API works for both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    When ROCm is installed, torch.cuda.is_available() returns True and
    all torch.cuda.* functions work seamlessly.

    Args:
        verbose: If True, print device detection info

    Returns:
        torch.device for the best available backend
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            device_name = torch.cuda.get_device_name(0)
            backend = get_gpu_backend()
            print(f"Device: GPU ({device_name})")
            print(f"Backend: {backend}")
        return device
    else:
        if verbose:
            print("Device: CPU (GPU not detected)")
        return torch.device("cpu")


def get_device_type() -> str:
    """
    Get device type string for autocast and GradScaler.

    PyTorch uses 'cuda' for both CUDA and ROCm backends.
    This is the string passed to torch.amp.autocast(device_type=...)
    and torch.amp.GradScaler(...).

    Returns:
        "cuda" if GPU available, "cpu" otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_gpu_backend() -> str:
    """
    Detect which GPU backend is being used.

    Returns:
        "CUDA" for NVIDIA GPUs
        "ROCm" for AMD GPUs
        "CPU" if no GPU available
    """
    if not torch.cuda.is_available():
        return "CPU"

    device_name = torch.cuda.get_device_name(0).upper()

    # Detect by vendor name in GPU string
    if "AMD" in device_name or "RADEON" in device_name or "INSTINCT" in device_name:
        return "ROCm"
    elif "NVIDIA" in device_name or "GEFORCE" in device_name or "TESLA" in device_name:
        return "CUDA"
    else:
        # Fallback: check for HIP runtime (ROCm indicator)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "ROCm"
        return "CUDA"


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or ROCm) is available."""
    return torch.cuda.is_available()


def get_gpu_name() -> str:
    """Get GPU device name for logging."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information.

    Returns:
        Dictionary with 'total', 'allocated', 'reserved', 'free' in GB
        Returns zeros if no GPU available
    """
    if not torch.cuda.is_available():
        return {'total': 0, 'allocated': 0, 'reserved': 0, 'free': 0}

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    free = total - reserved

    return {
        'total': total,
        'allocated': allocated,
        'reserved': reserved,
        'free': free
    }


def setup_gpu_environment(verbose: bool = False) -> str:
    """
    Configure GPU environment and detect backend.

    NOTE: Environment variables (PYTORCH_CUDA_ALLOC_CONF, PYTORCH_HIP_ALLOC_CONF)
    should be set BEFORE importing torch. This function is called AFTER torch
    import to detect the backend and configure torch-specific settings.

    Args:
        verbose: If True, print configuration info (unless SUPPRESS_GPU_OUTPUT env var is set)

    Returns:
        Backend name ("CUDA", "ROCm", or "CPU")
    """
    # Check if output should be suppressed (e.g., in worker processes)
    suppress_output = os.environ.get('SUPPRESS_GPU_OUTPUT', '0') == '1'
    
    backend = get_gpu_backend()

    if backend == "ROCm":
        # Ensure environment variables are set (should already be set before torch import)
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
        # Disable TF32 on AMD (not supported)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if verbose and not suppress_output:
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            print(f"Configured environment for ROCm (AMD GPU)")
            print(f"  Device: {device_name}")

    elif backend == "CUDA":
        # Ensure environment variable is set (should already be set before torch import)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        if verbose and not suppress_output:
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            print(f"Configured environment for CUDA (NVIDIA GPU)")
            print(f"  Device: {device_name}")
    else:
        if verbose and not suppress_output:
            print("No GPU detected, running on CPU")

    return backend


def test_rocm_functionality() -> bool:
    """
    Test if ROCm is actually functional by performing a simple matrix operation.
    
    Similar to CUDA cuBLAS test, this verifies that GPU operations work correctly.
    Some systems may detect GPU but have broken drivers/runtime.
    
    Returns:
        True if ROCm functionality test passes, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Create test tensors on GPU
        device = torch.device("cuda:0")
        a = torch.randn(100, 100, device=device, dtype=torch.float32)
        b = torch.randn(100, 100, device=device, dtype=torch.float32)
        
        # Perform matrix multiplication (tests ROCm/HIP functionality)
        c = torch.matmul(a, b)
        
        # Verify result is valid (not NaN or Inf)
        if torch.isnan(c).any() or torch.isinf(c).any():
            return False
        
        # Clean up
        del a, b, c
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"ROCm functionality test failed: {e}")
        return False


def empty_cache() -> None:
    """Clear GPU memory cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_amp_dtype(dtype_str: str = "float16") -> torch.dtype:
    """
    Convert dtype string to torch.dtype for autocast.

    Args:
        dtype_str: "bfloat16" or "float16"
                   bfloat16 recommended for MI300X, float16 for older GPUs

    Returns:
        torch.bfloat16 or torch.float16
    """
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    else:
        raise ValueError(f"Unknown AMP dtype: {dtype_str}. Use 'bfloat16' or 'float16'")


# Self-test when run directly
if __name__ == "__main__":
    print("=" * 50)
    print("Device Detection Test")
    print("=" * 50)

    device = get_device(verbose=True)
    device_type = get_device_type()
    backend = get_gpu_backend()

    print(f"\nDevice: {device}")
    print(f"Device type (for autocast): {device_type}")
    print(f"Backend: {backend}")

    if is_gpu_available():
        mem_info = get_gpu_memory_info()
        print(f"\nGPU Memory:")
        print(f"  Total: {mem_info['total']:.2f} GB")
        print(f"  Allocated: {mem_info['allocated']:.2f} GB")
        print(f"  Reserved: {mem_info['reserved']:.2f} GB")
        print(f"  Free: {mem_info['free']:.2f} GB")

        # Test environment setup
        print(f"\nEnvironment setup:")
        setup_gpu_environment(verbose=True)
        
        # Test ROCm functionality if ROCm backend
        if backend == "ROCm":
            print(f"\nROCm Functionality Test:")
            if test_rocm_functionality():
                print("  ✓ ROCm functionality test passed")
            else:
                print("  ❌ ROCm functionality test failed")

    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)
