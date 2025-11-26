"""
GPU Device Detection and Configuration
Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs transparently
"""
import torch
import os


def get_device_info():
    """
    Detect GPU backend and return device configuration.

    Returns:
        tuple: (device, backend_name, device_name)
        - device: torch.device object
        - backend_name: "CUDA", "ROCm", or "CPU"
        - device_name: GPU model name or "CPU"
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), "CPU", "CPU"

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    # Detect backend by GPU vendor
    if any(keyword in device_name.upper() for keyword in ["AMD", "RADEON", "INSTINCT"]):
        backend = "ROCm"
    elif any(keyword in device_name.upper() for keyword in ["NVIDIA", "GEFORCE", "TESLA", "RTX", "GTX"]):
        backend = "CUDA"
    else:
        # Fallback: check if HIP is available (ROCm indicator)
        backend = "ROCm" if hasattr(torch.version, 'hip') and torch.version.hip is not None else "CUDA"

    return device, backend, device_name


def setup_gpu_environment(backend: str):
    """
    Configure GPU-specific environment variables.

    Args:
        backend: "CUDA", "ROCm", or "CPU"

    Returns:
        str: The backend name
    """
    if backend == "ROCm":
        # ROCm-specific optimizations
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'  # Memory optimization

        # ROCm doesn't have TF32, so disable for consistency
        torch.backends.cuda.matmul.allow_tf32 = False

    elif backend == "CUDA":
        # CUDA-specific optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Optional: Enable TF32 for performance on NVIDIA Ampere+ GPUs
        # Uncomment the line below for ~20% speedup on RTX 30xx/40xx cards
        # torch.backends.cuda.matmul.allow_tf32 = True

    return backend


def get_gpu_memory_info():
    """
    Get GPU memory statistics (works for both CUDA and ROCm).

    Returns:
        dict: GPU memory information or None if no GPU available
    """
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'device_count': torch.cuda.device_count(),
        }
    return None


def get_backend_version():
    """
    Get GPU backend version information.

    Returns:
        str: Version string for the detected backend
    """
    if not torch.cuda.is_available():
        return "CPU only"

    # Check for ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return f"ROCm {torch.version.hip}"

    # Check for CUDA
    if hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
        return f"CUDA {torch.version.cuda}"

    return "Unknown GPU backend"


# Test the module
if __name__ == "__main__":
    print("=" * 60)
    print("GPU Device Detection Test")
    print("=" * 60)

    device, backend, name = get_device_info()
    print(f"\nDevice: {device}")
    print(f"Backend: {backend}")
    print(f"GPU Name: {name}")
    print(f"Backend Version: {get_backend_version()}")

    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"\nGPU Memory Info:")
        print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
        print(f"  Reserved: {mem_info['reserved_gb']:.2f} GB")
        print(f"  Device Count: {mem_info['device_count']}")

    setup_gpu_environment(backend)
    print(f"\n[OK] Environment configured for {backend}")
    print("=" * 60)
