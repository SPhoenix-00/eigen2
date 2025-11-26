"""
GPU Compatibility Tests for Project Eigen 2
Tests unified CUDA/ROCm support and device detection
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.device_utils import (
    get_device_info,
    setup_gpu_environment,
    get_gpu_memory_info,
    get_backend_version
)
from utils.config import Config


def test_device_detection():
    """Test GPU backend detection"""
    print("\n" + "="*60)
    print("TEST 1: Device Detection")
    print("="*60)

    device, backend, name = get_device_info()

    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print(f"GPU Name: {name}")
    print(f"Backend Version: {get_backend_version()}")

    # Validate backend is one of the expected values
    assert backend in ["CUDA", "ROCm", "CPU"], f"Invalid backend: {backend}"

    # If GPU is available, ensure device is cuda
    if backend in ["CUDA", "ROCm"]:
        assert device.type == "cuda", f"Expected cuda device, got {device.type}"

    print("✓ Device detection working correctly")
    return backend


def test_gpu_memory_operations():
    """Test basic GPU memory operations"""
    print("\n" + "="*60)
    print("TEST 2: GPU Memory Operations")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping memory tests")
        return

    mem_info = get_gpu_memory_info()
    assert mem_info is not None, "Failed to get GPU memory info"

    print(f"GPU Memory Info:")
    print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
    print(f"  Reserved: {mem_info['reserved_gb']:.2f} GB")
    print(f"  Device Count: {mem_info['device_count']}")

    # Test GPU memory allocation and operations
    print("\nTesting memory allocation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)

    assert y.device.type == 'cuda', "Tensor not on GPU"
    assert y.shape == (1000, 1000), "Unexpected tensor shape"

    # Clean up
    del x, y
    torch.cuda.empty_cache()

    print("✓ GPU memory operations working correctly")


def test_mixed_precision_training():
    """Test mixed precision training (AMP)"""
    print("\n" + "="*60)
    print("TEST 3: Mixed Precision Training")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping mixed precision tests")
        return

    from torch.amp import autocast, GradScaler

    device = torch.device('cuda')
    scaler = GradScaler('cuda')

    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test training step with mixed precision
    x = torch.randn(32, 100).to(device)
    target = torch.randint(0, 10, (32,)).to(device)

    with autocast(device_type='cuda'):
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Loss: {loss.item():.4f}")
    print("✓ Mixed precision training working correctly")

    # Clean up
    del model, optimizer, x, target
    torch.cuda.empty_cache()


def test_config_integration():
    """Test Config class integration with device detection"""
    print("\n" + "="*60)
    print("TEST 4: Config Integration")
    print("="*60)

    print(f"Config.DEVICE: {Config.DEVICE}")
    print(f"Config.GPU_BACKEND: {Config.GPU_BACKEND}")
    print(f"Config.GPU_NAME: {Config.GPU_NAME}")

    # Validate config values
    assert hasattr(Config, 'DEVICE'), "Config missing DEVICE attribute"
    assert hasattr(Config, 'GPU_BACKEND'), "Config missing GPU_BACKEND attribute"
    assert hasattr(Config, 'GPU_NAME'), "Config missing GPU_NAME attribute"

    assert Config.GPU_BACKEND in ["CUDA", "ROCm", "CPU"], f"Invalid backend in Config: {Config.GPU_BACKEND}"

    print("✓ Config integration working correctly")


def test_gradient_flow():
    """Test gradient computation on GPU"""
    print("\n" + "="*60)
    print("TEST 5: Gradient Flow")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping gradient tests")
        return

    device = torch.device('cuda')

    # Create simple network
    x = torch.randn(10, 10, requires_grad=True).to(device)
    w = torch.randn(10, 10, requires_grad=True).to(device)

    # Forward pass
    y = torch.matmul(x, w)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert x.grad is not None, "No gradient for x"
    assert w.grad is not None, "No gradient for w"
    assert not torch.isnan(x.grad).any(), "NaN in x gradient"
    assert not torch.isnan(w.grad).any(), "NaN in w gradient"

    print("✓ Gradient flow working correctly")

    # Clean up
    del x, w, y, loss
    torch.cuda.empty_cache()


def test_environment_setup():
    """Test GPU environment variable setup"""
    print("\n" + "="*60)
    print("TEST 6: Environment Setup")
    print("="*60)

    import os

    backend = Config.GPU_BACKEND

    if backend == "ROCm":
        # Check ROCm-specific environment variables
        assert 'PYTORCH_HIP_ALLOC_CONF' in os.environ, "PYTORCH_HIP_ALLOC_CONF not set"
        assert 'HSA_FORCE_FINE_GRAIN_PCIE' in os.environ, "HSA_FORCE_FINE_GRAIN_PCIE not set"
        print("✓ ROCm environment variables configured")

    elif backend == "CUDA":
        # Check CUDA-specific environment variables
        assert 'PYTORCH_CUDA_ALLOC_CONF' in os.environ, "PYTORCH_CUDA_ALLOC_CONF not set"
        print("✓ CUDA environment variables configured")

    else:
        print("⚠ CPU mode - no GPU environment variables expected")

    print("✓ Environment setup working correctly")


def run_all_tests():
    """Run all GPU compatibility tests"""
    print("\n" + "="*60)
    print("GPU COMPATIBILITY TEST SUITE")
    print("Project Eigen 2 - Unified CUDA/ROCm Support")
    print("="*60)

    try:
        backend = test_device_detection()
        test_gpu_memory_operations()
        test_mixed_precision_training()
        test_config_integration()
        test_gradient_flow()
        test_environment_setup()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print(f"GPU Backend: {backend}")
        print("="*60)

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
