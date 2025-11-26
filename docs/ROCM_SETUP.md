# ROCm Setup Guide for Project Eigen 2

This guide explains how to run Project Eigen 2 on AMD GPUs using ROCm.

## Overview

Project Eigen 2 supports **both NVIDIA (CUDA) and AMD (ROCm) GPUs** through a unified codebase. The code automatically detects your GPU backend and configures itself appropriately.

## Supported AMD GPUs

### Consumer GPUs (RDNA2/RDNA3)
- Radeon RX 6800
- Radeon RX 6900 XT
- Radeon RX 7900 XT
- Radeon RX 7900 XTX

### Datacenter GPUs
- AMD Instinct MI100
- AMD Instinct MI200 series
- AMD Instinct MI250/MI250X
- AMD Instinct MI300 series

## Installation

### 1. Install ROCm Drivers

**Ubuntu/Debian:**
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_latest.deb
sudo dpkg -i amdgpu-install_latest.deb
sudo apt update

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to video and render groups
sudo usermod -a -G render,video $USER
```

**RHEL/CentOS:**
```bash
# Add ROCm repository
sudo yum install https://repo.radeon.com/amdgpu-install/latest/rhel/8/amdgpu-install-latest.noarch.rpm

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to groups
sudo usermod -a -G render,video $USER
```

**Reboot after installation:**
```bash
sudo reboot
```

### 2. Verify ROCm Installation

```bash
# Check ROCm version
rocm-smi

# Should show your AMD GPU and ROCm version
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ROCm-specific requirements
pip install -r requirements_rocm.txt
```

## Verification

### Test GPU Detection

```bash
# Run device detection test
python utils/device_utils.py
```

Expected output:
```
============================================================
GPU Device Detection Test
============================================================

Device: cuda:0
Backend: ROCm
GPU Name: AMD Radeon RX 7900 XTX
Backend Version: ROCm 5.7

GPU Memory Info:
  Allocated: 0.00 GB
  Reserved: 0.00 GB
  Device Count: 1

✓ Environment configured for ROCm
============================================================
```

### Run Full Compatibility Tests

```bash
# Run comprehensive GPU compatibility tests
python tests/test_gpu_compatibility.py
```

All tests should pass with `✓` marks.

## Performance Expectations

| AMD GPU | Comparable NVIDIA GPU | Training Performance |
|---------|----------------------|---------------------|
| RX 7900 XTX | RTX 4080 | ~85-95% |
| RX 6900 XT | RTX 3090 | ~80-90% |
| MI250X | A100 80GB | ~90-100% |
| MI300X | H100 | ~85-95% |

*Performance measured on typical RL training workloads*

## Training

Training commands are **identical** for CUDA and ROCm:

```bash
# Start fresh training run
python main.py

# Resume from checkpoint
python main.py --resume

# Enable consistency mode
python main.py --consistency
```

The code automatically detects your AMD GPU and configures ROCm-specific optimizations.

## Environment Variables

The following environment variables are **automatically configured** by the code:

```bash
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  # Memory management
HSA_FORCE_FINE_GRAIN_PCIE=1                      # PCIe optimization
```

You can override these by setting them before running Python if needed.

## Troubleshooting

### GPU Not Detected

```bash
# Check if GPU is visible
rocm-smi

# Check PyTorch can see GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Out of Memory Errors

ROCm memory management is similar to CUDA. Try:

1. Reduce batch size in `utils/config.py`:
   ```python
   BATCH_SIZE = 8  # Reduce from 16
   ```

2. Enable gradient checkpointing (already enabled by default)

3. Monitor memory usage:
   ```bash
   watch -n 1 rocm-smi
   ```

### Performance Issues

1. **Check ROCm version**: ROCm 5.7+ recommended
   ```bash
   cat /opt/rocm/.info/version
   ```

2. **Enable PCIe resizable BAR** (BIOS setting) for best performance

3. **Update GPU drivers**:
   ```bash
   sudo amdgpu-install --usecase=rocm --update
   ```

### Build Issues

If you encounter build issues with PyTorch:

```bash
# Try using pre-built wheels
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm5.7
```

## Model Compatibility

✅ **Model checkpoints are fully compatible** between CUDA and ROCm:
- Train on NVIDIA GPU → Deploy on AMD GPU ✓
- Train on AMD GPU → Deploy on NVIDIA GPU ✓
- No conversion needed

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Installation Guide](https://pytorch.org/get-started/locally/)
- [AMD GPU Compatibility Matrix](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)

## Support

For ROCm-specific issues:
1. Check [ROCm GitHub Issues](https://github.com/RadeonOpenCompute/ROCm/issues)
2. Visit [AMD Community Forums](https://community.amd.com/)
3. Check PyTorch ROCm compatibility

For project-specific issues:
1. Run compatibility tests: `python tests/test_gpu_compatibility.py`
2. Check GPU detection: `python utils/device_utils.py`
3. Review logs for backend-specific messages
