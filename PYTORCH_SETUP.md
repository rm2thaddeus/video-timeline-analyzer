# PyTorch GPU Setup Guide

## Overview

This guide helps you set up PyTorch with GPU support in a way that avoids version conflicts. By using virtual environments, we can isolate dependencies and ensure that different projects can use different PyTorch versions without interfering with each other.

## Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with compatible drivers installed
- Git (for version control)

## Quick Setup

We've created a script that automates the setup process. To use it:

1. Run the setup script:
   ```bash
   python setup_env.py
   ```

2. Activate the virtual environment:
   - Windows:
     ```bash
     venv_pytorch\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv_pytorch/bin/activate
     ```

3. Verify GPU support:
   ```bash
   python test_gpu.py
   ```

## Manual Setup (Alternative)

If you prefer to set up the environment manually:

1. Create a virtual environment:
   ```bash
   python -m venv venv_pytorch
   ```

2. Activate the virtual environment:
   - Windows:
     ```bash
     venv_pytorch\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv_pytorch/bin/activate
     ```

3. Install PyTorch with GPU support:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
   ```

4. Verify GPU support:
   ```bash
   python test_gpu.py
   ```

## Understanding PyTorch Versions and CUDA

PyTorch comes in different versions that are compatible with specific CUDA versions. The current setup uses CUDA 11.8, which is widely compatible with most NVIDIA GPUs.

If you need a different CUDA version, you can modify the `cuda_version` parameter in the `install_requirements` function in `setup_env.py`.

Common CUDA versions:
- cu118: CUDA 11.8
- cu121: CUDA 12.1
- cu117: CUDA 11.7
- cpu: No CUDA support (CPU only)

## Troubleshooting

### Multiple PyTorch Versions

If you have multiple PyTorch versions installed in your system, you might encounter conflicts. Using virtual environments as described in this guide helps isolate dependencies and avoid conflicts.

### CUDA Version Mismatch

If you see errors related to CUDA version mismatch, make sure your NVIDIA drivers are compatible with the CUDA version you're trying to use. You can check your NVIDIA driver version with:

```bash
nvidia-smi
```

### ImportError or ModuleNotFoundError

If you encounter import errors, make sure you've activated the virtual environment before running your code.

## Best Practices

1. **Always use virtual environments** for Python projects to isolate dependencies.
2. **Avoid installing PyTorch globally** to prevent version conflicts.
3. **Use flexible version specifications** in requirements.txt when possible.
4. **Test GPU availability** after installation to ensure everything is working correctly.

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [Virtual Environment Documentation](https://docs.python.org/3/library/venv.html) 