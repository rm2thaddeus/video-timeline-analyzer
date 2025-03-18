# PyTorch with CUDA Global Installation Guide

This guide provides instructions for installing PyTorch with CUDA support globally on your system, ensuring it's available for all projects without requiring virtual environments.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- Python 3.x installed

## Installation Options

### Option 1: Automated Installation (Recommended)

1. **Run the installation script with administrator privileges**:
   - Right-click on `install_pytorch_cuda_admin.bat` and select "Run as administrator"
   - This will install PyTorch with CUDA support globally

2. **Verify the installation**:
   ```
   python verify_pytorch_cuda.py
   ```

### Option 2: Manual Installation

If the automated installation doesn't work, follow these steps:

1. **Open a Command Prompt or PowerShell with administrator privileges**

2. **Uninstall any existing PyTorch installations**:
   ```
   pip uninstall -y torch torchvision torchaudio
   ```

3. **Install PyTorch with CUDA support**:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify the installation**:
   ```
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"
   ```

## Fixing the test_gpu.py Script

To ensure the `test_gpu.py` script works with the global PyTorch installation:

1. **Run the fix script**:
   ```
   python fix_test_gpu.py
   ```

2. **Run the GPU test script**:
   ```
   python scripts/test_gpu.py
   ```

## Troubleshooting

### CUDA Not Available

If PyTorch is installed but CUDA is not available:

1. **Check NVIDIA driver installation**:
   ```
   nvidia-smi
   ```

2. **Verify CUDA compatibility**:
   - Ensure your NVIDIA driver supports the CUDA version required by PyTorch
   - For PyTorch 2.x, CUDA 11.8 or 12.1 is recommended

3. **Check environment variables**:
   - Ensure `CUDA_PATH` and related environment variables are set correctly

### Permission Issues

If you encounter permission errors during installation:

1. **Use administrator privileges**:
   - Run Command Prompt or PowerShell as administrator
   - Use the provided `install_pytorch_cuda_admin.bat` script

2. **Check Python installation**:
   - Ensure Python is installed for all users if installing packages globally

## Additional Resources

- [PyTorch Official Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [PyTorch Forums](https://discuss.pytorch.org/)

## Version Compatibility

| PyTorch Version | Compatible CUDA Versions |
|-----------------|--------------------------|
| 2.6.0           | 11.8, 12.1               |
| 2.0.0           | 11.7, 11.8               |
| 1.13.1          | 11.6, 11.7               |

Choose the appropriate CUDA version based on your NVIDIA driver and PyTorch version requirements. 