# Development Environment Setup

This guide provides detailed instructions for setting up the development environment for the Video Timeline Analyzer project, with special emphasis on GPU configuration and dependency management.

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

1. **Python 3.8+** (3.10 recommended)
2. **Git**
3. **FFmpeg 4.0+**
4. **CUDA Toolkit 11.7+** (for NVIDIA GPUs) or **ROCm 5.0+** (for AMD GPUs)

## 1. GPU Configuration

### 1.1 NVIDIA GPU Setup

#### Windows

1. Download and install the latest [NVIDIA GPU Drivers](https://www.nvidia.com/Download/index.aspx)
2. Install [CUDA Toolkit 11.7+](https://developer.nvidia.com/cuda-downloads)
3. Optionally install [cuDNN](https://developer.nvidia.com/cudnn) for improved performance

Verify installation:
```bash
nvcc --version
nvidia-smi
```

#### Linux

```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-xxx  # replace xxx with latest version

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run

# Add CUDA to path
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:
```bash
nvcc --version
nvidia-smi
```

#### macOS (Limited CUDA Support)

macOS has limited CUDA support. For Apple Silicon (M1/M2), use the PyTorch MPS backend instead.

```bash
# For Apple Silicon
# No CUDA installation needed, will use MPS backend
```

### 1.2 AMD GPU Setup (ROCm)

#### Linux (Ubuntu)

```bash
# Add ROCm apt repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update

# Install ROCm
sudo apt install rocm-dev

# Add user to video group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Set environment variables
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:
```bash
rocminfo
```

## 2. Virtual Environment Setup

We'll use Conda or Python's venv to create an isolated environment for the project.

### 2.1 Using Conda (Recommended)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n video_timeline python=3.10
conda activate video_timeline

# Install core dependencies with GPU support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 2.2 Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Activate environment (Linux/macOS)
source venv/bin/activate

# Install pip and setuptools
pip install --upgrade pip setuptools wheel
```

## 3. Project Setup

### 3.1 Clone Repository

```bash
git clone https://github.com/rm2thaddeus/video-timeline-analyzer.git
cd video-timeline-analyzer
```

### 3.2 Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3.3 GPU Detection Script

Create a script to detect and configure available GPUs:

```python
# src/utils/gpu_utils.py

import os
import platform
import logging
from typing import Tuple, Optional

import torch

logger = logging.getLogger(__name__)

def detect_gpu() -> Tuple[bool, Optional[str], int]:
    """
    Detect available GPU and return configuration information.
    
    Returns:
        Tuple containing:
        - Boolean indicating if GPU is available
        - String with GPU type ('cuda', 'mps', 'rocm', None)
        - Integer with device count (0 if no GPU)
    """
    has_gpu = False
    gpu_type = None
    device_count = 0
    
    # Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        has_gpu = True
        gpu_type = 'cuda'
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            logger.info(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
            logger.info(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Check for Apple Silicon (MPS)
    elif platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        has_gpu = True
        gpu_type = 'mps'
        device_count = 1
        logger.info("Apple Silicon (MPS) GPU available")
    
    # Check for AMD ROCm
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        has_gpu = True
        gpu_type = 'rocm'
        device_count = torch.hip.device_count()
        logger.info(f"ROCm GPU available with {device_count} devices")
    
    # No GPU available
    else:
        logger.warning("No GPU detected. Using CPU only (performance will be significantly reduced)")
    
    # Log environment variables for debugging
    logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.debug(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'Not set')}")
    
    return has_gpu, gpu_type, device_count

def get_optimal_device() -> torch.device:
    """
    Get the optimal device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: The optimal device for computation
    """
    has_gpu, gpu_type, _ = detect_gpu()
    
    if not has_gpu:
        return torch.device('cpu')
    
    if gpu_type == 'cuda':
        return torch.device('cuda')
    elif gpu_type == 'mps':
        return torch.device('mps')
    elif gpu_type == 'rocm':
        return torch.device('cuda')  # ROCm uses the 'cuda' device type in PyTorch
    
    return torch.device('cpu')

def optimize_gpu_settings():
    """Configure optimal settings for the detected GPU."""
    has_gpu, gpu_type, _ = detect_gpu()
    
    if not has_gpu:
        return
    
    # NVIDIA CUDA settings
    if gpu_type == 'cuda':
        # Set TF32 for faster computation on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # Leave some GPU memory for other processes
            torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Apple Silicon MPS settings  
    elif gpu_type == 'mps':
        # MPS-specific optimizations would go here
        pass
    
    # AMD ROCm settings
    elif gpu_type == 'rocm':
        torch.backends.cudnn.benchmark = True
```

## 4. Testing GPU Setup

Create a test script to verify that GPU is correctly configured:

```bash
# Create test script
cat > test_gpu.py << 'EOL'
#!/usr/bin/env python

import torch
import time
from src.utils.gpu_utils import detect_gpu, get_optimal_device

def main():
    print("Testing GPU setup...")
    has_gpu, gpu_type, device_count = detect_gpu()
    
    if not has_gpu:
        print("No GPU detected. Running on CPU only.")
        device = torch.device('cpu')
    else:
        print(f"GPU detected: {gpu_type} with {device_count} devices")
        device = get_optimal_device()
    
    print(f"Using device: {device}")
    
    # Create test tensors
    size = 5000
    print(f"Running matrix multiplication test ({size}x{size})...")
    
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    torch.matmul(a, b)
    
    # Benchmark
    start_time = time.time()
    for _ in range(3):
        c = torch.matmul(a, b)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 3
    print(f"Average time: {avg_time:.4f} seconds")
    print("GPU test completed successfully!")

if __name__ == "__main__":
    main()
EOL

# Run the test
python test_gpu.py
```

## 5. Common Issues & Troubleshooting

### 5.1 NVIDIA CUDA Issues

#### "CUDA not available" despite NVIDIA GPU

1. Verify drivers are installed: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure PyTorch is installed with CUDA support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

#### Out of Memory Errors

1. Reduce batch sizes
2. Implement gradient checkpointing
3. Use mixed precision training
4. Clear cache periodically:
   ```python
   torch.cuda.empty_cache()
   ```

### 5.2 AMD ROCm Issues

#### ROCm Installation Problems

1. Verify hardware compatibility: [ROCm Hardware Requirements](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
2. Check kernel compatibility on Linux
3. Verify user permissions for /dev/kfd and /dev/dri/

#### Performance Issues

1. Update to latest ROCm drivers
2. Check thermal throttling: `rocm-smi`
3. Optimize model for ROCm architecture

### 5.3 Apple Silicon (M1/M2) Issues

#### MPS Backend Not Working

1. Ensure using PyTorch 1.12+ with macOS 12.3+
2. Check installation:
   ```python
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

## 6. Environment Variables

Set these environment variables for optimal GPU performance:

### 6.1 NVIDIA CUDA

```bash
# Linux/macOS
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU (0, 1, etc.)
export TF_FORCE_GPU_ALLOW_GROWTH=true  # For TensorFlow

# Windows
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 6.2 AMD ROCm

```bash
# Linux
export HIP_VISIBLE_DEVICES=0  # Use specific GPU
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For compatibility with older software
```

### 6.3 Memory Management

```bash
# Limit GPU memory growth (PyTorch handles this internally)
# For TensorFlow:
export TF_MEMORY_ALLOCATION=0.8  # Use 80% of GPU memory
```

## 7. Recommended Development Tools

- **Visual Studio Code** with Python extension
- **NVIDIA Nsight** for CUDA debugging (NVIDIA GPUs)
- **ROCm-SMI** for AMD GPU monitoring
- **PyTorch Profiler** for model optimization
- **Weights & Biases** for experiment tracking

## 8. Next Steps

After successfully setting up your development environment:

1. Complete the project structure by creating core directories
2. Implement the GPU utility module in `src/utils/gpu_utils.py`
3. Begin developing core functionality according to the roadmap

For any issues, consult the troubleshooting section or create an issue on the project's GitHub repository.