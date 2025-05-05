# Development Environment Setup

This guide provides detailed instructions for setting up the development environment for the Video Timeline Analyzer project, with special emphasis on GPU configuration and dependency management.

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

1. **Python 3.8+** (3.9 recommended)
2. **Git**
3. **FFmpeg 4.0+**
4. **CUDA Toolkit 11.7+** (for NVIDIA GPUs) or **ROCm 5.0+** (for AMD GPUs)

## 1. Scene Detection & GPU Configuration (PyTorch, Cross-Platform)

### 1.1 TransNetV2 Scene Detection (PyTorch, Hugging Face Weights)

Scene detection now uses the PyTorch implementation of TransNetV2, with official weights published on Hugging Face. This workflow is fully cross-platform (Windows, WSL2, Linux, macOS) and hardware-agnostic (CPU or GPU).

**Key steps:**
1. Ensure you have Python 3.8+, Git, and FFmpeg installed.
2. Install PyTorch (with or without GPU support, as appropriate for your hardware):
   - [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
3. Download the TransNetV2 weights using the provided script:
   ```bash
   python src/scene_detection/download_transnetv2_weights.py
   ```
4. The pipeline will automatically use GPU if available, else CPU. No TensorFlow or CUDA setup is required for scene detection.

**Hardware-agnostic model loading:**
```python
import torch
from transnetv2_pytorch import TransNetV2
model = TransNetV2()
state_dict = torch.load("transnetv2-pytorch-weights.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(state_dict)
model.eval()
if torch.cuda.is_available():
    model.cuda()
```

**Legacy/Advanced GPU Setup:**
If you require advanced GPU configuration (e.g., for other models or full scientific stack), see the troubleshooting and legacy notes at the end of this document, or refer to the [Roadmap](ROADMAP.md).

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

### 3.3 Download TransNetV2 Weights

Download the PyTorch weights for TransNetV2 using the provided script:
```bash
python src/scene_detection/download_transnetv2_weights.py
```

### 3.4 (Optional) GPU Detection Script

The project includes a utility for detecting and configuring available GPUs in `src/utils/gpu_utils.py`.
See the [Roadmap](ROADMAP.md) for details.

## 4. Testing GPU Setup

You can test your GPU setup using the provided utility in `src/utils/gpu_utils.py` or by running a simple PyTorch test. See the [Roadmap](ROADMAP.md) for details.

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

1. Download the TransNetV2 weights as described above.
2. Complete the project structure by creating core directories.
3. Begin developing core functionality according to the [Roadmap](ROADMAP.md).

For any issues, consult the troubleshooting section or create an issue on the project's GitHub repository.