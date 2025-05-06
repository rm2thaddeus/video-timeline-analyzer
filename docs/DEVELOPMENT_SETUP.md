# Development Environment Setup

This guide provides detailed instructions for setting up the development environment for the Video Timeline Analyzer project, with special emphasis on GPU configuration and dependency management for Windows-native development.

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

1. **Python 3.8+** (3.9 recommended)
2. **Git**
3. **FFmpeg 4.0+**
4. **CUDA Toolkit 11.7+** (for NVIDIA GPUs, if using GPU acceleration)

## 1. Scene Detection & GPU Configuration (PyTorch, Windows-Native)

### 1.1 TransNetV2 Scene Detection (PyTorch, Hugging Face Weights)

Scene detection now uses the PyTorch implementation of TransNetV2, with official weights published on Hugging Face. This workflow is fully Windows-native and hardware-agnostic (CPU or GPU).

**Key steps:**
1. Ensure you have Python 3.8+, Git, and FFmpeg installed.
2. Install PyTorch (with or without GPU support, as appropriate for your hardware):
   - [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
3. Download the TransNetV2 weights using the provided script:
   ```powershell
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

## 2. Virtual Environment Setup

We'll use Python's venv to create an isolated environment for the project.

```powershell
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Install pip and setuptools
pip install --upgrade pip setuptools wheel
```

## 3. Project Setup

### 3.1 Clone Repository

```powershell
git clone https://github.com/rm2thaddeus/video-timeline-analyzer.git
cd video-timeline-analyzer
```

### 3.2 Install Dependencies

```powershell
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3.3 Download TransNetV2 Weights

Download the PyTorch weights for TransNetV2 using the provided script:
```powershell
python src/scene_detection/download_transnetv2_weights.py
```

### 3.4 (Optional) GPU Detection Script

The project includes a utility for detecting and configuring available GPUs in `src/utils/gpu_utils.py`.
See the [Roadmap](ROADMAP.md) for details.

## 4. Testing GPU Setup

You can test your GPU setup using the provided utility in `src/utils/gpu_utils.py` or by running a simple PyTorch test. See the [Roadmap](ROADMAP.md) for details.

## 5. Common Issues & Troubleshooting (Windows)

### 5.1 NVIDIA CUDA Issues

#### "CUDA not available" despite NVIDIA GPU

1. Verify drivers are installed: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Ensure PyTorch is installed with CUDA support:
   ```powershell
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

## 6. Environment Variables (Windows)

Set these environment variables for optimal GPU performance:

### 6.1 NVIDIA CUDA

```powershell
set CUDA_VISIBLE_DEVICES=0
```

## 7. Recommended Development Tools

- **Visual Studio Code** with Python extension
- **NVIDIA Nsight** for CUDA debugging (NVIDIA GPUs)
- **PyTorch Profiler** for model optimization

## 8. Next Steps

After successfully setting up your development environment:

1. Download the TransNetV2 weights as described above.
2. Complete the project structure by creating core directories.
3. Begin developing core functionality according to the [Roadmap](ROADMAP.md).

For any issues, consult the troubleshooting section or create an issue on the project's GitHub repository.

## 3. Visual Embedding Extraction (Hugging Face TimeSformer)

A new module (`src/visual_analysis/embedding_models/hf_timesformer.py`) provides robust video embedding extraction using Hugging Face's TimeSformer model. This implementation:
- Manually preprocesses video frames (resize to 224x224, RGB conversion, ImageNet normalization)
- Avoids dependency on `TimesformerImageProcessor` or `TimesformerFeatureExtractor`
- Is robust to Hugging Face API changes
- Returns a 768-dimensional embedding for each video

**Usage Example:**
```python
from src.visual_analysis.embedding_models.hf_timesformer import TimeSformerVideoEmbedder
embedder = TimeSformerVideoEmbedder()
embedding = embedder.get_video_embedding('path/to/video.mp4')
```

To test, run:
```bash
python scripts/test_hf_timesformer_embedding.py
```