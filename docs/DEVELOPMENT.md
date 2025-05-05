/*
📌 Purpose – This document provides step-by-step instructions to set up the development environment for the Video Timeline Analyzer project, ensuring consistency and reproducibility.
🔄 Latest Changes – Added development setup instructions, unit testing commands, and CI/CD pipeline overview.
⚙️ Key Logic – Contains detailed environment setup commands and testing instructions.
📂 Expected File Path – docs/DEVELOPMENT.md
🧠 Reasoning – Facilitates a clean and reproducible development setup following best practices recommended in the roadmap.
*/

# Development Setup

## Environment Setup

1. **Virtual Environment:**
   Create a virtual environment and activate it.

   ```bash
   python -m venv venv
   ```

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Unix/macOS:
     ```bash
     source venv/bin/activate
     ```

2. **Install Dependencies:**
   Install all required packages using:

   ```bash
   pip install -r video-timeline-analyzer/requirements.txt
   ```

## Validating GPU Utilities

Test the GPU detection module to ensure that your system is correctly set up:

```bash
python -c "from src/utils/gpu_utils import get_optimal_device; print('GPU Device:', get_optimal_device())"
```

## Unit Testing Framework

The project uses pytest for unit testing. To run tests, simply execute:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## CI/CD Pipeline

A GitHub Actions workflow has been configured to run tests and enforce code quality on every push. Check out the configuration files in the `.github/workflows` directory for more details.

## Further Documentation

For detailed architecture and future roadmap details, refer to:

- [Architecture](ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)

## Scene Detection (TransNetV2, PyTorch, Cross-Platform)

Scene detection now uses the PyTorch implementation of TransNetV2, with official weights published on Hugging Face. This workflow is fully cross-platform (Windows, WSL2, Linux, macOS) and hardware-agnostic (CPU or GPU).

**Key steps:**
1. Install PyTorch (with or without GPU support, as appropriate for your hardware):
   - [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
2. Download the TransNetV2 weights using the provided script:
   ```bash
   python src/scene_detection/download_transnetv2_weights.py
   ```
3. The pipeline will automatically use GPU if available, else CPU. No TensorFlow or CUDA setup is required for scene detection.

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

See the [Roadmap](ROADMAP.md) for details and troubleshooting. 