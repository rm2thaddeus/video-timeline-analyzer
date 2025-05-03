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

> **GPU Compatibility Note:**  
> For full GPU acceleration of both TensorFlow (scene detection) and PyTorch-based models (Whisper, embeddings), you must use:
> - TensorFlow 2.10.0 (last Windows GPU version) with CUDA 11.2 and cuDNN 8.1, and a compatible PyTorch (e.g., 1.12.1+cu112).
> - Or, for the latest TensorFlow and PyTorch GPU support, use WSL2 + Ubuntu, which allows both frameworks to use the latest CUDA stack.
> See the [Roadmap](ROADMAP.md) for details and troubleshooting. 