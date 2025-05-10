/*
📌 Purpose – Defines the modular, reproducible system architecture for the Video Timeline Analyzer, with a focus on variable-granularity metadata alignment and backend-first development.
🔄 Latest Changes – Clarified Windows-native, PyTorch-only backend; removed references to TensorFlow, PySceneDetect, and WSL2/Linux. Emphasized modular, pipeline-oriented backend and hardware-agnostic model loading for Windows. Updated to reflect hardware constraints: Docker integration with large frameworks (PyTorch, TensorFlow) is currently infeasible due to disk space. Added recommendations for lightweight CUDA alternatives (CuPy, Numba) and rationale for minimizing dependencies while retaining GPU acceleration.
⚙️ Key Logic – All pipeline outputs are parsed and aligned into a variable-granularity DataFrame, which is the canonical source for Qdrant ingestion and downstream analysis. Scene detection is modular: TransNet V2 (PyTorch-only, hardware-agnostic, Windows-native).
📂 Expected File Path – docs/ARCHITECTURE.md
🧠 Reasoning – Ensures maintainability, extensibility, and reproducibility for scientific video analysis by making the DataFrame the central, queryable structure and prioritizing backend development for Windows.
*/

# Video Timeline Analyzer: System Architecture

## 1. Overview

The Video Timeline Analyzer is a modular, pipeline-oriented backend application for extracting rich, interpretable metadata from video files. The central architectural principle is the use of a variable-granularity metadata DataFrame, which aggregates and aligns all outputs from the analysis pipelines (scene detection, audio transcription, frame extraction, embeddings, etc.). This DataFrame is the canonical, queryable structure for all downstream tasks, including vector database (Qdrant) ingestion.

**Current focus:** Backend development only (no UI yet). All logic is modular and pipeline-oriented, with PyTorch-only scene detection (TransNet V2, Windows-native, hardware-agnostic).

---

## 2. Architectural Principles

- **Modularity:** Each component is self-contained with well-defined interfaces.
- **Reproducibility:** All processing steps are deterministic, parameterized, and documented. The DataFrame is persisted for reproducibility.
- **Variable Granularity:** Metadata fields may apply globally, per scene, per frame, or per transcript segment. The DataFrame is designed to flexibly accommodate this.
- **Alignment Logic:** All pipeline outputs are parsed and aligned into the DataFrame, with explicit logic for handling overlaps and variable segmentations.
- **Extensibility:** New analysis modules and metadata fields can be added as needed.
- **Efficiency:** Parallelization and caching for large-scale video processing.
- **Resilience:** Robust error handling.

> **Note on GPU Stack Compatibility:**  
> The pipeline is Windows-native and uses PyTorch for all deep learning modules (scene detection, Whisper, embeddings). No TensorFlow, PySceneDetect, or WSL2/Linux is required. Hardware-agnostic model loading ensures automatic use of GPU if available, else CPU.

---

## 3. High-Level Architecture

```
+------------------+    +---------------------+    +----------------+
| Video Ingestion  | -> | Analysis Pipelines  | -> | Metadata       |
+------------------+    +---------------------+    | DataFrame      |
                         |                |         +----------------+
                         v                v                |
                    | Audio     |   | Visual    |          |
                    | Pipeline  |   | Pipeline  |          |
                    +-----------+   +-----------+          |
                         |                |                |
                         +----------------+                |
                                  |                        |
                                  v                        v
                            +-------------------------------+
                            |   Qdrant Vector Database      |
                            +-------------------------------+
```

## 3a. Containerization, Technology Choices, and Disk Management

### Rationale for Docker
- **Reproducibility:** Docker ensures that all contributors and deployments use the exact same environment, eliminating "works on my machine" issues.
- **GPU Acceleration:** The Dockerfile is based on NVIDIA CUDA images, enabling seamless GPU access for scientific workloads.
- **Dependency Management:** All dependencies (CUDA, cuDNN, PyTorch, OpenCV, etc.) are installed in a controlled, isolated environment, reducing conflicts and setup time.

### Exclusion of TensorFlow and WSL2
- **TensorFlow:** Initially considered, but excluded due to its large disk footprint, additional complexity, and lack of necessity for the current PyTorch-based pipeline. All deep learning modules are now PyTorch-only, simplifying the stack and improving maintainability.
- **WSL2:** Early development under WSL2 led to significant disk space consumption and operational complexity. The project is now fully Windows-native, with Docker providing all necessary isolation and compatibility.

### Docker Image Size and Disk Space Management
- **Image Size:** The initial GPU-accelerated Docker image exceeded 35GB, primarily due to large dependencies and lack of cleanup. This prompted the adoption of aggressive pruning and cleanup commands (e.g., `apt-get clean`, removing pip/apt caches) in the Dockerfile.
- **Disk Management:** Lessons from WSL2 and Docker image bloat have led to the adoption of best practices for disk and image management, now documented in DEVELOPMENT_SETUP.md and enforced by Cursor Project Rules. Regular use of Docker system prune and careful volume management are recommended.
- **Hardware Constraints & Alternatives:** If disk space is a limiting factor, consider omitting large frameworks (PyTorch, TensorFlow) from your Docker builds and using lightweight CUDA libraries (CuPy, Numba) where possible. This can significantly reduce image size and make Docker integration more feasible on constrained hardware. Reassess project requirements to determine if full deep learning frameworks are necessary for your use case.

### Current State
- **PyTorch-Only, Windows-Native, Docker-Based:** The backend is now fully PyTorch-based, running natively on Windows via Docker containers. All contributors are encouraged to use the provided Dockerfile and follow the Cursor Project Rules for reproducibility and efficiency.
- **Best Practices Enforced:** All build and operational processes include explicit pruning and cleanup steps. Contributors should reference DEVELOPMENT_SETUP.md, ROADMAP.md, and the Cursor Project Rules for up-to-date operational guidance.

### Lessons Learned
- Avoid unnecessary dependencies and layers in the Dockerfile.
- Always include cleanup steps after installing packages.
- Monitor disk usage regularly, especially when using WSL2 or large Docker images.
- Prefer a unified, minimal stack (PyTorch-only) for maintainability and efficiency.

For further details, see [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md), [ROADMAP.md](ROADMAP.md), and the Cursor Project Rules.

---

## 4. Core Components

### 4.1 Video Ingestion & Pre-Processing
- **Purpose:** Load video, extract frames/audio, and basic metadata.
- **Technologies:** FFmpeg (video/audio extraction), OpenCV (frame handling)

### 4.2 Scene Detection (PyTorch-Only, Windows-Native)
- **Purpose:** Segment video into scenes for timeline structure.
- **Logic:** Use TransNet V2 (PyTorch-only, hardware-agnostic, Windows-native) for all scene detection.
- **Technologies:** TransNet V2 (PyTorch), OpenCV
- **Rationale:** Maximizes speed and accuracy on GPU systems, ensures compatibility everywhere on Windows.

### 4.3 Audio Analysis Pipeline
- **Purpose:** Extract and analyze audio for speech, sentiment, and events.
- **Technologies:** OpenAI Whisper (transcription), Hugging Face Transformers (sentiment), librosa (audio features)
- **Outputs:**
    - **JSON**: Segment-level output (start/end, text) for direct ingestion into the variable-granularity DataFrame (default, always saved).
    - **SRT**: Optional subtitle file for human alignment and review (enabled by argument).
    - **WAV**: Optional full audio extraction (enabled by argument).

### 4.4 Visual Analysis Pipeline
- **Purpose:** Extract semantic and emotional context from frames and scenes.
- **Technologies:** CLIP (embeddings/tags), BLIP-2 (captioning), DeepFace (facial emotion), **Hugging Face TimeSformer (scene-level video embeddings, robust manual preprocessing; see DEVELOPMENT_SETUP.md)**
- **Implementation Note:** The pipeline now uses Hugging Face TimeSformer for all scene-level embedding extraction, ensuring maintainability and reproducibility. The legacy/custom TimeSformer model is deprecated.

### 4.5 Metadata DataFrame Construction (Central Step)
- **Purpose:** Parse and align all pipeline outputs (scenes, transcripts, frames, embeddings, etc.) into a single DataFrame with variable granularity.
- **Logic:**
    - Each row = one scene (primary unit)
    - Columns for global, scene, frame, and transcript-segment metadata
    - Lists/arrays for variable-length fields (e.g., key frames, transcript segments)
    - **Audio JSON output**: Each transcript segment (from Whisper) is a dict with start, end, text, and optionally word-level details, enabling precise alignment and flexible DataFrame construction.
    - Alignment logic is parameterized and documented
    - DataFrame is persisted (e.g., Parquet) for reproducibility
    - **Implementation:** The canonical DataFrame constructor is in `src/metadata/metadata_constructor.py` and is robust to scene boundary formats (dicts with 'start_time'/'end_time' or lists).

### 4.6 Qdrant Vector Database Integration
- **Purpose:** Store each scene as a point in Qdrant, with multi-vector support (text/image embeddings) and all other metadata as payload.
- **Technologies:** Qdrant (vector DB), pandas (DataFrame handling)

### 4.7 Interactive Timeline UI (Deferred)
- **Purpose:** Visualize video, scenes, and metadata for exploration.
- **Technologies:** PyQt5 (desktop), FastAPI (web, optional)
- **Status:** Deferred until backend is complete.

---

## 5. Data Flow

1. **Video Input:** Load video, extract frames/audio, and metadata.
2. **Scene Detection:** Segment into scenes (TransNet V2 if CUDA, else PySceneDetect).
3. **Parallel Processing:**
   - **Audio:** Transcribe, analyze sentiment, extract features.
   - **Visual:** Extract key frames, generate captions/tags, analyze faces.
4. **Metadata Alignment:** Parse and align all outputs into the variable-granularity DataFrame.
5. **Qdrant Ingestion:** Store each scene as a point with multi-vector embeddings and full payload.
6. **UI (Deferred):** Render interactive timeline (future work).

---

## 6. Reproducibility & Parameterization
- All steps are parameterized (scene detection sensitivity, frame extraction rate, transcript segment length, etc.)
- All parameters and random seeds are logged and version-controlled
- The DataFrame is persisted for reproducibility
- Alignment logic is explicitly documented

---

## 7. Extensibility & Future Work
- Plugin system for new analyzers and metadata fields (planned)
- Distributed/cloud processing (future)
- UI development (future)

---

## 8. References
- [Technical Specifications](SPECIFICATIONS.md)
- [Development Setup](DEVELOPMENT_SETUP.md)
- [Project Roadmap](ROADMAP.md)
- [TransNet V2 (Shot Boundary Detection)](https://github.com/soCzech/TransNetV2)
- [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/)

---

*This architecture ensures that all metadata is centrally aligned and queryable, supporting flexible, reproducible, and extensible scientific video analysis. Backend-first development ensures a robust foundation before UI work begins.*

## Windows Compatibility and Weights Management

The `de-novo-windows` branch enables native Windows development by using PyTorch weights for TransNetV2 from Hugging Face. The weights are not committed to the repository; instead, use the script at `src/scene_detection/download_transnetv2_weights.py` to download them.

**Hardware-agnostic model loading:**
The pipeline automatically detects CUDA availability and loads the model on GPU if available, else CPU:
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