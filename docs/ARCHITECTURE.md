/*
📌 Purpose – Defines the modular, reproducible system architecture for the Video Timeline Analyzer, with a focus on variable-granularity metadata alignment and backend-first development.
🔄 Latest Changes – Clarified Windows-native, PyTorch-only backend; removed references to TensorFlow, PySceneDetect, and WSL2/Linux. Emphasized modular, pipeline-oriented backend and hardware-agnostic model loading for Windows.
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
- **Purpose:** Extract semantic and emotional context from frames.
- **Technologies:** CLIP (embeddings/tags), BLIP-2 (captioning), DeepFace (facial emotion), **Hugging Face TimeSformer (video embeddings, manual preprocessing; see DEVELOPMENT_SETUP.md)**

### 4.5 Metadata DataFrame Construction (Central Step)
- **Purpose:** Parse and align all pipeline outputs (scenes, transcripts, frames, embeddings, etc.) into a single DataFrame with variable granularity.
- **Logic:**
    - Each row = one scene (primary unit)
    - Columns for global, scene, frame, and transcript-segment metadata
    - Lists/arrays for variable-length fields (e.g., key frames, transcript segments)
    - **Audio JSON output**: Each transcript segment (from Whisper) is a dict with start, end, text, and optionally word-level details, enabling precise alignment and flexible DataFrame construction.
    - Alignment logic is parameterized and documented
    - DataFrame is persisted (e.g., Parquet) for reproducibility

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