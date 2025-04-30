/*
📌 Purpose – Defines the modular, reproducible system architecture for the Video Timeline Analyzer, with a focus on variable-granularity metadata alignment.
🔄 Latest Changes – Added explicit centrality of the metadata DataFrame, clarified pipeline output parsing/alignment, and emphasized reproducibility and parameterization.
⚙️ Key Logic – All pipeline outputs are parsed and aligned into a variable-granularity DataFrame, which is the canonical source for Qdrant ingestion and downstream analysis.
📂 Expected File Path – docs/ARCHITECTURE.md
🧠 Reasoning – Ensures maintainability, extensibility, and reproducibility for scientific video analysis by making the DataFrame the central, queryable structure.
*/

# Video Timeline Analyzer: System Architecture

## 1. Overview

The Video Timeline Analyzer is a modular, pipeline-oriented application for extracting rich, interpretable metadata from video files and presenting it via an interactive timeline. The central architectural principle is the use of a variable-granularity metadata DataFrame, which aggregates and aligns all outputs from the analysis pipelines (scene detection, audio transcription, frame extraction, embeddings, etc.). This DataFrame is the canonical, queryable structure for all downstream tasks, including vector database (Qdrant) ingestion.

---

## 2. Architectural Principles

- **Modularity:** Each component is self-contained with well-defined interfaces.
- **Reproducibility:** All processing steps are deterministic, parameterized, and documented. The DataFrame is persisted for reproducibility.
- **Variable Granularity:** Metadata fields may apply globally, per scene, per frame, or per transcript segment. The DataFrame is designed to flexibly accommodate this.
- **Alignment Logic:** All pipeline outputs are parsed and aligned into the DataFrame, with explicit logic for handling overlaps and variable segmentations.
- **Extensibility:** New analysis modules and metadata fields can be added as needed.
- **Efficiency:** Parallelization and caching for large-scale video processing.
- **Resilience:** Fallbacks for critical steps; robust error handling.

---

## 3. High-Level Architecture

```
+------------------+    +---------------------+    +----------------+
| Video Ingestion  | -> | Analysis Pipelines  | -> | Metadata       |
+------------------+    +---------------------+    | DataFrame      |
                         |                |         +----------------+
                         v                v                |
                    +-----------+   +-----------+          |
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

### 4.2 Scene Detection
- **Purpose:** Segment video into scenes for timeline structure.
- **Technologies:** PySceneDetect (primary), OpenCV (fallback)

### 4.3 Audio Analysis Pipeline
- **Purpose:** Extract and analyze audio for speech, sentiment, and events.
- **Technologies:** OpenAI Whisper (transcription), Hugging Face Transformers (sentiment), librosa (audio features)

### 4.4 Visual Analysis Pipeline
- **Purpose:** Extract semantic and emotional context from frames.
- **Technologies:** CLIP (embeddings/tags), BLIP-2 (captioning), DeepFace (facial emotion)

### 4.5 Metadata DataFrame Construction (Central Step)
- **Purpose:** Parse and align all pipeline outputs (scenes, transcripts, frames, embeddings, etc.) into a single DataFrame with variable granularity.
- **Logic:**
    - Each row = one scene (primary unit)
    - Columns for global, scene, frame, and transcript-segment metadata
    - Lists/arrays for variable-length fields (e.g., key frames, transcript segments)
    - Alignment logic is parameterized and documented
    - DataFrame is persisted (e.g., Parquet) for reproducibility

### 4.6 Qdrant Vector Database Integration
- **Purpose:** Store each scene as a point in Qdrant, with multi-vector support (text/image embeddings) and all other metadata as payload.
- **Technologies:** Qdrant (vector DB), pandas (DataFrame handling)

### 4.7 Interactive Timeline UI (Optional)
- **Purpose:** Visualize video, scenes, and metadata for exploration.
- **Technologies:** PyQt5 (desktop), FastAPI (web, optional)

---

## 5. Data Flow

1. **Video Input:** Load video, extract frames/audio, and metadata.
2. **Scene Detection:** Segment into scenes.
3. **Parallel Processing:**
   - **Audio:** Transcribe, analyze sentiment, extract features.
   - **Visual:** Extract key frames, generate captions/tags, analyze faces.
4. **Metadata Alignment:** Parse and align all outputs into the variable-granularity DataFrame.
5. **Qdrant Ingestion:** Store each scene as a point with multi-vector embeddings and full payload.
6. **UI (Optional):** Render interactive timeline.

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
- Mobile/web UI (optional)

---

## 8. References
- [Technical Specifications](SPECIFICATIONS.md)
- [Development Setup](DEVELOPMENT_SETUP.md)
- [Project Roadmap](ROADMAP.md)

---

*This architecture ensures that all metadata is centrally aligned and queryable, supporting flexible, reproducible, and extensible scientific video analysis.*