# 📦 Video Timeline Analyzer: Full Pipeline Runner

---

## 📌 Purpose
Automates the full backend pipeline for scientific video analysis:
- Scene detection (TransNet V2, PyTorch)
- Audio transcription (Whisper)
- Visual embedding extraction (TimeSformer)
- Metadata DataFrame construction
- Qdrant vector database ingestion

## 🔄 Latest Changes
- Initial creation. Runs all pipeline stages in sequence for a given video, storing outputs in a dedicated directory.

## ⚙️ Key Logic
- Orchestrates all modules, manages file paths, logs progress, and ensures reproducibility.
- All outputs are stored in `data/outputs/<video_name>/`.
- The canonical DataFrame is saved as a Parquet file for downstream analysis.

## 📂 Expected File Path
`src/pipeline/run_full_pipeline.py`

## 🧠 Reasoning
Provides a reproducible, automated entrypoint for scientific video analysis, minimizing manual steps and errors.

---

## 🚀 Usage

### 1. **Activate your virtual environment**
```powershell
.\venv\Scripts\activate
```

### 2. **Run the pipeline**
```bash
python src/pipeline/run_full_pipeline.py "data/video submission.mp4"
```

#### Optional arguments:
- `--output_root`: Root directory for outputs (default: `data/outputs`)
- `--collection_name`: Qdrant collection name (default: `video_scenes`)
- `--vector_size`: Embedding vector size (default: 768)

### 3. **Outputs**
- All outputs are stored in `data/outputs/<video_name>/`:
  - `<video_name>_scenes.json`: Scene boundaries
  - `<video_name>_whisper.json`: Whisper transcription
  - `<video_name>_embeddings.json`: Scene-level embeddings
  - `<video_name>_canonical_metadata.parquet`: Canonical DataFrame

---

## 🧪 Requirements
- Python 3.8+
- All dependencies in `requirements.txt` (install in your venv)
- Qdrant server running (local or remote)

---

## 📝 Notes
- The pipeline will fail fast if any step fails or an output is missing.
- For batch processing, adapt the script to loop over multiple videos.
- For troubleshooting, check the output directory for intermediate files and logs.

---

## 📚 References
- See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` for full pipeline details. 