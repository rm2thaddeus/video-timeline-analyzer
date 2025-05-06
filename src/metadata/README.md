# Metadata & DataFrame Module

This folder contains code for constructing, aligning, and exporting the central metadata DataFrame.

**Main module:**
- `metadata_constructor.py`: Loads scene boundaries, Whisper audio segments, and visual embeddings; aligns them by scene; constructs a canonical DataFrame for downstream analysis and Qdrant ingestion.
- Main function: `construct_metadata_dataframe(scene_json_path, whisper_json_path, embeddings_json_path, ...)`

**Typical tasks:**
- Aligning scene, audio, and visual metadata
- Building the canonical DataFrame for analysis and storage
- Exporting DataFrame to Parquet, CSV, or for Qdrant ingestion

**Inputs:**
- Scene boundaries (JSON)
- Whisper segments (JSON)
- Visual embeddings (JSON)

**Output:**
- pandas DataFrame with one row per scene, including transcript, embedding, and (optionally) action scores

If you change the DataFrame structure, document and version-control it here for scientific reproducibility and pipeline compatibility. 