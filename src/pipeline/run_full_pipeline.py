"""
📌 Purpose – Automate the full backend pipeline: scene detection, audio transcription, embedding extraction, metadata construction, and Qdrant ingestion.
🔄 Latest Changes – Initial creation. Runs all pipeline stages in sequence for a given video, storing outputs in a dedicated directory.
⚙️ Key Logic – Orchestrates all modules, manages file paths, logs progress, and ensures reproducibility.
📂 Expected File Path – src/pipeline/run_full_pipeline.py
🧠 Reasoning – Provides a reproducible, automated entrypoint for scientific video analysis, minimizing manual steps and errors.
"""

import os
import sys
import subprocess
import shutil
import pandas as pd
from pathlib import Path

# --- CONFIGURABLE PARAMETERS ---
DEFAULT_OUTPUT_ROOT = "data/outputs"
SCENE_DET_SCRIPT = "src/scene_detection/scene_detection.py"
WHISPER_SCRIPT = "src/audio_analysis/whisper_transcribe.py"
EMBED_SCRIPT = "src/visual_analysis/timesformer_analysis.py"
METADATA_SCRIPT = "src/metadata/metadata_constructor.py"

# --- MAIN PIPELINE FUNCTION ---
def run_pipeline(video_path, output_root=DEFAULT_OUTPUT_ROOT, collection_name="video_scenes", vector_size=768):
    video_path = Path(video_path)
    video_stem = video_path.stem.replace(" ", "_")
    out_dir = Path(output_root) / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scene Detection
    scene_json = out_dir / f"{video_stem}_scenes.json"
    print(f"[Pipeline] Running scene detection...")
    subprocess.run([
        sys.executable, SCENE_DET_SCRIPT,
        str(video_path),
        str(out_dir),
        "--save_json", "True"
    ], check=True)
    assert scene_json.exists(), f"Scene detection failed: {scene_json} not found."

    # 2. Audio Transcription
    whisper_json = out_dir / f"{video_stem}_whisper.json"
    print(f"[Pipeline] Running audio transcription...")
    subprocess.run([
        sys.executable, WHISPER_SCRIPT,
        str(video_path),
        "--output_dir", str(out_dir),
        "--output_json", "True",
        "--output_srt", "False"
    ], check=True)
    assert whisper_json.exists(), f"Whisper transcription failed: {whisper_json} not found."

    # 3. Embedding Extraction
    embeddings_json = out_dir / f"{video_stem}_embeddings.json"
    print(f"[Pipeline] Running visual embedding extraction...")
    subprocess.run([
        sys.executable, EMBED_SCRIPT,
        "--video", str(video_path),
        "--scenes", str(scene_json),
        "--output", str(embeddings_json)
    ], check=True)
    assert embeddings_json.exists(), f"Embedding extraction failed: {embeddings_json} not found."

    # 4. Metadata Construction
    print(f"[Pipeline] Constructing canonical metadata DataFrame...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("metadata_constructor", METADATA_SCRIPT)
    metadata_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata_mod)
    df = metadata_mod.construct_metadata_dataframe(
        str(scene_json), str(whisper_json), str(embeddings_json)
    )
    df_path = out_dir / f"{video_stem}_canonical_metadata.parquet"
    df.to_parquet(df_path, index=False)
    print(f"[Pipeline] Canonical DataFrame saved to {df_path}")

    # 5. Qdrant Ingestion
    print(f"[Pipeline] Ingesting DataFrame into Qdrant...")
    from src.db.qdrant_client import QdrantClientWrapper
    wrapper = QdrantClientWrapper(collection_name=collection_name, vector_size=vector_size)
    wrapper.ingest_dataframe(df)
    print(f"[Pipeline] Ingestion complete.")

    print(f"[Pipeline] All outputs in: {out_dir}")
    return out_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full video analysis pipeline.")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root directory for outputs.")
    parser.add_argument("--collection_name", type=str, default="video_scenes", help="Qdrant collection name.")
    parser.add_argument("--vector_size", type=int, default=768, help="Embedding vector size.")
    args = parser.parse_args()
    run_pipeline(args.video_path, args.output_root, args.collection_name, args.vector_size) 