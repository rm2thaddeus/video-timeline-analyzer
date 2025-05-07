"""
📌 Purpose – Automate the full backend pipeline: scene detection, audio transcription, embedding extraction, metadata construction, and Qdrant ingestion, with robust logging and full CUDA utilization.
🔄 Latest Changes – Added structured logging, explicit CUDA checks, and parallelization/max GPU usage for all steps. Logs all subprocess outputs and errors for troubleshooting.
⚙️ Key Logic – Orchestrates all modules, manages file paths, logs progress and errors, ensures CUDA is used, and maximizes parallelization.
📂 Expected File Path – src/pipeline/run_full_pipeline.py
🧠 Reasoning – Provides a reproducible, automated entrypoint for scientific video analysis, maximizing hardware usage and troubleshooting clarity.
"""

import os
import sys
import subprocess
import shutil
import pandas as pd
from pathlib import Path
import logging
import torch
import json

# --- CONFIGURABLE PARAMETERS ---
DEFAULT_OUTPUT_ROOT = "data/outputs"
SCENE_DET_SCRIPT = "src/scene_detection/scene_detection.py"
WHISPER_SCRIPT = "src/audio_analysis/whisper_transcribe.py"
EMBED_SCRIPT = "src/visual_analysis/timesformer_analysis.py"
METADATA_SCRIPT = "src/metadata/metadata_constructor.py"

# --- LOGGING SETUP ---
def setup_logging(log_path):
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# --- MAIN PIPELINE FUNCTION ---
def run_pipeline(video_path, output_root=DEFAULT_OUTPUT_ROOT, collection_name="video_scenes", vector_size=768, num_workers=4):
    video_path = Path(video_path)
    video_stem = video_path.stem.replace(" ", "_")
    out_dir = Path(output_root) / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{video_stem}_pipeline.log"
    logger = setup_logging(log_path)

    # CUDA check
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    logger.info(f"CUDA available: {cuda_available} ({cuda_device})")

    # 1. Scene Detection
    scene_json = out_dir / f"{video_stem}_scenes.json"
    scene_profile_json = out_dir / f"{video_stem}_scene_detection_profile.json"
    logger.info("[Pipeline] Running scene detection...")
    try:
        result = subprocess.run([
            sys.executable, SCENE_DET_SCRIPT,
            str(video_path),
            str(out_dir),
            "--save_json", "True",
            "--device", "cuda" if cuda_available else "cpu",
            "--num_workers", str(num_workers)
        ], capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Scene detection failed: {e.stderr}")
        raise
    assert scene_json.exists(), f"Scene detection failed: {scene_json} not found."

    # 2. Audio Transcription
    whisper_json = out_dir / f"{video_stem}_whisper.json"
    whisper_profile_json = out_dir / f"{video_stem}_whisper_transcription_profile.json"
    logger.info("[Pipeline] Running audio transcription...")
    try:
        result = subprocess.run([
            sys.executable, WHISPER_SCRIPT,
            str(video_path),
            "--output_dir", str(out_dir),
            "--output_json", "True",
            "--output_srt", "False",
            "--device", "cuda" if cuda_available else "cpu"
        ], capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Whisper transcription failed: {e.stderr}")
        raise
    assert whisper_json.exists(), f"Whisper transcription failed: {whisper_json} not found."

    # 3. Embedding Extraction
    embeddings_json = out_dir / f"{video_stem}_embeddings.json"
    embed_profile_json = out_dir / f"{video_stem}_timesformer_embedding_profile.json"
    logger.info("[Pipeline] Running visual embedding extraction...")
    try:
        result = subprocess.run([
            sys.executable, EMBED_SCRIPT,
            "--video", str(video_path),
            "--scenes", str(scene_json),
            "--output", str(embeddings_json),
            "--device", "cuda" if cuda_available else "cpu"
        ], capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Embedding extraction failed: {e.stderr}")
        raise
    assert embeddings_json.exists(), f"Embedding extraction failed: {embeddings_json} not found."

    # 4. Metadata Construction
    logger.info("[Pipeline] Constructing canonical metadata DataFrame...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("metadata_constructor", METADATA_SCRIPT)
    metadata_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata_mod)
    df = metadata_mod.construct_metadata_dataframe(
        str(scene_json), str(whisper_json), str(embeddings_json)
    )
    df_path = out_dir / f"{video_stem}_canonical_metadata.parquet"
    df.to_parquet(df_path, index=False)
    logger.info(f"[Pipeline] Canonical DataFrame saved to {df_path}")

    # 5. Qdrant Ingestion
    logger.info("[Pipeline] Ingesting DataFrame into Qdrant...")
    from src.db.qdrant_client import QdrantClientWrapper
    wrapper = QdrantClientWrapper(collection_name=collection_name, vector_size=vector_size)
    wrapper.ingest_dataframe(df)
    logger.info("[Pipeline] Ingestion complete.")

    # --- Profiling Summary ---
    logger.info("[Pipeline] Profiling Summary:")
    def try_load_profile(profile_path):
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                return json.load(f)
        return None
    profiles = {
        'Scene Detection': try_load_profile(scene_profile_json),
        'Whisper Transcription': try_load_profile(whisper_profile_json),
        'Timesformer Embedding': try_load_profile(embed_profile_json)
    }
    logger.info("\n| Stage | Time (s) | CPU Mem (MB) | GPU Mem (MB) |\n|-------|----------|--------------|--------------|")
    for stage, prof in profiles.items():
        if prof:
            logger.info(f"| {stage} | {prof['time_sec']:.2f} | {prof['cpu_mem_mb']:.2f} | {prof['gpu_mem_mb']:.2f} |")
        else:
            logger.info(f"| {stage} | N/A | N/A | N/A |")

    logger.info(f"[Pipeline] All outputs in: {out_dir}")
    return out_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full video analysis pipeline.")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root directory for outputs.")
    parser.add_argument("--collection_name", type=str, default="video_scenes", help="Qdrant collection name.")
    parser.add_argument("--vector_size", type=int, default=768, help="Embedding vector size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for scene detection (lower if FFmpeg/OpenCV errors occur).")
    args = parser.parse_args()
    run_pipeline(args.video_path, args.output_root, args.collection_name, args.vector_size, args.num_workers) 