"""
📌 Purpose: Configuration module for the video processing pipeline
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Centralized configuration for pipeline components
📂 Expected File Path: test_pipeline/config.py
🧠 Reasoning: Separate configuration from implementation for better maintainability
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = ROOT_DIR / "metadata"
SCENES_DIR = ROOT_DIR / "scenes"
AUDIO_CHUNKS_DIR = ROOT_DIR / "audio_chunks"
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
FRAMES_DIR = ROOT_DIR / "frames"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
LOGS_DIR = ROOT_DIR / "logs"

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTIONS = {
    "transcript_embeddings": {
        "size": 512,  # CLIP text embedding dimension
        "distance": "cosine"
    },
    "frame_embeddings": {
        "size": 512,  # CLIP image embedding dimension
        "distance": "cosine"
    }
}

# Video processing settings
FRAME_SAMPLE_RATE = 1  # Extract 1 frame per X seconds
SCENE_THRESHOLD = 0.35  # Content detection threshold for scene detection
MAX_FRAME_DIMENSION = 720  # Resize frames to this maximum dimension

# Audio processing
AUDIO_CHUNK_DURATION = 30  # Duration of audio chunks in seconds
AUDIO_CHUNK_OVERLAP = 2  # Overlap between chunks in seconds

# Model configurations
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
CLIP_MODEL = "ViT-B/32"  # CLIP model variant

# GPU settings
USE_GPU = True  # Use GPU for processing if available
GPU_BATCH_SIZE = 16  # Batch size for GPU processing 