"""
📌 Purpose: Configuration settings for CUDA-accelerated video processing
🔄 Latest Changes: Initial configuration setup
⚙️ Key Logic: Defines constants and settings for GPU optimization
📂 Expected File Path: src/config/cuda_config.py
🧠 Reasoning: Centralized configuration for CUDA-related settings
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Directory paths
METADATA_DIR = ROOT_DIR / "metadata"
SCENES_DIR = ROOT_DIR / "scenes"
AUDIO_CHUNKS_DIR = ROOT_DIR / "audio_chunks"
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
FRAMES_DIR = ROOT_DIR / "frames"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "output"

# CUDA optimization settings
USE_CUDNN_BENCHMARK = True
USE_MIXED_PRECISION = True
USE_PINNED_MEMORY = True
GPU_MEMORY_FRACTION = 0.8  # Default fraction of GPU memory to use
OPTIMIZE_MEMORY_USAGE = True

# Model settings
WHISPER_MODEL = "medium"  # Default Whisper model size
PARALLEL_PROCESSING = True  # Enable parallel processing where possible

# Frame processing settings
FRAME_BATCH_SIZE = 32  # Number of frames to process in a batch
FRAME_RESIZE_DIM = (224, 224)  # Default frame dimensions for models

# Scene detection settings
SCENE_THRESHOLD = 0.35  # Threshold for scene change detection
MIN_SCENE_LENGTH = 15  # Minimum number of frames for a scene
BLACK_FRAME_THRESHOLD = 0.05  # Threshold for black frame detection

# Audio processing settings
AUDIO_SAMPLE_RATE = 16000  # Sample rate for audio processing
AUDIO_CHANNELS = 1  # Number of audio channels to use

# Cache settings
ENABLE_CACHE = True
CACHE_DIR = ROOT_DIR / "cache"
MAX_CACHE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB

# Ensure all directories exist
for directory in [
    METADATA_DIR, SCENES_DIR, AUDIO_CHUNKS_DIR, TRANSCRIPTS_DIR,
    FRAMES_DIR, EMBEDDINGS_DIR, LOGS_DIR, OUTPUT_DIR, CACHE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True) 