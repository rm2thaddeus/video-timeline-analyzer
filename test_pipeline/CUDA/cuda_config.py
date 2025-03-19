"""
📌 Purpose: Configuration settings for CUDA-accelerated video processing pipeline
🔄 Latest Changes: Simplified scene detection settings and removed unused batch parameters
⚙️ Key Logic: Central place for all configurable settings and hyperparameters
📂 Expected File Path: test_pipeline/CUDA/cuda_config.py
🧠 Reasoning: Centralize configuration in one file for easier adjustment and tuning
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_DIR = ROOT_DIR / "metadata"
SCENES_DIR = ROOT_DIR / "scenes"
AUDIO_CHUNKS_DIR = ROOT_DIR / "audio_chunks"
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
FRAMES_DIR = ROOT_DIR / "frames"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "output"  # Added output directory for final results

# GPU configuration
USE_CUDA = True  # Set to False to force CPU
GPU_ID = 0  # Which GPU to use (if multiple are available)
GPU_MEMORY_FRACTION = 0.4  # Fraction of GPU memory to use (0.0-1.0)
USE_MIXED_PRECISION = True  # Use mixed precision for faster computation
USE_CUDNN_BENCHMARK = True  # Use cuDNN benchmarking for faster convolutions
USE_PINNED_MEMORY = True  # Use pinned memory for faster CPU-GPU transfers
OPTIMIZE_MEMORY_USAGE = True  # Apply aggressive memory optimization techniques

# Multi-processing/threading
PARALLEL_PROCESSING = True  # Process audio extraction and frame extraction in parallel
NUM_WORKERS = 2  # Number of workers for DataLoader

# Scene detection
SCENE_THRESHOLD = 0.35  # Threshold for scene change detection (0.0-1.0)
SCENE_MIN_LENGTH = 15  # Minimum number of frames between scene changes
BLACK_THRESHOLD = 0.05  # Threshold for black frame detection (0.0-1.0)

# Audio processing
WHISPER_MODEL = "tiny"  # "tiny", "base", "small", "medium", "large"
AUDIO_CHUNK_DURATION = 30  # Length of audio chunks to process at once (in seconds)
AUDIO_CHUNK_OVERLAP = 0.5  # Overlap between audio chunks (in seconds)

# Frame processing
CLIP_MODEL = "ViT-B/32"  # CLIP model for frame embeddings
MAX_FRAME_DIMENSION = 512  # Max dimension for frame extraction (preserve aspect ratio)
FRAMES_PER_SCENE = 3  # Number of frames to extract per scene

# Parallelization settings
MAX_PARALLEL_TASKS = 2  # Maximum number of parallel tasks
