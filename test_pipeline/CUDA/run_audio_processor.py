"""
📌 Purpose: Test script for the CUDA-optimized audio processor
🔄 Latest Changes: Initial implementation for testing word-level audio transcription
⚙️ Key Logic: Runs the audio processor on a video file with GPU acceleration
📂 Expected File Path: test_pipeline/CUDA/run_audio_processor.py
🧠 Reasoning: Provides a simple interface to test the audio processing pipeline
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

import torch
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the CUDA-optimized audio processor
from audio_processor_cuda import AudioProcessorCUDA
import cuda_config as config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / f"audio_processor_cuda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("audio_processor_cuda_test")

def ensure_directories():
    """Create all necessary directories for the audio processor."""
    directories = [
        config.AUDIO_CHUNKS_DIR,
        config.TRANSCRIPTS_DIR,
        config.LOGS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def process_video(video_path, output_dir=None, chunk_duration=30.0, overlap=2.0, max_workers=8):
    """
    Process a video file with the CUDA-optimized audio processor.
    
    Args:
        video_path: Path to the video file
        output_dir: Base directory for output files (optional)
        chunk_duration: Duration of each audio chunk in seconds
        overlap: Overlap between chunks in seconds
        max_workers: Maximum number of parallel workers for audio segmentation
        
    Returns:
        Dictionary with processing results
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"Starting CUDA audio processing for {video_path}")
    
    # Setup device for GPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        # Log GPU memory before processing
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
        free_mem = total_mem - allocated_mem
        logger.info(f"GPU Memory - Total: {total_mem:.2f}GB, Allocated: {allocated_mem:.2f}GB, Free: {free_mem:.2f}GB")
    else:
        logger.warning(f"CUDA not available, using device: {device.type}")
    
    # Initialize the audio processor with CUDA optimizations
    audio_processor = AudioProcessorCUDA(
        model_name=config.WHISPER_MODEL,
        device=device,
        use_mixed_precision=config.USE_MIXED_PRECISION,
        batch_size=config.GPU_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        use_pinned_memory=config.USE_PINNED_MEMORY
    )
    
    # Process the video
    start_time = time.time()
    
    result = audio_processor.process_video_cuda(
        video_path=video_path,
        output_dir=output_dir or config.ROOT_DIR,
        chunk_duration=chunk_duration,
        overlap=overlap,
        max_workers=max_workers
    )
    
    processing_time = time.time() - start_time
    
    # Log results
    logger.info(f"Audio processing completed in {processing_time:.2f}s")
    logger.info(f"Processed {len(result['segments'])} audio segments")
    logger.info(f"Generated {len(result['transcriptions'])} transcriptions")
    logger.info(f"Word count: {len(result['merged_transcription']['words'])}")
    
    # Print summary of the transcription
    text = result['merged_transcription']['text']
    logger.info(f"Transcription summary ({len(text.split())} words): {text[:200]}...")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with the CUDA-optimized audio processor")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--chunk-duration", type=float, default=30.0, help="Duration of each audio chunk in seconds")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap between chunks in seconds")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers for audio segmentation")
    parser.add_argument("--output-dir", help="Base directory for output files")
    args = parser.parse_args()
    
    ensure_directories()
    process_video(
        args.video_path, 
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        max_workers=args.max_workers
    ) 