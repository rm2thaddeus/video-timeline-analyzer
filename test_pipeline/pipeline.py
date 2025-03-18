"""
📌 Purpose: Main pipeline script for video processing
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Orchestrates the entire video processing workflow
📂 Expected File Path: test_pipeline/pipeline.py
🧠 Reasoning: Centralized entry point for the video processing pipeline
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import torch
from datetime import datetime

# Import pipeline components
from processors.metadata_extractor import extract_metadata
from processors.scene_detector import SceneDetector
from processors.audio_processor import AudioProcessor
from processors.frame_processor import FrameProcessor
from utils.gpu_utils import setup_device
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("video_pipeline")

def ensure_directories():
    """Create all necessary directories for the pipeline."""
    directories = [
        config.METADATA_DIR,
        config.SCENES_DIR,
        config.AUDIO_CHUNKS_DIR,
        config.TRANSCRIPTS_DIR,
        config.FRAMES_DIR,
        config.EMBEDDINGS_DIR,
        config.LOGS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def process_video(video_path, output_dir=None):
    """
    Process a video through the entire pipeline.
    
    Args:
        video_path: Path to the video file
        output_dir: Base directory for output files (optional)
    
    Returns:
        Dictionary with processing results
    """
    video_path = Path(video_path)
    video_id = video_path.stem
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"Starting video processing pipeline for {video_path}")
    
    # Setup device for GPU processing
    device, device_props = setup_device()
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.info(f"Using device: {device.type}")
    
    # Step 1: Extract metadata
    logger.info("Step 1: Extracting video metadata")
    metadata = extract_metadata(video_path, config.METADATA_DIR)
    logger.info(f"Video duration: {metadata['duration_seconds']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")
    
    # Step 2: Detect scenes
    logger.info("Step 2: Detecting scenes")
    scene_detector = SceneDetector(threshold=config.SCENE_THRESHOLD)
    scenes = scene_detector.detect_scenes(video_path, config.SCENES_DIR)
    logger.info(f"Detected {len(scenes)} scenes")
    
    # Step 3: Process audio and transcribe
    logger.info("Step 3: Processing audio and transcribing")
    audio_processor = AudioProcessor(model_name=config.WHISPER_MODEL, device=device)
    audio_result = audio_processor.extract_audio(video_path, config.AUDIO_CHUNKS_DIR)
    logger.info(f"Extracted audio to {audio_result}")
    
    # Step 4: Extract and embed frames
    logger.info("Step 4: Extracting and embedding frames")
    frame_processor = FrameProcessor(model_name=config.CLIP_MODEL, device=device)
    frames = frame_processor.extract_frames(
        video_path, 
        scenes, 
        config.FRAMES_DIR,
        frames_per_scene=3,
        max_dimension=config.MAX_FRAME_DIMENSION
    )
    logger.info(f"Extracted {len(frames)} frames")
    
    # Compile results
    result = {
        "video_id": video_id,
        "metadata": metadata,
        "scenes": scenes,
        "frames": frames,
        "audio": audio_result
    }
    
    # Save pipeline result
    result_file = config.ROOT_DIR / f"{video_id}_pipeline_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Pipeline completed successfully. Results saved to {result_file}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument("video_path", help="Path to the video file to process")
    args = parser.parse_args()
    
    ensure_directories()
    process_video(args.video_path)

if __name__ == "__main__":
    main() 