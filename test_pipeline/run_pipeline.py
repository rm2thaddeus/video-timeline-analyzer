"""
📌 Purpose: Script to run the video processing pipeline
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Initialize directories and run the pipeline on a sample video
📂 Expected File Path: test_pipeline/run_pipeline.py
🧠 Reasoning: Simple entry point for testing the pipeline
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path to allow importing the pipeline module
sys.path.append(str(Path(__file__).parent))

from pipeline import ensure_directories, process_video
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "pipeline_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("video_pipeline.run")

def main():
    # Ensure all directories exist
    ensure_directories()
    
    # Path to the sample video
    video_path = r"C:\Users\aitor\Downloads\videotest.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Sample video not found at {video_path}")
        print(f"Error: Sample video not found at {video_path}")
        return
    
    logger.info(f"Starting pipeline with video: {video_path}")
    print(f"Starting pipeline with video: {video_path}")
    
    try:
        # Process the video
        result = process_video(video_path)
        
        print("\nPipeline completed successfully!")
        print(f"Metadata saved to: {config.METADATA_DIR}")
        print(f"Scene boundaries saved to: {config.SCENES_DIR}")
        print(f"Audio chunks saved to: {config.AUDIO_CHUNKS_DIR}")
        print(f"Transcripts saved to: {config.TRANSCRIPTS_DIR}")
        print(f"Frames saved to: {config.FRAMES_DIR}")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        print(f"Error running pipeline: {e}")

if __name__ == "__main__":
    main() 