"""
📌 Purpose: Simple wrapper script to run the video pipeline
🔄 Latest Changes: Updated imports and configuration handling
⚙️ Key Logic: Import and execute the pipeline with proper parameters
📂 Expected File Path: run_video_pipeline.py
🧠 Reasoning: Simplify running the pipeline to avoid PowerShell issues
"""

import os
import sys
import logging
from pathlib import Path
import time

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pipeline_runner")

# Set video path
VIDEO_PATH = r"C:\Users\aitor\Downloads\videotest.mp4"

def main():
    """Run the video pipeline with optimal settings."""
    logger.info(f"Running pipeline for video: {VIDEO_PATH}")
    
    # Ensure that all required directories exist
    from test_pipeline.CUDA.run_pipeline import ensure_directories
    ensure_directories()
    
    # Create output directory
    output_dir = os.path.join("output", Path(VIDEO_PATH).stem)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import the pipeline function
        from test_pipeline.CUDA.pipeline import process_video
        
        # Process video with optimal settings for 6GB GPU
        logger.info(f"Starting video processing with output to: {output_dir}")
        start_time = time.time()
        
        result = process_video(
            video_path=VIDEO_PATH,
            output_dir=output_dir,
            frames_per_scene=3,
            frame_batch_size=8,
            whisper_model="small",
            clip_model="ViT-B/32",
            skip_existing=False,
            force_cpu=False,
            max_frame_dimension=720,
            verbose=True,
            gpu_memory_fraction=0.8,
            optimize_memory=True,
            use_mixed_precision=True,
            audio_batch_size=4,
            min_scene_length=0.5,
            max_frames_in_memory=100
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log results
        logger.info(f"Processing complete in {processing_time:.2f} seconds!")
        logger.info(f"Video ID: {result.get('video_id', 'unknown')}")
        logger.info(f"Number of scenes: {len(result.get('scenes', []))}")
        logger.info(f"Number of frames: {len(result.get('frames', []))}")
        
        # Log where to find the unified dataframe
        metadata_file_path = result.get('metadata_file_path', None)
        if metadata_file_path:
            logger.info(f"Unified metadata dataframe saved to: {metadata_file_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 