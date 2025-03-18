"""
📌 Purpose: Test script for the optimized audio processor with CUDA acceleration
🔄 Latest Changes: Fixed device properties handling for better compatibility
⚙️ Key Logic: Tests audio extraction, transcription, and SRT generation with GPU
📂 Expected File Path: test_pipeline/CUDA/test_audio_processor.py
🧠 Reasoning: Validates the optimized audio processor with real-world video input
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the optimized audio processor
from CUDA.audio_processor_cuda import AudioProcessorCUDA
from utils.gpu_utils import setup_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_audio_processor")

def test_audio_processor(video_path, output_dir=None, model_name="base", gpu_memory_fraction=0.5):
    """
    Test the optimized audio processor with a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output files (optional)
        model_name: Whisper model size (tiny, base, small, medium, large)
        gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Testing audio processor with video: {video_path}")
    logger.info(f"Using Whisper model: {model_name}")
    
    # Setup device
    device, device_props = setup_device()
    if device.type == "cuda":
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        # Handle both dictionary and object forms of device_props
        if isinstance(device_props, dict):
            if 'total_memory' in device_props:
                logger.info(f"GPU Memory: {device_props['total_memory'] / (1024**3):.2f} GB")
        else:
            try:
                logger.info(f"GPU Memory: {device_props.total_memory / (1024**3):.2f} GB")
            except AttributeError:
                logger.info("GPU memory information not available")
        logger.info(f"Setting GPU memory fraction to {gpu_memory_fraction}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Initialize audio processor with optimized settings
    processor = AudioProcessorCUDA(
        model_name=model_name,
        device=device,
        use_mixed_precision=True,
        batch_size=8,  # Smaller batch size for stability
        num_workers=4,
        use_pinned_memory=True,
        gpu_memory_fraction=gpu_memory_fraction
    )
    
    try:
        # Process video
        result = processor.process_video_cuda(
            video_path=str(video_path),
            output_dir=output_dir,
            chunk_duration=30.0,  # 30 second chunks
            overlap=1.0,  # 1 second overlap
            max_workers=4
        )
        
        # Print results
        logger.info("Audio processing completed successfully!")
        logger.info(f"Processing time: {result['processing_time_seconds']:.2f} seconds")
        logger.info(f"SRT file: {result['srt_path']}")
        logger.info(f"JSON file: {result['json_path']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during audio processing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the optimized audio processor")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output_dir", help="Directory to save output files")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size")
    parser.add_argument("--gpu_memory", type=float, default=0.5, 
                        help="Fraction of GPU memory to use (0.0-1.0)")
    
    args = parser.parse_args()
    
    test_audio_processor(
        video_path=args.video_path,
        output_dir=args.output_dir,
        model_name=args.model,
        gpu_memory_fraction=args.gpu_memory
    ) 