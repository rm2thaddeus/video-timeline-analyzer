"""
📌 Purpose: Test script to run the CUDA-accelerated video processing pipeline
🔄 Latest Changes: Added command-line interface and error handling
⚙️ Key Logic: Configures and executes the optimized pipeline with GPU acceleration
📂 Expected File Path: test_pipeline/CUDA/run_pipeline.py
🧠 Reasoning: Provides an easy interface to test and run the optimized video processing pipeline
"""

import os
import sys
import logging
import argparse
import traceback
import torch
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the pipeline
from CUDA.pipeline import process_video, ensure_directories
import CUDA.cuda_config as config

# Configure more detailed logging with colored output if available
try:
    import coloredlogs
    coloredlogs.install(
        level=logging.INFO,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        isatty=True
    )
    has_colored_logs = True
except ImportError:
    # Configure standard logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    has_colored_logs = False

logger = logging.getLogger("run_pipeline")

# Ensure all related loggers are properly configured
for logger_name in ["run_pipeline", "video_pipeline_cuda", "scene_detector_cuda"]:
    log = logging.getLogger(logger_name)
    
    # Make sure we have a console handler
    has_console_handler = False
    for handler in log.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
            has_console_handler = True
            break
    
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        log.addHandler(console_handler)
    
    # Ensure logger passes through messages
    log.setLevel(logging.INFO)
    log.propagate = True

def print_cuda_info():
    """Print CUDA and GPU information"""
    if torch.cuda.is_available():
        logger.info(f"[CUDA] Available: Yes")
        logger.info(f"[CUDA] Version: {torch.version.cuda}")
        logger.info(f"[CUDA] GPU count: {torch.cuda.device_count()}")
        
        # Get info for each device
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"[DEVICE {i}] {device_properties.name}")
            logger.info(f"  - Total memory: {device_properties.total_memory / (1024**3):.2f} GB")
            logger.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
    else:
        logger.info("[CUDA] Available: No")
        logger.info("[INFO] Running in CPU mode")

def setup_configuration(args):
    """Apply command-line arguments to configuration"""
    logger.info("[CONFIG] Applying configuration settings...")
    
    if args.gpu_memory_fraction:
        config.GPU_MEMORY_FRACTION = args.gpu_memory_fraction
        logger.info(f"[CONFIG] Set GPU memory fraction to {config.GPU_MEMORY_FRACTION}")
    
    if args.batch_size:
        config.GPU_BATCH_SIZE = args.batch_size
        logger.info(f"[CONFIG] Set GPU batch size to {config.GPU_BATCH_SIZE}")
    
    if args.sequential:
        config.PARALLEL_PROCESSING = False
        logger.info("[CONFIG] Disabled parallel processing - running sequential mode")
    
    if args.mixed_precision is not None:
        config.USE_MIXED_PRECISION = args.mixed_precision
        logger.info(f"[CONFIG] Set mixed precision to {config.USE_MIXED_PRECISION}")
    
    if args.scene_threshold:
        config.SCENE_THRESHOLD = args.scene_threshold
        logger.info(f"[CONFIG] Set scene threshold to {config.SCENE_THRESHOLD}")
    
    if args.whisper_model:
        config.WHISPER_MODEL = args.whisper_model
        logger.info(f"[CONFIG] Set Whisper model to {config.WHISPER_MODEL}")
    
    if args.scene_batch_size:
        config.SCENE_BATCH_SIZE = args.scene_batch_size
        logger.info(f"[CONFIG] Set scene batch size to {config.SCENE_BATCH_SIZE}")

def main():
    parser = argparse.ArgumentParser(description="Run the CUDA-accelerated video processing pipeline")
    
    # Required arguments
    parser.add_argument("video_path", help="Path to the video file to process", nargs="?")
    
    # Optional arguments
    parser.add_argument("--output_dir", help="Directory to save output files (default: auto)")
    parser.add_argument("--gpu_memory_fraction", type=float, help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--batch_size", type=int, help="Batch size for GPU processing")
    parser.add_argument("--sequential", action="store_true", help="Disable parallel processing")
    parser.add_argument("--whisper_model", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size")
    parser.add_argument("--mixed_precision", type=bool, help="Use mixed precision (FP16)")
    parser.add_argument("--scene_threshold", type=float, help="Threshold for scene detection")
    parser.add_argument("--scene_batch_size", type=int, help="Batch size for scene detection")
    parser.add_argument("--info", action="store_true", help="Print CUDA and GPU information")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        for log_name in ["video_pipeline_cuda", "scene_detector_cuda"]:
            logging.getLogger(log_name).setLevel(logging.DEBUG)
        logger.info("[INFO] Verbose logging enabled")
    
    # Print CUDA info if requested
    if args.info:
        print_cuda_info()
        return
    
    # Check if video path is provided
    if not args.video_path:
        parser.print_help()
        logger.error("[ERROR] No video path provided. Please specify a video file to process.")
        sys.exit(1)
        
    # Validate input path
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
    
    # Apply configuration changes
    setup_configuration(args)
    
    # Ensure output directories exist
    ensure_directories()
    
    try:
        logger.info(f"[START] Starting video processing for {video_path}")
        logger.info("[CONFIG] Pipeline configuration:")
        logger.info(f"  - GPU memory fraction: {config.GPU_MEMORY_FRACTION}")
        logger.info(f"  - GPU batch size: {config.GPU_BATCH_SIZE}")
        logger.info(f"  - Parallel processing: {config.PARALLEL_PROCESSING}")
        logger.info(f"  - Mixed precision: {config.USE_MIXED_PRECISION}")
        logger.info(f"  - Scene threshold: {config.SCENE_THRESHOLD}")
        logger.info(f"  - Whisper model: {config.WHISPER_MODEL}")
        logger.info(f"  - Scene batch size: {config.SCENE_BATCH_SIZE}")
        
        # Run the pipeline
        result = process_video(video_path, args.output_dir)
        
        logger.info(f"[SUCCESS] Processing completed successfully!")
        logger.info(f"[OUTPUT] Results saved to {result['audio_processing']['srt_path']}")
        logger.info(f"[SCENES] Number of scenes detected: {len(result['scenes'])}")
        logger.info(f"[FRAMES] Number of frames extracted: {len(result['frames'])}")
        
    except KeyboardInterrupt:
        logger.info("[STOP] Processing stopped by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"[ERROR] Error during processing: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 