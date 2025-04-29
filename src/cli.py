"""
Command-Line Interface for Video Timeline Analyzer.

This module provides a CLI for interacting with the Video Timeline Analyzer.

📌 Purpose: Provide command-line interface for the application
🔄 Latest Changes: Added GPU-accelerated video processing commands
⚙️ Key Logic: Argument parsing, command routing
📂 Expected File Path: src/cli.py
🧠 Reasoning: CLI interface allows for scriptable usage and automation
"""

import argparse
import logging
import os
import sys
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.utils.gpu_utils import detect_gpu, setup_device
from src.models.schema import AnalysisConfig
from src.video_processing.pipeline import VideoProcessingPipeline
from src.video_processing.scene_detection import DetectionMethod
from src.video_processing.frame_extraction import FrameExtractionMethod

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("video_timeline")

def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Video Timeline Analyzer - Generate interactive timelines from videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process a video file to generate timeline metadata"
    )
    process_parser.add_argument(
        "video_path", type=str, help="Path to the video file to process"
    )
    process_parser.add_argument(
        "--output-dir", "-o", type=str, help="Directory to store output files",
        default="output"
    )
    process_parser.add_argument(
        "--scene-method", type=str, choices=["content", "threshold", "hybrid", "custom"],
        default="content", help="Scene detection method"
    )
    process_parser.add_argument(
        "--scene-threshold", type=float, default=30.0,
        help="Threshold for scene change detection"
    )
    process_parser.add_argument(
        "--min-scene-length", type=float, default=1.0,
        help="Minimum scene length in seconds"
    )
    process_parser.add_argument(
        "--frame-method", type=str, 
        choices=["first", "middle", "representative", "uniform", "all"],
        default="representative", help="Frame extraction method"
    )
    process_parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for processing"
    )
    process_parser.add_argument(
        "--use-gpu", action="store_true", default=True,
        help="Use GPU for processing if available"
    )
    process_parser.add_argument(
        "--no-gpu", action="store_false", dest="use_gpu",
        help="Disable GPU processing"
    )
    process_parser.add_argument(
        "--half-precision", action="store_true",
        help="Use half precision (FP16) to reduce memory usage"
    )
    process_parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum number of worker processes"
    )
    process_parser.add_argument(
        "--no-audio", action="store_true", help="Skip audio analysis"
    )
    process_parser.add_argument(
        "--no-visual", action="store_true", help="Skip visual analysis"
    )
    
    # Batch processing command
    batch_parser = subparsers.add_parser(
        "process-batch", help="Process multiple video files"
    )
    batch_parser.add_argument(
        "input_dir", type=str, help="Directory containing video files to process"
    )
    batch_parser.add_argument(
        "--output-dir", "-o", type=str, help="Directory to store output files",
        default="output"
    )
    batch_parser.add_argument(
        "--extensions", type=str, default="mp4,avi,mkv,mov",
        help="Comma-separated list of video file extensions to process"
    )
    batch_parser.add_argument(
        "--scene-method", type=str, choices=["content", "threshold", "hybrid", "custom"],
        default="content", help="Scene detection method"
    )
    batch_parser.add_argument(
        "--scene-threshold", type=float, default=30.0,
        help="Threshold for scene change detection"
    )
    batch_parser.add_argument(
        "--frame-method", type=str, 
        choices=["first", "middle", "representative", "uniform", "all"],
        default="representative", help="Frame extraction method"
    )
    batch_parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for processing"
    )
    batch_parser.add_argument(
        "--use-gpu", action="store_true", default=True,
        help="Use GPU for processing if available"
    )
    batch_parser.add_argument(
        "--no-gpu", action="store_false", dest="use_gpu",
        help="Disable GPU processing"
    )
    batch_parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum number of worker processes"
    )
    
    # Extract scenes command
    scenes_parser = subparsers.add_parser(
        "extract-scenes", help="Extract scenes from a video file"
    )
    scenes_parser.add_argument(
        "video_path", type=str, help="Path to the video file to process"
    )
    scenes_parser.add_argument(
        "--output-dir", "-o", type=str, help="Directory to store output files",
        default="output"
    )
    scenes_parser.add_argument(
        "--method", type=str, choices=["content", "threshold", "hybrid", "custom"],
        default="content", help="Scene detection method"
    )
    scenes_parser.add_argument(
        "--threshold", type=float, default=30.0,
        help="Threshold for scene change detection"
    )
    scenes_parser.add_argument(
        "--min-scene-length", type=float, default=1.0,
        help="Minimum scene length in seconds"
    )
    scenes_parser.add_argument(
        "--extract-frames", action="store_true",
        help="Extract key frames from detected scenes"
    )
    scenes_parser.add_argument(
        "--use-gpu", action="store_true", default=True,
        help="Use GPU for processing if available"
    )
    scenes_parser.add_argument(
        "--no-gpu", action="store_false", dest="use_gpu",
        help="Disable GPU processing"
    )
    
    # Extract frames command
    frames_parser = subparsers.add_parser(
        "extract-frames", help="Extract frames from a video file"
    )
    frames_parser.add_argument(
        "video_path", type=str, help="Path to the video file to process"
    )
    frames_parser.add_argument(
        "--output-dir", "-o", type=str, help="Directory to store output files",
        default="output"
    )
    frames_parser.add_argument(
        "--method", type=str, 
        choices=["first", "middle", "representative", "uniform", "all"],
        default="uniform", help="Frame extraction method"
    )
    frames_parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Interval between frames in seconds (for uniform method)"
    )
    frames_parser.add_argument(
        "--max-frames", type=int, default=1000,
        help="Maximum number of frames to extract"
    )
    frames_parser.add_argument(
        "--use-gpu", action="store_true", default=True,
        help="Use GPU for processing if available"
    )
    frames_parser.add_argument(
        "--no-gpu", action="store_false", dest="use_gpu",
        help="Disable GPU processing"
    )
    frames_parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for processing"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Display information about system setup"
    )
    info_parser.add_argument(
        "--gpu", action="store_true", help="Show detailed GPU information"
    )
    
    # UI command
    ui_parser = subparsers.add_parser(
        "ui", help="Launch the user interface"
    )
    ui_parser.add_argument(
        "--type", choices=["desktop", "web"], default="desktop",
        help="Type of UI to launch"
    )
    ui_parser.add_argument(
        "--port", type=int, default=8000, help="Port for web UI server"
    )
    
    # Version command
    subparsers.add_parser(
        "version", help="Show version information"
    )
    
    return parser

def process_command(args: argparse.Namespace) -> int:
    """
    Handle the 'process' command to analyze a video.
    
    Args:
        args: Command-line arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Processing video: {args.video_path}")
    
    # Check if video file exists
    if not os.path.isfile(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability if requested
    if args.use_gpu:
        gpu_info = detect_gpu()
        if gpu_info["detected"]:
            logger.info(f"Using GPU ({gpu_info['gpu_type']}) for processing")
        else:
            logger.warning("GPU requested but not available, falling back to CPU")
    
    # Create configuration
    config = AnalysisConfig(
        scene_detection_method=args.scene_method,
        scene_threshold=args.scene_threshold,
        min_scene_length=args.min_scene_length,
        extract_key_frames=True,
        key_frames_method=args.frame_method,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        transcribe_audio=not args.no_audio,
        detect_faces=not args.no_visual
    )
    
    # Create and run pipeline
    pipeline = VideoProcessingPipeline(config)
    
    try:
        results = pipeline.process_video(
            video_path=args.video_path,
            output_dir=output_dir,
            save_results=True
        )
        
        if results:
            logger.info(f"Successfully processed video with {len(results.timeline.scenes)} scenes")
            logger.info(f"Results saved to: {output_dir}")
            logger.info(f"Processing time: {results.processing_time:.2f} seconds")
            return 0
        else:
            logger.error("Failed to process video")
            return 1
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return 1

def process_batch_command(args: argparse.Namespace) -> int:
    """
    Handle the 'process-batch' command to analyze multiple videos.
    
    Args:
        args: Command-line arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Processing videos in directory: {args.input_dir}")
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Find video files
    extensions = args.extensions.split(",")
    video_paths = []
    
    for ext in extensions:
        pattern = f"*.{ext.strip()}"
        video_paths.extend(list(Path(args.input_dir).glob(pattern)))
    
    # Convert to strings
    video_paths = [str(path) for path in video_paths]
    
    if not video_paths:
        logger.error(f"No video files found in: {args.input_dir}")
        return 1
    
    logger.info(f"Found {len(video_paths)} video files")
    
    # Check GPU availability if requested
    if args.use_gpu:
        gpu_info = detect_gpu()
        if gpu_info["detected"]:
            logger.info(f"Using GPU ({gpu_info['gpu_type']}) for processing")
        else:
            logger.warning("GPU requested but not available, falling back to CPU")
    
    # Create configuration
    config = AnalysisConfig(
        scene_detection_method=args.scene_method,
        scene_threshold=args.scene_threshold,
        extract_key_frames=True,
        key_frames_method=args.frame_method,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    # Create and run pipeline
    pipeline = VideoProcessingPipeline(config)
    
    try:
        results = pipeline.process_videos_batch(
            video_paths=video_paths,
            output_dir=output_dir,
            save_results=True
        )
        
        # Log results
        success_count = sum(1 for r in results.values() if r is not None)
        fail_count = len(results) - success_count
        
        logger.info(f"Batch processing complete: {success_count} succeeded, {fail_count} failed")
        logger.info(f"Results saved to: {output_dir}")
        
        return 0 if fail_count == 0 else 1
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return 1

def extract_scenes_command(args: argparse.Namespace) -> int:
    """
    Handle the 'extract-scenes' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Extracting scenes from: {args.video_path}")
    
    # Check if video file exists
    if not os.path.isfile(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability if requested
    if args.use_gpu:
        gpu_info = detect_gpu()
        if gpu_info["detected"]:
            logger.info(f"Using GPU ({gpu_info['gpu_type']}) for processing")
        else:
            logger.warning("GPU requested but not available, falling back to CPU")
    
    # Import the scene detection module
    from src.video_processing.scene_detection import GPUAcceleratedSceneDetector
    
    # Create scene detector
    detector = GPUAcceleratedSceneDetector(
        detection_method=args.method,
        threshold=args.threshold,
        min_scene_length=args.min_scene_length,
        use_gpu=args.use_gpu
    )
    
    try:
        # Detect scenes
        scenes = detector.detect_scenes(
            args.video_path,
            stats_file=os.path.join(output_dir, "scene_stats.csv"),
            extract_frames=args.extract_frames,
            output_dir=output_dir if args.extract_frames else None
        )
        
        # Save scenes as JSON
        scenes_data = [scene.dict() for scene in scenes]
        
        with open(os.path.join(output_dir, "scenes.json"), "w") as f:
            json.dump(scenes_data, f, indent=2, default=str)
        
        logger.info(f"Detected {len(scenes)} scenes")
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error extracting scenes: {str(e)}")
        return 1

def extract_frames_command(args: argparse.Namespace) -> int:
    """
    Handle the 'extract-frames' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        int: Exit code
    """
    logger.info(f"Extracting frames from: {args.video_path}")
    
    # Check if video file exists
    if not os.path.isfile(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return 1
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability if requested
    if args.use_gpu:
        gpu_info = detect_gpu()
        if gpu_info["detected"]:
            logger.info(f"Using GPU ({gpu_info['gpu_type']}) for processing")
        else:
            logger.warning("GPU requested but not available, falling back to CPU")
    
    # Import the frame extraction module
    from src.video_processing.frame_extraction import FrameExtractor
    
    # Create frame extractor
    extractor = FrameExtractor(
        batch_size=args.batch_size,
        use_gpu=args.use_gpu
    )
    
    try:
        # Extract frames
        if args.method == "uniform":
            frames = extractor.extract_uniform_frames(
                args.video_path,
                output_dir,
                interval=args.interval,
                max_frames=args.max_frames
            )
        else:
            # For other methods, we need scenes first
            from src.video_processing.scene_detection import GPUAcceleratedSceneDetector
            
            # Create a temporary directory for scene detection
            scenes_dir = os.path.join(output_dir, "scenes")
            os.makedirs(scenes_dir, exist_ok=True)
            
            # Detect scenes
            detector = GPUAcceleratedSceneDetector(
                detection_method="content",
                use_gpu=args.use_gpu
            )
            
            scenes = detector.detect_scenes(
                args.video_path,
                extract_frames=False
            )
            
            # Extract frames from scenes
            frames_dict = extractor.extract_frames_from_scenes(
                args.video_path,
                scenes,
                output_dir,
                method=args.method,
                max_frames_per_scene=min(5, max(1, args.max_frames // max(1, len(scenes))))
            )
            
            # Flatten frames list
            frames = []
            for scene_frames in frames_dict.values():
                frames.extend(scene_frames)
        
        logger.info(f"Extracted {len(frames)} frames")
        logger.info(f"Frames saved to: {output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return 1

def info_command(args: argparse.Namespace) -> int:
    """
    Handle the 'info' command to display system information.
    
    Args:
        args: Command-line arguments
        
    Returns:
        int: Exit code
    """
    from src import __version__
    
    print(f"Video Timeline Analyzer v{__version__}")
    print(f"Python: {sys.version}")
    
    # GPU information
    gpu_info = detect_gpu()
    if gpu_info["detected"]:
        print(f"GPU: {gpu_info['gpu_type']} ({gpu_info['device_count']} device(s))")
    else:
        print("GPU: Not available")
    
    if args.gpu and gpu_info["detected"]:
        if gpu_info['gpu_type'] == 'cuda':
            for i in range(gpu_info['device_count']):
                import torch
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    Memory: {memory:.2f} GB")
                
                # Show memory usage
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"    Memory Usage: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    return 0

def ui_command(args: argparse.Namespace) -> int:
    """
    Handle the 'ui' command to launch the user interface.
    
    Args:
        args: Command-line arguments
        
    Returns:
        int: Exit code
    """
    ui_type = args.type
    
    if ui_type == "desktop":
        logger.info("Launching desktop UI")
        # TODO: Implement desktop UI launch
        logger.info("Desktop UI not yet implemented")
    elif ui_type == "web":
        port = args.port
        logger.info(f"Launching web UI on port {port}")
        # TODO: Implement web UI launch
        logger.info("Web UI not yet implemented")
    
    return 0

def version_command(_: argparse.Namespace) -> int:
    """
    Handle the 'version' command to display version information.
    
    Args:
        _: Command-line arguments (unused)
        
    Returns:
        int: Exit code
    """
    from src import __version__
    print(f"Video Timeline Analyzer v{__version__}")
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code
    """
    parser = setup_parser()
    args = parser.parse_args(argv)
    
    # Handle case where no command is provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    if args.command == "process":
        return process_command(args)
    elif args.command == "process-batch":
        return process_batch_command(args)
    elif args.command == "extract-scenes":
        return extract_scenes_command(args)
    elif args.command == "extract-frames":
        return extract_frames_command(args)
    elif args.command == "info":
        return info_command(args)
    elif args.command == "ui":
        return ui_command(args)
    elif args.command == "version":
        return version_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())