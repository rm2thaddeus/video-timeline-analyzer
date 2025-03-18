"""
📌 Purpose: Main pipeline script for video processing with CUDA optimization
🔄 Latest Changes: Enhanced memory management with batch processing and better error handling
⚙️ Key Logic: Orchestrates the entire video processing workflow with CUDA acceleration
📂 Expected File Path: test_pipeline/CUDA/pipeline.py
🧠 Reasoning: Centralized entry point for the CUDA-accelerated video processing pipeline
"""

import os
import sys
import gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging
import argparse
import concurrent.futures
import traceback
from pathlib import Path
import torch
import torch.multiprocessing as mp
from datetime import datetime
from contextlib import nullcontext
import time
from typing import Dict, Any, Optional

# Import pipeline components with fallback for module location
try:
    from processors.metadata_extractor import extract_metadata
except ModuleNotFoundError:
    from video_processing.metadata_extractor import extract_metadata

# Import our metadata dataframe manager
try:
    from processors.metadata_dataframe import MetadataManager
except ModuleNotFoundError:
    from video_processing.metadata_dataframe import MetadataManager

# Use our optimized scene detector
from CUDA.scene_detector import SceneDetector

# Import our optimized audio processor
from CUDA.audio_processor_cuda import AudioProcessorCUDA

try:
    from processors.frame_processor import FrameProcessor
except ModuleNotFoundError:
    from video_processing.frame_processor import FrameProcessor

from utils.gpu_utils import setup_device

# Import config with correct relative path
try:
    from CUDA import cuda_config as config
except ModuleNotFoundError:
    from . import cuda_config as config

# Setup logging with console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / f"cuda_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("video_pipeline_cuda")

# Ensure all logs go to console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add console handler to all relevant loggers
for logger_name in ["video_pipeline_cuda", "scene_detector_cuda"]:
    log = logging.getLogger(logger_name)
    log.addHandler(console_handler)
    log.setLevel(logging.INFO)

def _adapt_scenes_for_frame_processor(scenes):
    """
    Adapt scene data format to match what's expected by the frame processor.
    
    Args:
        scenes: List of scenes in our format
        
    Returns:
        List of scenes in the format expected by frame_processor
    """
    logger.debug(f"Adapting {len(scenes)} scenes for frame processor")
    adapted_scenes = []
    
    for scene in scenes:
        # Create a copy of the scene with the required keys
        adapted_scene = {
            "scene_id": scene["id"],  # Convert 'id' to 'scene_id'
            "start_frame": scene["start_frame"],
            "end_frame": scene["end_frame"],
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration_frames": scene["duration_frames"],
            "duration_seconds": scene["duration_seconds"]
        }
        
        # Add any additional keys that might be expected
        if "resolution" in scene:
            adapted_scene["resolution"] = scene["resolution"]
            
        adapted_scenes.append(adapted_scene)
    
    return adapted_scenes

def ensure_directories():
    """Create all necessary directories for the pipeline."""
    ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    directories = [
        getattr(config, 'METADATA_DIR', ROOT_DIR / "metadata"),
        getattr(config, 'SCENES_DIR', ROOT_DIR / "scenes"),
        getattr(config, 'AUDIO_CHUNKS_DIR', ROOT_DIR / "audio_chunks"),
        getattr(config, 'TRANSCRIPTS_DIR', ROOT_DIR / "transcripts"),
        getattr(config, 'FRAMES_DIR', ROOT_DIR / "frames"),
        getattr(config, 'EMBEDDINGS_DIR', ROOT_DIR / "embeddings"),
        getattr(config, 'LOGS_DIR', ROOT_DIR / "logs"),
        getattr(config, 'OUTPUT_DIR', ROOT_DIR / "output")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def clean_gpu_memory():
    """Aggressively clean GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[CLEAN] GPU memory cleaned")

def setup_cuda_optimizations(device):
    """
    Set up CUDA optimizations based on configuration.
    
    Args:
        device: PyTorch device
    """
    if device.type == "cuda":
        # Extract device index
        device_idx = 0 if device.index is None else device.index
        
        # Set CUDA optimization flags
        if config.USE_CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
        
        # Deterministic mode can slow things down, so we disable it for performance
        torch.backends.cudnn.deterministic = False
        
        # Enable TF32 on Ampere GPUs for better performance
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        if config.OPTIMIZE_MEMORY_USAGE:
            # More aggressive memory allocation
            clean_gpu_memory()
            # Set memory fraction to use
            torch.cuda.set_per_process_memory_fraction(config.GPU_MEMORY_FRACTION, device_idx)
        
        logger.info(f"[CUDA] Optimizations enabled:")
        logger.info(f"  - cuDNN benchmark: {config.USE_CUDNN_BENCHMARK}")
        logger.info(f"  - Mixed precision: {config.USE_MIXED_PRECISION}")
        logger.info(f"  - Pinned memory: {config.USE_PINNED_MEMORY}")
        logger.info(f"  - Memory fraction: {config.GPU_MEMORY_FRACTION}")
        logger.info(f"  - TF32 acceleration: {hasattr(torch.backends.cuda, 'matmul')}")
    else:
        logger.warning("[WARN] CUDA not available, optimizations not applied")

def _process_audio_direct(video_path, video_id, model_name="tiny"):
    """
    Process audio directly without chunking to avoid tensor dimension issues.
    
    Args:
        video_path: Path to the video file
        video_id: Video ID/name
        model_name: Whisper model name
        
    Returns:
        Dictionary with processing results
    """
    # Initialize result
    audio_result = {
        "video_id": video_id,
        "srt_path": "",
        "json_path": "",
        "processing_time_seconds": 0
    }
    
    logger.info(f"[AUDIO] Processing audio for {video_id} with Whisper {model_name} model")
    
    # Extract audio first
    audio_file = config.AUDIO_CHUNKS_DIR / f"{video_id}.wav"
    try:
        # Import subprocess here to avoid circular import
        import subprocess
        
        # Simple FFmpeg command to extract audio
        command = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian format
            "-ar", "16000",  # 16 kHz sample rate (Whisper's expected rate)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            str(audio_file)
        ]
        
        logger.info(f"[AUDIO] Extracting audio with FFmpeg...")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"[AUDIO] Audio extracted to {audio_file}")
        
        # Create transcription using OpenAI whisper directly
        import whisper
        
        start_time = time.time()
        logger.info(f"[AUDIO] Loading Whisper {model_name} model...")
        model = whisper.load_model(model_name, device="cpu")
        
        # Transcribe audio
        logger.info(f"[AUDIO] Transcribing audio with Whisper {model_name} model...")
        result = model.transcribe(str(audio_file))
        
        # Save SRT file
        srt_file = config.TRANSCRIPTS_DIR / f"{video_id}.srt"
        with open(srt_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"]):
                # Format: index, timestamp, text
                f.write(f"{i+1}\n")
                start = segment["start"]
                end = segment["end"]
                # Format timestamp as HH:MM:SS,mmm
                start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
                end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{segment['text']}\n\n")
        
        # Save JSON file
        json_file = config.TRANSCRIPTS_DIR / f"{video_id}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        processing_time = time.time() - start_time
        logger.info(f"[AUDIO] Transcription completed in {processing_time:.2f} seconds")
        logger.info(f"[AUDIO] SRT file saved to {srt_file}")
        logger.info(f"[AUDIO] JSON file saved to {json_file}")
        
        # Update audio result
        audio_result = {
            "video_id": video_id,
            "srt_path": str(srt_file),
            "json_path": str(json_file),
            "processing_time_seconds": processing_time
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Error in direct audio processing: {e}")
        logger.error(traceback.format_exc())
    
    return audio_result

def process_video(
    video_path: str,
    output_dir: Optional[str] = None,
    frames_per_scene: int = 3,
    frame_batch_size: int = 8,
    whisper_model: str = "small",
    clip_model: str = "ViT-B/32",
    skip_existing: bool = False,
    force_cpu: bool = False,
    max_frame_dimension: int = 720,
    verbose: bool = False,
    gpu_memory_fraction: float = 0.8,
    optimize_memory: bool = True,
    use_mixed_precision: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a video file with the pipeline.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save pipeline outputs (if None, a default is used)
        frames_per_scene: Number of frames to extract per scene
        frame_batch_size: Batch size for frame processing
        whisper_model: Whisper model size for transcription (tiny, base, small, medium, large)
        clip_model: CLIP model for frame embeddings (ViT-B/32, ViT-B/16, etc.)
        skip_existing: Skip processing if output files already exist
        force_cpu: Force using CPU even if GPU is available
        max_frame_dimension: Maximum dimension for extracting frames
        verbose: Enable verbose logging
        gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        optimize_memory: Use memory-efficient processing algorithms
        use_mixed_precision: Use mixed precision (FP16) for faster processing
        **kwargs: Additional arguments passed to components
        
    Returns:
        Dictionary with processing results
    """
    # Start performance tracking
    start_time = time.time()
    performance_metrics = {
        "start_time": datetime.now().isoformat(),
        "video_path": video_path,
    }
    
    # Setup device
    device, is_cuda_available = setup_device(force_cpu)
    logger.info(f"Using device: {device} (CUDA available: {is_cuda_available})")
    
    if device.type == "cuda":
        setup_cuda_optimizations(device)
    
    # Ensure directories exist
    ensure_directories()
    
    # Default output directory if not provided
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Get video file details
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate a unique identifier for this video
    video_id = video_path.stem
    logger.info(f"Processing video: {video_path} (ID: {video_id})")
    
    # Extract metadata for the video
    metadata = extract_metadata(str(video_path), str(config.METADATA_DIR))
    performance_metrics["metadata_extraction_time"] = time.time() - start_time
    logger.info(f"Video duration: {metadata.get('duration_seconds', 'Unknown')} seconds")
    
    # Create result dict
    result = {
        "video_id": video_id,
        "metadata": metadata,
        "scenes": None,
        "frames": None,
        "transcript": None,
        "unified_dataframe": None
    }
    
    # Detect scenes
    scene_start_time = time.time()
    scene_detector = SceneDetector(
        device=device,
        batch_size=int(config.SCENE_BATCH_SIZE),
        checkpoint_interval=int(config.SCENE_CHECKPOINT_INTERVAL)
    )
    
    # Detect scenes with progress
    logger.info("Detecting scenes...")
    scenes = scene_detector.detect_scenes(
        str(video_path),
        output_dir=str(config.SCENES_DIR)
    )
    
    # Add performance metrics
    scene_time = time.time() - scene_start_time
    performance_metrics["scene_detection_time"] = scene_time
    
    # Store scene information
    result["scenes"] = scenes
    logger.info(f"Detected {len(scenes)} scenes in {scene_time:.2f} seconds")
    
    # Clean GPU memory before next step
    clean_gpu_memory()
    
    # Extract and process audio with enhanced parallelism
    audio_start_time = time.time()
    
    # Initialize the audio processor
    audio_processor = AudioProcessorCUDA(
        model_name=whisper_model,
        device=device,
        batch_size=kwargs.get("audio_batch_size", 4),
        use_mixed_precision=use_mixed_precision,
        gpu_memory_fraction=gpu_memory_fraction * 0.8,  # Reserve some memory for other tasks
        num_workers=min(4, os.cpu_count() or 2)
    )
    
    # Process audio
    logger.info("Processing audio and transcribing...")
    transcript_data = audio_processor.process_video_cuda(
        str(video_path),
        output_dir=str(config.TRANSCRIPTS_DIR),
        chunk_duration=config.AUDIO_CHUNK_DURATION,
        overlap=config.AUDIO_CHUNK_OVERLAP,
        max_workers=min(kwargs.get("audio_batch_size", 4), config.MAX_PARALLEL_TASKS)
    )
    
    # Add performance metrics
    audio_time = time.time() - audio_start_time
    performance_metrics["audio_processing_time"] = audio_time
    
    # Store transcript data
    result["transcript"] = transcript_data
    logger.info(f"Audio processed in {audio_time:.2f} seconds")
    
    # Clean GPU memory again
    clean_gpu_memory()
    
    # Extract frames and generate embeddings
    frames_start_time = time.time()
    
    # Adjust scene data format for frame processor
    adapted_scenes = _adapt_scenes_for_frame_processor(scenes)
    
    # Initialize frame processor with CLIP model
    frame_processor = FrameProcessor(
        model_name=clip_model,
        device=device,
        memory_efficient=optimize_memory,
        max_frames_in_memory=kwargs.get("max_frames_in_memory", 100)
    )
    
    # Process frames
    logger.info("Extracting and embedding frames...")
    frame_results = frame_processor.process_video(
        str(video_path),
        adapted_scenes,
        frames_dir=str(config.FRAMES_DIR),
        embeddings_dir=str(config.EMBEDDINGS_DIR),
        frames_per_scene=frames_per_scene,
        max_dimension=max_frame_dimension
    )
    
    # Add performance metrics
    frames_time = time.time() - frames_start_time
    performance_metrics["frame_processing_time"] = frames_time
    
    # Store frame results
    result["frames"] = frame_results
    logger.info(f"Processed {len(frame_results)} frames in {frames_time:.2f} seconds")
    
    # Clean GPU memory one more time
    clean_gpu_memory()
    
    # Create unified metadata dataframe
    metadata_start_time = time.time()
    logger.info("Creating unified metadata dataframe...")
    
    # Initialize metadata manager
    metadata_manager = MetadataManager()
    
    # Create dataframe from various data sources
    unified_df = metadata_manager.create_dataframe(
        video_id=video_id,
        frames_data=frame_results,
        transcript_data=transcript_data,
        scene_data=scenes,
        video_metadata=metadata
    )
    
    # Add performance metrics
    metadata_manager.add_performance_metrics(performance_metrics)
    
    # Save the dataframe (without embedding arrays to save space)
    metadata_file_path = metadata_manager.save_dataframe(
        output_dir=str(config.METADATA_DIR),
        include_embeddings=False
    )
    
    # Calculate metadata creation time
    metadata_time = time.time() - metadata_start_time
    performance_metrics["metadata_dataframe_time"] = metadata_time
    logger.info(f"Created metadata dataframe with {len(unified_df)} entries in {metadata_time:.2f} seconds")
    
    # Store dataframe in result
    result["unified_dataframe"] = unified_df
    result["metadata_file_path"] = metadata_file_path
    
    # Final performance metrics
    total_time = time.time() - start_time
    performance_metrics["total_processing_time"] = total_time
    result["performance_metrics"] = performance_metrics
    
    logger.info(f"Video processing completed in {total_time:.2f} seconds")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with the CUDA pipeline")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output_dir", help="Directory to save output files")
    parser.add_argument("--gpu_memory", type=float, default=0.4, 
                        help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--sequential", action="store_true",
                        help="Disable parallel processing and run sequentially")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size")
    
    args = parser.parse_args()
    
    # Update configuration
    if args.gpu_memory:
        config.GPU_MEMORY_FRACTION = args.gpu_memory
    
    if args.sequential:
        config.PARALLEL_PROCESSING = False
    
    if args.model:
        config.WHISPER_MODEL = args.model
    
    try:
        ensure_directories()
        process_video(args.video_path, args.output_dir)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1) 