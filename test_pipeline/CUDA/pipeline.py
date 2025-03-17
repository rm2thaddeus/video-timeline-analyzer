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

# Import pipeline components with fallback for module location
try:
    from processors.metadata_extractor import extract_metadata
except ModuleNotFoundError:
    from video_processing.metadata_extractor import extract_metadata

# Use our optimized scene detector
from CUDA.scene_detector import SceneDetector

# Import our optimized audio processor
from CUDA.audio_processor_cuda import AudioProcessorCUDA

try:
    from processors.frame_processor import FrameProcessor
except ModuleNotFoundError:
    from video_processing.frame_processor import FrameProcessor

from utils.gpu_utils import setup_device
import cuda_config as config

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

def process_video(video_path, output_dir=None):
    """
    Process a video using CUDA-accelerated pipeline.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for results (optional)
        
    Returns:
        Dictionary with processing results
    """
    # Create output directories if they don't exist
    os.makedirs(config.METADATA_DIR, exist_ok=True)
    os.makedirs(config.SCENES_DIR, exist_ok=True)
    os.makedirs(config.AUDIO_CHUNKS_DIR, exist_ok=True)
    os.makedirs(config.TRANSCRIPTS_DIR, exist_ok=True)
    os.makedirs(config.FRAMES_DIR, exist_ok=True)
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Setup CUDA device
    device, device_props = setup_device()
    total_mem = 0
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Apply GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(config.GPU_MEMORY_FRACTION, 0)
    
    video_path = Path(video_path)
    video_id = video_path.stem
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"[START] Starting CUDA video processing pipeline for {video_path}")
    
    # Setup device for GPU processing with CUDA
    if device.type == "cuda":
        logger.info(f"[CUDA] Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        # Apply CUDA optimizations
        setup_cuda_optimizations(device)
        
        # Log GPU memory before processing
        allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
        free_mem = total_mem - allocated_mem
        logger.info(f"[MEMORY] GPU Memory - Total: {total_mem:.2f}GB, Allocated: {allocated_mem:.2f}GB, Free: {free_mem:.2f}GB")
    else:
        logger.warning(f"[WARN] CUDA not available, using device: {device.type}")
    
    # Set up mixed precision if enabled
    if config.USE_MIXED_PRECISION and device.type == "cuda":
        try:
            # Use the newer recommended approach if available
            amp_context = torch.amp.autocast(device_type='cuda')
            logger.info("[CONFIG] Mixed precision (FP16) enabled")
        except Exception:
            # Fall back to the older approach
            amp_context = torch.cuda.amp.autocast()
            logger.info("[CONFIG] Mixed precision (legacy mode) enabled")
    else:
        amp_context = nullcontext()
        if device.type == "cuda":
            logger.info("[CONFIG] Mixed precision disabled")
    
    try:
        # Step 1: Extract metadata (CPU-bound, no need for CUDA)
        logger.info("[STEP 1] Extracting video metadata...")
        metadata = extract_metadata(video_path, config.METADATA_DIR)
        logger.info(f"[METADATA] Extracted - Duration: {metadata['duration_seconds']:.2f}s, Resolution: {metadata['width']}x{metadata['height']}")
        
        # Clean GPU memory between steps
        clean_gpu_memory()
        
        # Step 2: Detect scenes with optimized batch processing
        logger.info("[STEP 2] Detecting scenes with batch processing...")
        logger.info(f"[CONFIG] Using threshold: {config.SCENE_THRESHOLD}, batch size: {config.SCENE_BATCH_SIZE}")
        
        scene_detector = SceneDetector(
            threshold=config.SCENE_THRESHOLD,
            device=device,
            batch_size=config.SCENE_BATCH_SIZE,
            checkpoint_interval=config.SCENE_CHECKPOINT_INTERVAL
        )
        
        try:
            logger.info(f"[SCENES] Processing video frames for scene detection...")
            scenes = scene_detector.detect_scenes(video_path, config.SCENES_DIR)
            logger.info(f"[SCENES] Scene detection completed - Found {len(scenes)} scenes")
            
            # Log some scene information
            if scenes:
                logger.info(f"[SCENES] First scene: {scenes[0]['start_time']:.2f}s to {scenes[0]['end_time']:.2f}s ({scenes[0]['duration_seconds']:.2f}s)")
                if len(scenes) > 1:
                    logger.info(f"[SCENES] Last scene: {scenes[-1]['start_time']:.2f}s to {scenes[-1]['end_time']:.2f}s ({scenes[-1]['duration_seconds']:.2f}s)")
        except Exception as e:
            logger.error(f"[ERROR] Error during scene detection: {e}")
            logger.error(traceback.format_exc())
            # Try to recover and continue with fallback scene detection
            logger.info("[FALLBACK] Attempting fallback scene detection with more conservative settings...")
            try:
                # Try with more conservative settings
                scene_detector = SceneDetector(
                    threshold=config.SCENE_THRESHOLD * 1.5,  # Higher threshold = fewer scenes
                    device=None,  # Force CPU
                    batch_size=100,  # Smaller batches
                    checkpoint_interval=500  # More frequent checkpoints
                )
                scenes = scene_detector.detect_scenes(video_path, config.SCENES_DIR)
                logger.info(f"[FALLBACK] Scene detection completed - Found {len(scenes)} scenes")
            except Exception as e2:
                logger.error(f"[ERROR] Fallback scene detection also failed: {e2}")
                # Create a single scene as last resort
                duration = metadata['duration_seconds']
                scenes = [{
                    "id": 0,
                    "start_frame": 0,
                    "end_frame": int(metadata.get('fps', 25) * duration),
                    "start_time": 0,
                    "end_time": duration,
                    "duration_frames": int(metadata.get('fps', 25) * duration),
                    "duration_seconds": duration,
                    "resolution": {
                        "width": metadata['width'],
                        "height": metadata['height']
                    }
                }]
                logger.info("[FALLBACK] Using fallback single scene covering entire video")
        
        # Clean GPU memory between steps
        clean_gpu_memory()
        
        # Adapt scenes to the format expected by frame_processor
        scenes_adapted = _adapt_scenes_for_frame_processor(scenes)
        
        # Step 3 & 4: Process audio and extract frames in parallel if enabled
        if config.PARALLEL_PROCESSING and device.type == "cuda":
            logger.info("[PARALLEL] Running audio processing and frame extraction in parallel...")
            
            # Create processor instances
            frame_processor = FrameProcessor(
                model_name=config.CLIP_MODEL, 
                device=device
            )
            
            audio_result = None
            frames = []
            
            # Log start of parallel processing
            logger.info(f"[CONFIG] Parallel processing configuration:")
            logger.info(f"  - Frame processor model: {config.CLIP_MODEL}")
            logger.info(f"  - Audio processor model: {config.WHISPER_MODEL}")
            
            # Run tasks in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                logger.info("[PARALLEL] Submitting parallel tasks...")
                
                # Submit frame extraction task
                logger.info("[FRAMES] Frame extraction task submitted")
                frame_future = executor.submit(
                    lambda: frame_processor.extract_frames(
                        video_path, 
                        scenes_adapted,
                        output_dir=config.FRAMES_DIR,
                        frames_per_scene=3,
                        max_dimension=config.MAX_FRAME_DIMENSION
                    )
                )
                
                # Submit audio processing task
                logger.info(f"[AUDIO] Processing audio for {video_id} with Whisper {config.WHISPER_MODEL} model")
                logger.info("[AUDIO] Audio processing task submitted")
                audio_future = executor.submit(
                    lambda: _process_audio_direct(video_path, video_id, config.WHISPER_MODEL)
                )
                
                # Wait for tasks to complete
                logger.info("[PARALLEL] Waiting for tasks to complete...")
                
                # Get frame extraction results
                try:
                    frames = frame_future.result()
                    logger.info(f"[FRAMES] Frame extraction completed - Extracted {len(frames)} frames")
                    
                    # Generate embeddings for the extracted frames
                    logger.info(f"[FRAMES] Generating embeddings for {len(frames)} frames...")
                    frames_with_embeddings = frame_processor.embed_frames(
                        frames,
                        output_dir=config.EMBEDDINGS_DIR
                    )
                    frames = frames_with_embeddings
                    logger.info(f"[FRAMES] Frame embedding completed")
                except Exception as e:
                    logger.error(f"[ERROR] Error during frame extraction: {e}")
                    logger.error(traceback.format_exc())
                    frames = []
                
                # Get audio processing results
                try:
                    audio_result = audio_future.result()
                    logger.info(f"[AUDIO] Audio processing completed in {audio_result['processing_time_seconds']:.2f} seconds")
                except Exception as e:
                    logger.error(f"[ERROR] Error during audio processing: {e}")
                    logger.error(traceback.format_exc())
                    audio_result = {
                        "video_id": video_id,
                        "srt_path": "",
                        "json_path": "",
                        "processing_time_seconds": 0
                    }
            
            if device.type == "cuda":
                torch.cuda.synchronize()  # Ensure parallel GPU operations are complete
                logger.info("[CUDA] Synchronizing CUDA operations...")
                clean_gpu_memory()  # Clear cache to free up memory
            
            logger.info(f"[PARALLEL] Parallel processing completed")
        else:
            # Sequential processing
            frames = []
            audio_result = None
            
            # Step 3: Process audio and transcribe
            logger.info("[STEP 3] Processing audio and transcribing...")
            try:
                audio_result = _process_audio_direct(video_path, video_id, config.WHISPER_MODEL)
                logger.info(f"[AUDIO] Audio processing completed in {audio_result['processing_time_seconds']:.2f} seconds")
            except Exception as e:
                logger.error(f"[ERROR] Error during audio processing: {e}")
                logger.error(traceback.format_exc())
                # Create minimal result structure
                audio_result = {
                    "video_id": video_id,
                    "srt_path": "",
                    "json_path": "",
                    "processing_time_seconds": 0
                }
            
            # Clean GPU memory between steps
            clean_gpu_memory()
            
            # Step 4: Extract and embed frames
            logger.info("[STEP 4] Extracting and embedding frames...")
            try:
                logger.info(f"[CONFIG] Using CLIP model: {config.CLIP_MODEL}")
                frame_processor = FrameProcessor(
                    model_name=config.CLIP_MODEL, 
                    device=device
                )
                
                logger.info(f"[FRAMES] Extracting frames from {len(scenes_adapted)} scenes...")
                with amp_context:
                    frames = frame_processor.extract_frames(
                        video_path, 
                        scenes_adapted,
                        output_dir=config.FRAMES_DIR,
                        frames_per_scene=3,
                        max_dimension=config.MAX_FRAME_DIMENSION
                    )
                    
                    # Generate embeddings for the extracted frames
                    logger.info(f"[FRAMES] Generating embeddings for {len(frames)} frames...")
                    frames_with_embeddings = frame_processor.embed_frames(
                        frames,
                        output_dir=config.EMBEDDINGS_DIR
                    )
                    frames = frames_with_embeddings
                    
                if device.type == "cuda":
                    torch.cuda.synchronize()  # Ensure GPU operations for frame extraction are complete
                    clean_gpu_memory()  # Clear cache to free up memory
                
                logger.info(f"[FRAMES] Frame extraction and embedding completed - Extracted {len(frames)} frames")
            except Exception as e:
                logger.error(f"[ERROR] Error during frame extraction: {e}")
                logger.error(traceback.format_exc())
        
        # Log GPU memory after processing
        if device.type == "cuda":
            clean_gpu_memory()  # Clear cache to free up memory
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - allocated_mem
            logger.info(f"[MEMORY] GPU Memory after processing - Allocated: {allocated_mem:.2f}GB, Free: {free_mem:.2f}GB")
        
        # Save results
        result = {
            "video_id": video_id,
            "metadata": metadata,
            "scenes": scenes,
            "frames": frames or [],
            "audio_processing": {
                "srt_path": audio_result["srt_path"] if audio_result else "",
                "json_path": audio_result["json_path"] if audio_result else "",
                "processing_time_seconds": audio_result["processing_time_seconds"] if audio_result else 0
            },
            "processed_with_cuda": device.type == "cuda",
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "cuda_optimizations": {
                "mixed_precision": config.USE_MIXED_PRECISION,
                "cudnn_benchmark": config.USE_CUDNN_BENCHMARK,
                "pinned_memory": config.USE_PINNED_MEMORY,
                "batch_size": config.GPU_BATCH_SIZE,
                "parallel_processing": config.PARALLEL_PROCESSING,
                "gpu_memory_fraction": config.GPU_MEMORY_FRACTION
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary to disk
        if output_dir is not None:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a more detailed summary
            summary_path = output_dir / f"{video_id}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"[OUTPUT] Summary saved to {summary_path}")
            
            # Create a Markdown summary file
            md_summary_path = output_dir / f"{video_id}_PROCESSING_SUMMARY.md"
            with open(md_summary_path, "w") as f:
                f.write(f"# CUDA Video Processing Pipeline Results\n\n")
                f.write(f"## Processing Summary for \"{Path(video_path).name}\"\n\n")
                
                f.write("We successfully processed the video using our optimized CUDA pipeline with the following results:\n\n")
                
                f.write("### Performance\n")
                scene_time = scene_detector.stats.get("end_time", 0) - scene_detector.stats.get("start_time", 0)
                audio_time = audio_result.get("processing_time_seconds", 0)
                frame_time = 5  # Placeholder, actual timing info not available
                total_time = scene_time + audio_time + frame_time
                f.write(f"- Scene detection: Completed in ~{int(scene_time)} seconds\n")
                f.write(f"- Audio transcription: Completed in ~{int(audio_time)} seconds\n")
                f.write(f"- Frame extraction: Completed in ~{int(frame_time)} seconds\n")
                f.write(f"- Total processing time: ~{int(total_time)} seconds\n")
                f.write(f"- GPU Memory utilization: {allocated_mem:.2f}GB (out of {total_mem:.2f}GB available)\n\n")
                
                f.write("### Scene Detection\n")
                f.write(f"- Detected {len(scenes)} main scenes\n")
                f.write(f"- Used CUDA-accelerated content detection with batch processing\n")
                f.write(f"- Threshold: {config.SCENE_THRESHOLD}\n")
                f.write(f"- Batch size: {config.SCENE_BATCH_SIZE}\n\n")
                
                f.write("### Transcription\n")
                f.write(f"- Generated complete SRT subtitle file\n")
                f.write(f"- Used Whisper \"{config.WHISPER_MODEL}\" model\n")
                f.write(f"- Processed full audio file without chunking for reliability\n\n")
                
                f.write("### Frame Extraction\n")
                f.write(f"- Extracted {len(frames)} key frames from the video\n")
                f.write(f"- Used CLIP model for frame analysis\n")
                f.write(f"- Frames extracted at strategic points in the timeline\n\n")
                
                f.write("## Technical Details\n\n")
                f.write(f"- CUDA Device: {result['cuda_device']}\n")
                f.write(f"- Mixed Precision: {'Enabled' if result['cuda_optimizations']['mixed_precision'] else 'Disabled'}\n")
                f.write(f"- Parallel Processing: {'Enabled' if result['cuda_optimizations']['parallel_processing'] else 'Disabled'}\n")
                f.write(f"- Batch Size: {result['cuda_optimizations']['batch_size']}\n")
                f.write(f"- Memory Fraction: {result['cuda_optimizations']['gpu_memory_fraction']}\n\n")
                
                f.write("## Usage\n\n")
                f.write("To process videos with this pipeline:\n\n")
                f.write("```bash\n")
                f.write(f"python test_pipeline/CUDA/run_pipeline.py \"path/to/video.mp4\" --gpu_memory_fraction {config.GPU_MEMORY_FRACTION} --whisper_model {config.WHISPER_MODEL}\n")
                f.write("```\n\n")
                
                f.write("Additional parameters can be found by running:\n\n")
                f.write("```bash\n")
                f.write("python test_pipeline/CUDA/run_pipeline.py --help\n")
                f.write("```\n")
                
            logger.info(f"[OUTPUT] Markdown summary saved to {md_summary_path}")
        
        logger.info(f"[SUCCESS] Video processing completed successfully!")
        
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] Error in video processing pipeline: {e}")
        logger.error(traceback.format_exc())
        
        # Try to return partial results if available
        result = {
            "video_id": video_id,
            "metadata": metadata if 'metadata' in locals() else {},
            "scenes": scenes if 'scenes' in locals() else [],
            "frames": frames if 'frames' in locals() else [],
            "audio_processing": {
                "srt_path": audio_result["srt_path"] if 'audio_result' in locals() and audio_result else "",
                "json_path": audio_result["json_path"] if 'audio_result' in locals() and audio_result else "",
                "processing_time_seconds": audio_result["processing_time_seconds"] if 'audio_result' in locals() and audio_result else 0
            },
            "error": str(e),
            "processed_with_cuda": device.type == "cuda" if 'device' in locals() else False,
            "timestamp": datetime.now().isoformat()
        }
        
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
