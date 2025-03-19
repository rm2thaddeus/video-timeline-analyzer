"""
📌 Purpose: Main pipeline script for video processing with CUDA optimization
🔄 Latest Changes: Fixed audio processing imports and improved error handling
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
from datetime import datetime, timedelta
import time
import shutil
from typing import Dict, Any, Optional, Union, List
from tqdm import tqdm
import subprocess
import whisper
import ffmpeg
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration

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
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("video_pipeline_cuda")

# Ensure all logs go to console as well
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add console handler to all relevant loggers
for logger_name in ["video_pipeline_cuda", "scene_detector_cuda"]:
    log = logging.getLogger(logger_name)
    log.addHandler(console_handler)
    log.setLevel(logging.INFO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

def copy_output_files(video_id: str, output_dir: str) -> None:
    """Copy all generated files to the output directory.
    
    Args:
        video_id: Video ID for file naming
        output_dir: Output directory path
    """
    # Create output directory structure
    output_dir = Path(output_dir)
    output_metadata_dir = output_dir / "metadata"
    output_scenes_dir = output_dir / "scenes"
    output_transcripts_dir = output_dir / "transcripts"
    output_screenshots_dir = output_dir / "screenshots"
    
    # Create directories
    for dir_path in [output_dir, output_metadata_dir, output_scenes_dir, output_transcripts_dir, output_screenshots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Copy metadata files
    metadata_file = Path("test_pipeline/metadata") / f"{video_id}_metadata.json"
    if metadata_file.exists():
        shutil.copy2(metadata_file, output_metadata_dir / f"{video_id}_metadata.json")
    
    metadata_csv = Path("test_pipeline/metadata") / f"{video_id}_metadata.csv"
    if metadata_csv.exists():
        shutil.copy2(metadata_csv, output_metadata_dir / f"{video_id}_metadata.csv")
    
    # Copy scene files
    scenes_file = Path("test_pipeline/scenes") / f"{video_id}_scenes.json"
    if scenes_file.exists():
        shutil.copy2(scenes_file, output_scenes_dir / f"{video_id}_scenes.json")
    
    # Copy screenshots
    screenshots_dir = Path("test_pipeline/scenes/screenshots")
    if screenshots_dir.exists():
        for screenshot in screenshots_dir.glob("scene_*.jpg"):
            shutil.copy2(screenshot, output_screenshots_dir / screenshot.name)
    
    # Copy transcripts
    for ext in [".srt", ".json"]:
        transcript_file = Path("test_pipeline/transcripts") / f"{video_id}{ext}"
        if transcript_file.exists():
            shutil.copy2(transcript_file, output_transcripts_dir / f"{video_id}{ext}")

def _process_audio_direct(video_path: Union[str, Path], model_name: str = "small") -> None:
    """Process audio directly with whisper model without chunking."""
    video_id = Path(video_path).stem
    audio_path = f"test_pipeline/audio_chunks/{video_id}.wav"
    
    # Extract audio using FFmpeg
    logging.info("[AUDIO] Extracting audio with FFmpeg...")
    subprocess.run([
        'ffmpeg', '-y', '-i', str(video_path), 
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        audio_path
    ], capture_output=True)
    logging.info(f"[AUDIO] Audio extracted to {audio_path}")
    
    try:
        # Load model
        logging.info(f"[AUDIO] Loading Whisper {model_name} model...")
        model = whisper.load_model(model_name)  # Use the provided model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logging.info(f"[AUDIO] Using {device} for processing")
        
        # Load audio
        audio = whisper.load_audio(audio_path)
        
        # Transcribe with Whisper's built-in chunking
        logging.info("[AUDIO] Transcribing audio...")
        result = model.transcribe(
            audio,
            task="transcribe",
            beam_size=5,  # Increased from 1 for better accuracy
            best_of=5,    # Increased from 1 for better accuracy
            temperature=0.2,  # Slight randomness for better results
            fp16=torch.cuda.is_available()  # Use FP16 if GPU available for speed
        )
        
        # Save results
        os.makedirs("test_pipeline/transcripts", exist_ok=True)
        
        # Save as SRT
        srt_path = f"test_pipeline/transcripts/{video_id}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start = timedelta(seconds=float(segment['start']))
                end = timedelta(seconds=float(segment['end']))
                f.write(f"{i}\n")
                f.write(f"{str(start).replace('.', ',')} --> {str(end).replace('.', ',')}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        
        # Save as JSON
        json_path = f"test_pipeline/transcripts/{video_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result['segments'], f, ensure_ascii=False, indent=2)
            
        logging.info(f"[AUDIO] Transcription completed: {len(result['segments'])} segments saved")
            
    except Exception as e:
        logging.error(f"[AUDIO] Error during transcription: {str(e)}")
        traceback.print_exc()

def _extract_scene_frames(video_path: Path, scene: dict) -> List[np.ndarray]:
    """Extract start, middle and end frames from a scene.
    
    Args:
        video_path: Path to the video file
        scene: Scene dictionary containing start_frame and end_frame
    
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_positions = [
        scene["start_frame"],  # Start
        scene["start_frame"] + (scene["end_frame"] - scene["start_frame"]) // 2,  # Middle
        scene["end_frame"]  # End
    ]
    
    for frame_pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

def _process_frames_with_models(frames: List[np.ndarray], video_id: str, scene_id: int) -> Dict:
    """Process frames with CLIP and BLIP-2 models.
    
    Args:
        frames: List of frames as numpy arrays
        video_id: Video identifier
        scene_id: Scene number
    
    Returns:
        Dictionary containing embeddings and metadata
    """
    try:
        # Initialize models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # CLIP setup
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # BLIP-2 setup
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        frame_data = []
        frame_types = ["start", "middle", "end"]
        
        # Process each frame
        for frame, frame_type in zip(frames, frame_types):
            # Save frame as image
            frame_path = f"test_pipeline/frames/{video_id}_scene_{scene_id}_{frame_type}.jpg"
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            Image.fromarray(frame).save(frame_path)
            
            # Get CLIP embeddings
            clip_inputs = clip_processor(images=frame, return_tensors="pt").to(device)
            clip_features = clip_model.get_image_features(**clip_inputs)
            embeddings = clip_features.detach().cpu().numpy()
            
            # Get BLIP-2 caption
            blip_inputs = blip_processor(Image.fromarray(frame), return_tensors="pt").to(device)
            caption = blip_model.generate(**blip_inputs)
            caption_text = blip_processor.decode(caption[0], skip_special_tokens=True)
            
            # Get detailed scene description using BLIP-2
            prompt = "Describe this scene in detail, including any notable objects, actions, or emotions:"
            inputs = blip_processor(Image.fromarray(frame), text=prompt, return_tensors="pt").to(device)
            detailed_description = blip_model.generate(**inputs)
            description_text = blip_processor.decode(detailed_description[0], skip_special_tokens=True)
            
            frame_data.append({
                "frame_type": frame_type,
                "frame_path": frame_path,
                "embeddings": embeddings.tolist(),
                "caption": caption_text,
                "detailed_description": description_text
            })
            
        return {
            "scene_id": scene_id,
            "frames": frame_data
        }
        
    except Exception as e:
        logger.error(f"Error processing frames for scene {scene_id}: {str(e)}")
        return None

def process_video(video_path: Union[str, Path], output_dir: str, whisper_model: str = "small") -> None:
    """Process a video file with the pipeline.
    
    Args:
        video_path: Path to the video file (string or Path object)
        output_dir: Output directory for processed files
        whisper_model: Whisper model name to use
    """
    start_time = time.time()
    try:
        # Convert string path to Path object if needed
        video_path = Path(video_path) if isinstance(video_path, str) else video_path
        
        # Get video ID from filename
        video_id = video_path.stem
        logger.info(f"Processing {video_path} (ID: {video_id})")
        
        # Extract metadata
        metadata = extract_metadata(video_path)
        
        # Detect scenes
        logger.info("Detecting scenes...")
        scene_detector = SceneDetector()
        start_scene_time = time.time()
        scenes = scene_detector.detect_scenes(video_path)
        scene_time = time.time() - start_scene_time
        logger.info(f"Detected {len(scenes)} scenes in {scene_time:.2f} seconds")
        
        # Clean GPU memory after scene detection
        clean_gpu_memory()
        
        # Process audio directly without chunking
        _process_audio_direct(str(video_path), whisper_model)
        
        # Clean GPU memory after audio processing
        clean_gpu_memory()
        
        # Process frames for each scene
        logger.info("Processing frames with CLIP and BLIP-2...")
        scene_data = []
        for i, scene in enumerate(scenes):
            frames = _extract_scene_frames(video_path, scene)
            if frames:
                scene_metadata = _process_frames_with_models(frames, video_id, i)
                if scene_metadata:
                    scene_data.append(scene_metadata)
        
        # Save scene data with embeddings and descriptions
        scene_data_path = f"test_pipeline/frames/{video_id}_scene_data.json"
        os.makedirs(os.path.dirname(scene_data_path), exist_ok=True)
        with open(scene_data_path, 'w') as f:
            json.dump(scene_data, f, indent=2)
        
        logger.info(f"Processed {len(scene_data)} scenes with CLIP and BLIP-2")
        
        # Clean GPU memory after frame processing
        clean_gpu_memory()
        
        # Copy all generated files to output directory
        logger.info(f"Copying generated files to output directory: {output_dir}")
        copy_output_files(video_id, output_dir)
        logger.info("All files copied to output directory successfully")
        
        # Log total processing time
        end_time = time.time()
        logger.info(f"Video processing completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

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