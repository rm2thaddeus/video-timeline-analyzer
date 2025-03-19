"""
📌 Purpose: Main pipeline script for video processing with CUDA optimization
🔄 Latest Changes: Fixed duplicate imports and BLIP model configuration
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
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

# Import GPU utilities
from utils.gpu_utils import get_memory_info, detect_gpu, setup_device
from utils.logging_config import setup_logger

# Import pipeline components with fallback for module location
try:
    from processors.metadata_extractor import extract_metadata
    from CUDA.audio_processor_cuda import AudioProcessorCUDA
    from CUDA.scene_detector import SceneDetector
except ImportError:
    from test_pipeline.processors.metadata_extractor import extract_metadata
    from test_pipeline.CUDA.audio_processor_cuda import AudioProcessorCUDA
    from test_pipeline.CUDA.scene_detector import SceneDetector

# Import our metadata dataframe manager
try:
    from processors.metadata_dataframe import MetadataManager
except ModuleNotFoundError:
    from video_processing.metadata_dataframe import MetadataManager

try:
    from processors.frame_processor import FrameProcessor
except ModuleNotFoundError:
    from video_processing.frame_processor import FrameProcessor

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

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_scenes(video_path: str) -> List[Dict]:
    """Detect scenes in video using SceneDetector."""
    scene_detector = SceneDetector()
    scenes = scene_detector.detect_scenes(video_path)
    return scenes

def _extract_audio(video_path: str, video_id: str) -> Optional[str]:
    """Extract audio from video using FFmpeg."""
    try:
        output_path = f"test_pipeline/audio_chunks/{video_id}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use FFmpeg to extract audio
        cmd = f'ffmpeg -y -i "{video_path}" -ac 1 -ar 16000 "{output_path}"'
        result = os.system(cmd)
        
        if result == 0:
            return output_path
        else:
            logging.error("Failed to extract audio")
            return None
    except Exception as e:
        logging.error(f"Error extracting audio: {str(e)}")
        return None

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

def _process_audio_direct(video_path: Union[str, Path], model_name: str = "small") -> Optional[List[Dict]]:
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
        return result['segments']
            
    except Exception as e:
        logging.error(f"[AUDIO] Error during transcription: {str(e)}")
        traceback.print_exc()
        return None

def _extract_scene_frames(video_path: str, scenes: List[Dict], output_dir: Path) -> Dict[str, np.ndarray]:
    """Extract frames from each scene in the video.
    
    Args:
        video_path: Path to video file
        scenes: List of scenes with frame information
        output_dir: Output directory path
        
    Returns:
        Dictionary mapping scene IDs to frames
    """
    try:
        # Create frames directory
        frames_dir = output_dir / "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return {}
        
        frames_dict = {}
        
        # Process each scene
        for scene in scenes:
            scene_id = scene["id"]
            start_frame = scene["start_frame"]
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read frame
            ret, frame = cap.read()
            if ret:
                # Save frame
                frame_path = frames_dir / f"{scene_id}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames_dict[scene_id] = frame
            else:
                logger.error(f"Failed to read frame for scene {scene_id}")
        
        # Release video capture
        cap.release()
        
        return frames_dict
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return {}

def _process_frames_with_models(frames_dict: Dict[str, np.ndarray], video_id: str, output_dir: Path, model_type: str = "medium") -> Dict[str, Any]:
    """Process frames with CLIP and BLIP models.
    
    Args:
        frames_dict: Dictionary of frames to process
        video_id: Video ID for file naming
        output_dir: Output directory path
        model_type: Model type to use (small, medium, large)
        
    Returns:
        Dictionary containing frame analysis results
    """
    try:
        # Create frames directory
        frames_dir = output_dir / "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Initialize models
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        blip_processor, blip_model = setup_blip_model()
        
        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model = clip_model.to(device)
        blip_model = blip_model.to(device)
        
        frame_data = {}
        
        # Process each frame
        for scene_id, frame in frames_dict.items():
            try:
                # Convert frame to PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Get CLIP embeddings
                clip_inputs = clip_processor(images=frame_pil, return_tensors="pt").to(device)
                clip_features = clip_model.get_image_features(**clip_inputs)
                clip_features = clip_features.detach().cpu().numpy()
                
                # Get BLIP caption
                blip_inputs = blip_processor(frame_pil, return_tensors="pt").to(device)
                blip_output = blip_model.generate(**blip_inputs)
                caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)
                
                frame_data[scene_id] = {
                    "frame_path": str(frames_dir / f"{scene_id}.jpg"),
                    "clip_features": clip_features.tolist(),
                    "caption": caption
                }
                
                # Clean GPU memory
                del clip_features, blip_output
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing frame for scene {scene_id}: {str(e)}")
                continue
        
        # Save frame data
        frame_data_path = frames_dir / f"{video_id}_frames.json"
        with open(frame_data_path, 'w') as f:
            json.dump(frame_data, f)
        
        return frame_data
        
    except Exception as e:
        logger.error(f"Error in frame processing: {str(e)}")
        return {}

def process_video(video_path: str, output_dir: str, model_type: str = "medium") -> Dict[str, Any]:
    """Process video with CUDA acceleration.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory path
        model_type: Model type to use (small, medium, large)
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Convert output_dir to Path
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        frames_dir = output_dir / "frames"
        audio_dir = output_dir / "audio"
        transcripts_dir = output_dir / "transcripts"
        metadata_dir = output_dir / "metadata"
        
        for dir_path in [frames_dir, audio_dir, transcripts_dir, metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Get video ID from filename
        video_id = Path(video_path).stem
        
        # Extract metadata
        logger.info(f"Processing {video_path} (ID: {video_id})")
        metadata = extract_metadata(video_path)
        
        # Save metadata
        metadata_path = metadata_dir / f"{video_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Detect scenes
        logger.info("[SCENE] Detecting scenes...")
        scenes = detect_scenes(video_path)
        logger.info(f"[SCENE] Detected {len(scenes)} scenes")
        
        # Extract audio
        logger.info("[AUDIO] Extracting audio...")
        audio_path = audio_dir / f"{video_id}.wav"
        cmd = f'ffmpeg -y -i "{video_path}" -ac 1 -ar 16000 "{audio_path}"'
        subprocess.run(cmd, shell=True, check=True)
        
        # Process audio
        logger.info(f"[AUDIO] Processing transcription with {model_type} model...")
        audio_processor = AudioProcessorCUDA(model_type=model_type)
        transcription = audio_processor.process_audio(str(audio_path))
        
        # Save transcription
        transcript_path = transcripts_dir / f"{video_id}.json"
        with open(transcript_path, 'w') as f:
            json.dump(transcription, f)
        
        # Extract and process frames
        logger.info("[FRAMES] Extracting and processing frames...")
        frames_dict = _extract_scene_frames(video_path, scenes, output_dir)
        frame_data = _process_frames_with_models(frames_dict, video_id, output_dir, model_type)
        
        # Save scene data
        scene_data_path = frames_dir / f"{video_id}_scene_data.json"
        scene_data = {
            "video_id": video_id,
            "total_scenes": len(scenes),
            "processing_time": time.time() - start_time,
            "scene_data": scenes,
            "summary": {
                "total_frames_analyzed": len(frame_data),
                "scene_timestamps": [scene["start_time"] for scene in scenes]
            }
        }
        with open(scene_data_path, 'w') as f:
            json.dump(scene_data, f)
        
        return {
            "video_id": video_id,
            "metadata": metadata,
            "scenes": scenes,
            "transcription": transcription,
            "frame_data": frame_data
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        traceback.print_exc()
        return {}

def setup_blip_model():
    """Set up BLIP model with proper configuration."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Configure generation parameters
    model.config.max_new_tokens = 50  # Allow longer captions
    model.config.num_beams = 4  # Use beam search
    model.config.length_penalty = 1.0  # Balanced length penalty
    model.config.no_repeat_ngram_size = 3  # Avoid repetition
    
    return processor, model.to(device)

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