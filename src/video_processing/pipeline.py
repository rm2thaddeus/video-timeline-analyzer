"""
Video Processing Pipeline.

This module orchestrates the video processing pipeline, bringing together scene detection
and frame extraction in an optimized workflow that leverages GPU acceleration.

📌 Purpose: Coordinate the video processing pipeline
🔄 Latest Changes: Initial implementation with GPU optimization
⚙️ Key Logic: Pipeline management with memory-efficient batch processing
📂 Expected File Path: src/video_processing/pipeline.py
🧠 Reasoning: Centralized pipeline orchestration allows for efficient resource 
              management and prevents memory issues with large videos
"""

import os
import logging
import tempfile
import json
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from enum import Enum
from pathlib import Path
import time
import datetime
import traceback
import subprocess # Added for running ffmpeg

import cv2
import numpy as np
import torch

from src.models.schema import Scene, Frame, VideoMetadata, AnalysisConfig, AnalysisResults
from src.utils.gpu_utils import setup_device, get_memory_info, clear_gpu_memory, memory_stats
from src.video_processing.loader import extract_video_metadata, create_video_capture
from src.video_processing.scene_detection import detect_scenes, DetectionMethod, GPUAcceleratedSceneDetector
from src.video_processing.frame_extraction import FrameExtractor, FrameExtractionMethod

logger = logging.getLogger(__name__)

# Get the optimal device (GPU if available, otherwise CPU)
DEVICE = setup_device()

# --- Helper Function for Audio Extraction ---
def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """Extracts audio from video using ffmpeg."""
    logger.info(f"Extracting audio from {video_path} to {output_audio_path}")
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # Disable video recording
        '-acodec', 'pcm_s16le', # Standard WAV format
        '-ar', '16000', # Sample rate expected by Whisper
        '-ac', '1', # Mono channel
        '-y', # Overwrite output file if it exists
        output_audio_path
    ]
    try:
        # Use subprocess.run for better error handling and capture
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.debug(f"ffmpeg stdout: {result.stdout}")
        logger.debug(f"ffmpeg stderr: {result.stderr}")
        logger.info(f"Audio extracted successfully to {output_audio_path}")
        return True
    except FileNotFoundError:
        logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your system PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed with exit code {e.returncode}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio extraction: {e}")
        return False
# --- End Helper Function ---

class VideoProcessingPipeline:
    """Orchestrates the video processing pipeline with GPU optimization."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the video processing pipeline.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config or AnalysisConfig()
        self.device = setup_device() if self.config.use_gpu else torch.device('cpu')
        
        # Initialize components
        self.scene_detector = GPUAcceleratedSceneDetector(
            detection_method=self.config.scene_detection_method,
            threshold=self.config.scene_threshold,
            min_scene_length=self.config.min_scene_length,
            use_gpu=self.config.use_gpu
        )
        
        self.frame_extractor = FrameExtractor(
            batch_size=self.config.batch_size,
            use_gpu=self.config.use_gpu,
            use_half_precision=False,  # Half precision can be enabled for specific models later
            num_workers=max(1, self.config.max_workers // 2)  # Use half the workers for loading
        )
        
        logger.info(f"Initialized video processing pipeline (device: {self.device})")
    
    def process_video(
        self,
        video_path: str,
        output_dir: str, # This is now the BASE output directory
        save_results: bool = True
    ) -> Optional[AnalysisResults]:
        """
        Process a video through the pipeline.
        
        Args:
            video_path: Path to the video file
            output_dir: Base directory to save all video outputs
            save_results: Whether to save results to disk
            
        Returns:
            AnalysisResults object, or None if processing fails
        """
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        video_file_path = Path(video_path)
        video_stem = video_file_path.stem

        # --- Create video-specific output directory --- 
        video_output_dir = Path(output_dir) / video_stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to: {video_output_dir}")

        # --- Create all expected output subdirectories within video_output_dir --- 
        scenes_dir = video_output_dir / "scenes"
        frames_dir = video_output_dir / "frames"
        audio_dir = video_output_dir / "audio"
        transcripts_dir = video_output_dir / "transcripts"
        metadata_dir = video_output_dir / "metadata"
        screenshots_dir = video_output_dir / "screenshots"

        for dir_path in [scenes_dir, frames_dir, audio_dir, 
                         transcripts_dir, metadata_dir, screenshots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output subdirectories exist under: {video_output_dir}")
        # --- End Directory Creation ---

        video_metadata = extract_video_metadata(video_path)
        if video_metadata is None:
            logger.error(f"Failed to extract metadata from video: {video_path}")
            return None
        logger.debug("Video metadata extracted successfully.")

        # --- Save raw metadata --- 
        if save_results:
            try:
                metadata_path = metadata_dir / "video_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(video_metadata.dict(), f, indent=2, default=str)
                logger.debug(f"Saved raw video metadata to {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to save raw video metadata: {e}")
        # --- End Save raw metadata ---

        logger.info("Step 1: Scene Detection")
        try:
            scenes = self.scene_detector.detect_scenes(
                video_path,
                stats_file=str(scenes_dir / "scene_stats.csv") if save_results else None
            )
        except Exception as e:
            logger.error(f"Exception during scene detection: {e}")
            logger.error(traceback.format_exc())
            return None
        if scenes is None or len(scenes) == 0:
            logger.warning(f"No scenes detected in video: {video_path}. Skipping further processing.")
            results = AnalysisResults(
                video_metadata=video_metadata,
                timeline=Timeline(video_id=1, video_metadata=video_metadata, scenes=[]),
                config=self.config,
                processing_time=time.time() - start_time,
                error_log=[f"No scenes detected at {datetime.datetime.now()}"]
            )
            if save_results:
                 self._save_results_json(results, str(video_output_dir / "results.json")) # Save main results here
            return results
        
        if save_results:
            self._save_scenes_json(scenes, str(scenes_dir / "scenes.json"))

        # --- Step 1.5: Audio Extraction --- 
        audio_file_path = None
        if self.config.transcribe_audio:
            logger.info("Step 1.5: Audio Extraction")
            audio_file_name = video_stem + ".wav"
            audio_file_path = audio_dir / audio_file_name
            if not extract_audio(video_path, str(audio_file_path)):
                 logger.warning(f"Audio extraction failed for {video_path}. Transcription will be skipped.")
                 audio_file_path = None
            else:
                 logger.info(f"Audio extracted to {audio_file_path}")
        else:
             logger.info("Skipping audio extraction (transcribe_audio is False).")
        # --- End Audio Extraction ---

        logger.info("Step 2: Frame Extraction")
        if self.config.extract_key_frames and scenes:
            logger.debug("Starting frame extraction...")
            # NOTE: frame_extractor needs to handle saving frames relative to frames_dir
            # and potentially update scene.key_frames[...].file_path accordingly.
            # Assuming FrameExtractor's save logic correctly uses the provided output_dir.
            self.frame_extractor.extract_frames_from_scenes(
                video_path,
                scenes,
                str(frames_dir), # Pass the correct frames directory
                method=self.config.key_frames_method,
                max_frames_per_scene=1
            )
            logger.debug("Frame extraction completed.")
        else:
             logger.info("Skipping frame extraction.")

        # --- Step 3: Transcription (Placeholder) --- 
        if audio_file_path:
            logger.info("Step 3: Audio Transcription (Placeholder)")
            transcript_path = transcripts_dir / (video_stem + ".txt")
            try:
                with open(transcript_path, 'w') as f:
                    f.write("Transcription placeholder - Implement Whisper integration here.\n")
                logger.info(f"Placeholder transcript saved to {transcript_path}")
            except Exception as e:
                logger.error(f"Failed to write placeholder transcript: {e}")
        else:
            logger.info("Skipping transcription (audio not available or not requested).")
        # --- End Transcription --- 

        # --- Step 4: Other Analyses (Placeholders) ---
        # TODO: Implement Face Detection/Recognition (update Frame objects)
        # TODO: Implement Caption Generation (update Frame objects)
        # TODO: Implement Tag Generation (update Frame/Scene objects)
        # TODO: Implement Screenshot logic (save to screenshots_dir)
        # --- End Other Analyses --- 

        logger.debug("Aggregating final results...")
        video_metadata.processed_at = datetime.datetime.now()
        from src.models.schema import Timeline
        timeline = Timeline(
            video_id=1,
            video_metadata=video_metadata,
            scenes=scenes
        )
        results = AnalysisResults(
            video_metadata=video_metadata,
            timeline=timeline,
            config=self.config,
            processing_time=time.time() - start_time
        )
        if save_results:
            # Save the main results JSON in the video-specific directory
            self._save_results_json(results, str(video_output_dir / "results.json"))
        logger.info(f"Completed video processing in {results.processing_time:.2f} seconds")
        return results
    
    def process_videos_batch(
        self,
        video_paths: List[str],
        output_dir: str,
        save_results: bool = True
    ) -> Dict[str, AnalysisResults]:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of paths to video files
            output_dir: Directory to save output
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary mapping video paths to AnalysisResults objects
        """
        logger.info(f"Processing {len(video_paths)} videos")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process videos
        results = {}
        
        for video_path in video_paths:
            # Create output directory for this video
            video_name = Path(video_path).stem
            video_output_dir = os.path.join(output_dir, video_name)
            
            try:
                # Process video
                video_results = self.process_video(
                    video_path,
                    video_output_dir,
                    save_results
                )
                
                results[video_path] = video_results
                
                # Clear GPU memory after each video
                if self.device.type == 'cuda':
                    clear_gpu_memory()
            
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {str(e)}")
                results[video_path] = None
        
        return results
    
    def _save_scenes_json(self, scenes: List[Scene], output_path: str) -> None:
        """
        Save scenes to a JSON file.
        
        Args:
            scenes: List of Scene objects
            output_path: Path to save the JSON file
        """
        try:
            # Convert to dictionary
            scenes_data = [scene.dict() for scene in scenes]
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(scenes_data, f, indent=2, default=str)
            
            logger.debug(f"Saved scenes to: {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving scenes to JSON: {str(e)}")
    
    def _save_results_json(self, results: AnalysisResults, output_path: str) -> None:
        """
        Save analysis results to a JSON file.
        
        Args:
            results: AnalysisResults object
            output_path: Path to save the JSON file
        """
        try:
            # Convert to dictionary
            results_data = results.dict()
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.debug(f"Saved results to: {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving results to JSON: {str(e)}")


class GPUMemoryManager:
    """
    Manages GPU memory efficiently for the video processing pipeline.
    
    This class implements strategies to avoid out-of-memory errors and optimize
    GPU memory usage during video processing.
    """
    
    def __init__(
        self,
        device: torch.device,
        max_memory_usage: float = 0.9,  # Maximum fraction of GPU memory to use
        check_interval: int = 10  # Check memory every N operations
    ):
        """
        Initialize the GPU memory manager.
        
        Args:
            device: PyTorch device
            max_memory_usage: Maximum fraction of GPU memory to use
            check_interval: Check memory every N operations
        """
        self.device = device
        self.max_memory_usage = max_memory_usage
        self.check_interval = check_interval
        self.operation_counter = 0
        
        logger.info(f"Initialized GPU memory manager (device: {device}, max usage: {max_memory_usage*100:.0f}%)")
    
    def __enter__(self):
        """Context manager entry."""
        # Check available memory at start
        if self.device.type == 'cuda':
            self._log_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Clean up resources
        if self.device.type == 'cuda':
            clear_gpu_memory()
            self._log_memory_stats()
    
    def check_memory(self, force_clear: bool = False) -> bool:
        """
        Check GPU memory usage and clear cache if necessary.
        
        Args:
            force_clear: Whether to force clearing the cache
            
        Returns:
            bool: True if memory is available, False if approaching limits
        """
        if self.device.type != 'cuda':
            return True
        
        self.operation_counter += 1
        
        # Check memory periodically or when forced
        if force_clear or self.operation_counter % self.check_interval == 0:
            stats = memory_stats()
            
            # Calculate memory usage ratio
            if 'free_gb' in stats and stats['free_gb'] > 0:
                reserved_ratio = stats['reserved_gb'] / (stats['reserved_gb'] + stats['free_gb'])
                
                # Log memory usage
                logger.debug(f"GPU memory usage: {reserved_ratio*100:.1f}% ({stats['reserved_gb']:.2f}GB/{stats['reserved_gb']+stats['free_gb']:.2f}GB)")
                
                # Clear cache if approaching limit
                if reserved_ratio > self.max_memory_usage or force_clear:
                    logger.debug(f"Clearing GPU memory cache (usage: {reserved_ratio*100:.1f}%)")
                    clear_gpu_memory()
                    return False
        
        return True
    
    def _log_memory_stats(self) -> None:
        """Log GPU memory statistics."""
        if self.device.type != 'cuda':
            return
        
        stats = memory_stats()
        logger.debug(
            f"GPU memory - Allocated: {stats['allocated_gb']:.2f}GB, "
            f"Reserved: {stats['reserved_gb']:.2f}GB, "
            f"Free: {stats['free_gb']:.2f}GB"
        )