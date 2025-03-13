"""
Scene Detection Module with GPU Acceleration.

This module provides functions for detecting scene changes in videos,
optimized for GPU acceleration when available.

📌 Purpose: Detect scene boundaries in videos with GPU acceleration
🔄 Latest Changes: Initial implementation with GPU support
⚙️ Key Logic: Uses PySceneDetect with custom GPU-accelerated content detector
📂 Expected File Path: src/video_processing/scene_detection.py
🧠 Reasoning: Scene detection is computationally intensive and benefits from
              GPU acceleration for histogram comparison and content analysis
"""

import os
import logging
import tempfile
from typing import List, Optional, Tuple, Dict, Any, Union

import cv2
import numpy as np
import torch
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector
from scenedetect.scene_detector import SceneDetector
from scenedetect.stats_manager import StatsManager
from scenedetect.frame_timecode import FrameTimecode

from src.models.schema import Scene, Frame, VideoMetadata
from src.utils.gpu_utils import get_optimal_device, memory_stats, clear_gpu_memory

logger = logging.getLogger(__name__)

# Get the optimal device (GPU if available, otherwise CPU)
DEVICE = get_optimal_device()
USE_GPU = DEVICE.type != 'cpu'

class GPUContentDetector(ContentDetector):
    """
    GPU-accelerated content detector for scene detection.
    
    This extends the standard ContentDetector from PySceneDetect to use
    GPU acceleration for histogram calculation and comparison when available.
    """
    
    def __init__(self, threshold: float = 30.0, min_scene_len: int = 15,
                 weights: Optional[List[float]] = None, luma_only: bool = False,
                 kernel_size: Optional[Tuple[int, int]] = None):
        """
        Initialize GPU-accelerated content detector.
        
        Args:
            threshold: Threshold for scene cut detection (higher values = less sensitive)
            min_scene_len: Minimum scene length in frames
            weights: Weights for the HSV channels (default: [0.5, 0.3, 0.2])
            luma_only: Whether to use only luma channel (Y) for comparison
            kernel_size: Size of Gaussian kernel for frame preprocessing
        """
        super().__init__(threshold, min_scene_len)
        self.weights = weights if weights is not None else [0.5, 0.3, 0.2]
        self.luma_only = luma_only
        self.kernel_size = kernel_size
        self.use_gpu = USE_GPU
        
        if self.use_gpu:
            logger.info(f"Using GPU ({DEVICE.type}) for content detection")
            # Convert weights to tensor on GPU
            self.weights_tensor = torch.tensor(self.weights, device=DEVICE)
        else:
            logger.info("Using CPU for content detection")
    
    def _calculate_frame_score(self, frame_img: np.ndarray, 
                              frame_num: int) -> float:
        """
        Calculate frame score using GPU acceleration if available.
        
        Args:
            frame_img: Frame image as numpy array
            frame_num: Frame number
            
        Returns:
            float: Frame score
        """
        # Convert frame to HSV color space
        if self.luma_only:
            # Use only luma channel (Y) from YUV
            frame_hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2YUV)[:, :, 0]
        else:
            # Use full HSV
            frame_hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur if kernel size is specified
        if self.kernel_size is not None:
            frame_hsv = cv2.GaussianBlur(frame_hsv, self.kernel_size, 0)
        
        if self.use_gpu:
            # Move frame to GPU and calculate histogram
            if self.luma_only:
                # Single channel histogram
                frame_tensor = torch.from_numpy(frame_hsv).to(DEVICE)
                hist = torch.histc(frame_tensor.float(), bins=256, min=0, max=255)
                hist = hist / hist.sum()  # Normalize
            else:
                # Multi-channel histogram
                frame_tensor = torch.from_numpy(frame_hsv).to(DEVICE)
                h_hist = torch.histc(frame_tensor[:, :, 0].float(), bins=256, min=0, max=255)
                s_hist = torch.histc(frame_tensor[:, :, 1].float(), bins=256, min=0, max=255)
                v_hist = torch.histc(frame_tensor[:, :, 2].float(), bins=256, min=0, max=255)
                
                # Normalize histograms
                h_hist = h_hist / h_hist.sum()
                s_hist = s_hist / s_hist.sum()
                v_hist = v_hist / v_hist.sum()
                
                # Store histograms for next frame comparison
                if frame_num == 0:
                    self.last_hsv = (h_hist, s_hist, v_hist)
                    return 0.0
                
                # Calculate weighted difference between current and last frame
                h_diff = torch.sum(torch.abs(h_hist - self.last_hsv[0]))
                s_diff = torch.sum(torch.abs(s_hist - self.last_hsv[1]))
                v_diff = torch.sum(torch.abs(v_hist - self.last_hsv[2]))
                
                # Apply weights
                score = (h_diff * self.weights_tensor[0] + 
                         s_diff * self.weights_tensor[1] + 
                         v_diff * self.weights_tensor[2])
                
                # Update last frame histograms
                self.last_hsv = (h_hist, s_hist, v_hist)
                
                return score.item() * 100.0  # Scale to match ContentDetector
        
        # Fall back to CPU implementation if GPU is not available
        return super()._calculate_frame_score(frame_img, frame_num)


def detect_scenes(video_path: str, 
                 threshold: float = 30.0,
                 min_scene_len: float = 1.0,
                 method: str = 'content',
                 luma_only: bool = False,
                 stats_file: Optional[str] = None) -> List[Scene]:
    """
    Detect scenes in a video using GPU acceleration when available.
    
    Args:
        video_path: Path to the video file
        threshold: Threshold for scene cut detection (higher values = less sensitive)
        min_scene_len: Minimum scene length in seconds
        method: Detection method ('content', 'threshold', 'hybrid')
        luma_only: Whether to use only luma channel for comparison
        stats_file: Path to save detection statistics
        
    Returns:
        List[Scene]: List of detected scenes
    """
    logger.info(f"Detecting scenes in video: {video_path}")
    logger.info(f"Method: {method}, Threshold: {threshold}, Min Scene Length: {min_scene_len}s")
    
    # Open video and get basic properties
    video = open_video(video_path)
    fps = video.frame_rate
    duration = video.duration
    frame_count = video.frame_count
    
    # Convert min_scene_len from seconds to frames
    min_scene_len_frames = int(min_scene_len * fps)
    
    # Create stats manager
    stats_manager = StatsManager()
    
    # Create scene manager
    scene_manager = SceneManager(stats_manager)
    
    # Add appropriate detector based on method
    if method == 'content':
        # Use GPU-accelerated content detector if available
        detector = GPUContentDetector(threshold=threshold, 
                                     min_scene_len=min_scene_len_frames,
                                     luma_only=luma_only)
        scene_manager.add_detector(detector)
    elif method == 'threshold':
        detector = ThresholdDetector(threshold=threshold, 
                                    min_scene_len=min_scene_len_frames)
        scene_manager.add_detector(detector)
    elif method == 'hybrid':
        # Use both content and threshold detectors
        content_detector = GPUContentDetector(threshold=threshold,
                                            min_scene_len=min_scene_len_frames,
                                            luma_only=luma_only)
        threshold_detector = ThresholdDetector(threshold=threshold * 0.7,  # Lower threshold
                                             min_scene_len=min_scene_len_frames)
        scene_manager.add_detector(content_detector)
        scene_manager.add_detector(threshold_detector)
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    # Detect scenes
    logger.info(f"Processing {frame_count} frames...")
    
    # Use a progress callback to track detection progress
    def progress_callback(current_frame: int, total_frames: int):
        if current_frame % 500 == 0 or current_frame == total_frames - 1:
            percent = 100.0 * current_frame / total_frames
            logger.info(f"Scene detection progress: {percent:.1f}% ({current_frame}/{total_frames})")
            
            # Log GPU memory usage if available
            if USE_GPU:
                mem_stats = memory_stats()
                logger.debug(f"GPU Memory: {mem_stats['allocated_gb']:.2f}GB allocated, "
                            f"{mem_stats['reserved_gb']:.2f}GB reserved")
    
    # Detect scenes
    scene_manager.detect_scenes(video, callback=progress_callback)
    
    # Get scene list
    scene_list = scene_manager.get_scene_list()
    
    # Save stats if requested
    if stats_file:
        stats_manager.save_to_csv(stats_file)
    
    # Convert scene list to our Scene model
    scenes = []
    for i, (start_time, end_time) in enumerate(scene_list):
        start_seconds = start_time.get_seconds()
        end_seconds = end_time.get_seconds()
        
        scene = Scene(
            id=i,
            start_time=start_seconds,
            end_time=end_seconds,
            key_frames=[],  # Will be populated later
            transcript_segments=[],  # Will be populated later
            audio_events=[],  # Will be populated later
            tags=[],  # Will be populated later
        )
        scenes.append(scene)
    
    logger.info(f"Detected {len(scenes)} scenes")
    
    # Clean up GPU memory if used
    if USE_GPU:
        clear_gpu_memory()
    
    return scenes


def extract_keyframes(video_path: str, 
                     scenes: List[Scene],
                     strategy: str = 'representative',
                     max_frames_per_scene: int = 3,
                     output_dir: Optional[str] = None) -> List[Scene]:
    """
    Extract key frames from each scene.
    
    Args:
        video_path: Path to the video file
        scenes: List of scenes
        strategy: Strategy for key frame extraction ('first', 'middle', 'representative', 'uniform')
        max_frames_per_scene: Maximum number of frames to extract per scene
        output_dir: Directory to save extracted frames
        
    Returns:
        List[Scene]: Updated list of scenes with key frames
    """
    logger.info(f"Extracting key frames using strategy: {strategy}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return scenes
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each scene
    for i, scene in enumerate(scenes):
        logger.info(f"Processing scene {i+1}/{len(scenes)}")
        
        # Calculate frame numbers for this scene
        start_frame = int(scene.start_time * fps)
        end_frame = int(scene.end_time * fps)
        scene_frames = end_frame - start_frame
        
        # Skip if scene is too short
        if scene_frames <= 0:
            logger.warning(f"Scene {i} has no frames, skipping")
            continue
        
        # Determine which frames to extract based on strategy
        frames_to_extract = []
        
        if strategy == 'first':
            # Extract first frame
            frames_to_extract = [start_frame]
        
        elif strategy == 'middle':
            # Extract middle frame
            middle_frame = start_frame + scene_frames // 2
            frames_to_extract = [middle_frame]
        
        elif strategy == 'uniform':
            # Extract frames uniformly distributed across the scene
            if scene_frames <= max_frames_per_scene:
                # If scene is short, extract all frames
                frames_to_extract = list(range(start_frame, end_frame + 1))
            else:
                # Extract frames uniformly
                step = scene_frames // max_frames_per_scene
                frames_to_extract = [start_frame + i * step for i in range(max_frames_per_scene)]
        
        elif strategy == 'representative':
            # TODO: Implement more sophisticated representative frame selection
            # For now, use uniform sampling as a placeholder
            if scene_frames <= max_frames_per_scene:
                frames_to_extract = list(range(start_frame, end_frame + 1))
            else:
                step = scene_frames // max_frames_per_scene
                frames_to_extract = [start_frame + i * step for i in range(max_frames_per_scene)]
        
        else:
            logger.error(f"Unknown key frame extraction strategy: {strategy}")
            return scenes
        
        # Extract the selected frames
        key_frames = []
        for frame_num in frames_to_extract:
            # Set position to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_num}")
                continue
            
            # Calculate timestamp
            timestamp = frame_num / fps
            
            # Save frame if output directory is specified
            frame_path = None
            if output_dir:
                frame_path = os.path.join(output_dir, f"scene_{i:04d}_frame_{len(key_frames):04d}.jpg")
                cv2.imwrite(frame_path, frame)
            
            # Create Frame object
            key_frame = Frame(
                timestamp=timestamp,
                file_path=frame_path,
                faces=[],  # Will be populated later
                tags=[],  # Will be populated later
            )
            
            key_frames.append(key_frame)
        
        # Update scene with key frames
        scene.key_frames = key_frames
    
    # Release video capture
    cap.release()
    
    logger.info(f"Extracted {sum(len(scene.key_frames) for scene in scenes)} key frames")
    
    return scenes


def batch_process_frames(frames: List[np.ndarray], 
                        batch_size: int = 16) -> List[np.ndarray]:
    """
    Process frames in batches for GPU efficiency.
    
    Args:
        frames: List of frames as numpy arrays
        batch_size: Batch size for processing
        
    Returns:
        List[np.ndarray]: Processed frames
    """
    # This is a placeholder for batch processing logic
    # In a real implementation, this would apply some GPU-accelerated
    # processing to the frames in batches
    
    processed_frames = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Convert batch to tensor
        if USE_GPU:
            # Move batch to GPU
            batch_tensor = torch.stack([torch.from_numpy(frame).to(DEVICE) for frame in batch])
            
            # Apply processing (placeholder)
            # In a real implementation, this would apply some model or processing
            processed_batch = batch_tensor
            
            # Move back to CPU and convert to numpy
            processed_batch = processed_batch.cpu().numpy()
        else:
            # CPU processing (placeholder)
            processed_batch = batch
        
        processed_frames.extend(processed_batch)
    
    return processed_frames