"""
Scene Detection Module.

This module provides GPU-accelerated scene detection capabilities using PySceneDetect
with custom enhancements for improved performance.

📌 Purpose: Detect scene boundaries in videos with GPU acceleration
🔄 Latest Changes: Initial implementation with GPU optimization
⚙️ Key Logic: Uses content-aware and threshold-based detection methods
📂 Expected File Path: src/video_processing/scene_detection.py
🧠 Reasoning: Scene detection is compute-intensive and benefits greatly from GPU
              acceleration, especially for content-aware algorithms
"""

import os
import logging
import tempfile
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
import concurrent.futures
from pathlib import Path

import cv2
import numpy as np
import torch
from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector, ThresholdDetector
from scenedetect.scene_detector import SceneDetector
from scenedetect.stats_manager import StatsManager

from src.models.schema import Scene, Frame, AnalysisConfig, VideoMetadata
from src.utils.gpu_utils import get_optimal_device, clear_gpu_memory
from src.video_processing.loader import extract_video_metadata, create_video_capture, extract_frame, save_frame

logger = logging.getLogger(__name__)

# Get the optimal device (GPU if available, otherwise CPU)
DEVICE = get_optimal_device()


class DetectionMethod(str, Enum):
    """Scene detection methods."""
    
    CONTENT = "content"
    THRESHOLD = "threshold"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class GPUContentDetector:
    """Custom content detector with GPU acceleration for frame comparison."""
    
    def __init__(self, threshold: float = 30.0, batch_size: int = 8, use_gpu: bool = True):
        """
        Initialize the GPU-accelerated content detector.
        
        Args:
            threshold: Threshold for scene change detection
            batch_size: Number of frames to process at once
            use_gpu: Whether to use GPU acceleration
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = get_optimal_device() if use_gpu else torch.device('cpu')
        self.prev_frame_hsv = None
        self.diff_scaler = 1.0 / 255.0 * 100.0
        
        logger.info(f"Initialized GPU content detector (device: {self.device}, threshold: {threshold})")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Process a frame and compute content difference from previous frame.
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Tuple containing:
            - Content difference score
            - Visualization frame (if requested)
        """
        with torch.no_grad():
            # Convert frame to HSV
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if self.prev_frame_hsv is None:
                self.prev_frame_hsv = frame_hsv
                return 0.0, None
            
            # Convert frames to tensors
            curr_tensor = torch.from_numpy(frame_hsv).to(self.device).float()
            prev_tensor = torch.from_numpy(self.prev_frame_hsv).to(self.device).float()
            
            # Calculate mean absolute difference for each channel
            abs_diff = torch.abs(curr_tensor - prev_tensor)
            mean_diff = torch.mean(abs_diff, dim=(0, 1))
            
            # Weight the channels (higher weight to hue and saturation)
            weighted_diff = mean_diff[0] * 0.5 + mean_diff[1] * 0.3 + mean_diff[2] * 0.2
            content_val = weighted_diff.item() * self.diff_scaler
            
            # Update previous frame
            self.prev_frame_hsv = frame_hsv
            
            return content_val, None
    
    def is_scene_change(self, score: float) -> bool:
        """
        Determine if a content difference score indicates a scene change.
        
        Args:
            score: Content difference score
            
        Returns:
            bool: True if score indicates a scene change
        """
        return score >= self.threshold
    
    def reset(self):
        """Reset the detector state."""
        self.prev_frame_hsv = None
        # Clear GPU memory if needed
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


class GPUAcceleratedSceneDetector:
    """Scene detector with GPU acceleration capabilities."""
    
    def __init__(
        self,
        detection_method: Union[str, DetectionMethod] = DetectionMethod.CONTENT,
        threshold: float = 30.0,
        min_scene_length: float = 1.0,
        batch_size: int = 8,
        use_gpu: bool = True
    ):
        """
        Initialize the scene detector.
        
        Args:
            detection_method: Method for scene detection
            threshold: Threshold for scene change detection
            min_scene_length: Minimum scene length in seconds
            batch_size: Number of frames to process at once
            use_gpu: Whether to use GPU acceleration
        """
        self.detection_method = detection_method if isinstance(detection_method, DetectionMethod) else DetectionMethod(detection_method)
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.batch_size = batch_size
        self.use_gpu = use_gpu and DEVICE.type in ['cuda', 'mps']
        
        logger.info(f"Initialized scene detector (method: {self.detection_method}, threshold: {threshold}, GPU: {self.use_gpu})")
    
    def detect_scenes(
        self, 
        video_path: str, 
        stats_file: Optional[str] = None,
        extract_frames: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Scene]:
        """
        Detect scenes in a video.
        
        Args:
            video_path: Path to the video file
            stats_file: Path to save detection statistics
            extract_frames: Whether to extract key frames for scenes
            output_dir: Directory to save extracted frames
            
        Returns:
            List of Scene objects
        """
        # Extract video metadata
        video_metadata = extract_video_metadata(video_path)
        if video_metadata is None:
            logger.error(f"Failed to extract metadata from video: {video_path}")
            return []
        
        # Create output directory if needed
        if extract_frames and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Choose detection method
        if self.detection_method == DetectionMethod.CONTENT:
            scenes = self._detect_content_aware(video_path, video_metadata, stats_file)
        elif self.detection_method == DetectionMethod.THRESHOLD:
            scenes = self._detect_threshold(video_path, video_metadata, stats_file)
        elif self.detection_method == DetectionMethod.HYBRID:
            scenes = self._detect_hybrid(video_path, video_metadata, stats_file)
        elif self.detection_method == DetectionMethod.CUSTOM:
            scenes = self._detect_custom_gpu(video_path, video_metadata)
        else:
            logger.error(f"Unknown detection method: {self.detection_method}")
            return []
        
        # Extract key frames if requested
        if extract_frames and scenes and output_dir:
            self._extract_scene_frames(video_path, scenes, output_dir)
        
        return scenes
    
    def _detect_content_aware(
        self, 
        video_path: str, 
        video_metadata: VideoMetadata,
        stats_file: Optional[str] = None
    ) -> List[Scene]:
        """
        Detect scenes using PySceneDetect's content-aware detector.
        
        Args:
            video_path: Path to the video file
            video_metadata: Video metadata
            stats_file: Path to save detection statistics
            
        Returns:
            List of Scene objects
        """
        logger.info(f"Using content-aware scene detection on: {video_path}")
        
        # Create video manager and stats manager
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        
        # Add stats file if provided
        if stats_file:
            stats_manager.save_to_csv(stats_file)
        
        # Create scene manager
        scene_manager = SceneManager(stats_manager)
        
        # Add content detector
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        # Start video manager and perform detection
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        # Convert to our Scene model
        scenes = []
        for i, (start_time, end_time) in enumerate(scene_list):
            start_time_secs = start_time.get_seconds()
            end_time_secs = end_time.get_seconds()
            
            # Skip scenes that are too short
            if end_time_secs - start_time_secs < self.min_scene_length:
                continue
            
            scene = Scene(
                id=i + 1,  # 1-based IDs
                start_time=start_time_secs,
                end_time=end_time_secs,
                key_frames=[]
            )
            scenes.append(scene)
        
        logger.info(f"Detected {len(scenes)} scenes")
        return scenes
    
    def _detect_threshold(
        self, 
        video_path: str, 
        video_metadata: VideoMetadata,
        stats_file: Optional[str] = None
    ) -> List[Scene]:
        """
        Detect scenes using PySceneDetect's threshold detector.
        
        Args:
            video_path: Path to the video file
            video_metadata: Video metadata
            stats_file: Path to save detection statistics
            
        Returns:
            List of Scene objects
        """
        logger.info(f"Using threshold-based scene detection on: {video_path}")
        
        # Create video manager and stats manager
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        
        # Add stats file if provided
        if stats_file:
            stats_manager.save_to_csv(stats_file)
        
        # Create scene manager
        scene_manager = SceneManager(stats_manager)
        
        # Add threshold detector
        scene_manager.add_detector(ThresholdDetector(threshold=self.threshold, min_percent=0.95))
        
        # Start video manager and perform detection
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        # Convert to our Scene model
        scenes = []
        for i, (start_time, end_time) in enumerate(scene_list):
            start_time_secs = start_time.get_seconds()
            end_time_secs = end_time.get_seconds()
            
            # Skip scenes that are too short
            if end_time_secs - start_time_secs < self.min_scene_length:
                continue
            
            scene = Scene(
                id=i + 1,  # 1-based IDs
                start_time=start_time_secs,
                end_time=end_time_secs,
                key_frames=[]
            )
            scenes.append(scene)
        
        logger.info(f"Detected {len(scenes)} scenes")
        return scenes
    
    def _detect_hybrid(
        self, 
        video_path: str, 
        video_metadata: VideoMetadata,
        stats_file: Optional[str] = None
    ) -> List[Scene]:
        """
        Detect scenes using both content and threshold detectors.
        
        Args:
            video_path: Path to the video file
            video_metadata: Video metadata
            stats_file: Path to save detection statistics
            
        Returns:
            List of Scene objects
        """
        logger.info(f"Using hybrid scene detection on: {video_path}")
        
        # Get scenes from both methods
        content_scenes = self._detect_content_aware(video_path, video_metadata, stats_file)
        
        # Temporarily change threshold for threshold detection
        original_threshold = self.threshold
        self.threshold = original_threshold * 1.5  # Higher threshold for more conservative cuts
        threshold_scenes = self._detect_threshold(video_path, video_metadata, None)
        self.threshold = original_threshold  # Restore original threshold
        
        # Merge scenes from both methods
        all_boundaries = set()
        
        # Add boundaries from content detector
        for scene in content_scenes:
            all_boundaries.add(scene.start_time)
            all_boundaries.add(scene.end_time)
        
        # Add boundaries from threshold detector
        for scene in threshold_scenes:
            all_boundaries.add(scene.start_time)
            all_boundaries.add(scene.end_time)
        
        # Convert to sorted list and create scenes
        boundaries = sorted(list(all_boundaries))
        
        scenes = []
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            
            # Skip scenes that are too short
            if end_time - start_time < self.min_scene_length:
                continue
            
            scene = Scene(
                id=i + 1,  # 1-based IDs
                start_time=start_time,
                end_time=end_time,
                key_frames=[]
            )
            scenes.append(scene)
        
        logger.info(f"Detected {len(scenes)} scenes using hybrid approach")
        return scenes
    
    def _detect_custom_gpu(
        self, 
        video_path: str, 
        video_metadata: VideoMetadata
    ) -> List[Scene]:
        """
        Detect scenes using custom GPU-accelerated implementation.
        
        Args:
            video_path: Path to the video file
            video_metadata: Video metadata
            
        Returns:
            List of Scene objects
        """
        logger.info(f"Using custom GPU-accelerated scene detection on: {video_path}")
        
        # Create video capture
        cap = create_video_capture(video_path)
        if cap is None:
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Create detector
        detector = GPUContentDetector(threshold=self.threshold, use_gpu=self.use_gpu)
        
        # Process frames
        frame_idx = 0
        scene_boundaries = [0.0]  # Start with the beginning of the video
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                score, _ = detector.process_frame(frame)
                
                # Check for scene change
                if frame_idx > 0 and detector.is_scene_change(score):
                    time_secs = frame_idx / fps
                    logger.debug(f"Scene change detected at {time_secs:.2f}s (frame {frame_idx}, score: {score:.2f})")
                    scene_boundaries.append(time_secs)
                
                frame_idx += 1
                
                # Print progress
                if frame_idx % 100 == 0:
                    progress = frame_idx / frame_count * 100 if frame_count > 0 else 0
                    logger.debug(f"Processing: {progress:.1f}% ({frame_idx}/{frame_count})")
        
        finally:
            # Release resources
            cap.release()
            detector.reset()
            if self.use_gpu:
                clear_gpu_memory()
        
        # Add the end of the video as the final boundary
        scene_boundaries.append(duration)
        
        # Create scenes
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            start_time = scene_boundaries[i]
            end_time = scene_boundaries[i + 1]
            
            # Skip scenes that are too short
            if end_time - start_time < self.min_scene_length:
                continue
            
            scene = Scene(
                id=i + 1,  # 1-based IDs
                start_time=start_time,
                end_time=end_time,
                key_frames=[]
            )
            scenes.append(scene)
        
        logger.info(f"Detected {len(scenes)} scenes with custom GPU method")
        return scenes
    
    def _extract_scene_frames(
        self, 
        video_path: str, 
        scenes: List[Scene], 
        output_dir: str,
        frame_extraction_method: str = "representative"
    ) -> None:
        """
        Extract key frames for each scene.
        
        Args:
            video_path: Path to the video file
            scenes: List of Scene objects
            output_dir: Directory to save extracted frames
            frame_extraction_method: Method for key frame extraction
        """
        logger.info(f"Extracting key frames for {len(scenes)} scenes")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create video capture
        cap = create_video_capture(video_path)
        if cap is None:
            logger.error(f"Failed to open video for frame extraction: {video_path}")
            return
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for scene in scenes:
                # Determine frame extraction strategy
                if frame_extraction_method == "first":
                    # Extract first frame of scene
                    timestamps = [scene.start_time]
                elif frame_extraction_method == "middle":
                    # Extract middle frame of scene
                    timestamps = [(scene.start_time + scene.end_time) / 2]
                elif frame_extraction_method == "representative":
                    # Extract multiple frames from the scene
                    scene_duration = scene.end_time - scene.start_time
                    frames_to_extract = max(1, min(3, int(scene_duration)))
                    
                    if frames_to_extract == 1:
                        timestamps = [(scene.start_time + scene.end_time) / 2]
                    else:
                        # Evenly spaced frames
                        step = scene_duration / (frames_to_extract + 1)
                        timestamps = [scene.start_time + step * (i + 1) for i in range(frames_to_extract)]
                else:
                    logger.error(f"Unknown frame extraction method: {frame_extraction_method}")
                    timestamps = [scene.start_time]
                
                # Extract frames
                scene_frames = []
                for i, timestamp in enumerate(timestamps):
                    frame = extract_frame(cap, timestamp)
                    if frame is not None:
                        # Save frame
                        frame_path = os.path.join(
                            output_dir, 
                            f"scene_{scene.id:04d}_frame_{i:02d}_{timestamp:.2f}s.jpg"
                        )
                        success = save_frame(frame, frame_path)
                        
                        if success:
                            # Add to scene
                            scene_frames.append(Frame(
                                timestamp=timestamp,
                                file_path=frame_path
                            ))
                            logger.debug(f"Saved frame: {frame_path}")
                        else:
                            logger.error(f"Failed to save frame: {frame_path}")
                
                # Update scene with extracted frames
                scene.key_frames = scene_frames
        
        finally:
            # Release resources
            cap.release()
            if self.use_gpu:
                clear_gpu_memory()
        
        logger.info(f"Extracted key frames for {len(scenes)} scenes")


def detect_scenes_batch(
    video_paths: List[str],
    output_dir: str,
    detection_method: str = "content",
    threshold: float = 30.0,
    min_scene_length: float = 1.0,
    extract_frames: bool = True,
    use_gpu: bool = True,
    max_workers: int = 1
) -> Dict[str, List[Scene]]:
    """
    Detect scenes in multiple videos in parallel.
    
    Args:
        video_paths: List of paths to video files
        output_dir: Directory to save output
        detection_method: Method for scene detection
        threshold: Threshold for scene change detection
        min_scene_length: Minimum scene length in seconds
        extract_frames: Whether to extract key frames
        use_gpu: Whether to use GPU acceleration
        max_workers: Maximum number of workers for parallel processing
        
    Returns:
        Dictionary mapping video paths to lists of Scene objects
    """
    logger.info(f"Detecting scenes in {len(video_paths)} videos")
    
    # Disable GPU parallelism if max_workers > 1 (to avoid OOM)
    parallel_use_gpu = use_gpu and max_workers == 1
    
    # Process videos
    results = {}
    
    if max_workers > 1:
        # Process videos in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for video_path in video_paths:
                # Create output directory for this video
                video_name = Path(video_path).stem
                video_output_dir = os.path.join(output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                
                # Create detector for this video
                detector = GPUAcceleratedSceneDetector(
                    detection_method=detection_method,
                    threshold=threshold,
                    min_scene_length=min_scene_length,
                    use_gpu=parallel_use_gpu
                )
                
                # Submit task
                futures[executor.submit(
                    detector.detect_scenes,
                    video_path,
                    None,
                    extract_frames,
                    video_output_dir if extract_frames else None
                )] = video_path
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                video_path = futures[future]
                try:
                    scenes = future.result()
                    results[video_path] = scenes
                except Exception as e:
                    logger.error(f"Error detecting scenes in {video_path}: {str(e)}")
                    results[video_path] = []
    else:
        # Process videos sequentially
        detector = GPUAcceleratedSceneDetector(
            detection_method=detection_method,
            threshold=threshold,
            min_scene_length=min_scene_length,
            use_gpu=use_gpu
        )
        
        for video_path in video_paths:
            # Create output directory for this video
            video_name = Path(video_path).stem
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            try:
                scenes = detector.detect_scenes(
                    video_path,
                    None,
                    extract_frames,
                    video_output_dir if extract_frames else None
                )
                results[video_path] = scenes
            except Exception as e:
                logger.error(f"Error detecting scenes in {video_path}: {str(e)}")
                results[video_path] = []
    
    logger.info(f"Completed scene detection for {len(results)} videos")
    return results