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
from enum import Enum, auto
from collections import defaultdict

import cv2
import numpy as np
import torch
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector
from scenedetect.scene_detector import SceneDetector
from scenedetect.stats_manager import StatsManager
from scenedetect.frame_timecode import FrameTimecode

from src.models.schema import Scene, Frame, VideoMetadata
from src.utils.gpu_utils import setup_device, memory_stats, clear_gpu_memory

logger = logging.getLogger(__name__)

# Get the optimal device (GPU if available, otherwise CPU)
DEVICE = setup_device()
USE_GPU = DEVICE.type != 'cpu'

class DetectionMethod(Enum):
    """
    Enum for scene detection methods.
    """
    CONTENT = auto()
    THRESHOLD = auto()
    HYBRID = auto()
    CUSTOM = auto()

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
        super().__init__(threshold=threshold, min_scene_len=min_scene_len)
        self.weights = weights if weights is not None else [0.5, 0.3, 0.2]
        self.luma_only = luma_only
        self.kernel_size = kernel_size
        self.last_hist = None
        self.last_hsv = None
        self.max_score = 0.0
        self.last_scores = []  # Keep track of recent scores
        self.score_window = 5  # Number of frames to average
        self.use_gpu = USE_GPU
        
        if self.use_gpu:
            logger.info(f"Using GPU ({DEVICE.type}) for content detection")
            self.weights_tensor = torch.tensor(self.weights, device=DEVICE)
        else:
            logger.info("Using CPU for content detection")
    
    def _calculate_frame_score(self, frame_img: Union[np.ndarray, int], frame_num: int) -> float:
        """
        Calculate frame score using GPU acceleration if available.
        
        Args:
            frame_img: Frame image as numpy array or frame number
            frame_num: Frame number
            
        Returns:
            float: Frame score
        """
        if isinstance(frame_img, int):
            return 0.0
        
        if len(frame_img.shape) == 2 or frame_img.shape[2] == 1:
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
        
        # Convert frame to HSV color space
        if self.luma_only:
            frame_hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2YUV)[:, :, 0]
        else:
            frame_hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur if kernel size is specified
        if self.kernel_size is not None:
            frame_hsv = cv2.GaussianBlur(frame_hsv, self.kernel_size, 0)
        
        if self.use_gpu:
            # Move frame to GPU and calculate histogram
            if self.luma_only:
                frame_tensor = torch.from_numpy(frame_hsv).to(DEVICE)
                hist = torch.histc(frame_tensor.float(), bins=256, min=0, max=255)
                hist = hist / hist.sum()  # Normalize
                
                if frame_num == 0:
                    self.last_hist = hist
                    return 0.0
                
                # Calculate difference and apply scaling
                score = torch.sum(torch.abs(hist - self.last_hist)).item() * 100.0
                
                # Update last histogram
                self.last_hist = hist
                
            else:
                frame_tensor = torch.from_numpy(frame_hsv).to(DEVICE)
                h_hist = torch.histc(frame_tensor[:, :, 0].float(), bins=256, min=0, max=255)
                s_hist = torch.histc(frame_tensor[:, :, 1].float(), bins=256, min=0, max=255)
                v_hist = torch.histc(frame_tensor[:, :, 2].float(), bins=256, min=0, max=255)
                
                # Normalize histograms
                h_hist = h_hist / h_hist.sum()
                s_hist = s_hist / s_hist.sum()
                v_hist = v_hist / v_hist.sum()
                
                if frame_num == 0:
                    self.last_hsv = (h_hist, s_hist, v_hist)
                    return 0.0
                
                # Calculate weighted difference
                h_diff = torch.sum(torch.abs(h_hist - self.last_hsv[0]))
                s_diff = torch.sum(torch.abs(s_hist - self.last_hsv[1]))
                v_diff = torch.sum(torch.abs(v_hist - self.last_hsv[2]))
                
                score = (h_diff * self.weights_tensor[0] + 
                        s_diff * self.weights_tensor[1] + 
                        v_diff * self.weights_tensor[2])
                
                score = score.item() * 100.0
                
                # Update last histograms
                self.last_hsv = (h_hist, s_hist, v_hist)
            
            # Keep track of recent scores
            self.last_scores.append(score)
            if len(self.last_scores) > self.score_window:
                self.last_scores.pop(0)
            
            # Calculate average score over window
            avg_score = sum(self.last_scores) / len(self.last_scores)
            
            # Log scores periodically
            if frame_num % 100 == 0:
                logger.debug(f"Frame {frame_num} - Raw score: {score:.2f}, Avg score: {avg_score:.2f}")
            
            # Track maximum score
            if avg_score > self.max_score:
                self.max_score = avg_score
                logger.debug(f"New max score: {avg_score:.2f} at frame {frame_num}")
            
            # Return the average score
            return avg_score
        
        # Fall back to CPU implementation
        return super()._calculate_frame_score(frame_img, frame_num)

class GPUAcceleratedSceneDetector:
    """
    GPU-accelerated scene detector that wraps different detection methods.
    """
    
    def __init__(self, 
                 detection_method: Union[DetectionMethod, str] = DetectionMethod.CONTENT,
                 threshold: float = 30.0,
                 min_scene_length: float = 1.0,
                 luma_only: bool = False,
                 use_gpu: bool = True):
        """
        Initialize the scene detector.
        
        Args:
            detection_method: Method to use for scene detection (DetectionMethod enum or string)
            threshold: Threshold for scene cut detection
            min_scene_length: Minimum scene length in seconds
            luma_only: Whether to use only luma channel for comparison
            use_gpu: Whether to use GPU acceleration if available
        """
        # Convert string to enum if needed
        if isinstance(detection_method, str):
            try:
                self.detection_method = DetectionMethod[detection_method.upper()]
            except KeyError:
                logger.warning(f"Unknown detection method: {detection_method}, using CONTENT")
                self.detection_method = DetectionMethod.CONTENT
        else:
            self.detection_method = detection_method
            
        self.threshold = threshold
        self.min_scene_len = min_scene_length
        self.luma_only = luma_only
        self.use_gpu = use_gpu and USE_GPU  # Only use GPU if available
        
        logger.info(f"Initializing GPU-accelerated scene detector:")
        logger.info(f"  Method: {self.detection_method.name}")
        logger.info(f"  Threshold: {threshold}")
        logger.info(f"  Min Scene Length: {min_scene_length}s")
        logger.info(f"  Using GPU: {self.use_gpu}")
    
    def detect_scenes(self, video_path: str, stats_file: Optional[str] = None) -> List[Scene]:
        """
        Detect scenes in a video.
        
        Args:
            video_path: Path to the video file
            stats_file: Path to save detection statistics
            
        Returns:
            List[Scene]: List of detected scenes
        """
        method_str = self.detection_method.name.lower()
        return detect_scenes(
            video_path=video_path,
            threshold=self.threshold,
            min_scene_len=self.min_scene_len,
            method=method_str,
            luma_only=self.luma_only,
            stats_file=stats_file
        )

def detect_scenes(video_path: str,
                 method: Union[DetectionMethod, str] = DetectionMethod.CONTENT,
                 threshold: float = 30.0,
                 min_scene_len: float = 1.0,
                 luma_only: bool = False,
                 stats_file: Optional[str] = None) -> List[Scene]:
    """
    Detect scenes in a video using the specified method.
    
    Args:
        video_path: Path to the video file
        method: Detection method to use
        threshold: Threshold for scene cut detection (higher values = less sensitive)
        min_scene_len: Minimum scene length in seconds
        luma_only: Whether to use only luma channel for comparison
        stats_file: Path to save detection stats
        
    Returns:
        List[Scene]: List of detected scenes
    """
    # Convert method to enum if string
    if isinstance(method, str):
        method = DetectionMethod[method.upper()]
    
    # Log detection parameters
    logger.info(f"Detecting scenes in video: {video_path}")
    logger.info(f"Method: {method.name.lower()}, Threshold: {threshold}, Min Scene Length: {min_scene_len}s")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create stats manager and scene manager
    base_timecode = FrameTimecode(timecode=0, fps=fps)
    stats_manager = StatsManager(base_timecode=base_timecode)
    scene_manager = SceneManager(stats_manager=stats_manager)
    
    # Convert min scene length to frames
    min_scene_len_frames = int(min_scene_len * fps)
    
    # Configure detector based on method
    if method == DetectionMethod.CONTENT:
        if USE_GPU:
            logger.info("Using GPU (cuda) for content detection")
            detector = GPUContentDetector(
                threshold=threshold * 0.5,  # Adjust threshold for better sensitivity
                min_scene_len=min_scene_len_frames,
                luma_only=luma_only
            )
        else:
            logger.info("Using CPU for content detection")
            detector = ContentDetector(
                threshold=threshold,
                min_scene_len=min_scene_len_frames
            )
        detector.stats_manager = stats_manager
        scene_manager.add_detector(detector)
    
    elif method == DetectionMethod.THRESHOLD:
        detector = ThresholdDetector(
            threshold=threshold * 0.1,  # Lower threshold for better sensitivity
            min_scene_len=min_scene_len_frames,
            fps=fps
        )
        detector.set_stats_manager(stats_manager)
        scene_manager.add_detector(detector)
    
    elif method == DetectionMethod.HYBRID:
        # Use both content and threshold detectors with adjusted thresholds
        content_detector = GPUContentDetector(
            threshold=threshold * 0.5,  # Lower threshold for content detector
            min_scene_len=min_scene_len_frames,
            luma_only=luma_only
        )
        threshold_detector = ThresholdDetector(
            threshold=threshold * 0.1,  # Lower threshold for threshold detector
            min_scene_len=min_scene_len_frames,
            fps=fps
        )
        content_detector.stats_manager = stats_manager
        threshold_detector.set_stats_manager(stats_manager)
        scene_manager.add_detector(content_detector)
        scene_manager.add_detector(threshold_detector)
    else:
        logger.error(f"Unknown detection method: {method}")
        cap.release()
        return []
    
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
                logger.debug(f"GPU Memory: {mem_stats.get('allocated', 0):.2f}GB allocated")
    
    # Process frames
    scene_cuts = []
    frame_num = 0
    last_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with all detectors
        for detector in scene_manager._detector_list:
            if detector.is_processing_required:
                if last_frame is not None:
                    cuts = detector.process_frame(frame, frame_num)
                    if cuts:
                        scene_cuts.extend(cuts)
                last_frame = frame
                    
        frame_num += 1
        
        # Update progress
        if frame_num % 500 == 0 or frame_num == frame_count - 1:
            progress_callback(frame_num, frame_count)
    
    # Release video capture
    cap.release()
    
    # Post-process with all detectors
    for detector in scene_manager._detector_list:
        cuts = detector.post_process(frame_num)
        if cuts:
            scene_cuts.extend(cuts)
    
    # Sort and deduplicate scene cuts
    scene_cuts = sorted(list(set(scene_cuts)))
    
    # Convert scene cuts to scenes
    scenes = []
    for i in range(len(scene_cuts)):
        start_frame = scene_cuts[i-1] if i > 0 else 0
        end_frame = scene_cuts[i]
        
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        scene = Scene(
            id=i,
            start_time=start_time,
            end_time=end_time,
            key_frames=[],  # Will be populated later
            transcript_segments=[],  # Will be populated later
            audio_events=[],  # Will be populated later
            tags=[],  # Will be populated later
        )
        scenes.append(scene)
    
    # Add final scene if needed
    if scene_cuts:
        start_frame = scene_cuts[-1]
        end_frame = frame_count - 1
        
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        scene = Scene(
            id=len(scenes),
            start_time=start_time,
            end_time=end_time,
            key_frames=[],
            transcript_segments=[],
            audio_events=[],
            tags=[],
        )
        scenes.append(scene)
    elif frame_count > 0:
        # If no cuts were detected, treat the entire video as one scene
        scene = Scene(
            id=0,
            start_time=0,
            end_time=frame_count / fps,
            key_frames=[],
            transcript_segments=[],
            audio_events=[],
            tags=[],
        )
        scenes.append(scene)
    
    logger.info(f"Detected {len(scenes)} scenes")
    
    # Save stats if requested
    if stats_file:
        stats_manager.save_to_csv(stats_file)
    
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

class ThresholdDetector(SceneDetector):
    """
    Detector that uses a simple threshold on the mean pixel intensity difference
    between consecutive frames to detect scene cuts.
    """
    def __init__(self, threshold: float = 30.0, min_scene_len: int = 15, fps: Optional[float] = None):
        super().__init__()
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.last_frame = None
        self.last_frame_num = None
        self.stats_manager = None
        self.fps = fps
        self._is_processing_required = True
        self._last_scene_cut = 0

    def process_frame(self, frame_img: np.ndarray, frame_num: Union[int, Tuple[int, int]]) -> List[int]:
        """Process a frame and detect scene cuts."""
        # Get the actual frame number if it's a tuple
        current_frame = frame_num[0] if isinstance(frame_num, tuple) else frame_num

        # Skip if this is the first frame
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            self.last_frame_num = current_frame
            return []

        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference between frames
        frame_delta = cv2.absdiff(self.last_frame, gray_frame)
        score = float(np.mean(frame_delta))

        # Save score in stats manager
        if self.stats_manager is not None:
            metric_dict = {'delta_mean': score}
            if self.fps is not None:
                frame_timecode = FrameTimecode(current_frame, self.fps)
                self.stats_manager.register_metrics(frame_timecode=frame_timecode, metric_dict=metric_dict)
            else:
                self.stats_manager.register_metrics(frame_timecode=None, metric_dict=metric_dict)

        # Check if this frame is a scene cut
        cut_list = []
        if score >= self.threshold and (current_frame - self._last_scene_cut) >= self.min_scene_len:
            cut_list.append(current_frame)
            self._last_scene_cut = current_frame

        # Update last frame
        self.last_frame = gray_frame
        self.last_frame_num = current_frame

        return cut_list

    def post_process(self, frame_num: int) -> List[int]:
        """Perform any post-processing steps and return a final list of cuts."""
        return []

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get the metrics associated with this detector."""
        return {'delta_mean': []}

    def set_stats_manager(self, stats_manager: StatsManager):
        """Set the stats manager for this detector."""
        self.stats_manager = stats_manager

    @property
    def is_processing_required(self) -> bool:
        """Check if processing is required."""
        return self._is_processing_required

class StatsManager:
    """Manages frame metrics for scene detection."""
    
    def __init__(self, base_timecode: Optional[FrameTimecode] = None):
        """Initialize the stats manager.
        
        Args:
            base_timecode: Base timecode for frame timing
        """
        self.metrics = defaultdict(list)
        self.base_timecode = base_timecode
        self._base_timecode = base_timecode  # For compatibility with PySceneDetect
    
    def register_metrics(self, frame_timecode: Optional[FrameTimecode] = None, metric_dict: Optional[Dict[str, float]] = None):
        """Register metrics for a frame.
        
        Args:
            frame_timecode: Frame timecode (optional)
            metric_dict: Dictionary of metrics to register
        """
        if metric_dict is None:
            return
            
        if frame_timecode is None:
            frame_timecode = self.base_timecode
        
        if frame_timecode is None:
            logger.warning("No frame timecode available for metrics registration")
            return
        
        for metric_key, value in metric_dict.items():
            self.metrics[metric_key].append(value)
    
    def metrics_exist(self, metric_key: str, frame_number: Optional[int] = None, metric_dict: Optional[Dict[str, float]] = None) -> bool:
        """Check if metrics exist for a given key.
        
        Args:
            metric_key: Key to check
            frame_number: Optional frame number to check
            metric_dict: Optional metric dictionary to check
            
        Returns:
            bool: True if metrics exist
        """
        if metric_dict is not None and metric_key in metric_dict:
            return True
        if frame_number is not None and metric_key in self.metrics:
            return len(self.metrics[metric_key]) > frame_number
        return metric_key in self.metrics and len(self.metrics[metric_key]) > 0
    
    def save_to_csv(self, csv_file: str):
        """Save metrics to a CSV file.
        
        Args:
            csv_file: Path to save CSV file
        """
        import pandas as pd
        
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Save to CSV
        df.to_csv(csv_file, index=False)