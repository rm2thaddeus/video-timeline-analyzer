"""
📌 Purpose: Detect scene changes in video files
🔄 Latest Changes: Updated to use black screen detection by default
⚙️ Key Logic: Use OpenCV histogram comparison and black frame detection
📂 Expected File Path: test_pipeline/processors/scene_detector.py
🧠 Reasoning: Enhanced to detect scene cuts via black frames
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("video_pipeline.scene_detector")

class SceneDetector:
    def __init__(
        self, 
        threshold: float = 0.35, 
        min_scene_len: int = 15, 
        black_threshold: float = 0.05,
        device=None
    ):
        """
        Initialize scene detector with detection parameters.
        
        Args:
            threshold: Histogram difference threshold (0.0-1.0)
            min_scene_len: Minimum scene length in frames
            black_threshold: Threshold for black frame detection (0.0-1.0)
            device: Torch device (if provided, used for potential GPU acceleration)
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.black_threshold = black_threshold
        self.device = device
        if self.device is not None and self.device.type == "cuda":
            logger.info("SceneDetector: GPU acceleration not implemented. Running on CPU despite CUDA device.")
        logger.info(f"Scene detector initialized with threshold={threshold}, min_scene_len={min_scene_len}, black_threshold={black_threshold}")
    
    def _is_black_frame(self, frame: np.ndarray) -> bool:
        """
        Check if a frame is predominantly black.
        
        Args:
            frame: Input frame
            
        Returns:
            bool: True if frame is black
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Calculate mean brightness
        mean_brightness = np.mean(gray) / 255.0
        
        return mean_brightness < self.black_threshold
    
    def detect_scenes(self, video_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect scenes in a video using black frame detection and histogram comparison.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the scene boundaries JSON file (optional)
            
        Returns:
            List of scene dictionaries with start/end frame numbers and timestamps
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Create subdirectory for scene screenshots
            screenshots_dir = output_dir / "screenshots"
            screenshots_dir.mkdir(exist_ok=True)
        
        logger.info(f"Detecting scenes in {video_path}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video has {frame_count} frames at {fps} FPS")
            
            # Initialize variables
            prev_hist = None
            scene_boundaries = [0]  # First frame is always a scene boundary
            current_frame = 0
            last_screenshot = None
            
            # Process frames for scene detection
            with tqdm(total=frame_count, desc="Detecting scenes") as pbar:
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                        
                    # Check for black frame
                    is_black = self._is_black_frame(frame)
                    
                    # Convert to grayscale and calculate histogram
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    
                    # Compare with previous histogram
                    if prev_hist is not None:
                        # Calculate histogram correlation
                        corr = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                        
                        # Detect scene boundary if:
                        # 1. Frame is black (scene cut) OR
                        # 2. Significant histogram change AND minimum scene length met
                        if (is_black or corr < (1.0 - self.threshold)) and (current_frame - scene_boundaries[-1]) >= self.min_scene_len:
                            scene_boundaries.append(current_frame)
                            
                            # Save screenshot of the last frame before the cut
                            if output_dir and last_screenshot is not None:
                                screenshot_path = screenshots_dir / f"scene_{len(scene_boundaries)-1:04d}.jpg"
                                cv2.imwrite(str(screenshot_path), last_screenshot)
                            
                            logger.debug(f"Scene boundary detected at frame {current_frame} (time: {current_frame/fps:.2f}s)")
                    
                    prev_hist = hist
                    last_screenshot = frame.copy()  # Store frame for potential screenshot
                    current_frame += 1
                    pbar.update(1)
                    
                    # Log progress periodically
                    if current_frame % 500 == 0:
                        logger.debug(f"Processed {current_frame}/{frame_count} frames ({current_frame/frame_count*100:.1f}%)")
            
            # Add last frame as scene boundary if not already added
            if scene_boundaries[-1] != frame_count - 1:
                scene_boundaries.append(frame_count - 1)
                
                # Save screenshot of the last scene
                if output_dir and last_screenshot is not None:
                    screenshot_path = screenshots_dir / f"scene_{len(scene_boundaries)-1:04d}.jpg"
                    cv2.imwrite(str(screenshot_path), last_screenshot)
            
            cap.release()
            
            # Convert boundaries to scenes
            scenes = []
            for i in range(len(scene_boundaries) - 1):
                start_frame = scene_boundaries[i]
                end_frame = scene_boundaries[i + 1]
                
                # Calculate timestamps
                start_time = start_frame / fps
                end_time = end_frame / fps
                duration = end_time - start_time
                
                scenes.append({
                    "id": i,
                    "video_id": video_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration,
                    "duration_frames": end_frame - start_frame,
                    "fps": fps,
                    "resolution": {"width": width, "height": height},
                    "screenshot_path": str(screenshots_dir / f"scene_{i:04d}.jpg") if output_dir else None
                })
            
            # Save scene boundaries to JSON file if output_dir is provided
            if output_dir:
                output_file = output_dir / f"{video_id}_scenes.json"
                with open(output_file, 'w') as f:
                    json.dump(scenes, f, indent=2)
                logger.info(f"Saved {len(scenes)} scene boundaries to {output_file}")
            
            return scenes
            
        except Exception as e:
            logger.error(f"Error during scene detection: {e}")
            raise 