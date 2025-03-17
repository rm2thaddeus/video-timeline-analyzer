"""
📌 Purpose: Detect scene changes in video files
🔄 Latest Changes: Updated to use only OpenCV for scene detection
⚙️ Key Logic: Use OpenCV histogram comparison to identify scene boundaries
📂 Expected File Path: test_pipeline/processors/scene_detector.py
🧠 Reasoning: Simplified implementation using only OpenCV without PySceneDetect
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
    def __init__(self, threshold: float = 0.35, min_scene_len: int = 15, device=None):
        """
        Initialize scene detector with detection parameters.
        
        Args:
            threshold: Histogram difference threshold (0.0-1.0)
            min_scene_len: Minimum scene length in frames
            device: Torch device (if provided, used for potential GPU acceleration)
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.device = device
        if self.device is not None and self.device.type == "cuda":
            logger.info("SceneDetector: GPU acceleration not implemented. Running on CPU despite CUDA device.")
        logger.info(f"Scene detector initialized with threshold={threshold}, min_scene_len={min_scene_len}")
    
    def detect_scenes(self, video_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect scenes in a video using OpenCV histogram comparison.
        
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
        
        logger.info(f"Detecting scenes with OpenCV in {video_path} (threshold={self.threshold})")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video has {frame_count} frames at {fps} FPS")
            
            # Initialize variables
            prev_hist = None
            scene_boundaries = [0]  # First frame is always a scene boundary
            current_frame = 0
            
            # Process frames for scene detection
            with tqdm(total=frame_count, desc="Detecting scenes") as pbar:
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                        
                    # Convert to grayscale and calculate histogram
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    
                    # Compare with previous histogram
                    if prev_hist is not None:
                        # Calculate histogram correlation (1.0 means identical, 0.0 means completely different)
                        corr = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                        
                        # If correlation is below threshold, mark as scene boundary
                        if corr < (1.0 - self.threshold) and (current_frame - scene_boundaries[-1]) >= self.min_scene_len:
                            scene_boundaries.append(current_frame)
                            logger.debug(f"Scene boundary detected at frame {current_frame} (time: {current_frame/fps:.2f}s)")
                    
                    prev_hist = hist
                    current_frame += 1
                    pbar.update(1)
                    
                    # Log progress periodically
                    if current_frame % 500 == 0:
                        logger.debug(f"Processed {current_frame}/{frame_count} frames ({current_frame/frame_count*100:.1f}%)")
            
            # Add last frame as scene boundary if not already added
            if scene_boundaries[-1] != frame_count - 1:
                scene_boundaries.append(frame_count - 1)
                
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
                    "scene_id": i,
                    "video_id": video_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration
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