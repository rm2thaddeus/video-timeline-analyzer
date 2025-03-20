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
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video has {total_frames} frames at {fps} FPS")
        
        # Initialize variables
        scenes = []
        current_scene_start = 0
        prev_hist = None
        frame_num = 0
        
        # Process frames
        pbar = tqdm(total=total_frames, desc="Detecting scenes")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale and calculate histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compare with previous histogram
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if diff < (1 - self.threshold) and (frame_num - current_scene_start) >= self.min_scene_len:
                    # Scene change detected
                    scenes.append({
                        "id": f"scene{len(scenes):03d}",
                        "start_frame": current_scene_start,
                        "end_frame": frame_num - 1,
                        "start_time": current_scene_start / fps,
                        "end_time": (frame_num - 1) / fps,
                        "duration": (frame_num - 1 - current_scene_start) / fps
                    })
                    current_scene_start = frame_num
            
            prev_hist = hist
            frame_num += 1
            pbar.update(1)
        
        # Add final scene
        if frame_num - current_scene_start >= self.min_scene_len:
            scenes.append({
                "id": f"scene{len(scenes):03d}",
                "start_frame": current_scene_start,
                "end_frame": frame_num - 1,
                "start_time": current_scene_start / fps,
                "end_time": (frame_num - 1) / fps,
                "duration": (frame_num - 1 - current_scene_start) / fps
            })
        
        # Close video capture
        cap.release()
        pbar.close()
        
        # Save scene data if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            scene_data = {
                "video_id": video_id,
                "total_frames": total_frames,
                "fps": fps,
                "scenes": scenes
            }
            scene_data_path = output_dir / f"{video_id}_scenes.json"
            with open(scene_data_path, "w") as f:
                json.dump(scene_data, f, indent=2)
        
        return scenes 