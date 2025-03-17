"""
📌 Purpose: CUDA-optimized scene detection with batch processing and checkpointing
🔄 Latest Changes: Implemented batch processing, checkpointing, and proper CUDA utilization
⚙️ Key Logic: Uses PySceneDetect with GPU acceleration and batch processing for memory efficiency
📂 Expected File Path: test_pipeline/CUDA/scene_detector.py
🧠 Reasoning: Optimize scene detection for large videos by breaking processing into manageable chunks with checkpoint support
"""

import os
import sys
import json
import time
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For video processing
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_detector import SceneDetector as PySceneDetector
from scenedetect.frame_timecode import FrameTimecode

logger = logging.getLogger("scene_detector_cuda")

# Make sure we have a console handler
has_console_handler = False
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
        has_console_handler = True
        break

if not has_console_handler:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

class VideoFrameDataset(Dataset):
    """Dataset to load video frames in batches for efficient processing"""
    
    def __init__(self, video_path: str, max_frames: Optional[int] = None, 
                 start_frame: int = 0, batch_size: int = 32):
        """
        Initialize the dataset for loading video frames.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to process (None for all)
            start_frame: Starting frame index (for resuming from checkpoints)
            batch_size: Batch size for loading frames
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set starting frame
        self.start_frame = start_frame
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Set max frames
        if max_frames is not None and max_frames > 0:
            self.max_frames = min(max_frames, self.frame_count - start_frame)
        else:
            self.max_frames = self.frame_count - start_frame
            
        self.batch_size = batch_size
        
        logger.info(f"📽️ Video info: {self.width}x{self.height}, {self.fps}fps, "
                   f"{self.frame_count} total frames, starting at frame {start_frame}")
        if start_frame > 0:
            logger.info(f"⏩ Resuming from frame {start_frame} ({(start_frame/self.frame_count)*100:.1f}% into video)")
    
    def __len__(self):
        return (self.max_frames + self.batch_size - 1) // self.batch_size  # Ceiling division
    
    def __getitem__(self, idx):
        """Get a batch of frames"""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.max_frames)
        
        frames = []
        frame_indices = []
        
        # Read the frames
        for i in range(start_idx, end_idx):
            success, frame = self.cap.read()
            if not success:
                break
            
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
            frame_indices.append(self.start_frame + i)
        
        # Stack the frames
        if len(frames) > 0:
            frames = np.stack(frames)
            # Convert to tensor
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            # Empty tensor with the right shape
            frames = torch.zeros((0, 3, self.height, self.width), dtype=torch.float32)
            
        return {
            "frames": frames,
            "indices": torch.tensor(frame_indices, dtype=torch.long)
        }
    
    def close(self):
        """Close the video capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ContentDetectorCUDA(nn.Module):
    """CUDA-accelerated content detector for scene detection"""
    
    def __init__(self, threshold: float = 30.0):
        """
        Initialize the content detector with the given threshold.
        
        Args:
            threshold: Detection threshold (higher = less sensitive)
        """
        super().__init__()
        self.threshold = threshold
        self.prev_frame = None
        self.prev_hsv = None
        self.prev_gray = None
        
        # For statistics
        self.cuts = []
        
        logger.info(f"🔍 Content detector initialized with threshold {threshold}")
    
    def _calculate_frame_score(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the content score for detecting scene changes.
        
        Args:
            frame: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of scores for each frame in the batch
        """
        batch_size = frame.shape[0]
        if batch_size == 0:
            return torch.tensor([], device=frame.device)
        
        # Create a placeholder for the scores
        scores = torch.zeros(batch_size, device=frame.device)
        
        # Process the first frame separately
        if self.prev_frame is None:
            self.prev_frame = frame[0].unsqueeze(0)
            return scores
        
        # Convert to HSV for better color-based content detection
        # Calculate delta hsv for first frame with previous batch's last frame
        frame_rgb = frame[0].unsqueeze(0)
        delta_hsv = torch.abs(self.prev_frame - frame_rgb).mean(dim=[1, 2, 3])
        scores[0] = delta_hsv * 100.0  # Scale for better comparison with threshold
        
        # Process remaining frames in the batch
        if batch_size > 1:
            # Calculate delta hsv for remaining frames in the batch
            delta_hsv = torch.abs(frame[1:] - frame[:-1]).mean(dim=[1, 2, 3])
            scores[1:] = delta_hsv * 100.0
        
        # Save the last frame of the batch for next batch processing
        self.prev_frame = frame[-1].unsqueeze(0)
        
        return scores
    
    def forward(self, frames: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of frames to detect scene changes.
        
        Args:
            frames: Input tensor of shape [B, C, H, W]
            indices: Tensor of frame indices
            
        Returns:
            Tuple of (detected_indices, scores)
        """
        scores = self._calculate_frame_score(frames)
        
        # Detect cuts based on threshold
        cuts = torch.where(scores > self.threshold)[0]
        
        # Get the corresponding indices
        if len(cuts) > 0:
            cut_indices = indices[cuts]
            # Store statistics
            for idx in cut_indices.cpu().numpy():
                self.cuts.append(int(idx))
            return cut_indices, scores[cuts]
        else:
            return torch.tensor([], device=frames.device, dtype=torch.long), torch.tensor([], device=frames.device)


class SceneDetector:
    """Scene detector for videos using CUDA acceleration and batch processing"""
    
    def __init__(self, threshold: float = 30.0, min_scene_len: int = 15, 
                 device: Optional[torch.device] = None, batch_size: int = 128,
                 checkpoint_interval: int = 5000):
        """
        Initialize the scene detector.
        
        Args:
            threshold: Detection threshold (higher = less sensitive)
            min_scene_len: Minimum scene length in frames
            device: Torch device to use (None for auto-detection)
            batch_size: Number of frames to process at once (adjust based on available VRAM)
            checkpoint_interval: Number of frames after which to save checkpoint
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        # Create CUDA detector if we're on GPU
        if self.device.type == "cuda":
            logger.info(f"🖥️ Using GPU acceleration with batch size {batch_size}")
        else:
            logger.info(f"🖥️ Using CPU with batch size {batch_size}")
        
        self.detector = ContentDetectorCUDA(threshold=threshold)
        
        # Stats
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_frames": 0,
            "processed_frames": 0,
            "cut_frames": [],
            "scenes": []
        }
        
        logger.info(f"⚙️ Scene detector configuration:")
        logger.info(f"  • Threshold: {threshold}")
        logger.info(f"  • Min scene length: {min_scene_len} frames")
        logger.info(f"  • Device: {self.device}")
        logger.info(f"  • Batch size: {batch_size} frames")
        logger.info(f"  • Checkpoint interval: {checkpoint_interval} frames")
    
    def detect_scenes(self, video_path: str, output_dir: Optional[Union[str, Path]] = None, 
                      checkpoint_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect scenes in the video with batch processing and checkpointing.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save scene detection results
            checkpoint_path: Path to load/save checkpoint (None to disable)
            
        Returns:
            List of detected scenes with metadata
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
        # Get video metadata using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Set or create the checkpoint path
        if checkpoint_path is None and output_dir is not None:
            checkpoint_path = output_dir / f"{Path(video_path).stem}_scene_checkpoint.json"
            
        # Initialize or load from checkpoint
        start_frame = 0
        scenes = []
        video_id = Path(video_path).stem
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                start_frame = checkpoint_data.get("processed_frames", 0)
                scenes = checkpoint_data.get("scenes", [])
                
                if start_frame > 0:
                    logger.info(f"⏩ Resuming from checkpoint at frame {start_frame}/{frame_count} "
                               f"({start_frame/frame_count*100:.1f}%)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load checkpoint: {e}")
                start_frame = 0
                scenes = []
        
        # Reset stats
        self.stats = {
            "start_time": time.time(),
            "end_time": None,
            "total_frames": frame_count,
            "processed_frames": 0,
            "cut_frames": [],
            "scenes": []
        }
        
        # Setup detector
        self.detector = ContentDetectorCUDA(threshold=self.threshold).to(self.device)
        
        # Create the dataset and dataloader
        logger.info(f"🎬 Starting scene detection for video: {video_path}")
        logger.info(f"🔄 Processing {frame_count} frames at {fps} fps ({duration:.2f} seconds)")
        
        try:
            dataset = VideoFrameDataset(
                video_path=video_path,
                start_frame=start_frame,
                batch_size=self.batch_size
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=1,  # We're already batching in the dataset
                shuffle=False,
                num_workers=0  # No multiprocessing needed as we batch in dataset
            )
            
            # Detect scenes
            logger.info(f"🔍 Analyzing video in batches of {self.batch_size} frames...")
            
            detection_start_time = time.time()
            frame_times = []  # For tracking performance
            detection_counts = 0  # Number of detected transitions
            last_checkpoint_frame = start_frame
            last_log_time = time.time()
            log_interval = 5.0  # Log progress every 5 seconds
            
            # Process batches
            for i, batch in enumerate(dataloader):
                batch_start_time = time.time()
                frames = batch["frames"].squeeze(0).to(self.device)
                indices = batch["indices"].squeeze(0).to(self.device)
                
                if frames.shape[0] == 0:
                    continue
                
                # Process the batch
                with torch.no_grad():
                    cut_indices, scores = self.detector(frames, indices)
                
                # Update stats
                batch_size = frames.shape[0]
                self.stats["processed_frames"] += batch_size
                detection_counts += len(cut_indices)
                
                if len(cut_indices) > 0:
                    # Add the cut frames to stats
                    self.stats["cut_frames"].extend(cut_indices.cpu().numpy().tolist())
                
                # Calculate performance metrics
                batch_time = time.time() - batch_start_time
                frame_times.append(batch_time / batch_size)  # Time per frame
                
                # Log progress periodically
                current_time = time.time()
                if current_time - last_log_time > log_interval:
                    progress = self.stats["processed_frames"] / frame_count
                    avg_fps = batch_size / batch_time
                    eta = (frame_count - self.stats["processed_frames"]) / avg_fps
                    
                    logger.info(f"🔄 Progress: {progress*100:.1f}% - "
                               f"Frame {self.stats['processed_frames']}/{frame_count} - "
                               f"Found {detection_counts} potential transitions - "
                               f"Processing at {avg_fps:.1f} fps - "
                               f"ETA: {eta:.1f}s")
                    
                    last_log_time = current_time
                
                # Save checkpoint periodically
                if (self.stats["processed_frames"] - last_checkpoint_frame >= self.checkpoint_interval and 
                    checkpoint_path is not None):
                    
                    # Convert cut frames to scenes
                    logger.info(f"💾 Saving checkpoint at frame {self.stats['processed_frames']}...")
                    
                    # Save checkpoint data
                    checkpoint_data = {
                        "video_id": video_id,
                        "processed_frames": self.stats["processed_frames"],
                        "cut_frames": self.detector.cuts,
                        "timestamp": time.time()
                    }
                    
                    try:
                        with open(checkpoint_path, 'w') as f:
                            json.dump(checkpoint_data, f)
                        logger.info(f"✅ Checkpoint saved to {checkpoint_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to save checkpoint: {e}")
                    
                    last_checkpoint_frame = self.stats["processed_frames"]
            
            # Processing complete
            detection_time = time.time() - detection_start_time
            
            dataset.close()
            
            # Update stats
            self.stats["end_time"] = time.time()
            
            # Calculate metrics
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            logger.info(f"✅ Scene detection completed in {detection_time:.2f} seconds")
            logger.info(f"📊 Stats: Processed {self.stats['processed_frames']} frames at {fps:.1f} fps")
            logger.info(f"📊 Found {len(self.detector.cuts)} potential scene transitions")
            
            # Build scenes from detected cuts
            logger.info(f"🔍 Building scene list from {len(self.detector.cuts)} transitions...")
            scenes = self._build_scenes_from_cuts(
                self.detector.cuts, fps, frame_count, width, height
            )
            
            # Save final results
            if output_dir is not None:
                output_file = output_dir / f"{video_id}_scenes.json"
                
                try:
                    with open(output_file, 'w') as f:
                        json.dump({
                            "video_id": video_id,
                            "fps": fps,
                            "frame_count": frame_count,
                            "duration": duration,
                            "resolution": {"width": width, "height": height},
                            "threshold": self.threshold,
                            "min_scene_len": self.min_scene_len,
                            "scene_count": len(scenes),
                            "processing_time": detection_time,
                            "scenes": scenes
                        }, f, indent=2)
                    logger.info(f"💾 Scene data saved to {output_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to save scene data: {e}")
            
            # Return the scenes
            self.stats["scenes"] = scenes
            logger.info(f"✅ Detected {len(scenes)} scenes")
            
            return scenes
            
        except Exception as e:
            logger.error(f"❌ Error during scene detection: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try fallback detection using PySceneDetect
            logger.info(f"⚠️ Attempting fallback scene detection using PySceneDetect...")
            
            try:
                adapter = PySceneDetectAdapter(video_path, threshold=self.threshold, min_scene_len=self.min_scene_len)
                adapter.detect()
                scenes = adapter.get_scenes()
                
                logger.info(f"✅ Fallback detection completed. Found {len(scenes)} scenes.")
                return scenes
            except Exception as e2:
                logger.error(f"❌ Fallback detection also failed: {e2}")
                # Return empty list
                return []
    
    def _build_scenes_from_cuts(self, cuts: List[int], fps: float, frame_count: int, 
                              width: int, height: int) -> List[Dict[str, Any]]:
        """
        Build scene list from detected cut frames.
        
        Args:
            cuts: List of frames where cuts occur
            fps: Frames per second
            frame_count: Total number of frames
            width: Video width
            height: Video height
            
        Returns:
            List of scenes
        """
        logger.info(f"🔄 Building scenes from {len(cuts)} cuts...")
        
        # Sort cuts and add start and end frames
        sorted_cuts = sorted(cuts)
        
        # Insert start frame if the first cut is not at the beginning
        if not sorted_cuts or sorted_cuts[0] > 0:
            sorted_cuts.insert(0, 0)
            
        # Add end frame if the last cut is not at the end
        if not sorted_cuts or sorted_cuts[-1] < frame_count - 1:
            sorted_cuts.append(frame_count - 1)
        
        # Build scenes list
        scenes = []
        for i in range(len(sorted_cuts) - 1):
            start_frame = sorted_cuts[i]
            end_frame = sorted_cuts[i + 1] - 1  # End frame is exclusive
            
            # Skip scenes that are too short
            if end_frame - start_frame < self.min_scene_len:
                continue
            
            # Calculate scene properties
            start_time = start_frame / fps if fps > 0 else 0
            end_time = end_frame / fps if fps > 0 else 0
            duration_frames = end_frame - start_frame + 1
            duration_seconds = duration_frames / fps if fps > 0 else 0
            
            # Only add scenes that are long enough
            if duration_seconds >= 0.5:  # At least 0.5 seconds
                scenes.append({
                    "id": len(scenes),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_frames": duration_frames,
                    "duration_seconds": duration_seconds,
                    "resolution": {
                        "width": width,
                        "height": height
                    }
                })
        
        # If we have no scenes, add one covering the whole video
        if not scenes:
            logger.warning("⚠️ No scenes detected, creating one scene for the entire video")
            scenes.append({
                "id": 0,
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "start_time": 0,
                "end_time": frame_count / fps if fps > 0 else 0,
                "duration_frames": frame_count,
                "duration_seconds": frame_count / fps if fps > 0 else 0,
                "resolution": {
                    "width": width,
                    "height": height
                }
            })
        
        # Log scene info
        logger.info(f"✅ Created {len(scenes)} scenes from {len(cuts)} cuts")
        
        return scenes


class PySceneDetectAdapter:
    """Adapter for PySceneDetect for fallback processing"""
    
    def __init__(self, video_path, threshold=30.0, min_scene_len=15):
        self.video_path = video_path
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.scenes = []
    
    def detect(self):
        """Detect scenes using PySceneDetect"""
        logger.info(f"🔄 Using PySceneDetect with threshold {self.threshold}")
        
        # Create video manager and scene manager
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        
        # Add content detector
        scene_manager.add_detector(
            ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
        )
        
        # Start video manager
        video_manager.start()
        
        # Detect scenes
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        
        # Stop video manager
        video_manager.release()
        
        # Save scenes
        fps = video_manager.get_framerate()
        frame_count = video_manager.get_base_timecode().get_frames()
        duration = frame_count / fps
        
        self.scenes = []
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames() - 1
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration_frames = end_frame - start_frame + 1
            duration_seconds = duration_frames / fps
            
            self.scenes.append({
                "id": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "duration_frames": duration_frames,
                "duration_seconds": duration_seconds
            })
        
        logger.info(f"✅ PySceneDetect found {len(self.scenes)} scenes")
    
    def get_scenes(self):
        """Get detected scenes"""
        return self.scenes 