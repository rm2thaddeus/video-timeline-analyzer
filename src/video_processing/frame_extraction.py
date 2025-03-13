"""
Frame Extraction Module.

This module provides utilities for extracting frames from videos with GPU acceleration
and efficient memory usage.

📌 Purpose: Extract and process frames from videos with GPU optimization
🔄 Latest Changes: Initial implementation with GPU/batch processing
⚙️ Key Logic: Uses batched processing and half precision where appropriate
📂 Expected File Path: src/video_processing/frame_extraction.py
🧠 Reasoning: Frame processing is parallelizable and benefits from GPU acceleration,
              especially for batched operations
"""

import os
import logging
import tempfile
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from enum import Enum
from pathlib import Path
import math
import time
import concurrent.futures

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.schema import Scene, Frame, VideoMetadata
from src.utils.gpu_utils import get_optimal_device, clear_gpu_memory, memory_stats
from src.video_processing.loader import create_video_capture, extract_frame, save_frame

logger = logging.getLogger(__name__)

# Get the optimal device (GPU if available, otherwise CPU)
DEVICE = get_optimal_device()


class FrameExtractionMethod(str, Enum):
    """Methods for extracting frames from scenes."""
    
    FIRST = "first"  # First frame of the scene
    MIDDLE = "middle"  # Middle frame of the scene
    REPRESENTATIVE = "representative"  # Multiple representative frames
    UNIFORM = "uniform"  # Uniformly spaced frames
    KEYFRAME = "keyframe"  # Frames with significant changes
    ALL = "all"  # All frames (with sampling)


class VideoFrameDataset(Dataset):
    """Dataset for efficient frame extraction from video."""
    
    def __init__(
        self,
        video_path: str,
        timestamps: Optional[List[float]] = None,
        frame_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        max_frames: int = 1000
    ):
        """
        Initialize the dataset.
        
        Args:
            video_path: Path to the video file
            timestamps: List of timestamps to extract (in seconds)
            frame_indices: List of frame indices to extract
            transform: Transform to apply to frames
            max_frames: Maximum number of frames to extract
        """
        self.video_path = video_path
        self.transform = transform
        
        # Open video file
        self.cap = create_video_capture(video_path)
        if self.cap is None:
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Determine frames to extract
        if timestamps is not None:
            # Convert timestamps to frame indices
            self.frame_indices = [min(self.frame_count - 1, max(0, int(t * self.fps))) for t in timestamps]
            self.timestamps = timestamps
        elif frame_indices is not None:
            # Use provided frame indices
            self.frame_indices = [min(self.frame_count - 1, max(0, idx)) for idx in frame_indices]
            # Convert frame indices to timestamps
            self.timestamps = [idx / self.fps if self.fps > 0 else 0 for idx in self.frame_indices]
        else:
            # Extract all frames (with limit)
            step = max(1, self.frame_count // max_frames)
            self.frame_indices = list(range(0, self.frame_count, step))
            # Convert frame indices to timestamps
            self.timestamps = [idx / self.fps if self.fps > 0 else 0 for idx in self.frame_indices]
        
        logger.debug(f"Created VideoFrameDataset for {video_path} with {len(self.frame_indices)} frames")
    
    def __len__(self) -> int:
        """Get the number of frames."""
        return len(self.frame_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int]:
        """
        Get a frame by index.
        
        Args:
            idx: Index of the frame to retrieve
            
        Returns:
            Tuple containing:
            - Frame as tensor (C, H, W)
            - Timestamp in seconds
            - Frame index
        """
        frame_idx = self.frame_indices[idx]
        timestamp = self.timestamps[idx]
        
        # Set position to the frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = self.cap.read()
        if not ret:
            # Return black frame if frame cannot be read
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transform if provided
        if self.transform:
            frame_rgb = self.transform(frame_rgb)
        else:
            # Convert to tensor (H, W, C) -> (C, H, W)
            frame_rgb = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        
        return frame_rgb, timestamp, frame_idx
    
    def close(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FrameExtractor:
    """Extract frames from videos with GPU acceleration."""
    
    def __init__(
        self,
        batch_size: int = 8,
        use_gpu: bool = True,
        use_half_precision: bool = False,
        num_workers: int = 2
    ):
        """
        Initialize the frame extractor.
        
        Args:
            batch_size: Number of frames to process at once
            use_gpu: Whether to use GPU acceleration
            use_half_precision: Whether to use half precision (FP16)
            num_workers: Number of workers for data loading
        """
        self.batch_size = batch_size
        self.device = get_optimal_device() if use_gpu else torch.device('cpu')
        self.use_half_precision = use_half_precision and self.device.type == 'cuda'
        self.num_workers = num_workers
        
        logger.info(
            f"Initialized frame extractor (device: {self.device}, "
            f"batch_size: {batch_size}, half precision: {use_half_precision})"
        )
    
    def extract_frames_from_scenes(
        self,
        video_path: str,
        scenes: List[Scene],
        output_dir: str,
        method: Union[str, FrameExtractionMethod] = FrameExtractionMethod.REPRESENTATIVE,
        max_frames_per_scene: int = 3,
        transform: Optional[Callable] = None,
        return_tensors: bool = False
    ) -> Dict[int, List[Frame]]:
        """
        Extract frames from scenes.
        
        Args:
            video_path: Path to the video file
            scenes: List of Scene objects
            output_dir: Directory to save extracted frames
            method: Method for extracting frames
            max_frames_per_scene: Maximum number of frames to extract per scene
            transform: Transform to apply to frames
            return_tensors: Whether to return frame tensors in Frame objects
            
        Returns:
            Dictionary mapping scene IDs to lists of Frame objects
        """
        logger.info(f"Extracting frames from {len(scenes)} scenes (method: {method})")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video metadata
        cap = create_video_capture(video_path)
        if cap is None:
            logger.error(f"Failed to open video: {video_path}")
            return {}
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Convert method to enum if string
            extraction_method = method if isinstance(method, FrameExtractionMethod) else FrameExtractionMethod(method)
            
            # Generate timestamps for each scene
            all_timestamps = []
            scene_timestamps = {}
            
            for scene in scenes:
                scene_id = scene.id
                timestamps = self._get_scene_timestamps(
                    scene,
                    extraction_method,
                    max_frames_per_scene,
                    fps
                )
                scene_timestamps[scene_id] = timestamps
                all_timestamps.extend(timestamps)
            
            # Create dataset
            dataset = VideoFrameDataset(
                video_path,
                timestamps=all_timestamps,
                transform=transform
            )
            
            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.device.type == 'cuda'
            )
            
            # Extract frames
            extracted_frames = {}
            
            for batch_frames, batch_timestamps, batch_indices in data_loader:
                # Move batch to device
                batch_frames = batch_frames.to(self.device)
                
                # Convert to half precision if required
                if self.use_half_precision:
                    batch_frames = batch_frames.half()
                
                # Process batch (in this case, just pass through)
                # Additional processing could be added here
                processed_frames = batch_frames
                
                # Create Frame objects
                for i in range(len(batch_timestamps)):
                    timestamp = batch_timestamps[i].item()
                    frame_idx = batch_indices[i].item()
                    
                    # Find which scene this timestamp belongs to
                    scene_id = None
                    for sid, timestamps in scene_timestamps.items():
                        if timestamp in timestamps:
                            scene_id = sid
                            break
                    
                    if scene_id is None:
                        logger.warning(f"Could not find scene for timestamp {timestamp}")
                        continue
                    
                    # Initialize scene frames list if needed
                    if scene_id not in extracted_frames:
                        extracted_frames[scene_id] = []
                    
                    # Convert frame to NumPy and save
                    frame_np = (
                        processed_frames[i].cpu().float().numpy() * 255
                    ).astype(np.uint8).transpose(1, 2, 0)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    
                    # Save frame
                    frame_path = os.path.join(
                        output_dir,
                        f"scene_{scene_id:04d}_frame_{len(extracted_frames[scene_id]):02d}_{timestamp:.2f}s.jpg"
                    )
                    success = save_frame(frame_np, frame_path)
                    
                    if success:
                        # Create Frame object
                        frame = Frame(
                            timestamp=timestamp,
                            file_path=frame_path
                        )
                        
                        # Add tensor if requested
                        if return_tensors:
                            frame.tensor = processed_frames[i].cpu()
                        
                        # Add to scene frames
                        extracted_frames[scene_id].append(frame)
                        logger.debug(f"Saved frame: {frame_path}")
                    else:
                        logger.error(f"Failed to save frame: {frame_path}")
                
                # Clear CUDA cache periodically
                if self.device.type == 'cuda' and len(extracted_frames) % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Update scenes with extracted frames
            for scene in scenes:
                if scene.id in extracted_frames:
                    scene.key_frames = extracted_frames[scene.id]
            
            return extracted_frames
        
        finally:
            # Release resources
            cap.release()
            dataset.close()
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def extract_uniform_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: float = 1.0,  # Extract one frame every N seconds
        max_frames: int = 1000,
        transform: Optional[Callable] = None,
        return_tensors: bool = False
    ) -> List[Frame]:
        """
        Extract frames at uniform intervals.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            interval: Interval between frames in seconds
            max_frames: Maximum number of frames to extract
            transform: Transform to apply to frames
            return_tensors: Whether to return frame tensors in Frame objects
            
        Returns:
            List of Frame objects
        """
        logger.info(f"Extracting uniform frames from {video_path} (interval: {interval}s)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video metadata
        cap = create_video_capture(video_path)
        if cap is None:
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Calculate frame indices
            frame_step = int(interval * fps)
            if frame_step < 1:
                frame_step = 1
            
            # Cap number of frames
            num_frames = min(max_frames, frame_count // frame_step + 1)
            
            # Generate frame indices
            frame_indices = [i * frame_step for i in range(num_frames)]
            
            # Create dataset
            dataset = VideoFrameDataset(
                video_path,
                frame_indices=frame_indices,
                transform=transform
            )
            
            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.device.type == 'cuda'
            )
            
            # Extract frames
            extracted_frames = []
            
            for batch_frames, batch_timestamps, batch_indices in data_loader:
                # Move batch to device
                batch_frames = batch_frames.to(self.device)
                
                # Convert to half precision if required
                if self.use_half_precision:
                    batch_frames = batch_frames.half()
                
                # Process batch (in this case, just pass through)
                # Additional processing could be added here
                processed_frames = batch_frames
                
                # Create Frame objects
                for i in range(len(batch_timestamps)):
                    timestamp = batch_timestamps[i].item()
                    frame_idx = batch_indices[i].item()
                    
                    # Convert frame to NumPy and save
                    frame_np = (
                        processed_frames[i].cpu().float().numpy() * 255
                    ).astype(np.uint8).transpose(1, 2, 0)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    
                    # Save frame
                    frame_path = os.path.join(
                        output_dir,
                        f"frame_{len(extracted_frames):04d}_{timestamp:.2f}s.jpg"
                    )
                    success = save_frame(frame_np, frame_path)
                    
                    if success:
                        # Create Frame object
                        frame = Frame(
                            timestamp=timestamp,
                            file_path=frame_path
                        )
                        
                        # Add tensor if requested
                        if return_tensors:
                            frame.tensor = processed_frames[i].cpu()
                        
                        # Add to frames
                        extracted_frames.append(frame)
                        logger.debug(f"Saved frame: {frame_path}")
                    else:
                        logger.error(f"Failed to save frame: {frame_path}")
                
                # Clear CUDA cache periodically
                if self.device.type == 'cuda' and len(extracted_frames) % 100 == 0:
                    torch.cuda.empty_cache()
            
            return extracted_frames
        
        finally:
            # Release resources
            cap.release()
            dataset.close()
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def _get_scene_timestamps(
        self,
        scene: Scene,
        method: FrameExtractionMethod,
        max_frames: int,
        fps: float
    ) -> List[float]:
        """
        Get timestamps for frames to extract from a scene.
        
        Args:
            scene: Scene object
            method: Method for extracting frames
            max_frames: Maximum number of frames to extract
            fps: Frames per second
            
        Returns:
            List of timestamps to extract
        """
        start_time = scene.start_time
        end_time = scene.end_time
        duration = end_time - start_time
        
        if method == FrameExtractionMethod.FIRST:
            # First frame of the scene
            return [start_time]
        
        elif method == FrameExtractionMethod.MIDDLE:
            # Middle frame of the scene
            return [(start_time + end_time) / 2]
        
        elif method == FrameExtractionMethod.REPRESENTATIVE:
            # Multiple representative frames
            frames_to_extract = min(max_frames, max(1, int(duration)))
            
            if frames_to_extract == 1:
                return [(start_time + end_time) / 2]
            else:
                # Evenly spaced frames
                return [
                    start_time + (i + 1) * duration / (frames_to_extract + 1)
                    for i in range(frames_to_extract)
                ]
        
        elif method == FrameExtractionMethod.UNIFORM:
            # Uniformly spaced frames
            frame_interval = 1.0  # One frame per second
            frames_to_extract = min(max_frames, max(1, int(duration / frame_interval) + 1))
            
            return [
                start_time + i * frame_interval
                for i in range(frames_to_extract)
                if start_time + i * frame_interval < end_time
            ]
        
        elif method == FrameExtractionMethod.ALL:
            # All frames (with sampling)
            frame_step = max(1, int(fps * duration / max_frames))
            
            # Convert to frame indices and back to timestamps for accuracy
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            return [
                frame_idx / fps
                for frame_idx in range(start_frame, end_frame, frame_step)
            ]
        
        else:
            logger.error(f"Unknown extraction method: {method}")
            return [start_time]


class BatchFrameProcessor:
    """Process frames in batches with GPU acceleration."""
    
    def __init__(
        self,
        batch_size: int = 16,
        use_gpu: bool = True,
        use_half_precision: bool = False,
        preload_frames: bool = False,
        max_frames_in_memory: int = 100
    ):
        """
        Initialize the batch frame processor.
        
        Args:
            batch_size: Number of frames to process at once
            use_gpu: Whether to use GPU acceleration
            use_half_precision: Whether to use half precision (FP16)
            preload_frames: Whether to preload frames into memory
            max_frames_in_memory: Maximum number of frames to keep in memory
        """
        self.batch_size = batch_size
        self.device = get_optimal_device() if use_gpu else torch.device('cpu')
        self.use_half_precision = use_half_precision and self.device.type == 'cuda'
        self.preload_frames = preload_frames
        self.max_frames_in_memory = max_frames_in_memory
        
        # Frame cache
        self.frame_cache = {}
        
        logger.info(
            f"Initialized batch frame processor (device: {self.device}, "
            f"batch_size: {batch_size}, half precision: {use_half_precision})"
        )
    
    def process_frames(
        self,
        frames: List[Frame],
        processor: Callable[[torch.Tensor], torch.Tensor],
        output_dir: Optional[str] = None,
        save_output: bool = False
    ) -> List[torch.Tensor]:
        """
        Process frames in batches.
        
        Args:
            frames: List of Frame objects
            processor: Function to process frames
            output_dir: Directory to save processed frames
            save_output: Whether to save processed frames
            
        Returns:
            List of processed frame tensors
        """
        logger.info(f"Processing {len(frames)} frames in batches of {self.batch_size}")
        
        if save_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Preload frames if requested
        if self.preload_frames:
            self._preload_frames(frames)
        
        # Process frames in batches
        processed_frames = []
        
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            
            # Load frame tensors
            batch_tensors = []
            for frame in batch_frames:
                # Check if frame is in cache
                if self.preload_frames and frame.file_path in self.frame_cache:
                    tensor = self.frame_cache[frame.file_path]
                else:
                    # Load frame from file
                    image = cv2.imread(frame.file_path)
                    
                    if image is None:
                        logger.error(f"Failed to load frame: {frame.file_path}")
                        continue
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Convert to tensor
                    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                
                batch_tensors.append(tensor)
            
            # Skip empty batches
            if not batch_tensors:
                continue
            
            # Create batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Convert to half precision if required
            if self.use_half_precision:
                batch_tensor = batch_tensor.half()
            
            # Process batch
            try:
                processed_batch = processor(batch_tensor)
                
                # Convert back to float if needed
                if self.use_half_precision:
                    processed_batch = processed_batch.float()
                
                # Move back to CPU
                processed_batch = processed_batch.cpu()
                
                # Save processed frames if requested
                if save_output and output_dir:
                    for j, frame in enumerate(batch_frames):
                        if j >= len(processed_batch):
                            break
                        
                        # Convert to NumPy and save
                        processed_np = (
                            processed_batch[j].numpy() * 255
                        ).astype(np.uint8).transpose(1, 2, 0)
                        
                        # Convert RGB to BGR for OpenCV
                        processed_np = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)
                        
                        # Save frame
                        output_path = os.path.join(
                            output_dir,
                            f"processed_{os.path.basename(frame.file_path)}"
                        )
                        cv2.imwrite(output_path, processed_np)
                
                # Add to results
                processed_frames.extend([tensor for tensor in processed_batch])
            
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
            
            # Clear CUDA cache periodically
            if self.device.type == 'cuda' and (i // self.batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Clear frame cache
        self.frame_cache.clear()
        
        # Clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return processed_frames
    
    def _preload_frames(self, frames: List[Frame]) -> None:
        """
        Preload frames into memory.
        
        Args:
            frames: List of Frame objects
        """
        logger.info(f"Preloading {min(len(frames), self.max_frames_in_memory)} frames")
        
        # Clear existing cache
        self.frame_cache.clear()
        
        # Limit number of frames to preload
        frames_to_load = frames[:self.max_frames_in_memory]
        
        for frame in frames_to_load:
            # Load frame from file
            image = cv2.imread(frame.file_path)
            
            if image is None:
                logger.error(f"Failed to preload frame: {frame.file_path}")
                continue
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            
            # Add to cache
            self.frame_cache[frame.file_path] = tensor