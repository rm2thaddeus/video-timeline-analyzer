"""
📌 Purpose: Extract and embed frames from video files
🔄 Latest Changes: Improved frame sampling for better overview, optimized GPU memory usage
⚙️ Key Logic: Extract frames based on scene boundaries and embed with CLIP
📂 Expected File Path: test_pipeline/processors/frame_processor.py
🧠 Reasoning: Separate frame processing for better code organization
"""

import os
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import concurrent.futures
import gc

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Import CLIP for embedding
import clip

# Fix the import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.gpu_utils import setup_device, get_optimal_batch_size

logger = logging.getLogger("video_pipeline.frame_processor")

class FrameProcessor:
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[torch.device] = None,
                 memory_efficient: bool = True, max_frames_in_memory: int = 100):
        """
        Initialize frame processor with CLIP model for embedding.
        
        Args:
            model_name: CLIP model variant
            device: Torch device for model inference
            memory_efficient: Use memory-efficient processing
            max_frames_in_memory: Maximum number of frames to keep in memory at once
        """
        self.model_name = model_name
        self.memory_efficient = memory_efficient
        self.max_frames_in_memory = max_frames_in_memory
        
        if device is None:
            self.device, _ = setup_device()
        else:
            self.device = device
            
        logger.info(f"Initializing CLIP model '{model_name}' on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Get optimal batch size based on available GPU memory
        suggested_batch_size = get_optimal_batch_size("clip") if self.device.type == "cuda" else 1
        
        # For 6GB GPU, we need to limit batch size further to avoid OOM
        if self.device.type == "cuda":
            # Get device properties
            device_idx = 0 if self.device.index is None else self.device.index
            device_props = torch.cuda.get_device_properties(device_idx)
            
            # Adjust batch size based on available memory
            mem_gb = device_props.total_memory / (1024**3)
            logger.info(f"GPU memory: {mem_gb:.2f} GB")
            
            # For smaller GPUs (6GB or less), use a more conservative batch size
            if mem_gb <= 6.5:  # Allow some margin
                suggested_batch_size = min(suggested_batch_size, 8)
                logger.info(f"Limited batch size to {suggested_batch_size} due to GPU memory constraints")
        
        self.batch_size = suggested_batch_size
        logger.info(f"CLIP model loaded successfully. Using batch size: {self.batch_size}")
        
        # Enable mixed precision for efficiency if on CUDA
        self.use_mixed_precision = self.device.type == "cuda"
        if self.use_mixed_precision:
            logger.info("Enabling mixed precision (FP16) for CLIP inference")
            self.model = self.model.half()
        
    def _clean_gpu_memory(self):
        """Clean GPU memory to prevent OOM errors"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
    def _sample_frames_from_scene(self, 
                                  scene: Dict[str, Any], 
                                  cap: cv2.VideoCapture,
                                  frames_per_scene: int,
                                  min_frames_per_scene: int = 1,
                                  max_frames_per_scene: int = 10,
                                  scene_duration_threshold: float = 5.0) -> List[int]:
        """
        Sample representative frames from a scene based on its duration and content.
        
        Args:
            scene: Scene dictionary containing duration and frame information
            cap: OpenCV video capture object
            frames_per_scene: Target number of frames per scene
            min_frames_per_scene: Minimum frames to extract per scene
            max_frames_per_scene: Maximum frames to extract per scene
            scene_duration_threshold: Minimum duration (seconds) to use more than min_frames
            
        Returns:
            List of frame indices to extract
        """
        start_frame = scene["start_frame"]
        end_frame = scene["end_frame"]
        duration_frames = end_frame - start_frame
        duration_seconds = scene.get("duration_seconds", duration_frames / 30.0)  # Assume 30fps if not provided
        
        # For very short scenes, extract just 1 frame
        if duration_seconds < 1.0:
            return [int(start_frame + duration_frames // 2)]  # Middle frame
            
        # For short scenes, extract fewer frames
        if duration_seconds < scene_duration_threshold:
            actual_frames = min_frames_per_scene
        else:
            # Scale the number of frames based on scene duration
            # More frames for longer scenes, but capped at max_frames_per_scene
            scaled_frames = int(frames_per_scene * min(1.0, duration_seconds / 20.0))
            actual_frames = min(max(min_frames_per_scene, scaled_frames), max_frames_per_scene)
        
        # Cap to the actual number of frames available
        actual_frames = min(actual_frames, duration_frames)
        
        if actual_frames <= 1:
            # Just the middle frame
            return [int(start_frame + duration_frames // 2)]
        
        # For important-looking scenes (e.g., with faces or text), extract more frames
        # This requires content analysis, but we'd need to sample frames first
        # Let's implement a simple uniform sampling for now
        
        frame_indices = []
        step = duration_frames / actual_frames
        
        # Ensure we get frames distributed across the scene
        for i in range(actual_frames):
            frame_idx = int(start_frame + i * step)
            frame_indices.append(frame_idx)
            
        return frame_indices
        
    def extract_frames(self, video_path: str, scene_data: List[Dict[str, Any]], 
                       output_dir: Optional[str] = None, frames_per_scene: int = 3,
                       max_dimension: int = 720, 
                       smart_sampling: bool = True) -> List[Dict[str, Any]]:
        """
        Extract representative frames from scenes in a video.
        
        Args:
            video_path: Path to the video file
            scene_data: List of scene dictionaries from SceneDetector
            output_dir: Directory to save extracted frames
            frames_per_scene: Target number of frames to extract per scene
            max_dimension: Maximum dimension for frame resizing
            smart_sampling: Use smart sampling based on scene content
            
        Returns:
            List of dictionaries with frame information
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Extracting frames from {video_path}")
        
        frames = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video has {frame_count} frames at {fps} FPS")
            
            # Process each scene
            for scene in tqdm(scene_data, desc="Processing scenes"):
                scene_id = scene["scene_id"] if "scene_id" in scene else scene.get("id", 0)
                start_frame = scene["start_frame"]
                end_frame = scene["end_frame"]
                
                # Use improved frame sampling
                if smart_sampling:
                    frame_indices = self._sample_frames_from_scene(
                        scene, cap, frames_per_scene,
                        min_frames_per_scene=1,
                        max_frames_per_scene=10,
                        scene_duration_threshold=5.0
                    )
                else:
                    # Old sampling logic - uniform distribution
                    duration_frames = end_frame - start_frame
                    
                    # Calculate frame indices to extract
                    if frames_per_scene >= duration_frames:
                        # If scene is shorter than requested frames, extract all frames
                        frame_indices = list(range(start_frame, end_frame + 1))
                    else:
                        # Extract evenly spaced frames
                        step = duration_frames / frames_per_scene
                        frame_indices = [int(start_frame + i * step) for i in range(frames_per_scene)]
                
                # Extract frames
                for frame_idx in frame_indices:
                    # Set video position to the desired frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning(f"Could not read frame {frame_idx}")
                        continue
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame if needed
                    height, width = frame_rgb.shape[:2]
                    if height > max_dimension or width > max_dimension:
                        if height > width:
                            new_height = max_dimension
                            new_width = int(width * (max_dimension / height))
                        else:
                            new_width = max_dimension
                            new_height = int(height * (max_dimension / width))
                        
                        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                    
                    # Save frame
                    frame_filename = f"{video_id}_scene{scene_id:03d}_frame{frame_idx:05d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    # Convert to PIL Image and save
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image.save(frame_path)
                    
                    # Add frame info to list
                    frame_info = {
                        "video_id": video_id,
                        "scene_id": scene_id,
                        "frame_number": frame_idx,
                        "timestamp": frame_idx / fps,
                        "path": str(frame_path)
                    }
                    
                    frames.append(frame_info)
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {len(scene_data)} scenes")
            
            # Save frame info to JSON
            if output_dir:
                frames_json = output_dir / f"{video_id}_frames.json"
                with open(frames_json, 'w') as f:
                    json.dump(frames, f, indent=2)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error during frame extraction: {e}")
            raise
    
    def _process_batch(self, batch_frames: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of frames with CLIP embedding.
        
        Args:
            batch_frames: List of frame dictionaries to process
            output_dir: Directory to save embeddings
            
        Returns:
            List of dictionaries with embeddings added
        """
        batch_images = []
        valid_indices = []
        
        # Load and preprocess images
        for i, frame in enumerate(batch_frames):
            try:
                image = Image.open(frame["path"])
                preprocessed_image = self.preprocess(image)
                batch_images.append(preprocessed_image)
                valid_indices.append(i)
            except Exception as e:
                logger.error(f"Error preprocessing frame {frame['path']}: {e}")
        
        if not batch_images:
            return []
        
        # Stack images into a batch tensor
        image_tensor = torch.stack(batch_images).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            # Use mixed precision if enabled and on CUDA
            if self.use_mixed_precision and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    image_features = self.model.encode_image(image_tensor)
            else:
                image_features = self.model.encode_image(image_tensor)
                
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Convert to numpy and add to results
        embeddings = image_features.cpu().numpy()
        
        # Create result frames with embeddings
        result_frames = []
        for j, idx in enumerate(valid_indices):
            if j < len(embeddings):
                frame = batch_frames[idx]
                frame_with_embedding = frame.copy()
                frame_with_embedding["embedding"] = embeddings[j].tolist()
                
                # Save individual embedding
                if output_dir:
                    video_id = frame["video_id"]
                    embedding_filename = f"{video_id}_scene{frame['scene_id']:03d}_frame{frame['frame_number']:05d}_embedding.npy"
                    embedding_path = output_dir / embedding_filename
                    np.save(embedding_path, embeddings[j])
                    frame_with_embedding["embedding_path"] = str(embedding_path)
                
                result_frames.append(frame_with_embedding)
        
        # Clean memory
        del image_tensor, image_features, embeddings, batch_images
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return result_frames
    
    def embed_frames(self, frames: List[Dict[str, Any]], output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate CLIP embeddings for extracted frames.
        
        Args:
            frames: List of frame dictionaries from extract_frames()
            output_dir: Directory to save embeddings
            
        Returns:
            List of dictionaries with frame information and embeddings
        """
        if not frames:
            logger.warning("No frames to embed")
            return []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        video_id = frames[0]["video_id"]
        logger.info(f"Generating CLIP embeddings for {len(frames)} frames with batch size {self.batch_size}")
        
        # Process frames in batches
        embedded_frames = []
        
        # If memory efficient mode and too many frames, process in chunks
        if self.memory_efficient and len(frames) > self.max_frames_in_memory:
            logger.info(f"Using memory-efficient processing for {len(frames)} frames")
            
            # Process in chunks to manage memory
            for chunk_start in tqdm(range(0, len(frames), self.max_frames_in_memory), 
                                   desc="Processing frame chunks"):
                chunk_end = min(chunk_start + self.max_frames_in_memory, len(frames))
                chunk_frames = frames[chunk_start:chunk_end]
                
                # Process batch within chunk
                for batch_start in range(0, len(chunk_frames), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(chunk_frames))
                    batch_frames = chunk_frames[batch_start:batch_end]
                    
                    # Process batch
                    batch_results = self._process_batch(batch_frames, output_dir)
                    embedded_frames.extend(batch_results)
                    
                # Clean memory between chunks
                self._clean_gpu_memory()
        else:
            # Standard batch processing
            for i in tqdm(range(0, len(frames), self.batch_size), desc="Embedding frames"):
                batch_frames = frames[i:i+self.batch_size]
                batch_results = self._process_batch(batch_frames, output_dir)
                embedded_frames.extend(batch_results)
        
        # Save all embeddings
        if output_dir and embedded_frames:
            embeddings_json = output_dir / f"{video_id}_frame_embeddings.json"
            
            # Create a copy without the actual embeddings for JSON storage
            frames_for_json = []
            for frame in embedded_frames:
                frame_copy = frame.copy()
                if "embedding" in frame_copy:
                    del frame_copy["embedding"]
                frames_for_json.append(frame_copy)
                
            with open(embeddings_json, 'w') as f:
                json.dump(frames_for_json, f, indent=2)
        
        logger.info(f"Generated embeddings for {len(embedded_frames)} frames")
        return embedded_frames
    
    def process_video(self, video_path: str, scene_data: List[Dict[str, Any]], 
                     frames_dir: Optional[str] = None, embeddings_dir: Optional[str] = None,
                     frames_per_scene: int = 3, max_dimension: int = 720,
                     parallel_extraction: bool = False) -> List[Dict[str, Any]]:
        """
        Process a video: extract frames and generate embeddings.
        
        Args:
            video_path: Path to the video file
            scene_data: List of scene dictionaries from SceneDetector
            frames_dir: Directory to save extracted frames
            embeddings_dir: Directory to save embeddings
            frames_per_scene: Number of frames to extract per scene
            max_dimension: Maximum dimension for frame resizing
            parallel_extraction: Use parallel processing for frame extraction
            
        Returns:
            List of dictionaries with frame information and embeddings
        """
        # Extract frames
        frames = self.extract_frames(
            video_path, 
            scene_data, 
            frames_dir, 
            frames_per_scene, 
            max_dimension,
            smart_sampling=True
        )
        
        # Generate embeddings
        embedded_frames = self.embed_frames(frames, embeddings_dir)
        
        return embedded_frames 