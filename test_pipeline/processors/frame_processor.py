"""
📌 Purpose: Extract and embed frames from video files
🔄 Latest Changes: Fixed import issue
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
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[torch.device] = None):
        """
        Initialize frame processor with CLIP model for embedding.
        
        Args:
            model_name: CLIP model variant
            device: Torch device for model inference
        """
        self.model_name = model_name
        
        if device is None:
            self.device, _ = setup_device()
        else:
            self.device = device
            
        logger.info(f"Initializing CLIP model '{model_name}' on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Get optimal batch size based on available GPU memory
        self.batch_size = get_optimal_batch_size("clip") if self.device.type == "cuda" else 1
        logger.info(f"CLIP model loaded successfully. Using batch size: {self.batch_size}")
        
    def extract_frames(self, video_path: str, scene_data: List[Dict[str, Any]], 
                       output_dir: Optional[str] = None, frames_per_scene: int = 3,
                       max_dimension: int = 720) -> List[Dict[str, Any]]:
        """
        Extract representative frames from scenes in a video.
        
        Args:
            video_path: Path to the video file
            scene_data: List of scene dictionaries from SceneDetector
            output_dir: Directory to save extracted frames
            frames_per_scene: Number of frames to extract per scene
            max_dimension: Maximum dimension for frame resizing
            
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
                scene_id = scene["scene_id"]
                start_frame = scene["start_frame"]
                end_frame = scene["end_frame"]
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
        logger.info(f"Generating CLIP embeddings for {len(frames)} frames")
        
        # Process frames in batches
        embedded_frames = []
        
        for i in tqdm(range(0, len(frames), self.batch_size), desc="Embedding frames"):
            batch_frames = frames[i:i+self.batch_size]
            batch_images = []
            
            # Load and preprocess images
            for frame in batch_frames:
                try:
                    image = Image.open(frame["path"])
                    preprocessed_image = self.preprocess(image)
                    batch_images.append(preprocessed_image)
                except Exception as e:
                    logger.error(f"Error preprocessing frame {frame['path']}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Stack images into a batch tensor
            image_tensor = torch.stack(batch_images).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy and add to results
            embeddings = image_features.cpu().numpy()
            
            for j, frame in enumerate(batch_frames):
                if j < len(embeddings):
                    frame_with_embedding = frame.copy()
                    frame_with_embedding["embedding"] = embeddings[j].tolist()
                    
                    # Save individual embedding
                    if output_dir:
                        embedding_filename = f"{video_id}_scene{frame['scene_id']:03d}_frame{frame['frame_number']:05d}_embedding.npy"
                        embedding_path = output_dir / embedding_filename
                        np.save(embedding_path, embeddings[j])
                        frame_with_embedding["embedding_path"] = str(embedding_path)
                    
                    embedded_frames.append(frame_with_embedding)
        
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
                     frames_per_scene: int = 3, max_dimension: int = 720) -> List[Dict[str, Any]]:
        """
        Process a video: extract frames and generate embeddings.
        
        Args:
            video_path: Path to the video file
            scene_data: List of scene dictionaries from SceneDetector
            frames_dir: Directory to save extracted frames
            embeddings_dir: Directory to save embeddings
            frames_per_scene: Number of frames to extract per scene
            max_dimension: Maximum dimension for frame resizing
            
        Returns:
            List of dictionaries with frame information and embeddings
        """
        # Extract frames
        frames = self.extract_frames(
            video_path, 
            scene_data, 
            frames_dir, 
            frames_per_scene, 
            max_dimension
        )
        
        # Generate embeddings
        embedded_frames = self.embed_frames(frames, embeddings_dir)
        
        return embedded_frames 