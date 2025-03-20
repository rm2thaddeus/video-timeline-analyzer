"""
📌 Purpose: Unified metadata dataframe for video processing pipeline
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Create and manage a dataframe containing aligned frame, transcript, and embedding data
📂 Expected File Path: test_pipeline/processors/metadata_dataframe.py
🧠 Reasoning: Central repository for all metadata with efficient alignment of multi-rate sampled data
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from tqdm import tqdm

logger = logging.getLogger("video_pipeline.metadata_dataframe")

class MetadataManager:
    """
    Manager for unified video metadata including frames, transcripts, and embeddings.
    Handles alignment between different sampling rates and creates a unified dataframe.
    """
    
    def __init__(self):
        """Initialize the metadata manager"""
        self.df = None
        self.video_id = None
        self.performance_metrics = {}
        
    def create_dataframe(self, 
                         video_id: str,
                         frames_data: List[Dict[str, Any]],
                         transcript_data: Dict[str, Any],
                         scene_data: List[Dict[str, Any]] = None,
                         video_metadata: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Create a unified metadata dataframe from various data sources.
        
        Args:
            video_id: Identifier for the video
            frames_data: List of dictionaries containing frame data with embeddings
            transcript_data: Dictionary containing transcript segments with timestamps
            scene_data: Optional list of scene boundaries
            video_metadata: Optional video metadata
            
        Returns:
            Pandas DataFrame with aligned metadata
        """
        self.video_id = video_id
        
        # Create DataFrame from frames data
        logger.info(f"Creating metadata dataframe for {video_id} with {len(frames_data)} frames")
        
        # Extract relevant fields from frames data
        frames_for_df = []
        for frame in frames_data:
            frame_entry = {
                'video_id': video_id,
                'frame_number': frame.get('frame_number'),
                'timestamp': frame.get('timestamp'),
                'scene_id': frame.get('scene_id'),
                'frame_path': frame.get('path'),
                'embedding_path': frame.get('embedding_path', None),
            }
            
            # Add embedding if available in memory
            if 'embedding' in frame:
                # Store it separately to avoid DataFrame memory issues
                embedding = np.array(frame['embedding'])
                frame_entry['has_embedding'] = True
                frame_entry['embedding_dims'] = embedding.shape[0] if embedding is not None else 0
            else:
                frame_entry['has_embedding'] = False
                frame_entry['embedding_dims'] = 0
                
            frames_for_df.append(frame_entry)
        
        # Create dataframe from frames
        self.df = pd.DataFrame(frames_for_df)
        
        # Add transcript data by finding the closest transcript segment for each frame
        if transcript_data and 'segments' in transcript_data:
            logger.info(f"Aligning {len(transcript_data['segments'])} transcript segments with frames")
            
            # Create a lookup for transcript segments by start time
            transcript_segments = transcript_data['segments']
            
            # Sort segments by start time
            transcript_segments.sort(key=lambda x: x.get('start', 0))
            
            # Add transcript column initialized to empty strings
            self.df['transcript'] = ''
            self.df['transcript_start'] = -1.0
            self.df['transcript_end'] = -1.0
            
            # For each frame, find the corresponding transcript segment
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Aligning transcripts"):
                frame_time = row['timestamp']
                
                # Find the segment that contains this timestamp
                matching_segment = None
                for segment in transcript_segments:
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    
                    if start_time <= frame_time <= end_time:
                        matching_segment = segment
                        break
                
                # If no exact match, find the closest one
                if matching_segment is None and transcript_segments:
                    # Find segment with closest start time
                    closest_segment = min(transcript_segments, 
                                         key=lambda x: abs(x.get('start', 0) - frame_time))
                    matching_segment = closest_segment
                
                # Add transcript text to the dataframe
                if matching_segment:
                    self.df.at[idx, 'transcript'] = matching_segment.get('text', '')
                    self.df.at[idx, 'transcript_start'] = matching_segment.get('start', -1)
                    self.df.at[idx, 'transcript_end'] = matching_segment.get('end', -1)
        
        # Add scene information
        if scene_data:
            logger.info(f"Adding scene information from {len(scene_data)} scenes")
            
            # Add scene columns
            self.df['scene_start_frame'] = -1
            self.df['scene_end_frame'] = -1
            self.df['scene_start_time'] = -1.0
            self.df['scene_end_time'] = -1.0
            
            # Match scenes to frames
            for idx, row in self.df.iterrows():
                scene_id = row['scene_id']
                
                # Find matching scene
                matching_scene = next((s for s in scene_data if s.get('id', None) == scene_id or 
                                      s.get('scene_id', None) == scene_id), None)
                
                if matching_scene:
                    self.df.at[idx, 'scene_start_frame'] = matching_scene.get('start_frame', -1)
                    self.df.at[idx, 'scene_end_frame'] = matching_scene.get('end_frame', -1)
                    self.df.at[idx, 'scene_start_time'] = matching_scene.get('start_time', -1.0)
                    self.df.at[idx, 'scene_end_time'] = matching_scene.get('end_time', -1.0)
        
        # Add video metadata if provided
        if video_metadata:
            # Add global video metadata as separate columns prefixed with 'video_'
            metadata_to_add = {
                'video_duration': video_metadata.get('duration_seconds', 0),
                'video_width': video_metadata.get('width', 0),
                'video_height': video_metadata.get('height', 0),
                'video_fps': video_metadata.get('frame_rate', 0),
                'video_total_frames': video_metadata.get('total_frames', 0)
            }
            
            for key, value in metadata_to_add.items():
                self.df[key] = value
        
        logger.info(f"Created metadata dataframe with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def add_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add processing performance metrics to the metadata.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        self.performance_metrics.update(metrics)
        
    def load_clip_embeddings(self, embeddings_dir: str) -> None:
        """
        Load CLIP embeddings from files and add to dataframe.
        
        Args:
            embeddings_dir: Directory containing embedding files
        """
        if self.df is None:
            logger.warning("No dataframe to load embeddings into")
            return
            
        embeddings_dir = Path(embeddings_dir)
        logger.info(f"Loading CLIP embeddings from {embeddings_dir}")
        
        # Find embedding paths in the dataframe
        embedding_paths = self.df['embedding_path'].dropna().unique().tolist()
        
        if not embedding_paths:
            # Try to find embeddings based on frame info
            logger.info("No embedding paths in dataframe, searching for embeddings based on frame info")
            for idx, row in self.df.iterrows():
                if 'frame_path' in row and row['frame_path']:
                    frame_path = Path(row['frame_path'])
                    frame_name = frame_path.stem
                    embedding_path = embeddings_dir / f"{frame_name}_embedding.npy"
                    
                    if embedding_path.exists():
                        self.df.at[idx, 'embedding_path'] = str(embedding_path)
                        self.df.at[idx, 'has_embedding'] = True
            
            # Update list of embedding paths
            embedding_paths = self.df['embedding_path'].dropna().unique().tolist()
        
        # Load embeddings
        logger.info(f"Loading {len(embedding_paths)} embeddings")
        
        # Add a column to store the numpy arrays (not serialized to JSON)
        self.df['_embedding_array'] = None
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading embeddings"):
            if pd.notna(row['embedding_path']) and row['embedding_path']:
                try:
                    embedding = np.load(row['embedding_path'])
                    self.df.at[idx, '_embedding_array'] = embedding
                    self.df.at[idx, 'embedding_dims'] = embedding.shape[0]
                    self.df.at[idx, 'has_embedding'] = True
                except Exception as e:
                    logger.warning(f"Error loading embedding from {row['embedding_path']}: {e}")
    
    def save_dataframe(self, output_dir: str, include_embeddings: bool = False) -> str:
        """
        Save the dataframe to disk in JSON and CSV formats.
        
        Args:
            output_dir: Directory to save the dataframe
            include_embeddings: Whether to include the embedding arrays in JSON
            
        Returns:
            Path to the saved JSON file
        """
        if self.df is None:
            logger.warning("No dataframe to save")
            return None
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Base filename
        base_filename = f"{self.video_id}_metadata"
        
        # Create a copy of the dataframe for saving
        df_for_save = self.df.copy()
        
        # Add performance metrics
        metrics_df = pd.DataFrame([self.performance_metrics])
        
        # Save as CSV
        csv_path = output_dir / f"{base_filename}.csv"
        df_for_save.to_csv(csv_path, index=False)
        logger.info(f"Saved metadata to CSV: {csv_path}")
        
        # Save metrics as CSV
        metrics_csv_path = output_dir / f"{base_filename}_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        # Prepare for JSON serialization
        # Remove actual embedding arrays unless requested
        if not include_embeddings:
            if '_embedding_array' in df_for_save.columns:
                df_for_save = df_for_save.drop(columns=['_embedding_array'])
        else:
            # Convert numpy arrays to lists for JSON serialization
            def convert_embeddings(row):
                if pd.notna(row.get('_embedding_array')) and row['_embedding_array'] is not None:
                    return row['_embedding_array'].tolist()
                return None
                
            if '_embedding_array' in df_for_save.columns:
                df_for_save['embedding'] = df_for_save.apply(convert_embeddings, axis=1)
                df_for_save = df_for_save.drop(columns=['_embedding_array'])
        
        # Convert to dictionary for JSON
        json_data = {
            'video_id': self.video_id,
            'performance_metrics': self.performance_metrics,
            'frames': df_for_save.to_dict(orient='records')
        }
        
        # Save as JSON
        json_path = output_dir / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved metadata to JSON: {json_path}")
        
        return str(json_path)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the current dataframe.
        
        Returns:
            Pandas DataFrame with metadata
        """
        return self.df
    
    def get_embeddings_tensor(self) -> torch.Tensor:
        """
        Get all embeddings as a PyTorch tensor.
        
        Returns:
            Tensor containing all embeddings
        """
        if self.df is None or '_embedding_array' not in self.df.columns:
            return None
            
        # Filter to rows with embeddings
        embeddings_df = self.df.dropna(subset=['_embedding_array'])
        
        if len(embeddings_df) == 0:
            return None
            
        # Convert to list of numpy arrays
        embeddings_list = embeddings_df['_embedding_array'].tolist()
        
        # Stack into a single array
        embeddings_array = np.stack(embeddings_list)
        
        # Convert to tensor
        embeddings_tensor = torch.tensor(embeddings_array)
        
        return embeddings_tensor
    
    @staticmethod
    def merge_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple metadata dataframes.
        
        Args:
            dataframes: List of dataframes to merge
            
        Returns:
            Merged dataframe
        """
        if not dataframes:
            return None
            
        return pd.concat(dataframes, ignore_index=True)

    def process_metadata(
        self,
        video_id: str,
        metadata: Dict[str, Any],
        scenes: List[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Process metadata from various sources into a unified dataframe.
        
        Args:
            video_id: Unique identifier for the video
            metadata: Video metadata dictionary
            scenes: List of scene dictionaries
            audio_result: Audio processing results (optional)
            
        Returns:
            pandas DataFrame with processed metadata
        """
        # Create base dataframe from scenes
        df = pd.DataFrame(scenes)
        
        # Add video metadata as columns
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                df[f"video_{key}"] = value
        
        # Add audio information if available
        if audio_result:
            df["audio_file"] = audio_result.get("audio_file", "")
            df["transcript_file"] = audio_result.get("transcript_json", "")
            df["srt_file"] = audio_result.get("transcript_srt", "")
        
        # Add video ID
        df["video_id"] = video_id
        
        # Calculate additional metrics
        df["scene_duration"] = df["end_time"] - df["start_time"]
        df["frame_rate"] = df["duration_frames"] / df["duration_seconds"]
        
        # Sort by start time
        df = df.sort_values("start_time")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df 