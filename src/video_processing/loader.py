"""
Video Loader Module.

This module provides functions for loading and extracting metadata from video files.

📌 Purpose: Load video files and extract basic metadata
🔄 Latest Changes: Initial implementation
⚙️ Key Logic: Uses FFmpeg and OpenCV to load video files and extract metadata
📂 Expected File Path: src/video_processing/loader.py
🧠 Reasoning: Centralizes video loading logic and provides consistent
              interface for interacting with video files
"""

import os
import logging
import subprocess
from typing import Dict, Optional, Tuple, Any

import cv2
import ffmpeg
import numpy as np

from src.models.schema import VideoMetadata
from src.utils.gpu_utils import setup_device

logger = logging.getLogger(__name__)


def validate_video_file(file_path: str) -> bool:
    """
    Validate that a file exists and is a valid video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        probe = ffmpeg.probe(file_path)
        video_stream = next((stream for stream in probe['streams'] 
                             if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            logger.error(f"No video stream found in file: {file_path}")
            return False
        return True
    except ffmpeg.Error as e:
        logger.error(f"Error probing video file: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        return False


def extract_video_metadata(file_path: str) -> Optional[VideoMetadata]:
    """
    Extract metadata from a video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        VideoMetadata object if successful, None otherwise
    """
    if not validate_video_file(file_path):
        logger.error(f"extract_video_metadata: Validation failed for file: {file_path}")
        return None
    
    try:
        probe = ffmpeg.probe(file_path)
        video_stream = next((stream for stream in probe['streams'] 
                             if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            logger.error(f"extract_video_metadata: No video stream found in file: {file_path}")
            return None
        
        # Extract video properties
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Calculate duration
        duration = float(video_stream.get('duration', 0))
        if duration == 0 and 'format' in probe and 'duration' in probe['format']:
            duration = float(probe['format']['duration'])
        
        # Extract FPS
        fps_parts = video_stream.get('r_frame_rate', '30/1').split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        
        # Extract codec
        codec = video_stream.get('codec_name', 'unknown')
        
        # Create metadata object
        metadata = VideoMetadata(
            filename=os.path.basename(file_path),
            file_path=os.path.abspath(file_path),
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            codec=codec
        )
        
        return metadata
    
    except ffmpeg.Error as e:
        logger.error(f"extract_video_metadata: ffmpeg.Error for file {file_path}: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        return None
    except Exception as e:
        logger.error(f"extract_video_metadata: Unexpected error for file {file_path}: {str(e)}")
        return None


def create_video_capture(file_path: str) -> Optional[cv2.VideoCapture]:
    """
    Create an OpenCV VideoCapture object for a video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise
    """
    if not validate_video_file(file_path):
        return None
    
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {file_path}")
            return None
        return cap
    except Exception as e:
        logger.error(f"Error creating video capture: {str(e)}")
        return None


def extract_frame(cap: cv2.VideoCapture, timestamp: float) -> Optional[np.ndarray]:
    """
    Extract a specific frame from a video at the given timestamp.
    
    Args:
        cap: OpenCV VideoCapture object
        timestamp: Timestamp in seconds
        
    Returns:
        numpy.ndarray containing the frame if successful, None otherwise
    """
    try:
        # Calculate frame number from timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        # Set position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to extract frame at timestamp {timestamp}s")
            return None
        
        return frame
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        return None


def save_frame(frame: np.ndarray, output_path: str) -> bool:
    """
    Save a frame as an image file.
    
    Args:
        frame: Frame as numpy.ndarray
        output_path: Path to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return cv2.imwrite(output_path, frame)
    except Exception as e:
        logger.error(f"Error saving frame: {str(e)}")
        return False


def extract_audio(video_path: str, output_path: str, format: str = 'wav') -> bool:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file
        format: Audio format (default: 'wav')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract audio using ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le' if format == 'wav' else 'copy')
            .run(quiet=True, overwrite_output=True)
        )
        
        return os.path.exists(output_path)
    except ffmpeg.Error as e:
        logger.error(f"Error extracting audio: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error extracting audio: {str(e)}")
        return False


# Consider GPU availability for video decoding when possible
DEVICE = setup_device()
if DEVICE.type == 'cuda':
    logger.info("CUDA device available for video processing")
    # Note: OpenCV needs to be built with CUDA support for this to work
    # For now, we're just detecting it, but not changing behavior yet
    # In the future, we could use GPU-accelerated video decoding