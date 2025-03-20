"""
📌 Purpose: Process and transcribe audio from video files
🔄 Latest Changes: Fixed import issue
⚙️ Key Logic: Extract audio, segment it, and transcribe with Whisper
📂 Expected File Path: test_pipeline/processors/audio_processor.py
🧠 Reasoning: Separate audio processing from other components for better modularity
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import torch
import numpy as np
import whisper
from tqdm import tqdm

# Fix the import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.gpu_utils import setup_device

logger = logging.getLogger("video_pipeline.audio_processor")

class AudioProcessor:
    def __init__(self, model_name: str = "base", device: Optional[torch.device] = None):
        """
        Initialize audio processor with Whisper model for transcription.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Torch device for model inference
        """
        self.model_name = model_name
        
        if device is None:
            self.device, _ = setup_device()
        else:
            self.device = device
            
        logger.info(f"Initializing Whisper model '{model_name}' on {self.device}")
        self.model = whisper.load_model(model_name).to(self.device)
        logger.info(f"Whisper model loaded successfully")
        
    def extract_audio(self, video_path: str, output_dir: Optional[str] = None) -> str:
        """
        Extract audio from a video file using FFmpeg.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the extracted audio file (optional)
            
        Returns:
            Path to the extracted audio file
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            audio_path = output_dir / f"{video_id}_audio.wav"
        else:
            # Use a temporary file if no output directory is specified
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = Path(temp_file.name)
            temp_file.close()
        
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        
        try:
            # Extract audio using FFmpeg
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit little-endian audio
                "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
                "-ac", "1",  # Mono
                "-y",  # Overwrite output file if it exists
                str(audio_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio extraction completed successfully")
            
            return str(audio_path)
            
        except subprocess.SubprocessError as e:
            logger.error(f"FFmpeg error during audio extraction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}")
            raise
    
    def segment_audio(self, audio_path: str, output_dir: str, 
                      chunk_duration: float = 30.0, overlap: float = 2.0) -> List[Dict[str, Any]]:
        """
        Segment audio file into chunks for transcription.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save the audio chunks
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of dictionaries with chunk information
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        audio_id = audio_path.stem
        
        logger.info(f"Segmenting audio {audio_path} into {chunk_duration}s chunks with {overlap}s overlap")
        
        try:
            # Get audio duration using FFprobe
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(json.loads(result.stdout)["format"]["duration"])
            
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Calculate segment boundaries
            segments = []
            start_time = 0.0
            
            while start_time < duration:
                end_time = min(start_time + chunk_duration, duration)
                
                chunk_id = f"{audio_id}_chunk_{len(segments):03d}"
                chunk_path = output_dir / f"{chunk_id}.wav"
                
                # Extract chunk using FFmpeg
                cmd = [
                    "ffmpeg",
                    "-ss", str(start_time),
                    "-i", str(audio_path),
                    "-t", str(end_time - start_time),
                    "-c:a", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",
                    str(chunk_path)
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)
                
                segments.append({
                    "chunk_id": chunk_id,
                    "audio_id": audio_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "path": str(chunk_path)
                })
                
                # Move to next segment with overlap
                start_time = end_time - overlap
                
                if start_time >= duration:
                    break
            
            logger.info(f"Created {len(segments)} audio chunks")
            
            # Save segment info
            segments_file = output_dir / f"{audio_id}_segments.json"
            with open(segments_file, 'w') as f:
                json.dump(segments, f, indent=2)
                
            return segments
            
        except subprocess.SubprocessError as e:
            logger.error(f"FFmpeg/FFprobe error during audio segmentation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during audio segmentation: {e}")
            raise
    
    def transcribe_segments(self, segments: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio segments using Whisper.
        
        Args:
            segments: List of segment dictionaries from segment_audio()
            output_dir: Directory to save transcription results
            
        Returns:
            List of dictionaries with transcription results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Transcribing {len(segments)} audio segments with Whisper model '{self.model_name}'")
        
        transcriptions = []
        
        for segment in tqdm(segments, desc="Transcribing segments"):
            chunk_id = segment["chunk_id"]
            audio_path = segment["path"]
            
            try:
                # Transcribe with Whisper
                result = self.model.transcribe(
                    audio_path,
                    temperature=0.0,
                    word_timestamps=True,
                )
                
                # Create transcription entry
                transcription = {
                    "chunk_id": chunk_id,
                    "audio_id": segment["audio_id"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "text": result["text"],
                    "segments": result["segments"],
                    "language": result.get("language", "")
                }
                
                # Save individual transcription
                output_file = output_dir / f"{chunk_id}_transcript.json"
                with open(output_file, 'w') as f:
                    json.dump(transcription, f, indent=2)
                
                transcriptions.append(transcription)
                
            except Exception as e:
                logger.error(f"Error transcribing segment {chunk_id}: {e}")
                # Continue with other segments instead of failing completely
                continue
        
        # Save combined transcriptions
        if transcriptions:
            audio_id = transcriptions[0]["audio_id"]
            combined_file = output_dir / f"{audio_id}_transcripts.json"
            with open(combined_file, 'w') as f:
                json.dump(transcriptions, f, indent=2)
                
            logger.info(f"Transcribed {len(transcriptions)} segments successfully")
        
        return transcriptions
    
    def process_video(self, video_path: str, output_dir: Optional[str] = None, 
                      chunk_duration: float = 30.0, overlap: float = 2.0) -> Dict[str, Any]:
        """
        Process a video file: extract audio, segment, and transcribe.
        
        Args:
            video_path: Path to the video file
            output_dir: Base directory for output files
            chunk_duration: Duration of each audio chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            Dictionary with processing results
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        
        if output_dir:
            output_dir = Path(output_dir)
            audio_dir = output_dir / "audio_chunks"
            transcript_dir = output_dir / "transcripts"
        else:
            # Use temporary directories if no output directory is specified
            temp_dir = tempfile.mkdtemp()
            audio_dir = Path(temp_dir) / "audio_chunks"
            transcript_dir = Path(temp_dir) / "transcripts"
        
        audio_dir.mkdir(exist_ok=True)
        transcript_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing video {video_path} for audio transcription")
        
        try:
            # Extract audio
            audio_path = self.extract_audio(video_path, audio_dir)
            
            # Segment audio
            segments = self.segment_audio(audio_path, audio_dir, chunk_duration, overlap)
            
            # Transcribe segments
            transcriptions = self.transcribe_segments(segments, transcript_dir)
            
            result = {
                "video_id": video_id,
                "audio_path": audio_path,
                "segments": segments,
                "transcriptions": transcriptions
            }
            
            # Save processing result
            result_file = transcript_dir / f"{video_id}_processing_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.info(f"Video audio processing completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during video audio processing: {e}")
            raise 