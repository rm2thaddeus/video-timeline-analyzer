"""
📌 Purpose: CUDA-accelerated audio processing with Whisper
🔄 Latest Changes: Added model type support and improved error handling
⚙️ Key Logic: Process audio with Whisper model using CUDA acceleration
📂 Expected File Path: test_pipeline/CUDA/audio_processor_cuda.py
🧠 Reasoning: Optimized audio processing for GPU acceleration
"""

import os
import json
import torch
import whisper
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import gc

# Import GPU utils from the correct path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_utils import get_memory_info, detect_gpu, setup_device
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

class AudioProcessorCUDA:
    """CUDA-accelerated audio processing class."""
    
    def __init__(self, model_type: str = "medium") -> None:
        """Initialize the audio processor.
        
        Args:
            model_type: Whisper model type (tiny, base, small, medium, large)
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Whisper {model_type} model on {self.device}")
        self.model = whisper.load_model(model_type).to(self.device)
        
        # Get GPU information
        gpu_info = detect_gpu()
        memory_info = get_memory_info()
        
        if gpu_info["detected"]:
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"Memory allocated: {memory_info['allocated_gb']:.2f} GB")
            logger.info(f"Memory free: {memory_info['free_gb']:.2f} GB")
            logger.info(f"Using mixed precision (FP16)")
        else:
            logger.warning("No GPU detected, using CPU")
        
        logger.info(f"AudioProcessorCUDA initialized with model_type={model_type}, device={self.device}")

    def _ensure_dirs(self, video_id: str) -> Tuple[Path, Path, Path]:
        """Ensure all necessary directories exist."""
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        audio_dir = base_dir / "transcripts" / "audio_chunks"
        chunks_dir = audio_dir / f"{video_id}_chunks"
        transcript_dir = base_dir / "transcripts" / "transcripts"
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        return audio_dir, chunks_dir, transcript_dir

    def process_video(self, video_path: str, chunk_duration: int = 30, chunk_overlap: float = 0.5) -> Dict:
        video_id = Path(video_path).stem
        audio_dir, chunks_dir, transcript_dir = self._ensure_dirs(video_id)
        
        # Extract audio
        audio_path = audio_dir / f"{video_id}.wav"
        logger.info(f"Extracting audio from {video_path}")
        
        try:
            import subprocess
            command = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-ac', '1',
                '-ar', '16000',
                str(audio_path),
                '-hide_banner',
                '-loglevel', 'error'
            ]
            subprocess.run(command, check=True, capture_output=True)
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Failed to extract audio from {video_path}")
            logger.info(f"Audio extraction succeeded: {audio_path}")
            
            # Load and split audio
            audio, sr = librosa.load(str(audio_path), sr=16000)
            duration = len(audio) / sr
            chunk_samples = int(chunk_duration * sr)
            overlap_samples = int(chunk_overlap * sr)
            
            # Create chunks directory if it doesn't exist
            chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # Create chunks
            chunks = []
            for i in range(0, len(audio), chunk_samples - overlap_samples):
                chunk = audio[i:i + chunk_samples]
                if len(chunk) < sr:  # Skip chunks shorter than 1 second
                    continue
                
                chunk_path = chunks_dir / f"chunk_{len(chunks):04d}.wav"
                sf.write(str(chunk_path), chunk, sr)
                chunks.append({
                    "path": str(chunk_path),
                    "start": i / sr,
                    "end": min((i + len(chunk)) / sr, duration)
                })
            
            logger.info(f"Split {video_id}.wav ({duration:.2f}s) into {len(chunks)} chunks")
            
            # Transcribe chunks
            num_workers = min(2, len(chunks))
            logger.info(f"Transcribing {len(chunks)} chunks using {num_workers} workers")
            
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._transcribe_audio_chunk, chunk) for chunk in chunks]
                for future in tqdm(as_completed(futures), total=len(chunks), desc="Transcribing"):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error in transcription worker: {str(e)}")
            
            # Sort results by start time
            results.sort(key=lambda x: x["start"])
            
            # Create transcripts directory if it doesn't exist
            transcript_dir.mkdir(parents=True, exist_ok=True)
            
            # Save transcripts
            transcript_json = transcript_dir / f"{video_id}.json"
            transcript_srt = transcript_dir / f"{video_id}.srt"
            
            with open(transcript_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Create SRT file
            with open(transcript_srt, "w", encoding="utf-8") as f:
                for i, segment in enumerate(results, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._format_timestamp(segment['start'])} --> {self._format_timestamp(segment['end'])}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
            
            logger.info(f"Created SRT file: {transcript_srt}")
            logger.info(f"Transcription saved to {transcript_json} and {transcript_srt}")
            
            return {
                "video_id": video_id,
                "duration": duration,
                "num_chunks": len(chunks),
                "transcript_json": str(transcript_json),
                "transcript_srt": str(transcript_srt)
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _transcribe_audio_chunk(self, chunk: Dict) -> Optional[Dict]:
        """Transcribe a single audio chunk."""
        try:
            result = self.model.transcribe(
                chunk["path"],
                task="transcribe",  # Use transcribe to maintain original language
                language=None,  # Let the model auto-detect the language
                fp16=True
            )
            
            # Get the detected language from the first segment
            detected_lang = result["language"] if "language" in result else None
            
            return {
                "start": chunk["start"],
                "end": chunk["end"],
                "text": result["text"],
                "language": detected_lang
            }
        except Exception as e:
            logger.error(f"Error transcribing chunk: {str(e)}")
            return None

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio file with Whisper model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcription data
        """
        try:
            logger.info(f"Processing audio file: {audio_path}")
            
            # Load and transcribe audio
            result = self.model.transcribe(audio_path)
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"]
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                "text": "",
                "segments": [],
                "language": None,
                "error": str(e)
            }
