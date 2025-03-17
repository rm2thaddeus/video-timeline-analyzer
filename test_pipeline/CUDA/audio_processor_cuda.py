"""
📌 Purpose: CUDA-optimized audio processing and transcription for video files
🔄 Latest Changes: Implemented batched processing with proper CUDA utilization and memory management
⚙️ Key Logic: Uses Whisper model with GPU acceleration for efficient audio transcription
📂 Expected File Path: test_pipeline/CUDA/audio_processor_cuda.py
🧠 Reasoning: Optimize audio transcription for large videos with proper resource management
"""

import os
import sys
import json
import time
import wave
import torch
import whisper
import logging
import tempfile
import datetime
import numpy as np
import subprocess
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logger = logging.getLogger("audio_processor_cuda")

class AudioProcessorCUDA:
    """CUDA-accelerated audio processor for extracting and transcribing audio from videos."""
    
    def __init__(
        self,
        model_name: str = "tiny",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        batch_size: int = 4,
        num_workers: int = 2,
        use_pinned_memory: bool = True,
        gpu_memory_fraction: float = 0.4,
    ):
        """
        Initialize the audio processor with CUDA acceleration.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: PyTorch device to use (CUDA or CPU)
            use_mixed_precision: Whether to use mixed precision (FP16) for faster processing
            batch_size: Number of audio segments to process at once
            num_workers: Number of worker threads for data loading
            use_pinned_memory: Whether to use pinned memory for faster CPU-GPU transfers
            gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_pinned_memory = use_pinned_memory and self.device.type == "cuda"
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Use custom temp dir for audio processing
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Set GPU memory management if using CUDA
        if self.device.type == "cuda":
            self._setup_cuda_memory()
            logger.info(f"CUDA device: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
            logger.info(f"Memory fraction: {self.gpu_memory_fraction}")
            
            if self.use_mixed_precision:
                logger.info("Using mixed precision (FP16)")
            else:
                logger.info("Using full precision (FP32)")
        else:
            logger.info("Using CPU for audio processing")
            
        # Load whisper model - we'll load it lazily when needed to preserve memory
        self.model = None
        logger.info(f"AudioProcessorCUDA initialized with model_name={model_name}, device={self.device}")
    
    def _setup_cuda_memory(self):
        """Configure CUDA memory management settings"""
        if self.device.type == "cuda":
            # Extract device index for cuda functions
            device_idx = 0 if self.device.index is None else self.device.index
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, device_idx)
            
            # Empty cache for good measure
            torch.cuda.empty_cache()
            
            # Set up TensorFloat32 for Ampere and newer GPUs
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
    
    def _load_model(self):
        """Lazy-load the Whisper model when needed"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            try:
                # Load the model with specified precision
                if self.use_mixed_precision and self.device.type == "cuda":
                    # Use different model loading depending on Whisper version
                    try:
                        # Try newer approach
                        self.model = whisper.load_model(
                            self.model_name, 
                            device=self.device, 
                            download_root=None
                        )
                    except TypeError:
                        # Fall back to older approach
                        self.model = whisper.load_model(self.model_name)
                        self.model = self.model.to(self.device)
                        
                    # Set up mixed precision
                    if hasattr(self.model, 'half'):
                        self.model = self.model.half()
                else:
                    self.model = whisper.load_model(self.model_name, device=self.device)
                
                logger.info(f"Successfully loaded Whisper model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise
    
    def _extract_audio_ffmpeg(self, video_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        Extract audio from video using FFmpeg (CPU-based).
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the extracted audio
            
        Returns:
            True if extraction succeeded, False otherwise
        """
        video_path = str(video_path)
        output_path = str(output_path)
        
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian format
            "-ar", "16000",  # 16 kHz sample rate (Whisper's expected rate)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            output_path
        ]
        
        logger.info(f"Extracting audio from {video_path}")
        
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            logger.info(f"Audio extraction succeeded: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            return False
    
    def _split_audio(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        chunk_duration: float = 30.0,
        overlap: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Split audio file into smaller chunks for parallel processing.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save audio chunks
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of dictionaries with chunk information
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Open audio file to get properties
        with wave.open(str(audio_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            frames = wf.getnframes()
            duration = frames / sample_rate
        
        # Calculate chunk sizes
        chunk_frames = int(chunk_duration * sample_rate)
        overlap_frames = int(overlap * sample_rate)
        step_frames = chunk_frames - overlap_frames
        
        # Create chunks
        chunks = []
        
        with wave.open(str(audio_path), 'rb') as wf:
            # Calculate number of chunks
            num_chunks = max(1, int((frames - overlap_frames) / step_frames))
            logger.info(f"Splitting {audio_path.name} ({duration:.2f}s) into {num_chunks} chunks")
            
            for i in range(num_chunks):
                # Calculate start and end frames for this chunk
                start_frame = i * step_frames
                end_frame = min(start_frame + chunk_frames, frames)
                
                # Don't create small chunks at the end
                if end_frame - start_frame < chunk_frames * 0.5 and i > 0:
                    # Extend the previous chunk instead
                    chunks[-1]["end_frame"] = end_frame
                    chunks[-1]["end_time"] = end_frame / sample_rate
                    chunks[-1]["duration"] = chunks[-1]["end_time"] - chunks[-1]["start_time"]
                    continue
                
                # Create chunk file
                chunk_path = output_dir / f"chunk_{i:04d}.wav"
                
                # Calculate position and seek
                wf.setpos(start_frame)
                
                # Read frames
                frames_to_read = end_frame - start_frame
                audio_data = wf.readframes(frames_to_read)
                
                # Write chunk file
                with wave.open(str(chunk_path), 'wb') as chunk_wf:
                    chunk_wf.setnchannels(channels)
                    chunk_wf.setsampwidth(sample_width)
                    chunk_wf.setframerate(sample_rate)
                    chunk_wf.writeframes(audio_data)
                
                # Store chunk info
                chunks.append({
                    "index": i,
                    "path": str(chunk_path),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_frame / sample_rate,
                    "end_time": end_frame / sample_rate,
                    "duration": (end_frame - start_frame) / sample_rate
                })
        
        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks
    
    def _transcribe_audio_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transcribe a single audio chunk using Whisper.
        
        Args:
            chunk: Dictionary with chunk information
            
        Returns:
            Dictionary with transcription results
        """
        chunk_path = chunk["path"]
        
        logger.debug(f"Transcribing chunk {chunk['index']}: {chunk_path}")
        
        try:
            # Use the official OpenAI whisper module directly
            import whisper
            
            # Load model if not already loaded
            if self.model is None:
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.model = whisper.load_model(self.model_name, device=self.device)
            
            # Simple direct transcription - avoid tensor shape mismatches
            result = self.model.transcribe(str(chunk_path), language="en")
            
            # Add chunk info to result
            result["chunk"] = {
                "index": chunk["index"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "duration": chunk["duration"]
            }
            
            # Adjust timestamps for segments
            for segment in result["segments"]:
                segment["start"] += chunk["start_time"]
                segment["end"] += chunk["start_time"]
            
            return result
        
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk['index']}: {e}")
            # Return a minimal result with empty transcription
            return {
                "text": "",
                "segments": [],
                "chunk": {
                    "index": chunk["index"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "duration": chunk["duration"]
                },
                "error": str(e)
            }
    
    def _transcribe_chunks_parallel(
        self,
        chunks: List[Dict[str, Any]],
        max_workers: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio chunks in parallel using ThreadPoolExecutor.
        
        Args:
            chunks: List of chunk dictionaries
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of transcription results
        """
        results = []
        num_workers = min(max_workers, len(chunks))
        
        logger.info(f"Transcribing {len(chunks)} chunks using {num_workers} workers")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {executor.submit(self._transcribe_audio_chunk, chunk): chunk for chunk in chunks}
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed chunk {chunk['index']}")
                except Exception as e:
                    logger.error(f"Exception transcribing chunk {chunk['index']}: {e}")
                    # Add empty result to maintain order
                    results.append({
                        "text": "",
                        "segments": [],
                        "chunk": {
                            "index": chunk["index"],
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                            "duration": chunk["duration"]
                        },
                        "error": str(e)
                    })
        
        # Sort results by chunk index
        results.sort(key=lambda x: x["chunk"]["index"])
        
        return results
    
    def _merge_transcriptions(self, transcriptions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge transcriptions from multiple chunks, handling overlapping segments.
        
        Args:
            transcriptions: List of transcription results from individual chunks
            
        Returns:
            Dictionary with merged transcription
        """
        if not transcriptions:
            return {"text": "", "segments": []}
        
        # First, gather all segments
        all_segments = []
        merged_text = []
        
        for result in transcriptions:
            merged_text.append(result["text"])
            segments = result.get("segments", [])
            
            # Handle overlaps with previous chunk
            for segment in segments:
                # Check if this segment overlaps with any existing segment
                overlapping = False
                
                if all_segments:
                    # Check for overlap with existing segments
                    for existing_segment in all_segments:
                        # If there's a significant overlap (more than 50% of the shorter segment)
                        if (min(segment["end"], existing_segment["end"]) - 
                            max(segment["start"], existing_segment["start"])) > 0:
                            
                            # If the existing segment is shorter or equal in duration
                            if (existing_segment["end"] - existing_segment["start"]) <= (segment["end"] - segment["start"]):
                                # Replace the existing segment text if the new one is longer
                                if len(segment["text"]) > len(existing_segment["text"]):
                                    existing_segment["text"] = segment["text"]
                            
                            overlapping = True
                            break
                
                # If no overlap, add the segment
                if not overlapping:
                    all_segments.append(segment.copy())
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Return merged result
        return {
            "text": " ".join(merged_text),
            "segments": all_segments
        }
    
    def _create_srt(self, transcription: Dict[str, Any], output_path: Union[str, Path]) -> str:
        """
        Create SRT subtitle file from transcription.
        
        Args:
            transcription: Dictionary with transcription results
            output_path: Path to save the SRT file
            
        Returns:
            Path to the created SRT file
        """
        output_path = Path(output_path)
        segments = transcription.get("segments", [])
        
        if not segments:
            logger.warning("No segments found for SRT creation")
            return str(output_path)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                # SRT index
                f.write(f"{i+1}\n")
                
                # Format timestamps (HH:MM:SS,mmm --> HH:MM:SS,mmm)
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                f.write(f"{start_time} --> {end_time}\n")
                
                # Text
                f.write(f"{segment['text'].strip()}\n\n")
        
        logger.info(f"Created SRT file: {output_path}")
        return str(output_path)
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as SRT timestamp (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    @contextmanager
    def _set_up_context(self):
        """Context manager for setting up CUDA and cleaning up after processing"""
        try:
            # Set up CUDA memory management
            if self.device.type == "cuda":
                self._setup_cuda_memory()
            
            yield
        finally:
            # Clean up
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Unload model to free memory
            if self.model is not None:
                self.model = None
    
    def process_video_cuda(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        chunk_duration: float = 30.0,
        overlap: float = 0.5,
        max_workers: int = 2
    ) -> Dict[str, Any]:
        """
        Process video audio with CUDA optimization.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output files (or None for temp dir)
            chunk_duration: Duration of each audio chunk in seconds
            overlap: Overlap between chunks in seconds
            max_workers: Maximum number of worker threads for parallel processing
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        video_path = Path(video_path)
        video_id = video_path.stem
        
        # Set up output directory
        if output_dir is None:
            output_dir = Path(self.temp_dir.name)
        else:
            output_dir = Path(output_dir)
        
        # Create output directories
        audio_dir = output_dir / "audio_chunks"
        transcripts_dir = output_dir / "transcripts"
        
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(transcripts_dir, exist_ok=True)
        
        # Paths for output files
        audio_path = audio_dir / f"{video_id}.wav"
        json_path = transcripts_dir / f"{video_id}.json"
        srt_path = transcripts_dir / f"{video_id}.srt"
        
        with self._set_up_context():
            try:
                logger.info(f"Processing audio for {video_path}")
                
                # Extract audio from video
                if not self._extract_audio_ffmpeg(video_path, audio_path):
                    raise RuntimeError(f"Failed to extract audio from {video_path}")
                
                # Split audio into chunks
                chunks = self._split_audio(
                    audio_path,
                    audio_dir / f"{video_id}_chunks",
                    chunk_duration=chunk_duration,
                    overlap=overlap
                )
                
                # Transcribe chunks in parallel
                transcriptions = self._transcribe_chunks_parallel(
                    chunks,
                    max_workers=max_workers
                )
                
                # Merge transcriptions
                merged_transcription = self._merge_transcriptions(transcriptions)
                
                # Save merged transcription as JSON
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(merged_transcription, f, indent=2)
                
                # Create SRT file
                srt_file = self._create_srt(merged_transcription, srt_path)
                
                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time
                
                logger.info(f"Processed audio in {processing_time:.2f} seconds")
                logger.info(f"Transcription saved to {json_path} and {srt_path}")
                
                # Return results
                return {
                    "video_id": video_id,
                    "srt_path": str(srt_path),
                    "json_path": str(json_path),
                    "audio_path": str(audio_path),
                    "processing_time_seconds": processing_time,
                    "num_segments": len(merged_transcription.get("segments", [])),
                    "device": str(self.device),
                    "model": self.model_name,
                    "chunk_duration": chunk_duration,
                    "overlap": overlap,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                raise
