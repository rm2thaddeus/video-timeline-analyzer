"""
Audio Transcription Module with GPU Acceleration.

This module provides functions for transcribing audio from videos
using OpenAI's Whisper model with GPU acceleration.

📌 Purpose: Transcribe audio with GPU acceleration
🔄 Latest Changes: Initial implementation with GPU support
⚙️ Key Logic: Uses Whisper model with GPU acceleration and memory optimization
📂 Expected File Path: src/audio_analysis/transcription.py
🧠 Reasoning: Speech-to-text is computationally intensive and benefits from
              GPU acceleration, especially for large models
"""

import os
import logging
import tempfile
from typing import List, Optional, Dict, Any, Union, Tuple

import numpy as np
import torch
import whisper
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

from src.models.schema import TranscriptSegment, SentimentType, Scene
from src.utils.gpu_utils import get_optimal_device, memory_stats, clear_gpu_memory
from src.video_processing.loader import extract_audio

logger = logging.getLogger(__name__)

# Get the optimal device (GPU if available, otherwise CPU)
DEVICE = get_optimal_device()
USE_GPU = DEVICE.type != 'cpu'

# Configure Whisper model size based on available GPU memory
def get_optimal_whisper_model_size() -> str:
    """
    Determine the optimal Whisper model size based on available GPU memory.
    
    Returns:
        str: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    """
    if not USE_GPU:
        logger.info("No GPU available, using 'small' Whisper model")
        return "small"
    
    # Get available GPU memory
    mem_stats = memory_stats()
    available_gb = mem_stats.get('free_gb', 0)
    
    # Model size recommendations based on VRAM:
    # - tiny: ~1GB VRAM
    # - base: ~1GB VRAM
    # - small: ~2GB VRAM
    # - medium: ~5GB VRAM
    # - large: ~10GB VRAM
    
    if available_gb < 1.5:
        logger.info("Limited GPU memory (<1.5GB), using 'tiny' Whisper model")
        return "tiny"
    elif available_gb < 2.5:
        logger.info("Limited GPU memory (<2.5GB), using 'base' Whisper model")
        return "base"
    elif available_gb < 5.0:
        logger.info("Moderate GPU memory (<5GB), using 'small' Whisper model")
        return "small"
    elif available_gb < 9.0:
        logger.info("Good GPU memory (<9GB), using 'medium' Whisper model")
        return "medium"
    else:
        logger.info("Abundant GPU memory (>9GB), using 'large' Whisper model")
        return "large"


class TranscriptionManager:
    """
    Manager for audio transcription with GPU acceleration.
    
    This class handles loading and using the Whisper model for transcription,
    with optimizations for GPU memory usage.
    """
    
    def __init__(self, model_size: Optional[str] = None, 
                device: Optional[torch.device] = None,
                use_fp16: bool = True):
        """
        Initialize the transcription manager.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use for computation
            use_fp16: Whether to use half-precision (FP16) for inference
        """
        self.device = device if device is not None else DEVICE
        self.use_gpu = self.device.type != 'cpu'
        
        # Determine model size if not specified
        if model_size is None:
            model_size = get_optimal_whisper_model_size()
        
        self.model_size = model_size
        self.use_fp16 = use_fp16 and self.use_gpu  # Only use FP16 with GPU
        
        # Model will be loaded on first use
        self.model = None
        
        logger.info(f"Initialized transcription manager with model_size={model_size}, "
                   f"device={self.device}, use_fp16={self.use_fp16}")
    
    def load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is not None:
            return
        
        logger.info(f"Loading Whisper model '{self.model_size}'...")
        
        try:
            # Load model with appropriate precision
            if self.use_fp16:
                self.model = whisper.load_model(
                    self.model_size, 
                    device=self.device,
                    download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
                ).half()  # Use half precision (FP16)
            else:
                self.model = whisper.load_model(
                    self.model_size, 
                    device=self.device,
                    download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
                )
            
            logger.info(f"Whisper model '{self.model_size}' loaded successfully")
            
            # Log memory usage
            if self.use_gpu:
                mem_stats = memory_stats()
                logger.info(f"GPU Memory after model load: {mem_stats['allocated_gb']:.2f}GB allocated, "
                           f"{mem_stats['reserved_gb']:.2f}GB reserved")
        
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def unload_model(self):
        """Unload the model to free GPU memory."""
        if self.model is not None:
            logger.info("Unloading Whisper model to free memory")
            self.model = None
            
            # Clear CUDA cache if using GPU
            if self.use_gpu:
                clear_gpu_memory()
                
                # Log memory usage
                mem_stats = memory_stats()
                logger.info(f"GPU Memory after model unload: {mem_stats['allocated_gb']:.2f}GB allocated, "
                           f"{mem_stats['reserved_gb']:.2f}GB reserved")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None,
                  task: str = "transcribe", verbose: bool = False,
                  word_timestamps: bool = False) -> List[TranscriptSegment]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            task: Task to perform ('transcribe' or 'translate')
            verbose: Whether to print detailed output
            word_timestamps: Whether to generate word-level timestamps
            
        Returns:
            List[TranscriptSegment]: List of transcript segments with timestamps
        """
        # Load model if not already loaded
        self.load_model()
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            # Set transcription options
            options = {
                "task": task,
                "verbose": verbose,
            }
            
            if language:
                options["language"] = language
            
            # Add word timestamps option if available and requested
            # Note: word_timestamps requires WhisperX or newer Whisper versions
            if word_timestamps and hasattr(whisper, "transcribe") and "word_timestamps" in whisper.transcribe.__code__.co_varnames:
                options["word_timestamps"] = True
            
            # Transcribe audio
            result = self.model.transcribe(audio_path, **options)
            
            # Convert result to TranscriptSegment objects
            segments = []
            
            for segment in result["segments"]:
                transcript_segment = TranscriptSegment(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"].strip(),
                )
                segments.append(transcript_segment)
            
            logger.info(f"Transcription complete: {len(segments)} segments")
            
            return segments
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
        
        finally:
            # Optionally unload model to free memory
            # Uncomment if you want to free memory after each transcription
            # self.unload_model()
            pass


class SentimentAnalyzer:
    """
    Analyzer for sentiment in transcribed text with GPU acceleration.
    
    This class handles loading and using a sentiment analysis model,
    with optimizations for GPU memory usage.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                device: Optional[torch.device] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the sentiment analysis model
            device: Device to use for computation
        """
        self.device = device if device is not None else DEVICE
        self.use_gpu = self.device.type != 'cpu'
        self.model_name = model_name
        
        # Model will be loaded on first use
        self.pipeline = None
        
        logger.info(f"Initialized sentiment analyzer with model={model_name}, device={self.device}")
    
    def load_model(self):
        """Load the sentiment analysis model if not already loaded."""
        if self.pipeline is not None:
            return
        
        logger.info(f"Loading sentiment analysis model '{self.model_name}'...")
        
        try:
            # Load model with appropriate device
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.use_gpu else -1  # 0 for GPU, -1 for CPU
            )
            
            logger.info(f"Sentiment analysis model loaded successfully")
            
            # Log memory usage
            if self.use_gpu:
                mem_stats = memory_stats()
                logger.info(f"GPU Memory after model load: {mem_stats['allocated_gb']:.2f}GB allocated, "
                           f"{mem_stats['reserved_gb']:.2f}GB reserved")
        
        except Exception as e:
            logger.error(f"Error loading sentiment analysis model: {str(e)}")
            raise
    
    def unload_model(self):
        """Unload the model to free GPU memory."""
        if self.pipeline is not None:
            logger.info("Unloading sentiment analysis model to free memory")
            self.pipeline = None
            
            # Clear CUDA cache if using GPU
            if self.use_gpu:
                clear_gpu_memory()
                
                # Log memory usage
                mem_stats = memory_stats()
                logger.info(f"GPU Memory after model unload: {mem_stats['allocated_gb']:.2f}GB allocated, "
                           f"{mem_stats['reserved_gb']:.2f}GB reserved")
    
    def analyze_sentiment(self, text: str) -> Tuple[SentimentType, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple[SentimentType, float]: Sentiment type and confidence score
        """
        # Load model if not already loaded
        self.load_model()
        
        try:
            # Skip empty or very short text
            if not text or len(text.strip()) < 3:
                return SentimentType.NEUTRAL, 0.5
            
            # Analyze sentiment
            result = self.pipeline(text)[0]
            
            # Map label to SentimentType
            label = result["label"].lower()
            score = result["score"]
            
            if "positive" in label:
                sentiment_type = SentimentType.POSITIVE
            elif "negative" in label:
                sentiment_type = SentimentType.NEGATIVE
            else:
                sentiment_type = SentimentType.NEUTRAL
            
            return sentiment_type, score
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return SentimentType.NEUTRAL, 0.5
    
    def analyze_transcript_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Analyze sentiment of transcript segments.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List[TranscriptSegment]: Updated transcript segments with sentiment
        """
        logger.info(f"Analyzing sentiment for {len(segments)} transcript segments")
        
        # Load model if not already loaded
        self.load_model()
        
        # Process segments in batches for efficiency
        batch_size = 16
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            
            # Extract text from segments
            texts = [segment.text for segment in batch]
            
            try:
                # Skip empty texts
                valid_indices = [i for i, text in enumerate(texts) if text and len(text.strip()) >= 3]
                valid_texts = [texts[i] for i in valid_indices]
                
                if valid_texts:
                    # Analyze sentiment for valid texts
                    results = self.pipeline(valid_texts)
                    
                    # Update segments with sentiment
                    for idx, result in zip(valid_indices, results):
                        label = result["label"].lower()
                        score = result["score"]
                        
                        if "positive" in label:
                            sentiment_type = SentimentType.POSITIVE
                        elif "negative" in label:
                            sentiment_type = SentimentType.NEGATIVE
                        else:
                            sentiment_type = SentimentType.NEUTRAL
                        
                        batch[idx].sentiment = sentiment_type
                        batch[idx].sentiment_score = score
                
                # Set neutral sentiment for invalid texts
                for i in range(len(batch)):
                    if i not in valid_indices:
                        batch[i].sentiment = SentimentType.NEUTRAL
                        batch[i].sentiment_score = 0.5
            
            except Exception as e:
                logger.error(f"Error analyzing sentiment for batch: {str(e)}")
                # Set neutral sentiment for all segments in the batch
                for segment in batch:
                    segment.sentiment = SentimentType.NEUTRAL
                    segment.sentiment_score = 0.5
        
        logger.info("Sentiment analysis complete")
        
        return segments


def transcribe_video(video_path: str, output_dir: Optional[str] = None,
                    model_size: Optional[str] = None,
                    language: Optional[str] = None,
                    analyze_sentiment: bool = True) -> List[TranscriptSegment]:
    """
    Transcribe audio from a video file and optionally analyze sentiment.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted audio and transcripts
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g., 'en', 'es', 'fr')
        analyze_sentiment: Whether to analyze sentiment of transcribed text
        
    Returns:
        List[TranscriptSegment]: List of transcript segments with timestamps and sentiment
    """
    logger.info(f"Transcribing video: {video_path}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract audio from video
    audio_path = os.path.join(output_dir, "audio.wav") if output_dir else tempfile.mktemp(suffix=".wav")
    
    try:
        # Extract audio
        extract_audio(video_path, audio_path)
        
        # Initialize transcription manager
        transcription_manager = TranscriptionManager(model_size=model_size)
        
        # Transcribe audio
        segments = transcription_manager.transcribe(
            audio_path,
            language=language,
            verbose=True
        )
        
        # Analyze sentiment if requested
        if analyze_sentiment:
            sentiment_analyzer = SentimentAnalyzer()
            segments = sentiment_analyzer.analyze_transcript_segments(segments)
        
        # Save transcript to file if output directory is specified
        if output_dir:
            # Save as SRT
            srt_path = os.path.join(output_dir, "transcript.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    f.write(f"{i+1}\n")
                    
                    # Format timestamps as HH:MM:SS,mmm
                    start_time_str = format_timestamp(segment.start_time)
                    end_time_str = format_timestamp(segment.end_time)
                    
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"{segment.text}\n\n")
            
            # Save as JSON
            json_path = os.path.join(output_dir, "transcript.json")
            import json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([segment.dict() for segment in segments], f, indent=2)
        
        return segments
    
    finally:
        # Clean up temporary audio file
        if not output_dir and os.path.exists(audio_path):
            os.remove(audio_path)


def format_timestamp(seconds: float) -> str:
    """
    Format timestamp as HH:MM:SS,mmm for SRT files.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def assign_transcripts_to_scenes(scenes: List[Scene], 
                               transcripts: List[TranscriptSegment]) -> List[Scene]:
    """
    Assign transcript segments to scenes based on timestamps.
    
    Args:
        scenes: List of scenes
        transcripts: List of transcript segments
        
    Returns:
        List[Scene]: Updated list of scenes with transcript segments
    """
    logger.info(f"Assigning {len(transcripts)} transcript segments to {len(scenes)} scenes")
    
    # Sort scenes by start time
    sorted_scenes = sorted(scenes, key=lambda s: s.start_time)
    
    # Assign each transcript segment to the appropriate scene
    for transcript in transcripts:
        # Find the scene that contains this transcript
        assigned = False
        
        for scene in sorted_scenes:
            # Check if transcript is within scene boundaries
            if (transcript.start_time >= scene.start_time and 
                transcript.end_time <= scene.end_time):
                # Transcript is fully within scene
                scene.transcript_segments.append(transcript)
                assigned = True
                break
            elif (transcript.start_time < scene.end_time and 
                  transcript.end_time > scene.start_time):
                # Transcript overlaps with scene
                # Assign to the scene where the majority of the transcript lies
                transcript_duration = transcript.end_time - transcript.start_time
                
                # Calculate overlap with scene
                overlap_start = max(transcript.start_time, scene.start_time)
                overlap_end = min(transcript.end_time, scene.end_time)
                overlap_duration = overlap_end - overlap_start
                
                # If more than 50% of the transcript is in this scene, assign it
                if overlap_duration > transcript_duration * 0.5:
                    scene.transcript_segments.append(transcript)
                    assigned = True
                    break
        
        # If transcript wasn't assigned to any scene, assign it to the closest one
        if not assigned and transcripts:
            # Find scene with closest start time
            closest_scene = min(
                sorted_scenes, 
                key=lambda s: min(
                    abs(transcript.start_time - s.start_time),
                    abs(transcript.start_time - s.end_time)
                )
            )
            closest_scene.transcript_segments.append(transcript)
    
    # Sort transcript segments within each scene by start time
    for scene in sorted_scenes:
        scene.transcript_segments.sort(key=lambda t: t.start_time)
    
    logger.info("Transcript assignment complete")
    
    return sorted_scenes