"""
Core data models for the Video Timeline Analyzer.

This module defines the Pydantic models representing the core entities
in the application data model.

📌 Purpose: Define data structures for video processing
🔄 Latest Changes: Initial implementation of core models
⚙️ Key Logic: Pydantic models with validation and relationships
📂 Expected File Path: src/models/schema.py
🧠 Reasoning: Strongly typed models enable validation, serialization,
              and clear interfaces between components
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any

from pydantic import BaseModel, Field, validator


class VideoMetadata(BaseModel):
    """Metadata for a video file."""
    
    filename: str
    file_path: str
    duration: float = Field(..., description="Duration in seconds")
    width: int
    height: int
    fps: float
    codec: str
    created_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None


class SentimentType(str, Enum):
    """Types of sentiment classifications."""
    
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EmotionType(str, Enum):
    """Types of emotion classifications."""
    
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech."""
    
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str
    speaker: Optional[str] = None
    sentiment: Optional[SentimentType] = None
    sentiment_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('end_time')
    def end_time_must_be_after_start_time(cls, v, values):
        """Validate that end_time is after start_time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class BoundingBox(BaseModel):
    """A bounding box for an object in a frame."""
    
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")


class Face(BaseModel):
    """A detected face in a frame."""
    
    bbox: BoundingBox
    emotions: Dict[EmotionType, float] = Field(
        ..., description="Emotion probabilities"
    )
    age: Optional[float] = None
    gender: Optional[str] = None


class Frame(BaseModel):
    """A video frame with associated metadata."""
    
    timestamp: float = Field(..., description="Timestamp in seconds")
    file_path: Optional[str] = None
    faces: List[Face] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    caption: Optional[str] = None
    embedding: Optional[List[float]] = None


class AudioEvent(BaseModel):
    """A detected audio event."""
    
    event_type: str = Field(..., description="Type of audio event (e.g., 'laughter', 'applause')")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Scene(BaseModel):
    """A scene in a video."""
    
    id: Optional[int] = None
    video_id: Optional[int] = None
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    key_frames: List[Frame] = Field(default_factory=list)
    transcript_segments: List[TranscriptSegment] = Field(default_factory=list)
    audio_events: List[AudioEvent] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    viral_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    @validator('end_time')
    def end_time_must_be_after_start_time(cls, v, values):
        """Validate that end_time is after start_time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class Timeline(BaseModel):
    """A timeline of scenes for a video."""
    
    video_id: int
    video_metadata: VideoMetadata
    scenes: List[Scene] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    

class AnalysisConfig(BaseModel):
    """Configuration for video analysis pipeline."""
    
    # Scene detection settings
    scene_detection_method: str = Field("content", description="Method for scene detection ('content', 'threshold', 'hybrid')")
    scene_threshold: float = Field(30.0, description="Threshold for scene change detection")
    min_scene_length: float = Field(1.0, description="Minimum scene length in seconds")
    
    # Audio analysis settings
    transcribe_audio: bool = True
    diarize_speakers: bool = False
    analyze_audio_sentiment: bool = True
    detect_audio_events: bool = True
    
    # Visual analysis settings
    extract_key_frames: bool = True
    key_frames_method: str = Field("representative", description="Method for key frame extraction ('first', 'middle', 'representative')")
    detect_faces: bool = True
    generate_captions: bool = True
    generate_tags: bool = True
    
    # Performance settings
    use_gpu: bool = True
    batch_size: int = Field(16, description="Batch size for processing")
    max_workers: int = Field(4, description="Maximum number of worker processes")


class AnalysisResults(BaseModel):
    """Results from the video analysis pipeline."""
    
    video_metadata: VideoMetadata
    timeline: Timeline
    config: AnalysisConfig
    processing_time: float = Field(..., description="Total processing time in seconds")
    error_log: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True