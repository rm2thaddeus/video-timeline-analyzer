# Video Timeline Analyzer: Technical Specifications

This document provides detailed technical specifications for each component of the Video Timeline Analyzer, including requirements, interfaces, and performance considerations.

## 1. System Requirements

### 1.1 Hardware Requirements

- **Processor**: Quad-core CPU or better
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: 10GB for application, additional space for processed videos
- **GPU**: CUDA-compatible NVIDIA GPU with 4GB+ VRAM (strongly recommended)
  - Alternative: AMD GPU with ROCm support
  - Fallback: CPU-only mode (with reduced performance)

### 1.2 Software Requirements

- **Operating System**: 
  - Windows 10/11 (64-bit)
  - macOS 11.0+ (Big Sur or newer)
  - Linux (Ubuntu 20.04+, CentOS 8+)
- **Python**: 3.8+ (3.10 recommended)
- **Additional Software**:
  - FFmpeg (4.0+)
  - CUDA Toolkit 11.7+ (for NVIDIA GPUs)
  - ROCm 5.0+ (for AMD GPUs)

## 2. Component Specifications

### 2.1 Video Ingestion & Pre-Processing

#### 2.1.1 Supported Video Formats

- **Container Formats**: MP4, MKV, AVI, MOV, WebM
- **Video Codecs**: H.264, H.265/HEVC, VP9, AV1
- **Maximum Resolution**: 4K (3840×2160)
- **Frame Rates**: Variable, up to 60fps
- **Color Depth**: 8-bit and 10-bit

#### 2.1.2 Performance Requirements

- **Loading Time**: Maximum 30 seconds for a 1-hour HD video
- **Memory Usage**: Maximum 2GB RAM for a 1-hour HD video during loading
- **Throughput**: Process at least 20MB/s of video data

#### 2.1.3 Interfaces

```python
class VideoIngestor:
    def load_video(file_path: str) -> VideoContainer:
        """Load a video file and prepare it for analysis."""
        
    def extract_frames(video: VideoContainer, frame_rate: Optional[float] = None) -> List[Frame]:
        """Extract frames at specified rate or key frames only."""
        
    def extract_audio(video: VideoContainer) -> AudioStream:
        """Extract audio stream from video container."""
```

### 2.2 Scene Change Detection

#### 2.2.1 Detection Methods

- **Deep Learning (TransNet V2, PyTorch-only):** Uses the official PyTorch implementation of TransNet V2 for robust, hardware-agnostic scene boundary detection. This is the only supported method in the Windows-native branch.

#### 2.2.2 Performance Metrics

- **Precision**: >85% correct scene boundaries
- **Recall**: >80% of actual scene changes detected
- **Processing Speed**: Minimum 2x real-time (e.g., 30-min video processed in 15 min)

#### 2.2.3 Interfaces

```python
class SceneDetector:
    def detect_scenes(video: VideoContainer,
                      batch_size: int = 100,
                      threshold: float = 0.5,
                      min_scene_len: int = 15,
                      output_format: Literal['json', 'csv', 'both'] = 'json') -> List[Scene]:
        """Detect scene changes and return list of scenes with both frame and time boundaries."""

    def extract_keyframes(scenes: List[Scene],
                         strategy: Literal['first', 'middle', 'representative'] = 'representative') -> Dict[Scene, List[Frame]]:
        """Extract key frames from each scene."""
```

**Output Specification:**
- **Default:** JSON file with a list of scenes, each containing:
    - `start_frame` (int)
    - `end_frame` (int)
    - `start_time` (float, seconds)
    - `end_time` (float, seconds)
- **Optional:** CSV file with the same fields, if requested.

**Note:** No other scene detection backends (e.g., PySceneDetect, threshold-based, hybrid) are supported in this branch. All logic is PyTorch-only, hardware-agnostic, and Windows-native.

### 2.3 Audio Analysis Pipeline

#### 2.3.1 Speech-to-Text Requirements

- **Supported Languages**: English (primary), Spanish, French, German, Japanese
- **Word Error Rate**: <10% for clear speech
- **Timestamp Accuracy**: Within 500ms
- **Speaker Diarization**: Optional, with >80% speaker identification accuracy

#### 2.3.2 Sentiment Analysis Requirements

- **Classification**: Positive, Negative, Neutral
- **Confidence Score**: 0-1 scale
- **Granularity**: Sentence-level or paragraph-level
- **Accuracy**: >80% agreement with human judgment

#### 2.3.3 Interfaces

```python
class AudioPipeline:
    def transcribe(audio: AudioStream, 
                  language: str = 'en',
                  word_timestamps: bool = False,
                  diarize: bool = False) -> List[TranscriptSegment]:
        """Transcribe audio to text with timestamps."""
        
    def analyze_sentiment(transcript: List[TranscriptSegment]) -> List[SentimentScore]:
        """Analyze sentiment of transcribed text."""
        
    def extract_audio_features(audio: AudioStream) -> Dict[str, List[float]]:
        """Extract audio features for emotion/event detection."""
        
    def detect_audio_events(audio: AudioStream, 
                           event_types: List[str] = ['laughter', 'applause', 'music']) -> List[AudioEvent]:
        """Detect specified audio events with timestamps."""
```

### 2.4 Visual Analysis Pipeline

#### 2.4.1 Scene Context Analysis Requirements

- **Tag Generation**: 5-10 relevant tags per scene
- **Caption Quality**: Grammatically correct, descriptive sentences
- **Processing Time**: <5 seconds per frame on GPU, <20 seconds on CPU

#### 2.4.2 Facial Analysis Requirements

- **Face Detection**: >90% accuracy for frontal faces
- **Emotion Recognition**: 7 basic emotions (happy, sad, angry, fear, disgust, surprise, neutral)
- **Confidence Scores**: 0-1 scale per emotion
- **Processing Time**: <1 second per face on GPU, <3 seconds on CPU

#### 2.4.3 Interfaces

```python
class VisualPipeline:
    def generate_tags(frame: Frame) -> List[str]:
        """Generate tags describing the frame content."""
        
    def generate_caption(frame: Frame) -> str:
        """Generate a natural language caption for the frame."""
        
    def detect_faces(frame: Frame) -> List[Face]:
        """Detect faces in the frame."""
        
    def analyze_facial_emotions(face: Face) -> Dict[str, float]:
        """Analyze emotions expressed in the face."""
        
    def create_embedding(frame: Frame) -> List[float]:
        """Create vector embedding for the frame."""
    
    def create_video_embedding(video_path: str) -> List[float]:
        """Create a robust video embedding using Hugging Face TimeSformer with manual preprocessing (see DEVELOPMENT_SETUP.md)."""
```

### 2.5 Data Fusion & Scoring

#### 2.5.1 "Viral" Scoring Algorithm

- **Input Signals**:
  - Scene transition speed (cuts per minute)
  - Audio volume variance
  - Sentiment extremes and shifts
  - Facial emotion intensity
  - Audio event density (laughter, applause)
- **Score Range**: 0-100
- **Threshold**: Scenes with scores >70 highlighted as potential viral moments

#### 2.5.2 Interfaces

```python
class DataFusion:
    def align_metadata(scenes: List[Scene], 
                      transcripts: List[TranscriptSegment],
                      visual_metadata: Dict[Scene, VisualMetadata],
                      audio_metadata: Dict[TimeRange, AudioMetadata]) -> List[EnrichedScene]:
        """Align all metadata by timestamp and scene boundaries."""
        
    def calculate_viral_score(scene: EnrichedScene) -> float:
        """Calculate viral potential score for the scene."""
```

### 2.6 Data Storage & Retrieval

#### 2.6.1 Database Schema

- **Tables**:
  - Videos (id, path, duration, resolution, created_at)
  - Scenes (id, video_id, start_time, end_time, viral_score)
  - KeyFrames (id, scene_id, timestamp, path)
  - Transcripts (id, scene_id, start_time, end_time, text, speaker, sentiment)
  - Tags (id, scene_id, tag, confidence)
  - Captions (id, frame_id, text)
  - Embeddings (id, frame_id, type, vector)
  - FacialEmotions (id, frame_id, bbox, emotions)
  - AudioEvents (id, scene_id, type, start_time, end_time, confidence)

#### 2.6.2 Performance Requirements

- **Query Time**: <100ms for timeline metadata retrieval
- **Storage Efficiency**: <100MB database size per hour of HD video
- **Vector Search**: <500ms for similarity search across 1000 scenes

#### 2.6.3 Interfaces

```python
class DataStorage:
    def save_video_metadata(video: VideoContainer) -> int:
        """Save video metadata and return video_id."""
        
    def save_scenes(video_id: int, scenes: List[EnrichedScene]) -> None:
        """Save all scene data for a video."""
        
    def get_timeline(video_id: int) -> Timeline:
        """Retrieve timeline data for the video."""
        
    def search_scenes(query: str, embedding: Optional[List[float]] = None) -> List[Scene]:
        """Search scenes by text or vector similarity."""
```

### 2.7 Interactive Timeline UI

#### 2.7.1 UI Requirements

- **Video Player**: Controls for play, pause, seek, volume
- **Timeline**: Visual representation with scene markers
- **Subtitle Display**: Synchronized text overlay
- **Metadata Panel**: Display scene information, tags, sentiment
- **Highlight Indicators**: Visual cues for high-scoring scenes
- **Responsive Layout**: Adapt to different screen sizes

#### 2.7.2 Performance Requirements

- **Startup Time**: <3 seconds to load UI
- **Timeline Responsiveness**: <100ms response to user interactions
- **Smooth Playback**: No frame drops during normal playback
- **Memory Usage**: <200MB for UI components

#### 2.7.3 Interfaces

```python
class TimelineUI:
    def initialize(video_id: int) -> None:
        """Initialize the UI with the specified video."""
        
    def display_timeline(timeline: Timeline) -> None:
        """Display the timeline with all markers and metadata."""
        
    def play_scene(scene_id: int) -> None:
        """Jump to and play the specified scene."""
        
    def display_metadata(scene_id: int) -> None:
        """Display metadata for the specified scene."""
```

## 3. External Dependencies

### 3.1 Core Dependencies

- **PySceneDetect**: Scene detection
- **OpenAI Whisper**: Speech-to-text
- **CLIP**: Visual embedding and tagging
- **BLIP-2**: Image captioning
- **Hugging Face Transformers**: NLP tasks
- **DeepFace**: Facial analysis
- **FFmpeg**: Video/audio processing
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **SQLite**: Database storage
- **FAISS/ChromaDB**: Vector search

### 3.2 UI Dependencies

- **Desktop UI**: PyQt5/PySide6 or Tkinter
- **Web UI**: FastAPI, HTML5, JavaScript, Chart.js

## 4. Performance Benchmarks

### 4.1 Processing Time Targets

| Component | 1-minute HD Video | 1-hour HD Video |
|-----------|-------------------|-----------------|
| Video Loading | <5 seconds | <30 seconds |
| Scene Detection | <10 seconds | <10 minutes |
| Transcription | <30 seconds | <15 minutes |
| Visual Analysis | <1 minute | <30 minutes |
| Data Fusion | <5 seconds | <1 minute |
| Total Processing | <2 minutes | <1 hour |

### 4.2 Resource Usage Targets

| Resource | Target |
|----------|--------|
| CPU Usage | <80% average across all cores |
| GPU Memory | <80% of available VRAM |
| RAM Usage | <4GB for 1-hour HD video |
| Disk I/O | <50MB/s sustained write |

## 5. Quality Metrics

### 5.1 Scene Detection Quality

- **Precision**: >85% (proportion of detected scenes that are actual scenes)
- **Recall**: >80% (proportion of actual scenes that are detected)
- **F1 Score**: >82% (harmonic mean of precision and recall)

### 5.2 Transcription Quality

- **Word Error Rate**: <10% for clear speech, <20% for challenging audio
- **Timestamp Accuracy**: >90% within 500ms of actual word timing

### 5.3 Sentiment Analysis Quality

- **Accuracy**: >80% agreement with human judgment
- **F1 Score**: >75% for multi-class sentiment classification

### 5.4 Viral Score Correlation

- **Human Agreement**: >70% correlation with human ratings of "highlight quality"
- **Engagement Correlation**: >60% correlation with actual engagement metrics (when available)

## 6. Compliance & Ethics

### 6.1 Privacy Considerations

- **Facial Recognition**: Processing done locally, no persistent storage of identifiable features
- **Content Analysis**: No transmission of video content to external services without explicit consent
- **Data Storage**: Option to anonymize or delete processed data

### 6.2 Bias Mitigation

- **Models Selection**: Preference for models evaluated for demographic biases
- **Sentiment Analysis**: Calibration across different languages and cultural contexts
- **Facial Analysis**: Testing across diverse datasets to ensure equitable performance