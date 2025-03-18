# CUDA-Optimized Audio Processor with Word-Level Accuracy

## Overview

This module provides a GPU-accelerated audio processing pipeline for video files, featuring:

- **Parallel audio chunking**: Segments audio files in parallel for faster processing
- **GPU acceleration**: Utilizes CUDA for maximum performance
- **Word-level accuracy**: Provides precise word-level timestamps for transcriptions
- **Memory optimization**: Implements batch processing and memory management techniques
- **Mixed precision**: Uses FP16 computation for faster processing when available

## Key Components

1. **AudioProcessorCUDA**: Main class that handles audio extraction, segmentation, and transcription
2. **Parallel Processing**: Uses ThreadPoolExecutor for concurrent audio chunk creation
3. **Batched Transcription**: Processes audio chunks in batches for optimal GPU utilization
4. **Word-Level Timestamps**: Provides accurate start/end times for each word in the transcription
5. **Memory Management**: Implements CUDA cache clearing and pinned memory for efficient data transfer

## Usage

### Basic Usage

```python
from audio_processor_cuda import AudioProcessorCUDA
import torch

# Initialize processor
processor = AudioProcessorCUDA(
    model_name="base",  # Whisper model size (tiny, base, small, medium, large)
    device=torch.device("cuda"),
    use_mixed_precision=True,
    batch_size=16
)

# Process a video file
result = processor.process_video_cuda(
    video_path="path/to/video.mp4",
    chunk_duration=20.0,  # Duration of each audio chunk in seconds
    overlap=1.5,          # Overlap between chunks in seconds
    max_workers=8         # Number of parallel workers for segmentation
)

# Access word-level transcription
words = result["merged_transcription"]["words"]
for word in words:
    print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

### Command Line Usage

Run the included test script:

```bash
# Windows Command Prompt
run_audio_test.bat

# PowerShell
.\run_audio_test.ps1

# Direct Python execution
python run_audio_processor.py "path/to/video.mp4" --chunk-duration 20 --overlap 1.5 --max-workers 8
```

## Performance Optimization

The processor includes several optimizations for maximum GPU performance:

1. **Parallel Audio Chunking**: Creates audio segments in parallel using multiple worker threads
2. **Batched Processing**: Groups audio chunks into batches for efficient GPU utilization
3. **Mixed Precision**: Uses FP16 computation when available for faster processing
4. **Pinned Memory**: Utilizes pinned memory for faster CPU-GPU data transfer
5. **Memory Management**: Clears CUDA cache between batches to prevent memory buildup
6. **Non-blocking Transfers**: Uses non-blocking data transfers when possible

## Integration with Pipeline

The audio processor is integrated into the main CUDA pipeline and can be used either:

1. As a standalone component for audio-only processing
2. As part of the full video processing pipeline, running in parallel with frame extraction

## Output Format

The processor generates several outputs:

1. **Audio chunks**: Segmented audio files for processing
2. **Individual transcriptions**: JSON files for each audio chunk with word-level data
3. **Merged transcription**: Combined JSON file with all words and segments aligned to video timeline
4. **Processing results**: Comprehensive JSON with processing statistics and metadata

## Word-Level Data Structure

Each word in the transcription includes:

```json
{
  "word": "example",
  "start": 10.25,
  "end": 10.75,
  "probability": 0.98
}
```

This enables precise alignment of text with audio/video for applications like:
- Subtitle generation with exact timing
- Interactive transcripts
- Audio/video search with precise seeking
- Content analysis with temporal alignment 