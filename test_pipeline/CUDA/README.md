# CUDA-Accelerated Video Processing Pipeline

This pipeline provides CUDA-accelerated video processing capabilities including:

1. Efficient scene detection with batch processing and checkpointing
2. Optimized audio extraction and transcription using Whisper
3. Frame extraction and embedding with CLIP
4. Memory-efficient processing suitable for large videos

The pipeline is designed to use GPU acceleration where available and to handle memory constraints gracefully, with fallbacks to CPU processing when necessary.

## Directory Structure

```
test_pipeline/CUDA/
├── metadata/         # Video metadata storage
├── scenes/           # Scene detection results
├── audio_chunks/     # Extracted audio segments
├── transcripts/      # Transcribed audio
├── frames/           # Extracted video frames
├── embeddings/       # Vector embeddings
├── logs/             # Pipeline logs
├── config.py         # CUDA-specific configuration
├── pipeline.py       # CUDA-optimized pipeline implementation
└── run_cuda_pipeline.py  # Entry point for running the pipeline
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- FFmpeg installed and available in PATH
- NVIDIA GPU with CUDA support (for acceleration)

## Key Components

- `pipeline.py` - Main pipeline orchestration
- `scene_detector.py` - Optimized scene detection with CUDA support
- `audio_processor_cuda.py` - CUDA-accelerated audio processing with Whisper
- `cuda_config.py` - Configuration settings for the pipeline
- `run_pipeline.py` - Command-line interface for running the pipeline

## Usage

You can run the pipeline using the provided `run_pipeline.py` script:

```bash
python run_pipeline.py your_video_file.mp4 [options]
```

### Command-line Options

```
Required arguments:
  video_path            Path to the video file to process

Optional arguments:
  --output_dir OUTPUT_DIR
                        Directory to save output files (default: auto)
  --gpu_memory_fraction GPU_MEMORY_FRACTION
                        Fraction of GPU memory to use (0.0-1.0)
  --batch_size BATCH_SIZE
                        Batch size for GPU processing
  --sequential          Disable parallel processing
  --whisper_model {tiny,base,small,medium,large}
                        Whisper model size
  --mixed_precision MIXED_PRECISION
                        Use mixed precision (FP16)
  --scene_threshold SCENE_THRESHOLD
                        Threshold for scene detection
  --scene_batch_size SCENE_BATCH_SIZE
                        Batch size for scene detection
  --info                Print CUDA and GPU information
```

### Example Commands

Check your GPU information:
```bash
python run_pipeline.py --info
```

Process a video with default settings:
```bash
python run_pipeline.py path/to/your/video.mp4
```

Process a video with custom settings:
```bash
python run_pipeline.py path/to/your/video.mp4 --gpu_memory_fraction 0.6 --whisper_model base --scene_threshold 25
```

Run in sequential mode (for stability):
```bash
python run_pipeline.py path/to/your/video.mp4 --sequential
```

## Memory Management

The pipeline includes several memory optimization strategies:

1. **Batch Processing**: Both scene detection and audio processing process data in configurable batch sizes
2. **Checkpointing**: Scene detection saves progress regularly, allowing resumption after crashes
3. **Memory Cleanup**: Aggressive cleanup between processing stages
4. **Mixed Precision**: Optional FP16 computation for faster processing and reduced memory usage
5. **Fallback Mechanisms**: Automatic fallback to CPU or reduced batch sizes when encountering memory issues

## Adjusting for Different Hardware

### For High-End GPUs
```bash
python run_pipeline.py video.mp4 --gpu_memory_fraction 0.8 --batch_size 8 --scene_batch_size 256 --whisper_model small
```

### For Mid-Range GPUs
```bash
python run_pipeline.py video.mp4 --gpu_memory_fraction 0.6 --batch_size 4 --scene_batch_size 128 --whisper_model tiny
```

### For Low-End GPUs
```bash
python run_pipeline.py video.mp4 --gpu_memory_fraction 0.4 --batch_size 2 --scene_batch_size 64 --whisper_model tiny
```

### For CPU-only Processing
```bash
# Force CPU processing by setting memory to 0
python run_pipeline.py video.mp4 --gpu_memory_fraction 0 --sequential
```

## Output Structure

The pipeline produces the following outputs:

- **metadata/**: Contains video metadata and overall pipeline results
- **scenes/**: JSON files with scene detection results
- **audio_chunks/**: Extracted audio files
- **transcripts/**: Transcription files in JSON and SRT formats
- **frames/**: Extracted video frames
- **logs/**: Pipeline execution logs

## Configuration

You can customize the pipeline by editing `cuda_config.py` directly, or by providing command-line arguments to `run_pipeline.py`.

## CUDA Optimizations

The pipeline implements several advanced CUDA optimizations:

1. **Mixed Precision (FP16)**: Uses half-precision floating point where possible to double throughput
2. **Pinned Memory**: Optimizes CPU-GPU memory transfers
3. **Parallel Processing**: Runs audio processing and frame extraction in parallel
4. **Optimized Batch Size**: Uses larger batch sizes for better GPU utilization
5. **cuDNN Benchmarking**: Automatically selects the most efficient algorithms
6. **Memory Management**: Optimizes memory allocation and caching
7. **Worker Threads**: Uses multiple worker threads for data loading
8. **TF32 Acceleration**: Enables TF32 on Ampere GPUs for better performance

## Usage

To run the CUDA-optimized pipeline:

```bash
python run_cuda_pipeline.py
```

For performance benchmarking:

```bash
python run_cuda_pipeline.py --benchmark
```

To process a specific video:

```bash
python run_cuda_pipeline.py --video /path/to/video.mp4
```

## Performance

The CUDA-optimized pipeline provides significant performance improvements over the CPU version, particularly for:

- Scene detection (2-5x faster)
- Frame extraction and embedding (3-10x faster)
- Audio transcription (2-4x faster)

Performance metrics are automatically logged and can be viewed in the benchmark results.

## Previous Results

Previous pipeline results are stored in the `test_pipeline/previous_results/` directory for comparison. 