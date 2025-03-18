# Previous Pipeline Results

This directory contains the results from previous runs of the video processing pipeline. These results are stored here for reference and comparison with new pipeline runs.

## Directory Structure

```
test_pipeline/previous_results/
├── metadata/         # Video metadata from previous runs
├── scenes/           # Scene detection results from previous runs
├── audio_chunks/     # Extracted audio segments from previous runs
├── transcripts/      # Transcribed audio from previous runs
├── frames/           # Extracted video frames from previous runs
├── embeddings/       # Vector embeddings from previous runs
├── logs/             # Pipeline logs from previous runs
└── videotest_pipeline_result.json  # Complete pipeline results
```

## Usage

These results can be used for:

1. Comparing performance between different pipeline versions
2. Debugging issues in the pipeline
3. Analyzing the effectiveness of different processing parameters
4. Benchmarking CUDA vs. non-CUDA performance

## CUDA Pipeline

The current active pipeline is located in the `test_pipeline/CUDA/` directory, which contains a CUDA-optimized version of the video processing pipeline. 