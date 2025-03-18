"""
📌 Purpose: Script to run the video processing pipeline with CUDA
🔄 Latest Changes: Added performance benchmarking and optimized CUDA settings
⚙️ Key Logic: Initialize directories and run the pipeline on a sample video with CUDA
📂 Expected File Path: test_pipeline/CUDA/run_cuda_pipeline.py
🧠 Reasoning: Entry point for testing the pipeline with CUDA acceleration
"""

import os
import sys
import logging
import time
from pathlib import Path
import torch
import psutil
import argparse

# Add the parent directory to the path to allow importing the pipeline module
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

# Set CUDA environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pipeline import ensure_directories, process_video
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "cuda_pipeline_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("video_pipeline.cuda_run")

def print_system_info():
    """Print system information including CPU, RAM, and GPU details."""
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    
    # RAM info
    ram = psutil.virtual_memory()
    ram_total = ram.total / (1024**3)  # GB
    ram_available = ram.available / (1024**3)  # GB
    
    # GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        # CUDA info
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else "Unknown"
    else:
        gpu_count = 0
        gpu_name = "N/A"
        gpu_mem = 0
        cuda_version = "N/A"
        cudnn_version = "N/A"
    
    # Print system info
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"CPU: {cpu_count} cores ({cpu_count_logical} logical)")
    if cpu_freq:
        print(f"CPU Frequency: {cpu_freq.current:.2f} MHz")
    print(f"RAM: {ram_total:.2f} GB (Available: {ram_available:.2f} GB)")
    print(f"GPU Count: {gpu_count}")
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_mem:.2f} GB")
    print(f"CUDA Version: {cuda_version}")
    print(f"cuDNN Version: {cudnn_version}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*50)
    print("CUDA OPTIMIZATION SETTINGS")
    print("="*50)
    print(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    print(f"cuDNN Benchmark: {config.USE_CUDNN_BENCHMARK}")
    print(f"Pinned Memory: {config.USE_PINNED_MEMORY}")
    print(f"Batch Size: {config.GPU_BATCH_SIZE}")
    print(f"Number of Workers: {config.NUM_WORKERS}")
    print(f"Parallel Processing: {config.PARALLEL_PROCESSING}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run the CUDA-optimized video processing pipeline")
    parser.add_argument("--video", type=str, help="Path to the video file (optional)")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    
    # Ensure all directories exist
    ensure_directories()
    
    # Path to the sample video
    video_path = args.video if args.video else r"C:\Users\aitor\Downloads\videotest.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Sample video not found at {video_path}")
        print(f"Error: Sample video not found at {video_path}")
        return
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU instead.")
        print("Warning: CUDA is not available. Using CPU instead.")
    
    logger.info(f"Starting CUDA pipeline with video: {video_path}")
    print(f"Starting CUDA pipeline with video: {video_path}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Process the video
        result = process_video(video_path)
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\nCUDA Pipeline completed successfully!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        # Calculate processing speed
        video_duration = result["metadata"]["duration_seconds"]
        processing_ratio = video_duration / processing_time
        print(f"Video duration: {video_duration:.2f} seconds")
        print(f"Processing speed: {processing_ratio:.2f}x real-time")
        
        print(f"\nMetadata saved to: {config.METADATA_DIR}")
        print(f"Scene boundaries saved to: {config.SCENES_DIR}")
        print(f"Audio chunks saved to: {config.AUDIO_CHUNKS_DIR}")
        print(f"Transcripts saved to: {config.TRANSCRIPTS_DIR}")
        print(f"Frames saved to: {config.FRAMES_DIR}")
        
        # Run benchmark if requested
        if args.benchmark:
            print("\n" + "="*50)
            print("PERFORMANCE BENCHMARK")
            print("="*50)
            print(f"Video Resolution: {result['metadata']['width']}x{result['metadata']['height']}")
            print(f"Video Duration: {video_duration:.2f} seconds")
            print(f"Total Processing Time: {processing_time:.2f} seconds")
            print(f"Processing Speed: {processing_ratio:.2f}x real-time")
            print(f"Scenes Detected: {len(result['scenes'])}")
            print(f"Frames Extracted: {len(result['frames'])}")
            print(f"Frames Per Second: {len(result['frames']) / processing_time:.2f}")
            print("="*50)
            
            # Compare with previous results if available
            previous_results_path = Path(__file__).parent.parent / "previous_results" / "videotest_pipeline_result.json"
            if previous_results_path.exists():
                import json
                with open(previous_results_path, 'r') as f:
                    prev_result = json.load(f)
                
                # Calculate improvement
                if "processing_time" in prev_result:
                    prev_time = prev_result["processing_time"]
                    improvement = (prev_time - processing_time) / prev_time * 100
                    print(f"\nPerformance Improvement: {improvement:.2f}% faster than previous run")
            
            # Save benchmark results
            result["processing_time"] = processing_time
            result["processing_ratio"] = processing_ratio
            result["frames_per_second"] = len(result["frames"]) / processing_time
            
            # Update the result file
            result_path = config.METADATA_DIR / f"{result['video_id']}_cuda_pipeline_result.json"
            with open(result_path, 'w') as f:
                import json
                json.dump(result, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error running CUDA pipeline: {e}", exc_info=True)
        print(f"Error running CUDA pipeline: {e}")

if __name__ == "__main__":
    main() 