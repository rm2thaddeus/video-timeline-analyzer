#!/usr/bin/env python
"""
GPU Test Script for Video Timeline Analyzer.

This script tests the GPU configuration and performance metrics essential
for the metadata processing pipeline. It performs the following tests:
  - GPU compute performance via matrix multiplication and convolution benchmarks
  - Video decoding speed using NVDEC acceleration via ffmpeg (if available)
  - PCIe bandwidth test using nvidia-smi to query GPU properties (if available)

Outputs all test results in JSON format.

📌 Purpose: Extended GPU test for metadata pipeline diagnostics
🔄 Latest Changes: Rewritten to fix broken import of gpu_utils; fallback implementations provided inline. Confirmed working for capturing GPU and system metrics.
📂 Expected File Path: scripts/test_gpu.py
🧠 Reasoning: To ensure that GPU and system configurations are optimal for metadata processing.
"""

import os
import sys
import time
import argparse
import subprocess
import json
import logging
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    import numpy as np
except ImportError as e:
    logger.error(f"Error importing PyTorch or numpy: {e}")
    sys.exit(1)

# Attempt to import GPU utilities; if unavailable, define fallback functions.
try:
    from src.utils.gpu_utils import detect_gpu, get_optimal_device, memory_stats, clear_gpu_memory
except ImportError as e:
    logger.warning(f"Could not import gpu_utils: {e}. Using fallback implementations.")
    def detect_gpu():
        is_avail = torch.cuda.is_available()
        dev_type = "cuda" if is_avail else "cpu"
        count = torch.cuda.device_count() if is_avail else 0
        return is_avail, dev_type, count

    def get_optimal_device():
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def memory_stats():
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(0)
            free = total - allocated
            return {"total_gb": total / (1024**3), "allocated_gb": allocated / (1024**3), "free_gb": free / (1024**3)}
        return {}

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

TORCH_AVAILABLE = True

def check_ffmpeg_availability(ffmpeg_path):
    """
    Check if ffmpeg is available and has NVDEC support.
    
    Args:
        ffmpeg_path: Path to the ffmpeg executable
        
    Returns:
        Tuple containing:
        - Boolean indicating if ffmpeg is available
        - Boolean indicating if NVDEC is supported
        - Error message if any
    """
    try:
        # Check if ffmpeg is available
        process = subprocess.Popen([ffmpeg_path, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return False, False, f"FFmpeg not available: {stderr.decode('utf-8')}"
        
        # Check for NVDEC support
        process = subprocess.Popen([ffmpeg_path, '-codecs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return True, False, f"Error checking FFmpeg codecs: {stderr.decode('utf-8')}"
        
        output = stdout.decode('utf-8')
        has_nvdec = 'h264_nvdec' in output or 'hevc_nvdec' in output
        
        return True, has_nvdec, None
    except Exception as e:
        return False, False, str(e)

def check_nvidia_smi_availability():
    """
    Check if nvidia-smi is available.
    
    Returns:
        Tuple containing:
        - Boolean indicating if nvidia-smi is available
        - Error message if any
    """
    try:
        process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return False, f"nvidia-smi not available: {stderr.decode('utf-8')}"
        
        return True, None
    except Exception as e:
        return False, str(e)

def benchmark_matrix_multiplication(device, size=5000, iterations=3):
    """
    Benchmark matrix multiplication performance.
    
    Args:
        device: torch.device to use for computation
        size: Size of the square matrices
        iterations: Number of iterations to run
        
    Returns:
        Average execution time in seconds
    """
    logger.info(f"Running matrix multiplication benchmark ({size}x{size}) on {device}...")
    
    try:
        # Adjust matrix size based on available memory
        if device.type == 'cuda':
            # For GPUs with limited memory, use a smaller matrix size
            free_memory = memory_stats().get('free_gb', 0)
            if free_memory > 0 and free_memory < 2:  # Less than 2GB free
                adjusted_size = min(size, 2000)
                logger.info(f"Adjusting matrix size to {adjusted_size} due to limited GPU memory")
                size = adjusted_size
        
        # Create random matrices
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        torch.matmul(a, b)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        return avg_time
    except Exception as e:
        logger.error(f"Error in matrix multiplication benchmark: {e}")
        return {"error": str(e)}

def benchmark_convolution(device, iterations=3):
    """
    Benchmark convolution performance (common in video/image processing).
    
    Args:
        device: torch.device to use for computation
        iterations: Number of iterations to run
        
    Returns:
        Average execution time in seconds
    """
    logger.info(f"Running convolution benchmark on {device}...")
    
    try:
        # Create input tensor: [batch_size, channels, height, width]
        input_tensor = torch.randn(16, 3, 224, 224, device=device)
        
        # Create convolution layer
        conv_layer = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=True
        ).to(device)
        
        # Warmup
        conv_layer(input_tensor)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            output = conv_layer(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        return avg_time
    except Exception as e:
        logger.error(f"Error in convolution benchmark: {e}")
        return {"error": str(e)}

def benchmark_video_decoding(ffmpeg_path, input_video, has_nvdec):
    """
    Benchmark video decoding speed using hardware acceleration via ffmpeg.
    
    Args:
        ffmpeg_path: Path to the ffmpeg executable
        input_video: Path to sample video for decoding benchmark
        has_nvdec: Boolean indicating if NVDEC is supported
        
    Returns:
        Dictionary with results or error
    """
    results = {}
    
    # Check if input video exists
    if not os.path.exists(input_video):
        return {"error": f"Input video not found: {input_video}"}
    
    # Test with hardware acceleration if available
    if has_nvdec:
        logger.info(f"Running NVDEC video decoding benchmark with {ffmpeg_path} on {input_video}...")
        
        try:
            # Command to decode video using ffmpeg with NVDEC
            command = [ffmpeg_path, '-i', input_video, '-c:v', 'h264_nvdec', '-f', 'null', '-']
            
            # Benchmark
            start_time = time.time()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8')
                logger.error(f"Error decoding video with NVDEC: {error_msg}")
                results["nvdec"] = {"error": error_msg}
            else:
                end_time = time.time()
                results["nvdec"] = end_time - start_time
        except Exception as e:
            logger.error(f"Exception in NVDEC video decoding: {e}")
            results["nvdec"] = {"error": str(e)}
    
    # Always test with software decoding for comparison
    logger.info(f"Running software video decoding benchmark with {ffmpeg_path} on {input_video}...")
    
    try:
        # Command to decode video using ffmpeg with software decoding
        command = [ffmpeg_path, '-i', input_video, '-c:v', 'h264', '-f', 'null', '-']
        
        # Benchmark
        start_time = time.time()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8')
            logger.error(f"Error decoding video with software: {error_msg}")
            results["software"] = {"error": error_msg}
        else:
            end_time = time.time()
            results["software"] = end_time - start_time
    except Exception as e:
        logger.error(f"Exception in software video decoding: {e}")
        results["software"] = {"error": str(e)}
    
    return results

def get_pci_bandwidth():
    """
    Get PCIe bandwidth information via nvidia-smi.
    
    Returns:
        Dictionary with PCIe bandwidth information or error
    """
    logger.info("Getting PCIe bandwidth information via nvidia-smi...")
    
    try:
        # Command to get PCIe bandwidth information
        command = ['nvidia-smi', '-q', '-d', 'PCI']
        
        # Execute command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Parse output
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8')
            logger.error(f"Error getting PCIe bandwidth information: {error_msg}")
            return {"error": error_msg}
        
        lines = stdout.decode('utf-8').splitlines()
        pci_info = {}
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if ':' in stripped_line:
                key, value = stripped_line.split(':', 1)
                pci_info[key.strip()] = value.strip()
        return pci_info
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Exception while parsing PCIe data: {error_msg}")
        return {"error": error_msg}

def main():
    parser = argparse.ArgumentParser(description="GPU and system performance test with JSON output")
    parser.add_argument('--size', type=int, default=2000, help="Matrix size for multiplication benchmark")
    parser.add_argument('--iterations', type=int, default=3, help="Number of iterations for benchmarks")
    parser.add_argument('--ffmpeg-path', type=str, default="ffmpeg", help="Path to the ffmpeg executable")
    parser.add_argument('--input-video', type=str, default="C:\\Users\\aitor\\Downloads\\videotest.mp4", 
                        help="Path to sample video for decoding benchmark")
    parser.add_argument('--skip-video', action='store_true', help="Skip video decoding benchmark")
    parser.add_argument('--skip-pci', action='store_true', help="Skip PCIe bandwidth test")
    args = parser.parse_args()

    results = {
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else "Not available"
        }
    }

    # Check if PyTorch is available
    if not TORCH_AVAILABLE:
        results["error"] = "PyTorch is not available. Please install PyTorch with GPU support."
        print(json.dumps(results, indent=4))
        return

    # GPU Detection
    has_gpu, gpu_type, device_count = detect_gpu()
    if not has_gpu:
        optimal_device = torch.device('cpu')
        results["gpu"] = {"detected": False, "device": "cpu"}
    else:
        optimal_device = get_optimal_device()
        results["gpu"] = {
            "detected": True, 
            "gpu_type": gpu_type, 
            "device_count": device_count, 
            "optimal_device": str(optimal_device)
        }

    # Memory Information
    if optimal_device.type != 'cpu':
        mem_stats = memory_stats()
        results["memory_info"] = mem_stats

    # GPU Compute Benchmarks
    results["benchmarks"] = {}
    
    # Matrix multiplication benchmark
    matrix_result = benchmark_matrix_multiplication(optimal_device, size=args.size, iterations=args.iterations)
    results["benchmarks"]["matrix_multiplication"] = matrix_result
    
    # Convolution benchmark
    conv_result = benchmark_convolution(optimal_device, iterations=args.iterations)
    results["benchmarks"]["convolution"] = conv_result

    # NVDEC Video Decoding Benchmark
    if not args.skip_video:
        # Check if ffmpeg is available and has NVDEC support
        ffmpeg_available, has_nvdec, ffmpeg_error = check_ffmpeg_availability(args.ffmpeg_path)
        
        if ffmpeg_available:
            video_results = benchmark_video_decoding(args.ffmpeg_path, args.input_video, has_nvdec)
            results["benchmarks"]["video_decoding"] = video_results
        else:
            results["benchmarks"]["video_decoding"] = {"error": ffmpeg_error}
    else:
        results["benchmarks"]["video_decoding"] = {"skipped": True}

    # PCIe Bandwidth Information
    if not args.skip_pci and gpu_type == 'cuda':
        # Check if nvidia-smi is available
        nvidia_smi_available, nvidia_smi_error = check_nvidia_smi_availability()
        
        if nvidia_smi_available:
            pci_info = get_pci_bandwidth()
            results["pci_bandwidth"] = pci_info
        else:
            results["pci_bandwidth"] = {"error": nvidia_smi_error}
    else:
        results["pci_bandwidth"] = {"skipped": True}

    # Clean up
    if optimal_device.type != 'cpu':
        clear_gpu_memory()

    # Print results as JSON
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()