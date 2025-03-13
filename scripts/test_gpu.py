#!/usr/bin/env python
"""
GPU Test Script for Video Timeline Analyzer.

This script tests the GPU detection and benchmarks performance to ensure
proper configuration for the application.

📌 Purpose: Test and verify GPU detection and performance
🔄 Latest Changes: Initial implementation
⚙️ Key Logic: Detects GPU, runs matrix multiplication benchmark
📂 Expected File Path: scripts/test_gpu.py
🧠 Reasoning: Critical to confirm GPU is properly configured before
              running computationally intensive video analysis
"""

import os
import sys
import time
import argparse

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.utils.gpu_utils import detect_gpu, get_optimal_device, memory_stats, clear_gpu_memory

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
    print(f"Running matrix multiplication benchmark ({size}x{size}) on {device}...")
    
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    torch.matmul(a, b)
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    
    # Synchronize if using GPU
    if device.type in ['cuda', 'mps']:
        torch.cuda.synchronize() if device.type == 'cuda' else None
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time

def benchmark_convolution(device, iterations=3):
    """
    Benchmark convolution performance (common in video/image processing).
    
    Args:
        device: torch.device to use for computation
        iterations: Number of iterations to run
        
    Returns:
        Average execution time in seconds
    """
    print(f"Running convolution benchmark on {device}...")
    
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
    
    # Synchronize if using GPU
    if device.type in ['cuda', 'mps']:
        torch.cuda.synchronize() if device.type == 'cuda' else None
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time

def print_memory_info(device):
    """Print memory information for the given device."""
    if device.type == 'cpu':
        print("CPU memory information not available")
        return
    
    if device.type == 'cuda':
        stats = memory_stats()
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
        print(f"  Free:      {stats['free_gb']:.2f} GB")
    
    if device.type == 'mps':
        print("MPS Memory: Shared with system memory")

def main():
    """Main function to run the GPU test script."""
    parser = argparse.ArgumentParser(description='Test GPU configuration and performance')
    parser.add_argument('--size', type=int, default=5000, help='Size of matrices for multiplication test')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations for benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all tests including on CPU for comparison')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Video Timeline Analyzer - GPU Test")
    print("=" * 50)
    
    # Detect GPU
    print("\n[1/4] Testing GPU Detection...")
    has_gpu, gpu_type, device_count = detect_gpu()
    
    if not has_gpu:
        print("No GPU detected. Running on CPU only.")
        optimal_device = torch.device('cpu')
    else:
        print(f"GPU detected: {gpu_type} with {device_count} device(s)")
        optimal_device = get_optimal_device()
    
    print(f"Using device: {optimal_device}")
    
    # Memory information
    print("\n[2/4] Memory Information...")
    print_memory_info(optimal_device)
    
    # Matrix multiplication benchmark
    print("\n[3/4] Matrix Multiplication Benchmark...")
    matrix_mul_time = benchmark_matrix_multiplication(
        optimal_device,
        size=args.size,
        iterations=args.iterations
    )
    print(f"Average time: {matrix_mul_time:.4f} seconds")
    
    # Convolution benchmark
    print("\n[4/4] Convolution Benchmark...")
    conv_time = benchmark_convolution(
        optimal_device,
        iterations=args.iterations
    )
    print(f"Average time: {conv_time:.4f} seconds")
    
    # If requested, also run on CPU for comparison
    if args.all and optimal_device.type != 'cpu':
        print("\n[EXTRA] Running comparison benchmarks on CPU...")
        cpu_device = torch.device('cpu')
        
        # CPU matrix multiplication
        cpu_matrix_time = benchmark_matrix_multiplication(
            cpu_device,
            size=args.size,
            iterations=args.iterations
        )
        print(f"CPU Matrix Multiplication: {cpu_matrix_time:.4f} seconds")
        print(f"Speedup: {cpu_matrix_time / matrix_mul_time:.2f}x")
        
        # CPU convolution
        cpu_conv_time = benchmark_convolution(
            cpu_device,
            iterations=args.iterations
        )
        print(f"CPU Convolution: {cpu_conv_time:.4f} seconds")
        print(f"Speedup: {cpu_conv_time / conv_time:.2f}x")
    
    # Clean up
    clear_gpu_memory()
    
    print("\n" + "=" * 50)
    print("GPU test completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()