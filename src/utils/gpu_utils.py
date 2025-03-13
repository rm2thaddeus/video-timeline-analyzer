"""
GPU Utilities for Video Timeline Analyzer.

This module provides utilities for detecting and configuring GPU hardware,
optimizing performance, and falling back to CPU when necessary.

📌 Purpose: Detect available GPU hardware and configure optimal settings
🔄 Latest Changes: Initial implementation
⚙️ Key Logic: Detects NVIDIA CUDA, AMD ROCm, or Apple Silicon MPS and 
              configures PyTorch accordingly
📂 Expected File Path: src/utils/gpu_utils.py
🧠 Reasoning: GPU acceleration is critical for deep learning models used in 
              video analysis; this module provides a unified interface for 
              different GPU types
"""

import os
import platform
import logging
from typing import Tuple, Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)

def detect_gpu() -> Tuple[bool, Optional[str], int]:
    """
    Detect available GPU and return configuration information.
    
    Returns:
        Tuple containing:
        - Boolean indicating if GPU is available
        - String with GPU type ('cuda', 'mps', 'rocm', None)
        - Integer with device count (0 if no GPU)
    """
    has_gpu = False
    gpu_type = None
    device_count = 0
    
    # Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        has_gpu = True
        gpu_type = 'cuda'
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            logger.info(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
            logger.info(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Check for Apple Silicon (MPS)
    elif platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        has_gpu = True
        gpu_type = 'mps'
        device_count = 1
        logger.info("Apple Silicon (MPS) GPU available")
    
    # Check for AMD ROCm
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        has_gpu = True
        gpu_type = 'rocm'
        device_count = torch.hip.device_count()
        logger.info(f"ROCm GPU available with {device_count} devices")
    
    # No GPU available
    else:
        logger.warning("No GPU detected. Using CPU only (performance will be significantly reduced)")
    
    # Log environment variables for debugging
    logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.debug(f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'Not set')}")
    
    return has_gpu, gpu_type, device_count

def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed information about available GPU(s).
    
    Returns:
        Dictionary containing detailed GPU information
    """
    info = {
        'has_gpu': False,
        'gpu_type': None,
        'device_count': 0,
        'devices': [],
    }
    
    has_gpu, gpu_type, device_count = detect_gpu()
    info['has_gpu'] = has_gpu
    info['gpu_type'] = gpu_type
    info['device_count'] = device_count
    
    # Get detailed device information
    if has_gpu:
        if gpu_type == 'cuda':
            for i in range(device_count):
                device_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'compute_capability': torch.cuda.get_device_capability(i),
                    'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
                }
                info['devices'].append(device_info)
        elif gpu_type == 'mps':
            info['devices'].append({
                'index': 0,
                'name': 'Apple Silicon GPU',
                'compute_capability': 'MPS',
                'total_memory_gb': 'Shared with system',
            })
        elif gpu_type == 'rocm':
            for i in range(device_count):
                info['devices'].append({
                    'index': i,
                    'name': f'AMD GPU {i}',
                    'compute_capability': 'ROCm',
                    'total_memory_gb': 'Unknown',
                })
    
    return info

def get_optimal_device() -> torch.device:
    """
    Get the optimal device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: The optimal device for computation
    """
    has_gpu, gpu_type, _ = detect_gpu()
    
    if not has_gpu:
        return torch.device('cpu')
    
    if gpu_type == 'cuda':
        return torch.device('cuda')
    elif gpu_type == 'mps':
        return torch.device('mps')
    elif gpu_type == 'rocm':
        return torch.device('cuda')  # ROCm uses the 'cuda' device type in PyTorch
    
    return torch.device('cpu')

def optimize_gpu_settings() -> None:
    """
    Configure optimal settings for the detected GPU.
    
    This function applies performance optimizations based on the detected GPU type.
    """
    has_gpu, gpu_type, _ = detect_gpu()
    
    if not has_gpu:
        logger.info("No GPU detected, skipping GPU optimization")
        return
    
    # NVIDIA CUDA settings
    if gpu_type == 'cuda':
        # Set TF32 for faster computation on Ampere+ GPUs
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("Enabled TF32 for CUDA matrix multiplication")
        
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for cuDNN")
        
        # Enable cuDNN benchmarking for optimal performance
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode")
        
        # Set memory allocation strategy
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # Leave some GPU memory for other processes
            torch.cuda.set_per_process_memory_fraction(0.9)
            logger.info("Set CUDA memory fraction to 0.9")
    
    # Apple Silicon MPS settings  
    elif gpu_type == 'mps':
        logger.info("Using Apple Silicon MPS backend")
        # MPS-specific optimizations would go here
    
    # AMD ROCm settings
    elif gpu_type == 'rocm':
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode for ROCm")

def memory_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dictionary with memory statistics (allocated, reserved, free)
    """
    has_gpu, gpu_type, _ = detect_gpu()
    
    if not has_gpu or gpu_type != 'cuda':
        return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0}
    
    stats = {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'free_gb': 0  # Will be calculated if possible
    }
    
    # Try to get total memory to calculate free memory
    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        stats['free_gb'] = total_memory - stats['reserved_gb']
    except:
        pass
    
    return stats

def clear_gpu_memory() -> None:
    """Clear unused GPU memory cache to free up resources."""
    has_gpu, gpu_type, _ = detect_gpu()
    
    if not has_gpu:
        return
    
    if gpu_type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA memory cache")
    elif gpu_type == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        logger.info("Cleared MPS memory cache")

# Run GPU detection at module import time
GPU_AVAILABLE, GPU_TYPE, GPU_COUNT = detect_gpu()

# Log GPU information at module import
if GPU_AVAILABLE:
    logger.info(f"GPU detected: {GPU_TYPE} with {GPU_COUNT} device(s)")
else:
    logger.warning("No GPU detected. Running on CPU only (performance will be significantly reduced)")

# Apply optimal settings at module import time
optimize_gpu_settings()