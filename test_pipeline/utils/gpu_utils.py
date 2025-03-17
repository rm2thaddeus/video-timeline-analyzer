"""
📌 Purpose: GPU utilities for the video processing pipeline
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Functions to detect and utilize GPU resources
📂 Expected File Path: test_pipeline/utils/gpu_utils.py
🧠 Reasoning: Centralize GPU handling for consistent resource utilization
"""

import os
import torch
import logging
import subprocess
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("video_pipeline.gpu_utils")

def detect_gpu() -> Dict[str, Any]:
    """
    Detect available GPU resources and return relevant information.
    
    Returns:
        Dict with GPU information including:
        - detected: Whether a GPU was detected
        - gpu_type: Type of GPU (cuda, mps, or cpu)
        - device_count: Number of available devices
        - optimal_device: Recommended device for processing
    """
    gpu_info = {
        "detected": False,
        "gpu_type": "cpu",
        "device_count": 0,
        "optimal_device": "cpu"
    }
    
    # Check for CUDA
    if torch.cuda.is_available():
        gpu_info["detected"] = True
        gpu_info["gpu_type"] = "cuda"
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["optimal_device"] = "cuda"
        logger.info(f"CUDA GPU detected. Device count: {gpu_info['device_count']}")
        
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_info["detected"] = True
        gpu_info["gpu_type"] = "mps"
        gpu_info["device_count"] = 1
        gpu_info["optimal_device"] = "mps"
        logger.info("Apple Silicon MPS device detected")
    
    else:
        logger.warning("No GPU detected, falling back to CPU")
    
    return gpu_info

def get_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information (if available).
    
    Returns:
        Dict with memory information in GB:
        - total_gb: Total GPU memory
        - allocated_gb: Currently allocated memory
        - free_gb: Available memory
    """
    memory_info = {
        "total_gb": 0.0,
        "allocated_gb": 0.0,
        "free_gb": 0.0
    }
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory
        
        # Convert to GB
        memory_info["total_gb"] = total_memory / (1024**3)
        memory_info["allocated_gb"] = allocated_memory / (1024**3)
        memory_info["free_gb"] = free_memory / (1024**3)
        
        logger.info(f"GPU Memory - Total: {memory_info['total_gb']:.2f}GB, "
                   f"Allocated: {memory_info['allocated_gb']:.2f}GB, "
                   f"Free: {memory_info['free_gb']:.2f}GB")
    else:
        logger.warning("No CUDA device available for memory reporting")
    
    return memory_info

def get_pcie_bandwidth() -> Dict[str, Any]:
    """
    Get PCIe bandwidth information using nvidia-smi if available.
    
    Returns:
        Dict with PCIe information or error message
    """
    bandwidth_info = {"error": ""}
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            gen, width = result.stdout.strip().split(", ")
            bandwidth_info = {
                "generation": gen,
                "width": width,
                "theoretical_bandwidth_gbps": float(gen) * float(width) * 0.985
            }
            logger.info(f"PCIe Bandwidth: Gen {gen}, Width {width}, "
                       f"Theoretical: {bandwidth_info['theoretical_bandwidth_gbps']:.2f} GB/s")
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        bandwidth_info["error"] = str(e)
        logger.error(f"Error getting PCIe bandwidth information: {e}")
    
    return bandwidth_info

def get_optimal_batch_size(model_name: str) -> int:
    """
    Calculate optimal batch size based on available GPU memory and model.
    
    Args:
        model_name: Name of the model to use (e.g., 'clip', 'whisper')
        
    Returns:
        Recommended batch size
    """
    memory_info = get_memory_info()
    
    # Default conservative batch sizes based on model and available memory
    if not torch.cuda.is_available():
        return 1
    
    free_memory_gb = memory_info["free_gb"]
    
    # Model-specific batch size calculation
    if model_name.lower() == 'clip':
        # CLIP typically uses ~1.5-2GB for ViT-B/32 with batch size of 16
        return max(1, min(32, int(free_memory_gb / 0.125)))
    elif model_name.lower() == 'whisper':
        # Whisper base model uses ~1GB for 30-second audio segments
        return max(1, min(16, int(free_memory_gb / 0.25)))
    else:
        # Generic conservative estimate
        return max(1, min(8, int(free_memory_gb / 0.5)))

def setup_device() -> Tuple[torch.device, Optional[Dict]]:
    """
    Set up the optimal device for processing.
    
    Returns:
        Tuple of (device, device_properties)
    """
    gpu_info = detect_gpu()
    
    if gpu_info["detected"]:
        device_str = gpu_info["optimal_device"]
        device = torch.device(device_str)
        
        if device_str == "cuda":
            properties = {
                "name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "memory": get_memory_info()
            }
            logger.info(f"Using GPU: {properties['name']}")
            return device, properties
        elif device_str == "mps":
            properties = {"name": "Apple Silicon MPS"}
            logger.info("Using Apple Silicon GPU via MPS")
            return device, properties
    
    logger.warning("Using CPU for computation")
    return torch.device("cpu"), None 