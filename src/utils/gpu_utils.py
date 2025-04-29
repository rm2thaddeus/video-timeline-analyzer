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
        - reserved_gb: Reserved GPU memory
        - allocated_gb: Currently allocated memory
        - free_gb: Available memory
    """
    memory_info = {
        "total_gb": 0.0,
        "reserved_gb": 0.0,
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
        memory_info["reserved_gb"] = reserved_memory / (1024**3)
        memory_info["allocated_gb"] = allocated_memory / (1024**3)
        memory_info["free_gb"] = free_memory / (1024**3)
        
        logger.info(f"GPU Memory - Total: {memory_info['total_gb']:.2f}GB, "
                   f"Reserved: {memory_info['reserved_gb']:.2f}GB, "
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

def get_optimal_batch_size(model, input_shape, target_memory_usage=0.7) -> int:
    """
    Calculate optimal batch size based on available GPU memory and model size.
    Args:
        model: torch.nn.Module
        input_shape: tuple, shape of a single input sample (excluding batch)
        target_memory_usage: float, fraction of free memory to use
    Returns:
        int: Recommended batch size
    """
    if not torch.cuda.is_available():
        return 1
    memory_info = get_memory_info()
    free_memory_gb = memory_info["free_gb"] * target_memory_usage
    # Estimate memory per sample (rough, based on model size and input)
    try:
        sample = torch.randn((1,) + tuple(input_shape)).cuda()
        with torch.no_grad():
            model = model.cuda()
            output = model(sample)
        mem_per_sample = torch.cuda.memory_allocated() / (1024**3)
        torch.cuda.empty_cache()
        if mem_per_sample == 0:
            mem_per_sample = 0.05  # fallback estimate
    except Exception:
        mem_per_sample = 0.05  # fallback estimate
    batch_size = max(1, int(free_memory_gb / mem_per_sample))
    return batch_size

def setup_device(force_cpu: bool = False) -> torch.device:
    """
    Set up the optimal device for processing.
    
    Args:
        force_cpu: Whether to force CPU usage even if GPU is available
        
    Returns:
        The optimal torch.device for processing
    """
    if force_cpu:
        logger.warning("Forcing CPU usage as requested")
        return torch.device("cpu")
        
    gpu_info = detect_gpu()
    
    if gpu_info["detected"] and not force_cpu:
        device_str = gpu_info["optimal_device"]
        device = torch.device(device_str)
        
        if device_str == "cuda":
            properties = {
                "name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "memory": get_memory_info()
            }
            logger.info(f"Using GPU: {properties['name']}")
            return device
        elif device_str == "mps":
            properties = {"name": "Apple Silicon MPS"}
            logger.info("Using Apple Silicon GPU via MPS")
            return device
    
    logger.warning("Using CPU for computation")
    return torch.device("cpu")

def clear_gpu_memory():
    """
    Clear GPU memory cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory cache")

def memory_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dict with memory statistics in GB
    """
    return get_memory_info()

def get_optimal_device() -> torch.device:
    """
    Compatibility wrapper to return the optimal torch.device (GPU if available, else CPU).
    Returns:
        torch.device: The best available device for computation.
    """
    return setup_device(force_cpu=False)

def setup_cuda(gpu_memory_fraction: float = 1.0) -> torch.device:
    """
    Compatibility wrapper for legacy code/tests. Sets up CUDA device if available.
    Args:
        gpu_memory_fraction (float): Fraction of GPU memory to use (currently ignored, for API compatibility).
    Returns:
        torch.device: The CUDA device if available, else CPU.
    """
    import torch
    device = setup_device(force_cpu=False)
    if torch.cuda.is_available():
        import torch.backends.cudnn
        torch.backends.cudnn.benchmark = True
    # Optionally, could add logic to restrict GPU memory usage if needed
    return device

# Compatibility alias for legacy code/tests
get_gpu_memory_info = get_memory_info

def optimize_gpu_memory(model, use_half_precision: bool = True):
    """
    Move model to CUDA and optionally convert to half precision (float16).
    Args:
        model: torch.nn.Module to optimize
        use_half_precision: Whether to use float16
    Returns:
        torch.nn.Module: Optimized model
    """
    if torch.cuda.is_available():
        model = model.cuda()
        if use_half_precision:
            model = model.half()
    return model

def batch_to_device(batch, device):
    """
    Move a tensor, list of tensors, or dict of tensors to the specified device.
    Args:
        batch: torch.Tensor, list, or dict
        device: torch.device
    Returns:
        The batch moved to the specified device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [batch_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}") 