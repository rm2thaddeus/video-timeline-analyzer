"""
📌 Purpose: GPU utilities for the video processing pipeline
🔄 Latest Changes: Enhanced GPU detection to check for NVIDIA drivers even if PyTorch doesn't detect CUDA
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

def check_nvidia_drivers() -> bool:
    """
    Check if NVIDIA drivers are installed and working by running nvidia-smi.
    
    Returns:
        Boolean indicating if NVIDIA drivers are available
    """
    try:
        process = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return process.returncode == 0
    except Exception:
        return False

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
    
    # Check for CUDA via PyTorch
    if torch.cuda.is_available():
        gpu_info["detected"] = True
        gpu_info["gpu_type"] = "cuda"
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["optimal_device"] = "cuda"
        logger.info(f"CUDA GPU detected via PyTorch. Device count: {gpu_info['device_count']}")
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_info["detected"] = True
        gpu_info["gpu_type"] = "mps"
        gpu_info["device_count"] = 1
        gpu_info["optimal_device"] = "mps"
        logger.info("Apple Silicon MPS device detected")
    
    # Check for NVIDIA drivers even if PyTorch doesn't detect CUDA
    elif check_nvidia_drivers():
        gpu_info["detected"] = True
        gpu_info["gpu_type"] = "cuda"
        gpu_info["device_count"] = 1  # Assume at least one GPU if nvidia-smi works
        gpu_info["optimal_device"] = "cpu"  # Still use CPU since PyTorch can't use CUDA
        logger.warning("NVIDIA GPU detected via nvidia-smi, but PyTorch doesn't have CUDA support. "
                      "Using CPU for computation. Consider reinstalling PyTorch with CUDA support.")
    
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

def get_nvidia_smi_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information using nvidia-smi if available.
    
    Returns:
        Dict with memory information in GB from nvidia-smi
    """
    memory_info = {
        "total_gb": 0.0,
        "used_gb": 0.0,
        "free_gb": 0.0
    }
    
    try:
        process = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        output = process.stdout.strip().split(',')
        if len(output) >= 3:
            memory_info["total_gb"] = float(output[0].strip()) / 1024  # Convert MiB to GiB
            memory_info["used_gb"] = float(output[1].strip()) / 1024
            memory_info["free_gb"] = float(output[2].strip()) / 1024
            
            logger.info(f"GPU Memory (nvidia-smi) - Total: {memory_info['total_gb']:.2f}GB, "
                       f"Used: {memory_info['used_gb']:.2f}GB, "
                       f"Free: {memory_info['free_gb']:.2f}GB")
    except Exception as e:
        logger.error(f"Error getting memory information from nvidia-smi: {e}")
    
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
    # Try to get memory info from PyTorch first
    memory_info = get_memory_info()
    
    # If PyTorch doesn't provide memory info but we have NVIDIA drivers, try nvidia-smi
    if memory_info["total_gb"] == 0 and check_nvidia_drivers():
        memory_info = get_nvidia_smi_memory_info()
    
    # Default conservative batch sizes based on model and available memory
    if not torch.cuda.is_available() and not check_nvidia_drivers():
        return 1
    
    free_memory_gb = memory_info.get("free_gb", 0)
    
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
    
    if gpu_info["detected"] and gpu_info["optimal_device"] != "cpu":
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
    
    # If we have NVIDIA drivers but PyTorch doesn't have CUDA support
    if gpu_info["detected"] and gpu_info["optimal_device"] == "cpu" and gpu_info["gpu_type"] == "cuda":
        logger.warning("Using CPU for computation, but NVIDIA GPU is available. "
                      "Consider reinstalling PyTorch with CUDA support for better performance.")
        
        # Get GPU info from nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            gpu_name = result.stdout.strip()
            
            properties = {
                "name": gpu_name,
                "memory": get_nvidia_smi_memory_info()
            }
            
            return torch.device("cpu"), properties
        except Exception:
            pass
    
    logger.warning("Using CPU for computation")
    return torch.device("cpu"), None 