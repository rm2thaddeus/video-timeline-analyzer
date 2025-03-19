"""
📌 Purpose: GPU utilities for managing CUDA operations and memory
🔄 Latest Changes: Initial implementation of GPU utilities
⚙️ Key Logic: Provides functions for GPU setup, memory management, and device selection
📂 Expected File Path: src/utils/gpu_utils.py
🧠 Reasoning: Centralizes GPU-related operations for better resource management
"""

import os
import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from loguru import logger
import gc

def setup_cuda(gpu_memory_fraction=0.8, enable_benchmark=True):
    """
    Configure CUDA settings and initialize GPU environment.
    
    Args:
        gpu_memory_fraction (float): Fraction of GPU memory to use (0.0-1.0)
        enable_benchmark (bool): Whether to enable cuDNN benchmark mode
    
    Returns:
        torch.device: Selected device (GPU or CPU)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using CPU instead.")
        return torch.device('cpu')
    
    # Set device
    device = torch.device('cuda')
    
    # Configure cuDNN
    if enable_benchmark:
        cudnn.benchmark = True
    cudnn.deterministic = True
    
    # Set memory fraction
    if gpu_memory_fraction > 0:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        max_memory = int(total_memory * gpu_memory_fraction)
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
    
    # Log GPU info
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
    logger.info(f"Allocated memory: {max_memory / 1e9:.2f} GB")
    
    return device

def clear_gpu_memory():
    """
    Clear GPU memory cache and run garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        cuda.synchronize()
        logger.debug("GPU memory cleared")

def get_gpu_memory_info():
    """
    Get current GPU memory usage information.
    
    Returns:
        dict: Memory information including total, allocated, and free memory
    """
    if not torch.cuda.is_available():
        return None
    
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = reserved - allocated
    
    return {
        'total_gb': total / 1e9,
        'reserved_gb': reserved / 1e9,
        'allocated_gb': allocated / 1e9,
        'free_gb': free / 1e9
    }

def optimize_gpu_memory(model, use_half_precision=True):
    """
    Optimize model memory usage with optional half-precision.
    
    Args:
        model (torch.nn.Module): PyTorch model to optimize
        use_half_precision (bool): Whether to use FP16 precision
    
    Returns:
        torch.nn.Module: Optimized model
    """
    if not torch.cuda.is_available():
        return model
    
    if use_half_precision:
        model = model.half()
    
    return model.cuda()

def batch_to_device(batch, device):
    """
    Move a batch of data to the specified device.
    
    Args:
        batch: Input batch (can be tensor, list, tuple, or dict)
        device: Target device
    
    Returns:
        Batch data on the target device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [batch_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    return batch

def get_optimal_batch_size(model, input_shape, target_memory_usage=0.7):
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        model (torch.nn.Module): Model to analyze
        input_shape (tuple): Shape of a single input
        target_memory_usage (float): Target memory usage fraction
    
    Returns:
        int: Estimated optimal batch size
    """
    if not torch.cuda.is_available():
        return 1
    
    clear_gpu_memory()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory * target_memory_usage
    
    # Test with batch size of 1
    try:
        sample_input = torch.randn(1, *input_shape).cuda()
        with torch.no_grad():
            _ = model(sample_input)
        memory_per_sample = torch.cuda.memory_allocated()
        optimal_batch_size = max(1, int(available_memory / memory_per_sample))
    except Exception as e:
        logger.warning(f"Error estimating batch size: {e}")
        optimal_batch_size = 1
    finally:
        clear_gpu_memory()
    
    return optimal_batch_size
