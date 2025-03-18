"""
📌 Purpose – Test GPU utilities for PyTorch compatibility.
🔄 Latest Changes – Enhanced to provide more comprehensive information about PyTorch configuration.
⚙️ Key Logic – Use torch methods to verify GPU detection, memory, and compatibility.
📂 Expected File Path – /test_gpu.py
🧠 Reasoning – Simple script to confirm that GPU is accessible and that PyTorch is correctly installed with CUDA support.
"""

import os
import sys
import platform
import json
from pathlib import Path

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    TORCH_AVAILABLE = False

def test_pytorch_config():
    """Test PyTorch configuration and GPU availability."""
    results = {
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else "Not available"
        }
    }
    
    if not TORCH_AVAILABLE:
        results["error"] = "PyTorch is not available. Please install PyTorch."
        return results
    
    # Check CUDA availability
    results["cuda"] = {
        "is_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        results["cuda"]["current_device"] = current_device
        results["cuda"]["device_name"] = torch.cuda.get_device_name(current_device)
        results["cuda"]["compute_capability"] = torch.cuda.get_device_capability(current_device)
        
        # Get memory info
        results["cuda"]["memory"] = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9
        }
        
        # Test if we can create a tensor on GPU
        try:
            test_tensor = torch.ones(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            results["cuda"]["tensor_creation_test"] = "Success"
        except Exception as e:
            results["cuda"]["tensor_creation_test"] = f"Failed: {str(e)}"
    
    # Check for MPS (Apple Silicon) support
    if platform.system() == 'Darwin' and hasattr(torch.backends, 'mps'):
        results["mps"] = {
            "is_available": torch.backends.mps.is_available(),
            "is_built": torch.backends.mps.is_built()
        }
        
        if torch.backends.mps.is_available():
            try:
                test_tensor = torch.ones(1000, 1000).to('mps')
                del test_tensor
                results["mps"]["tensor_creation_test"] = "Success"
            except Exception as e:
                results["mps"]["tensor_creation_test"] = f"Failed: {str(e)}"
    
    # Check for ROCm support
    if hasattr(torch, 'hip'):
        results["rocm"] = {
            "is_available": torch.hip.is_available() if hasattr(torch.hip, 'is_available') else False
        }
    
    # Check for cuDNN
    if hasattr(torch.backends, 'cudnn'):
        results["cudnn"] = {
            "is_available": torch.backends.cudnn.is_available(),
            "version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "enabled": torch.backends.cudnn.enabled
        }
    
    # Environment variables
    results["environment_variables"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
    }
    
    return results

if __name__ == "__main__":
    results = test_pytorch_config()
    print(json.dumps(results, indent=4))
    
    # Print a summary for quick reference
    print("\n=== PyTorch GPU Support Summary ===")
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"CUDA available: Yes, using {device_name}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("GPU available: Yes, using Apple Silicon MPS")
        elif hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available') and torch.hip.is_available():
            print("GPU available: Yes, using AMD ROCm")
        else:
            print("GPU available: No, using CPU only")
    else:
        print("PyTorch is not installed or not working correctly") 