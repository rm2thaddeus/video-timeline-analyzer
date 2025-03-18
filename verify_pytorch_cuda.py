"""
Purpose – Verify PyTorch installation with CUDA support
Latest Changes – Created script to verify global PyTorch CUDA installation
Key Logic – Check PyTorch version, CUDA availability, and run a simple GPU tensor operation
Expected File Path – verify_pytorch_cuda.py
Reasoning – Provides a comprehensive check of the PyTorch CUDA installation
"""

import os
import sys
import platform
import json

def print_separator():
    print("=" * 70)

def verify_pytorch_cuda():
    print_separator()
    print("PyTorch CUDA Installation Verification")
    print_separator()
    
    results = {}
    
    # System information
    results["system"] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_path": sys.executable
    }
    
    # Try importing PyTorch
    try:
        import torch
        import torchvision
        
        results["pytorch"] = {
            "installed": True,
            "version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_available": torch.cuda.is_available()
        }
        
        print(f"Python version: {results['system']['python_version']}")
        print(f"Python path: {results['system']['python_path']}")
        print(f"PyTorch version: {results['pytorch']['version']}")
        print(f"Torchvision version: {results['pytorch']['torchvision_version']}")
        print(f"CUDA available: {results['pytorch']['cuda_available']}")
        
        if torch.cuda.is_available():
            results["cuda"] = {
                "version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0)
            }
            
            print(f"CUDA version: {results['cuda']['version']}")
            print(f"CUDA device count: {results['cuda']['device_count']}")
            print(f"Current CUDA device: {results['cuda']['current_device']}")
            print(f"CUDA device name: {results['cuda']['device_name']}")
            
            # Run a simple GPU test
            print_separator()
            print("Running GPU tensor test...")
            
            # Create a tensor on GPU
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            
            # Perform matrix multiplication
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            z = torch.matmul(x, y)
            end_time.record()
            
            # Synchronize CUDA
            torch.cuda.synchronize()
            
            # Calculate elapsed time
            elapsed_time = start_time.elapsed_time(end_time)
            results["gpu_test"] = {
                "operation": "1000x1000 matrix multiplication",
                "elapsed_ms": elapsed_time
            }
            
            print(f"GPU test completed in {elapsed_time:.2f} ms")
            print("GPU test PASSED!")
        else:
            print("CUDA is not available. GPU tests skipped.")
            results["error"] = "CUDA is not available"
    
    except ImportError as e:
        print(f"Error importing PyTorch: {e}")
        results["error"] = f"Error importing PyTorch: {e}"
    
    print_separator()
    print("Results summary:")
    print(json.dumps(results, indent=2))
    print_separator()
    
    return results

if __name__ == "__main__":
    verify_pytorch_cuda() 