"""
📌 Purpose – Verify PyTorch installation and CUDA support.
🔄 Latest Changes – Created script to verify PyTorch installation.
⚙️ Key Logic – Check PyTorch version, CUDA availability, and run a simple tensor operation.
📂 Expected File Path – /project_root/verify_pytorch.py
🧠 Reasoning – Provides a comprehensive check of the PyTorch installation.
"""

import torch
import torchvision
import platform
import sys


def print_separator():
    print("-" * 50)


def verify_pytorch():
    print_separator()
    print("PyTorch Installation Verification")
    print_separator()
    
    # System information
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    print_separator()
    
    # Create a simple tensor
    print("Creating a simple tensor...")
    x = torch.rand(5, 3)
    print(x)
    
    # Try a CUDA tensor if available
    if torch.cuda.is_available():
        print("\nCreating a CUDA tensor...")
        y = torch.rand(5, 3).cuda()
        print(y)
        
        # Simple operation
        print("\nPerforming a simple operation on CUDA...")
        z = y * 2
        print(z)
    
    print_separator()
    print("Verification completed successfully!")
    print_separator()


if __name__ == "__main__":
    verify_pytorch() 