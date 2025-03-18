#!/usr/bin/env python
# Purpose: Configure Python to use the global PyTorch installation with CUDA support
# Latest Changes: Initial script to modify Python path
# Key Logic: Adds the global site-packages to sys.path before importing torch
# Expected File Path: /project_root/use_global_torch.py
# Reasoning: Allows using CUDA-enabled PyTorch from within the virtual environment

import sys
import os

# Add the global site-packages directory to the Python path
global_site_packages = r'C:\Users\aitor\AppData\Local\Programs\Python\Python39\lib\site-packages'
if global_site_packages not in sys.path:
    sys.path.insert(0, global_site_packages)

# Now import torch and verify CUDA is available
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}")
print(f"PyTorch path: {torch.__file__}")

# This script can be imported at the beginning of your Python files to use the global PyTorch
# Example usage in other scripts:
# import use_global_torch
# import torch  # This will now use the global PyTorch installation 