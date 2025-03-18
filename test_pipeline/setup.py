"""
📌 Purpose: Setup script to install dependencies
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Install required packages for the video processing pipeline
📂 Expected File Path: test_pipeline/setup.py
🧠 Reasoning: Simplify dependency installation
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies for the video processing pipeline."""
    print("Installing dependencies for the video processing pipeline...")
    
    # Get the requirements file path
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"Error: Requirements file not found at {requirements_path}")
        return False
    
    try:
        # Install dependencies using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

if __name__ == "__main__":
    install_dependencies() 