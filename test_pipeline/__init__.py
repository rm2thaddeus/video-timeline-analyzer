"""
📌 Purpose: Initialize the test_pipeline package
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Make test_pipeline directory a proper Python package
📂 Expected File Path: test_pipeline/__init__.py
🧠 Reasoning: Proper Python package structure for better imports
"""

from .pipeline import process_video, ensure_directories

__version__ = "0.1.0"

__all__ = [
    'process_video',
    'ensure_directories'
] 