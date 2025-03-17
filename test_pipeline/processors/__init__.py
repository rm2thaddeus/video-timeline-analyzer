"""
📌 Purpose: Initialize the processors package
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Make processors directory a proper Python package
📂 Expected File Path: test_pipeline/processors/__init__.py
🧠 Reasoning: Proper Python package structure for better imports
"""

# This file makes 'processors' a Python package.

from .metadata_extractor import extract_metadata
from .scene_detector import SceneDetector
from .audio_processor import AudioProcessor
from .frame_processor import FrameProcessor

__all__ = [
    'extract_metadata',
    'SceneDetector',
    'AudioProcessor',
    'FrameProcessor'
] 