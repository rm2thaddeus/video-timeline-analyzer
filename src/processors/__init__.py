"""
📌 Purpose: Export processor modules for the video processing pipeline
🔄 Latest Changes: Added MetadataManager export
⚙️ Key Logic: Import and re-export processor classes for easier imports
📂 Expected File Path: test_pipeline/processors/__init__.py
🧠 Reasoning: Simplify imports for pipeline components
"""

# This file makes 'processors' a Python package.

from .frame_processor import FrameProcessor
from .scene_detector import SceneDetector
from .audio_processor import AudioProcessor
from .metadata_extractor import extract_metadata
from .metadata_dataframe import MetadataManager

__all__ = [
    'FrameProcessor',
    'SceneDetector',
    'AudioProcessor',
    'extract_metadata',
    'MetadataManager'
] 