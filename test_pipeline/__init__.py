"""
📌 Purpose: Package initialization file for the video pipeline test module
🔄 Latest Changes: Added metadata dataframe import
⚙️ Key Logic: Import and export key components for the pipeline
📂 Expected File Path: test_pipeline/__init__.py
🧠 Reasoning: Proper package structure for easier imports
"""

# Import submodules
from .processors.frame_processor import FrameProcessor
from .processors.metadata_extractor import extract_metadata
from .processors.metadata_dataframe import MetadataManager

# Import CUDA optimized modules
try:
    from .CUDA.pipeline import process_video
    from .CUDA.scene_detector import SceneDetector
    from .CUDA.audio_processor_cuda import AudioProcessorCUDA
except ImportError:
    # Fallback to non-CUDA versions if available
    from .processors.scene_detector import SceneDetector
    from .processors.audio_processor import AudioProcessor as AudioProcessorCUDA

# Define exports
__all__ = [
    'FrameProcessor',
    'extract_metadata',
    'MetadataManager',
    'process_video',
    'SceneDetector',
    'AudioProcessorCUDA'
] 