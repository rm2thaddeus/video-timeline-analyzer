"""
📌 Purpose: Package initialization file for the CUDA-accelerated video processing components
🔄 Latest Changes: Added module exports for clean imports
⚙️ Key Logic: Makes key classes available at the package level
📂 Expected File Path: test_pipeline/CUDA/__init__.py
🧠 Reasoning: Simplify imports from other modules in the codebase
"""

from CUDA.audio_processor_cuda import AudioProcessorCUDA
from CUDA.scene_detector import SceneDetector
from CUDA.pipeline import process_video, ensure_directories

__all__ = [
    'AudioProcessorCUDA', 
    'SceneDetector',
    'process_video',
    'ensure_directories'
]
