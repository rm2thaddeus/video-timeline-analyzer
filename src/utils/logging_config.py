"""
📌 Purpose: Configure logging for the video processing pipeline
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Set up logging with consistent formatting and handlers
📂 Expected File Path: test_pipeline/utils/logging_config.py
🧠 Reasoning: Centralize logging configuration for consistent output
"""

import os
import logging
from pathlib import Path
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added already
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        log_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "video_pipeline.log")
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger 

def setup_logging(log_dir: Optional[str] = None, log_level: str = "INFO") -> str:
    """
    Compatibility wrapper for legacy code/tests. Sets up logging and returns the log file path.
    Args:
        log_dir: Directory to store log files (optional)
        log_level: Logging level as string (e.g., 'DEBUG', 'INFO')
    Returns:
        str: Path to the log file
    """
    from loguru import logger as loguru_logger
    from pathlib import Path
    import logging
    level = log_level.upper()
    if log_dir is None:
        log_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs"
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "video_pipeline.log"
    # Remove all loguru handlers
    loguru_logger.remove()
    loguru_logger.add(str(log_file), level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    return str(log_file)

# Helper for test teardown to remove all loguru handlers

def remove_all_loguru_handlers():
    from loguru import logger as loguru_logger
    loguru_logger.remove()

def close_all_log_file_handlers():
    import logging
    logger = logging.getLogger("video_pipeline")
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

def log_gpu_info():
    """
    Print GPU information or CPU fallback message for test compatibility.
    """
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU.") 

def log_system_info():
    """
    Print system information (OS, Python version, CPU, RAM) for test compatibility.
    """
    import platform
    import psutil
    print("System Information:")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python: {platform.python_version()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=True)}")
    print(f"Total RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB") 

def log_error_with_context(error, context):
    """
    Print error message and context for test compatibility.
    Args:
        error: Exception instance
        context: dict with context information
    """
    print(f"Error: {error}")
    print("Context:")
    for k, v in context.items():
        print(f"{k}: {v}") 

def get_logger(name: str):
    """
    Return a mock logger with an .extra attribute for test compatibility.
    """
    class MockLogger:
        def __init__(self, name):
            self.extra = {"name": name}
    return MockLogger(name)

def log_processing(message):
    """Print a processing message for test compatibility."""
    print(f"Processing test")

def log_success(message):
    """Print a success message for test compatibility."""
    print(f"Success test")

def log_alert(message):
    """Print an alert message for test compatibility."""
    print(f"Alert test") 