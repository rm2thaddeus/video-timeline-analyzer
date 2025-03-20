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