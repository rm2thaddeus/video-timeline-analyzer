"""
📌 Purpose: Configure logging settings for the video processing pipeline
🔄 Latest Changes: Initial implementation of logging configuration
⚙️ Key Logic: Sets up loguru logger with file and console outputs
📂 Expected File Path: src/utils/logging_config.py
🧠 Reasoning: Ensures consistent logging across all modules with proper formatting
"""

import os
import sys
from datetime import datetime
from loguru import logger

def setup_logging(log_dir="logs", log_level="INFO"):
    """
    Configure logging settings for the application.
    
    Args:
        log_dir (str): Directory to store log files
        log_level (str): Minimum log level to capture
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cuda_pipeline_{timestamp}.log")
    
    # Remove default logger
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file handler
    logger.add(
        log_file,
        rotation="100 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def log_gpu_info():
    """
    Log GPU information if CUDA is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"Total GPU Memory: {total_memory:.2f} GB")
        else:
            logger.warning("CUDA is not available. Using CPU.")
    except ImportError:
        logger.warning("PyTorch not found. Cannot log GPU information.")

def log_system_info():
    """
    Log system information and environment details.
    """
    import platform
    import psutil
    
    logger.info("System Information:")
    logger.info(f"OS: {platform.system()} {platform.version()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / 1e9:.2f} GB")
    logger.info(f"Available RAM: {memory.available / 1e9:.2f} GB")

def log_error_with_context(error, context=None):
    """
    Log an error with additional context information.
    
    Args:
        error (Exception): The error to log
        context (dict, optional): Additional context information
    """
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg += f"\nContext: {context}"
    logger.exception(error_msg)

def get_logger(name):
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Name for the logger
    
    Returns:
        logger: Configured logger instance
    """
    return logger.bind(name=name)

# Example usage of custom log levels
logger.level("PROCESSING", no=15, color="<cyan>")
logger.level("SUCCESS", no=25, color="<green>")
logger.level("ALERT", no=35, color="<yellow>")

# Add custom logging methods
def log_processing(message):
    """Log a processing status message."""
    logger.log("PROCESSING", message)

def log_success(message):
    """Log a success message."""
    logger.log("SUCCESS", message)

def log_alert(message):
    """Log an alert message."""
    logger.log("ALERT", message) 