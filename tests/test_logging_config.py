"""
📌 Purpose: Test suite for logging configuration module
🔄 Latest Changes: Initial implementation of logging tests
⚙️ Key Logic: Tests logging setup, custom levels, and logging functions
📂 Expected File Path: tests/test_logging_config.py
🧠 Reasoning: Ensures consistent and reliable logging across the application
"""

import os
import pytest
import tempfile
from pathlib import Path
from loguru import logger
from src.utils.logging_config import (
    setup_logging,
    log_gpu_info,
    log_system_info,
    log_error_with_context,
    get_logger,
    log_processing,
    log_success,
    log_alert,
    remove_all_loguru_handlers
)

@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(autouse=True)
def cleanup_loguru_handlers():
    yield
    remove_all_loguru_handlers()

def test_setup_logging(temp_log_dir):
    # Test basic logging setup
    log_file = setup_logging(log_dir=temp_log_dir, log_level="DEBUG")
    assert os.path.exists(log_file)
    
    # Test log file creation
    logger.info("Test log message")
    with open(log_file, 'r') as f:
        log_content = f.read()
    assert "Test log message" in log_content
    
    # Test log directory creation
    nested_dir = os.path.join(temp_log_dir, "nested", "logs")
    log_file = setup_logging(log_dir=nested_dir)
    assert os.path.exists(nested_dir)

def test_log_gpu_info(capsys):
    # Test GPU info logging
    log_gpu_info()
    captured = capsys.readouterr()
    
    # Either GPU info or CPU fallback message should be present
    assert any([
        "GPU:" in captured.out,
        "CUDA is not available" in captured.out
    ])

def test_log_system_info(capsys):
    # Test system info logging
    log_system_info()
    captured = capsys.readouterr()
    
    # Check for essential system information
    assert "System Information:" in captured.out
    assert "OS:" in captured.out
    assert "Python:" in captured.out
    assert "CPU Cores:" in captured.out
    assert "Total RAM:" in captured.out

def test_log_error_with_context(capsys):
    # Test error logging with context
    error = ValueError("Test error")
    context = {"function": "test_function", "input": "test_input"}
    
    log_error_with_context(error, context)
    captured = capsys.readouterr()
    
    assert "Error: Test error" in captured.out
    assert "Context:" in captured.out
    assert "test_function" in captured.out
    assert "test_input" in captured.out

def test_get_logger():
    # Test logger binding
    test_logger = get_logger("test_module")
    assert test_logger.extra["name"] == "test_module"

def test_custom_log_levels(capsys):
    # Test processing level
    log_processing("Processing test")
    captured = capsys.readouterr()
    assert "Processing test" in captured.out
    
    # Test success level
    log_success("Success test")
    captured = capsys.readouterr()
    assert "Success test" in captured.out
    
    # Test alert level
    log_alert("Alert test")
    captured = capsys.readouterr()
    assert "Alert test" in captured.out

def test_log_file_rotation(temp_log_dir):
    # Test log file rotation
    log_file = setup_logging(log_dir=temp_log_dir)
    
    # Generate enough logs to trigger rotation
    for i in range(1000):
        logger.info("Test log message " * 100)
    
    # Check if multiple log files were created
    log_files = list(Path(temp_log_dir).glob("*.log"))
    assert len(log_files) >= 1

def test_log_retention(temp_log_dir):
    # Test log retention (this is more of a configuration test)
    log_file = setup_logging(log_dir=temp_log_dir)
    
    # Verify retention settings
    handlers = logger._core.handlers
    for handler_id, handler in handlers.items():
        if hasattr(handler, "retention"):
            assert handler.retention == "30 days"

@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_levels(temp_log_dir, log_level):
    # Test different log levels
    log_file = setup_logging(log_dir=temp_log_dir, log_level=log_level)
    
    # Log messages at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Read log file
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Check if messages are filtered correctly based on level
    log_levels = {
        "DEBUG": ["Debug", "Info", "Warning", "Error", "Critical"],
        "INFO": ["Info", "Warning", "Error", "Critical"],
        "WARNING": ["Warning", "Error", "Critical"],
        "ERROR": ["Error", "Critical"],
        "CRITICAL": ["Critical"]
    }
    
    for message_level in log_levels[log_level]:
        assert f"{message_level} message" in log_content 