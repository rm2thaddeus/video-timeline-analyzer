"""
📌 Purpose: Test script to verify logging functionality in the CUDA pipeline
🔄 Latest Changes: Created test script
⚙️ Key Logic: Configures and verifies logging functionality
📂 Expected File Path: test_pipeline/CUDA/test_logging.py
🧠 Reasoning: Provides a simple way to verify logging is working properly
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Get our configuration
import CUDA.cuda_config as config

def setup_logging():
    """Set up proper console logging"""
    try:
        # Try to import coloredlogs for better console output
        import coloredlogs
        coloredlogs.install(
            level=logging.INFO,
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print("Using colored logs - OK")
    except ImportError:
        # Configure standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        print("Using standard logs (coloredlogs not installed)")
    
    # Ensure all our pipeline loggers are configured properly
    for logger_name in ["run_pipeline", "video_pipeline_cuda", "scene_detector_cuda"]:
        log = logging.getLogger(logger_name)
        
        # Make sure we have a console handler
        has_console_handler = False
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                has_console_handler = True
                break
        
        if not has_console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            log.addHandler(console_handler)
        
        # Ensure logger passes through messages
        log.setLevel(logging.INFO)
        log.propagate = True

def test_logging():
    """Test logging to verify all handlers are working"""
    # Set up logging
    setup_logging()
    
    # Get our loggers
    main_logger = logging.getLogger("run_pipeline")
    pipeline_logger = logging.getLogger("video_pipeline_cuda")
    scene_logger = logging.getLogger("scene_detector_cuda")
    
    # Test logging to each logger
    print("\n===== Testing Logging =====")
    main_logger.info("[INFO] This is a test message from the main logger")
    pipeline_logger.info("[PIPELINE] This is a test message from the pipeline logger")
    scene_logger.info("[SCENES] This is a test message from the scene detector logger")
    
    # Verify logging to file
    log_file = config.LOGS_DIR / f"test_logging.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    for logger in [main_logger, pipeline_logger, scene_logger]:
        logger.addHandler(file_handler)
    
    main_logger.info("[FILE] This message should appear in the log file")
    pipeline_logger.info("[FILE] This message should also appear in the log file")
    scene_logger.info("[FILE] This message should also appear in the log file")
    
    print(f"Log file created at: {log_file}")
    
    # Test different log levels
    main_logger.debug("[DEBUG] This is a DEBUG message (might not show with default settings)")
    main_logger.info("[INFO] This is an INFO message")
    main_logger.warning("[WARN] This is a WARNING message")
    main_logger.error("[ERROR] This is an ERROR message")
    
    # Test bracketed format used in pipeline
    print("\n===== Testing Tagged Message Style =====")
    print("[INFO] This message uses a consistent prefix format")
    print("[WARN] This warning message should display well in Windows terminals")
    print("[ERROR] This error message should display well in Windows terminals")
    print("[DEBUG] This debug message should display well in Windows terminals")
    
    print("\n===== Test Complete =====")
    print(f"Check the log file at {log_file} to verify file logging is working")

if __name__ == "__main__":
    # Ensure our log directory exists
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Run the test
    test_logging()
    
    print("\nTo run the pipeline with the new logging, use:")
    print("python test_pipeline/CUDA/run_pipeline.py path/to/your/video.mp4\n") 