#!/bin/bash
# 📌 Purpose – Entrypoint script for the Video Timeline Analyzer Docker container
# 🔄 Latest Changes – Initial creation. Handles initialization and proper command execution.
# ⚙️ Key Logic – 
#    1. Sets up environment and initializes any required components
#    2. Handles special commands (dev mode, specific scripts)
#    3. Executes the provided command or defaults to an appropriate action
# 📂 Expected File Path – ./entrypoint.sh
# 🧠 Reasoning – Provides flexibility in how the container can be used while ensuring proper initialization

set -e
export PYTHONPATH="/app"

# Define functions
initialize_app() {
    echo "Initializing Video Timeline Analyzer..."
    
    # Check if TransNetV2 weights exist, download if not
    if [ ! -f "/app/models/transnetv2_weights/transnetv2-pytorch-weights.pth" ]; then
        echo "TransNetV2 weights not found, downloading..."
        python3 /app/src/scene_detection/transnetv2_repo/inference-pytorch/download_transnetv2_weights.py
    fi
    
    # Add any other initialization steps here
}

# Initialize the application
initialize_app

# If no command is provided, start in development mode
if [ "$#" -eq 0 ]; then
    echo "No command provided. Starting in interactive mode..."
    exec tail -f /dev/null
fi

# Handle special commands
case "$1" in
    dev)
        echo "Starting in development mode..."
        exec tail -f /dev/null
        ;;
    process)
        # Shift to remove the 'process' argument
        shift
        echo "Running full video analysis pipeline..."
        exec python3 /app/src/pipeline/run_full_pipeline.py "$@"
        ;;
    scene)
        # Shift to remove the 'scene' argument
        shift
        echo "Running scene detection only..."
        exec python3 /app/src/scene_detection/scene_detection.py "$@"
        ;;
    analyze)
        # Shift to remove the 'analyze' argument
        shift
        echo "Running video analysis..."
        exec python3 /app/scripts/analyze_video.py "$@"
        ;;
    *)
        # Execute the provided command
        echo "Executing command: $@"
        exec "$@"
        ;;
esac 