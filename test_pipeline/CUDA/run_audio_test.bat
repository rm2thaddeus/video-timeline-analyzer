@echo off
REM Run the CUDA-optimized audio processor on the test video
echo Running CUDA-optimized audio processor with word-level accuracy...

REM Activate virtual environment if it exists
if exist ..\..\venv\Scripts\activate.bat (
    call ..\..\venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found, using system Python
)

REM Run the audio processor
python run_audio_processor.py "C:\Users\aitor\Downloads\videotest.mp4" --chunk-duration 20 --overlap 1.5 --max-workers 8

REM Check if the command was successful
if %ERRORLEVEL% EQU 0 (
    echo Audio processing completed successfully!
) else (
    echo Error: Audio processing failed with error code %ERRORLEVEL%
)

REM Pause to see the output
pause 