# Run the CUDA-optimized audio processor on the test video
Write-Host "Running CUDA-optimized audio processor with word-level accuracy..." -ForegroundColor Green

# Activate virtual environment if it exists
$venvPath = "..\..\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $venvPath
} else {
    Write-Host "Warning: Virtual environment not found, using system Python" -ForegroundColor Yellow
}

# Run the audio processor
try {
    Write-Host "Starting audio processing..." -ForegroundColor Cyan
    python run_audio_processor.py "C:\Users\aitor\Downloads\videotest.mp4" --chunk-duration 20 --overlap 1.5 --max-workers 8
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Audio processing completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Error: Audio processing failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Wait for user input before closing
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 