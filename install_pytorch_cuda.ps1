# Install PyTorch with CUDA Support
Write-Host "Installing PyTorch with CUDA 12.1 support globally..." -ForegroundColor Green

# Try to uninstall existing PyTorch installations
Write-Host "Removing existing PyTorch installations..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA support
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

Write-Host "`nSetup completed!" -ForegroundColor Green 