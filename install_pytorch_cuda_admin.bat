@echo off
echo Running PyTorch CUDA installation with administrator privileges...
powershell -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0install_pytorch_cuda.ps1\"' -Verb RunAs"
pause 