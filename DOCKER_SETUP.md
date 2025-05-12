# 📌 Docker Setup Guide for Video Timeline Analyzer

This guide explains how to set up and run the Video Timeline Analyzer application using Docker.

## Prerequisites

1. Install Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop/)
2. For GPU support: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Starting Docker Desktop

### Windows
1. Search for "Docker Desktop" in the Start menu and click to launch
2. Alternatively, run from PowerShell:
   ```
   Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
   ```

> Note: If the above path doesn't work, locate where Docker Desktop is installed on your system.

### macOS
1. Open Docker Desktop from the Applications folder
2. Alternatively, run from Terminal:
   ```
   open -a Docker
   ```

### Linux
Docker Desktop is not required on Linux. Just ensure the Docker daemon is running:
```
sudo systemctl start docker
```

## Building and Running the Application

### First-time setup

1. Make sure Docker Desktop is running
2. Create the `models/transnetv2_weights` directory:
   ```
   mkdir -p models/transnetv2_weights
   ```
3. Build the Docker images:
   ```
   docker-compose build
   ```

### Development Mode

```
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

This will:
- Mount your local source code into the container
- Run the application in development mode
- Enable hot reloading of code changes

### Production Mode

```
docker-compose up
```

## Processing Videos

To process a video:

```
docker-compose run app process --input /app/data/your_video.mp4 [other options]
```

## Troubleshooting

### Docker Desktop Not Found

If you get an error like `The system cannot find the file specified` when starting Docker Desktop:

1. Check if Docker Desktop is installed by looking in Program Files or Applications folder
2. Reinstall Docker Desktop if needed
3. Add Docker Desktop to your PATH

### NVIDIA GPU Support Issues

If GPU is not recognized:

1. Ensure NVIDIA drivers are up to date
2. Check that NVIDIA Container Toolkit is installed
3. Verify Docker daemon configuration has NVIDIA runtime enabled

### Volume Mount Issues

If volumes aren't properly mounting:

1. Ensure paths in docker-compose files are correct for your system
2. Check that directories exist on your host system
3. Try using absolute paths instead of relative paths 