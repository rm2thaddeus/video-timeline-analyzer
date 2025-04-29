# Video Timeline Analyzer Roadmap

<!--
📌 Purpose – Project development roadmap and environment setup guidance
🔄 Latest Changes – Updated environment setup instructions: removed reference to shared venv, now recommend local venv in project root. Note added: PyTorch must be installed manually for compatibility.
⚙️ Key Logic – Phased development, reproducibility, and environment management
📂 Expected File Path – ROADMAP.md (project root)
🧠 Reasoning – Ensures clarity and reproducibility for all contributors, avoids CUDA/PyTorch conflicts.
-->

This roadmap outlines the development path for the Video Timeline Analyzer project, providing a clear timeline of planned features, milestones, and progress tracking.

## Environment Setup (Updated)

- Always use a local virtual environment for this project to avoid version conflicts and ensure reproducibility.
- Prerequisites:
  - Python 3.9 or higher
  - NVIDIA GPU with compatible drivers and CUDA installed
- Create and activate a local virtual environment in the project root:
  ```sh
  python -m venv .venv
  .venv\Scripts\activate  # On Windows
  source .venv/bin/activate  # On macOS/Linux
  ```
- Install dependencies from `requirements.txt` (PyTorch line is commented out):
  ```sh
  pip install -r requirements.txt
  ```
- **Install PyTorch in the venv:**
  - Use the official selector to get the right command for your system and CUDA version: https://pytorch.org/get-started/locally/
  - Example (CUDA 12.1):
    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
- **Test GPU availability after installation:**
  ```sh
  python scripts/test_gpu.py
  ```
- All dependencies, including PyTorch, must be installed in the venv for reproducibility.
- Remove any references to the old shared venv (`C:\WINDOWS\System32\pytorch-env`).

## PyTorch & CUDA Troubleshooting

- **Multiple PyTorch Versions:**
  - Use venvs to avoid conflicts. Do not install PyTorch globally.
- **CUDA Version Mismatch:**
  - Check your NVIDIA driver and CUDA version with `nvidia-smi`.
  - Use the correct PyTorch build for your CUDA version.
- **ImportError or ModuleNotFoundError:**
  - Ensure the venv is activated before running any code.
- **Best Practices:**
  1. Always use venvs for Python projects.
  2. Avoid global PyTorch installs.
  3. Use flexible version specs in requirements.txt.
  4. Test GPU availability after setup.

## Phase 1: Project Setup & Core Infrastructure (Week 1)

- [x] Create GitHub repository
- [x] Set up initial documentation (README, ROADMAP)
- [x] Establish project structure and architecture
- [x] Set up dev environment configuration (requirements.txt, environment.yml) using a local venv in the project root
- [x] Address dependency management between PyTorch and Python to ensure smooth GPU integration
- [x] Implement GPU detection and utilization (diagnostic validated by test_gpu.py)
- [x] Write initial unit tests for GPU utilities
- [x] Create unit testing framework
- [ ] Implement CI/CD pipeline (GitHub Actions)

## Phase 2: Video Processing & Scene Detection (Weeks 2-3)

- [x] Implement video file ingestion module and integrate it into the processing pipeline
- [x] Implement scene detection using OpenCV-based detector with CUDA optimization
- [x] Create scene metadata extraction framework with complete timestamping and resolution details
- [x] Implement frame extraction for key moments with integration into the overall video pipeline
- [x] Set up FFmpeg integration for audio extraction (integrated with scene detection)
- [x] Implement CUDA acceleration for video processing pipeline
- [x] Optimize memory management for GPU processing
- [x] Add mixed precision support for better performance
- [ ] Implement batch processing for multiple videos
- [ ] Add support for different video formats and codecs

## Phase 3: Audio Analysis & Transcription (Weeks 3-4)

- [x] Integrate OpenAI Whisper transcription with CUDA acceleration
- [x] Implement configurable model selection (tiny to large) with optimized parameters
- [x] Develop subtitle generation with SRT outputs and precise timestamps
- [x] Add CUDA optimization for audio processing pipeline
- [x] Add support for WhisperX for word-level precision
- [x] Implement automatic cleanup of intermediate audio chunks
- [ ] Optional: Implement speaker diarization
- [ ] Enhance audio feature extraction (volume, pitch, tone)
- [ ] Implement transcript sentiment analysis

## Phase 4: Visual Context Analysis (Weeks 5-6)

- [x] Integrate CLIP for scene tag generation and visual embeddings
- [x] Implement BLIP-2 for descriptive captions with CUDA acceleration
- [x] Optimize BLIP model configuration for better captions
- [ ] Develop facial detection and tracking
- [ ] Integrate DeepFace for emotion recognition
- [ ] Create visual metadata aggregation from multiple sources
- [ ] Implement vector embeddings for scenes
- [ ] Set up semantic similarity search

## Phase 5: Data Fusion & Storage (Weeks 7-8)

- [ ] Design unified database schema (SQLite) for storing all metadata
- [ ] Implement JSON export/import functionality for data interchange
- [ ] Develop a "viral moment" scoring algorithm
- [ ] Create a metadata fusion pipeline
- [ ] Implement vector storage with FAISS/ChromaDB
- [ ] Set up a caching mechanism for processed videos
- [ ] Develop a robust data versioning system

## Phase 6: User Interface Development (Weeks 9-11)

- [ ] Design UI wireframes and mockups
- [ ] Implement video player component
- [ ] Develop interactive timeline visualization
- [ ] Create subtitle overlay system
- [ ] Implement scene metadata display panel
- [ ] Develop navigation controls
- [ ] Refine UX for metadata exploration

## Phase 7: Integration & System Optimization (Weeks 12-13)

- [x] Integrate GPU diagnostic metrics
- [x] Implement dynamic GPU memory management
- [x] Add mixed precision training support
- [ ] Seamlessly integrate all components
- [ ] Implement parallel processing strategies
- [ ] Develop advanced caching strategies
- [ ] Implement robust error handling

## Phase 8: Testing & Refinement (Weeks 14-15)

- [ ] Conduct integration testing
- [ ] Perform usability testing
- [ ] Optimize UI responsiveness
- [ ] Identify and resolve edge cases
- [ ] Enhance error messages
- [ ] Update documentation
- [ ] Create example projects

## Phase 9: Deployment & Distribution (Week 16)

- [ ] Package the application
- [ ] Create installer scripts
- [ ] Develop user documentation
- [ ] Implement update mechanism
- [ ] Prepare tutorials
- [ ] Release version 1.0

## Future Enhancements (Post v1.0)

- [ ] Extend multi-language support
- [ ] Explore custom training
- [ ] Develop web-based deployment
- [ ] Integrate cloud storage
- [ ] Enable batch processing
- [ ] Develop API
- [ ] Create mobile companion app

## Progress Tracking

| Phase   | Completion | Status      |
|---------|------------|-------------|
| Phase 1 | 100%       | Complete    |
| Phase 2 | ~90%       | Near Complete (CUDA implementation done) |
| Phase 3 | ~85%       | In Progress (CUDA transcription complete) |
| Phase 4 | ~30%       | In Progress (CLIP/BLIP integration done) |
| Phase 5 | 0%         | Not Started |
| Phase 6 | 0%         | Not Started |
| Phase 7 | ~40%       | In Progress (GPU optimizations) |
| Phase 8 | 0%         | Not Started |
| Phase 9 | 0%         | Not Started |

_Last Updated: 2025-03-19_

## Recent Achievements

1. Successfully implemented CUDA acceleration for:
   - Video processing pipeline
   - Scene detection
   - Audio transcription
   - Frame extraction and analysis
   - CLIP and BLIP model integration

2. Optimized GPU memory management:
   - Added dynamic memory allocation
   - Implemented mixed precision support
   - Added memory cleanup routines
   - Configured optimal batch sizes

3. Enhanced model configurations:
   - Optimized BLIP caption generation
   - Improved Whisper transcription accuracy
   - Added configurable model selection

## Next Steps

1. Complete the migration of CUDA implementation to main project:
   - Move files from test_pipeline to src structure
   - Update import paths
   - Add comprehensive tests
   - Update documentation

2. Focus on data storage and management:
   - Design database schema
   - Implement vector storage
   - Create data versioning system

3. Begin UI development:
   - Design wireframes
   - Implement basic video player
   - Create timeline visualization

This roadmap will be regularly reviewed and updated throughout the development process to reflect progress, changes in priorities, and new insights.