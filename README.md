# Video Timeline Analyzer

A powerful video analysis tool that leverages CUDA acceleration for efficient processing of video content, including scene detection, audio transcription, and visual analysis.

## 🚀 Features

- **CUDA-Accelerated Processing**
  - GPU-optimized video frame extraction
  - CUDA-accelerated scene detection
  - Hardware-accelerated audio transcription
  - Efficient memory management

- **Advanced Analysis**
  - Scene detection with black frame handling
  - Audio transcription using Whisper
  - Visual analysis with CLIP and BLIP
  - Frame-by-frame processing

- **Comprehensive Output**
  - Scene timestamps and metadata
  - Audio transcriptions (SRT and JSON)
  - Frame captions and embeddings
  - Visual analysis results

## 🛠 Requirements

- Python 3.8+
- CUDA-capable GPU
- FFmpeg
- PyTorch with CUDA support
- Additional dependencies in `requirements.txt`

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-timeline-analyzer.git
cd video-timeline-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

Basic usage:
```bash
python src/processors/cuda_pipeline.py video_path --output_dir output/video_name --model medium
```

Options:
- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--gpu_memory`: Fraction of GPU memory to use (0.0-1.0)
- `--sequential`: Disable parallel processing
- `--output_dir`: Directory for output files

## 📁 Project Structure

```
video-timeline-analyzer/
├── src/
│   ├── processors/
│   │   ├── cuda_pipeline.py
│   │   ├── scene_detector.py
│   │   └── audio_processor.py
│   ├── utils/
│   │   ├── gpu_utils.py
│   │   └── logging_config.py
│   ├── config/
│   │   └── cuda_config.py
│   └── models/
├── tests/
├── docs/
└── output/
```

## 🔧 Configuration

CUDA settings can be configured in `src/config/cuda_config.py`:
- GPU memory usage
- Model parameters
- Processing settings
- Cache configuration

## 📊 Output Structure

```
output/video_name/
├── frames/
│   ├── scene_XXX.jpg
│   └── frames_data.json
├── audio/
│   └── audio.wav
├── transcripts/
│   ├── transcript.srt
│   └── transcript.json
└── metadata/
    └── metadata.json
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for audio transcription
- CLIP and BLIP for visual analysis
- FFmpeg for media processing
- PyTorch team for CUDA support

## 📈 Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development plans and progress tracking.