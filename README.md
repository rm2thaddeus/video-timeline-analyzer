# Video Timeline Analyzer

A sophisticated video analysis application that generates interactive timelines with rich metadata including scene detection, transcribed subtitles, sentiment analysis, and viral moment scoring.

## 🌟 Features

- **Scene Change Detection**: Automatically identify and mark scene transitions
- **Speech-to-Text**: Generate accurate, timestamped transcripts
- **Scene Context Analysis**: Extract visual metadata and scene descriptions 
- **Sentiment & Emotion Analysis**: Analyze text, faces, and audio for emotional content
- **Viral Moment Detection**: Algorithmically identify potentially viral segments
- **Interactive Timeline UI**: Navigate through video with rich metadata cues
- **Video Summarization**: Generate comprehensive video synopsis

## 🔧 Technologies

- **Scene Detection**: PySceneDetect with OpenCV fallback
- **Transcription**: OpenAI Whisper with WhisperX enhancements
- **Visual Analysis**: CLIP for tagging, BLIP-2 for captions
- **Sentiment Analysis**: Hugging Face Transformers (BERT/DistilBERT)
- **Facial Emotion**: DeepFace
- **Audio Analysis**: pyAudioAnalysis/librosa
- **Data Storage**: SQLite + JSON with optional FAISS/ChromaDB vector search
- **UI Options**: Desktop (PyQt5/Tkinter) or Web (FastAPI + JavaScript)

## 🚀 Getting Started

> Detailed setup instructions will be added once the initial development is completed.

## 📋 Project Status

This project is currently in the initial development phase. See the [ROADMAP.md](ROADMAP.md) for current progress and planned features.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Architecture

![System Architecture](docs/architecture_diagram.png)

*Architecture diagram will be added during development*

## 👥 Contributing

Contribution guidelines will be added as the project matures.