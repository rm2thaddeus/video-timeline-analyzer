"""
📌 Purpose: Extract metadata from video files
🔄 Latest Changes: Initial creation
⚙️ Key Logic: Use ffprobe to extract video and audio metadata
📂 Expected File Path: test_pipeline/processors/metadata_extractor.py
🧠 Reasoning: Separate metadata extraction for clean modular design
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("video_pipeline.metadata_extractor")

def extract_metadata(video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract metadata from a video file using ffprobe.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the metadata JSON file (optional)
        
    Returns:
        Dictionary containing video metadata
    """
    video_path = Path(video_path)
    video_id = video_path.stem
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Extracting metadata from {video_path}")
    
    try:
        # Run ffprobe to get video metadata in JSON format
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-print_format", "json",
            "-show_format", 
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        ffprobe_data = json.loads(result.stdout)
        
        # Process and organize the metadata
        metadata = {
            "video_id": video_id,
            "filename": video_path.name,
            "format": ffprobe_data.get("format", {}),
            "streams": ffprobe_data.get("streams", []),
            "video_stream": None,
            "audio_stream": None,
            "duration_seconds": float(ffprobe_data.get("format", {}).get("duration", 0)),
            "file_size_bytes": int(ffprobe_data.get("format", {}).get("size", 0)),
        }
        
        # Extract specific video and audio stream info
        for stream in ffprobe_data.get("streams", []):
            if stream.get("codec_type") == "video" and metadata["video_stream"] is None:
                metadata["video_stream"] = stream
                metadata["width"] = int(stream.get("width", 0))
                metadata["height"] = int(stream.get("height", 0))
                metadata["frame_rate"] = eval(stream.get("r_frame_rate", "0/1"))
                metadata["total_frames"] = int(stream.get("nb_frames", 0)) or int(metadata["duration_seconds"] * metadata["frame_rate"])
                
            elif stream.get("codec_type") == "audio" and metadata["audio_stream"] is None:
                metadata["audio_stream"] = stream
                metadata["audio_codec"] = stream.get("codec_name")
                metadata["audio_channels"] = int(stream.get("channels", 0))
                metadata["audio_sample_rate"] = int(stream.get("sample_rate", 0))
        
        # Save metadata to JSON file if output_dir is provided
        if output_dir:
            output_file = output_dir / f"{video_id}_metadata.json"
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {output_file}")
        
        return metadata
        
    except subprocess.SubprocessError as e:
        logger.error(f"FFprobe error: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during metadata extraction: {e}")
        raise 