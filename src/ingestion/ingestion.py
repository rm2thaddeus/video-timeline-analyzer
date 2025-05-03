"""
📌 Purpose – Ingest video files of any format, extract metadata, frames, and audio for downstream analysis.
🔄 Latest Changes – Initial module creation for robust, format-agnostic ingestion using ffmpeg.
⚙️ Key Logic – Uses ffmpeg for all extraction tasks; optionally leverages GPU acceleration if available.
📂 Expected File Path – src/ingestion/ingestion.py
🧠 Reasoning – Centralizes all video/audio loading and extraction, ensuring reproducibility and modularity.
"""

import subprocess
import os
from typing import Dict, Optional


def get_video_metadata(video_path: str) -> Dict[str, str]:
    """
    Extract basic metadata from a video file using ffmpeg.
    Args:
        video_path (str): Path to the video file.
    Returns:
        Dict[str, str]: Dictionary with metadata (duration, resolution, framerate, etc.)
    """
    import json
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=width,height,codec_name,avg_frame_rate',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    info = json.loads(result.stdout)
    # Parse and flatten info as needed
    metadata = {}
    if 'format' in info:
        metadata['duration'] = info['format'].get('duration', None)
    if 'streams' in info:
        for stream in info['streams']:
            if stream.get('codec_name') in ['h264', 'hevc', 'vp9', 'av1']:
                metadata['width'] = stream.get('width')
                metadata['height'] = stream.get('height')
                metadata['framerate'] = stream.get('avg_frame_rate')
    return metadata


def extract_frames(video_path: str, output_dir: str, use_cuda: bool = False, fps: Optional[int] = None) -> None:
    """
    Extract frames from a video file using ffmpeg.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        use_cuda (bool): If True, use GPU acceleration (if available).
        fps (Optional[int]): Frames per second to extract (None = all frames).
    """
    os.makedirs(output_dir, exist_ok=True)
    # Build ffmpeg command
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_path]
    if use_cuda:
        # Use hardware-accelerated decoding if available (NVIDIA)
        cmd = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-i', video_path]
    if fps:
        cmd += ['-vf', f'fps={fps}']
    cmd += [os.path.join(output_dir, 'frame_%06d.png')]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.decode()}")


def extract_audio(video_path: str, output_audio_path: str) -> None:
    """
    Extract the audio track from a video file using ffmpeg.
    Args:
        video_path (str): Path to the video file.
        output_audio_path (str): Path to save the extracted audio (e.g., .wav or .mp3).
    """
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_audio_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.decode()}")


# Example usage (for testing, not for production):
if __name__ == "__main__":
    video = "../data/example.mp4"
    print(get_video_metadata(video))
    extract_frames(video, "../data/frames", use_cuda=False, fps=1)
    extract_audio(video, "../data/example.wav") 