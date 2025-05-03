"""
# TEMPORARY TEST SCRIPT (TO DELETE AFTER USE)
# This script tests the ingestion module on all videos in the data/ folder.
# It extracts metadata, frames, and audio, saving outputs to data/temp/.
# DO NOT COMMIT THIS FILE. DELETE AFTER RUNNING.
"""

import os
from ingestion import get_video_metadata, extract_frames, extract_audio

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
temp_dir = os.path.join(data_dir, "temp")
videos = [f for f in os.listdir(data_dir) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]

for video in videos:
    video_path = os.path.join(data_dir, video)
    print(f"Testing: {video}")
    metadata = get_video_metadata(video_path)
    print("Metadata:", metadata)
    frames_dir = os.path.join(temp_dir, "frames_" + os.path.splitext(video)[0])
    extract_frames(video_path, frames_dir, use_cuda=False, fps=1)
    print(f"Extracted frames to {frames_dir}")
    audio_path = os.path.join(temp_dir, "audio_" + os.path.splitext(video)[0] + ".wav")
    extract_audio(video_path, audio_path)
    print(f"Extracted audio to {audio_path}")
    print("-" * 40) 