"""
# TEMPORARY TEST SCRIPT (TO DELETE AFTER USE)
# This script tests the modular scene detection interface on all videos in the data/ folder.
# It calls detect_scenes and prints the result for each video, saving outputs to data/temp/.
# DO NOT COMMIT THIS FILE. DELETE AFTER RUNNING.
"""

import os
from scene_detection import detect_scenes_transnetv2_pytorch

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
temp_dir = os.path.join(data_dir, "temp")
videos = [f for f in os.listdir(data_dir) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]

for video in videos:
    video_path = os.path.join(data_dir, video)
    print(f"Testing scene detection (PyTorch): {video}")
    scenes = detect_scenes_transnetv2_pytorch(video_path, temp_dir)
    print(f"Detected scenes for {video}: {scenes}")
    print("-" * 40) 