'''
📌 Purpose – Standalone script to run scene detection and TimeSformer-based visual analysis on a video, saving scene-level embeddings and action scores as JSON.
🔄 Latest Changes – Initial creation. Integrates modular backend pipeline for reproducible, Windows-native batch analysis.
⚙️ Key Logic – Runs scene detection, then visual analysis, and saves results for downstream use.
📂 Expected File Path – scripts/run_timesformer_analysis.py
🧠 Reasoning – Enables reproducible, automated, and modular batch processing for scientific video analysis.
'''

import os
import sys
import json
import argparse
from src.scene_detection.scene_detection import detect_scenes
from src.visual_analysis.timesformer_analysis import extract_scene_embeddings_and_actions

def main():
    parser = argparse.ArgumentParser(description="Run scene detection and TimeSformer visual analysis on a video.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, default='data/scene_detection_output', help='Directory to save outputs')
    parser.add_argument('--extract_actions', action='store_true', help='Extract action scores (default: True)')
    args = parser.parse_args()

    video_path = args.video_path
    output_dir = args.output_dir
    extract_actions = args.extract_actions

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    print(f"[INFO] Running scene detection on: {video_path}")
    scene_dicts = detect_scenes(video_path, output_dir, num_workers=1)
    scene_boundaries = [(d['start_time'], d['end_time']) for d in scene_dicts]
    print(f"[INFO] Detected {len(scene_boundaries)} scenes.")

    print(f"[INFO] Running TimeSformer visual analysis...")
    results = extract_scene_embeddings_and_actions(
        video_path,
        scene_boundaries,
        extract_actions=extract_actions
    )

    output_json = os.path.join(output_dir, f"{base}_timesformer_analysis.json")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved results to {output_json}")

if __name__ == "__main__":
    main() 