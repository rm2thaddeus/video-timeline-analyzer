"""
📌 Purpose – Modular scene detection interface for the backend pipeline. Uses only TransNet V2 (PyTorch) for scene detection, with parallelized frame extraction and robust output.
🔄 Latest Changes – Refactored to remove TensorFlow/PySceneDetect, output both frame and time boundaries, parallelized frame extraction, and save results as CSV and JSON.
⚙️ Key Logic – Uses torch.cuda.is_available() to select GPU, parallelizes frame extraction, and outputs scene boundaries in frames and seconds.
📂 Expected File Path – src/scene_detection/scene_detection.py
🧠 Reasoning – Ensures fast, accurate, and hardware-aware scene detection in a modular, backend-first pipeline, with robust output for downstream use.
"""

import os
import sys
import numpy as np
import cv2
import torch
import json
import csv
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

def detect_scenes(video_path: str, output_dir: str, batch_size: int = 100, threshold: float = 0.5, min_scene_len: int = 15, num_workers: int = 4, save_json: bool = True, save_csv: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """
    Detect scenes in a video using TransNet V2 (PyTorch).
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save scene detection outputs.
        batch_size (int, optional): Number of frames per batch for inference. Default is 100.
        threshold (float, optional): Threshold for scene boundary detection.
        min_scene_len (int, optional): Minimum length of a scene in frames.
        num_workers (int, optional): Number of threads for parallel frame extraction.
        save_json (bool, optional): Whether to save results as JSON. Default is True.
        save_csv (bool, optional): Whether to save results as CSV. Default is False.
        **kwargs: Additional arguments for backend-specific options.
    Returns:
        List of dicts with scene boundaries (frame and time).
    """
    scenes = detect_scenes_transnetv2_pytorch(
        video_path, output_dir, batch_size=batch_size, threshold=threshold, min_scene_len=min_scene_len, num_workers=num_workers
    )
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    if save_json:
        json_path = os.path.join(output_dir, f"{base}_scenes.json")
        with open(json_path, "w") as f:
            json.dump(scenes, f, indent=2)
    if save_csv:
        csv_path = os.path.join(output_dir, f"{base}_scenes.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["start_frame", "end_frame", "start_time", "end_time"])
            writer.writeheader()
            for scene in scenes:
                writer.writerow(scene)
    print(f"[SceneDetection] Detected {len(scenes)} scenes for {video_path}")
    return scenes

def _extract_frame(args):
    cap, idx = args
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    frame_resized = cv2.resize(frame, (48, 27))
    return frame_resized

def _get_fps_and_nframes(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n_frames

def detect_scenes_transnetv2_pytorch(video_path: str, output_dir: str, batch_size: int = 100, threshold: float = 0.5, min_scene_len: int = 15, num_workers: int = 4, **kwargs) -> List[Dict[str, Any]]:
    """
    Detect scenes using TransNet V2 (PyTorch, GPU-accelerated).
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save outputs.
        batch_size (int): Number of frames per batch for inference.
        threshold (float): Threshold for scene boundary detection.
        min_scene_len (int): Minimum length of a scene in frames.
        num_workers (int): Number of threads for parallel frame extraction.
        **kwargs: Additional options for TransNet V2.
    Returns:
        List of dicts with scene boundaries (frame and time).
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), "transnetv2_repo", "inference-pytorch"))
    from transnetv2_pytorch import TransNetV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(os.path.dirname(__file__), "transnetv2_repo", "inference-pytorch", "transnetv2-pytorch-weights.pth")
    if not os.path.exists(weights_path):
        import requests
        url = "https://huggingface.co/ByteDance/shot2story/resolve/main/transnetv2-pytorch-weights.pth"
        print(f"[TransNetV2-PyTorch] Downloading weights from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(weights_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[TransNetV2-PyTorch] Downloaded weights to {weights_path}.")

    # Get video properties
    fps, n_frames = _get_fps_and_nframes(video_path)
    if n_frames == 0:
        print(f"[TransNetV2-PyTorch] No frames found in {video_path}.")
        return []

    # Parallel frame extraction
    cap = cv2.VideoCapture(video_path)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        frames = list(executor.map(_extract_frame, [(cap, idx) for idx in range(n_frames)]))
    cap.release()
    frames = [f for f in frames if f is not None]
    frames = np.array(frames, dtype=np.uint8)
    n_frames = len(frames)
    if n_frames == 0:
        print(f"[TransNetV2-PyTorch] No frames extracted from {video_path}.")
        return []

    # Load model and weights
    model = TransNetV2()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Batch inference
    all_preds = []
    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = frames[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).unsqueeze(0).to(device)  # shape: [1, T, 27, 48, 3]
            single_frame_pred, _ = model(batch_tensor)
            preds = torch.sigmoid(single_frame_pred).cpu().numpy()[0, :, 0]
            all_preds.append(preds)
    predictions = np.concatenate(all_preds)

    # Post-process predictions to scene boundaries
    scenes = predictions_to_scenes(predictions, threshold=threshold, min_scene_len=min_scene_len)

    # Convert to frame and time boundaries
    scene_dicts = []
    for start, end in scenes:
        start_time = start / fps
        end_time = end / fps
        scene_dicts.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "start_time": float(start_time),
            "end_time": float(end_time)
        })
    return scene_dicts

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5, min_scene_len: int = 15) -> List[tuple]:
    """
    Convert model predictions to scene boundaries.
    Args:
        predictions (np.ndarray): 1D array of probabilities per frame.
        threshold (float): Probability threshold for scene boundary.
        min_scene_len (int): Minimum length of a scene in frames.
    Returns:
        List of (start_frame, end_frame) tuples.
    """
    boundaries = np.where(predictions > threshold)[0]
    if len(boundaries) == 0:
        return [(0, len(predictions)-1)]
    scenes = []
    prev = 0
    for b in boundaries:
        if b - prev >= min_scene_len:
            scenes.append((prev, b-1))
            prev = b
    if prev < len(predictions)-1:
        scenes.append((prev, len(predictions)-1))
    return scenes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scene detection using TransNet V2 (PyTorch)")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save outputs.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for scene boundary detection.")
    parser.add_argument("--min_scene_len", type=int, default=15, help="Minimum scene length in frames.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for frame extraction.")
    parser.add_argument("--save_json", type=lambda x: (str(x).lower() == 'true'), default=True, help="Save results as JSON.")
    parser.add_argument("--save_csv", type=lambda x: (str(x).lower() == 'true'), default=False, help="Save results as CSV.")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", None], help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    # Set device if specified
    if args.device:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[SceneDetection] CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        else:
            device = args.device
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    detect_scenes(
        video_path=args.video_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
        num_workers=args.num_workers,
        save_json=args.save_json,
        save_csv=args.save_csv,
        device=device
    ) 