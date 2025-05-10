"""
📌 Purpose – Modular scene detection interface for the backend pipeline. Uses only TransNet V2 (PyTorch) for scene detection, with parallelized frame extraction and robust output.
🔄 Latest Changes – Refactored for thread-safe frame extraction (per-worker VideoCapture), batched inference, and integrated profiling utility.
⚙️ Key Logic – Each worker opens its own VideoCapture, batches frames for inference, and profiles resource usage.
📂 Expected File Path – src/scene_detection/scene_detection.py
🧠 Reasoning – Ensures fast, accurate, and hardware-aware scene detection with robust profiling for optimization.
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
import psutil
import platform
from src.utils.profiling import profile_stage
import time

def _extract_frame_worker(args):
    video_path, idx, use_cuda = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if use_cuda:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_resized = cv2.cuda.resize(gpu_frame, (48, 27), interpolation=cv2.INTER_LINEAR)
            frame_resized = gpu_resized.download()
        except Exception as e:
            print(f"[SceneDetection][WARNING] CUDA resize failed, falling back to CPU: {e}")
            frame_resized = cv2.resize(frame, (48, 27))
    else:
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

    # Check for OpenCV CUDA
    use_cuda = False
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            use_cuda = True
            print("[SceneDetection] Using OpenCV CUDA for resizing.")
    except Exception:
        use_cuda = False

    # Parallel, thread-safe frame extraction with timing
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        frames = list(executor.map(_extract_frame_worker, [(video_path, idx, use_cuda) for idx in range(n_frames)]))
    t1 = time.time()
    frames = [f for f in frames if f is not None]
    frames = np.array(frames, dtype=np.uint8)
    n_frames = len(frames)
    if n_frames == 0:
        print(f"[TransNetV2-PyTorch] No frames extracted from {video_path}.")
        return []
    print(f"[SceneDetection][Profiling] Frame extraction: {n_frames} frames in {t1-t0:.2f} sec (FPS: {n_frames/(t1-t0):.2f})")

    # Load model and weights
    model = TransNetV2()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Batched inference with timing
    t2 = time.time()
    all_preds = []
    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = frames[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).unsqueeze(0).to(device)  # shape: [1, T, 27, 48, 3]
            single_frame_pred, _ = model(batch_tensor)
            preds = torch.sigmoid(single_frame_pred).cpu().numpy()[0, :, 0]
            all_preds.append(preds)
    t3 = time.time()
    predictions = np.concatenate(all_preds)
    print(f"[SceneDetection][Profiling] Model inference: {n_frames} frames in {t3-t2:.2f} sec (FPS: {n_frames/(t3-t2):.2f}), batch size: {batch_size}")

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

def detect_scenes(video_path: str, output_dir: str, batch_size: int = 100, threshold: float = 0.5, min_scene_len: int = 15, num_workers: int = 4, save_json: bool = True, save_csv: bool = False, **kwargs) -> List[Dict[str, Any]]:
    from src.utils.profiling import profile_stage
    with profile_stage("scene_detection"):
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

def detect_scenes_all_in_memory(video_path: str, output_dir: str, batch_size: int = 100, threshold: float = 0.5, min_scene_len: int = 15, save_json: bool = True, save_csv: bool = False, device: str = None, **kwargs):
    """
    Detect scenes by loading all frames into memory, then running inference in one batch (de-novo style).
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save outputs.
        batch_size (int): Batch size for inference (not used, all frames in one batch).
        threshold (float): Threshold for scene boundary detection.
        min_scene_len (int): Minimum length of a scene in frames.
        save_json (bool): Save results as JSON.
        save_csv (bool): Save results as CSV.
        device (str): 'cuda' or 'cpu'.
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), "transnetv2_repo", "inference-pytorch"))
    from transnetv2_pytorch import TransNetV2

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights_path = os.path.join(os.path.dirname(__file__), "transnetv2_repo", "inference-pytorch", "transnetv2-pytorch-weights.pth")
    model = TransNetV2()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load all frames into memory
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (48, 27))
        frames.append(frame_resized)
    cap.release()
    frames = np.array(frames, dtype=np.uint8)
    print(f"[AllInMemory] Loaded {len(frames)} frames into memory. RAM usage: {psutil.virtual_memory().percent}%")

    # Run inference in one batch
    with torch.no_grad():
        batch_tensor = torch.from_numpy(frames).unsqueeze(0).to(device)  # [1, T, 27, 48, 3]
        single_frame_pred, _ = model(batch_tensor)
        predictions = torch.sigmoid(single_frame_pred).cpu().numpy()[0, :, 0]

    # Post-process predictions to scene boundaries
    scenes = predictions_to_scenes(predictions, threshold=threshold, min_scene_len=min_scene_len)
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0] + "_allinmem"
    if save_json:
        json_path = os.path.join(output_dir, f"{base}_scenes.json")
        with open(json_path, "w") as f:
            import json
            json.dump([{'start_frame': int(s[0]), 'end_frame': int(s[1])} for s in scenes], f, indent=2)
    if save_csv:
        csv_path = os.path.join(output_dir, f"{base}_scenes.csv")
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["start_frame", "end_frame"])
            writer.writeheader()
            for s in scenes:
                writer.writerow({"start_frame": int(s[0]), "end_frame": int(s[1])})
    print(f"[AllInMemory] Detected {len(scenes)} scenes for {video_path}")
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
    parser.add_argument("--mode", type=str, default="threaded", choices=["threaded", "all_in_memory"], help="Scene detection mode.")
    args = parser.parse_args()

    if args.mode == "all_in_memory":
        detect_scenes_all_in_memory(
            video_path=args.video_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            save_json=args.save_json,
            save_csv=args.save_csv,
            device=args.device
        )
    else:
        # Existing threaded approach
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