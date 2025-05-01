"""
📌 Purpose – Modular scene detection interface for the backend pipeline. Automatically selects TransNet V2 (if CUDA is available) or PySceneDetect (CPU fallback).
🔄 Latest Changes – Initial creation of modular interface with backend selection logic.
⚙️ Key Logic – Uses torch.cuda.is_available() to select the best backend for scene detection.
📂 Expected File Path – src/scene_detection/scene_detection.py
🧠 Reasoning – Ensures fast, accurate, and hardware-aware scene detection in a modular, backend-first pipeline.
"""

def detect_scenes(video_path, output_dir, **kwargs):
    """
    Detect scenes in a video using the best available backend.
    Uses TransNet V2 if CUDA is available, else falls back to PySceneDetect.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save scene detection outputs.
        **kwargs: Additional arguments for backend-specific options.
    Returns:
        List of scene boundaries (to be defined in implementation).
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        print("[SceneDetection] CUDA detected: using TransNet V2 for scene detection.")
        return detect_scenes_transnetv2(video_path, output_dir, **kwargs)
    else:
        print("[SceneDetection] CUDA not detected: using PySceneDetect for scene detection.")
        return detect_scenes_pyscenedetect(video_path, output_dir, **kwargs)


def detect_scenes_transnetv2(video_path, output_dir, **kwargs):
    """
    Detect scenes using TransNet V2 (deep learning, GPU-accelerated).
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save outputs.
        **kwargs: Additional options for TransNet V2.
    Returns:
        List of (start_frame, end_frame) tuples for each detected scene.
    """
    import sys
    import os
    import numpy as np
    import cv2
    # Add the inference directory to sys.path
    sys.path.append(os.path.join(os.path.dirname(__file__), "transnetv2_repo", "inference"))
    from transnetv2 import TransNetV2

    # Load model (weights are downloaded automatically by TransNetV2)
    model = TransNetV2()

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to 48x27 as required by TransNetV2
        frame_resized = cv2.resize(frame, (48, 27))
        frames.append(frame_resized)
    cap.release()
    frames = np.array(frames)

    # Run model
    predictions, _ = model.predict_frames(frames)
    scenes = model.predictions_to_scenes(predictions)

    # Optionally save scene boundaries to a file in output_dir
    os.makedirs(output_dir, exist_ok=True)
    scene_file = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_scenes.txt")
    with open(scene_file, "w") as f:
        for start, end in scenes:
            f.write(f"{start},{end}\n")

    print(f"[TransNetV2] Detected {len(scenes)} scenes for {video_path}")
    return scenes


def detect_scenes_pyscenedetect(video_path, output_dir, **kwargs):
    """
    Detect scenes using PySceneDetect (CPU, robust fallback).
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save outputs.
        **kwargs: Additional options for PySceneDetect.
    Returns:
        List of (start_frame, end_frame) tuples for each detected scene.
    """
    import os
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector

    # Create video manager and scene manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    # Start scene detection
    video_manager.set_downscale_factor(1)
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    # Convert scene_list to (start_frame, end_frame) tuples
    scenes = []
    for start_time, end_time in scene_list:
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames() - 1  # end is exclusive
        scenes.append((start_frame, end_frame))

    # Save scene boundaries to a file in output_dir
    os.makedirs(output_dir, exist_ok=True)
    scene_file = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_scenes.txt")
    with open(scene_file, "w") as f:
        for start, end in scenes:
            f.write(f"{start},{end}\n")

    print(f"[PySceneDetect] Detected {len(scenes)} scenes for {video_path}")
    return scenes 