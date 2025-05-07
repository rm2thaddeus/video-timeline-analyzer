"""
📌 Purpose – Extract scene-level embeddings from video using Hugging Face TimeSformer for the Video Timeline Analyzer pipeline.
🔄 Latest Changes – Refactored for batched scene embedding extraction, half-precision inference, async CUDA streams, and integrated profiling utility.
⚙️ Key Logic – Batches scenes, uses half-precision, leverages CUDA streams, and profiles resource usage for each embedding extraction run.
📂 Expected File Path – src/visual_analysis/timesformer_analysis.py
🧠 Reasoning – Maximizes throughput and reproducibility for visual analysis, ready for DataFrame alignment and Qdrant ingestion.
"""

import os
import json
import torch
import cv2
import numpy as np
from transformers import TimesformerModel
from typing import List, Tuple, Dict, Any
from src.utils.profiling import profile_stage

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class TimeSformerSceneEmbedder:
    def __init__(self, model_name: str = 'facebook/timesformer-base-finetuned-k400', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TimesformerModel.from_pretrained(model_name)
        if self.device == 'cuda':
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.stream = torch.cuda.Stream() if self.device == 'cuda' else None

    def extract_frames_for_scene(self, video_path: str, start_time: float, end_time: float, num_frames: int = 8) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        # Pad if not enough frames
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.float32))
        return frames

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
            frame = np.transpose(frame, (2, 0, 1))
            processed.append(frame)
        frames_np = np.stack(processed, axis=0)
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        frames_tensor = torch.from_numpy(frames_np).unsqueeze(0).to(dtype)  # [1, num_frames, 3, 224, 224]
        return frames_tensor

    def get_scene_embeddings_batch(self, video_path: str, scene_boundaries: List[Tuple[float, float]], num_frames: int = 8, batch_size: int = 4) -> List[np.ndarray]:
        embeddings = []
        for i in range(0, len(scene_boundaries), batch_size):
            batch_scenes = scene_boundaries[i:i+batch_size]
            batch_inputs = []
            for (start, end) in batch_scenes:
                frames = self.extract_frames_for_scene(video_path, start, end, num_frames=num_frames)
                batch_inputs.append(self.preprocess_frames(frames))
            inputs = torch.cat(batch_inputs, dim=0).to(self.device)  # [B, num_frames, 3, 224, 224]
            with torch.no_grad():
                if self.stream:
                    with torch.cuda.stream(self.stream):
                        outputs = self.model(pixel_values=inputs)
                        batch_emb = outputs.last_hidden_state[:, 0].cpu().numpy()
                else:
                    outputs = self.model(pixel_values=inputs)
                    batch_emb = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.extend(batch_emb)
        return embeddings

def extract_scene_embeddings(
    video_path: str,
    scene_boundaries: List[Tuple[float, float]],
    model_name: str = 'facebook/timesformer-base-finetuned-k400',
    device: str = None,
    batch_size: int = 4
) -> List[Dict[str, Any]]:
    embedder = TimeSformerSceneEmbedder(model_name=model_name, device=device)
    with profile_stage("timesformer_embedding"):
        embeddings = embedder.get_scene_embeddings_batch(video_path, scene_boundaries, batch_size=batch_size)
    results = []
    for (start, end), emb in zip(scene_boundaries, embeddings):
        results.append({
            'scene': (start, end),
            'embedding': emb.tolist()
        })
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract scene-level embeddings using Hugging Face TimeSformer.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--scenes", type=str, required=True, help="Path to scene boundaries JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument("--model_name", type=str, default='facebook/timesformer-base-finetuned-k400', help="Hugging Face model name.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for scene embedding extraction.")
    args = parser.parse_args()

    with open(args.scenes, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    if isinstance(scenes, dict) and 'scenes' in scenes:
        scenes = scenes['scenes']
    if isinstance(scenes, list) and all(isinstance(s, dict) and 'start_time' in s and 'end_time' in s for s in scenes):
        scene_boundaries = [(float(s['start_time']), float(s['end_time'])) for s in scenes]
    else:
        scene_boundaries = [(float(s[0]), float(s[1])) if isinstance(s, (list, tuple)) else (float(s['start']), float(s['end'])) for s in scenes]

    results = extract_scene_embeddings(
        args.video,
        scene_boundaries,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2) 