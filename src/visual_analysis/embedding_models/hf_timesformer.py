"""
📌 Purpose – Extract video embeddings using Hugging Face's TimeSformer model for downstream analysis (e.g., search, tagging, similarity).
🔄 Latest Changes – Manual preprocessing added to remove dependency on TimesformerImageProcessor.
⚙️ Key Logic – Loads TimeSformer from Hugging Face, manually preprocesses frames, computes embeddings.
📂 Expected File Path – src/visual_analysis/embedding_models/hf_timesformer.py
🧠 Reasoning – Ensures robust, maintainable embedding extraction regardless of Hugging Face API changes.
"""

import torch
from transformers import TimesformerModel
import cv2
import numpy as np
from typing import List, Optional

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class TimeSformerVideoEmbedder:
    """
    Extracts video embeddings using Hugging Face's TimeSformer with manual preprocessing.
    
    Args:
        model_name (str): Hugging Face model repo id (default: 'facebook/timesformer-base-finetuned-k400').
        device (str): 'cuda' or 'cpu'. If None, auto-detects.
    """
    def __init__(self, model_name: str = 'facebook/timesformer-base-finetuned-k400', device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TimesformerModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """
        Extracts evenly spaced frames from a video file.
        Args:
            video_path (str): Path to video file.
            num_frames (int): Number of frames to extract.
        Returns:
            List[np.ndarray]: List of frames as numpy arrays (BGR format).
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if len(frames) < num_frames:
            raise ValueError(f"Could only extract {len(frames)} frames from {video_path}")
        return frames

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Manually preprocess frames for TimeSformer: resize, convert, normalize, stack.
        Args:
            frames (List[np.ndarray]): List of frames (BGR, uint8).
        Returns:
            torch.Tensor: Shape [1, num_frames, 3, 224, 224]
        """
        processed = []
        for frame in frames:
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to float32 and scale to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            # Normalize
            frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
            # HWC to CHW
            frame = np.transpose(frame, (2, 0, 1))
            processed.append(frame)
        # Stack to [num_frames, 3, 224, 224]
        frames_np = np.stack(processed, axis=0)
        # Add batch dimension and permute to [1, num_frames, 3, 224, 224]
        frames_tensor = torch.from_numpy(frames_np).unsqueeze(0).float()
        return frames_tensor

    def get_video_embedding(self, video_path: str, num_frames: int = 8) -> np.ndarray:
        """
        Extracts a video embedding from the given video file.
        Args:
            video_path (str): Path to video file.
            num_frames (int): Number of frames to use for embedding.
        Returns:
            np.ndarray: Embedding vector (1D array).
        """
        frames = self.extract_frames(video_path, num_frames=num_frames)
        inputs = self.preprocess_frames(frames).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=inputs)
            # Use the [CLS] token embedding as the video representation
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy().squeeze()
        return embedding

# Example usage
def example_extract_embedding():
    video_path = "path/to/your/video.mp4"
    embedder = TimeSformerVideoEmbedder()
    embedding = embedder.get_video_embedding(video_path)
    print(f"Embedding shape: {embedding.shape}")
    print(embedding)

if __name__ == "__main__":
    example_extract_embedding() 