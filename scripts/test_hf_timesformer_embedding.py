"""
📌 Purpose – Test Hugging Face TimeSformer embedding extraction on a sample video.
🔄 Latest Changes – Initial test script for modular video embedding extraction.
⚙️ Key Logic – Loads the video, extracts frames, computes embedding, prints shape and summary statistics.
📂 Expected File Path – scripts/test_hf_timesformer_embedding.py
🧠 Reasoning – Provides a simple, reproducible way to verify the new embedding module and dependencies.
"""

import sys
import numpy as np
from pathlib import Path

# Import the embedder from the project
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'visual_analysis' / 'embedding_models'))
from hf_timesformer import TimeSformerVideoEmbedder

VIDEO_PATH = r"C:/Users/aitor/Video_Timeline/data/video submission.mp4"

if __name__ == "__main__":
    print(f"Testing TimeSformer embedding extraction on: {VIDEO_PATH}")
    try:
        embedder = TimeSformerVideoEmbedder()
        embedding = embedder.get_video_embedding(VIDEO_PATH)
        print(f"\nEmbedding shape: {embedding.shape}")
        print(f"Embedding mean: {np.mean(embedding):.4f}, std: {np.std(embedding):.4f}")
        print(f"First 10 values: {embedding[:10]}")
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        sys.exit(1) 