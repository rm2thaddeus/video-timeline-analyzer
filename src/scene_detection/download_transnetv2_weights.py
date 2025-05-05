"""
📌 Purpose – Download TransNetV2 PyTorch weights from Hugging Face for reproducible setup.
🔄 Latest Changes – Initial creation for Windows/WSL2 compatibility.
⚙️ Key Logic – Downloads weights only if not already present.
📂 Expected File Path – src/scene_detection/download_transnetv2_weights.py
🧠 Reasoning – Ensures all users can easily obtain the required weights without manual steps or bloating the repository.
"""

import os
import requests

WEIGHTS_URL = "https://huggingface.co/ByteDance/shot2story/resolve/main/transnetv2-pytorch-weights.pth"
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "transnetv2_repo", "inference-pytorch", "transnetv2-pytorch-weights.pth")

def download_weights(url: str = WEIGHTS_URL, dest_path: str = WEIGHTS_PATH):
    if os.path.exists(dest_path):
        print(f"Weights already exist at {dest_path}. Skipping download.")
        return
    print(f"Downloading weights from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded weights to {dest_path}.")

if __name__ == "__main__":
    download_weights() 