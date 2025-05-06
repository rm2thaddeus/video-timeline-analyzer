"""
📌 Purpose – Construct a DataFrame aligning scene boundaries, audio transcripts, and visual embeddings for each scene, ready for downstream analysis and Qdrant ingestion.
🔄 Latest Changes – Initial creation. Modular, Windows-native, and fully aligned with backend pipeline and DataFrame-centric architecture.
⚙️ Key Logic – Loads scene boundaries, Whisper segments, and visual embeddings; aligns by time; aggregates transcript and embeddings per scene; outputs a pandas DataFrame.
📂 Expected File Path – src/metadata/metadata_constructor.py
🧠 Reasoning – Ensures reproducible, modular, and extensible metadata construction for scientific video analysis pipelines.
"""

import os
import json
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


def load_scene_boundaries(scene_json_path: str) -> List[Tuple[float, float]]:
    """
    Load scene boundaries from a JSON file (list of [start, end] pairs or dicts with 'start_time'/'end_time').
    Args:
        scene_json_path: Path to scene detection output JSON.
    Returns:
        List of (start, end) tuples in seconds.
    """
    with open(scene_json_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    # Accepts either list of dicts or list of lists
    if isinstance(scenes, dict) and 'scenes' in scenes:
        scenes = scenes['scenes']
    if isinstance(scenes, list) and all(isinstance(s, dict) and 'start_time' in s and 'end_time' in s for s in scenes):
        return [(float(s['start_time']), float(s['end_time'])) for s in scenes]
    return [(float(s[0]), float(s[1])) if isinstance(s, (list, tuple)) else (float(s['start']), float(s['end'])) for s in scenes]


def load_whisper_segments(whisper_json_path: str) -> List[Dict[str, Any]]:
    """
    Load Whisper transcription segments from JSON.
    Args:
        whisper_json_path: Path to Whisper output JSON.
    Returns:
        List of segment dicts (with 'start', 'end', 'text', ...).
    """
    with open(whisper_json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    return segments


def load_visual_embeddings(embeddings_json_path: str) -> List[Dict[str, Any]]:
    """
    Load visual embeddings (and optionally action scores) from JSON.
    Args:
        embeddings_json_path: Path to visual embedding output JSON.
    Returns:
        List of dicts with 'scene', 'embedding', and optionally 'action_scores'.
    """
    with open(embeddings_json_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    return embeddings


def aggregate_transcript_for_scene(segments: List[Dict[str, Any]], start: float, end: float) -> str:
    """
    Aggregate Whisper transcript text for a given scene window.
    Args:
        segments: List of Whisper segments.
        start: Scene start time (s).
        end: Scene end time (s).
    Returns:
        Concatenated transcript string for the scene.
    """
    texts = [seg['text'].strip() for seg in segments if seg['start'] < end and seg['end'] > start]
    return ' '.join(texts)


def construct_metadata_dataframe(
    scene_json_path: str,
    whisper_json_path: str,
    embeddings_json_path: str,
    include_action_scores: bool = True
) -> pd.DataFrame:
    """
    Construct a DataFrame aligning scenes, audio, and visual embeddings.
    Args:
        scene_json_path: Path to scene detection output JSON.
        whisper_json_path: Path to Whisper output JSON.
        embeddings_json_path: Path to visual embedding output JSON.
        include_action_scores: If True, include action scores in DataFrame.
    Returns:
        pd.DataFrame with columns: ['scene_start', 'scene_end', 'transcript', 'embedding', 'action_scores' (optional)]
    """
    scenes = load_scene_boundaries(scene_json_path)
    segments = load_whisper_segments(whisper_json_path)
    embeddings = load_visual_embeddings(embeddings_json_path)
    rows = []
    for i, (start, end) in enumerate(scenes):
        transcript = aggregate_transcript_for_scene(segments, start, end)
        emb = embeddings[i]['embedding'] if i < len(embeddings) else None
        action_scores = embeddings[i].get('action_scores') if (include_action_scores and i < len(embeddings)) else None
        row = {
            'scene_start': start,
            'scene_end': end,
            'transcript': transcript,
            'embedding': emb
        }
        if include_action_scores:
            row['action_scores'] = action_scores
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


# Example usage (for testing, not for production):
if __name__ == "__main__":
    # Example file paths (replace with actual paths)
    scene_json = "../scene_detection/example_scenes.json"
    whisper_json = "../audio_analysis/example_whisper.json"
    embeddings_json = "../visual_analysis/example_embeddings.json"
    df = construct_metadata_dataframe(scene_json, whisper_json, embeddings_json)
    print(df.head()) 