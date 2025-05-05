"""
📌 Purpose – Modular audio transcription using Whisper-medium, with dual output: SRT (subtitle) and JSON (granular, DataFrame-ready). Now also outputs the full extracted audio as WAV for downstream use, and uses overlapping chunking for robust segment boundaries.
🔄 Latest Changes – Added process-based parallel chunk transcription for full GPU utilization; 'num_workers' parameter controls parallelism.
⚙️ Key Logic – Extracts audio, splits into overlapping chunks (default 30s, 2s overlap), transcribes each in parallel with Whisper-medium, deduplicates segments, and saves all outputs (audio, SRT, JSON).
📂 Expected File Path – src/audio_analysis/whisper_transcribe.py
🧠 Reasoning – Ensures both human usability (SRT), scientific, granular metadata (JSON), and reusable audio for further analysis or review, with robust handling of chunk boundaries and improved speed via GPU parallelism.
"""

import os
import json
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_full_audio(
    video_path: str,
    output_path: str
) -> None:
    """
    Extract the full audio track from a video and save as WAV (16kHz mono, PCM).
    Args:
        video_path (str): Path to video file.
        output_path (str): Path to save the extracted audio (WAV).
    """
    import ffmpeg
    (
        ffmpeg
        .input(video_path)
        .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )

def extract_audio_chunks(
    video_path: str,
    chunk_length: float = 30.0,
    overlap: float = 2.0,
    tmp_dir: str = "_audio_chunks_tmp"
) -> List[Dict[str, float]]:
    """
    Extract audio from video and split into overlapping chunks.
    Args:
        video_path (str): Path to video file.
        chunk_length (float): Length of each chunk in seconds.
        overlap (float): Overlap between chunks in seconds.
        tmp_dir (str): Directory to store temporary audio chunks.
    Returns:
        List of dicts: [{'path': chunk_path, 'start': float, 'end': float}]
    """
    import ffmpeg
    import math
    import shutil
    # Clean/create temp dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    # Get audio duration
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    step = chunk_length - overlap
    n_chunks = max(1, math.ceil((duration - overlap) / step))
    chunk_infos = []
    for i in range(n_chunks):
        start = i * step
        end = min(start + chunk_length, duration)
        out_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
        (
            ffmpeg
            .input(video_path, ss=start, t=chunk_length)
            .output(out_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        chunk_infos.append({'path': out_path, 'start': start, 'end': end})
    return chunk_infos

def _transcribe_chunk_worker(args):
    chunk_path, chunk_start, model_size, device, language = args
    import whisper
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(chunk_path, language=language, verbose=False)
    segments = []
    for seg in result["segments"]:
        seg = seg.copy()
        seg["start"] += chunk_start
        seg["end"] += chunk_start
        segments.append(seg)
    return segments

def transcribe_audio_whisper_chunked(
    video_path: str,
    model_size: str = "medium",
    device: Optional[str] = None,
    language: Optional[str] = None,
    chunk_length: float = 30.0,
    overlap: float = 2.0,
    num_workers: int = 2
) -> List[Dict[str, Any]]:
    """
    Transcribe audio from a video file using Whisper, with overlapping chunking and parallel processing for robust boundaries and fast GPU utilization.
    Args:
        video_path (str): Path to the video file.
        model_size (str): Whisper model size (default: "medium").
        device (str, optional): 'cuda' or 'cpu'. If None, auto-detect.
        language (str, optional): Language code for forced transcription.
        chunk_length (float): Length of each chunk in seconds.
        overlap (float): Overlap between chunks in seconds.
        num_workers (int): Number of parallel processes for chunk transcription (default: 2). Too many may cause GPU OOM.
    Returns:
        List of segments, each a dict with start, end, text.
    Note:
        Output is segment-level (not word-level) for speed and practicality.
    """
    import torch
    import shutil
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chunk_infos = extract_audio_chunks(video_path, chunk_length=chunk_length, overlap=overlap)
    all_segments = []
    # Prepare arguments for each chunk
    chunk_args = [(chunk['path'], chunk['start'], model_size, device, language) for chunk in chunk_infos]
    # Parallel transcription
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_transcribe_chunk_worker, arg) for arg in chunk_args]
        for future in as_completed(futures):
            all_segments.extend(future.result())
    # Deduplicate: keep only the first occurrence of each segment (by start time and text)
    deduped = []
    seen = set()
    for seg in all_segments:
        key = (round(seg["start"], 2), seg["text"].strip())
        if key not in seen:
            deduped.append(seg)
            seen.add(key)
    # Clean up temp files
    if chunk_infos:
        shutil.rmtree(os.path.dirname(chunk_infos[0]['path']))
    # Sort deduped segments by start time
    deduped.sort(key=lambda s: s["start"])
    return deduped

def transcribe_audio_whisper(
    video_path: str,
    model_size: str = "medium",
    device: Optional[str] = None,
    language: Optional[str] = None,
    chunk_length: float = 30.0,
    overlap: float = 2.0,
    num_workers: int = 2
) -> List[Dict[str, Any]]:
    """
    Transcribe audio from a video file using Whisper, with overlapping chunking and parallel processing for robust boundaries and fast GPU utilization.
    Args:
        video_path (str): Path to the video file.
        model_size (str): Whisper model size (default: "medium").
        device (str, optional): 'cuda' or 'cpu'. If None, auto-detect.
        language (str, optional): Language code for forced transcription.
        chunk_length (float): Length of each chunk in seconds.
        overlap (float): Overlap between chunks in seconds.
        num_workers (int): Number of parallel processes for chunk transcription (default: 2). Too many may cause GPU OOM.
    Returns:
        List of segments, each a dict with start, end, text.
    Note:
        Output is segment-level (not word-level) for speed and practicality.
    """
    return transcribe_audio_whisper_chunked(video_path, model_size, device, language, chunk_length, overlap, num_workers)

def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Convert Whisper segments to SRT format.
    Args:
        segments (list): List of segment dicts from Whisper.
    Returns:
        str: SRT formatted string.
    """
    def format_timestamp(ts: float) -> str:
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        text = seg['text'].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)

def save_transcription_outputs(
    segments: List[Dict[str, Any]],
    output_prefix: str
) -> None:
    """
    Save both SRT and JSON outputs for a transcription.
    Args:
        segments (list): List of segment dicts from Whisper.
        output_prefix (str): Path prefix for output files (no extension).
    """
    # Save SRT
    srt_str = segments_to_srt(segments)
    with open(f"{output_prefix}.srt", "w", encoding="utf-8") as f:
        f.write(srt_str)
    # Save JSON
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

def transcribe_and_save(
    video_path: str,
    output_dir: str,
    model_size: str = "medium",
    device: Optional[str] = None,
    language: Optional[str] = None,
    chunk_length: float = 30.0,
    overlap: float = 2.0,
    num_workers: int = 2
) -> List[Dict[str, Any]]:
    """
    Transcribe a video and save SRT, JSON, and full audio outputs.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save outputs.
        model_size (str): Whisper model size (default: "medium").
        device (str, optional): 'cuda' or 'cpu'. If None, auto-detect.
        language (str, optional): Language code for forced transcription.
        chunk_length (float): Length of each chunk in seconds.
        overlap (float): Overlap between chunks in seconds.
        num_workers (int): Number of parallel processes for chunk transcription (default: 2). Too many may cause GPU OOM.
    Returns:
        List of segment dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_prefix = os.path.join(output_dir, base + "_whisper")
    audio_path = os.path.join(output_dir, base + "_audio.wav")
    # Extract and save full audio
    extract_full_audio(video_path, audio_path)
    # Transcribe and save SRT/JSON
    segments = transcribe_audio_whisper(video_path, model_size, device, language, chunk_length, overlap, num_workers)
    save_transcription_outputs(segments, output_prefix)
    return segments 