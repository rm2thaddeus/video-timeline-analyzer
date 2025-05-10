"""
📌 Purpose – Modular audio transcription using Whisper-small (default), with JSON output (for DataFrame construction) and optional SRT/WAV. Auto-estimates max workers for GPU. Highly configurable for scientific pipelines.
🔄 Latest Changes – Refactored for single-model loading, in-memory chunking (FFmpeg pipe), and integrated profiling utility. Added class-based interface for maintainability.
⚙️ Key Logic – Loads Whisper model once, extracts audio chunks in memory, transcribes in batch, deduplicates segments, and profiles resource usage.
📂 Expected File Path – src/audio_analysis/whisper_transcribe.py
🧠 Reasoning – Maximizes efficiency, minimizes disk I/O, and enables robust profiling for scientific reproducibility.
"""

import os
import json
from typing import List, Dict, Any, Optional
from io import BytesIO
import math
import ffmpeg
import torch
import shutil
from src.utils.profiling import profile_stage

class WhisperTranscriber:
    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        import whisper
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)
        self.model.eval()

    def extract_audio_chunk(self, video_path: str, start: float, duration: float) -> bytes:
        out, _ = (
            ffmpeg
            .input(video_path, ss=start, t=duration)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .run(capture_stdout=True, capture_stderr=True)
        )
        return out

    def transcribe_chunk(self, audio_bytes: bytes, chunk_start: float, language: Optional[str] = None) -> List[Dict[str, Any]]:
        import whisper
        import tempfile
        import os
        # Write bytes to a temp file and pass the path directly to Whisper
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()
            result = self.model.transcribe(tmp.name, language=language, verbose=False)
        finally:
            os.unlink(tmp.name)
        segments = []
        for seg in result["segments"]:
            seg = seg.copy()
            seg["start"] += chunk_start
            seg["end"] += chunk_start
            segments.append(seg)
        return segments

    def transcribe_video(self, video_path: str, chunk_length: float = 30.0, overlap: float = 2.0, language: Optional[str] = None) -> List[Dict[str, Any]]:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        step = chunk_length - overlap
        n_chunks = max(1, math.ceil((duration - overlap) / step))
        all_segments = []
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_length, duration)
            audio_bytes = self.extract_audio_chunk(video_path, start, chunk_length)
            segments = self.transcribe_chunk(audio_bytes, start, language=language)
            all_segments.extend(segments)
        return self._deduplicate_segments(all_segments)

    @staticmethod
    def _deduplicate_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for seg in segments:
            key = (round(seg["start"], 2), seg["text"].strip())
            if key not in seen:
                deduped.append(seg)
                seen.add(key)
        deduped.sort(key=lambda s: s["start"])
        return deduped

# --- SRT and Output Utilities (unchanged) ---
def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
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
    output_prefix: str,
    output_json: bool = True,
    output_srt: bool = False
) -> None:
    if output_srt:
        srt_str = segments_to_srt(segments)
        with open(f"{output_prefix}.srt", "w", encoding="utf-8") as f:
            f.write(srt_str)
    if output_json:
        with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

# --- Main Entrypoint with Profiling ---
def transcribe_and_save(
    video_path: str,
    output_dir: str,
    model_size: str = "small",
    device: Optional[str] = None,
    language: Optional[str] = None,
    chunk_length: float = 30.0,
    overlap: float = 2.0,
    output_json: bool = True,
    output_srt: bool = False,
    output_wav: bool = False
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_prefix = os.path.join(output_dir, base + "_whisper")
    audio_path = os.path.join(output_dir, base + "_audio.wav")
    if output_wav:
        # Use original extract_full_audio for full WAV
        import ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
    transcriber = WhisperTranscriber(model_size=model_size, device=device)
    with profile_stage("whisper_transcription"):
        segments = transcriber.transcribe_video(
            video_path, chunk_length=chunk_length, overlap=overlap, language=language
        )
    save_transcription_outputs(segments, output_prefix, output_json=output_json, output_srt=output_srt)
    return segments

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Transcribe audio from video using Whisper with in-memory chunking and profiling.")
    parser.add_argument("video_path", type=str, help="Path to input video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--model_size", type=str, default="small", help="Whisper model size (default: small)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--language", type=str, default=None, help="Language code for forced transcription.")
    parser.add_argument("--chunk_length", type=float, default=30.0, help="Length of each chunk in seconds.")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap between chunks in seconds.")
    parser.add_argument("--output_json", type=lambda x: (str(x).lower() == 'true'), default=True, help="Save JSON output (default: True).")
    parser.add_argument("--output_srt", type=lambda x: (str(x).lower() == 'true'), default=False, help="Save SRT output (default: False).")
    parser.add_argument("--output_wav", type=lambda x: (str(x).lower() == 'true'), default=False, help="Save full audio WAV (default: False).")
    args = parser.parse_args()
    try:
        segments = transcribe_and_save(
            video_path=args.video_path,
            output_dir=args.output_dir,
            model_size=args.model_size,
            device=args.device,
            language=args.language,
            chunk_length=args.chunk_length,
            overlap=args.overlap,
            output_json=args.output_json,
            output_srt=args.output_srt,
            output_wav=args.output_wav
        )
        base = os.path.splitext(os.path.basename(args.video_path))[0]
        output_prefix = os.path.join(args.output_dir, base + "_whisper.json")
        print(f"[WhisperTranscribe] Success. Output: {output_prefix}")
    except Exception as e:
        print(f"[WhisperTranscribe] ERROR: {e}", file=sys.stderr)
        sys.exit(1) 