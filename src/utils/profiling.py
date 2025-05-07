"""
📌 Purpose – Utility for profiling CPU, RAM, and GPU usage in pipeline stages, using psutil and torch.profiler.
🔄 Latest Changes – Initial creation for pipeline-wide profiling integration.
⚙️ Key Logic – Context manager for timing, memory, and GPU profiling; outputs metrics as dict or JSON.
📂 Expected File Path – src/utils/profiling.py
🧠 Reasoning – Centralizes profiling logic for maintainability and reproducibility across all modules.
"""

import time
import json
import psutil
import torch
from contextlib import contextmanager
from typing import Optional, Dict, Any

@contextmanager
def profile_stage(stage_name: str, output_json: Optional[str] = None):
    process = psutil.Process()
    start_time = time.time()
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    gpu_mem_start = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    try:
        yield
    finally:
        end_time = time.time()
        end_mem = process.memory_info().rss / 1024 / 1024  # MB
        gpu_mem_end = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        metrics = {
            'stage': stage_name,
            'time_sec': end_time - start_time,
            'cpu_mem_mb': end_mem - start_mem,
            'gpu_mem_mb': gpu_mem_end - gpu_mem_start,
            'cpu_mem_peak_mb': psutil.virtual_memory().peak_wset / 1024 / 1024 if hasattr(psutil.virtual_memory(), 'peak_wset') else None
        }
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            print(f"[Profiling] {stage_name}: {metrics}") 