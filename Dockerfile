FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip opencv-python-headless

FROM base AS builder

WORKDIR /app

COPY requirements.txt /app/
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-cache-dir \
    torch==2.2.0+cu118 \
    torchvision==0.17.0+cu118 \
    torchaudio==2.2.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY src /app/src

FROM base

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /app/src /app/src

CMD ["python3", "src/pipeline/run_full_pipeline.py", "/app/data/video_submission.mp4"]
