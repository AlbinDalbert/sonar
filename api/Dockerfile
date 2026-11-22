# syntax=docker/dockerfile:1.7
# torch's cu124 wheels ship their own CUDA/cuDNN (nvidia-* pip packages) and the
# nvidia runtime mounts the host driver, so a CUDA base image would only duplicate them.
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    QWEN_LOCAL_FILES_ONLY=1 \
    QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    QWEN_DEVICE=cuda:0 \
    QWEN_DTYPE=bfloat16 \
    QWEN_ATTN_IMPL=sdpa

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg libsndfile1 sox git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /app/requirements.txt && \
    python3 -m pip install --no-deps qwen-tts==0.1.1

COPY app /app/app

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8888/healthz || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
