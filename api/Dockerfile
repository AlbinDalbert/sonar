# syntax=docker/dockerfile:1.7
# torch's cu124 wheels ship their own CUDA/cuDNN (nvidia-* pip packages) and the
# nvidia runtime mounts the host driver, so a CUDA base image would only duplicate them.
# Pinned to a digest so routine Ubuntu rebuilds don't bust the ~3 GB PyTorch layer cache.
ARG BASE_IMAGE=ubuntu:22.04@sha256:f64b2ffc2f10ab8f8505aa56b2664bd846cbb9df87c7e2532e30f7667021c0e6
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg libsndfile1 sox git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch + CUDA wheels in their own layer so changing app deps
# in requirements.txt does not force re-downloading ~3 GB of torch/nvidia wheels.
COPY requirements-torch.txt /app/requirements-torch.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /app/requirements-torch.txt

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt && \
    python3 -m pip install --no-deps qwen-tts==0.1.1

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    QWEN_LOCAL_FILES_ONLY=1 \
    QWEN_DEVICE=cuda:0 \
    QWEN_DTYPE=bfloat16 \
    QWEN_ATTN_IMPL=sdpa

COPY app /app/app

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8888/healthz || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
