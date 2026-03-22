# ============================================================
# Trading Bot — Multi-stage Docker build
# Stages:
#   1. base   : CUDA 12.1 + Python 3.10 + system deps
#   2. deps   : Python packages (cache-friendly layer)
#   3. app    : Application code
#
# Build:
#   docker build -t trading-bot:latest .
#
# Run (GPU):
#   docker run --gpus all -v $(pwd)/data:/app/data trading-bot:latest
#
# Run (CPU):
#   docker run -v $(pwd)/data:/app/data trading-bot:latest
# ============================================================

# ──────────────────────────────────────────────────────────
# Stage 1: base — CUDA runtime + Python 3.10
# ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Alias python3.10 → python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ──────────────────────────────────────────────────────────
# Stage 2: deps — Install Python packages
# This layer is cached as long as requirements.txt is unchanged.
# ──────────────────────────────────────────────────────────
FROM base AS deps

# Copy only requirements first (maximise cache hit rate)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 wheels first (avoid index conflicts)
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the requirements
# Some heavy optional packages are skipped in the Docker image to keep it lean.
# Comment out lines you don't need to reduce image size.
RUN pip install -r requirements.txt \
    || true  # tolerant — optional deps may fail (e.g. torchrl)

# ──────────────────────────────────────────────────────────
# Stage 3: app — Copy source code
# ──────────────────────────────────────────────────────────
FROM deps AS app

# Create expected directories
RUN mkdir -p \
    /app/data/raw \
    /app/data/cache \
    /app/checkpoints \
    /app/mlruns \
    /app/logs \
    /app/results

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.py 2>/dev/null || true

# Default environment variables (override at runtime)
ENV CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=/app/mlruns

# Healthcheck: verify Python + torch import
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; print('GPU:', torch.cuda.is_available())" || exit 1

# Default entrypoint — override with docker-compose `command:`
ENTRYPOINT ["python"]
CMD ["-m", "training.train_pipeline", "--config", "config/local_3060ti.yaml"]
