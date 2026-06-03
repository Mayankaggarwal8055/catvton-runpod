# syntax=docker/dockerfile:1.4
# ↑ Required for --mount=type=cache (BuildKit). Docker Desktop enables this by default.

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    MODELS_DIR=/workspace/models

WORKDIR /workspace

# ─────────────────────────────────────────────────────────────────────────────
# LAYER ORDER = build & push speed.
# Docker only re-uploads layers that changed. Put STABLE things early so they
# cache forever. Put CHANGING things (handler.py) LAST so a code-only rebuild
# only pushes a few KB instead of gigabytes of model weights.
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. System packages ── almost never changes ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl \
        libgl1 libglib2.0-0 \
        build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 2. CatVTON repo ── --depth 1 skips full history (~10× faster clone) ────
RUN git clone --depth 1 https://github.com/Mayankaggarwal8055/CatVTON.git /workspace/CatVTON
WORKDIR /workspace/CatVTON

# ── 3. CatVTON requirements (torch/torchvision stripped — already in base) ──
# --mount=type=cache means pip's HTTP cache survives between builds.
# Re-running this layer (e.g. after requirements.txt changes) takes seconds
# instead of minutes because wheels are already on disk.
RUN --mount=type=cache,target=/root/.cache/pip \
    sed -i '/torch/d;/torchvision/d' requirements.txt && \
    pip install -r requirements.txt

# ── 4. torchvision CUDA wheel (separate layer for better cache granularity) ─
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps \
        "torchvision==0.16.0+cu118" \
        --index-url https://download.pytorch.org/whl/cu118

# ── 5. HuggingFace / diffusion stack — pinned to versions that work ─────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "diffusers==0.25.0" \
        "transformers==4.44.0" \
        "peft==0.17.0" \
        "accelerate==0.33.0" \
        "huggingface_hub>=0.23.0"

# ── 6. Face restoration stack ────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "gfpgan==1.3.8" \
        "realesrgan==0.3.0" \
        "basicsr==1.4.2" \
        "facexlib==0.3.0" \
        "opencv-python-headless>=4.9.0" \
        "mediapipe==0.10.9"

# ── 7. Force numpy 1.26 AFTER gfpgan ────────────────────────────────────────
# gfpgan pulls in numpy 2.x which breaks basicsr. Always reinstall last.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --force-reinstall "numpy==1.26.4" && \
    python -c "import gfpgan, peft, accelerate, cv2; print('[verify] imports OK')"

# ── 9. Handler — changes every iteration, lives in its own final layer ───────
# On every push this is the ONLY layer Docker Hub needs to upload (~KB).
COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]