# syntax=docker/dockerfile:1.4
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    MODELS_DIR=/workspace/models

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl \
        libgl1 libglib2.0-0 \
        build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Mayankaggarwal8055/CatVTON.git /workspace/CatVTON
WORKDIR /workspace/CatVTON

# torchvision
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps \
        "torchvision==0.16.0+cu118" \
        --index-url https://download.pytorch.org/whl/cu118

# ── Install all dependencies in one transaction ──
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "diffusers==0.27.2" \
        "huggingface_hub==0.25.0" \
        "accelerate==0.33.0" \
        "transformers==4.36.2" \
        "numpy==1.26.4" \
        "opencv-python==4.10.0.84" \
        "opencv-python-headless==4.10.0.84" \
        "Pillow==10.3.0" \
        "scipy==1.13.1" \
        "scikit-image==0.24.0" \
        "tqdm==4.66.4" \
        "matplotlib==3.9.1" \
        "fvcore==0.1.5.post20221221" \
        "cloudpickle==3.0.0" \
        "omegaconf==2.3.0" \
        "pycocotools==2.0.8" \
        "av==12.3.0" \
        "gfpgan==1.3.8" \
        "realesrgan==0.3.0" \
        "basicsr==1.4.2" \
        "facexlib==0.3.0" \
        "mediapipe==0.10.9" \
        "yacs==0.1.8" \
        "antlr4-python3-runtime==4.9.3" \
        "iopath==0.1.10" \
        "tabulate==0.10.0" \
        "termcolor==3.3.0" \
        "portalocker==3.2.0"

# Install peft last with --no-deps to prevent downgrades
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "peft==0.17.0"

# Force numpy 1.26.4 (belt and suspenders)
RUN pip install --force-reinstall "numpy==1.26.4"

# Verify
RUN python -c "import diffusers, peft; print('OK')"

COPY handler.py /workspace/CatVTON/handler.py
CMD ["python", "-u", "/workspace/CatVTON/handler.py"]