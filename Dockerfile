FROM python:3.10-slim

WORKDIR /workspace

# ── System deps ─────────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0 are required by OpenCV (used by DensePose/SCHP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch with CUDA 11.8 ──────────────────────────────────────────────────
# Version pins ensure reproducibility. Upgrade torch with care — CUDA compat.
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ── Clone CatVTON ───────────────────────────────────────────────────────────
RUN git clone https://github.com/Zheng-Chong/CatVTON /workspace/CatVTON

# Fix repo references (original used zheng-chong, model files reference zhengchong)
RUN sed -i 's/zheng-chong\/CatVTON/zhengchong\/CatVTON/g' /workspace/CatVTON/model/pipeline.py
RUN sed -i 's/zheng-chong\/CatVTON/zhengchong\/CatVTON/g' /workspace/CatVTON/model/cloth_masker.py

WORKDIR /workspace/CatVTON

# ── Install CatVTON requirements first ──────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Force-reinstall specific versions for compatibility ─────────────────────
# CatVTON's requirements.txt may pull newer versions that break the pipeline.
# These pins are known to work together.
RUN pip install --no-cache-dir --force-reinstall \
    "diffusers==0.25.0" \
    "transformers==4.36.2" \
    "huggingface_hub==0.19.4"

# ── Additional deps ─────────────────────────────────────────────────────────
# rembg provides CPU-based background removal as a fallback mask generator
# fvcore/iopath/yacs/cloudpickle/pycocotools/omegaconf are required by DensePose/SCHP
# Pillow >= 10 for ImageOps improvements
RUN pip install --no-cache-dir \
    "runpod==1.6.0" \
    "fvcore" \
    "iopath" \
    "yacs" \
    "cloudpickle" \
    "pycocotools" \
    "omegaconf==2.3.0" \
    "Pillow>=10.0.0" \
    "requests>=2.31.0"

# ── Copy handler ────────────────────────────────────────────────────────────
COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]
