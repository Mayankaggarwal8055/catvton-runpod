FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install base dependencies first
RUN pip install --no-cache-dir \
    "numpy==1.24.4" \
    "huggingface_hub==0.19.4" \
    "Pillow==10.0.0" \
    "requests==2.31.0" \
    "runpod==1.6.0"

# Install ML dependencies separately
RUN pip install --no-cache-dir \
    "diffusers==0.25.0" \
    "transformers==4.36.2" \
    "accelerate==0.25.0"

# Install vision dependencies separately
RUN pip install --no-cache-dir \
    "scipy==1.11.4" \
    "opencv-python-headless" \
    "einops" \
    "timm" \
    "yacs" \
    "fvcore" \
    "iopath" \
    "cloudpickle" \
    "pycocotools"

# Clone CatVTON
RUN git clone https://github.com/Zheng-Chong/CatVTON /workspace/CatVTON

WORKDIR /workspace/CatVTON

RUN pip install --no-cache-dir -r requirements.txt || true

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]
