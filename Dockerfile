FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gcc g++ libgl1 libglib2.0-0 libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install huggingface_hub first with old version that has cached_download
RUN pip install --no-cache-dir "huggingface_hub==0.19.4"

# Install fvcore (needed by detectron2 inside CatVTON)
RUN pip install --no-cache-dir \
    "fvcore==0.1.5.post20221221" \
    "iopath==0.1.9" \
    "omegaconf==2.3.0" \
    "pycocotools==2.0.7"

# Install all other dependencies
RUN pip install --no-cache-dir \
    "diffusers==0.25.0" \
    "transformers==4.36.2" \
    "accelerate==0.25.0" \
    "runpod==1.6.0" \
    "requests==2.31.0" \
    "Pillow==10.0.0" \
    "scipy==1.11.4" \
    "opencv-python-headless==4.8.1.78" \
    "numpy==1.24.4" \
    "einops==0.7.0" \
    "timm==0.9.12" \
    "av==11.0.0" \
    "cloudpickle==3.0.0" \
    "yacs==0.1.8"

# Clone CatVTON
RUN git clone https://github.com/Zheng-Chong/CatVTON /workspace/CatVTON

WORKDIR /workspace/CatVTON

# Install CatVTON's own requirements on top (this handles any remaining deps)
RUN pip install --no-cache-dir -r requirements.txt || true

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]
