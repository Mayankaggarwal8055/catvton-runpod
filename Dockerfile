FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install exact versions - huggingface_hub must be old enough to have cached_download
RUN pip install --no-cache-dir \
    "huggingface_hub==0.19.4" \
    "diffusers==0.25.0" \
    "transformers==4.36.2" \
    "accelerate==0.25.0" \
    "runpod==1.6.0" \
    "requests==2.31.0" \
    "Pillow==10.0.0" \
    "scipy==1.11.4" \
    "opencv-python-headless==4.8.1.78" \
    "numpy==1.24.4"

# Clone CatVTON
RUN git clone https://github.com/Zheng-Chong/CatVTON /workspace/CatVTON

WORKDIR /workspace/CatVTON

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]
