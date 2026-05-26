FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Clone CatVTON
RUN git clone https://github.com/Zheng-Chong/CatVTON /workspace/CatVTON

WORKDIR /workspace/CatVTON

# Install exact versions that work with CatVTON
RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    Pillow==10.0.0 \
    requests==2.31.0 \
    runpod==1.6.0 \
    opencv-python-headless==4.8.0.76 \
    scipy==1.11.0 \
    numpy==1.24.0

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]
