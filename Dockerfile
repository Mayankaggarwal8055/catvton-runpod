FROM runpod/base:0.4.0-py3.11

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first with exact version
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install diffusers and dependencies
RUN pip install --no-cache-dir \
    "diffusers==0.27.2" \
    "transformers==4.38.2" \
    "accelerate==0.27.2" \
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
