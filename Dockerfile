FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 \
    build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Zheng-Chong/CatVTON.git

WORKDIR /workspace/CatVTON

RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    huggingface_hub==0.19.4 \
    accelerate==0.25.0

RUN pip install --no-cache-dir \
    runpod==1.6.0 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.9 \
    omegaconf==2.3.0 \
    yacs \
    cloudpickle \
    pycocotools

# ✅ THE FIX: restore CUDA torchvision LAST
# Some transitive dep in requirements.txt overwrites it with a CPU build
# This pins back the exact version matching torch 2.1.0 + CUDA 11.8
RUN pip install --no-cache-dir --force-reinstall \
    "torchvision==0.16.0+cu118" \
    --index-url https://download.pytorch.org/whl/cu118

# Verify the fix worked before shipping the image
RUN python -c "import torch, torchvision; from torchvision.ops import nms; print('torchvision CUDA OK:', torchvision.__version__)"

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]