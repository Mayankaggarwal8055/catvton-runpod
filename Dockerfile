FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone CatVTON
RUN git clone https://github.com/Zheng-Chong/CatVTON /workspace/CatVTON

RUN sed -i 's/zheng-chong\/CatVTON/zhengchong\/CatVTON/g' /workspace/CatVTON/model/pipeline.py
RUN sed -i 's/zheng-chong\/CatVTON/zhengchong\/CatVTON/g' /workspace/CatVTON/model/cloth_masker.py

WORKDIR /workspace/CatVTON

# Install CatVTON requirements FIRST
RUN pip install --no-cache-dir -r requirements.txt

# Now force-reinstall correct diffusers version (overrides whatever requirements.txt installed)
RUN pip install --no-cache-dir --force-reinstall \
    "diffusers==0.25.0" \
    "transformers==4.36.2" \
    "huggingface_hub==0.19.4"

# Install remaining deps
RUN pip install --no-cache-dir \
    "runpod==1.6.0" \
    "fvcore" \
    "iopath" \
    "yacs" \
    "cloudpickle" \
    "pycocotools" \
    "omegaconf==2.3.0"

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]
