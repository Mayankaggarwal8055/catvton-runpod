FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Zheng-Chong/CatVTON.git

WORKDIR /workspace/CatVTON

RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# ✅ FIXED: removed --force-reinstall (was pulling torch back in via diffusers deps)
# Use --no-deps for the pinned stack — versions are explicit, no resolution needed
RUN pip install --no-cache-dir --no-deps \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    huggingface_hub==0.19.4 \
    accelerate==0.25.0

# ✅ FIXED: pinned fvcore + iopath (unpinned = resolver hangs 10-20 min)
# ✅ FIXED: merged into one RUN block to cut resolver overhead
RUN pip install --no-cache-dir \
    runpod==1.6.0 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.9 \
    omegaconf==2.3.0 \
    yacs \
    cloudpickle \
    pycocotools

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]