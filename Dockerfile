FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Added build-essential + python3-dev (required for pycocotools wheel compilation)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Zheng-Chong/CatVTON.git

WORKDIR /workspace/CatVTON

RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Removed --no-deps: pip can now install missing sub-deps without touching torch
# (torch is already installed + not in requirements.txt, so pip leaves it alone)
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

# Catches import errors at BUILD TIME instead of silently failing at runtime
RUN python -c "import runpod, torch, diffusers, transformers; print('core imports OK')"
RUN python -c "import sys; sys.path.insert(0,'/workspace/CatVTON'); from model.pipeline import CatVTONPipeline; from model.cloth_masker import AutoMasker; print('CatVTON imports OK')"

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]