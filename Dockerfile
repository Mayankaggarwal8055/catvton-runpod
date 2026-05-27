FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone CatVTON
RUN git clone https://github.com/Zheng-Chong/CatVTON.git

WORKDIR /workspace/CatVTON

# Remove incompatible torch packages from requirements
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Force compatible versions
RUN pip install --no-cache-dir --force-reinstall \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    huggingface_hub==0.19.4 \
    accelerate==0.25.0

# Extra dependencies required by CatVTON
RUN pip install --no-cache-dir \
    runpod==1.6.0 \
    fvcore \
    iopath \
    omegaconf==2.3.0 \
    yacs \
    cloudpickle \
    pycocotools

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]