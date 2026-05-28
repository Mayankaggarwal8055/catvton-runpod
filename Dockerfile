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
    accelerate==0.30.0

RUN pip install --no-cache-dir \
    runpod==1.6.0 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.9 \
    omegaconf==2.3.0 \
    yacs \
    cloudpickle \
    pycocotools

RUN pip install --no-cache-dir \
    "torchvision==0.16.0+cu118" \
    --index-url https://download.pytorch.org/whl/cu118 \
    --no-deps

RUN pip install --no-cache-dir "numpy==1.26.4"

# ✅ peft LAST — after everything else so nothing can overwrite it
RUN pip install --no-cache-dir "peft==0.10.0"

# ✅ Verify peft and accelerate are correct versions before shipping
RUN python -c "import peft, accelerate; print('peft:', peft.__version__, 'accelerate:', accelerate.__version__)"

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]