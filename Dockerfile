FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Zheng-Chong/CatVTON /workspace/CatVTON

WORKDIR /workspace/CatVTON

# Force uninstall system diffusers first, then install correct version
RUN pip uninstall -y diffusers transformers accelerate && \
    pip install --no-cache-dir \
    "diffusers==0.27.2" \
    "transformers==4.38.2" \
    "accelerate==0.27.2" \
    "runpod==1.6.0" \
    "requests==2.31.0" \
    "Pillow==10.0.0" \
    "scipy==1.11.4" \
    "opencv-python-headless==4.8.1.78"

COPY handler.py /workspace/CatVTON/handler.py

CMD ["python", "-u", "/workspace/CatVTON/handler.py"]
