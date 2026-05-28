import runpod
import torch
import base64
import requests
import sys

from PIL import Image
from io import BytesIO

sys.path.insert(0, "/workspace/CatVTON")

pipe = None
automasker = None
mask_processor = None

def load_model():
    global pipe, automasker, mask_processor

    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    from diffusers.image_processor import VaeImageProcessor
    from huggingface_hub import snapshot_download
    import os

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    local_catvton = "/workspace/models/catvton"
    local_sd = "/workspace/models/sd-inpainting"

    # Download models if not already present
    if not os.path.exists(local_catvton):
        print("[CatVTON] Downloading CatVTON weights...")
        snapshot_download(
            repo_id="zhengchong/CatVTON",
            local_dir=local_catvton,
            local_dir_use_symlinks=False,
            token=token
        )

    if not os.path.exists(local_sd):
        print("[CatVTON] Downloading SD inpainting...")
        snapshot_download(
            repo_id="booksforcharlie/stable-diffusion-inpainting",
            local_dir=local_sd,
            local_dir_use_symlinks=False,
            token=token
        )

    print("[CatVTON] Loading pipeline...")
    pipe = CatVTONPipeline(
        base_ckpt=local_sd,
        attn_ckpt=local_catvton,
        attn_ckpt_version="mix",
        device="cuda"
    )

    print("[CatVTON] Loading AutoMasker...")
    automasker = AutoMasker(
        densepose_ckpt=local_catvton,
        schp_ckpt=local_catvton,
        device="cuda"
    )

    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True
    )
    print("[CatVTON] Ready")


def handler(job):
    global pipe, automasker, mask_processor

    try:
        if pipe is None or automasker is None or mask_processor is None:
            load_model()

        job_input = job["input"]
        person_url = job_input.get("person_image")
        garment_url = job_input.get("garment_image")
        cloth_type = job_input.get("cloth_type", "upper")
        steps = int(job_input.get("steps", 20))

        if not person_url:
            return {"error": "Missing person_image"}
        if not garment_url:
            return {"error": "Missing garment_image"}

        print("[CatVTON] Downloading images...")
        person_image = download_image(person_url)
        garment_image = download_image(garment_url)
        person_image = person_image.resize((768, 1024))
        garment_image = garment_image.resize((768, 1024))

        print("[CatVTON] Generating mask...")
        mask = automasker(person_image, cloth_type)["mask"]

        print("[CatVTON] Running inference...")
        result = pipe(
            image=person_image,
            condition_image=garment_image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=2.5,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )[0]

        print("[CatVTON] Inference complete")
        return {
            "status": "success",
            "image_base64": image_to_base64(result)
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        }


def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


runpod.serverless.start({"handler": handler})