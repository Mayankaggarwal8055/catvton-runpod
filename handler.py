import runpod
import torch
import base64
import requests
import sys
import os

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

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    local_catvton = "/workspace/models/catvton"
    local_sd = "/workspace/models/sd-inpainting"

    os.makedirs(local_catvton, exist_ok=True)
    os.makedirs(local_sd, exist_ok=True)

    if not os.path.exists(os.path.join(local_catvton, "SCHP")):
        print("[CatVTON] Downloading CatVTON weights...")
        snapshot_download(
            repo_id="zhengchong/CatVTON",
            local_dir=local_catvton,
            local_dir_use_symlinks=False,
            token=token
        )

    if not os.path.exists(os.path.join(local_sd, "unet")):
        print("[CatVTON] Downloading SD inpainting...")
        snapshot_download(
            repo_id="booksforcharlie/stable-diffusion-inpainting",
            local_dir=local_sd,
            local_dir_use_symlinks=False,
            token=token
        )

    # ✅ Exact same paths as app.py uses
    densepose_path = os.path.join(local_catvton, "DensePose")
    schp_path = os.path.join(local_catvton, "SCHP")

    print("[CatVTON] Loading pipeline...")
    pipe = CatVTONPipeline(
        base_ckpt=local_sd,
        attn_ckpt=local_catvton,
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device="cuda",
        skip_safety_check=True   # ✅ skip safety checker — needs CLIP + extra model
    )

    print("[CatVTON] Loading AutoMasker...")
    automasker = AutoMasker(
        densepose_ckpt=densepose_path,
        schp_ckpt=schp_path,
        device="cuda"
    )

    # ✅ mask_processor lives inside AutoMasker already, but we need it for blur
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True
    )

    print("[CatVTON] Ready")


def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


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
        guidance_scale = float(job_input.get("guidance_scale", 2.5))
        seed = int(job_input.get("seed", 42))

        if not person_url:
            return {"error": "Missing person_image"}
        if not garment_url:
            return {"error": "Missing garment_image"}

        print("[CatVTON] Downloading images...")
        person_image = download_image(person_url)
        garment_image = download_image(garment_url)

        # ✅ Exact same resize logic as app.py
        from utils import resize_and_crop, resize_and_padding
        person_image = resize_and_crop(person_image, (768, 1024))
        garment_image = resize_and_padding(garment_image, (768, 1024))

        print("[CatVTON] Generating mask...")
        # ✅ automasker returns dict, get mask, then blur — exactly like app.py
        mask = automasker(person_image, cloth_type)["mask"]
        mask = mask_processor.blur(mask, blur_factor=9)

        print("[CatVTON] Running inference...")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # ✅ pipe returns a list of PIL images — [0] is correct per pipeline.py
        result = pipe(
            image=person_image,
            condition_image=garment_image,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
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


runpod.serverless.start({"handler": handler})