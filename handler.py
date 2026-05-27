import runpod
import torch
import base64
import requests
import sys
import time
import os
from PIL import Image
from io import BytesIO

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

sys.path.insert(0, '/workspace/CatVTON')

# Use a dict to store models so global scope works correctly
models = {
    "pipe": None,
    "automasker": None,
    "mask_processor": None,
    "loaded": False
}

def load_model():
    from huggingface_hub import login, snapshot_download
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    from diffusers.image_processor import VaeImageProcessor

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("[CatVTON] Downloading model weights...")
    local_model_path = snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir="/workspace/hf_cache/zhengchong_CatVTON",
        ignore_patterns=[],
    )
    print(f"[CatVTON] Model downloaded to: {local_model_path}")

    print("[CatVTON] Loading pipeline...")
    models["pipe"] = CatVTONPipeline(
        attn_ckpt_version="mix",
        attn_ckpt=local_model_path,
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        device="cuda",
    )

    densepose_path = os.path.join(local_model_path, "DensePose")
    schp_path = os.path.join(local_model_path, "SCHP")
    print("[CatVTON] Loading AutoMasker...")
    models["automasker"] = AutoMasker(
        densepose_ckpt=densepose_path,
        schp_ckpt=schp_path,
        device="cuda",
    )

    models["mask_processor"] = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )
    models["loaded"] = True
    print("[CatVTON] Ready.")

def get_image(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def handler(job):
    try:
        if not models["loaded"]:
            load_model()

        inp = job.get("input", {})
        person_url  = inp.get("person_image")
        garment_url = inp.get("garment_image")
        cloth_type  = inp.get("cloth_type", "upper")
        steps       = int(inp.get("steps", 20))

        if not person_url or not garment_url:
            return {"error": "person_image and garment_image are required"}

        print("[CatVTON] Downloading images...")
        person_img = get_image(person_url)
        cloth_img  = get_image(garment_url)

        person_img = person_img.resize((768, 1024))
        cloth_img  = cloth_img.resize((768, 1024))

        print("[CatVTON] Generating mask...")
        mask = models["automasker"](person_img, cloth_type)["mask"]
        mask_image = models["mask_processor"].blur(mask, blur_factor=9)

        start = time.time()
        print(f"[CatVTON] Running inference ({steps} steps)...")

        result = models["pipe"](
            image=person_img,
            condition_image=cloth_img,
            mask_image=mask_image,
            num_inference_steps=steps,
            guidance_scale=2.5,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )[0]

        print(f"[CatVTON] Done in {time.time()-start:.2f}s")

        return {
            "status": "success",
            "image_base64": to_base64(result)
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        }

runpod.serverless.start({"handler": handler})
