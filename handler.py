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

MODEL = None

def load_model():
    global MODEL
    from huggingface_hub import snapshot_download
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    from diffusers.image_processor import VaeImageProcessor

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    local_dir = "/workspace/hf_cache/zhengchong_CatVTON"
    snapshot_download(repo_id="zhengchong/CatVTON", local_dir=local_dir, ignore_patterns=[])

    pipe = CatVTONPipeline(
        attn_ckpt_version="mix",
        attn_ckpt=local_dir,
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        device="cuda",
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(local_dir, "DensePose"),
        schp_ckpt=os.path.join(local_dir, "SCHP"),
        device="cuda",
    )
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True,
    )
    MODEL = {"pipe": pipe, "automasker": automasker, "mask_processor": mask_processor}
    print("[CatVTON] Ready.")

def get_image(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def handler(job):
    global MODEL
    try:
        if MODEL is None:
            load_model()

        inp = job.get("input", {})
        person_url = inp.get("person_image")
        garment_url = inp.get("garment_image")
        cloth_type = inp.get("cloth_type", "upper_body")
        steps = int(inp.get("steps", 20))

        if not person_url or not garment_url:
            return {"error": "person_image and garment_image are required"}

        # Map cloth types from Next.js API to CatVTON AutoMasker types
        type_map = {"upper_body": "upper", "lower_body": "lower", "dresses": "overall"}
        catvton_type = type_map.get(cloth_type, "upper")

        person_img = get_image(person_url).resize((768, 1024))
        cloth_img = get_image(garment_url).resize((768, 1024))

        # Generate mask
        mask_data = MODEL["automasker"](person_img, catvton_type)
        mask_img = MODEL["mask_processor"].blur(mask_data["mask"], blur_factor=9)

        start = time.time()
        print(f"[CatVTON] Running inference ({steps} steps)...")

        result = MODEL["pipe"](
            image=person_img,
            condition_image=cloth_img,
            mask=mask_img,
            num_inference_steps=steps,
            guidance_scale=2.5,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )[0]

        print(f"[CatVTON] Done in {time.time()-start:.2f}s")

        buf = BytesIO()
        result.save(buf, format="PNG")
        return {"status": "success", "image_base64": base64.b64encode(buf.getvalue()).decode()}

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "trace": traceback.format_exc()}

# Warmup at startup
try:
    load_model()
except Exception as e:
    print(f"[CatVTON] Startup load failed (will retry on first job): {e}")

runpod.serverless.start({"handler": handler})
