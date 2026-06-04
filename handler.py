import os, io, base64, random, traceback, sys
import runpod, requests, numpy as np, torch
from PIL import Image

CLOTH_TYPE_MAP = {
    "upper_body": "upper",
    "lower_body": "lower",
    "dresses":    "overall",
    "upper":      "upper",
    "lower":      "lower",
    "overall":    "overall",
}

TARGET_SIZE = (768, 1024)
pipeline = None
gfpgan_restorer = None

def load_models():
    global pipeline, gfpgan_restorer
    if pipeline is not None:
        return
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker

    models_dir = os.environ.get("MODELS_DIR", "/workspace/models")
    catvton_path = os.path.join(models_dir, "catvton")
    sd_path = os.path.join(models_dir, "sd-inpainting")

    from huggingface_hub import snapshot_download
    import urllib.request

    # Download CatVTON weights if missing
    if not os.path.exists(os.path.join(catvton_path, "SCHP")):
        print("[handler] Downloading CatVTON weights...")
        snapshot_download("zhengchong/CatVTON", local_dir=catvton_path, local_dir_use_symlinks=False)

    # Download SD inpainting if missing
    if not os.path.exists(os.path.join(sd_path, "unet")):
        print("[handler] Downloading SD inpainting...")
        snapshot_download("booksforcharlie/stable-diffusion-inpainting", local_dir=sd_path, local_dir_use_symlinks=False)

    # Download GFPGAN if missing
    gfpgan_path = os.environ.get("GFPGAN_MODEL_PATH", "/workspace/models/gfpgan/GFPGANv1.3.pth")
    if not os.path.exists(gfpgan_path):
        os.makedirs(os.path.dirname(gfpgan_path), exist_ok=True)
        print("[handler] Downloading GFPGAN...")
        urllib.request.urlretrieve("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth", gfpgan_path)

    pipeline = CatVTONPipeline(
        base_ckpt=sd_path,
        attn_ckpt=catvton_path,
        attn_ckpt_version="mix",
        weight_dtype=torch.float16,
        device="cuda",
        skip_safety_check=True,
    )
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.vae.enable_slicing()
    pipeline.automasker = AutoMasker(
        densepose_ckpt=os.path.join(catvton_path, "DensePose"),
        schp_ckpt=os.path.join(catvton_path, "SCHP"),
        device="cuda",
    )

    from gfpgan import GFPGANer
    gfpgan_restorer = GFPGANer(
        model_path=gfpgan_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device="cuda",
    )
    print("[handler] Models ready")

def download_image(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def resize_and_pad(img):
    img = img.copy()
    img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", TARGET_SIZE, (255,255,255))
    paste_x = (TARGET_SIZE[0] - img.size[0]) // 2
    paste_y = (TARGET_SIZE[1] - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def prepare_mask(mask_img, cloth_type, person_img):
    if mask_img is not None:
        mask = mask_img.convert("L")
    else:
        result = pipeline.automasker(person_img, cloth_type)
        mask = result["mask"]
    if mask.size != TARGET_SIZE:
        mask = mask.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    return mask

def restore_face(image):
    if gfpgan_restorer is None:
        return image
    img_cv = np.array(image.convert("RGB"))[:,:,::-1].copy()
    _, _, restored = gfpgan_restorer.enhance(img_cv, has_aligned=False, only_center_face=True, paste_back=True)
    return Image.fromarray(restored[:,:,::-1]).convert("RGB") if restored is not None else image

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def handler(job):
    try:
        inp = job.get("input", {})
        person_url = inp.get("person_image_url") or inp.get("person_image")
        garment_url = inp.get("garment_image_url") or inp.get("garment_image")
        mask_url = inp.get("mask_image_url") or inp.get("mask_image")
        cloth_type_raw = inp.get("cloth_type", "upper_body")

        if not person_url or not garment_url:
            return {"status":"error","error":"Missing image URLs","error_code":"INVALID_INPUT"}

        cloth_type = CLOTH_TYPE_MAP.get(cloth_type_raw, "upper")
        person_img = resize_and_pad(download_image(person_url))
        garment_img = resize_and_pad(download_image(garment_url))
        mask_img = download_image(mask_url) if mask_url else None
        mask = prepare_mask(mask_img, cloth_type, person_img)

        seed = inp.get("seed", random.randint(0, 2**31-1))
        steps = inp.get("steps", 20)
        guidance = inp.get("guidance_scale", 7.5)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipeline(
            image=person_img,
            condition_image=garment_img,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )[0]

        result = restore_face(result)
        return {"status":"success","image_base64":image_to_base64(result),"cloth_type_used":cloth_type,"steps_used":steps,"seed":seed}
    except Exception as e:
        traceback.print_exc()
        return {"status":"error","error":str(e),"error_code":"INFERENCE_ERROR"}

# Startup with error capture
try:
    if not os.environ.get("RUNPOD_WARMUP_DISABLE"):
        load_models()
    runpod.serverless.start({"handler": handler})
except Exception:
    print("\n[FATAL] Worker startup failed:\n")
    traceback.print_exc()
    sys.stdout.flush()
    raise