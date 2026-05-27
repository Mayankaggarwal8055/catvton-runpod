"""
RunPod Serverless handler for CatVTON Virtual Try-On

Accepts:
  - person_image  (required) — URL to the standardized person photo
  - garment_image (required) — URL to the prepared garment photo
  - mask_image    (optional) — pre-generated mask from preprocessing service (fallback)
  - cloth_type    (optional) — "upper_body" | "lower_body" | "dresses" (default "upper_body")
  - steps         (optional) — inference steps (default 20)
  - guidance_scale (optional) — CFG scale (default 2.5)
  - force_dc      (optional) — force dress code processing (default false)

Returns:
  - On success: { "status": "success", "image_base64": "<base64 PNG>" }
  - On error:   { "status": "error", "error": "<message>", "trace": "<stack>" }
"""

import runpod
import torch
import base64
import requests
import sys
import time
import os
import gc
import subprocess
import shutil
import traceback
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

sys.path.insert(0, "/workspace/CatVTON")

models = {
    "pipe": None,
    "automasker": None,
    "mask_processor": None,
    "loaded": False,
}

# ── Cloth type mapping ─────────────────────────────────────────────────────
# The API sends upper_body / lower_body / dresses (from compatibility checks)
# but CatVTON's AutoMasker expects upper / lower / overall. Mismatch here
# is the #1 cause of mask failures.

CLOTH_TYPE_MAP = {
    "upper_body": "upper",
    "lower_body": "lower",
    "dresses": "overall",
    "full_body": "overall",
}

TARGET_SIZE = (768, 1024)


# ── Model Loading ──────────────────────────────────────────────────────────

def load_model():
    """Load CatVTON pipeline, AutoMasker, and mask processor from HuggingFace."""
    from huggingface_hub import login, snapshot_download
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    from diffusers.image_processor import VaeImageProcessor

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    local_dir = "/workspace/hf_cache/zhengchong_CatVTON"

    # Clear stale LFS pointer files from incomplete downloads
    if os.path.exists(local_dir):
        schp_test = os.path.join(local_dir, "SCHP", "exp-schp-201908301523-atr.pth")
        if os.path.exists(schp_test) and os.path.getsize(schp_test) < 1024:
            print("[CatVTON] Found stale LFS pointer files — clearing cache...")
            shutil.rmtree(local_dir)
        elif os.path.exists(local_dir) and not os.path.exists(
            os.path.join(local_dir, "SCHP")
        ):
            print("[CatVTON] Missing SCHP directory — clearing cache...")
            shutil.rmtree(local_dir)

    print("[CatVTON] Downloading model weights...")
    local_model_path = snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir=local_dir,
        ignore_patterns=[],
    )
    print(f"[CatVTON] Model downloaded to: {local_model_path}")

    # Debug — verify model files
    for pattern in ["*.pth", "*.bin", "*.safetensors"]:
        result = subprocess.run(
            ["find", local_model_path, "-name", pattern],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            print(f"[CatVTON] {pattern} files:\n{result.stdout[:2000]}")

    print("[CatVTON] Loading pipeline...")
    models["pipe"] = CatVTONPipeline(
        attn_ckpt_version="mix",
        attn_ckpt=local_model_path,
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        device="cuda",
    )

    densepose_path = os.path.join(local_model_path, "DensePose")
    schp_path = os.path.join(local_model_path, "SCHP")

    # Verify DensePose and SCHP paths exist
    if not os.path.exists(densepose_path):
        print(f"[CatVTON] WARNING: DensePose path not found at {densepose_path}")
    if not os.path.exists(schp_path):
        print(f"[CatVTON] WARNING: SCHP path not found at {schp_path}")

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
    print("[CatVTON] Model loaded. Warming up...")
    _warmup_model()
    print("[CatVTON] Ready.")


@torch.no_grad()
def _warmup_model():
    """Run a quick dummy warmup inference to avoid cold-start timeout on first job."""
    try:
        dummy_person = Image.new("RGB", TARGET_SIZE, (128, 128, 128))
        dummy_cloth = Image.new("RGB", TARGET_SIZE, (200, 200, 200))
        dummy_mask = Image.new("L", TARGET_SIZE, 128)

        _ = models["pipe"](
            image=dummy_person,
            condition_image=dummy_cloth,
            mask=dummy_mask,
            num_inference_steps=1,
            guidance_scale=2.5,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        print("[CatVTON] Warmup complete.")
    except Exception as e:
        print(f"[CatVTON] Warmup failed (non-fatal): {e}")


# ── Image Helpers ──────────────────────────────────────────────────────────

def _download_image(url: str, max_retries: int = 3) -> Image.Image:
    """Download an image from URL and return as PIL RGB."""
    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            # Validate image
            if img.width < 64 or img.height < 64:
                raise ValueError(
                    f"Image too small: {img.width}x{img.height}"
                )
            return img
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[CatVTON] Download attempt {attempt+1} failed, retrying in {wait}s: {e}")
                time.sleep(wait)
    raise RuntimeError(f"Failed to download image after {max_retries} attempts: {last_error}")


def _resize_standard(img: Image.Image, target: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """
    Resize image to target size while preserving aspect ratio.
    Uses ImageOps.fit to crop-center the image without distortion.
    """
    return ImageOps.fit(img, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64-encoded PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _gpu_cleanup():
    """Clear GPU memory between runs to prevent OOM accumulation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Mask Generation (with fallback chain) ───────────────────────────────────

def generate_mask_automasker(
    person_img: Image.Image,
    cloth_type: str,
) -> Image.Image | None:
    """Try generating mask using CatVTON's AutoMasker (DensePose + SCHP)."""
    try:
        print(f"[CatVTON] AutoMasker generating mask (cloth_type={cloth_type})...")
        mask = models["automasker"](person_img, cloth_type)["mask"]
        mask_image = models["mask_processor"].blur(mask, blur_factor=9)
        return mask_image
    except Exception as e:
        print(f"[CatVTON] AutoMasker failed: {e}")
        return None


def generate_mask_fallback(person_img: Image.Image) -> Image.Image:
    """
    Ultimate fallback: create a simple full-frame mask.
    This is not as good as AutoMasker but prevents job failure.
    """
    print("[CatVTON] Using fallback full-frame mask...")
    # Create a centered elliptical mask covering ~70% of the image
    mask = Image.new("L", TARGET_SIZE, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2
    rx, ry = int(TARGET_SIZE[0] * 0.38), int(TARGET_SIZE[1] * 0.45)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return mask


# ── Handler ─────────────────────────────────────────────────────────────────

def handler(job):
    """
    Main RunPod handler — orchestrates CatVTON try-on with graceful degradation.
    """
    try:
        inp = job.get("input", {})
        person_url = inp.get("person_image")
        garment_url = inp.get("garment_image")
        mask_url = inp.get("mask_image")  # optional — pre-generated mask
        cloth_type_raw = inp.get("cloth_type", "upper_body")
        steps = int(inp.get("steps", 20))
        guidance_scale = float(inp.get("guidance_scale", 2.5))
        force_dc = str(inp.get("force_dc", "false")).lower() == "true"

        if not person_url or not garment_url:
            return {"error": "person_image and garment_image are required"}

        # ── Map cloth type ──────────────────────────────────────────
        # CRITICAL: AutoMasker expects 'upper' / 'lower' / 'overall'
        # but API sends 'upper_body' / 'lower_body' / 'dresses'
        masker_type = CLOTH_TYPE_MAP.get(cloth_type_raw, "upper")

        print(f"[CatVTON] cloth_type_raw={cloth_type_raw}, masker_type={masker_type}")

        # ── Load model if not loaded ────────────────────────────────
        if not models["loaded"]:
            print("[CatVTON] Cold start — loading model...")
            load_model()

        # ── Download images ─────────────────────────────────────────
        print("[CatVTON] Downloading person image...")
        person_img = _download_image(person_url)
        print("[CatVTON] Downloading garment image...")
        cloth_img = _download_image(garment_url)

        # ── Standardize to target size ──────────────────────────────
        person_img = _resize_standard(person_img)
        cloth_img = _resize_standard(cloth_img)

        # ── Generate mask with fallback chain ──────────────────────
        # Priority: 1) AutoMasker  2) pre-provided mask  3) simple fallback
        mask_image = None

        # Attempt 1: AutoMasker (best quality)
        if models["automasker"] is not None:
            mask_image = generate_mask_automasker(person_img, masker_type)

        # Attempt 2: Use pre-generated mask from preprocessing service
        if mask_image is None and mask_url:
            try:
                print(f"[CatVTON] Using pre-generated mask from preprocessing...")
                mask_img = _download_image(mask_url).convert("L")
                mask_img = _resize_standard(mask_img)
                mask_image = models["mask_processor"].blur(mask_img, blur_factor=9) \
                    if models["mask_processor"] else mask_img
            except Exception as e:
                print(f"[CatVTON] Pre-generated mask download/process failed: {e}")

        # Attempt 3: Simple fallback mask
        if mask_image is None:
            print("[CatVTON] All mask methods failed — using simple fallback mask")
            mask_image = generate_mask_fallback(person_img)

        # ── Run inference ───────────────────────────────────────────
        start = time.time()
        print(f"[CatVTON] Running inference ({steps} steps, guidance={guidance_scale})...")

        # Clean up before inference to maximize available memory
        _gpu_cleanup()

        result = models["pipe"](
            image=person_img,
            condition_image=cloth_img,
            mask=mask_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )[0]

        elapsed = time.time() - start
        print(f"[CatVTON] Inference complete in {elapsed:.2f}s")

        # Clean up after inference
        del person_img, cloth_img, mask_image
        _gpu_cleanup()

        return {
            "status": "success",
            "image_base64": _to_base64(result),
        }

    except requests.exceptions.RequestException as e:
        _gpu_cleanup()
        return {
            "status": "error",
            "error": f"Image download failed: {e}",
            "trace": traceback.format_exc(),
        }
    except torch.cuda.OutOfMemoryError as e:
        _gpu_cleanup()
        return {
            "status": "error",
            "error": f"GPU out of memory. Try reducing steps or image size: {e}",
            "trace": traceback.format_exc(),
        }
    except Exception as e:
        _gpu_cleanup()
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc(),
        }


# ── Start server ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[CatVTON] Starting RunPod serverless handler...")
    # Pre-load model at startup so first job doesn't time out
    try:
        print("[CatVTON] Pre-loading model at startup...")
        load_model()
    except Exception as e:
        print(f"[CatVTON] Startup model loading failed (will load on first request): {e}")

    runpod.serverless.start({"handler": handler})
