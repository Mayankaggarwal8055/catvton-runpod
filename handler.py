from __future__ import annotations

# SECTION 1: Imports
import base64
import io
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests
import runpod
import torch
from gfpgan import GFPGANer
from PIL import Image

LOCAL_CATVTON_ROOT = Path(__file__).resolve().parent.parent / "CatVTON"
DEFAULT_CATVTON_ROOT = Path("/workspace/CatVTON")
CATVTON_ROOT = Path(os.getenv("CATVTON_ROOT", str(DEFAULT_CATVTON_ROOT)))
if not CATVTON_ROOT.exists() and LOCAL_CATVTON_ROOT.exists():
    CATVTON_ROOT = LOCAL_CATVTON_ROOT
if str(CATVTON_ROOT) not in sys.path:
    sys.path.insert(0, str(CATVTON_ROOT))

from model.pipeline import CatVTONPipeline  # noqa: E402


# SECTION 2: Constants
WIDTH = 768
HEIGHT = 1024
TARGET_SIZE = (WIDTH, HEIGHT)
REQUEST_TIMEOUT_SECONDS = 45

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/workspace/models"))
CATVTON_MODEL_DIR = MODELS_DIR / "catvton"
BASE_MODEL_DIR = MODELS_DIR / "stable-diffusion-inpainting"
GFPGAN_MODEL_PATH = Path(
    os.getenv("GFPGAN_MODEL_PATH", str(MODELS_DIR / "gfpgan" / "GFPGANv1.3.pth"))
)

CATVTON_MODEL_ID = os.getenv("CATVTON_MODEL_ID", "zhengchong/CatVTON")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "booksforcharlie/stable-diffusion-inpainting")

CLOTH_TYPE_MAP = {
    "upper_body": "upper",
    "lower_body": "lower",
    "dresses": "overall",
    "upper": "upper",
    "lower": "lower",
    "overall": "overall",
}


# SECTION 3: Global model variables
pipeline: CatVTONPipeline | None = None
gfpgan_restorer: GFPGANer | None = None
device = "cuda" if torch.cuda.is_available() else "cpu"


# SECTION 4: Model loading function
def model_path_or_id(local_path: Path, repo_id: str) -> str:
    return str(local_path) if local_path.exists() else repo_id


def weight_dtype() -> torch.dtype:
    if device != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_models() -> None:
    global pipeline, gfpgan_restorer
    if pipeline is not None and gfpgan_restorer is not None:
        return
    if device != "cuda":
        raise RuntimeError("CatVTON RunPod worker requires a CUDA GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    pipeline = CatVTONPipeline(
        base_ckpt=model_path_or_id(BASE_MODEL_DIR, BASE_MODEL_ID),
        attn_ckpt=model_path_or_id(CATVTON_MODEL_DIR, CATVTON_MODEL_ID),
        attn_ckpt_version="mix",
        weight_dtype=weight_dtype(),
        device=device,
        skip_safety_check=True,
        use_tf32=True,
    )

    if not GFPGAN_MODEL_PATH.exists():
        raise RuntimeError(f"GFPGAN model not found: {GFPGAN_MODEL_PATH}")
    gfpgan_restorer = GFPGANer(
        model_path=str(GFPGAN_MODEL_PATH),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=device,
    )


# SECTION 5: Image utilities
def error_response(message: str, code: str) -> dict[str, str]:
    return {"status": "error", "error": message, "error_code": code}


def read_input_url(data: dict[str, Any], *keys: str, required: bool = True) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if required:
        raise ValueError(f"Missing required input: {' or '.join(keys)}")
    return None


def download_image(url: str, mode: str) -> Image.Image:
    if url.startswith("data:image"):
        _, encoded = url.split(",", 1)
        raw = base64.b64decode(encoded)
    else:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "TryLix-RunPod/1.0"},
        )
        response.raise_for_status()
        raw = response.content
    return Image.open(io.BytesIO(raw)).convert(mode)


def resize_and_pad(image: Image.Image, size: tuple[int, int], fill: int | tuple[int, int, int]) -> Image.Image:
    resample = Image.Resampling.NEAREST if image.mode == "L" else Image.Resampling.LANCZOS
    image = image.copy()
    image.thumbnail(size, resample)
    canvas = Image.new(image.mode, size, fill)
    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def prepare_mask(mask: Image.Image | None) -> tuple[Image.Image, str, list[str]]:
    if mask is None:
        return Image.new("L", TARGET_SIZE, 255), "full", ["mask_image_url missing; used full-body mask"]
    mask = resize_and_pad(mask.convert("L"), TARGET_SIZE, 0)
    mask_array = np.array(mask)
    mask_array = np.where(mask_array >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_array, mode="L"), "provided", []


def encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def normalize_seed(value: Any) -> int:
    if value is None:
        return random.randint(0, 2**31 - 1)
    return int(value)


def normalize_steps(value: Any) -> int:
    steps = int(value if value is not None else 20)
    if steps < 1 or steps > 100:
        raise ValueError("steps must be between 1 and 100")
    return steps


def normalize_guidance(value: Any) -> float:
    guidance = float(value if value is not None else 7.5)
    if guidance < 0 or guidance > 20:
        raise ValueError("guidance_scale must be between 0 and 20")
    return guidance


# SECTION 6: Inference function
def run_catvton(
    person_image: Image.Image,
    garment_image: Image.Image,
    mask_image: Image.Image,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    load_models()
    assert pipeline is not None

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        result = pipeline(
            image=person_image,
            condition_image=garment_image,
            mask=mask_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )[0]
    return result.convert("RGB")


# SECTION 7: Face restoration function
def restore_face(image: Image.Image) -> Image.Image:
    load_models()
    assert gfpgan_restorer is not None

    rgb = np.array(image.convert("RGB"))
    bgr = rgb[:, :, ::-1]
    _, _, restored_bgr = gfpgan_restorer.enhance(
        bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    if restored_bgr is None:
        raise RuntimeError("GFPGAN did not return a restored image")
    restored_rgb = restored_bgr[:, :, ::-1]
    return Image.fromarray(restored_rgb.astype(np.uint8), mode="RGB")


# SECTION 8: RunPod handler function
def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        data = job.get("input")
        if not isinstance(data, dict):
            raise ValueError("RunPod job input must be an object")

        person_url = read_input_url(data, "person_image_url", "person_image")
        garment_url = read_input_url(data, "garment_image_url", "garment_image")
        mask_url = read_input_url(data, "mask_image_url", "mask_image", required=False)
        raw_cloth_type = str(data.get("cloth_type") or "upper_body").strip().lower()
        cloth_type = CLOTH_TYPE_MAP.get(raw_cloth_type)
        if cloth_type is None:
            raise ValueError(f"Unsupported cloth_type: {raw_cloth_type}")

        steps = normalize_steps(data.get("steps"))
        guidance_scale = normalize_guidance(data.get("guidance_scale"))
        seed = normalize_seed(data.get("seed"))
    except Exception as exc:
        return error_response(str(exc), "INVALID_INPUT")

    try:
        person_image = resize_and_pad(download_image(person_url, "RGB"), TARGET_SIZE, (255, 255, 255))
        garment_image = resize_and_pad(download_image(garment_url, "RGB"), TARGET_SIZE, (255, 255, 255))
        mask_source = download_image(mask_url, "L") if mask_url else None
        mask_image, mask_type, warnings = prepare_mask(mask_source)
    except Exception as exc:
        return error_response(f"Failed to download or decode input images: {exc}", "DOWNLOAD_ERROR")

    try:
        result = run_catvton(person_image, garment_image, mask_image, steps, guidance_scale, seed)
        result = restore_face(result)
        return {
            "status": "success",
            "image_base64": encode_png_base64(result),
            "cloth_type_used": cloth_type,
            "mask_type_used": mask_type,
            "steps_used": steps,
            "seed": seed,
            "warnings": warnings,
        }
    except Exception as exc:
        return error_response(f"CatVTON inference failed: {exc}", "INFERENCE_ERROR")


# SECTION 9: RunPod server start
if __name__ == "__main__":
    load_models()
    runpod.serverless.start({"handler": handler})
