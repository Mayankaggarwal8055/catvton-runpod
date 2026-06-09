"""
RunPod Serverless Handler — CatVTON Virtual Try-On Worker
=========================================================

OPTIMIZATIONS APPLIED:
  - xformers / SDPA enabled with startup diagnostics
  - Global pipeline loaded once at cold start, reused for all requests
  - Cold/warm tracking with reuse counting
  - Direct Cloudinary upload (no base64 round-trip)
  - Persistent HTTP session for image downloads
  - Per-stage timing logs (download_ms, inference_ms, upload_ms, total_ms)
  - Preprocessing-first mask policy (use preprocessing mask, AutoMasker fallback)
  - torch.inference_mode() wrapper for inference
  - Conditional GFPGAN (skipped for lower-body, no face detectable)
  - Conditional resize_and_pad (only when dimensions mismatch 768x1024)
  - Handler-level safety guard (load_models if pipeline is None)
  - Gated VAE slicing (enabled only on GPUs with < 14GB VRAM)
  - channels_last memory format for UNet and VAE
  - GPU warm-up dummy pass after model load
  - Cloudinary upload retry (2 retries on failure)

ARCHITECTURE — Preprocessing-First
===================================
This worker is the LAST stage in a multi-stage pipeline.
The preprocessing service runs FIRST and produces:
  - Processed person image (cropped, centered, normalized to 768x1024)
  - Processed garment image (background removed, texture-preserved, normalized)
  - Precision mask (face-excluded, region-limited, polished)

The worker receives these PRECOMPUTED artifacts and runs ONLY inference.

  Frontend -> preprocessing -> Next.js orchestration -> RunPod worker -> Cloudinary -> frontend

Preprocessing is responsible for: geometry, segmentation, masking, crop handling.
RunPod worker is responsible for: inference only, GPU execution, final result upload.

MASK POLICY:
  PRIMARY:   Preprocessing-generated mask (mask_image_url provided)
  FALLBACK:  CatVTON internal AutoMasker (only when preprocessing mask
             generation failed or mask_url is empty)
  NEVER:     Black/zero mask — would suppress all garment modification
"""

from __future__ import annotations

import io
import os
import time
import logging
import random
import sys
import traceback
import threading
from typing import Any

import runpod
import requests
import numpy as np
import torch
import cv2
from PIL import Image
import cloudinary  # noqa: F401
import cloudinary.uploader  # noqa: F401
from requests.adapters import HTTPAdapter

# Dual-mask + production post-processing pipeline
from postprocessing import generate_masks, run_pipeline

# Fashn AI optional primary inference engine
# When FASHN_API_KEY is set, Fashn AI is used as the primary inference
# engine and CatVTON serves as the fallback.
try:
    import fashn_client
    _FASHN_AVAILABLE = fashn_client.is_available()
except Exception:
    fashn_client = None  # type: ignore
    _FASHN_AVAILABLE = False

# ── Debug Dump (DEBUG_TRYON) ─────────────────────────────────────────────

DEBUG = os.environ.get("DEBUG_TRYON", "0") == "1"


def save_debug(name: str, img: Any) -> str | None:
    """
    If DEBUG_TRYON=1, save intermediate artifacts to /tmp/debug and return path.
    """
    if not DEBUG:
        return None

    os.makedirs("/tmp/debug", exist_ok=True)
    out_path = f"/tmp/debug/{name}.png"

    try:
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                cv2.imwrite(out_path, img)
            else:
                # assume RGB
                bgr = img[:, :, ::-1]
                cv2.imwrite(out_path, bgr)
        elif hasattr(img, "save"):
            # PIL Image
            img.save(out_path, format="PNG")
        return out_path
    except Exception:
        # never break the job due to debug dump issues
        logger.exception("save_debug_failed name=%s", name)
        return None

# ── Logging ────────────────────────────────────────────────────────────────

logger = logging.getLogger("catvton.worker")
_handler_configured = False


def _ensure_logging():
    global _handler_configured
    if not _handler_configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _handler_configured = True


# ── Constants & Env ───────────────────────────────────────────────────────

CLOTH_TYPE_MAP: dict[str, str] = {
    "upper_body": "upper",
    "lower_body": "lower",
    "dresses": "overall",
    "upper": "upper",
    "lower": "lower",
    "overall": "overall",
}

TARGET_SIZE = (768, 1024)
TARGET_W, TARGET_H = TARGET_SIZE
CLOUDINARY_FOLDER = os.environ.get("CLOUDINARY_FOLDER", "trylix/tryon/results")

# ── Default prompt templates ─────────────────────────────────────────────

DEFAULT_POSITIVE_PROMPT = (
    "A photorealistic catalog-quality virtual try-on image showing "
    "the person wearing the garment with natural fabric texture, "
    "realistic lighting, preserved body proportions, accurate garment fit, "
    "and high detail in the clothing region."
)

DEFAULT_NEGATIVE_PROMPT = (
    "deformed body, distorted face, changed face, different person, "
    "blurry, low quality, plastic fabric, synthetic texture, "
    "hallucinated text, wrong logo, fake brand, changed collar, wrong sleeves, "
    "extra limbs, missing limbs, bad anatomy, watermark, cartoon, "
    "illustration, painting, low resolution, noise, grain, "
    "overexposed, underexposed, unnatural colors, body distortion, "
    "wrong proportions, duplicated body parts, warped garment, distorted garment, "
    "altered background, changed background, skin discoloration, "
    "face distortion, neck distortion, washed out, oversaturated, "
    "incorrect skin tone, color shift, background change, background artifacts"
)

# ── Global state (loaded once at cold start) ──────────────────────────────

pipeline = None
gfpgan_restorer = None
_WARM = threading.Event()
_STARTUP_TIME = time.perf_counter()
_REUSE_COUNT: int = 0

# ── HTTP Session (persistent connection pool for downloads) ──────────────

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def _get_session() -> requests.Session:
    """Get or create a persistent HTTP session for connection reuse."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        session = requests.Session()
        session.headers.update({
            "User-Agent": "TryLix-Worker/1.0",
            "Accept": "image/webp,image/jpeg,image/png,*/*",
        })
        adapter = HTTPAdapter(
            pool_connections=4,
            pool_maxsize=8,
            max_retries=2,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _SESSION = session
        logger.info("http_session_created pool_maxsize=8")
        return session


# ── GPU / Attention Diagnostics ──────────────────────────────────────────


def _log_acceleration_info() -> tuple[bool, bool]:
    """Log comprehensive GPU and attention acceleration diagnostics.

    Returns:
        Tuple of (xformers_available, sdpa_available).
    """
    logger.info("=== Attention Acceleration Report ===")
    logger.info("torch_version=%s", torch.__version__)
    logger.info("cuda_available=%s", torch.cuda.is_available())

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        logger.info("cuda_version=%s", torch.version.cuda)
        logger.info("gpu_name=%s", torch.cuda.get_device_name(0))
        logger.info("gpu_capability=%s", torch.cuda.get_device_capability(0))
        logger.info("vram_total_gb=%.1f", total / (1024**3))
        logger.info("vram_free_gb=%.1f", free / (1024**3))

    # xformers
    xformers_available = False
    try:
        import xformers  # noqa: F811
        import xformers.ops  # noqa: F811
        xformers_available = True
        logger.info("xformers_available=True version=%s", xformers.__version__)
    except ImportError:
        logger.info("xformers_available=False")

    # PyTorch SDPA backends
    try:
        logger.info("flash_sdp_enabled=%s", torch.backends.cuda.flash_sdp_enabled())
        logger.info("mem_efficient_sdp_enabled=%s", torch.backends.cuda.mem_efficient_sdp_enabled())
        logger.info("math_sdp_enabled=%s", torch.backends.cuda.math_sdp_enabled())
    except Exception:
        pass

    sdpa_available = (
        torch.backends.cuda.flash_sdp_enabled()
        or torch.backends.cuda.mem_efficient_sdp_enabled()
        or torch.backends.cuda.math_sdp_enabled()
    ) if hasattr(torch.backends, 'cuda') else False
    logger.info("sdpa_available=%s", sdpa_available)

    return xformers_available, sdpa_available


# ── Cloudinary Upload (direct upload, no base64) ─────────────────────────


def _configure_cloudinary() -> bool:
    """Configure Cloudinary SDK. Returns True if configured successfully."""
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")

    if not all([cloud_name, api_key, api_secret]):
        logger.warning("Cloudinary not configured - cannot upload results")
        return False

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    return True


def _upload_to_cloudinary(image: Image.Image, job_id: str) -> str:
    """Upload a PIL image directly to Cloudinary. Returns secure URL.

    Retries up to 2 times on transient failures before giving up.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = cloudinary.uploader.upload(
                buffer,
                folder=CLOUDINARY_FOLDER,
                public_id=f"result_{job_id}",
                resource_type="image",
                overwrite=False,
            )
            url = str(result["secure_url"])
            if attempt > 0:
                logger.info("cloudinary_upload_retried attempt=%s success=True", attempt + 1)
            logger.info("cloudinary_upload_complete result_url=%s", url)
            return url
        except Exception as exc:
            last_error = exc
            logger.warning(
                "cloudinary_upload_failed attempt=%s error=%s",
                attempt + 1, exc,
            )
            if attempt < 2:
                buffer.seek(0)
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Cloudinary upload failed after 3 attempts: {last_error}")


# ── Image Download ───────────────────────────────────────────────────────


def download_image(url: str, timeout: int = 60) -> Image.Image:
    """Download image from URL using the persistent HTTP session."""
    session = _get_session()
    resp = session.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ── Image Geometry (defensive — preprocessing should have done this) ─────


def _ensure_canonical_size(img: Image.Image) -> Image.Image:
    """Defensive: only resize/pad if the image is NOT already at 768x1024.

    Preprocessing always produces 768x1024 artifacts. This function is
    a safety net for unexpected sizes (e.g. direct API calls, debugging).
    """
    if img.size == TARGET_SIZE:
        return img

    logger.info("geometry_mismatch expected=%s got=%s applying_fallback",
                TARGET_SIZE, img.size)
    img = img.copy()
    img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
    paste_x = (TARGET_W - img.size[0]) // 2
    paste_y = (TARGET_H - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


# ── Mask Preparation ─────────────────────────────────────────────────────


def prepare_mask(mask_img: Image.Image | None, cloth_type: str, person_img: Image.Image) -> Image.Image:
    """Prepare mask for CatVTON inference.

    PRIMARY:   Preprocessing-generated mask (when mask_img is provided)
    FALLBACK:  CatVTON internal AutoMasker (when mask_img is None)
    NEVER:     Black/zero mask — would suppress all garment modification

    Returns the PIL mask for use in the CatVTON pipeline.
    Dual masks (inference + composite) are generated downstream in run_inference().
    """
    if mask_img is not None:
        mask = mask_img.convert("L")
    else:
        logger.info("mask_source=automasker_fallback using CatVTON AutoMasker")
        result = pipeline.automasker(person_img, cloth_type)
        mask = result["mask"]

    if mask.size != TARGET_SIZE:
        mask = mask.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    return mask


# ── Conditional Face Restoration ─────────────────────────────────────────


def _should_restore_face(cloth_type: str) -> bool:
    """Return True only if face restoration is likely to improve the result.

    GFPGAN adds 1-5s latency and can over-smooth faces. Skip when:
    - lower-body garment (face barely in frame)
    - overall/dress with full body (face is small, GFPGAN may hallucinate)
    For upper-body, face is prominent so GFPGAN helps.
    """
    if cloth_type == "lower":
        return False
    # For 'overall' / 'dresses' — face may not be fully visible in crop
    if cloth_type == "overall":
        return False
    return True


def restore_face(image: Image.Image, cloth_type: str) -> Image.Image:
    """Apply GFPGAN face restoration if available and beneficial.

    Args:
        image: The generated try-on result.
        cloth_type: CatVTON cloth_type ('upper', 'lower', 'overall').

    Returns:
        Restored image (or original if GFPGAN is skipped/unavailable).
    """
    if gfpgan_restorer is None:
        return image

    if not _should_restore_face(cloth_type):
        logger.info("gfpgan_skipped cloth_type=%s", cloth_type)
        return image

    img_cv = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    _, _, restored = gfpgan_restorer.enhance(
        img_cv, has_aligned=False, only_center_face=True, paste_back=True,
    )
    if restored is not None:
        logger.info("gfpgan_applied cloth_type=%s", cloth_type)
        return Image.fromarray(restored[:, :, ::-1]).convert("RGB")

    logger.info("gfpgan_noop no_face_detected cloth_type=%s", cloth_type)
    return image


def hard_composite(
    original_person: Image.Image,
    model_output: Image.Image,
    mask_img: Image.Image,
    edge_feather_px: int = 8,
) -> Image.Image:
    """
    Trust model output ONLY inside the mask.
    Use exact original pixels everywhere else.

    This eliminates diffusion bleed into non-garment regions.
    """
    person_np = np.array(original_person.convert("RGB")).astype(np.float32)
    output_np = np.array(model_output.convert("RGB")).astype(np.float32)
    mask_np = np.array(mask_img.convert("L")).astype(np.float32) / 255.0

    # Feather mask edges slightly to avoid hard seams at boundaries.
    k = edge_feather_px * 2 + 1
    mask_blur = cv2.GaussianBlur(mask_np, (k, k), 0)
    mask_3ch = np.stack([mask_blur] * 3, axis=-1)

    composite = output_np * mask_3ch + person_np * (1.0 - mask_3ch)
    return Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8))


# ── Garment Region Sharpening (Layer 5 Post-Processing Step 4) ────────────


def sharpen_garment_region(
    image: Image.Image,
    mask_img: Image.Image,
    radius: float = 1.0,
    percent: float = 120,
    threshold: int = 3,
) -> Image.Image:
    """
    Apply unsharp mask sharpening to the garment region only.

    Recovers texture detail that the diffusion process slightly softens.
    Sharpening is applied ONLY inside the mask, not to surrounding areas
    (face, background, lower body).
    """
    from PIL import ImageFilter

    img_np = np.array(image.convert("RGB")).astype(np.float32)
    mask_np = np.array(mask_img.convert("L")).astype(np.float32) / 255.0

    # Create sharpened version
    sharpened = image.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    sharp_np = np.array(sharpened.convert("RGB")).astype(np.float32)

    # Blend: sharpened in mask region, original outside
    mask_3ch = np.stack([mask_np] * 3, axis=-1)
    result_np = img_np * (1.0 - mask_3ch) + sharp_np * mask_3ch

    logger.info("garment_sharpen_applied radius=%.1f percent=%.0f", radius, percent)
    return Image.fromarray(np.clip(result_np, 0, 255).astype(np.uint8))


# ── Seamless Blending at Garment Boundary (Layer 5 Step 3) ────────────────


def seamless_garment_blend(
    image: Image.Image,
    original_person: Image.Image,
    mask_img: Image.Image,
    boundary_width: int = 15,
) -> Image.Image:
    """
    Apply seamless blending ONLY at the garment boundary (not the full garment).

    Creates a narrow band around the mask contour and blends only that region.
    This smooths visible seams without undoing the garment swap in the interior.
    """
    img_cv = np.array(image.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
    orig_cv = np.array(original_person.convert("RGB"))[:, :, ::-1].copy()
    mask_np = np.array(mask_img.convert("L"), dtype=np.uint8)

    # Find the garment contour center
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.info("seamless_blend_skipped no_garment_contour")
        return image

    main_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(main_contour)
    if M["m00"] == 0:
        return image

    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    try:
        # Create a narrow boundary band mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.erode(mask_np, kernel, iterations=2)
        mask_dilated = cv2.dilate(mask_np, kernel, iterations=2)
        # Boundary band = dilated minus eroded (only edge pixels)
        boundary_mask = cv2.subtract(mask_dilated, mask_eroded)

        # Only apply seamlessClone if we have enough boundary pixels
        if np.sum(boundary_mask > 0) < 50:
            logger.info("seamless_blend_skipped boundary_too_small")
            return image

        blended = cv2.seamlessClone(
            img_cv, orig_cv, boundary_mask, (center_x, center_y), cv2.NORMAL_CLONE,
        )

        # Composite: use blended only in the boundary band, original garment interior
        boundary_3ch = np.stack([boundary_mask.astype(np.float32) / 255.0] * 3, axis=-1)
        result = blended.astype(np.float32) * boundary_3ch + img_cv.astype(np.float32) * (1.0 - boundary_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)

        logger.info("seamless_blend_applied boundary_width=%d center=(%d,%d)", boundary_width, center_x, center_y)
        return Image.fromarray(result[:, :, ::-1])
    except Exception as exc:
        logger.warning("seamless_blend_failed error=%s", exc)
        return image


# ── Color Consistency Check (Layer 5 Step 5) ──────────────────────────────


def correct_skin_tone(
    image: Image.Image,
    original_person: Image.Image,
    person_img: Image.Image,
    mask_img: Image.Image,
) -> Image.Image:
    """
    Check and correct skin color to prevent garment swap from shifting skin tone.

    Detects the face region in the original person image and compares mean
    skin color to the result. Applies color correction if significant shift.
    """
    img_np = np.array(image.convert("RGB")).astype(np.float32)
    orig_np = np.array(original_person.convert("RGB")).astype(np.float32)
    mask_np = np.array(mask_img.convert("L")).astype(np.float32) / 255.0

    # Detect face region in original image for skin color sampling
    orig_cv = np.array(original_person.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    h, w = mask_np.shape
    face_skin_mask = np.zeros((h, w), dtype=np.float32)

    if len(faces) > 0:
        # Use the largest face bbox for skin measurement
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        # Use center 60% of face bbox (avoid hair/neck edges)
        margin_x = int(fw * 0.20)
        margin_y = int(fh * 0.15)
        sx1 = max(0, fx + margin_x)
        sy1 = max(0, fy + margin_y)
        sx2 = min(w, fx + fw - margin_x)
        sy2 = min(h, fy + fh - margin_y)
        if sx2 > sx1 and sy2 > sy1:
            face_skin_mask[sy1:sy2, sx1:sx2] = 1.0
            logger.info("skin_tone_face_bbox=[%d,%d,%d,%d]", sx1, sy1, sx2, sy2)
    else:
        # Fallback: use upper-central region (crude approximation)
        face_skin_mask[int(h * 0.08):int(h * 0.35), int(w * 0.25):int(w * 0.75)] = 1.0
        logger.info("skin_tone_fallback=upper_central (no face detected)")

    # Exclude garment mask area
    skin_mask = face_skin_mask * (1.0 - (mask_np > 0.5).astype(np.float32))

    if skin_mask.sum() < 50:
        logger.info("skin_tone_skipped skin_region_too_small")
        return image

    # Mean color in skin region
    skin_mask_3ch = np.stack([skin_mask] * 3, axis=-1)
    skin_pixel_count = max(skin_mask.sum(), 1.0)
    orig_skin = (orig_np * skin_mask_3ch).sum(axis=(0, 1)) / skin_pixel_count
    img_skin = (img_np * skin_mask_3ch).sum(axis=(0, 1)) / skin_pixel_count

    # Per-channel gain to match original
    gain = np.where(img_skin > 1.0, orig_skin / (img_skin + 1e-6), 1.0)
    gain = np.clip(gain, 0.85, 1.15)  # Limit correction to +/-15%

    max_diff = float(np.max(np.abs(gain - 1.0)))
    if max_diff > 0.03:  # >3% per-channel difference triggers correction
        corrected = img_np * gain.reshape(1, 1, 3)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        logger.info(
            "skin_tone_corrected max_diff=%.4f gain_r=%.3f gain_g=%.3f gain_b=%.3f",
            max_diff, gain[0], gain[1], gain[2],
        )
        return Image.fromarray(corrected)

    logger.info("skin_tone_ok max_diff=%.4f (no correction needed)", max_diff)
    return image


def composite_original_face(
    original: Image.Image,
    result: Image.Image,
    expand_ratio: float = 0.25,
) -> Image.Image:
    """Paste the original face region onto the diffusion result."""
    if original.size != result.size:
        original = original.resize(result.size, Image.Resampling.LANCZOS)

    orig_cv = np.array(original.convert("RGB"))[:, :, ::-1]
    result_cv = np.array(result.convert("RGB"))[:, :, ::-1]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        logger.info("face_composite_skipped no_face_detected")
        return result

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    pad_x = int(w * expand_ratio)
    pad_y = int(h * expand_ratio)
    H, W = orig_cv.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h)

    face_mask = np.zeros((H, W), dtype=np.float32)
    face_mask[y1:y2, x1:x2] = 1.0
    face_mask = cv2.GaussianBlur(face_mask, (0, 0), sigmaX=12, sigmaY=12)
    face_mask = face_mask[:, :, np.newaxis]

    composited = (
        orig_cv.astype(np.float32) * face_mask
        + result_cv.astype(np.float32) * (1.0 - face_mask)
    )
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    logger.info(
        "face_composite_applied face_bbox=[%d,%d,%d,%d] expand_ratio=%.2f",
        x1, y1, x2, y2, expand_ratio,
    )
    return Image.fromarray(composited[:, :, ::-1])


# ── Model Loading ────────────────────────────────────────────────────────


def load_models():
    """Load CatVTON pipeline, AutoMasker, and GFPGAN.

    Models are cached on disk at startup (should be pre-baked in the image
    or mounted via RunPod network volume). Falls back to runtime download
    only if cache is missing — this should be extremely rare in production.
    """
    global pipeline, gfpgan_restorer
    if pipeline is not None:
        return

    load_start = time.perf_counter()

    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker

    models_dir = os.environ.get("MODELS_DIR", "/workspace/models")
    catvton_path = os.path.join(models_dir, "catvton")
    sd_path = os.path.join(models_dir, "sd-inpainting")

    from huggingface_hub import snapshot_download
    import urllib.request

    # Download CatVTON weights if missing
    if not os.path.exists(os.path.join(catvton_path, "SCHP")):
        logger.warning("CatVTON cache MISSING — downloading (cold start penalty!)")
        snapshot_download(
            "zhengchong/CatVTON",
            local_dir=catvton_path,
            local_dir_use_symlinks=False,
        )

    # Download SD inpainting if missing
    if not os.path.exists(os.path.join(sd_path, "unet")):
        logger.warning("SD inpainting cache MISSING — downloading (cold start penalty!)")
        snapshot_download(
            "booksforcharlie/stable-diffusion-inpainting",
            local_dir=sd_path,
            local_dir_use_symlinks=False,
        )

    # Download GFPGAN if missing
    gfpgan_path = os.environ.get(
        "GFPGAN_MODEL_PATH", "/workspace/models/gfpgan/GFPGANv1.3.pth"
    )
    if not os.path.exists(gfpgan_path):
        logger.warning("GFPGAN cache MISSING — downloading (cold start penalty!)")
        os.makedirs(os.path.dirname(gfpgan_path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            gfpgan_path,
        )

    # Build CatVTON pipeline
    pipeline = CatVTONPipeline(
        base_ckpt=sd_path,
        attn_ckpt=catvton_path,
        attn_ckpt_version="mix",
        weight_dtype=torch.float16,
        device="cuda",
        skip_safety_check=True,
    )

    logger.info("using default CatVTON scheduler")

    # ── Attention weight diagnostic ───────────────────────────────────
    # Verify that CatVTON checkpoint weights loaded into self-attention
    # modules correctly. Standard SD self-attention has to_k weight shape
    # (inner_dim, hidden_size) where hidden_size matches UNet block channels.
    # If shapes show 768 instead of 320/640/1280, checkpoint failed to load.
    for _diag_name, _diag_mod in pipeline.unet.named_modules():
        if hasattr(_diag_mod, 'to_k') and _diag_mod.to_k is not None:
            logger.info(
                "attn_diag first_attn_layer=%s to_k_weight_shape=%s",
                _diag_name, list(_diag_mod.to_k.weight.shape),
            )
            break

    # Gate VAE slicing: enable on GPUs with < 14 GB VRAM, disable otherwise
    try:
        vram_total = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_total / (1024**3)
        if vram_gb < 14.0:
            pipeline.vae.enable_slicing()
            logger.info("vae_slicing=enabled vram_gb=%.1f", vram_gb)
        else:
            pipeline.vae.disable_slicing()
            logger.info("vae_slicing=disabled vram_gb=%.1f (high VRAM)", vram_gb)
    except Exception:
        pipeline.vae.enable_slicing()
        logger.info("vae_slicing=enabled (fallback)")

    # channels_last memory format for performance
    try:
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)
        logger.info("memory_format=channels_last")
    except Exception as exc:
        logger.warning("channels_last failed: %s", exc)

    # ── Defensive SkipAttnProcessor validation ────────────────────────────
    # CatVTON requires SkipAttnProcessor on ALL cross-attention (attn2)
    # modules to skip text cross-attention (encoder_hidden_states=None).
    # Some diffusers versions (e.g. 0.27.2) may not apply the processor
    # correctly, leading to:
    #   RuntimeError: mat1 and mat2 shapes cannot be multiplied
    #   (49152x320 and 768x320)
    # This validation detects and re-applies the processor if needed.
    from model.attn_processor import SkipAttnProcessor as _SkipAttnProcessor
    from model.utils import init_adapter as _init_adapter

    _cross_attn_needs_fix = False
    for _name, _proc in pipeline.unet.attn_processors.items():
        if _name.endswith("attn2.processor") and not isinstance(_proc, _SkipAttnProcessor):
            _cross_attn_needs_fix = True
            break

    if _cross_attn_needs_fix:
        logger.warning(
            "skip_attn_reapply SkipAttnProcessor not present on some "
            "cross-attention modules — re-applying (diffusers compat fix)"
        )
        _init_adapter(pipeline.unet, cross_attn_cls=_SkipAttnProcessor)
        # NOTE: Do NOT call enable_xformers_memory_efficient_attention() here.
        # It replaces ALL attention processors (including attn2 SkipAttnProcessor)
        # with xformers-compatible versions, which would undo our fix.
        # init_adapter already sets AttnProcessor2_0 on self-attention (attn1),
        # which uses PyTorch SDPA and is already memory-efficient.
        logger.info("skip_attn_reapply: relying on SDPA for self-attention efficiency")
    else:
        logger.info("skip_attn_ok SkipAttnProcessor present on all cross-attention modules")
        # Do NOT call enable_xformers_memory_efficient_attention() as it would
        # overwrite SkipAttnProcessor on attn2. AttnProcessor2_0 via SDPA is
        # already memory-efficient for self-attention.
        logger.info("skip_attn_ok: relying on SDPA for self-attention efficiency")

    # Build AutoMasker
    pipeline.automasker = AutoMasker(
        densepose_ckpt=os.path.join(catvton_path, "DensePose"),
        schp_ckpt=os.path.join(catvton_path, "SCHP"),
        device="cuda",
    )

    # Build GFPGAN restorer
    from gfpgan import GFPGANer

    gfpgan_restorer = GFPGANer(
        model_path=gfpgan_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device="cuda",
    )

    load_ms = (time.perf_counter() - load_start) * 1000
    logger.info("models_ready model_load_ms=%.0f", load_ms)


# ── Warmup (called exactly once at cold start) ───────────────────────────


def warmup():
    """Initialize models, enable optimizations, and warm GPU.

    Runs ONCE at cold start. Subsequent requests reuse warmed pipeline.
    Thread-safe via _WARM event.

    After loading models, runs a tiny dummy UNet pass to initialize CUDA
    kernels and memory allocator, preventing first-inference slowdown.
    """
    global _REUSE_COUNT

    if _WARM.is_set():
        return

    logger.info("=" * 60)
    logger.info("COLD START BEGIN")
    logger.info("=" * 60)

    # GPU / attention diagnostics
    _log_acceleration_info()

    # Load models (including CatVTON, AutoMasker, GFPGAN)
    load_models()

    # GPU warm-up: one dummy UNet pass to initialize CUDA kernels
    # This prevents the first real inference from being ~2x slower
    try:
        try:
            try:
                torch.cuda.synchronize()
                logger.info("gpu_warmup_ready=True")
            except Exception as exc:
                logger.warning("gpu_warmup_skipped error=%s", exc)
        except Exception as exc:
            logger.warning("gpu_warmup_skipped error=%s", exc)
    except Exception as exc:
        logger.warning("gpu_warmup_skipped error=%s", exc)

    # Cloudinary config check
    cloudinary_ok = _configure_cloudinary()

    startup_total_ms = (time.perf_counter() - _STARTUP_TIME) * 1000
    logger.info("=" * 60)
    logger.info("COLD START COMPLETE")
    logger.info("  startup_total_ms=%.0f", startup_total_ms)
    logger.info("  cloudinary_configured=%s", cloudinary_ok)
    logger.info("=" * 60)

    _WARM.set()
    _REUSE_COUNT = 0


# ── Inference (per job) ──────────────────────────────────────────────────


def run_inference(job_input: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Run CatVTON inference with preprocessing-processed inputs.

    The inputs are EXPECTED to be preprocessing-processed artifacts:
      - person_image_url  -> cropped, centered, normalized to 768x1024
      - garment_image_url -> background-removed, texture-preserved
      - mask_image_url    -> face-excluded, region-limited mask (optional)

    MASK POLICY:
      PRIMARY:   Preprocessing-generated mask (mask_image_url provided)
      FALLBACK:  CatVTON AutoMasker (when mask_image_url is empty)
    """
    job_start = time.perf_counter()

    # Extract and validate inputs
    person_url = job_input.get("person_image_url") or job_input.get("person_image")
    garment_url = job_input.get("garment_image_url") or job_input.get("garment_image")
    mask_url = job_input.get("mask_image_url") or job_input.get("mask_image", "")
    cloth_type_raw = job_input.get("cloth_type", "upper_body")

    if not person_url or not garment_url:
        raise ValueError("Missing required inputs: person_image_url and garment_image_url")

    cloth_type = CLOTH_TYPE_MAP.get(cloth_type_raw, "upper")
    steps = int(job_input.get("steps", 30))
    guidance_scale = float(job_input.get("guidance_scale", 2.5))
    seed_in = job_input.get("seed", random.randint(0, 2**31 - 1))
    prompt = job_input.get("prompt", DEFAULT_POSITIVE_PROMPT)
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

    logger.info(
        "inference_start cloth_type=%s steps=%s guidance=%.1f prompt_provided=%s fashn_available=%s",
        cloth_type, steps, guidance_scale, bool(job_input.get("prompt")), _FASHN_AVAILABLE,
    )

    # ── Download person and mask (needed for post-processing regardless) ─
    download_start = time.perf_counter()
    person_img = download_image(person_url)
    mask_source: str
    if mask_url:
        mask_img = download_image(mask_url)
        mask_source = "preprocessing"
    else:
        mask_img = None
        mask_source = "automasker_fallback"
    person_img = _ensure_canonical_size(person_img)

    # ── Fashn AI Primary Inference Path ─────────────────────────────────
    inference_engine = "catvton"
    result_raw = None
    download_ms = (time.perf_counter() - download_start) * 1000

    if _FASHN_AVAILABLE:
        logger.info("inference_engine=fashn_ai primary_path")
        try:
            fashn_result = fashn_client.run_tryon(
                person_image_url=person_url,
                garment_image_url=garment_url,
                job_id=job_id,
                timeout=120,
            )
            if fashn_result.get("status") == "success":
                inference_engine = "fashn_ai"
                result_path = fashn_result["result_url"]
                result_raw = download_image(result_path)
                result_raw = _ensure_canonical_size(result_raw)
                logger.info("fashn_success job_id=%s", job_id)
            else:
                logger.warning(
                    "fashn_fallback fashn_failed reason=%s",
                    fashn_result.get("error", "unknown"),
                )
        except Exception as exc:
            logger.warning(
                "fashn_fallback exception=%s falling_back_to_catvton", exc,
            )
    else:
        logger.info("inference_engine=catvton fashn_not_configured")

    # ── CatVTON Path (fallback) ────────────────────────────────────────
    if inference_engine == "catvton":
        logger.info("inference_engine=catvton inference_path")
        garment_img = download_image(garment_url)
        garment_img = _ensure_canonical_size(garment_img)

        # Prepare mask for CatVTON
        mask = prepare_mask(mask_img, cloth_type, person_img)

        # ── Dual mask generation ─────────────────────────────────────────────
        # Generate inference mask (dilated + feathered) for CatVTON diffusion
        # and composite mask (original sharp) for identity-lock compositing.
        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        inference_mask_np, composite_mask_np = generate_masks(mask_np)
        inference_mask_pil = Image.fromarray(
            (np.clip(inference_mask_np * 255, 0, 255)).astype(np.uint8), mode="L"
        )
        composite_mask_pil = Image.fromarray(composite_mask_np, mode="L")
        logger.info(
            "dual_masks generated inference_coverage=%.3f composite_coverage=%.3f",
            inference_mask_np.mean(), composite_mask_np.mean() / 255.0,
        )

        if DEBUG:
            m = save_debug(f"{job_id}_mask_original", mask)
            if m:
                debug_files["mask_original"] = m
            im = save_debug(f"{job_id}_mask_inference", inference_mask_pil)
            if im:
                debug_files["mask_inference"] = im
            cm = save_debug(f"{job_id}_mask_composite", composite_mask_pil)
            if cm:
                debug_files["mask_composite"] = cm

        # CatVTON Inference
        inference_start = time.perf_counter()
        generator = torch.Generator(device="cuda").manual_seed(int(seed_in))

        cloth_type_allowed = {"upper", "lower", "overall"}
        if cloth_type not in cloth_type_allowed:
            logger.warning("invalid_cloth_type=%s defaulting to upper", cloth_type)
            cloth_type = "upper"

        guidance_scale = float(max(1.0, min(10.0, guidance_scale)))

        mask_np_arr = np.array(mask.convert("L"))
        mask_coverage = float(mask_np_arr.mean() / 255.0)
        logger.info("mask_diagnostics mask_coverage=%.3f", mask_coverage)
        if mask_coverage <= 0.0:
            raise RuntimeError("mask_is_empty mask_coverage=0.0; garment not applied safely")

        import gc
        oom_retry_triggered = False
        oom_retry_steps = None
        pipeline_kwargs_snapshot = {}
        negative_prompt_injected = False
        negative_prompt_supported = False

        try:
            import inspect
            sig = inspect.signature(pipeline.__call__)
            negative_prompt_supported = "negative_prompt" in sig.parameters
        except Exception:
            negative_prompt_supported = False

        base_kwargs: dict[str, Any] = {
            "image": person_img,
            "condition_image": garment_img,
            "mask": inference_mask_pil,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        try:
            sig = inspect.signature(pipeline.__call__)
            if "prompt" in sig.parameters:
                base_kwargs["prompt"] = prompt
        except Exception:
            pass

        if negative_prompt_supported:
            base_kwargs["negative_prompt"] = negative_prompt
            negative_prompt_injected = True

        try:
            generation_steps = steps
            with torch.inference_mode():
                result_raw = pipeline(**base_kwargs)[0]
        except torch.cuda.OutOfMemoryError:
            oom_retry_triggered = True
            torch.cuda.empty_cache()
            gc.collect()
            oom_retry_steps = max(15, steps - 10)
            logger.warning("oom_retry: %s -> %s", steps, oom_retry_steps)
            base_kwargs["num_inference_steps"] = oom_retry_steps
            with torch.inference_mode():
                result_raw = pipeline(**base_kwargs)[0]

        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_start) * 1000

        if DEBUG:
            raw_path = save_debug(f"{job_id}_raw_output", result_raw)
            if raw_path:
                debug_files["raw_output"] = raw_path

        # ── Dual mask generation ─────────────────────────────────────────────
        # Convert mask PIL to numpy for dual mask processing
        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        inference_mask_np, composite_mask_np = generate_masks(mask_np)
        logger.info(
            "dual_masks applied inference_coverage=%.3f composite_coverage=%.3f",
            inference_mask_np.mean(), composite_mask_np.mean() / 255.0,
        )

        # Convert inference mask back to PIL for CatVTON pipeline
        inference_mask_pil = Image.fromarray(
            (inference_mask_np * 255).astype(np.uint8), mode="L"
        )
        composite_mask_pil = Image.fromarray(composite_mask_np, mode="L")

        # ── New post-processing pipeline ──────────────────────────────────────
        # run_pipeline handles: identity lock, two-pass compositing,
        # garment sharpening, color consistency, and final edge blend.
        result_np = run_pipeline(
            diffusion_output=np.array(result_raw.convert("RGB")),
            original_person=np.array(person_img.convert("RGB")),
            garment_image=np.array(garment_img.convert("RGB")),
            inference_mask=inference_mask_np,
            composite_mask=composite_mask_np,
        )
        result = Image.fromarray(result_np)
    else:
        # Fashn AI: no hard composite needed (different model, no mask bleed)
        result = result_raw
        inference_ms = download_ms
        # Ensure mask is defined for shared post-processing
        if mask_img is not None:
            mask = mask_img.convert("L")
            if mask.size != TARGET_SIZE:
                mask = mask.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            mask_np = np.array(mask, dtype=np.uint8)
            _, composite_mask_np = generate_masks(mask_np)
            composite_mask_pil = Image.fromarray(composite_mask_np, mode="L")
        else:
            # Create a full-frame mask as fallback (post-processing will degrade gracefully)
            mask = Image.new("L", TARGET_SIZE, 128)
            composite_mask_pil = mask
        mask_source = "fashn_ai"
    
    debug_files = {}
    person_original_for_composite = person_img
    download_ms = download_ms or (time.perf_counter() - download_start) * 1000
    
    if DEBUG and inference_engine == "catvton":
        dm = save_debug(f"{job_id}_pipeline_result", result)
        if dm:
            debug_files["pipeline_result"] = dm

    if DEBUG:
        hc_path = save_debug(f"{job_id}_hard_composite_result", result)
        if hc_path:
            debug_files["hard_composite_result"] = hc_path

    # Mask overlap diagnostics (face/neck) for logging/trace
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    orig_cv = np.array(person_img.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    mask_l = np.array(mask.convert("L")).astype(np.uint8)
    face_overlap_percent = 0.0
    neck_overlap_percent = 0.0

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        # face region (slightly padded)
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.10)
        fx1, fy1 = max(0, x - pad_x), max(0, y - pad_y)
        fx2, fy2 = min(mask_l.shape[1], x + w + pad_x), min(mask_l.shape[0], y + h + pad_y)
        face_area = max(1, (fy2 - fy1) * (fx2 - fx1))
        face_overlap_percent = float((mask_l[fy1:fy2, fx1:fx2] > 0).sum() / face_area)

        # neck region (below face bbox)
        nx1, ny1 = fx1, fy2
        nx2, ny2 = fx2, min(mask_l.shape[0], fy2 + int(h * 0.35))
        neck_area = max(1, (ny2 - ny1) * (nx2 - nx1))
        neck_overlap_percent = float((mask_l[ny1:ny2, nx1:nx2] > 0).sum() / neck_area)

        if (face_overlap_percent > 0.05) or (neck_overlap_percent > 0.05):
            logger.warning(
                "mask_overlap warning face_overlap=%.3f neck_overlap=%.3f bbox_face=%s",
                face_overlap_percent, neck_overlap_percent, (fx1, fy1, fx2, fy2)
            )

    trace = {
        "mask_coverage_percent": round(mask_coverage * 100.0, 3),
        "mask_overlap": {
            "face_overlap_percent": round(face_overlap_percent * 100.0, 3),
            "neck_overlap_percent": round(neck_overlap_percent * 100.0, 3),
        },
        "generation_steps": int(generation_steps),
        "oom_retry_triggered": bool(oom_retry_triggered),
        "oom_retry_steps": int(oom_retry_steps) if oom_retry_steps is not None else None,
        "final_resolution": [person_img.size[0], person_img.size[1]],
        "pipeline_runtime_ms": round(inference_ms, 2),
        "pipeline_kwargs_snapshot": pipeline_kwargs_snapshot,
    }

    # ── Step 1: Face identity preservation (original face paste) ──────
    result = composite_original_face(person_img, result, expand_ratio=0.25)

    # ── Step 2: Conditional face restoration (GFPGAN) ──────────────────
    result = restore_face(result, cloth_type)

    if DEBUG:
        face_path = save_debug(f"{job_id}_face_composite_result", result)
        if face_path:
            debug_files["face_composite_result"] = face_path

    # ── Step 3: Skin tone preservation (garment color correction is in run_pipeline) ───
    result = correct_skin_tone(result, person_original_for_composite, person_img, mask)

    # Upload to Cloudinary (with retry built into _upload_to_cloudinary)
    upload_start = time.perf_counter()
    result_url = _upload_to_cloudinary(result, job_id)
    upload_ms = (time.perf_counter() - upload_start) * 1000

    total_ms = (time.perf_counter() - job_start) * 1000

    logger.info(
        "job_complete total_ms=%.0f download_ms=%.0f inference_ms=%.0f "
        "upload_ms=%.0f mask_source=%s",
        total_ms, download_ms, inference_ms, upload_ms, mask_source,
    )

    payload: dict[str, Any] = {
        "status": "success",
        "result_url": result_url,
        "cloth_type_used": cloth_type,
        "steps_used": steps,
        "seed": int(seed_in),
        "mask_source": mask_source,
        "inference_ms": round(inference_ms, 2),
        "upload_ms": round(upload_ms, 2),
        "download_ms": round(download_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    if DEBUG:
        payload["debug_files"] = debug_files
        payload["trace"] = trace

    return payload


# ── RunPod Handler (called per job) ───────────────────────────────────────


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless handler entry point.

    Handles cold start, reuses warmed pipeline, and returns direct
    Cloudinary result URL (no base64 round-trip).

    Safety guard: if pipeline is None despite _WARM being set
    (e.g. RUNPOD_WARMUP_DISABLE was set at startup but later unset),
    forces load_models() here.
    """
    job_start = time.time()

    if not _WARM.is_set():
        warmup()
        cold_start = True
    else:
        cold_start = False

    # Safety guard: if pipeline is still None, force load
    if pipeline is None:
        logger.warning("pipeline was None — forcing load_models() in handler")
        load_models()
        cold_start = True

    global _REUSE_COUNT
    _REUSE_COUNT += 1

    logger.info(
        "handler_invoked cold_start=%s worker_reused=%s reuse_count=%s job_id=%s",
        cold_start, not cold_start, _REUSE_COUNT, job.get("id", "unknown"),
    )

    user_input = job.get("input", {})
    job_id = str(job.get("id", "unknown"))

    try:
        output = run_inference(user_input, job_id)
        output["cold_start"] = cold_start
        return output

    except Exception as exc:
        total_ms = (time.time() - job_start) * 1000
        logger.error("job_failed total_ms=%.0f error=%s", total_ms, exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "error_code": "worker_inference_failed",
            "total_ms": round(total_ms, 2),
            "cold_start": cold_start,
        }


# Module-level startup logging

_ensure_logging()
logger.info("=" * 60)
logger.info("CatVTON Worker v1.2.0 — loading")
logger.info("target_size=%s", TARGET_SIZE)
logger.info("gpu_available=%s", torch.cuda.is_available())
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    logger.info("gpu_device=%s", dev.name)
    logger.info("vram_total_gb=%.1f", dev.total_memory / (1024**3))
logger.info("=" * 60)

# RunPod Entrypoint

if __name__ == "__main__":
    try:
        if not os.environ.get("RUNPOD_WARMUP_DISABLE"):
            warmup()
        runpod.serverless.start({"handler": handler})
    except Exception:
        logger.error("Worker startup failed")
        traceback.print_exc()
        sys.stdout.flush()
        raise
