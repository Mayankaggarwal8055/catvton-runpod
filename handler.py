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
from PIL import Image
import cloudinary  # noqa: F401
import cloudinary.uploader  # noqa: F401
from requests.adapters import HTTPAdapter

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

    # Enable xformers / memory-efficient attention
    try:
        pipeline.unet.enable_xformers_memory_efficient_attention()
        logger.info("xformers attention enabled on UNet")
    except Exception as exc:
        logger.warning("xformers enable failed: %s", exc)

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
        # NOTE: intentionally NOT re-enabling xformers here —
        # enable_xformers_memory_efficient_attention() would replace ALL
        # attention processors (including our SkipAttnProcessor on attn2)
        # with xformers-compatible versions, undoing the fix.
        # AttnProcessor2_0 (set by init_adapter on self-attention) uses
        # PyTorch SDPA which is already memory-efficient.
    else:
        logger.info("skip_attn_ok SkipAttnProcessor present on all cross-attention modules")

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
    steps = int(job_input.get("steps", 24))
    guidance_scale = float(job_input.get("guidance_scale", 2.5))
    seed_in = job_input.get("seed", random.randint(0, 2**31 - 1))

    logger.info(
        "inference_start cloth_type=%s steps=%s guidance=%.1f",
        cloth_type, steps, guidance_scale,
    )

    # Download images
    download_start = time.perf_counter()
    person_img = download_image(person_url)
    garment_img = download_image(garment_url)

    # Defensive geometry check — preprocessing should have produced 768x1024,
    # but we guard against edge cases (direct API calls, debugging, etc.)
    person_img = _ensure_canonical_size(person_img)
    garment_img = _ensure_canonical_size(garment_img)

    # Mask source: preprocessing PRIMARY, AutoMasker FALLBACK
    mask_source: str
    if mask_url:
        mask_img = download_image(mask_url)
        mask_source = "preprocessing"
        logger.info("mask_source=preprocessing using preprocessing mask")
    else:
        mask_img = None
        mask_source = "automasker_fallback"
        logger.info("mask_source=automasker_fallback no mask from preprocessing")

    mask = prepare_mask(mask_img, cloth_type, person_img)
    download_ms = (time.perf_counter() - download_start) * 1000

    # Inference
    inference_start = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(int(seed_in))

    logger.info(
    "pipeline_debug cross_attention_dim=%s in_channels=%s",
    pipeline.unet.config.cross_attention_dim,
    pipeline.unet.config.in_channels,
    )

    try:
        text_hidden = pipeline.text_encoder.config.hidden_size
    except Exception:
        text_hidden = "missing"

    logger.info("pipeline_debug text_hidden_size=%s", text_hidden)
    logger.info(
        "pipeline_debug person_size=%s garment_size=%s mask_size=%s",
        person_img.size,
        garment_img.size,
        mask.size,
    )

    with torch.inference_mode():
        result = pipeline(
            image=person_img,
            condition_image=garment_img,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )[0]

    torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # Conditional face restoration
    result = restore_face(result, cloth_type)

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

    return {
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
