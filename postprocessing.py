"""
Dual Mask + Production Post-Processing Pipeline for CatVTON RunPod Worker

DUAL MASK STRATEGY:
  INFERENCE MASK (dilated 35px + feathered 15px):
    - Passed to CatVTON diffusion model as the inpainting mask
    - Dilation gives the model ~35px of drawing space beyond garment boundary
    - Prevents hemline leakage, orange strip artifact, and latent bleed
    - Feathering creates soft edges that blend naturally

  COMPOSITE MASK (original sharp):
    - Used for hard compositing after inference
    - Restores original person pixels EXACTLY outside the garment boundary
    - Prevents torso drift, background modifications, and face distortion

POST-PROCESSING PIPELINE (in order):
  1. Identity Lock: Force original person pixels back into non-mask regions
  2. Two-Pass Compositing: Hard composite + alpha blend with feathered mask
  3. Garment Sharpening: Unsharp mask only in garment region (recovers texture)
  4. Color Consistency: Match garment region color to original garment image
  5. Final Edge Feathering: Soft blend at composite boundary
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("catvton.postprocessing")

# Constants
DILATION_KERNEL = 5
DILATION_ITERATIONS = 7
FEATHER_SIGMA = 15
FEATHER_KERNEL = 21
SHARPEN_RADIUS = 1.0
SHARPEN_PERCENT = 120
SHARPEN_THRESHOLD = 3
COLOR_GAIN_CLAMP = 0.15


def generate_masks(
    base_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate both masks from a single base mask.

    Args:
        base_mask: Original sharp mask (HxW, uint8 0-255)

    Returns:
        inference_mask: Dilated + feathered (HxW, float32 0-1)
        composite_mask: Sharp binary (HxW, uint8 0-255)
    """
    if base_mask is None or base_mask.size == 0:
        raise ValueError("base_mask cannot be empty")

    # Clamp to binary
    _, base_binary = cv2.threshold(base_mask, 24, 255, cv2.THRESH_BINARY)

    # --- Composite mask (original sharp) ---
    composite_mask = base_binary.copy()

    # --- Inference mask (dilated + feathered) ---
    kernel = np.ones((DILATION_KERNEL, DILATION_KERNEL), np.uint8)
    dilated = cv2.dilate(base_binary, kernel, iterations=DILATION_ITERATIONS)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_k)

    ksize = FEATHER_KERNEL if FEATHER_KERNEL % 2 == 1 else FEATHER_KERNEL + 1
    feathered = cv2.GaussianBlur(
        dilated.astype(np.float32), (ksize, ksize), FEATHER_SIGMA
    )
    inference_mask = feathered / 255.0

    coverage = float(inference_mask.mean())
    logger.info(
        "dual_masks generated inference_dilation=%dpx "
        "feather_sigma=%.1f coverage=%.1f%%",
        DILATION_KERNEL * DILATION_ITERATIONS,
        FEATHER_SIGMA,
        coverage * 100.0,
    )
    if coverage < 0.08 or coverage > 0.50:
        logger.warning(
            "inference_mask_coverage=%.2f outside range", coverage * 100.0
        )

    return inference_mask, composite_mask


def identity_lock(
    diffusion: np.ndarray,
    original: np.ndarray,
    composite_mask: np.ndarray,
) -> np.ndarray:
    """
    Restore original pixels outside the composite mask.

    Critical safeguard against torso drift.
    """
    try:
        if diffusion.shape != original.shape:
            raise ValueError(f"Shape mismatch: {diffusion.shape} vs {original.shape}")
        m = composite_mask.astype(np.float32) / 255.0
        locked = diffusion.astype(np.float32) * m[..., np.newaxis] \
               + original.astype(np.float32) * (1.0 - m[..., np.newaxis])
        logger.info("identity_lock applied mask_mean=%.3f", m.mean())
        return locked.astype(np.uint8)
    except Exception as e:
        logger.error("identity_lock failed: %s", e)
        return diffusion


def sharpen_garment(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Sharpen only the garment region to recover diffusion-softened texture."""
    try:
        pil_img = Image.fromarray(image)
        sharp = pil_img.filter(
            ImageFilter.UnsharpMask(
                radius=SHARPEN_RADIUS, percent=SHARPEN_PERCENT, threshold=SHARPEN_THRESHOLD
            )
        )
        sharp_np = np.array(sharp, dtype=np.float32)
        m = mask.astype(np.float32) / 255.0
        result = image.astype(np.float32) * (1.0 - m[..., np.newaxis]) \
               + sharp_np * m[..., np.newaxis]
        logger.info("garment_sharpen applied")
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.error("garment_sharpen failed: %s", e)
        return image


def correct_color(
    output: np.ndarray,
    garment: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Correct garment color drift via per-channel gain (clamped +/- 15%)."""
    try:
        m = mask.astype(np.float32) / 255.0
        m_sum = max(m.sum(), 1.0)
        m_3ch = m[..., np.newaxis]

        g_mean = (garment.astype(np.float32) * m_3ch).sum(axis=(0, 1)) / m_sum
        o_mean = (output.astype(np.float32) * m_3ch).sum(axis=(0, 1)) / m_sum

        gain = np.where(o_mean > 1.0, g_mean / (o_mean + 1e-6), 1.0)
        gain = np.clip(gain, 1.0 - COLOR_GAIN_CLAMP, 1.0 + COLOR_GAIN_CLAMP)

        max_diff = float(np.max(np.abs(gain - 1.0)))
        if max_diff <= 0.03:
            logger.info("color_ok max_diff=%.4f", max_diff)
            return output

        corrected = output.astype(np.float32) * gain.reshape(1, 1, 3)
        logger.info(
            "color_corrected max_diff=%.4f gain=(%.3f,%.3f,%.3f)",
            max_diff, gain[0], gain[1], gain[2],
        )
        return np.clip(corrected, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.error("color_correct failed: %s", e)
        return output


def alpha_blend(
    fg: np.ndarray,
    bg: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Alpha blend foreground onto background."""
    result = fg.astype(np.float32) * alpha[..., np.newaxis] \
           + bg.astype(np.float32) * (1.0 - alpha[..., np.newaxis])
    return np.clip(result, 0, 255).astype(np.uint8)


def run_pipeline(
    diffusion_output: np.ndarray,
    original_person: np.ndarray,
    garment_image: np.ndarray,
    inference_mask: np.ndarray,
    composite_mask: np.ndarray,
) -> np.ndarray:
    """
    Full post-processing pipeline (5 steps in order).

    1. Identity Lock
    2. Two-Pass Compositing (hard composite + alpha blend)
    3. Garment Sharpening
    4. Color Consistency
    5. Final soft edge blend
    """
    try:
        # Step 1: Identity Lock
        s1 = identity_lock(diffusion_output, original_person, composite_mask)

        # Step 2: Two-Pass Compositing
        cm = composite_mask.astype(np.float32) / 255.0
        hard = (
            diffusion_output.astype(np.float32) * cm[..., np.newaxis]
            + original_person.astype(np.float32) * (1.0 - cm[..., np.newaxis])
        )
        s2 = alpha_blend(hard, s1, inference_mask)

        # Step 3: Garment Sharpening
        s3 = sharpen_garment(s2, composite_mask)

        # Step 4: Color Consistency
        s4 = correct_color(s3, garment_image, composite_mask)

        # Step 5: Final soft edge
        final = alpha_blend(s4, s1, inference_mask)

        logger.info("pipeline completed")
        return final

    except Exception as e:
        logger.error("pipeline failed: %s", e)
        return identity_lock(diffusion_output, original_person, composite_mask)
