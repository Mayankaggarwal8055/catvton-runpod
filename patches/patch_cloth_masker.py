"""Patch CatVTON AutoMasker fallback masks during the Docker build."""

from __future__ import annotations

import os
from pathlib import Path


TARGET = Path(os.environ.get("CATVTON_CLOTH_MASKER", "/workspace/CatVTON/model/cloth_masker.py"))

HELPER = '''

def postprocess_automasker_mask(mask_pil: Image.Image, cloth_type: str) -> Image.Image:
    """Match preprocessing mask cleanup for AutoMasker fallback masks."""
    mask_np = np.array(mask_pil.convert("L"), dtype=np.uint8)

    if cloth_type == "upper":
        face_cutoff = int(mask_np.shape[0] * 0.25)
        mask_np[:face_cutoff, :] = 0

    kernel = np.ones((3, 3), np.uint8)
    mask_np = cv2.erode(mask_np, kernel, iterations=2)
    mask_np = cv2.dilate(mask_np, kernel, iterations=3)
    mask_np = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=2)
    return Image.fromarray(mask_np.astype(np.uint8))
'''

CALL_BEFORE = """        return {
            'mask': mask,
"""

CALL_AFTER = """        mask = postprocess_automasker_mask(mask, mask_type)
        return {
            'mask': mask,
"""


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if "def postprocess_automasker_mask" not in text:
        text = text.replace("\nclass AutoMasker:", f"{HELPER}\nclass AutoMasker:")
    if "postprocess_automasker_mask(mask, mask_type)" not in text:
        text = text.replace(CALL_BEFORE, CALL_AFTER)
    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
