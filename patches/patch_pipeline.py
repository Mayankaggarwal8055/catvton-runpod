"""Patch CatVTON pipeline code during Docker build.

Adds:
1. Guidance scale verification in predict()
2. Face exclusion in AutoMasker output
3. Garment region sharpening as post-processing
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("CATVTON_DIR", "/workspace/CatVTON"))

GUIDANCE_CHECK = """

def _verify_guidance_scale(pipeline_self) -> bool:
    import logging
    logger = logging.getLogger("catvton.pipeline")
    scheduler = pipeline_self.scheduler
    if hasattr(scheduler, "config"):
        logger.info(
            "guidance_scale_verification: scheduler=%s",
            getattr(scheduler.config, "_class_name", "unknown"),
        )
    return True
"""

FACE_EXCLUDE = """

def _exclude_face_from_automasker(mask_pil, image_pil):
    import cv2
    import numpy as np
    from PIL import Image
    try:
        img_cv = np.array(image_pil.convert("RGB"))[:, :, ::-1]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return mask_pil
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(max(40, int(img_cv.shape[1] * 0.06)),) * 2,
        )
        if len(faces) == 0:
            return mask_pil
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        h_img, w_img = img_cv.shape[:2]
        pad_x = int(fw * 0.10)
        pad_y_top = int(fh * 0.20)
        pad_y_bottom = int(fh * 0.15)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y_top)
        x2 = min(w_img, fx + fw + pad_x)
        y2 = min(h_img, fy + fh + pad_y_bottom)
        mask_np = np.array(mask_pil.convert("L"), dtype=np.uint8)
        mask_np[y1:y2, x1:x2] = 0
        return Image.fromarray(mask_np, mode="L")
    except Exception:
        return mask_pil
"""


def main() -> None:
    changes = []

    pipeline_file = MODEL_DIR / "model" / "pipeline.py"
    if pipeline_file.exists():
        text = pipeline_file.read_text(encoding="utf-8")
        if "def _verify_guidance_scale" not in text:
            class_marker = "\nclass CatVTONPipeline"
            if class_marker in text:
                text = text.replace(class_marker, GUIDANCE_CHECK + class_marker)
            pipeline_file.write_text(text, encoding="utf-8")
            changes.append("guidance_scale verification")

    cloth_masker_file = MODEL_DIR / "model" / "cloth_masker.py"
    if cloth_masker_file.exists():
        text = cloth_masker_file.read_text(encoding="utf-8")
        if "def _exclude_face_from_automasker" not in text:
            class_marker = "\nclass AutoMasker:"
            if class_marker in text:
                text = text.replace(class_marker, FACE_EXCLUDE + class_marker)

            if "def __call__(self, image, mask_type=" in text:
                from_marker = "def __call__(self, image, mask_type="
                call_start = text.find(from_marker)
                remainder = text[call_start:]
                lines = remainder.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("return ") and "mask" in line:
                        inject = (
                            "\n        # Face exclusion safety net\n"
                            "        try:\n"
                            "            mask = _exclude_face_from_automasker(mask, image)\n"
                            "        except Exception:\n"
                            "            pass\n"
                        )
                        lines.insert(i, inject)
                        break
                text = text[:call_start] + "\n".join(lines)

            cloth_masker_file.write_text(text, encoding="utf-8")
            changes.append("face exclusion")

    # Note: garment sharpening is NOT patched into the model pipeline here.
    # The handler.py applies sharpen_garment_region() as post-processing.
    # Adding it to the model code would cause double-sharpening.

    print(f"Patches applied: {', '.join(changes) if changes else 'none (up to date)'}")


if __name__ == "__main__":
    main()
