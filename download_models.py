from huggingface_hub import snapshot_download
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/workspace/models"))
GFPGAN_DIR = MODELS_DIR / "gfpgan"
GFPGAN_PATH = GFPGAN_DIR / "GFPGANv1.3.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

def download_models():
    print("[download] CatVTON weights...")
    snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir=str(MODELS_DIR / "catvton"),
        local_dir_use_symlinks=False,
    )

    print("[download] SD inpainting base model...")
    snapshot_download(
        repo_id="booksforcharlie/stable-diffusion-inpainting",
        local_dir=str(MODELS_DIR / "sd-inpainting"),
        local_dir_use_symlinks=False,
    )

    print("[download] GFPGAN...")
    GFPGAN_DIR.mkdir(parents=True, exist_ok=True)
    if not GFPGAN_PATH.exists():
        urllib.request.urlretrieve(GFPGAN_URL, GFPGAN_PATH)
        print(f"[download] GFPGAN saved to {GFPGAN_PATH}")
    else:
        print("[download] GFPGAN already present")

if __name__ == "__main__":
    download_models()
    print("All models downloaded.")