from huggingface_hub import snapshot_download
import os
import sys

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

if token:
    print("[models] HF token found")
else:
    print("[models] WARNING: No HF token — may hit rate limits")

try:
    print("[models] Downloading stable-diffusion-inpainting...")
    snapshot_download(
        repo_id="booksforcharlie/stable-diffusion-inpainting",
        local_dir="/workspace/models/sd-inpainting",
        local_dir_use_symlinks=False,   # ✅ required in Docker — symlinks fail in layers
        token=token
    )
    print("[models] SD inpainting OK")
except Exception as e:
    print(f"[models] FAILED: {e}", file=sys.stderr)
    sys.exit(1)

try:
    print("[models] Downloading CatVTON weights...")
    snapshot_download(
        repo_id="zhengchong/CatVTON",
        local_dir="/workspace/models/catvton",
        local_dir_use_symlinks=False,   # ✅ same
        token=token
    )
    print("[models] CatVTON OK")
except Exception as e:
    print(f"[models] FAILED: {e}", file=sys.stderr)
    sys.exit(1)

print("[models] All done")