from huggingface_hub import snapshot_download
import os

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

print("Downloading SD inpainting...")
snapshot_download(
    repo_id="booksforcharlie/stable-diffusion-inpainting",
    local_dir="/workspace/models/sd-inpainting",
    token=token
)

print("Downloading CatVTON weights...")
snapshot_download(
    repo_id="zheng-chong/CatVTON",
    local_dir="/workspace/models/catvton",
    token=token
)

print("All models downloaded.")