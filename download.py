from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="stable-diffusion-v1-5",
    local_dir_use_symlinks=False,
    resume_download=True,
    # mirror = "https://hf-mirror.com"
)