#!/usr/bin/env python
"""Download checkpoint from HuggingFace Hub"""
from huggingface_hub import hf_hub_download
import os

repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
filename = "stage1_improved/epoch_12.pt"

print(f"📥 Downloading {filename} from HuggingFace...")

# Create directory
os.makedirs("/root/checkpoints/stage1_improved", exist_ok=True)

# Download
local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir="/root/checkpoints",
    local_dir_use_symlinks=False
)

print(f"✅ Downloaded to: {local_path}")

# Verify
file_size_gb = os.path.getsize(local_path) / (1024**3)
print(f"📊 File size: {file_size_gb:.2f} GB")
