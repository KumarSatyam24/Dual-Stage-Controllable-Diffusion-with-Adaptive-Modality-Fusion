#!/usr/bin/env python3
"""
Manual checkpoint uploader - Upload specific checkpoints and delete them locally
"""

from huggingface_hub import HfApi
from pathlib import Path
import sys

def upload_and_delete(checkpoint_path, repo_id, repo_path):
    """Upload checkpoint to HF and delete locally if successful."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ File not found: {checkpoint_path}")
        return False
    
    print(f"📤 Uploading {checkpoint_path.name} to {repo_id}/{repo_path}...")
    print(f"   Size: {checkpoint_path.stat().st_size / 1e9:.2f} GB")
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ Upload successful!")
        
        # Delete local file
        print(f"🗑️  Deleting local file...")
        checkpoint_path.unlink()
        print(f"✅ Deleted {checkpoint_path.name}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"⚠️  Local file NOT deleted (keeping it safe)")
        print()
        return False

if __name__ == "__main__":
    repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
    
    checkpoints = [
        "/root/checkpoints/stage1_with_ssim/epoch_20.pt"
    ]
    
    print("🚀 Starting manual checkpoint upload and cleanup")
    print(f"   Repository: {repo_id}")
    print()
    
    success_count = 0
    for ckpt_path in checkpoints:
        repo_path = f"stage1_with_ssim/{Path(ckpt_path).name}"
        if upload_and_delete(ckpt_path, repo_id, repo_path):
            success_count += 1
    
    print(f"📊 Summary: {success_count}/{len(checkpoints)} uploaded and deleted")
    print(f"🔗 View at: https://huggingface.co/{repo_id}/tree/main/stage1_with_ssim")
