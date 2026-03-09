#!/usr/bin/env python3
"""
Automatic Checkpoint Uploader to HuggingFace Hub

Monitors checkpoint directory and automatically uploads new checkpoints
to HuggingFace Hub without interrupting training.

Usage:
    python auto_upload_checkpoints.py &
    
Author: RAGAF-Diffusion Research Team
Date: March 9, 2026
"""

import os
import time
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import argparse

def upload_checkpoint(api, repo_id, local_path, repo_path):
    """Upload a single checkpoint to HuggingFace."""
    try:
        print(f"📤 Uploading {local_path.name} to {repo_id}/{repo_path}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ Uploaded {local_path.name} successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {local_path.name}: {e}")
        return False

def monitor_and_upload(
    checkpoint_dir="/root/checkpoints/stage1_with_ssim",
    repo_id="DrRORAL/ragaf-diffusion-checkpoints",
    subfolder="stage1_with_ssim",
    check_interval=300  # Check every 5 minutes
):
    """Monitor checkpoint directory and upload new files."""
    
    print("🚀 Starting automatic checkpoint uploader")
    print(f"   Monitoring: {checkpoint_dir}")
    print(f"   Uploading to: {repo_id}/{subfolder}")
    print(f"   Check interval: {check_interval}s")
    print()
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Ensure repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✅ Repository {repo_id} exists")
    except:
        print(f"⚠️  Repository {repo_id} not found, creating...")
        try:
            create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"✅ Created repository {repo_id}")
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return
    
    # Track uploaded files
    uploaded_files = set()
    checkpoint_path = Path(checkpoint_dir)
    
    # Initial upload of existing files
    if checkpoint_path.exists():
        for ckpt_file in checkpoint_path.glob("*.pt"):
            repo_path = f"{subfolder}/{ckpt_file.name}"
            if upload_checkpoint(api, repo_id, ckpt_file, repo_path):
                uploaded_files.add(ckpt_file.name)
        print(f"\n✅ Initial upload complete: {len(uploaded_files)} files")
    
    print(f"\n👀 Monitoring for new checkpoints...")
    print(f"   (Press Ctrl+C to stop)")
    print()
    
    # Monitor loop
    try:
        while True:
            time.sleep(check_interval)
            
            if not checkpoint_path.exists():
                continue
            
            # Check for new checkpoint files
            for ckpt_file in checkpoint_path.glob("*.pt"):
                if ckpt_file.name not in uploaded_files:
                    print(f"\n🆕 New checkpoint detected: {ckpt_file.name}")
                    repo_path = f"{subfolder}/{ckpt_file.name}"
                    
                    if upload_checkpoint(api, repo_id, ckpt_file, repo_path):
                        uploaded_files.add(ckpt_file.name)
                        print(f"✅ Total uploaded: {len(uploaded_files)} files\n")
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Stopping checkpoint uploader")
        print(f"   Total files uploaded: {len(uploaded_files)}")
        print(f"   Repository: https://huggingface.co/{repo_id}/tree/main/{subfolder}")

def main():
    parser = argparse.ArgumentParser(description='Auto-upload checkpoints to HuggingFace')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='/root/checkpoints/stage1_with_ssim',
                        help='Directory to monitor for checkpoints')
    parser.add_argument('--repo_id', type=str,
                        default='DrRORAL/ragaf-diffusion-checkpoints',
                        help='HuggingFace repository ID')
    parser.add_argument('--subfolder', type=str,
                        default='stage1_with_ssim',
                        help='Subfolder in the repository')
    parser.add_argument('--check_interval', type=int,
                        default=300,
                        help='Check interval in seconds (default: 300 = 5 minutes)')
    
    args = parser.parse_args()
    
    monitor_and_upload(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        subfolder=args.subfolder,
        check_interval=args.check_interval
    )

if __name__ == '__main__':
    main()
