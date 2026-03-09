"""
🤗 Upload Stage 1 Checkpoints to Hugging Face Hub
Simple Python script to upload all checkpoints with metadata
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import json
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to Hugging Face Hub")
    parser.add_argument("--repo-name", type=str, default="stage1-dual-stage-diffusion",
                        help="Repository name on Hugging Face")
    parser.add_argument("--checkpoint-dir", type=str, 
                        default="/root/checkpoints/stage1_with_ssim",
                        help="Directory containing checkpoints")
    parser.add_argument("--username", type=str, default=None,
                        help="Hugging Face username (optional, will auto-detect)")
    args = parser.parse_args()
    
    print("🤗 Hugging Face Checkpoint Upload")
    print("=" * 60)
    
    # Initialize API
    api = HfApi()
    
    # Get username
    try:
        user_info = api.whoami()
        username = args.username or user_info['name']
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print(f"❌ Not authenticated! Please run: huggingface-cli login")
        print(f"   Error: {e}")
        return
    
    # Setup
    checkpoint_dir = Path(args.checkpoint_dir)
    repo_id = f"{username}/{args.repo_name}"
    
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    print(f"📦 Repository: {repo_id}")
    print()
    
    # Check checkpoints exist
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"📊 Found {len(checkpoints)} checkpoints:")
    total_size_gb = 0
    for ckpt in checkpoints:
        size_gb = ckpt.stat().st_size / (1024**3)
        total_size_gb += size_gb
        print(f"   {ckpt.name}: {size_gb:.2f} GB")
    
    print(f"\n📦 Total size: {total_size_gb:.2f} GB")
    print()
    
    # Confirm upload
    response = input("⚠️  This will upload large files. Continue? (yes/NO): ")
    if response.lower() != "yes":
        print("❌ Cancelled")
        return
    
    print()
    
    # Create repository
    try:
        print("📁 Creating repository...")
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
        print("✅ Repository ready!")
    except Exception as e:
        print(f"⚠️  Repository creation issue: {e}")
        print("   Continuing with upload...")
    
    print()
    
    # Create README
    readme_content = f"""---
license: apache-2.0
tags:
- diffusion
- controlnet
- sketch-to-image
- dual-stage
---

# Stage 1 - Dual-Stage Controllable Diffusion

Stage 1 checkpoint for dual-stage controllable diffusion with adaptive modality fusion.

## Model Details

- **Architecture:** ControlNet + Sketch Encoder + VAE
- **Training Date:** {datetime.now().strftime("%Y-%m-%d")}
- **Epochs Trained:** {len(checkpoints)}
- **Batch Size:** 4
- **Learning Rate:** 5e-6 (cosine annealing)

## Loss Components

- MSE Loss
- Perceptual Loss (LPIPS)
- SSIM Loss (weight: 0.05)

## Validation Metrics

Based on 100-sample validation:
- SSIM: ~0.15-0.17
- PSNR: ~10.0-10.5 dB
- LPIPS: ~0.70-0.75

## Checkpoints

{len(checkpoints)} epoch checkpoints available in `checkpoints/` directory.

## Usage

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/epoch_18.pt")

# Extract models
stage1_model = checkpoint['stage1_model']
# ... use in your pipeline
```

## Training Details

- Dataset: Sketchy Database
- Validation Frequency: Every 2 epochs
- Best Checkpoint: epoch_18.pt (based on SSIM)

## Citation

If you use this model, please cite:
[Your paper/project details]
"""
    
    # Save README
    readme_path = "/tmp/README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print("📝 Uploading README...")
    try:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print("✅ README uploaded!")
    except Exception as e:
        print(f"⚠️  README upload failed: {e}")
    
    print()
    
    # Create and upload metadata
    metadata = {
        "model_name": "Stage 1 - Dual-Stage Controllable Diffusion",
        "training_date": datetime.now().isoformat(),
        "epochs_trained": len(checkpoints),
        "architecture": "ControlNet + Sketch Encoder + VAE",
        "loss_components": ["MSE", "Perceptual (LPIPS)", "SSIM"],
        "hyperparameters": {
            "ssim_weight": 0.05,
            "batch_size": 4,
            "learning_rate": "5e-6",
            "scheduler": "CosineAnnealingLR",
            "validation_samples": 100,
        },
        "best_checkpoint": "epoch_18.pt",
        "checkpoints": []
    }
    
    # Upload each checkpoint
    for i, ckpt_path in enumerate(checkpoints, 1):
        epoch_num = int(ckpt_path.stem.split("_")[1])
        size_gb = ckpt_path.stat().st_size / (1024**3)
        
        print(f"📤 [{i}/{len(checkpoints)}] Uploading {ckpt_path.name} ({size_gb:.2f} GB)...")
        
        try:
            upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=f"checkpoints/{ckpt_path.name}",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"   ✅ Uploaded!")
            
            # Add to metadata
            metadata["checkpoints"].append({
                "epoch": epoch_num,
                "filename": ckpt_path.name,
                "size_gb": round(size_gb, 2)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
        
        print()
    
    # Upload metadata
    print("📝 Uploading metadata...")
    metadata_path = "/tmp/checkpoint_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, indent=2, fp=f)
    
    try:
        upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="checkpoint_metadata.json",
            repo_id=repo_id,
            repo_type="model",
        )
        print("✅ Metadata uploaded!")
    except Exception as e:
        print(f"⚠️  Metadata upload failed: {e}")
    
    print()
    print("=" * 60)
    print("🎉 Upload Complete!")
    print(f"🔗 View at: https://huggingface.co/{repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    main()
