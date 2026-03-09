#!/bin/bash

# 🤗 Upload Stage 1 Checkpoints to Hugging Face Hub

set -e

echo "🤗 Hugging Face Checkpoint Upload Script"
echo "========================================"
echo ""

# Configuration
CHECKPOINT_DIR="/root/checkpoints/stage1_with_ssim"
REPO_NAME="stage1-dual-stage-diffusion"  # Change this to your desired repo name
HF_USERNAME="${HF_USERNAME:-your-username}"  # Set your HF username

echo "📋 Configuration:"
echo "   Checkpoint Dir: $CHECKPOINT_DIR"
echo "   Repository: $HF_USERNAME/$REPO_NAME"
echo ""

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# Check if logged in
echo "🔐 Checking Hugging Face authentication..."
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "❌ Not logged in to Hugging Face!"
    echo ""
    echo "Please run one of:"
    echo "  1. huggingface-cli login"
    echo "  2. export HF_TOKEN=your_token_here"
    echo ""
    exit 1
fi

echo "✅ Authenticated!"
echo ""

# List available checkpoints
echo "📊 Available checkpoints:"
ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
echo ""

# Ask for confirmation
read -p "Upload all these checkpoints? (yes/NO): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Cancelled"
    exit 0
fi

# Create upload script
cat > /tmp/upload_checkpoints.py << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
import json
from datetime import datetime

def upload_checkpoints():
    # Configuration
    checkpoint_dir = Path("/root/checkpoints/stage1_with_ssim")
    repo_id = f"{os.environ.get('HF_USERNAME', 'your-username')}/stage1-dual-stage-diffusion"
    
    print(f"🚀 Uploading to: {repo_id}")
    print()
    
    # Initialize API
    api = HfApi()
    
    # Create repository (if doesn't exist)
    try:
        print("📁 Creating/accessing repository...")
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print("✅ Repository ready!")
    except Exception as e:
        print(f"⚠️  Repository issue: {e}")
        print("Continuing with upload...")
    
    print()
    
    # Find all checkpoints
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    print(f"📦 Found {len(checkpoints)} checkpoints")
    print()
    
    # Create metadata
    metadata = {
        "model_name": "Stage 1 - Dual-Stage Controllable Diffusion",
        "training_date": datetime.now().isoformat(),
        "epochs_trained": len(checkpoints),
        "architecture": "ControlNet + Sketch Encoder + VAE",
        "loss_components": ["MSE", "Perceptual (LPIPS)", "SSIM"],
        "ssim_weight": 0.05,
        "batch_size": 4,
        "learning_rate": "5e-6 (cosine annealing)",
        "validation_samples": 100,
        "checkpoints": []
    }
    
    # Upload each checkpoint
    for i, ckpt_path in enumerate(checkpoints, 1):
        epoch_num = int(ckpt_path.stem.split("_")[1])
        print(f"📤 [{i}/{len(checkpoints)}] Uploading {ckpt_path.name}...")
        
        try:
            # Get file size
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.1f} MB")
            
            # Upload
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
                "size_mb": round(size_mb, 1)
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
    upload_checkpoints()
PYTHON_SCRIPT

# Run upload
echo "🚀 Starting upload..."
echo ""
python /tmp/upload_checkpoints.py

echo ""
echo "✅ Done!"
