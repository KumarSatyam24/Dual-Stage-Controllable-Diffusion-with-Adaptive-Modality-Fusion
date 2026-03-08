#!/bin/bash

# Quick Start Script for Improved Stage-1 Training with WandB
# This will retrain the model with optimized hyperparameters

echo "================================================================================"
echo "🚀 IMPROVED STAGE-1 TRAINING - QUICK START"
echo "================================================================================"
echo ""
echo "This script will retrain Stage-1 with improved hyperparameters:"
echo "  ✅ Learning rate: 1e-4 → 1e-5 (10x lower)"
echo "  ✅ Epochs: 10 → 20 (2x longer)"
echo "  ✅ Perceptual loss: Added (LPIPS)"
echo "  ✅ UNet: Unfrozen for full adaptation"
echo "  ✅ LoRA rank: 4 → 8 (2x capacity)"
echo "  ✅ Early stopping: Enabled"
echo "  ✅ WandB tracking: Enabled"
echo ""
echo "================================================================================"
echo ""

# Check if WandB is installed
if ! python3 -c "import wandb" 2>/dev/null; then
    echo "📦 Installing WandB..."
    pip install wandb -q
    echo "   ✅ WandB installed"
fi

# Check WandB login using Python
echo "🔐 Checking WandB login status..."
WANDB_CHECK=$(python3 << 'PYEOF'
import wandb
try:
    api = wandb.Api()
    user = str(api.viewer)
    print(f"LOGGED_IN:{user}")
except Exception as e:
    print("NOT_LOGGED_IN")
PYEOF
)

if [[ $WANDB_CHECK == LOGGED_IN:* ]]; then
    USER=$(echo $WANDB_CHECK | cut -d: -f2- | sed "s/<User //g" | sed "s/>//g" | tr -d "'")
    echo "   ✅ Logged in to WandB as: $USER"
    echo "   ✅ WandB tracking enabled"
else
    echo "   ⚠️  Not logged in to WandB"
    echo ""
    echo "To enable WandB tracking, run: wandb login"
    echo "Or continue without WandB by pressing 'y'"
    echo ""
    read -p "Continue without WandB? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please login first: wandb login"
        exit 1
    fi
    NO_WANDB="--no_wandb"
fi

# Change to project directory
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Check if dataset exists
if [ ! -d "/workspace/sketchy" ]; then
    echo "❌ ERROR: Dataset not found at /workspace/sketchy"
    echo ""
    echo "Please ensure the Sketchy dataset is available at /workspace/sketchy"
    exit 1
fi

echo "✅ Dataset found"
echo ""

# Check GPU
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Ask for confirmation
echo "⏱️  Estimated training time: 10-15 hours on RTX 5090"
echo "💾 Disk space needed: ~40GB"
echo ""
read -p "Do you want to start training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Training cancelled."
    exit 1
fi

echo ""
echo "================================================================================"
echo "🏋️  STARTING TRAINING"
echo "================================================================================"
echo ""

# Run training
python3 train_improved_stage1.py \
    --learning_rate 1e-5 \
    --epochs 20 \
    --batch_size 4 \
    --lora_rank 8 \
    --checkpoint_dir /root/checkpoints/stage1_improved \
    --hf_repo DrRORAL/ragaf-diffusion-checkpoints \
    $NO_WANDB

echo ""
echo "================================================================================"
echo "✅ TRAINING SCRIPT COMPLETED"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check checkpoints: ls /root/checkpoints/stage1_improved/"
echo "  2. Review training log: cat /root/checkpoints/stage1_improved/training_log.json"
echo "  3. View WandB dashboard: https://wandb.ai/your-username/ragaf-diffusion-stage1"
echo "  4. Run validation: python validate_epochs.py --checkpoint_dir /root/checkpoints/stage1_improved"
echo ""
echo "================================================================================"
