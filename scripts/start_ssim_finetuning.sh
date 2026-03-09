#!/bin/bash

# 🎯 SSIM Fine-tuning Launcher
# Run this script after Phase 2 training completes

set -e

echo "🎯 SSIM Fine-tuning Setup"
echo "========================="
echo ""

# Configuration
CHECKPOINT_DIR="/root/checkpoints/stage1_with_ssim"
FINETUNE_DIR="/root/checkpoints/ssim_finetuned"
LOG_FILE="train_ssim_finetune.log"

# Parse arguments
STRATEGY="${1:-aggressive}"  # aggressive, gradual, or only

# Check if Phase 2 checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Phase 2 checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Find best checkpoint
BEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "best_model.pt" -o -name "epoch_*.pt" | sort -V | tail -1)

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "✅ Found checkpoint: $BEST_CHECKPOINT"
echo ""

# Create finetune directory
mkdir -p "$FINETUNE_DIR"

# Backup current best checkpoint
BACKUP_PATH="/root/checkpoints/backups/before_ssim_finetune_$(date +%Y%m%d_%H%M%S).pt"
mkdir -p /root/checkpoints/backups
echo "💾 Backing up checkpoint to: $BACKUP_PATH"
cp "$BEST_CHECKPOINT" "$BACKUP_PATH"
echo ""

# Configure based on strategy
case "$STRATEGY" in
    aggressive)
        echo "🔥 Using AGGRESSIVE SSIM fine-tuning strategy"
        SSIM_WEIGHT=0.3
        PERCEPTUAL_WEIGHT=0.3
        MSE_WEIGHT=0.4
        LEARNING_RATE=1e-6
        EPOCHS=15
        EXPERIMENT_NAME="ssim-finetune-aggressive"
        ;;
    gradual)
        echo "📈 Using GRADUAL SSIM fine-tuning strategy"
        SSIM_WEIGHT=0.15
        PERCEPTUAL_WEIGHT=0.4
        MSE_WEIGHT=0.45
        LEARNING_RATE=2e-6
        EPOCHS=20
        EXPERIMENT_NAME="ssim-finetune-gradual"
        ;;
    only)
        echo "🎯 Using SSIM-ONLY fine-tuning strategy"
        SSIM_WEIGHT=1.0
        PERCEPTUAL_WEIGHT=0.0
        MSE_WEIGHT=0.0
        LEARNING_RATE=5e-7
        EPOCHS=10
        EXPERIMENT_NAME="ssim-only-polish"
        echo "⚠️  WARNING: Pure SSIM training may hurt perceptual quality!"
        ;;
    *)
        echo "❌ Error: Unknown strategy '$STRATEGY'"
        echo "Usage: $0 [aggressive|gradual|only]"
        exit 1
        ;;
esac

echo ""
echo "Configuration:"
echo "  SSIM Weight:       $SSIM_WEIGHT"
echo "  Perceptual Weight: $PERCEPTUAL_WEIGHT"
echo "  MSE Weight:        $MSE_WEIGHT"
echo "  Learning Rate:     $LEARNING_RATE"
echo "  Epochs:            $EPOCHS"
echo "  Experiment Name:   $EXPERIMENT_NAME"
echo ""

# Confirm before starting
read -p "🚀 Start SSIM fine-tuning? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Check if training is already running
if ps aux | grep -v grep | grep -q "train_stage1_with_ssim.py"; then
    echo "⚠️  WARNING: Another training process is already running!"
    ps aux | grep -v grep | grep "train_stage1_with_ssim.py"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Cancelled"
        exit 1
    fi
fi

# Start training
cd ~/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

echo "🚀 Starting SSIM fine-tuning..."
nohup python train_stage1_with_ssim.py \
    --resume_from "$BEST_CHECKPOINT" \
    --batch_size 4 \
    --ssim_weight "$SSIM_WEIGHT" \
    --perceptual_weight "$PERCEPTUAL_WEIGHT" \
    --mse_weight "$MSE_WEIGHT" \
    --learning_rate "$LEARNING_RATE" \
    --validation_samples 100 \
    --epochs "$EPOCHS" \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_dir "$FINETUNE_DIR" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo ""
echo "✅ SSIM fine-tuning started!"
echo "   PID: $TRAIN_PID"
echo "   Log: $LOG_FILE"
echo "   Checkpoint dir: $FINETUNE_DIR"
echo ""
echo "📊 Monitor with:"
echo "   tail -f $LOG_FILE"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🛑 Stop training with:"
echo "   kill $TRAIN_PID"
echo ""

# Create a status file
cat > ssim_finetune_status.txt << EOF
SSIM Fine-tuning Status
=======================
Started: $(date)
PID: $TRAIN_PID
Strategy: $STRATEGY
Checkpoint: $BEST_CHECKPOINT
Log: $LOG_FILE
Output: $FINETUNE_DIR

Configuration:
- SSIM Weight: $SSIM_WEIGHT
- Perceptual Weight: $PERCEPTUAL_WEIGHT
- MSE Weight: $MSE_WEIGHT
- Learning Rate: $LEARNING_RATE
- Epochs: $EPOCHS
- Experiment: $EXPERIMENT_NAME
EOF

echo "📝 Status saved to: ssim_finetune_status.txt"
echo ""
echo "🎯 Good luck with fine-tuning!"
