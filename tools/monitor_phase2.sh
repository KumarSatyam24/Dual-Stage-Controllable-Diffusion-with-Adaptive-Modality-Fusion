#!/bin/bash

# 📊 Phase 2 Training Monitor
# Check training status and progress

echo "🔍 Phase 2 Training Status Monitor"
echo "===================================="
echo ""

# Check if training is running
TRAIN_PID=$(ps aux | grep "train_stage1_with_ssim.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ Training is NOT running!"
    echo ""
    echo "Last log entries:"
    tail -20 ~/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/train_phase2.log 2>/dev/null || echo "No log file found"
    exit 1
fi

echo "✅ Training is RUNNING"
echo "   PID: $TRAIN_PID"
echo ""

# Check GPU usage
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F, '{printf "   GPU %s: %s%%  |  Mem: %sMB/%sMB  |  Temp: %s°C  |  Power: %sW\n", $1, $3, $5, $6, $7, $8}'
echo ""

# Training process info
echo "⚙️  Process Info:"
ps -p $TRAIN_PID -o pid,user,%cpu,%mem,etime,cmd --no-headers | \
    awk '{printf "   CPU: %s%%  |  MEM: %s%%  |  Runtime: %s\n", $3, $4, $5}'
echo ""

# Check for checkpoint files
echo "💾 Checkpoints:"
CKPT_DIR="/root/checkpoints/stage1_with_ssim"
if [ -d "$CKPT_DIR" ]; then
    CKPT_COUNT=$(find "$CKPT_DIR" -name "*.pt" 2>/dev/null | wc -l)
    LATEST_CKPT=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null | head -1)
    
    if [ $CKPT_COUNT -gt 0 ]; then
        echo "   Total checkpoints: $CKPT_COUNT"
        if [ -n "$LATEST_CKPT" ]; then
            CKPT_NAME=$(basename "$LATEST_CKPT")
            CKPT_SIZE=$(du -h "$LATEST_CKPT" | cut -f1)
            CKPT_TIME=$(stat -c %y "$LATEST_CKPT" 2>/dev/null | cut -d'.' -f1)
            echo "   Latest: $CKPT_NAME ($CKPT_SIZE)"
            echo "   Created: $CKPT_TIME"
        fi
    else
        echo "   ⚠️  No checkpoints found yet"
    fi
else
    echo "   ⚠️  Checkpoint directory not found"
fi
echo ""

# Check WandB run
echo "📈 WandB:"
WANDB_DIR="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/wandb"
LATEST_RUN=$(ls -td "$WANDB_DIR"/run-* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
    RUN_NAME=$(basename "$LATEST_RUN")
    RUN_TIME=$(stat -c %y "$LATEST_RUN" 2>/dev/null | cut -d' ' -f1-2 | cut -d'.' -f1)
    echo "   Run: $RUN_NAME"
    echo "   Started: $RUN_TIME"
    
    # Try to get run URL from log
    DEBUG_LOG="$LATEST_RUN/logs/debug.log"
    if [ -f "$DEBUG_LOG" ]; then
        RUN_URL=$(grep "View run" "$DEBUG_LOG" 2>/dev/null | tail -1 | grep -oP 'https://[^ ]+' || echo "")
        if [ -n "$RUN_URL" ]; then
            echo "   URL: $RUN_URL"
        fi
    fi
else
    echo "   ⚠️  No WandB run found"
fi
echo ""

# Estimated progress (very rough)
echo "⏱️  Estimated Progress:"
LOG_FILE="$HOME/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/train_phase2.log"
if [ -f "$LOG_FILE" ]; then
    # Try to find epoch info in log
    LAST_EPOCH=$(grep -oP "Epoch \K[0-9]+" "$LOG_FILE" 2>/dev/null | tail -1)
    if [ -n "$LAST_EPOCH" ]; then
        echo "   Last epoch seen in log: $LAST_EPOCH / 25"
        PROGRESS=$((LAST_EPOCH * 100 / 25))
        echo "   Progress: ~$PROGRESS%"
    else
        echo "   ℹ️  No epoch information in log yet"
        echo "   (Training may still be initializing)"
    fi
else
    echo "   ⚠️  Log file not found"
fi
echo ""

# Quick tips
echo "💡 Monitoring Commands:"
echo "   Full status:    ./monitor_phase2.sh"
echo "   Watch GPU:      watch -n 2 nvidia-smi"
echo "   Watch log:      tail -f $LOG_FILE"
echo "   Check process:  ps aux | grep train_stage1"
echo "   Kill training:  kill $TRAIN_PID"
echo ""

# Disk space check
echo "💿 Disk Space:"
df -h /root/checkpoints 2>/dev/null | tail -1 | awk '{printf "   Checkpoints: %s used / %s total (%s available)\n", $3, $2, $4}'
echo ""

echo "✅ Monitoring complete"
echo ""
echo "🔄 Run this script periodically to check progress"
echo "   Example: watch -n 300 ./monitor_phase2.sh  (every 5 min)"
