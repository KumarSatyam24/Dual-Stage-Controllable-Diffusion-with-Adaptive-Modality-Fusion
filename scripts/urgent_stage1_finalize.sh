#!/bin/bash

# 🚨 URGENT: Stage 1 Finalization Script
# Run this to make immediate decision for March 31 deadline

set -e

echo "🚨 DEADLINE TRACKER"
echo "==================="
echo "Today: March 9, 2026"
echo "Final Deadline: March 31, 2026"
echo "Days Remaining: 22"
echo ""

# Check current training
echo "📊 Current Training Status:"
echo "-------------------------"
if ps aux | grep -v grep | grep -q "train_stage1_with_ssim.py"; then
    echo "✅ Phase 2 training IS RUNNING"
    ps aux | grep -v grep | grep "train_stage1_with_ssim.py" | head -1
    echo ""
    
    # Check GPU
    echo "🔥 GPU Status:"
    nvidia-smi | grep -A 1 "GPU  Name"
    echo ""
    
    # Check checkpoints
    echo "💾 Latest Checkpoints:"
    ls -lht /root/checkpoints/stage1_with_ssim/ 2>/dev/null | head -5 || echo "No checkpoints yet"
    echo ""
else
    echo "❌ No training running"
    echo ""
fi

echo ""
echo "🎯 DECISION TIME"
echo "================"
echo ""
echo "To meet March 31 deadline, choose your path:"
echo ""
echo "Option A: WAIT for Phase 2 to complete (~10 PM tonight)"
echo "  - Pros: Best Stage 1 checkpoint, no wasted compute"
echo "  - Cons: Delays Stage 2 start by 10 hours"
echo "  - Stage 1 done: 11:30 PM today"
echo "  - Stage 2 starts: Tomorrow 10 AM"
echo ""
echo "Option B: STOP NOW and use epoch_12.pt (FASTEST)"
echo "  - Pros: Start Stage 2 TODAY, max time for Stage 2"
echo "  - Cons: Miss potential improvements from Phase 2"
echo "  - Stage 1 done: 2 PM today"
echo "  - Stage 2 starts: TODAY 3 PM"
echo ""
echo "Option C: STOP and aggressive 6-hour fine-tune"
echo "  - Pros: One more optimization attempt"
echo "  - Cons: Risky, may not improve much"
echo "  - Stage 1 done: 9 PM today"
echo "  - Stage 2 starts: Tomorrow 8 AM"
echo ""

read -p "Enter your choice (A/B/C): " choice

case "$choice" in
    [Aa])
        echo ""
        echo "✅ OPTION A SELECTED: Wait for Phase 2 completion"
        echo ""
        echo "Action Plan:"
        echo "1. Continue monitoring current training"
        echo "2. Set alarm for 10 PM to check completion"
        echo "3. Evaluate results at 11 PM"
        echo "4. Finalize Stage 1 checkpoint by midnight"
        echo "5. Start Stage 2 tomorrow morning"
        echo ""
        echo "📊 Starting continuous monitor..."
        echo "Press Ctrl+C to exit monitoring"
        echo ""
        sleep 3
        
        # Monitor loop
        while true; do
            clear
            echo "=== PHASE 2 TRAINING MONITOR ==="
            echo "Time: $(date)"
            echo "Target completion: ~10:00 PM"
            echo ""
            
            if ps aux | grep -v grep | grep -q "train_stage1_with_ssim.py"; then
                echo "✅ Training ACTIVE"
                nvidia-smi | grep -E "MiB.*MiB|python"
                echo ""
                echo "Latest checkpoint:"
                ls -lht /root/checkpoints/stage1_with_ssim/ 2>/dev/null | head -2 || echo "Checking..."
            else
                echo "🎉 TRAINING COMPLETED!"
                echo ""
                echo "Next steps:"
                echo "1. Validate final checkpoint"
                echo "2. Select best model"
                echo "3. Prepare for Stage 2"
                break
            fi
            
            echo ""
            echo "Refreshing in 60 seconds... (Ctrl+C to exit)"
            sleep 60
        done
        ;;
        
    [Bb])
        echo ""
        echo "⚡ OPTION B SELECTED: Stop NOW and use epoch_12"
        echo ""
        read -p "Are you sure? This will stop current training. (yes/NO): " confirm
        
        if [ "$confirm" = "yes" ]; then
            echo ""
            echo "🛑 Stopping current training..."
            pkill -f train_stage1_with_ssim.py || echo "No training to stop"
            sleep 2
            
            echo "💾 Finalizing epoch_12 as Stage 1 checkpoint..."
            mkdir -p /root/checkpoints/stage1_final
            cp /root/checkpoints/stage1_improved/epoch_12.pt \
               /root/checkpoints/stage1_final/stage1_final.pt
            
            echo "✅ Stage 1 checkpoint finalized!"
            echo "📍 Location: /root/checkpoints/stage1_final/stage1_final.pt"
            echo ""
            echo "📝 Creating Stage 1 completion report..."
            cat > /root/STAGE1_FINAL_REPORT.txt << EOF
Stage 1 Training - FINAL REPORT
================================
Date: $(date)
Decision: Use epoch_12.pt from initial training
Reason: Time-critical path for March 31 deadline

Checkpoint: /root/checkpoints/stage1_final/stage1_final.pt

Known Metrics (from epoch 12):
- SSIM: ~0.11-0.24 (depending on validation set)
- PSNR: ~9.8 dB
- LPIPS: ~0.71

Status: READY FOR STAGE 2 ✅
Next: Stage 2 training setup
EOF
            
            echo "✅ STAGE 1 COMPLETE!"
            echo ""
            echo "🚀 Ready to start Stage 2!"
            echo "Run: python train_stage2.py --stage1_checkpoint /root/checkpoints/stage1_final/stage1_final.pt"
            
        else
            echo "❌ Cancelled. Current training continues."
        fi
        ;;
        
    [Cc])
        echo ""
        echo "🔥 OPTION C SELECTED: Aggressive 6-hour fine-tune"
        echo ""
        read -p "Are you sure? This restarts training. (yes/NO): " confirm
        
        if [ "$confirm" = "yes" ]; then
            echo "🛑 Stopping current training..."
            pkill -f train_stage1_with_ssim.py || echo "No training to stop"
            sleep 2
            
            echo "🚀 Starting aggressive SSIM fine-tune..."
            cd ~/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion
            
            nohup python train_stage1_with_ssim.py \
                --resume_from /root/checkpoints/stage1_improved/epoch_12.pt \
                --batch_size 4 \
                --ssim_weight 0.3 \
                --perceptual_weight 0.3 \
                --mse_weight 0.4 \
                --learning_rate 1e-6 \
                --validation_samples 100 \
                --epochs 10 \
                --experiment_name "urgent-ssim-finetune" \
                > train_urgent_finetune.log 2>&1 &
            
            TRAIN_PID=$!
            echo "✅ Training started! PID: $TRAIN_PID"
            echo "📊 Estimated completion: ~9 PM"
            echo "📝 Log: train_urgent_finetune.log"
            echo ""
            echo "Monitor with: tail -f train_urgent_finetune.log"
        else
            echo "❌ Cancelled. Current training continues."
        fi
        ;;
        
    *)
        echo ""
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "📅 Timeline reminder:"
echo "- TODAY: Finalize Stage 1"
echo "- March 10-20: Stage 2 training"
echo "- March 21-28: Evaluation & documentation"
echo "- March 29-30: Buffer/polish"
echo "- March 31: DEADLINE ⏰"
echo ""
