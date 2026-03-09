#!/bin/bash
# Quick Accuracy Evaluation for Stage 1 Model

echo "========================================================================"
echo "🎯 STAGE 1 MODEL ACCURACY EVALUATION"
echo "========================================================================"
echo ""
echo "This script evaluates your trained Stage 1 model and computes:"
echo "  • Overall Accuracy (0-100%)"
echo "  • Edge Consistency (sketch fidelity)"
echo "  • Category Accuracy (True Positive Rate)"
echo ""

CHECKPOINT="/root/checkpoints/stage1/epoch_10.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo "Please specify the correct checkpoint path"
    exit 1
fi

echo "📁 Using checkpoint: $CHECKPOINT"
echo ""

# Option 1: Quick evaluation (100 samples)
echo "Option 1: Quick Evaluation (100 samples, ~5-10 minutes)"
echo "   python evaluate_stage1_accuracy.py --checkpoint $CHECKPOINT --num_samples 100"
echo ""

# Option 2: Thorough evaluation (500 samples)
echo "Option 2: Thorough Evaluation (500 samples, ~30-40 minutes)"
echo "   python evaluate_stage1_accuracy.py --checkpoint $CHECKPOINT --num_samples 500"
echo ""

# Option 3: Compare guidance scales
echo "Option 3: Compare Guidance Scales"
echo "   python evaluate_stage1_accuracy.py --checkpoint $CHECKPOINT --guidance_scale 7.5"
echo "   python evaluate_stage1_accuracy.py --checkpoint $CHECKPOINT --guidance_scale 2.5"
echo ""

read -p "Run quick evaluation now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "🚀 Starting evaluation..."
    echo ""
    python evaluate_stage1_accuracy.py \
        --checkpoint "$CHECKPOINT" \
        --num_samples 100 \
        --guidance_scale 2.5
    
    echo ""
    echo "✅ Evaluation complete!"
    echo ""
    echo "Results saved to: stage1_evaluation_guidance2.5.json"
    echo ""
fi

echo ""
echo "========================================================================"
echo "📚 For more information, see: ACCURACY_EVALUATION_GUIDE.md"
echo "========================================================================"
