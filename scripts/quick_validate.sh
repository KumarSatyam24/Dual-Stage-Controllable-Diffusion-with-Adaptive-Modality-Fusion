#!/bin/bash

# Quick Epoch Validation Script
# Tests epoch 2 with 10 samples to verify everything works

echo "========================================================================"
echo "🧪 Quick Epoch Validation Test"
echo "========================================================================"
echo ""
echo "Testing: Epoch 2 with 10 samples"
echo "Time: ~5 minutes"
echo ""
echo "This will:"
echo "  1. Download epoch_2.pt from HuggingFace (if needed)"
echo "  2. Generate 10 validation images"
echo "  3. Compute PSNR, SSIM, LPIPS, FID, CLIP metrics"
echo "  4. Save results to validation_results/epoch_2/"
echo ""
echo "========================================================================"
echo ""

cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

python validate_epochs.py \
    --epochs 2 \
    --num_samples 10 \
    --output_dir validation_results

echo ""
echo "========================================================================"
echo "✅ Validation Complete!"
echo "========================================================================"
echo ""
echo "Results saved to: validation_results/epoch_2/"
echo ""
echo "View results:"
echo "  ls validation_results/epoch_2/"
echo "  cat validation_results/epoch_2/metrics.json | jq"
echo ""
echo "Next steps:"
echo "  1. Check metrics in validation_results/epoch_2/metrics.json"
echo "  2. View sample images: validation_results/epoch_2/sample_*.png"
echo "  3. Run full validation: python validate_epochs.py --epochs 2 4 6 8 10 --num_samples 50"
echo ""
echo "========================================================================"
