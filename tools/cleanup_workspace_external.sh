#!/bin/bash
# Cleanup /workspace directory
# Remove old test outputs and comparison grids

set -e

echo "=== /workspace Directory Cleanup ==="
echo ""
echo "This will remove:"
echo "  - Old test outputs (test_outputs_*)"
echo "  - Comparison grid images (comparison_grid_*.png, *_comparison.png)"
echo "  - Old log files (*.log)"
echo "  - Test output files"
echo ""
echo "This will KEEP:"
echo "  ✓ sketchy/ (dataset)"
echo "  ✓ checkpoints/ (model checkpoints)"
echo "  ✓ dataset/ (if exists)"
echo ""

read -p "Continue with cleanup? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

cd /workspace

echo ""
echo "=== Step 1: Removing old test output directories ==="
rm -rf test_outputs_all_categories
rm -rf test_outputs_corrected
rm -rf test_outputs_epoch10
rm -rf test_outputs_final_diverse
rm -rf test_outputs_guidance_comparison
rm -rf test_outputs_guidance_scales
rm -rf test_outputs_optimized
echo "✓ Removed 7 test output directories"

echo ""
echo "=== Step 2: Removing comparison grid images ==="
rm -f comparison_grid_*.png
rm -f *_comparison.png
rm -f optimized_grid_*.png
rm -f epoch2_test_output.png
rm -f fighter_jet_epoch2.png
rm -f final_diverse_comparison.png
echo "✓ Removed ~35+ comparison images"

echo ""
echo "=== Step 3: Removing old log files ==="
rm -f test_all_categories.log
rm -f test_corrected.log
rm -f test_output.txt
rm -f epoch2_test.log
rm -f train_stage1.log
rm -f train_stage1_FIXED.log
rm -f train_stage2.log
echo "✓ Removed 7 log files"

echo ""
echo "=== Step 4: Removing documentation (moved to project dir) ==="
rm -f GRIDS_LOCATION.md
rm -f HOW_TO_VIEW_RESULTS.md
rm -f view_all_grids.html
echo "✓ Removed 3 documentation files"

echo ""
echo "=== Cleanup Summary ==="
echo "Removed from /workspace:"
echo "  - 7 test output directories"
echo "  - ~35+ comparison images"
echo "  - 7 old log files"
echo "  - 3 documentation files"
echo ""
echo "Preserved in /workspace:"
echo "  ✓ sketchy/ (Sketchy dataset)"
echo "  ✓ checkpoints/ (model checkpoints)"
echo "  ✓ dataset/ (if exists)"
echo ""

# Show remaining files
echo "=== Remaining files in /workspace ==="
ls -lh /workspace | grep -v "^total" | grep -v "^d" || echo "Only directories remain"
echo ""

# Show disk space
echo "=== Disk Space ==="
df -h /workspace | tail -1

echo ""
echo "✅ /workspace cleanup complete!"
