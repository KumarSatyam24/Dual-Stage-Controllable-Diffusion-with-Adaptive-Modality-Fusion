#!/bin/bash
# Safe workspace cleanup script
# This will NOT interrupt training!

set -e

echo "=== Workspace Cleanup Script ==="
echo "This will remove obsolete files while preserving:"
echo "  - Current training (train.log, train_improved_stage1.py)"
echo "  - Active code (src/, models/, configs/, etc.)"
echo "  - Essential documentation"
echo ""

# Check if training is running
if ps aux | grep -q "[t]rain_improved_stage1.py"; then
    echo "✓ Training is running - will NOT touch any training files"
else
    echo "⚠ Training not detected as running"
fi

echo ""
read -p "Continue with cleanup? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo "=== Step 1: Removing duplicate validation scripts ==="
rm -vf evaluate_stage1_accuracy.py
rm -vf evaluate_stage1_metrics.py
rm -vf evaluate_stage1_simple.py
rm -vf evaluate_stage1_validation.py
rm -vf test_epoch_validation.py
rm -vf test_validation_quick.py
rm -vf debug_checkpoint.py

echo ""
echo "=== Step 2: Removing duplicate documentation ==="
rm -vf ACCURACY_EVALUATION_GUIDE.md
rm -vf ACCURACY_EXPLAINED.md
rm -vf HOW_TO_MEASURE_ACCURACY.md
rm -vf IMPROVEMENT_PLAN.md
rm -vf STAGE1_VALIDATION_SUMMARY.md
rm -vf VALIDATION_COMMANDS.md
rm -vf VALIDATION_METRICS_GUIDE.md
rm -vf VALIDATION_READY.md
rm -vf VALIDATION_RESULTS_ANALYSIS.md
rm -vf VALIDATION_SETUP_COMPLETE.md
rm -vf QUICK_START_SUMMARY.txt

echo ""
echo "=== Step 3: Removing old shell scripts ==="
rm -vf cleanup_project.sh
rm -vf evaluate.sh
rm -vf inference.sh
rm -vf quick_test.sh
rm -vf quick_validate.sh
rm -vf run_accuracy_evaluation.sh
rm -vf run_validation.sh
rm -vf view_grids.sh

echo ""
echo "=== Step 4: Removing old configuration ==="
rm -vf config_improved.py

echo ""
echo "=== Step 5: Removing old logs ==="
rm -vf logs/test_epoch10_results.log
rm -vf logs/test_stage1_results.log
rm -vf logs/train_stage1.log
rm -vf evaluation_output.log
rm -vf validation_test.log

echo ""
echo "=== Step 6: Cleaning Python cache ==="
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "=== Step 7: Removing test validation results ==="
# Keep the JSON for comparison, remove images
if [ -d "test_validation" ]; then
    find test_validation -type f -name "*.png" -delete
    find test_validation -type d -name "epoch_*" -exec rm -rf {} + 2>/dev/null || true
fi

echo ""
echo "=== Step 8: Cleaning old validation results ==="
echo "Keeping all_epochs_metrics.json and metrics_across_epochs.png for comparison"
if [ -d "validation_results" ]; then
    # Keep these for comparison
    mkdir -p validation_results_backup
    cp validation_results/all_epochs_metrics.json validation_results_backup/ 2>/dev/null || true
    cp validation_results/metrics_across_epochs.png validation_results_backup/ 2>/dev/null || true
    
    # Remove epoch directories with images
    rm -rf validation_results/epoch_*/
    
    # Remove other files
    rm -f validation_results/comparison_*.png
    rm -f validation_results/validation_metrics.json
    rm -f validation_results/validation_report.html
    
    # Restore backup
    cp validation_results_backup/* validation_results/ 2>/dev/null || true
    rm -rf validation_results_backup
fi

echo ""
echo "=== Cleanup Summary ==="
echo "Removed:"
echo "  - 7 duplicate validation scripts"
echo "  - 11 duplicate documentation files"
echo "  - 8 old shell scripts"
echo "  - Old logs and cache files"
echo "  - Old validation images"
echo ""
echo "Preserved:"
echo "  ✓ train_improved_stage1.py (current training)"
echo "  ✓ train.log (ACTIVE training output)"
echo "  ✓ validate_epochs.py (current validation)"
echo "  ✓ All source code (src/, models/, configs/, etc.)"
echo "  ✓ Essential docs (README*.md, TRAINING_IMPROVEMENTS.md, WANDB_SETUP.md)"
echo "  ✓ Metrics comparison data"
echo ""

# Check disk space saved
echo "=== Disk Space Check ==="
df -h . | tail -1

echo ""
echo "✅ Cleanup complete! Training continues unaffected."
echo ""
echo "Optional: Review archive/ directory manually:"
echo "  du -sh archive/"
echo "  # Delete if not needed: rm -rf archive/"
