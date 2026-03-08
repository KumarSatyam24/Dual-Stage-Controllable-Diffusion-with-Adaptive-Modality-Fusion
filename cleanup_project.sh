#!/bin/bash
# Project Cleanup and Optimization Script
# Removes unnecessary test files, old outputs, and organizes the project

echo "========================================================================"
echo "🧹 PROJECT CLEANUP AND OPTIMIZATION"
echo "========================================================================"
echo ""

PROJ_DIR="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
cd "$PROJ_DIR"

# Create organized directories
mkdir -p archive/{debug_scripts,old_tests,documentation}
mkdir -p tests/unit_tests
mkdir -p logs

echo "📊 Analyzing project files..."
echo ""

# Count files
total_py=$(find . -name "*.py" | wc -l)
total_md=$(find . -name "*.md" | wc -l)
total_sh=$(find . -name "*.sh" | wc -l)
total_log=$(find . -name "*.log" | wc -l)

echo "Current file count:"
echo "  Python files: $total_py"
echo "  Markdown docs: $total_md"
echo "  Shell scripts: $total_sh"
echo "  Log files: $total_log"
echo ""

echo "========================================================================"
echo "📦 CLEANUP CATEGORIES"
echo "========================================================================"
echo ""

# ============================================================================
# Category 1: Core Files (KEEP)
# ============================================================================
echo "✅ KEEP - Core Files:"
echo "   train.py                    - Main training script"
echo "   inference.py                - Main inference script"
echo "   models/*.py                 - Model architectures"
echo "   configs/config.py           - Configuration"
echo "   datasets/*.py               - Dataset loaders"
echo "   data/*.py                   - Data processing"
echo "   utils/*.py                  - Utility functions"
echo ""

# ============================================================================
# Category 2: Debug/Development Scripts (ARCHIVE)
# ============================================================================
echo "🗄️  ARCHIVE - Debug Scripts (move to archive/debug_scripts/):"
cat << 'EOF'
   test_real_controlnet.py         - ControlNet residual test
   test_12_residuals.py            - 12 residuals verification
   debug_unet_zip.py               - UNet debugging
   compare_residual_order.py       - Residual order comparison
   trace_unet_residual_order.py    - UNet tracing
   trace_unet_error.py             - Error tracing
   capture_unet_shapes.py          - Shape capturing
   test_sketch_shapes.py           - Shape testing
   test_sketch_injection.py        - Injection testing
   test_sketch_encoder.py          - Encoder testing
EOF
echo ""

# ============================================================================
# Category 3: Old Test Scripts (ARCHIVE)
# ============================================================================
echo "🗄️  ARCHIVE - Old Test Scripts (move to archive/old_tests/):"
cat << 'EOF'
   test_stage1_trained.py          - Old training test
   test_epoch2_from_hf.py          - Epoch 2 test (superseded)
   test_epoch2_custom.py           - Custom epoch 2 test
   test_epoch10.py                 - Epoch 10 test
   test_final_diverse.py           - Final diverse test
   test_final_comprehensive.py     - Comprehensive test
   test_corrected_prompts.py       - Prompt correction test
   test_guidance_scales.py         - Guidance scale test
   test_guidance_scales_simple.py  - Simple guidance test
EOF
echo ""

# ============================================================================
# Category 4: Useful Test Scripts (KEEP, move to tests/)
# ============================================================================
echo "✅ KEEP - Active Test Scripts (move to tests/):"
cat << 'EOF'
   test_all_categories.py          - Test all 125 categories
   regenerate_optimized.py         - Regenerate with optimal settings
   quick_test.py                   - Quick single test
   test_stage1.py                  - Stage 1 unit test
   test_inference.py               - Inference test
EOF
echo ""

# ============================================================================
# Category 5: Utility Scripts (KEEP, maybe rename)
# ============================================================================
echo "✅ KEEP - Utility Scripts:"
cat << 'EOF'
   verify_dataset.py               - Dataset verification
   verify_ide_setup.py             - IDE setup check
   check_sketchy_format.py         - Format checker
   download_dataset.py             - Dataset downloader
   monitor_sync.py                 - Sync monitor
   check_and_clean_hub.py          - HF Hub cleaner
EOF
echo ""

# ============================================================================
# Category 6: Documentation (ORGANIZE)
# ============================================================================
echo "📚 ORGANIZE - Documentation (keep in docs/):"
find . -maxdepth 1 -name "*.md" -type f | while read file; do
    echo "   $(basename $file)"
done
echo ""

# ============================================================================
# Category 7: Shell Scripts (KEEP/ORGANIZE)
# ============================================================================
echo "📜 Shell Scripts:"
find . -maxdepth 1 -name "*.sh" -type f | while read file; do
    echo "   $(basename $file)"
done
echo ""

echo "========================================================================"
echo "🎯 RECOMMENDED ACTIONS"
echo "========================================================================"
echo ""
echo "1. Archive debug scripts (don't need anymore):"
echo "   mv test_*controlnet*.py test_*residuals*.py debug*.py compare*.py trace*.py capture*.py archive/debug_scripts/"
echo ""
echo "2. Archive old test scripts (superseded by better versions):"
echo "   mv test_epoch*.py test_final*.py test_corrected*.py test_guidance_scales*.py archive/old_tests/"
echo ""
echo "3. Organize active tests:"
echo "   mv test_all_categories.py regenerate_optimized.py quick_test.py tests/"
echo "   mv test_stage1.py test_inference.py tests/unit_tests/"
echo ""
echo "4. Organize documentation:"
echo "   mkdir -p docs"
echo "   mv *.md docs/"
echo "   mv docs/README.md ./"  # Keep README at root
echo ""
echo "5. Clean up logs (move to logs/):"
echo "   mv *.log logs/ 2>/dev/null"
echo ""

# Create the actual cleanup script
cat > /tmp/cleanup_project.sh << 'CLEANUP_EOF'
#!/bin/bash
set -e

PROJ_DIR="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
cd "$PROJ_DIR"

echo "🧹 Starting cleanup..."

# Create directories
mkdir -p archive/debug_scripts
mkdir -p archive/old_tests
mkdir -p tests/unit_tests
mkdir -p docs
mkdir -p logs

# 1. Archive debug scripts
echo "📦 Archiving debug scripts..."
mv test_real_controlnet.py archive/debug_scripts/ 2>/dev/null || true
mv test_12_residuals.py archive/debug_scripts/ 2>/dev/null || true
mv debug_unet_zip.py archive/debug_scripts/ 2>/dev/null || true
mv compare_residual_order.py archive/debug_scripts/ 2>/dev/null || true
mv trace_unet_residual_order.py archive/debug_scripts/ 2>/dev/null || true
mv trace_unet_error.py archive/debug_scripts/ 2>/dev/null || true
mv capture_unet_shapes.py archive/debug_scripts/ 2>/dev/null || true
mv test_sketch_shapes.py archive/debug_scripts/ 2>/dev/null || true
mv test_sketch_injection.py archive/debug_scripts/ 2>/dev/null || true
mv test_sketch_encoder.py archive/debug_scripts/ 2>/dev/null || true

# 2. Archive old test scripts
echo "📦 Archiving old test scripts..."
mv test_stage1_trained.py archive/old_tests/ 2>/dev/null || true
mv test_epoch2_from_hf.py archive/old_tests/ 2>/dev/null || true
mv test_epoch2_custom.py archive/old_tests/ 2>/dev/null || true
mv test_epoch10.py archive/old_tests/ 2>/dev/null || true
mv test_final_diverse.py archive/old_tests/ 2>/dev/null || true
mv test_final_comprehensive.py archive/old_tests/ 2>/dev/null || true
mv test_corrected_prompts.py archive/old_tests/ 2>/dev/null || true
mv test_guidance_scales.py archive/old_tests/ 2>/dev/null || true
mv test_guidance_scales_simple.py archive/old_tests/ 2>/dev/null || true

# 3. Organize active tests
echo "✅ Organizing active tests..."
mv test_all_categories.py tests/ 2>/dev/null || true
mv regenerate_optimized.py tests/ 2>/dev/null || true
mv quick_test.py tests/ 2>/dev/null || true
mv test_stage1.py tests/unit_tests/ 2>/dev/null || true
mv test_inference.py tests/unit_tests/ 2>/dev/null || true

# 4. Organize documentation
echo "📚 Organizing documentation..."
for md in *.md; do
    if [ "$md" != "README.md" ] && [ -f "$md" ]; then
        mv "$md" docs/
    fi
done

# 5. Move logs
echo "📝 Organizing logs..."
mv *.log logs/ 2>/dev/null || true

# 6. Clean up compiled Python files
echo "🧹 Cleaning compiled files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

echo ""
echo "========================================================================"
echo "✅ CLEANUP COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 New structure:"
tree -L 2 -d "$PROJ_DIR" 2>/dev/null || find . -type d -maxdepth 2 | head -20

echo ""
echo "📦 Space freed:"
du -sh archive/ 2>/dev/null || echo "Archive directory size: minimal"

echo ""
echo "✨ Project is now cleaner and better organized!"
CLEANUP_EOF

chmod +x /tmp/cleanup_project.sh

echo "========================================================================"
echo "🚀 READY TO EXECUTE"
echo "========================================================================"
echo ""
echo "Review the plan above, then run:"
echo "   bash /tmp/cleanup_project.sh"
echo ""
echo "This will:"
echo "  ✅ Archive debug scripts (10 files)"
echo "  ✅ Archive old test scripts (9 files)"
echo "  ✅ Organize active tests into tests/"
echo "  ✅ Move documentation to docs/"
echo "  ✅ Move logs to logs/"
echo "  ✅ Clean up compiled Python files"
echo ""
echo "💡 Safe operation: Files are moved (not deleted)"
echo "   You can restore any file from archive/ if needed"
echo ""
