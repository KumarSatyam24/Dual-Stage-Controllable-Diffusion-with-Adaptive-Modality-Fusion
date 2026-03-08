#!/bin/bash
# Migrate large files from /workspace/ (10GB volume, 81% full)
# to /root/ (115GB disk, 62GB available)

echo "========================================================================"
echo "🔄 WORKSPACE TO DISK MIGRATION PLAN"
echo "========================================================================"
echo ""

# Current usage
echo "📊 Current Storage Usage:"
echo "   /workspace/: 8.1G / 10G (81% full) ⚠️"
echo "   /root/:      54G / 115G (47% full) ✅"
echo ""

# What to move
echo "========================================================================"
echo "📦 Items to Move:"
echo "========================================================================"
echo ""

echo "1. Sketchy Dataset: 1.2 GB"
echo "   FROM: /workspace/sketchy/"
echo "   TO:   /root/datasets/sketchy/"
echo "   Why: Dataset used for training/testing"
echo ""

echo "2. Test Outputs: ~100 MB total"
echo "   FROM: /workspace/test_outputs_*/"
echo "   TO:   /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/test_outputs_*/"
echo "   Why: Keep test results with project code"
echo ""

echo "3. Training Logs: 29 MB"
echo "   FROM: /workspace/*.log"
echo "   TO:   /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/logs/"
echo "   Why: Archive logs with project"
echo ""

echo "4. Comparison Grids: 16 MB (keep copies in workspace for easy access)"
echo "   Keep in: /workspace/*.png (for download)"
echo "   Archive: /root/.../archived_grids/"
echo ""

echo "========================================================================"
echo "💾 Space to Free: ~1.3 GB"
echo "   Workspace after: ~6.8G / 10G (68% full) ✅"
echo "========================================================================"
echo ""

# Create migration script
cat > /tmp/migrate.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting migration..."

# 1. Move Sketchy dataset
echo "📦 Moving Sketchy dataset..."
mkdir -p /root/datasets
if [ -d "/workspace/sketchy" ]; then
    mv /workspace/sketchy /root/datasets/
    ln -s /root/datasets/sketchy /workspace/sketchy
    echo "✅ Sketchy dataset moved and symlinked"
else
    echo "⚠️  Sketchy already moved or not found"
fi

# 2. Move test outputs
echo "📦 Moving test outputs..."
PROJ_DIR="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
for dir in /workspace/test_outputs_*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        mv "$dir" "$PROJ_DIR/"
        echo "✅ Moved $dirname"
    fi
done

# 3. Move training logs
echo "📦 Moving training logs..."
mkdir -p "$PROJ_DIR/logs"
for log in /workspace/train*.log /workspace/*test*.log; do
    if [ -f "$log" ]; then
        mv "$log" "$PROJ_DIR/logs/"
        echo "✅ Moved $(basename $log)"
    fi
done

# 4. Archive old comparison grids (keep copies in workspace)
echo "📦 Archiving old comparison grids..."
mkdir -p "$PROJ_DIR/archived_grids"
for grid in /workspace/comparison_grid_*.png; do
    if [ -f "$grid" ]; then
        cp "$grid" "$PROJ_DIR/archived_grids/"
    fi
done
echo "✅ Old grids archived (originals kept in workspace)"

# Keep optimized grids in workspace for easy access
echo "✅ Optimized grids remain in /workspace/ for download"

echo ""
echo "========================================================================"
echo "✅ MIGRATION COMPLETE!"
echo "========================================================================"
df -h /workspace | tail -1
df -h / | tail -1

EOF

chmod +x /tmp/migrate.sh

echo "========================================================================"
echo "🚀 Ready to Execute"
echo "========================================================================"
echo ""
echo "Review the plan above, then run:"
echo "   bash /tmp/migrate.sh"
echo ""
echo "This will:"
echo "  1. Move sketchy dataset to /root/datasets/ (1.2 GB freed)"
echo "  2. Move test outputs to project directory (~100 MB freed)"
echo "  3. Move logs to project logs/ directory (29 MB freed)"
echo "  4. Archive old grids (keep in workspace for download)"
echo ""
echo "Total space freed: ~1.3 GB"
echo "Workspace after: ~6.8G / 10G (68% full)"
echo ""
