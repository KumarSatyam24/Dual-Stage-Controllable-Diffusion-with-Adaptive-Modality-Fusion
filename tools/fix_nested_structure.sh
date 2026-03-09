#!/bin/bash

# 🧹 Fix Nested Directory Structure and Clean for GitHub

set -e

echo "🧹 Cleaning Repository Structure for GitHub"
echo "============================================"
echo ""

# Paths
OUTER_DIR="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
INNER_DIR="$OUTER_DIR/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
TEMP_DIR="/tmp/repo_cleanup_temp"

echo "📂 Current structure:"
echo "   Outer: $OUTER_DIR"
echo "   Inner: $INNER_DIR"
echo ""

# Check if nested directory exists
if [ ! -d "$INNER_DIR" ]; then
    echo "✅ No nested directory found. Structure is already clean!"
    exit 0
fi

echo "⚠️  Found nested duplicate directory!"
echo ""

# Ask for confirmation
read -p "This will move all files from inner directory to outer. Continue? (yes/NO): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting cleanup..."
echo ""

# Create temp directory
mkdir -p "$TEMP_DIR"

# Step 1: Move everything from inner directory to temp
echo "📦 Step 1: Moving files to temporary location..."
mv "$INNER_DIR"/* "$TEMP_DIR/" 2>/dev/null || true
mv "$INNER_DIR"/.* "$TEMP_DIR/" 2>/dev/null || true

# Step 2: Remove the now-empty inner directory
echo "🗑️  Step 2: Removing nested directory..."
rmdir "$INNER_DIR"

# Step 3: Move everything from temp back to outer directory
echo "📥 Step 3: Moving files to correct location..."
cd "$TEMP_DIR"
for item in * .[^.]* ..?*; do
    if [ -e "$item" ] && [ "$item" != "." ] && [ "$item" != ".." ]; then
        # Check if item exists in outer dir
        if [ -e "$OUTER_DIR/$item" ]; then
            echo "   ⚠️  Conflict: $item exists in both locations"
            echo "   Keeping outer version, skipping inner version"
        else
            mv "$item" "$OUTER_DIR/"
            echo "   ✅ Moved: $item"
        fi
    fi
done 2>/dev/null || true

# Step 4: Clean up temp
echo "🧹 Step 4: Cleaning temporary files..."
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Directory structure fixed!"
echo ""
echo "📂 New structure:"
ls -la "$OUTER_DIR" | head -20

echo ""
echo "🎉 Cleanup complete!"
echo ""
echo "Next steps for GitHub:"
echo "  1. Review .gitignore file"
echo "  2. Remove checkpoints and large files"
echo "  3. git add ."
echo "  4. git commit -m 'Clean repository structure'"
echo "  5. git push"
