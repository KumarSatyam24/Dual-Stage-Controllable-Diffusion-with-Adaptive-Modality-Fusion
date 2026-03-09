#!/bin/bash

# 🧹 Clean Project for GitHub Upload
# This script removes unnecessary files and prepares the repo for GitHub

set -e

echo "🧹 Cleaning Project for GitHub"
echo "=" * 60
echo ""

PROJECT_ROOT="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
cd "$PROJECT_ROOT"

echo "📁 Current directory: $(pwd)"
echo ""

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# PyCharm
.idea/

# VS Code
.vscode/

# Data & Checkpoints (DO NOT UPLOAD TO GITHUB!)
*.pt
*.pth
*.ckpt
*.safetensors
checkpoints/
outputs/
wandb/
validation_results/
test_validation/

# Dataset (DO NOT UPLOAD TO GITHUB!)
data/
datasets/
sketchy/
*.tar.gz
*.zip

# Logs
*.log
logs/
nohup.out

# Temporary files
*.tmp
*.swp
*.swo
*~
.DS_Store
tmp/

# Cache
.cache/
*.cache

# Training artifacts
train_*.log
*.png
*.jpg
*.jpeg
!docs/**/*.png
!docs/**/*.jpg

# System
.Trash-*
EOF
    echo "✅ .gitignore created!"
else
    echo "✅ .gitignore already exists"
fi

echo ""
echo "🗑️  Removing files that shouldn't be in GitHub..."
echo ""

# Remove Python cache
echo "  Removing __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove Jupyter checkpoints
echo "  Removing .ipynb_checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# Remove logs (keep in gitignore, don't need in repo)
echo "  Removing log files..."
find . -maxdepth 1 -type f -name "*.log" -delete 2>/dev/null || true
rm -f nohup.out 2>/dev/null || true

# Remove temporary files
echo "  Removing temporary files..."
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name "*.swp" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true

# Remove .DS_Store (Mac)
echo "  Removing .DS_Store files..."
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

echo ""
echo "✅ Cleanup complete!"
echo ""

# Show what will be committed
echo "📊 Files that will be in GitHub:"
echo "================================"
git status --short 2>/dev/null || ls -la | grep -v "^d" | awk '{print $9}' | grep -v "^\." | head -20

echo ""
echo "📊 Current directory size:"
du -sh . 2>/dev/null || echo "Size calculation unavailable"

echo ""
echo "⚠️  IMPORTANT: The following should NOT be in GitHub:"
echo "  - Checkpoints (*.pt, *.pth files)"
echo "  - Dataset files (data/, sketchy/)"
echo "  - Training logs (*.log)"
echo "  - WandB runs (wandb/)"
echo "  - Validation results (validation_results/)"
echo ""
echo "✅ These are now in .gitignore and won't be uploaded!"
echo ""

# Check for large files
echo "🔍 Checking for large files (>50MB)..."
find . -type f -size +50M 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  ⚠️  Large file: $file ($size)"
done || echo "  ✅ No large files found"

echo ""
echo "📝 Recommended .gitattributes for better Git handling:"
cat > .gitattributes << 'EOF'
# Python
*.py text eol=lf

# Shell scripts
*.sh text eol=lf

# Markdown
*.md text eol=lf

# JSON
*.json text eol=lf

# YAML
*.yaml text eol=lf
*.yml text eol=lf

# Images (binary)
*.png binary
*.jpg binary
*.jpeg binary

# Archives (binary)
*.tar.gz binary
*.zip binary
EOF

echo "✅ .gitattributes created!"
echo ""

echo "🚀 Ready for GitHub!"
echo ""
echo "Next steps:"
echo "  1. Review files: git status"
echo "  2. Add files: git add ."
echo "  3. Commit: git commit -m 'Clean project for GitHub'"
echo "  4. Push: git push origin main"
echo ""
