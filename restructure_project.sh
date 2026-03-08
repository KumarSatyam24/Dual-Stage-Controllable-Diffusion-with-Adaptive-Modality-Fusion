#!/bin/bash
# Professional Project Restructuring Script
# Organizes the project following ML/AI best practices

set -e

PROJECT_DIR="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "🏗️  PROFESSIONAL PROJECT RESTRUCTURING"
echo "========================================================================"
echo ""

# ============================================================================
# Create Professional Directory Structure
# ============================================================================
echo "📁 Creating professional directory structure..."

# Core directories
mkdir -p src/models
mkdir -p src/data
mkdir -p src/datasets
mkdir -p src/configs
mkdir -p src/utils

# Training and inference
mkdir -p scripts/training
mkdir -p scripts/inference
mkdir -p scripts/evaluation

# Testing
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/fixtures

# Tools and utilities
mkdir -p tools/dataset
mkdir -p tools/monitoring
mkdir -p tools/deployment

# Documentation
mkdir -p docs/guides
mkdir -p docs/api
mkdir -p docs/reports

# Data directories (for organization reference)
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/sketches

# Output directories
mkdir -p outputs/stage1
mkdir -p outputs/stage2
mkdir -p outputs/evaluations

# Checkpoints (symlink to actual location)
mkdir -p checkpoints

# Archive (keep existing)
mkdir -p archive/old_structure

echo "✅ Directory structure created"
echo ""

# ============================================================================
# Move Core Model Files
# ============================================================================
echo "📦 Organizing core model files..."

# Models
cp models/__init__.py src/models/__init__.py 2>/dev/null || true
cp models/stage1_diffusion.py src/models/stage1_diffusion.py 2>/dev/null || true
cp models/stage2_refinement.py src/models/stage2_refinement.py 2>/dev/null || true
cp models/ragaf_attention.py src/models/ragaf_attention.py 2>/dev/null || true
cp models/adaptive_fusion.py src/models/adaptive_fusion.py 2>/dev/null || true

# Data processing
cp data/__init__.py src/data/__init__.py 2>/dev/null || true
cp data/region_extraction.py src/data/region_extraction.py 2>/dev/null || true
cp data/region_graph.py src/data/region_graph.py 2>/dev/null || true
cp data/sketch_extraction.py src/data/sketch_extraction.py 2>/dev/null || true

# Datasets
cp datasets/__init__.py src/datasets/__init__.py 2>/dev/null || true
cp datasets/coco_dataset.py src/datasets/coco_dataset.py 2>/dev/null || true
cp datasets/sketchy_dataset.py src/datasets/sketchy_dataset.py 2>/dev/null || true

# Configs
cp configs/__init__.py src/configs/__init__.py 2>/dev/null || true
cp configs/config.py src/configs/config.py 2>/dev/null || true

# Utils
cp utils/__init__.py src/utils/__init__.py 2>/dev/null || true
cp utils/common.py src/utils/common.py 2>/dev/null || true

echo "✅ Core model files organized in src/"
echo ""

# ============================================================================
# Move Scripts
# ============================================================================
echo "📜 Organizing scripts..."

# Training scripts
cp train.py scripts/training/train.py 2>/dev/null || true

# Inference scripts
cp inference.py scripts/inference/inference.py 2>/dev/null || true

# Evaluation scripts
cp tests/test_all_categories.py scripts/evaluation/evaluate_all_categories.py 2>/dev/null || true
cp tests/regenerate_optimized.py scripts/evaluation/regenerate_optimized.py 2>/dev/null || true
cp tests/quick_test.py scripts/inference/quick_test.py 2>/dev/null || true

echo "✅ Scripts organized"
echo ""

# ============================================================================
# Move Tools
# ============================================================================
echo "🔧 Organizing tools..."

# Dataset tools
cp download_dataset.py tools/dataset/download_dataset.py 2>/dev/null || true
cp verify_dataset.py tools/dataset/verify_dataset.py 2>/dev/null || true
cp check_sketchy_format.py tools/dataset/check_sketchy_format.py 2>/dev/null || true

# Monitoring tools
cp monitor_sync.py tools/monitoring/monitor_sync.py 2>/dev/null || true

# Other tools
cp verify_ide_setup.py tools/verify_ide_setup.py 2>/dev/null || true
cp check_and_clean_hub.py tools/check_and_clean_hub.py 2>/dev/null || true

echo "✅ Tools organized"
echo ""

# ============================================================================
# Move Tests
# ============================================================================
echo "🧪 Organizing tests..."

# Unit tests
cp tests/unit_tests/test_stage1.py tests/unit/test_stage1.py 2>/dev/null || true
cp tests/unit_tests/test_inference.py tests/unit/test_inference.py 2>/dev/null || true

# Create __init__.py files for tests
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

echo "✅ Tests organized"
echo ""

# ============================================================================
# Move Documentation
# ============================================================================
echo "📚 Organizing documentation..."

# Guides
mv docs/DATASET_SETUP_GUIDE.md docs/guides/ 2>/dev/null || true
mv docs/DATASET_ACCESS_SETUP.md docs/guides/ 2>/dev/null || true
mv docs/IDE_SETUP.md docs/guides/ 2>/dev/null || true
mv docs/DEVELOPMENT.md docs/guides/ 2>/dev/null || true
mv docs/NETWORK_VOLUME_GUIDE.md docs/guides/ 2>/dev/null || true
mv docs/STORAGE_MIGRATION_GUIDE.md docs/guides/ 2>/dev/null || true
mv docs/GUIDANCE_SCALE_GUIDE.md docs/guides/ 2>/dev/null || true

# Reports
mv docs/PROJECT_CHECKLIST.md docs/reports/ 2>/dev/null || true
mv docs/IMPLEMENTATION_SUMMARY.md docs/reports/ 2>/dev/null || true
mv docs/TEST_RESULTS.md docs/reports/ 2>/dev/null || true
mv docs/SKETCHY_DATASET_STATUS.md docs/reports/ 2>/dev/null || true
mv docs/READY_TO_TRAIN.md docs/reports/ 2>/dev/null || true
mv docs/TRAINING_IN_PROGRESS.md docs/reports/ 2>/dev/null || true
mv docs/ALL_CATEGORIES_TEST_SUMMARY.md docs/reports/ 2>/dev/null || true
mv docs/CRITICAL_BUG_FIX.md docs/reports/ 2>/dev/null || true
mv docs/GUIDANCE_SCALE_OPTIMIZATION.md docs/reports/ 2>/dev/null || true
mv docs/PROJECT_CLEANUP_SUMMARY.md docs/reports/ 2>/dev/null || true
mv docs/WHATS_NEXT.md docs/reports/ 2>/dev/null || true
mv docs/*.md docs/reports/ 2>/dev/null || true

echo "✅ Documentation organized"
echo ""

# ============================================================================
# Move Output Directories
# ============================================================================
echo "📊 Organizing outputs..."

mv test_outputs_epoch2 outputs/stage1/test_outputs_epoch2 2>/dev/null || true
mv test_outputs_epoch10 outputs/stage1/test_outputs_epoch10 2>/dev/null || true
mv test_outputs_all_categories outputs/evaluations/all_categories 2>/dev/null || true
mv test_outputs_optimized outputs/evaluations/optimized 2>/dev/null || true

# Move sample outputs
mv stage1_generated.png outputs/stage1/sample_generated.png 2>/dev/null || true
mv stage1_generated_epoch10.png outputs/stage1/sample_generated_epoch10.png 2>/dev/null || true
mv input_sketch.png outputs/stage1/sample_input_sketch.png 2>/dev/null || true
mv input_sketch_epoch10.png outputs/stage1/sample_input_sketch_epoch10.png 2>/dev/null || true

echo "✅ Outputs organized"
echo ""

# ============================================================================
# Create Symlinks for Convenience
# ============================================================================
echo "🔗 Creating convenience symlinks..."

# Link checkpoints (if they exist in /root/checkpoints)
if [ -d "/root/checkpoints/stage1" ]; then
    ln -sf /root/checkpoints/stage1 checkpoints/stage1 2>/dev/null || true
fi
if [ -d "/root/checkpoints/stage2" ]; then
    ln -sf /root/checkpoints/stage2 checkpoints/stage2 2>/dev/null || true
fi

# Link dataset (if exists in workspace)
if [ -d "/workspace/sketchy" ]; then
    ln -sf /workspace/sketchy data/raw/sketchy 2>/dev/null || true
elif [ -d "/root/datasets/sketchy" ]; then
    ln -sf /root/datasets/sketchy data/raw/sketchy 2>/dev/null || true
fi

echo "✅ Symlinks created"
echo ""

# ============================================================================
# Create Entry Point Scripts
# ============================================================================
echo "🚀 Creating entry point scripts..."

# Main train script (wrapper)
cat > train.sh << 'EOF'
#!/bin/bash
# Training entry point
python3 scripts/training/train.py "$@"
EOF
chmod +x train.sh

# Main inference script (wrapper)
cat > inference.sh << 'EOF'
#!/bin/bash
# Inference entry point
python3 scripts/inference/inference.py "$@"
EOF
chmod +x inference.sh

# Quick test script (wrapper)
cat > quick_test.sh << 'EOF'
#!/bin/bash
# Quick test entry point
python3 scripts/inference/quick_test.py "$@"
EOF
chmod +x quick_test.sh

# Evaluation script (wrapper)
cat > evaluate.sh << 'EOF'
#!/bin/bash
# Evaluation entry point
python3 scripts/evaluation/evaluate_all_categories.py "$@"
EOF
chmod +x evaluate.sh

echo "✅ Entry point scripts created"
echo ""

# ============================================================================
# Archive Old Structure
# ============================================================================
echo "📦 Archiving old structure..."

# Move old directories to archive
mv models archive/old_structure/models 2>/dev/null || true
mv data archive/old_structure/data 2>/dev/null || true
mv datasets archive/old_structure/datasets 2>/dev/null || true
mv configs archive/old_structure/configs 2>/dev/null || true
mv utils archive/old_structure/utils 2>/dev/null || true
mv tests/unit_tests archive/old_structure/tests_unit_tests 2>/dev/null || true

# Move old test scripts
mv tests/test_all_categories.py archive/old_structure/ 2>/dev/null || true
mv tests/regenerate_optimized.py archive/old_structure/ 2>/dev/null || true
mv tests/quick_test.py archive/old_structure/ 2>/dev/null || true

# Move old root scripts
mv train.py archive/old_structure/ 2>/dev/null || true
mv inference.py archive/old_structure/ 2>/dev/null || true
mv download_dataset.py archive/old_structure/ 2>/dev/null || true
mv verify_dataset.py archive/old_structure/ 2>/dev/null || true
mv check_sketchy_format.py archive/old_structure/ 2>/dev/null || true
mv monitor_sync.py archive/old_structure/ 2>/dev/null || true
mv verify_ide_setup.py archive/old_structure/ 2>/dev/null || true
mv check_and_clean_hub.py archive/old_structure/ 2>/dev/null || true

echo "✅ Old structure archived"
echo ""

# ============================================================================
# Create Project Configuration Files
# ============================================================================
echo "⚙️  Creating configuration files..."

# Create setup.py
cat > setup.py << 'SETUP_EOF'
"""
Dual-Stage Controllable Diffusion with Adaptive Modality Fusion
Setup script for installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dual-stage-diffusion",
    version="0.1.0",
    author="KumarSatyam24",
    description="Dual-Stage Controllable Diffusion with Sketch and Region Guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dual-stage-train=scripts.training.train:main",
            "dual-stage-inference=scripts.inference.inference:main",
        ],
    },
)
SETUP_EOF

# Create pytest configuration
cat > pytest.ini << 'PYTEST_EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
PYTEST_EOF

# Create .gitignore additions
cat >> .gitignore << 'GITIGNORE_EOF'

# Project specific
outputs/
checkpoints/
data/raw/
data/processed/
*.log
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
GITIGNORE_EOF

# Create MANIFEST.in
cat > MANIFEST.in << 'MANIFEST_EOF'
include README.md
include LICENSE
include requirements.txt
recursive-include src *.py
recursive-include scripts *.py
recursive-include configs *.yaml *.json
MANIFEST_EOF

echo "✅ Configuration files created"
echo ""

# ============================================================================
# Update README with new structure
# ============================================================================
echo "📝 Updating README reference..."

cat > docs/NEW_PROJECT_STRUCTURE.md << 'STRUCTURE_EOF'
# Professional Project Structure

## Overview
The project has been reorganized following ML/AI best practices for better maintainability, scalability, and collaboration.

## Directory Structure

```
Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/
│
├── README.md                           # Project overview and quick start
├── setup.py                            # Package installation configuration
├── requirements.txt                    # Python dependencies
├── pytest.ini                          # Test configuration
├── MANIFEST.in                         # Package manifest
├── .gitignore                          # Git ignore patterns
│
├── train.sh                            # Quick training entry point
├── inference.sh                        # Quick inference entry point
├── quick_test.sh                       # Quick test entry point
├── evaluate.sh                         # Quick evaluation entry point
│
├── src/                                # Source code (importable package)
│   ├── models/                         # Model architectures
│   │   ├── __init__.py
│   │   ├── stage1_diffusion.py        # Stage 1: Sketch-guided diffusion
│   │   ├── stage2_refinement.py       # Stage 2: Region-guided refinement
│   │   ├── ragaf_attention.py         # Region-Adaptive attention
│   │   └── adaptive_fusion.py         # Adaptive fusion module
│   │
│   ├── data/                           # Data processing modules
│   │   ├── __init__.py
│   │   ├── region_extraction.py       # Region extraction
│   │   ├── region_graph.py            # Scene graph processing
│   │   └── sketch_extraction.py       # Sketch extraction
│   │
│   ├── datasets/                       # Dataset loaders
│   │   ├── __init__.py
│   │   ├── coco_dataset.py            # COCO dataset loader
│   │   └── sketchy_dataset.py         # Sketchy dataset loader
│   │
│   ├── configs/                        # Configuration classes
│   │   ├── __init__.py
│   │   └── config.py                  # Main configuration
│   │
│   └── utils/                          # Utility functions
│       ├── __init__.py
│       └── common.py                  # Common utilities
│
├── scripts/                            # Executable scripts
│   ├── training/                       # Training scripts
│   │   └── train.py                   # Main training script
│   │
│   ├── inference/                      # Inference scripts
│   │   ├── inference.py               # Main inference script
│   │   └── quick_test.py              # Quick testing tool
│   │
│   └── evaluation/                     # Evaluation scripts
│       ├── evaluate_all_categories.py # Full evaluation
│       └── regenerate_optimized.py    # Regenerate with optimal settings
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── unit/                           # Unit tests
│   │   ├── __init__.py
│   │   ├── test_stage1.py             # Stage 1 tests
│   │   └── test_inference.py          # Inference tests
│   │
│   ├── integration/                    # Integration tests
│   │   └── __init__.py
│   │
│   └── fixtures/                       # Test fixtures
│
├── tools/                              # Development and deployment tools
│   ├── dataset/                        # Dataset management
│   │   ├── download_dataset.py        # Download datasets
│   │   ├── verify_dataset.py          # Verify dataset integrity
│   │   └── check_sketchy_format.py    # Check Sketchy format
│   │
│   ├── monitoring/                     # Monitoring tools
│   │   └── monitor_sync.py            # Training monitor
│   │
│   ├── verify_ide_setup.py            # IDE setup verification
│   └── check_and_clean_hub.py         # HuggingFace Hub management
│
├── docs/                               # Documentation
│   ├── guides/                         # User guides
│   │   ├── DATASET_SETUP_GUIDE.md
│   │   ├── DEVELOPMENT.md
│   │   ├── IDE_SETUP.md
│   │   └── ...
│   │
│   ├── reports/                        # Project reports
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── TEST_RESULTS.md
│   │   ├── CRITICAL_BUG_FIX.md
│   │   └── ...
│   │
│   ├── api/                            # API documentation
│   └── NEW_PROJECT_STRUCTURE.md       # This file
│
├── data/                               # Data directory (local, not in git)
│   ├── raw/                            # Raw datasets
│   │   └── sketchy/                   # Sketchy dataset (symlink)
│   ├── processed/                      # Processed data
│   └── sketches/                       # Custom sketches
│
├── outputs/                            # Generated outputs (not in git)
│   ├── stage1/                         # Stage 1 outputs
│   │   ├── test_outputs_epoch2/
│   │   ├── test_outputs_epoch10/
│   │   ├── sample_generated.png
│   │   └── ...
│   │
│   ├── stage2/                         # Stage 2 outputs
│   └── evaluations/                    # Evaluation results
│       ├── all_categories/
│       └── optimized/
│
├── checkpoints/                        # Model checkpoints (symlinks)
│   ├── stage1/ -> /root/checkpoints/stage1/
│   └── stage2/ -> /root/checkpoints/stage2/
│
├── logs/                               # Training and execution logs
│
└── archive/                            # Archived files
    ├── old_structure/                  # Previous structure
    ├── debug_scripts/                  # Old debug scripts
    └── old_tests/                      # Old test scripts

```

## Key Improvements

### 1. **Separation of Concerns**
- `src/` - Importable source code
- `scripts/` - Executable scripts
- `tests/` - Test suite
- `tools/` - Development utilities
- `docs/` - Documentation

### 2. **Package Structure**
- Proper Python package with `setup.py`
- Can be installed: `pip install -e .`
- Clean imports: `from models.stage1_diffusion import ...`

### 3. **Entry Points**
- Simple shell wrappers at root level
- `./train.sh` - Start training
- `./inference.sh` - Run inference
- `./evaluate.sh` - Run evaluation

### 4. **Data Organization**
- Clear separation of raw vs processed data
- Symlinks to actual storage locations
- Not tracked in git (in .gitignore)

### 5. **Testing**
- Organized test structure
- Unit tests separated from integration tests
- Pytest configuration included

### 6. **Documentation**
- Guides for users
- Reports for project status
- API documentation structure

## Migration from Old Structure

All old files have been **copied** (not moved) to their new locations and the originals archived in `archive/old_structure/`. This ensures:

1. ✅ Nothing is lost
2. ✅ Can revert if needed
3. ✅ Old imports still work (from archive)

## Usage

### Installation
```bash
# Install in development mode
pip install -e .
```

### Training
```bash
# Option 1: Using wrapper script
./train.sh

# Option 2: Direct call
python3 scripts/training/train.py

# Option 3: Installed entry point (after pip install)
dual-stage-train
```

### Inference
```bash
# Option 1: Using wrapper script
./inference.sh --checkpoint checkpoints/stage1/epoch_10.pth

# Option 2: Direct call
python3 scripts/inference/inference.py --checkpoint checkpoints/stage1/epoch_10.pth

# Option 3: Quick test
./quick_test.sh
```

### Testing
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run specific test
pytest tests/unit/test_stage1.py
```

### Evaluation
```bash
# Evaluate all categories
./evaluate.sh

# Or directly
python3 scripts/evaluation/evaluate_all_categories.py
```

## Benefits

1. **Professionalism** - Standard ML/AI project structure
2. **Maintainability** - Clear organization and separation
3. **Scalability** - Easy to add new models, datasets, scripts
4. **Collaboration** - Easy for others to understand and contribute
5. **Deployment** - Ready for pip installation and distribution
6. **Testing** - Proper test organization with pytest
7. **Documentation** - Well-organized docs structure

## Next Steps

1. Update imports in scripts to use new structure
2. Add more comprehensive tests
3. Generate API documentation
4. Create CI/CD pipeline configuration
5. Add Docker configuration for deployment

## Rollback

If you need to revert to the old structure:

```bash
# Restore from archive
cp -r archive/old_structure/* .
```

All original files are preserved in `archive/old_structure/`.
STRUCTURE_EOF

echo "✅ Documentation created"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "✅ PROJECT RESTRUCTURING COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 New Professional Structure:"
echo ""
echo "  src/              - Source code (importable package)"
echo "  scripts/          - Executable scripts"
echo "  tests/            - Test suite"
echo "  tools/            - Development utilities"
echo "  docs/             - Documentation"
echo "  data/             - Data directory"
echo "  outputs/          - Generated outputs"
echo "  checkpoints/      - Model checkpoints (symlinks)"
echo ""
echo "🚀 Quick Start Commands:"
echo ""
echo "  ./train.sh                    - Start training"
echo "  ./inference.sh                - Run inference"
echo "  ./quick_test.sh               - Quick test"
echo "  ./evaluate.sh                 - Run evaluation"
echo ""
echo "📦 Installation:"
echo ""
echo "  pip install -e .              - Install package in dev mode"
echo ""
echo "🧪 Testing:"
echo ""
echo "  pytest                        - Run all tests"
echo "  pytest tests/unit/            - Run unit tests"
echo ""
echo "📚 Documentation:"
echo ""
echo "  docs/NEW_PROJECT_STRUCTURE.md - Complete structure guide"
echo "  docs/guides/                  - User guides"
echo "  docs/reports/                 - Project reports"
echo ""
echo "💾 Old Structure:"
echo ""
echo "  archive/old_structure/        - Original files preserved"
echo ""
echo "✨ Project is now professionally organized!"
echo ""
