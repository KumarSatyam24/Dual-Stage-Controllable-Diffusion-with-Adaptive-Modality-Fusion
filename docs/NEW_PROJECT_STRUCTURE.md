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
