# /workspace Directory Cleanup Analysis

## Current Contents (Unnecessary Files)

### 🗑️ **Test Output Directories** (7 directories, ~1-2 GB)
These are old inference test results that are no longer needed:
- ❌ `test_outputs_all_categories/`
- ❌ `test_outputs_corrected/`
- ❌ `test_outputs_epoch10/`
- ❌ `test_outputs_final_diverse/`
- ❌ `test_outputs_guidance_comparison/`
- ❌ `test_outputs_guidance_scales/`
- ❌ `test_outputs_optimized/`

### 🗑️ **Comparison Images** (~35+ files, ~50-100 MB)
Old test output images from various experiments:
- ❌ `comparison_grid_01.png` through `comparison_grid_16.png` (16 files)
- ❌ `optimized_grid_01.png` through `optimized_grid_03.png` (3 files)
- ❌ `airplane_guidance_comparison.png`
- ❌ `bicycle_guidance_comparison.png`
- ❌ `chair_guidance_comparison.png`
- ❌ `cup_comparison.png`
- ❌ `eyeglasses_comparison.png`
- ❌ `eyeglasses_guidance_comparison.png`
- ❌ `fighter_jet_epoch2.png`
- ❌ `final_diverse_comparison.png`
- ❌ `hat_comparison.png`
- ❌ `knife_comparison.png`
- ❌ `shoe_comparison.png`
- ❌ `spoon_comparison.png`
- ❌ `epoch2_test_output.png`

### 🗑️ **Old Log Files** (7 files, ~10-50 MB)
Training and test logs from old experiments:
- ❌ `test_all_categories.log`
- ❌ `test_corrected.log`
- ❌ `test_output.txt`
- ❌ `epoch2_test.log`
- ❌ `train_stage1.log`
- ❌ `train_stage1_FIXED.log`
- ❌ `train_stage2.log`

### 🗑️ **Documentation Files** (3 files, ~5 KB)
Old documentation that should be in project directory:
- ❌ `GRIDS_LOCATION.md`
- ❌ `HOW_TO_VIEW_RESULTS.md`
- ❌ `view_all_grids.html`

## Files to KEEP

### ✅ **Essential Directories**
- ✅ `sketchy/` - Sketchy dataset (required for training)
- ✅ `checkpoints/` - Model checkpoints (if any)
- ✅ `dataset/` - Dataset files (if exists)

## Quick Cleanup Commands

### Option 1: Run cleanup script
```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion
bash cleanup_workspace_external.sh
```

### Option 2: Manual cleanup (if you want control)
```bash
cd /workspace

# Remove test output directories
rm -rf test_outputs_*

# Remove comparison images
rm -f comparison_grid_*.png
rm -f *_comparison.png
rm -f optimized_grid_*.png
rm -f epoch2_test_output.png
rm -f fighter_jet_epoch2.png
rm -f final_diverse_comparison.png

# Remove old logs
rm -f *.log test_output.txt

# Remove documentation
rm -f GRIDS_LOCATION.md HOW_TO_VIEW_RESULTS.md view_all_grids.html

# Verify what's left
ls -lh
```

### Option 3: One-liner (fastest)
```bash
cd /workspace && rm -rf test_outputs_* && rm -f comparison_grid_*.png *_comparison.png optimized_grid_*.png epoch2_test_output.png fighter_jet_epoch2.png final_diverse_comparison.png *.log test_output.txt GRIDS_LOCATION.md HOW_TO_VIEW_RESULTS.md view_all_grids.html && echo "✅ Cleanup complete!" && ls -lh
```

## Estimated Space Savings

- **Test output directories:** ~1-2 GB
- **Comparison images:** ~50-100 MB
- **Log files:** ~10-50 MB
- **Documentation:** ~5 KB

**Total:** ~1-2 GB freed

## Safety

✅ **100% Safe** - None of these files are needed for:
- Current training (running in `/root/...`)
- Dataset (sketchy/ is preserved)
- Checkpoints (checkpoints/ is preserved)

These are all old test/experiment artifacts.
