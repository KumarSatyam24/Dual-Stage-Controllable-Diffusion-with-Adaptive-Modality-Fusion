# Workspace Cleanup Analysis

## Files Safe to Delete

### 1. **Duplicate/Obsolete Validation Scripts** (7 files, ~50KB)
These are replaced by `validate_epochs.py`:
- ❌ `evaluate_stage1_accuracy.py` - Old validation script
- ❌ `evaluate_stage1_metrics.py` - Duplicate functionality
- ❌ `evaluate_stage1_simple.py` - Basic version
- ❌ `evaluate_stage1_validation.py` - Another duplicate
- ❌ `test_epoch_validation.py` - Test script
- ❌ `test_validation_quick.py` - Quick test
- ❌ `debug_checkpoint.py` - Debugging script

**Keep:** `validate_epochs.py` (current production script)

### 2. **Duplicate Documentation** (12 files, ~100KB)
Multiple overlapping guides:
- ❌ `ACCURACY_EVALUATION_GUIDE.md` - Covered in TRAINING_IMPROVEMENTS.md
- ❌ `ACCURACY_EXPLAINED.md` - Redundant
- ❌ `HOW_TO_MEASURE_ACCURACY.md` - Duplicate
- ❌ `IMPROVEMENT_PLAN.md` - Old plan (completed)
- ❌ `STAGE1_VALIDATION_SUMMARY.md` - Outdated
- ❌ `VALIDATION_COMMANDS.md` - Covered in README_IMPROVEMENTS.md
- ❌ `VALIDATION_METRICS_GUIDE.md` - Duplicate
- ❌ `VALIDATION_READY.md` - Outdated status
- ❌ `VALIDATION_RESULTS_ANALYSIS.md` - Old analysis
- ❌ `VALIDATION_SETUP_COMPLETE.md` - Outdated status
- ❌ `QUICK_START_SUMMARY.txt` - Replaced by README_IMPROVEMENTS.md

**Keep:** 
- `README.md` (main documentation)
- `README_IMPROVEMENTS.md` (current quick start)
- `TRAINING_IMPROVEMENTS.md` (current analysis)
- `WANDB_SETUP.md` (WandB guide)

### 3. **Old Shell Scripts** (6 files, ~20KB)
Replaced by new scripts:
- ❌ `cleanup_project.sh` - Old cleanup
- ❌ `evaluate.sh` - Old evaluation
- ❌ `inference.sh` - Not used
- ❌ `quick_test.sh` - Old test
- ❌ `quick_validate.sh` - Old validation
- ❌ `run_accuracy_evaluation.sh` - Old evaluation
- ❌ `run_validation.sh` - Old validation
- ❌ `view_grids.sh` - Viewer script

**Keep:**
- `start_improved_training.sh` (current training script)
- `quickstart.sh` (if still relevant)
- `train.sh` (if used)

### 4. **Old Configuration** (1 file, ~5KB)
- ❌ `config_improved.py` - Not used (using configs/config.py)

### 5. **Old Validation Results** (2 directories, ~500MB potential)
These contain results from FAILED model (SSIM=0.25):
- ❌ `validation_results/` - Old failed validation results
- ❌ `test_validation/` - Test validation results

**Keep:** Only the JSON metrics for comparison reference

### 6. **Archive Directory** (entire directory, size unknown)
Contains old code that's been refactored:
- ⚠️ `archive/` - Review before deletion (may have useful reference code)

### 7. **Log Files** (3 files, variable size)
Old training/test logs:
- ❌ `logs/test_epoch10_results.log`
- ❌ `logs/test_stage1_results.log`
- ❌ `logs/train_stage1.log`
- ❌ `evaluation_output.log`
- ❌ `validation_test.log`

**Keep:** `train.log` (current training output)

### 8. **Python Cache** (1 directory)
- ❌ `__pycache__/` - Should be in .gitignore anyway

### 9. **WandB Cache** (1 directory, large)
- ⚠️ `wandb/` - Local WandB cache (can be regenerated, check size first)

## Files to KEEP

### Essential Code
- ✅ `train_improved_stage1.py` - Current training script
- ✅ `validate_epochs.py` - Current validation script
- ✅ All files in `src/`, `configs/`, `data/`, `datasets/`, `models/`, `utils/`
- ✅ `requirements.txt`, `setup.py`, `pytest.ini`

### Essential Documentation
- ✅ `README.md` - Main docs
- ✅ `README_IMPROVEMENTS.md` - Quick start guide
- ✅ `TRAINING_IMPROVEMENTS.md` - Analysis and fixes
- ✅ `WANDB_SETUP.md` - WandB documentation

### Essential Scripts
- ✅ `start_improved_training.sh` - Training launcher
- ✅ `train.log` - Current training output (ACTIVE!)

### Reference Data (Optional Keep)
- ⚠️ `validation_results/all_epochs_metrics.json` - Failed model metrics (for comparison)
- ⚠️ `validation_results/metrics_across_epochs.png` - Visualization of failure

## Estimated Space Savings

- **Duplicate Scripts:** ~50 KB
- **Duplicate Docs:** ~100 KB
- **Old Validation Results:** ~500 MB (images + checkpoints)
- **Old Logs:** ~10-50 MB
- **Archive:** Unknown (check size)
- **WandB Cache:** Unknown (check size)

**Total Estimated:** ~500-1000 MB

## Safe Cleanup Command

Run this to clean up safely without affecting training:

```bash
# See cleanup script: cleanup_workspace.sh
```
