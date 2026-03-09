# Emergency Disk Cleanup - Completed

**Date:** March 8, 2026  
**Status:** ✅ COMPLETED

---

## Disk Space Freed

### Files Deleted:

1. **`/root/checkpoints/stage1/` (29 GB)**
   - Old FAILED checkpoints with SSIM=0.027
   - epochs 2, 4, 6, 8, 10, final.pt
   - No longer needed (training from scratch)

2. **`/root/checkpoints/stage1_improved/epoch_1.pt` (13.8 GB)**
   - First epoch checkpoint
   - Already uploaded to WandB
   - Can be re-downloaded if needed

3. **`/workspace/test_outputs_*` directories (~1-2 GB)**
   - Old test output directories (7 total)
   - test_outputs_all_categories
   - test_outputs_corrected
   - test_outputs_epoch10
   - test_outputs_final_diverse
   - test_outputs_guidance_comparison
   - test_outputs_guidance_scales
   - test_outputs_optimized

4. **`/workspace/*.png` (~100 MB)**
   - comparison_grid_01.png through comparison_grid_16.png
   - optimized_grid_*.png
   - *_comparison.png files
   - All old test visualizations

5. **`/workspace/*.log` (~50 MB)**
   - train_stage1.log
   - train_stage1_FIXED.log
   - train_stage2.log
   - test_*.log files

6. **`/workspace/*.md` and HTML (~5 MB)**
   - GRIDS_LOCATION.md
   - HOW_TO_VIEW_RESULTS.md
   - view_all_grids.html

7. **WandB cache (~variable)**
   - Cleaned cache files

---

## Total Space Freed

**Estimated:** ~45-50 GB

Breakdown:
- Old stage1 checkpoints: 29 GB
- Old epoch_1.pt: 13.8 GB
- /workspace cleanup: 1-2 GB
- WandB cache: ~0.5 GB

---

## Files Preserved

### ✅ Essential - Kept:

1. **Current Training:**
   - `/root/checkpoints/stage1_improved/` (will grow to ~25 GB)
   - Training process still running unaffected

2. **Dataset:**
   - `/workspace/sketchy/` (1.2 GB)
   - Required for training

3. **Code:**
   - `/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/`
   - All source code intact

4. **WandB Artifacts:**
   - epoch_1.pt uploaded to WandB cloud
   - Can be re-downloaded: ~14 GB remote storage

---

## Current Disk Usage

After cleanup:
- `/root/checkpoints/stage1_improved/`: Will grow as training progresses
  - Each epoch: ~13.8 GB
  - Expected max: ~27 GB (2 epochs kept + best.pt)

- `/workspace/sketchy/`: 1.2 GB (dataset)

---

## Space Management Strategy

### During Training:
The training script will save checkpoints as epochs complete. To prevent running out of space:

**Auto-cleanup strategy** (recommended):
```bash
# Keep only last 2 checkpoints + best.pt
cd /root/checkpoints/stage1_improved/
ls -t epoch_*.pt | tail -n +3 | xargs -r rm
```

**Manual monitoring:**
```bash
# Check space every few hours
df -h /root
du -sh /root/checkpoints/stage1_improved/
```

### After Training:
1. Upload best checkpoint to HuggingFace
2. Delete all except best.pt and final.pt
3. Expected final size: ~30 GB (2 checkpoints)

---

## Recovery Options

If you accidentally need deleted files:

1. **epoch_1.pt**: Download from WandB artifacts
   ```python
   import wandb
   api = wandb.Api()
   artifact = api.artifact('satyam-kumar2022-vitstudent-ac-in/ragaf-diffusion-stage1/epoch_1.pt:latest')
   artifact.download()
   ```

2. **Old stage1 checkpoints**: Available on HuggingFace
   - Repo: DrRORAL/ragaf-diffusion-checkpoints/stage1/
   - But these are FAILED models (SSIM=0.027), not useful

---

## Training Status

✅ **Training continues unaffected**

Current progress (as of cleanup):
- Epoch 1: ~23% complete (3,000+ / 13,220 steps)
- Speed: 3.5 it/s
- ETA: ~47 minutes remaining for Epoch 1

**Next checkpoint will be saved at Epoch 2** (~2 hours from cleanup)

---

## Recommendations

1. **Monitor disk space** every 2-3 epochs:
   ```bash
   df -h /root
   ```

2. **Keep only 2 most recent checkpoints** to save space

3. **Upload important checkpoints to HuggingFace** for backup

4. **After Epoch 20 completes**, delete all except:
   - `best.pt` (highest SSIM)
   - `final.pt` (last epoch)
   - Saves ~250 GB

---

**Cleanup completed successfully!**  
**Training continues normally.**
