# Training Restart Plan - Stage 1 with SSIM Loss

**Date:** March 9, 2026  
**Status:** Ready to restart training from Epoch 12 with improvements

## 🔍 What We Learned from 1000-Sample Validation

### Key Findings:
1. **Epochs 8 and 12 are identical** (SSIM 0.161 ± 0.10)
2. **10-sample validation is dangerously unreliable** (±15-20% error)
3. **Model plateaued after Epoch 4-6** - no real improvement since then
4. **Root cause:** Training optimizes MSE+LPIPS, but we measure SSIM (not in loss!)

### Performance Summary:
```
Initial failed model: SSIM 0.027
After 12 epochs:      SSIM 0.161 (+492% improvement)
Target goal:          SSIM 0.60
Progress:             27% of goal achieved
```

## 🛠️ Improvements Made

### 1. **Added SSIM Loss to Training Objective**
```python
# OLD: loss = MSE + 0.1 * LPIPS
# NEW: loss = MSE + 0.1 * LPIPS + 0.3 * (1 - SSIM)
```
- **Why:** Model now optimizes what we actually measure!
- **Expected:** Should break through the 0.16 plateau

### 2. **Increased Batch Size: 4 → 8**
```python
batch_size = 8  # Was 4
```
- **Benefits:**
  - 2x more stable gradients
  - ~25% faster training (fewer iterations per epoch)
  - Better learning dynamics with SSIM loss
  - Still fits in 32GB VRAM (~30GB estimated usage)
- **Fallback:** Can reduce to batch_size=6 if OOM

### 3. **Larger Validation Sample Size: 10 → 100**
```python
validation_samples = 100  # Was 10
```
- **Reliability:** ±7-10% uncertainty (vs ±15-20% with 10 samples)
- **Cost:** ~5-7 minutes per validation (vs 1 minute)
- **Worth it:** Much more trustworthy metrics

### 4. **Fixed Random Seed for Validation**
```python
torch.manual_seed(42)  # Reproducible validation samples
```
- **Why:** Same 100 samples every epoch = fair comparison
- **Benefit:** Can track true progress without sampling noise

### 5. **Adjusted Learning Rate: 4e-6 → 5e-6**
```python
learning_rate = 5e-6  # Was stuck at 4e-6
```
- **Why:** Model got stuck in local minimum at 4e-6
- **Benefit:** More exploration capacity to find better solutions

### 6. **Extended Training: 20 → 25 Epochs**
```python
num_epochs = 25  # Was 20
```
- **Why:** Starting from Epoch 12, gives 13 more epochs
- **Goal:** Reach SSIM 0.30-0.40 (2x improvement)

## 📊 Expected Performance Trajectory

### Conservative Estimate:
```
Epoch 12 (baseline): SSIM 0.161
Epoch 15:            SSIM 0.20-0.22 (+25%)
Epoch 18:            SSIM 0.24-0.28 (+50%)
Epoch 21:            SSIM 0.28-0.32 (+75%)
Epoch 25:            SSIM 0.32-0.38 (+100%)
```

### Optimistic (if SSIM loss works well):
```
Epoch 15: SSIM 0.23-0.26
Epoch 18: SSIM 0.28-0.32
Epoch 21: SSIM 0.34-0.38
Epoch 25: SSIM 0.40-0.45 (publishable!)
```

## 🚀 Start Training Command

```bash
cd ~/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

python train_stage1_with_ssim.py \
    --resume_from /root/checkpoints/stage1_improved/epoch_12.pt \
    --batch_size 8 \
    --ssim_weight 0.3 \
    --validation_samples 100 \
    --epochs 25 \
    --lr 5e-6
```

**If you get OOM (out of memory), try batch_size=6:**
```bash
python train_stage1_with_ssim.py \
    --resume_from /root/checkpoints/stage1_improved/epoch_12.pt \
    --batch_size 6 \
    --ssim_weight 0.3 \
    --validation_samples 100 \
    --epochs 25 \
    --lr 5e-6
```

## ⏱️ Training Timeline

**Per Epoch Estimate:**
- With batch_size=8: ~45 minutes
- Validation (100 samples): ~6 minutes
- Total per epoch: ~51 minutes

**Full Training (Epoch 12→25):**
- 13 epochs × 51 min = **~11 hours**
- Validation epochs (2,4,6...): 7 validations
- **Completion:** ~11-12 hours from start

**Expected completion:** March 9, evening (~6-7 PM)

## 📊 Monitoring Progress

### What to Watch:

1. **Training Loss Components:**
   ```
   MSE loss:        Should decrease steadily
   LPIPS loss:      Should decrease
   SSIM loss:       Should decrease (NEW - this is key!)
   Total loss:      Combination of above
   ```

2. **Validation Metrics (Every 2 Epochs):**
   ```
   SSIM:   Primary metric - should steadily increase
   PSNR:   Secondary - monitor for correlation
   LPIPS:  Should decrease
   ```

3. **Key Milestones:**
   - **Epoch 14:** SSIM should be > 0.18 (improvement from 0.161)
   - **Epoch 16:** SSIM should be > 0.22 (clear progress)
   - **Epoch 20:** SSIM should be > 0.28 (doubling baseline)
   - **Epoch 25:** SSIM target: 0.35+ (2x improvement)

### Red Flags to Watch:

⚠️ **Stop training if:**
- SSIM doesn't improve by Epoch 16 (still ~0.16)
- Training loss increases (divergence)
- OOM errors (reduce batch_size to 6)
- SSIM starts decreasing after initial improvement

## 💾 Checkpoint Management

**Storage:** Currently 88GB free (good buffer)

**Strategy:**
- Keep only: `latest_epoch.pt` + `best.pt`
- Delete old epoch checkpoints after new one saves
- Each checkpoint: ~13GB

**Manual cleanup after each validation:**
```bash
# When epoch_14 saves, delete epoch_12
rm /root/checkpoints/stage1_with_ssim/epoch_12.pt

# When epoch_16 saves, delete epoch_14
rm /root/checkpoints/stage1_with_ssim/epoch_14.pt
# etc.
```

## 📈 Success Criteria

### Minimum Success:
- ✅ SSIM reaches 0.25+ by Epoch 25 (+55% from baseline)
- ✅ Training stable (no divergence)
- ✅ Validation metrics correlate with training loss

### Good Success:
- ✅ SSIM reaches 0.30-0.35 by Epoch 25 (+87-117%)
- ✅ Consistent improvement every 2 epochs
- ✅ Model learns to optimize structural similarity

### Excellent Success:
- ✅ SSIM reaches 0.38-0.42 by Epoch 25 (+136-161%)
- ✅ Approaches half of target goal (0.60)
- ✅ Publishable quality results

## 🎯 Next Steps After Training

1. **Final Comprehensive Validation** (1000 samples):
   ```bash
   python validate_comprehensive.py \
       --checkpoint /root/checkpoints/stage1_with_ssim/best.pt \
       --num_samples 1000 \
       --device cuda
   ```

2. **Compare Results:**
   - Baseline (Epoch 12): SSIM 0.161
   - Best (Epoch ?): SSIM 0.X
   - Improvement: +X%

3. **Visual Inspection:**
   - Check generated images in `validation_examples/`
   - Assess quality subjectively
   - Identify remaining failure modes

4. **Decision:**
   - If SSIM ≥ 0.35: Proceed to Stage 2 (refinement) ✅
   - If SSIM 0.25-0.35: Acceptable, consider more training
   - If SSIM < 0.25: Need architectural changes

## 📝 Notes

- **WandB Tracking:** Enabled automatically
- **HuggingFace Backup:** Checkpoints uploaded to DrRORAL/ragaf-diffusion-checkpoints
- **Logs:** Check `train_with_ssim.log` for progress
- **GPU:** RTX 5090 (32GB) - should handle batch_size=8

---

**Ready to start!** Run the command above to begin improved training. 🚀
