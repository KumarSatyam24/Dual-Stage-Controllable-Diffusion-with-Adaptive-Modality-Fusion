# 🔧 Stage-1 Model Improvement Plan

## 🚨 ROOT CAUSE IDENTIFIED

After analyzing the checkpoint and training code, I've identified the critical issues:

### **Issue #1: Learning Rate Too High** ⚠️⚠️⚠️
- **Current:** 1e-4 (0.0001)
- **Problem:** This is 10x too high for fine-tuning diffusion models
- **Solution:** Reduce to 1e-5 or 5e-6

### **Issue #2: Only Sketch Encoder Training**
- **Current:** Only 124 sketch-related parameters being trained
- **Problem:** Base UNet is frozen, limiting adaptation
- **Solution:** Unfreeze UNet or use LoRA for efficient fine-tuning

### **Issue #3: No Perceptual Loss**
- **Current:** Only MSE loss (pixel-level)
- **Problem:** MSE doesn't capture structural/perceptual quality
- **Solution:** Add LPIPS perceptual loss

### **Issue #4: Short Training**
- **Current:** 10 epochs
- **Problem:** May not be enough for convergence
- **Solution:** Train for 20-30 epochs with early stopping

---

## 🎯 IMPROVEMENT STRATEGY

### Phase 1: Quick Fixes (2-3 hours)
1. ✅ Lower learning rate to 1e-5
2. ✅ Add perceptual loss (LPIPS)
3. ✅ Unfreeze more UNet layers
4. ✅ Increase training to 20 epochs

### Phase 2: Enhanced Training (8-10 hours)
1. Train with improved configuration
2. Monitor metrics every 2 epochs
3. Early stop if no improvement

### Phase 3: Validation (30 min)
1. Run epoch-by-epoch validation
2. Confirm SSIM > 0.5, FID < 100

---

## 📋 IMPLEMENTATION CHECKLIST

- [ ] Create improved training configuration
- [ ] Create enhanced training script
- [ ] Test on 1 epoch first
- [ ] Run full 20-epoch training
- [ ] Validate results
- [ ] Compare before/after metrics

---

## 🎓 EXPECTED IMPROVEMENTS

### Target Metrics After Retraining:

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| SSIM | 0.25 | 0.55-0.65 | +120-160% |
| LPIPS | 0.75 | 0.35-0.45 | -40-53% |
| FID | 280 | 50-80 | -71-82% |
| PSNR | 8.9 dB | 22-26 dB | +147-192% |

---

## 🚀 NEXT STEPS

1. Create `train_improved.py` with all fixes
2. Create `config_improved.py` with better hyperparameters
3. Run quick 1-epoch test
4. If successful, run full 20-epoch training
5. Validate and compare results

---

**Status:** Ready to implement improvements
**Estimated Time:** 10-12 hours total
**Success Probability:** High (80%+)
