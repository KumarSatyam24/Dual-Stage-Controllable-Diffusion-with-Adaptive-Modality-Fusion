# Stage 1 Validation Results Analysis

**Date:** March 7, 2026  
**Checkpoint:** final.pt (10 epochs)  
**Samples Evaluated:** 10  
**Image Size:** 256x256

---

## 📊 Results Summary

### Pixel-Level Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **PSNR** | 8.84 dB | 22-30 dB | ❌ **POOR** |
| **SSIM** | 0.264 | 0.55-0.75 | ❌ **POOR** |
| **MSE** | 9756 | Lower is better | ❌ High error |
| **MAE** | 84.5 | Lower is better | ❌ High error |

### Perceptual Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **LPIPS** | 0.760 | 0.25-0.45 | ❌ **POOR** |

### Distribution Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **FID** | 384.9 | 25-50 | ❌ **VERY POOR** |
| **IS** | 1.0 | 3-5 | ❌ **VERY POOR** |

---

## 🔍 Analysis

### What These Results Mean

**❌ The model is NOT performing well**

1. **PSNR: 8.84 dB** (Target: 22-30 dB)
   - Extremely low - indicates generated images are very different from ground truth
   - Should be at least 20 dB for acceptable quality

2. **SSIM: 0.264** (Target: 0.55-0.75)
   - Very poor structural similarity
   - Model is NOT preserving sketch structure well
   - Should be above 0.5 for acceptable structure preservation

3. **LPIPS: 0.760** (Target: 0.25-0.45)
   - Very high perceptual distance
   - Generated images look very different from real photos
   - Lower is better, this is quite high

4. **FID: 384.9** (Target: 25-50)
   - Extremely high - indicates very poor quality and diversity
   - Should be below 50 for good quality
   - This suggests the model's distribution is far from real images

5. **Inception Score: 1.0** (Target: 3-5)
   - Minimum possible score (1.0)
   - Indicates lack of diversity or quality in generations
   - Suggests all images look very similar or unclear

---

## 🤔 Possible Causes

### 1. **Image Size Mismatch During Training?**
- Model was trained on one size but evaluated on another
- Check if training used 512x512 but current config is 256x256

### 2. **Model Not Fully Trained**
- 10 epochs might not be enough
- Loss might not have converged

### 3. **Guidance Scale Issue**
- Currently using guidance_scale=2.5
- Might be too low for this model

### 4. **Checkpoint Loading Issue**
- Weights might not have loaded correctly
- Check if checkpoint corresponds to current architecture

### 5. **Dataset Issue**
- Test set might be different from training set
- Distribution shift between train and test

---

## 🔧 Recommended Actions

### Immediate Checks

1. **Verify Training Was Successful**
   ```bash
   # Check training logs
   cat logs/stage1_training.log | grep "loss"
   
   # Look for convergence
   # Loss should decrease over epochs
   ```

2. **Test with Different Guidance Scales**
   ```bash
   # Try higher guidance
   ./run_validation.sh --quick --guidance 5.0 --output val_g5.0
   ./run_validation.sh --quick --guidance 7.5 --output val_g7.5
   
   # Compare results
   cat val_g5.0/validation_metrics.json | grep -A2 '"ssim"'
   cat val_g7.5/validation_metrics.json | grep -A2 '"ssim"'
   ```

3. **Visually Inspect Outputs**
   ```bash
   # Look at the generated images
   ls -lh validation_results/comparison_*.png
   
   # Open one to see what's wrong
   # Check if images are blurry, wrong structure, or just noise
   ```

4. **Test Earlier Checkpoints**
   ```bash
   # Try epoch 2 (if available)
   ./run_validation.sh --checkpoint /root/checkpoints/stage1/epoch_2.pt --samples 5
   
   # Try epoch 5
   ./run_validation.sh --checkpoint /root/checkpoints/stage1/epoch_5.pt --samples 5
   ```

5. **Check Training Configuration**
   ```bash
   # Verify what image size was used during training
   cat src/configs/config.py | grep "image_size"
   
   # Check if it matches current setting (256)
   ```

---

## 📈 What Good Results Should Look Like

For comparison, good Stage 1 results would be:

```json
{
  "psnr": {"mean": 25.0},        // vs your 8.84 ❌
  "ssim": {"mean": 0.65},        // vs your 0.26 ❌
  "lpips": {"mean": 0.35},       // vs your 0.76 ❌
  "fid": 35.0,                   // vs your 384.9 ❌
  "inception_score": {"mean": 4.0} // vs your 1.0 ❌
}
```

Your model is performing **significantly worse** than expected across all metrics.

---

## 🎯 Next Steps

### Priority 1: Visual Inspection
Look at the generated images to understand what's wrong:
```bash
cd validation_results
# Open comparison images
```

### Priority 2: Check Training
```bash
# Review training logs
tail -100 logs/stage1_training.log

# Check if training completed successfully
# Look for final loss values
```

### Priority 3: Test Different Settings
```bash
# Test with higher guidance scale
./run_validation.sh --quick --guidance 7.5

# Test earlier checkpoint
./run_validation.sh --checkpoint /root/checkpoints/stage1/epoch_5.pt --samples 5
```

### Priority 4: Retrain if Necessary
If the model is fundamentally broken:
```bash
# Consider retraining with correct settings
python3 scripts/training/train.py --stage stage1 --epochs 20
```

---

## 💡 Understanding the Metrics

### Why All Metrics Are Poor

When **all** metrics are poor (not just one or two), it usually means:

1. **Model didn't train properly**
   - Loss didn't converge
   - Learning rate too high/low
   - Architecture mismatch

2. **Wrong checkpoint loaded**
   - Checkpoint from different experiment
   - Incompatible architecture

3. **Severe distribution shift**
   - Test set very different from training
   - Data preprocessing mismatch

4. **Fundamental issue**
   - Model architecture problem
   - Configuration error

---

## 📊 Comparison with Expected Performance

| Metric | Your Score | Expected | Ratio |
|--------|-----------|----------|-------|
| PSNR | 8.84 dB | 25 dB | **35% of target** |
| SSIM | 0.26 | 0.65 | **40% of target** |
| LPIPS | 0.76 | 0.35 | **217% worse** |
| FID | 384.9 | 35 | **1100% worse** |
| IS | 1.0 | 4.0 | **25% of target** |

The model is performing at approximately **25-40% of expected quality**.

---

## ✅ Action Items

- [ ] Visual inspection of generated images
- [ ] Check training logs for convergence
- [ ] Test with different guidance scales (5.0, 7.5)
- [ ] Test earlier checkpoints (epoch 2, 5)
- [ ] Verify image size consistency (training vs eval)
- [ ] Review training loss curve
- [ ] Check if sketch conditioning is working
- [ ] Consider retraining if fundamentally broken

---

## 🚨 Bottom Line

**The Stage 1 model is NOT working properly.**

These metrics indicate a serious problem. Before proceeding to Stage 2, you need to:

1. **Diagnose the issue** - Look at generated images
2. **Check training** - Did it complete successfully?
3. **Test fixes** - Try different settings
4. **Retrain if needed** - Model may need retraining

**Do not proceed to Stage 2 until Stage 1 is working!**
