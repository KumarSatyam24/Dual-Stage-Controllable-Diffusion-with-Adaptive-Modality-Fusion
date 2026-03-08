# 🎯 Stage 1 Validation - Quick Reference Card

## 📋 TL;DR

**To validate your Stage 1 model:**

```bash
./run_validation.sh --quick  # 10 samples, 5 mins
```

**This compares generated images with ground truth photos and gives you:**
- ✅ PSNR (pixel similarity)
- ✅ SSIM (structure preservation) 
- ✅ LPIPS (perceptual quality)
- ✅ FID (overall quality)
- ✅ Visual comparisons

---

## 🎨 What Gets Evaluated

```
Input:      [Sketch of airplane] ────────┐
                                          │
Stage 1 Model generates:                  │
                                          ▼
Output:     [Photo of airplane] ──── Compare ──── [Real airplane photo]
                                          │        (ground truth)
                                          ▼
                                      Metrics:
                                      - PSNR: 26.3 dB ✅
                                      - SSIM: 0.68 ✅
                                      - LPIPS: 0.32 ✅
                                      - FID: 34.5 ✅
```

---

## 📊 Target Metrics (Stage 1)

| Metric | Range | Target | Meaning |
|--------|-------|--------|---------|
| **PSNR** | 0-50+ dB | **22-30 dB** | Pixel accuracy |
| **SSIM** | -1 to 1 | **0.55-0.75** | Structure preserved ⭐ |
| **LPIPS** | 0-1 | **0.25-0.45** | Looks realistic ⭐ |
| **FID** | 0-∞ | **25-50** | Overall quality ⭐ |
| **IS** | 1-10+ | **3-5** | Diversity |

⭐ = Most important for sketch→photo

---

## ✅ Good Results Look Like

```
================================================================================
📊 VALIDATION METRICS RESULTS
================================================================================

PSNR:  26.3 dB   ✅ GOOD (target: 22-30)
SSIM:  0.68      ✅ EXCELLENT (target: 0.55-0.75)
LPIPS: 0.32      ✅ GOOD (target: 0.25-0.45)
FID:   34.5      ✅ GOOD (target: 25-50)
IS:    4.2       ✅ GOOD (target: 3-5)

🎉 Stage 1 model is performing well!
```

---

## ❌ Poor Results Look Like

```
================================================================================
📊 VALIDATION METRICS RESULTS
================================================================================

PSNR:  18.2 dB   ❌ POOR (target: 22-30)
SSIM:  0.42      ❌ POOR (target: 0.55-0.75)
LPIPS: 0.58      ❌ POOR (target: 0.25-0.45)
FID:   68.3      ❌ POOR (target: 25-50)
IS:    2.1       ❌ POOR (target: 3-5)

⚠️  Model needs more training or debugging
```

---

## 🚀 Usage Examples

### Quick Test (Recommended First)

```bash
./run_validation.sh --quick
# 10 samples, ~5 minutes
# Good for quick check
```

### Standard Evaluation

```bash
./run_validation.sh
# 50 samples (default), ~20 minutes
# Good for reliable metrics
```

### Full Evaluation

```bash
./run_validation.sh --full
# 200 samples, ~80 minutes
# Best for final report
```

### Custom

```bash
./run_validation.sh --samples 100 --guidance 2.5 --output my_results
```

### Compare Epochs

```bash
# Compare different checkpoints
./run_validation.sh --checkpoint /root/checkpoints/stage1/epoch_2.pt --output val_epoch2
./run_validation.sh --checkpoint /root/checkpoints/stage1/epoch_5.pt --output val_epoch5
./run_validation.sh --checkpoint /root/checkpoints/stage1/final.pt --output val_final
```

---

## 📁 Output Files

```
validation_results/
├── validation_metrics.json     ← All numeric results
├── comparison_0000.png         ← [Sketch | Generated | GT]
├── comparison_0010.png         ← Visual comparisons
├── comparison_0020.png         ← (every 10 samples)
└── ...
```

---

## 🤔 Common Questions

### "What about True Positive / False Positive?"

❌ **Not applicable** - those are for classification (cat vs dog)

✅ **Use instead** - similarity metrics (how close to ground truth?)

### "Is 0.65 SSIM good?"

✅ **YES!** For sketch→photo, 0.65 SSIM is excellent structure preservation

### "My PSNR is only 24 dB"

✅ **That's good!** Sketch→photo isn't meant to have 40+ dB like compression

### "How do I know if my model is good?"

**Check 3 key metrics:**
1. SSIM > 0.6 ✅ (preserves structure)
2. LPIPS < 0.4 ✅ (looks realistic)
3. FID < 50 ✅ (good quality)

**If all 3 pass → your model is working well!**

---

## 📈 What Each Metric Tells You

```
┌─────────────────────────────────────────┐
│ SSIM (Structural Similarity)            │
├─────────────────────────────────────────┤
│ Does generated image follow sketch      │
│ structure?                               │
│                                          │
│ 0.68 ✅ = YES, structure preserved      │
│ 0.42 ❌ = NO, structure lost            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ LPIPS (Perceptual Similarity)           │
├─────────────────────────────────────────┤
│ Does it look like a real photo?         │
│                                          │
│ 0.32 ✅ = YES, looks realistic          │
│ 0.58 ❌ = NO, looks fake                │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ FID (Frechet Inception Distance)        │
├─────────────────────────────────────────┤
│ Is overall quality good?                │
│                                          │
│ 34.5 ✅ = YES, good quality             │
│ 68.3 ❌ = NO, poor quality              │
└─────────────────────────────────────────┘
```

---

## 🎯 Decision Tree

```
Run: ./run_validation.sh
         │
         ▼
    Check Results
         │
    ┌────┴────┐
    │         │
SSIM > 0.6?   │
    │         │
   YES       NO
    │         │
    ▼         ▼
LPIPS < 0.4?  Need more
    │         training
   YES
    │
    ▼
  FID < 50?
    │
   YES
    │
    ▼
✅ MODEL IS GOOD!
Ready for Stage 2
```

---

## 📝 Example Report

```markdown
# Stage 1 Model Validation Report

**Date:** March 7, 2026
**Checkpoint:** final.pt (10 epochs)
**Samples:** 50

## Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| PSNR   | 26.3 dB | 22-30 dB | ✅ Pass |
| SSIM   | 0.68 | 0.55-0.75 | ✅ Pass |
| LPIPS  | 0.32 | 0.25-0.45 | ✅ Pass |
| FID    | 34.5 | 25-50 | ✅ Pass |
| IS     | 4.2 | 3-5 | ✅ Pass |

## Conclusion

All metrics within target range. Model successfully:
- Preserves sketch structure (SSIM: 0.68)
- Generates realistic images (LPIPS: 0.32)
- Maintains good quality (FID: 34.5)

**Status:** ✅ Ready for Stage 2 training
```

---

## 🔧 Troubleshooting

### "Command not found"

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion
chmod +x run_validation.sh
```

### "Module not found"

```bash
pip install lpips scikit-image scipy matplotlib
```

### "CUDA out of memory"

```bash
./run_validation.sh --samples 10  # Reduce samples
```

---

## ⏱️ Time Estimates

| Samples | Time | Use Case |
|---------|------|----------|
| 10 | ~5 min | Quick test |
| 50 | ~20 min | Standard eval |
| 100 | ~40 min | Thorough eval |
| 200 | ~80 min | Full report |

---

## 🎉 Bottom Line

**To validate Stage 1:**

1. Run: `./run_validation.sh --quick`
2. Check: SSIM > 0.6, LPIPS < 0.4, FID < 50
3. If all pass → ✅ Model is good!
4. View: `validation_results/comparison_*.png`

**That's it!** No True/False Positives needed. These metrics tell you if your generative model is working.

---

## 📚 More Info

- **Detailed guide:** `VALIDATION_METRICS_GUIDE.md`
- **Complete summary:** `STAGE1_VALIDATION_SUMMARY.md`
- **Quick reference:** This file!

---

**Ready to validate? Just run:**

```bash
./run_validation.sh --quick
```

**🚀 Let's see how good your Stage 1 model is!**
