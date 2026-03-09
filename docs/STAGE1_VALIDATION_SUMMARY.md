# Stage 1 Model Validation - Complete Guide

## Quick Start

### Run Validation (Recommended: 50 samples)

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Option 1: Using wrapper script (easiest)
./run_validation.sh

# Option 2: Quick test (10 samples, ~5 minutes)
./run_validation.sh --quick

# Option 3: Full evaluation (200 samples, ~40 minutes)
./run_validation.sh --full

# Option 4: Custom settings
./run_validation.sh --samples 100 --guidance 2.5 --output my_results
```

### Run Directly with Python

```bash
python3 evaluate_stage1_validation.py \
    --checkpoint /root/checkpoints/stage1/final.pt \
    --num_samples 50 \
    --guidance_scale 2.5 \
    --output_dir validation_results
```

---

## What Does This Evaluate?

### Your Question: "How do I find accuracy?"

For **generative models** (like Stage 1), accuracy is measured differently than classification:

**❌ Not Used:**
- True Positive / False Positive (those are for classifiers)
- Binary accuracy (correct/incorrect)

**✅ Instead We Use:**
- **Similarity Metrics**: How close is generated image to ground truth photo?
- **Quality Metrics**: How good is the generation quality?
- **Perceptual Metrics**: Does it look realistic to humans?

---

## Metrics Computed

### 1. PSNR (Peak Signal-to-Noise Ratio)
**Simple pixel-level similarity**
- Range: 0-50+ dB
- Target for Stage 1: **22-30 dB**
- Higher = more pixel-accurate

### 2. SSIM (Structural Similarity)
**Structural preservation**
- Range: -1 to 1
- Target for Stage 1: **0.55-0.75**
- Higher = better structure match
- **Most important for sketch → photo**

### 3. LPIPS (Perceptual Similarity)
**How it looks to humans**
- Range: 0-1
- Target for Stage 1: **0.25-0.45**
- Lower = more perceptually similar
- **Best metric for realism**

### 4. FID (Frechet Inception Distance)
**Overall quality & diversity**
- Range: 0-∞
- Target for Stage 1: **25-50**
- Lower = better

### 5. Inception Score (IS)
**Quality and variety**
- Range: 1-10+
- Target for Stage 1: **3-5**
- Higher = better

### 6. MSE & MAE
**Raw pixel errors**
- Lower = better
- Used mainly for debugging

---

## Understanding Results

### Sample Output

```
================================================================================
📊 VALIDATION METRICS RESULTS
================================================================================

📝 Evaluated 50 samples

┌─────────────────────────────────────────────────────────────────────┐
│ PIXEL-LEVEL METRICS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ PSNR (Peak Signal-to-Noise Ratio)                                   │
│   Mean:  26.342 dB  ← Your score                                    │
│   Target: 22-30 dB  ← Good range for Stage 1                        │
│   ✅ GOOD! Within target range                                      │
│                                                                      │
│ SSIM (Structural Similarity Index)                                   │
│   Mean:  0.6845     ← Your score                                    │
│   Target: 0.55-0.75 ← Good range                                    │
│   ✅ EXCELLENT! Structure well preserved                            │
│                                                                      │
│ LPIPS (Learned Perceptual Similarity)                               │
│   Mean:  0.3214     ← Your score (lower is better)                  │
│   Target: 0.25-0.45 ← Good range                                    │
│   ✅ GOOD! Perceptually similar to real photos                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DISTRIBUTION METRICS                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ FID (Frechet Inception Distance)                                     │
│   Score:  34.567    ← Your score                                    │
│   Target: 25-50     ← Good range                                    │
│   ✅ GOOD! Quality is acceptable                                    │
│                                                                      │
│ IS (Inception Score)                                                 │
│   Mean:  4.123      ← Your score                                    │
│   Target: 3-5       ← Good range                                    │
│   ✅ GOOD! Diverse and high-quality generations                     │
└─────────────────────────────────────────────────────────────────────┘
```

### How to Interpret

| If PSNR is... | It means... |
|---------------|-------------|
| < 20 dB | ❌ Poor quality - model not learning |
| 22-30 dB | ✅ Good for sketch→photo task |
| > 30 dB | 🌟 Excellent (rare for this task) |

| If SSIM is... | It means... |
|---------------|-------------|
| < 0.5 | ❌ Not preserving sketch structure |
| 0.55-0.75 | ✅ Good structure preservation |
| > 0.75 | 🌟 Excellent structure match |

| If LPIPS is... | It means... |
|----------------|-------------|
| > 0.5 | ❌ Doesn't look realistic |
| 0.25-0.45 | ✅ Perceptually good |
| < 0.25 | 🌟 Very realistic looking |

| If FID is... | It means... |
|--------------|-------------|
| > 50 | ❌ Poor overall quality |
| 25-50 | ✅ Acceptable quality |
| < 25 | 🌟 Excellent quality |

---

## Visual Output

The script saves comparison images like:

```
validation_results/
├── comparison_0000.png  ← [Sketch | Generated | Ground Truth]
├── comparison_0010.png
├── comparison_0020.png
├── ...
└── validation_metrics.json  ← All numeric results
```

Each comparison shows:
1. **Input Sketch** (left)
2. **Generated Image** (middle) with PSNR/SSIM scores
3. **Ground Truth Photo** (right)

---

## Comparing Different Checkpoints

```bash
# Test epoch 2
./run_validation.sh --checkpoint /root/checkpoints/stage1/epoch_2.pt \
                    --samples 30 --output val_epoch2

# Test epoch 5
./run_validation.sh --checkpoint /root/checkpoints/stage1/epoch_5.pt \
                    --samples 30 --output val_epoch5

# Test epoch 10 (final)
./run_validation.sh --checkpoint /root/checkpoints/stage1/final.pt \
                    --samples 30 --output val_epoch10

# Compare
echo "Epoch 2 SSIM:" && cat val_epoch2/validation_metrics.json | grep -A2 '"ssim"'
echo "Epoch 5 SSIM:" && cat val_epoch5/validation_metrics.json | grep -A2 '"ssim"'
echo "Epoch 10 SSIM:" && cat val_epoch10/validation_metrics.json | grep -A2 '"ssim"'
```

---

## Expected Performance

### Stage 1 (Sketch-Guided Generation)

Since sketch → photo is a very challenging task:

**Realistic Targets:**
- PSNR: 22-28 dB
- SSIM: 0.55-0.75
- LPIPS: 0.25-0.45
- FID: 25-50
- IS: 3-5

**Why not perfect?**
- Sketch is very abstract (low detail)
- Ground truth photo has specific textures/colors
- Many valid photorealistic interpretations exist
- Model must "hallucinate" realistic details

**These targets mean:**
✅ Structure is preserved (SSIM ~ 0.65)
✅ Looks photorealistic (LPIPS ~ 0.35)
✅ Overall quality is good (FID ~ 35)

---

## FAQ

### Q: Is 60% SSIM bad?

**A:** No! For sketch→photo, 0.60 SSIM is **good**. It means structure is preserved while allowing realistic texture generation.

### Q: My PSNR is 25 dB, is that low?

**A:** No! 25 dB for sketch→photo is **excellent**. This isn't image compression where we expect 40+ dB.

### Q: Can I get 95%+ accuracy?

**A:** Generative models don't have "accuracy" like classifiers. But:
- If you want **category accuracy** (does generated image match the category?), use the other script: `evaluate_stage1_accuracy.py`
- That measures if CLIP can correctly identify the category of generated images

### Q: Which metric should I trust most?

**A:** For sketch→photo:
1. **SSIM** (0.55-0.75) - Most important: structure preservation
2. **LPIPS** (0.25-0.45) - Second: perceptual realism
3. **FID** (25-50) - Third: overall quality

### Q: How long does evaluation take?

**A:** Approximately:
- 10 samples: ~5 minutes
- 50 samples: ~20 minutes
- 100 samples: ~40 minutes
- 200 samples: ~80 minutes

---

## Troubleshooting

### Error: "No module named 'lpips'"

```bash
pip install lpips scikit-image
```

### Error: "CUDA out of memory"

Reduce num_samples:
```bash
./run_validation.sh --samples 20
```

### Error: "Dataset not found"

Check dataset path in config:
```bash
ls /workspace/sketchy/
```

---

## What's Next?

After validation:

1. **If metrics are good** (SSIM > 0.6, FID < 50):
   - ✅ Stage 1 is working well!
   - ✅ Ready to move to Stage 2 training
   - ✅ Can also test with optimized guidance scale (2.5)

2. **If metrics are poor** (SSIM < 0.5, FID > 60):
   - ❌ May need more training epochs
   - ❌ Check if sketch conditioning is working
   - ❌ Review training loss curves

3. **Generate evaluation report:**
   ```bash
   cat validation_results/validation_metrics.json
   ```

---

## Summary

### ✅ What You Get

1. **Quantitative metrics** comparing generated vs ground truth
2. **Visual comparisons** showing side-by-side results
3. **JSON results** for programmatic analysis
4. **Clear targets** to know if your model is good

### 🎯 Running It

```bash
# Quick test (10 samples, 5 mins)
./run_validation.sh --quick

# Standard evaluation (50 samples, 20 mins)
./run_validation.sh

# Full evaluation (200 samples, 80 mins)
./run_validation.sh --full
```

### 📊 Understanding Results

- **SSIM > 0.6**: Structure preserved ✅
- **LPIPS < 0.4**: Looks realistic ✅
- **FID < 50**: Good quality ✅

**If your model hits these targets, Stage 1 is working well!**

---

## Files Created

1. **`evaluate_stage1_validation.py`** - Main evaluation script
2. **`run_validation.sh`** - Convenient wrapper script
3. **`VALIDATION_METRICS_GUIDE.md`** - Detailed metrics explanation
4. **`STAGE1_VALIDATION_SUMMARY.md`** - This file

**Ready to evaluate your model!** 🚀
