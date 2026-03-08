# Stage 1 Validation Metrics Guide

## Overview

This guide explains how to evaluate your Stage 1 model by comparing generated images with ground truth photos from the Sketchy dataset.

## What Are Validation Metrics?

Unlike classification models (which have True Positives/Negatives), **generative models** are evaluated by comparing generated images with real images using various similarity metrics.

## Metrics Explained

### 1. **PSNR (Peak Signal-to-Noise Ratio)**
- **What it measures**: Pixel-level similarity
- **Range**: 0-50+ dB (higher is better)
- **Interpretation**:
  - < 20 dB: Poor quality
  - 20-30 dB: Acceptable
  - 30-40 dB: Good quality
  - > 40 dB: Excellent
- **Use case**: Basic quality check

### 2. **SSIM (Structural Similarity Index)**
- **What it measures**: Structural similarity (luminance, contrast, structure)
- **Range**: -1 to 1 (higher is better)
- **Interpretation**:
  - < 0.5: Poor similarity
  - 0.5-0.7: Moderate similarity
  - 0.7-0.9: Good similarity
  - > 0.9: Excellent similarity
- **Use case**: Better than PSNR for perceptual quality

### 3. **LPIPS (Learned Perceptual Image Patch Similarity)**
- **What it measures**: Perceptual similarity using deep features
- **Range**: 0-1 (lower is better)
- **Interpretation**:
  - < 0.1: Very similar
  - 0.1-0.3: Moderately similar
  - 0.3-0.5: Somewhat different
  - > 0.5: Very different
- **Use case**: Best metric for human perception
- **Note**: Requires `pip install lpips`

### 4. **MSE (Mean Squared Error)**
- **What it measures**: Average squared pixel difference
- **Range**: 0-∞ (lower is better)
- **Interpretation**: Raw pixel error
- **Use case**: Loss function during training

### 5. **MAE (Mean Absolute Error)**
- **What it measures**: Average absolute pixel difference
- **Range**: 0-255 (lower is better)
- **Interpretation**: Average pixel deviation
- **Use case**: Simpler alternative to MSE

### 6. **FID (Frechet Inception Distance)**
- **What it measures**: Distribution similarity between real and generated images
- **Range**: 0-∞ (lower is better)
- **Interpretation**:
  - < 10: Excellent
  - 10-30: Good
  - 30-50: Acceptable
  - > 50: Poor
- **Use case**: Overall quality and diversity assessment
- **Important**: Requires many samples (>50) for stability

### 7. **IS (Inception Score)**
- **What it measures**: Quality and diversity of generated images
- **Range**: 1-10+ (higher is better)
- **Interpretation**:
  - < 2: Poor
  - 2-4: Acceptable
  - 4-6: Good
  - > 6: Excellent
- **Use case**: Measures both sharpness and variety

---

## How to Run Validation

### Basic Usage

```bash
# Evaluate with 100 samples
python evaluate_stage1_validation.py \
    --checkpoint /root/checkpoints/stage1/final.pt \
    --num_samples 100 \
    --guidance_scale 2.5 \
    --output_dir validation_results
```

### Quick Test (10 samples)

```bash
python evaluate_stage1_validation.py \
    --checkpoint /root/checkpoints/stage1/final.pt \
    --num_samples 10 \
    --output_dir validation_quick
```

### Full Evaluation (500 samples)

```bash
python evaluate_stage1_validation.py \
    --checkpoint /root/checkpoints/stage1/final.pt \
    --num_samples 500 \
    --output_dir validation_full
```

---

## Output

### 1. Console Output

```
================================================================================
📊 VALIDATION METRICS RESULTS
================================================================================

📝 Evaluated 100 samples

┌─────────────────────────────────────────────────────────────────────┐
│ PIXEL-LEVEL METRICS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ PSNR (Peak Signal-to-Noise Ratio)                                   │
│   Mean:  24.567 dB  (Higher is better)                              │
│   Std:   3.234 dB                                                    │
│   Range: [18.45, 32.67] dB                                           │
│                                                                      │
│ SSIM (Structural Similarity Index)                                   │
│   Mean:  0.7234     (Range: -1 to 1, higher is better)              │
│   Std:   0.0892                                                      │
│   Range: [0.543, 0.891]                                              │
...
```

### 2. JSON Results (`validation_metrics.json`)

```json
{
  "num_samples": 100,
  "psnr": {
    "mean": 24.567,
    "std": 3.234,
    "min": 18.45,
    "max": 32.67
  },
  "ssim": {
    "mean": 0.7234,
    "std": 0.0892,
    "min": 0.543,
    "max": 0.891
  },
  ...
}
```

### 3. Visual Comparisons

Saved to `validation_results/comparison_XXXX.png`:
- Shows: [Input Sketch] [Generated Image] [Ground Truth Photo]
- Includes PSNR and SSIM scores for that pair

---

## Interpreting Results for Your Model

### Good Model Performance

```
PSNR:  25-35 dB (sketch-to-photo is challenging)
SSIM:  0.6-0.8 (structural preservation)
LPIPS: 0.2-0.4 (perceptually similar)
FID:   20-40 (good distribution match)
IS:    3-5 (reasonable quality/diversity)
```

### What Each Metric Tells You

| Metric | What It Means for Your Model |
|--------|------------------------------|
| **High PSNR** | Generated images are pixel-accurate to ground truth |
| **High SSIM** | Model preserves structure well from sketch to photo |
| **Low LPIPS** | Generated images look perceptually similar to real photos |
| **Low FID** | Generated distribution matches real photo distribution |
| **High IS** | Model generates diverse, high-quality images |

---

## Common Questions

### Q: Why not True Positive/False Positive?

**A:** Those metrics are for **classification** (predicting categories). Your model is **generative** (creating images), so we measure similarity to ground truth instead.

### Q: Can I still measure category accuracy?

**A:** Yes! Use the other script (`evaluate_stage1_accuracy.py`) which uses CLIP to classify generated images and compute accuracy.

### Q: Which metric is most important?

**A:** For your sketch-to-photo model:
1. **SSIM** - Does it preserve sketch structure?
2. **LPIPS** - Does it look realistic?
3. **FID** - Is the overall quality good?

### Q: What's a realistic target for Stage 1?

**A:** Since sketch-to-photo is very challenging:
- PSNR: 22-28 dB
- SSIM: 0.55-0.75
- LPIPS: 0.25-0.45
- FID: 25-50

These are reasonable targets. Perfect reconstruction is not expected!

---

## Comparing Different Checkpoints

```bash
# Test epoch 2
python evaluate_stage1_validation.py \
    --checkpoint /root/checkpoints/stage1/epoch_2.pt \
    --num_samples 50 \
    --output_dir validation_epoch2

# Test epoch 5
python evaluate_stage1_validation.py \
    --checkpoint /root/checkpoints/stage1/epoch_5.pt \
    --num_samples 50 \
    --output_dir validation_epoch5

# Test final
python evaluate_stage1_validation.py \
    --checkpoint /root/checkpoints/stage1/final.pt \
    --num_samples 50 \
    --output_dir validation_final

# Compare results
cat validation_epoch2/validation_metrics.json
cat validation_epoch5/validation_metrics.json
cat validation_final/validation_metrics.json
```

---

## Dependencies

The script uses standard computer vision metrics. Install if needed:

```bash
pip install scikit-image scipy lpips matplotlib
```

---

## Summary

**For generative models like yours:**
- ❌ No True Positive/False Positive (that's for classification)
- ✅ Use similarity metrics (PSNR, SSIM, LPIPS, FID, IS)
- ✅ Compare generated images with ground truth photos
- ✅ Multiple metrics give comprehensive quality assessment

**Run the evaluation to get quantitative proof your model works!**
