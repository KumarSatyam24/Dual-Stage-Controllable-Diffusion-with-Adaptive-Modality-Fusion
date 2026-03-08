# 📊 Stage-1 Epoch-by-Epoch Validation Guide

## Overview

This validation pipeline evaluates your Stage-1 sketch-guided diffusion model across multiple training epochs. It automatically:

1. ✅ Downloads checkpoints from HuggingFace (or uses local)
2. ✅ Generates images from validation sketches  
3. ✅ Computes quantitative metrics (FID, CLIP, LPIPS, SSIM, PSNR)
4. ✅ Creates visual comparisons
5. ✅ Plots training progress across epochs
6. ✅ Generates HTML report

---

## 🚀 Quick Start

### Basic Usage

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Validate all epochs (auto-detects from HuggingFace or local)
python validate_epochs.py \
    --hf_repo DrRORAL/ragaf-diffusion-checkpoints \
    --dataset_root /workspace/sketchy \
    --num_samples 50

# Quick validation (fewer samples)
python validate_epochs.py --num_samples 20

# Validate specific epochs only
python validate_epochs.py --epochs 1 2 5 10 --num_samples 30
```

---

## 📋 Complete Usage

### Command-Line Arguments

```bash
python validate_epochs.py [OPTIONS]

Options:
  --hf_repo TEXT              HuggingFace repository ID 
                              Default: DrRORAL/ragaf-diffusion-checkpoints
  
  --dataset_root TEXT         Path to Sketchy dataset
                              Default: /workspace/sketchy
  
  --output_dir TEXT           Output directory for results
                              Default: validation_results
  
  --epochs INT [INT ...]      Specific epochs to validate (default: all)
                              Example: --epochs 1 2 5 10
  
  --num_samples INT           Number of validation samples per epoch
                              Default: 50 (20-100 recommended)
  
  --guidance_scale FLOAT      Guidance scale for generation
                              Default: 2.5 (optimal value)
  
  --device TEXT               Device: cuda or cpu
                              Default: cuda
```

### Examples

#### 1. Validate All Checkpoints from HuggingFace

```bash
python validate_epochs.py \
    --hf_repo DrRORAL/ragaf-diffusion-checkpoints \
    --num_samples 50 \
    --output_dir validation_results
```

#### 2. Validate Specific Epochs

```bash
# Only validate epoch 1, 5, and 10
python validate_epochs.py --epochs 1 5 10 --num_samples 30
```

#### 3. Quick Validation (10 samples per epoch)

```bash
python validate_epochs.py --num_samples 10 --output_dir quick_val
```

#### 4. Full Validation (100 samples, high confidence)

```bash
python validate_epochs.py --num_samples 100 --output_dir full_validation
```

#### 5. Custom HuggingFace Repository

```bash
python validate_epochs.py \
    --hf_repo username/your-model-name \
    --dataset_root /path/to/sketchy \
    --num_samples 50
```

---

## 📁 Output Structure

After running validation, you'll get:

```
validation_results/
├── epoch_1/
│   ├── metrics.json                 # Quantitative metrics for epoch 1
│   ├── sample_0000.png             # Comparison: sketch | generated | ground truth
│   ├── sample_0005.png
│   ├── sample_0010.png
│   └── ...
│
├── epoch_2/
│   ├── metrics.json
│   ├── sample_0000.png
│   └── ...
│
├── epoch_5/
│   └── ...
│
├── epoch_10/
│   └── ...
│
├── metrics_across_epochs.png       # 📊 Visual plots of all metrics
├── validation_report.html          # 📝 Comprehensive HTML report
└── all_epochs_metrics.json         # 📄 All metrics in JSON format
```

---

## 📊 Metrics Explained

### 1. PSNR (Peak Signal-to-Noise Ratio)
- **Range:** 0-50+ dB
- **Higher is better**
- **Interpretation:**
  - < 20 dB: Poor quality
  - 20-25 dB: Acceptable for sketch→photo
  - 25-30 dB: Good
  - > 30 dB: Excellent
- **What it measures:** Pixel-level accuracy

### 2. SSIM (Structural Similarity Index)
- **Range:** -1 to 1
- **Higher is better**
- **Interpretation:**
  - < 0.5: Poor structure preservation
  - 0.5-0.6: Acceptable
  - 0.6-0.7: Good ✅
  - > 0.7: Excellent
- **What it measures:** Structure preservation from sketch
- **Most important metric for Stage-1!**

### 3. LPIPS (Learned Perceptual Similarity)
- **Range:** 0-1
- **Lower is better**
- **Interpretation:**
  - < 0.2: Very similar to ground truth
  - 0.2-0.4: Good perceptual similarity ✅
  - 0.4-0.6: Moderate
  - > 0.6: Poor
- **What it measures:** Perceptual realism (how it looks to humans)

### 4. FID (Fréchet Inception Distance)
- **Range:** 0-∞
- **Lower is better**
- **Interpretation:**
  - < 20: Excellent
  - 20-40: Good ✅
  - 40-60: Acceptable
  - > 60: Poor
- **What it measures:** Overall image quality and diversity
- **Note:** Requires 50+ samples for stability

### 5. CLIP Similarity
- **Range:** 0-1
- **Higher is better**
- **Interpretation:**
  - > 0.7: Excellent semantic alignment
  - 0.5-0.7: Good
  - < 0.5: Poor
- **What it measures:** How well generated image matches text prompt

---

## 📈 Understanding the Plots

### Metrics Across Epochs Plot

The script generates a 6-panel figure:

1. **Top-Left:** PSNR over epochs (should increase)
2. **Top-Middle:** SSIM over epochs (should increase)
3. **Top-Right:** LPIPS over epochs (should decrease)
4. **Bottom-Left:** FID over epochs (should decrease)
5. **Bottom-Middle:** CLIP Similarity (should increase)
6. **Bottom-Right:** Summary table of best epoch per metric

**What to look for:**
- ✅ **Upward trend** in PSNR, SSIM, CLIP
- ✅ **Downward trend** in LPIPS, FID
- ✅ **Convergence** around epoch 5-10
- ❌ **Overfitting:** Metrics get worse after peaking

---

## 🎯 Interpreting Results

### Good Stage-1 Training

```json
{
  "epoch": 10,
  "psnr": {"mean": 26.5},     // ✅ Good (22-30 range)
  "ssim": {"mean": 0.68},     // ✅ Excellent (>0.6)
  "lpips": {"mean": 0.32},    // ✅ Good (<0.4)
  "fid": 35.2,                // ✅ Good (<50)
  "clip_similarity": {"mean": 0.72}  // ✅ Excellent (>0.7)
}
```

**Conclusion:** Model is learning structure well, ready for Stage-2!

### Poor Stage-1 Training

```json
{
  "epoch": 10,
  "psnr": {"mean": 18.2},     // ❌ Too low
  "ssim": {"mean": 0.42},     // ❌ Poor structure
  "lpips": {"mean": 0.65},    // ❌ Not realistic
  "fid": 125.3,               // ❌ Poor quality
  "clip_similarity": {"mean": 0.45}  // ❌ Poor alignment
}
```

**Conclusion:** Needs more training or debugging

---

## 🔧 How It Works

### 1. Checkpoint Detection

The script automatically:
- Lists files in HuggingFace repo
- Finds `epoch_1.pt`, `epoch_2.pt`, ..., `final.pt`
- Falls back to local `/root/checkpoints/stage1/` if HF unavailable

### 2. Model Loading

For each checkpoint:
- Downloads from HuggingFace (or uses local)
- Loads model weights
- Creates inference pipeline
- Sets guidance scale (default: 2.5)

### 3. Image Generation

For each validation sample:
- Loads sketch from Sketchy test set
- Generates image using diffusion model
- Seeds are fixed for reproducibility
- Uses configured image size (256x256)

### 4. Metric Computation

Per sample:
- PSNR: Pixel similarity
- SSIM: Structural similarity
- LPIPS: Perceptual distance

Aggregate (all samples):
- FID: Distribution similarity
- CLIP: Text-image alignment

### 5. Visualization

Every 5th sample:
- Saves comparison image: [Sketch | Generated | Ground Truth]
- Includes PSNR/SSIM scores

### 6. Reporting

- JSON metrics per epoch
- Combined plot across epochs
- HTML report with tables and recommendations

---

## ⏱️ Time Estimates

| Samples/Epoch | Time/Epoch | Total (10 epochs) |
|---------------|------------|-------------------|
| 10 | ~5 min | ~50 min |
| 20 | ~10 min | ~100 min |
| 50 | ~25 min | ~250 min (4 hours) |
| 100 | ~50 min | ~500 min (8 hours) |

**Recommendation:** Start with 20 samples for quick validation, then run 50 for final report.

---

## 🎯 Decision Making

### Based on Results

#### Scenario 1: SSIM Increasing Steadily

```
Epoch 1: SSIM = 0.45
Epoch 5: SSIM = 0.62
Epoch 10: SSIM = 0.68  ✅
```

**Action:** ✅ Training is working! Can proceed to Stage-2.

#### Scenario 2: SSIM Plateaued

```
Epoch 1: SSIM = 0.45
Epoch 5: SSIM = 0.58
Epoch 10: SSIM = 0.59  (no improvement)
```

**Action:** ⚠️ Model converged. If SSIM < 0.6, may need:
- More training data
- Different learning rate
- Architecture changes

#### Scenario 3: SSIM Decreasing (Overfitting)

```
Epoch 1: SSIM = 0.45
Epoch 5: SSIM = 0.65
Epoch 10: SSIM = 0.58  ❌ (decreased!)
```

**Action:** ❌ Overfitting! Use checkpoint from epoch 5 instead.

#### Scenario 4: FID Too High

```
Epoch 10: FID = 150  ❌
```

**Action:** ⚠️ Poor image quality. Check:
- Is guidance scale optimal? (try 2.5)
- Enough training epochs?
- Data augmentation needed?

---

## 📝 Sample Outputs

### Metrics JSON (epoch_10/metrics.json)

```json
{
  "epoch": 10,
  "num_samples": 50,
  "psnr": {
    "mean": 26.342,
    "std": 3.124,
    "min": 19.45,
    "max": 32.67
  },
  "ssim": {
    "mean": 0.6845,
    "std": 0.0892,
    "min": 0.543,
    "max": 0.891
  },
  "lpips": {
    "mean": 0.3214,
    "std": 0.0654
  },
  "fid": 34.567,
  "clip_similarity": {
    "mean": 0.7234,
    "std": 0.0543
  }
}
```

### HTML Report Features

- 📊 Interactive table with best epochs highlighted
- 📈 Embedded metrics plot
- 🎯 Automatic recommendations based on results
- 📁 Links to sample images per epoch

---

## 🔍 Troubleshooting

### Error: "No checkpoints found"

**Solution:**
```bash
# Check local checkpoints
ls /root/checkpoints/stage1/

# Or specify HF repo explicitly
python validate_epochs.py --hf_repo YOUR_USERNAME/YOUR_REPO
```

### Error: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size (one image at a time, but fewer samples)
python validate_epochs.py --num_samples 20

# Or use CPU (slower)
python validate_epochs.py --device cpu
```

### Error: "Dataset not found"

**Solution:**
```bash
# Check dataset path
ls /workspace/sketchy/

# Or specify correct path
python validate_epochs.py --dataset_root /path/to/sketchy
```

### Missing Metrics (LPIPS/CLIP)

**Solution:**
```bash
# Install missing libraries
pip install lpips
pip install git+https://github.com/openai/CLIP.git
```

---

## 🎨 Visual Examples

### Good Training Progression

```
Epoch 1: Blurry, lacks structure
Epoch 5: Structure emerging, textures rough
Epoch 10: Clear structure, realistic textures ✅
```

### Overfitting

```
Epoch 1: Blurry but diverse
Epoch 5: Sharp and diverse ✅
Epoch 10: Sharp but repetitive/memorized ❌
```

---

## 📚 Required Libraries

```bash
# Core
pip install torch torchvision
pip install diffusers transformers
pip install huggingface-hub

# Metrics
pip install scikit-image scipy
pip install lpips
pip install git+https://github.com/openai/CLIP.git

# Visualization
pip install matplotlib seaborn
pip install tqdm
pip install opencv-python
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## 🚀 Advanced Usage

### Custom Validation Set

Modify the script to use custom sketches:

```python
# In validate_epochs.py, line ~90
self.val_dataset = SketchyDataset(
    root_dir=dataset_root,
    split='custom',  # Your custom split
    categories=['airplane', 'car'],  # Specific categories
    image_size=self.config['data'].image_size
)
```

### Early Stopping

Monitor SSIM in real-time during training:

```python
# After each epoch during training
validator = EpochValidator(...)
metrics = validator.validate_epoch(epoch=current_epoch, ...)

if metrics['ssim']['mean'] > 0.7:
    print("✅ Target SSIM reached! Can stop training.")
```

### Compare Different Guidance Scales

```bash
# Test different guidance scales
python validate_epochs.py --guidance_scale 1.0 --output_dir val_g1.0
python validate_epochs.py --guidance_scale 2.5 --output_dir val_g2.5
python validate_epochs.py --guidance_scale 5.0 --output_dir val_g5.0

# Compare FID scores
cat val_g*/all_epochs_metrics.json | grep '"fid"'
```

---

## 📊 Integration with Training

### Option 1: Run After Training

```bash
# Train model
python train.py --stage stage1 --epochs 10

# Then validate
python validate_epochs.py --num_samples 50
```

### Option 2: Validate During Training

Add to `train.py`:

```python
# After saving checkpoint
if epoch % 2 == 0:  # Every 2 epochs
    from validate_epochs import EpochValidator
    validator = EpochValidator(...)
    metrics = validator.validate_epoch(epoch, pipeline, num_samples=20)
    
    # Log to wandb/tensorboard
    wandb.log({"val_ssim": metrics['ssim']['mean']})
```

---

## 🎯 Summary

### What This Script Does

✅ **Automatically validates all epochs**  
✅ **Computes 5+ metrics per epoch**  
✅ **Creates visual comparisons**  
✅ **Plots training progress**  
✅ **Generates HTML report**  
✅ **Works with HuggingFace or local checkpoints**  

### What You Get

📊 **Quantitative proof** your model is learning  
📈 **Visual tracking** of training progress  
🎯 **Clear decision points** (continue training? move to Stage-2?)  
📝 **Professional report** for documentation/papers  

### Quick Command

```bash
python validate_epochs.py --num_samples 50
```

**That's it!** The script handles everything else automatically.

---

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Verify dataset path: `ls /workspace/sketchy/`
3. Verify checkpoints: `ls /root/checkpoints/stage1/`
4. Check GPU memory: `nvidia-smi`
5. Review error messages in console output

---

**Happy Validating! 🎉**

Your Stage-1 model evaluation is now fully automated and comprehensive!
