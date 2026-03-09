# ✅ Epoch Validation Pipeline - Ready to Use!

## 🎉 Setup Complete

The epoch-by-epoch validation pipeline has been configured for your HuggingFace repository!

**HuggingFace Repository:** `DrRORAL/ragaf-diffusion-checkpoints`  
**Checkpoints Location:** `stage1/` subfolder  
**Available Epochs:** 2, 4, 6, 8, 10, final

---

## 🚀 Quick Start

### 1. Test with Single Epoch (5 minutes)

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

python validate_epochs.py --epochs 2 --num_samples 10
```

This will:
- Download `epoch_2.pt` from HuggingFace (if not already local)
- Generate 10 validation images
- Compute PSNR, SSIM, LPIPS, FID, CLIP metrics
- Save results to `validation_results/epoch_2/`

### 2. Validate All Epochs (2-3 hours)

```bash
python validate_epochs.py --epochs 2 4 6 8 10 --num_samples 50
```

This will:
- Validate epochs 2, 4, 6, 8, and 10
- 50 samples per epoch (reliable metrics)
- Create comparison plots across epochs
- Generate HTML report

### 3. Quick Test (just to verify it works)

```bash
python validate_epochs.py --epochs 2 --num_samples 5
```

---

## 📊 What You'll Get

After running validation, check:

```bash
# View the metrics plot
xdg-open validation_results/metrics_across_epochs.png

# Open HTML report
firefox validation_results/validation_report.html

# Check JSON metrics
cat validation_results/all_epochs_metrics.json | jq
```

**Output Structure:**
```
validation_results/
├── epoch_2/
│   ├── metrics.json
│   ├── sample_0000.png  # [Sketch | Generated | Ground Truth]
│   ├── sample_0005.png
│   └── ...
├── epoch_4/
│   └── ...
├── epoch_10/
│   └── ...
├── metrics_across_epochs.png  # 📊 Training curves
├── validation_report.html      # 📝 Full report
└── all_epochs_metrics.json     # 📄 All data
```

---

## 🎯 Interpreting Results

### What to Look For

The plot will show 5 metrics across epochs:

1. **SSIM (Structural Similarity)** 📈 Should increase
   - Target: > 0.60 for good structure
   - This is the MOST important metric for Stage-1!

2. **LPIPS (Perceptual Distance)** 📉 Should decrease
   - Target: < 0.40 for realistic images
   - Measures how "human-like" images look

3. **FID (Fréchet Inception Distance)** 📉 Should decrease
   - Target: < 50 for good quality
   - Overall image quality metric

4. **PSNR (Peak Signal-to-Noise)** 📈 Should increase
   - Target: 22-30 dB acceptable
   - Pixel-level accuracy

5. **CLIP Similarity** 📈 Should increase
   - Target: > 0.70 for semantic alignment
   - Text-image correspondence

### Example Good Training

```
Epoch 2:  SSIM=0.45, LPIPS=0.68, FID=150
Epoch 4:  SSIM=0.52, LPIPS=0.55, FID=95
Epoch 6:  SSIM=0.61, LPIPS=0.42, FID=62
Epoch 8:  SSIM=0.68, LPIPS=0.35, FID=42  ✅ Good!
Epoch 10: SSIM=0.70, LPIPS=0.32, FID=38  ✅ Excellent!
```

**Conclusion:** Model is learning well! Ready for Stage-2.

### Example Overfitting

```
Epoch 2:  SSIM=0.45, LPIPS=0.68, FID=150
Epoch 4:  SSIM=0.55, LPIPS=0.52, FID=85
Epoch 6:  SSIM=0.65, LPIPS=0.38, FID=48  ✅ Best epoch
Epoch 8:  SSIM=0.62, LPIPS=0.42, FID=55  ⚠️ Getting worse
Epoch 10: SSIM=0.58, LPIPS=0.48, FID=68  ❌ Overfitting!
```

**Conclusion:** Use checkpoint from epoch 6 instead!

---

## 🔧 Command Options

```bash
python validate_epochs.py [OPTIONS]

Required: None (uses defaults)

Options:
  --hf_repo TEXT              HuggingFace repository
                              Default: DrRORAL/ragaf-diffusion-checkpoints
  
  --dataset_root TEXT         Path to Sketchy dataset
                              Default: /workspace/sketchy
  
  --output_dir TEXT           Output directory
                              Default: validation_results
  
  --epochs INT [INT ...]      Specific epochs to validate
                              Default: all available (2, 4, 6, 8, 10, final)
                              Example: --epochs 2 6 10
  
  --num_samples INT           Samples per epoch
                              Default: 50
                              Recommended: 20-100
  
  --guidance_scale FLOAT      Guidance scale for generation
                              Default: 2.5 (optimal)
  
  --device TEXT               Device: cuda or cpu
                              Default: cuda
```

---

## ⏱️ Time Estimates

| Configuration | Time per Epoch | Total (5 epochs) |
|--------------|----------------|------------------|
| Quick test (5 samples) | ~3 min | ~15 min |
| Standard (20 samples) | ~10 min | ~50 min |
| Reliable (50 samples) | ~25 min | ~2 hours |
| Publication (100 samples) | ~50 min | ~4 hours |

**GPU:** RTX 5090  
**Generation:** ~30 seconds per image (50 diffusion steps)

---

## 📋 Checklist

Before running full validation:

- [x] HuggingFace repo configured: `DrRORAL/ragaf-diffusion-checkpoints`
- [x] Checkpoints verified: epochs 2, 4, 6, 8, 10, final available
- [x] Dataset accessible: `/workspace/sketchy/`
- [x] Dependencies installed: seaborn, lpips, clip
- [ ] GPU available: `nvidia-smi`
- [ ] Sufficient disk space: ~2GB per epoch for outputs

---

## 🎬 Recommended Workflow

### Step 1: Quick Sanity Check (5 min)

```bash
python validate_epochs.py --epochs 2 --num_samples 5
```

**Goal:** Verify everything works (downloads, generation, metrics)

### Step 2: Compare Early vs Late Epochs (20 min)

```bash
python validate_epochs.py --epochs 2 10 --num_samples 20
```

**Goal:** See if training improved from epoch 2 to 10

### Step 3: Full Validation (2-3 hours)

```bash
python validate_epochs.py --epochs 2 4 6 8 10 --num_samples 50
```

**Goal:** Get reliable metrics across all epochs

### Step 4: Analyze Results

```bash
# View plots
xdg-open validation_results/metrics_across_epochs.png

# Check best epoch
cat validation_results/all_epochs_metrics.json | jq '.[] | {epoch, ssim: .ssim.mean, fid}'

# View sample images
ls validation_results/epoch_*/sample_*.png
```

### Step 5: Make Decision

Based on results:
- ✅ **SSIM > 0.6 on epoch 10?** → Proceed to Stage-2!
- ⚠️ **SSIM peaked earlier?** → Use that checkpoint instead
- ❌ **SSIM < 0.5 everywhere?** → Need to debug training

---

## 🔍 Troubleshooting

### "No checkpoints found"

Check if files exist:
```bash
python3 -c "from huggingface_hub import list_repo_files; files = list_repo_files('DrRORAL/ragaf-diffusion-checkpoints'); print([f for f in files if 'stage1' in f])"
```

### "CUDA out of memory"

Reduce samples:
```bash
python validate_epochs.py --epochs 2 --num_samples 10
```

### "Dataset not found"

Check dataset path:
```bash
ls /workspace/sketchy/photo/
ls /workspace/sketchy/sketch/
```

If not there, specify correct path:
```bash
python validate_epochs.py --dataset_root /path/to/sketchy
```

---

## 📚 Documentation

Full guides available:
- **Complete Guide:** `docs/EPOCH_VALIDATION_GUIDE.md`
- **Quick Commands:** `VALIDATION_COMMANDS.md`
- **Metrics Explanation:** `VALIDATION_METRICS_GUIDE.md`

---

## 🎯 Next Steps

1. **Run quick test:** `python validate_epochs.py --epochs 2 --num_samples 5`
2. **Wait 5 minutes** for it to complete
3. **Check results:** `ls validation_results/epoch_2/`
4. **If successful,** run full validation with all epochs
5. **Analyze training curve** to pick best checkpoint
6. **Proceed to Stage-2** if SSIM > 0.6!

---

## 💡 Pro Tips

1. **Start small:** Always test with 5-10 samples first
2. **Compare endpoints:** Run epochs 2 and 10 first to see improvement
3. **Watch SSIM:** This metric matters most for Stage-1 structure learning
4. **Save outputs:** Keep validation results for paper figures
5. **Use guidance 2.5:** This is optimal for Stage-1 (tested)

---

## 🎉 Ready to Go!

Everything is configured and ready. Just run:

```bash
python validate_epochs.py --epochs 2 --num_samples 10
```

And you'll have comprehensive validation of your Stage-1 model! 🚀

---

**Last Updated:** March 8, 2026  
**HuggingFace Repo:** DrRORAL/ragaf-diffusion-checkpoints  
**Status:** ✅ Ready to use
