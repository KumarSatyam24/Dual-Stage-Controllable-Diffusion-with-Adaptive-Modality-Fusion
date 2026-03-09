# 🚀 Quick Validation Commands

## Epoch-by-Epoch Validation

### Basic (Recommended)
```bash
python validate_epochs.py --num_samples 50
```

### Quick Test (10 samples)
```bash
python validate_epochs.py --num_samples 10
```

### Specific Epochs Only
```bash
python validate_epochs.py --epochs 1 5 10 --num_samples 30
```

### Full Validation (100 samples)
```bash
python validate_epochs.py --num_samples 100 --output_dir full_validation
```

### Custom HuggingFace Repo
```bash
python validate_epochs.py \
    --hf_repo DrRORAL/ragaf-diffusion-checkpoints \
    --num_samples 50
```

---

## Single Checkpoint Validation

### Quick (10 samples)
```bash
./run_validation.sh --quick
```

### Standard (50 samples)
```bash
./run_validation.sh
```

### Full (200 samples)
```bash
./run_validation.sh --full
```

---

## View Results

### Open HTML Report
```bash
firefox validation_results/validation_report.html
```

### View Metrics Plot
```bash
xdg-open validation_results/metrics_across_epochs.png
```

### Check JSON Metrics
```bash
cat validation_results/all_epochs_metrics.json | jq
```

---

## Troubleshooting

### Check GPU
```bash
nvidia-smi
```

### Check Dataset
```bash
ls /workspace/sketchy/photo/
ls /workspace/sketchy/sketch/
```

### Check Checkpoints
```bash
ls /root/checkpoints/stage1/
```

### View Latest Log
```bash
tail -f validation_results/validation.log
```

---

## What to Expect

| Command | Samples | Time | Purpose |
|---------|---------|------|---------|
| Quick test | 10 | ~5 min | Initial check |
| Standard | 50 | ~25 min | Reliable metrics |
| Full | 100 | ~50 min | Publication-ready |

---

## Decision Tree

```
Run: python validate_epochs.py --num_samples 50
└─> Check: validation_results/metrics_across_epochs.png

    ├─> SSIM > 0.6 AND FID < 50?
    │   └─> ✅ Ready for Stage-2!
    │
    ├─> SSIM increasing steadily?
    │   └─> ⏳ Train more epochs
    │
    ├─> SSIM decreasing after epoch X?
    │   └─> ⚠️ Use checkpoint from epoch X (overfitting)
    │
    └─> SSIM flat < 0.5?
        └─> ❌ Debug training (LR, data, architecture)
```

---

## Target Metrics (Stage-1)

✅ **Good Performance:**
- SSIM: 0.60-0.75
- LPIPS: 0.25-0.40
- FID: 25-50
- PSNR: 22-30 dB

⚠️ **Acceptable:**
- SSIM: 0.50-0.60
- LPIPS: 0.40-0.50
- FID: 50-80

❌ **Needs Work:**
- SSIM: < 0.50
- LPIPS: > 0.50
- FID: > 80

---

**TIP:** Always start with `--num_samples 20` for quick check, then run `--num_samples 50` for final report!
