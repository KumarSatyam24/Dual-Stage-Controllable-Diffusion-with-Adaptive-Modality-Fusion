# 📊 Weights & Biases (WandB) Tracking Setup

## Overview

The improved Stage-1 training script now includes **Weights & Biases** integration for comprehensive experiment tracking, visualization, and comparison.

---

## 🚀 Quick Setup

### 1. Install WandB

```bash
pip install wandb
```

### 2. Login to WandB

```bash
wandb login
```

This will prompt you for an API key. Get your key from: https://wandb.ai/authorize

**Or set it directly:**
```bash
export WANDB_API_KEY=your_api_key_here
```

### 3. Run Training with WandB

```bash
python train_improved_stage1.py
```

That's it! WandB tracking is enabled by default.

---

## 📈 What Gets Tracked

### Training Metrics (Every Batch)
- **Total Loss** - Combined MSE + perceptual loss
- **MSE Loss** - Reconstruction loss
- **Perceptual Loss** - LPIPS perceptual similarity
- **Learning Rate** - Current learning rate (with scheduler)

### Validation Metrics (Every 2 Epochs)
- **SSIM** (Structural Similarity) - Target: > 0.60
- **PSNR** (Peak Signal-to-Noise Ratio) - Target: > 22 dB
- **LPIPS** (Perceptual Distance) - Target: < 0.40

### Generated Images
- **3 sample images** per validation run
- Shows model's progress in generating realistic photos from sketches

### Final Metrics
- **Best SSIM** achieved during training
- **Total epochs** trained
- **Training time** in seconds

### Configuration
All hyperparameters are logged:
- Learning rate
- Batch size
- LoRA rank
- Perceptual loss weight
- Architecture details

---

## 🎯 Usage Examples

### Basic Training (Default WandB)

```bash
python train_improved_stage1.py
```

### Custom WandB Project Name

```bash
python train_improved_stage1.py \
    --wandb_project "my-sketch-diffusion" \
    --wandb_run_name "experiment-1-lr1e5"
```

### Disable WandB

```bash
python train_improved_stage1.py --no_wandb
```

### Custom Configuration with WandB

```bash
python train_improved_stage1.py \
    --learning_rate 5e-6 \
    --epochs 30 \
    --lora_rank 16 \
    --wandb_run_name "high-lora-rank-experiment"
```

---

## 📊 WandB Dashboard Features

### 1. **Live Training Curves**
Monitor in real-time:
- Loss trends (should decrease)
- SSIM trends (should increase)
- Learning rate schedule

### 2. **Image Gallery**
View generated images during training to see quality improvement

### 3. **Comparison Across Runs**
Compare different hyperparameters:
- Which learning rate works best?
- Does higher LoRA rank help?
- Impact of perceptual loss weight

### 4. **System Metrics**
Automatically tracks:
- GPU usage
- CPU usage
- Memory consumption
- Training speed

---

## 🔍 Viewing Your Runs

After training starts, WandB will print:
```
✅ WandB initialized: https://wandb.ai/your-username/ragaf-diffusion-stage1/runs/abc123
```

Click the link to view your run in the browser!

**Or visit:** https://wandb.ai/your-username/ragaf-diffusion-stage1

---

## 📋 WandB Project Structure

```
ragaf-diffusion-stage1/  (default project)
├── runs/
│   ├── stage1-lr1e-05-0308-1430/
│   │   ├── train/loss
│   │   ├── train/mse_loss
│   │   ├── train/perceptual_loss
│   │   ├── val/ssim
│   │   ├── val/psnr
│   │   ├── val/lpips
│   │   └── val/generated_images
│   │
│   ├── experiment-2/
│   └── experiment-3/
```

---

## 🎨 Custom Tags

The training script automatically adds tags:
- `stage1` - Stage-1 model
- `sketch-guided` - Sketch-conditional generation
- `diffusion` - Diffusion model
- `improved` - Improved hyperparameters

These help filter and organize experiments.

---

## 💡 Best Practices

### 1. **Meaningful Run Names**
Use descriptive names that identify the experiment:
```bash
--wandb_run_name "lr5e6-lora16-no-freeze"
```

### 2. **Group Related Experiments**
Use consistent project names for related experiments:
```bash
--wandb_project "ragaf-stage1-ablation"
```

### 3. **Add Notes**
The script automatically adds notes explaining the experiment purpose

### 4. **Compare Runs**
After multiple experiments, use WandB's comparison tool to find optimal hyperparameters

---

## 🔧 Troubleshooting

### "wandb not installed"

```bash
pip install wandb
```

### "Not logged in"

```bash
wandb login
```

Then paste your API key from https://wandb.ai/authorize

### "Want to use offline mode"

```bash
export WANDB_MODE=offline
python train_improved_stage1.py
```

Later sync:
```bash
wandb sync wandb/offline-run-*
```

### "Disable WandB completely"

```bash
python train_improved_stage1.py --no_wandb
```

---

## 📊 Example Dashboard

After a few epochs, you'll see:

**Training Loss:**
```
Epoch 1: 0.1234
Epoch 2: 0.0987
Epoch 3: 0.0856  ← Decreasing ✅
```

**Validation SSIM:**
```
Epoch 2: 0.32
Epoch 4: 0.41
Epoch 6: 0.48  ← Increasing ✅
```

**Generated Images:**
- Epoch 2: Blurry, lacks structure
- Epoch 6: Clearer, better structure
- Epoch 10: Sharp, realistic ✅

---

## 🎯 Success Metrics

Watch these metrics in WandB:

| Metric | Initial | Target | Status |
|--------|---------|--------|--------|
| SSIM | 0.25 | > 0.60 | 📈 Track improvement |
| LPIPS | 0.75 | < 0.40 | 📉 Track reduction |
| FID | 280 | < 50 | 📉 Track reduction |
| Training Loss | - | Stable decrease | 📉 Monitor convergence |

---

## 🔗 Useful Links

- **WandB Docs:** https://docs.wandb.ai/
- **Python API:** https://docs.wandb.ai/ref/python
- **Best Practices:** https://docs.wandb.ai/guides/track/best-practices
- **Dashboard:** https://wandb.ai/

---

## 📝 Example WandB Config

The script automatically logs this configuration:

```python
{
    "learning_rate": 1e-5,
    "num_epochs": 20,
    "batch_size": 4,
    "use_perceptual_loss": True,
    "perceptual_weight": 0.1,
    "freeze_unet": False,
    "lora_rank": 8,
    "architecture": "Stage1SketchGuidedDiffusion",
    "base_model": "stabilityai/stable-diffusion-2-1-base",
    "dataset": "Sketchy",
    "optimizer": "AdamW",
    "scheduler": "DDPM"
}
```

---

## ✅ Quick Checklist

- [ ] WandB installed: `pip install wandb`
- [ ] Logged in: `wandb login`
- [ ] Training started with WandB enabled
- [ ] Dashboard link printed in console
- [ ] Metrics appearing in dashboard
- [ ] Images logging correctly

---

## 🎉 You're All Set!

WandB will now automatically track all your experiments, making it easy to:
- Monitor training progress in real-time
- Compare different hyperparameters
- Share results with collaborators
- Generate reports for papers

**Happy training!** 🚀

---

**Note:** WandB tracking is **enabled by default**. To disable it, use `--no_wandb` flag.
