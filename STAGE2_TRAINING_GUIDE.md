# Stage 2 Training Guide

## ✅ Pre-Training Checklist

- [x] Stage 1 checkpoint downloaded: `/root/checkpoints/stage1_with_ssim/epoch_18.pt` (13GB)
- [x] Python dependencies installed
- [x] HuggingFace authenticated
- [x] GitHub remote attached: `https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion`
- [x] Stage 2 model verified:
  - ✅ RAGAF attention module ready
  - ✅ Adaptive fusion module ready
  - ✅ Feature projection layer created (512 → 4 latent channels)
  - ✅ Trainable parameters: 866M (mostly UNet)
  - ✅ Feature injection via residual conditioning implemented

---

## 📊 What Changed in Stage 2

### 1. **Feature Projection Layer Added**
```python
# New in Stage2SemanticRefinement.__init__
self.feature_projection = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.SiLU(),
    nn.Linear(hidden_dim, 4)  # Project to latent space (B, 4, H/8, W/8)
)
```

### 2. **Forward Pass Now Uses Fused Features**
```python
# In Stage2SemanticRefinement.forward()
# Project fused features to latent space
fused_latents = self.feature_projection(fused_region_features)  # (N, 4)
# Expand to match latent shape: (B, 4, H/8, W/8)
fused_latents_expanded = fused_latents.unsqueeze(-1).unsqueeze(-1).expand_as(latents)
# Inject via residual: latents + α * fused_latents
conditioned_latents = latents + 0.1 * fused_latents_expanded
# Use conditioned latents in UNet
noise_pred = self.unet(conditioned_latents, timestep, encoder_hidden_states=..., return_dict=False)[0]
```

### 3. **Batch Processing Fixed**
```python
# In RAGAFDiffusionTrainer.train_stage2_step()
# Now processes full batch, not just first item
# Properly handles tokenization and embedding for all text prompts
```

### 4. **Validation Loop Added**
```python
# New method: validate_stage2()
# Generates sample images at checkpoint intervals
# Compares Stage 1 vs Stage 2 outputs
# Logs SSIM, LPIPS differences to WandB
```

---

## 🚀 Starting Training

### **Quick Start (5 minutes)**

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Stage 2 only (recommended for first run)
nohup python scripts/training/train.py \
  --stage stage2 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --epochs 10 \
  --checkpoint_dir /root/checkpoints/stage2 \
  > train_stage2.log 2>&1 &

echo "Stage 2 training started!"
tail -f train_stage2.log
```

### **Full Dual-Stage Training**

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Train both stages
nohup python scripts/training/train.py \
  --stage both \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --epochs 10 \
  --checkpoint_dir /root/checkpoints \
  > train_dual.log 2>&1 &

echo "Dual-stage training started!"
```

### **With Custom Config**

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Edit default config first
python src/configs/config.py

# Then run with config
nohup python scripts/training/train.py \
  --stage stage2 \
  --batch_size 8 \
  --epochs 15 \
  > train_stage2_custom.log 2>&1 &
```

---

## 📈 Monitoring Training

### **Real-time Log Monitoring**

```bash
# Watch training progress
tail -f train_stage2.log

# Check specific metrics
grep "loss:" train_stage2.log | tail -20

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk usage
df -h /root/checkpoints
```

### **Expected Training Time**

- **Stage 2 only**: 10 epochs × ~45 min/epoch = ~7.5 hours
- **Dual-stage**: Stage 1 (10 ep) + Stage 2 (10 ep) = ~15 hours

### **Expected Resources**

- GPU: 20-28GB VRAM (RTX 5090: ~80% usage)
- Disk: ~50GB for checkpoints (auto-uploaded to HF Hub)
- RAM: ~16GB

---

## 💾 Checkpoint Management

### **Automatic Behavior**

```python
# Config from src/configs/config.py:
save_every_n_epochs: int = 2        # Save epoch 2, 4, 6, 8, 10
push_to_hub: bool = True            # Auto-upload to HuggingFace
hub_repo_id: str = "DrRORAL/ragaf-diffusion-checkpoints"

# NEW: Aggressive cleanup for 100GB container storage
# After each upload: Delete old checkpoints, keep only 2 most recent
# Result: Max storage ~39GB (2 checkpoints + final)
# Free space: ~61GB available
```

### **Smart Storage Strategy**

For your 100GB container, the training now:

1. ✅ Saves checkpoint locally (~13GB)
2. ✅ Uploads to HuggingFace Hub (~30s)
3. ✅ **Deletes old checkpoints** (keeps only 2 + final)
4. ✅ Maintains ~60GB free space

**Timeline Example**:
```
Epoch 2: Save → Upload → Keep epoch_2.pt (no deletion yet)
Epoch 4: Save → Upload → Delete epoch_2.pt, keep epoch_4.pt
Epoch 6: Save → Upload → Delete epoch_4.pt, keep epoch_6.pt  
...
Epoch 10: Save → Upload → Delete epoch_8.pt, keep epoch_10.pt
Final: Save final.pt → Keep all 3 (for resume capability)
```

**Storage at any point**: 
- Epoch checkpoints: ~13GB (1 most recent) or ~26GB (2 recent)
- Plus final.pt: ~13GB
- Total: ~26-39GB max ✅

### **Manual Commands**

```bash
# List saved checkpoints
ls -lh /root/checkpoints/stage2/

# Monitor storage during training
watch -n 10 'du -sh /root/checkpoints/stage2/'

# Download old checkpoint from HF if deleted
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id="DrRORAL/ragaf-diffusion-checkpoints",
    filename="stage2/epoch_8.pt",  # Any checkpoint
    local_dir="/root/checkpoints/stage2"
)
EOF
```

### **Emergency Cleanup**

If you run out of space:

```bash
# Delete all old epochs (keep final)
rm /root/checkpoints/stage2/epoch_*.pt

# Or delete everything and let training continue
rm -rf /root/checkpoints/stage2/*
# Training will still have HF backup and can resume
```

**See**: [`STORAGE_MANAGEMENT_GUIDE.md`](STORAGE_MANAGEMENT_GUIDE.md) for detailed storage strategy

---

## 🔍 Troubleshooting

### **Out of Memory (OOM)**
```bash
# Reduce batch size
python scripts/training/train.py --stage stage2 --batch_size 2 --epochs 10

# Or reduce hidden dimensions in config
# src/configs/config.py: hidden_dim: int = 256  # Down from 512
```

### **CUDA out of memory on graph processing**
```bash
# Reduce max regions in DataConfig
# src/configs/config.py: max_num_regions: int = 10  # Down from 20
```

### **Training is slow**
```bash
# Increase num_workers for data loading
python scripts/training/train.py --stage stage2 --batch_size 8 --epochs 10

# Check if GPU is actually being used
nvidia-smi  # Should show >50% GPU usage
```

### **Loss not decreasing**
- Check learning rate: try `1e-5` instead of `1e-4`
- Check data: verify dataset is loading with `python -c "from datasets.sketchy_dataset import *; print('OK')"`
- Check model: verify Stage 2 forward pass with `python verify_stage2.py`

---

## 📊 Expected Results

### **Stage 2 Training Metrics**

After 10 epochs, expect:
- **MSE Loss**: 0.3 → 0.15 (50% reduction)
- **LPIPS**: 0.2 → 0.12 (40% reduction)
- **Validation SSIM**: 0.161 → 0.25 (55% improvement over Stage 1)
- **Training time**: ~7.5 hours

### **Validation Outputs**

Stage 2 validation will save:
- `stage2_validation_epoch_X.png`: Before/after comparison
- `stage2_metrics_epoch_X.json`: Detailed metrics
- `stage2_attention_maps_epoch_X.png`: Region-text alignment visualization

---

## 🛠️ Configuration Reference

Edit `src/configs/config.py` to customize:

```python
# Model
hidden_dim: int = 512              # RAGAF hidden dimension
num_attention_heads: int = 8       # Multi-head attention heads
fusion_method: str = "learned"     # Adaptive fusion method

# Training
learning_rate: float = 1e-4        # Initial LR (will decay)
stage2_epochs: int = 10            # Number of epochs
batch_size: int = 4                # Batch size
mixed_precision: str = "bf16"      # BF16 for RTX 5090
save_every_n_epochs: int = 2       # Save frequency

# Data
image_size: int = 256              # Image resolution
max_num_regions: int = 20          # Max regions per image
batch_size: int = 4                # Batch size
```

---

## ✅ Next Steps

1. **Start Training**: Run the Quick Start command above
2. **Monitor**: Watch `train_stage2.log` for loss curves
3. **Validate**: Check stage2 checkpoints every 2 epochs
4. **Compare**: Visualize Stage 1 vs Stage 2 outputs
5. **Iterate**: Adjust hyperparameters based on results
6. **Deploy**: Use final checkpoint for inference

---

## 📞 Support

If issues arise:
1. Check `train_stage2.log` for error messages
2. Verify checkpoint exists: `ls -lh /root/checkpoints/stage1_with_ssim/epoch_18.pt`
3. Test Stage 2 model: `python verify_stage2.py`
4. Check Git status: `git -C /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion status`

---

**Status**: ✅ Ready to train
**Last updated**: March 19, 2026
**Checkpoint**: epoch_18.pt (Stage 1 SSIM 0.161)
