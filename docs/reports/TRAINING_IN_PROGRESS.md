# 🚀 Training In Progress!

## ✅ Training Status: RUNNING

Your RAGAF-Diffusion model is now training on the Sketchy dataset!

---

## 📊 Training Configuration

### Models Loaded
- ✅ **Stable Diffusion v1.5** (base model)
- ✅ **VAE** (Variational Autoencoder) - frozen
- ✅ **Text Encoder** (CLIP) - frozen  
- ✅ **Stage 1**: Sketch-Guided Diffusion with LoRA (rank=4)
- ✅ **Stage 2**: Semantic Refinement with RAGAF attention

### Dataset
- **Name**: Sketchy
- **Training samples**: 30,685 sketch-photo pairs
- **Batch size**: 4
- **Workers**: 4

### Training Settings
- **Stage**: Both (Stage 1 + Stage 2)
- **Epochs**: 10 per stage
- **Learning rate**: 1e-4
- **Mixed precision**: FP16
- **Device**: CUDA (GPU)
- **Checkpoints**: Saving to `/workspace/outputs`

---

## 📝 Training Log

The full training output is being logged to:
```
/workspace/training.log
```

### Monitor Training Progress

```bash
# Watch the training log in real-time
tail -f /workspace/training.log

# Check last 50 lines
tail -50 /workspace/training.log

# Search for specific epoch
grep "Epoch" /workspace/training.log
```

### Check GPU Usage

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi
```

### View Checkpoints

```bash
# List saved checkpoints
ls -lh /workspace/outputs/

# Check checkpoint details
ls -lh /workspace/outputs/*.pt
```

---

## 🎯 Expected Training Time

With the current configuration:
- **Stage 1**: ~7,672 iterations per epoch
- **Stage 2**: Similar iterations
- **Total**: ~20 epochs (10 per stage)

Depending on your GPU:
- **RTX 4090**: ~2-4 hours
- **RTX 3090**: ~4-6 hours  
- **A100**: ~1-2 hours

---

## 📈 What's Happening Now

The training process is:

1. **Loading batches** of sketch-photo pairs
2. **Stage 1 Training**: Learning to generate coarse images from sketches
3. **Stage 2 Training**: Refining with semantic information and RAGAF attention
4. **Saving checkpoints** every 2 epochs
5. **Logging metrics** every 10 steps

---

## 🛑 Stop Training

If you need to stop:

```bash
# Find the training process
ps aux | grep train.py

# Stop gracefully (Ctrl+C in the terminal or kill the process)
```

---

## ✨ After Training

Once complete, you'll find:

### Model Checkpoints
```
/workspace/outputs/
├── stage1_epoch_2.pt
├── stage1_epoch_4.pt
├── ...
├── stage1_final.pt
├── stage2_epoch_2.pt
├── ...
└── stage2_final.pt
```

### Use Trained Model

```bash
# Run inference with your trained model
python inference.py \
    --sketch-path /workspace/datasets/sketchy/sketch/tx_000000000000/bear/n02132136_1290-1.png \
    --prompt "A photo of a bear" \
    --stage1-checkpoint /workspace/outputs/stage1_final.pt \
    --stage2-checkpoint /workspace/outputs/stage2_final.pt \
    --output-path /workspace/outputs/generated_bear.png
```

---

## 🔍 Troubleshooting

### Check if training is still running:
```bash
ps aux | grep train.py
```

### Check CUDA memory:
```bash
nvidia-smi
```

### If training crashes:
Check the log for errors:
```bash
tail -100 /workspace/training.log
```

---

## 💡 Tips

- Training will continue even if you disconnect from the pod
- Checkpoints are saved to persistent storage (`/workspace`)
- You can resume from a checkpoint if interrupted
- Monitor GPU temperature to ensure it's not overheating

---

**Training started at**: `date`  
**Log file**: `/workspace/training.log`  
**Output directory**: `/workspace/outputs/`

Good luck with your training! 🎨
