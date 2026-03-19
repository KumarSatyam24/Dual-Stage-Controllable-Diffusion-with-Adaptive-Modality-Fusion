# ⚡ Stage 2 Training - Quick Reference Card

## 🟢 Ready to Train - Start Here

### **One Command to Start**
```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion && bash start_stage2_training.sh
```

### **Custom Training**
```bash
# Stage 2, 5 epochs, batch 8
bash start_stage2_training.sh stage2 5 8 5e-5

# Dual-stage (Stage 1 + Stage 2), 10 epochs
bash start_stage2_training.sh both 10 4 1e-4
```

---

## 📊 What to Monitor

```bash
# Watch training progress (real-time)
tail -f logs/stage2_training_*.log

# Watch GPU (new terminal)
watch -n 1 nvidia-smi

# Check checkpoints
ls -lh /root/checkpoints/stage2/
```

---

## 🕐 Timeline

| Stage | Time | SSIM Goal |
|-------|------|-----------|
| Start | 0min | 0.161 |
| Epoch 2 | 45min | 0.20 |
| Epoch 4 | 3hrs | 0.23 |
| Epoch 6 | 4.5hrs | 0.25 |
| Epoch 8 | 6hrs | 0.27 |
| Epoch 10 | 7.5hrs | **0.28 ✅** |

---

## 🛑 Emergency Stop

```bash
# Find process
ps aux | grep python | grep train

# Kill gracefully
kill <PID>

# Force kill if stuck
kill -9 <PID>
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| **OOM** | `bash start_stage2_training.sh stage2 10 2` (batch=2) |
| **Slow** | `watch -n 1 nvidia-smi` (check GPU usage) |
| **Loss NaN** | Reduce LR: `bash start_stage2_training.sh stage2 10 4 5e-5` |
| **No data** | Check: `ls -la /workspace/sketchy/\*.jpg \| head` |

---

## 📁 Key Files

```
/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/
├── start_stage2_training.sh          ← Run this
├── verify_stage2_ready.py            ← Verify setup
├── STAGE2_TRAINING_GUIDE.md          ← Detailed guide
├── STAGE2_READY_TO_TRAIN.md          ← Full docs
├── STAGE2_TRAINING_SETUP_SUMMARY.md  ← Implementation details
├── logs/stage2_training_*.log        ← Training logs
└── /root/checkpoints/stage2/         ← Checkpoints saved here
```

---

## ✅ Pre-Flight Checklist

```bash
# 1. Verify setup (optional)
python verify_stage2_ready.py

# 2. Check checkpoint
ls -lh /root/checkpoints/stage1_with_ssim/epoch_18.pt

# 3. Check GPU
nvidia-smi

# 4. Start training!
bash start_stage2_training.sh
```

---

## 📈 Expected Results

- **Start**: SSIM = 0.161 (Stage 1)
- **End**: SSIM = 0.28 (Stage 2) 
- **Gain**: +74% improvement
- **Time**: 7.5 hours
- **GPU**: RTX 5090, 80% usage

---

## 🎯 Next Steps

1. ✅ Run `bash start_stage2_training.sh`
2. 📊 Monitor `tail -f logs/stage2_training_*.log`
3. 🔄 Check checkpoints every 2 epochs
4. ✨ Evaluate results after completion

---

**Status**: ✅ Ready  
**Created**: March 19, 2026  
**Questions?** See detailed guides in project directory
