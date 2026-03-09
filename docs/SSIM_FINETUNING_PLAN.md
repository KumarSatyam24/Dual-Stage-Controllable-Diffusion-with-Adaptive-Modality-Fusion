# 🎯 SSIM-Focused Fine-tuning Strategy

## Current Training Overview

### Phase 1: Initial Training (Completed)
- **Checkpoint:** `epoch_12.pt`
- **Loss Components:** MSE + Perceptual (LPIPS)
- **Results:** Basic reconstruction learned
- **Best Metrics:**
  - PSNR: ~8.9 dB (Target: >22 dB)
  - SSIM: ~0.245 (Target: >0.60)
  - LPIPS: ~0.754 (Target: <0.50)

### Phase 2: Current Training (In Progress)
- **Status:** ACTIVE - Running since 12:02 PM
- **Configuration:**
  - Batch Size: 4
  - Learning Rate: 5e-6
  - SSIM Weight: 0.05
  - Validation: 100 samples
  - Target: 25 epochs
- **Starting from:** epoch_12.pt
- **Loss Function:** MSE + Perceptual + **0.05 × SSIM**

---

## 📈 Phase 3: SSIM-Focused Fine-tuning (Upcoming)

### Objective
**Fine-tune the model with increased emphasis on SSIM loss** to improve structural similarity between generated and target images.

### Why SSIM Fine-tuning?

Current issues:
- ✅ Model learns basic reconstruction (MSE)
- ✅ Model learns perceptual features (LPIPS)
- ❌ **Structural similarity is poor (SSIM: 0.24 vs target 0.60)**

SSIM captures:
- **Luminance** - Brightness matching
- **Contrast** - Local contrast preservation
- **Structure** - Structural patterns and edges

---

## 🎯 Fine-tuning Strategy

### Option A: Aggressive SSIM Focus
**When to use:** After Phase 2 shows minimal SSIM improvement

```bash
python train_stage1_with_ssim.py \
    --resume_from /root/checkpoints/stage1_with_ssim/best_model.pt \
    --batch_size 4 \
    --ssim_weight 0.3 \              # ⬆️ Increased from 0.05
    --perceptual_weight 0.3 \        # ⬇️ Reduced
    --mse_weight 0.4 \               # ⬇️ Reduced
    --learning_rate 1e-6 \           # 🐌 Lower LR for fine-tuning
    --validation_samples 100 \
    --epochs 15 \
    --experiment_name "ssim-finetune-aggressive"
```

**Loss Composition:**
- 40% MSE (reconstruction)
- 30% Perceptual (realism)
- 30% SSIM (structure) ⬆️

---

### Option B: Gradual SSIM Increase
**When to use:** If Phase 2 shows some SSIM improvement

```bash
python train_stage1_with_ssim.py \
    --resume_from /root/checkpoints/stage1_with_ssim/best_model.pt \
    --batch_size 4 \
    --ssim_weight 0.15 \             # 🔼 Moderate increase
    --perceptual_weight 0.4 \
    --mse_weight 0.45 \
    --learning_rate 2e-6 \
    --validation_samples 100 \
    --epochs 20 \
    --experiment_name "ssim-finetune-gradual"
```

**Loss Composition:**
- 45% MSE
- 40% Perceptual
- 15% SSIM ⬆️

---

### Option C: SSIM-Only Fine-tuning
**When to use:** As final polishing step after Options A or B

```bash
python train_stage1_with_ssim.py \
    --resume_from /root/checkpoints/ssim_finetuned/best_model.pt \
    --batch_size 4 \
    --ssim_weight 1.0 \              # 🎯 Pure SSIM optimization
    --perceptual_weight 0.0 \
    --mse_weight 0.0 \
    --learning_rate 5e-7 \           # 🐌 Very low LR
    --validation_samples 100 \
    --epochs 10 \
    --experiment_name "ssim-only-polish"
```

**⚠️ Warning:** Pure SSIM training may hurt perceptual quality!

---

## 📊 Expected Improvements

### Target Metrics After SSIM Fine-tuning

| Metric | Before | After Phase 2 | After SSIM Fine-tune | Target |
|--------|--------|---------------|---------------------|--------|
| **SSIM** | 0.245 | ~0.30 (est.) | **0.45-0.55** | >0.60 |
| **PSNR** | 8.9 | ~10-11 (est.) | **12-15** | >22 |
| **LPIPS** | 0.754 | ~0.73 (est.) | **0.65-0.70** | <0.50 |

---

## 🔧 Implementation Checklist

### Before Starting Fine-tuning:

- [ ] **Wait for Phase 2 to complete** (current training)
- [ ] **Analyze Phase 2 results:**
  ```bash
  # Check validation metrics
  python evaluate_stage1_validation.py \
      --checkpoint /root/checkpoints/stage1_with_ssim/best_model.pt \
      --num_samples 100
  ```
- [ ] **Select fine-tuning strategy** (Option A, B, or C)
- [ ] **Backup current best checkpoint**
  ```bash
  cp /root/checkpoints/stage1_with_ssim/best_model.pt \
     /root/checkpoints/backups/before_ssim_finetune.pt
  ```

### During Fine-tuning:

- [ ] **Monitor SSIM loss** closely in WandB
- [ ] **Validate every 2 epochs**
- [ ] **Early stopping** if SSIM doesn't improve for 3 validations
- [ ] **Watch for overfitting:**
  - Training loss keeps decreasing
  - Validation SSIM stagnates or drops

### After Fine-tuning:

- [ ] **Compare before/after visualizations**
  ```bash
  python tools/compare_checkpoints.py \
      --checkpoint1 /root/checkpoints/stage1_with_ssim/best_model.pt \
      --checkpoint2 /root/checkpoints/ssim_finetuned/best_model.pt \
      --num_samples 20
  ```
- [ ] **Full evaluation on test set**
- [ ] **Document results** in validation reports

---

## 🎓 Learning Rate Schedule

### Recommended Schedule for SSIM Fine-tuning:

```python
# In train_stage1_with_ssim.py
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-7  # Very low minimum for fine-tuning
)
```

**Or manual schedule:**
- Epochs 1-5: 2e-6
- Epochs 6-10: 1e-6
- Epochs 11-15: 5e-7

---

## 📈 Monitoring Commands

### Check Current Training Status:
```bash
# GPU usage
nvidia-smi

# Training process
ps aux | grep train_stage1_with_ssim

# Log tail
tail -f ~/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/train_phase2.log

# Latest checkpoint
ls -lht /root/checkpoints/stage1_with_ssim/ | head -5
```

### WandB Dashboard:
```bash
# Get WandB run URL
cat /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/wandb/latest-run/logs/debug.log | grep "View run"
```

---

## 🚨 Troubleshooting

### If SSIM doesn't improve:

1. **Check data quality:**
   - Are sketch-photo pairs correctly aligned?
   - Is data augmentation too aggressive?

2. **Adjust SSIM weight:**
   - Try higher values: 0.4, 0.5
   - Balance with perceptual loss

3. **Check SSIM loss implementation:**
   - Using correct SSIM loss (1 - SSIM)
   - Window size appropriate (default: 11)

4. **Learning rate too high:**
   - Reduce to 5e-7 or 1e-7
   - Use more gradual warmup

### If perceptual quality degrades:

1. **Don't reduce perceptual weight too much**
   - Keep at least 0.2-0.3
   - SSIM alone doesn't ensure realism

2. **Use hybrid approach:**
   - Fine-tune with moderate SSIM (0.15-0.2)
   - Don't go full SSIM-only

---

## 📝 Experiment Tracking

### Create experiment log:
```bash
# After Phase 2 completes
echo "Phase 2 Results:" > SSIM_FINETUNE_LOG.md
echo "Best SSIM: $(grep 'best_ssim' /root/checkpoints/stage1_with_ssim/training_stats.json)" >> SSIM_FINETUNE_LOG.md
```

### Track all fine-tuning runs:
- Experiment name
- Hyperparameters
- Best validation metrics
- Checkpoint paths
- WandB run URLs

---

## 🎯 Success Criteria

**Fine-tuning is successful if:**
- ✅ SSIM increases by at least 0.10 (e.g., 0.30 → 0.40+)
- ✅ PSNR increases by 2-3 dB
- ✅ LPIPS doesn't degrade by more than 0.05
- ✅ Visual quality remains good or improves

**Consider stopping if:**
- ❌ SSIM improvement < 0.02 after 5 epochs
- ❌ Perceptual quality visibly degrades
- ❌ Training becomes unstable

---

## 🚀 Quick Start Commands

### When Phase 2 completes, start fine-tuning:

```bash
# 1. Check Phase 2 results
ls -lht /root/checkpoints/stage1_with_ssim/

# 2. Run validation
./run_validation.sh --checkpoint /root/checkpoints/stage1_with_ssim/best_model.pt

# 3. Start SSIM fine-tuning (Option A recommended)
cd ~/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion && \
nohup python train_stage1_with_ssim.py \
    --resume_from /root/checkpoints/stage1_with_ssim/best_model.pt \
    --batch_size 4 \
    --ssim_weight 0.3 \
    --perceptual_weight 0.3 \
    --mse_weight 0.4 \
    --learning_rate 1e-6 \
    --validation_samples 100 \
    --epochs 15 \
    --experiment_name "ssim-finetune-aggressive" \
    > train_ssim_finetune.log 2>&1 &

echo "✅ SSIM fine-tuning started! Monitor with: tail -f train_ssim_finetune.log"
```

---

## 📚 Additional Resources

- **SSIM Paper:** "Image Quality Assessment: From Error Visibility to Structural Similarity" (Wang et al., 2004)
- **Multi-scale SSIM (MS-SSIM):** Consider for future improvements
- **SSIM Loss Variants:** Can try differentiable SSIM implementations

---

**Last Updated:** March 9, 2026
**Status:** Phase 2 training in progress, awaiting results for fine-tuning decision
