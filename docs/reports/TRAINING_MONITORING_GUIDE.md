# Pre-Training Checklist & Storage Management

## Current Storage Status

```
/ (root filesystem):     115 GB total, 11 GB used, 105 GB free  ✅ PLENTY OF SPACE
/workspace (network):     10 GB total,  8 GB used,  2 GB free   ⚠️  TIGHT!
```

## Storage Requirements for Training

**Each checkpoint:** ~4.5 GB  
**Training plan:** Save every 2 epochs (epoch_2, 4, 6, 8, 10, final)  
**Total checkpoints:** 6 files × 4.5 GB = **27 GB needed**

### ✅ You're SAFE - Checkpoints save to `/root/checkpoints/`
- This is on the **root filesystem (115 GB)** with **105 GB free**
- 27 GB needed < 105 GB available ✅
- Previous config mistake (saving to `/workspace`) has been fixed!

## Things to Monitor Before Training Ends

### 1. ⚠️ **Disk Space** (Check Every ~2 Hours)
```bash
# Quick check
df -h / | tail -1

# Detailed breakdown
du -sh /root/checkpoints/stage1/
du -sh /root/old_checkpoints_broken/
```

**Action if space gets low (<20 GB free):**
- Delete old broken checkpoints: `rm -rf /root/old_checkpoints_broken/`
- This frees 9 GB immediately

### 2. 🔄 **Training Progress** (Check Hourly)
```bash
# View live log
tail -f /workspace/train_stage1_FIXED.log

# Check current epoch
grep "Epoch" /workspace/train_stage1_FIXED.log | tail -5

# Check GPU usage
nvidia-smi
```

### 3. 📤 **HuggingFace Hub Uploads** (Check After Each Checkpoint)
```bash
# List uploaded files
huggingface-cli scan-cache DrRORAL/ragaf-diffusion-checkpoints

# Check if upload succeeded
grep "Successfully uploaded" /workspace/train_stage1_FIXED.log
```

**Action if upload fails:**
- Checkpoints still saved locally at `/root/checkpoints/stage1/`
- Can manually upload later with: `huggingface-cli upload ...`

### 4. 🔥 **GPU Temperature/Throttling**
```bash
# Check every few hours
nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics --format=csv
```

**Normal:** 70-85°C  
**Action if >90°C:** Training might slow down (throttling)

### 5. 💾 **Checkpoint Integrity**
After training completes, verify checkpoints aren't corrupted:
```bash
python3 << 'EOF'
import torch
for epoch in [2, 4, 6, 8, 10]:
    path = f"/root/checkpoints/stage1/epoch_{epoch}.pt"
    try:
        ckpt = torch.load(path, map_location='cpu')
        print(f"✅ epoch_{epoch}.pt - OK ({len(ckpt.keys())} keys)")
    except Exception as e:
        print(f"❌ epoch_{epoch}.pt - CORRUPTED: {e}")
EOF
```

## Recommended Actions NOW (Before Training Ends)

### ✅ Immediate (Do Now):
1. **Delete old HF Hub checkpoints** (type `yes` in the waiting prompt)
   - Frees 27 GB on HuggingFace
   - Avoids confusion with broken model

2. **Set up monitoring cron/script:**
```bash
# Create monitoring script
cat > /root/monitor_training.sh << 'SCRIPT'
#!/bin/bash
echo "=== Training Monitor $(date) ==="
echo "Disk space:"
df -h / | grep overlay
echo ""
echo "Checkpoints:"
ls -lh /root/checkpoints/stage1/*.pt 2>/dev/null | wc -l
echo "files saved"
echo ""
echo "Latest log lines:"
tail -3 /workspace/train_stage1_FIXED.log
echo "================================"
SCRIPT
chmod +x /root/monitor_training.sh

# Run every hour
watch -n 3600 /root/monitor_training.sh
```

### 📋 Optional (Nice to Have):
1. **Clean up old broken checkpoints locally** (after HF deletion confirms):
```bash
rm -rf /root/old_checkpoints_broken/  # Frees 9 GB
```

2. **Set up email/Discord webhook** for training completion notification
   - Training will take ~6 hours
   - Nice to get notified when done

3. **Prepare test script** for when first checkpoint (epoch_2) is ready:
```bash
# Will test if sketch conditioning actually works this time
python3 test_stage1_trained.py  # Run after epoch_2 saves
```

## Critical Warnings

### 🚨 DO NOT:
1. ❌ Delete `/root/checkpoints/` during training
2. ❌ Stop training before epoch 2 (no checkpoint saved yet)
3. ❌ Fill up `/workspace` to 100% (can crash system)
4. ❌ Restart machine without checking if training finished

### ✅ DO:
1. ✅ Keep terminal/tmux session alive (don't close SSH)
2. ✅ Monitor disk space if training exceeds 8 hours
3. ✅ Test epoch_2 checkpoint to verify sketch conditioning works
4. ✅ Keep HF Hub token active (already in env vars)

## Post-Training Checklist

After training completes:
- [ ] Verify all 6 checkpoints saved successfully
- [ ] Check all uploaded to HuggingFace Hub
- [ ] Test epoch_10 with real sketch input
- [ ] Compare old broken model vs new fixed model outputs
- [ ] Delete local old_checkpoints_broken/ to free 9 GB
- [ ] Document results before starting Stage 2

---

**Current Status:** ✅ Storage is adequate, training is safe to continue  
**ETA:** ~6 hours from start (check log for current progress)  
**Next checkpoint:** epoch_2 (~1.2 hours if started at 21:13)
