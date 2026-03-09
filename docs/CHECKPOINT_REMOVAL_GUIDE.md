# Checkpoint Removal Guide

## Summary

Based on your disk usage (from earlier: 39GB in stage1_improved), here's what's stored and what to do:

## Current Checkpoint Status

### 1. `/root/checkpoints/stage1_improved/` (~39 GB)
**Contents:**
- Multiple epoch_*.pt files (~13.8 GB each)
- Likely has: epoch_1.pt, epoch_2.pt, possibly epoch_3.pt
- training_log.json (small)

**Status on HuggingFace/WandB:**
- ✅ epoch_1.pt uploaded to WandB
- ✅ New epochs auto-upload to WandB
- ❌ NOT on HuggingFace main repo (only old failed checkpoints there)

**Action:** **DELETE ALL - Not needed locally**
```bash
rm -rf /root/checkpoints/stage1_improved/*.pt
rm -f /root/checkpoints/stage1_improved/training_log.json
```

**Why safe:**
- Checkpoints are on WandB cloud
- Training will create new ones as it progresses
- Can download from WandB if needed later

---

### 2. `/root/checkpoints/stage2/` (~190 MB)
**Contents:**
- Old Stage-2 checkpoints (not relevant)

**Action:** **DELETE**
```bash
rm -rf /root/checkpoints/stage2/
```

---

### 3. `/root/checkpoints/stage1/` (already deleted - 29 GB freed)
**Status:** ✅ Already removed in earlier cleanup

---

## HuggingFace Status

### What's on HuggingFace:
**DrRORAL/ragaf-diffusion-checkpoints/stage1/**
- epoch_2.pt, epoch_4.pt, epoch_6.pt, epoch_8.pt, epoch_10.pt, final.pt
- These are the **FAILED models** (SSIM=0.027)
- **Not worth keeping** - they don't work

**Action:** Can delete from HF later (not urgent, doesn't use local space)

### What's on WandB:
**satyam-kumar2022-vitstudent-ac-in/ragaf-diffusion-stage1**
- epoch_1.pt artifact (uploaded)
- Future epochs will auto-upload
- **This is your backup** - safe to delete local copies

---

## Commands to Run

### Option 1: Delete everything (Recommended)
```bash
# Stop and enter project directory
cd /root/checkpoints/

# Remove all checkpoint files
rm -rf stage1_improved/*.pt
rm -rf stage2/
rm -f stage1_improved/training_log.json

# Verify space freed
du -sh stage1_improved/
df -h /root
```

**Expected result:** ~39 GB freed

### Option 2: Keep only latest checkpoint
```bash
cd /root/checkpoints/stage1_improved/

# Keep only the most recent checkpoint
ls -t epoch_*.pt | tail -n +2 | xargs rm
ls -t *.pt | grep -v "^best\|^final" | tail -n +2 | xargs rm

# Verify
ls -lh *.pt
```

**Expected result:** ~25-30 GB freed (keeps 1 epoch)

---

## What Happens After Deletion?

1. **Training continues normally**
   - Saves new checkpoints as epochs complete
   - Each checkpoint: ~13.8 GB
   - Auto-uploads to WandB

2. **Space management going forward:**
   - After each epoch completes, consider deleting previous epoch
   - Keep only: latest epoch + best.pt
   - Expected disk usage: ~15-30 GB (manageable)

3. **Recovery if needed:**
   ```bash
   # Download from WandB
   wandb artifact get satyam-kumar2022-vitstudent-ac-in/ragaf-diffusion-stage1/epoch_1.pt:latest
   ```

---

## Disk Space Breakdown

**Before cleanup:**
- stage1_improved: 39 GB
- stage2: 0.2 GB
- **Total:** ~39 GB

**After cleanup:**
- stage1_improved: ~0 GB (empty, will grow as training continues)
- stage2: 0 GB (deleted)
- **Total freed:** ~39 GB

---

## Quick Commands for You

**To delete all checkpoints NOW:**
```bash
cd /root/checkpoints/
rm -rf stage1_improved/*.pt stage2/ stage1_improved/training_log.json
echo "Checkpoints deleted"
du -sh stage1_improved/
df -h | grep root
```

**To verify training is still running:**
```bash
ps aux | grep train_improved_stage1.py | grep -v grep
tail -20 nohup.out
```

---

## Recommendation

**DELETE ALL LOCAL CHECKPOINTS** because:
1. ✅ They're backed up on WandB
2. ✅ Training will create new ones automatically
3. ✅ Frees up ~39 GB immediately
4. ✅ Old checkpoints (epochs 1-3) are less useful than later ones anyway
5. ✅ Can always download from WandB if absolutely needed

**The 10GB+ you're seeing is likely:**
- Multiple epoch_*.pt files (~13.8 GB each)
- Training has completed 2-3 epochs, so 2-3 × 13.8 GB = 28-42 GB

---

## Run This Now

```bash
#!/bin/bash
cd /root/checkpoints/
rm -rf stage1_improved/*.pt stage2/
echo "✅ Deleted all checkpoints"
echo "Space freed:"
df -h | grep root | awk '{print $4 " available"}'
```
