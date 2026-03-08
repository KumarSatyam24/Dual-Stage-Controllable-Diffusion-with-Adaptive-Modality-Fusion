# What You Need to Do Before Training Ends

## ✅ **Good News: You're Safe!**

### Storage is Fine:
- **Root filesystem:** 105 GB free (checkpoints save here)
- **Needed:** 27 GB for all checkpoints
- **Margin:** 78 GB extra → ✅ **No worries!**

### Training Location is Correct:
- Checkpoints: `/root/checkpoints/stage1/` ✅ (correct filesystem)
- Old mistake (saving to `/workspace`) is already fixed
- HF Hub auto-upload enabled for cloud backup

---

## 🎯 **Actions Required NOW:**

### 1. Delete Old Broken Checkpoints from HuggingFace Hub

**Status:** Script is waiting for your response

**Action:** Go to the terminal where it says:
```
Delete all old checkpoints? (yes/no):
```

**Type:** `yes` and press Enter

**Why:** 
- Removes 6 broken checkpoints from cloud (27 GB)
- Prevents confusion between old broken model and new fixed model
- New checkpoints will upload automatically during training

---

### 2. Monitor Training Progress (Optional but Recommended)

**Every 1-2 hours, run:**
```bash
/root/monitor.sh
```

**Or set up auto-refresh:**
```bash
watch -n 300 /root/monitor.sh   # Updates every 5 minutes
```

**What to watch for:**
- ✅ "Training running" message
- ✅ Checkpoints appearing every ~1.2 hours
- ✅ Root disk stays below 80% usage
- ⚠️ Any error messages in log

---

### 3. Test First Checkpoint (When Available)

**After epoch_2 saves (~1-2 hours from start):**
```bash
python3 test_stage1_trained.py
```

**This verifies:**
- Sketch conditioning is actually working this time
- Output matches sketch structure (not like the broken model)
- Quality improves with each checkpoint

---

## 🚨 **Things That Could Go Wrong (Unlikely):**

### Problem: Training Crashes
**Check:** `tail -100 /workspace/train_stage1_FIXED.log`  
**Solution:** Restart with: `python3 train.py --stage stage1`

### Problem: Disk Full (Very Unlikely with 105 GB free)
**Check:** `df -h /`  
**Solution:** Delete old backups: `rm -rf /root/old_checkpoints_broken/`

### Problem: HF Upload Fails
**Impact:** Checkpoints still saved locally → You're safe!  
**Solution:** Manual upload later with `huggingface-cli upload`

---

## 📋 **After Training Completes (~6 Hours):**

1. **Verify all checkpoints exist:**
   ```bash
   ls -lh /root/checkpoints/stage1/
   # Should see: epoch_2.pt, epoch_4.pt, epoch_6.pt, epoch_8.pt, epoch_10.pt, final.pt
   ```

2. **Test the final model:**
   ```bash
   python3 test_stage1_trained.py
   # Compare output with broken model - should be MUCH better!
   ```

3. **Clean up old backups (optional):**
   ```bash
   rm -rf /root/old_checkpoints_broken/  # Frees 9 GB
   ```

4. **Proceed to Stage 2** (only if Stage 1 sketch conditioning works!)

---

## 🎮 **Quick Commands Reference:**

```bash
# Check if training is running
ps aux | grep "python3 train.py" | grep -v grep

# View live training log
tail -f /workspace/train_stage1_FIXED.log

# Check disk space
df -h /

# Monitor status
/root/monitor.sh

# Test checkpoint
python3 test_stage1_trained.py

# GPU usage
nvidia-smi
```

---

## ⏰ **Timeline Estimate:**

- **Started:** ~21:13 (March 6, 2026)
- **Epoch 2:** ~22:25 (1.2 hours) → First checkpoint saved
- **Epoch 4:** ~23:37
- **Epoch 6:** ~00:49 (March 7)
- **Epoch 8:** ~02:01
- **Epoch 10:** ~03:13
- **Complete:** ~03:30 → 6+ hours total

---

## ✅ **TL;DR - You Only Need To:**

1. **NOW:** Type `yes` in the terminal to delete old HF Hub checkpoints
2. **Optional:** Run `/root/monitor.sh` every few hours to check progress
3. **When done:** Test the model to verify sketch conditioning works!

**Everything else is automatic!** 🎉

The training will:
- ✅ Save checkpoints to `/root/checkpoints/` (plenty of space)
- ✅ Upload to HuggingFace Hub automatically
- ✅ Complete in ~6 hours
- ✅ Log everything to `/workspace/train_stage1_FIXED.log`

**You can safely go to sleep/work and check back later!**
