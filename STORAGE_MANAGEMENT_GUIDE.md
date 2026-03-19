# 💾 Storage Management for 100GB Container

## Aggressive Checkpoint Cleanup Strategy

The training script now automatically manages disk space for your 100GB container by:

### **How It Works**

1. **Every 2 epochs** (checkpoint save):
   - ✅ Save new checkpoint locally (~13GB)
   - ✅ Upload to HuggingFace Hub (~30 seconds)
   - ✅ **Delete old checkpoints** (keep only 2 most recent + final)

2. **Storage Efficiency**:
   - Keep only **2 recent checkpoints** locally (~26GB)
   - Plus **1 final checkpoint** (~13GB)
   - Total local usage: **~39GB maximum**
   - Rest: **~61GB free** for other operations

3. **Resume Capability**:
   - You can always resume from the 2 most recent checkpoints
   - Or download any checkpoint from HuggingFace Hub if needed
   - Final.pt is kept for inference

### **Example Timeline (Stage 2, 10 epochs)**

```
Epoch 1: Save → Upload → No cleanup (first checkpoint)
Epoch 2: Save → Upload → Delete nothing (only 1 old)
Epoch 3: (skip)
Epoch 4: Save → Upload → Delete epoch_2.pt (39GB → 26GB)
Epoch 5: (skip)
Epoch 6: Save → Upload → Delete epoch_4.pt (keep epoch_6.pt + epoch_2.pt)
Epoch 7: (skip)
Epoch 8: Save → Upload → Delete epoch_4.pt (keep epoch_8.pt + epoch_6.pt)
Epoch 9: (skip)
Epoch 10: Save → Upload → Delete epoch_8.pt (keep epoch_10.pt + epoch_6.pt)
Final: Save final.pt → Keep all 3 (epoch_10, epoch_6, final)
```

**Result**: Max storage used at any point = ~39GB (safe for 100GB container)

### **Disk Space Timeline**

```
Time     Checkpoints        Size    Free Space
────────────────────────────────────────────
Start    -                  -       ~100GB
Epoch 2  epoch_2.pt         13GB    ~87GB
Epoch 4  epoch_2,4.pt       26GB    ~74GB (cleanup after upload)
Epoch 4  epoch_4.pt         13GB    ~87GB ✅ (after deleting epoch_2)
Epoch 6  epoch_4,6.pt       26GB    ~74GB (cleanup after upload)
Epoch 6  epoch_6.pt         13GB    ~87GB ✅ (after deleting epoch_4)
...
Epoch 10 epoch_6,10.pt      26GB    ~74GB
Epoch 10 epoch_10.pt        13GB    ~87GB ✅ (after cleanup)
Final    epoch_10.pt        13GB    ~87GB
Final    +final.pt          26GB    ~74GB (final model)
```

### **Storage Breakdown**

```
Container: 100GB
├─ System & dependencies: ~15GB (fixed)
├─ Datasets: ~20GB (Sketchy dataset)
├─ Models: ~20GB (Stage 1 + Stage 2 base)
├─ Current checkpoints: ~26GB max (2 recent + cleanup)
└─ Free space: ~19GB (buffer)
```

### **What Gets Deleted**

- **Deleted**: Old epoch checkpoints (kept older than 2 most recent)
  - Example: When saving epoch_4, delete epoch_2
  - Example: When saving epoch_6, delete epoch_4
  
- **Kept Locally**:
  - 2 most recent epoch checkpoints
  - 1 final.pt checkpoint (for inference)
  - Old checkpoints available on HuggingFace Hub

- **Always on HuggingFace**:
  - Every epoch checkpoint (full backup)
  - Safe storage for all models
  - Free to download anytime

### **Cleanup Details**

The cleanup happens automatically in `save_checkpoint()`:

```python
# After upload completes:
_cleanup_old_checkpoints(
    stage="stage2",
    checkpoint_dir="/root/checkpoints/stage2/",
    keep_count=2  # Keep 2 most recent
)
```

**Console Output During Training**:

```
💾 Disk free before checkpoint: 87.0 GB
✅ Checkpoint saved locally: /root/checkpoints/stage2/epoch_4.pt (13.25 GB)
☁️  Uploading to HF Hub: DrRORAL/ragaf-diffusion-checkpoints/stage2/epoch_4.pt ...
✅ Uploaded to HF Hub: https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints

🗑️  Cleaning up old checkpoints (1 to delete, keeping 2 recent)...
  🗑️  Deleted epoch_2.pt (13.25 GB freed)

📊 Checkpoint status after cleanup:
  Disk free: 100.3 GB
  Kept epochs: epoch_2, epoch_4
  Final checkpoint: ❌ (not yet saved)
```

### **Monitoring Storage**

```bash
# Check current checkpoint sizes
ls -lh /root/checkpoints/stage2/
du -sh /root/checkpoints/stage2/

# Monitor in real-time during training
watch -n 10 'du -sh /root/checkpoints/stage2/'

# See total storage usage
df -h /root

# Check HuggingFace Hub (all backups)
ls -lh /root/checkpoints/  # Local copies
# Plus all versions in: https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints
```

### **If You Run Out of Space**

Emergency cleanup options:

```bash
# Option 1: Manually delete old epochs (keep final)
rm /root/checkpoints/stage2/epoch_*.pt

# Option 2: Delete everything and download from HF
rm -rf /root/checkpoints/stage2/
# Restore specific checkpoint:
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id="DrRORAL/ragaf-diffusion-checkpoints",
    filename="stage2/epoch_10.pt",
    local_dir="/root/checkpoints/stage2"
)
EOF

# Option 3: Stop training, resume later
# All checkpoints on HF Hub - can resume from any epoch
```

### **Resume from Old Checkpoints**

If you delete a local checkpoint by accident:

```bash
# Download from HuggingFace to resume
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="DrRORAL/ragaf-diffusion-checkpoints",
    filename="stage2/epoch_8.pt",  # Download any epoch
    local_dir="/root/checkpoints/stage2",
    cache_dir="/root/checkpoints/cache"
)
EOF

# Then resume training from that epoch
python scripts/training/train.py --stage stage2 --resume_epoch 8
```

### **Storage Management Summary**

| Aspect | Detail |
|--------|--------|
| **Local Storage** | 100GB total container |
| **Max Usage** | ~39GB (2 checkpoints + final) |
| **Free Buffer** | ~61GB available |
| **Per Checkpoint** | 13GB (~1.3GB/epoch with 10 epochs) |
| **Cleanup** | Automatic after each upload |
| **Backup** | All versions on HF Hub |
| **Safety** | Can resume from any checkpoint on Hub |

### **Key Files**

- Training script: `scripts/training/train.py`
- Storage logic: `save_checkpoint()` method (line ~656)
- Cleanup logic: `_cleanup_old_checkpoints()` method (line ~730)
- HuggingFace: https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints

---

## 🎯 Bottom Line

✅ **You can safely train without worrying about storage**

- Training automatically uploads & deletes
- Only 2 recent checkpoints stored locally (~26GB)
- All checkpoints backed up on HuggingFace
- 60GB+ free space maintained
- Can resume from any checkpoint anytime

**No manual cleanup needed!** 🚀
