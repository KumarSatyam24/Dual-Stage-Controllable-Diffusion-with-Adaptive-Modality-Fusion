# ✅ Dataset Access Setup Complete

## 📍 Your Dataset Location

Your **Sketchy dataset** is now accessible at:
```
/workspace/datasets/sketchy/
```

## 🗂️ Dataset Structure Verified

```
/workspace/datasets/sketchy/
├── photo/
│   └── tx_000000000000/
│       ├── airplane/
│       ├── alarm_clock/
│       ├── ant/
│       ├── ape/
│       ├── apple/
│       └── ... (125 categories)
└── sketch/
    └── tx_000000000000/
        ├── airplane/
        ├── alarm_clock/
        ├── ant/
        └── ... (125 categories)
```

## ⚙️ Configuration Updated

The `configs/config.py` file has been updated to use your network volume:

```python
sketchy_root: str = "/workspace/datasets/sketchy"  # RunPod network volume
coco_root: str = "/workspace/datasets/coco"  # RunPod network volume
```

## 🚀 Next Steps

### 1. Install Dependencies (if not already done)

```bash
cd /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion
pip install -r requirements.txt
```

### 2. Set Environment Variable (Optional for scripts)

Add to your shell or use before running scripts:

```bash
export SKETCHY_ROOT=/workspace/datasets/sketchy
```

Or add to `~/.bashrc` for persistence:

```bash
echo 'export SKETCHY_ROOT=/workspace/datasets/sketchy' >> ~/.bashrc
source ~/.bashrc
```

### 3. Verify Dataset

Once dependencies are installed:

```bash
python verify_dataset.py
```

### 4. Start Training

```bash
# Train both stages
python train.py --dataset sketchy

# Or train specific stage
python train.py --dataset sketchy --train-stage stage1
```

## 📊 Storage Information

- **Network Volume Mount**: `/workspace`
- **Volume Size**: 1.7 PB total, 649 TB used
- **Dataset Location**: `/workspace/datasets/sketchy/`
- **Original Zip**: `/workspace/myfile.zip` (1.1 GB - can be deleted to save space)

## 🔍 Useful Commands

### Check dataset size:
```bash
du -sh /workspace/datasets/sketchy/
```

### Count images per category:
```bash
# Count photos
ls /workspace/datasets/sketchy/photo/tx_000000000000/airplane/ | wc -l

# Count sketches
ls /workspace/datasets/sketchy/sketch/tx_000000000000/airplane/ | wc -l
```

### List all categories:
```bash
ls /workspace/datasets/sketchy/photo/tx_000000000000/
```

### Remove original zip file (optional - saves 1.1 GB):
```bash
rm /workspace/myfile.zip
```

## ✨ Benefits of Network Volume Setup

✅ **Persistent Storage**: Data survives pod restarts  
✅ **Large Capacity**: Network volume has much more space  
✅ **Fast Access**: Optimized for training workloads  
✅ **Shared Access**: Can mount on multiple pods  

## 🎯 Ready to Train!

Your dataset is now properly set up and ready for training. The configuration file already points to the correct location, so you can start training immediately after installing dependencies.

---

**Need help?** Check `NETWORK_VOLUME_GUIDE.md` for more details about RunPod network volumes.
