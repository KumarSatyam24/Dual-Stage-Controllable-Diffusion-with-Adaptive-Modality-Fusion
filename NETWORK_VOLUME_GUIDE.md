# Accessing Network Volume on RunPod

## 📍 Network Volume Location

Your RunPod network volume is mounted at: **`/workspace`**

Current contents:
- `myfile.zip` (1.1 GB compressed)

---

## 🗂️ Setting Up Your Dataset

### Option 1: Extract Dataset to Network Volume (Recommended)

This keeps your dataset on the persistent network volume:

```bash
# Create dataset directory on network volume
mkdir -p /workspace/datasets

# Extract your dataset
cd /workspace
unzip myfile.zip -d /workspace/datasets/

# Check the extracted structure
ls -la /workspace/datasets/
```

### Option 2: Create Symbolic Link

If you want to keep data on the network volume but access it from the project:

```bash
# Extract to network volume
mkdir -p /workspace/datasets
cd /workspace
unzip myfile.zip -d /workspace/datasets/

# Create symbolic link from project
ln -s /workspace/datasets /root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/datasets_link
```

### Option 3: Direct Path Configuration

Simply update `configs/config.py` to point directly to `/workspace/datasets/`:

```python
sketchy_root: str = "/workspace/datasets/sketchy"
coco_root: str = "/workspace/datasets/coco"
```

---

## 🔧 Quick Setup Commands

### 1. Extract and inspect your dataset:

```bash
# Peek inside the zip without extracting
unzip -l /workspace/myfile.zip | head -50

# Create datasets directory
mkdir -p /workspace/datasets

# Extract the zip file
cd /workspace
unzip myfile.zip -d /workspace/datasets/

# Check what was extracted
ls -lah /workspace/datasets/
```

### 2. Update your configuration:

Edit `configs/config.py` and change the dataset paths:

```python
# For Sketchy dataset
sketchy_root: str = "/workspace/datasets/sketchy"

# For COCO dataset
coco_root: str = "/workspace/datasets/coco"
```

### 3. Verify the dataset structure:

```bash
# Run the verification script with network volume path
python verify_dataset.py --dataset sketchy --data-root /workspace/datasets/sketchy
```

---

## ✅ Benefits of Using Network Volume

1. **Persistence**: Data survives pod restarts
2. **Large Storage**: Network volumes typically have much more space than container storage
3. **Speed**: Mounted over fast network, good for training
4. **Shared Access**: Can be mounted on multiple pods if needed

---

## 📊 Storage Usage

Check your network volume usage:

```bash
# See total space and usage
df -h /workspace

# See size of specific directories
du -sh /workspace/*
du -sh /workspace/datasets/*
```

---

## 🚨 Important Notes

- **Container storage** (`/root/`, etc.) is **temporary** and will be lost when pod stops
- **Network volume** (`/workspace/`) is **persistent** across pod restarts
- Always keep datasets on `/workspace/` for RunPod
- The network volume is shared across all your pods in the same region

---

## 🔍 Troubleshooting

### Check if network volume is mounted:
```bash
df -h | grep workspace
ls -la /workspace
```

### Check dataset after extraction:
```bash
# For Sketchy dataset
ls -la /workspace/datasets/sketchy/
ls -la /workspace/datasets/sketchy/sketch/
ls -la /workspace/datasets/sketchy/photo/
```

### Test dataset loading:
```bash
python verify_dataset.py --dataset sketchy --data-root /workspace/datasets/sketchy
```
