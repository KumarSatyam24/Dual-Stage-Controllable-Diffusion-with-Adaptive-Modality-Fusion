# Dataset Setup Guide for RAGAF-Diffusion

This guide explains how to download and set up the Sketchy and MS COCO datasets for training.

---

## 📋 Sketchy Dataset

### Required Directory Structure

The Sketchy dataset should be organized in the following format:

```
sketchy/
├── sketch/
│   └── tx_000000000000/
│       ├── airplane/
│       │   ├── n000001.png
│       │   ├── n000002.png
│       │   └── ...
│       ├── ant/
│       │   ├── n000001.png
│       │   └── ...
│       ├── bear/
│       ├── bicycle/
│       └── ... (125 categories total)
└── photo/
    └── tx_000000000000/
        ├── airplane/
        │   ├── n000001.jpg
        │   ├── n000002.jpg
        │   └── ...
        ├── ant/
        │   ├── n000001.jpg
        │   └── ...
        ├── bear/
        ├── bicycle/
        └── ... (125 categories total)
```

### Key Points:
- **Sketch files**: PNG format (grayscale), located in `sketch/tx_000000000000/{category}/`
- **Photo files**: JPG format (RGB), located in `photo/tx_000000000000/{category}/`
- **File naming**: Sketches and photos must have matching names (e.g., `n000001.png` ↔ `n000001.jpg`)
- **Total size**: ~10GB
- **Categories**: 125 object categories (airplane, ant, bear, bicycle, etc.)
- **Pairs**: ~75,000 sketch-photo pairs

---

## 🔽 Download Sketchy Dataset

### Option 1: Official Website (Recommended)

1. **Visit**: https://sketchy.eye.gatech.edu/

2. **Download**: Click on "Download Dataset" button

3. **Extract**: Unzip the downloaded file
   ```bash
   unzip sketchy_database_extended.zip -d ~/datasets/sketchy
   ```

4. **Verify structure**: Check that you have the correct folders
   ```bash
   ls ~/datasets/sketchy/
   # Should show: sketch/ photo/ (and possibly info.mat, etc.)
   
   ls ~/datasets/sketchy/sketch/tx_000000000000/
   # Should show: airplane/ ant/ bear/ bicycle/ ... (125 categories)
   ```

### Option 2: Google Drive (Alternative)

The Sketchy dataset is also available on Google Drive:
- Search for "Sketchy dataset extended" on Google
- Download from academic mirrors or shared drives
- Extract as shown above

### Option 3: Academic Torrent

Some researchers share the dataset via academic torrents or institutional repositories. Check with your university's data resources.

---

## 📋 MS COCO Dataset

### Required Directory Structure

The MS COCO dataset should be organized as follows:

```
coco/
├── train2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ... (~118,000 images)
├── val2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ... (~5,000 images)
└── annotations/
    ├── captions_train2017.json
    ├── captions_val2017.json
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Key Points:
- **Images**: JPG format (RGB)
- **Annotations**: JSON format with captions and instance segmentations
- **Total size**: ~25GB (images + annotations)
- **Train images**: ~118,000
- **Val images**: ~5,000

---

## 🔽 Download MS COCO Dataset

### Official Download (Recommended)

1. **Visit**: https://cocodataset.org/#download

2. **Download images**:
   ```bash
   # Create directory
   mkdir -p ~/datasets/coco
   cd ~/datasets/coco
   
   # Download train2017 images (~18GB)
   wget http://images.cocodataset.org/zips/train2017.zip
   
   # Download val2017 images (~1GB)
   wget http://images.cocodataset.org/zips/val2017.zip
   
   # Download annotations (~241MB)
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   ```

3. **Extract**:
   ```bash
   unzip train2017.zip
   unzip val2017.zip
   unzip annotations_trainval2017.zip
   
   # Clean up zip files (optional)
   rm *.zip
   ```

4. **Verify structure**:
   ```bash
   ls ~/datasets/coco/
   # Should show: train2017/ val2017/ annotations/
   
   ls ~/datasets/coco/train2017/ | wc -l
   # Should show: ~118,000 files
   ```

---

## ⚙️ Set Environment Variables

After downloading both datasets, set the environment variables:

### Temporary (Current Session Only)
```bash
export SKETCHY_ROOT=~/datasets/sketchy
export COCO_ROOT=~/datasets/coco
```

### Permanent (Add to Shell Profile)

**For Zsh (macOS default):**
```bash
echo 'export SKETCHY_ROOT=~/datasets/sketchy' >> ~/.zshrc
echo 'export COCO_ROOT=~/datasets/coco' >> ~/.zshrc
source ~/.zshrc
```

**For Bash:**
```bash
echo 'export SKETCHY_ROOT=~/datasets/sketchy' >> ~/.bashrc
echo 'export COCO_ROOT=~/datasets/coco' >> ~/.bashrc
source ~/.bashrc
```

### Verify Environment Variables
```bash
echo $SKETCHY_ROOT
# Should output: /Users/yourusername/datasets/sketchy

echo $COCO_ROOT
# Should output: /Users/yourusername/datasets/coco
```

---

## ✅ Validate Dataset Setup

Run this Python script to verify your datasets are correctly formatted:

```python
import os
from pathlib import Path

def validate_sketchy(root):
    """Validate Sketchy dataset structure."""
    root = Path(root)
    
    print("🔍 Validating Sketchy dataset...")
    
    # Check directories exist
    sketch_dir = root / "sketch" / "tx_000000000000"
    photo_dir = root / "photo" / "tx_000000000000"
    
    if not sketch_dir.exists():
        print(f"❌ Missing: {sketch_dir}")
        return False
    if not photo_dir.exists():
        print(f"❌ Missing: {photo_dir}")
        return False
    
    # Count categories
    sketch_cats = [d.name for d in sketch_dir.iterdir() if d.is_dir()]
    photo_cats = [d.name for d in photo_dir.iterdir() if d.is_dir()]
    
    print(f"✅ Found {len(sketch_cats)} sketch categories")
    print(f"✅ Found {len(photo_cats)} photo categories")
    
    # Count pairs in first category
    if sketch_cats:
        cat = sketch_cats[0]
        sketches = list((sketch_dir / cat).glob("*.png"))
        photos = list((photo_dir / cat).glob("*.jpg"))
        print(f"✅ Category '{cat}': {len(sketches)} sketches, {len(photos)} photos")
    
    return True

def validate_coco(root):
    """Validate MS COCO dataset structure."""
    root = Path(root)
    
    print("\n🔍 Validating MS COCO dataset...")
    
    # Check directories
    train_dir = root / "train2017"
    val_dir = root / "val2017"
    ann_dir = root / "annotations"
    
    if not train_dir.exists():
        print(f"❌ Missing: {train_dir}")
        return False
    if not val_dir.exists():
        print(f"❌ Missing: {val_dir}")
        return False
    if not ann_dir.exists():
        print(f"❌ Missing: {ann_dir}")
        return False
    
    # Count images
    train_images = list(train_dir.glob("*.jpg"))
    val_images = list(val_dir.glob("*.jpg"))
    
    print(f"✅ Found {len(train_images)} training images")
    print(f"✅ Found {len(val_images)} validation images")
    
    # Check annotations
    captions_train = ann_dir / "captions_train2017.json"
    instances_train = ann_dir / "instances_train2017.json"
    
    if captions_train.exists():
        print(f"✅ Found captions_train2017.json")
    else:
        print(f"❌ Missing: captions_train2017.json")
        
    if instances_train.exists():
        print(f"✅ Found instances_train2017.json")
    else:
        print(f"❌ Missing: instances_train2017.json")
    
    return True

# Run validation
sketchy_root = os.getenv("SKETCHY_ROOT")
coco_root = os.getenv("COCO_ROOT")

if sketchy_root:
    validate_sketchy(sketchy_root)
else:
    print("⚠️  SKETCHY_ROOT environment variable not set")

if coco_root:
    validate_coco(coco_root)
else:
    print("⚠️  COCO_ROOT environment variable not set")
```

Save this as `validate_datasets.py` and run:
```bash
python validate_datasets.py
```

---

## 📊 Dataset Statistics

### Sketchy Dataset
- **Categories**: 125 object classes
- **Total pairs**: ~75,000 sketch-photo pairs
- **Image size**: Variable (will be resized to 512×512 by dataloader)
- **Format**: Sketches (PNG, grayscale), Photos (JPG, RGB)
- **Use case**: Single-object structure-preserving generation

### MS COCO Dataset
- **Categories**: 80 object classes
- **Train images**: 118,287
- **Val images**: 5,000
- **Captions**: 5 captions per image
- **Format**: JPG, RGB
- **Use case**: Complex multi-object scenes, text-to-image generation

---

## 🎯 Quick Start After Setup

Once datasets are downloaded and environment variables are set:

### Test Dataset Loaders
```bash
cd /Users/satyamkumar/RAGAF-Diffusion/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Test Sketchy dataset loader
python -c "
from datasets.sketchy_dataset import SketchyDataset
import os

dataset = SketchyDataset(
    root_dir=os.getenv('SKETCHY_ROOT'),
    split='train',
    image_size=512
)
print(f'✅ Sketchy dataset loaded: {len(dataset)} samples')

# Load one sample
sample = dataset[0]
print(f'✅ Sample keys: {sample.keys()}')
print(f'   - Sketch shape: {sample[\"sketch\"].shape}')
print(f'   - Photo shape: {sample[\"photo\"].shape}')
print(f'   - Category: {sample[\"category\"]}')
print(f'   - Prompt: {sample[\"text_prompt\"]}')
"

# Test COCO dataset loader
python -c "
from datasets.coco_dataset import COCODataset
import os

dataset = COCODataset(
    root_dir=os.getenv('COCO_ROOT'),
    split='train',
    image_size=512
)
print(f'✅ COCO dataset loaded: {len(dataset)} samples')

# Load one sample
sample = dataset[0]
print(f'✅ Sample keys: {sample.keys()}')
print(f'   - Sketch shape: {sample[\"sketch\"].shape}')
print(f'   - Photo shape: {sample[\"photo\"].shape}')
print(f'   - Caption: {sample[\"text_prompt\"]}')
"
```

### Start Training
```bash
# Train on Sketchy dataset only
python train.py --dataset sketchy

# Train on COCO dataset only
python train.py --dataset coco

# Train on both datasets (default)
python train.py --dataset both
```

---

## 🚨 Troubleshooting

### Issue: "Dataset not found" error
**Solution**: 
1. Verify directory structure matches exactly as shown above
2. Check environment variables are set: `echo $SKETCHY_ROOT`
3. Use absolute paths, not relative paths

### Issue: "No matching photo found for sketch"
**Solution**: 
- Sketchy dataset requires exact name matching (n000001.png ↔ n000001.jpg)
- Verify both sketch/ and photo/ directories have the same categories
- Check file extensions (.png for sketches, .jpg for photos)

### Issue: "Missing annotations file"
**Solution**: 
- Download the full annotations_trainval2017.zip from COCO website
- Extract into the coco/ directory
- Verify `annotations/captions_train2017.json` exists

### Issue: Slow dataset loading
**Solution**: 
- Don't use `preload_graphs=True` (very memory intensive)
- Use SSD storage instead of HDD for faster I/O
- Consider reducing `max_num_regions` in RegionExtractor

---

## 💡 Tips for Cloud Training (RunPod, AWS, etc.)

### Upload Datasets to Cloud Storage
```bash
# Compress datasets for faster upload
tar -czf sketchy.tar.gz -C ~/datasets sketchy/
tar -czf coco.tar.gz -C ~/datasets coco/

# Upload to cloud (example: scp)
scp sketchy.tar.gz user@runpod-instance:/workspace/datasets/
scp coco.tar.gz user@runpod-instance:/workspace/datasets/

# On cloud instance: extract
tar -xzf sketchy.tar.gz -C /workspace/datasets/
tar -xzf coco.tar.gz -C /workspace/datasets/
```

### Use Persistent Storage
- **RunPod**: Use Network Volumes to persist datasets across pod sessions
- **AWS**: Use EBS volumes or S3 buckets
- **Lambda Labs**: Datasets persist in `/home` directory

---

## ✅ Checklist

- [ ] Sketchy dataset downloaded (~10GB)
- [ ] Sketchy directory structure verified
- [ ] MS COCO dataset downloaded (~25GB)  
- [ ] COCO directory structure verified
- [ ] Environment variables set (`SKETCHY_ROOT`, `COCO_ROOT`)
- [ ] Dataset validation script passed
- [ ] Dataset loaders tested successfully
- [ ] Ready to start training!

---

**Questions or issues?** Check the troubleshooting section or review the dataset loader code in `datasets/sketchy_dataset.py` and `datasets/coco_dataset.py`.
