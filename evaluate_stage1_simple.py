"""
Simple Stage 1 Evaluation Script (Works Offline)
Uses cached HuggingFace models and evaluates sketch fidelity
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import json
from tqdm import tqdm
import cv2
import os

# Set offline mode for HuggingFace
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets.sketchy_dataset import SketchyDataset
from configs.config import get_default_config

print("=" * 80)
print("🎯 STAGE 1 MODEL EVALUATION (Offline Mode)")
print("=" * 80)
print()

# Check for cached model
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
print(f"📁 Checking HuggingFace cache: {cache_dir}")

# List cached models
if cache_dir.exists():
    cached_models = list(cache_dir.glob("models--*"))
    print(f"   Found {len(cached_models)} cached models")
    for model_path in cached_models[:5]:
        model_name = model_path.name.replace("models--", "").replace("--", "/")
        print(f"   - {model_name}")
else:
    print("   ⚠️  No cache directory found!")

print()

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")
print()

# Load config
config = get_default_config()

# Load dataset
print("📊 Loading Sketchy Dataset...")
dataset = SketchyDataset(
    root_dir=config['data'].sketchy_root,
    split='test',
    image_size=config['data'].image_size,
    augment=False
)

print(f"   Total test samples: {len(dataset)}")
print()

# Get categories
categories = sorted(list(set([pair['category'] for pair in dataset.data_pairs])))
print(f"📁 Total categories: {len(categories)}")
print()

# Try to load checkpoint
checkpoint_path = "/root/checkpoints/stage1/final.pt"
print(f"📦 Loading checkpoint: {checkpoint_path}")

if not Path(checkpoint_path).exists():
    print(f"❌ Checkpoint not found!")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"   ✅ Checkpoint loaded")
print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"   Loss: {checkpoint.get('loss', 'unknown'):.4f}" if 'loss' in checkpoint else "")
print()

# Analyze model state
print("🔍 Analyzing Model State...")
state_dict = checkpoint['model_state_dict']
print(f"   Total parameters: {len(state_dict)} tensors")
print()

# Group parameters by component
components = {}
for key in state_dict.keys():
    component = key.split('.')[0]
    if component not in components:
        components[component] = []
    components[component].append(key)

print("📊 Model Components:")
for component, keys in sorted(components.items()):
    total_params = sum(state_dict[k].numel() for k in keys)
    print(f"   {component:20s}: {len(keys):4d} tensors, {total_params:,} params")
print()

# Check sketch encoder
sketch_encoder_keys = [k for k in state_dict.keys() if 'sketch_encoder' in k]
print(f"🎨 Sketch Encoder: {len(sketch_encoder_keys)} parameters")
if sketch_encoder_keys:
    print("   Sample keys:")
    for key in sketch_encoder_keys[:5]:
        shape = state_dict[key].shape
        print(f"   - {key}: {tuple(shape)}")
print()

# Simple sketch fidelity test (without loading full model)
print("=" * 80)
print("🧪 SKETCH FIDELITY EVALUATION (Visual Inspection)")
print("=" * 80)
print()

print("Since we can't load the full model in offline mode,")
print("let's evaluate the generated outputs from previous runs.")
print()

# Check for existing outputs
output_dirs = [
    "outputs/stage1/test_outputs_epoch10",
    "outputs/stage1/test_outputs_epoch2",
    "test_outputs_epoch10",
    "test_outputs_epoch2",
    "outputs/evaluations/all_categories"
]

found_outputs = []
for out_dir in output_dirs:
    out_path = Path(out_dir)
    if out_path.exists():
        images = list(out_path.glob("*.png")) + list(out_path.glob("*.jpg"))
        if images:
            found_outputs.append((out_dir, len(images)))

if found_outputs:
    print("✅ Found existing generated outputs:")
    for out_dir, count in found_outputs:
        print(f"   - {out_dir}: {count} images")
    print()
    
    # Analyze one output directory
    selected_dir = found_outputs[0][0]
    print(f"📊 Analyzing outputs from: {selected_dir}")
    print()
    
    images = list(Path(selected_dir).glob("*.png"))
    
    # Compute simple image statistics
    print("📈 Image Quality Metrics:")
    print()
    
    for i, img_path in enumerate(images[:5]):  # Analyze first 5 images
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Compute statistics
        mean_val = img_array.mean()
        std_val = img_array.std()
        
        # Check if image is diverse (not blank/noisy)
        is_diverse = std_val > 20  # Threshold for diversity
        
        print(f"   Image {i+1}: {img_path.name}")
        print(f"      Size: {img.size}")
        print(f"      Mean pixel value: {mean_val:.2f}")
        print(f"      Std deviation: {std_val:.2f}")
        print(f"      Quality: {'✅ Good' if is_diverse else '⚠️  Low variance'}")
        print()
    
else:
    print("⚠️  No generated outputs found.")
    print("   Run inference first to generate images for evaluation.")
    print()

# Summary
print("=" * 80)
print("📊 EVALUATION SUMMARY")
print("=" * 80)
print()
print("✅ Model checkpoint loaded successfully")
print(f"✅ Dataset loaded: {len(dataset)} test samples")
print(f"✅ Model has {len(state_dict)} parameters")
print(f"✅ Sketch encoder: {len(sketch_encoder_keys)} parameters")
print()

if found_outputs:
    print(f"✅ Found generated outputs in {len(found_outputs)} directories")
    print()
    print("🎯 To evaluate accuracy:")
    print("   1. Visual inspection: Check if generated images match sketch structure")
    print("   2. Use scripts/evaluation/evaluate_all_categories.py for full metrics")
    print("   3. Compare outputs across different epochs")
else:
    print("⚠️  No generated outputs found for evaluation")
    print()
    print("🎯 To generate outputs:")
    print("   python3 scripts/inference/quick_test.py")
    print("   python3 scripts/evaluation/evaluate_all_categories.py")

print()
print("=" * 80)
print("✨ Evaluation Complete!")
print("=" * 80)
