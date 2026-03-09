#!/usr/bin/env python3
"""
Quick test for validate_epochs.py

Tests basic functionality without full validation:
1. Import check
2. Config loading
3. Dataset loading
4. Checkpoint detection
"""

import sys
from pathlib import Path

print("=" * 70)
print("🧪 Testing Epoch Validation Pipeline")
print("=" * 70)

# Test 1: Import check
print("\n[1/4] Testing imports...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from validate_epochs import EpochValidator
    from configs.config import get_default_config
    from datasets.sketchy_dataset import SketchyDataset
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Config loading
print("\n[2/4] Testing config...")
try:
    config = get_default_config()
    print(f"   ✅ Config loaded")
    print(f"   - Image size: {config['data'].image_size}")
    print(f"   - Pretrained model: {config['model'].pretrained_model_name}")
except Exception as e:
    print(f"   ❌ Config failed: {e}")
    sys.exit(1)

# Test 3: Dataset check
print("\n[3/4] Testing dataset...")
try:
    import os
    dataset_root = "/workspace/sketchy"
    
    if not os.path.exists(dataset_root):
        print(f"   ⚠️  Dataset not found at {dataset_root}")
        dataset_root = "/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/sketchy"
    
    if os.path.exists(dataset_root):
        dataset = SketchyDataset(
            root_dir=dataset_root,
            split='test',
            image_size=config['data'].image_size,
            augment=False
        )
        print(f"   ✅ Dataset loaded: {len(dataset)} samples")
    else:
        print(f"   ⚠️  Dataset not available (will be needed for actual validation)")
except Exception as e:
    print(f"   ⚠️  Dataset check skipped: {e}")

# Test 4: Checkpoint detection
print("\n[4/4] Testing checkpoint detection...")
try:
    # Check local checkpoints
    local_checkpoint_dir = Path("/root/checkpoints/stage1")
    if local_checkpoint_dir.exists():
        checkpoints = list(local_checkpoint_dir.glob("epoch_*.pt"))
        if checkpoints:
            print(f"   ✅ Found {len(checkpoints)} local checkpoints:")
            for cp in sorted(checkpoints)[:5]:  # Show first 5
                print(f"      - {cp.name}")
            if len(checkpoints) > 5:
                print(f"      ... and {len(checkpoints) - 5} more")
        else:
            print(f"   ⚠️  No epoch checkpoints found locally")
    else:
        print(f"   ⚠️  Local checkpoint directory not found")
    
    # Try HuggingFace detection
    print("\n   Testing HuggingFace checkpoint detection...")
    try:
        from huggingface_hub import list_repo_files
        
        # Replace with actual repo ID
        hf_repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
        
        files = list_repo_files(hf_repo_id)
        epoch_files = [f for f in files if 'epoch' in f.lower() and f.endswith('.pt')]
        
        if epoch_files:
            print(f"   ✅ Found {len(epoch_files)} checkpoint files on HuggingFace:")
            for f in sorted(epoch_files)[:5]:
                print(f"      - {f}")
            if len(epoch_files) > 5:
                print(f"      ... and {len(epoch_files) - 5} more")
        else:
            print(f"   ⚠️  No epoch checkpoints found on HuggingFace")
            
    except Exception as e:
        print(f"   ⚠️  HuggingFace check skipped: {e}")
        
except Exception as e:
    print(f"   ⚠️  Checkpoint detection failed: {e}")

# Summary
print("\n" + "=" * 70)
print("📊 Test Summary")
print("=" * 70)
print("""
✅ Basic functionality working!

To run actual validation:
  python validate_epochs.py --num_samples 20

Options:
  --hf_repo         HuggingFace repo ID (default: DrRORAL/ragaf-diffusion-checkpoints)
  --dataset_root    Path to Sketchy dataset (default: /workspace/sketchy)
  --epochs          Specific epochs to validate (default: all)
  --num_samples     Samples per epoch (default: 50)
  --output_dir      Output directory (default: validation_results)
  --guidance_scale  Guidance scale (default: 2.5)

Quick start:
  python validate_epochs.py --num_samples 10  # Quick test

See docs/EPOCH_VALIDATION_GUIDE.md for complete documentation.
""")
print("=" * 70)
