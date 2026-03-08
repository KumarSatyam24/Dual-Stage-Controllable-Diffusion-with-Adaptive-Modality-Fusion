#!/usr/bin/env python3
"""
Quick verification script to check if your Sketchy dataset is ready for training.
Run this anytime to verify your setup.

Usage:
    python verify_dataset.py
"""

import os
import sys
from pathlib import Path

def main():
    print("="*70)
    print("🔍 RAGAF-Diffusion Dataset Verification")
    print("="*70)
    
    # Check environment variable
    sketchy_root = os.getenv("SKETCHY_ROOT")
    
    if not sketchy_root:
        print("\n❌ SKETCHY_ROOT environment variable not set!")
        print("\nPlease run:")
        print("  export SKETCHY_ROOT=/path/to/sketchy")
        print("  # Or add to ~/.zshrc for permanent setup")
        return False
    
    print(f"\n✅ SKETCHY_ROOT: {sketchy_root}")
    
    # Check if path exists
    if not Path(sketchy_root).exists():
        print(f"\n❌ Path does not exist: {sketchy_root}")
        return False
    
    print(f"✅ Path exists")
    
    # Try to load dataset
    try:
        print("\n📦 Testing dataset loader...")
        from datasets.sketchy_dataset import SketchyDataset
        
        # Load small subset for quick test
        dataset = SketchyDataset(
            root_dir=sketchy_root,
            split='train',
            categories=['airplane', 'apple', 'bear'],
            image_size=512,
            augment=False,
            preload_graphs=False
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples (subset)")
        
        # Load one sample
        sample = dataset[0]
        print(f"✅ Sample loaded successfully")
        print(f"   - Sketch: {sample['sketch'].shape}")
        print(f"   - Photo: {sample['photo'].shape}")
        print(f"   - Category: {sample['category']}")
        print(f"   - Prompt: {sample['text_prompt']}")
        
        # Load full dataset counts
        print("\n📊 Loading full dataset statistics...")
        dataset_full = SketchyDataset(
            root_dir=sketchy_root,
            split='train',
            categories=None,  # All categories
            image_size=512,
            augment=False,
            preload_graphs=False
        )
        
        dataset_val = SketchyDataset(
            root_dir=sketchy_root,
            split='val',
            categories=None,
            image_size=512,
            augment=False,
            preload_graphs=False
        )
        
        dataset_test = SketchyDataset(
            root_dir=sketchy_root,
            split='test',
            categories=None,
            image_size=512,
            augment=False,
            preload_graphs=False
        )
        
        total = len(dataset_full) + len(dataset_val) + len(dataset_test)
        
        print(f"\n📈 Full Dataset Statistics:")
        print(f"   Train:      {len(dataset_full):,} samples")
        print(f"   Validation: {len(dataset_val):,} samples")
        print(f"   Test:       {len(dataset_test):,} samples")
        print(f"   Total:      {total:,} samples")
        
        print("\n" + "="*70)
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
        print("="*70)
        print("\nYou can now run:")
        print("  python train.py --dataset sketchy")
        print("\nFor GPU training (recommended), use RunPod or cloud GPU")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure you're in the project directory and dependencies are installed")
        return False
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nCheck that your dataset structure matches the expected format")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
