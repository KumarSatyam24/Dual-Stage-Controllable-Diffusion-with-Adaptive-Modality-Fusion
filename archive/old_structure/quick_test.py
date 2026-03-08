#!/usr/bin/env python3
"""
Quick sketch tester - Test any sketch with any prompt.
This makes it easy to test different sketches interactively.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

from test_epoch2_custom import test_sketch, download_checkpoint
import torch

def quick_test(sketch_path, prompt, output_name=None):
    """Quick test with a sketch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download checkpoint (uses cache if available)
    checkpoint_path = download_checkpoint()
    
    # Set output path
    if output_name is None:
        base_name = os.path.splitext(os.path.basename(sketch_path))[0]
        output_name = f"{base_name}_epoch2.png"
    
    output_path = f"test_outputs_epoch2/{output_name}"
    
    # Test
    print("\n" + "="*70)
    print("🎨 EPOCH 2 SKETCH TEST")
    print("="*70)
    
    result = test_sketch(sketch_path, prompt, output_path, checkpoint_path, device)
    
    print("\n" + "="*70)
    print(f"✅ Output saved: {result}")
    print("="*70)
    print(f"\n📂 Copy to workspace: cp {result} /workspace/")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 quick_test.py <sketch_path> <prompt> [output_name]")
        print("\nExamples:")
        print('  python3 quick_test.py input_sketch.png "a red fighter jet"')
        print('  python3 quick_test.py my_sketch.png "a beautiful sunset" my_output.png')
        sys.exit(1)
    
    sketch = sys.argv[1]
    prompt = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    
    quick_test(sketch, prompt, output)
