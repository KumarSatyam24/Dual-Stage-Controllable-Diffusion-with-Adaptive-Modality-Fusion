#!/usr/bin/env python3
"""
Quick validation test - simplified version for debugging
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from datasets.sketchy_dataset import SketchyDataset
from configs.config import get_default_config

print("=" * 80)
print("Quick Validation Test")
print("=" * 80)

# Load config
print("\n1. Loading config...")
config = get_default_config()
print(f"   Dataset root: {config['data'].sketchy_root}")
print(f"   Image size: {config['data'].image_size}")

# Load dataset
print("\n2. Loading dataset...")
dataset = SketchyDataset(
    root_dir=config['data'].sketchy_root,
    split='test',
    image_size=config['data'].image_size,
    augment=False
)
print(f"   Total samples: {len(dataset)}")

# Test loading one sample
print("\n3. Loading one sample...")
data = dataset[0]
print(f"   Sketch shape: {data['sketch'].shape}")
print(f"   Photo shape: {data['photo'].shape}")
print(f"   Prompt: {data['text_prompt']}")
print(f"   Category: {data['category']}")

# Load model
print("\n4. Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

model_config = config['model']
print(f"   Model name: {model_config.pretrained_model_name}")
print(f"   Sketch encoder channels: {model_config.sketch_encoder_channels}")

try:
    model = Stage1SketchGuidedDiffusion(
        pretrained_model_name=model_config.pretrained_model_name,
        sketch_encoder_channels=model_config.sketch_encoder_channels,
        freeze_base_unet=model_config.freeze_stage1_unet,
        use_lora=model_config.use_lora,
        lora_rank=model_config.lora_rank
    ).to(device)
    print("   ✅ Model created successfully")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)

# Load checkpoint
print("\n5. Loading checkpoint...")
checkpoint_path = "/root/checkpoints/stage1/final.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✅ Checkpoint loaded")
except Exception as e:
    print(f"   ❌ Checkpoint loading failed: {e}")
    sys.exit(1)

# Create pipeline
print("\n6. Creating pipeline...")
try:
    pipeline = Stage1DiffusionPipeline(
        model=model,
        num_inference_steps=20,  # Reduced for testing
        guidance_scale=2.5,
        device=device
    )
    print("   ✅ Pipeline created")
except Exception as e:
    print(f"   ❌ Pipeline creation failed: {e}")
    sys.exit(1)

# Test generation
print("\n7. Testing generation...")
sketch = data['sketch'].unsqueeze(0)  # Add batch dimension
prompt = data['text_prompt']
img_size = config['data'].image_size  # Use configured image size

print(f"   Sketch input shape: {sketch.shape}")
print(f"   Prompt: {prompt}")
print(f"   Image size: {img_size}")

try:
    with torch.no_grad():
        generated = pipeline.generate(
            sketch=sketch,
            text_prompt=prompt,
            height=img_size,  # Use dataset image size
            width=img_size,   # Use dataset image size
            seed=42
        )
    print(f"   ✅ Generation successful!")
    print(f"   Generated shape: {generated.shape}")
except Exception as e:
    print(f"   ❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ All tests passed!")
print("=" * 80)
