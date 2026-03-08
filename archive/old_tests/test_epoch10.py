#!/usr/bin/env python3
"""
Test Stage 1 Epoch 10 (final) checkpoint from HuggingFace Hub.
"""

import sys
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

import torch
import os
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms
import numpy as np

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from configs.config import get_default_config

def download_checkpoint_epoch10():
    """Download epoch 10 checkpoint from HF Hub."""
    cache_path = "/root/.cache/huggingface/models--DrRORAL--ragaf-diffusion-checkpoints/snapshots"
    
    # Check if already downloaded
    if os.path.exists(cache_path):
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                if file == "epoch_10.pt" and "stage1" in root:
                    checkpoint_path = os.path.join(root, file)
                    print(f"✅ Using cached checkpoint: {checkpoint_path}")
                    return checkpoint_path
    
    print("📥 Downloading epoch 10 checkpoint from HuggingFace Hub...")
    repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
    filename = "stage1/epoch_10.pt"
    
    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir="/root/.cache/huggingface",
        )
        print(f"✅ Downloaded to: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        # Try final.pt as fallback
        print("📥 Trying final.pt instead...")
        filename = "stage1/final.pt"
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir="/root/.cache/huggingface",
        )
        print(f"✅ Downloaded to: {checkpoint_path}")
        return checkpoint_path

def test_epoch10(sketch_path, prompt, output_path):
    """Test epoch 10 with a sketch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Download checkpoint
    checkpoint_path = download_checkpoint_epoch10()
    
    # Load config
    config_dict = get_default_config()
    model_config = config_dict['model']
    
    # Initialize model
    print("📦 Loading Stage 1 model...")
    model = Stage1SketchGuidedDiffusion(
        pretrained_model_name=model_config.pretrained_model_name,
        sketch_encoder_channels=model_config.sketch_encoder_channels,
        freeze_base_unet=model_config.freeze_stage1_unet,
        use_lora=model_config.use_lora,
        lora_rank=model_config.lora_rank
    )
    
    # Load checkpoint
    print(f"⚙️  Loading checkpoint from epoch 10...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded checkpoint")
    
    model = model.to(device)
    model.eval()
    
    # Create pipeline with inference settings
    pipeline = Stage1DiffusionPipeline(
        model, 
        num_inference_steps=50,  # Higher quality for final model
        guidance_scale=7.5,
        device=device
    )
    
    # Load and preprocess sketch
    print(f"📥 Loading sketch: {sketch_path}")
    sketch = Image.open(sketch_path).convert('L')  # Grayscale
    sketch = sketch.resize((512, 512), Image.LANCZOS)  # Full resolution
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
    # Run inference
    print(f"🔮 Generating image with prompt: '{prompt}'")
    with torch.no_grad():
        output_tensor = pipeline.generate(
            text_prompt=prompt,
            sketch=sketch_tensor,
            height=512,
            width=512,
        )
    
    # Convert tensor to PIL Image
    output_numpy = output_tensor.squeeze(0).cpu().numpy()
    output_numpy = (output_numpy * 255).astype('uint8')
    output_numpy = output_numpy.transpose(1, 2, 0)
    output_image = Image.fromarray(output_numpy)
    
    # Save result
    output_image.save(output_path)
    print(f"✅ Saved to: {output_path}")
    
    return output_path

def main():
    print("="*70)
    print("🎯 TESTING EPOCH 10 (FINAL) CHECKPOINT")
    print("="*70)
    
    # Base directory
    base_dir = "/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
    
    # Create output directory
    os.makedirs(f"{base_dir}/test_outputs_epoch10", exist_ok=True)
    
    # Test cases with absolute paths
    sketch_path = f"{base_dir}/input_sketch.png"
    test_cases = [
        (sketch_path, "an airplane flying in the blue sky", "airplane_basic.png"),
        (sketch_path, "a military fighter jet with camouflage paint", "fighter_jet.png"),
        (sketch_path, "a commercial passenger airplane", "passenger_plane.png"),
        (sketch_path, "a vintage propeller airplane from the 1940s", "vintage_plane.png"),
    ]
    
    results = []
    
    for sketch, prompt, output_name in test_cases:
        if not os.path.exists(sketch):
            print(f"⚠️  Sketch not found: {sketch}")
            continue
        
        output_path = f"{base_dir}/test_outputs_epoch10/{output_name}"
        
        try:
            print(f"\n{'='*70}")
            result = test_epoch10(sketch, prompt, output_path)
            results.append(result)
            print(f"✅ Test complete: {output_name}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ EPOCH 10 TESTING COMPLETE!")
    print("="*70)
    print(f"\n📊 Generated {len(results)} images:")
    for r in results:
        print(f"   - {r}")
    
    print(f"\n📂 All outputs in: {base_dir}/test_outputs_epoch10/")
    print(f"📂 Copy to workspace: cp {base_dir}/test_outputs_epoch10/*.png /workspace/")
    
    print("\n🔍 Key Questions to Check:")
    print("   1. Does output follow sketch structure? ✅")
    print("   2. Is quality better than epoch 2? 📈")
    print("   3. Are different prompts producing different results? 🎨")
    print("   4. Is it photorealistic and detailed? 🖼️")

if __name__ == "__main__":
    main()
