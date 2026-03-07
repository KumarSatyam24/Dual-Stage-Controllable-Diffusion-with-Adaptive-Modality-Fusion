#!/usr/bin/env python3
"""
Test Stage 1 Epoch 2 checkpoint with custom sketches.
Usage: python3 test_epoch2_custom.py --sketch <path> --prompt <text> [--output <path>]
"""

import torch
import os
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from configs.config import get_default_config

# Global cache for model
_model_cache = None
_pipeline_cache = None

def load_model_once(checkpoint_path, device):
    """Load model once and cache it."""
    global _model_cache, _pipeline_cache
    
    if _model_cache is not None:
        print("✅ Using cached model")
        return _model_cache, _pipeline_cache
    
    print("📦 Loading Stage 1 model...")
    config_dict = get_default_config()
    model_config = config_dict['model']
    
    model = Stage1SketchGuidedDiffusion(
        pretrained_model_name=model_config.pretrained_model_name,
        sketch_encoder_channels=model_config.sketch_encoder_channels,
        freeze_base_unet=model_config.freeze_stage1_unet,
        use_lora=model_config.use_lora,
        lora_rank=model_config.lora_rank
    )
    
    # Load checkpoint
    print(f"⚙️  Loading checkpoint...")
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
        num_inference_steps=20,  # Quick test
        guidance_scale=7.5,
        device=device
    )
    
    _model_cache = model
    _pipeline_cache = pipeline
    
    return model, pipeline

def test_sketch(sketch_path, prompt, output_path, checkpoint_path, device):
    """Test with a custom sketch."""
    print(f"\n🎨 Testing with:")
    print(f"   Sketch: {sketch_path}")
    print(f"   Prompt: {prompt}")
    
    # Load model
    model, pipeline = load_model_once(checkpoint_path, device)
    
    # Load and preprocess sketch
    print("📥 Loading sketch...")
    sketch = Image.open(sketch_path).convert('L')  # Convert to grayscale (1 channel)
    
    # Resize to 256x256
    sketch = sketch.resize((256, 256), Image.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)  # Shape: (1, 1, 256, 256)
    
    # Run inference
    print("🔮 Generating image...")
    with torch.no_grad():
        output_tensor = pipeline.generate(
            text_prompt=prompt,
            sketch=sketch_tensor,
            height=256,
            width=256,
        )
    
    # Convert tensor to PIL Image
    # output_tensor is (1, 3, H, W) in range [0, 1]
    output_numpy = output_tensor.squeeze(0).cpu().numpy()  # (3, H, W)
    output_numpy = (output_numpy * 255).astype('uint8')  # Convert to [0, 255]
    output_numpy = output_numpy.transpose(1, 2, 0)  # (H, W, 3)
    output_image = Image.fromarray(output_numpy)
    
    # Save result
    output_image.save(output_path)
    print(f"✅ Saved to: {output_path}")
    
    return output_path

def download_checkpoint():
    """Download epoch 2 checkpoint from HF Hub."""
    cache_path = "/root/.cache/huggingface/models--DrRORAL--ragaf-diffusion-checkpoints/snapshots"
    
    # Check if already downloaded
    if os.path.exists(cache_path):
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                if file == "epoch_2.pt" and "stage1" in root:
                    checkpoint_path = os.path.join(root, file)
                    print(f"✅ Using cached checkpoint: {checkpoint_path}")
                    return checkpoint_path
    
    print("📥 Downloading epoch 2 checkpoint from HuggingFace Hub...")
    repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
    filename = "stage1/epoch_2.pt"
    
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir="/root/.cache/huggingface",
    )
    print(f"✅ Downloaded to: {checkpoint_path}")
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser(description="Test epoch 2 with custom sketch")
    parser.add_argument("--sketch", type=str, help="Path to sketch image")
    parser.add_argument("--prompt", type=str, help="Text prompt")
    parser.add_argument("--output", type=str, help="Output path (optional)")
    parser.add_argument("--test-all", action="store_true", help="Test with multiple sketches")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Download checkpoint
    checkpoint_path = download_checkpoint()
    
    # Create output directory
    os.makedirs("test_outputs_epoch2", exist_ok=True)
    
    if args.test_all:
        # Test with multiple examples
        test_cases = [
            ("input_sketch.png", "an airplane flying in the blue sky", "test_outputs_epoch2/airplane_output.png"),
            ("input_sketch.png", "a fighter jet in combat", "test_outputs_epoch2/fighter_jet_output.png"),
            ("input_sketch.png", "a commercial airplane landing", "test_outputs_epoch2/commercial_plane_output.png"),
        ]
        
        print("\n" + "="*60)
        print("Testing Multiple Variations")
        print("="*60)
        
        for sketch, prompt, output in test_cases:
            if os.path.exists(sketch):
                try:
                    test_sketch(sketch, prompt, output, checkpoint_path, device)
                    print(f"✅ Completed: {output}\n")
                except Exception as e:
                    print(f"❌ Failed: {e}\n")
            else:
                print(f"⚠️  Sketch not found: {sketch}\n")
    
    elif args.sketch and args.prompt:
        # Test with custom sketch and prompt
        output = args.output or "test_outputs_epoch2/custom_output.png"
        
        print("\n" + "="*60)
        print("Custom Test")
        print("="*60)
        
        try:
            test_sketch(args.sketch, args.prompt, output, checkpoint_path, device)
            print("\n" + "="*60)
            print("✅ TEST SUCCESSFUL!")
            print("="*60)
        except Exception as e:
            print("\n" + "="*60)
            print("❌ TEST FAILED!")
            print("="*60)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("Usage:")
        print("  Single test:   python3 test_epoch2_custom.py --sketch path.png --prompt 'text'")
        print("  Multiple test: python3 test_epoch2_custom.py --test-all")
        print("\nExamples:")
        print("  python3 test_epoch2_custom.py --sketch input_sketch.png --prompt 'a red airplane'")
        print("  python3 test_epoch2_custom.py --test-all")

if __name__ == "__main__":
    main()
