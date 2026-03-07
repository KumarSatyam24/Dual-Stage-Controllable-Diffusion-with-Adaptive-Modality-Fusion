#!/usr/bin/env python3
"""
Comprehensive test of the final checkpoint (epoch 10).
Tests with different prompts and styles on available sketches.
"""

import torch
import os
import sys
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from torchvision import transforms

sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from configs.config import get_default_config

def download_checkpoint(epoch="final"):
    """Download checkpoint from HF Hub."""
    print(f"📥 Downloading {epoch} checkpoint from HuggingFace Hub...")
    
    # Check cache first
    cache_path = "/root/.cache/huggingface/models--DrRORAL--ragaf-diffusion-checkpoints/snapshots"
    if os.path.exists(cache_path):
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                if file == f"{epoch}.pt" and "stage1" in root:
                    checkpoint_path = os.path.join(root, file)
                    print(f"✅ Using cached checkpoint: {checkpoint_path}")
                    return checkpoint_path
    
    repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
    filename = f"stage1/{epoch}.pt"
    
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
        return None

def load_model(checkpoint_path, device):
    """Load the model from checkpoint."""
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
    
    # Create pipeline
    pipeline = Stage1DiffusionPipeline(
        model,
        num_inference_steps=50,  # High quality for final test
        guidance_scale=7.5,
        device=device
    )
    
    return model, pipeline

def load_sketch(sketch_path, device):
    """Load and preprocess sketch."""
    sketch = Image.open(sketch_path).convert('L')  # Grayscale
    sketch = sketch.resize((256, 256), Image.LANCZOS)
    
    transform = transforms.Compose([transforms.ToTensor()])
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
    return sketch_tensor, sketch

def generate_image(pipeline, sketch_tensor, prompt, device):
    """Generate image from sketch and prompt."""
    with torch.no_grad():
        output_tensor = pipeline.generate(
            text_prompt=prompt,
            sketch=sketch_tensor,
            height=256,
            width=256,
        )
    
    # Convert to PIL
    output_numpy = output_tensor.squeeze(0).cpu().numpy()
    output_numpy = (output_numpy * 255).astype('uint8')
    output_numpy = output_numpy.transpose(1, 2, 0)
    output_image = Image.fromarray(output_numpy)
    
    return output_image

def create_comparison_grid(images, labels, output_path):
    """Create a grid of images with labels."""
    from PIL import ImageDraw, ImageFont
    
    n_images = len(images)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    img_size = 256
    padding = 20
    label_height = 40
    
    grid_width = cols * img_size + (cols + 1) * padding
    grid_height = rows * (img_size + label_height) + (rows + 1) * padding
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (img_size + padding)
        y = padding + row * (img_size + label_height + padding)
        
        # Paste image
        grid.paste(img, (x, y + label_height))
        
        # Draw label
        draw.text((x + img_size // 2, y + label_height // 2), label, 
                 fill='black', font=font, anchor='mm')
    
    grid.save(output_path)
    print(f"✅ Saved comparison grid: {output_path}")

def main():
    print("=" * 70)
    print("COMPREHENSIVE FINAL CHECKPOINT TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Download checkpoint
    checkpoint_path = download_checkpoint("final")
    if not checkpoint_path:
        print("❌ Failed to download checkpoint")
        return
    
    # Load model
    model, pipeline = load_model(checkpoint_path, device)
    
    # Create output directory
    output_dir = "/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/final_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test cases with the airplane sketch
    sketch_path = "/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/input_sketch.png"
    
    if not os.path.exists(sketch_path):
        print(f"❌ Sketch not found: {sketch_path}")
        return
    
    print(f"\n📥 Loading sketch: {sketch_path}")
    sketch_tensor, sketch_pil = load_sketch(sketch_path, device)
    
    # Define test prompts
    test_prompts = [
        # Different airplane types
        ("fighter_jet", "a military fighter jet in flight"),
        ("passenger_plane", "a large commercial passenger airplane"),
        ("vintage_plane", "a vintage propeller airplane from the 1940s"),
        
        # Different styles
        ("photorealistic", "a photorealistic airplane flying in the sky"),
        ("cartoon", "a cartoon style airplane illustration"),
        ("watercolor", "an airplane painted in watercolor style"),
        
        # Different contexts
        ("sunset", "an airplane flying during a beautiful sunset"),
        ("clouds", "an airplane flying through puffy white clouds"),
        ("stormy", "an airplane flying through dramatic storm clouds"),
        
        # Different colors/attributes
        ("red_plane", "a bright red airplane"),
        ("blue_plane", "a blue and white airplane"),
        ("stealth", "a sleek black stealth aircraft"),
    ]
    
    print(f"\n🎨 Generating {len(test_prompts)} variations...")
    print("=" * 70)
    
    images = [sketch_pil.convert('RGB')]
    labels = ["Input Sketch"]
    
    for idx, (name, prompt) in enumerate(test_prompts, 1):
        print(f"\n[{idx}/{len(test_prompts)}] {name}")
        print(f"   Prompt: {prompt}")
        
        # Generate
        output_image = generate_image(pipeline, sketch_tensor, prompt, device)
        
        # Save individual
        output_path = f"{output_dir}/{name}.png"
        output_image.save(output_path)
        print(f"   ✅ Saved: {output_path}")
        
        images.append(output_image)
        labels.append(name.replace('_', ' ').title())
    
    # Create comparison grid
    print(f"\n📊 Creating comparison grid...")
    grid_path = f"{output_dir}/comparison_grid.png"
    create_comparison_grid(images, labels, grid_path)
    
    # Copy to workspace
    print(f"\n📂 Copying results to workspace...")
    os.system(f"cp -r {output_dir} /workspace/")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to:")
    print(f"   - {output_dir}/")
    print(f"   - /workspace/final_test_results/")
    print(f"\n🖼️  Key files:")
    print(f"   - comparison_grid.png - All results in one image")
    print(f"   - Individual images: fighter_jet.png, passenger_plane.png, etc.")
    print(f"\n🔍 Review the outputs to verify:")
    print(f"   1. All outputs follow the airplane sketch structure ✈️")
    print(f"   2. Different prompts produce different variations 🎨")
    print(f"   3. Quality is high (10 epochs of training) 📈")

if __name__ == "__main__":
    main()
