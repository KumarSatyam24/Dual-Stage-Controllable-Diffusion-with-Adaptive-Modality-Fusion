#!/usr/bin/env python3
"""
Test different guidance scales to find optimal value for Stage 1.
Uses random sketches from dataset.
"""

import torch
import os
import sys
import glob
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms

sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from configs.config import get_default_config
from huggingface_hub import hf_hub_download

def load_model(checkpoint_path, device):
    """Load model once."""
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded checkpoint")
    
    model = model.to(device)
    model.eval()
    
    return model

def get_random_sketch(category):
    """Get a random sketch from a category."""
    sketch_dir = f"/workspace/sketchy/sketch/tx_000000000000/{category}"
    if not os.path.exists(sketch_dir):
        return None
    
    sketches = glob.glob(f"{sketch_dir}/*.png")
    if not sketches:
        return None
    
    return random.choice(sketches)

def generate_with_scale(model, sketch_path, prompt, guidance_scale, device):
    """Generate image with specific guidance scale."""
    # Load sketch
    sketch = Image.open(sketch_path).convert('L')
    sketch = sketch.resize((256, 256), Image.LANCZOS)
    
    transform = transforms.ToTensor()
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
    # Create pipeline with specific guidance scale
    pipeline = Stage1DiffusionPipeline(
        model,
        num_inference_steps=30,
        guidance_scale=guidance_scale,
        device=device
    )
    
    # Generate
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
    
    return sketch, output_image

def create_comparison_grid(results, category, output_dir):
    """
    Create comparison grid showing results at different guidance scales.
    Layout: 3 rows × 6 columns
    Row 1: Sketch (repeated)
    Row 2: Generated images at scales [1.5, 2.0, 2.5, 3.0, 5.0, 7.5]
    Row 3: Scale labels
    """
    img_size = 256
    cols = 6
    margin = 10
    text_height = 40
    
    # Calculate grid dimensions
    grid_width = cols * img_size + (cols + 1) * margin
    grid_height = 2 * img_size + 3 * margin + 2 * text_height
    
    # Create white canvas
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Add title
    title = f"{category.upper()} - Guidance Scale Comparison"
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((grid_width - title_width) // 2, margin), title, fill='black', font=font)
    
    # Add results
    sketch_img = results[0][0]  # All have same sketch
    
    for idx, (sketch, gen_img, scale) in enumerate(results):
        x_offset = margin + idx * (img_size + margin)
        
        # Sketch (top row)
        sketch_y = margin + text_height
        grid.paste(sketch_img.resize((img_size, img_size)), (x_offset, sketch_y))
        
        # Generated image (middle row)
        gen_y = sketch_y + img_size + margin
        grid.paste(gen_img.resize((img_size, img_size)), (x_offset, gen_y))
        
        # Scale label (bottom)
        label = f"Scale {scale}"
        label_bbox = draw.textbbox((0, 0), label, font=small_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_x = x_offset + (img_size - label_width) // 2
        label_y = gen_y + img_size + margin // 2
        draw.text((label_x, label_y), label, fill='black', font=small_font)
    
    # Add row labels
    draw.text((margin, sketch_y + img_size // 2), "Input Sketch →", fill='red', font=small_font)
    draw.text((margin, gen_y + img_size // 2), "Generated →", fill='blue', font=small_font)
    
    # Save
    grid_path = os.path.join(output_dir, f"{category}_guidance_comparison.png")
    grid.save(grid_path)
    print(f"  ✅ Saved: {grid_path}")
    
    return grid_path

def main():
    print("=" * 70)
    print("🎯 GUIDANCE SCALE COMPARISON TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Download checkpoint
    print("\n📥 Downloading final checkpoint...")
    checkpoint_path = hf_hub_download(
        repo_id="DrRORAL/ragaf-diffusion-checkpoints",
        filename="stage1/final.pt",
        cache_dir="/root/.cache/huggingface",
    )
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Create output directory
    output_dir = "test_outputs_guidance_scales"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test categories (problematic ones identified by user)
    test_categories = {
        "eyeglasses": "eyeglasses",  # Was adding person
        "airplane": "airplane",       # Multiple objects
        "chair": "chair",            # Structure changes
        "bicycle": "bicycle",        # Structure changes
    }
    
    # Guidance scales to test
    guidance_scales = [1.5, 2.0, 2.5, 3.0, 5.0, 7.5]
    
    print(f"\n🧪 Testing {len(test_categories)} categories with {len(guidance_scales)} guidance scales")
    print(f"📊 Scales: {guidance_scales}")
    print("=" * 70)
    
    for category, prompt in test_categories.items():
        print(f"\n🎨 Category: {category}")
        
        # Get random sketch
        sketch_path = get_random_sketch(category)
        if not sketch_path:
            print(f"  ⚠️  No sketches found for {category}")
            continue
        
        print(f"  📄 Using: {os.path.basename(sketch_path)}")
        print(f"  💬 Prompt: '{prompt}'")
        
        results = []
        
        # Test each guidance scale
        for scale in guidance_scales:
            print(f"    Testing scale {scale}...", end=" ", flush=True)
            
            sketch, gen_img = generate_with_scale(
                model, sketch_path, prompt, scale, device
            )
            
            # Save individual result
            output_path = os.path.join(output_dir, f"{category}_scale_{scale}.png")
            gen_img.save(output_path)
            
            results.append((sketch, gen_img, scale))
            print("✅")
        
        # Create comparison grid
        print(f"  📊 Creating comparison grid...")
        create_comparison_grid(results, category, output_dir)
    
    # Copy to workspace
    print("\n" + "=" * 70)
    print("✅ GUIDANCE SCALE TEST COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Output directory: {output_dir}/")
    print(f"\n📊 Results:")
    print(f"   - Individual images: {output_dir}/<category>_scale_X.X.png")
    print(f"   - Comparison grids: {output_dir}/<category>_guidance_comparison.png")
    
    print(f"\n📋 Copying comparison grids to workspace...")
    os.system(f"cp {output_dir}/*_guidance_comparison.png /workspace/")
    print(f"   ✅ Available in /workspace/")
    
    print("\n" + "=" * 70)
    print("🎯 ANALYSIS GUIDE:")
    print("=" * 70)
    print("""
For each category, observe:

1. Sketch Fidelity (Most Important for Stage 1):
   - Does output preserve sketch structure?
   - Same number of objects as sketch?
   - Correct pose/orientation?

2. Added Objects:
   - Lower scales (1.5-3.0): Less likely to add objects
   - Higher scales (5.0-7.5): May add context/background

3. Detail vs Accuracy Trade-off:
   - Lower scales: Simpler, more accurate to sketch
   - Higher scales: More details, may drift from sketch

4. Recommended for Stage 1:
   - Guidance scale: 2.0 - 3.0
   - Reason: Best balance of sketch fidelity + quality
   - Avoids adding unwanted objects

5. Next Steps:
   - Based on results, choose optimal scale
   - Regenerate all 125 categories with that scale
   - Use minimal prompts (single word: category name)
""")

if __name__ == "__main__":
    main()
