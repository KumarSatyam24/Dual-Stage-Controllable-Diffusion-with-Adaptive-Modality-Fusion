#!/usr/bin/env python3
"""
Test final Stage 1 checkpoint on all 125 Sketchy categories.
Creates comparison grids showing sketch → generated image for 8 categories per grid.
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

# Dataset path - correct structure: sketchy/sketch/tx_000000000000/<category>/*.png
DATASET_PATH = "/workspace/sketchy/sketch/tx_000000000000"

def get_all_categories():
    """Get all 125 categories from the dataset."""
    categories = []
    if os.path.exists(DATASET_PATH):
        categories = [d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))]
        categories.sort()
    return categories

def get_random_sketch(category):
    """Get a random sketch from a category."""
    category_path = os.path.join(DATASET_PATH, category)
    sketch_files = glob.glob(f"{category_path}/*.png")
    
    if sketch_files:
        return random.choice(sketch_files)
    return None

def load_model(checkpoint_path, device):
    """Load the model once."""
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
    
    print(f"⚙️  Loading checkpoint: {checkpoint_path}")
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
    
    pipeline = Stage1DiffusionPipeline(
        model,
        num_inference_steps=30,
        guidance_scale=7.5,
        device=device
    )
    
    return model, pipeline

def generate_from_sketch(sketch_path, prompt, pipeline, device):
    """Generate image from sketch."""
    # Load sketch as grayscale
    sketch = Image.open(sketch_path).convert('L')
    sketch = sketch.resize((256, 256), Image.LANCZOS)
    
    # Convert to tensor
    transform = transforms.ToTensor()
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
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

def create_comparison_grid(results, grid_num, output_dir):
    """
    Create a comparison grid showing sketch → generated image.
    Grid layout: 2 rows × 8 columns (sketch on top, generated below)
    
    Args:
        results: List of (category, sketch_img, generated_img) tuples (up to 8)
        grid_num: Grid number for filename
        output_dir: Output directory
    """
    img_size = 256
    cols = 8
    margin = 10
    text_height = 30
    
    # Calculate grid dimensions
    grid_width = cols * img_size + (cols + 1) * margin
    grid_height = 2 * img_size + 3 * margin + 2 * text_height
    
    # Create white canvas
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Add each result to the grid
    for idx, (category, sketch_img, gen_img) in enumerate(results):
        if idx >= cols:
            break
        
        x_offset = margin + idx * (img_size + margin)
        
        # Sketch (top row)
        sketch_y = margin + text_height
        grid.paste(sketch_img.resize((img_size, img_size)), (x_offset, sketch_y))
        
        # Category label above sketch
        text_bbox = draw.textbbox((0, 0), category, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (img_size - text_width) // 2
        draw.text((text_x, margin), category, fill='black', font=font)
        
        # "Sketch" label
        draw.text((x_offset + 5, sketch_y + img_size - 20), "Sketch", fill='white', font=small_font)
        
        # Generated image (bottom row)
        gen_y = sketch_y + img_size + margin + text_height
        grid.paste(gen_img.resize((img_size, img_size)), (x_offset, gen_y))
        
        # "Generated" label
        draw.text((x_offset + 5, gen_y + img_size - 20), "Generated", fill='white', font=small_font)
    
    # Save grid
    grid_path = os.path.join(output_dir, f"comparison_grid_{grid_num:02d}.png")
    grid.save(grid_path)
    print(f"  ✅ Saved grid {grid_num}: {grid_path}")
    
    return grid_path

def main():
    print("=" * 70)
    print("🎨 TESTING ALL 125 SKETCHY CATEGORIES")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Get all categories
    print("\n📂 Finding categories...")
    categories = get_all_categories()
    print(f"✅ Found {len(categories)} categories")
    
    if len(categories) == 0:
        print("❌ No categories found! Check dataset path.")
        return
    
    # Download checkpoint
    print("\n📥 Downloading final checkpoint...")
    try:
        checkpoint_path = hf_hub_download(
            repo_id="DrRORAL/ragaf-diffusion-checkpoints",
            filename="stage1/final.pt",
            cache_dir="/root/.cache/huggingface",
        )
        print(f"✅ Checkpoint ready: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Failed to download checkpoint: {e}")
        return
    
    # Load model
    model, pipeline = load_model(checkpoint_path, device)
    
    # Create output directory
    output_dir = "test_outputs_all_categories"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 Generating images for all {len(categories)} categories...")
    print("=" * 70)
    
    all_results = []
    successful = 0
    failed = 0
    
    for idx, category in enumerate(categories, 1):
        try:
            # Get random sketch
            sketch_path = get_random_sketch(category)
            if sketch_path is None:
                print(f"⚠️  [{idx}/{len(categories)}] {category}: No sketches found")
                failed += 1
                continue
            
            # Generate prompt
            prompt = f"a photorealistic {category.replace('_', ' ')}"
            
            print(f"🎨 [{idx}/{len(categories)}] {category}...", end=" ", flush=True)
            
            # Generate image
            sketch_img, gen_img = generate_from_sketch(sketch_path, prompt, pipeline, device)
            
            # Save individual result
            result_path = os.path.join(output_dir, f"{category}.png")
            gen_img.save(result_path)
            
            # Store for grid
            all_results.append((category, sketch_img, gen_img))
            successful += 1
            
            print(f"✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
    
    # Create comparison grids (8 categories per grid)
    print("\n" + "=" * 70)
    print("📊 Creating comparison grids...")
    print("=" * 70)
    
    grids_created = 0
    for i in range(0, len(all_results), 8):
        batch = all_results[i:i+8]
        grid_num = (i // 8) + 1
        create_comparison_grid(batch, grid_num, output_dir)
        grids_created += 1
    
    # Create summary
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)
    print(f"📊 Results:")
    print(f"   Total categories: {len(categories)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Grids created: {grids_created}")
    print(f"\n📂 Output directory: {output_dir}/")
    print(f"   - Individual images: {output_dir}/<category>.png")
    print(f"   - Comparison grids: {output_dir}/comparison_grid_XX.png")
    print("\n💡 View the grids to see sketch → generated comparisons!")
    print("   Each grid shows 8 categories (sketch top, generated bottom)")
    
    # Copy a few sample grids to workspace
    print("\n📋 Copying sample grids to workspace...")
    for i in range(1, min(4, grids_created + 1)):
        src = f"{output_dir}/comparison_grid_{i:02d}.png"
        dst = f"/workspace/comparison_grid_{i:02d}.png"
        if os.path.exists(src):
            os.system(f"cp {src} {dst}")
            print(f"   ✅ {dst}")

if __name__ == "__main__":
    main()
