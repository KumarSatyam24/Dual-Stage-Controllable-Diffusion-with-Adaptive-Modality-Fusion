#!/usr/bin/env python3
"""
Re-test Stage 1 with corrected prompts and settings.
Focus on preserving sketch structure without adding context.
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

DATASET_PATH = "/workspace/sketchy/sketch/tx_000000000000"

# Problematic categories (wearables and items prone to context leakage)
PROBLEMATIC_CATEGORIES = [
    "eyeglasses",
    "hat", 
    "shoe",
    "watch",
    "ring",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon"
]

# Different prompt strategies to test
PROMPT_STRATEGIES = {
    "minimal": lambda cat: f"{cat.replace('_', ' ')}",
    "isolated": lambda cat: f"{cat.replace('_', ' ')}, isolated object, white background",
    "simple": lambda cat: f"simple {cat.replace('_', ' ')}, no background",
    "structure_only": lambda cat: f"{cat.replace('_', ' ')} shape, simple outline"
}

# Different guidance scales to test
GUIDANCE_SCALES = [3.0, 5.0, 7.5]

def load_model(checkpoint_path, device, guidance_scale):
    """Load model with specified guidance scale."""
    print(f"📦 Loading model with guidance_scale={guidance_scale}...")
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
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    pipeline = Stage1DiffusionPipeline(
        model,
        num_inference_steps=30,
        guidance_scale=guidance_scale,  # Variable guidance scale
        device=device
    )
    
    return pipeline

def get_sketch(category):
    """Get a sketch for a category."""
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(category_path):
        return None
    sketch_files = glob.glob(f"{category_path}/*.png")
    return random.choice(sketch_files) if sketch_files else None

def generate_from_sketch(sketch_path, prompt, pipeline, device):
    """Generate image from sketch."""
    sketch = Image.open(sketch_path).convert('L')
    sketch = sketch.resize((256, 256), Image.LANCZOS)
    
    transform = transforms.ToTensor()
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = pipeline.generate(
            text_prompt=prompt,
            sketch=sketch_tensor,
            height=256,
            width=256,
        )
    
    output_numpy = output_tensor.squeeze(0).cpu().numpy()
    output_numpy = (output_numpy * 255).astype('uint8')
    output_numpy = output_numpy.transpose(1, 2, 0)
    output_image = Image.fromarray(output_numpy)
    
    return sketch, output_image

def create_comparison_grid(category, sketch_img, results, output_path):
    """
    Create comparison grid showing:
    - Sketch (left)
    - Old result (guidance=7.5, photorealistic prompt)
    - New results (different prompts and guidance scales)
    
    Layout: 1 sketch + N generated images in a row
    """
    img_size = 256
    margin = 10
    text_height = 40
    
    num_results = len(results) + 2  # sketch + old + new results
    grid_width = num_results * img_size + (num_results + 1) * margin
    grid_height = img_size + 2 * margin + text_height
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Title
    title = f"Category: {category.replace('_', ' ').title()}"
    draw.text((margin, 5), title, fill='black', font=font)
    
    x_offset = margin
    y_offset = margin + text_height
    
    # Sketch
    grid.paste(sketch_img.resize((img_size, img_size)), (x_offset, y_offset))
    draw.text((x_offset + 5, y_offset + 5), "SKETCH", fill='white', font=font)
    x_offset += img_size + margin
    
    # Results
    for label, img in results:
        grid.paste(img.resize((img_size, img_size)), (x_offset, y_offset))
        
        # Multi-line label
        lines = label.split('\n')
        for i, line in enumerate(lines):
            draw.text((x_offset + 5, y_offset + 5 + i * 15), line, fill='white', font=font)
        
        x_offset += img_size + margin
    
    grid.save(output_path)
    return output_path

def main():
    print("=" * 70)
    print("🔬 CORRECTED STAGE 1 TESTING")
    print("Testing with minimal prompts and lower guidance scales")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Download checkpoint
    print("\n📥 Downloading checkpoint...")
    checkpoint_path = hf_hub_download(
        repo_id="DrRORAL/ragaf-diffusion-checkpoints",
        filename="stage1/final.pt",
        cache_dir="/root/.cache/huggingface",
    )
    
    output_dir = "test_outputs_corrected"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 Testing {len(PROBLEMATIC_CATEGORIES)} problematic categories...")
    print("=" * 70)
    
    for category in PROBLEMATIC_CATEGORIES:
        print(f"\n📸 Testing: {category}")
        print("-" * 70)
        
        sketch_path = get_sketch(category)
        if not sketch_path:
            print(f"  ⚠️  No sketch found, skipping")
            continue
        
        sketch_img = Image.open(sketch_path).convert('L').resize((256, 256))
        
        # Test different configurations
        results = []
        
        # 1. Original (baseline - what we used before)
        print("  🔹 Baseline (guidance=7.5, 'photorealistic')...")
        pipeline_old = load_model(checkpoint_path, device, guidance_scale=7.5)
        _, img_old = generate_from_sketch(
            sketch_path,
            f"a photorealistic {category.replace('_', ' ')}",
            pipeline_old,
            device
        )
        results.append(("OLD\nG=7.5\nphotorealistic", img_old))
        del pipeline_old
        torch.cuda.empty_cache()
        
        # 2. Test minimal prompt with lower guidance
        print("  🔹 Minimal prompt (guidance=3.0)...")
        pipeline_min = load_model(checkpoint_path, device, guidance_scale=3.0)
        _, img_min = generate_from_sketch(
            sketch_path,
            PROMPT_STRATEGIES["minimal"](category),
            pipeline_min,
            device
        )
        results.append(("MINIMAL\nG=3.0\nno context", img_min))
        del pipeline_min
        torch.cuda.empty_cache()
        
        # 3. Test isolated prompt
        print("  🔹 Isolated prompt (guidance=5.0)...")
        pipeline_iso = load_model(checkpoint_path, device, guidance_scale=5.0)
        _, img_iso = generate_from_sketch(
            sketch_path,
            PROMPT_STRATEGIES["isolated"](category),
            pipeline_iso,
            device
        )
        results.append(("ISOLATED\nG=5.0\nwhite bg", img_iso))
        del pipeline_iso
        torch.cuda.empty_cache()
        
        # 4. Test structure-only prompt
        print("  🔹 Structure-only prompt (guidance=3.0)...")
        pipeline_struct = load_model(checkpoint_path, device, guidance_scale=3.0)
        _, img_struct = generate_from_sketch(
            sketch_path,
            PROMPT_STRATEGIES["structure_only"](category),
            pipeline_struct,
            device
        )
        results.append(("STRUCTURE\nG=3.0\nshape only", img_struct))
        del pipeline_struct
        torch.cuda.empty_cache()
        
        # Create comparison grid
        output_path = os.path.join(output_dir, f"{category}_comparison.png")
        create_comparison_grid(category, sketch_img, results, output_path)
        print(f"  ✅ Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE")
    print("=" * 70)
    print(f"\n📂 Results saved in: {output_dir}/")
    print(f"📊 Compare OLD vs NEW approaches for each category")
    print("\n💡 Look for:")
    print("   - Does minimal prompt preserve sketch structure better?")
    print("   - Does lower guidance scale reduce context leakage?")
    print("   - Which strategy works best for wearable items?")
    
    # Copy to workspace
    print("\n📋 Copying results to workspace...")
    os.system(f"cp {output_dir}/*.png /workspace/")
    print(f"✅ Available in /workspace/ for easy viewing")

if __name__ == "__main__":
    main()
