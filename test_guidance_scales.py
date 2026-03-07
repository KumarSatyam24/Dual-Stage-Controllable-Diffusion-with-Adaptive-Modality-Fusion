#!/usr/bin/env python3
"""
Test different guidance scales to find optimal value for Stage 1.
Compares how different scales affect sketch fidelity and output quality.
"""

import torch
import os
import sys
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

def test_guidance_scale(model, sketch_path, category, prompt, guidance_scales, device, output_dir):
    """Test multiple guidance scales on same sketch."""
    print(f"\n🎨 Testing category: {category}")
    print(f"   Prompt: '{prompt}'")
    print(f"   Scales: {guidance_scales}")
    
    # Load sketch
    sketch = Image.open(sketch_path).convert('L')
    sketch = sketch.resize((256, 256), Image.LANCZOS)
    
    transform = transforms.ToTensor()
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
    results = []
    
    for scale in guidance_scales:
        print(f"   Testing scale {scale}...", end=" ", flush=True)
        
        # Create pipeline with specific guidance scale
        pipeline = Stage1DiffusionPipeline(
            model,
            num_inference_steps=30,
            guidance_scale=scale,
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
        
        # Save individual result
        output_path = os.path.join(output_dir, f"{category}_scale_{scale}.png")
        output_image.save(output_path)
        
        results.append((scale, sketch, output_image))
        print("✅")
    
    return results

def create_comparison_grid(results, category, prompt, output_dir):
    """Create comparison grid showing different guidance scales."""
    num_scales = len(results)
    img_size = 256
    margin = 10
    text_height = 40
    
    # Grid: 2 rows (sketch + generated) × num_scales columns
    grid_width = num_scales * img_size + (num_scales + 1) * margin
    grid_height = 2 * img_size + 3 * margin + 2 * text_height
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Title
    title = f"{category}: '{prompt}'"
    draw.text((margin, margin//2), title, fill='black', font=font)
    
    for idx, (scale, sketch_img, gen_img) in enumerate(results):
        x_offset = margin + idx * (img_size + margin)
        
        # Sketch (top row)
        sketch_y = margin + text_height
        grid.paste(sketch_img.resize((img_size, img_size)), (x_offset, sketch_y))
        
        # Scale label above sketch
        scale_label = f"Scale: {scale}"
        text_bbox = draw.textbbox((0, 0), scale_label, font=small_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (img_size - text_width) // 2
        draw.text((text_x, sketch_y - 20), scale_label, fill='black', font=small_font)
        
        # Generated (bottom row)
        gen_y = sketch_y + img_size + margin + text_height
        grid.paste(gen_img.resize((img_size, img_size)), (x_offset, gen_y))
    
    # Save grid
    grid_path = os.path.join(output_dir, f"{category}_guidance_comparison.png")
    grid.save(grid_path)
    print(f"  ✅ Saved comparison grid: {grid_path}")
    
    return grid_path

def main():
    print("=" * 70)
    print("🎯 GUIDANCE SCALE COMPARISON TEST")
    print("=" * 70)
    print("\nTesting different guidance scales to find optimal value for Stage 1")
    print("Focus: Sketch fidelity vs text following")
    
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
    output_dir = "test_outputs_guidance_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test cases - problematic categories identified by user
    test_cases = [
        ("eyewear", "/workspace/sketchy/sketch/tx_000000000000/eyeglasses/n03379051_7923-13.png", "eyewear"),
        ("airplane", "/workspace/sketchy/sketch/tx_000000000000/airplane/n02691156_583-11.png", "an airplane"),
        ("chair", "/workspace/sketchy/sketch/tx_000000000000/chair/n03001627_1639-2.png", "a chair"),
        ("bicycle", "/workspace/sketchy/sketch/tx_000000000000/bicycle/n02834778_1820-8.png", "a bicycle"),
    ]
    
    # Guidance scales to test
    guidance_scales = [1.5, 2.0, 2.5, 3.0, 5.0, 7.5]
    
    print(f"\n🧪 Testing {len(test_cases)} categories with {len(guidance_scales)} guidance scales")
    print("=" * 70)
    
    for category, sketch_path, prompt in test_cases:
        if not os.path.exists(sketch_path):
            print(f"⚠️  Sketch not found: {sketch_path}")
            continue
        
        # Test different scales
        results = test_guidance_scale(
            model, sketch_path, category, prompt, 
            guidance_scales, device, output_dir
        )
        
        # Create comparison grid
        create_comparison_grid(results, category, prompt, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ GUIDANCE SCALE COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Output directory: {output_dir}/")
    print(f"\n📊 Generated files:")
    print(f"   - Individual results: {output_dir}/<category>_scale_X.X.png")
    print(f"   - Comparison grids: {output_dir}/<category>_guidance_comparison.png")
    
    # Copy to workspace
    print("\n📋 Copying comparison grids to workspace...")
    import glob
    comparison_grids = glob.glob(f"{output_dir}/*_guidance_comparison.png")
    for grid in comparison_grids:
        dst = f"/workspace/{os.path.basename(grid)}"
        os.system(f"cp {grid} {dst}")
        print(f"   ✅ {dst}")
    
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATIONS:")
    print("=" * 70)
    print("""
1. View the comparison grids in /workspace/
2. For each category, observe:
   - Which scale preserves sketch structure best?
   - Which scale avoids adding extra objects?
   - Which scale produces cleanest results?
   
3. Expected observations:
   - Lower scales (1.5-3.0): Better sketch fidelity
   - Higher scales (5.0-7.5): More details but may distort
   
4. Recommended for Stage 1:
   - Guidance scale: 2.5 or 3.0
   - Reason: Good balance of sketch following + quality
   
5. Next step:
   - Regenerate all 125 categories with optimal scale
   - Use minimal prompts (category name only)
    """)

if __name__ == "__main__":
    main()
