#!/usr/bin/env python3
"""
Test the final Stage 1 checkpoint on diverse sketches from the Sketchy dataset.
This script samples different categories to showcase the model's generalization.
"""

import torch
import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from torchvision import transforms

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from configs.config import get_default_config


# Categories to test with prompts
TEST_CATEGORIES = [
    ("airplane", "a commercial airplane flying in the blue sky"),
    ("car", "a red sports car on a highway"),
    ("cat", "a fluffy cat sitting on a sofa"),
    ("dog", "a golden retriever dog in a park"),
    ("horse", "a brown horse running in a field"),
    ("bicycle", "a mountain bike on a trail"),
    ("chair", "a modern wooden chair in a living room"),
    ("tree", "a large oak tree in a meadow"),
    ("house", "a beautiful cottage with a garden"),
    ("bird", "a colorful bird perched on a branch"),
]


def find_sketch_paths(dataset_path="/workspace/sketchy/sketch/tx_000000000000"):
    """Find available sketch paths for each category."""
    sketch_paths = {}
    
    for category, _ in TEST_CATEGORIES:
        category_path = Path(dataset_path) / category
        
        if category_path.exists():
            # Get all PNG files in the category
            sketches = list(category_path.glob("*.png"))
            
            if sketches:
                # Randomly select one sketch
                selected = random.choice(sketches)
                sketch_paths[category] = selected
                print(f"✅ Found {category}: {selected.name}")
            else:
                print(f"⚠️  No sketches found for {category}")
        else:
            print(f"⚠️  Category not found: {category}")
    
    return sketch_paths


def download_checkpoint(checkpoint_name="final.pt"):
    """Download final checkpoint from HF Hub."""
    cache_path = "/root/.cache/huggingface/models--DrRORAL--ragaf-diffusion-checkpoints/snapshots"
    
    # Check if already downloaded
    if os.path.exists(cache_path):
        for root, dirs, files in os.walk(cache_path):
            if checkpoint_name in files and "stage1" in root:
                checkpoint_path = os.path.join(root, checkpoint_name)
                print(f"✅ Using cached checkpoint: {checkpoint_path}")
                return checkpoint_path
    
    print(f"📥 Downloading {checkpoint_name} from HuggingFace Hub...")
    repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
    filename = f"stage1/{checkpoint_name}"
    
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
        # Try epoch_10.pt as fallback
        if checkpoint_name != "epoch_10.pt":
            print("🔄 Trying epoch_10.pt instead...")
            return download_checkpoint("epoch_10.pt")
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
    
    # Create pipeline with good quality settings
    pipeline = Stage1DiffusionPipeline(
        model, 
        num_inference_steps=50,  # Higher quality
        guidance_scale=7.5,
        device=device
    )
    
    return model, pipeline


def test_sketch(sketch_path, prompt, output_path, pipeline, device):
    """Generate image from sketch."""
    print(f"\n🎨 Testing: {sketch_path.name}")
    print(f"   Prompt: {prompt}")
    
    # Load and preprocess sketch
    sketch = Image.open(sketch_path).convert('L')  # Grayscale
    sketch = sketch.resize((256, 256), Image.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)  # (1, 1, 256, 256)
    
    # Generate
    with torch.no_grad():
        output_tensor = pipeline.generate(
            text_prompt=prompt,
            sketch=sketch_tensor,
            height=256,
            width=256,
        )
    
    # Convert to PIL
    output_numpy = output_tensor.squeeze(0).cpu().numpy()  # (3, H, W)
    output_numpy = (output_numpy * 255).astype('uint8')
    output_numpy = output_numpy.transpose(1, 2, 0)  # (H, W, 3)
    output_image = Image.fromarray(output_numpy)
    
    # Save
    output_image.save(output_path)
    print(f"   ✅ Saved: {output_path}")
    
    return sketch, output_image


def create_comparison_grid(results, output_path):
    """Create a grid showing sketch vs generated image for all categories."""
    print("\n📊 Creating comparison grid...")
    
    n_samples = len(results)
    n_cols = 4  # 2 columns per sample (sketch + generated)
    n_rows = (n_samples + 1) // 2
    
    img_size = 256
    grid_width = n_cols * img_size
    grid_height = n_rows * img_size
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    
    for idx, (category, sketch_img, generated_img, prompt) in enumerate(results):
        row = idx // 2
        col = (idx % 2) * 2
        
        # Place sketch (convert to RGB if grayscale)
        if sketch_img.mode == 'L':
            sketch_img = sketch_img.convert('RGB')
        grid.paste(sketch_img, (col * img_size, row * img_size))
        
        # Place generated image
        grid.paste(generated_img, ((col + 1) * img_size, row * img_size))
    
    grid.save(output_path)
    print(f"✅ Grid saved: {output_path}")
    
    return grid


def main():
    print("=" * 70)
    print("🎨 FINAL CHECKPOINT - DIVERSE SKETCHES TEST")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Setup output directory
    output_dir = Path("test_outputs_final_diverse")
    output_dir.mkdir(exist_ok=True)
    
    # Download checkpoint
    print("\n" + "=" * 70)
    print("📥 DOWNLOADING CHECKPOINT")
    print("=" * 70)
    checkpoint_path = download_checkpoint("final.pt")
    if checkpoint_path is None:
        print("❌ Failed to download checkpoint")
        return
    
    # Load model
    print("\n" + "=" * 70)
    print("📦 LOADING MODEL")
    print("=" * 70)
    model, pipeline = load_model(checkpoint_path, device)
    
    # Find sketches
    print("\n" + "=" * 70)
    print("🔍 FINDING SKETCHES")
    print("=" * 70)
    sketch_paths = find_sketch_paths()
    
    if not sketch_paths:
        print("❌ No sketches found in dataset!")
        print("💡 Make sure the dataset is mounted at /network_volume/datasets/sketchy/")
        return
    
    # Test each category
    print("\n" + "=" * 70)
    print("🎨 GENERATING IMAGES")
    print("=" * 70)
    
    results = []
    for category, prompt in TEST_CATEGORIES:
        if category not in sketch_paths:
            continue
        
        sketch_path = sketch_paths[category]
        output_path = output_dir / f"{category}_output.png"
        
        try:
            sketch_img, generated_img = test_sketch(
                sketch_path, prompt, output_path, pipeline, device
            )
            
            # Save sketch for reference
            sketch_copy_path = output_dir / f"{category}_sketch.png"
            sketch_img.save(sketch_copy_path)
            
            results.append((category, sketch_img, generated_img, prompt))
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Create comparison grid
    if results:
        print("\n" + "=" * 70)
        print("📊 CREATING COMPARISON GRID")
        print("=" * 70)
        grid_path = output_dir / "comparison_grid.png"
        create_comparison_grid(results, grid_path)
        
        # Also copy to workspace
        workspace_grid = Path("/workspace/final_diverse_comparison.png")
        grid_path_obj = Path(grid_path)
        if grid_path_obj.exists():
            import shutil
            shutil.copy(grid_path, workspace_grid)
            print(f"✅ Copied to workspace: {workspace_grid}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"   Tested categories: {len(results)}")
    print(f"   Output directory: {output_dir}")
    print(f"   Comparison grid: {output_dir}/comparison_grid.png")
    print(f"   Workspace copy: /workspace/final_diverse_comparison.png")
    
    print("\n📋 Tested categories:")
    for category, _, _, prompt in results:
        print(f"   ✅ {category}: {prompt}")
    
    print("\n🔍 What to observe:")
    print("   1. Does each output follow its sketch structure?")
    print("   2. Are the prompts interpreted correctly?")
    print("   3. How is the quality across different object types?")
    print("   4. Compare with epoch 2 results - improvement?")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    main()
