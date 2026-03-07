#!/usr/bin/env python3
"""
Test Stage 1 Epoch 2 checkpoint from HuggingFace Hub.
Downloads the checkpoint and runs inference to verify sketch conditioning works.
"""

import torch
import os
from PIL import Image
from huggingface_hub import hf_hub_download

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from configs.config import get_default_config

def download_checkpoint():
    """Download epoch 2 checkpoint from HF Hub."""
    print("📥 Downloading epoch 2 checkpoint from HuggingFace Hub...")
    
    repo_id = "DrRORAL/ragaf-diffusion-checkpoints"
    filename = "stage1/epoch_2.pt"
    
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
        print("\n💡 Make sure:")
        print("   1. You're logged in: huggingface-cli login")
        print("   2. Epoch 2 has been uploaded to HF Hub")
        print("   3. Repository exists: https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints")
        return None

def test_checkpoint(checkpoint_path):
    """Test the checkpoint with inference."""
    print("\n🧪 Testing checkpoint...")
    
    # Load config
    config_dict = get_default_config()
    model_config = config_dict['model']
    
    # Initialize model
    print("📦 Loading Stage 1 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage1SketchGuidedDiffusion(
        pretrained_model_name=model_config.pretrained_model_name,
        sketch_encoder_channels=model_config.sketch_encoder_channels,
        freeze_base_unet=model_config.freeze_stage1_unet,
        use_lora=model_config.use_lora,
        lora_rank=model_config.lora_rank
    )
    
    # Load checkpoint
    print(f"⚙️  Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded checkpoint (no epoch info)")
    
    model = model.to(device)
    model.eval()
    
    # Create pipeline for easier inference
    pipeline = Stage1DiffusionPipeline(model, device=device)
    
    # Test with sample data
    print("\n🎨 Running test inference...")
    
    # Load test sketch - use existing one
    sketch_path = "input_sketch.png"
    if os.path.exists(sketch_path):
        print(f"📝 Using existing sketch: {sketch_path}")
        sketch = Image.open(sketch_path).convert('RGB')
    else:
        print(f"⚠️  Sketch not found, creating simple test sketch...")
        sketch = Image.new('RGB', (256, 256), 'white')
        # Draw a simple shape for testing
        from PIL import ImageDraw
        draw = ImageDraw.Draw(sketch)
        draw.rectangle([64, 96, 192, 160], outline='black', width=3)
    
    # Run inference
    text_prompt = "an airplane in the sky"
    
    # Convert sketch to tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)  # (1, 1, 256, 256)
    
    with torch.no_grad():
        output_tensor = pipeline.generate(
            sketch=sketch_tensor,
            text_prompt=text_prompt,
            height=256,
            width=256,
            seed=42,
        )
    
    # Convert output tensor to PIL image
    # Assuming output is (1, 3, H, W) in range [-1, 1] or [0, 1]
    output_np = output_tensor[0].cpu().permute(1, 2, 0).numpy()
    if output_np.min() < 0:
        output_np = (output_np + 1) / 2  # [-1, 1] -> [0, 1]
    output_np = (output_np * 255).clip(0, 255).astype('uint8')
    output = Image.fromarray(output_np)
    
    # Save result
    output_dir = "test_outputs_epoch2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/epoch2_test_output.png"
    output.save(output_path)
    
    print(f"✅ Test complete! Output saved to: {output_path}")
    print("\n📊 What to check:")
    print("   1. Does the output follow the sketch structure?")
    print("   2. Is it better than the broken model (abstract shapes)?")
    print("   3. Does it look coherent (not just noise)?")
    print("\n💡 Compare with:")
    print("   - Old broken output: test_data/output_airplane.png")
    print("   - This should show sketch guidance working!")
    
    return output_path

def main():
    print("=" * 60)
    print("Stage 1 Epoch 2 Checkpoint Test")
    print("=" * 60)
    
    # Download checkpoint
    checkpoint_path = download_checkpoint()
    if checkpoint_path is None:
        return
    
    # Test checkpoint
    try:
        output_path = test_checkpoint(checkpoint_path)
        print("\n" + "=" * 60)
        print("✅ TEST SUCCESSFUL!")
        print("=" * 60)
        print(f"\n🖼️  View output: {output_path}")
        print("\n🔍 Key question: Does it follow the sketch better than the old model?")
        print("\n📂 To view the image, download it or use:")
        print(f"   Display on server: feh {output_path}")
        print(f"   Or copy to workspace: cp {output_path} /workspace/")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
