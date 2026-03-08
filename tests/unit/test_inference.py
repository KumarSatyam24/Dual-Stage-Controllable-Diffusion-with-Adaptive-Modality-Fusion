"""
Full Inference Test - Generate Image from Sketch

This script tests the complete inference pipeline:
1. Load a real sketch from the dataset
2. Run the diffusion process
3. Generate and save the output image
"""

import torch
import os
from pathlib import Path
from models.stage1_diffusion import Stage1SketchGuidedDiffusion
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_sketch(sketch_path: str, size=512):
    """Load and preprocess a sketch image."""
    sketch_img = Image.open(sketch_path).convert('L')
    sketch_img = sketch_img.resize((size, size))
    
    # Convert to tensor [1, 1, H, W] normalized to [0, 1]
    sketch_array = np.array(sketch_img) / 255.0
    sketch_tensor = torch.from_numpy(sketch_array).float().unsqueeze(0).unsqueeze(0)
    
    return sketch_tensor, sketch_img


def generate_image(model, sketch, text_prompt, num_inference_steps=50, guidance_scale=7.5, device='cuda'):
    """
    Generate image from sketch using DDPM sampling.
    
    Args:
        model: Stage1SketchGuidedDiffusion model
        sketch: Sketch tensor [1, 1, H, W]
        text_prompt: Text description
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        device: Device to run on
    """
    print(f"\n🎨 Generating image...")
    print(f"   Prompt: '{text_prompt}'")
    print(f"   Steps: {num_inference_steps}")
    print(f"   Guidance: {guidance_scale}")
    
    model = model.to(device)
    model.eval()
    sketch = sketch.to(device)
    
    # Encode sketch and text
    with torch.no_grad():
        sketch_features = model.encode_sketch(sketch)
        text_embeddings = model.encode_text([text_prompt])
        
        # For classifier-free guidance, also encode unconditional
        uncond_embeddings = model.encode_text([""])
        
        # Combine for guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Double sketch features for guidance
        sketch_features_doubled = []
        for feat in sketch_features:
            sketch_features_doubled.append(torch.cat([feat, feat]))
    
    # Initialize random latents
    latents = torch.randn(1, 4, 64, 64).to(device)
    
    # Set up scheduler (DDPM)
    num_train_timesteps = 1000
    beta_start = 0.00085
    beta_end = 0.012
    
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    
    # Create timestep schedule
    timesteps = torch.linspace(num_train_timesteps - 1, 0, num_inference_steps).long()
    
    # Denoising loop
    print("\n🔄 Denoising...")
    for i, t in enumerate(tqdm(timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        
        # Expand latents for guidance
        latent_model_input = torch.cat([latents] * 2)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model(
                latents=latent_model_input,
                timestep=t_tensor,
                sketch_features=sketch_features_doubled,
                text_embeddings=text_embeddings,
                return_dict=False
            )
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous latents
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        
        # Compute x_0 from noise prediction
        pred_original_sample = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        
        # Compute direction to x_t
        pred_sample_direction = (1 - alpha_t_prev).sqrt() * noise_pred
        
        # Compute x_{t-1}
        latents = alpha_t_prev.sqrt() * pred_original_sample + pred_sample_direction
    
    # Decode latents to image
    print("\n🖼️  Decoding to image...")
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = model.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)


def main():
    """Main inference test."""
    print("="*60)
    print("Full Inference Test - Generate Image from Sketch")
    print("="*60)
    
    # Load checkpoint
    checkpoint_path = "/workspace/outputs/stage1/epoch_2.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n📦 Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = Stage1SketchGuidedDiffusion(
        pretrained_model_name="runwayml/stable-diffusion-v1-5",
        freeze_base_unet=False,
        use_lora=True,
        lora_rank=4
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from epoch {checkpoint['epoch'] + 1}")
    
    # Find a sketch to test
    dataset_root = Path("/workspace/datasets/sketchy")
    sketch_dir = dataset_root / "sketch" / "tx_000000000000"
    
    # Find any available sketch
    import subprocess
    result = subprocess.run(
        ['find', str(sketch_dir), '-name', '*.png', '-type', 'f'],
        capture_output=True, text=True
    )
    
    sketch_files = result.stdout.strip().split('\n')
    sketch_files = [f for f in sketch_files if f]  # Remove empty strings
    
    if not sketch_files:
        print("❌ No sketches found in dataset!")
        return
    
    # Use the first sketch
    sketch_path = Path(sketch_files[0])
    
    print(f"\n📍 Using sketch: {sketch_path.name}")
    print(f"   Category: {sketch_path.parent.name}")
    
    # Load sketch
    sketch, sketch_img = load_sketch(str(sketch_path))
    print(f"✅ Loaded sketch: {sketch.shape}")
    
    # Generate image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    category_name = sketch_path.parent.name.replace("_", " ")
    text_prompt = f"a photo of a {category_name}"
    
    generated_img = generate_image(
        model=model,
        sketch=sketch,
        text_prompt=text_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        device=device
    )
    
    # Save results
    output_dir = Path("/workspace/outputs/test_inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sketch
    sketch_img.save(output_dir / "input_sketch.png")
    
    # Save generated image
    generated_img.save(output_dir / "generated_image.png")
    
    # Create comparison
    comparison = Image.new('RGB', (1024, 512))
    comparison.paste(sketch_img.convert('RGB'), (0, 0))
    comparison.paste(generated_img, (512, 0))
    comparison.save(output_dir / "comparison.png")
    
    print("\n" + "="*60)
    print("✅ INFERENCE TEST COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   - input_sketch.png")
    print(f"   - generated_image.png")
    print(f"   - comparison.png")
    print(f"\n💡 View the results to evaluate model performance!")


if __name__ == "__main__":
    main()
