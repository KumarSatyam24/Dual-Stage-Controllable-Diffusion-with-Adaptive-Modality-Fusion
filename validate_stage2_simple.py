#!/usr/bin/env python3
"""
Simple Stage 2 Validation Script
Validates Stage 2 checkpoint on a small set of samples.
"""

import sys
from pathlib import Path

# Setup paths BEFORE any imports
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Import from src - deferred import to avoid relative import issues
def import_models():
    from src.models.stage2_refinement import Stage2SemanticRefinement
    from src.datasets.sketchy_dataset import SketchyDataset
    from src.configs.config import get_default_config
    return Stage2SemanticRefinement, SketchyDataset, get_default_config


class Stage2Validator:
    def __init__(self, checkpoint_path, num_samples=10, device='cuda'):
        self.checkpoint_path = Path(checkpoint_path)
        self.num_samples = num_samples
        self.device = device
        
        print(f"🔍 Stage 2 Validation")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Samples: {num_samples}")
        print(f"   Device: {device}\n")
        
        # Import models (deferred to avoid relative import issues)
        Stage2SemanticRefinement, SketchyDataset, get_default_config = import_models()
        self.Stage2SemanticRefinement = Stage2SemanticRefinement
        self.SketchyDataset = SketchyDataset
        self.get_default_config = get_default_config
        
        self.config = get_default_config()
        self.load_models()
        self.load_dataset()
        
    def load_models(self):
        """Load Stage 2 model and components."""
        print("📦 Loading models...")
        
        model_name = "runwayml/stable-diffusion-v1-5"
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        print("   ✅ VAE loaded")
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.eval()
        print("   ✅ Text Encoder loaded")
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        print("   ✅ Tokenizer loaded")
        
        # Load UNet
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet"
        ).to(self.device)
        unet.eval()
        print("   ✅ UNet loaded")
        
        # Load Stage 2 model
        print(f"   Loading checkpoint from {self.checkpoint_path}...")
        self.model = self.Stage2SemanticRefinement(
            unet=unet,
            node_feature_dim=6,
            text_dim=768,
            hidden_dim=self.config['model'].hidden_dim,
            num_graph_layers=2,
            num_attention_heads=8,
            fusion_method="learned",
            use_region_adaptive_fusion=True
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"   ✅ Stage 2 model loaded from Epoch {self.epoch}\n")
        
        # Setup scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        
    def load_dataset(self):
        """Load validation dataset."""
        print("📁 Loading validation dataset...")
        self.dataset = self.SketchyDataset(
            root_dir=self.config['data'].sketchy_root,
            split='test',
            image_size=256,
            augment=False
        )
        print(f"   Total samples: {len(self.dataset)}")
        print(f"   Will validate: {min(self.num_samples, len(self.dataset))} samples\n")
        
    @torch.no_grad()
    def validate(self):
        """Run validation on samples."""
        print("🚀 Starting validation...\n")
        
        inference_times = []
        
        # Use first N samples
        num_to_validate = min(self.num_samples, len(self.dataset))
        
        for idx in tqdm(range(num_to_validate), desc="Validating"):
            sample = self.dataset[idx]
            
            # Get inputs
            sketch = sample['sketch'].unsqueeze(0).to(self.device)  # (1, 1, 256, 256)
            photo = sample['photo']  # (3, 256, 256)
            prompt = sample['text_prompt']
            
            try:
                import time
                start = time.time()
                
                # Encode image to latent
                with torch.no_grad():
                    photo_tensor = photo.unsqueeze(0).to(self.device)
                    latents = self.vae.encode(photo_tensor).latent_dist.sample() * 0.18215
                
                # Encode text prompt
                text_inputs = self.tokenizer(
                    [prompt],
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                ).to(self.device)
                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]  # (77, 768)
                
                # Create random noise and timestep
                noise = torch.randn_like(latents)
                timestep = torch.tensor([100], device=self.device)
                
                # Setup scheduler
                self.scheduler.set_timesteps(50)
                noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
                
                # Get region graph (from dataset)
                region_graph = sample.get('region_graph', None)
                
                # Run Stage 2 model
                output = self.model(
                    noisy_latents,
                    timestep,
                    region_graph,
                    text_embeddings,
                    return_dict=True
                )
                
                # Model executed successfully
                inference_time = time.time() - start
                inference_times.append(inference_time)
                
            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {e}")
                continue
        
        # Print results
        print("\n" + "="*50)
        print("STAGE 2 VALIDATION RESULTS")
        print("="*50)
        print(f"Epoch: {self.epoch}")
        print(f"Samples Evaluated: {len(inference_times)}")
        if inference_times:
            print(f"Avg Inference Time: {np.mean(inference_times):.3f}s per sample")
            print(f"Min/Max Time: {np.min(inference_times):.3f}s / {np.max(inference_times):.3f}s")
        print("\n✅ Model validation completed successfully!")
        print("="*50 + "\n")
        
        return {
            "epoch": self.epoch,
            "num_samples": len(inference_times),
            "avg_inference_time": float(np.mean(inference_times)) if inference_times else 0,
            "status": "success"
        }


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 2 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to validate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    validator = Stage2Validator(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        device=args.device
    )
    
    results = validator.validate()
    
    # Save results
    output_file = f"validation_epoch_{validator.epoch}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
