#!/usr/bin/env python3
"""
Comprehensive Stage 2 Validation Script
Computes SSIM, PSNR, and LPIPS metrics and saves comparison images.
Provides overall and category-wise statistics.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add project root and src to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS not available. Install with 'pip install lpips'")

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion
from src.models.stage2_refinement import Stage2SemanticRefinement
from src.datasets.sketchy_dataset import SketchyDataset
from src.configs.config import get_default_config

class Stage2ComprehensiveValidator:
    def __init__(self, stage1_ckpt, stage2_ckpt, num_samples=100, device='cuda', output_dir='validation_stage2'):
        self.device = device
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stage1_ckpt = stage1_ckpt
        self.stage2_ckpt = stage2_ckpt
        
        self.config = get_default_config()
        self.load_models()
        self.load_dataset()
        
        if LPIPS_AVAILABLE:
            print("📦 Loading LPIPS model...")
            self.lpips_model = LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

    def load_models(self):
        print("📦 Loading models...")
        model_name = self.config['model'].pretrained_model_name
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device).eval()
        
        # Load Text Encoder & Tokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        
        # Load Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        # Load Stage 1
        print(f"   Loading Stage 1 from {self.stage1_ckpt}...")
        self.stage1 = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            sketch_encoder_channels=self.config['model'].sketch_encoder_channels,
            use_lora=True,
            lora_rank=8
        ).to(self.device).eval()
        
        s1_checkpoint = torch.load(self.stage1_ckpt, map_location=self.device, weights_only=False)
        self.stage1.load_state_dict(s1_checkpoint['model_state_dict'], strict=False)
        
        # Load Stage 2
        print(f"   Loading Stage 2 from {self.stage2_ckpt}...")
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(self.device)
        self.stage2 = Stage2SemanticRefinement(
            unet=unet,
            node_feature_dim=self.config['model'].node_feature_dim,
            text_dim=self.config['model'].text_dim,
            hidden_dim=self.config['model'].hidden_dim
        ).to(self.device).eval()
        
        s2_checkpoint = torch.load(self.stage2_ckpt, map_location=self.device, weights_only=False)
        self.stage2.load_state_dict(s2_checkpoint['model_state_dict'], strict=False)
        self.epoch = s2_checkpoint.get('epoch', 'unknown')
        
        print("✅ All models loaded successfully\n")

    def load_dataset(self):
        print("📁 Loading Sketchy dataset...")
        self.dataset = SketchyDataset(
            root_dir=self.config['data'].sketchy_root,
            split='test',
            image_size=self.config['data'].image_size,
            augment=False
        )
        print(f"   Total test samples: {len(self.dataset)}")
        self.num_samples = min(self.num_samples, len(self.dataset))
        print(f"   Evaluating on {self.num_samples} samples\n")

    def tensor_to_numpy(self, tensor):
        """Convert [-1, 1] tensor to [0, 255] numpy array."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return img

    @torch.no_grad()
    def validate(self, save_examples=10):
        print(f"🚀 Starting Stage 2 validation (Epoch {self.epoch})...")
        
        metrics = {
            "ssim": [],
            "psnr": [],
            "lpips": []
        }
        category_metrics = defaultdict(lambda: {"ssim": [], "psnr": [], "lpips": [], "count": 0})
        
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        indices = torch.randperm(len(self.dataset))[:self.num_samples]
        
        example_dir = self.output_dir / "examples"
        example_dir.mkdir(exist_ok=True)
        
        for i, idx in enumerate(tqdm(indices, desc="Validating")):
            sample = self.dataset[idx]
            sketch = sample['sketch'].unsqueeze(0).to(self.device)
            photo = sample['photo'].unsqueeze(0).to(self.device)
            prompt = sample['text_prompt']
            category = sample.get('category', 'unknown')
            
            try:
                # 1. Stage 1 Generation
                sketch_features = self.stage1.encode_sketch(sketch)
                text_emb_s1 = self.stage1.encode_text([prompt])
                
                latent = torch.randn(1, 4, 32, 32, device=self.device)
                self.scheduler.set_timesteps(50)
                
                for t in self.scheduler.timesteps:
                    t_tensor = torch.tensor([t], device=self.device)
                    noise_pred = self.stage1(latent, t_tensor, sketch_features, text_emb_s1)
                    latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                
                stage1_img = self.vae.decode(latent / 0.18215).sample
                stage1_img = stage1_img.clamp(-1, 1)
                
                # 2. Stage 2 Refinement
                # For validation, we'll use a simplified refinement: 
                # Add noise to Stage 1 output and denoise for a few steps
                refinement_steps = 20
                strength = 0.5
                self.scheduler.set_timesteps(50)
                
                # Encode stage 1 output to latents
                s1_latent = self.vae.encode(stage1_img).latent_dist.sample() * 0.18215
                
                # Determine starting timestep
                t_start = int(50 * strength)
                timesteps = self.scheduler.timesteps[-t_start:]
                
                # Add noise
                noise = torch.randn_like(s1_latent)
                latent = self.scheduler.add_noise(s1_latent, noise, timesteps[0])
                
                # Text embeddings for Stage 2
                text_inputs = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt").to(self.device)
                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                
                region_graph = sample.get('region_graph', None)
                
                # Refinement loop
                for t in timesteps:
                    t_tensor = torch.tensor([t], device=self.device)
                    noise_pred = self.stage2(latent, t_tensor, region_graph, text_embeddings)
                    latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                
                final_img = self.vae.decode(latent / 0.18215).sample
                final_img = final_img.clamp(-1, 1)
                
                # 3. Compute Metrics
                gen_np = self.tensor_to_numpy(final_img)
                gt_np = self.tensor_to_numpy(photo)
                s1_np = self.tensor_to_numpy(stage1_img)
                
                ssim_val = ssim(gen_np, gt_np, channel_axis=2)
                psnr_val = psnr(gt_np, gen_np)
                
                if self.lpips_model:
                    lpips_val = self.lpips_model(final_img, photo).item()
                else:
                    lpips_val = 0.0
                
                # Store metrics
                metrics["ssim"].append(ssim_val)
                metrics["psnr"].append(psnr_val)
                metrics["lpips"].append(lpips_val)
                
                category_metrics[category]["ssim"].append(ssim_val)
                category_metrics[category]["psnr"].append(psnr_val)
                category_metrics[category]["lpips"].append(lpips_val)
                category_metrics[category]["count"] += 1
                
                # 4. Save Examples
                if i < save_examples:
                    self.save_comparison(
                        sketch[0, 0], s1_np, gen_np, gt_np, 
                        prompt, category, i, 
                        ssim_val, psnr_val, lpips_val,
                        example_dir
                    )
                    
            except Exception as e:
                print(f"⚠️ Error processing sample {idx}: {e}")
                continue

        # 5. Compute Statistics
        results = self.compute_statistics(metrics, category_metrics)
        
        # Add metadata
        results["metadata"] = {
            "num_samples": len(metrics["ssim"]),
            "checkpoint": str(self.stage2_ckpt),
            "epoch": self.epoch,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        output_file = self.output_dir / f"stage2_validation_epoch_{self.epoch}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n✅ Validation complete. Results saved to {output_file}")
        self.print_summary(results)
        
        return results

    def save_comparison(self, sketch, s1, s2, gt, prompt, category, idx, ssim_v, psnr_v, lpips_v, output_dir):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Sketch (invert for better visibility)
        sketch_np = sketch.cpu().numpy()
        axes[0].imshow(sketch_np, cmap='gray_r')
        axes[0].set_title("Input Sketch")
        axes[0].axis('off')
        
        # Stage 1
        axes[1].imshow(s1)
        axes[1].set_title("Stage 1 Output")
        axes[1].axis('off')
        
        # Stage 2
        axes[2].imshow(s2)
        axes[2].set_title(f"Stage 2 Refined\nSSIM: {ssim_v:.3f}, PSNR: {psnr_v:.1f}")
        axes[2].axis('off')
        
        # Ground Truth
        axes[3].imshow(gt)
        axes[3].set_title(f"Ground Truth\nCategory: {category}")
        axes[3].axis('off')
        
        plt.suptitle(f"Prompt: {prompt}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"example_{idx:03d}_{category}.png", dpi=150)
        plt.close()

    def compute_statistics(self, metrics, category_metrics):
        def get_stats(scores):
            if not scores: return {}
            return {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75))
            }
        
        overall = {
            "ssim": get_stats(metrics["ssim"]),
            "psnr": get_stats(metrics["psnr"]),
            "lpips": get_stats(metrics["lpips"])
        }
        
        cat_wise = {}
        for cat, scores in category_metrics.items():
            cat_wise[cat] = {
                "ssim_mean": float(np.mean(scores["ssim"])),
                "psnr_mean": float(np.mean(scores["psnr"])),
                "lpips_mean": float(np.mean(scores["lpips"])),
                "num_samples": scores["count"]
            }
            
        return {
            "overall": overall,
            "category_wise": cat_wise
        }

    def print_summary(self, results):
        print("\n" + "="*50)
        print("STAGE 2 VALIDATION SUMMARY")
        print("="*50)
        o = results["overall"]
        print(f"SSIM:  {o['ssim']['mean']:.4f} ± {o['ssim']['std']:.4f}")
        print(f"PSNR:  {o['psnr']['mean']:.2f} ± {o['psnr']['std']:.2f} dB")
        if LPIPS_AVAILABLE:
            print(f"LPIPS: {o['lpips']['mean']:.4f} ± {o['lpips']['std']:.4f}")
        print("-" * 50)
        print(f"Categories evaluated: {len(results['category_wise'])}")
        # Print top 5 categories by SSIM
        sorted_cats = sorted(results['category_wise'].items(), key=lambda x: x[1]['ssim_mean'], reverse=True)
        print("\nTop 5 Categories (by SSIM):")
        for cat, stats in sorted_cats[:5]:
            print(f"  {cat:15s}: SSIM={stats['ssim_mean']:.4f}, PSNR={stats['psnr_mean']:.2f}")
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Stage 2 Validation")
    parser.add_argument("--stage1", type=str, default="./checkpoints/stage1_with_ssim/epoch_18.pt", help="Stage 1 checkpoint")
    parser.add_argument("--stage2", type=str, required=True, help="Stage 2 checkpoint")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output", type=str, default="validation_stage2_results", help="Output directory")
    
    args = parser.parse_args()
    
    validator = Stage2ComprehensiveValidator(
        stage1_ckpt=args.stage1,
        stage2_ckpt=args.stage2,
        num_samples=args.samples,
        device=args.device,
        output_dir=args.output
    )
    
    validator.validate()

if __name__ == "__main__":
    main()
