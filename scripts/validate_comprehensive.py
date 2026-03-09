#!/usr/bin/env python3
"""
Comprehensive Validation Script
Run detailed validation on 1000+ samples to get reliable metrics.
Can be run while training is in progress (uses separate process).
"""

import sys
from pathlib import Path
# Add project root and src to Python path
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

# Models and data
from models.stage1_diffusion import Stage1SketchGuidedDiffusion
from datasets.sketchy_dataset import SketchyDataset
from configs.config import get_default_config
from diffusers import AutoencoderKL, DDIMScheduler
from lpips import LPIPS


class ComprehensiveValidator:
    def __init__(self, checkpoint_path, num_samples=1000, device='cuda'):
        self.checkpoint_path = Path(checkpoint_path)
        self.num_samples = num_samples
        self.device = device
        
        print(f"🔍 Comprehensive Validation")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Samples: {num_samples}")
        print(f"   Device: {device}")
        print()
        
        # Load model
        self.load_model()
        
        # Load dataset
        self.load_dataset()
        
        # Setup metrics
        self.lpips_model = LPIPS(net='alex').to(device)
        self.lpips_model.eval()
        
    def load_model(self):
        """Load model from checkpoint."""
        print("📦 Loading model...")
        
        config = get_default_config()
        model_name = config['model'].pretrained_model_name
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        
        # Load Stage1 model (on CPU first to avoid OOM)
        self.model = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            sketch_encoder_channels=config['model'].sketch_encoder_channels,
            freeze_base_unet=False,
            use_lora=True,
            lora_rank=8
        )
        
        # Load checkpoint (always load to CPU first, then move)
        print(f"   Loading checkpoint from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device after loading
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"   ✅ Loaded checkpoint from Epoch {self.epoch}")
        
        # Setup scheduler
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler"
        )
        
    def load_dataset(self):
        """Load validation dataset."""
        print("\n📁 Loading validation dataset...")
        
        config = get_default_config()
        self.val_dataset = SketchyDataset(
            root_dir=config['data'].sketchy_root,
            split='test',
            image_size=config['data'].image_size,
            augment=False
        )
        
        print(f"   Total validation samples: {len(self.val_dataset)}")
        print(f"   Will evaluate: {min(self.num_samples, len(self.val_dataset))} samples")
        print(f"   Coverage: {100 * min(self.num_samples, len(self.val_dataset)) / len(self.val_dataset):.2f}%")
        
    @torch.no_grad()
    def validate(self, save_examples=5):
        """Run comprehensive validation."""
        print(f"\n🚀 Starting validation...")
        print(f"   Using DDIM sampling with 50 steps")
        print()
        
        # Metrics storage
        ssim_scores = []
        psnr_scores = []
        lpips_scores = []
        
        # Category-wise metrics
        category_metrics = {}
        
        # Limit to available samples
        num_samples = min(self.num_samples, len(self.val_dataset))
        
        # Random sampling (reproducible)
        torch.manual_seed(42)
        indices = torch.randperm(len(self.val_dataset))[:num_samples]
        
        # Create output directory for examples
        output_dir = Path("validation_examples")
        output_dir.mkdir(exist_ok=True)
        
        # Progress bar
        for idx_num, idx in enumerate(tqdm(indices, desc="Validating")):
            sample = self.val_dataset[idx]
            sketch = sample['sketch'].unsqueeze(0).to(self.device)
            photo = sample['photo']
            prompt = sample['text_prompt']
            category = sample.get('category', 'unknown')
            
            # Generate image
            generated = self.generate_image(sketch, prompt)
            
            # Convert to numpy for metrics
            gen_np = self.tensor_to_numpy(generated)
            photo_np = self.tensor_to_numpy(photo)
            
            # Compute PSNR
            psnr_val = psnr(photo_np, gen_np, data_range=255)
            psnr_scores.append(psnr_val)
            
            # Compute SSIM (grayscale)
            gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
            photo_gray = cv2.cvtColor(photo_np, cv2.COLOR_RGB2GRAY)
            ssim_val = ssim(photo_gray, gen_gray, data_range=255)
            ssim_scores.append(ssim_val)
            
            # Compute LPIPS
            gen_tensor = torch.from_numpy(gen_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 127.5 - 1
            photo_tensor = photo.unsqueeze(0).to(self.device)
            lpips_val = self.lpips_model(gen_tensor, photo_tensor).item()
            lpips_scores.append(lpips_val)
            
            # Track category-wise metrics
            if category not in category_metrics:
                category_metrics[category] = {'ssim': [], 'psnr': [], 'lpips': []}
            category_metrics[category]['ssim'].append(ssim_val)
            category_metrics[category]['psnr'].append(psnr_val)
            category_metrics[category]['lpips'].append(lpips_val)
            
            # Save example images
            if idx_num < save_examples:
                self.save_example(
                    sketch[0], gen_np, photo_np, 
                    prompt, idx_num, ssim_val, psnr_val, lpips_val,
                    output_dir
                )
        
        # Compute statistics
        results = self.compute_statistics(
            ssim_scores, psnr_scores, lpips_scores, category_metrics
        )
        
        # Save results
        self.save_results(results, output_dir)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def generate_image(self, sketch, prompt):
        """Generate image using DDIM sampling."""
        # Encode inputs
        sketch_features = self.model.encode_sketch(sketch)
        text_embeddings = self.model.encode_text([prompt])
        
        # Initialize latent
        latent = torch.randn(1, 4, 32, 32, device=self.device)
        
        # Setup DDIM scheduler
        self.noise_scheduler.set_timesteps(50)
        
        # Denoise
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.tensor([t], device=self.device)
            noise_pred = self.model(latent, timesteps, sketch_features, text_embeddings)
            latent = self.noise_scheduler.step(noise_pred, t, latent).prev_sample
        
        # Decode
        generated = self.vae.decode(latent / 0.18215).sample[0]
        return generated
    
    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy uint8 image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        np_img = tensor.cpu().numpy().transpose(1, 2, 0)
        np_img = ((np_img + 1) / 2 * 255).astype(np.uint8)
        return np_img
    
    def save_example(self, sketch, generated, photo, prompt, idx, ssim_val, psnr_val, lpips_val, output_dir):
        """Save example images."""
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sketch
        sketch_np = sketch[0].cpu().numpy()
        sketch_np = (sketch_np * 255).astype(np.uint8)
        axes[0].imshow(sketch_np, cmap='gray')
        axes[0].set_title('Input Sketch')
        axes[0].axis('off')
        
        # Generated
        axes[1].imshow(generated)
        axes[1].set_title(f'Generated\nSSIM: {ssim_val:.3f} | PSNR: {psnr_val:.1f}')
        axes[1].axis('off')
        
        # Ground truth
        axes[2].imshow(photo)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        # Add prompt as suptitle
        plt.suptitle(f'"{prompt}"', fontsize=10)
        plt.tight_layout()
        
        # Save
        save_path = output_dir / f"example_{idx:03d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def compute_statistics(self, ssim_scores, psnr_scores, lpips_scores, category_metrics):
        """Compute comprehensive statistics."""
        results = {
            'overall': {
                'ssim': {
                    'mean': float(np.mean(ssim_scores)),
                    'std': float(np.std(ssim_scores)),
                    'median': float(np.median(ssim_scores)),
                    'min': float(np.min(ssim_scores)),
                    'max': float(np.max(ssim_scores)),
                    'q25': float(np.percentile(ssim_scores, 25)),
                    'q75': float(np.percentile(ssim_scores, 75))
                },
                'psnr': {
                    'mean': float(np.mean(psnr_scores)),
                    'std': float(np.std(psnr_scores)),
                    'median': float(np.median(psnr_scores)),
                    'min': float(np.min(psnr_scores)),
                    'max': float(np.max(psnr_scores)),
                    'q25': float(np.percentile(psnr_scores, 25)),
                    'q75': float(np.percentile(psnr_scores, 75))
                },
                'lpips': {
                    'mean': float(np.mean(lpips_scores)),
                    'std': float(np.std(lpips_scores)),
                    'median': float(np.median(lpips_scores)),
                    'min': float(np.min(lpips_scores)),
                    'max': float(np.max(lpips_scores)),
                    'q25': float(np.percentile(lpips_scores, 25)),
                    'q75': float(np.percentile(lpips_scores, 75))
                }
            },
            'category_wise': {},
            'metadata': {
                'num_samples': len(ssim_scores),
                'checkpoint': str(self.checkpoint_path),
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Category-wise statistics (top 10 categories by sample count)
        sorted_categories = sorted(
            category_metrics.items(), 
            key=lambda x: len(x[1]['ssim']), 
            reverse=True
        )[:10]
        
        for category, metrics in sorted_categories:
            results['category_wise'][category] = {
                'ssim_mean': float(np.mean(metrics['ssim'])),
                'psnr_mean': float(np.mean(metrics['psnr'])),
                'lpips_mean': float(np.mean(metrics['lpips'])),
                'num_samples': len(metrics['ssim'])
            }
        
        return results
    
    def save_results(self, results, output_dir):
        """Save results to JSON."""
        output_path = output_dir / f"validation_results_epoch{self.epoch}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    def print_summary(self, results):
        """Print comprehensive summary."""
        overall = results['overall']
        
        print("\n" + "="*70)
        print(f"📊 COMPREHENSIVE VALIDATION RESULTS - Epoch {self.epoch}")
        print("="*70)
        
        print(f"\n📈 Overall Metrics ({results['metadata']['num_samples']} samples):")
        print(f"\n   SSIM:")
        print(f"      Mean:   {overall['ssim']['mean']:.4f} ± {overall['ssim']['std']:.4f}")
        print(f"      Median: {overall['ssim']['median']:.4f}")
        print(f"      Range:  [{overall['ssim']['min']:.4f}, {overall['ssim']['max']:.4f}]")
        print(f"      IQR:    [{overall['ssim']['q25']:.4f}, {overall['ssim']['q75']:.4f}]")
        
        print(f"\n   PSNR:")
        print(f"      Mean:   {overall['psnr']['mean']:.2f} ± {overall['psnr']['std']:.2f} dB")
        print(f"      Median: {overall['psnr']['median']:.2f} dB")
        print(f"      Range:  [{overall['psnr']['min']:.2f}, {overall['psnr']['max']:.2f}] dB")
        print(f"      IQR:    [{overall['psnr']['q25']:.2f}, {overall['psnr']['q75']:.2f}] dB")
        
        print(f"\n   LPIPS:")
        print(f"      Mean:   {overall['lpips']['mean']:.4f} ± {overall['lpips']['std']:.4f}")
        print(f"      Median: {overall['lpips']['median']:.4f}")
        print(f"      Range:  [{overall['lpips']['min']:.4f}, {overall['lpips']['max']:.4f}]")
        print(f"      IQR:    [{overall['lpips']['q25']:.4f}, {overall['lpips']['q75']:.4f}]")
        
        if results['category_wise']:
            print(f"\n📊 Top Categories:")
            print(f"   {'Category':<20} {'SSIM':<12} {'PSNR':<12} {'LPIPS':<12} {'Samples'}")
            print(f"   {'-'*70}")
            for category, metrics in list(results['category_wise'].items())[:10]:
                print(f"   {category:<20} "
                      f"{metrics['ssim_mean']:>6.4f}      "
                      f"{metrics['psnr_mean']:>6.2f} dB    "
                      f"{metrics['lpips_mean']:>6.4f}      "
                      f"{metrics['num_samples']:>3}")
        
        print("\n" + "="*70)
        print(f"✅ Validation complete!")
        print(f"📁 Examples saved in: validation_examples/")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive validation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to validate (default: 1000)')
    parser.add_argument('--save_examples', type=int, default=10,
                        help='Number of example images to save (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Run validation
    validator = ComprehensiveValidator(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        device=args.device
    )
    
    results = validator.validate(save_examples=args.save_examples)


if __name__ == '__main__':
    main()
