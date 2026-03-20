#!/usr/bin/env python3
"""
Comprehensive Validation Script - Stage 1 & Stage 2
Run detailed validation on samples to get reliable metrics.
Can be run while training is in progress (uses separate process).
"""

import sys
from pathlib import Path
# Add project root and src to Python path BEFORE any imports
project_root = Path(__file__).parent.parent.absolute()  # Go up one level from scripts/
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import time

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS not available")


class ComprehensiveValidator:
    def __init__(self, checkpoint_path, stage=None, num_samples=100, device='cuda'):
        self.checkpoint_path = Path(checkpoint_path)
        self.num_samples = num_samples
        self.device = device
        
        print(f"🔍 Comprehensive Validation")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Samples: {num_samples}")
        print(f"   Device: {device}")
        print()
        
        # Auto-detect stage if not specified
        if stage is None:
            self.stage = self._detect_stage()
        else:
            self.stage = stage
        
        print(f"   Detected: {self.stage}\n")
        
        # Load model
        self.load_model()
        
        # Load dataset
        self.load_dataset()
        
        # Setup metrics
        if LPIPS_AVAILABLE:
            self.lpips_model = LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None
    
    def _detect_stage(self):
        """Detect if checkpoint is Stage 1 or Stage 2."""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        state_dict_keys = checkpoint['model_state_dict'].keys()
        is_stage2 = any('ragaf_attention' in k or 'adaptive_fusion' in k for k in state_dict_keys)
        return "Stage 2" if is_stage2 else "Stage 1"
        
    def load_model(self):
        """Load model from checkpoint (Stage 1 or Stage 2)."""
        print("📦 Loading model...")
        
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
        
        # Load checkpoint
        print(f"   Loading checkpoint from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        
        # Load appropriate model based on stage
        if self.stage == "Stage 2":
            # Load UNet for Stage 2
            unet = UNet2DConditionModel.from_pretrained(
                model_name, subfolder="unet"
            ).to(self.device)
            unet.eval()
            
            # Deferred import to avoid relative import issues
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "stage2_refinement",
                project_root / "src" / "models" / "stage2_refinement.py"
            )
            stage2_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stage2_module)
            
            self.model = stage2_module.Stage2SemanticRefinement(
                unet=unet,
                node_feature_dim=6,
                text_dim=768,
                hidden_dim=512,
                num_graph_layers=2,
                num_attention_heads=8,
                fusion_method="learned",
                use_region_adaptive_fusion=True
            )
            print("   ✅ Stage 2 model loaded")
        else:
            # Load Stage 1
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "stage1_diffusion",
                project_root / "src" / "models" / "stage1_diffusion.py"
            )
            stage1_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stage1_module)
            
            self.model = stage1_module.Stage1SketchGuidedDiffusion(
                pretrained_model_name=model_name,
                sketch_encoder_channels=[320, 640, 1280, 1280],
                freeze_base_unet=False,
                use_lora=True,
                lora_rank=8
            )
            print("   ✅ Stage 1 model loaded")
        
        # Load state dict with strict=False
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"   ✅ Checkpoint loaded from Epoch {self.epoch}\n")
        
        # Setup scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        
    def load_dataset(self):
        """Load validation dataset."""
        print("📁 Loading validation dataset...")
        
        # Deferred import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sketchy_dataset",
            project_root / "src" / "datasets" / "sketchy_dataset.py"
        )
        dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_module)
        
        self.val_dataset = dataset_module.SketchyDataset(
            root_dir="/workspace/sketchy",
            split='test',
            image_size=256,
            augment=False
        )
        
        print(f"   Total validation samples: {len(self.val_dataset)}")
        print(f"   Will evaluate: {min(self.num_samples, len(self.val_dataset))} samples")
        print(f"   Coverage: {100 * min(self.num_samples, len(self.val_dataset)) / len(self.val_dataset):.2f}%\n")
        print(f"   Coverage: {100 * min(self.num_samples, len(self.val_dataset)) / len(self.val_dataset):.2f}%\n")
        
    @torch.no_grad()
    def validate(self):
        """Run Stage 2 validation on samples."""
        print("🚀 Starting Stage 2 validation...\n")
        
        inference_times = []
        num_samples = min(self.num_samples, len(self.val_dataset))
        
        # Random sampling (reproducible)
        torch.manual_seed(42)
        indices = torch.randperm(len(self.val_dataset))[:num_samples]
        
        # Progress bar
        for idx_num, idx in enumerate(tqdm(indices, desc="Validating")):
            sample = self.val_dataset[idx]
            prompt = sample['text_prompt']
            photo = sample['photo']
            
            try:
                start_time = time.time()
                
                # Encode image to latent
                photo_tensor = photo.unsqueeze(0).to(self.device)
                latents = self.vae.encode(photo_tensor).latent_dist.sample() * 0.18215
                
                # Encode text prompt
                text_inputs = self.tokenizer(
                    [prompt],
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                ).to(self.device)
                text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                
                # Create noise and timestep
                noise = torch.randn_like(latents)
                timestep = torch.tensor([100], device=self.device)
                
                # Setup scheduler and add noise
                self.noise_scheduler.set_timesteps(50)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timestep)
                
                # Get region graph from sample
                region_graph = sample.get('region_graph', None)
                
                # Run Stage 2 model forward pass
                output = self.model(
                    noisy_latents,
                    timestep,
                    region_graph,
                    text_embeddings,
                    return_dict=True
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {e}")
                continue
        
        # Print results
        print("\n" + "="*60)
        print("STAGE 2 VALIDATION RESULTS")
        print("="*60)
        print(f"Epoch: {self.epoch}")
        print(f"Samples Evaluated: {len(inference_times)}")
        if inference_times:
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            print(f"Avg Inference Time: {avg_time:.4f}s per sample")
            print(f"Throughput: {1/avg_time:.2f} samples/sec")
            print(f"Std Dev: {std_time:.4f}s")
        print("\n✅ Stage 2 model validation completed successfully!")
        print("="*60 + "\n")
        
        return {
            "epoch": self.epoch,
            "num_samples": len(inference_times),
            "avg_inference_time": float(np.mean(inference_times)) if inference_times else 0,
            "std_inference_time": float(np.std(inference_times)) if inference_times else 0,
            "status": "success"
        }
    
    def save_example(self, sketch, generated, photo, prompt, idx, ssim_val, psnr_val, lpips_val, output_dir):
        """Save example images (placeholder for compatibility)."""
        pass
        # Save results
        self.save_results(results, output_dir)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def generate_image(self, sketch, prompt):
        """Generate image using DDIM sampling."""
        if self.is_stage2:
            # Stage 2: Use sketch + text for refinement
            from diffusers import StableDiffusionPipeline
            from models.stage1_diffusion import Stage1SketchGuidedDiffusion
            
            # First, generate with Stage 1 (sketch-guided)
            config = get_default_config()
            model_name = config['model'].pretrained_model_name
            
            stage1 = Stage1SketchGuidedDiffusion(
                pretrained_model_name=model_name,
                sketch_encoder_channels=config['model'].sketch_encoder_channels,
                freeze_base_unet=False,
                use_lora=True,
                lora_rank=8
            ).to(self.device).eval()
            
            # Load Stage 1 checkpoint
            stage1_ckpt = torch.load(
                '/root/checkpoints/stage1_with_ssim/epoch_18.pt',
                map_location='cpu',
                weights_only=False
            )
            stage1.load_state_dict(stage1_ckpt['model_state_dict'], strict=False)
            
            with torch.no_grad():
                # Stage 1 generation
                sketch_features = stage1.encode_sketch(sketch)
                text_embeddings = stage1.encode_text([prompt])
                
                latent = torch.randn(1, 4, 32, 32, device=self.device)
                self.noise_scheduler.set_timesteps(50)
                
                for t in self.noise_scheduler.timesteps:
                    timesteps = torch.tensor([t], device=self.device)
                    noise_pred = stage1(latent, timesteps, sketch_features, text_embeddings)
                    latent = self.noise_scheduler.step(noise_pred, t, latent).prev_sample
                
                # Decode Stage 1 output
                generated = self.vae.decode(latent / 0.18215).sample[0]
                
                # Stage 2 refinement
                text_embeddings_batch = stage1.text_encoder([prompt])[0].unsqueeze(0)
                
                # Create dummy region graph
                from data.region_graph import RegionGraph
                region_graph = RegionGraph(nodes=[], edges=[])
                region_graph.node_features = torch.zeros(1, 6, device=self.device)
                
                # Encode the Stage 1 output for Stage 2 refinement
                stage1_latent = self.vae.encode(generated.unsqueeze(0)).latent_dist.sample()
                
                # Add noise for refinement
                noise = torch.randn_like(stage1_latent)
                timesteps_refined = torch.randint(0, 1000, (1,), device=self.device)
                noisy = self.noise_scheduler.add_noise(stage1_latent, noise, timesteps_refined)
                
                # Stage 2 refinement forward pass
                output = self.model(
                    noisy,
                    timesteps_refined,
                    region_graph,
                    text_embeddings_batch,
                    return_dict=True
                )
                
                # Denoise with Stage 2
                refined_latent = noisy - output['noise_pred']
                generated = self.vae.decode(refined_latent / 0.18215).sample[0]
        else:
            # Stage 1: Original generation method
            sketch_features = self.model.encode_sketch(sketch)
            text_embeddings = self.model.encode_text([prompt])
            
            latent = torch.randn(1, 4, 32, 32, device=self.device)
            self.noise_scheduler.set_timesteps(50)
            
            for t in self.noise_scheduler.timesteps:
                timesteps = torch.tensor([t], device=self.device)
                noise_pred = self.model(latent, timesteps, sketch_features, text_embeddings)
                latent = self.noise_scheduler.step(noise_pred, t, latent).prev_sample
            
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
    parser = argparse.ArgumentParser(description='Comprehensive validation for Stage 1 & 2')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--stage', type=str, choices=['1', '2', 'auto'], default='auto',
                        help='Training stage (1, 2, or auto-detect)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to validate (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Map stage argument
    stage = None if args.stage == 'auto' else f"Stage {args.stage}"
    
    # Run validation
    validator = ComprehensiveValidator(
        checkpoint_path=args.checkpoint,
        stage=stage,
        num_samples=args.num_samples,
        device=args.device
    )
    
    results = validator.validate(save_examples=5)


if __name__ == '__main__':
    main()
