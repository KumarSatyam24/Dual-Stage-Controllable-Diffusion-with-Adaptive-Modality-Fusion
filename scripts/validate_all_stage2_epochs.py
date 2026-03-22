#!/usr/bin/env python3
"""
Validate All Stage 2 Epochs
Compares Stage 1 vs Stage 2 refinement across all epochs to find the best checkpoint.
Generates images for the same classes as Stage 1 validation for direct comparison.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add project root to path
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


# Same classes as Stage 1 validation
TARGET_CLASSES = [
    'couch', 'chicken', 'saw', 'seagull', 'rifle',
    'elephant', 'sheep', 'dog', 'snake', 'zebra'
]


class Stage2EpochComparator:
    def __init__(self, stage1_ckpt, stage2_ckpts, num_samples_per_class=25, device='cuda', output_dir='stage2_comparison'):
        self.device = device
        self.num_samples_per_class = num_samples_per_class
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stage1_ckpt = stage1_ckpt
        self.stage2_ckpts = stage2_ckpts

        self.config = get_default_config()

        # Load common components
        self.load_common_models()

        # Load dataset
        self.load_dataset()

        # LPIPS
        if LPIPS_AVAILABLE:
            print("📦 Loading LPIPS model...")
            self.lpips_model = LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

    def load_common_models(self):
        """Load models that are shared across all epochs."""
        print("📦 Loading common models...")
        model_name = self.config['model'].pretrained_model_name

        # VAE
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device).eval()

        # Text Encoder & Tokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        # Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        # Stage 1 Model
        print(f"   Loading Stage 1 from {self.stage1_ckpt}...")
        self.stage1 = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            sketch_encoder_channels=self.config['model'].sketch_encoder_channels,
            use_lora=True,
            lora_rank=8
        ).to(self.device).eval()

        s1_checkpoint = torch.load(self.stage1_ckpt, map_location=self.device, weights_only=False)
        self.stage1.load_state_dict(s1_checkpoint['model_state_dict'], strict=False)

        print("✅ Common models loaded\n")

    def load_dataset(self):
        """Load dataset and filter by target classes."""
        print("📁 Loading Sketchy dataset...")
        self.dataset = SketchyDataset(
            root_dir=self.config['data'].sketchy_root,
            split='test',
            image_size=self.config['data'].image_size,
            augment=False
        )

        # Group samples by category
        self.samples_by_class = defaultdict(list)
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            category = sample.get('category', 'unknown')
            if category in TARGET_CLASSES:
                self.samples_by_class[category].append(idx)

        print(f"   Found samples for classes:")
        for cls in TARGET_CLASSES:
            print(f"      {cls}: {len(self.samples_by_class[cls])} samples")
        print()

    def load_stage2_model(self, ckpt_path):
        """Load a specific Stage 2 checkpoint."""
        model_name = self.config['model'].pretrained_model_name
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(self.device)

        stage2 = Stage2SemanticRefinement(
            unet=unet,
            node_feature_dim=self.config['model'].node_feature_dim,
            text_dim=self.config['model'].text_dim,
            hidden_dim=self.config['model'].hidden_dim
        ).to(self.device).eval()

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        stage2.load_state_dict(checkpoint['model_state_dict'], strict=False)
        epoch = checkpoint.get('epoch', 'unknown')

        return stage2, epoch

    def tensor_to_numpy(self, tensor):
        """Convert [-1, 1] tensor to [0, 255] numpy array."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return img

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image."""
        return Image.fromarray(self.tensor_to_numpy(tensor))

    @torch.no_grad()
    def generate_stage1(self, sketch, prompt):
        """Generate image using Stage 1."""
        sketch_features = self.stage1.encode_sketch(sketch)
        text_emb = self.stage1.encode_text([prompt])

        latent = torch.randn(1, 4, 32, 32, device=self.device, generator=torch.Generator(device=self.device).manual_seed(42))
        self.scheduler.set_timesteps(50)

        for t in self.scheduler.timesteps:
            t_tensor = torch.tensor([t], device=self.device)
            noise_pred = self.stage1(latent, t_tensor, sketch_features, text_emb)
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        img = self.vae.decode(latent / 0.18215).sample.clamp(-1, 1)
        return img

    @torch.no_grad()
    def refine_stage2(self, stage1_img, prompt, region_graph, stage2_model):
        """Refine Stage 1 output using Stage 2."""
        # Encode to latents
        s1_latent = self.vae.encode(stage1_img).latent_dist.sample() * 0.18215

        # Refinement parameters
        refinement_steps = 30
        strength = 0.5
        self.scheduler.set_timesteps(50)

        # Starting timestep
        t_start = int(50 * strength)
        timesteps = self.scheduler.timesteps[-t_start:]

        # Add noise
        noise = torch.randn_like(s1_latent, generator=torch.Generator(device=self.device).manual_seed(42))
        latent = self.scheduler.add_noise(s1_latent, noise, timesteps[0])

        # Text embeddings
        text_inputs = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt").to(self.device)
        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

        # Refinement loop
        for t in timesteps:
            t_tensor = torch.tensor([t], device=self.device)
            noise_pred = stage2_model(latent, t_tensor, region_graph, text_embeddings)
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        refined = self.vae.decode(latent / 0.18215).sample.clamp(-1, 1)
        return refined

    def compute_metrics(self, gen_img, gt_img):
        """Compute SSIM, PSNR, LPIPS."""
        gen_np = self.tensor_to_numpy(gen_img)
        gt_np = self.tensor_to_numpy(gt_img)

        ssim_val = ssim(gen_np, gt_np, channel_axis=2, data_range=255)
        psnr_val = psnr(gt_np, gen_np, data_range=255)

        if self.lpips_model:
            lpips_val = self.lpips_model(gen_img, gt_img).item()
        else:
            lpips_val = 0.0

        return {"ssim": ssim_val, "psnr": psnr_val, "lpips": lpips_val}

    def save_comparison_grid(self, sketch, stage1_img, stage2_img, gt_img, prompt, category, sample_idx, metrics_s1, metrics_s2, epoch, save_dir):
        """Save a comparison grid: Sketch | Stage1 | Stage2 | GT."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Sketch
        sketch_np = sketch.detach().cpu().numpy()
        axes[0].imshow(sketch_np, cmap='gray')
        axes[0].set_title('Sketch Input', fontsize=10)
        axes[0].axis('off')

        # Stage 1
        s1_np = self.tensor_to_numpy(stage1_img)
        axes[1].imshow(s1_np)
        axes[1].set_title(f'Stage 1\nSSIM: {metrics_s1["ssim"]:.3f} | PSNR: {metrics_s1["psnr"]:.2f}', fontsize=9)
        axes[1].axis('off')

        # Stage 2
        s2_np = self.tensor_to_numpy(stage2_img)
        axes[2].imshow(s2_np)
        axes[2].set_title(f'Stage 2 (Epoch {epoch})\nSSIM: {metrics_s2["ssim"]:.3f} | PSNR: {metrics_s2["psnr"]:.2f}', fontsize=9)
        axes[2].axis('off')

        # Ground Truth
        gt_np = self.tensor_to_numpy(gt_img)
        axes[3].imshow(gt_np)
        axes[3].set_title('Ground Truth', fontsize=10)
        axes[3].axis('off')

        plt.suptitle(f'{category}: {prompt}', fontsize=12, y=0.98)
        plt.tight_layout()

        save_path = save_dir / f'{category}_{sample_idx:03d}_epoch{epoch}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def validate_epoch(self, stage2_ckpt):
        """Validate a single Stage 2 epoch."""
        ckpt_name = Path(stage2_ckpt).stem
        print(f"\n{'='*80}")
        print(f"Validating: {ckpt_name}")
        print(f"{'='*80}\n")

        # Load Stage 2 model
        stage2_model, epoch = self.load_stage2_model(stage2_ckpt)

        # Create output directory
        epoch_dir = self.output_dir / f'epoch_{epoch}'
        epoch_dir.mkdir(exist_ok=True)
        examples_dir = epoch_dir / 'examples'
        examples_dir.mkdir(exist_ok=True)

        # Metrics storage
        all_metrics_s1 = {"ssim": [], "psnr": [], "lpips": []}
        all_metrics_s2 = {"ssim": [], "psnr": [], "lpips": []}
        category_metrics_s1 = defaultdict(lambda: {"ssim": [], "psnr": [], "lpips": []})
        category_metrics_s2 = defaultdict(lambda: {"ssim": [], "psnr": [], "lpips": []})

        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Process each class
        for category in TARGET_CLASSES:
            indices = self.samples_by_class[category]
            n_samples = min(self.num_samples_per_class, len(indices))

            # Use same random selection as Stage 1
            selected_indices = np.random.choice(indices, n_samples, replace=False)

            print(f"Processing {category}: {n_samples} samples...")

            for i, idx in enumerate(tqdm(selected_indices, desc=f"  {category}")):
                try:
                    sample = self.dataset[idx]
                    sketch = sample['sketch'].unsqueeze(0).to(self.device)
                    photo = sample['photo'].unsqueeze(0).to(self.device)
                    prompt = sample['text_prompt']
                    region_graph = sample.get('region_graph', None)

                    # Generate Stage 1
                    stage1_img = self.generate_stage1(sketch, prompt)

                    # Compute Stage 1 metrics
                    metrics_s1 = self.compute_metrics(stage1_img, photo)
                    all_metrics_s1["ssim"].append(metrics_s1["ssim"])
                    all_metrics_s1["psnr"].append(metrics_s1["psnr"])
                    all_metrics_s1["lpips"].append(metrics_s1["lpips"])
                    category_metrics_s1[category]["ssim"].append(metrics_s1["ssim"])
                    category_metrics_s1[category]["psnr"].append(metrics_s1["psnr"])
                    category_metrics_s1[category]["lpips"].append(metrics_s1["lpips"])

                    # Refine with Stage 2
                    stage2_img = self.refine_stage2(stage1_img, prompt, region_graph, stage2_model)

                    # Compute Stage 2 metrics
                    metrics_s2 = self.compute_metrics(stage2_img, photo)
                    all_metrics_s2["ssim"].append(metrics_s2["ssim"])
                    all_metrics_s2["psnr"].append(metrics_s2["psnr"])
                    all_metrics_s2["lpips"].append(metrics_s2["lpips"])
                    category_metrics_s2[category]["ssim"].append(metrics_s2["ssim"])
                    category_metrics_s2[category]["psnr"].append(metrics_s2["psnr"])
                    category_metrics_s2[category]["lpips"].append(metrics_s2["lpips"])

                    # Save comparison (first 5 per class)
                    if i < 5:
                        self.save_comparison_grid(
                            sketch[0, 0], stage1_img, stage2_img, photo,
                            prompt, category, i, metrics_s1, metrics_s2, epoch, examples_dir
                        )

                except Exception as e:
                    print(f"   ⚠️  Error processing {category} sample {i}: {e}")
                    continue

        # Compute statistics
        results = self.compute_statistics(all_metrics_s1, all_metrics_s2, category_metrics_s1, category_metrics_s2, epoch, stage2_ckpt)

        # Save results
        results_file = epoch_dir / f'validation_results_epoch_{epoch}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {results_file}")

        # Clean up
        del stage2_model
        torch.cuda.empty_cache()

        return results

    def compute_statistics(self, metrics_s1, metrics_s2, cat_metrics_s1, cat_metrics_s2, epoch, ckpt_path):
        """Compute overall and per-category statistics."""

        def stats_dict(values):
            if len(values) == 0:
                return {}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75))
            }

        results = {
            "stage1": {
                "overall": {
                    "ssim": stats_dict(metrics_s1["ssim"]),
                    "psnr": stats_dict(metrics_s1["psnr"]),
                    "lpips": stats_dict(metrics_s1["lpips"])
                },
                "category_wise": {}
            },
            "stage2": {
                "overall": {
                    "ssim": stats_dict(metrics_s2["ssim"]),
                    "psnr": stats_dict(metrics_s2["psnr"]),
                    "lpips": stats_dict(metrics_s2["lpips"])
                },
                "category_wise": {}
            },
            "improvement": {
                "ssim_improvement": float(np.mean(metrics_s2["ssim"]) - np.mean(metrics_s1["ssim"])),
                "psnr_improvement": float(np.mean(metrics_s2["psnr"]) - np.mean(metrics_s1["psnr"])),
                "lpips_improvement": float(np.mean(metrics_s1["lpips"]) - np.mean(metrics_s2["lpips"]))  # Lower is better
            },
            "metadata": {
                "num_samples": len(metrics_s1["ssim"]),
                "checkpoint": str(ckpt_path),
                "epoch": epoch,
                "timestamp": datetime.now().isoformat()
            }
        }

        # Category-wise stats
        for category in TARGET_CLASSES:
            if category in cat_metrics_s1:
                results["stage1"]["category_wise"][category] = {
                    "ssim_mean": float(np.mean(cat_metrics_s1[category]["ssim"])),
                    "psnr_mean": float(np.mean(cat_metrics_s1[category]["psnr"])),
                    "lpips_mean": float(np.mean(cat_metrics_s1[category]["lpips"])),
                    "num_samples": len(cat_metrics_s1[category]["ssim"])
                }

            if category in cat_metrics_s2:
                results["stage2"]["category_wise"][category] = {
                    "ssim_mean": float(np.mean(cat_metrics_s2[category]["ssim"])),
                    "psnr_mean": float(np.mean(cat_metrics_s2[category]["psnr"])),
                    "lpips_mean": float(np.mean(cat_metrics_s2[category]["lpips"])),
                    "num_samples": len(cat_metrics_s2[category]["ssim"])
                }

        return results

    def run_comparison(self):
        """Run validation on all Stage 2 checkpoints."""
        all_results = {}

        print(f"\n🚀 Starting Stage 2 Epoch Comparison")
        print(f"   Stage 1 checkpoint: {self.stage1_ckpt}")
        print(f"   Stage 2 checkpoints: {len(self.stage2_ckpts)}")
        print(f"   Samples per class: {self.num_samples_per_class}")
        print(f"   Target classes: {', '.join(TARGET_CLASSES)}\n")

        for ckpt in self.stage2_ckpts:
            results = self.validate_epoch(ckpt)
            epoch_name = Path(ckpt).stem
            all_results[epoch_name] = results

        # Create summary
        self.create_summary(all_results)

        return all_results

    def create_summary(self, all_results):
        """Create a summary comparing all epochs."""
        print(f"\n{'='*80}")
        print("SUMMARY: Stage 2 Epoch Comparison")
        print(f"{'='*80}\n")

        summary = []
        for epoch_name, results in all_results.items():
            epoch = results['metadata']['epoch']
            s1_ssim = results['stage1']['overall']['ssim']['mean']
            s2_ssim = results['stage2']['overall']['ssim']['mean']
            s1_psnr = results['stage1']['overall']['psnr']['mean']
            s2_psnr = results['stage2']['overall']['psnr']['mean']
            s1_lpips = results['stage1']['overall']['lpips']['mean']
            s2_lpips = results['stage2']['overall']['lpips']['mean']

            summary.append({
                'epoch': epoch,
                'stage1_ssim': s1_ssim,
                'stage2_ssim': s2_ssim,
                'ssim_improvement': s2_ssim - s1_ssim,
                'stage1_psnr': s1_psnr,
                'stage2_psnr': s2_psnr,
                'psnr_improvement': s2_psnr - s1_psnr,
                'stage1_lpips': s1_lpips,
                'stage2_lpips': s2_lpips,
                'lpips_improvement': s1_lpips - s2_lpips  # Lower LPIPS is better
            })

        # Sort by SSIM improvement
        summary.sort(key=lambda x: x['ssim_improvement'], reverse=True)

        print(f"{'Epoch':<10} {'S1 SSIM':<10} {'S2 SSIM':<10} {'Δ SSIM':<10} {'S1 PSNR':<10} {'S2 PSNR':<10} {'Δ PSNR':<10} {'S1 LPIPS':<10} {'S2 LPIPS':<10} {'Δ LPIPS':<10}")
        print("-" * 110)

        for item in summary:
            print(f"{item['epoch']:<10} {item['stage1_ssim']:<10.4f} {item['stage2_ssim']:<10.4f} {item['ssim_improvement']:<10.4f} "
                  f"{item['stage1_psnr']:<10.2f} {item['stage2_psnr']:<10.2f} {item['psnr_improvement']:<10.2f} "
                  f"{item['stage1_lpips']:<10.4f} {item['stage2_lpips']:<10.4f} {item['lpips_improvement']:<10.4f}")

        # Save summary
        summary_file = self.output_dir / 'comparison_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Summary saved to {summary_file}")

        # Print best epoch
        best = summary[0]
        print(f"\n🏆 Best Epoch: {best['epoch']}")
        print(f"   SSIM improvement: {best['ssim_improvement']:.4f}")
        print(f"   PSNR improvement: {best['psnr_improvement']:.2f}")
        print(f"   LPIPS improvement: {best['lpips_improvement']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Validate all Stage 2 epochs and compare with Stage 1")
    parser.add_argument("--stage1", type=str, default="./checkpoints/stage1_with_ssim/epoch_18.pt", help="Stage 1 checkpoint")
    parser.add_argument("--stage2-dir", type=str, default="./checkpoints/stage2", help="Directory containing Stage 2 checkpoints")
    parser.add_argument("--samples-per-class", type=int, default=25, help="Number of samples per class (default: 25)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="stage2_comparison_results", help="Output directory")

    args = parser.parse_args()

    # Find all Stage 2 checkpoints
    stage2_dir = Path(args.stage2_dir)
    stage2_ckpts = sorted(list(stage2_dir.glob("epoch_*.pt")) + list(stage2_dir.glob("final.pt")))

    if len(stage2_ckpts) == 0:
        print(f"❌ No Stage 2 checkpoints found in {stage2_dir}")
        return

    print(f"Found {len(stage2_ckpts)} Stage 2 checkpoints:")
    for ckpt in stage2_ckpts:
        print(f"  - {ckpt.name}")
    print()

    # Run comparison
    comparator = Stage2EpochComparator(
        stage1_ckpt=args.stage1,
        stage2_ckpts=stage2_ckpts,
        num_samples_per_class=args.samples_per_class,
        device=args.device,
        output_dir=args.output
    )

    comparator.run_comparison()


if __name__ == "__main__":
    main()
