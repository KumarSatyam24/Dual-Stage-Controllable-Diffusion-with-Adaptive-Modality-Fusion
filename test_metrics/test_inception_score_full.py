#!/usr/bin/env python3
"""
Inception Score (IS) Metric Test on Full Sketchy Test Set

Computes Inception Score for generated images using Inception V3.
IS measures both the quality (confidence of predictions) and diversity
of generated images. Higher IS indicates better quality and diversity.

Typical range: 1.0 to 10.0+, higher is better

Usage:
    python test_metrics/test_inception_score_full.py \
        --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
        --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
        --output_dir ./results/is_test \
        --num_samples 100 \
        --splits 10
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import json
import csv
from datetime import datetime
import time
import warnings
from scipy.stats import entropy
warnings.filterwarnings('ignore')

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def optimize_prompt(prompt, category):
    """
    Optimizes a simple prompt for better CLIP score alignment.

    Args:
        prompt: Original simple prompt (e.g., "A photo of a dog")
        category: Object category (e.g., "dog")

    Returns:
        Optimized prompt for CLIP
    """
    return (
        f"A high-quality, realistic photograph of a {category}, "
        f"based on the description: {prompt}, with natural lighting, "
        f"sharp details, and accurate colors."
    )


def get_inception_model(device='cuda'):
    """Load Inception V3 model for feature extraction."""
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.eval()
        model = model.to(device)
        return model
    except Exception as e:
        print(f"⚠️  Could not load Inception V3: {e}")
        return None


class InceptionScoreValidator:
    """Standalone Inception Score metric validator for full test set."""

    def __init__(
        self,
        stage1_checkpoint,
        stage2_checkpoint,
        output_dir="./results/is_test",
        batch_size=4,
        num_samples=-1,
        image_size=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        device='cuda',
        splits=10
    ):
        self.stage1_checkpoint = Path(stage1_checkpoint)
        self.stage2_checkpoint = Path(stage2_checkpoint)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
        self.splits = splits

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.results = {
            'stage2_is_mean': None,
            'stage2_is_std': None,
            'stage1_is_mean': None,
            'stage1_is_std': None,
            'per_sample_results': []
        }

        print("=" * 70)
        print("📊 Inception Score (IS) Metric Validation - Full Sketchy Test Set")
        print("=" * 70)
        print(f"   Stage 1: {self.stage1_checkpoint}")
        print(f"   Stage 2: {self.stage2_checkpoint}")
        print(f"   Output:  {self.output_dir}")
        print(f"   Device:  {device}")
        print(f"   Splits:  {self.splits}")
        print("=" * 70)
        print()

        self.load_models()
        self.load_dataset()

    def load_models(self):
        """Load Stage 1 and Stage 2 models."""
        print("📦 Loading models...")
        model_name = "runwayml/stable-diffusion-v1-5"

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(self.device).eval()
        print("   ✅ VAE loaded")

        # Text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(self.device).eval()
        print("   ✅ Text Encoder loaded")

        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )

        # Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Stage 1
        print("\n   Loading Stage 1...")
        from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion

        self.stage1 = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            sketch_encoder_channels=[320, 640, 1280, 1280],
            freeze_base_unet=False,
            use_lora=True,
            lora_rank=8
        ).to(self.device).eval()

        ckpt1 = torch.load(self.stage1_checkpoint, map_location="cpu", weights_only=False)
        self.stage1.load_state_dict(ckpt1["model_state_dict"], strict=False)
        print(f"   ✅ Stage 1 loaded")

        # Stage 2
        print("\n   Loading Stage 2...")
        from src.models.stage2_refinement import Stage2SemanticRefinement

        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet"
        ).to(self.device)

        self.stage2 = Stage2SemanticRefinement(unet=unet).to(self.device).eval()

        ckpt2 = torch.load(self.stage2_checkpoint, map_location="cpu", weights_only=False)
        self.stage2.load_state_dict(ckpt2["model_state_dict"], strict=False)
        print(f"   ✅ Stage 2 loaded")

        # Inception model
        print("\n   Loading Inception V3...")
        self.inception = get_inception_model(self.device)
        if self.inception is not None:
            print("   ✅ Inception V3 loaded")
        else:
            raise RuntimeError("Failed to load Inception V3 model")
        print()

    def load_dataset(self):
        """Load Sketchy test dataset."""
        print("📁 Loading dataset...")
        from src.datasets.sketchy_dataset import SketchyDataset

        self.dataset = SketchyDataset(
            root_dir="/workspace/sketchy",
            split="test",
            image_size=self.image_size,
            augment=False
        )

        if self.num_samples == -1:
            self.num_samples = len(self.dataset)
        else:
            self.num_samples = min(self.num_samples, len(self.dataset))

        print(f"   ✅ Dataset loaded: {len(self.dataset)} total samples")
        print(f"   📊 Will evaluate: {self.num_samples} samples")
        print()

    @torch.no_grad()
    def generate(self, sketch, text_prompt, region_graphs, categories=None):
        """Generate image using full pipeline with optimized prompts."""
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]
            if categories is None:
                categories = [None]
        elif categories is None:
            categories = [None] * len(text_prompt)

        # Optimize prompts for better CLIP alignment
        optimized_prompts = []
        for prompt, category in zip(text_prompt, categories):
            if category is not None:
                optimized_prompts.append(optimize_prompt(prompt, category))
            else:
                optimized_prompts.append(prompt)

        sketch = sketch.to(self.device)

        # Stage 1: Sketch-guided generation
        stage1_output = self._run_stage1(sketch, optimized_prompts)

        # Stage 2: Semantic refinement
        stage2_output = self._run_stage2(stage1_output, sketch, optimized_prompts, region_graphs)

        return stage2_output, stage1_output

    def _run_stage1(self, sketch, text_prompt):
        """Run Stage 1 generation."""
        B = sketch.shape[0]

        text_inputs = self.tokenizer(
            text_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(
            text_inputs.input_ids.to(self.device)
        )[0]

        uncond_inputs = self.tokenizer(
            [""] * B,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(
            uncond_inputs.input_ids.to(self.device)
        )[0]

        down_res, mid_res = self.stage1.encode_sketch(sketch)
        down_res_cfg = [torch.cat([r, r]) for r in down_res]
        mid_res_cfg = torch.cat([mid_res, mid_res])

        latents = torch.randn(
            B, 4, self.image_size // 8, self.image_size // 8,
            device=self.device
        )

        self.scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

            noise_pred = self.stage1.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res_cfg,
                mid_block_additional_residual=mid_res_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def _run_stage2(self, stage1_output, sketch, text_prompt, region_graphs):
        """Run Stage 2 refinement."""
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]

        B = len(text_prompt)

        stage1_latent = self.vae.encode(stage1_output * 2 - 1).latent_dist.sample()
        stage1_latent = stage1_latent * 0.18215

        text_inputs = self.tokenizer(
            text_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        uncond_inputs = self.tokenizer(
            [""] * B,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]

        scheduler = self.scheduler
        scheduler.set_timesteps(30)
        timesteps = scheduler.timesteps[-15:]

        noise = torch.randn_like(stage1_latent)
        latents = scheduler.add_noise(stage1_latent, noise, timesteps[0])

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])
            stage1_latent_cfg = torch.cat([stage1_latent] * 2)

            # Duplicate region_graphs for CFG (uncond + cond)
            region_graphs_cfg = region_graphs + region_graphs

            # Ensure timestep is a tensor on the correct device
            if isinstance(t, torch.Tensor):
                t_tensor = t.to(self.device)
            else:
                t_tensor = torch.tensor(t, device=self.device)
            # Expand timestep to batch size (B*2 for CFG)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.unsqueeze(0).expand(latent_model_input.shape[0])

            noise_pred = self.stage2(
                latent_model_input,
                t_tensor,
                region_graphs_cfg,
                encoder_hidden_states,
                stage1_latents=stage1_latent_cfg,
                return_dict=False
            )

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        refined = self.vae.decode(latents / 0.18215).sample
        refined = (refined / 2 + 0.5).clamp(0, 1)

        return refined

    def compute_inception_score(self, images, splits=10):
        """
        Compute Inception Score for generated images.

        IS = exp(E_x[KL(p(y|x) || p(y))])

        Args:
            images: Tensor of shape (N, 3, H, W) in range [0, 1]
            splits: Number of splits for computing mean and std

        Returns:
            mean: Mean IS across splits
            std: Standard deviation of IS across splits
        """
        N = images.shape[0]

        # Move images to the same device as Inception model
        images = images.to(self.device)

        # Resize and normalize for Inception
        # Inception expects 299x299 images
        images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images_normalized = (images_resized - mean) / std

        # Get predictions
        preds = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = images_normalized[i:i+batch_size]
                pred = self.inception(batch)
                pred = F.softmax(pred, dim=1)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Compute IS using vectorized entropy
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)  # marginal distribution p(y)
            # Vectorized KL divergence computation
            scores = entropy(part.T, py[:, None])
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    def validate(self):
        """Run validation on full test set."""
        print("🔄 Starting Inception Score validation...")
        print(f"   Total samples: {self.num_samples}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Splits: {self.splits}")
        print()

        start_time = time.time()
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        # Collect all generated images for IS computation
        stage2_images = []
        stage1_images = []
        all_metadata = []

        for batch_idx in tqdm(range(num_batches), desc="Generating images"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            batch_data = []
            for i in range(start_idx, end_idx):
                try:
                    sample = self.dataset[i]
                    batch_data.append(sample)
                except Exception as e:
                    print(f"Error loading sample {i}: {e}")
                    continue

            if len(batch_data) == 0:
                continue

            sketches = torch.stack([d['sketch'] for d in batch_data])
            prompts = [d['text_prompt'] for d in batch_data]
            file_ids = [d['file_id'] for d in batch_data]
            categories = [d['category'] for d in batch_data]
            region_graphs = [d['region_graph'] for d in batch_data]

            try:
                generated, stage1 = self.generate(sketches, prompts, region_graphs, categories)

                stage2_images.append(generated.cpu())
                stage1_images.append(stage1.cpu())

                for i, file_id in enumerate(file_ids):
                    all_metadata.append({
                        'file_id': file_id,
                        'category': categories[i],
                        'prompt': prompts[i]
                    })

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Save intermediate results every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"\n   Processed {(batch_idx + 1) * self.batch_size} samples...")

        print("\n📊 Computing Inception Score...")

        # Compute IS for Stage 2
        if len(stage2_images) > 0:
            stage2_tensor = torch.cat(stage2_images, dim=0)
            print(f"   Computing IS for {stage2_tensor.shape[0]} Stage 2 images...")
            is_mean_s2, is_std_s2 = self.compute_inception_score(stage2_tensor, splits=self.splits)
            self.results['stage2_is_mean'] = is_mean_s2
            self.results['stage2_is_std'] = is_std_s2
            print(f"   Stage 2 IS: {is_mean_s2:.2f} ± {is_std_s2:.2f}")

        # Compute IS for Stage 1
        if len(stage1_images) > 0:
            stage1_tensor = torch.cat(stage1_images, dim=0)
            print(f"   Computing IS for {stage1_tensor.shape[0]} Stage 1 images...")
            is_mean_s1, is_std_s1 = self.compute_inception_score(stage1_tensor, splits=self.splits)
            self.results['stage1_is_mean'] = is_mean_s1
            self.results['stage1_is_std'] = is_std_s1
            print(f"   Stage 1 IS: {is_mean_s1:.2f} ± {is_std_s1:.2f}")

        elapsed = time.time() - start_time
        print(f"\n✅ Inception Score validation complete in {elapsed/3600:.2f} hours")
        self.save_results(elapsed)

    def save_results(self, elapsed_time):
        """Save final results."""
        print("\n💾 Saving Inception Score results...")

        # Compute summary
        stage2_mean = self.results['stage2_is_mean'] if self.results['stage2_is_mean'] is not None else 0
        stage2_std = self.results['stage2_is_std'] if self.results['stage2_is_std'] is not None else 0
        stage1_mean = self.results['stage1_is_mean'] if self.results['stage1_is_mean'] is not None else 0
        stage1_std = self.results['stage1_is_std'] if self.results['stage1_is_std'] is not None else 0

        summary = {
            'stage2_is': {
                'mean': float(stage2_mean),
                'std': float(stage2_std)
            },
            'stage1_is': {
                'mean': float(stage1_mean),
                'std': float(stage1_std)
            },
            'is_improvement': float(stage2_mean - stage1_mean)
        }

        # Save JSON
        results_path = self.output_dir / "is_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'summary': summary,
                'config': {
                    'stage1_checkpoint': str(self.stage1_checkpoint),
                    'stage2_checkpoint': str(self.stage2_checkpoint),
                    'num_samples': self.num_samples,
                    'image_size': self.image_size,
                    'num_inference_steps': self.num_inference_steps,
                    'guidance_scale': self.guidance_scale,
                    'splits': self.splits
                },
                'execution_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # Print report
        print("\n" + "=" * 70)
        print("📊 INCEPTION SCORE VALIDATION RESULTS")
        print("=" * 70)
        print(f"\nStage 1 IS: {stage1_mean:.2f} ± {stage1_std:.2f}")
        print(f"Stage 2 IS: {stage2_mean:.2f} ± {stage2_std:.2f}")
        print(f"Improvement: {summary['is_improvement']:.2f}")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - JSON: {results_path}")




def main():
    parser = argparse.ArgumentParser(description="Inception Score (IS) Metric Test on Full Sketchy Test Set")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/is_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--splits", type=int, default=10, help="Number of splits for IS computation")

    args = parser.parse_args()

    validator = InceptionScoreValidator(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        splits=args.splits
    )

    validator.validate()


if __name__ == "__main__":
    main()
