#!/usr/bin/env python3
"""
Edge Similarity Metric Test on Full Sketchy Test Set

Computes edge consistency between generated images and ground truth using Canny edge detection.
Measures how well edges are preserved in the generation process.
Uses SSIM on edge maps for similarity measurement.

Usage:
    python test_metrics/test_edge_similarity_full.py \
        --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
        --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
        --output_dir ./results/edge_similarity_test \
        --num_samples -1
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
import csv
from datetime import datetime
import time
import warnings
import cv2
warnings.filterwarnings('ignore')

from skimage.metrics import structural_similarity as compute_ssim
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class EdgeSimilarityValidator:
    """Standalone Edge Similarity metric validator for full test set."""

    def __init__(
        self,
        stage1_checkpoint,
        stage2_checkpoint,
        output_dir="./results/edge_similarity_test",
        batch_size=4,
        num_samples=-1,
        image_size=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        device='cuda',
        canny_low=50,
        canny_high=150
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
        self.canny_low = canny_low
        self.canny_high = canny_high

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'stage2_edge_scores': [],
            'stage1_edge_scores': [],
            'per_sample_results': []
        }

        print("=" * 70)
        print("📊 Edge Similarity Metric Validation - Full Sketchy Test Set")
        print("=" * 70)
        print(f"   Stage 1: {self.stage1_checkpoint}")
        print(f"   Stage 2: {self.stage2_checkpoint}")
        print(f"   Output:  {self.output_dir}")
        print(f"   Device:  {device}")
        print(f"   Canny:   low={canny_low}, high={canny_high}")
        print("=" * 70)
        print()

        self.load_models()
        self.load_dataset()

    def load_models(self):
        """Load Stage 1 and Stage 2 models."""
        print("📦 Loading models...")
        model_name = "runwayml/stable-diffusion-v1-5"

        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(self.device).eval()
        print("   ✅ VAE loaded")

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(self.device).eval()
        print("   ✅ Text Encoder loaded")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )

        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

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

        print("\n   Loading Stage 2...")
        from src.models.stage2_refinement import Stage2SemanticRefinement

        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet"
        ).to(self.device)

        self.stage2 = Stage2SemanticRefinement(unet=unet).to(self.device).eval()

        ckpt2 = torch.load(self.stage2_checkpoint, map_location="cpu", weights_only=False)
        self.stage2.load_state_dict(ckpt2["model_state_dict"], strict=False)
        print(f"   ✅ Stage 2 loaded")
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
    def generate(self, sketch, text_prompt, region_graphs):
        """Generate image using full pipeline."""
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]

        sketch = sketch.to(self.device)
        stage1_output = self._run_stage1(sketch, text_prompt)
        stage2_output = self._run_stage2(stage1_output, sketch, text_prompt, region_graphs)
        return stage2_output, stage1_output

    def _run_stage1(self, sketch, text_prompt):
        """Run Stage 1 generation."""
        B = sketch.shape[0]

        text_inputs = self.tokenizer(
            text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        uncond_inputs = self.tokenizer([""] * B, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]

        down_res, mid_res = self.stage1.encode_sketch(sketch)
        down_res_cfg = [torch.cat([r, r]) for r in down_res]
        mid_res_cfg = torch.cat([mid_res, mid_res])

        latents = torch.randn(B, 4, self.image_size // 8, self.image_size // 8, device=self.device)
        self.scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

            noise_pred = self.stage1.unet(
                latent_model_input, t, encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res_cfg, mid_block_additional_residual=mid_res_cfg
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
        stage1_latent = self.vae.encode(stage1_output * 2 - 1).latent_dist.sample() * 0.18215

        text_inputs = self.tokenizer(text_prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        uncond_inputs = self.tokenizer([""] * B, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]

        scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        scheduler.set_timesteps(30)
        timesteps = scheduler.timesteps[-15:]

        noise = torch.randn_like(stage1_latent)
        latents = scheduler.add_noise(stage1_latent, noise, timesteps[0])

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])
            stage1_latent_cfg = torch.cat([stage1_latent] * 2)

            noise_pred = self.stage2(
                latent_model_input, t.to(self.device), region_graphs,
                encoder_hidden_states, stage1_latents=stage1_latent_cfg, return_dict=False
            )

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        refined = self.vae.decode(latents / 0.18215).sample
        refined = (refined / 2 + 0.5).clamp(0, 1)
        return refined

    def compute_edge_similarity(self, generated, ground_truth):
        """Compute edge similarity using Canny edge detection."""
        gen_np = generated.cpu().numpy()
        gt_np = ground_truth.cpu().numpy()

        edge_scores = []
        for i in range(gen_np.shape[0]):
            gen_img = (np.transpose(gen_np[i], (1, 2, 0)) * 255).astype(np.uint8)
            gt_img = (np.transpose(gt_np[i], (1, 2, 0)) * 255).astype(np.uint8)

            # Convert to grayscale
            gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
            gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)

            # Canny edges
            gen_edges = cv2.Canny(gen_gray, self.canny_low, self.canny_high)
            gt_edges = cv2.Canny(gt_gray, self.canny_low, self.canny_high)

            # SSIM on edges
            edge_sim = compute_ssim(gt_edges, gen_edges, data_range=255)
            edge_scores.append(edge_sim)

        return np.mean(edge_scores)

    def validate(self):
        """Run validation on full test set."""
        print("🔄 Starting Edge Similarity validation...")
        print(f"   Total samples: {self.num_samples}")
        print(f"   Batch size: {self.batch_size}")
        print()

        start_time = time.time()
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Computing Edge Similarity"):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            batch_data = []
            for i in range(start_idx, end_idx):
                try:
                    batch_data.append(self.dataset[i])
                except Exception as e:
                    print(f"Error loading sample {i}: {e}")
                    continue

            if len(batch_data) == 0:
                continue

            sketches = torch.stack([d['sketch'] for d in batch_data])
            photos_gt = torch.stack([d['photo'] for d in batch_data])
            prompts = [d['text_prompt'] for d in batch_data]
            file_ids = [d['file_id'] for d in batch_data]
            categories = [d['category'] for d in batch_data]
            region_graphs = [d['region_graph'] for d in batch_data]

            photos_gt = (photos_gt + 1) / 2

            try:
                generated, stage1 = self.generate(sketches, prompts, region_graphs)

                edge_stage2 = self.compute_edge_similarity(generated, photos_gt)
                edge_stage1 = self.compute_edge_similarity(stage1, photos_gt)

                self.results['stage2_edge_scores'].append(edge_stage2)
                self.results['stage1_edge_scores'].append(edge_stage1)

                for i, file_id in enumerate(file_ids):
                    self.results['per_sample_results'].append({
                        'file_id': file_id,
                        'category': categories[i],
                        'prompt': prompts[i],
                        'stage2_edge_similarity': edge_stage2,
                        'stage1_edge_similarity': edge_stage1
                    })

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

            if (batch_idx + 1) % 10 == 0:
                self.save_intermediate_results()

        elapsed = time.time() - start_time
        print(f"\n✅ Edge Similarity validation complete in {elapsed/3600:.2f} hours")
        self.save_results(elapsed)

    def save_intermediate_results(self):
        """Save intermediate results."""
        temp_path = self.output_dir / "_intermediate_results.json"
        with open(temp_path, 'w') as f:
            json.dump({
                'stage2_edge_mean': float(np.mean(self.results['stage2_edge_scores'])) if self.results['stage2_edge_scores'] else 0,
                'stage1_edge_mean': float(np.mean(self.results['stage1_edge_scores'])) if self.results['stage1_edge_scores'] else 0,
                'num_samples': len(self.results['per_sample_results'])
            }, f, indent=2)

    def save_results(self, elapsed_time):
        """Save final results."""
        print("\n💾 Saving Edge Similarity results...")

        summary = {
            'stage2_edge_similarity': {
                'mean': float(np.mean(self.results['stage2_edge_scores'])),
                'std': float(np.std(self.results['stage2_edge_scores'])),
                'min': float(np.min(self.results['stage2_edge_scores'])),
                'max': float(np.max(self.results['stage2_edge_scores']))
            },
            'stage1_edge_similarity': {
                'mean': float(np.mean(self.results['stage1_edge_scores'])),
                'std': float(np.std(self.results['stage1_edge_scores'])),
                'min': float(np.min(self.results['stage1_edge_scores'])),
                'max': float(np.max(self.results['stage1_edge_scores']))
            },
            'edge_improvement': float(np.mean(self.results['stage2_edge_scores']) - np.mean(self.results['stage1_edge_scores']))
        }

        results_path = self.output_dir / "edge_similarity_results.json"
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
                    'canny_low': self.canny_low,
                    'canny_high': self.canny_high
                },
                'execution_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        csv_path = self.output_dir / "per_sample_edge_similarity.csv"
        if self.results['per_sample_results']:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results['per_sample_results'][0].keys())
                writer.writeheader()
                writer.writerows(self.results['per_sample_results'])

        print("\n" + "=" * 70)
        print("📊 EDGE SIMILARITY VALIDATION RESULTS")
        print("=" * 70)
        print(f"\nStage 1 Edge Sim: {summary['stage1_edge_similarity']['mean']:.4f} ± {summary['stage1_edge_similarity']['std']:.4f}")
        print(f"Stage 2 Edge Sim: {summary['stage2_edge_similarity']['mean']:.4f} ± {summary['stage2_edge_similarity']['std']:.4f}")
        print(f"Improvement:      {summary['edge_improvement']:.4f}")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - JSON: {results_path}")
        print(f"  - CSV:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Edge Similarity Metric Test on Full Sketchy Test Set")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/edge_similarity_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--canny_low", type=int, default=50)
    parser.add_argument("--canny_high", type=int, default=150)

    args = parser.parse_args()

    validator = EdgeSimilarityValidator(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        canny_low=args.canny_low,
        canny_high=args.canny_high
    )

    validator.validate()


if __name__ == "__main__":
    main()
