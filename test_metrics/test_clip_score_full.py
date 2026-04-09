#!/usr/bin/env python3
"""
CLIP Score Metric Test on Full Sketchy Test Set

Computes semantic alignment between generated images and text prompts using CLIP.
CLIP Score ranges from 0 to 100, higher values indicate better text-image alignment.

Usage:
    python test_metrics/test_clip_score_full.py \
        --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
        --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
        --output_dir ./results/clip_score_test \
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
warnings.filterwarnings('ignore')

from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not available. Install: pip install git+https://github.com/openai/CLIP.git")


class CLIPScoreValidator:
    """Standalone CLIP Score metric validator for full test set."""

    def __init__(
        self,
        stage1_checkpoint,
        stage2_checkpoint,
        output_dir="./results/clip_score_test",
        batch_size=4,
        num_samples=-1,
        image_size=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        device='cuda',
        clip_model='ViT-B/32'
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
        self.clip_model_name = clip_model

        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP is required but not installed.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'stage2_clip_scores': [],
            'stage1_clip_scores': [],
            'per_sample_results': []
        }

        print("=" * 70)
        print("📊 CLIP Score Metric Validation - Full Sketchy Test Set")
        print("=" * 70)
        print(f"   Stage 1: {self.stage1_checkpoint}")
        print(f"   Stage 2: {self.stage2_checkpoint}")
        print(f"   Output:  {self.output_dir}")
        print(f"   Device:  {device}")
        print(f"   CLIP Model: {clip_model}")
        print("=" * 70)
        print()

        self.load_models()
        self.load_dataset()
        self.setup_clip()

    def setup_clip(self):
        """Initialize CLIP model."""
        print("📊 Setting up CLIP...")
        self.clip_model, self.clip_preprocess = clip.load(self.clip_model_name, device=self.device)
        self.clip_model.eval()
        print(f"   ✅ CLIP ({self.clip_model_name}) ready")
        print()

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

    def compute_clip_score(self, generated, text_prompts):
        """Compute CLIP Score metric."""
        clip_scores = []

        for i, prompt in enumerate(text_prompts):
            # Prepare image
            img = generated[i]
            img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)

            # Encode
            image_features = self.clip_model.encode_image(img_tensor)
            text_tokens = clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)

            # Normalize and compute cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).item()

            clip_scores.append(score)

        return np.mean(clip_scores)

    def validate(self):
        """Run validation on full test set."""
        print("🔄 Starting CLIP Score validation...")
        print(f"   Total samples: {self.num_samples}")
        print(f"   Batch size: {self.batch_size}")
        print()

        start_time = time.time()
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Computing CLIP Score"):
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
            photos_gt = torch.stack([d['photo'] for d in batch_data])  # Not used for CLIP
            prompts = [d['text_prompt'] for d in batch_data]
            file_ids = [d['file_id'] for d in batch_data]
            categories = [d['category'] for d in batch_data]
            region_graphs = [d['region_graph'] for d in batch_data]

            try:
                generated, stage1 = self.generate(sketches, prompts, region_graphs)

                clip_stage2 = self.compute_clip_score(generated, prompts)
                clip_stage1 = self.compute_clip_score(stage1, prompts)

                self.results['stage2_clip_scores'].append(clip_stage2)
                self.results['stage1_clip_scores'].append(clip_stage1)

                for i, file_id in enumerate(file_ids):
                    self.results['per_sample_results'].append({
                        'file_id': file_id,
                        'category': categories[i],
                        'prompt': prompts[i],
                        'stage2_clip_score': clip_stage2,
                        'stage1_clip_score': clip_stage1
                    })

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

            if (batch_idx + 1) % 10 == 0:
                self.save_intermediate_results()

        elapsed = time.time() - start_time
        print(f"\n✅ CLIP Score validation complete in {elapsed/3600:.2f} hours")
        self.save_results(elapsed)

    def save_intermediate_results(self):
        """Save intermediate results."""
        temp_path = self.output_dir / "_intermediate_results.json"
        with open(temp_path, 'w') as f:
            json.dump({
                'stage2_clip_mean': float(np.mean(self.results['stage2_clip_scores'])) if self.results['stage2_clip_scores'] else 0,
                'stage1_clip_mean': float(np.mean(self.results['stage1_clip_scores'])) if self.results['stage1_clip_scores'] else 0,
                'num_samples': len(self.results['per_sample_results'])
            }, f, indent=2)

    def save_results(self, elapsed_time):
        """Save final results."""
        print("\n💾 Saving CLIP Score results...")

        summary = {
            'stage2_clip_score': {
                'mean': float(np.mean(self.results['stage2_clip_scores'])),
                'std': float(np.std(self.results['stage2_clip_scores'])),
                'min': float(np.min(self.results['stage2_clip_scores'])),
                'max': float(np.max(self.results['stage2_clip_scores']))
            },
            'stage1_clip_score': {
                'mean': float(np.mean(self.results['stage1_clip_scores'])),
                'std': float(np.std(self.results['stage1_clip_scores'])),
                'min': float(np.min(self.results['stage1_clip_scores'])),
                'max': float(np.max(self.results['stage1_clip_scores']))
            },
            'clip_improvement': float(np.mean(self.results['stage2_clip_scores']) - np.mean(self.results['stage1_clip_scores']))
        }

        results_path = self.output_dir / "clip_score_results.json"
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
                    'clip_model': self.clip_model_name
                },
                'execution_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        csv_path = self.output_dir / "per_sample_clip_scores.csv"
        if self.results['per_sample_results']:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results['per_sample_results'][0].keys())
                writer.writeheader()
                writer.writerows(self.results['per_sample_results'])

        print("\n" + "=" * 70)
        print("📊 CLIP SCORE VALIDATION RESULTS")
        print("=" * 70)
        print(f"\nStage 1 CLIP Score: {summary['stage1_clip_score']['mean']:.4f} ± {summary['stage1_clip_score']['std']:.4f}")
        print(f"Stage 2 CLIP Score: {summary['stage2_clip_score']['mean']:.4f} ± {summary['stage2_clip_score']['std']:.4f}")
        print(f"Improvement:        {summary['clip_improvement']:.4f}")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - JSON: {results_path}")
        print(f"  - CSV:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="CLIP Score Metric Test on Full Sketchy Test Set")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/clip_score_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])

    args = parser.parse_args()

    validator = CLIPScoreValidator(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        clip_model=args.clip_model
    )

    validator.validate()


if __name__ == "__main__":
    main()
