#!/usr/bin/env python3
"""
FID/KID Metric Test on Full Sketchy Test Set

Computes Fréchet Inception Distance (FID) and Kernel Inception Distance (KID).
These are distribution-level metrics that measure the distance between generated
and real image distributions using Inception V3 features.

FID: Lower is better (typical range: 10-100+)
KID: Lower is better, more stable than FID for small samples

Usage:
    python test_metrics/test_fid_kid_full.py \
        --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
        --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
        --output_dir ./results/fid_kid_test \
        --num_samples -1
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
import shutil
from PIL import Image
warnings.filterwarnings('ignore')

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from torch_fidelity.metrics import calculate_metrics
    TORCH_FIDELITY_AVAILABLE = True
except ImportError:
    TORCH_FIDELITY_AVAILABLE = False
    print("⚠️  torch-fidelity not available. Install: pip install torch-fidelity")


class KIDMetric:
    """Kernel Inception Distance (KID) implementation."""

    def __init__(self, feature_dim=2048, device='cuda'):
        self.device = device
        self.feature_dim = feature_dim

        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            self.inception.fc = nn.Identity()
            self.inception = self.inception.to(device)
            self.inception.eval()
        except:
            print("⚠️  Could not load Inception V3 for KID")
            self.inception = None

    def extract_features(self, images):
        """Extract Inception features from images."""
        if self.inception is None:
            return None

        # images: (B, 3, H, W) in range [0, 1]
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std

        with torch.no_grad():
            features = self.inception(images)

        return features

    def polynomial_kernel(self, x, y, degree=3, gamma=None, coef0=1):
        """Compute polynomial kernel for MMD."""
        if gamma is None:
            gamma = 1.0 / x.size(1)
        K = (gamma * (x @ y.t()) + coef0) ** degree
        return K

    def compute_kid(self, real_features, fake_features, num_subsets=100, max_subset_size=1000):
        """Compute KID between real and fake features."""
        n = real_features.size(0)
        m = fake_features.size(0)
        subset_size = min(max_subset_size, n, m)
        if subset_size < 2:
            return float("nan"), float("nan")

        kids = []
        for _ in range(num_subsets):
            idx_real = torch.randperm(n)[:subset_size]
            idx_fake = torch.randperm(m)[:subset_size]

            real_subset = real_features[idx_real]
            fake_subset = fake_features[idx_fake]

            K_real_real = self.polynomial_kernel(real_subset, real_subset)
            K_fake_fake = self.polynomial_kernel(fake_subset, fake_subset)
            K_real_fake = self.polynomial_kernel(real_subset, fake_subset)

            mmd_squared = (K_real_real.sum() - K_real_real.diag().sum()) / (subset_size * (subset_size - 1))
            mmd_squared += (K_fake_fake.sum() - K_fake_fake.diag().sum()) / (subset_size * (subset_size - 1))
            mmd_squared -= 2 * K_real_fake.mean()

            kids.append(mmd_squared.item())

        kid_mean = np.mean(kids)
        kid_std = np.std(kids)

        return kid_mean, kid_std


class FIDKIDValidator:
    """Standalone FID/KID metric validator for full test set."""

    def __init__(
        self,
        stage1_checkpoint,
        stage2_checkpoint,
        output_dir="./results/fid_kid_test",
        batch_size=4,
        num_samples=-1,
        image_size=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        device='cuda',
        compute_fid=True,
        compute_kid=True,
        kid_subset_size=100
    ):
        self.stage1_checkpoint = Path(stage1_checkpoint)
        self.stage2_checkpoint = Path(stage2_checkpoint)
        self.output_dir = Path(output_dir)
        self.batch_dir = self.output_dir / "temp_images"
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
        self.compute_fid = compute_fid and TORCH_FIDELITY_AVAILABLE
        self.compute_kid = compute_kid
        self.kid_subset_size = kid_subset_size

        if not TORCH_FIDELITY_AVAILABLE and compute_fid:
            print("⚠️  torch-fidelity not available, FID will be skipped")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        # For KID accumulation
        self.real_features_kid = []
        self.fake_features_kid = []
        self.kid_metric = None

        if self.compute_kid:
            self.kid_metric = KIDMetric(device=self.device)

        print("=" * 70)
        print("📊 FID/KID Metric Validation - Full Sketchy Test Set")
        print("=" * 70)
        print(f"   Stage 1: {self.stage1_checkpoint}")
        print(f"   Stage 2: {self.stage2_checkpoint}")
        print(f"   Output:  {self.output_dir}")
        print(f"   Device:  {device}")
        print(f"   FID:     {self.compute_fid}")
        print(f"   KID:     {self.compute_kid}")
        print("=" * 70)
        print()

        self.load_models()
        self.load_dataset()


    def load_models(self):
        """Load Stage 1 and Stage 2 models."""
        print("📦 Loading models...")
        model_name = "runwayml/stable-diffusion-v1-5"

        # Ensure CUDA availability
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("⚠️ CUDA not available. Falling back to CPU.")
            self.device = "cpu"

        # ------------------ VAE ------------------
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(self.device).eval()
        print("   ✅ VAE loaded")

        # ------------------ Text Encoder ------------------
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(self.device).eval()
        print("   ✅ Text Encoder loaded")

        # ------------------ Tokenizer ------------------
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )

        # ------------------ Schedulers ------------------
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        print("   ✅ Stage 1 Scheduler loaded")

        self.stage2_scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        print("   ✅ Stage 2 Scheduler loaded")

        # ------------------ Stage 1 ------------------
        print("\n   Loading Stage 1...")
        from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion

        self.stage1 = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            sketch_encoder_channels=[320, 640, 1280, 1280],
            freeze_base_unet=False,
            use_lora=True,
            lora_rank=8
        ).to(self.device).eval()

        ckpt1 = torch.load(
            self.stage1_checkpoint,
            map_location="cpu",
            weights_only=False
        )
        self.stage1.load_state_dict(ckpt1["model_state_dict"], strict=False)
        print("   ✅ Stage 1 loaded")

        # ------------------ Stage 2 ------------------
        print("\n   Loading Stage 2...")
        from src.models.stage2_refinement import Stage2SemanticRefinement

        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet"
        ).to(self.device)

        self.stage2 = Stage2SemanticRefinement(unet=unet).to(self.device).eval()

        ckpt2 = torch.load(
            self.stage2_checkpoint,
            map_location="cpu",
            weights_only=False
        )
        self.stage2.load_state_dict(ckpt2["model_state_dict"], strict=False)
        print("   ✅ Stage 2 loaded\n")




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

        scheduler = self.stage2_scheduler
        scheduler.set_timesteps(30)
        timesteps = scheduler.timesteps[-15:]

        # Ensure stage1_latent is on the correct device
        stage1_latent = stage1_latent.to(self.device)
        noise = torch.randn_like(stage1_latent)
        latents = scheduler.add_noise(stage1_latent, noise, timesteps[0])

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])
            stage1_latent_cfg = torch.cat([stage1_latent] * 2)

            # Duplicate region_graphs for CFG (uncond + cond)
            region_graphs_cfg = region_graphs + region_graphs

            # Ensure timestep is a tensor on the correct device, properly shaped for batch
            if isinstance(t, torch.Tensor):
                t_tensor = t.to(self.device)
            else:
                t_tensor = torch.tensor(t, device=self.device)
            # Expand timestep to batch size (B*2 for CFG)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.unsqueeze(0).expand(latent_model_input.shape[0])

            # Ensure all inputs are on the correct device
            latent_model_input = latent_model_input.to(self.device)
            encoder_hidden_states = encoder_hidden_states.to(self.device)
            stage1_latent_cfg = stage1_latent_cfg.to(self.device)

            noise_pred = self.stage2(
                latent_model_input, t_tensor, region_graphs_cfg,
                encoder_hidden_states, stage1_latents=stage1_latent_cfg, return_dict=False
            )

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        refined = self.vae.decode(latents / 0.18215).sample
        refined = (refined / 2 + 0.5).clamp(0, 1)
        return refined

    def validate(self):
        """Run validation on full test set."""
        print("🔄 Starting FID/KID validation...")
        print(f"   Total samples: {self.num_samples}")
        print(f"   Batch size: {self.batch_size}")
        print()

        start_time = time.time()

        # Create temp directories for FID
        if self.compute_fid:
            temp_gen_dir = self.output_dir / "temp_generated"
            temp_gt_dir = self.output_dir / "temp_ground_truth"
            temp_gen_dir.mkdir(exist_ok=True)
            temp_gt_dir.mkdir(exist_ok=True)

        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating images for FID/KID"):
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
            region_graphs = [d['region_graph'] for d in batch_data]

            photos_gt = (photos_gt + 1) / 2

            try:
                generated, _ = self.generate(sketches, prompts, region_graphs)

                # Accumulate KID features
                if self.compute_kid and self.kid_metric is not None:
                    real_feats = self.kid_metric.extract_features(photos_gt.to(self.device))
                    fake_feats = self.kid_metric.extract_features(generated.to(self.device))

                    if real_feats is not None and fake_feats is not None:
                        self.real_features_kid.append(real_feats.cpu())
                        self.fake_features_kid.append(fake_feats.cpu())

                # Save images for FID
                if self.compute_fid:
                    for i, file_id in enumerate(file_ids):
                        gen_img = generated[i].detach().cpu().clamp(0, 1).numpy()
                        gt_img = photos_gt[i].detach().cpu().clamp(0, 1).numpy()

                        gen_img_pil = Image.fromarray((gen_img.transpose(1, 2, 0) * 255).astype(np.uint8))
                        gt_img_pil = Image.fromarray((gt_img.transpose(1, 2, 0) * 255).astype(np.uint8))
                        gen_img_pil.save(temp_gen_dir / f"{file_id}.png")
                        gt_img_pil.save(temp_gt_dir / f"{file_id}.png")

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        elapsed = time.time() - start_time

        # Compute FID/KID
        results = {}

        if self.compute_fid and TORCH_FIDELITY_AVAILABLE:
            print("\n🧮 Computing FID and KID with torch-fidelity...")
            try:
                metrics = calculate_metrics(
                    input1=str(temp_gen_dir),
                    input2=str(temp_gt_dir),
                    cuda=self.device.startswith('cuda'),
                    isc=False,
                    fid=True,
                    kid=True,
                    verbose=False,
                )
                results['fid'] = metrics.get('frechet_inception_distance')
                results['kid_mean'] = metrics.get('kernel_inception_distance_mean')
                results['kid_std'] = metrics.get('kernel_inception_distance_std')
                print(f"   ✅ FID: {results['fid']:.4f}")
                print(f"   ✅ KID: {results['kid_mean']:.4f} ± {results['kid_std']:.4f}")
            except Exception as e:
                print(f"   ❌ Error calculating FID/KID: {e}")

        # Custom KID computation
        if self.compute_kid and len(self.real_features_kid) > 0:
            print("\n🧮 Computing custom KID...")
            real_features = torch.cat(self.real_features_kid, dim=0)
            fake_features = torch.cat(self.fake_features_kid, dim=0)
            kid_mean, kid_std = self.kid_metric.compute_kid(
                real_features, fake_features, num_subsets=100, max_subset_size=self.kid_subset_size
            )
            results['custom_kid_mean'] = kid_mean
            results['custom_kid_std'] = kid_std
            print(f"   ✅ Custom KID: {kid_mean:.4f} ± {kid_std:.4f}")

        # Cleanup
        if self.compute_fid:
            shutil.rmtree(temp_gen_dir)
            shutil.rmtree(temp_gt_dir)

        print(f"\n✅ FID/KID validation complete in {elapsed/3600:.2f} hours")
        self.save_results(results, elapsed)

    def save_results(self, results, elapsed_time):
        """Save final results."""
        print("\n💾 Saving FID/KID results...")

        results_path = self.output_dir / "fid_kid_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': results,
                'config': {
                    'stage1_checkpoint': str(self.stage1_checkpoint),
                    'stage2_checkpoint': str(self.stage2_checkpoint),
                    'num_samples': self.num_samples,
                    'image_size': self.image_size,
                    'num_inference_steps': self.num_inference_steps,
                    'guidance_scale': self.guidance_scale,
                    'compute_fid': self.compute_fid,
                    'compute_kid': self.compute_kid,
                    'kid_subset_size': self.kid_subset_size
                },
                'execution_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 FID/KID VALIDATION RESULTS")
        print("=" * 70)
        if 'fid' in results:
            print(f"\nFID:          {results['fid']:.4f}")
        if 'kid_mean' in results:
            print(f"KID:          {results['kid_mean']:.4f} ± {results['kid_std']:.4f}")
        if 'custom_kid_mean' in results:
            print(f"Custom KID:   {results['custom_kid_mean']:.4f} ± {results['custom_kid_std']:.4f}")
        print("=" * 70)
        print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="FID/KID Metric Test on Full Sketchy Test Set")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/fid_kid_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_fid", action="store_true", default=True)
    parser.add_argument("--no-compute_fid", dest="compute_fid", action="store_false")
    parser.add_argument("--compute_kid", action="store_true", default=True)
    parser.add_argument("--no-compute_kid", dest="compute_kid", action="store_false")
    parser.add_argument("--kid_subset_size", type=int, default=100)

    args = parser.parse_args()

    validator = FIDKIDValidator(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        compute_fid=args.compute_fid,
        compute_kid=args.compute_kid,
        kid_subset_size=args.kid_subset_size
    )

    validator.validate()


if __name__ == "__main__":
    main()
