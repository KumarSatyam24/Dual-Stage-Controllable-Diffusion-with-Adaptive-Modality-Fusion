#!/usr/bin/env python3
"""
Unified Evaluation Script for SSIM, PSNR, and LPIPS
with CLIP-Optimized Prompts on the Full Sketchy Test Set.

This script evaluates Stage 1 and Stage 2 outputs using:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

It also enhances prompts using a CLIP-optimized template.

Usage:
    python test_lpips_psnr_ssim.py \
        --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
        --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
        --output_dir ./results/all_metrics_test
"""

import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

warnings.filterwarnings("ignore")

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# Optional LPIPS import
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


# ============================================================
# Prompt Optimization
# ============================================================
def optimize_prompt(prompt, category):
    """
    Optimizes a simple prompt for better CLIP score alignment.
    """
    return (
        f"A high-quality, realistic photograph of a {category}, "
        f"based on the description: {prompt}, with natural lighting, "
        f"sharp details, and accurate colors."
    )


def extract_category(sample):
    """
    Extracts category from dataset sample.
    """
    if "category" in sample:
        return sample["category"]

    for key in ["photo_path", "sketch_path"]:
        if key in sample:
            return Path(sample[key]).parent.name

    return "object"


# ============================================================
# Metric Utilities
# ============================================================
def tensor_to_numpy(img):
    """Convert tensor (B,C,H,W) to numpy (B,H,W,C) in [0,1]."""
    img = img.detach().cpu().clamp(0, 1)
    return img.permute(0, 2, 3, 1).numpy()


def compute_psnr(gen, gt):
    """Compute average PSNR."""
    gen_np = tensor_to_numpy(gen)
    gt_np = tensor_to_numpy(gt)
    scores = [
        peak_signal_noise_ratio(gt_np[i], gen_np[i], data_range=1.0)
        for i in range(len(gen_np))
    ]
    return float(np.mean(scores))


def compute_ssim(gen, gt):
    """Compute average SSIM."""
    gen_np = tensor_to_numpy(gen)
    gt_np = tensor_to_numpy(gt)
    scores = [
        structural_similarity(
            gt_np[i],
            gen_np[i],
            channel_axis=2,
            data_range=1.0,
        )
        for i in range(len(gen_np))
    ]
    return float(np.mean(scores))


# ============================================================
# Validator Class
# ============================================================
class UnifiedMetricsValidator:
    def __init__(
        self,
        stage1_checkpoint,
        stage2_checkpoint,
        output_dir,
        batch_size=4,
        num_samples=-1,
        image_size=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        device="cuda",
        lpips_net="alex",
    ):
        self.stage1_checkpoint = Path(stage1_checkpoint)
        self.stage2_checkpoint = Path(stage2_checkpoint)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device if torch.cuda.is_available() else "cpu"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "stage1_psnr": [],
            "stage2_psnr": [],
            "stage1_ssim": [],
            "stage2_ssim": [],
            "stage1_lpips": [],
            "stage2_lpips": [],
            "prompt_samples": []
        }

        print("\n" + "=" * 70)
        print("📊 Unified Metrics Validation (PSNR | SSIM | LPIPS)")
        print("=" * 70)

        self.load_models()
        self.load_dataset()
        self.setup_lpips(lpips_net)

    # ========================================================
    # Model Loading
    # ========================================================
    def load_models(self):
        print("📦 Loading models...")

        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        model_name = "runwayml/stable-diffusion-v1-5"

        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(self.device).eval()

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(self.device).eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )

        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
        from src.models.stage2_refinement import Stage2SemanticRefinement, Stage2RefinementPipeline

        self.stage1 = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            sketch_encoder_channels=[320, 640, 1280, 1280],
            freeze_base_unet=False,
            use_lora=True,
            lora_rank=8,
        ).to(self.device).eval()

        ckpt1 = torch.load(self.stage1_checkpoint, map_location="cpu", weights_only=False)
        self.stage1.load_state_dict(
            ckpt1.get("model_state_dict", ckpt1),
            strict=False
        )

        # Create Stage 1 pipeline
        self.stage1_pipeline = Stage1DiffusionPipeline(
            model=self.stage1,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            device=self.device
        )

        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet"
        ).to(self.device)

        self.stage2 = Stage2SemanticRefinement(unet=unet).to(self.device).eval()

        ckpt2 = torch.load(self.stage2_checkpoint, map_location="cpu", weights_only=False)
        self.stage2.load_state_dict(
            ckpt2.get("model_state_dict", ckpt2),
            strict=False
        )

        # Create Stage 2 pipeline
        self.stage2_pipeline = Stage2RefinementPipeline(
            stage2_model=self.stage2,
            vae=self.vae,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            device=self.device
        )

        print("✅ Models and pipelines loaded successfully.\n")

    # ========================================================
    # LPIPS Setup
    # ========================================================
    def setup_lpips(self, net="alex"):
        if not LPIPS_AVAILABLE:
            raise ImportError("Install LPIPS: pip install lpips")

        print("📊 Initializing LPIPS...")
        self.lpips_model = lpips.LPIPS(net=net).to(self.device)
        self.lpips_model.eval()
        print("✅ LPIPS ready.\n")

    # ========================================================
    # Dataset Loading
    # ========================================================
    def load_dataset(self):
        print("📁 Loading dataset...")
        from src.datasets.sketchy_dataset import SketchyDataset

        self.dataset = SketchyDataset(
            root_dir="/workspace/sketchy",
            split="test",
            image_size=self.image_size,
            augment=False,
        )

        if self.num_samples == -1:
            self.num_samples = len(self.dataset)
        else:
            self.num_samples = min(self.num_samples, len(self.dataset))

        print(f"✅ Loaded {self.num_samples} samples.\n")

    # ========================================================
    # Image Generation
    # ========================================================
    @torch.no_grad()
    def generate(self, sketches, prompts, region_graphs):
        """Generate images using pipelines (batch processing via loop)."""
        batch_size = sketches.shape[0]
        stage1_outputs = []
        stage2_outputs = []

        for i in range(batch_size):
            sketch = sketches[i:i+1].to(self.device)
            prompt = prompts[i]
            region_graph = region_graphs[i]

            # Stage 1 generation
            stage1_img = self.stage1_pipeline.generate(
                sketch=sketch,
                text_prompt=prompt,
                height=self.image_size,
                width=self.image_size,
            )

            # Encode text for Stage 2
            text_embeddings = self.stage1.encode_text([prompt])

            # Stage 2 refinement
            stage2_img = self.stage2_pipeline.refine(
                stage1_image=stage1_img,
                region_graph=region_graph,
                text_prompt=prompt,
                text_embeddings=text_embeddings,
            )

            stage1_outputs.append(stage1_img)
            stage2_outputs.append(stage2_img)

        # Stack into batch tensors
        stage1_output = torch.cat(stage1_outputs, dim=0)
        stage2_output = torch.cat(stage2_outputs, dim=0)

        return stage2_output, stage1_output

    # ========================================================
    # LPIPS Computation
    # ========================================================
    def compute_lpips(self, gen, gt):
        gen = gen.to(self.device) * 2 - 1
        gt = gt.to(self.device) * 2 - 1
        return float(self.lpips_model(gen, gt).mean().item())

    # ========================================================
    # Validation Loop
    # ========================================================
    def validate(self):
        print("🚀 Starting evaluation...\n")
        start_time = time.time()

        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches)):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            batch = [self.dataset[i] for i in range(start, end)]

            sketches = torch.stack([b["sketch"] for b in batch])
            photos_gt = torch.stack([b["photo"] for b in batch])
            region_graphs = [b["region_graph"] for b in batch]

            # Convert from [-1,1] to [0,1]
            photos_gt = (photos_gt + 1) / 2

            # Optimize prompts
            prompts = []
            for b in batch:
                original_prompt = b["text_prompt"]
                category = extract_category(b)
                optimized = optimize_prompt(original_prompt, category)
                prompts.append(optimized)

                if len(self.results["prompt_samples"]) < 5:
                    self.results["prompt_samples"].append({
                        "original": original_prompt,
                        "optimized": optimized,
                        "category": category
                    })

            # Generate outputs
            stage2, stage1 = self.generate(
                sketches, prompts, region_graphs
            )

            # Clamp outputs
            stage1 = torch.clamp(stage1, 0, 1)
            stage2 = torch.clamp(stage2, 0, 1)

            # Compute metrics
            self.results["stage1_psnr"].append(compute_psnr(stage1, photos_gt))
            self.results["stage2_psnr"].append(compute_psnr(stage2, photos_gt))
            self.results["stage1_ssim"].append(compute_ssim(stage1, photos_gt))
            self.results["stage2_ssim"].append(compute_ssim(stage2, photos_gt))
            self.results["stage1_lpips"].append(self.compute_lpips(stage1, photos_gt))
            self.results["stage2_lpips"].append(self.compute_lpips(stage2, photos_gt))

        elapsed = time.time() - start_time
        self.save_results(elapsed)

    # ========================================================
    # Save Results
    # ========================================================
    def save_results(self, elapsed):
        summary = {
            "stage1_psnr": float(np.mean(self.results["stage1_psnr"])),
            "stage2_psnr": float(np.mean(self.results["stage2_psnr"])),
            "stage1_ssim": float(np.mean(self.results["stage1_ssim"])),
            "stage2_ssim": float(np.mean(self.results["stage2_ssim"])),
            "stage1_lpips": float(np.mean(self.results["stage1_lpips"])),
            "stage2_lpips": float(np.mean(self.results["stage2_lpips"])),
        }

        summary["psnr_improvement"] = (
            summary["stage2_psnr"] - summary["stage1_psnr"]
        )
        summary["ssim_improvement"] = (
            summary["stage2_ssim"] - summary["stage1_ssim"]
        )
        summary["lpips_improvement"] = (
            summary["stage1_lpips"] - summary["stage2_lpips"]
        )

        output_path = self.output_dir / "test_lpips_psnr_ssim.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "prompt_samples": self.results["prompt_samples"],
                    "execution_time_seconds": elapsed,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        print("\n" + "=" * 70)
        print("📊 FINAL RESULTS")
        print("=" * 70)
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")
        print("=" * 70)
        print(f"📁 Results saved to: {output_path}\n")


# ============================================================
# Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified SSIM, PSNR, LPIPS Evaluation with Optimized Prompts"
    )
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./results/lpips_psnr_ssim_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lpips_net", type=str, default="alex")

    args = parser.parse_args()

    validator = UnifiedMetricsValidator(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        lpips_net=args.lpips_net,
    )

    validator.validate()


if __name__ == "__main__":
    main()