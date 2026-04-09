#!/usr/bin/env python3
"""
Baseline: Run ControlNet on Sketchy test set for comparison.

Usage:
    python run_baseline_controlnet.py \
        --output_dir ./baseline_results/controlnet \
        --num_samples 1000 \
        --batch_size 4
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
import csv
from datetime import datetime
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  diffusers not available")

try:
    from skimage.metrics import structural_similarity as compute_ssim
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class ControlNetBaseline:
    """ControlNet baseline for comparison."""

    def __init__(
        self,
        output_dir="./baseline_results/controlnet",
        batch_size=4,
        num_samples=-1,
        image_size=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        device='cuda'
    ):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {
            'ssim_scores': [],
            'psnr_scores': [],
            'lpips_scores': [],
            'clip_scores': [],
            'edge_similarity_scores': [],
            'per_sample_results': []
        }

        print("=" * 70)
        print("🎨 ControlNet Baseline Evaluation")
        print("=" * 70)

        self.load_model()
        self.load_dataset()
        self.setup_metrics()

    def load_model(self):
        """Load ControlNet model."""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers required")

        print("📦 Loading ControlNet...")

        # Load ControlNet for sketch/edge conditioning
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",  # Use Canny edge ControlNet
            torch_dtype=torch.float16
        ).to(self.device)

        # Load SD pipeline with ControlNet
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        print("   ✅ ControlNet loaded")
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

        print(f"   ✅ Dataset loaded: {len(self.dataset)} total")
        print(f"   📊 Will evaluate: {self.num_samples} samples")
        print()

    def setup_metrics(self):
        """Setup metrics."""
        print("📊 Setting up metrics...")

        if LPIPS_AVAILABLE:
            self.lpips_model = LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            print("   ✅ LPIPS ready")

        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            print("   ✅ CLIP ready")

        print()

    @torch.no_grad()
    def generate(self, sketch, prompt):
        """Generate with ControlNet."""
        # Convert sketch to PIL
        images = []
        for i in range(sketch.shape[0]):
            sk = sketch[i].squeeze().cpu().numpy()
            sk_pil = Image.fromarray((sk * 255).astype(np.uint8))
            images.append(sk_pil)

        # Generate
        outputs = self.pipe(
            prompt,
            images,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.image_size,
            width=self.image_size
        ).images

        # Convert back to tensors
        output_tensors = []
        for img in outputs:
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            output_tensors.append(img_tensor)

        return torch.stack(output_tensors)

    def compute_metrics(self, generated, ground_truth, prompt):
        """Compute metrics."""
        metrics = {}
        gen_np = generated.cpu().numpy()
        gt_np = ground_truth.cpu().numpy()

        # SSIM
        if SKIMAGE_AVAILABLE:
            ssim_scores = []
            for i in range(gen_np.shape[0]):
                gen_img = np.transpose(gen_np[i], (1, 2, 0))
                gt_img = np.transpose(gt_np[i], (1, 2, 0))
                score = compute_ssim(gt_img, gen_img, channel_axis=2, data_range=1.0)
                ssim_scores.append(score)
            metrics['ssim'] = np.mean(ssim_scores)

        # PSNR
        if SKIMAGE_AVAILABLE:
            psnr_scores = []
            for i in range(gen_np.shape[0]):
                gen_img = np.transpose(gen_np[i], (1, 2, 0))
                gt_img = np.transpose(gt_np[i], (1, 2, 0))
                score = compute_psnr(gt_img, gen_img, data_range=1.0)
                psnr_scores.append(score)
            metrics['psnr'] = np.mean(psnr_scores)

        # LPIPS
        if self.lpips_model is not None:
            gen_lpips = generated * 2 - 1
            gt_lpips = ground_truth * 2 - 1
            lpips_score = self.lpips_model(gen_lpips, gt_lpips).mean().item()
            metrics['lpips'] = lpips_score

        # CLIP
        if self.clip_model is not None and isinstance(prompt, list):
            import cv2
            clip_scores = []
            for i, p in enumerate(prompt):
                img = generated[i]
                img_pil = Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)

                image_features = self.clip_model.encode_image(img_tensor)
                text_tokens = clip.tokenize([p]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)

                score = (image_features @ text_features.T).item()
                clip_scores.append(score)
            metrics['clip_score'] = np.mean(clip_scores)

        # Edge similarity
        import cv2
        edge_scores = []
        for i in range(gen_np.shape[0]):
            gen_img = (np.transpose(gen_np[i], (1, 2, 0)) * 255).astype(np.uint8)
            gt_img = (np.transpose(gt_np[i], (1, 2, 0)) * 255).astype(np.uint8)

            gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
            gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)

            gen_edges = cv2.Canny(gen_gray, 50, 150)
            gt_edges = cv2.Canny(gt_gray, 50, 150)

            if SKIMAGE_AVAILABLE:
                edge_sim = compute_ssim(gt_edges, gen_edges, data_range=255)
            else:
                edge_sim = np.sum(gen_edges == gt_edges) / gen_edges.size
            edge_scores.append(edge_sim)

        metrics['edge_similarity'] = np.mean(edge_scores)

        return metrics

    def run(self):
        """Run ControlNet baseline."""
        print("🔄 Starting ControlNet baseline...")

        for i in tqdm(range(self.num_samples), desc="ControlNet"):
            try:
                sample = self.dataset[i]

                sketch = sample['sketch'].unsqueeze(0).to(self.device)
                photo_gt = sample['photo'].unsqueeze(0).to(self.device)
                prompt = sample['text_prompt']
                file_id = sample['file_id']
                category = sample['category']

                # Normalize GT
                photo_gt = (photo_gt + 1) / 2

                # Generate
                generated = self.generate(sketch, [prompt])

                # Compute metrics
                metrics = self.compute_metrics(generated, photo_gt, [prompt])

                # Store
                for key in ['ssim', 'psnr', 'lpips', 'clip_score', 'edge_similarity']:
                    if key in metrics:
                        self.results[f'{key}_scores'].append(metrics[key])

                self.results['per_sample_results'].append({
                    'file_id': file_id,
                    'category': category,
                    'prompt': prompt,
                    **{k: metrics[k] for k in ['ssim', 'psnr', 'lpips', 'clip_score', 'edge_similarity'] if k in metrics}
                })

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

        self.save_results()

    def save_results(self):
        """Save results."""
        print("\n💾 Saving results...")

        summary = {}
        for key in ['ssim_scores', 'psnr_scores', 'lpips_scores', 'clip_scores', 'edge_similarity_scores']:
            if self.results[key]:
                summary[key.replace('_scores', '')] = {
                    'mean': float(np.mean(self.results[key])),
                    'std': float(np.std(self.results[key])),
                    'min': float(np.min(self.results[key])),
                    'max': float(np.max(self.results[key]))
                }

        # JSON
        with open(self.output_dir / "controlnet_results.json", 'w') as f:
            json.dump({
                'summary': summary,
                'config': {
                    'method': 'ControlNet',
                    'num_samples': self.num_samples,
                    'image_size': self.image_size,
                    'num_inference_steps': self.num_inference_steps,
                    'guidance_scale': self.guidance_scale
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # CSV
        if self.results['per_sample_results']:
            with open(self.output_dir / "controlnet_per_sample.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results['per_sample_results'][0].keys())
                writer.writeheader()
                writer.writerows(self.results['per_sample_results'])

        print("\n" + "=" * 70)
        print("📊 CONTROLNET BASELINE RESULTS")
        print("=" * 70)
        for metric, stats in summary.items():
            print(f"{metric.upper():20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./baseline_results/controlnet")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    baseline = ControlNetBaseline(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device
    )
    baseline.run()


if __name__ == "__main__":
    main()
