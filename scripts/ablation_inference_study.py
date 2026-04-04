#!/usr/bin/env python3
"""
Inference-only ablation study runner for dual-stage controllable diffusion.

What it does:
- Loads trained Stage 1 + Stage 2 checkpoints
- Uses a fixed sample set (prompt/image/sketch) for fair comparison
- Runs multiple inference-time ablation variants (no retraining)
- Computes CLIP Score, FID, SSIM (+ LPIPS if available)
- Saves per-variant outputs + metrics + side-by-side visualizations
- Saves final cross-variant comparison table (JSON + CSV)

Default output layout:
results/
  baseline_full/
    images/
    comparisons/
    metrics.json
  no_stage2/
    images/
    comparisons/
    metrics.json
  ...
  ablation_comparison.json
  ablation_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

try:
    from lpips import LPIPS

    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.region_extraction import RegionExtractor
from src.data.region_graph import RegionGraphBuilder
from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion
from src.models.stage2_refinement import Stage2SemanticRefinement


# ------------------------------
# Utilities
# ------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """Convert CHW tensor to PIL RGB. Supports ranges [-1,1] or [0,1]."""
    if img.dim() == 4:
        img = img.squeeze(0)

    x = img.detach().cpu().float()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)

    if x.shape[0] == 1:
        arr = (x.squeeze(0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")

    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def sanitize_name(text: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip().lower()]
    return "".join(keep).strip("_")


# ------------------------------
# Dataset (fixed deterministic sample set)
# ------------------------------

class FixedSampleSketchyDataset(Dataset):
    """Deterministically picks one sketch-photo sample per selected category."""

    def __init__(self, root_dir: str, image_size: int = 256, num_classes: int = 100, seed: int = 42):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed

        self.photo_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.sketch_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.region_extractor = RegionExtractor(min_region_area=100, max_num_regions=50)
        self.graph_builder = RegionGraphBuilder(
            graph_type="adjacency", feature_type="spatial", image_size=(image_size, image_size)
        )

        self.selected_classes = self._get_fixed_classes()
        self.data_pairs = self._load_one_per_class()

    def _get_fixed_classes(self) -> List[str]:
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        if not sketch_dir.exists():
            raise FileNotFoundError(f"Sketch directory not found: {sketch_dir}")

        all_categories = sorted([d.name for d in sketch_dir.iterdir() if d.is_dir()])
        rng = random.Random(self.seed)
        rng.shuffle(all_categories)
        selected = all_categories[: min(self.num_classes, len(all_categories))]
        return sorted(selected)

    def _load_one_per_class(self) -> List[Dict[str, Any]]:
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        photo_dir = self.root_dir / "photo" / "tx_000000000000"

        pairs: List[Dict[str, Any]] = []
        for class_name in self.selected_classes:
            class_sketch_dir = sketch_dir / class_name
            class_photo_dir = photo_dir / class_name
            if not class_sketch_dir.exists() or not class_photo_dir.exists():
                continue

            sketch_files = sorted(class_sketch_dir.glob("*.png"))
            if not sketch_files:
                continue

            # deterministic pick per class
            sketch_path = sketch_files[min(2, len(sketch_files) - 1)]
            sketch_stem = sketch_path.stem
            photo_base = sketch_stem.rsplit("-", 1)[0] if "-" in sketch_stem else sketch_stem
            photo_path = class_photo_dir / f"{photo_base}.jpg"
            if not photo_path.exists():
                continue

            pairs.append(
                {
                    "sketch_path": str(sketch_path),
                    "photo_path": str(photo_path),
                    "category": class_name,
                    "file_id": sketch_stem,
                }
            )

        return pairs

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.data_pairs[idx]

        sketch_pil = Image.open(pair["sketch_path"]).convert("L")
        photo_pil = Image.open(pair["photo_path"]).convert("RGB")

        sketch = self.sketch_transform(sketch_pil)
        photo = self.photo_transform(photo_pil)

        category = pair["category"].replace("_", " ")
        text_prompt = f"A natural photo of a {category} aligned with the given sketch"

        sketch_np = (sketch.squeeze(0).numpy() * 255).astype(np.uint8)
        regions = self.region_extractor.extract_regions(sketch_np)
        region_graph = self.graph_builder.build_graph(regions)

        return {
            "sketch": sketch,
            "photo": photo,
            "text_prompt": text_prompt,
            "region_graph": region_graph,
            "category": pair["category"],
            "file_id": pair["file_id"],
        }


# ------------------------------
# Metrics
# ------------------------------

class MetricsSuite:
    def __init__(self, device: str = "cuda"):
        self.device = device

        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval().to(device)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if LPIPS_AVAILABLE:
            self.lpips_model = LPIPS(net="alex").to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

    def compute_ssim(self, gen_img: torch.Tensor, gt_img: torch.Tensor) -> float:
        gen = gen_img.detach().cpu().float().numpy()
        gt = gt_img.detach().cpu().float().numpy()

        if gen.shape[0] == 3:
            gen = np.transpose(gen, (1, 2, 0))
        if gt.shape[0] == 3:
            gt = np.transpose(gt, (1, 2, 0))

        gen = np.clip(gen, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        gen_gray = cv2.cvtColor((gen * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gt_gray = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return float(ssim(gen_gray, gt_gray, data_range=255))

    def compute_lpips(self, gen_img: torch.Tensor, gt_img: torch.Tensor) -> Optional[float]:
        if self.lpips_model is None:
            return None

        # LPIPS expects normalized [-1,1]
        if gen_img.min() >= 0:
            gen = gen_img * 2.0 - 1.0
        else:
            gen = gen_img

        if gt_img.min() >= 0:
            gt = gt_img * 2.0 - 1.0
        else:
            gt = gt_img

        with torch.no_grad():
            val = self.lpips_model(gen.unsqueeze(0).to(self.device), gt.unsqueeze(0).to(self.device)).item()
        return float(val)

    def compute_clip_score(self, images: Sequence[torch.Tensor], prompts: Sequence[str]) -> float:
        scores: List[float] = []
        with torch.no_grad():
            for img, prompt in zip(images, prompts):
                pil = tensor_to_pil(img)
                inputs = self.clip_processor(
                    text=[prompt], images=pil, return_tensors="pt", padding=True
                ).to(self.device)
                out = self.clip_model(**inputs)
                scores.append(float(out.logits_per_image.item()))
        return float(np.mean(scores)) if scores else 0.0

    def _extract_inception_features(self, images: Sequence[torch.Tensor]) -> np.ndarray:
        prep = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        feats: List[np.ndarray] = []
        with torch.no_grad():
            for img in images:
                pil = tensor_to_pil(img)
                x = prep(pil).unsqueeze(0).to(self.device)
                f = self.inception(x).cpu().numpy()[0]
                feats.append(f)

        return np.array(feats)

    def compute_fid(self, real_images: Sequence[torch.Tensor], generated_images: Sequence[torch.Tensor]) -> float:
        real_feats = self._extract_inception_features(real_images)
        gen_feats = self._extract_inception_features(generated_images)

        mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
        mu_g, sigma_g = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

        diff = mu_r - mu_g
        covmean = sqrtm(sigma_r @ sigma_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
        return float(fid)


# ------------------------------
# Ablation definitions
# ------------------------------

@dataclass
class AblationVariant:
    name: str
    use_stage2: bool = True
    adaptive_residual_alpha: bool = True
    fixed_residual_alpha: Optional[float] = None
    no_text_conditioning_stage2: bool = False
    refinement_strength: float = 0.5


@dataclass
class SampleCacheItem:
    idx: int
    category: str
    file_id: str
    prompt: str
    sketch: torch.Tensor
    gt_image: torch.Tensor  # [0,1]
    region_graph: Any
    stage1_image: torch.Tensor  # [0,1]
    stage1_latents: torch.Tensor


# ------------------------------
# Runner
# ------------------------------

class AblationInferenceRunner:
    def __init__(
        self,
        stage1_checkpoint: str,
        stage2_checkpoint: str,
        dataset_root: str,
        output_root: str,
        num_samples: int,
        device: str,
        image_size: int,
        num_inference_steps: int,
        num_refinement_steps: int,
        seed: int,
    ):
        self.stage1_checkpoint = Path(stage1_checkpoint)
        self.stage2_checkpoint = Path(stage2_checkpoint)
        self.dataset_root = dataset_root
        self.output_root = Path(output_root)
        self.num_samples = num_samples
        self.device = device
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.num_refinement_steps = num_refinement_steps
        self.seed = seed

        self.output_root.mkdir(parents=True, exist_ok=True)

        set_global_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.metrics = MetricsSuite(device=device)
        self._load_models()
        self._load_dataset()

    def _load_models(self) -> None:
        model_name = "runwayml/stable-diffusion-v1-5"

        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)
        self.vae.eval()

        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(self.device)
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        self.stage1_model = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name, freeze_base_unet=True
        ).to(self.device)
        ckpt1 = torch.load(self.stage1_checkpoint, map_location="cpu", weights_only=False)
        self.stage1_model.load_state_dict(ckpt1["model_state_dict"], strict=False)
        self.stage1_model.eval()

        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(self.device)
        self.stage2_model = Stage2SemanticRefinement(
            unet=unet,
            node_feature_dim=6,
            text_dim=768,
            hidden_dim=512,
            num_graph_layers=2,
            num_attention_heads=8,
            fusion_method="learned",
            use_region_adaptive_fusion=True,
            residual_alpha=0.2,
        ).to(self.device)
        ckpt2 = torch.load(self.stage2_checkpoint, map_location="cpu", weights_only=False)
        self.stage2_model.load_state_dict(ckpt2["model_state_dict"], strict=False)
        self.stage2_model.eval()

        self.stage1_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.stage2_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.stage1_scheduler.set_timesteps(self.num_inference_steps)
        self.stage2_scheduler.set_timesteps(self.num_refinement_steps)

    def _load_dataset(self) -> None:
        dataset = FixedSampleSketchyDataset(
            root_dir=self.dataset_root,
            image_size=self.image_size,
            num_classes=max(100, self.num_samples),
            seed=self.seed,
        )
        if len(dataset) == 0:
            raise RuntimeError("Dataset produced zero samples. Check dataset path and structure.")

        num = min(self.num_samples, len(dataset))
        self.dataset = dataset

        # Deterministic fixed sample indices for all ablations
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        self.sample_indices = torch.randperm(len(dataset), generator=rng)[:num].tolist()

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        return self.text_encoder(tokens.input_ids)[0]

    @torch.no_grad()
    def generate_stage1(self, sketch: torch.Tensor, prompt: str, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        sketch = sketch.to(self.device)
        down_res, mid_res = self.stage1_model.encode_sketch(sketch)

        text_emb = self._encode_text(prompt)
        uncond_emb = self._encode_text("")
        enc = torch.cat([uncond_emb, text_emb], dim=0)

        latents = torch.randn(
            (1, 4, self.image_size // 8, self.image_size // 8),
            device=self.device,
            dtype=torch.float32,
            generator=g,
        )
        latents = latents * self.stage1_scheduler.init_noise_sigma

        down_cfg = [torch.cat([torch.zeros_like(r), r], dim=0) for r in down_res]
        mid_cfg = torch.cat([torch.zeros_like(mid_res), mid_res], dim=0)

        for t in self.stage1_scheduler.timesteps:
            latent_in = torch.cat([latents, latents], dim=0)
            latent_in = self.stage1_scheduler.scale_model_input(latent_in, t)

            noise_pred = self.stage1_model(latent_in, t, (down_cfg, mid_cfg), enc)
            noise_u, noise_c = noise_pred.chunk(2)
            noise_pred = noise_u + 7.5 * (noise_c - noise_u)
            latents = self.stage1_scheduler.step(noise_pred, t, latents).prev_sample

        decoded = self.vae.decode((1.0 / 0.18215) * latents).sample
        image = (decoded / 2 + 0.5).clamp(0, 1)
        return image.detach().cpu(), latents.detach().cpu()

    @torch.no_grad()
    def generate_stage2_variant(
        self,
        stage1_latents: torch.Tensor,
        region_graph: Any,
        prompt: str,
        seed: int,
        variant: AblationVariant,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not variant.use_stage2:
            decoded = self.vae.decode((1.0 / 0.18215) * stage1_latents.to(self.device)).sample
            image = (decoded / 2 + 0.5).clamp(0, 1)
            return image.detach().cpu(), {"alpha_history": [], "adaptive_enabled": False}

        # Move graph to device
        if hasattr(region_graph, "node_features"):
            region_graph.node_features = region_graph.node_features.to(self.device)
        if hasattr(region_graph, "edge_index"):
            region_graph.edge_index = region_graph.edge_index.to(self.device)
        if hasattr(region_graph, "edge_attr") and region_graph.edge_attr is not None:
            region_graph.edge_attr = region_graph.edge_attr.to(self.device)

        stage1_latents = stage1_latents.to(self.device)

        cond_prompt = "" if variant.no_text_conditioning_stage2 else prompt
        text_emb = self._encode_text(cond_prompt)
        uncond_emb = self._encode_text("")
        enc = torch.cat([uncond_emb, text_emb], dim=0)

        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        strength = float(np.clip(variant.refinement_strength, 0.0, 1.0))
        init_timestep = max(1, int(self.num_refinement_steps * strength))
        timesteps = self.stage2_scheduler.timesteps[-init_timestep:]

        noise = torch.randn(stage1_latents.shape, device=self.device, generator=g)
        latents = self.stage2_scheduler.add_noise(stage1_latents, noise, timesteps[0])

        stage1_cfg = torch.cat([torch.zeros_like(stage1_latents), stage1_latents], dim=0)
        guidance_scale = 2.5

        original_alpha = float(self.stage2_model.residual_alpha)
        alpha = float(
            variant.fixed_residual_alpha
            if variant.fixed_residual_alpha is not None
            else original_alpha
        )

        alpha_min, alpha_max = 0.05, 0.30
        ema_momentum = 0.9
        t_low, t_high = 0.08, 0.15
        running_delta: Optional[float] = None
        alpha_history: List[float] = []

        for t in timesteps:
            self.stage2_model.residual_alpha = float(np.clip(alpha, alpha_min, alpha_max))
            alpha_history.append(float(self.stage2_model.residual_alpha))

            latent_in = torch.cat([latents, latents], dim=0)
            latent_in = self.stage2_scheduler.scale_model_input(latent_in, t)

            out = self.stage2_model(
                latent_in,
                t.to(self.device),
                region_graph,
                enc,
                stage1_latents=stage1_cfg,
                return_dict=True,
            )
            noise_pred = out["noise_pred"]
            noise_u, noise_c = noise_pred.chunk(2)
            noise_pred = noise_u + guidance_scale * (noise_c - noise_u)
            latents = self.stage2_scheduler.step(noise_pred, t, latents).prev_sample

            if variant.adaptive_residual_alpha and variant.fixed_residual_alpha is None:
                modulation = out.get("modulation_map", None)
                if modulation is not None:
                    delta = float(modulation.detach().abs().mean().item())
                    if running_delta is None:
                        running_delta = delta
                    else:
                        running_delta = ema_momentum * running_delta + (1.0 - ema_momentum) * delta

                    if running_delta > t_high:
                        alpha *= 0.97
                    elif running_delta < t_low:
                        alpha *= 1.01
                    alpha = float(np.clip(alpha, alpha_min, alpha_max))

        self.stage2_model.residual_alpha = original_alpha

        decoded = self.vae.decode((1.0 / 0.18215) * latents).sample
        image = (decoded / 2 + 0.5).clamp(0, 1)

        return image.detach().cpu(), {
            "alpha_history": alpha_history,
            "adaptive_enabled": bool(variant.adaptive_residual_alpha),
            "final_alpha": float(alpha_history[-1]) if alpha_history else float(original_alpha),
        }

    def _create_side_by_side(
        self,
        sketch: torch.Tensor,
        stage1_img: torch.Tensor,
        variant_img: torch.Tensor,
        title: str,
    ) -> Image.Image:
        sketch_pil = tensor_to_pil(sketch)
        s1_pil = tensor_to_pil(stage1_img)
        v_pil = tensor_to_pil(variant_img)

        w, h = sketch_pil.size
        canvas = Image.new("RGB", (w * 3, h + 36), color="white")
        draw = ImageDraw.Draw(canvas)

        canvas.paste(sketch_pil, (0, 28))
        canvas.paste(s1_pil, (w, 28))
        canvas.paste(v_pil, (2 * w, 28))

        draw.text((10, 8), "Input (Sketch)", fill="black")
        draw.text((w + 10, 8), "Stage 1", fill="black")
        draw.text((2 * w + 10, 8), title, fill="black")

        return canvas

    def _summarize_variant(
        self,
        variant: AblationVariant,
        per_sample: List[Dict[str, Any]],
        gt_images: List[torch.Tensor],
        generated_images: List[torch.Tensor],
        prompts: List[str],
    ) -> Dict[str, Any]:
        ssim_vals = [r["ssim"] for r in per_sample]
        lpips_vals = [r["lpips"] for r in per_sample if r["lpips"] is not None]

        fid = self.metrics.compute_fid(gt_images, generated_images)
        clip_score = self.metrics.compute_clip_score(generated_images, prompts)

        summary = {
            "variant": asdict(variant),
            "num_samples": len(per_sample),
            "metrics": {
                "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
                "ssim_std": float(np.std(ssim_vals)) if ssim_vals else 0.0,
                "fid": float(fid),
                "clip_score": float(clip_score),
                "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
                "lpips_std": float(np.std(lpips_vals)) if lpips_vals else None,
            },
            "per_sample": per_sample,
            "timestamp": datetime.now().isoformat(),
        }
        return summary

    def _write_final_comparison(self, summaries: Dict[str, Dict[str, Any]]) -> None:
        comparison_json = self.output_root / "ablation_comparison.json"
        with open(comparison_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

        comparison_csv = self.output_root / "ablation_comparison.csv"
        fieldnames = [
            "experiment",
            "ssim_mean",
            "ssim_std",
            "fid",
            "clip_score",
            "lpips_mean",
            "lpips_std",
            "num_samples",
        ]

        with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for exp_name, payload in summaries.items():
                m = payload["metrics"]
                writer.writerow(
                    {
                        "experiment": exp_name,
                        "ssim_mean": m["ssim_mean"],
                        "ssim_std": m["ssim_std"],
                        "fid": m["fid"],
                        "clip_score": m["clip_score"],
                        "lpips_mean": m["lpips_mean"],
                        "lpips_std": m["lpips_std"],
                        "num_samples": payload["num_samples"],
                    }
                )

    def print_summary_table(self, summaries: Dict[str, Dict[str, Any]]) -> None:
        print("\n" + "=" * 96)
        print("Ablation Summary")
        print("=" * 96)
        print(
            f"{'Experiment':<28} {'SSIM':>10} {'FID':>10} {'CLIP':>10} {'LPIPS':>10} {'N':>6}"
        )
        print("-" * 96)
        for exp_name, payload in summaries.items():
            m = payload["metrics"]
            lpips_val = m["lpips_mean"]
            lpips_str = f"{lpips_val:.4f}" if lpips_val is not None else "N/A"
            print(
                f"{exp_name:<28} {m['ssim_mean']:>10.4f} {m['fid']:>10.3f} {m['clip_score']:>10.3f} {lpips_str:>10} {payload['num_samples']:>6}"
            )
        print("=" * 96 + "\n")

    def build_variants(self) -> List[AblationVariant]:
        return [
            AblationVariant(name="baseline_full", use_stage2=True, adaptive_residual_alpha=True, refinement_strength=0.5),
            AblationVariant(name="no_stage2", use_stage2=False, adaptive_residual_alpha=False, refinement_strength=0.0),
            AblationVariant(
                name="fixed_residual_alpha",
                use_stage2=True,
                adaptive_residual_alpha=False,
                fixed_residual_alpha=0.2,
                refinement_strength=0.5,
            ),
            AblationVariant(
                name="no_text_conditioning_stage2",
                use_stage2=True,
                adaptive_residual_alpha=True,
                no_text_conditioning_stage2=True,
                refinement_strength=0.5,
            ),
            AblationVariant(name="refinement_strength_0_2", use_stage2=True, adaptive_residual_alpha=True, refinement_strength=0.2),
            AblationVariant(name="refinement_strength_0_4", use_stage2=True, adaptive_residual_alpha=True, refinement_strength=0.4),
            AblationVariant(name="refinement_strength_0_6", use_stage2=True, adaptive_residual_alpha=True, refinement_strength=0.6),
        ]

    def run(self) -> Dict[str, Dict[str, Any]]:
        print("\nPreparing fixed sample cache (Stage 1 once, reused for all variants)...")
        sample_cache: List[SampleCacheItem] = []

        for order_idx, ds_idx in enumerate(tqdm(self.sample_indices, desc="Stage1 cache")):
            sample = self.dataset[ds_idx]
            sketch = sample["sketch"].unsqueeze(0)
            gt_img = ((sample["photo"] + 1.0) / 2.0).clamp(0, 1)
            prompt = sample["text_prompt"]

            stage1_seed = self.seed + order_idx
            stage1_img, stage1_latents = self.generate_stage1(sketch, prompt, seed=stage1_seed)

            sample_cache.append(
                SampleCacheItem(
                    idx=order_idx,
                    category=sample["category"],
                    file_id=sample["file_id"],
                    prompt=prompt,
                    sketch=sketch.squeeze(0).cpu(),
                    gt_image=gt_img.cpu(),
                    region_graph=sample["region_graph"],
                    stage1_image=stage1_img.squeeze(0).cpu(),
                    stage1_latents=stage1_latents.cpu(),
                )
            )

        summaries: Dict[str, Dict[str, Any]] = {}
        variants = self.build_variants()

        for v_idx, variant in enumerate(variants):
            exp_name = sanitize_name(variant.name)
            exp_dir = self.output_root / exp_name
            image_dir = exp_dir / "images"
            cmp_dir = exp_dir / "comparisons"
            image_dir.mkdir(parents=True, exist_ok=True)
            cmp_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{v_idx + 1}/{len(variants)}] Running variant: {exp_name}")

            per_sample_results: List[Dict[str, Any]] = []
            generated_images: List[torch.Tensor] = []
            gt_images: List[torch.Tensor] = []
            prompts: List[str] = []

            for item in tqdm(sample_cache, desc=f"Variant {exp_name}"):
                infer_seed = self.seed + 100_000 + item.idx
                variant_img, debug_info = self.generate_stage2_variant(
                    stage1_latents=item.stage1_latents,
                    region_graph=item.region_graph,
                    prompt=item.prompt,
                    seed=infer_seed,
                    variant=variant,
                )
                variant_img = variant_img.squeeze(0).cpu()

                ssim_val = self.metrics.compute_ssim(variant_img, item.gt_image)
                lpips_val = self.metrics.compute_lpips(variant_img, item.gt_image)

                # Save generated image
                out_img_path = image_dir / f"{item.idx:03d}_{item.category}_{item.file_id}.png"
                tensor_to_pil(variant_img).save(out_img_path)

                # Save side-by-side (input -> stage1 -> variant)
                cmp_img = self._create_side_by_side(
                    sketch=item.sketch,
                    stage1_img=item.stage1_image,
                    variant_img=variant_img,
                    title=exp_name,
                )
                cmp_path = cmp_dir / f"{item.idx:03d}_{item.category}_{item.file_id}.png"
                cmp_img.save(cmp_path)

                generated_images.append(variant_img)
                gt_images.append(item.gt_image)
                prompts.append(item.prompt)

                per_sample_results.append(
                    {
                        "idx": item.idx,
                        "category": item.category,
                        "file_id": item.file_id,
                        "prompt": item.prompt,
                        "ssim": float(ssim_val),
                        "lpips": float(lpips_val) if lpips_val is not None else None,
                        "debug": debug_info,
                        "image_path": str(out_img_path),
                        "comparison_path": str(cmp_path),
                    }
                )

            summary = self._summarize_variant(
                variant=variant,
                per_sample=per_sample_results,
                gt_images=gt_images,
                generated_images=generated_images,
                prompts=prompts,
            )

            metrics_path = exp_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            summaries[exp_name] = summary
            print(
                f"   -> SSIM: {summary['metrics']['ssim_mean']:.4f}, "
                f"FID: {summary['metrics']['fid']:.3f}, "
                f"CLIP: {summary['metrics']['clip_score']:.3f}, "
                f"LPIPS: {summary['metrics']['lpips_mean'] if summary['metrics']['lpips_mean'] is not None else 'N/A'}"
            )

        self._write_final_comparison(summaries)
        self.print_summary_table(summaries)
        return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference-only ablation study runner")
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="Path to Stage 2 checkpoint")
    parser.add_argument("--dataset_root", type=str, default="/workspace/sketchy", help="Sketchy root dir")
    parser.add_argument("--output_root", type=str, default="results", help="Root output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of fixed samples")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_refinement_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runner = AblationInferenceRunner(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        num_samples=args.num_samples,
        device=args.device,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        num_refinement_steps=args.num_refinement_steps,
        seed=args.seed,
    )
    runner.run()


if __name__ == "__main__":
    main()
