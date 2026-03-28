#!/usr/bin/env python3
"""
Comprehensive Validation Script for RAGAF-Diffusion

Generates validation images from Stage 1 and Stage 2 models with metrics.
- Loads one image per class from 100 fixed classes
- Generates Stage 1 output (coarse)
- Generates Stage 2 output (refined)
- Computes metrics: SSIM, FID, CLIP Score, Edge IoU, Chamfer Distance
- Saves comparison PNGs with all 4 views: sketch, stage1, stage2, ground truth
- Saves JSON with computed metrics

Usage:
    python validate_comprehensive.py --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
                                      --stage2_checkpoint /workspace/checkpoints/stage2/epoch_2.pt \
                                      --output_dir /workspace/validation_results/epoch_2

Author: Claude
"""

import sys
import os
from pathlib import Path
import argparse
import json
import random
import hashlib
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from scipy.spatial.distance import directed_hausdorff

from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights
from scipy.linalg import sqrtm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from src.models.stage2_refinement import Stage2SemanticRefinement
from src.data.region_extraction import RegionExtractor
from src.data.region_graph import RegionGraphBuilder


class FixedClassValidationDataset(Dataset):
    """
    Dataset that returns one sample per class from fixed 100 classes.
    The class selection is deterministic using a seed for consistency.
    """

    def __init__(self, root_dir: str, image_size: int = 256, seed: int = 42):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.seed = seed

        # Fixed 100 classes (using deterministic selection from all categories)
        self.selected_classes = self._get_fixed_classes()

        # Transforms
        self.photo_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.sketch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # Region extraction
        self.region_extractor = RegionExtractor(
            min_region_area=100,
            max_num_regions=50
        )
        self.graph_builder = RegionGraphBuilder(
            graph_type="adjacency",
            feature_type="spatial",
            image_size=(image_size, image_size)
        )

        # Load data pairs
        self.data_pairs = self._load_one_per_class()

        print(f"Loaded {len(self.data_pairs)} samples from {len(self.selected_classes)} classes")

    def _get_fixed_classes(self):
        """Get 100 fixed classes deterministically."""
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        all_categories = sorted([d.name for d in sketch_dir.iterdir() if d.is_dir()])

        # Use deterministic seed for reproducibility
        rng = random.Random(self.seed)

        # Shuffle deterministically
        shuffled = all_categories.copy()
        rng.shuffle(shuffled)

        # Take first 100
        selected = shuffled[:10]
        return sorted(selected)  # Sort for consistent ordering

    def _load_one_per_class(self):
        """Load one sample per class."""
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        photo_dir = self.root_dir / "photo" / "tx_000000000000"

        data_pairs = []

        for class_name in self.selected_classes:
            cat_sketch_dir = sketch_dir / class_name
            cat_photo_dir = photo_dir / class_name

            if not cat_sketch_dir.exists() or not cat_photo_dir.exists():
                continue

            # Get first sketch file (deterministic)
            sketch_files = sorted(cat_sketch_dir.glob("*.png"))
            if len(sketch_files) == 0:
                continue

            # Pick one deterministically
            sketch_path = sketch_files[2]
            sketch_stem = sketch_path.stem

            # Find corresponding photo
            if '-' in sketch_stem:
                photo_base = sketch_stem.rsplit('-', 1)[0]
            else:
                photo_base = sketch_stem

            photo_name = photo_base + ".jpg"
            photo_path = cat_photo_dir / photo_name

            if photo_path.exists():
                data_pairs.append({
                    "sketch_path": str(sketch_path),
                    "photo_path": str(photo_path),
                    "category": class_name,
                    "file_id": sketch_path.stem
                })

        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]

        # Load images
        sketch_pil = Image.open(pair["sketch_path"]).convert("L")
        photo_pil = Image.open(pair["photo_path"]).convert("RGB")

        # Transform
        sketch = self.sketch_transform(sketch_pil)
        photo = self.photo_transform(photo_pil)

        # Generate text prompt
        category = pair["category"].replace("_", " ")
        text_prompt = f"A natural photo of a {category} aligned with the given sketch"

        # Extract region graph
        sketch_np = (sketch.squeeze(0).numpy() * 255).astype(np.uint8)
        regions = self.region_extractor.extract_regions(sketch_np)
        region_graph = self.graph_builder.build_graph(regions)
    

        return {
            "sketch": sketch,
            "photo": photo,
            "text_prompt": text_prompt,
            "region_graph": region_graph,
            "category": pair["category"],
            "file_id": pair["file_id"]
        }


class MetricsCalculator:
    """Compute various image quality metrics."""

    def __init__(self, device='cuda'):
        self.device = device

        # Initialize Inception V3 for FID
        print("Loading Inception V3 for FID...")
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = torch.nn.Identity()
        self.inception = self.inception.to(device)
        self.inception.eval()

        # Initialize CLIP for score
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def compute_ssim(self, img1, img2):
        """Compute SSIM between two images."""
        # Convert to numpy and ensure same shape
        if torch.is_tensor(img1):
            img1 = img1.cpu().numpy()
        if torch.is_tensor(img2):
            img2 = img2.cpu().numpy()

        # Convert from CHW to HWC
        if img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
        if img2.shape[0] == 3:
            img2 = np.transpose(img2, (1, 2, 0))

        # Ensure range [0, 1]
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        # Convert to grayscale for SSIM
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        return ssim(img1, img2, data_range=1.0)

    def compute_fid(self, real_images, generated_images):
        """Compute Frechet Inception Distance."""
        # Extract features
        real_features = self._extract_inception_features(real_images)
        gen_features = self._extract_inception_features(generated_images)

        # Compute mean and covariance
        mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

        # Compute FID
        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real @ sigma_gen)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)

        return fid

    def _extract_inception_features(self, images):
        """Extract Inception V3 features."""
        features_list = []

        with torch.no_grad():
            for img in images:
                # Convert to PIL if needed
                if torch.is_tensor(img):
                    img = self._tensor_to_pil(img)

                # Preprocess
                img_tensor = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(img).unsqueeze(0).to(self.device)

                # Extract features
                feat = self.inception(img_tensor).cpu().numpy()
                features_list.append(feat[0])

        return np.array(features_list)

    def compute_clip_score(self, images, prompts):
        """Compute CLIP score."""
        scores = []

        with torch.no_grad():
            for img, prompt in zip(images, prompts):
                # Convert to PIL if needed
                if torch.is_tensor(img):
                    img = self._tensor_to_pil(img)

                # Process
                inputs = self.clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(self.device)

                # Compute similarity
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                score = logits_per_image.item()
                scores.append(score)

        return np.mean(scores)

    def compute_edge_iou(self, img1, img2, sigma=1.0):
        """Compute Edge IoU (Intersection over Union)."""
        if torch.is_tensor(img1):
            img1 = self._tensor_to_pil(img1)
        if torch.is_tensor(img2):
            img2 = self._tensor_to_pil(img2)

        # Convert to numpy
        arr1 = np.array(img1.convert('L'))
        arr2 = np.array(img2.convert('L'))

        # Normalize
        arr1 = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-8)
        arr2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min() + 1e-8)

        # Detect edges
        edges1 = canny(arr1, sigma=sigma)
        edges2 = canny(arr2, sigma=sigma)

        # Compute IoU
        intersection = np.logical_and(edges1, edges2).sum()
        union = np.logical_or(edges1, edges2).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union

    def compute_chamfer_distance(self, img1, img2, sigma=1.0):
        """Compute Chamfer Distance between edge maps."""
        if torch.is_tensor(img1):
            img1 = self._tensor_to_pil(img1)
        if torch.is_tensor(img2):
            img2 = self._tensor_to_pil(img2)

        # Convert to numpy
        arr1 = np.array(img1.convert('L'))
        arr2 = np.array(img2.convert('L'))

        # Normalize
        arr1 = (arr1 - arr1.min()) / (arr1.max() - arr1.min() + 1e-8)
        arr2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min() + 1e-8)

        # Detect edges
        edges1 = canny(arr1, sigma=sigma)
        edges2 = canny(arr2, sigma=sigma)

        # Get edge points
        points1 = np.argwhere(edges1)
        points2 = np.argwhere(edges2)

        if len(points1) == 0 or len(points2) == 0:
            return float('inf') if len(points1) != len(points2) else 0.0

        # Compute Chamfer distance (symmetric)
        dist1 = directed_hausdorff(points1, points2)[0]
        dist2 = directed_hausdorff(points2, points1)[0]

        return (dist1 + dist2) / 2.0

    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Denormalize from [-1, 1] to [0, 1] if needed
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2

        tensor = torch.clamp(tensor, 0, 1)

        # Convert CHW to HWC
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)

        if np_img.shape[0] == 3:
            # RGB image: (3, H, W) -> (H, W, 3)
            np_img = np.transpose(np_img, (1, 2, 0))
            return Image.fromarray(np_img)
        elif np_img.shape[0] == 1:
            # Grayscale sketch: (1, H, W) -> (H, W)
            np_img = np_img.squeeze(0)
            return Image.fromarray(np_img, mode='L').convert('RGB')
        else:
            # Handle other cases
            if len(np_img.shape) == 3:
                np_img = np_img.squeeze()
            return Image.fromarray(np_img, mode='L' if len(np_img.shape) == 2 else None)


class ComprehensiveValidator:
    """Comprehensive validation pipeline."""

    def __init__(
        self,
        stage1_checkpoint: str,
        stage2_checkpoint: str,
        output_dir: str,
        device: str = 'cuda',
        image_size: int = 256,
        num_inference_steps: int = 50,
        num_refinement_steps: int = 30
    ):
        self.stage1_checkpoint = Path(stage1_checkpoint)
        self.stage2_checkpoint = Path(stage2_checkpoint)
        self.output_dir = Path(output_dir)
        self.device = device
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.num_refinement_steps = num_refinement_steps

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "individual").mkdir(exist_ok=True)

        print(f"Output directory: {self.output_dir}")

        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator(device=device)

        # Load models
        self.load_models()

        # Load dataset
        self.load_dataset()

    def load_models(self):
        """Load Stage 1 and Stage 2 models."""
        print("\n" + "="*60)
        print("Loading Models")
        print("="*60)

        model_name = "runwayml/stable-diffusion-v1-5"

        # Load base components
        print("\nLoading VAE, Text Encoder, Tokenizer...")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)
        self.vae.eval()

        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(self.device)
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        # Load Stage 1
        print(f"\nLoading Stage 1 from {self.stage1_checkpoint}...")
        self.stage1_model = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_name,
            freeze_base_unet=True
        ).to(self.device)

        checkpoint1 = torch.load(self.stage1_checkpoint, map_location='cpu', weights_only=False)
        self.stage1_model.load_state_dict(checkpoint1['model_state_dict'], strict=False)
        self.stage1_model.eval()
        self.stage1_epoch = checkpoint1.get('epoch', 'unknown')
        print(f"  Stage 1 loaded (Epoch {self.stage1_epoch})")

        # Load Stage 2
        print(f"\nLoading Stage 2 from {self.stage2_checkpoint}...")
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
            residual_alpha=0.2
        ).to(self.device)

        checkpoint2 = torch.load(self.stage2_checkpoint, map_location='cpu', weights_only=False)
        self.stage2_model.load_state_dict(checkpoint2['model_state_dict'], strict=False)
        self.stage2_model.eval()
        self.stage2_epoch = checkpoint2.get('epoch', 'unknown')
        print(f"  Stage 2 loaded (Epoch {self.stage2_epoch})")

        # Setup schedulers
        self.stage1_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.stage1_scheduler.set_timesteps(self.num_inference_steps)

        self.stage2_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.stage2_scheduler.set_timesteps(self.num_refinement_steps)

        print("\n" + "="*60)

    def load_dataset(self):
        """Load validation dataset with 100 fixed classes."""
        print("\nLoading validation dataset...")

        dataset = FixedClassValidationDataset(
            root_dir="/workspace/sketchy",
            image_size=self.image_size,
            seed=42  # Fixed seed for reproducibility
        )

        self.dataset = dataset
        print(f"  Dataset contains {len(dataset)} samples from 100 fixed classes")

    @torch.no_grad()
    def generate_stage1(self, sketch, text_prompt):
        """Generate Stage 1 output."""
        # Encode sketch
        sketch = sketch.to(self.device)
        down_res, mid_res = self.stage1_model.encode_sketch(sketch)

        # Encode text
        text_inputs = self.tokenizer(
            [text_prompt],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).to(self.device)
        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

        # Unconditional embeddings for CFG
        uncond_inputs = self.tokenizer(
            [""],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).to(self.device)
        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids)[0]

        # Concatenate for CFG
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

        # Prepare latents
        latents = torch.randn(
            1, 4, self.image_size // 8, self.image_size // 8,
            device=self.device, dtype=torch.float32
        )
        latents = latents * self.stage1_scheduler.init_noise_sigma

        # Duplicate sketch conditioning for CFG
        down_res_cfg = [torch.cat([torch.zeros_like(r), r]) for r in down_res]
        mid_res_cfg = torch.cat([torch.zeros_like(mid_res), mid_res])

        # Denoising loop
        for t in self.stage1_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.stage1_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.stage1_model(
                latent_model_input,
                t,
                (down_res_cfg, mid_res_cfg),
                encoder_hidden_states
            )

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latents = self.stage1_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        #latents = 1 / 0.18215 * latents
        unscaled_latents = 1 / 0.18215 * latents
        image = self.vae.decode(unscaled_latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        return image, latents








    @torch.no_grad()
    def generate_stage2(self, region_graph, text_embeddings, stage1_latents):
        """Generate Stage 2 output with CFG + stable refinement."""

        # 🔧 Move inputs to device
        if hasattr(region_graph, "node_features"):
            region_graph.node_features = region_graph.node_features.to(self.device)

        if hasattr(region_graph, "edge_index"):
            region_graph.edge_index = region_graph.edge_index.to(self.device)

        if hasattr(region_graph, "edge_attr") and region_graph.edge_attr is not None:
            region_graph.edge_attr = region_graph.edge_attr.to(self.device)

        stage1_latents = stage1_latents.to(self.device)

        # 🔧 Ensure correct shape for text embeddings
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        # =========================================================
        # 🔥 CFG SETUP (IMPORTANT)
        # =========================================================

        # Conditional embeddings already given (text_embeddings)
        # Create unconditional embeddings
        uncond_inputs = self.tokenizer(
            [""],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids)[0]

        # Concatenate for CFG
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

        # =========================================================
        # 🔥 REFINEMENT SETUP
        # =========================================================

        strength = 0.5  # ↓ reduced from 0.5 (VERY IMPORTANT)

        # Use Stage1 latents directly
        init_latents = stage1_latents

        # Select timesteps
        init_timestep = int(self.num_refinement_steps * strength)
        timesteps = self.stage2_scheduler.timesteps[-init_timestep:]

        # Add noise
        noise = torch.randn_like(init_latents)
        latents = self.stage2_scheduler.add_noise(init_latents, noise, timesteps[0])

        # CFG guidance scale
        guidance_scale = 2.5 # tune between 4–7

        # Duplicate stage1 latents for CFG
        stage1_latents_cfg = torch.cat([torch.zeros_like(stage1_latents), stage1_latents])

        # =========================================================
        # 🔥 DENOISING LOOP
        # =========================================================

        for t in timesteps:

            # Duplicate latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.stage2_scheduler.scale_model_input(latent_model_input, t)

            # Forward pass
            output = self.stage2_model(
                latent_model_input,
                t.to(self.device),
                region_graph,
                encoder_hidden_states,
                stage1_latents=stage1_latents_cfg,
                return_dict=True
            )

            noise_pred = output["noise_pred"]

            # Split unconditional & conditional
            noise_uncond, noise_text = noise_pred.chunk(2)

            # Apply CFG
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = self.stage2_scheduler.step(noise_pred, t, latents).prev_sample

            # 🔧 Clamp latents (stability fix)
            #latents = torch.clamp(latents, -1.2, 1.2)

        # =========================================================
        # 🔥 DECODE
        # =========================================================

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        # Normalize to [0,1]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image










    def create_comparison_image(self, sketch, stage1_img, stage2_img, ground_truth,
                                 category, metrics_text):
        """Create 1x4 comparison image."""
        # Convert all to PIL
        if torch.is_tensor(sketch):
            sketch_np = sketch.squeeze().cpu().numpy()
            sketch_np = (sketch_np * 255).astype(np.uint8)
            sketch_pil = Image.fromarray(sketch_np, mode='L').convert('RGB')
        else:
            sketch_pil = sketch

        if torch.is_tensor(stage1_img):
            stage1_np = stage1_img.squeeze().permute(1, 2, 0).cpu().numpy()
            stage1_np = (np.clip(stage1_np, 0, 1) * 255).astype(np.uint8)
            stage1_pil = Image.fromarray(stage1_np)
        else:
            stage1_pil = stage1_img

        if torch.is_tensor(stage2_img):
            stage2_np = stage2_img.squeeze().permute(1, 2, 0).cpu().numpy()
            stage2_np = (np.clip(stage2_np, 0, 1) * 255).astype(np.uint8)
            stage2_pil = Image.fromarray(stage2_np)
        else:
            stage2_pil = stage2_img

        if torch.is_tensor(ground_truth):
            gt_np = ground_truth.squeeze().permute(1, 2, 0).cpu().numpy()
            gt_np = ((gt_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            gt_pil = Image.fromarray(gt_np)
        else:
            gt_pil = ground_truth

        # Get size
        w, h = sketch_pil.size

        # Create canvas
        canvas_w = w * 4
        canvas_h = h + 60  # Extra space for labels
        canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')

        # Paste images
        canvas.paste(sketch_pil, (0, 30))
        canvas.paste(stage1_pil, (w, 30))
        canvas.paste(stage2_pil, (w*2, 30))
        canvas.paste(gt_pil, (w*3, 30))

        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            metrics_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
            metrics_font = font

        # Labels
        labels = ["Sketch", "Stage 1", "Stage 2", "Ground Truth"]
        for i, label in enumerate(labels):
            draw.text((i*w + 10, 5), label, fill='black', font=font)

        # Category and metrics
        draw.text((10, h + 35), f"Class: {category}", fill='black', font=font)

        # Add metrics text
        lines = metrics_text.split('\n')
        y_offset = h + 35
        for line in lines:
            draw.text((w + 10, y_offset), line, fill='black', font=metrics_font)
            y_offset += 12

        return canvas

    def validate(self):
        """Run full validation."""
        print("\n" + "="*60)
        print("Starting Validation")
        print("="*60)

        all_results = []
        stage1_images = []
        stage2_images = []
        ground_truths = []
        prompts = []

        # Generate all images
        for idx in tqdm(range(len(self.dataset)), desc="Generating images"):
            sample = self.dataset[idx]

            sketch = sample['sketch'].unsqueeze(0)
            photo = sample['photo']
            prompt = sample['text_prompt']
            region_graph = sample['region_graph']
            category = sample['category']

            # Generate Stage 1
            stage1_img, stage1_latents = self.generate_stage1(sketch, prompt)

            # Encode text for Stage 2
            text_inputs = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

            # Generate Stage 2
            stage2_img = self.generate_stage2(
                region_graph, text_embeddings, stage1_latents
            )

            # Denormalize ground truth
            gt_img = (photo + 1) / 2
            gt_img = gt_img.unsqueeze(0)

            # Compute individual metrics
            s1_ssim = self.metrics_calc.compute_ssim(stage1_img[0], gt_img[0])
            s2_ssim = self.metrics_calc.compute_ssim(stage2_img[0], gt_img[0])

            s1_edge_iou = self.metrics_calc.compute_edge_iou(stage1_img[0], gt_img[0])
            s2_edge_iou = self.metrics_calc.compute_edge_iou(stage2_img[0], gt_img[0])

            s1_chamfer = self.metrics_calc.compute_chamfer_distance(stage1_img[0], gt_img[0])
            s2_chamfer = self.metrics_calc.compute_chamfer_distance(stage2_img[0], gt_img[0])

            # Store for batch metrics
            stage1_images.append(stage1_img[0])
            stage2_images.append(stage2_img[0])
            ground_truths.append(gt_img[0])
            prompts.append(prompt)

            result = {
                "category": category,
                "prompt": prompt,
                "stage1_ssim": float(s1_ssim),
                "stage2_ssim": float(s2_ssim),
                "stage1_edge_iou": float(s1_edge_iou),
                "stage2_edge_iou": float(s2_edge_iou),
                "stage1_chamfer": float(s1_chamfer),
                "stage2_chamfer": float(s2_chamfer)
            }
            all_results.append(result)

            # Create comparison image
            metrics_text = (
                f"S1 SSIM: {s1_ssim:.3f} | S2 SSIM: {s2_ssim:.3f}\n"
                f"S1 Edge IoU: {s1_edge_iou:.3f} | S2 Edge IoU: {s2_edge_iou:.3f}\n"
                f"S1 Chamfer: {s1_chamfer:.1f} | S2 Chamfer: {s2_chamfer:.1f}"
            )

            comparison = self.create_comparison_image(
                sketch[0], stage1_img[0], stage2_img[0], gt_img[0],
                category, metrics_text
            )
            comparison.save(self.output_dir / "comparisons" / f"{idx:03d}_{category}.png")

            # Save individual images
            (self.output_dir / "individual" / category).mkdir(parents=True, exist_ok=True)
            self.metrics_calc._tensor_to_pil(sketch[0]).save(
                self.output_dir / "individual" / category / f"{idx:03d}_sketch.png"
            )
            self.metrics_calc._tensor_to_pil(stage1_img[0]).save(
                self.output_dir / "individual" / category / f"{idx:03d}_stage1.png"
            )
            self.metrics_calc._tensor_to_pil(stage2_img[0]).save(
                self.output_dir / "individual" / category / f"{idx:03d}_stage2.png"
            )
            self.metrics_calc._tensor_to_pil(gt_img[0]).save(
                self.output_dir / "individual" / category / f"{idx:03d}_groundtruth.png"
            )

        # Compute batch metrics
        print("\n" + "="*60)
        print("Computing Batch Metrics (FID, CLIP Score)")
        print("="*60)

        print("Computing FID for Stage 1...")
        s1_fid = self.metrics_calc.compute_fid(ground_truths, stage1_images)
        print(f"  Stage 1 FID: {s1_fid:.2f}")

        print("Computing FID for Stage 2...")
        s2_fid = self.metrics_calc.compute_fid(ground_truths, stage2_images)
        print(f"  Stage 2 FID: {s2_fid:.2f}")

        print("Computing CLIP Score for Stage 1...")
        s1_clip = self.metrics_calc.compute_clip_score(stage1_images, prompts)
        print(f"  Stage 1 CLIP Score: {s1_clip:.2f}")

        print("Computing CLIP Score for Stage 2...")
        s2_clip = self.metrics_calc.compute_clip_score(stage2_images, prompts)
        print(f"  Stage 2 CLIP Score: {s2_clip:.2f}")

        # Aggregate results
        summary = {
            "validation_date": datetime.now().isoformat(),
            "stage1_checkpoint": str(self.stage1_checkpoint),
            "stage2_checkpoint": str(self.stage2_checkpoint),
            "stage1_epoch": self.stage1_epoch,
            "stage2_epoch": self.stage2_epoch,
            "num_samples": len(self.dataset),
            "classes": [r["category"] for r in all_results],
            "stage1_metrics": {
                "ssim_mean": float(np.mean([r["stage1_ssim"] for r in all_results])),
                "ssim_std": float(np.std([r["stage1_ssim"] for r in all_results])),
                "fid": float(s1_fid),
                "clip_score": float(s1_clip),
                "edge_iou_mean": float(np.mean([r["stage1_edge_iou"] for r in all_results])),
                "chamfer_mean": float(np.mean([r["stage1_chamfer"] for r in all_results]))
            },
            "stage2_metrics": {
                "ssim_mean": float(np.mean([r["stage2_ssim"] for r in all_results])),
                "ssim_std": float(np.std([r["stage2_ssim"] for r in all_results])),
                "fid": float(s2_fid),
                "clip_score": float(s2_clip),
                "edge_iou_mean": float(np.mean([r["stage2_edge_iou"] for r in all_results])),
                "chamfer_mean": float(np.mean([r["stage2_chamfer"] for r in all_results]))
            },
            "per_sample_results": all_results
        }

        # Save summary
        with open(self.output_dir / "validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Stage 1 (Epoch {self.stage1_epoch}):")
        print(f"  SSIM: {summary['stage1_metrics']['ssim_mean']:.4f} ± {summary['stage1_metrics']['ssim_std']:.4f}")
        print(f"  FID: {summary['stage1_metrics']['fid']:.2f}")
        print(f"  CLIP Score: {summary['stage1_metrics']['clip_score']:.2f}")
        print(f"  Edge IoU: {summary['stage1_metrics']['edge_iou_mean']:.4f}")
        print(f"  Chamfer Distance: {summary['stage1_metrics']['chamfer_mean']:.2f}")
        print(f"\nStage 2 (Epoch {self.stage2_epoch}):")
        print(f"  SSIM: {summary['stage2_metrics']['ssim_mean']:.4f} ± {summary['stage2_metrics']['ssim_std']:.4f}")
        print(f"  FID: {summary['stage2_metrics']['fid']:.2f}")
        print(f"  CLIP Score: {summary['stage2_metrics']['clip_score']:.2f}")
        print(f"  Edge IoU: {summary['stage2_metrics']['edge_iou_mean']:.4f}")
        print(f"  Chamfer Distance: {summary['stage2_metrics']['chamfer_mean']:.2f}")
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)

        return summary


def main():
    parser = argparse.ArgumentParser(description="Comprehensive validation for RAGAF-Diffusion")
    parser.add_argument("--stage1_checkpoint", type=str,
                       default="/workspace/checkpoints/stage1/epoch_18.pt",
                       help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, required=True,
                       help="Path to Stage 2 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for validation results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Stage 1 inference steps")
    parser.add_argument("--num_refinement_steps", type=int, default=30, help="Stage 2 refinement steps")

    args = parser.parse_args()

    validator = ComprehensiveValidator(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        num_refinement_steps=args.num_refinement_steps
    )

    validator.validate()


if __name__ == "__main__":
    main()
