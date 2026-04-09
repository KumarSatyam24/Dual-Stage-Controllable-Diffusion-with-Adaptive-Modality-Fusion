#!/usr/bin/env python3
"""
Validation on Unseen Classes for RAGAF-Diffusion

Validates on 25 classes NOT used during training (from the 125 total Sketchy classes).
Loads 5 images per class = 125 total validation samples.

Usage:
    python validate_unseen_classes.py --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
                                     --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
                                     --output_dir /workspace/validation_results/unseen_classes_v3

Author: Claude
"""

import sys
import os
from pathlib import Path
import argparse
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from scipy.spatial import cKDTree

from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights
from scipy.linalg import sqrtm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

# Import LPIPS for perceptual metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS metric will be skipped.")

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from src.models.stage2_refinement import Stage2SemanticRefinement
from src.data.region_extraction import RegionExtractor
from src.data.region_graph import RegionGraphBuilder


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UnseenClassValidationDataset(Dataset):
    """
    Dataset for validating on unseen classes (not in training set).
    Selects 25 classes from the 125 total that are NOT in the training set of 100.
    Loads N images per class for validation.
    """

    def __init__(self, root_dir: str, image_size: int = 256, seed: int = 42, images_per_class: int = 5):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.seed = seed
        self.images_per_class = images_per_class

        # Get the 100 training classes
        self.training_classes = self._get_training_classes()

        # Get the 25 unseen classes (not in training)
        self.unseen_classes = self._get_unseen_classes()

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
        self.data_pairs = self._load_data_pairs()

        print(f"Loaded {len(self.data_pairs)} samples from {len(self.unseen_classes)} unseen classes")
        print(f"Classes: {', '.join(sorted(self.unseen_classes))}")

    def _get_training_classes(self):
        """Get the 100 classes used for training (same as FixedClassValidationDataset)."""
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        all_categories = sorted([d.name for d in sketch_dir.iterdir() if d.is_dir()])

        # Use deterministic seed for reproducibility (same as training)
        rng = random.Random(self.seed)
        shuffled = all_categories.copy()
        rng.shuffle(shuffled)

        # Take first 100 (these were used for training)
        selected = shuffled[:100]
        return set(selected)

    def _get_unseen_classes(self):
        """Get the 25 classes NOT used for training."""
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        all_categories = sorted([d.name for d in sketch_dir.iterdir() if d.is_dir()])

        # Use deterministic seed for reproducibility
        rng = random.Random(self.seed)
        shuffled = all_categories.copy()
        rng.shuffle(shuffled)

        # Take the last 25 (not in training set)
        unseen = shuffled[100:]
        return sorted(unseen)

    def _load_data_pairs(self):
        """Load N samples per unseen class."""
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        photo_dir = self.root_dir / "photo" / "tx_000000000000"

        data_pairs = []

        for class_name in self.unseen_classes:
            cat_sketch_dir = sketch_dir / class_name
            cat_photo_dir = photo_dir / class_name

            if not cat_sketch_dir.exists() or not cat_photo_dir.exists():
                print(f"Warning: Directory missing for {class_name}")
                continue

            # Get sketch files (deterministic order)
            sketch_files = sorted(cat_sketch_dir.glob("*.png"))

            if len(sketch_files) == 0:
                print(f"Warning: No sketches found for {class_name}")
                continue

            # Take first N images per class
            selected_sketches = sketch_files[:self.images_per_class]

            for sketch_path in selected_sketches:
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
                else:
                    print(f"Warning: Photo not found for {sketch_path}")

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
        text_prompt = f"A detailed realistic photograph of a {category}, high quality, sharp focus, natural lighting"

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

        # Initialize LPIPS if available
        if LPIPS_AVAILABLE:
            print("Loading LPIPS (AlexNet)...")
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

    def compute_ssim(self, img1, img2):
        """Compute SSIM between two images."""
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
            return ssim(img1, img2, channel_axis=2, data_range=1.0)
        if len(img2.shape) == 3 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        return ssim(img1, img2, data_range=1.0)

    def compute_lpips(self, img1, img2):
        """Compute LPIPS perceptual distance."""
        if self.lpips_model is None:
            return None

        # Ensure tensors on device
        if not torch.is_tensor(img1):
            img1 = transforms.ToTensor()(img1).unsqueeze(0)
        if not torch.is_tensor(img2):
            img2 = transforms.ToTensor()(img2).unsqueeze(0)

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # LPIPS expects values in [-1, 1]
        if img1.max() <= 1.0:
            img1 = img1 * 2 - 1
        if img2.max() <= 1.0:
            img2 = img2 * 2 - 1

        with torch.no_grad():
            dist = self.lpips_model(img1, img2)

        return dist.item()

    def compute_fid(self, real_images, generated_images, batch_size=32):
        """Compute Frechet Inception Distance with batched feature extraction."""
        # Extract features in batches
        real_features = self._extract_inception_features_batched(real_images, batch_size)
        gen_features = self._extract_inception_features_batched(generated_images, batch_size)

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

    def _extract_inception_features_batched(self, images, batch_size=32):
        """Extract Inception V3 features in batches for efficiency."""
        all_features = []
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_tensors = []

                for img in batch_images:
                    if torch.is_tensor(img):
                        img = self._tensor_to_pil(img)
                    batch_tensors.append(preprocess(img).unsqueeze(0))

                batch = torch.cat(batch_tensors, dim=0).to(self.device)
                features = self.inception(batch).cpu().numpy()
                all_features.append(features)

        return np.concatenate(all_features, axis=0)

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
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                score = (image_embeds * text_embeds).sum().item()
                scores.append(score)

        return np.mean(scores)

    def compute_edge_iou(self, img1, img2, sigma=1.5):
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

    def compute_chamfer_distance(self, img1, img2, sigma=1.5):
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

        # Compute Chamfer distance (symmetric) using k-d trees
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)

        dist1, _ = tree1.query(points2)
        dist2, _ = tree2.query(points1)

        return dist1.mean() + dist2.mean()

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


class UnseenClassValidator:
    """Validation pipeline for unseen classes."""

    def __init__(
        self,
        stage1_checkpoint: str,
        stage2_checkpoint: str,
        output_dir: str,
        device: str = 'cuda',
        image_size: int = 256,
        num_inference_steps: int = 50,
        num_refinement_steps: int = 30,
        images_per_class: int = 5,
        seed: int = 42
    ):
        self.stage1_checkpoint = Path(stage1_checkpoint)
        self.stage2_checkpoint = Path(stage2_checkpoint)
        self.output_dir = Path(output_dir)
        self.device = device
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.num_refinement_steps = num_refinement_steps
        self.images_per_class = images_per_class
        self.seed = seed

        # Set seeds for reproducibility
        set_seed(seed)

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
        """Load validation dataset with 25 unseen classes."""
        print("\nLoading unseen class validation dataset...")

        dataset = UnseenClassValidationDataset(
            root_dir="/workspace/sketchy",
            image_size=self.image_size,
            seed=42,  # Same seed as training for consistent split
            images_per_class=self.images_per_class
        )

        self.dataset = dataset
        print(f"  Dataset contains {len(dataset)} samples from 25 unseen classes")

    @torch.no_grad()
    def generate_stage1(self, sketch, text_prompt, generator=None):
        """Generate Stage 1 output with deterministic latents."""
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

        # Prepare latents with fixed generator for reproducibility
        latents = torch.randn(
            1, 4, self.image_size // 8, self.image_size // 8,
            device=self.device, dtype=torch.float32,
            generator=generator
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
        unscaled_latents = 1 / 0.18215 * latents
        with torch.cuda.amp.autocast():  # Mixed precision for VAE decode
            image = self.vae.decode(unscaled_latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        return image, latents

    @torch.no_grad()
    def generate_stage2(self, region_graph, text_embeddings, stage1_latents, generator=None):
        """Generate Stage 2 output with CFG + consistent conditioning."""

        # Move inputs to device
        if hasattr(region_graph, "node_features"):
            region_graph.node_features = region_graph.node_features.to(self.device)

        if hasattr(region_graph, "edge_index"):
            region_graph.edge_index = region_graph.edge_index.to(self.device)

        if hasattr(region_graph, "edge_attr") and region_graph.edge_attr is not None:
            region_graph.edge_attr = region_graph.edge_attr.to(self.device)

        stage1_latents = stage1_latents.to(self.device)

        # Ensure correct shape for text embeddings
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        # CFG SETUP
        uncond_inputs = self.tokenizer(
            [""],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        uncond_embeddings = self.text_encoder(uncond_inputs.input_ids)[0]

        # Concatenate for CFG
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

        # Refinement SETUP
        strength = 0.7

        # Use Stage1 latents directly
        init_latents = stage1_latents

        # Select timesteps
        init_timestep = int(self.num_refinement_steps * strength)
        timesteps = self.stage2_scheduler.timesteps[-init_timestep:]

        # Add noise with fixed generator
        noise = torch.randn_like(init_latents) * 0.3
        if generator is not None:
            noise = torch.randn(
                init_latents.shape,
                device=self.device, dtype=torch.float32,
                generator=generator
            ) * 0.4
        latents = self.stage2_scheduler.add_noise(init_latents, noise, timesteps[0])

        # CFG guidance scale
        guidance_scale = 6

        # Duplicate stage1 latents for CFG
        stage1_latents_cfg = torch.cat([stage1_latents, stage1_latents])

        # Duplicate region graph for CFG consistency
        region_graph_cfg = region_graph
        if hasattr(region_graph, 'node_features'):
            # Duplicate node features for CFG
            region_graph_cfg.node_features = torch.cat([
                region_graph.node_features,
                region_graph.node_features
            ], dim=0)

        # Denoising LOOP
        for t in timesteps:
            # Duplicate latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.stage2_scheduler.scale_model_input(latent_model_input, t)

            # Forward pass
            output = self.stage2_model(
                latent_model_input,
                t.to(self.device),
                region_graph_cfg,  # Use duplicated graph
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

        # Decode with mixed precision
        latents = 1 / 0.18215 * latents
        with torch.cuda.amp.autocast():
            image = self.vae.decode(latents).sample

        # Normalize to [0,1]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def create_comparison_image(self, sketch, stage1_img, stage2_img, ground_truth,
                                 category, file_id, metrics_text, refinement_gains=None):
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
        canvas_h = h + 100  # Extra space for labels and gains
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
            gain_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font = ImageFont.load_default()
            metrics_font = font
            gain_font = font

        # Labels
        labels = ["Sketch", "Stage 1", "Stage 2", "Ground Truth"]
        for i, label in enumerate(labels):
            draw.text((i*w + 10, 5), label, fill='black', font=font)

        # Category and file_id
        draw.text((10, h + 35), f"Class: {category}", fill='black', font=font)
        draw.text((10, h + 55), f"ID: {file_id}", fill='black', font=font)

        # Add metrics text
        lines = metrics_text.split('\n')
        y_offset = h + 35
        for line in lines:
            draw.text((w + 10, y_offset), line, fill='black', font=metrics_font)
            y_offset += 12

        # Add refinement gains if provided
        if refinement_gains:
            gain_text = (
                f"SSIM: {refinement_gains['ssim_gain']:+.3f} | "
                f"Edge: {refinement_gains['edge_gain']:+.3f} | "
                f"CLIP: {refinement_gains['clip_gain']:+.2f}"
            )
            draw.text((10, h + 80), f"Stage 2 Gains: {gain_text}", fill='green', font=gain_font)

        return canvas

    def validate(self):
        """Run validation on unseen classes."""
        print("\n" + "="*60)
        print("Starting Unseen Class Validation")
        print("="*60)

        # Create fixed generator for reproducible latents
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

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
            file_id = sample['file_id']

            # Generate Stage 1 with fixed generator
            stage1_img, stage1_latents = self.generate_stage1(sketch, prompt, generator=generator)

            # Encode text for Stage 2
            text_inputs = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

            # Generate Stage 2 with same generator
            stage2_img = self.generate_stage2(
                region_graph, text_embeddings, stage1_latents, generator=generator
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

            # Compute LPIPS if available
            s1_lpips = None
            s2_lpips = None
            if self.metrics_calc.lpips_model is not None:
                s1_lpips = self.metrics_calc.compute_lpips(stage1_img[0], gt_img[0])
                s2_lpips = self.metrics_calc.compute_lpips(stage2_img[0], gt_img[0])

            # Compute refinement gains
            ssim_gain = s2_ssim - s1_ssim
            edge_gain = s2_edge_iou - s1_edge_iou

            # Store for batch metrics
            stage1_images.append(stage1_img[0])
            stage2_images.append(stage2_img[0])
            ground_truths.append(gt_img[0])
            prompts.append(prompt)

            result = {
                "category": category,
                "file_id": file_id,
                "prompt": prompt,
                "stage1_ssim": float(s1_ssim),
                "stage2_ssim": float(s2_ssim),
                "stage1_edge_iou": float(s1_edge_iou),
                "stage2_edge_iou": float(s2_edge_iou),
                "stage1_chamfer": float(s1_chamfer),
                "stage2_chamfer": float(s2_chamfer),
                "refinement_gains": {
                    "ssim_gain": float(ssim_gain),
                    "edge_gain": float(edge_gain)
                }
            }

            if s1_lpips is not None:
                result["stage1_lpips"] = float(s1_lpips)
                result["stage2_lpips"] = float(s2_lpips)
                result["refinement_gains"]["lpips_gain"] = float(s2_lpips - s1_lpips)

            all_results.append(result)

            # Create refinement gains dict for visualization
            refinement_gains = {
                "ssim_gain": float(ssim_gain),
                "edge_gain": float(edge_gain),
                "clip_gain": 0.0  # Will be updated after CLIP scores
            }

            # Create comparison image
            metrics_text = (
                f"S1 SSIM: {s1_ssim:.3f} | S2 SSIM: {s2_ssim:.3f}\n"
                f"S1 Edge IoU: {s1_edge_iou:.3f} | S2 Edge IoU: {s2_edge_iou:.3f}\n"
                f"S1 Chamfer: {s1_chamfer:.1f} | S2 Chamfer: {s2_chamfer:.1f}"
            )
            if s1_lpips is not None:
                metrics_text += f"\nS1 LPIPS: {s1_lpips:.3f} | S2 LPIPS: {s2_lpips:.3f}"

            comparison = self.create_comparison_image(
                sketch[0], stage1_img[0], stage2_img[0], gt_img[0],
                category, file_id, metrics_text, refinement_gains
            )
            comparison.save(self.output_dir / "comparisons" / f"{idx:03d}_{category}_{file_id}.png")

            # Save individual images
            (self.output_dir / "individual" / category).mkdir(parents=True, exist_ok=True)
            self.metrics_calc._tensor_to_pil(sketch[0]).save(
                self.output_dir / "individual" / category / f"{file_id}_sketch.png"
            )
            self.metrics_calc._tensor_to_pil(stage1_img[0]).save(
                self.output_dir / "individual" / category / f"{file_id}_stage1.png"
            )
            self.metrics_calc._tensor_to_pil(stage2_img[0]).save(
                self.output_dir / "individual" / category / f"{file_id}_stage2.png"
            )
            self.metrics_calc._tensor_to_pil(gt_img[0]).save(
                self.output_dir / "individual" / category / f"{file_id}_groundtruth.png"
            )

        # Compute batch metrics
        print("\n" + "="*60)
        print("Computing Batch Metrics (FID, CLIP Score)")
        print("="*60)

        print("Computing FID for Stage 1...")
        s1_fid = self.metrics_calc.compute_fid(ground_truths, stage1_images, batch_size=32)
        print(f"  Stage 1 FID: {s1_fid:.2f}")

        print("Computing FID for Stage 2...")
        s2_fid = self.metrics_calc.compute_fid(ground_truths, stage2_images, batch_size=32)
        print(f"  Stage 2 FID: {s2_fid:.2f}")

        print("Computing CLIP Score for Stage 1...")
        s1_clip = self.metrics_calc.compute_clip_score(stage1_images, prompts)
        print(f"  Stage 1 CLIP Score: {s1_clip:.2f}")

        print("Computing CLIP Score for Stage 2...")
        s2_clip = self.metrics_calc.compute_clip_score(stage2_images, prompts)
        print(f"  Stage 2 CLIP Score: {s2_clip:.2f}")

        # Compute average LPIPS
        s1_lpips_mean = None
        s2_lpips_mean = None
        if self.metrics_calc.lpips_model is not None:
            s1_lpips_list = [r["stage1_lpips"] for r in all_results if "stage1_lpips" in r]
            s2_lpips_list = [r["stage2_lpips"] for r in all_results if "stage2_lpips" in r]
            if s1_lpips_list:
                s1_lpips_mean = np.mean(s1_lpips_list)
                s2_lpips_mean = np.mean(s2_lpips_list)
                print(f"  Stage 1 LPIPS: {s1_lpips_mean:.4f}")
                print(f"  Stage 2 LPIPS: {s2_lpips_mean:.4f}")

        # Per-class metrics
        class_metrics = {}
        for result in all_results:
            cls = result["category"]
            if cls not in class_metrics:
                class_metrics[cls] = {
                    "count": 0,
                    "stage1_ssim_sum": 0,
                    "stage2_ssim_sum": 0,
                    "stage1_edge_iou_sum": 0,
                    "stage2_edge_iou_sum": 0,
                    "ssim_gain_sum": 0,
                    "edge_gain_sum": 0,
                }
            class_metrics[cls]["count"] += 1
            class_metrics[cls]["stage1_ssim_sum"] += result["stage1_ssim"]
            class_metrics[cls]["stage2_ssim_sum"] += result["stage2_ssim"]
            class_metrics[cls]["stage1_edge_iou_sum"] += result["stage1_edge_iou"]
            class_metrics[cls]["stage2_edge_iou_sum"] += result["stage2_edge_iou"]
            class_metrics[cls]["ssim_gain_sum"] += result["refinement_gains"]["ssim_gain"]
            class_metrics[cls]["edge_gain_sum"] += result["refinement_gains"]["edge_gain"]

        per_class_summary = {}
        for cls, metrics in class_metrics.items():
            count = metrics["count"]
            per_class_summary[cls] = {
                "count": count,
                "stage1_ssim": metrics["stage1_ssim_sum"] / count,
                "stage2_ssim": metrics["stage2_ssim_sum"] / count,
                "stage1_edge_iou": metrics["stage1_edge_iou_sum"] / count,
                "stage2_edge_iou": metrics["stage2_edge_iou_sum"] / count,
                "ssim_gain": metrics["ssim_gain_sum"] / count,
                "edge_gain": metrics["edge_gain_sum"] / count,
            }

        # Compute overall refinement gains
        all_ssim_gains = [r["refinement_gains"]["ssim_gain"] for r in all_results]
        all_edge_gains = [r["refinement_gains"]["edge_gain"] for r in all_results]
        clip_gain = s2_clip - s1_clip
        fid_gain = s2_fid - s1_fid  # Lower is better for FID

        refinement_summary = {
            "ssim_gain_mean": float(np.mean(all_ssim_gains)),
            "ssim_gain_std": float(np.std(all_ssim_gains)),
            "edge_gain_mean": float(np.mean(all_edge_gains)),
            "edge_gain_std": float(np.std(all_edge_gains)),
            "clip_gain": float(clip_gain),
            "fid_delta": float(fid_gain)
        }

        # Aggregate results
        summary = {
            "validation_type": "unseen_classes",
            "validation_date": datetime.now().isoformat(),
            "seed": self.seed,
            "num_classes": 25,
            "images_per_class": self.images_per_class,
            "total_samples": len(self.dataset),
            "stage1_checkpoint": str(self.stage1_checkpoint),
            "stage2_checkpoint": str(self.stage2_checkpoint),
            "stage1_epoch": self.stage1_epoch,
            "stage2_epoch": self.stage2_epoch,
            "unseen_classes": sorted(list(self.dataset.unseen_classes)),
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
            "refinement_gains": refinement_summary,
            "per_class_metrics": per_class_summary,
            "per_sample_results": all_results
        }

        if s1_lpips_mean is not None:
            summary["stage1_metrics"]["lpips_mean"] = float(s1_lpips_mean)
            summary["stage2_metrics"]["lpips_mean"] = float(s2_lpips_mean)
            summary["refinement_gains"]["lpips_gain"] = float(s2_lpips_mean - s1_lpips_mean)

        # Save summary
        with open(self.output_dir / "validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("UNSEEN CLASS VALIDATION SUMMARY")
        print("="*60)
        print(f"Classes: 25 unseen classes")
        print(f"Samples: {len(self.dataset)} total ({self.images_per_class} per class)")
        print(f"Seed: {self.seed} (reproducible)")
        print(f"\nUnseen classes: {', '.join(sorted(self.dataset.unseen_classes)[:10])}...")
        print(f"\nStage 1 (Epoch {self.stage1_epoch}):")
        print(f"  SSIM: {summary['stage1_metrics']['ssim_mean']:.4f} ± {summary['stage1_metrics']['ssim_std']:.4f}")
        print(f"  FID: {summary['stage1_metrics']['fid']:.2f}")
        print(f"  CLIP Score: {summary['stage1_metrics']['clip_score']:.2f}")
        print(f"  Edge IoU: {summary['stage1_metrics']['edge_iou_mean']:.4f}")
        print(f"  Chamfer Distance: {summary['stage1_metrics']['chamfer_mean']:.2f}")
        if s1_lpips_mean is not None:
            print(f"  LPIPS: {s1_lpips_mean:.4f}")
        print(f"\nStage 2 (Epoch {self.stage2_epoch}):")
        print(f"  SSIM: {summary['stage2_metrics']['ssim_mean']:.4f} ± {summary['stage2_metrics']['ssim_std']:.4f}")
        print(f"  FID: {summary['stage2_metrics']['fid']:.2f}")
        print(f"  CLIP Score: {summary['stage2_metrics']['clip_score']:.2f}")
        print(f"  Edge IoU: {summary['stage2_metrics']['edge_iou_mean']:.4f}")
        print(f"  Chamfer Distance: {summary['stage2_metrics']['chamfer_mean']:.2f}")
        if s2_lpips_mean is not None:
            print(f"  LPIPS: {s2_lpips_mean:.4f}")
        print(f"\n{'='*60}")
        print("REFINEMENT GAINS (Stage 2 over Stage 1):")
        print(f"{'='*60}")
        print(f"  SSIM Gain: {refinement_summary['ssim_gain_mean']:+.4f} ± {refinement_summary['ssim_gain_std']:.4f}")
        print(f"  Edge IoU Gain: {refinement_summary['edge_gain_mean']:+.4f} ± {refinement_summary['edge_gain_std']:.4f}")
        print(f"  CLIP Gain: {refinement_summary['clip_gain']:+.2f}")
        print(f"  FID Delta: {refinement_summary['fid_delta']:+.2f} (lower is better)")
        if "lpips_gain" in refinement_summary:
            lpips_gain = refinement_summary["lpips_gain"]
            print(f"  LPIPS Gain: {lpips_gain:+.4f} (lower is better)")
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)

        return summary


def main():
    parser = argparse.ArgumentParser(description="Unseen class validation for RAGAF-Diffusion")
    parser.add_argument("--stage1_checkpoint", type=str,
                       default="/workspace/checkpoints/stage1/epoch_18.pt",
                       help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str,default="/workspace/checkpoints/stage1/epoch_6.pt", required=True,
                       help="Path to Stage 2 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for validation results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Stage 1 inference steps")
    parser.add_argument("--num_refinement_steps", type=int, default=40, help="Stage 2 refinement steps")
    parser.add_argument("--images_per_class", type=int, default=10, help="Number of images per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    validator = UnseenClassValidator(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        image_size=args.image_size,
        num_inference_steps=args.num_inference_steps,
        num_refinement_steps=args.num_refinement_steps,
        images_per_class=args.images_per_class,
        seed=args.seed
    )

    validator.validate()


if __name__ == "__main__":
    main()
