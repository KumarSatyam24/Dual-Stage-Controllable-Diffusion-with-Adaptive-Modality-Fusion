"""
Main Training Script for RAGAF-Diffusion

This script trains the RAGAF-Diffusion model in a dual-stage manner:
1. Stage 1: Sketch-guided diffusion for coarse structure
2. Stage 2: Semantic refinement with RAGAF attention

Supports:
- Single-stage or dual-stage training
- Mixed precision (fp16/bf16)
- Gradient accumulation
- Checkpointing and resuming
- Weights & Biases logging
- RunPod cloud training

Author: RAGAF-Diffusion Research Team
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
from tqdm.auto import tqdm
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from torchvision.utils import make_grid, save_image
import json

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import ModelConfig, DataConfig, TrainingConfig, get_default_config
from src.datasets.sketchy_dataset import SketchyDataset, collate_fn as sketchy_collate
from src.datasets.coco_dataset import COCODataset, collate_fn as coco_collate
from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion
from src.models.stage2_refinement import Stage2SemanticRefinement
from src.data.sketch_extraction import SketchExtractor
from src.data.region_extraction import RegionExtractor
from src.data.region_graph import RegionGraphBuilder


class RAGAFDiffusionTrainer:
    """
    Trainer for RAGAF-Diffusion model.
    
    Handles both Stage 1 and Stage 2 training with all necessary components.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
            training_config: Training configuration
        """
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        
        # Force CUDA device before Accelerator init so it picks up the GPU
        # (Accelerate may default to CPU when invoked via plain `python` without `accelerate launch`)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        # Initialize accelerator for distributed training and mixed precision
        self.accelerator = Accelerator(
            mixed_precision=training_config.mixed_precision,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            log_with="wandb" if training_config.use_wandb else None,
            project_dir=training_config.checkpoint_dir
        )

        # Sanity-check device — if Accelerate still reports CPU despite CUDA being available, override
        if str(self.accelerator.device) == "cpu" and torch.cuda.is_available():
            print("WARNING: Accelerate reported CPU but CUDA is available. Forcing cuda:0.")
            self.accelerator.state.device = torch.device("cuda", 0)
        
        # Setup logging
        if self.accelerator.is_main_process:
            if training_config.use_wandb:
                import wandb
                wandb.init(
                    project=training_config.wandb_project,
                    name=training_config.wandb_run_name,
                    config={
                        "model": vars(model_config),
                        "data": vars(data_config),
                        "training": vars(training_config)
                    }
                )
        
        # Initialize models
        self.setup_models()

        # Initialize shared components (VAE, text encoder, scheduler)
        self.setup_shared_components()

        # Initialize datasets
        self.setup_datasets()

        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize LPIPS for perceptual loss (Stage 2)
        if self.training_config.train_stage in ["stage2", "both"]:
            print("Initializing LPIPS perceptual loss...")
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.accelerator.device)
            self.lpips_loss.requires_grad_(False)
            
            # Adaptive Refinement Control State
            self.running_delta_mean = 0.0
            self.delta_ema_momentum = 0.9
            self.alpha_history = []
        
        print(f"Trainer initialized on device: {self.accelerator.device}")
        print(f"Mixed precision: {training_config.mixed_precision}")
        print(f"Training stage: {training_config.train_stage}")
    
    def setup_models(self):
        """Setup models for training."""
        print("Loading pretrained models...")
        
        # Load Stage 1 model (always needed for Stage 2 conditioning)
        self.stage1_model = Stage1SketchGuidedDiffusion(
            pretrained_model_name=self.model_config.pretrained_model_name,
            sketch_encoder_channels=self.model_config.sketch_encoder_channels,
            freeze_base_unet=self.model_config.freeze_stage1_unet,
            use_lora=self.model_config.use_lora,
            lora_rank=self.model_config.lora_rank
        )
        
        # Load Stage 1 weights if we are in Stage 2
        if self.training_config.train_stage == "stage2":
            # Check if checkpoint exists in config or environment
            stage1_path = getattr(self.training_config, "stage1_checkpoint", None)
            if not stage1_path:
                # Try to find it in common locations
                stage1_path = "/workspace/checkpoints/stage1/epoch_18.pt"
            
            if os.path.exists(stage1_path):
                print(f"Loading Stage 1 weights from {stage1_path}")
                ckpt = torch.load(stage1_path, map_location="cpu", weights_only=False)
                self.stage1_model.load_state_dict(ckpt["model_state_dict"])
            else:
                print(f"WARNING: Stage 1 checkpoint not found at {stage1_path}. Stage 2 training might be suboptimal.")
        
        # Stage 1 model should be frozen during Stage 2 training
        if self.training_config.train_stage == "stage2":
            self.stage1_model.requires_grad_(False)
            self.stage1_model.eval()
            self.stage1_model.to(self.accelerator.device)  # Keep on GPU (RTX 5090 32GB)
        
        # Stage 2 model
        if self.training_config.train_stage in ["stage2", "both"]:
            # Load UNet
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(
                self.model_config.pretrained_model_name,
                subfolder="unet"
            )
            
            # Gradient checkpointing — needed for Stage 2 since Stage 1 also occupies GPU
            unet.enable_gradient_checkpointing()
            # Flash Attention via xFormers for throughput (optional speedup)
            try:
                unet.set_use_memory_efficient_attention_xformers(True)
                print("✅ xFormers attention enabled")
            except Exception:
                # Fallback to PyTorch's native SDPA
                pass
            
            self.stage2_model = Stage2SemanticRefinement(
                unet=unet,
                node_feature_dim=self.model_config.node_feature_dim,
                text_dim=self.model_config.text_dim,
                hidden_dim=self.model_config.hidden_dim,
                num_graph_layers=self.model_config.num_graph_layers,
                num_attention_heads=self.model_config.num_attention_heads,
                fusion_method=self.model_config.fusion_method,
                use_region_adaptive_fusion=self.model_config.use_region_adaptive_fusion,
                use_residual=True,
                concatenate_stage1=True,
                residual_alpha=getattr(self.model_config, "residual_alpha", 0.2)
            )
        else:
            self.stage2_model = None
    
    def setup_shared_components(self):
        """Setup shared components (VAE, text encoder, scheduler)."""
        # Shared components
        self.vae = AutoencoderKL.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="vae"
        )
        self.vae.requires_grad_(False)  # Freeze VAE
        self.vae.to(self.accelerator.device)  # Move to device

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="text_encoder"
        )
        self.text_encoder.requires_grad_(False)  # Freeze text encoder
        self.text_encoder.to(self.accelerator.device)  # Move to device

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="tokenizer"
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="scheduler"
        )

        # DDIM scheduler for Stage 1 inference — created once and reused
        from diffusers import DDIMScheduler
        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            self.model_config.pretrained_model_name, subfolder="scheduler"
        )

        print("Shared components loaded successfully")

    def compute_metrics(self, pred_images, gt_images):
        """Compute SSIM and PSNR for a batch of images."""
        pred_np = pred_images.detach().cpu().numpy().transpose(0, 2, 3, 1) # [B, H, W, 3]
        gt_np = gt_images.detach().cpu().numpy().transpose(0, 2, 3, 1)

        # Scale from [-1, 1] to [0, 1]
        pred_np = (pred_np + 1) / 2
        gt_np = (gt_np + 1) / 2

        ssim_scores = []
        psnr_scores = []

        for i in range(pred_np.shape[0]):
            s = ssim(gt_np[i], pred_np[i], data_range=1.0, channel_axis=2)
            p = psnr(gt_np[i], pred_np[i], data_range=1.0)
            ssim_scores.append(s)
            psnr_scores.append(p)

        return np.mean(ssim_scores), np.mean(psnr_scores)
    
    def setup_datasets(self):
        """Setup datasets and dataloaders."""
        print(f"Loading {self.data_config.dataset_name} dataset...")
        
        # Initialize extractors
        sketch_extractor = SketchExtractor(method=self.data_config.sketch_method)
        region_extractor = RegionExtractor(
            min_region_area=self.data_config.min_region_area,
            max_num_regions=self.data_config.max_num_regions
        )
        graph_builder = RegionGraphBuilder(
            graph_type=self.data_config.graph_type,
            image_size=(self.data_config.image_size, self.data_config.image_size)
        )
        
        # Create datasets
        if self.data_config.dataset_name == "sketchy":
            train_dataset = SketchyDataset(
                root_dir=self.data_config.sketchy_root,
                split="train",
                image_size=self.data_config.image_size,
                region_extractor=region_extractor,
                graph_builder=graph_builder,
                augment=self.data_config.use_augmentation
            )
            collate = sketchy_collate
            
        elif self.data_config.dataset_name == "coco":
            train_dataset = COCODataset(
                root_dir=self.data_config.coco_root,
                split="train",
                image_size=self.data_config.image_size,
                sketch_method=self.data_config.sketch_method,
                sketch_extractor=sketch_extractor,
                region_extractor=region_extractor,
                graph_builder=graph_builder,
                augment=self.data_config.use_augmentation,
                cache_sketches=self.data_config.cache_sketches
            )
            collate = coco_collate
        else:
            raise ValueError(f"Unknown dataset: {self.data_config.dataset_name}")
        
        # Subsample training dataset if limited (e.g., for faster Stage 2 iterations)
        dataset_limit = getattr(self.training_config, "dataset_limit", None)
        if dataset_limit and dataset_limit < len(train_dataset):
            print(f"📊 Subsampling training set: {len(train_dataset)} -> {dataset_limit} samples")
            indices = torch.randperm(len(train_dataset))[:dataset_limit]
            from torch.utils.data import Subset
            train_dataset = Subset(train_dataset, indices)

        # Create train dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=collate
        )

        # Create val dataloader (separate split to avoid train-set validation leakage)
        try:
            if self.data_config.dataset_name == "sketchy":
                val_dataset = SketchyDataset(
                    root_dir=self.data_config.sketchy_root,
                    split="val",
                    image_size=self.data_config.image_size,
                    region_extractor=region_extractor,
                    graph_builder=graph_builder,
                    augment=False
                )
            else:
                val_dataset = COCODataset(
                    root_dir=self.data_config.coco_root,
                    split="val",
                    image_size=self.data_config.image_size,
                    sketch_method=self.data_config.sketch_method,
                    sketch_extractor=sketch_extractor,
                    region_extractor=region_extractor,
                    graph_builder=graph_builder,
                    augment=False,
                    cache_sketches=self.data_config.cache_sketches
                )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.data_config.batch_size,
                shuffle=False,
                num_workers=self.data_config.num_workers,
                pin_memory=self.data_config.pin_memory,
                collate_fn=collate
            )
            print(f"Val dataset loaded: {len(val_dataset)} samples")

            # Create fixed validation batch for consistent monitoring (100 images)
            print("Creating fixed validation subset (100 samples)...")
            self.fixed_val_samples = []
            fixed_count = 0
            for batch in self.val_dataloader:
                self.fixed_val_samples.append(batch)
                fixed_count += batch["photo"].shape[0]
                if fixed_count >= 100:
                    break
            print(f"✅ Fixed validation subset ready: {fixed_count} samples")

        except Exception as e:
            print(f"Warning: Could not create val dataloader ({e}). Falling back to train set for validation.")
            self.val_dataloader = self.train_dataloader
            self.fixed_val_samples = []

        print(f"Dataset loaded: {len(train_dataset)} samples")
    
    def setup_optimizers(self):
        """Setup optimizers and learning rate schedulers."""
        # Stage 1 optimizer
        if self.stage1_model is not None:
            self.optimizer_stage1 = torch.optim.AdamW(
                self.stage1_model.get_trainable_parameters(),
                lr=self.training_config.learning_rate,
                betas=(self.training_config.adam_beta1, self.training_config.adam_beta2),
                eps=self.training_config.adam_epsilon,
                weight_decay=self.training_config.adam_weight_decay
            )
            
            self.lr_scheduler_stage1 = get_scheduler(
                self.training_config.lr_scheduler,
                optimizer=self.optimizer_stage1,
                num_warmup_steps=self.training_config.lr_warmup_steps,
                num_training_steps=len(self.train_dataloader) * self.training_config.stage1_epochs
            )
        
        # Stage 2 optimizer
        if self.stage2_model is not None:
            self.optimizer_stage2 = torch.optim.AdamW(
                self.stage2_model.get_trainable_parameters(),
                lr=self.training_config.learning_rate,
                betas=(self.training_config.adam_beta1, self.training_config.adam_beta2),
                eps=self.training_config.adam_epsilon,
                weight_decay=self.training_config.adam_weight_decay
            )
            
            self.lr_scheduler_stage2 = get_scheduler(
                self.training_config.lr_scheduler,
                optimizer=self.optimizer_stage2,
                num_warmup_steps=self.training_config.lr_warmup_steps,
                num_training_steps=len(self.train_dataloader) * self.training_config.stage2_epochs
            )
    
    def train_stage1_step(self, batch: Dict, model=None) -> Dict:
        """
        Single training step for Stage 1.
        
        Args:
            batch: Batch of data
            model: Model to use (if None, uses self.stage1_model)
        
        Returns:
            Dict with loss and metrics
        """
        # Use provided model or fall back to self.stage1_model
        stage1_model = model if model is not None else self.stage1_model
        
        sketches = batch["sketch"].to(self.accelerator.device)
        photos = batch["photo"].to(self.accelerator.device)
        text_prompts = batch["text_prompt"]

        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(photos).latent_dist.sample()
            latents = latents * 0.18215
        
        # Detach latents (VAE gradients not needed)
        latents = latents.detach()
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Sample random timestep
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        )
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode sketch
        sketch_features = stage1_model.encode_sketch(sketches)

        # Encode text
        text_embeddings = stage1_model.encode_text(text_prompts)

        # Predict noise
        noise_pred = stage1_model(
            noisy_latents,
            timesteps,
            sketch_features,
            text_embeddings
        )
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}
    
    def _generate_stage1_latents(self, sketch_features, text_embeddings, num_steps=5):
        """Generate coarse latents from Stage 1 for Stage 2 conditioning."""
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Initialize with noise — match stage1 training resolution (256x256 → 32x32 latents)
        latents = torch.randn(batch_size, 4, 32, 32, device=device)

        # Reuse shared DDIM scheduler — set_timesteps is cheap vs. from_pretrained
        self.ddim_scheduler.set_timesteps(num_steps)

        for t in self.ddim_scheduler.timesteps:
            # Scale model input
            latent_model_input = self.ddim_scheduler.scale_model_input(latents, t)

            # Predict noise
            noise_pred = self.stage1_model(
                latent_model_input,
                t.to(device),
                sketch_features,
                text_embeddings
            )

            # Step scheduler
            latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents

    def compute_edge_similarity(self, img1, img2):
        """Compute edge similarity using Sobel filters to assess structural preservation."""
        # imgs are [B, 3, H, W] in [-1, 1]
        def sobel_edges(x):
            x = (x + 1) / 2 # [-1,1] -> [0,1]
            # convert to grayscale
            x = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
            
            # kernels
            kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3).float()
            ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).view(1, 1, 3, 3).float()
            
            gx = F.conv2d(x, kx, padding=1)
            gy = F.conv2d(x, ky, padding=1)
            mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
            return mag
        
        edges1 = sobel_edges(img1)
        edges2 = sobel_edges(img2)
        
        # Pearson correlation or simple MSE between edge maps
        similarity = F.cosine_similarity(edges1.flatten(1), edges2.flatten(1), dim=1).mean().item()
        return similarity

    def train_stage2_step(self, batch: Dict, model=None) -> Dict:
        """
        Refined Stage 2 training. RTX 5090 32GB: Stage 1 stays on GPU, no CPU offloading.
        """
        stage2_model = model if model is not None else self.stage2_model
        photos = batch["photo"].to(self.accelerator.device)
        sketches = batch["sketch"].to(self.accelerator.device)
        text_prompts = batch["text_prompt"]

        # 1. Stage-1 Output (on GPU — 32GB VRAM has headroom for both stages)
        with torch.no_grad():
            sketch_features = self.stage1_model.encode_sketch(sketches)
            text_embeddings = self.stage1_model.encode_text(text_prompts)
            # Use 4 steps for faster Stage 1 conditioning during training (down from 10)
            stage1_latents = self._generate_stage1_latents(sketch_features, text_embeddings, num_steps=8)

            # Encode Ground Truth to latents
            gt_latents = self.vae.encode(photos).latent_dist.sample() * 0.18215

        # 2. Sample noise — full timestep range so Stage 2 learns all noise levels
        T_MAX = self.noise_scheduler.config.num_train_timesteps  # 1000
        timesteps = torch.randint(0, T_MAX, (photos.shape[0],), device=photos.device)
        
        # Add noise to GT latents
        #noise = torch.randn_like(gt_latents)
        #noisy_latents = self.noise_scheduler.add_noise(gt_latents, noise, timesteps)



        if torch.rand(1)<0.7:
            base_latents = stage1_latents.detach()
        else:
            base_latents = (0.5 * gt_latents + 0.5 * stage1_latents).detach()
            
            
        #base_latents = stage1_latents.detach() #comment this if necessary
        noise = torch.randn_like(base_latents)
        noisy_latents = self.noise_scheduler.add_noise(base_latents, noise, timesteps)

        # 3. Forward through Stage 2 with Stage-1 conditioning
        # Get region graph (handle batch)
        region_graphs = batch["region_graph"]
        
        output_dict = stage2_model(
            noisy_latents,
            timesteps,
            region_graphs,
            text_embeddings,
            stage1_latents=stage1_latents,
            return_dict=True
        )
        noise_pred = output_dict["noise_pred"]
        
        # 4. Losses and Delta Magnitude Monitoring (CRITICAL)
        # Estimate x0 (denoised latents) for identity and perceptual loss
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(photos.device)
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sigma_t = (1 - alpha_t).sqrt()
        pred_x0_latents = (noisy_latents - sigma_t * noise_pred) / alpha_t.sqrt()
        
        # Delta Magnitude Tracking
        latent_delta = pred_x0_latents - stage1_latents
        delta_mean = torch.mean(torch.abs(latent_delta)).item()
        delta_std = torch.std(latent_delta).item()

        # A. Diffusion loss (MSE)
        loss_diffusion = F.mse_loss(noise_pred, noise)
        
        # B. Identity Preservation Loss (L1)
        # Encourage Stage-2 to stay close to Stage-1 structure
        loss_identity = (0.5 * F.l1_loss(pred_x0_latents, stage1_latents)+ 0.5 * F.mse_loss(pred_x0_latents, stage1_latents))
        
        # C. Perceptual Loss (LPIPS) and L_delta
        # Decode pred and stage1 latents in a single batched VAE call
        with torch.set_grad_enabled(True):
            self._s2_step_count = getattr(self, "_s2_step_count", 0) + 1
            combined_latents = torch.cat([pred_x0_latents, stage1_latents], dim=0) / 0.18215
            if self._s2_step_count % 50 == 0:
                with torch.no_grad():
                    combined_images = self.vae.decode(combined_latents).sample
                combined_images = torch.clamp(combined_images, -1, 1)
                
                b = pred_x0_latents.shape[0]
                pred_images = combined_images[:b]
                s1_images = combined_images[b:].detach()

                loss_perceptual = self.lpips_loss(pred_images, photos).mean()
                image_delta = pred_images - s1_images
                loss_delta = torch.mean(torch.abs(image_delta))
            else:
                pred_images = None
                s1_images = None
                loss_perceptual = torch.tensor(0.0, device=photos.device)
                loss_delta = torch.tensor(0.0, device=photos.device)
                

            
        
        
        
        # Final combined loss
        # Use configurable weights from training_config
        lambda_id = getattr(self.training_config, "lambda_identity", 0.2)
        lambda_lpips = getattr(self.training_config, "lambda_lpips", 0.1)
        lambda_delta = getattr(self.training_config, "lambda_delta", 0.05)

        # --- Decode predicted image for SSIM ---
        with torch.no_grad():
            pred_for_ssim = self.vae.decode(pred_x0_latents / 0.18215).sample
            pred_for_ssim = torch.clamp(pred_for_ssim, -1, 1)

        # Convert to [0,1]
        pred_ssim = (pred_for_ssim + 1) / 2
        gt_ssim = (photos + 1) / 2

        # Compute SSIM (batch average)
        ssim_val = 0
        for i in range(pred_ssim.shape[0]):
            ssim_val += ssim(
                gt_ssim[i].permute(1,2,0).cpu().numpy(),
                pred_ssim[i].permute(1,2,0).cpu().numpy(),
                data_range=1.0,
                channel_axis=2
            )
        ssim_val /= pred_ssim.shape[0]

        ssim_loss = 1 - ssim_val
        ssim_loss = torch.tensor(ssim_loss, device=photos.device)
        
        lambda_ssim = 0.15

        total_loss = (
            loss_diffusion 
            + lambda_id * loss_identity 
            + lambda_lpips * loss_perceptual
            + lambda_delta * loss_delta
            + lambda_ssim * ssim_loss   
        )
        
        # Compute Stage-2 metrics — CPU-heavy, gated to every 10 steps
        self._s2_step_count = getattr(self, '_s2_step_count', 0) + 1
        if self._s2_step_count % 10 == 0 and pred_images is not None:
            s2_ssim, s2_psnr = self.compute_metrics(pred_images.detach(), photos)
            s1_ssim, s1_psnr = self.compute_metrics(s1_images.detach(), photos)
            edge_sim = self.compute_edge_similarity(pred_images.detach(), s1_images.detach())
        else:
            s2_ssim = s2_psnr = s1_ssim = s1_psnr = edge_sim = 0.0

        return {
            "loss": total_loss,
            "loss_diffusion": loss_diffusion.detach(),
            "loss_identity": loss_identity.detach(),
            "loss_perceptual": loss_perceptual.detach(),
            "loss_delta": loss_delta.detach(),
            "metrics": {
                "s2_ssim": s2_ssim,
                "s2_psnr": s2_psnr,
                "s1_ssim": s1_ssim,
                "delta_ssim": s2_ssim - s1_ssim,
                "edge_sim": edge_sim,
                "delta_mean": delta_mean,
                "delta_std": delta_std
            },
            # Return images for grid generation
            "images": {
                "sketch": sketches,
                "s1": s1_images,
                "s2": pred_images,
                "gt": photos
            }
        }
    
    def _generate_stage2_latents(self, stage2_model, stage1_latents, region_graphs, text_embeddings, num_steps=10):
        """Perform full multi-step refinement for Stage 2 (inference mode)."""
        batch_size = stage1_latents.shape[0]
        device = stage1_latents.device
        
        # Start from pure noise
        latents = torch.randn_like(stage1_latents)
        self.ddim_scheduler.set_timesteps(num_steps)
        
        for t in self.ddim_scheduler.timesteps:
            latent_model_input = self.ddim_scheduler.scale_model_input(latents, t)
            
            # Stage 2 takes stage1_latents as conditioning
            noise_pred = stage2_model(
                latent_model_input,
                t.to(device),
                region_graphs,
                text_embeddings,
                stage1_latents=stage1_latents,
                return_dict=True
            )["noise_pred"]
            
            latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents

    def validate_stage2(self, model) -> Dict:
        """
        Validate Stage 2 by performing FULL inference (starting from noise)
        to prevent ground-truth leakage in visual reports.
        """
        # Unwrap model for inference
        model_to_eval = self.accelerator.unwrap_model(model)
        model_to_eval.eval()
        device = self.accelerator.device
        
        all_metrics = {
            "val_loss": [],
            "val_s2_ssim": [],
            "val_s2_psnr": [],
            "val_s1_ssim": [],
            "val_s1_psnr": [],
            "val_edge_sim": []
        }
        
        # Store a few images for visualization (first 8 samples)
        viz_images = {"sketch": [], "s1": [], "s2": [], "gt": []}
        
        with torch.no_grad():
            for batch in self.fixed_val_samples:
                photos = batch["photo"].to(device)
                sketches = batch["sketch"].to(device)
                text_prompts = batch["text_prompt"]
                region_graphs = batch["region_graph"]
                bs = photos.shape[0]

                # 1. Full Stage-1 Generation (Starting from noise)
                sketch_features = self.stage1_model.encode_sketch(sketches)
                text_embeddings = self.stage1_model.encode_text(text_prompts)
                s1_latents = self._generate_stage1_latents(sketch_features, text_embeddings, num_steps=10)
                s1_images = self.vae.decode(s1_latents / 0.18215).sample
                s1_images = torch.clamp(s1_images, -1, 1)

                # 2. Full Stage-2 Refinement (Starting from noise, NOT noisy GT)
                # This ensures the visual report shows the model's ACTUAL generation ability
                # Slice region_graphs for current batch
                current_graphs = region_graphs[:bs] if isinstance(region_graphs, list) else region_graphs
                
                s2_latents = self._generate_stage2_latents(
                    model_to_eval, s1_latents, current_graphs, text_embeddings, num_steps=10
                )
                pred_images = self.vae.decode(s2_latents / 0.18215).sample
                pred_images = torch.clamp(pred_images, -1, 1)

                # Metrics (compare prediction against ground truth)
                s2_ssim, s2_psnr = self.compute_metrics(pred_images, photos)
                s1_ssim, s1_psnr = self.compute_metrics(s1_images, photos)
                edge_sim = self.compute_edge_similarity(pred_images, s1_images)
                
                # Still calculate loss for tracking (needs GT-based noise prediction)
                gt_latents = self.vae.encode(photos).latent_dist.sample() * 0.18215
                torch.manual_seed(42)
                noise = torch.randn_like(gt_latents)
                t_val = torch.ones((bs,), device=device, dtype=torch.long) * 500
                noisy_gt = self.noise_scheduler.add_noise(gt_latents, noise, t_val)
                noise_pred = model_to_eval(noisy_gt, t_val, current_graphs, text_embeddings, stage1_latents=s1_latents)["noise_pred"]
                
                all_metrics["val_loss"].append(F.mse_loss(noise_pred, noise).item())
                all_metrics["val_s2_ssim"].append(s2_ssim)
                all_metrics["val_s2_psnr"].append(s2_psnr)
                all_metrics["val_s1_ssim"].append(s1_ssim)
                all_metrics["val_s1_psnr"].append(s1_psnr)
                all_metrics["val_edge_sim"].append(edge_sim)
                
                # Viz subset (max 8)
                if len(viz_images["gt"]) < 8:
                    rem = 8 - len(viz_images["gt"])
                    viz_images["sketch"].append(sketches[:rem].cpu())
                    viz_images["s1"].append(s1_images[:rem].cpu())
                    viz_images["s2"].append(pred_images[:rem].cpu())
                    viz_images["gt"].append(photos[:rem].cpu())

        # Average metrics
        avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        
        # Concat viz images
        viz_grid = {k: torch.cat(v, dim=0) for k, v in viz_images.items()}
        
        model.train()
        return {"avg_metrics": avg_metrics, "viz_images": viz_grid}

    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting RAGAF-Diffusion Training")
        print("="*60)
        
        # Train Stage 1
        if self.training_config.train_stage in ["stage1", "both"]:
            print("\n[Stage 1] Sketch-Guided Diffusion Training")
            self.train_stage(
                stage="stage1",
                model=self.stage1_model,
                optimizer=self.optimizer_stage1,
                lr_scheduler=self.lr_scheduler_stage1,
                num_epochs=self.training_config.stage1_epochs,
                train_step_fn=self.train_stage1_step,
                start_epoch=self.training_config.resume_from_epoch
            )
        
        # Train Stage 2
        if self.training_config.train_stage in ["stage2", "both"]:
            print("\n[Stage 2] Semantic Refinement Training")
            self.train_stage(
                stage="stage2",
                model=self.stage2_model,
                optimizer=self.optimizer_stage2,
                lr_scheduler=self.lr_scheduler_stage2,
                num_epochs=self.training_config.stage2_epochs,
                train_step_fn=self.train_stage2_step,
                start_epoch=self.training_config.resume_from_epoch
            )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def train_stage(
        self,
        stage: str,
        model,
        optimizer,
        lr_scheduler,
        num_epochs: int,
        train_step_fn,
        start_epoch: int = 0
    ):
        """
        Train a single stage.

        Args:
            stage: Stage name
            model: Model to train
            optimizer: Optimizer
            lr_scheduler: LR scheduler
            num_epochs: Number of epochs
            train_step_fn: Training step function
            start_epoch: Epoch to resume from (0-indexed)
        """
        # Prepare for distributed training
        for name, p in model.named_parameters():
            if 'unet' in name or 'sketch_encoder' in name:
                p.requires_grad = True

        # Ensure auxiliary models are on the correct device
        if hasattr(self, 'lpips_loss') and self.lpips_loss is not None:
            self.lpips_loss.to(self.accelerator.device)

        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            model, optimizer, self.train_dataloader, lr_scheduler
        )

        # Resume: load checkpoint weights if start_epoch > 0
        # ✅ NEW RESUME LOGIC (independent of start_epoch)
        resume_path = getattr(self.training_config, "resume_from_checkpoint", None)

        if resume_path is not None and os.path.exists(resume_path):
            print(f"▶️ Resuming from checkpoint: {resume_path}")
    
            ckpt = torch.load(resume_path, map_location=self.accelerator.device, weights_only=False)
    
            unwrapped = self.accelerator.unwrap_model(model)
            unwrapped.load_state_dict(ckpt["model_state_dict"])
    
            # ✅ Set correct start epoch
            start_epoch = ckpt.get("epoch", 0) + 1
    
            print(f"✅ Loaded weights from epoch {ckpt['epoch'] + 1}")
            print(f"✅ Resuming training from epoch {start_epoch}")

        else:
            print("⚠️ No checkpoint found. Training from scratch.")

        global_step = start_epoch * len(train_dataloader)
        
        # Best model tracking and early stopping
        best_ssim = -1.0
        patience_counter = 0
        patience = getattr(self.training_config, "early_stopping_patience", 5)
        
        # Diagnostics history
        comparison_history = []
        
        # Identity sanity check helper (Stage 2)
        def run_identity_check(model):
            model.eval()
            with torch.no_grad():
                # Get a single sample
                batch = next(iter(self.train_dataloader))
                photos = batch["photo"][:1].to(self.accelerator.device)
                sketches = batch["sketch"][:1].to(self.accelerator.device)
                text_prompts = batch["text_prompt"][:1]
                
                # Encode Stage 1
                sketch_features = self.stage1_model.encode_sketch(sketches)
                text_embeddings = self.stage1_model.encode_text(text_prompts)
                stage1_latents = self._generate_stage1_latents(sketch_features, text_embeddings, num_steps=3)
                
                # Zero noise, t=0
                dummy_t = torch.zeros(1, device=self.accelerator.device, dtype=torch.long)
                
                output = model(
                    stage1_latents, # Use stage1 directly as noisy input
                    dummy_t,
                    batch["region_graph"][0],
                    text_embeddings,
                    stage1_latents=stage1_latents,
                    return_dict=True
                )
                
                # If identity is preserved, noise_pred (delta) should be small
                delta = output["noise_pred"]
                error = torch.mean(torch.abs(delta)).item()
            model.train()
            return error

        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_ssim = []
            epoch_delta_means = []
            
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                disable=not self.accelerator.is_main_process,
                desc=f"[{stage.upper()}] Epoch {epoch+1}/{num_epochs}"
            )
            
            for step, batch in progress_bar:
                with self.accelerator.accumulate(model):
                    # Training step
                    outputs = train_step_fn(batch, model=model)
                    loss = outputs["loss"]
                    
                    # Backward
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping (Value: 1.0)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            1.0
                        )
                    
                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Metrics tracking
                epoch_loss += loss.item()
                if "metrics" in outputs:
                    epoch_ssim.append(outputs["metrics"]["s2_ssim"])
                    epoch_delta_means.append(outputs["metrics"]["delta_mean"])
                
                global_step += 1
                
                # Adaptive scaling automatic logic (Stage 2)
                if stage == "stage2" and "metrics" in outputs and self.accelerator.is_main_process:
                    d_mean = outputs["metrics"]["delta_mean"]
                    
                    # Update EMA
                    if self.running_delta_mean == 0:
                        self.running_delta_mean = d_mean
                    else:
                        self.running_delta_mean = (self.delta_ema_momentum * self.running_delta_mean) + (1 - self.delta_ema_momentum) * d_mean
                    
                    # Adaptive update with hysteresis and cooldown (every 200 steps)
                    # Thresholds calibrated to observed EMA delta range (~0.3–0.8)
                    # Cooldown prevents continuous decay when delta is stuck above threshold
                    if global_step % 200 == 0:
                        t_high = getattr(self.training_config, "delta_threshold_high", 0.75)
                        t_low = getattr(self.training_config, "delta_threshold_low", 0.30)
                        cooldown = getattr(self, "_alpha_cooldown_steps", 0)

                        unwrapped_model = self.accelerator.unwrap_model(model)
                        old_alpha = unwrapped_model.residual_alpha

                        if self.running_delta_mean > t_high and cooldown == 0:
                            # Delta too large — reduce alpha slightly, then enforce cooldown
                            unwrapped_model.residual_alpha *= 0.95
                            self._alpha_cooldown_steps = 600  # ~3 update intervals
                            print(f"\n[Adaptive] EMA Delta ({self.running_delta_mean:.3f}) > {t_high}. Reducing residual_alpha: {old_alpha:.4f} -> {unwrapped_model.residual_alpha:.4f}")
                        elif self.running_delta_mean < t_low and cooldown == 0:
                            # Delta too small — refinement is being suppressed, increase alpha
                            unwrapped_model.residual_alpha *= 1.02
                            self._alpha_cooldown_steps = 600
                            print(f"\n[Adaptive] EMA Delta ({self.running_delta_mean:.3f}) < {t_low}. Increasing residual_alpha: {old_alpha:.4f} -> {unwrapped_model.residual_alpha:.4f}")
                        else:
                            # Inside dead band or in cooldown — leave alpha alone
                            if cooldown > 0:
                                self._alpha_cooldown_steps = max(0, cooldown - 200)

                        # Clamp residual_alpha ∈ [0.05, 0.5]
                        unwrapped_model.residual_alpha = max(0.05, min(0.25, unwrapped_model.residual_alpha))
                
                # Minimal Logging
                if global_step % self.training_config.log_every_n_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = lr_scheduler.get_last_lr()[0]
                    logs = {
                        f"{stage}/loss": avg_loss,
                        f"{stage}/lr": lr,
                    }
                    if self.training_config.use_wandb and self.accelerator.is_main_process:
                        import wandb
                        wandb.log(logs)

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                    })

            # 1. Save periodic checkpoint FIRST (as requested)
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                self.save_checkpoint(stage, model, epoch)

            # 2. End of Epoch Evaluation and Diagnostics
            if stage == "stage2":
                print(f"\n[Epoch {epoch+1}] Running validation on fixed subset...")
                val_results = self.validate_stage2(model)
                avg_metrics = val_results["avg_metrics"]
                avg_val_ssim = avg_metrics["val_s2_ssim"]

                if self.accelerator.is_main_process:
                    print(f"✅ Val SSIM: {avg_val_ssim:.4f}, Val LPIPS-Loss: {avg_metrics['val_loss']:.4f}")

                    # Log to WandB
                    if self.training_config.use_wandb:
                        import wandb
                        # Log metrics
                        wandb_logs = {f"val/{k}": v for k, v in avg_metrics.items()}
                        wandb_logs["epoch"] = epoch + 1

                        # Log images
                        viz = val_results["viz_images"]
                        sketches = viz["sketch"].repeat(1, 3, 1, 1) if viz["sketch"].shape[1] == 1 else viz["sketch"]
                        if sketches.max() <= 1.0: sketches = sketches * 2 - 1

                        combined = torch.cat([sketches, viz["s1"], viz["s2"], viz["gt"]], dim=0)
                        grid = make_grid(combined, nrow=viz["gt"].shape[0], normalize=True, value_range=(-1, 1))

                        wandb_logs["visuals/comparison"] = wandb.Image(grid, caption=f"Epoch {epoch+1}: Sketch | Stage-1 | Stage-2 (Refined) | Ground Truth")
                        wandb.log(wandb_logs)

                    # Identity Sanity Check (Stage 2)
                    id_err = run_identity_check(model)
                    print(f"[Identity Check] Error: {id_err:.6f}")

                    # Save comparison log for this epoch
                    curr_alpha = float(getattr(self.accelerator.unwrap_model(model), "residual_alpha", 0.0))
                    epoch_log = {
                        "epoch": epoch + 1,
                        "ssim_val": float(avg_val_ssim),
                        "identity_error": id_err,
                        "residual_alpha": curr_alpha,
                        **avg_metrics
                    }
                    comparison_history.append(epoch_log)

                    log_path = Path(self.training_config.checkpoint_dir) / stage / "comparison_logs.json"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "w") as f:
                        json.dump(comparison_history, f, indent=2)

                eval_metric = avg_val_ssim
            else:
                # Stage 1: simpler val
                eval_metric = 0 # Implement if needed

            # Save Best Model
            if eval_metric > best_ssim:
                best_ssim = eval_metric
                patience_counter = 0
                print(f"🌟 New best evaluation score! Saving best_{stage}.pt")
                self.save_checkpoint(stage, model, epoch, filename=f"best_{stage}.pt")
            else:
                patience_counter += 1

                
            # Early Stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without SSIM improvement.")
                break
        
        # Save final checkpoint
        self.save_checkpoint(stage, model, num_epochs, final=True)

    def save_visual_grid(self, images_dict, stage, step):
        """Save a grid of images for visual validation."""
        # Convert grayscale sketch to 3-channel
        sketches = images_dict["sketch"].repeat(1, 3, 1, 1) if images_dict["sketch"].shape[1] == 1 else images_dict["sketch"]
        
        # Rescale sketches to [-1, 1] for grid consistency if they are [0, 1]
        if sketches.max() <= 1.0:
            sketches = sketches * 2 - 1
            
        combined = torch.cat([sketches, images_dict["s1"], images_dict["s2"], images_dict["gt"]], dim=0)
        grid = make_grid(combined, nrow=images_dict["gt"].shape[0], normalize=True, value_range=(-1, 1))
        
        out_dir = Path(self.training_config.checkpoint_dir) / "visuals" / stage
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, out_dir / f"step_{step}.png")
    
    def _find_resume_checkpoint(self, stage: str, start_epoch: int) -> Optional[str]:
        """Find the best local checkpoint to resume from."""
        checkpoint_dir = Path(self.training_config.checkpoint_dir) / stage
        # Look for the checkpoint just before start_epoch
        for ep in range(start_epoch, 0, -1):
            path = checkpoint_dir / f"epoch_{ep}.pt"
            if path.exists():
                return str(path)
        return None

    def save_checkpoint(self, stage: str, model, epoch: int, final: bool = False, filename: Optional[str] = None):
        """
        Save model checkpoint locally, upload to HuggingFace Hub, then aggressively cleanup old checkpoints.
        """
        if not self.accelerator.is_main_process:
            return

        # Check free disk space before saving (need at least 2 GB)
        import shutil
        checkpoint_dir = Path(self.training_config.checkpoint_dir) / stage
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is not None:
            # use provided filename
            pass
        elif final:
            filename = "final.pt"
        else:
            filename = f"epoch_{epoch+1}.pt"

        path = checkpoint_dir / filename

        # Unwrap model — save ONLY model weights (not optimizer) to save ~60% space
        unwrapped_model = self.accelerator.unwrap_model(model)
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "config": {
                "model": vars(self.model_config),
                "data": vars(self.data_config),
                "training": vars(self.training_config)
            }
        }

        # Save to a temp file first, then rename (avoids corrupt partial writes)
        tmp_path = path.with_suffix(".tmp")
        try:
            torch.save(checkpoint_data, tmp_path)
            tmp_path.rename(path)
            saved_size_gb = path.stat().st_size / 1e9
            print(f"✅ Checkpoint saved locally: {path} ({saved_size_gb:.2f} GB)")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"❌ Failed to save checkpoint: {e}")
            return

        # --- HuggingFace Hub upload (background thread — does not block training) ---
        if getattr(self.training_config, "push_to_hub", False):
            import threading

            def _upload():
                try:
                    from huggingface_hub import HfApi, create_repo
                    token = getattr(self.training_config, "hub_token", None) or os.getenv("HF_TOKEN")
                    repo_id = self.training_config.hub_repo_id
                    api = HfApi(token=token)
                    try:
                        create_repo(repo_id, repo_type="model", private=True, token=token, exist_ok=True)
                    except Exception:
                        pass
                    hub_path = f"{stage}/{filename}"
                    print(f"☁️  [BG] Uploading to HF Hub: {repo_id}/{hub_path} ...")
                    api.upload_file(
                        path_or_fileobj=str(path),
                        path_in_repo=hub_path,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"[{stage}] {'Final' if final else f'Epoch {epoch+1}'} checkpoint"
                    )
                    print(f"✅ [BG] Uploaded to HF Hub: https://huggingface.co/{repo_id}")
                    self._cleanup_old_checkpoints(stage, checkpoint_dir, keep_count=2)
                except Exception as e:
                    print(f"⚠️  HF Hub upload failed — keeping local copy: {e}")

            threading.Thread(target=_upload, daemon=True).start()
        else:
            print("ℹ️  push_to_hub=False, checkpoint kept locally only.")

    def _cleanup_old_checkpoints(self, stage: str, checkpoint_dir: Path, keep_count: int = 2):
        """
        Delete old checkpoints, keeping only the most recent N + final.pt.
        
        This aggressively manages disk space for 100GB container storage.
        
        Args:
            stage: Stage name (stage1, stage2, etc)
            checkpoint_dir: Directory containing checkpoints
            keep_count: Number of recent checkpoints to keep (default: 2)
        """
        import shutil
        
        # Get all epoch checkpoints (not final)
        epoch_checkpoints = sorted(
            checkpoint_dir.glob("epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        
        # Calculate how many to delete
        num_to_delete = max(0, len(epoch_checkpoints) - keep_count)
        
        if num_to_delete > 0:
            print(f"\n🗑️  Cleaning up old checkpoints ({num_to_delete} to delete, keeping {keep_count} recent)...")
            
            # Delete oldest checkpoints
            for old_ckpt in epoch_checkpoints[:num_to_delete]:
                try:
                    old_size_gb = old_ckpt.stat().st_size / 1e9
                    old_ckpt.unlink()
                    
                    # Log deletion
                    freed_space = old_size_gb
                    print(f"  🗑️  Deleted {old_ckpt.name} ({freed_space:.2f} GB freed)")
                    
                except Exception as e:
                    print(f"  ⚠️  Failed to delete {old_ckpt.name}: {e}")
            
            # Check final disk space
            free_bytes = shutil.disk_usage(str(checkpoint_dir)).free
            free_gb = free_bytes / 1e9
            
            # Count remaining checkpoints
            remaining = sorted(checkpoint_dir.glob("epoch_*.pt"))
            remaining_str = ", ".join([p.stem for p in remaining[-keep_count:]])
            
            print(f"\n📊 Checkpoint status after cleanup:")
            print(f"  Disk free: {free_gb:.1f} GB")
            print(f"  Kept epochs: {remaining_str}")
            if checkpoint_dir.joinpath("final.pt").exists():
                print(f"  Final checkpoint: ✅ final.pt")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train RAGAF-Diffusion")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--stage", type=str, default="both", choices=["stage1", "stage2", "both"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--dataset_limit", type=int, default=None, help="Limit training dataset to N samples")
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Resume stage1 training from this epoch (e.g. 4 to skip epochs 1-4)")
    parser.add_argument("--stage2_checkpoint",type=int,default=None,help="Path to Stage 2 checkpoint to resume from")
    
    parser.add_argument("--resume_checkpoint",type=str,default=None,help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    
    if args.config is not None:
        from configs.config import load_config
        loaded_config = load_config(args.config)
        # TODO: Merge loaded config with default
    
    # Override with CLI args
    if args.stage is not None:
        config["training"].train_stage = args.stage
    if args.batch_size is not None:
        config["data"].batch_size = args.batch_size
    if args.learning_rate is not None:
        config["training"].learning_rate = args.learning_rate
    if args.epochs is not None:
        config["training"].stage1_epochs = args.epochs
        config["training"].stage2_epochs = args.epochs
    if args.checkpoint_dir is not None:
        config["training"].checkpoint_dir = args.checkpoint_dir
    if args.grad_accum is not None:
        config["training"].gradient_accumulation_steps = args.grad_accum
    if args.save_every is not None:
        config["training"].save_every_n_epochs = args.save_every
    if args.dataset_limit is not None:
        config["training"].dataset_limit = args.dataset_limit
    if args.resume_epoch > 0:
        config["training"].resume_from_epoch = args.resume_epoch
        print(f"▶️  Will resume Stage 1 from epoch {args.resume_epoch}")
    if args.resume_checkpoint is not None:
        config["training"].resume_from_checkpoint = args.resume_checkpoint
    
    # Check for WandB availability
    try:
        import wandb
        wandb_available = True
    except ImportError:
        wandb_available = False

    # Force WandB for this run if available
    if wandb_available:
        config["training"].use_wandb = True
    else:
        config["training"].use_wandb = False
        print("⚠️  WandB not installed. Logging to local logs only.")
    
    # Print configuration summary
    print("\n" + "=" * 40)
    print("🚀 FINAL TRAINING CONFIGURATION")
    print("-" * 40)
    print(f"  Stage:           {config['training'].train_stage}")
    print(f"  Batch Size:      {config['data'].batch_size}")
    print(f"  Grad Accum:      {config['training'].gradient_accumulation_steps}")
    print(f"  Effective Batch: {config['data'].batch_size * config['training'].gradient_accumulation_steps}")
    print(f"  Dataset Limit:   {getattr(config['training'], 'dataset_limit', 'Full')}")
    print(f"  Learning Rate:   {config['training'].learning_rate}")
    print(f"  Epochs:          {config['training'].stage1_epochs if config['training'].train_stage == 'stage1' else config['training'].stage2_epochs}")
    print(f"  Image Size:      {config['data'].image_size}")
    print(f"  Precision:       {config['training'].mixed_precision}")
    print(f"  LPIPS Loss:      {config['training'].lambda_lpips > 0}")
    print(f"  WandB:           {'Enabled ✅' if config['training'].use_wandb else 'Disabled ❌'}")
    print("=" * 40 + "\n")

    # Create trainer
    trainer = RAGAFDiffusionTrainer(
        model_config=config["model"],
        data_config=config["data"],
        training_config=config["training"]
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
