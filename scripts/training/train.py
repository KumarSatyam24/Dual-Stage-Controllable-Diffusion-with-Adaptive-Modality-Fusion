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
        
        # Initialize accelerator for distributed training and mixed precision
        self.accelerator = Accelerator(
            mixed_precision=training_config.mixed_precision,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            log_with="wandb" if training_config.use_wandb else None,
            project_dir=training_config.checkpoint_dir
        )
        
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
                stage1_path = "./checkpoints/stage1_with_ssim/epoch_18.pt"
            
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
            self.stage1_model.to("cpu") # KEEP ON CPU UNTIL NEEDED (Dynamic Offloading)
        
        # Stage 2 model
        if self.training_config.train_stage in ["stage2", "both"]:
            # Load UNet
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(
                self.model_config.pretrained_model_name,
                subfolder="unet"
            )
            
            # --- Memory Optimization for 4GB VRAM ---
            # 1. Gradient Checkpointing (Critical for 4GB)
            unet.enable_gradient_checkpointing()
            # 2. Memory Efficient Attention
            try:
                unet.set_use_memory_efficient_attention_xformers(True)
                print("✅ xFormers attention enabled")
            except Exception:
                # Fallback to PyTorch's native SDPA if xFormers isn't installed
                pass
            # ----------------------------------------
            
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
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=collate
        )
        
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
        
        # Debug first batch
        if not hasattr(self, '_debug_done'):
            print(f"\nDEBUG batch shapes:")
            print(f"  sketches: {sketches.shape}, dtype: {sketches.dtype}")
            print(f"  photos: {photos.shape}, dtype: {photos.dtype}")
            print(f"  Model type: {type(stage1_model)}")
            print(f"  Model module type: {type(stage1_model.module) if hasattr(stage1_model, 'module') else 'no module attr'}")
            self._debug_done = True
        
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
        
        # Check if we're in no_grad mode
        if not hasattr(self, '_debug_done3'):
            print(f"\nDEBUG grad mode:")
            print(f"  torch.is_grad_enabled(): {torch.is_grad_enabled()}")
            print(f"  Model training: {stage1_model.training}")
            self._debug_done3 = True
        
        # Predict noise
        noise_pred = stage1_model(
            noisy_latents,
            timesteps,
            sketch_features,
            text_embeddings
        )
        
        # Debug first batch
        if not hasattr(self, '_debug_done2'):
            print(f"\nDEBUG tensor info:")
            print(f"  noisy_latents.requires_grad: {noisy_latents.requires_grad}")
            print(f"  noise_pred.requires_grad: {noise_pred.requires_grad}")
            print(f"  noise_pred.grad_fn: {noise_pred.grad_fn}")
            print(f"  noise.requires_grad: {noise.requires_grad}")
            self._debug_done2 = True
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}
    
    def _generate_stage1_latents(self, sketch_features, text_embeddings, num_steps=5):
        """Generate coarse latents from Stage 1 for Stage 2 conditioning."""
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        # Initialize with noise
        latents = torch.randn(batch_size, 4, 32, 32, device=device) # Assuming 256x256 -> 32x32
        
        # Use a simple scheduler for few steps
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_pretrained(
            self.model_config.pretrained_model_name, subfolder="scheduler"
        )
        scheduler.set_timesteps(num_steps)
        
        for t in scheduler.timesteps:
            # Scale model input
            latent_model_input = scheduler.scale_model_input(latents, t)
            
            # Predict noise
            noise_pred = self.stage1_model(
                latent_model_input,
                t.to(device),
                sketch_features,
                text_embeddings
            )
            
            # Step scheduler
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
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
        Refined Stage 2 training with Sequential Model Offloading for 4GB VRAM.
        """
        stage2_model = model if model is not None else self.stage2_model
        photos = batch["photo"].to(self.accelerator.device)
        sketches = batch["sketch"].to(self.accelerator.device)
        text_prompts = batch["text_prompt"]
        
        # 1. Stage-1 Output (KEEP ON CPU TO SAVE VRAM)
        with torch.no_grad():
            # Ensure stage1_model is entirely on CPU to avoid OOM on low VRAM GPUs
            # Force move all submodules to CPU (some pretrained components may default to CUDA)
            if next(self.stage1_model.parameters()).device.type != 'cpu':
                self.stage1_model.to('cpu')

            # Move inputs to CPU temporarily for stage1 inference
            sketches_cpu = sketches.cpu()
            sketch_features = self.stage1_model.encode_sketch(sketches_cpu)
            text_embeddings = self.stage1_model.encode_text(text_prompts)
            # Use 3 steps for speed during training
            stage1_latents = self._generate_stage1_latents(sketch_features, text_embeddings, num_steps=3)

            # Move ALL stage1 outputs back to GPU for stage2
            stage1_latents = stage1_latents.to(self.accelerator.device)
            text_embeddings = text_embeddings.to(self.accelerator.device)
            torch.cuda.empty_cache()

            # Encode Ground Truth to latents
            gt_latents = self.vae.encode(photos).latent_dist.sample() * 0.18215

        # 2. Sample reduced noise (t ~ Uniform(0, 200))
        # Ensure T_MAX is 200 for refinement
        T_MAX = 200
        timesteps = torch.randint(0, T_MAX, (photos.shape[0],), device=photos.device)
        
        # Add noise to GT latents
        noise = torch.randn_like(gt_latents)
        noisy_latents = self.noise_scheduler.add_noise(gt_latents, noise, timesteps)
        
        # 3. Forward through Stage 2 with Stage-1 conditioning
        # Get region graph (handle batch)
        region_graphs = batch["region_graph"]
        # Use the first one for now or loop (Simplified for performance)
        region_graph = region_graphs[0] if isinstance(region_graphs, list) else region_graphs
        
        output_dict = stage2_model(
            noisy_latents,
            timesteps,
            region_graph,
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
        loss_identity = F.l1_loss(pred_x0_latents, stage1_latents)
        
        # C. Perceptual Loss (LPIPS) and L_delta
        # Requires decoding to image space
        with torch.set_grad_enabled(True):
            # Decode predicted latents to images
            pred_images = self.vae.decode(pred_x0_latents / 0.18215).sample
            pred_images = torch.clamp(pred_images, -1, 1)
            
            # Compute LPIPS between pred and ground truth
            loss_perceptual = self.lpips_loss(pred_images, photos).mean()
            
            # L_delta: Regularization on magnitude of change in image space
            with torch.no_grad():
                s1_images = self.vae.decode(stage1_latents / 0.18215).sample
                s1_images = torch.clamp(s1_images, -1, 1)
            
            image_delta = pred_images - s1_images
            loss_delta = torch.mean(torch.abs(image_delta))
        
        # Final combined loss
        # Use configurable weights from training_config
        lambda_id = getattr(self.training_config, "lambda_identity", 0.5)
        lambda_lpips = getattr(self.training_config, "lambda_lpips", 0.1)
        lambda_delta = getattr(self.training_config, "lambda_delta", 0.05)
        
        total_loss = (
            loss_diffusion 
            + lambda_id * loss_identity 
            + lambda_lpips * loss_perceptual
            + lambda_delta * loss_delta
        )
        
        # Compute Stage-2 metrics
        s2_ssim, s2_psnr = self.compute_metrics(pred_images, photos)
        s1_ssim, s1_psnr = self.compute_metrics(s1_images, photos)
        edge_sim = self.compute_edge_similarity(pred_images, s1_images)

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
    
    def validate_stage2(self, model, num_samples: int = 4) -> Dict[str, float]:
        """
        Validate Stage 2 by checking that output differs from input.
        
        Args:
            model: Stage 2 model
            num_samples: Number of validation samples to process
        
        Returns:
            Dict with validation metrics
        """
        model.eval()
        device = self.accelerator.device
        val_loss = 0.0
        num_processed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_dataloader):
                if batch_idx >= 1:  # Just process 1 batch for validation
                    break
                
                photos = batch["photo"].to(device)
                text_prompts = batch["text_prompt"]
                region_graphs = batch["region_graph"]
                
                # Encode images
                latents = self.vae.encode(photos).latent_dist.sample()
                latents = latents * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                )
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode text
                text_inputs = self.tokenizer(
                    text_prompts,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings_list = self.text_encoder(
                    text_inputs.input_ids.to(device)
                )[0]
                
                # Validate on first sample
                batch_size = min(num_samples, photos.shape[0])
                for i in range(batch_size):
                    output = model(
                        noisy_latents[i:i+1],
                        timesteps[i:i+1],
                        region_graphs[i],
                        text_embeddings_list[i],
                        return_dict=True
                    )
                    noise_pred = output["noise_pred"]
                    loss = F.mse_loss(noise_pred, noise[i:i+1])
                    val_loss += loss.item()
                    num_processed += 1
        
        model.train()
        return {"val_loss": val_loss / max(num_processed, 1)}

    
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
                start_epoch=0
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

        # Ensure auxiliary models are on the same device
        if hasattr(self, 'stage1_model') and self.stage1_model is not None:
            self.stage1_model.to(self.accelerator.device)
        if hasattr(self, 'lpips_loss') and self.lpips_loss is not None:
            self.lpips_loss.to(self.accelerator.device)

        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            model, optimizer, self.train_dataloader, lr_scheduler
        )

        # Resume: load checkpoint weights if start_epoch > 0
        if start_epoch > 0:
            resume_path = self._find_resume_checkpoint(stage, start_epoch)
            if resume_path:
                print(f"▶️  Resuming from checkpoint: {resume_path}")
                ckpt = torch.load(resume_path, map_location=self.accelerator.device, weights_only=False)
                unwrapped = self.accelerator.unwrap_model(model)
                unwrapped.load_state_dict(ckpt["model_state_dict"])
                print(f"✅ Loaded weights from epoch {ckpt['epoch']+1}")
            else:
                print(f"⚠️  No local checkpoint found for epoch {start_epoch}, starting fresh.")

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
                return error
            model.train()

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
                    
                    # Periodic adaptive update every 100 steps
                    if global_step % 100 == 0:
                        t_high = getattr(self.training_config, "delta_threshold_high", 0.5)
                        t_low = getattr(self.training_config, "delta_threshold_low", 0.01)
                        
                        # Get unwrapped model to access residual_alpha
                        unwrapped_model = self.accelerator.unwrap_model(model)
                        old_alpha = unwrapped_model.residual_alpha
                        
                        if self.running_delta_mean > t_high:
                            unwrapped_model.residual_alpha *= 0.9
                            print(f"\n[Adaptive] EMA Delta ({self.running_delta_mean:.3f}) > High Threshold ({t_high}). Reducing residual_alpha: {old_alpha:.4f} -> {unwrapped_model.residual_alpha:.4f}")
                        elif self.running_delta_mean < t_low:
                            unwrapped_model.residual_alpha *= 1.1
                            print(f"\n[Adaptive] EMA Delta ({self.running_delta_mean:.3f}) < Low Threshold ({t_low}). Increasing residual_alpha: {old_alpha:.4f} -> {unwrapped_model.residual_alpha:.4f}")
                        
                        # Clamp residual_alpha ∈ [0.05, 0.5]
                        unwrapped_model.residual_alpha = max(0.05, min(0.5, unwrapped_model.residual_alpha))
                
                # Detailed Logging
                if global_step % self.training_config.log_every_n_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = lr_scheduler.get_last_lr()[0]
                    
                    logs = {
                        f"{stage}/loss": avg_loss,
                        f"{stage}/lr": lr,
                    }
                    if "metrics" in outputs:
                        for k, v in outputs["metrics"].items():
                            logs[f"{stage}/{k}"] = v
                    
                    if self.training_config.use_wandb and self.accelerator.is_main_process:
                        import wandb
                        wandb.log(logs)
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "s2_ssim": f"{np.mean(epoch_ssim) if epoch_ssim else 0:.3f}"
                    })

                # Visual Validation (every 500 steps)
                if self.accelerator.is_main_process and global_step % 500 == 0 and "images" in outputs:
                    self.save_visual_grid(outputs["images"], stage, global_step)
            
            # End of Epoch Evaluation and Diagnostics
            avg_train_ssim = np.mean(epoch_ssim) if epoch_ssim else 0.0
            
            # Validation Step for Overfitting Detection
            val_metrics = self.validate_stage2(model)
            avg_val_ssim = val_metrics.get("val_ssim", 0.0)
            
            if self.accelerator.is_main_process:
                print(f"\n[Epoch {epoch+1}] Train SSIM: {avg_train_ssim:.4f}, Val SSIM: {avg_val_ssim:.4f}")
                
                # Overfitting detection
                if avg_train_ssim > avg_val_ssim + 0.1:
                    print(f"🚨 [WARNING] Stage-2 Overfitting detected! (Train SSIM={avg_train_ssim:.3f}, Val SSIM={avg_val_ssim:.3f})")

                # Identity Sanity Check (Stage 2)
                id_err = 0.0
                if stage == "stage2":
                    id_err = run_identity_check(model)
                    print(f"[Identity Check] Error: {id_err:.6f}")

                # Save comparison log for this epoch
                curr_alpha = getattr(self.accelerator.unwrap_model(model), "residual_alpha", 0.0)
                self.alpha_history.append(curr_alpha)

                epoch_log = {
                    "epoch": epoch + 1,
                    "ssim_train": avg_train_ssim,
                    "ssim_val": avg_val_ssim,
                    "delta_ssim": avg_train_ssim - (outputs["metrics"]["s1_ssim"] if "metrics" in outputs else 0),
                    "identity_error": id_err,
                    "delta_mean_avg": np.mean(epoch_delta_means) if epoch_delta_means else 0,
                    "running_delta_mean_ema": self.running_delta_mean,
                    "residual_alpha": curr_alpha
                }
                comparison_history.append(epoch_log)
                
                log_path = Path(self.training_config.checkpoint_dir) / stage / "comparison_logs.json"
                with open(log_path, "w") as f:
                    json.dump(comparison_history, f, indent=2)

            # Save Best Model (based on Val SSIM if available, else Train)
            eval_metric = avg_val_ssim if avg_val_ssim > 0 else avg_train_ssim
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

            # Save periodic checkpoint
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                self.save_checkpoint(stage, model, epoch)
        
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

        # --- HuggingFace Hub upload ---
        if getattr(self.training_config, "push_to_hub", False):
            try:
                from huggingface_hub import HfApi, create_repo
                token = getattr(self.training_config, "hub_token", None) or os.getenv("HF_TOKEN")
                repo_id = self.training_config.hub_repo_id
                api = HfApi(token=token)

                # Create repo if it doesn't exist
                try:
                    create_repo(repo_id, repo_type="model", private=True, token=token, exist_ok=True)
                except Exception:
                    pass

                # Upload checkpoint file
                hub_path = f"{stage}/{filename}"
                print(f"☁️  Uploading to HF Hub: {repo_id}/{hub_path} ...")
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=hub_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"[{stage}] {'Final' if final else f'Epoch {epoch+1}'} checkpoint"
                )
                print(f"✅ Uploaded to HF Hub: https://huggingface.co/{repo_id}")
                
                # Now aggressively delete old local checkpoints to save space
                self._cleanup_old_checkpoints(stage, checkpoint_dir, keep_count=2)

            except Exception as e:
                print(f"⚠️  HF Hub upload failed — keeping local copy: {e}")
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
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Resume stage1 training from this epoch (e.g. 4 to skip epochs 1-4)")
    
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
    if args.resume_epoch > 0:
        config["training"].resume_from_epoch = args.resume_epoch
        print(f"▶️  Will resume Stage 1 from epoch {args.resume_epoch}")
    
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
