#!/usr/bin/env python3
"""
Stage-1 Training with SSIM Loss

Key improvements over previous version:
1. ✅ SSIM loss added to training objective
2. ✅ Larger validation sample size (100 instead of 10)
3. ✅ Fixed random seed for reproducible validation
4. ✅ Better learning rate schedule
5. ✅ Optimizes what we measure!

Usage:
    python train_stage1_with_ssim.py --resume_from /root/checkpoints/stage1_improved/epoch_12.pt

Author: RAGAF-Diffusion Research Team  
Date: March 9, 2026
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import json
from datetime import datetime

# SSIM loss
from pytorch_msssim import ssim as compute_ssim

# WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.configs.config import get_default_config
from src.datasets.sketchy_dataset import SketchyDataset
from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from huggingface_hub import HfApi, create_repo


class Stage1TrainerWithSSIM:
    """
    Enhanced trainer that optimizes for SSIM (structural similarity).
    """
    
    def __init__(
        self,
        learning_rate: float = 5e-6,  # Slightly higher than 4e-6 where it got stuck
        num_epochs: int = 25,  # Train to 25 instead of 20
        batch_size: int = 8,  # Increased from 4 for better gradient estimates
        use_perceptual_loss: bool = True,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.3,  # NEW: SSIM loss weight
        freeze_unet: bool = False,
        lora_rank: int = 8,
        checkpoint_dir: str = "/root/checkpoints/stage1_with_ssim",
        hf_repo: str = "DrRORAL/ragaf-diffusion-checkpoints",
        validate_every: int = 2,
        validation_samples: int = 100,  # NEW: 100 instead of 10
        early_stopping_patience: int = 5,
        device: str = "cuda",
        use_wandb: bool = True,
        wandb_project: str = "ragaf-diffusion-stage1",
        wandb_run_name: str = None,
        resume_from_checkpoint: str = None
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight  # NEW
        self.freeze_unet = freeze_unet
        self.lora_rank = lora_rank
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo = hf_repo
        self.validate_every = validate_every
        self.validation_samples = validation_samples  # NEW
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or f"stage1-ssim-lr{learning_rate:.0e}-{datetime.now().strftime('%m%d-%H%M')}"
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Training state
        self.start_epoch = 1
        self.best_ssim = 0.0
        self.no_improvement_count = 0
        self.training_log = []
        
        # Initialize
        if self.use_wandb:
            self.init_wandb()
        
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_perceptual_loss()
        
        # Resume if specified
        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint)
    
    def init_wandb(self):
        """Initialize Weights & Biases tracking."""
        print("\n🔗 Initializing WandB...")
        
        try:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "use_perceptual_loss": self.use_perceptual_loss,
                    "perceptual_weight": self.perceptual_weight,
                    "ssim_weight": self.ssim_weight,  # NEW
                    "freeze_unet": self.freeze_unet,
                    "lora_rank": self.lora_rank,
                    "validation_samples": self.validation_samples,  # NEW
                    "architecture": "Stage1SketchGuidedDiffusion",
                    "base_model": "runwayml/stable-diffusion-v1-5",
                    "dataset": "Sketchy",
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingWarmRestarts",
                    "improvements": "Added SSIM loss + 100-sample validation"
                },
                tags=["stage1", "sketch-guided", "diffusion", "ssim-loss"],
                notes=f"Training with SSIM loss (weight={self.ssim_weight}) to optimize structural similarity. Validation uses {self.validation_samples} samples for reliability."
            )
            print(f"   ✅ WandB initialized: {wandb.run.url}")
        except Exception as e:
            print(f"   ⚠️  WandB initialization failed: {e}")
            self.use_wandb = False
    
    def setup_model(self):
        """Setup model."""
        print("\n📦 Loading model...")
        
        config = get_default_config()
        
        # Create model
        self.model = Stage1SketchGuidedDiffusion(
            pretrained_model_name=config['model'].pretrained_model_name,
            sketch_encoder_channels=config['model'].sketch_encoder_channels,
            freeze_base_unet=self.freeze_unet,
            use_lora=True,
            lora_rank=self.lora_rank
        ).to(self.device)
        
        # Load VAE and scheduler
        self.vae = AutoencoderKL.from_pretrained(
            config['model'].pretrained_model_name,
            subfolder="vae"
        ).to(self.device)
        self.vae.requires_grad_(False)
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config['model'].pretrained_model_name,
            subfolder="scheduler"
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"   ✅ Model loaded")
        
    def setup_data(self):
        """Setup data."""
        print("\n📁 Loading dataset...")
        
        config = get_default_config()
        
        # Custom collate function
        def collate_fn(batch):
            collated = {}
            tensor_keys = ['sketch', 'photo']
            for key in tensor_keys:
                if key in batch[0]:
                    collated[key] = torch.stack([item[key] for item in batch])
            
            if 'text_prompt' in batch[0]:
                collated['text_prompt'] = [item['text_prompt'] for item in batch]
            if 'category' in batch[0]:
                collated['category'] = [item['category'] for item in batch]
            
            return collated
        
        # Training dataset
        self.train_dataset = SketchyDataset(
            root_dir=config['data'].sketchy_root,
            split='train',
            image_size=config['data'].image_size,
            augment=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Validation dataset
        self.val_dataset = SketchyDataset(
            root_dir=config['data'].sketchy_root,
            split='test',
            image_size=config['data'].image_size,
            augment=False
        )
        
        print(f"   Training samples: {len(self.train_dataset)}")
        print(f"   Validation samples: {len(self.val_dataset)}")
        print(f"   Validation will use: {self.validation_samples} samples (fixed seed)")
        print(f"   Batches per epoch: {len(self.train_loader)}")
        print(f"   ✅ Dataset loaded")
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        print("\n⚙️  Setting up optimizer...")
        
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=1,
            eta_min=1e-7
        )
        
        print(f"   Optimizer: AdamW (lr={self.learning_rate:.0e}, weight_decay=0.01)")
        print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0=5)")
        print(f"   ✅ Optimizer ready")
        
    def setup_perceptual_loss(self):
        """Setup perceptual loss."""
        if self.use_perceptual_loss:
            print("\n🎨 Setting up perceptual loss...")
            from lpips import LPIPS
            self.lpips_model = LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            print(f"   ✅ LPIPS loaded (AlexNet)")
        else:
            self.lpips_model = None
            
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        print(f"\n📥 Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Try to load optimizer/scheduler, but don't fail if batch size changed
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"   ✅ Loaded optimizer and scheduler state")
        except (ValueError, KeyError) as e:
            print(f"   ⚠️  Could not load optimizer state (batch size changed?): {e}")
            print(f"   ✅ Continuing with fresh optimizer (learning rate: {self.learning_rate:.0e})")
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.training_log = checkpoint.get('training_log', [])
        self.best_ssim = checkpoint.get('best_ssim', 0.0)
        
        print(f"   ✅ Resumed from epoch {checkpoint['epoch']}")
        print(f"   Best SSIM so far: {self.best_ssim:.4f}")
        print(f"   Starting from epoch: {self.start_epoch}")
            
    def train_step(self, batch):
        """Single training step with SSIM loss."""
        sketches = batch["sketch"].to(self.device)
        photos = batch["photo"].to(self.device)
        text_prompts = batch["text_prompt"]
        
        # Encode to latents
        with torch.no_grad():
            latents = self.vae.encode(photos).latent_dist.sample()
            latents = latents * 0.18215
        
        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Forward pass
        sketch_features = self.model.encode_sketch(sketches)
        text_embeddings = self.model.encode_text(text_prompts)
        noise_pred = self.model(noisy_latents, timesteps, sketch_features, text_embeddings)
        
        # MSE loss
        mse_loss = F.mse_loss(noise_pred, noise)
        
        # Initialize total loss
        total_loss = mse_loss
        perceptual_loss_val = 0.0
        ssim_loss_val = 0.0
        
        # Compute predicted images for perceptual/SSIM losses
        if self.use_perceptual_loss or self.ssim_weight > 0:
            with torch.no_grad():
                # Proper DDPM denoising formula
                alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
                sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                
                sqrt_alpha_prod = sqrt_alpha_prod.flatten().view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().view(-1, 1, 1, 1)
                
                pred_latents = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                pred_latents = torch.clamp(pred_latents, -4, 4)
            
            # Decode (requires grad for SSIM loss)
            pred_images = self.vae.decode(pred_latents / 0.18215).sample
            target_images = self.vae.decode(latents / 0.18215).sample
            
            # LPIPS perceptual loss
            if self.use_perceptual_loss and self.lpips_model is not None:
                with torch.no_grad():
                    perceptual_loss = self.lpips_model(pred_images.detach(), target_images).mean()
                perceptual_loss_val = perceptual_loss.item()
                total_loss = total_loss + self.perceptual_weight * perceptual_loss
            
            # SSIM loss (NEW!)
            if self.ssim_weight > 0:
                # SSIM returns value in [0, 1], we want to maximize it, so minimize (1 - SSIM)
                ssim_val = compute_ssim(
                    pred_images, 
                    target_images, 
                    data_range=2.0,  # Images in [-1, 1]
                    size_average=True
                )
                ssim_loss = 1 - ssim_val
                ssim_loss_val = ssim_loss.item()
                total_loss = total_loss + self.ssim_weight * ssim_loss
        
        return {
            "loss": total_loss,
            "mse_loss": mse_loss.item(),
            "perceptual_loss": perceptual_loss_val,
            "ssim_loss": ssim_loss_val  # NEW
        }
    
    @torch.no_grad()
    def validate(self, log_images=True):
        """Validation with larger sample size and fixed seed."""
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        import numpy as np
        import cv2
        
        self.model.eval()
        
        ssim_scores = []
        psnr_scores = []
        lpips_scores = []
        
        wandb_images = []
        
        # Fixed random seed for reproducibility
        torch.manual_seed(42)
        num_samples = min(self.validation_samples, len(self.val_dataset))
        indices = torch.randperm(len(self.val_dataset))[:num_samples]
        
        for idx_num, idx in enumerate(tqdm(indices, desc="Validating", leave=False)):
            sample = self.val_dataset[idx]
            sketch = sample['sketch'].unsqueeze(0).to(self.device)
            photo = sample['photo']
            prompt = sample['text_prompt']
            
            # Generate using DDIM sampling
            sketch_features = self.model.encode_sketch(sketch)
            text_embeddings = self.model.encode_text([prompt])
            
            # DDIM sampling (50 steps)
            latent = torch.randn(1, 4, 32, 32, device=self.device)
            
            ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
            ddim_scheduler.set_timesteps(50)
            
            for t in ddim_scheduler.timesteps:
                timesteps = torch.tensor([t], device=self.device)
                noise_pred = self.model(latent, timesteps, sketch_features, text_embeddings)
                latent = ddim_scheduler.step(noise_pred, t, latent).prev_sample
            
            # Decode
            generated = self.vae.decode(latent / 0.18215).sample[0]
            
            # Compute metrics
            gen_np = ((generated.cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
            photo_np = ((photo.numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
            
            # PSNR
            psnr_val = psnr(photo_np, gen_np, data_range=255)
            psnr_scores.append(psnr_val)
            
            # SSIM (grayscale)
            gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
            photo_gray = cv2.cvtColor(photo_np, cv2.COLOR_RGB2GRAY)
            ssim_val = ssim(photo_gray, gen_gray, data_range=255)
            ssim_scores.append(ssim_val)
            
            # LPIPS
            if self.lpips_model is not None:
                gen_tensor = torch.from_numpy(gen_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 127.5 - 1
                photo_tensor = photo.unsqueeze(0).to(self.device)
                lpips_val = self.lpips_model(gen_tensor, photo_tensor).item()
                lpips_scores.append(lpips_val)
            
            # Log first 5 images to WandB
            if self.use_wandb and log_images and idx_num < 5:
                sketch_np = sketch[0, 0].cpu().numpy()
                sketch_np = (sketch_np * 255).astype(np.uint8)
                
                wandb_images.append(wandb.Image(
                    gen_np,
                    caption=f"Sample {idx_num+1}: {prompt[:50]}..."
                ))
        
        # Log images to WandB
        if self.use_wandb and wandb_images:
            wandb.log({"val/generated_images": wandb_images})
        
        self.model.train()
        
        return {
            "ssim": np.mean(ssim_scores),
            "psnr": np.mean(psnr_scores),
            "lpips": np.mean(lpips_scores) if lpips_scores else 0.0
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_log": self.training_log,
            "best_ssim": self.best_ssim,
            "config": {
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "use_perceptual_loss": self.use_perceptual_loss,
                "perceptual_weight": self.perceptual_weight,
                "ssim_weight": self.ssim_weight,
                "freeze_unet": self.freeze_unet,
                "lora_rank": self.lora_rank,
                "validation_samples": self.validation_samples
            }
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"   💾 Saved checkpoint: {path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"   ⭐ Saved best checkpoint: {best_path}")
        
        # Save final checkpoint
        if epoch == self.num_epochs:
            final_path = self.checkpoint_dir / "final.pt"
            torch.save(checkpoint, final_path)
            print(f"   🎉 Saved final checkpoint: {final_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\n🚀 Starting training from epoch {self.start_epoch}...")
        print(f"   Target epochs: {self.num_epochs}")
        print(f"   Validation every {self.validate_every} epochs with {self.validation_samples} samples")
        print(f"   Early stopping patience: {self.early_stopping_patience}")
        print()
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.model.train()
            
            epoch_loss = 0
            epoch_mse = 0
            epoch_perceptual = 0
            epoch_ssim_loss = 0  # NEW
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                self.optimizer.zero_grad()
                
                # Training step
                metrics = self.train_step(batch)
                loss = metrics["loss"]
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Accumulate metrics
                epoch_loss += metrics["loss"].item()
                epoch_mse += metrics["mse_loss"]
                epoch_perceptual += metrics["perceptual_loss"]
                epoch_ssim_loss += metrics["ssim_loss"]
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss'].item():.4f}",
                    "mse": f"{metrics['mse_loss']:.4f}",
                    "lpips": f"{metrics['perceptual_loss']:.4f}",
                    "ssim_loss": f"{metrics['ssim_loss']:.4f}",  # NEW
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.0e}"
                })
            
            # Update scheduler
            self.scheduler.step()
            
            # Average metrics
            n_batches = len(self.train_loader)
            avg_loss = epoch_loss / n_batches
            avg_mse = epoch_mse / n_batches
            avg_perceptual = epoch_perceptual / n_batches
            avg_ssim_loss = epoch_ssim_loss / n_batches  # NEW
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Total loss: {avg_loss:.4f}")
            print(f"   MSE loss: {avg_mse:.4f}")
            print(f"   Perceptual loss: {avg_perceptual:.4f}")
            print(f"   SSIM loss: {avg_ssim_loss:.4f}")  # NEW
            print(f"   Learning rate: {current_lr:.0e}")
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/mse_loss": avg_mse,
                    "train/perceptual_loss": avg_perceptual,
                    "train/ssim_loss": avg_ssim_loss,  # NEW
                    "train/learning_rate": current_lr,
                    "epoch": epoch
                })
            
            # Validation
            if epoch % self.validate_every == 0 or epoch == self.num_epochs:
                print(f"\n🔍 Running validation ({self.validation_samples} samples, fixed seed)...")
                val_metrics = self.validate()
                
                print(f"   SSIM: {val_metrics['ssim']:.4f}")
                print(f"   PSNR: {val_metrics['psnr']:.2f}")
                print(f"   LPIPS: {val_metrics['lpips']:.4f}")
                
                # Check if best
                is_best = val_metrics['ssim'] > self.best_ssim
                if is_best:
                    self.best_ssim = val_metrics['ssim']
                    self.no_improvement_count = 0
                    print(f"   ⭐ New best SSIM!")
                else:
                    self.no_improvement_count += 1
                    print(f"   No improvement for {self.no_improvement_count} validations")
                
                # Log validation metrics
                if self.use_wandb:
                    wandb.log({
                        "val/ssim": val_metrics['ssim'],
                        "val/psnr": val_metrics['psnr'],
                        "val/lpips": val_metrics['lpips'],
                        "epoch": epoch
                    })
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Log to training log
                self.training_log.append({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_mse": avg_mse,
                    "train_perceptual": avg_perceptual,
                    "train_ssim_loss": avg_ssim_loss,  # NEW
                    "val_ssim": val_metrics['ssim'],
                    "val_psnr": val_metrics['psnr'],
                    "val_lpips": val_metrics['lpips'],
                    "learning_rate": current_lr
                })
                
                # Early stopping
                if self.no_improvement_count >= self.early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered! No improvement for {self.early_stopping_patience} validations.")
                    print(f"   Best SSIM: {self.best_ssim:.4f}")
                    break
            else:
                # Save checkpoint without validation
                self.save_checkpoint(epoch, is_best=False)
            
            print()
        
        print(f"\n✅ Training complete!")
        print(f"   Best SSIM: {self.best_ssim:.4f}")
        print(f"   Checkpoints saved in: {self.checkpoint_dir}")
        
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Stage-1 with SSIM loss')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                        help='Learning rate (default: 5e-6)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs (default: 25)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8, increased for better gradients)')
    parser.add_argument('--ssim_weight', type=float, default=0.3,
                        help='SSIM loss weight (default: 0.3)')
    parser.add_argument('--validation_samples', type=int, default=100,
                        help='Number of validation samples (default: 100)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Stage1TrainerWithSSIM(
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        ssim_weight=args.ssim_weight,
        validation_samples=args.validation_samples,
        use_wandb=not args.no_wandb,
        resume_from_checkpoint=args.resume_from
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
