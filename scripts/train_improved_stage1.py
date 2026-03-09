#!/usr/bin/env python3
"""
Improved Stage-1 Training Script

This script retrains Stage-1 model with optimized hyperparameters to fix
the poor performance (SSIM=0.25, FID=280).

Key improvements:
1. Learning rate: 1e-4 → 1e-5 (10x lower)
2. Training epochs: 10 → 20 (2x longer)
3. Perceptual loss: Added LPIPS
4. UNet layers: Unfrozen for full adaptation
5. LoRA rank: 4 → 8 (2x capacity)
6. Early stopping: Added with SSIM monitoring

Usage:
    python train_improved_stage1.py
    
    Or with custom settings:
    python train_improved_stage1.py --learning_rate 5e-6 --epochs 30

Author: RAGAF-Diffusion Research Team
Date: March 8, 2026
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

# WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not installed. Run: pip install wandb")

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts" / "training"))

from src.configs.config import get_default_config
from src.datasets.sketchy_dataset import SketchyDataset
from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler
from huggingface_hub import HfApi, create_repo


class ImprovedStage1Trainer:
    """
    Improved trainer for Stage-1 with better hyperparameters.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-5,
        num_epochs: int = 20,
        batch_size: int = 4,
        use_perceptual_loss: bool = True,
        perceptual_weight: float = 0.1,
        freeze_unet: bool = False,
        lora_rank: int = 8,
        checkpoint_dir: str = "/root/checkpoints/stage1_improved",
        hf_repo: str = "DrRORAL/ragaf-diffusion-checkpoints",
        validate_every: int = 2,
        early_stopping_patience: int = 4,
        device: str = "cuda",
        use_wandb: bool = True,
        wandb_project: str = "ragaf-diffusion-stage1",
        wandb_run_name: str = None
    ):
        """
        Initialize improved trainer.
        
        Args:
            learning_rate: Learning rate (1e-5 recommended)
            num_epochs: Number of training epochs
            batch_size: Batch size
            use_perceptual_loss: Whether to use LPIPS loss
            perceptual_weight: Weight for perceptual loss
            freeze_unet: Whether to freeze UNet (False recommended)
            lora_rank: LoRA rank (8 recommended)
            checkpoint_dir: Where to save checkpoints
            hf_repo: HuggingFace repo for auto-upload
            validate_every: Validate every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            device: Device to use
            use_wandb: Whether to use Weights & Biases tracking
            wandb_project: WandB project name
            wandb_run_name: WandB run name (auto-generated if None)
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight
        self.freeze_unet = freeze_unet
        self.lora_rank = lora_rank
        self.checkpoint_dir = Path(checkpoint_dir)
        self.hf_repo = hf_repo
        self.validate_every = validate_every
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # WandB configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or f"stage1-lr{learning_rate:.0e}-{datetime.now().strftime('%m%d-%H%M')}"
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training log
        self.training_log = []
        self.best_ssim = 0.0
        self.epochs_without_improvement = 0
        
        print("=" * 80)
        print("IMPROVED STAGE-1 TRAINER INITIALIZED")
        print("=" * 80)
        print(f"Learning rate: {self.learning_rate:.0e}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Perceptual loss: {self.use_perceptual_loss}")
        print(f"Freeze UNet: {self.freeze_unet}")
        print(f"LoRA rank: {self.lora_rank}")
        print(f"Device: {self.device}")
        print(f"WandB tracking: {self.use_wandb}")
        if self.use_wandb:
            print(f"WandB project: {self.wandb_project}")
            print(f"WandB run: {self.wandb_run_name}")
        print("=" * 80)
        
        # Initialize WandB
        if self.use_wandb:
            self.init_wandb()
        
        # Initialize components
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_perceptual_loss()
    
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
                    "freeze_unet": self.freeze_unet,
                    "lora_rank": self.lora_rank,
                    "architecture": "Stage1SketchGuidedDiffusion",
                    "base_model": "stabilityai/stable-diffusion-2-1-base",
                    "dataset": "Sketchy",
                    "optimizer": "AdamW",
                    "scheduler": "DDPM"
                },
                tags=["stage1", "sketch-guided", "diffusion", "improved"],
                notes=f"Improved training with LR={self.learning_rate:.0e}, fixing poor metrics (SSIM=0.25→target 0.6)"
            )
            print(f"   ✅ WandB initialized: {wandb.run.url}")
        except Exception as e:
            print(f"   ⚠️  WandB initialization failed: {e}")
            print(f"   Continuing without WandB tracking...")
            self.use_wandb = False
        
    def setup_model(self):
        """Setup model."""
        print("\n📦 Loading model...")
        
        config = get_default_config()
        
        # Create model with improved settings
        self.model = Stage1SketchGuidedDiffusion(
            pretrained_model_name=config['model'].pretrained_model_name,
            sketch_encoder_channels=config['model'].sketch_encoder_channels,
            freeze_base_unet=self.freeze_unet,  # Use provided setting
            use_lora=True,
            lora_rank=self.lora_rank  # Use provided rank
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
        
        # Custom collate function to handle non-tensor objects like RegionGraph
        def collate_fn(batch):
            """Custom collate function that filters out non-tensor fields."""
            collated = {}
            # Only collate tensor/numeric fields
            tensor_keys = ['sketch', 'photo']
            for key in tensor_keys:
                if key in batch[0]:
                    collated[key] = torch.stack([item[key] for item in batch])
            
            # Keep string fields as lists
            if 'text_prompt' in batch[0]:
                collated['text_prompt'] = [item['text_prompt'] for item in batch]
            if 'category' in batch[0]:
                collated['category'] = [item['category'] for item in batch]
            
            # Skip region_graph and other non-tensor objects
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
            collate_fn=collate_fn  # Use custom collate to handle RegionGraph
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
        print(f"   Batches per epoch: {len(self.train_loader)}")
        print(f"   ✅ Dataset loaded")
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        print("\n⚙️  Configuring optimizer...")
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2
        )
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.train_loader) * 5,  # Restart every 5 epochs
            T_mult=2,
            eta_min=self.learning_rate * 0.1
        )
        
        print(f"   Optimizer: AdamW")
        print(f"   Learning rate: {self.learning_rate:.0e}")
        print(f"   Scheduler: CosineAnnealingWarmRestarts")
        print(f"   ✅ Optimizer configured")
        
    def setup_perceptual_loss(self):
        """Setup perceptual loss (LPIPS)."""
        if self.use_perceptual_loss:
            print("\n🎨 Loading LPIPS...")
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
                print(f"   ✅ LPIPS loaded (weight: {self.perceptual_weight})")
            except ImportError:
                print(f"   ⚠️  LPIPS not available. Install: pip install lpips")
                self.use_perceptual_loss = False
                self.lpips_model = None
        else:
            self.lpips_model = None
            
    def train_step(self, batch):
        """Single training step."""
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
        
        # Perceptual loss
        if self.use_perceptual_loss and self.lpips_model is not None:
            # Decode predicted and target
            with torch.no_grad():
                pred_latents = noisy_latents - noise_pred
                pred_images = self.vae.decode(pred_latents / 0.18215).sample
                target_images = self.vae.decode(latents / 0.18215).sample
            
            # Compute LPIPS
            perceptual_loss = self.lpips_model(pred_images, target_images).mean()
            
            # Combined loss
            total_loss = mse_loss + self.perceptual_weight * perceptual_loss
            
            return {
                "loss": total_loss,
                "mse_loss": mse_loss.item(),
                "perceptual_loss": perceptual_loss.item()
            }
        else:
            return {
                "loss": mse_loss,
                "mse_loss": mse_loss.item(),
                "perceptual_loss": 0.0
            }
    
    @torch.no_grad()
    def validate(self, num_samples=10, log_images=True):
        """Quick validation."""
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        import numpy as np
        import cv2
        
        self.model.eval()
        
        ssim_scores = []
        psnr_scores = []
        lpips_scores = []
        
        # Storage for WandB image logging
        wandb_images = []
        
        indices = torch.randperm(len(self.val_dataset))[:num_samples]
        
        for idx_num, idx in enumerate(indices):
            sample = self.val_dataset[idx]
            sketch = sample['sketch'].unsqueeze(0).to(self.device)
            photo = sample['photo']
            prompt = sample['text_prompt']
            
            # Generate
            sketch_features = self.model.encode_sketch(sketch)
            text_embeddings = self.model.encode_text([prompt])
            
            # Simple generation (simplified)
            latent = torch.randn(1, 4, 32, 32, device=self.device)
            
            for t in reversed(range(0, 50, 5)):  # Quick sampling
                timesteps = torch.tensor([t], device=self.device)
                noise_pred = self.model(latent, timesteps, sketch_features, text_embeddings)
                latent = latent - 0.1 * noise_pred
            
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
            
            # Log first 3 images to WandB
            if self.use_wandb and log_images and idx_num < 3:
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
                "freeze_unet": self.freeze_unet,
                "lora_rank": self.lora_rank
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
        
        # Upload to HuggingFace
        if self.hf_repo:
            try:
                self.upload_to_hf(path, f"stage1_improved/epoch_{epoch}.pt")
            except Exception as e:
                print(f"   ⚠️  HF upload failed: {e}")
    
    def upload_to_hf(self, local_path, hf_path):
        """Upload checkpoint to HuggingFace."""
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_path,
                repo_id=self.hf_repo,
                repo_type="model"
            )
            print(f"   ☁️  Uploaded to HF: {hf_path}")
        except Exception as e:
            print(f"   ⚠️  Upload failed: {e}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("STARTING IMPROVED STAGE-1 TRAINING")
        print("=" * 80)
        
        start_time = datetime.now()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}/{self.num_epochs}")
            print(f"{'='*80}")
            
            self.model.train()
            epoch_losses = []
            epoch_mse_losses = []
            epoch_perceptual_losses = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch_idx, batch in enumerate(pbar):
                # Train step
                self.optimizer.zero_grad()
                metrics = self.train_step(batch)
                loss = metrics["loss"]
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Log
                epoch_losses.append(loss.item())
                epoch_mse_losses.append(metrics["mse_loss"])
                epoch_perceptual_losses.append(metrics["perceptual_loss"])
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{metrics['mse_loss']:.4f}",
                    "lpips": f"{metrics['perceptual_loss']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.0e}"
                })
            
            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_mse = sum(epoch_mse_losses) / len(epoch_mse_losses)
            avg_perceptual = sum(epoch_perceptual_losses) / len(epoch_perceptual_losses)
            
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Total loss: {avg_loss:.4f}")
            print(f"   MSE loss: {avg_mse:.4f}")
            print(f"   Perceptual loss: {avg_perceptual:.4f}")
            print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']:.0e}")
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    "train/mse_loss": avg_mse,
                    "train/perceptual_loss": avg_perceptual,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Validation
            if epoch % self.validate_every == 0:
                print(f"\n🔍 Running validation...")
                val_metrics = self.validate(num_samples=10)
                print(f"   SSIM: {val_metrics['ssim']:.4f}")
                
                # Log validation metrics to WandB
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "val/ssim": val_metrics['ssim'],
                        "val/psnr": val_metrics['psnr'],
                        "val/lpips": val_metrics['lpips']
                    })
                
                # Check for improvement
                is_best = val_metrics['ssim'] > self.best_ssim
                if is_best:
                    self.best_ssim = val_metrics['ssim']
                    self.epochs_without_improvement = 0
                    print(f"   ⭐ New best SSIM!")
                else:
                    self.epochs_without_improvement += 1
                    print(f"   No improvement for {self.epochs_without_improvement} epochs")
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Log
                self.training_log.append({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "mse_loss": avg_mse,
                    "perceptual_loss": avg_perceptual,
                    "ssim": val_metrics['ssim'],
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
                
                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered (no improvement for {self.early_stopping_patience} epochs)")
                    break
            else:
                # Save checkpoint without validation
                self.save_checkpoint(epoch, is_best=False)
        
        # Training complete
        elapsed = datetime.now() - start_time
        print(f"\n" + "=" * 80)
        print(f"✅ TRAINING COMPLETE!")
        print(f"=" * 80)
        print(f"Total time: {elapsed}")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        
        # Log final metrics to WandB
        if self.use_wandb:
            wandb.log({
                "final/best_ssim": self.best_ssim,
                "final/total_epochs": epoch,
                "final/training_time_seconds": elapsed.total_seconds()
            })
            wandb.finish()
            print(f"✅ WandB run finished")
        
        # Save training log
        log_path = self.checkpoint_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f"Training log saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Improved Stage-1 Training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--no_perceptual_loss", action="store_true", help="Disable perceptual loss")
    parser.add_argument("--freeze_unet", action="store_true", help="Freeze UNet")
    parser.add_argument("--checkpoint_dir", type=str, default="/root/checkpoints/stage1_improved")
    parser.add_argument("--hf_repo", type=str, default="DrRORAL/ragaf-diffusion-checkpoints")
    
    # WandB arguments
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB tracking")
    parser.add_argument("--wandb_project", type=str, default="ragaf-diffusion-stage1", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ImprovedStage1Trainer(
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_perceptual_loss=not args.no_perceptual_loss,
        freeze_unet=args.freeze_unet,
        lora_rank=args.lora_rank,
        checkpoint_dir=args.checkpoint_dir,
        hf_repo=args.hf_repo,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
