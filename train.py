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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from configs.config import ModelConfig, DataConfig, TrainingConfig, get_default_config
from datasets.sketchy_dataset import SketchyDataset, collate_fn as sketchy_collate
from datasets.coco_dataset import COCODataset, collate_fn as coco_collate
from models.stage1_diffusion import Stage1SketchGuidedDiffusion
from models.stage2_refinement import Stage2SemanticRefinement
from data.sketch_extraction import SketchExtractor
from data.region_extraction import RegionExtractor
from data.region_graph import RegionGraphBuilder


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
        
        # Initialize datasets
        self.setup_datasets()
        
        # Initialize optimizers
        self.setup_optimizers()
        
        print(f"Trainer initialized on device: {self.accelerator.device}")
        print(f"Mixed precision: {training_config.mixed_precision}")
        print(f"Training stage: {training_config.train_stage}")
    
    def setup_models(self):
        """Setup models for training."""
        print("Loading pretrained models...")
        
        # Stage 1 model
        if self.training_config.train_stage in ["stage1", "both"]:
            self.stage1_model = Stage1SketchGuidedDiffusion(
                pretrained_model_name=self.model_config.pretrained_model_name,
                sketch_encoder_channels=self.model_config.sketch_encoder_channels,
                freeze_base_unet=self.model_config.freeze_stage1_unet,
                use_lora=self.model_config.use_lora,
                lora_rank=self.model_config.lora_rank
            )
        else:
            self.stage1_model = None
        
        # Stage 2 model
        if self.training_config.train_stage in ["stage2", "both"]:
            # Load UNet
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(
                self.model_config.pretrained_model_name,
                subfolder="unet"
            )
            
            self.stage2_model = Stage2SemanticRefinement(
                unet=unet,
                node_feature_dim=self.model_config.node_feature_dim,
                text_dim=self.model_config.text_dim,
                hidden_dim=self.model_config.hidden_dim,
                num_graph_layers=self.model_config.num_graph_layers,
                num_attention_heads=self.model_config.num_attention_heads,
                fusion_method=self.model_config.fusion_method,
                use_region_adaptive_fusion=self.model_config.use_region_adaptive_fusion
            )
        else:
            self.stage2_model = None
        
        # Shared components
        self.vae = AutoencoderKL.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="vae"
        )
        self.vae.requires_grad_(False)  # Freeze VAE
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="text_encoder"
        )
        self.text_encoder.requires_grad_(False)  # Freeze text encoder
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="tokenizer"
        )
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_config.pretrained_model_name,
            subfolder="scheduler"
        )
        
        print("Models loaded successfully")
    
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
    
    def train_stage1_step(self, batch: Dict) -> Dict:
        """
        Single training step for Stage 1.
        
        Args:
            batch: Batch of data
        
        Returns:
            Dict with loss and metrics
        """
        sketches = batch["sketch"].to(self.accelerator.device)
        photos = batch["photo"].to(self.accelerator.device)
        text_prompts = batch["text_prompt"]
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(photos).latent_dist.sample()
            latents = latents * 0.18215
        
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
        sketch_features = self.stage1_model.encode_sketch(sketches)
        
        # Encode text
        text_embeddings = self.stage1_model.encode_text(text_prompts)
        
        # Predict noise
        noise_pred = self.stage1_model(
            noisy_latents,
            timesteps,
            sketch_features,
            text_embeddings
        )
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}
    
    def train_stage2_step(self, batch: Dict) -> Dict:
        """
        Single training step for Stage 2.
        
        Args:
            batch: Batch of data
        
        Returns:
            Dict with loss and metrics
        """
        # Similar to Stage 1 but with RAGAF attention
        # TODO: Implement full Stage 2 training step
        
        photos = batch["photo"].to(self.accelerator.device)
        text_prompts = batch["text_prompt"]
        region_graphs = batch["region_graph"]
        
        # Encode images
        with torch.no_grad():
            latents = self.vae.encode(photos).latent_dist.sample()
            latents = latents * 0.18215
        
        # For now, simplified training (just UNet loss)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Process first item in batch (simplified)
        # Full implementation would batch process
        text_inputs = self.tokenizer(
            text_prompts[0],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.accelerator.device)
            )[0].squeeze(0)  # (77, 768)
        
        # Forward through Stage 2
        output = self.stage2_model(
            noisy_latents[:1],  # First item only for simplicity
            timesteps[:1],
            region_graphs[0],
            text_embeddings,
            return_dict=True
        )
        
        noise_pred = output["noise_pred"]
        loss = F.mse_loss(noise_pred, noise[:1])
        
        return {"loss": loss}
    
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
                train_step_fn=self.train_stage1_step
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
                train_step_fn=self.train_stage2_step
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
        train_step_fn
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
        """
        # Prepare for distributed training
        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            model, optimizer, self.train_dataloader, lr_scheduler
        )
        
        global_step = 0
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                disable=not self.accelerator.is_main_process,
                desc=f"[{stage.upper()}] Epoch {epoch+1}/{num_epochs}"
            )
            
            for step, batch in progress_bar:
                with self.accelerator.accumulate(model):
                    # Training step
                    outputs = train_step_fn(batch)
                    loss = outputs["loss"]
                    
                    # Backward
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            self.training_config.max_grad_norm
                        )
                    
                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Logging
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % self.training_config.log_every_n_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = lr_scheduler.get_last_lr()[0]
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}"
                    })
                    
                    if self.training_config.use_wandb and self.accelerator.is_main_process:
                        import wandb
                        wandb.log({
                            f"{stage}/loss": avg_loss,
                            f"{stage}/lr": lr,
                            f"{stage}/epoch": epoch,
                            "global_step": global_step
                        })
            
            # Save checkpoint
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                self.save_checkpoint(stage, model, epoch)
        
        # Save final checkpoint
        self.save_checkpoint(stage, model, num_epochs, final=True)
    
    def save_checkpoint(self, stage: str, model, epoch: int, final: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = Path(self.training_config.checkpoint_dir) / stage
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            path = checkpoint_dir / "final.pt"
        else:
            path = checkpoint_dir / f"epoch_{epoch+1}.pt"
        
        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "config": {
                "model": vars(self.model_config),
                "data": vars(self.data_config),
                "training": vars(self.training_config)
            }
        }, path)
        
        print(f"Checkpoint saved: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train RAGAF-Diffusion")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--stage", type=str, default="both", choices=["stage1", "stage2", "both"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    
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
