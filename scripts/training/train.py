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
    
    def train_stage2_step(self, batch: Dict, model=None) -> Dict:
        """
        Single training step for Stage 2 - FAST BATCH VERSION.
        Processes entire batch at once for maximum GPU utilization.
        
        Args:
            batch: Batch of data
            model: Model to use (if None, uses self.stage2_model)
        
        Returns:
            Dict with loss and metrics
        """
        # Use provided model or fall back to self.stage2_model
        stage2_model = model if model is not None else self.stage2_model
        
        photos = batch["photo"].to(self.accelerator.device)
        text_prompts = batch["text_prompt"]
        batch_size = photos.shape[0]
        
        # Encode images to latents (fully batched)
        with torch.no_grad():
            latents = self.vae.encode(photos).latent_dist.sample()
            latents = latents * 0.18215
        
        # Sample noise and timesteps (fully batched)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text prompts (fully batched)
        with torch.no_grad():
            text_inputs = self.tokenizer(
                text_prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings_batch = self.text_encoder(
                text_inputs.input_ids.to(self.accelerator.device)
            )[0]  # (B, 77, 768)
        
        # FAST BATCHED FORWARD PASS
        # Use full batch text embeddings for proper batching
        text_embeddings = text_embeddings_batch  # (B, 77, 768)
        
        # Get dummy region graph (use first sample)
        region_graphs = batch["region_graph"]
        region_graph = region_graphs[0] if isinstance(region_graphs, list) else region_graphs
        
        # Move region graph to device
        if hasattr(region_graph, 'node_features') and region_graph.node_features is not None:
            region_graph.node_features = region_graph.node_features.to(self.accelerator.device)
        
        # Forward through Stage 2 with entire batch
        output = stage2_model(
            noisy_latents,  # (B, 4, H/8, W/8) - FULLY BATCHED
            timesteps,      # (B,) - FULLY BATCHED
            region_graph,   # Same region graph for all
            text_embeddings,  # (B, 77, 768) - batched token embeddings
            return_dict=True
        )
        
        noise_pred = output["noise_pred"]
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}
    
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

        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            model, optimizer, self.train_dataloader, lr_scheduler
        )

        # Resume: load checkpoint weights if start_epoch > 0
        if start_epoch > 0:
            resume_path = self._find_resume_checkpoint(stage, start_epoch)
            if resume_path:
                print(f"▶️  Resuming from checkpoint: {resume_path}")
                ckpt = torch.load(resume_path, map_location=self.accelerator.device)
                unwrapped = self.accelerator.unwrap_model(model)
                unwrapped.load_state_dict(ckpt["model_state_dict"])
                print(f"✅ Loaded weights from epoch {ckpt['epoch']+1}")
            else:
                print(f"⚠️  No local checkpoint found for epoch {start_epoch}, starting fresh.")

        global_step = start_epoch * len(train_dataloader)

        for epoch in range(start_epoch, num_epochs):
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
                    # Training step (pass the wrapped model)
                    outputs = train_step_fn(batch, model=model)
                    loss = outputs["loss"]
                    
                    # Debug logging
                    if step == 0:
                        print(f"\nDEBUG - First step:")
                        print(f"  loss.requires_grad: {loss.requires_grad}")
                        print(f"  loss.grad_fn: {loss.grad_fn}")
                        print(f"  model.training: {model.training}")
                    
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
    
    def _find_resume_checkpoint(self, stage: str, start_epoch: int) -> Optional[str]:
        """Find the best local checkpoint to resume from."""
        checkpoint_dir = Path(self.training_config.checkpoint_dir) / stage
        # Look for the checkpoint just before start_epoch
        for ep in range(start_epoch, 0, -1):
            path = checkpoint_dir / f"epoch_{ep}.pt"
            if path.exists():
                return str(path)
        return None

    def save_checkpoint(self, stage: str, model, epoch: int, final: bool = False):
        """
        Save model checkpoint locally, upload to HuggingFace Hub, then aggressively cleanup old checkpoints.
        
        Strategy for 100GB container storage:
        1. Save checkpoint locally
        2. Upload to HuggingFace Hub
        3. Delete older checkpoints (keep only 2 most recent + final)
        4. This keeps disk usage minimal while maintaining resume capability
        """
        if not self.accelerator.is_main_process:
            return

        # Check free disk space before saving (need at least 2 GB)
        import shutil
        checkpoint_dir = Path(self.training_config.checkpoint_dir) / stage
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(str(checkpoint_dir)).free
        free_gb = free_bytes / 1e9
        print(f"💾 Disk free before checkpoint: {free_gb:.1f} GB")

        if final:
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
