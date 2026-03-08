"""
Improved Training Configuration for Stage-1 Model

This configuration addresses the issues identified in validation:
1. Lower learning rate (1e-5 instead of 1e-4)
2. More trainable parameters (unfreeze some UNet layers)
3. Perceptual loss added (LPIPS)
4. Longer training (20 epochs instead of 10)
5. Better validation schedule

Author: RAGAF-Diffusion Research Team
Date: March 8, 2026
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs.config import (
    ModelConfig, 
    DataConfig, 
    TrainingConfig, 
    InferenceConfig
)


@dataclass
class ImprovedModelConfig(ModelConfig):
    """Improved model configuration for better learning."""
    
    # CRITICAL FIX: Unfreeze UNet for better adaptation
    freeze_stage1_unet: bool = False  # Changed from True → False
    
    # Use LoRA for efficient fine-tuning
    use_lora: bool = True
    lora_rank: int = 8  # Increased from 4 → 8 for more capacity
    lora_alpha: int = 8  # Increased from 4 → 8
    
    # Sketch encoder - keep same
    sketch_encoder_channels: List[int] = field(default_factory=lambda: [320, 640, 1280, 1280])


@dataclass
class ImprovedTrainingConfig(TrainingConfig):
    """Improved training configuration."""
    
    # CRITICAL FIX #1: Lower learning rate
    learning_rate: float = 1e-5  # Changed from 1e-4 → 1e-5 (10x lower)
    
    # CRITICAL FIX #2: Longer training
    stage1_epochs: int = 20  # Changed from 10 → 20
    stage2_epochs: int = 10  # Changed from 5 → 10
    
    # CRITICAL FIX #3: Add perceptual loss weight
    use_perceptual_loss: bool = True
    perceptual_loss_weight: float = 0.1  # 0.1 * LPIPS + 1.0 * MSE
    
    # Better learning rate schedule
    lr_scheduler: str = "cosine_with_restarts"  # Better than plain cosine
    lr_warmup_steps: int = 1000  # Increased from 500
    lr_num_cycles: int = 3  # For cosine_with_restarts
    
    # Gradient clipping (prevent exploding gradients)
    max_grad_norm: float = 1.0
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 1e-2
    
    # Mixed precision - use bf16 for RTX 5090
    mixed_precision: str = "bf16"
    
    # Checkpointing - save every 2 epochs
    save_every_n_epochs: int = 2
    checkpoint_dir: str = "/root/checkpoints"
    
    # HuggingFace Hub
    push_to_hub: bool = True
    hub_repo_id: str = "DrRORAL/ragaf-diffusion-checkpoints"
    
    # Validation
    validate_every_n_epochs: int = 2  # Validate every 2 epochs
    num_validation_samples: int = 10  # Quick validation
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 4  # Stop if no improvement for 4 epochs
    early_stopping_metric: str = "ssim"  # Monitor SSIM
    early_stopping_min_delta: float = 0.01  # Min improvement threshold
    
    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = False  # Can enable if needed
    
    # Data augmentation during training
    use_advanced_augmentation: bool = True
    augmentation_strength: float = 0.3  # Moderate augmentation


@dataclass
class ImprovedDataConfig(DataConfig):
    """Improved data configuration."""
    
    # Dataset
    dataset_name: str = "sketchy"
    sketchy_root: str = "/workspace/sketchy"
    
    # Image size - keep at 256 for speed
    image_size: int = 256
    
    # Batch size - can handle 4 on RTX 5090
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation - enable for better generalization
    use_augmentation: bool = True
    
    # Cache
    cache_sketches: bool = True


def get_improved_config():
    """Get improved configuration for retraining."""
    return {
        "model": ImprovedModelConfig(),
        "data": ImprovedDataConfig(),
        "training": ImprovedTrainingConfig(),
        "inference": InferenceConfig()
    }


def print_config_comparison():
    """Print comparison between old and new config."""
    from src.configs.config import get_default_config
    
    old_config = get_default_config()
    new_config = get_improved_config()
    
    print("=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)
    
    print("\n📊 KEY CHANGES:")
    print("-" * 80)
    
    # Learning rate
    old_lr = old_config['training'].learning_rate
    new_lr = new_config['training'].learning_rate
    print(f"Learning Rate:")
    print(f"  OLD: {old_lr:.0e} (too high!)")
    print(f"  NEW: {new_lr:.0e} ✅ (10x lower)")
    
    # UNet freezing
    old_freeze = old_config['model'].freeze_stage1_unet
    new_freeze = new_config['model'].freeze_stage1_unet
    print(f"\nUNet Trainable:")
    print(f"  OLD: {'Frozen' if old_freeze else 'Trainable'} (only sketch encoder learned)")
    print(f"  NEW: {'Frozen' if new_freeze else 'Trainable'} ✅ (full model adaptation)")
    
    # LoRA rank
    old_rank = old_config['model'].lora_rank
    new_rank = new_config['model'].lora_rank
    print(f"\nLoRA Rank:")
    print(f"  OLD: {old_rank}")
    print(f"  NEW: {new_rank} ✅ (2x more capacity)")
    
    # Epochs
    old_epochs = old_config['training'].stage1_epochs
    new_epochs = new_config['training'].stage1_epochs
    print(f"\nTraining Epochs:")
    print(f"  OLD: {old_epochs}")
    print(f"  NEW: {new_epochs} ✅ (2x longer)")
    
    # Perceptual loss
    print(f"\nPerceptual Loss:")
    print(f"  OLD: Not used (only MSE)")
    print(f"  NEW: LPIPS + MSE ✅ (better perceptual quality)")
    
    # Early stopping
    print(f"\nEarly Stopping:")
    print(f"  OLD: Not used")
    print(f"  NEW: Enabled (patience=4 epochs) ✅")
    
    print("\n" + "=" * 80)
    print("EXPECTED IMPROVEMENTS:")
    print("=" * 80)
    
    print("""
Metric          Current    Target     Improvement
─────────────────────────────────────────────────
SSIM            0.25       0.55-0.65  +120-160%
LPIPS           0.75       0.35-0.45  -40-53%
FID             280        50-80      -71-82%
PSNR            8.9 dB     22-26 dB   +147-192%
─────────────────────────────────────────────────

Estimated training time: 10-15 hours on RTX 5090
Checkpoint size: ~2GB per epoch
Total disk space needed: ~40GB (20 epochs)
""")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config_comparison()
    
    print("\n📝 To use this improved configuration:")
    print("   python train_improved.py --config improved")
    print("\n   Or manually:")
    print("   python scripts/training/train.py \\")
    print("      --learning_rate 1e-5 \\")
    print("      --stage1_epochs 20 \\")
    print("      --unfreeze_unet \\")
    print("      --use_perceptual_loss")
