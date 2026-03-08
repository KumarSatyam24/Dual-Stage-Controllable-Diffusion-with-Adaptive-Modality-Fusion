"""
Training Configuration for RAGAF-Diffusion

Default hyperparameters and settings for training.
Modify these values or override via command-line arguments.

Author: RAGAF-Diffusion Research Team
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Base model
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    
    # Stage 1 (Sketch-guided)
    sketch_encoder_channels: List[int] = field(default_factory=lambda: [320, 640, 1280, 1280])
    freeze_stage1_unet: bool = True  # Freeze UNet, only train sketch encoder
    
    # Stage 2 (Semantic refinement)
    node_feature_dim: int = 6  # Spatial features from region extraction
    text_dim: int = 768  # CLIP text embedding dimension
    hidden_dim: int = 512
    num_graph_layers: int = 2
    num_attention_heads: int = 8
    
    # RAGAF attention
    ragaf_dropout: float = 0.1
    
    # Adaptive fusion
    fusion_method: str = "learned"  # "learned", "heuristic", "hybrid"
    use_region_adaptive_fusion: bool = True
    
    # LoRA (for efficient fine-tuning)
    use_lora: bool = True
    lora_rank: int = 4
    lora_alpha: int = 4


@dataclass
class DataConfig:
    """Data loading configuration."""
    
    # Datasets
    dataset_name: str = "sketchy"  # "sketchy", "coco", "both"
    sketchy_root: str = "/workspace/sketchy"  # vast.ai - Google Drive sync destination
    coco_root: str = "/workspace/coco"  # vast.ai container disk
    
    # Data processing
    image_size: int = 256  # Reduced from 512 → 4x faster, less VRAM, can upscale later
    sketch_method: str = "canny"  # For COCO auto-sketch generation
    
    # Region extraction
    min_region_area: int = 100
    max_num_regions: int = 20  # Reduced from 50 → faster graph processing
    graph_type: str = "adjacency"  # Fastest graph type (was "hybrid")
    
    # Data loading
    batch_size: int = 4  # Safe for RTX 5090 32GB (~31GB VRAM, ~1.6GB headroom)
    num_workers: int = 4  # More parallel data loading
    pin_memory: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    
    # Cache
    cache_sketches: bool = True  # Cache auto-generated sketches (COCO only)
    preload_graphs: bool = False  # Preload region graphs (memory intensive)


@dataclass
class TrainingConfig:
    """Training procedure configuration."""
    
    # Training stages
    train_stage: str = "both"  # "stage1", "stage2", "both"
    stage1_epochs: int = 10  # Increased from 2 - more epochs = better quality
    stage2_epochs: int = 5   # Increased from 2
    
    # Optimization
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 1e-2
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "constant", "linear", "cosine"
    lr_warmup_steps: int = 500
    
    # Gradient
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1  # No accumulation needed with batch_size=4
    
    # Mixed precision
    mixed_precision: str = "bf16"  # bf16 for RTX 5090 - saves ~50% VRAM
    
    # Diffusion
    num_train_timesteps: int = 1000
    noise_scheduler: str = "ddpm"  # "ddpm", "ddim"
    
    # Checkpointing
    save_every_n_epochs: int = 2  # Save every 2 epochs → epoch_2, epoch_4, epoch_6, epoch_8, final
    checkpoint_dir: str = "/root/checkpoints"  # 114 GB free on root filesystem
    resume_from_checkpoint: Optional[str] = None
    resume_from_epoch: int = 0  # Set to N to resume training from epoch N

    # HuggingFace Hub auto-upload (never lose checkpoints again)
    push_to_hub: bool = True
    hub_repo_id: str = "DrRORAL/ragaf-diffusion-checkpoints"  # Will be created if not exists
    hub_token: Optional[str] = None  # Uses HF_TOKEN env var if None
    
    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = False  # Disabled by default (install wandb if needed)
    wandb_project: str = "ragaf-diffusion"
    wandb_run_name: Optional[str] = None
    
    # Validation
    validate_every_n_epochs: int = 2
    num_validation_samples: int = 4
    
    # Device
    device: str = "cuda"  # "cuda", "cpu"
    
    # RunPod / Cloud training
    use_runpod: bool = False
    runpod_volume_path: str = "/workspace"


@dataclass
class InferenceConfig:
    """Inference configuration."""
    
    # Model checkpoint
    stage1_checkpoint: str = "./checkpoints/stage1_final.pt"
    stage2_checkpoint: str = "./checkpoints/stage2_final.pt"
    
    # Generation
    num_inference_steps: int = 50  # Stage 1
    num_refinement_steps: int = 30  # Stage 2
    guidance_scale: float = 7.5
    refinement_strength: float = 0.5
    
    # Output
    output_dir: str = "./outputs"
    save_intermediates: bool = True  # Save Stage 1 output
    
    # Visualization
    visualize_regions: bool = True
    visualize_attention: bool = True
    
    # Device
    device: str = "cuda"


# Default configurations
def get_default_config():
    """Get default configuration."""
    return {
        "model": ModelConfig(),
        "data": DataConfig(),
        "training": TrainingConfig(),
        "inference": InferenceConfig()
    }


def save_config(config_dict, path: str):
    """Save configuration to YAML file."""
    import yaml
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def load_config(path: str):
    """Load configuration from YAML file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Generate default config file
    print("Generating default configuration...")
    
    config = get_default_config()
    
    # Convert dataclasses to dict
    from dataclasses import asdict
    config_dict = {
        "model": asdict(config["model"]),
        "data": asdict(config["data"]),
        "training": asdict(config["training"]),
        "inference": asdict(config["inference"])
    }
    
    save_config(config_dict, "default_config.yaml")
    print("Saved default configuration to default_config.yaml")
    
    # Print summary
    print("\nConfiguration Summary:")
    print(f"  Model: {config['model'].pretrained_model_name}")
    print(f"  Dataset: {config['data'].dataset_name}")
    print(f"  Image size: {config['data'].image_size}")
    print(f"  Batch size: {config['data'].batch_size}")
    print(f"  Learning rate: {config['training'].learning_rate}")
    print(f"  Training stage: {config['training'].train_stage}")
    print(f"  Mixed precision: {config['training'].mixed_precision}")
