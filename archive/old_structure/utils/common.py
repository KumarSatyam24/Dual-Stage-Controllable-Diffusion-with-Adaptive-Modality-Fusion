"""
Utility Functions for RAGAF-Diffusion

Common helper functions for training, evaluation, and visualization.

Author: RAGAF-Diffusion Research Team
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
from PIL import Image


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get available device.
    
    Args:
        prefer_cuda: Prefer CUDA if available
    
    Returns:
        Device string
    """
    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_image_grid(
    images: List[torch.Tensor],
    path: str,
    nrow: int = 4,
    normalize: bool = True
):
    """
    Save a grid of images.
    
    Args:
        images: List of image tensors (C, H, W)
        path: Output path
        nrow: Number of images per row
        normalize: Whether to normalize to [0, 1]
    """
    from torchvision.utils import make_grid, save_image as tv_save_image
    
    # Stack images
    grid = torch.stack(images, dim=0)
    
    # Create grid
    grid = make_grid(grid, nrow=nrow, normalize=normalize)
    
    # Save
    tv_save_image(grid, path)


def visualize_attention_map(
    attention_map: np.ndarray,
    region_masks: List[np.ndarray],
    text_tokens: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize region-text attention map.
    
    Args:
        attention_map: Attention scores (N, T)
        region_masks: List of region masks
        text_tokens: List of text tokens
        save_path: Optional path to save visualization
    """
    N, T = attention_map.shape
    
    # Create figure
    fig, axes = plt.subplots(1, N + 1, figsize=(3 * (N + 1), 3))
    
    # Plot attention heatmap
    im = axes[0].imshow(attention_map, cmap='viridis', aspect='auto')
    axes[0].set_xlabel("Text Tokens")
    axes[0].set_ylabel("Regions")
    axes[0].set_title("Region-Text Attention")
    
    # Set ticks
    if len(text_tokens) <= 20:
        axes[0].set_xticks(range(len(text_tokens)))
        axes[0].set_xticklabels(text_tokens, rotation=90, fontsize=8)
    
    plt.colorbar(im, ax=axes[0])
    
    # Plot top-k most attended regions for each
    for i in range(min(N, len(axes) - 1)):
        if i < len(region_masks):
            axes[i + 1].imshow(region_masks[i], cmap='gray')
            
            # Get top attended tokens
            top_k = min(3, T)
            top_indices = np.argsort(attention_map[i])[-top_k:][::-1]
            top_tokens = [text_tokens[idx] if idx < len(text_tokens) else f"T{idx}" 
                         for idx in top_indices]
            
            axes[i + 1].set_title(f"R{i}: {', '.join(top_tokens)}", fontsize=9)
            axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_training_summary(
    checkpoint_dir: str,
    output_path: str
):
    """
    Create a summary of training progress from checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        output_path: Path to save summary
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    summary = []
    summary.append("RAGAF-Diffusion Training Summary")
    summary.append("=" * 60)
    
    # Check Stage 1
    stage1_dir = checkpoint_dir / "stage1"
    if stage1_dir.exists():
        summary.append("\nStage 1 Checkpoints:")
        checkpoints = sorted(stage1_dir.glob("*.pt"))
        for ckpt in checkpoints:
            summary.append(f"  - {ckpt.name}")
    
    # Check Stage 2
    stage2_dir = checkpoint_dir / "stage2"
    if stage2_dir.exists():
        summary.append("\nStage 2 Checkpoints:")
        checkpoints = sorted(stage2_dir.glob("*.pt"))
        for ckpt in checkpoints:
            summary.append(f"  - {ckpt.name}")
    
    summary.append("\n" + "=" * 60)
    
    # Save summary
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print('\n'.join(summary))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Tensor (C, H, W) in range [0, 1] or [-1, 1]
    
    Returns:
        PIL Image
    """
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Clamp
    tensor = tensor.clamp(0, 1)
    
    # To numpy
    arr = (tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # To PIL
    return Image.fromarray(arr)


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """
    Convert PIL Image to tensor.
    
    Args:
        image: PIL Image
        normalize: Whether to normalize to [-1, 1]
    
    Returns:
        Tensor (C, H, W)
    """
    arr = np.array(image).astype(np.float32) / 255.0
    
    if len(arr.shape) == 2:
        # Grayscale
        tensor = torch.from_numpy(arr).unsqueeze(0)
    else:
        # RGB
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    
    if normalize:
        tensor = tensor * 2 - 1
    
    return tensor


if __name__ == "__main__":
    # Test utilities
    print("Utility Functions for RAGAF-Diffusion")
    print("=" * 60)
    
    # Test device detection
    device = get_device()
    
    # Test seed setting
    set_seed(42)
    print(f"\nRandom seed set to 42")
    
    # Test attention visualization
    print("\nTesting attention visualization...")
    dummy_attn = np.random.rand(5, 10)
    dummy_masks = [np.random.randint(0, 2, (64, 64)) for _ in range(5)]
    dummy_tokens = [f"token{i}" for i in range(10)]
    
    visualize_attention_map(
        dummy_attn,
        dummy_masks,
        dummy_tokens,
        save_path="test_attention.png"
    )
    print("  Attention map saved to test_attention.png")
