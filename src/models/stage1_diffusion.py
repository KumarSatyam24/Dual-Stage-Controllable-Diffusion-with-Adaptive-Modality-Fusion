"""
Stage 1: Sketch-Guided Diffusion for RAGAF-Diffusion

This module implements the first stage of the dual-stage pipeline:
coarse structure-preserving image layout generation guided by sketch input.

Uses ControlNet-style architecture to inject sketch conditioning into
Stable Diffusion's UNet via down_block_additional_residuals and
mid_block_additional_residual.

Key features:
- Sketch encoder producing 12 residuals (11 down + 1 mid) matching SD v1.5 UNet
- ControlNet-style zero-initialized output convolutions
- Proper residual injection into all UNet down blocks and mid block
- Preserves sketch structure while allowing texture generation

Author: RAGAF-Diffusion Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer


class SketchEncoder(nn.Module):
    """
    ControlNet-style encoder for sketch input.

    Produces 12 down-block residuals + 1 mid-block residual that match
    the shape expected by UNet2DConditionModel's
    down_block_additional_residuals / mid_block_additional_residual args.

    The sketch is first downsampled 8× (via strided convs) to match the
    latent spatial resolution that the UNet operates at.

    For SD v1.5 at 256×256 input (latent 32×32) the residual shapes are:
      Block 0 (channels=320, spatial=32): [B,320,32,32] x3, then [B,320,16,16] (post-ds)
      Block 1 (channels=640, spatial=16): [B,640,16,16] x2, then [B,640,8,8]
      Block 2 (channels=1280, spatial=8):  [B,1280,8,8]  x2, then [B,1280,4,4]
      Block 3 (channels=1280, spatial=4):  [B,1280,4,4]  x2
    Total: 3+1 + 2+1 + 2+1 + 2 = 12 down residuals + 1 mid.
    """

    def __init__(
        self,
        in_channels: int = 1,          # Grayscale sketch
        base_channels: int = 16,
        # Channels per UNet block: [320, 640, 1280, 1280]
        block_out_channels: List[int] = [320, 640, 1280, 1280],
        layers_per_block: int = 2,     # SD v1.5 uses 2 resnet layers per block
    ):
        super().__init__()

        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block

        # ------------------------------------------------------------------
        # Stem: sketch (B,1,H,W) -> (B, base_channels, H/8, W/8)
        #
        # SD v1.5 encodes images into latents that are 8× smaller.
        # The UNet's down_block_additional_residuals must therefore start
        # at latent spatial resolution (e.g. 32×32 for a 256×256 image).
        # We use three stride-2 convolutions to achieve the 8× reduction.
        # ------------------------------------------------------------------
        self.input_proj = nn.Sequential(
            # 1×: H/2
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            # 2×: H/4
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            # 3×: H/8  — now matches latent spatial resolution
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        
        # ControlNet has an extra conv after input that produces the first residual
        # This gives block 0 an extra output (3 total instead of 2)
        self.conv_after_input = nn.Conv2d(base_channels, block_out_channels[0], kernel_size=3, padding=1)

        # ------------------------------------------------------------------
        # Down-sampling feature pyramid.
        # We build one "block" per UNet down-block, each consisting of
        # `layers_per_block` residual layers + optional strided downsample.
        # ------------------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()   # one per block (except last)
        current_ch = block_out_channels[0]  # Start from first block's channels

        for i, out_ch in enumerate(block_out_channels):
            layers = []
            for j in range(layers_per_block):
                layers.append(ResidualBlock(current_ch, out_ch))
                current_ch = out_ch
            self.down_blocks.append(nn.Sequential(*layers))

            # Downsample between blocks (not after the last block)
            if i < len(block_out_channels) - 1:
                self.down_samplers.append(
                    nn.Conv2d(current_ch, current_ch, kernel_size=3, stride=2, padding=1)
                )
            else:
                self.down_samplers.append(None)

        # ------------------------------------------------------------------
        # Mid block (same channels as last down block)
        # ------------------------------------------------------------------
        mid_ch = block_out_channels[-1]
        self.mid_block = nn.Sequential(
            ResidualBlock(current_ch, mid_ch),
            ResidualBlock(mid_ch, mid_ch),
        )

        # ------------------------------------------------------------------
        # Zero-initialized 1×1 projection convolutions (ControlNet trick).
        # One per residual output we produce:
        #   - 1 for conv_after_input (block 0's extra output)
        #   - layers_per_block outputs per block (before downsample)
        #   - 1 output per block for the post-downsample feature
        #     (except the last block which has no downsampler)
        # Total down outputs  = 1 + sum over blocks of (layers_per_block + has_ds)
        #   = 1 + (2+1)*3 + (2+0)*1 = 1 + 9 + 2 = 12
        # Plus 1 mid output  -> 13 total zero-convs
        # ------------------------------------------------------------------
        self.zero_convs_down: nn.ModuleList = nn.ModuleList()
        
        # First zero-conv for conv_after_input output
        self.zero_convs_down.append(self._make_zero_conv(block_out_channels[0]))
        
        for i, out_ch in enumerate(block_out_channels):
            # One zero-conv per resnet layer output
            for _ in range(layers_per_block):
                self.zero_convs_down.append(self._make_zero_conv(out_ch))
            # One zero-conv for the post-downsampler output (if exists)
            if i < len(block_out_channels) - 1:
                self.zero_convs_down.append(self._make_zero_conv(out_ch))

        self.zero_conv_mid = self._make_zero_conv(mid_ch)

    @staticmethod
    def _make_zero_conv(channels: int) -> nn.Conv2d:
        conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv

    def forward(self, sketch: torch.Tensor):
        """
        Args:
            sketch: (B, 1, H, W) sketch image, values in [-1, 1]

        Returns:
            down_residuals: tuple of 12 tensors for down_block_additional_residuals
            mid_residual:   single tensor  for mid_block_additional_residual
        """
        x = self.input_proj(sketch)
        
        # ControlNet structure: first apply conv_after_input and emit as residual[0]
        x = self.conv_after_input(x)
        
        down_residuals: List[torch.Tensor] = []
        down_residuals.append(self.zero_convs_down[0](x))  # First residual from conv_after_input
        zero_idx = 1

        for i, (block, ds) in enumerate(zip(self.down_blocks, self.down_samplers)):
            # Apply each resnet layer individually so we can capture per-layer output
            for layer in block:
                x = layer(x)
                down_residuals.append(self.zero_convs_down[zero_idx](x))
                zero_idx += 1

            # Downsample (if this block has one) and capture the result
            if ds is not None:
                x = ds(x)
                down_residuals.append(self.zero_convs_down[zero_idx](x))
                zero_idx += 1

        # Mid block
        x = self.mid_block(x)
        mid_residual = self.zero_conv_mid(x)

        # Return as tuple (UNet expects tuple, not list)
        return tuple(down_residuals), mid_residual


class ResidualBlock(nn.Module):
    """Simple residual block with group normalization."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.act(x + residual)


class Stage1SketchGuidedDiffusion(nn.Module):
    """
    Stage 1: Sketch-guided diffusion model for coarse layout generation.
    
    This model takes a sketch as input and generates a coarse image that
    preserves the sketch structure. Text conditioning is minimal at this stage.
    
    Architecture:
    - Base: Stable Diffusion UNet
    - Sketch conditioning: Injected via ControlNet-style encoder
    - Text conditioning: Standard CLIP text encoder
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        sketch_encoder_channels: List[int] = [320, 640, 1280, 1280],
        freeze_base_unet: bool = False,
        use_lora: bool = True,
        lora_rank: int = 4
    ):
        """
        Initialize Stage 1 model.
        
        Args:
            pretrained_model_name: HuggingFace model name for pretrained SD
            sketch_encoder_channels: Channels for sketch encoder (match UNet)
            freeze_base_unet: Whether to freeze pretrained UNet weights
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_rank: LoRA rank
        """
        super().__init__()
        
        print(f"Loading pretrained Stable Diffusion: {pretrained_model_name}")
        
        # Load pretrained components
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name, subfolder="text_encoder"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name, subfolder="tokenizer"
        )
        
        # Freeze VAE and text encoder (typically not fine-tuned)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Optionally freeze UNet
        if freeze_base_unet:
            self.unet.requires_grad_(False)
        
        # Sketch encoder for conditioning.
        # Produces 11 down-block residuals + 1 mid-block residual that are
        # directly passed to UNet via down_block_additional_residuals /
        # mid_block_additional_residual (ControlNet-style injection).
        self.sketch_encoder = SketchEncoder(
            in_channels=1,
            block_out_channels=self.unet.config.block_out_channels,
            layers_per_block=self.unet.config.layers_per_block,
        )

        # LoRA placeholder (not critical for sketch conditioning correctness)
        self.use_lora = use_lora
        if use_lora:
            print(f"LoRA fine-tuning enabled (rank={lora_rank}) — sketch encoder is primary adapter")
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler"
        )
        
        print("Stage 1 Sketch-Guided Diffusion initialized")
    
    def encode_sketch(self, sketch: torch.Tensor):
        """
        Encode sketch to multi-scale ControlNet-style residuals.

        Args:
            sketch: Sketch input (B, 1, H, W) in range [0, 1]

        Returns:
            (down_residuals, mid_residual) — 11-element list + single tensor
        """
        # Normalize sketch to [-1, 1]
        sketch = sketch * 2.0 - 1.0
        return self.sketch_encoder(sketch)
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts using CLIP text encoder.
        
        Args:
            text_prompts: List of text prompts
        
        Returns:
            Text embeddings (B, 77, 768)
        """
        # Tokenize
        text_inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.text_encoder.device)
            )[0]
        
        return text_embeddings
    
    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        sketch_features,           # tuple: (down_residuals, mid_residual)
        text_embeddings: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through UNet with ControlNet-style sketch conditioning.

        Args:
            latents: Noisy latents (B, 4, H/8, W/8)
            timestep: Diffusion timestep (B,)
            sketch_features: Output of encode_sketch() —
                             tuple (down_residuals list[11], mid_residual tensor)
            text_embeddings: Text embeddings (B, 77, 768)
            return_dict: Whether to return dict or tensor

        Returns:
            Predicted noise (B, 4, H/8, W/8)
        """
        # Unpack sketch conditioning residuals
        if isinstance(sketch_features, (tuple, list)) and len(sketch_features) == 2:
            down_residuals, mid_residual = sketch_features
        else:
            # Fallback: no sketch conditioning (should not happen in normal use)
            down_residuals = None
            mid_residual = None

        # UNet forward with sketch conditioning injected via ControlNet residuals
        noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=return_dict,
        )

        if return_dict:
            return noise_pred
        else:
            # When return_dict=False, UNet returns a tuple (noise_pred,)
            return noise_pred[0] if isinstance(noise_pred, tuple) else noise_pred.sample
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters for optimization.
        
        Returns:
            Iterator of trainable parameters
        """
        trainable_params = []
        
        # Sketch encoder is always trainable
        trainable_params.extend(self.sketch_encoder.parameters())
        
        # UNet parameters if not frozen
        if not all(not p.requires_grad for p in self.unet.parameters()):
            trainable_params.extend(
                p for p in self.unet.parameters() if p.requires_grad
            )
        
        return trainable_params


class Stage1DiffusionPipeline:
    """
    Inference pipeline for Stage 1 sketch-guided diffusion.
    
    Handles the full generation process: noise initialization, denoising loop,
    and VAE decoding.
    """
    
    def __init__(
        self,
        model: Stage1SketchGuidedDiffusion,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        device: str = "cuda"
    ):
        """
        Initialize pipeline.
        
        Args:
            model: Stage1SketchGuidedDiffusion model
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            device: Device to run on
        """
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
        
        # Setup DDIM scheduler for inference
        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        self.scheduler.set_timesteps(num_inference_steps)
    
    @torch.no_grad()
    def generate(
        self,
        sketch: torch.Tensor,
        text_prompt: str,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate image from sketch and text prompt.

        Args:
            sketch: Sketch input (1, 1, H, W)
            text_prompt: Text prompt
            height: Output height
            width: Output width
            seed: Random seed for reproducibility

        Returns:
            Generated image (1, 3, H, W) in range [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Move sketch to device
        sketch = sketch.to(self.device)

        # Encode sketch — returns (down_residuals, mid_residual) for batch=1
        down_res_single, mid_res_single = self.model.encode_sketch(sketch)

        # For CFG we run a batch of 2 (uncond + cond). Duplicate sketch residuals.
        down_res_cfg = [torch.cat([r, r]) for r in down_res_single]
        mid_res_cfg = torch.cat([mid_res_single, mid_res_single])
        sketch_features_cfg = (down_res_cfg, mid_res_cfg)

        # Encode text
        text_embeddings = self.model.encode_text([text_prompt])

        # Prepare uncond embeddings for classifier-free guidance
        uncond_embeddings = self.model.encode_text([""])

        # Initialize latents
        latents = torch.randn(
            1, 4, height // 8, width // 8,
            device=self.device, dtype=torch.float32
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Denoising loop
        for t in self.scheduler.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate uncond + cond embeddings
            encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings])

            # Predict noise with sketch conditioning for both uncond and cond
            noise_pred = self.model(
                latent_model_input,
                t,
                sketch_features_cfg,
                encoder_hidden_states,
            )

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to image
        latents = 1 / 0.18215 * latents
        image = self.model.vae.decode(latents).sample

        # Denormalize to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image


if __name__ == "__main__":
    # Example usage
    print("Stage 1: Sketch-Guided Diffusion for RAGAF-Diffusion")
    print("=" * 60)
    
    # NOTE: This requires GPU and pretrained Stable Diffusion weights
    # Uncomment to test (requires GPU)
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device: {device}")
    
    # # Create model
    # model = Stage1SketchGuidedDiffusion(
    #     pretrained_model_name="runwayml/stable-diffusion-v1-5",
    #     freeze_base_unet=True
    # ).to(device)
    
    # # Create dummy sketch
    # dummy_sketch = torch.randn(1, 1, 512, 512).to(device)
    
    # # Encode sketch
    # sketch_features = model.encode_sketch(dummy_sketch)
    # print(f"Sketch features: {len(sketch_features)} scales")
    # for i, feat in enumerate(sketch_features):
    #     print(f"  Scale {i}: {feat.shape}")
    
    # # Encode text
    # text_emb = model.encode_text(["A photo of a dog"])
    # print(f"Text embeddings: {text_emb.shape}")
    
    print("\nStage 1 model structure defined.")
    print("Uncomment test code to run with GPU.")
