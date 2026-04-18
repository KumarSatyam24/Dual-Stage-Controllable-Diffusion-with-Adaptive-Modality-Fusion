"""
================================================================================
THESIS APPENDIX — SAMPLE SOURCE CODE
================================================================================

Project : Dual-Stage Controllable Diffusion with Adaptive Modality Fusion
         (RAGAF-Diffusion)

Abstract: A dual-stage diffusion framework that combines ControlNet-style
          sketch conditioning with Region-Adaptive Graph-Attention Fusion
          (RAGAF) to generate semantically rich images that faithfully
          preserve the structure of an input sketch while obeying a free-form
          text prompt.

Pipeline Overview:
  1. Stage 1 – Sketch-Guided Coarse Generation
     A ControlNet-style Sketch Encoder injects multi-scale sketch residuals
     into a frozen Stable Diffusion UNet, producing a structurally-aligned
     coarse image.

  2. Stage 2 – Semantic Refinement with RAGAF
     The coarse latent is concatenated with a noisy latent and refined by a
     second UNet conditioned on:
       • RAGAF Attention  – graph attention over sketch regions + cross-
                            attention to CLIP text tokens
       • Adaptive Fusion  – timestep-aware, region-specific weighting that
                            gradually shifts guidance from sketch structure
                            (early timesteps) to text semantics (late
                            timesteps)

File structure of this appendix:
  §1   Dependencies & imports
  §2   Region graph data structure
  §3   Stage 1 – ControlNet-style Sketch Encoder & Diffusion Model
  §4   RAGAF Attention (graph attention + region-text cross-attention)
  §5   Adaptive Modality Fusion
  §6   Stage 2 – Semantic Refinement Model
  §7   Dataset loader (Sketchy)
  §8   Trainer  (Stage 1 & Stage 2 training steps)
  §9   Inference pipeline
  §10  Utility helpers

Author: RAGAF-Diffusion Research Team
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# §1  DEPENDENCIES & IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator


# ─────────────────────────────────────────────────────────────────────────────
# §2  REGION GRAPH DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class RegionGraph:
    """
    Lightweight container for a sketch-region graph.

    Each node represents one segmented sketch region.
    Node features encode spatial statistics (centroid x/y, area, aspect ratio,
    mean edge density, compactness) – all normalised to [0, 1].

    Attributes:
        num_nodes       : Number of regions (N).
        node_features   : (N, F) float tensor of per-region features.
        edge_index      : (2, E) long tensor – [source_nodes; target_nodes].
        edge_weights    : (E,)  float tensor of adjacency weights (optional).
        region_masks    : List of (H, W) binary numpy arrays, one per region.
        adjacency_matrix: (N, N) float tensor (optional, for convenience).
    """

    def __init__(
        self,
        num_nodes: int,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor],
        region_masks: List[np.ndarray],
        adjacency_matrix: Optional[torch.Tensor],
    ):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.region_masks = region_masks
        self.adjacency_matrix = adjacency_matrix

    def __repr__(self) -> str:
        return (
            f"RegionGraph(nodes={self.num_nodes}, "
            f"edges={self.edge_index.shape[1] if self.edge_index is not None else 0})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3  STAGE 1 – CONTROLNET-STYLE SKETCH ENCODER & DIFFUSION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Basic residual convolution block with Group Normalisation and SiLU.

    Used inside the SketchEncoder to build the multi-scale feature pyramid.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act   = nn.SiLU()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class SketchEncoder(nn.Module):
    """
    ControlNet-style encoder that converts a grayscale sketch into the 12 down-
    block residuals + 1 mid-block residual expected by SD v1.5's UNet.

    Architecture:
      • Stem  : 3 × stride-2 convolutions → 8× spatial reduction (matches the
                latent resolution produced by the VAE encoder).
      • Pyramid: 4 down-blocks mirroring the UNet's block_out_channels
                 [320, 640, 1280, 1280], each with 2 ResidualBlocks + optional
                 stride-2 downsample.
      • Mid   : 2 ResidualBlocks at the lowest resolution.
      • Zero-convs: Every residual output is passed through a zero-initialised
                    1×1 convolution (the ControlNet trick) so the model starts
                    as an identity at the beginning of training.

    Total residuals: 1 (stem) + 3×(2+1) + 1×2 + 1 (mid) = 12 down + 1 mid.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        block_out_channels: List[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.block_out_channels = list(block_out_channels)
        self.layers_per_block   = layers_per_block

        # Stem: sketch → latent spatial resolution (÷8)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels,   base_channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1), nn.SiLU(),
        )
        # Additional conv to bring channels up to block_out_channels[0]
        self.conv_after_input = nn.Conv2d(base_channels, block_out_channels[0], 3, padding=1)

        # Down-sampling feature pyramid
        self.down_blocks  = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        current_ch = block_out_channels[0]

        for i, out_ch in enumerate(block_out_channels):
            layers = []
            for _ in range(layers_per_block):
                layers.append(ResidualBlock(current_ch, out_ch))
                current_ch = out_ch
            self.down_blocks.append(nn.Sequential(*layers))
            if i < len(block_out_channels) - 1:
                self.down_samplers.append(nn.Conv2d(current_ch, current_ch, 3, stride=2, padding=1))
            else:
                self.down_samplers.append(None)

        # Mid block
        mid_ch = block_out_channels[-1]
        self.mid_block = nn.Sequential(
            ResidualBlock(current_ch, mid_ch),
            ResidualBlock(mid_ch,     mid_ch),
        )

        # Zero-initialised 1×1 projection convolutions
        self.zero_convs_down = nn.ModuleList()
        self.zero_convs_down.append(self._make_zero_conv(block_out_channels[0]))  # stem output
        for i, out_ch in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                self.zero_convs_down.append(self._make_zero_conv(out_ch))
            if i < len(block_out_channels) - 1:
                self.zero_convs_down.append(self._make_zero_conv(out_ch))
        self.zero_conv_mid = self._make_zero_conv(mid_ch)

    @staticmethod
    def _make_zero_conv(channels: int) -> nn.Conv2d:
        conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv

    def forward(self, sketch: torch.Tensor) -> Tuple[Tuple, torch.Tensor]:
        """
        Args:
            sketch: (B, 1, H, W) in [-1, 1].

        Returns:
            down_residuals : tuple of 12 tensors for UNet down_block_additional_residuals.
            mid_residual   : single tensor  for UNet mid_block_additional_residual.
        """
        x = self.conv_after_input(self.input_proj(sketch))

        down_residuals: List[torch.Tensor] = [self.zero_convs_down[0](x)]
        zero_idx = 1

        for block, ds in zip(self.down_blocks, self.down_samplers):
            for layer in block:
                x = layer(x)
                down_residuals.append(self.zero_convs_down[zero_idx](x))
                zero_idx += 1
            if ds is not None:
                x = ds(x)
                down_residuals.append(self.zero_convs_down[zero_idx](x))
                zero_idx += 1

        mid_residual = self.zero_conv_mid(self.mid_block(x))
        return tuple(down_residuals), mid_residual


class Stage1SketchGuidedDiffusion(nn.Module):
    """
    Stage 1: Sketch-conditioned diffusion model for coarse layout generation.

    Wraps a pretrained Stable Diffusion v1.5 UNet and injects sketch guidance
    via ControlNet-style residual connections produced by ``SketchEncoder``.
    The VAE and CLIP text encoder are frozen; only the sketch encoder (and
    optionally parts of the UNet via LoRA) are trained.
    """

    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        sketch_encoder_channels: List[int] = (320, 640, 1280, 1280),
        freeze_base_unet: bool = False,
        use_lora: bool = True,
        lora_rank: int = 4,
    ):
        super().__init__()

        # Load pretrained Stable Diffusion components
        self.vae          = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")
        self.unet         = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")
        self.tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")

        # Freeze components that are not trained
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if freeze_base_unet:
            self.unet.requires_grad_(False)

        # Sketch encoder (ControlNet-style)
        self.sketch_encoder = SketchEncoder(
            in_channels=1,
            block_out_channels=list(self.unet.config.block_out_channels),
            layers_per_block=self.unet.config.layers_per_block,
        )

        # Noise scheduler for training
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")
        self.use_lora = use_lora

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_sketch(self, sketch: torch.Tensor):
        """Normalise sketch to [-1, 1] and run through SketchEncoder."""
        return self.sketch_encoder(sketch * 2.0 - 1.0)

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Tokenise and encode text with frozen CLIP. Returns (B, 77, 768)."""
        inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return self.text_encoder(inputs.input_ids.to(self.text_encoder.device))[0]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        sketch_features: Tuple,
        text_embeddings: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Predict noise for a single denoising step.

        Args:
            latents          : Noisy latents (B, 4, H/8, W/8).
            timestep         : Diffusion timestep (B,).
            sketch_features  : Output of ``encode_sketch`` –
                               (down_residuals tuple, mid_residual tensor).
            text_embeddings  : CLIP embeddings (B, 77, 768).
            return_dict      : Return UNetOutput dict when True.

        Returns:
            Predicted noise tensor (B, 4, H/8, W/8).
        """
        down_residuals, mid_residual = sketch_features
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
        return noise_pred[0] if isinstance(noise_pred, tuple) else noise_pred.sample

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return all parameters that should receive gradient updates."""
        params = list(self.sketch_encoder.parameters())
        params += [p for p in self.unet.parameters() if p.requires_grad]
        return params


# ─────────────────────────────────────────────────────────────────────────────
# §4  RAGAF ATTENTION
#     Region-Adaptive Graph-Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────

class RegionGraphAttention(nn.Module):
    """
    Multi-head graph attention over sketch regions.

    For each edge (i → j) in the region adjacency graph we compute a
    scaled dot-product attention score between the target node's query and
    the source node's key, weighted by the edge's learned projection.
    Values are then scattered-summed back to target nodes.

    This is equivalent to a sparse variant of standard multi-head attention
    restricted to the graph neighbourhood of each node.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.q_proj   = nn.Linear(node_feature_dim, hidden_dim)
        self.k_proj   = nn.Linear(node_feature_dim, hidden_dim)
        self.v_proj   = nn.Linear(node_feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(1, num_heads)  # project scalar edge weight → per-head bias
        self.dropout  = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter_softmax(
        scores: torch.Tensor,   # (E, H)
        indices: torch.Tensor,  # (E,)  target node indices
        num_nodes: int,
    ) -> torch.Tensor:
        """Numerically-stable softmax scattered over target nodes."""
        max_scores = torch.full((num_nodes, scores.shape[1]), -1e9, device=scores.device)
        max_scores.scatter_reduce_(0, indices.unsqueeze(-1).expand_as(scores), scores, reduce="amax")
        exp_scores = torch.exp(scores - max_scores[indices])
        sum_exp = torch.zeros(num_nodes, scores.shape[1], device=scores.device)
        sum_exp.index_add_(0, indices, exp_scores)
        return exp_scores / (sum_exp[indices] + 1e-8)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,   # (N, D)
        edge_index: torch.Tensor,       # (2, E)
        edge_weights: Optional[torch.Tensor] = None,  # (E,)
    ) -> torch.Tensor:                  # (N, hidden_dim)

        N = node_features.shape[0]
        if edge_index.shape[1] == 0:
            return self.out_proj(node_features)

        Q = self.q_proj(node_features).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(node_features).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(node_features).view(N, self.num_heads, self.head_dim)

        src, tgt = edge_index[0], edge_index[1]
        attn = (Q[tgt] * K[src]).sum(-1) / self.scale  # (E, H)

        if edge_weights is not None and edge_weights.numel() == src.shape[0]:
            attn = attn + self.edge_proj(edge_weights.view(-1, 1))

        attn = self._scatter_softmax(attn, tgt, N)
        attn = self.dropout(attn)

        out = torch.zeros(N, self.num_heads, self.head_dim, device=node_features.device)
        out.index_add_(0, tgt, attn.unsqueeze(-1) * V[src])
        return self.out_proj(out.view(N, self.hidden_dim))


class RegionTextCrossAttention(nn.Module):
    """
    Cross-attention from sketch regions (queries) to CLIP text tokens (keys/values).

    Each region attends over all T text tokens and absorbs the weighted
    text embedding, making region features semantically aware.
    Returns both the updated region features and an (N, T) attention map
    for visualisation.
    """

    def __init__(
        self,
        region_dim: int,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.region_proj = nn.Linear(region_dim,  hidden_dim)
        self.text_proj   = nn.Linear(text_dim,    hidden_dim)
        self.q_proj      = nn.Linear(hidden_dim,  hidden_dim)
        self.k_proj      = nn.Linear(hidden_dim,  hidden_dim)
        self.v_proj      = nn.Linear(hidden_dim,  hidden_dim)
        self.out_proj    = nn.Linear(hidden_dim,  region_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        region_features: torch.Tensor,          # (N, region_dim)
        text_embeddings: torch.Tensor,           # (T, text_dim)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:     # (N, region_dim), (N, T)

        N = region_features.shape[0]
        T = text_embeddings.shape[0]

        Q = self.q_proj(self.region_proj(region_features))  # (N, H_dim)
        K = self.k_proj(self.text_proj(text_embeddings))     # (T, H_dim)
        V = self.v_proj(self.text_proj(text_embeddings))     # (T, H_dim)

        # Reshape → (heads, N/T, head_dim)
        def rsh(t, seq): return t.view(seq, self.num_heads, self.head_dim).transpose(0, 1)
        Q, K, V = rsh(Q, N), rsh(K, T), rsh(V, T)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (H, N, T)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[None, None], float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))   # (H, N, T)
        out  = torch.matmul(attn, V)                      # (H, N, head_dim)
        out  = out.transpose(0, 1).contiguous().view(N, self.hidden_dim)
        return self.out_proj(out), attn.mean(0)            # (N, region_dim), (N, T)


class RAGAFAttentionModule(nn.Module):
    """
    RAGAF: Region-Adaptive Graph-Attention Fusion module.

    Processes a RegionGraph through:
      1. Node embedding projection.
      2. L stacked graph-attention layers (message passing over the adjacency
         graph) with residual connections and LayerNorm.
      3. Region→text cross-attention to pull in text semantics.
      4. Final linear projection.

    The output is a per-region feature vector that encodes both the spatial
    structure of the sketch and the semantics of the text prompt.
    """

    def __init__(
        self,
        node_feature_dim: int = 6,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_graph_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        self.graph_layers = nn.ModuleList([
            RegionGraphAttention(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_graph_layers)
        ])
        self.graph_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_graph_layers)])

        self.cross_attention   = RegionTextCrossAttention(hidden_dim, text_dim, hidden_dim, num_heads, dropout)
        self.cross_attn_norm   = nn.LayerNorm(hidden_dim)
        self.output_proj       = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        region_graph: RegionGraph,
        text_embeddings: torch.Tensor,   # (T, text_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_features : (N, hidden_dim) – text-enriched region features.
            attn_map      : (N, T)          – region-to-token attention weights.
        """
        node_features = self.node_embedding(region_graph.node_features)

        # Ensure edge_weights are 1-D and on the correct device
        ew = region_graph.edge_weights
        if ew is not None:
            ew = ew.to(node_features.device).view(-1)

        # Stacked graph-attention with residual connections
        for layer, norm in zip(self.graph_layers, self.graph_norms):
            node_features = norm(
                node_features + layer(node_features, region_graph.edge_index, ew)
            )

        # Region–text cross-attention
        cross_out, attn_map = self.cross_attention(node_features, text_embeddings)
        node_features = self.cross_attn_norm(node_features + cross_out)

        return self.output_proj(node_features), attn_map


# ─────────────────────────────────────────────────────────────────────────────
# §5  ADAPTIVE MODALITY FUSION
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveFusionWeights(nn.Module):
    """
    Compute per-region, per-timestep fusion weights (w_sketch, w_text).

    Motivation:  During early denoising steps (high noise / high t) the model
    should prioritise the sketch for structural guidance.  As t decreases the
    text should gradually take over to fill in colours and fine details.

    Three strategies are supported:
      • "heuristic" : Linear schedule baked in – no learnable parameters.
      • "learned"   : Small MLP conditioned on a sinusoidal timestep embedding
                      and (optionally) the current region features.
      • "hybrid"    : 50/50 blend of the above two.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        region_feature_dim: int = 512,
        fusion_method: str = "learned",
        use_region_adaptive: bool = True,
    ):
        super().__init__()
        self.num_timesteps      = num_timesteps
        self.fusion_method      = fusion_method
        self.use_region_adaptive = use_region_adaptive
        self.time_embed_dim     = 128

        if fusion_method in ("learned", "hybrid"):
            in_dim = self.time_embed_dim + (region_feature_dim if use_region_adaptive else 0)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(in_dim, 256), nn.SiLU(),
                nn.Linear(256,    128), nn.SiLU(),
                nn.Linear(128,      2),             # [w_sketch, w_text]
            )

    # ------------------------------------------------------------------

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding (B,) → (B, time_embed_dim)."""
        half = self.time_embed_dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=t.device) * -freq)
        emb  = t[:, None].float() * freq[None]
        emb  = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return F.pad(emb, (0, self.time_embed_dim % 2))

    # ------------------------------------------------------------------

    def forward(
        self,
        timestep: torch.Tensor,
        region_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            timestep        : (B,) or scalar.
            region_features : (B, N, D) or (N, D), optional.

        Returns:
            sketch_weight, text_weight  – same shape as inputs broadcast over D.
        """
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep])
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        B = timestep.shape[0]

        if self.fusion_method == "heuristic":
            # Linear schedule: sketch dominant early, text dominant late
            t_norm        = timestep.float() / self.num_timesteps   # ∈ [0, 1]
            sketch_weight = 0.3 + 0.6 * t_norm
            text_weight   = 0.7 - 0.6 * t_norm
            if region_features is not None and self.use_region_adaptive:
                N = region_features.shape[-2]
                sketch_weight = sketch_weight.unsqueeze(-1).expand(B, N)
                text_weight   = text_weight.unsqueeze(-1).expand(B, N)
            return sketch_weight, text_weight

        # Learned / hybrid path
        time_emb = self._timestep_embedding(timestep)   # (B, time_embed_dim)

        if self.use_region_adaptive and region_features is not None:
            if region_features.dim() == 2:
                region_features = region_features.unsqueeze(0)
            N    = region_features.shape[1]
            t_ex = time_emb.unsqueeze(1).expand(-1, N, -1)          # (B, N, T_emb)
            inp  = torch.cat([t_ex, region_features], dim=-1)        # (B, N, T_emb+D)
            w    = F.softmax(self.fusion_mlp(inp), dim=-1)           # (B, N, 2)
            sw, tw = w[..., 0], w[..., 1]
        else:
            w    = F.softmax(self.fusion_mlp(time_emb), dim=-1)      # (B, 2)
            sw, tw = w[:, 0], w[:, 1]

        if self.fusion_method == "hybrid":
            # Blend with heuristic
            saved = self.fusion_method
            self.fusion_method = "heuristic"
            hsw, htw = self.forward(timestep, region_features)
            self.fusion_method = saved
            sw = 0.5 * sw + 0.5 * hsw
            tw = 0.5 * tw + 0.5 * htw

        return sw, tw


class AdaptiveModalityFusion(nn.Module):
    """
    Weighted combination of sketch-derived and text-derived region features.

    Given:
      • sketch_features : structural embedding of each region  (B, N, D)
      • text_features   : semantics-enriched embedding          (B, N, D)
      • timestep        : current denoising step               (B,)

    The module uses ``AdaptiveFusionWeights`` to compute (w_sketch, w_text) and
    returns:
        fused = w_sketch · sketch_features + w_text · text_features

    followed by a learnable MLP transform and LayerNorm.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_timesteps: int = 1000,
        fusion_method: str = "learned",
        use_region_adaptive: bool = True,
    ):
        super().__init__()
        self.fusion_weights  = AdaptiveFusionWeights(
            num_timesteps, feature_dim, fusion_method, use_region_adaptive
        )
        self.fusion_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        sketch_features: torch.Tensor,
        text_features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            fused        : (B, N, D) or (N, D) blended feature tensor.
            fusion_info  : dict with 'sketch_weight' and 'text_weight' for logging.
        """
        sw, tw = self.fusion_weights(timestep, sketch_features)
        sw = sw.unsqueeze(-1)
        tw = tw.unsqueeze(-1)
        fused = sw * sketch_features + tw * text_features
        fused = self.norm(self.fusion_transform(fused))
        return fused, {"sketch_weight": sw.squeeze(-1), "text_weight": tw.squeeze(-1)}


# ─────────────────────────────────────────────────────────────────────────────
# §6  STAGE 2 – SEMANTIC REFINEMENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Stage2SemanticRefinement(nn.Module):
    """
    Stage 2: RAGAF-conditioned semantic refinement.

    The Stage-1 coarse latent is concatenated to the noisy latent along the
    channel dimension (8 channels total), giving the UNet a direct reference
    to the structural output of Stage 1.  On top of this, RAGAF attention
    and Adaptive Modality Fusion inject region-specific, text-enriched
    modulation into the latent before it enters the UNet.

    Training objective: standard diffusion noise prediction (MSE) augmented
    with an L1 identity-preservation loss between the predicted clean latent
    and the Stage-1 latent.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        node_feature_dim: int = 6,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_graph_layers: int = 2,
        num_attention_heads: int = 8,
        fusion_method: str = "learned",
        use_region_adaptive_fusion: bool = True,
        num_timesteps: int = 1000,
        concatenate_stage1: bool = True,
        residual_alpha: float = 0.2,
    ):
        super().__init__()
        self.residual_alpha    = residual_alpha
        self.concatenate_stage1 = concatenate_stage1

        # Double UNet input channels to accommodate concatenated Stage-1 latent
        if concatenate_stage1:
            self._expand_unet_input(unet)
        self.unet = unet

        # Core RAGAF modules
        self.ragaf_attention  = RAGAFAttentionModule(
            node_feature_dim, text_dim, hidden_dim, num_graph_layers, num_attention_heads
        )
        self.adaptive_fusion  = AdaptiveModalityFusion(
            hidden_dim, num_timesteps, fusion_method, use_region_adaptive_fusion
        )

        # Projection: fused region feature → 4-channel latent modulation
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

        # Per-region refinement MLP (residual)
        self.refinement_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Fallback text projection for empty graphs
        self.text_proj = nn.Linear(text_dim, hidden_dim)

    # ------------------------------------------------------------------

    @staticmethod
    def _expand_unet_input(unet: UNet2DConditionModel):
        """Expand UNet conv_in from 4→8 channels (for Stage-1 concatenation)."""
        with torch.no_grad():
            old = unet.conv_in
            new = nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding)
            new.weight[:, :4] = old.weight.clone()
            new.weight[:, 4:] = torch.zeros_like(old.weight)
            new.bias = old.bias
            unet.conv_in = new
        if hasattr(unet, "config"):
            unet.config.in_channels = 8

    # ------------------------------------------------------------------

    def forward(
        self,
        latents: torch.Tensor,                          # (B, 4, H, W)
        timestep: torch.Tensor,                         # (B,)
        region_graph: Union[RegionGraph, List[RegionGraph]],
        text_embeddings: torch.Tensor,                  # (B, T, text_dim)
        stage1_latents: Optional[torch.Tensor] = None,  # (B, 4, H, W)
        return_dict: bool = False,
    ):
        B, _, H, W = latents.shape
        device, dtype = latents.device, latents.dtype

        # 1. Concatenate Stage-1 latent as additional 4 input channels
        if self.concatenate_stage1:
            s1 = stage1_latents if stage1_latents is not None else torch.zeros_like(latents)
            unet_input = torch.cat([latents, s1], dim=1)
        else:
            unet_input = latents

        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0).expand(B, -1, -1)

        rgs = [region_graph] * B if isinstance(region_graph, RegionGraph) else region_graph
        batch_modulation = torch.zeros_like(latents)

        # 2. Per-sample RAGAF processing
        for i, rg in enumerate(rgs):
            if rg.num_nodes == 0:
                # Empty graph – use mean text feature as global modulation
                feat = self.text_proj(text_embeddings[i].mean(0))  # (hidden_dim,)
                mod  = self.feature_projection(feat)                 # (4,)
                batch_modulation[i] = mod.view(4, 1, 1).expand(4, H, W)
                continue

            # Move graph tensors to device
            rg.node_features = rg.node_features.to(device)
            rg.edge_index    = rg.edge_index.to(device)
            if rg.edge_weights is not None:
                rg.edge_weights = rg.edge_weights.to(device)

            # RAGAF: text-enriched region features
            text_feats, _   = self.ragaf_attention(rg, text_embeddings[i])       # (N, H_dim)
            sketch_feats    = self.ragaf_attention.node_embedding(rg.node_features)  # (N, H_dim)

            # Adaptive fusion
            t_i = timestep[i] if timestep.dim() > 0 else timestep
            fused, _ = self.adaptive_fusion(sketch_feats, text_feats, t_i)       # (1, N, H_dim)
            fused    = fused.squeeze(0)                                           # (N, H_dim)
            fused    = fused + self.refinement_mlp(fused)                        # residual refinement

            # Project to latent-channel modulation
            mods = self.feature_projection(fused)  # (N, 4)
            for j, mask_np in enumerate(rg.region_masks):
                if j >= rg.num_nodes:
                    break
                mask = torch.from_numpy(mask_np).to(device=device, dtype=dtype)
                mask = F.interpolate(mask[None, None], size=(H, W), mode="nearest").squeeze()
                batch_modulation[i] += mods[j].view(4, 1, 1) * mask

        # 3. Add modulation to input latents
        if self.concatenate_stage1:
            unet_input[:, :4] += self.residual_alpha * batch_modulation
        else:
            unet_input += self.residual_alpha * batch_modulation

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(B)

        # 4. Standard UNet forward
        noise_pred = self.unet(
            unet_input, timestep,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]

        if not return_dict:
            return noise_pred
        return {"noise_pred": noise_pred, "modulation_map": batch_modulation}

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params  = list(self.ragaf_attention.parameters())
        params += list(self.adaptive_fusion.parameters())
        params += list(self.feature_projection.parameters())
        params += list(self.refinement_mlp.parameters())
        params += list(self.text_proj.parameters())
        params += [p for p in self.unet.parameters() if p.requires_grad]
        return params


# ─────────────────────────────────────────────────────────────────────────────
# §7  DATASET LOADER – SKETCHY
# ─────────────────────────────────────────────────────────────────────────────

class SketchyDataset(Dataset):
    """
    PyTorch Dataset for the Sketchy sketch-photo benchmark.

    Expected directory layout::

        <root_dir>/
          sketch/tx_000000000000/<category>/<id>.png
          photo/tx_000000000000/<category>/<id>.jpg

    Each sample yields a dict with keys:
        sketch       – (1, H, W) float tensor in [0, 1]
        photo        – (3, H, W) float tensor in [-1, 1]  (normalised for SD)
        text_prompt  – str, e.g. "A photo of an airplane"
        region_graph – RegionGraph (built on-the-fly or pre-cached)
        category     – str
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        image_size: int = 512,
        region_extractor=None,
        graph_builder=None,
        prompt_template: str = "A photo of a {category}",
        augment: bool = True,
    ):
        self.root_dir        = Path(root_dir)
        self.split           = split
        self.image_size      = image_size
        self.region_extractor = region_extractor
        self.graph_builder   = graph_builder
        self.prompt_template = prompt_template
        self.augment         = augment

        self.photo_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])
        self.sketch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # Collect (sketch_path, photo_path, category) tuples
        self.samples = self._collect_samples(categories)

    def _collect_samples(self, categories):
        sketch_root = self.root_dir / "sketch" / "tx_000000000000"
        photo_root  = self.root_dir / "photo"  / "tx_000000000000"
        if not sketch_root.exists():
            raise FileNotFoundError(f"Sketchy sketch root not found: {sketch_root}")

        all_cats = sorted(d.name for d in sketch_root.iterdir() if d.is_dir())
        if categories:
            all_cats = [c for c in all_cats if c in categories]

        # Simple deterministic train / val split (90 / 10 per category)
        samples = []
        for cat in all_cats:
            sk_files = sorted((sketch_root / cat).glob("*.png"))
            for sk in sk_files:
                ph_stem = sk.stem.split("-")[0]
                ph = photo_root / cat / f"{ph_stem}.jpg"
                if not ph.exists():
                    ph = photo_root / cat / f"{ph_stem}.JPEG"
                if ph.exists():
                    samples.append((sk, ph, cat))
        n = len(samples)
        cut = int(0.9 * n)
        return samples[:cut] if self.split == "train" else samples[cut:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sk_path, ph_path, category = self.samples[idx]
        sketch = Image.open(sk_path).convert("L")
        photo  = Image.open(ph_path).convert("RGB")

        if self.augment and self.split == "train" and random.random() > 0.5:
            sketch = transforms.functional.hflip(sketch)
            photo  = transforms.functional.hflip(photo)

        sketch_t = self.sketch_transform(sketch)   # (1, H, W)
        photo_t  = self.photo_transform(photo)      # (3, H, W) ∈ [-1, 1]

        # Build region graph from sketch
        region_graph = self._build_graph(sketch_t)

        return {
            "sketch":       sketch_t,
            "photo":        photo_t,
            "text_prompt":  self.prompt_template.format(category=category.replace("_", " ")),
            "region_graph": region_graph,
            "category":     category,
        }

    def _build_graph(self, sketch_tensor: torch.Tensor) -> RegionGraph:
        """Extract regions and build a graph from a single sketch tensor."""
        if self.region_extractor is None or self.graph_builder is None:
            # Return a trivial empty graph if no extractors are provided
            return RegionGraph(
                num_nodes=0,
                node_features=torch.zeros(0, 6),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_weights=None,
                region_masks=[],
                adjacency_matrix=None,
            )
        sketch_np = sketch_tensor.squeeze(0).numpy()
        regions   = self.region_extractor.extract(sketch_np)
        return self.graph_builder.build(sketch_np, regions)


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function that handles variable-size RegionGraphs."""
    return {
        "sketch":       torch.stack([b["sketch"]  for b in batch]),
        "photo":        torch.stack([b["photo"]   for b in batch]),
        "text_prompt":  [b["text_prompt"]          for b in batch],
        "region_graph": [b["region_graph"]         for b in batch],
        "category":     [b["category"]             for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# §8  TRAINER – STAGE 1 & STAGE 2 TRAINING STEPS
# ─────────────────────────────────────────────────────────────────────────────

class RAGAFDiffusionTrainer:
    """
    Trainer for the dual-stage RAGAF-Diffusion pipeline.

    Handles:
      • Mixed-precision training via 🤗 Accelerate.
      • Gradient accumulation.
      • W&B logging (optional).
      • Checkpoint save / resume.
      • Stage 1 and Stage 2 training loops.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, model_config, data_config, training_config):
        self.model_config    = model_config
        self.data_config     = data_config
        self.training_config = training_config

        # 🤗 Accelerate handles device placement, mixed precision, DDP
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.accelerator = Accelerator(
            mixed_precision=training_config.mixed_precision,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        )

        # Shared frozen components
        self.vae          = AutoencoderKL.from_pretrained(
            model_config.pretrained_model_name, subfolder="vae"
        ).requires_grad_(False).to(self.accelerator.device)

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_config.pretrained_model_name, subfolder="text_encoder"
        ).requires_grad_(False).to(self.accelerator.device)

        self.tokenizer    = CLIPTokenizer.from_pretrained(
            model_config.pretrained_model_name, subfolder="tokenizer"
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_config.pretrained_model_name, subfolder="scheduler"
        )
        self.ddim_scheduler  = DDIMScheduler.from_pretrained(
            model_config.pretrained_model_name, subfolder="scheduler"
        )

    # ------------------------------------------------------------------
    # Stage 1 training step
    # ------------------------------------------------------------------

    def train_stage1_step(self, batch: Dict, stage1_model: Stage1SketchGuidedDiffusion) -> Dict:
        """
        One gradient-accumulation step for Stage 1.

        Loss: ε-prediction MSE  L = ‖ε_θ(x_t, t, s, c) − ε‖²

        where  x_t is the noisy latent, t the timestep, s the sketch
        conditioning residuals, and c the CLIP text embedding.
        """
        device = self.accelerator.device
        photos  = batch["photo"].to(device)
        sketches = batch["sketch"].to(device)

        # Encode photo → latent space
        with torch.no_grad():
            latents = self.vae.encode(photos).latent_dist.sample() * 0.18215

        # Sample noise and timestep
        noise     = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Compute sketch conditioning and text embeddings
        sketch_features = stage1_model.encode_sketch(sketches)
        text_embeddings = stage1_model.encode_text(batch["text_prompt"])

        # Forward pass → noise prediction
        noise_pred = stage1_model(noisy_latents, timesteps, sketch_features, text_embeddings)

        # Diffusion loss
        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Stage 1 fast inference (used as conditioning for Stage 2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_stage1_latents(
        self,
        sketch_features: Tuple,
        text_embeddings: torch.Tensor,
        stage1_model: Stage1SketchGuidedDiffusion,
        num_steps: int = 12,
    ) -> torch.Tensor:
        """DDIM decode with ``num_steps`` steps to obtain Stage-1 coarse latents."""
        B, _, H, W = text_embeddings.shape[0], None, 32, 32
        device     = text_embeddings.device
        latents    = torch.randn(text_embeddings.shape[0], 4, H, W, device=device)
        self.ddim_scheduler.set_timesteps(num_steps)

        for t in self.ddim_scheduler.timesteps:
            lmi = self.ddim_scheduler.scale_model_input(latents, t)
            np_ = stage1_model(lmi, t.to(device), sketch_features, text_embeddings)
            latents = self.ddim_scheduler.step(np_, t, latents).prev_sample

        return latents

    # ------------------------------------------------------------------
    # Stage 2 training step
    # ------------------------------------------------------------------

    def train_stage2_step(
        self,
        batch: Dict,
        stage1_model: Stage1SketchGuidedDiffusion,
        stage2_model: Stage2SemanticRefinement,
    ) -> Dict:
        """
        One gradient-accumulation step for Stage 2.

        Losses:
          L_total = L_diffusion + λ_id · L_identity

        • L_diffusion  : standard noise-prediction MSE.
        • L_identity   : L1 + MSE between the predicted clean latent x̂₀ and
                         the Stage-1 coarse latent, to preserve structure.
        """
        device  = self.accelerator.device
        photos  = batch["photo"].to(device)
        sketches = batch["sketch"].to(device)

        # 1. Obtain Stage-1 coarse latents (no grad – Stage 1 is frozen)
        with torch.no_grad():
            sketch_features = stage1_model.encode_sketch(sketches)
            text_embeddings = stage1_model.encode_text(batch["text_prompt"])
            stage1_latents  = self._generate_stage1_latents(
                sketch_features, text_embeddings, stage1_model
            )
            gt_latents = self.vae.encode(photos).latent_dist.sample() * 0.18215

        # 2. Build noisy input (stochastic mix of Stage-1 and GT)
        base = stage1_latents if torch.rand(1).item() < 0.7 else (
            0.5 * gt_latents + 0.5 * stage1_latents
        )
        noise     = torch.randn_like(base)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (photos.shape[0],), device=device,
        )
        noisy_latents = self.noise_scheduler.add_noise(base.detach(), noise, timesteps)

        # 3. Stage 2 forward
        out = stage2_model(
            noisy_latents, timesteps,
            batch["region_graph"], text_embeddings,
            stage1_latents=stage1_latents.detach(),
            return_dict=True,
        )
        noise_pred = out["noise_pred"]

        # 4. Recover predicted clean latent x̂₀ for identity loss
        αs  = self.noise_scheduler.alphas_cumprod.to(device)
        α_t = αs[timesteps].view(-1, 1, 1, 1)
        σ_t = (1 - α_t).sqrt()
        pred_x0 = (noisy_latents - σ_t * noise_pred) / α_t.sqrt()

        # 5. Compute losses
        loss_diffusion = F.mse_loss(noise_pred, noise)
        loss_identity  = 0.5 * F.l1_loss(pred_x0, stage1_latents.detach()) + \
                         0.5 * F.mse_loss(pred_x0, stage1_latents.detach())
        loss_total     = loss_diffusion + 0.1 * loss_identity

        return {
            "loss":            loss_total,
            "loss_diffusion":  loss_diffusion.item(),
            "loss_identity":   loss_identity.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# §9  INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class RAGAFDiffusionInferencePipeline:
    """
    End-to-end inference: sketch + text prompt → generated image.

    Usage::

        pipeline = RAGAFDiffusionInferencePipeline(
            stage1_model, stage2_model, vae,
            num_stage1_steps=50, num_stage2_steps=30,
            guidance_scale=7.5,
        )
        image = pipeline.generate(sketch, "a golden retriever playing in a park")
    """

    def __init__(
        self,
        stage1_model: Stage1SketchGuidedDiffusion,
        stage2_model: Stage2SemanticRefinement,
        vae: AutoencoderKL,
        region_extractor=None,
        graph_builder=None,
        num_stage1_steps: int = 50,
        num_stage2_steps: int = 30,
        guidance_scale: float = 7.5,
        device: str = "cuda",
    ):
        self.stage1_model    = stage1_model.to(device).eval()
        self.stage2_model    = stage2_model.to(device).eval()
        self.vae             = vae.to(device).eval()
        self.region_extractor = region_extractor
        self.graph_builder   = graph_builder
        self.guidance_scale  = guidance_scale
        self.device          = device

        self.scheduler_s1 = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        self.scheduler_s1.set_timesteps(num_stage1_steps)

        self.scheduler_s2 = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        self.scheduler_s2.set_timesteps(num_stage2_steps)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        sketch: torch.Tensor,    # (1, 1, H, W) in [0, 1]
        text_prompt: str,
        height: int = 512,
        width:  int = 512,
        seed: Optional[int] = None,
    ) -> torch.Tensor:           # (1, 3, H, W) in [0, 1]
        """Generate a single image."""
        if seed is not None:
            torch.manual_seed(seed)

        sketch = sketch.to(self.device)

        # ── Stage 1: sketch-conditioned coarse generation ──────────────
        down_res, mid_res = self.stage1_model.encode_sketch(sketch)

        # Duplicate residuals for classifier-free guidance (uncond + cond)
        down_res_cfg = [torch.cat([r, r]) for r in down_res]
        mid_res_cfg  = torch.cat([mid_res, mid_res])

        text_emb  = self.stage1_model.encode_text([text_prompt])
        uncond_emb = self.stage1_model.encode_text([""])

        latents = (
            torch.randn(1, 4, height // 8, width // 8, device=self.device)
            * self.scheduler_s1.init_noise_sigma
        )

        for t in self.scheduler_s1.timesteps:
            lmi = self.scheduler_s1.scale_model_input(torch.cat([latents] * 2), t)
            enc = torch.cat([uncond_emb, text_emb])
            np_ = self.stage1_model(lmi, t, (down_res_cfg, mid_res_cfg), enc)
            np_u, np_c = np_.chunk(2)
            np_ = np_u + self.guidance_scale * (np_c - np_u)  # CFG
            latents = self.scheduler_s1.step(np_, t, latents).prev_sample

        stage1_latents = latents.clone()

        # ── Stage 2: semantic refinement ────────────────────────────────
        # Build region graph
        rg = self._build_region_graph(sketch)

        # Fresh noise for refinement
        latents = torch.randn_like(stage1_latents) * self.scheduler_s2.init_noise_sigma

        for t in self.scheduler_s2.timesteps:
            lmi = self.scheduler_s2.scale_model_input(latents, t)
            np_ = self.stage2_model(
                lmi, t.unsqueeze(0).to(self.device),
                rg, text_emb,
                stage1_latents=stage1_latents,
                return_dict=False,
            )
            latents = self.scheduler_s2.step(np_, t, latents).prev_sample

        # Decode to pixel space
        image = self.vae.decode(latents / 0.18215).sample
        return (image / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------

    def _build_region_graph(self, sketch: torch.Tensor) -> RegionGraph:
        """Build a RegionGraph from a sketch tensor. Falls back to empty graph."""
        if self.region_extractor is None or self.graph_builder is None:
            return RegionGraph(0, torch.zeros(0, 6),
                               torch.zeros(2, 0, dtype=torch.long), None, [], None)
        sketch_np = sketch.squeeze().cpu().numpy()
        regions   = self.region_extractor.extract(sketch_np)
        return self.graph_builder.build(sketch_np, regions)


# ─────────────────────────────────────────────────────────────────────────────
# §10  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) float tensor in [0,1] or [-1,1] to a PIL Image."""
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    arr = (tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def visualise_fusion_weights(
    sketch_weights: np.ndarray,
    text_weights:   np.ndarray,
    timesteps:      List[int],
    save_path: Optional[str] = None,
):
    """
    Plot how adaptive fusion weights evolve across diffusion timesteps.

    Args:
        sketch_weights : (T,) mean sketch weight per timestep.
        text_weights   : (T,) mean text weight per timestep.
        timesteps      : list of timestep values corresponding to each entry.
        save_path      : if provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timesteps, sketch_weights, label="Sketch weight", color="steelblue",  linewidth=2)
    ax.plot(timesteps, text_weights,   label="Text weight",   color="darkorange", linewidth=2)
    ax.set_xlabel("Diffusion Timestep t")
    ax.set_ylabel("Fusion Weight")
    ax.set_title("Adaptive Modality Fusion Weights across Timesteps")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def compute_edge_similarity(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Measure structural similarity between two images using Sobel edge maps.

    Both inputs should be (B, 3, H, W) in [-1, 1].
    Returns the mean cosine similarity between their edge magnitude maps.
    """
    def sobel(x):
        x = (x + 1) / 2                        # → [0, 1]
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]  # grey
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        return torch.sqrt(F.conv2d(x, kx, padding=1) ** 2 +
                          F.conv2d(x, ky, padding=1) ** 2 + 1e-6)

    return F.cosine_similarity(sobel(img1).flatten(1), sobel(img2).flatten(1)).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (no GPU or pretrained weights required)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("RAGAF-Diffusion – Thesis Source Code Self-Test")
    print("=" * 70)

    set_seed(42)
    device = "cpu"  # run on CPU for portability

    # ── Test 1: SketchEncoder (no pretrained UNet needed) ──────────────
    print("\n[1] SketchEncoder forward pass …")
    encoder = SketchEncoder(in_channels=1, base_channels=16)
    sketch  = torch.randn(1, 1, 256, 256)
    down_res, mid_res = encoder(sketch * 2 - 1)
    print(f"    ✓  down residuals : {len(down_res)} tensors")
    print(f"    ✓  mid residual   : {mid_res.shape}")

    # ── Test 2: RAGAF Attention ─────────────────────────────────────────
    print("\n[2] RAGAFAttentionModule forward pass …")
    rg = RegionGraph(
        num_nodes=8,
        node_features=torch.randn(8, 6),
        edge_index=torch.randint(0, 8, (2, 20)),
        edge_weights=torch.rand(20),
        region_masks=[np.zeros((64, 64)) for _ in range(8)],
        adjacency_matrix=None,
    )
    text_emb = torch.randn(77, 768)
    ragaf    = RAGAFAttentionModule(node_feature_dim=6, text_dim=768, hidden_dim=128, num_graph_layers=2)
    feats, attn = ragaf(rg, text_emb)
    print(f"    ✓  region features : {feats.shape}")
    print(f"    ✓  attention map   : {attn.shape}")

    # ── Test 3: Adaptive Modality Fusion ───────────────────────────────
    print("\n[3] AdaptiveModalityFusion at t=900, 500, 100 …")
    fusion   = AdaptiveModalityFusion(feature_dim=128)
    sk_feat  = torch.randn(1, 8, 128)
    tx_feat  = torch.randn(1, 8, 128)
    for t_val in [900, 500, 100]:
        t    = torch.tensor([t_val])
        out, info = fusion(sk_feat, tx_feat, t)
        sw = info["sketch_weight"].mean().item()
        tw = info["text_weight"].mean().item()
        print(f"    t={t_val:4d} → sketch_w={sw:.3f}  text_w={tw:.3f}  (sum={sw+tw:.3f})")

    # ── Test 4: Dataset collate helper ─────────────────────────────────
    print("\n[4] collate_fn with dummy batch …")
    dummy_batch = [
        {
            "sketch": torch.randn(1, 64, 64),
            "photo":  torch.randn(3, 64, 64),
            "text_prompt": f"A photo of cat {i}",
            "region_graph": RegionGraph(
                0, torch.zeros(0, 6),
                torch.zeros(2, 0, dtype=torch.long), None, [], None
            ),
            "category": "cat",
        }
        for i in range(4)
    ]
    collated = collate_fn(dummy_batch)
    print(f"    ✓  sketch batch : {collated['sketch'].shape}")
    print(f"    ✓  photo  batch : {collated['photo'].shape}")

    # ── Test 5: Utility helpers ─────────────────────────────────────────
    print("\n[5] Utility helpers …")
    img = tensor_to_pil(torch.rand(3, 64, 64))
    print(f"    ✓  tensor_to_pil  → PIL Image {img.size}")
    sim = compute_edge_similarity(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    print(f"    ✓  edge_similarity → {sim:.4f}")

    print("\n" + "=" * 70)
    print("All self-tests passed.  See individual modules for GPU-based tests.")
    print("=" * 70)
