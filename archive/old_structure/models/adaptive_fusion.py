"""
Adaptive Modality Fusion for RAGAF-Diffusion

This module implements dynamic, timestep-aware fusion between sketch and text
modalities. The fusion weights adapt across diffusion timesteps to balance:
- Early timesteps: Strong sketch guidance for structure
- Late timesteps: Strong text guidance for details and texture

Key innovations:
1. Timestep-conditioned fusion weights
2. Region-specific adaptive weighting
3. Learned vs. heuristic fusion strategies

Author: RAGAF-Diffusion Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class AdaptiveFusionWeights(nn.Module):
    """
    Compute adaptive fusion weights based on diffusion timestep and region features.
    
    Strategy:
    - Early timesteps (high noise): Prioritize sketch structure
    - Late timesteps (low noise): Prioritize text details
    - Region-specific: Different regions may need different balance
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        region_feature_dim: int = 512,
        fusion_method: str = "learned",  # "learned", "heuristic", "hybrid"
        use_region_adaptive: bool = True
    ):
        """
        Initialize adaptive fusion weights.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            region_feature_dim: Dimension of region features
            fusion_method: Method for computing fusion weights
            use_region_adaptive: Whether to adapt weights per region
        """
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.region_feature_dim = region_feature_dim
        self.fusion_method = fusion_method
        self.use_region_adaptive = use_region_adaptive
        
        if fusion_method in ["learned", "hybrid"]:
            # Learnable fusion network
            # Input: [timestep_embedding, region_features]
            # Output: fusion weights
            
            # Timestep embedding (sinusoidal)
            self.time_embed_dim = 128
            
            # MLP for fusion weight prediction
            if use_region_adaptive:
                # Region-specific weights
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(self.time_embed_dim + region_feature_dim, 256),
                    nn.SiLU(),
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, 2)  # [sketch_weight, text_weight]
                )
            else:
                # Global weights (same for all regions)
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(self.time_embed_dim, 256),
                    nn.SiLU(),
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, 2)
                )
    
    def timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: Timestep values (B,) in range [0, num_timesteps-1]
            dim: Embedding dimension
        
        Returns:
            Timestep embeddings (B, dim)
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if dim % 2 == 1:  # Pad if odd
            emb = F.pad(emb, (0, 1))
        
        return emb
    
    def forward(
        self,
        timestep: torch.Tensor,
        region_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive fusion weights.
        
        Args:
            timestep: Current timestep (B,) or scalar
            region_features: Region features (B, N, region_feature_dim) or (N, region_feature_dim)
        
        Returns:
            - Sketch weight (B, N) or (N,) or (B,)
            - Text weight (B, N) or (N,) or (B,)
        """
        # Ensure timestep is tensor
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep])
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        batch_size = timestep.shape[0]
        
        if self.fusion_method == "heuristic":
            # Simple heuristic: linear interpolation based on timestep
            # Early (t near 1000): sketch_weight=0.9, text_weight=0.1
            # Late (t near 0): sketch_weight=0.3, text_weight=0.7
            
            t_normalized = timestep.float() / self.num_timesteps  # [0, 1]
            
            # Sketch weight decreases over time
            sketch_weight = 0.3 + 0.6 * t_normalized
            # Text weight increases over time
            text_weight = 0.7 - 0.6 * t_normalized
            
            if region_features is not None and self.use_region_adaptive:
                # Broadcast to match region dimensions
                num_regions = region_features.shape[-2]
                sketch_weight = sketch_weight.unsqueeze(-1).expand(batch_size, num_regions)
                text_weight = text_weight.unsqueeze(-1).expand(batch_size, num_regions)
            
            return sketch_weight, text_weight
        
        elif self.fusion_method in ["learned", "hybrid"]:
            # Learned fusion weights
            time_emb = self.timestep_embedding(timestep, self.time_embed_dim)  # (B, time_embed_dim)
            
            if self.use_region_adaptive and region_features is not None:
                # Region-specific weights
                # region_features: (B, N, region_feature_dim) or (N, region_feature_dim)
                
                if region_features.dim() == 2:
                    # Add batch dimension
                    region_features = region_features.unsqueeze(0)
                
                num_regions = region_features.shape[1]
                
                # Expand time_emb to match regions
                time_emb_expanded = time_emb.unsqueeze(1).expand(-1, num_regions, -1)  # (B, N, time_embed_dim)
                
                # Concatenate
                fusion_input = torch.cat([time_emb_expanded, region_features], dim=-1)  # (B, N, time_embed_dim + region_dim)
                
                # Predict weights
                fusion_weights = self.fusion_mlp(fusion_input)  # (B, N, 2)
                
                # Softmax to ensure weights sum to 1
                fusion_weights = F.softmax(fusion_weights, dim=-1)
                
                sketch_weight = fusion_weights[..., 0]  # (B, N)
                text_weight = fusion_weights[..., 1]    # (B, N)
                
            else:
                # Global weights
                fusion_weights = self.fusion_mlp(time_emb)  # (B, 2)
                fusion_weights = F.softmax(fusion_weights, dim=-1)
                
                sketch_weight = fusion_weights[:, 0]  # (B,)
                text_weight = fusion_weights[:, 1]
            
            # If hybrid, blend with heuristic
            if self.fusion_method == "hybrid":
                heuristic_sketch, heuristic_text = self.forward_heuristic(timestep, region_features)
                
                # 50-50 blend (could be learnable)
                sketch_weight = 0.5 * sketch_weight + 0.5 * heuristic_sketch
                text_weight = 0.5 * text_weight + 0.5 * heuristic_text
            
            return sketch_weight, text_weight
    
    def forward_heuristic(
        self,
        timestep: torch.Tensor,
        region_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to call heuristic method."""
        original_method = self.fusion_method
        self.fusion_method = "heuristic"
        result = self.forward(timestep, region_features)
        self.fusion_method = original_method
        return result


class AdaptiveModalityFusion(nn.Module):
    """
    Adaptive fusion of sketch and text features for each region.
    
    Combines region-level sketch features and text-aligned features using
    adaptive weights that vary by timestep and region.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_timesteps: int = 1000,
        fusion_method: str = "learned",
        use_region_adaptive: bool = True
    ):
        """
        Initialize adaptive fusion module.
        
        Args:
            feature_dim: Feature dimension
            num_timesteps: Number of diffusion timesteps
            fusion_method: Fusion weight computation method
            use_region_adaptive: Use region-specific weights
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Adaptive fusion weights
        self.fusion_weights = AdaptiveFusionWeights(
            num_timesteps=num_timesteps,
            region_feature_dim=feature_dim,
            fusion_method=fusion_method,
            use_region_adaptive=use_region_adaptive
        )
        
        # Optional: Learnable fusion transformation
        self.fusion_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        sketch_features: torch.Tensor,
        text_features: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse sketch and text features adaptively.
        
        Args:
            sketch_features: Sketch-derived region features (B, N, feature_dim) or (N, feature_dim)
            text_features: Text-aligned region features (B, N, feature_dim) or (N, feature_dim)
            timestep: Current diffusion timestep (B,) or scalar
        
        Returns:
            - Fused features (B, N, feature_dim) or (N, feature_dim)
            - Dict with fusion weights for analysis
        """
        # Compute adaptive weights
        sketch_weight, text_weight = self.fusion_weights(timestep, sketch_features)
        
        # Ensure weights are broadcastable
        if sketch_features.dim() == 2:
            # (N, feature_dim)
            sketch_weight = sketch_weight.unsqueeze(-1)  # (N, 1)
            text_weight = text_weight.unsqueeze(-1)
        elif sketch_features.dim() == 3:
            # (B, N, feature_dim)
            sketch_weight = sketch_weight.unsqueeze(-1)  # (B, N, 1)
            text_weight = text_weight.unsqueeze(-1)
        
        # Weighted fusion
        fused = sketch_weight * sketch_features + text_weight * text_features
        
        # Apply fusion transform and norm
        fused = self.fusion_transform(fused)
        fused = self.norm(fused)
        
        # Return fusion weights for visualization/analysis
        fusion_info = {
            "sketch_weight": sketch_weight.squeeze(-1),
            "text_weight": text_weight.squeeze(-1)
        }
        
        return fused, fusion_info


class RegionFeatureInjection(nn.Module):
    """
    Inject region-level features back into spatial feature maps.
    
    This module takes region-level fused features and injects them back into
    the spatial feature maps used by the diffusion UNet.
    """
    
    def __init__(
        self,
        region_feature_dim: int = 512,
        spatial_feature_dim: int = 1280,  # UNet feature dimension
        injection_method: str = "add"  # "add", "concat", "gated"
    ):
        """
        Initialize feature injection module.
        
        Args:
            region_feature_dim: Dimension of region features
            spatial_feature_dim: Dimension of spatial features
            injection_method: Method for injecting features
        """
        super().__init__()
        
        self.region_feature_dim = region_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.injection_method = injection_method
        
        # Projection to match spatial feature dimension
        self.region_proj = nn.Linear(region_feature_dim, spatial_feature_dim)
        
        if injection_method == "concat":
            # If concatenating, need a projection to reduce back to original dim
            self.concat_proj = nn.Conv2d(
                spatial_feature_dim * 2, spatial_feature_dim, kernel_size=1
            )
        elif injection_method == "gated":
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(region_feature_dim + spatial_feature_dim, spatial_feature_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        region_features: torch.Tensor,
        region_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Inject region features into spatial feature map.
        
        Args:
            spatial_features: Spatial features (B, C, H, W)
            region_features: Region-level features (N, region_feature_dim)
            region_masks: List of region masks (each H, W)
        
        Returns:
            Updated spatial features (B, C, H, W)
        """
        B, C, H, W = spatial_features.shape
        N = region_features.shape[0]
        
        # Project region features
        region_features_proj = self.region_proj(region_features)  # (N, spatial_feature_dim)
        
        # Create spatial map of region features
        # For each region, apply its features to the corresponding spatial locations
        
        region_feature_map = torch.zeros(
            B, self.spatial_feature_dim, H, W,
            device=spatial_features.device,
            dtype=spatial_features.dtype
        )
        
        for i, mask in enumerate(region_masks):
            if i >= N:
                break
            
            # Resize mask to match feature map size
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze()  # (H, W)
            
            # Apply region features to masked locations
            region_feature_map[:, :, mask_resized > 0.5] = region_features_proj[i].unsqueeze(-1)
        
        # Inject features based on method
        if self.injection_method == "add":
            output = spatial_features + region_feature_map
        elif self.injection_method == "concat":
            concatenated = torch.cat([spatial_features, region_feature_map], dim=1)
            output = self.concat_proj(concatenated)
        elif self.injection_method == "gated":
            # Compute gate
            # This is simplified; proper implementation would be per-pixel gating
            output = spatial_features  # TODO: Implement proper gating
        else:
            output = spatial_features
        
        return output


if __name__ == "__main__":
    # Example usage
    print("Adaptive Modality Fusion for RAGAF-Diffusion")
    print("=" * 60)
    
    # Test adaptive fusion weights
    fusion_weights = AdaptiveFusionWeights(
        num_timesteps=1000,
        region_feature_dim=512,
        fusion_method="learned"
    )
    
    # Test at different timesteps
    timesteps = torch.tensor([900, 500, 100])  # Early, mid, late
    region_features = torch.randn(3, 10, 512)  # 3 batches, 10 regions
    
    print("Testing adaptive fusion weights:")
    for i, t in enumerate(timesteps):
        sketch_w, text_w = fusion_weights(t.unsqueeze(0), region_features[i:i+1])
        print(f"  Timestep {t.item()}:")
        print(f"    Sketch weight: {sketch_w.mean().item():.3f} ± {sketch_w.std().item():.3f}")
        print(f"    Text weight: {text_w.mean().item():.3f} ± {text_w.std().item():.3f}")
    
    # Test full fusion module
    print("\nTesting adaptive modality fusion:")
    fusion_module = AdaptiveModalityFusion(
        feature_dim=512,
        fusion_method="learned"
    )
    
    sketch_features = torch.randn(1, 10, 512)
    text_features = torch.randn(1, 10, 512)
    timestep = torch.tensor([500])
    
    fused, info = fusion_module(sketch_features, text_features, timestep)
    print(f"  Input shapes: {sketch_features.shape}, {text_features.shape}")
    print(f"  Output shape: {fused.shape}")
    print(f"  Fusion weights: sketch={info['sketch_weight'].mean():.3f}, text={info['text_weight'].mean():.3f}")
