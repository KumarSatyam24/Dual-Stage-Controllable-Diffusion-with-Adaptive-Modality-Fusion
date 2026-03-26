
"""
Stage 2: Semantic Refinement with RAGAF Fusion

This module implements the second stage of the dual-stage pipeline:
semantic refinement using text prompts while preserving sketch structure.

Combines:
- Coarse output from Stage 1
- RAGAF attention for region-text association
- Adaptive fusion to balance sketch structure and text details

Author: RAGAF-Diffusion Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from diffusers import UNet2DConditionModel

from src.models.ragaf_attention import RAGAFAttentionModule
from src.models.adaptive_fusion import AdaptiveModalityFusion, RegionFeatureInjection
from src.data.region_graph import RegionGraph


class Stage2SemanticRefinement(nn.Module):
    """
    Stage 2: Semantic refinement with RAGAF fusion.
    
    Takes the coarse output from Stage 1 and refines it using:
    1. Region-text association via RAGAF attention
    2. Adaptive fusion of sketch and text features
    3. Structure-preserving refinement
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
        use_residual: bool = True,
        concatenate_stage1: bool = True,
        residual_alpha: float = 0.2
    ):
        """
        Initialize Stage 2 model.
        
        Args:
            unet: Pretrained UNet from Stage 1 or separate UNet
            node_feature_dim: Dimension of region node features
            text_dim: Dimension of text embeddings
            hidden_dim: Hidden dimension for RAGAF
            num_graph_layers: Number of graph attention layers
            num_attention_heads: Number of attention heads
            fusion_method: Method for adaptive fusion
            use_region_adaptive_fusion: Use region-specific fusion weights
            num_timesteps: Number of diffusion timesteps
            use_residual: Whether to use residual learning
            concatenate_stage1: Whether to concatenate Stage-1 latents to noisy input
            residual_alpha: Scaling factor for residual refinement
        """
        super().__init__()
        
        self.unet = unet
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.concatenate_stage1 = concatenate_stage1
        self.residual_alpha = residual_alpha
        
        # Adjust UNet input channels if concatenating
        if self.concatenate_stage1:
            self._adjust_unet_input_channels()
        
        # RAGAF attention module
        self.ragaf_attention = RAGAFAttentionModule(
            node_feature_dim=node_feature_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            num_heads=num_attention_heads
        )
        
        # Adaptive fusion module
        self.adaptive_fusion = AdaptiveModalityFusion(
            feature_dim=hidden_dim,
            num_timesteps=num_timesteps,
            fusion_method=fusion_method,
            use_region_adaptive=use_region_adaptive_fusion
        )
        
        # Feature injection module
        self.feature_injection = RegionFeatureInjection(
            region_feature_dim=hidden_dim,
            spatial_feature_dim=1280,
            injection_method="add"
        )
        
        # Projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # Refinement MLP
        self.refinement_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        print(f"Stage 2 Semantic Refinement initialized. Residual: {use_residual}, Concat: {concatenate_stage1}")

    def _adjust_unet_input_channels(self):
        """Adjust UNet input channels to 8 for concatenation of Stage-1 latents."""
        with torch.no_grad():
            old_conv = self.unet.conv_in
            new_conv = nn.Conv2d(
                8,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding
            )
            new_conv.weight[:, :4, :, :] = old_conv.weight.clone()
            new_conv.weight[:, 4:, :, :] = torch.zeros_like(old_conv.weight)
            new_conv.bias = old_conv.bias
            self.unet.conv_in = new_conv
            if hasattr(self.unet, "config"):
                self.unet.config.in_channels = 8
    
    def compute_region_text_alignment(
        self,
        region_graph: RegionGraph,
        text_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute region-text alignment using RAGAF attention.
        
        Args:
            region_graph: RegionGraph object
            text_embeddings: Text embeddings (T, text_dim)
        
        Returns:
            - Text-aligned region features (N, hidden_dim)
            - Attention map (N, T)
        """
        # Apply RAGAF attention
        region_features, attn_map = self.ragaf_attention(
            region_graph,
            text_embeddings
        )
        
        return region_features, attn_map
    
    def fuse_modalities(
        self,
        sketch_region_features: torch.Tensor,
        text_region_features: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Adaptively fuse sketch and text features.
        
        Args:
            sketch_region_features: Sketch-derived features (N, hidden_dim)
            text_region_features: Text-aligned features (N, hidden_dim)
            timestep: Current timestep
        
        Returns:
            - Fused features (N, hidden_dim)
            - Fusion info dict
        """
        fused_features, fusion_info = self.adaptive_fusion(
            sketch_region_features,
            text_region_features,
            timestep
        )
        
        # Optional refinement
        fused_features = fused_features + self.refinement_mlp(fused_features)
        
        return fused_features, fusion_info
    
    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        region_graph: Union[RegionGraph, List[RegionGraph]],
        text_embeddings: torch.Tensor,
        stage1_latents: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Dict:
        """
        Forward pass through Stage 2 refinement.
        
        Args:
            latents: Noisy latents (B, 4, H/8, W/8)
            timestep: Diffusion timestep (B,) or scalar
            region_graph: RegionGraph object or list of RegionGraph objects
            text_embeddings: Text embeddings (B, T, text_dim) where B is batch size
            stage1_latents: Optional Stage-1 latents for conditioning (B, 4, H/8, W/8)
            return_dict: Whether to return detailed dict
        
        Returns:
            Dict with noise prediction and auxiliary outputs
        """
        batch_size = latents.shape[0]
        device = latents.device
        dtype = latents.dtype
        H, W = latents.shape[2:]
        
        # 1. Handle Stage-1 conditioning
        if self.concatenate_stage1:
            if stage1_latents is None:
                stage1_latents = torch.zeros_like(latents)
            unet_input = torch.cat([latents, stage1_latents], dim=1)  # (B, 8, H/8, W/8)
        else:
            unet_input = latents

        # 2. Ensure text_embeddings is properly batched
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Process each item in batch for region-aware features
        if isinstance(region_graph, RegionGraph):
            region_graphs = [region_graph] * batch_size
        else:
            region_graphs = region_graph

        batch_fused_features = []
        batch_modulation_map = torch.zeros_like(latents)

        for i in range(batch_size):
            rg = region_graphs[i]
            if rg.num_nodes == 0:
                # Handle empty graph: use global text features as fallback
                text_feat = text_embeddings[i].mean(0) # (D,)
                if text_feat.shape[-1] != self.hidden_dim:
                    if not hasattr(self, 'text_proj'):
                        self.text_proj = nn.Linear(text_feat.shape[-1], self.hidden_dim).to(device)
                    fused = self.text_proj(text_feat)
                else:
                    fused = text_feat
                
                # Global modulation
                mod = self.feature_projection(fused) # (4,)
                batch_modulation_map[i] = mod.view(4, 1, 1).expand(4, H, W)
                batch_fused_features.append(fused.unsqueeze(0))
                continue

            # Move graph components to device
            rg.node_features = rg.node_features.to(device)
            rg.edge_index = rg.edge_index.to(device)
            if rg.edge_weights is not None:
                rg.edge_weights = rg.edge_weights.to(device)

            # RAGAF Attention: text-aligned region features
            text_region_features, _ = self.ragaf_attention(rg, text_embeddings[i]) # (N, hidden_dim)

            # Sketch features from graph nodes (projected)
            sketch_region_features = self.ragaf_attention.node_embedding(rg.node_features) # (N, hidden_dim)

            # Adaptive Fusion
            # timestep for this item
            t_i = timestep[i] if timestep.dim() > 0 else timestep
            fused_features, _ = self.adaptive_fusion(
                sketch_region_features,
                text_region_features,
                t_i
            ) # (B=1, N, hidden_dim)
            fused_features = fused_features.squeeze(0) # (N, hidden_dim)

            # Refinement MLP (inside fuse_modalities-like logic)
            fused_features = fused_features + self.refinement_mlp(fused_features)
            batch_fused_features.append(fused_features)

            # Project to latent modulation
            region_modulations = self.feature_projection(fused_features) # (N, 4)
            # print(f"DEBUG: fused_features shape: {fused_features.shape}, region_modulations shape: {region_modulations.shape}")

            # Create spatial modulation map using masks
            for j, mask_np in enumerate(rg.region_masks):
                if j >= rg.num_nodes: break
                
                # Convert mask to tensor and resize
                mask = torch.from_numpy(mask_np).to(device=device, dtype=dtype)
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='nearest'
                ).squeeze() # (H, W)
                
                # Inject region modulation
                # (4,) * (H, W) -> (4, H, W)
                # print(f"DEBUG: j={j}, region_modulations[j] shape: {region_modulations[j].shape}")
                batch_modulation_map[i] += region_modulations[j].view(4, 1, 1) * mask_resized

        # 4. Apply modulation to UNet input or latents
        # Based on checklist: conditioned_latents = latents + alpha * fused_latents
        # We apply it to the latents part of unet_input
        if self.concatenate_stage1:
            unet_input[:, :4] = unet_input[:, :4] + self.residual_alpha * batch_modulation_map
        else:
            unet_input = unet_input + self.residual_alpha * batch_modulation_map
        
        # Ensure timestep is proper shape for UNet: (B,)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(batch_size)
        
        # 5. Standard UNet forward
        noise_pred = self.unet(
            unet_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            return_dict=False
        )[0]
        
        
        return {
            "noise_pred": noise_pred,
            "modulation_map": batch_modulation_map,
            "fused_features": batch_fused_features
        }

    def predict_latent_delta(self, unet_input, timestep, text_embeddings):
        """Explicitly predict the refinement delta for latent space."""
        # This can be used if we want to bypass the standard diffusion noise pred
        # and directly predict the latent refinement.
        noise_pred = self.unet(
            unet_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            return_dict=False
        )[0]
        return self.residual_alpha * noise_pred
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimization."""
        trainable_params = []
        
        # RAGAF attention
        trainable_params.extend(self.ragaf_attention.parameters())
        
        # Adaptive fusion
        trainable_params.extend(self.adaptive_fusion.parameters())
        
        # Feature projection (CRITICAL for Stage 2)
        trainable_params.extend(self.feature_projection.parameters())
        
        # Feature injection
        trainable_params.extend(self.feature_injection.parameters())
        
        # Refinement MLP
        trainable_params.extend(self.refinement_mlp.parameters())
        
        # UNet parameters (if not frozen)
        trainable_params.extend(
            p for p in self.unet.parameters() if p.requires_grad
        )
        
        return trainable_params


class Stage2RefinementPipeline:
    """
    Inference pipeline for Stage 2 semantic refinement.
    
    Takes Stage 1 output and refines it with text guidance.
    """
    
    def __init__(
        self,
        stage2_model: Stage2SemanticRefinement,
        vae,  # VAE decoder
        num_inference_steps: int = 30,  # Fewer steps than Stage 1
        guidance_scale: float = 7.5,
        device: str = "cuda"
    ):
        """
        Initialize Stage 2 pipeline.
        
        Args:
            stage2_model: Stage2SemanticRefinement model
            vae: VAE for encoding/decoding
            num_inference_steps: Number of refinement steps
            guidance_scale: Guidance scale
            device: Device
        """
        self.model = stage2_model
        self.vae = vae
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
    
    @torch.no_grad()
    def refine(
        self,
        stage1_image: torch.Tensor,
        region_graph: RegionGraph,
        text_prompt: str,
        text_embeddings: torch.Tensor,
        strength: float = 0.5,  # Refinement strength [0, 1]
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Refine Stage 1 output with text guidance.
        
        Args:
            stage1_image: Output from Stage 1 (1, 3, H, W) in [0, 1]
            region_graph: RegionGraph object
            text_prompt: Text prompt (for reference)
            text_embeddings: Pre-computed text embeddings
            strength: Refinement strength (higher = more change)
            seed: Random seed
        
        Returns:
            Refined image (1, 3, H, W) in [0, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Move to device
        stage1_image = stage1_image.to(self.device)
        
        # 1. Encode to latents
        stage1_image_normalized = stage1_image * 2 - 1  # [0,1] -> [-1,1]
        stage1_latents = self.vae.encode(stage1_image_normalized).latent_dist.sample()
        stage1_latents = stage1_latents * 0.18215
        
        # 2. Setup scheduler and timesteps
        # We only run for a fraction of steps based on strength
        # (similar to Stable Diffusion img2img)
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        scheduler.set_timesteps(self.num_inference_steps)
        
        # Determine starting timestep
        init_timestep = int(self.num_inference_steps * strength)
        timesteps = scheduler.timesteps[-init_timestep:]
        
        # 3. Add noise to Stage 1 latents
        noise = torch.randn_like(stage1_latents)
        latents = scheduler.add_noise(stage1_latents, noise, timesteps[0])
        
        # 4. Denoising loop
        for t in timesteps:
            # Scale model input
            latent_model_input = scheduler.scale_model_input(latents, t)
            
            # Predict noise
            # We pass stage1_latents as conditioning to the Stage 2 model
            noise_pred = self.model(
                latent_model_input,
                t.to(self.device),
                region_graph,
                text_embeddings,
                stage1_latents=stage1_latents,
                return_dict=False
            )
            
            # Perform guidance (if needed)
            # For simplicity, we assume text_embeddings already contains guidance if requested
            
            # Step scheduler
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
        # 5. Decode refined latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        
        print(f"Stage 2 refinement complete with strength {strength}")
        
        return image


if __name__ == "__main__":
    # Example usage
    print("Stage 2: Semantic Refinement with RAGAF Fusion")
    print("=" * 60)
    
    # This requires a pretrained UNet
    # For demonstration, we'll just show the structure
    
    print("\nStage 2 model structure:")
    print("1. RAGAF Attention: Region-text association")
    print("2. Adaptive Fusion: Dynamic sketch-text balancing")
    print("3. Feature Injection: Inject region features into UNet")
    print("4. Refinement: Generate final image")
    
    print("\nKey features:")
    print("- Region-aware text conditioning")
    print("- Timestep-adaptive fusion weights")
    print("- Structure preservation from Stage 1")
    
    # # Uncomment to test with actual UNet (requires GPU)
    # from diffusers import UNet2DConditionModel
    # 
    # unet = UNet2DConditionModel.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5",
    #     subfolder="unet"
    # )
    # 
    # stage2 = Stage2SemanticRefinement(
    #     unet=unet,
    #     node_feature_dim=6,
    #     text_dim=768,
    #     hidden_dim=512
    # )
    # 
    # print(f"\nTrainable parameters: {sum(p.numel() for p in stage2.get_trainable_parameters()):,}")
