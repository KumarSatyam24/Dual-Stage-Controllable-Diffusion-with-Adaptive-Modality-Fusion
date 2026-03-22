#!/usr/bin/env python3
"""
Verify Stage 2 model can be loaded and initialized correctly.
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

try:
    from src.models.stage2_refinement import Stage2SemanticRefinement
    from diffusers import UNet2DConditionModel
    import torch
    
    print("[*] Loading UNet from Stable Diffusion v1-5...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    )
    
    print("[*] Creating Stage 2 model...")
    stage2_model = Stage2SemanticRefinement(
        unet=unet,
        node_feature_dim=6,
        text_dim=768,
        hidden_dim=512,
        num_graph_layers=2,
        num_attention_heads=8,
        fusion_method="learned",
        use_region_adaptive_fusion=True
    )
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in stage2_model.parameters())
    trainable_params = sum(p.numel() for p in stage2_model.parameters() if p.requires_grad)
    
    print(f"✅ Stage 2 model verified")
    print(f"   Total parameters: {total_params/1e6:.1f}M")
    print(f"   Trainable parameters: {trainable_params/1e6:.1f}M")
    
except Exception as e:
    import traceback
    print(f"❌ Stage 2 model verification failed: {e}")
    traceback.print_exc()
    sys.exit(1)
