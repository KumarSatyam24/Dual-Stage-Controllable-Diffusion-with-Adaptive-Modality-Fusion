
import torch
import sys
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stage2_refinement import Stage2SemanticRefinement, Stage2RefinementPipeline
from src.data.region_graph import RegionGraph
from diffusers import UNet2DConditionModel, AutoencoderKL

def test_stage2_refine():
    print("Testing Stage 2 Refinement Pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Model
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    ).to(device)
    
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae"
    ).to(device)
    
    model = Stage2SemanticRefinement(
        unet=unet,
        node_feature_dim=6,
        text_dim=768,
        hidden_dim=512
    ).to(device)
    
    pipeline = Stage2RefinementPipeline(
        stage2_model=model,
        vae=vae,
        num_inference_steps=10, # small steps for test
        device=device
    )
    
    # 2. Setup Dummy Inputs
    stage1_image = torch.rand(1, 3, 256, 256).to(device) # [0, 1]
    text_embeddings = torch.randn(1, 77, 768).to(device)
    
    num_nodes = 3
    region_graph = RegionGraph(
        num_nodes=num_nodes,
        node_features=torch.randn(num_nodes, 6),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_weights=torch.rand(4),
        region_masks=[torch.randint(0, 2, (256, 256)).numpy() for _ in range(num_nodes)]
    )
    
    # 3. Run Refine
    print("Running refinement...")
    refined_image = pipeline.refine(
        stage1_image,
        region_graph,
        "A photo of a dog",
        text_embeddings,
        strength=0.5
    )
    
    print("✅ Refinement successful")
    print(f"   Refined image shape: {refined_image.shape}")
    
    if torch.any(refined_image != stage1_image):
        print("✅ Refined image is different from Stage 1 image")
    else:
        print("❌ Refined image is same as Stage 1 image!")

if __name__ == "__main__":
    test_stage2_refine()
