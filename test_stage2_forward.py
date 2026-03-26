
import torch
import sys
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stage2_refinement import Stage2SemanticRefinement
from src.data.region_graph import RegionGraph
from diffusers import UNet2DConditionModel

def test_stage2_forward():
    print("Testing Stage 2 Forward Pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Model
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    ).to(device)
    
    model = Stage2SemanticRefinement(
        unet=unet,
        node_feature_dim=6,
        text_dim=768,
        hidden_dim=512
    ).to(device)
    
    # 2. Setup Dummy Inputs
    batch_size = 2
    latents = torch.randn(batch_size, 4, 32, 32).to(device)
    timestep = torch.tensor([10, 20]).to(device)
    text_embeddings = torch.randn(batch_size, 77, 768).to(device)
    stage1_latents = torch.randn(batch_size, 4, 32, 32).to(device)
    
    # Create dummy region graphs
    region_graphs = []
    for i in range(batch_size):
        num_nodes = 3
        rg = RegionGraph(
            num_nodes=num_nodes,
            node_features=torch.randn(num_nodes, 6),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_weights=torch.rand(4),
            region_masks=[torch.randint(0, 2, (256, 256)).numpy() for _ in range(num_nodes)]
        )
        region_graphs.append(rg)
    
    # 3. Run Forward
    print("Running forward...")
    output = model(
        latents,
        timestep,
        region_graphs,
        text_embeddings,
        stage1_latents=stage1_latents,
        return_dict=True
    )
    
    print("✅ Forward pass successful")
    print(f"   Noise pred shape: {output['noise_pred'].shape}")
    print(f"   Modulation map shape: {output['modulation_map'].shape}")
    
    # Check if modulation map is not all zeros
    if torch.any(output['modulation_map'] != 0):
        print("✅ Modulation map is active")
    else:
        print("❌ Modulation map is all zeros!")

if __name__ == "__main__":
    test_stage2_forward()
