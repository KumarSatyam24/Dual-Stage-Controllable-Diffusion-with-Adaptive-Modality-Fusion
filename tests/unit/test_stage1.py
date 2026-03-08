"""
Test Stage 1 Trained Model

Quick test to verify the trained model can be loaded and used for inference.
"""

import torch
import os
from pathlib import Path
from models.stage1_diffusion import Stage1SketchGuidedDiffusion
from PIL import Image
import numpy as np

def load_checkpoint(checkpoint_path: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = Stage1SketchGuidedDiffusion(
        pretrained_model_name="runwayml/stable-diffusion-v1-5",
        freeze_base_unet=False,
        use_lora=True,
        lora_rank=4
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch'] + 1}")
    
    return model, checkpoint


def test_model_forward(model, device='cuda'):
    """Test model forward pass."""
    print("\n🧪 Testing model forward pass...")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    sketch = torch.randn(batch_size, 1, 512, 512).to(device)
    text_prompts = ["a photo of a cat"]
    
    with torch.no_grad():
        # Encode sketch
        sketch_features = model.encode_sketch(sketch)
        print(f"✅ Sketch encoding works: {len(sketch_features)} feature scales")
        
        # Encode text
        text_embeddings = model.encode_text(text_prompts)
        print(f"✅ Text encoding works: {text_embeddings.shape}")
        
        # Test UNet forward
        latents = torch.randn(batch_size, 4, 64, 64).to(device)
        timestep = torch.tensor([500]).to(device)
        
        noise_pred = model(
            latents=latents,
            timestep=timestep,
            sketch_features=sketch_features,
            text_embeddings=text_embeddings,
            return_dict=False
        )
        
        print(f"✅ UNet forward works: {noise_pred.shape}")
        print(f"✅ Output requires grad: {noise_pred.requires_grad}")
    
    print("\n✨ All forward pass tests passed!")
    return True


def test_sketch_encoding(model, sketch_path: str = None, device='cuda'):
    """Test sketch encoding with real sketch if available."""
    print("\n🖼️  Testing sketch encoding...")
    
    model = model.to(device)
    model.eval()
    
    if sketch_path and os.path.exists(sketch_path):
        # Load real sketch
        sketch_img = Image.open(sketch_path).convert('L')
        sketch_img = sketch_img.resize((512, 512))
        sketch_array = np.array(sketch_img) / 255.0
        sketch_tensor = torch.from_numpy(sketch_array).float().unsqueeze(0).unsqueeze(0).to(device)
        print(f"✅ Loaded sketch from {sketch_path}")
    else:
        # Create random sketch
        sketch_tensor = torch.randn(1, 1, 512, 512).to(device)
        print("ℹ️  Using random sketch (no real sketch provided)")
    
    with torch.no_grad():
        sketch_features = model.encode_sketch(sketch_tensor)
        
        print(f"✅ Encoded sketch to {len(sketch_features)} feature levels:")
        for i, feat in enumerate(sketch_features):
            print(f"   Level {i}: {feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}")
    
    return True


def main():
    """Main test function."""
    print("="*60)
    print("Testing Stage 1 Trained Model")
    print("="*60)
    
    # Check for checkpoint
    checkpoint_dir = Path("/workspace/outputs/stage1")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("Please train the model first!")
        return
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n📦 Found checkpoint: {latest_checkpoint.name}")
    
    # Load model
    model, checkpoint = load_checkpoint(str(latest_checkpoint))
    
    # Print checkpoint info
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Config: {checkpoint.get('config', 'Not saved')}")
    
    # Test device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    # Run tests
    print("\n" + "="*60)
    print("Running Tests")
    print("="*60)
    
    try:
        # Test 1: Forward pass
        test_model_forward(model, device)
        
        # Test 2: Sketch encoding
        # Try to find a real sketch for testing
        sketch_path = "/workspace/datasets/sketchy/sketch/tx_000000000000/cat/n02121808_11315-1.png"
        test_sketch_encoding(model, sketch_path, device)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n🎉 Your Stage 1 model is working correctly!")
        print(f"📍 Checkpoint: {latest_checkpoint}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Model statistics
    print("\n📊 Model Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")


if __name__ == "__main__":
    main()
