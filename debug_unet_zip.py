"""
Patch UNet to print exactly which residual causes the size mismatch.
"""
import torch, sys
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

from models.stage1_diffusion import SketchEncoder
from diffusers import UNet2DConditionModel

enc = SketchEncoder(block_out_channels=[320,640,1280,1280], layers_per_block=2).cuda()
sketch = torch.zeros(1, 1, 256, 256).cuda()
down_res, mid_res = enc(sketch)

unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder='unet'
).cuda()

# Patch the zip loop to print which index fails
import diffusers.models.unets.unet_2d_condition
orig_forward = unet.forward

def patched_forward(self, *args, **kwargs):
    # Call original but intercept at the zip point
    down_block_additional_residuals = kwargs.get('down_block_additional_residuals')
    
    if down_block_additional_residuals is not None:
        print(f"down_block_additional_residuals provided: {len(down_block_additional_residuals)} items")
        for i, r in enumerate(down_block_additional_residuals):
            print(f"  add_res[{i}]: {tuple(r.shape)}")
    
    # Call original - it will fail, but we'll see the print first
    return orig_forward(*args, **kwargs)

unet.forward = lambda *args, **kwargs: patched_forward(unet, *args, **kwargs)

latents = torch.zeros(1, 4, 32, 32).cuda()
t = torch.tensor([500]).cuda()
enc_hidden = torch.zeros(1, 77, 768).cuda()

print("Calling UNet with our sketch residuals...")
try:
    with torch.no_grad():
        out = unet(latents, t, enc_hidden,
                   down_block_additional_residuals=down_res,
                   mid_block_additional_residual=mid_res,
                   return_dict=False)
    print("SUCCESS!")
except RuntimeError as e:
    print(f"\nFAILED: {e}")
    print("\nThis means the UNet's internal down_block_res_samples has different shapes")
    print("than what we're providing. Need to trace UNet's actual forward pass.")
