"""Test updated 12-residual sketch encoder."""
import torch, sys
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

from models.stage1_diffusion import SketchEncoder
from diffusers import UNet2DConditionModel

print("Step 1: Create 12-residual sketch encoder")
enc = SketchEncoder(block_out_channels=[320,640,1280,1280], layers_per_block=2).cuda()
sketch = torch.zeros(1, 1, 256, 256).cuda()
down_res, mid_res = enc(sketch)

print(f"  Got {len(down_res)} down residuals (expect 12)")
for i, r in enumerate(down_res):
    print(f"    [{i}]: {tuple(r.shape)}")
print(f"  Mid: {tuple(mid_res.shape)}")

# Expected (from real ControlNet)
expected = [
    (1,320,32,32),(1,320,32,32),(1,320,32,32),(1,320,16,16),
    (1,640,16,16),(1,640,16,16),(1,640,8,8),
    (1,1280,8,8),(1,1280,8,8),(1,1280,4,4),
    (1,1280,4,4),(1,1280,4,4),
]

all_ok = True
for i, (r, e) in enumerate(zip(down_res, expected)):
    if tuple(r.shape) != e:
        all_ok = False
        print(f"  MISMATCH [{i}]: got {tuple(r.shape)}, expected {e}")

if all_ok:
    print("  ✅ All shapes match ControlNet output!")
else:
    print("  ❌ Shape mismatch")

print("\nStep 2: Test with UNet")
unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet').cuda()
latents = torch.zeros(1, 4, 32, 32).cuda()
t = torch.tensor([500]).cuda()
enc_hidden = torch.zeros(1, 77, 768).cuda()

try:
    with torch.no_grad():
        out = unet(latents, t, enc_hidden,
                   down_block_additional_residuals=down_res,
                   mid_block_additional_residual=mid_res,
                   return_dict=False)
    print(f"  ✅ SUCCESS: {tuple(out[0].shape)}")
    print("\n🎉 Sketch injection NOW WORKS!")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
