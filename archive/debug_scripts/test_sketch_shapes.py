"""
Verify SketchEncoder residual shapes match UNet expectations,
then do a live UNet forward pass with sketch injection.
"""
import sys, torch
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

print("Step 1: SketchEncoder shape check")
from models.stage1_diffusion import SketchEncoder

enc = SketchEncoder(block_out_channels=[320, 640, 1280, 1280], layers_per_block=2).cuda()
sketch = torch.zeros(1, 1, 256, 256).cuda()
down_res, mid_res = enc(sketch)

# Expected shapes (12 residuals matching ControlNet output)
expected = [
    (1,320,32,32),(1,320,32,32),(1,320,32,32),(1,320,16,16),
    (1,640,16,16),(1,640,16,16),(1,640,8,8),
    (1,1280,8,8),(1,1280,8,8),(1,1280,4,4),
    (1,1280,4,4),(1,1280,4,4),
]
all_ok = True
for i, (r, e) in enumerate(zip(down_res, expected)):
    ok = tuple(r.shape) == e
    if not ok:
        all_ok = False
        print(f"  MISMATCH [{i}]: got {tuple(r.shape)}, expected {e}")
print(f"  Down residuals: {len(down_res)}/12  {'ALL OK' if all_ok else 'SHAPE MISMATCH'}")
print(f"  Mid residual:   {tuple(mid_res.shape)}  {'OK' if tuple(mid_res.shape)==(1,1280,4,4) else 'MISMATCH'}")

print("\nStep 2: UNet forward WITH sketch residuals")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder='unet'
).cuda()
unet.eval()

latents   = torch.zeros(1, 4, 32, 32).cuda()
timestep  = torch.tensor([500]).cuda()
enc_hidden = torch.zeros(1, 77, 768).cuda()

try:
    with torch.no_grad():
        out = unet(latents, timestep, enc_hidden,
                   down_block_additional_residuals=down_res,
                   mid_block_additional_residual=mid_res,
                   return_dict=False)
    print(f"  Output shape: {tuple(out[0].shape)}")
    print("\n✅  SUCCESS — Sketch injection into UNet works correctly!")
except Exception as e:
    print(f"\n❌  FAILED — {type(e).__name__}: {e}")
