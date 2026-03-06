"""Full traceback for sketch injection error."""
import sys, torch, traceback
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

from models.stage1_diffusion import SketchEncoder
from diffusers import UNet2DConditionModel

enc = SketchEncoder(block_out_channels=[320,640,1280,1280], layers_per_block=2).cuda()
sketch = torch.zeros(1, 1, 256, 256).cuda()
down_res, mid_res = enc(sketch)

print("Our encoder output shapes:")
for i, r in enumerate(down_res):
    print(f"  [{i}]: {tuple(r.shape)}")
print(f"  mid: {tuple(mid_res.shape)}")

unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder='unet'
).cuda()
unet.eval()

latents    = torch.zeros(1, 4, 32, 32).cuda()
timestep   = torch.tensor([500]).cuda()
enc_hidden = torch.zeros(1, 77, 768).cuda()

print("\nRunning UNet forward with sketch residuals...")
try:
    with torch.no_grad():
        out = unet(latents, timestep, enc_hidden,
                   down_block_additional_residuals=down_res,
                   mid_block_additional_residual=mid_res,
                   return_dict=False)
    print(f"SUCCESS: {tuple(out[0].shape)}")
except Exception:
    traceback.print_exc()
