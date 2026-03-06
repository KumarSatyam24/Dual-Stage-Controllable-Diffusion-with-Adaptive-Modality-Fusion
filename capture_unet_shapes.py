"""Capture exact residual shapes from UNet down blocks."""
import torch, functools, sys
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder='unet'
).cuda()
unet.eval()

latents    = torch.zeros(1, 4, 32, 32).cuda()
timestep   = torch.tensor([500]).cuda()
enc_hidden = torch.zeros(1, 77, 768).cuda()

captured = []

for block in unet.down_blocks:
    _orig = block.forward
    def make_hook(orig):
        def hook(*args, **kwargs):
            result = orig(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                res_samples = result[1]
                for r in res_samples:
                    captured.append(tuple(r.shape))
            return result
        return hook
    block.forward = make_hook(_orig)

with torch.no_grad():
    unet(latents, timestep, enc_hidden, return_dict=False)

print("Exact residual shapes UNet expects (down_block_additional_residuals):")
for i, s in enumerate(captured):
    print(f"  [{i}]: {s}")
print(f"Total: {len(captured)}")
