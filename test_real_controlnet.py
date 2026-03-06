"""Test what actual ControlNet outputs and verify it works with UNet."""
import torch
from diffusers import ControlNetModel, UNet2DConditionModel

print("Loading ControlNet...")
cn = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny').cuda()
cn.eval()

# ControlNet expects latent-space input (4 channels) + conditioning image
sample = torch.zeros(1, 4, 32, 32).cuda()
t = torch.tensor([500]).cuda()
enc = torch.zeros(1, 77, 768).cuda()
controlnet_cond = torch.zeros(1, 3, 256, 256).cuda()

print("Running ControlNet forward...")
with torch.no_grad():
    down_samples, mid_sample = cn(
        sample, t, enc, controlnet_cond, return_dict=False
    )

print(f"\nControlNet outputs {len(down_samples)} down residuals:")
for i, s in enumerate(down_samples):
    print(f"  [{i}]: {tuple(s.shape)}")
print(f"Mid: {tuple(mid_sample.shape)}")

# Now test with UNet
print("\nLoading UNet...")
unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder='unet'
).cuda()
unet.eval()

print("Testing ControlNet residuals with UNet...")
try:
    with torch.no_grad():
        out = unet(sample, t, enc,
                   down_block_additional_residuals=down_samples,
                   mid_block_additional_residual=mid_sample,
                   return_dict=False)
    print(f"✅ SUCCESS: {tuple(out[0].shape)}")
    print("\nThis confirms the ControlNet interface works correctly!")
except Exception as e:
    print(f"❌ FAILED: {e}")
