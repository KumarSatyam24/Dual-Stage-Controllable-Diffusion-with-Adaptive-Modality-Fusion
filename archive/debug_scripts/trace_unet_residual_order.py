"""
Understand the exact order UNet collects down_block_res_samples.
"""
import torch
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder='unet'
).cuda()

# Inspect down blocks structure
print("UNet down blocks structure:")
for i, block in enumerate(unet.down_blocks):
    n_resnets = len(block.resnets)
    has_ds = hasattr(block, 'downsamplers') and block.downsamplers is not None
    print(f"  Block {i}: {n_resnets} resnets, downsampler={has_ds}")

# Now trace actual forward to see order
print("\nRunning forward to capture residual collection order...")

latents = torch.zeros(1, 4, 32, 32).cuda()
t = torch.tensor([500]).cuda()
enc = torch.zeros(1, 77, 768).cuda()

# Monkey-patch to intercept
collected_shapes = []
original_call = unet.down_blocks[0].__class__.__call__

def make_hook(block_idx):
    def hook(self, *args, **kwargs):
        result = original_call(self, *args, **kwargs)
        # result is (hidden_states, res_samples_tuple)
        if isinstance(result, tuple) and len(result) == 2:
            for r in result[1]:
                collected_shapes.append((block_idx, tuple(r.shape)))
        return result
    return hook

for i, block in enumerate(unet.down_blocks):
    block.__class__.__call__ = make_hook(i)

with torch.no_grad():
    _ = unet(latents, t, enc, return_dict=False)

print(f"\nCollected {len(collected_shapes)} residuals in this order:")
for i, (block_idx, shape) in enumerate(collected_shapes):
    print(f"  res[{i}]: block{block_idx}, shape={shape}")
