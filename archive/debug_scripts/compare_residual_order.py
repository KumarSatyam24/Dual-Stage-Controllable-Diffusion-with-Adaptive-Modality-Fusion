"""
Compare our encoder output order with UNet's internal residual order.
"""
import torch, sys
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

from models.stage1_diffusion import SketchEncoder
from diffusers import UNet2DConditionModel

# Our encoder
enc = SketchEncoder(block_out_channels=[320, 640, 1280, 1280], layers_per_block=2).cuda()
sketch = torch.zeros(1, 1, 256, 256).cuda()
our_down, our_mid = enc(sketch)

print("OUR ENCODER produces:")
for i, r in enumerate(our_down):
    print(f"  [{i}]: {tuple(r.shape)}")

# UNet internal residuals
unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder='unet'
).cuda()

latents = torch.zeros(1, 4, 32, 32).cuda()
t = torch.tensor([500]).cuda()
enc_hidden = torch.zeros(1, 77, 768).cuda()

# Capture UNet's internal down_block_res_samples
unet_residuals = []
orig_mid_forward = unet.mid_block.forward

def capture_down_res(*args, **kwargs):
    # At this point down_block_res_samples should be collected
    # We'll hook earlier in the down blocks loop
    return orig_mid_forward(*args, **kwargs)

# Hook each down block to capture its output residuals
for block_idx, block in enumerate(unet.down_blocks):
    orig_forward = block.forward
    def make_hook(idx, orig):
        def hook(*args, **kwargs):
            result = orig(*args, **kwargs)
            # result[1] is the tuple of residuals from this block
            if isinstance(result, tuple) and len(result) >= 2:
                for r in result[1]:
                    unet_residuals.append((idx, tuple(r.shape)))
            return result
        return hook
    block.forward = make_hook(block_idx, orig_forward)

with torch.no_grad():
    _ = unet(latents, t, enc_hidden, return_dict=False)

print("\nUNet INTERNALLY collects:")
for i, (block_idx, shape) in enumerate(unet_residuals):
    print(f"  [{i}]: {shape}  (from block{block_idx})")

print("\nCOMPARISON:")
all_match = True
for i in range(min(len(our_down), len(unet_residuals))):
    ours = tuple(our_down[i].shape)
    theirs = unet_residuals[i][1]
    match = ours == theirs
    if not match:
        all_match = False
    print(f"  [{i}]: ours={ours}, theirs={theirs}  {'✓' if match else '✗ MISMATCH'}")

if all_match:
    print("\n✅ All shapes match! The error must be elsewhere.")
else:
    print("\n❌ Shape mismatch found — need to reorder our encoder output!")
