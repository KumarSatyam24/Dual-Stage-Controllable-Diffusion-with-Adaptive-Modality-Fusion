"""Quick shape-check for the new SketchEncoder."""
import torch, sys
sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')
from models.stage1_diffusion import SketchEncoder

enc = SketchEncoder(
    in_channels=1,
    block_out_channels=[320, 640, 1280, 1280],
    layers_per_block=2,
)
sketch = torch.zeros(1, 1, 256, 256)
down_res, mid_res = enc(sketch * 2.0 - 1.0)

print(f"down residuals: {len(down_res)}  (expected 11)")
for i, r in enumerate(down_res):
    print(f"  down_res[{i:2d}]: {tuple(r.shape)}")
print(f"mid_res:        {tuple(mid_res.shape)}")

# Expected shapes (batch=1, 256x256 input, latent 32x32)
expected_down = [
    (1,320,256,256),(1,320,256,256),(1,320,128,128),   # block0 (2 resnets + ds)
    (1,640,128,128),(1,640,128,128),(1,640,64,64),     # block1
    (1,1280,64,64), (1,1280,64,64), (1,1280,32,32),   # block2
    (1,1280,32,32), (1,1280,32,32),                    # block3 (no ds)
]
print("\nShape validation:")
all_ok = True
for i, (got, exp) in enumerate(zip(down_res, expected_down)):
    ok = tuple(got.shape) == exp
    if not ok:
        all_ok = False
    print(f"  [{i}]: {'OK' if ok else 'MISMATCH'} — got {tuple(got.shape)}, expected {exp}")
print(f"\n{'ALL OK' if all_ok else 'SHAPE MISMATCH — CHECK ABOVE'}")
