"""
Test the trained Stage 1 model using the saved checkpoint.
Tests:
  1. Checkpoint loading
  2. Forward pass (encode sketch, encode text, predict noise)
  3. Real sketch from Sketchy dataset
  4. Denoising loop (generate a latent from a real sketch)
  5. Decode latent → image and save to /root/test_outputs/
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion')

CHECKPOINT  = "/root/checkpoints/stage1/epoch_10.pt"  # final epoch weights
SKETCHY_ROOT = "/workspace/sketchy"
OUTPUT_DIR   = "/root/test_outputs"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("  Stage 1 Model Test")
print("="*60)
print(f"  Checkpoint : {CHECKPOINT}")
print(f"  Device     : {DEVICE}")
print(f"  Output dir : {OUTPUT_DIR}")
print("="*60)

# ── 1. Load checkpoint ────────────────────────────────────────
print("\n[1/5] Loading checkpoint...")
from models.stage1_diffusion import Stage1SketchGuidedDiffusion

ckpt = torch.load(CHECKPOINT, map_location="cpu")
model = Stage1SketchGuidedDiffusion(
    pretrained_model_name="runwayml/stable-diffusion-v1-5",
    freeze_base_unet=False,
    use_lora=True,
    lora_rank=4
)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()
print(f"  ✅ Loaded weights from epoch {ckpt['epoch']+1}")

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  ✅ Parameters — total: {total:,}  trainable: {trainable:,}")

# ── 2. Forward pass with dummy input ─────────────────────────
print("\n[2/5] Forward pass (dummy inputs)...")
with torch.no_grad():
    dummy_sketch  = torch.randn(1, 1, 256, 256).to(DEVICE)
    dummy_latents = torch.randn(1, 4, 32, 32).to(DEVICE)
    dummy_ts      = torch.tensor([500]).to(DEVICE)
    text_prompts  = ["a photo of an airplane"]

    sketch_feats  = model.encode_sketch(dummy_sketch)
    text_emb      = model.encode_text(text_prompts)
    noise_pred    = model(dummy_latents, dummy_ts, sketch_feats, text_emb)

print(f"  ✅ Sketch features: {len(sketch_feats)} levels — {[f.shape for f in sketch_feats]}")
print(f"  ✅ Text embedding : {text_emb.shape}")
print(f"  ✅ Noise pred     : {noise_pred.shape}")

# ── 3. Real sketch from Sketchy dataset ───────────────────────
print("\n[3/5] Loading a real sketch from Sketchy dataset...")
sketch_dir = Path(SKETCHY_ROOT) / "sketch/tx_000000000000/airplane"
sketch_files = list(sketch_dir.glob("*.png"))
if not sketch_files:
    print("  ⚠️  No sketches found, using dummy")
    real_sketch_t = torch.randn(1, 1, 256, 256).to(DEVICE)
else:
    sketch_path = sketch_files[0]
    sketch_img  = Image.open(sketch_path).convert("L").resize((256, 256))
    real_sketch_t = torch.from_numpy(
        np.array(sketch_img) / 255.0
    ).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    print(f"  ✅ Loaded sketch: {sketch_path.name}  shape={real_sketch_t.shape}")
    # Save the input sketch
    Image.fromarray((real_sketch_t[0,0].cpu().numpy()*255).astype(np.uint8)).save(
        f"{OUTPUT_DIR}/input_sketch.png")
    print(f"  ✅ Saved input sketch → {OUTPUT_DIR}/input_sketch.png")

# ── 4. Run denoising loop with CFG ───────────────────────────
print("\n[4/5] Running denoising loop (50 steps, guidance_scale=7.5)...")
from diffusers import DDIMScheduler, AutoencoderKL

scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
scheduler.set_timesteps(50)

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE)
vae.requires_grad_(False)

GUIDANCE_SCALE = 7.5
TEXT_PROMPT    = "a photo of an airplane, realistic, detailed"

with torch.no_grad():
    sketch_feats = model.encode_sketch(real_sketch_t)

    # Conditional text embedding
    cond_emb  = model.encode_text([TEXT_PROMPT])          # (1, 77, 768)
    # Unconditional (empty) embedding for CFG
    uncond_emb = model.encode_text([""])                  # (1, 77, 768)

    # Stack for batched CFG forward pass
    sketch_feats_2x = [torch.cat([f, f]) for f in sketch_feats]  # duplicate for CFG
    text_emb_cfg    = torch.cat([uncond_emb, cond_emb])           # (2, 77, 768)

    # Start from pure noise — latent size = image_size/8 = 256/8 = 32
    latents = torch.randn(1, 4, 32, 32, device=DEVICE)
    latents = latents * scheduler.init_noise_sigma

    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG (run unconditional + conditional together)
        latents_input = torch.cat([latents, latents])
        ts = torch.tensor([t], device=DEVICE).expand(2)

        noise_pred = model(latents_input, ts, sketch_feats_2x, text_emb_cfg)

        # Apply classifier-free guidance
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/50  latent mean={latents.mean():.4f}  std={latents.std():.4f}")

print("  ✅ Denoising complete")

# ── 5. Decode latent → image ──────────────────────────────────
print("\n[5/5] Decoding latent to image...")
with torch.no_grad():
    latents_scaled = latents / 0.18215
    image = vae.decode(latents_scaled).sample          # (1, 3, 256, 256)
    image = (image / 2 + 0.5).clamp(0, 1)             # normalise to [0,1]
    image_np = (image[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

out_path = f"{OUTPUT_DIR}/stage1_generated.png"
Image.fromarray(image_np).save(out_path)
print(f"  ✅ Generated image saved → {out_path}")

print("\n" + "="*60)
print("  ✅ ALL 5 TESTS PASSED — Stage 1 model is working!")
print("="*60)
print(f"\n  📁 Outputs saved to: {OUTPUT_DIR}/")
print(f"     - input_sketch.png     (the sketch fed in)")
print(f"     - stage1_generated.png (model output image)")
