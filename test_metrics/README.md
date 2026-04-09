# Validation Metrics Test Suite

This directory contains standalone test programs for each validation metric used in the RAGAF-Diffusion project. Each script runs the full two-stage pipeline on the Sketchy dataset and computes a specific metric in isolation.

## Quick Start

```bash
# Test SSIM metric
python test_metrics/test_ssim_full.py \
    --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
    --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
    --num_samples 100

# Test all metrics (quick test with 100 samples)
for metric in ssim psnr lpips clip_score edge_similarity fid_kid dists; do
    python test_metrics/test_${metric}_full.py \
        --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
        --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
        --num_samples 100 \
        --output_dir ./results/${metric}_test
done
```

## Validation Metrics Overview

### Standard Metrics

| Metric | File | Range | Interpretation | Dependencies |
|--------|------|-------|----------------|--------------|
| **SSIM** | `test_ssim_full.py` | [-1, 1] | Higher is better (1 = identical) | scikit-image |
| **PSNR** | `test_psnr_full.py` | [0, ∞] dB | Higher is better (typical: 20-40) | scikit-image |
| **LPIPS** | `test_lpips_full.py` | [0, 1] | Lower is better (0 = identical) | lpips |
| **CLIP Score** | `test_clip_score_full.py` | [0, 100] | Higher is better (semantic alignment) | clip |
| **Edge Similarity** | `test_edge_similarity_full.py` | [0, 1] | Higher is better (edge preservation) | opencv-python, scikit-image |

### Advanced Metrics

| Metric | File | Range | Interpretation | Dependencies |
|--------|------|-------|----------------|--------------|
| **FID** | `test_fid_kid_full.py` | [0, ∞] | Lower is better (distribution distance) | torch-fidelity |
| **KID** | `test_fid_kid_full.py` | [0, ∞] | Lower is better (distribution distance) | torch-fidelity |
| **DISTS** | `test_dists_full.py` | [0, 1] | Lower is better (perceptual similarity) | DISTS-pytorch |

## Metric Descriptions

### SSIM (Structural Similarity Index Measure)

Measures perceptual similarity between two images by comparing luminance, contrast, and structure.

- **Range**: -1 to 1
- **Best**: 1.0
- **Formula**: `SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ`
- **When to use**: Pixel-level quality assessment

### PSNR (Peak Signal-to-Noise Ratio)

Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.

- **Range**: 0 to ∞ dB
- **Typical**: 20-40 dB for good quality
- **Formula**: `PSNR = 10 · log10(MAX² / MSE)`
- **When to use**: Traditional image quality metric, sensitive to pixel differences

### LPIPS (Learned Perceptual Image Patch Similarity)

Uses deep network features (AlexNet/VGG) to measure perceptual similarity, better aligned with human perception.

- **Range**: 0 to 1
- **Best**: 0.0
- **Networks**: alex (default), vgg, squeeze
- **When to use**: Perceptual quality assessment, robust to minor pixel shifts

### CLIP Score

Measures semantic alignment between image and text using CLIP embeddings.

- **Range**: 0 to 100 (scaled cosine similarity)
- **Best**: Higher values
- **Models**: ViT-B/32 (default), ViT-B/16, ViT-L/14, RN50, RN101
- **When to use**: Text-image alignment, semantic correctness

### Edge Similarity

Measures edge preservation using Canny edge detection and SSIM on edge maps.

- **Range**: 0 to 1
- **Best**: 1.0
- **Parameters**: Canny thresholds (default: low=50, high=150)
- **When to use**: Sketch-to-image quality, structure preservation

### FID (Fréchet Inception Distance)

Measures distance between feature distributions of real and generated images using Inception V3.

- **Range**: 0 to ∞
- **Best**: Lower values (good: <50, excellent: <20)
- **When to use**: Overall distribution quality, diversity assessment

### KID (Kernel Inception Distance)

Similar to FID but uses Maximum Mean Discrepancy with polynomial kernel, more stable for small samples.

- **Range**: 0 to ∞
- **Best**: Lower values (closer to 0)
- **When to use**: Small sample sizes, when FID is unstable

### DISTS (Deep Image Structure and Texture Similarity)

Separates structure and texture information using VGG features, better aligned with human perception than LPIPS.

- **Range**: 0 to 1
- **Best**: 0.0
- **When to use**: High-quality perceptual assessment, texture-sensitive tasks

## Installation

```bash
# Standard metrics
pip install scikit-image opencv-python

# LPIPS
pip install lpips

# CLIP
pip install git+https://github.com/openai/CLIP.git

# FID/KID (torch-fidelity)
pip install torch-fidelity

# DISTS
pip install DISTS-pytorch
```

## Common Arguments

All scripts support these arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage1_checkpoint` | Required | Path to Stage 1 checkpoint |
| `--stage2_checkpoint` | Required | Path to Stage 2 checkpoint |
| `--output_dir` | `./results/<metric>_test` | Output directory for results |
| `--batch_size` | 4 | Batch size for generation |
| `--num_samples` | -1 | Number of samples to evaluate (-1 for all) |
| `--image_size` | 256 | Image size for generation |
| `--num_inference_steps` | 50 | Number of diffusion steps for Stage 1 |
| `--guidance_scale` | 7.5 | Classifier-free guidance scale |
| `--device` | cuda | Device to use (cuda/cpu) |

## Output Format

Each script produces:

1. **JSON results file** (`<metric>_results.json`):
   ```json
   {
     "summary": {
       "stage2_<metric>": {
         "mean": 0.85,
         "std": 0.05,
         "min": 0.65,
         "max": 0.95
       },
       "stage1_<metric>": {...},
       "<metric>_improvement": 0.10
     },
     "config": {...},
     "execution_time_seconds": 3600,
     "timestamp": "2024-01-01T00:00:00"
   }
   ```

2. **CSV per-sample results** (`per_sample_<metric>.csv`):
   ```csv
   file_id,category,prompt,stage2_<metric>,stage1_<metric>
   n03770679_1,monitor,monitor,0.85,0.75
   ```

3. **Intermediate results** (saved every 10 batches):
   - `_intermediate_results.json` - Current running averages

## Running on Full Test Set

The Sketchy dataset test split contains 11,424 images. To evaluate on the full set:

```bash
python test_metrics/test_ssim_full.py \
    --stage1_checkpoint /workspace/checkpoints/stage1/epoch_18.pt \
    --stage2_checkpoint /workspace/checkpoints/stage2/epoch_6.pt \
    --num_samples -1 \
    --batch_size 8 \
    --output_dir ./results/ssim_full
```

**Note**: Full evaluation may take several hours depending on GPU and batch size.

## Running Specific Stage

All scripts compute metrics for both Stage 1 and Stage 2 outputs, allowing you to analyze the improvement from semantic refinement.

## Interpreting Results

### Typical Values for Sketch-to-Image

| Metric | Stage 1 | Stage 2 | Improvement |
|--------|---------|---------|-------------|
| SSIM | 0.60-0.70 | 0.70-0.80 | +0.10 |
| PSNR | 18-22 dB | 22-26 dB | +4 dB |
| LPIPS | 0.25-0.35 | 0.15-0.25 | -0.10 |
| CLIP Score | 0.25-0.30 | 0.28-0.32 | +0.03 |
| Edge Similarity | 0.65-0.75 | 0.75-0.85 | +0.10 |
| FID | 40-60 | 20-40 | -20 |
| DISTS | 0.15-0.25 | 0.10-0.15 | -0.05 |

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 2 or 1)
- Use `--num_samples` to test subset first

### Slow Generation
- Reduce `--num_inference_steps` (try 25 instead of 50)
- Use GPU with more VRAM

### Missing Dependencies
```bash
# Check what's installed
python -c "import lpips; print('LPIPS OK')"
python -c "import clip; print('CLIP OK')"
python -c "import DISTS_pytorch; print('DISTS OK')"
python -c "import torch_fidelity; print('torch-fidelity OK')"
```

## References

- **SSIM**: Wang et al., "Image quality assessment: from error visibility to structural similarity", TIP 2004
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- **FID**: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule", NeurIPS 2017
- **KID**: Bińkowski et al., "Demystifying MMD GANs", ICLR 2018
- **DISTS**: Ding et al., "Image Quality Assessment: Unifying Structure and Texture Similarity", TPAMI 2022
