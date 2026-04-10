# RAGAF-Diffusion: Dual-Stage Controllable Sketch-to-Image Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a **dual-stage diffusion pipeline** for sketch-to-image generation with **Region-Adaptive Graph-Attention Fusion (RAGAF)**.

---

## 1) Core Concept (What this project does)

The system separates generation into two complementary stages:

- **Stage 1 (Structure First):** sketch-guided diffusion generates a coarse, structure-aligned image.
- **Stage 2 (Semantics Next):** region-aware graph attention and adaptive fusion refine Stage 1 output using text semantics.

The key idea is to avoid uniform text conditioning across the whole image by modeling sketch regions and their relations explicitly.

---

## 2) Code-Backed Architecture (Where logic lives)

### Stage 1: Sketch-guided coarse generation
- `src/models/stage1_diffusion.py`
  - `SketchEncoder`: ControlNet-style encoder producing residuals for UNet down/mid blocks.
  - `Stage1SketchGuidedDiffusion`: Stable Diffusion v1.5 UNet + sketch residual injection.
  - `Stage1DiffusionPipeline`: DDIM-based inference loop with classifier-free guidance.

### Stage 2: Semantic refinement
- `src/models/stage2_refinement.py`
  - `Stage2SemanticRefinement`: takes noisy latents + Stage 1 latents + region graph + text embeddings.
  - Builds modulation maps from region-wise fused features and applies latent residual refinement.

### Region-aware fusion modules
- `src/models/ragaf_attention.py`
  - `RegionGraphAttention`: multi-head graph attention over sketch regions.
  - `RegionTextCrossAttention`: cross-attention from regions to text tokens.
  - `RAGAFAttentionModule`: stacked graph reasoning + region-text association.
- `src/models/adaptive_fusion.py`
  - `AdaptiveFusionWeights`: timestep-conditioned sketch/text fusion weights (`learned`, `heuristic`, `hybrid`).
  - `AdaptiveModalityFusion`: combines sketch region features and text-aligned features per timestep.

### Data and graph construction
- `src/data/sketch_extraction.py`: sketch extraction utilities (Canny/XDoG; `hed` is a known limitation and currently falls back to Canny, not true HED inference).
- `src/data/region_extraction.py`: connected-components-based region extraction with filtering/merging.
- `src/data/region_graph.py`: region graph builder (`adjacency`, `knn`, `radius`, `hybrid`) and node features.

### End-to-end scripts
- `scripts/training/train.py`: stage-wise training orchestration.
- `scripts/inference/inference.py`: full inference pipeline and outputs.
- `validate_comprehensive.py`: broad validation script.
- `test_metrics/*.py`: metric-specific evaluators.

---

## 3) Pipeline Logic (How generation works)

1. **Input sketch preprocessing**
   - Normalize sketch, extract connected regions, compute region features, build region graph.
2. **Stage 1 denoising**
   - Inject sketch-conditioned residuals into SD UNet blocks.
   - Produce a coarse image that preserves layout and structure.
3. **Stage 2 semantic refinement**
   - Encode region graph with graph attention.
   - Align region features with text token embeddings via cross-attention.
   - Compute timestep-aware adaptive fusion between sketch and text features.
   - Project fused region features to latent modulation and refine output.
4. **Final decode**
   - Decode refined latents to RGB image using VAE.

---

## 4) Validation Snapshot (from repository artifacts)

The following values are extracted from checked-in result files under `results/`.

| Metric | Stage 1 | Stage 2 | Change | Source |
|---|---:|---:|---:|---|
| CLIP Score (mean) | 32.3774 | 32.4223 | +0.0449 | `results/clip_score_full/clip_score_results.json` |
| DISTS (mean, lower better) | 0.3785 | 0.3771 | -0.0014 (reduction) | `results/dists_full/dists_results.json` |
| PSNR | 9.8789 | 9.9652 | +0.0863 | `results/lpips_psnr_ssim_test/test_lpips_psnr_ssim.json` |
| SSIM | 0.4340 | 0.4352 | +0.0012 | `results/lpips_psnr_ssim_test/test_lpips_psnr_ssim.json` |
| LPIPS (lower better) | 0.6242 | 0.6228 | -0.0014 (reduction) | `results/lpips_psnr_ssim_test/test_lpips_psnr_ssim.json` |
| Inception Score (mean) | 1.3374 | 1.3808 | +0.0433 | `results/is_test/is_results.json` |
| Edge Similarity (mean) | 0.21127 | 0.21104 | -0.00023 | `results/edge_similarity_test/edge_similarity_results.json` |

Additional distribution metrics:
- **FID:** 18.7028
- **KID mean:** 0.003278
- Source: `results/fid_kid_test/fid_kid_results.json`

> Note: some metrics show modest improvement, while edge similarity shows a very small degradation in this snapshot.
>
> CLIP scores above are on the script's scaled reporting convention (not raw cosine values).
>
> PSNR values are reported as produced by the current evaluation pipeline and should be compared primarily stage-to-stage within this project setup.

---

## 5) Reproducing Validation

Use metric-specific scripts in `test_metrics/`:

```bash
python test_metrics/test_lpips_psnr_ssim.py \
  --stage1_checkpoint /path/to/stage1.pt \
  --stage2_checkpoint /path/to/stage2.pt \
  --output_dir ./results/lpips_psnr_ssim_test

python test_metrics/test_clip_score_full.py \
  --stage1_checkpoint /path/to/stage1.pt \
  --stage2_checkpoint /path/to/stage2.pt \
  --output_dir ./results/clip_score_full

python test_metrics/test_fid_kid_full.py \
  --stage1_checkpoint /path/to/stage1.pt \
  --stage2_checkpoint /path/to/stage2.pt \
  --output_dir ./results/fid_kid_test
```

For metric definitions and dependencies, see `test_metrics/README.md`.

---

## 6) Quick Start

```bash
git clone https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion.git
cd Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
python scripts/training/train.py --stage stage1
python scripts/training/train.py --stage stage2
```

### Inference
```bash
python scripts/inference/inference.py \
  --sketch /path/to/sketch.png \
  --prompt "A realistic photo of the sketched object" \
  --stage1_checkpoint /path/to/stage1.pt \
  --stage2_checkpoint /path/to/stage2.pt \
  --output_dir ./outputs
```

---

## 7) Repository Layout

```
.
├── README.md
├── requirements.txt
├── app.py
├── scripts/
│   ├── training/train.py
│   ├── inference/inference.py
│   └── evaluation/
├── src/
│   ├── configs/
│   ├── data/
│   ├── datasets/
│   ├── models/
│   └── utils/
├── test_metrics/
├── tests/
├── tools/
├── results/
└── docs/
```

---

## 8) Validation in this agent environment

A pre-edit unit-test invocation was attempted:

```bash
pytest tests/unit -q
```

It failed here because `pytest` is not installed in the current sandbox (`pytest: command not found`).

---

## License

MIT License.
