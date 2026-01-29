# RAGAF-Diffusion: Dual-Stage Controllable Diffusion with Region-Adaptive Graph-Attention Fusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

RAGAF-Diffusion is a novel diffusion framework that explicitly models sketch-derived regions as nodes in a graph and uses **region-aware graph attention** to fuse sketch structure and text semantics during image generation.

### Key Innovations

1. **Dual-Stage Pipeline**
   - **Stage 1**: Sketch-guided diffusion for coarse structure-preserving layout
   - **Stage 2**: Semantic refinement using text prompts with structure preservation

2. **Region-Adaptive Graph-Attention Fusion (RAGAF)**
   - Automatic region extraction from sketches (no manual annotation)
   - Graph construction with regions as nodes, spatial relationships as edges
   - Graph attention for region-aware text conditioning
   - Text tokens influence only relevant regions

3. **Adaptive Modality Fusion**
   - Dynamic balance between sketch and text features across timesteps
   - Early timesteps: Strong sketch guidance for structure
   - Late timesteps: Strong text guidance for details
   - Region-specific adaptive weighting

## Architecture

```
Input: Sketch + Text Prompt
         │
         ├─► [Sketch Region Extraction]
         │   └─► Connected Components → Region Graph
         │
         ├─► [Stage 1: Sketch-Guided Diffusion]
         │   ├─► Sketch Encoder (ControlNet-style)
         │   ├─► UNet with Sketch Conditioning
         │   └─► Output: Coarse Structure-Preserving Image
         │
         └─► [Stage 2: Semantic Refinement]
             ├─► RAGAF Attention Module
             │   ├─► Graph Attention over Regions
             │   └─► Region-Text Cross-Attention
             ├─► Adaptive Fusion (timestep-aware)
             │   └─► Dynamic Sketch-Text Balancing
             └─► Output: Final Refined Image
```

## Datasets

- **Primary**: [Sketchy Dataset](https://sketchy.eye.gatech.edu/) - Sketch-photo pairs for 125 object categories
- **Secondary**: [MS COCO](https://cocodataset.org/) - Auto-generated sketches + captions for complex scenes

## Environment Setup

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory (RTX 3090/4090 recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/KumarSatyam24/RAGAF-Diffusion.git
cd RAGAF-Diffusion/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

### Dataset Setup

**Sketchy Dataset:**
```bash
# Download from https://sketchy.eye.gatech.edu/
# Expected structure:
# sketchy/
# ├── sketch/tx_000000000000/
# │   ├── airplane/
# │   └── ...
# └── photo/tx_000000000000/
#     ├── airplane/
#     └── ...

export SKETCHY_ROOT=/path/to/sketchy
```

**MS COCO:**
```bash
# Download COCO 2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Expected structure:
# coco/
# ├── train2017/
# ├── val2017/
# └── annotations/
#     ├── captions_train2017.json
#     └── captions_val2017.json

export COCO_ROOT=/path/to/coco
```

## Quick Start

### 1. Test Region Extraction

```bash
python data/region_extraction.py
```

This will test the region extraction pipeline on synthetic sketches.

### 2. Test Dataset Loading

```bash
# For Sketchy dataset
python datasets/sketchy_dataset.py

# For COCO dataset
python datasets/coco_dataset.py
```

### 3. Training

**Local Training (Development):**

```bash
# Train both stages
python train.py \
    --stage both \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --checkpoint_dir ./checkpoints

# Train Stage 1 only
python train.py --stage stage1 --epochs 10

# Train Stage 2 only
python train.py --stage stage2 --epochs 10
```

**RunPod Cloud Training:**

```bash
# On RunPod instance
git clone https://github.com/KumarSatyam24/RAGAF-Diffusion.git
cd RAGAF-Diffusion/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# Install dependencies
pip install -r requirements.txt

# Mount datasets to /workspace
# Set environment variables
export SKETCHY_ROOT=/workspace/sketchy
export COCO_ROOT=/workspace/coco

# Train with mixed precision
python train.py \
    --stage both \
    --batch_size 8 \
    --mixed_precision fp16 \
    --checkpoint_dir /workspace/checkpoints \
    --use_wandb
```

### 4. Inference

```bash
python inference.py \
    --sketch examples/dog_sketch.png \
    --prompt "A photo of a golden retriever dog" \
    --stage1_checkpoint ./checkpoints/stage1/final.pt \
    --stage2_checkpoint ./checkpoints/stage2/final.pt \
    --output dog_output \
    --seed 42
```

Output structure:
```
outputs/dog_output/
├── sketch.png              # Input sketch
├── regions.png             # Extracted regions visualization
├── stage1_output.png       # Stage 1 coarse output
├── stage2_output.png       # Stage 2 refined output
├── comparison.png          # Side-by-side comparison
└── prompt.txt              # Text prompt
```

## Project Structure

```
.
├── data/                           # Data processing utilities
│   ├── sketch_extraction.py       # Sketch extraction from images (Canny, XDoG)
│   ├── region_extraction.py       # Region extraction via connected components
│   └── region_graph.py            # Graph construction from regions
│
├── datasets/                       # Dataset loaders
│   ├── sketchy_dataset.py         # Sketchy dataset loader
│   └── coco_dataset.py            # MS COCO dataset loader
│
├── models/                         # Core model components
│   ├── stage1_diffusion.py        # Stage 1: Sketch-guided diffusion
│   ├── stage2_refinement.py       # Stage 2: Semantic refinement
│   ├── ragaf_attention.py         # RAGAF attention module
│   └── adaptive_fusion.py         # Adaptive modality fusion
│
├── configs/                        # Configuration files
│   └── config.py                  # Training/inference configs
│
├── utils/                          # Utility functions
│   └── common.py                  # Common helpers
│
├── train.py                        # Main training script
├── inference.py                    # Inference script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Configuration

Edit `configs/config.py` or create a YAML config file:

```python
# Generate default config
python configs/config.py

# Train with custom config
python train.py --config my_config.yaml
```

Key configuration options:
- `pretrained_model_name`: Base Stable Diffusion model
- `batch_size`: Batch size (adjust based on GPU memory)
- `learning_rate`: Learning rate (default: 1e-4)
- `mixed_precision`: Mixed precision training (fp16/bf16)
- `fusion_method`: Adaptive fusion method (learned/heuristic/hybrid)
- `use_lora`: Use LoRA for efficient fine-tuning

## Memory Optimization

For limited GPU memory:

```bash
# Use smaller batch size
--batch_size 2

# Use gradient accumulation
--gradient_accumulation_steps 4

# Use mixed precision
--mixed_precision fp16

# Freeze base UNet (train only RAGAF components)
--freeze_base_unet
```

## Monitoring

### Weights & Biases Integration

```bash
# Enable W&B logging
python train.py --use_wandb --wandb_project ragaf-diffusion

# Tracked metrics:
# - Training loss (stage1/stage2)
# - Learning rate
# - Fusion weights (sketch vs text)
# - Region-text attention maps
```

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir ./checkpoints

# View at http://localhost:6006
```

## Evaluation

Coming soon: Evaluation metrics including:
- FID (Fréchet Inception Distance)
- Sketch fidelity scores
- Text-image alignment (CLIP score)
- Region-text attention quality

## Troubleshooting

**CUDA Out of Memory:**
- Reduce batch size
- Enable gradient accumulation
- Use mixed precision (fp16)
- Freeze base UNet

**Slow Training:**
- Enable xFormers for efficient attention
- Use DataLoader with multiple workers
- Cache auto-generated sketches (COCO)

**Region Extraction Issues:**
- Adjust `min_region_area` threshold
- Try different sketch extraction methods
- Check sketch quality (should be clear, high contrast)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ragaf-diffusion-2026,
  title={RAGAF-Diffusion: Dual-Stage Controllable Diffusion with Region-Adaptive Graph-Attention Fusion},
  author={Kumar, Satyam},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the base diffusion model
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) for the diffusion framework
- [ControlNet](https://github.com/lllyasviel/ControlNet) for sketch conditioning inspiration
- Sketchy and MS COCO dataset creators

## Contact

- **Author**: Satyam Kumar
- **GitHub**: [@KumarSatyam24](https://github.com/KumarSatyam24)
- **Project**: [RAGAF-Diffusion](https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion)

## Research Status

🚧 **Active Development** - This is a research project under active development. Contributions and feedback are welcome!

### Roadmap
- [x] Core architecture implementation
- [x] Dataset loaders (Sketchy, COCO)
- [x] Training pipeline
- [x] Inference pipeline
- [ ] Pretrained model weights
- [ ] Evaluation suite
- [ ] Interactive demo
- [ ] Paper publication
