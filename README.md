# RAGAF-Diffusion: Dual-Stage Controllable Diffusion with Region-Adaptive Graph-Attention Fusion

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion?style=social)](https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion)

**Bridging Structure and Semantics: A Novel Diffusion Framework for Controllable Sketch-to-Image Generation**

</div>

---

## The Core Concept

**The Challenge:** How can we generate photorealistic images from sketches while preserving structural fidelity AND incorporating rich semantic details from text descriptions?

**Traditional Approaches Fall Short:**
- **Pure text-to-image**: Struggles with precise spatial control
- **Simple sketch conditioning**: Lacks semantic understanding of different regions
- **Uniform fusion**: Treats all image regions equally, missing context-specific requirements

**Our Innovation:** RAGAF-Diffusion treats sketch regions as a **semantic graph**, where each region can be intelligently fused with relevant text semantics through **graph attention**, enabling **region-aware**, **context-sensitive** generation.

---

## Conceptual Overview

### The Problem We Solve

Imagine you sketch a **house with a tree and a car**. You want:
- The **house** to be a "Victorian mansion"
- The **tree** to be a "cherry blossom in spring"
- The **car** to be a "vintage red sports car"

**Traditional methods** apply the entire text prompt uniformly across the image, leading to:
- Semantic confusion (tree features bleeding into the house)
- Structure-semantic mismatch (text details violating sketch structure)
- Poor controllability (can't target specific regions)

**RAGAF-Diffusion** solves this by:
1. **Automatically detecting regions** in your sketch (house, tree, car)
2. **Building a spatial graph** of region relationships
3. **Using graph attention** to determine which text tokens are relevant to each region
4. **Adaptively fusing** sketch structure and text semantics based on the denoising timestep

### The RAGAF Philosophy

```
 Sketch Structure →  Region Graph →  Text Semantics →  Photorealistic Image
 (What + Where)      (Relationships)    (How + Details)      (Structure + Beauty)
```

**Three Core Principles:**

1. **Region-Awareness**: Different parts of a sketch have different semantic needs
2. **Adaptive Fusion**: Balance between structure and semantics evolves during generation
3. **Graph Reasoning**: Spatial relationships matter for coherent image synthesis

---

## Technical Innovation: The RAGAF Framework

### 1. Dual-Stage Architecture Design

Our pipeline separates **structural generation** from **semantic refinement** for better controllability:

#### **Stage 1: Sketch-Guided Coarse Generation**
```
Purpose: Establish global structure and layout
Method:  ControlNet-style sketch conditioning
Output:  Structure-preserving coarse image
```

**Why separate stages?**
- **Focus**: Each stage optimizes for one objective (structure vs. semantics)
- **Flexibility**: Can use different guidance strengths for different generation goals
- **Quality**: Prevents structure-semantic conflicts during generation

#### **Stage 2: RAGAF Semantic Refinement**
```
Purpose: Add semantic details while preserving structure
Method:  Region-adaptive graph attention fusion
Output:  Photorealistic image with rich details
```

**The key insight**: Structure and semantics have different importance at different denoising timesteps!

---

### 2. Region-Adaptive Graph-Attention Fusion (RAGAF)

This is the **core innovation** of our framework:

#### **Step 1: Automatic Region Extraction**

Instead of treating sketches as monolithic images, we decompose them into **meaningful regions**:

```python
# Conceptual process
sketch → edge_detection → connected_components → regions
```

**Features per region** (6D vector):
- Centroid location (x, y) - normalized
- Area and perimeter
- Bounding box dimensions
- Shape compactness measure

**Why automatic?** No manual annotation required! Works with any sketch.

#### **Step 2: Graph Construction**

Regions aren't isolated - they have **spatial relationships**:

```
Graph G = (V, E) where:
- V (nodes) = Sketch regions with spatial features
- E (edges) = Relationships between regions
```

**Edge Types:**
1. **Adjacency**: Regions that touch or overlap
2. **Proximity**: K-nearest neighbors by centroid distance
3. **Containment**: Nested regions (e.g., window inside house)

**Why graphs?**
- Captures **spatial context** (tree is next to house)
- Enables **relational reasoning** (car should match road style)
- Models **part-whole relationships** (windows belong to house)

#### **Step 3: Graph Attention Mechanism**

Not all region relationships are equally important! We use **multi-head graph attention**:

```python
# Simplified concept
for each region i:
    attention_weights = softmax(Q_i @ K_neighbors / √d)
    updated_features_i = ∑ attention_weights * V_neighbors
```

**What this does:**
- Each region "attends" to relevant neighboring regions
- Learns which relationships matter (e.g., roof relates to walls)
- Propagates information across the graph

#### **Step 4: Region-Text Cross-Attention**

This is where **semantic control** happens:

```python
# For each region, compute attention with text tokens
attention_map[region_i, token_j] = relevance(region_i, token_j)
```

**The Magic:** Different text tokens influence different regions!

**Example with prompt: "A Victorian house with a cherry blossom tree"**

```
Text Token Attention Map:

"Victorian" → High attention to [House, Roof, Window] regions
            → Low attention to [Tree, Ground] regions

"cherry"    → High attention to [Tree foliage] region
            → Zero attention to [House] regions

"blossom"   → High attention to [Tree foliage] region
            → Low attention to [Tree trunk] region
```

**Why powerful?**
- **Targeted semantics**: "Victorian" only affects the house
- **No bleeding**: Tree style doesn't leak into house
- **Fine control**: Different parts get different semantic guidance

#### **Step 5: Adaptive Fusion Weights**

The **final innovation**: Fusion weights adapt based on **diffusion timestep**:

```python
α_sketch(t) = high when t is large  (early steps, noisy)
β_text(t)   = high when t is small  (late steps, denoised)

fused_features = α(t) * sketch_features + β(t) * text_features
```

**Intuition:**
- **Early timesteps** (t=1000 → 700): Image is very noisy
  - **Strong sketch guidance** (α=0.8): Establish correct structure
  - Weak text guidance (β=0.2): Don't add details yet

- **Middle timesteps** (t=700 → 300): Structure forming
  - **Balanced guidance** (α=0.5, β=0.5): Refine both structure and semantics

- **Late timesteps** (t=300 → 0): Near final image
  - **Strong text guidance** (β=0.8): Add rich semantic details
  - Weak sketch guidance (α=0.2): Allow flexibility for realism

---

## Advantages Over Existing Methods

### Comparison with State-of-the-Art

| Capability | Stable Diffusion | ControlNet | Sketch-guided GAN | **RAGAF-Diffusion** |
|------------|------------------|------------|-------------------|---------------------|
| **Structural Control** | ❌ Weak | ✅ Strong | ✅ Strong | ✅ **Strong** |
| **Semantic Control** | ✅ Strong | ✅ Strong | ❌ Limited | ✅ **Strong** |
| **Region Awareness** | ❌ None | ❌ None | ❌ None | ✅ **Full** |
| **Adaptive Fusion** | ❌ No | ❌ No | ❌ No | ✅ **Yes** |
| **Spatial Reasoning** | ❌ No | ❌ No | ❌ No | ✅ **Graph-based** |
| **Timestep Awareness** | ⚠️ Fixed | ⚠️ Fixed | N/A | ✅ **Adaptive** |
| **Multi-region Text** | ❌ No | ❌ No | ❌ No | ✅ **Yes** |

---

## Project Structure

```
Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── paper.md                          # Research paper draft
├── app.py                            # Streamlit demo application
├── config_improved.py                # Alternative configuration
│
├── src/                              # Source code
│   ├── configs/
│   │   └── config.py                 # Training/inference configs
│   ├── data/
│   │   ├── sketch_extraction.py      # Edge detection methods
│   │   ├── region_extraction.py      # Connected component analysis
│   │   └── region_graph.py           # Spatial graph construction
│   ├── datasets/
│   │   ├── sketchy_dataset.py        # Sketchy dataset loader
│   │   └── coco_dataset.py           # MS COCO dataset loader
│   ├── models/
│   │   ├── ragaf_attention.py        # RAGAF attention module
│   │   ├── adaptive_fusion.py        # Timestep-aware fusion
│   │   ├── stage1_diffusion.py       # Stage 1 sketch-guided diffusion
│   │   └── stage2_refinement.py      # Stage 2 semantic refinement
│   └── utils/
│       └── common.py                 # Helper functions
│
├── scripts/                          # Training and inference scripts
│   ├── training/
│   │   └── train.py                  # Main training script
│   ├── inference/
│   │   ├── inference.py              # Inference script
│   │   └── quick_test.py             # Quick testing utility
│   └── evaluation/
│       ├── evaluate_all_categories.py
│       └── regenerate_optimized.py
│
├── tests/                            # Unit tests
│   └── unit/
│       ├── test_inference.py
│       └── test_stage1.py
│
├── tools/                            # Dataset utilities
│   └── dataset/
│       ├── download_dataset.py
│       ├── verify_dataset.py
│       └── check_sketchy_format.py
│
├── docs/                             # Documentation
│   ├── guides/                       # User guides
│   └── reports/                      # Validation reports
│
├── results/                          # Generated outputs
├── validation_results/               # Validation outputs
└── wandb_logs/                       # Training logs
```

---

## Datasets

### Sketchy Dataset (Primary)

<div align="center">

| Metric | Value |
|--------|-------|
| **Total Pairs** | **75,481** |
| **Categories** | **125 objects** |
| **Train** | 52,514 samples (70%) |
| **Validation** | 11,532 samples (15%) |
| **Test** | 11,435 samples (15%) |
| **Size** | ~10 GB |

</div>

**Download:** [https://sketchy.eye.gatech.edu/](https://sketchy.eye.gatech.edu/)

**Categories Include:** airplane, apple, bear, bicycle, cat, dog, elephant, guitar, horse, house, motorcycle, penguin, piano, rabbit, shoe, tree, and 109 more!

### MS COCO (Secondary)

- **Purpose**: Multi-object complex scenes
- **Size**: ~25 GB (images + annotations)
- **Train**: 118,287 images
- **Val**: 5,000 images
- **Features**: 5 captions per image, auto-generated sketches

**Download:** [https://cocodataset.org/](https://cocodataset.org/)

> **Note**: You can train on **Sketchy only**. COCO is optional for multi-object experiments.

---

## Quick Start

### Installation

```bash
# 1. Clone repository
git clone https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion.git
cd Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python tools/dataset/verify_dataset.py  # If you have datasets downloaded
```

### Dataset Setup

**Option 1: Automatic Setup (Sketchy)**
```bash
# Download Sketchy dataset from https://sketchy.eye.gatech.edu/
# Extract to your preferred location

# Set environment variable
export SKETCHY_ROOT=/path/to/sketchy
# On Windows: set SKETCHY_ROOT=C:\path\to\sketchy

# Verify dataset format
python tools/dataset/check_sketchy_format.py /path/to/sketchy
```

**Option 2: Verify Setup**
```bash
# Run comprehensive validation
python tools/dataset/verify_dataset.py

# Expected output:
# ✅ SKETCHY_ROOT: /path/to/sketchy
# ✅ Dataset loaded: 52,514 training samples
# ✅ ALL CHECKS PASSED - READY FOR TRAINING!
```

---

## Training

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| CUDA | 11.8+ | 12.1+ |
| GPU Memory | 16GB | 24GB+ |
| GPU | RTX 3090 | RTX 4090 / A100 |
| RAM | 32GB | 64GB+ |
| Storage | 50GB | 100GB+ |

> **Mac Users**: Training on CPU is extremely slow. Use cloud GPU (RunPod, Lambda Labs, AWS).

### Training Commands

**Quick Start:**
```bash
# Train both stages on Sketchy dataset
python scripts/training/train.py --dataset sketchy

# Train with subset for quick testing
python scripts/training/train.py \
    --dataset sketchy \
    --categories airplane,apple,bear,cat,dog \
    --epochs 2
```

**Full Training:**
```bash
# Stage 1: Sketch-guided diffusion
python scripts/training/train.py \
    --stage stage1 \
    --dataset sketchy \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --checkpoint_dir ./checkpoints/stage1

# Stage 2: Semantic refinement
python scripts/training/train.py \
    --stage stage2 \
    --dataset sketchy \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --checkpoint_dir ./checkpoints/stage2

# Both stages (end-to-end)
python scripts/training/train.py \
    --stage both \
    --dataset sketchy \
    --batch_size 4 \
    --epochs 20
```

**Advanced Options:**
```bash
python scripts/training/train.py \
    --stage both \
    --dataset both \                    # Use both Sketchy and COCO
    --batch_size 8 \
    --gradient_accumulation_steps 2 \    # Effective batch size = 16
    --learning_rate 1e-4 \
    --mixed_precision bf16 \             # For RTX 4090/Blackwell GPUs
    --use_lora \                         # LoRA fine-tuning
    --lora_rank 8 \
    --use_wandb                          # Weights & Biases logging
```

### Cloud GPU Training (RunPod)

**Setup:**
```bash
# 1. Create RunPod account: https://runpod.io/
# 2. Select GPU: RTX 4090 or A100 recommended
# 3. SSH into instance

# 4. Clone and setup
git clone https://github.com/KumarSatyam24/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion.git
cd Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion
pip install -r requirements.txt

# 5. Upload datasets to /workspace/datasets
# 6. Set environment variables
export SKETCHY_ROOT=/workspace/datasets/sketchy
export COCO_ROOT=/workspace/datasets/coco

# 7. Start training
python scripts/training/train.py \
    --stage both \
    --batch_size 8 \
    --mixed_precision bf16 \
    --checkpoint_dir /workspace/checkpoints \
    --use_wandb
```

**Expected Training Time:**

| Stage | Dataset | GPU | Epochs | Time |
|-------|---------|-----|--------|------|
| Stage 1 | Sketchy | RTX 4090 | 10 | ~6 hours |
| Stage 2 | Sketchy | RTX 4090 | 10 | ~8 hours |
| Both | Sketchy | RTX 4090 | 20 | ~14 hours |
| Both | Sketchy + COCO | A100 | 20 | ~24 hours |

---

## Inference & Generation

### Basic Usage

```bash
python scripts/inference/inference.py \
    --sketch examples/dog_sketch.png \
    --prompt "A photo of a golden retriever dog" \
    --stage1_checkpoint ./checkpoints/stage1/final.pt \
    --stage2_checkpoint ./checkpoints/stage2/final.pt \
    --output dog_output \
    --seed 42
```

### Advanced Options

```bash
python scripts/inference/inference.py \
    --sketch my_sketch.png \
    --prompt "A beautiful sunset landscape with mountains" \
    --stage1_checkpoint checkpoints/stage1_best.pt \
    --stage2_checkpoint checkpoints/stage2_best.pt \
    --output landscape_output \
    --num_inference_steps 50 \          # More steps = higher quality
    --guidance_scale 7.5 \               # Classifier-free guidance
    --sketch_strength 0.8 \              # Sketch influence (0-1)
    --seed 42 \
    --save_intermediates                  # Save Stage 1 output
```

### Output Structure

```
outputs/dog_output/
├── sketch.png              # Input sketch (normalized)
├── regions.png             # Extracted regions visualization
├── region_graph.png        # Graph structure visualization
├── stage1_output.png       # Stage 1 coarse output
├── stage2_output.png       # Stage 2 refined output (final)
├── comparison.png          # Side-by-side comparison
├── attention_maps.png      # Region-text attention visualization
└── metadata.json           # Generation parameters
```

### Batch Inference

```bash
# Generate from multiple sketches
python scripts/inference/inference.py \
    --sketch_dir examples/sketches/ \
    --prompts_file examples/prompts.txt \
    --output_dir batch_outputs/ \
    --batch_size 4
```

---

## Streamlit Demo App

Launch an interactive web interface for sketch-to-image generation:

```bash
# Run the Streamlit app
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8501
```

**Features:**
- Draw sketches directly in the browser
- Enter text prompts for semantic control
- Adjust guidance scale and sketch strength
- Real-time region extraction visualization
- Compare Stage 1 and Stage 2 outputs

---

## Configuration

### Default Configuration

Key hyperparameters in `src/configs/config.py`:

```python
# Model
pretrained_model_name = "runwayml/stable-diffusion-v1-5"
hidden_dim = 512
num_graph_layers = 2
num_attention_heads = 8

# Training
learning_rate = 1e-5
batch_size = 2
stage1_epochs = 10
stage2_epochs = 10
mixed_precision = "bf16"  # bf16 for RTX 4090/Blackwell, fp16 for others

# Fusion
fusion_method = "learned"  # "learned", "heuristic", "hybrid"
use_region_adaptive_fusion = True

# LoRA (efficient fine-tuning)
use_lora = True
lora_rank = 4

# Stage 2 refinement
residual_alpha = 0.2  # Scaling for residual refinement
lambda_identity = 0.3
lambda_lpips = 0.15
```

---

## Evaluation

### Metrics

**Image Quality:**
- **FID** (Fréchet Inception Distance) - Overall image quality
- **IS** (Inception Score) - Image diversity and quality
- **LPIPS** - Perceptual similarity
- **SSIM** - Structural similarity

**Sketch Fidelity:**
- **Chamfer Distance** - Edge alignment with input sketch
- **IoU** - Region overlap with sketch regions

**Text Alignment:**
- **CLIP Score** - Text-image semantic alignment

**RAGAF-Specific:**
- **Attention Accuracy** - Region-text attention alignment
- **Fusion Balance** - Sketch vs text weight distribution
- **Graph Quality** - Region graph connectivity metrics

### Running Evaluation

```bash
# Evaluate on test set
python scripts/evaluation/evaluate_all_categories.py \
    --checkpoint checkpoints/best.pt \
    --test_split test \
    --output_dir evaluation_results/

# Run ablation studies
python scripts/ablation_inference_study.py \
    --stage1_checkpoint checkpoints/stage1/final.pt \
    --stage2_checkpoint checkpoints/stage2/final.pt \
    --output_dir ablation_results/
```

---

## Examples

### Example 1: Simple Object
```bash
python scripts/inference/inference.py \
    --sketch examples/apple_sketch.png \
    --prompt "A photo of a red apple on a wooden table" \
    --output apple_result
```

**Input Sketch** → **Stage 1 (Structure)** → **Stage 2 (Refined)**
```
   ┌─────┐          ┌─────┐              ┌─────┐
   │ ○○  │   ───►   │ Gray│      ───►    │Photo│
   │○  ○ │          │Apple│              │Apple│
   └─────┘          └─────┘              └─────┘
```

### Example 2: Animal with Details
```bash
python scripts/inference/inference.py \
    --sketch examples/dog_sketch.png \
    --prompt "A golden retriever dog sitting on grass in a park" \
    --guidance_scale 8.0 \
    --sketch_strength 0.75
```

### Example 3: Multiple Variations
```bash
# Generate 5 variations from the same sketch
for seed in {1..5}; do
    python scripts/inference/inference.py \
        --sketch examples/cat_sketch.png \
        --prompt "A fluffy white cat with blue eyes" \
        --output cat_var_$seed \
        --seed $seed
done
```

---

## Getting Help

1. **Check Documentation:**
   - `docs/guides/` - User guides and tutorials
   - `docs/reports/` - Validation and analysis reports

2. **Run Validation:**
   ```bash
   python tools/dataset/verify_dataset.py
   ```

3. **GitHub Issues:**
   - Search existing issues
   - Create new issue with error logs

4. **Debug Mode:**
   ```bash
   python scripts/training/train.py --debug --verbose
   ```

---

## Related Work

**Diffusion Models:**
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., NeurIPS 2020)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., CVPR 2022)

**Controllable Generation:**
- [ControlNet](https://arxiv.org/abs/2302.05543) (Zhang et al., ICCV 2023)
- [T2I-Adapter](https://arxiv.org/abs/2302.08453) (Mou et al., 2023)

**Graph Attention:**
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., ICLR 2018)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017)

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **Stable Diffusion**: CreativeML Open RAIL-M License
- **HuggingFace Transformers**: Apache License 2.0
- **PyTorch**: BSD License
- **Sketchy Dataset**: Academic use only
- **MS COCO**: Creative Commons Attribution 4.0

---

## Acknowledgments

This project builds upon excellent prior work:

- **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)** by CompVis - Base diffusion model architecture
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** - Diffusion model framework and utilities
- **[ControlNet](https://github.com/lllyasviel/ControlNet)** by Lvmin Zhang - Inspiration for sketch conditioning
- **[Sketchy Dataset](https://sketchy.eye.gatech.edu/)** by Georgia Tech - Sketch-photo paired dataset
- **[MS COCO](https://cocodataset.org/)** - Image-caption dataset
- **PyTorch Team** - Deep learning framework

---

<div align="center">

**Made with ❤️ by [Satyam Kumar](https://github.com/KumarSatyam24)**

</div>
