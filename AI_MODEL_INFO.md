# AI Model Information

## Base AI Model

This project uses **Stable Diffusion v1.5** as its foundation AI model.

### Primary Model Details

**Model Name:** `runwayml/stable-diffusion-v1-5`

**Model Type:** Latent Diffusion Model (Text-to-Image)

**Source:** [Hugging Face Model Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5)

**License:** CreativeML Open RAIL-M License

### Model Architecture Components

The RAGAF-Diffusion framework builds upon Stable Diffusion v1.5 with the following components:

#### 1. Core Stable Diffusion v1.5 Components

- **UNet2DConditionModel**: Main denoising network
  - 860M parameters (frozen during Stage 1 training)
  - Operates on latent space (256×256 → 32×32 latents)
  - 4 down blocks, 1 mid block, 4 up blocks

- **AutoencoderKL (VAE)**: Image encoder/decoder
  - Compresses images to 8x smaller latent representations
  - Frozen during training

- **CLIP Text Encoder**: Text conditioning
  - Model: `openai/clip-vit-large-patch14`
  - 768-dimensional text embeddings
  - 77 max token length
  - Frozen during training

- **Scheduler**: Noise scheduling
  - Training: DDPMScheduler
  - Inference: DDIMScheduler (50 steps) or DDPMScheduler (1000 steps)

#### 2. Custom RAGAF Components (Novel Contributions)

- **SketchEncoder** (~14M parameters)
  - ControlNet-style architecture
  - Input: Grayscale sketch (1, 256, 256)
  - Output: 12 down-block residuals + 1 mid-block residual
  - Channels: [320, 640, 1280, 1280] matching UNet architecture

- **RAGAF Attention Module** (~4.08M parameters)
  - Node feature dimension: 6 (spatial region features)
  - Hidden dimension: 512
  - Graph attention layers: 2
  - Multi-head attention: 8 heads
  - Text dimension: 768 (CLIP embeddings)

- **Adaptive Fusion Module**
  - Timestep-aware fusion weights
  - Learned fusion method (default)
  - Region-adaptive fusion enabled

### Model Specifications

| Component | Parameters | Training Status |
|-----------|-----------|-----------------|
| **Stable Diffusion v1.5 UNet** | 860M | Frozen (Stage 1), Fine-tuned with LoRA (Stage 2) |
| **VAE** | 83M | Frozen |
| **CLIP Text Encoder** | 123M | Frozen |
| **Sketch Encoder** | ~14M | Trainable (Stage 1) |
| **RAGAF Attention** | ~4.08M | Trainable (Stage 2) |
| **Adaptive Fusion** | <1M | Trainable (Stage 2) |
| **Total Trainable (Stage 1)** | ~14M | SketchEncoder only |
| **Total Trainable (Stage 2)** | ~5M + LoRA | RAGAF + Fusion + LoRA adapters |

### Training Configuration

**Stage 1: Sketch-Guided Diffusion**
- Base Model: Stable Diffusion v1.5
- Trainable: SketchEncoder only
- Frozen: UNet, VAE, CLIP
- Epochs: 10-18
- Optimizer: AdamW (lr=1e-4)

**Stage 2: Semantic Refinement**
- Base Model: Stage 1 checkpoint
- Trainable: RAGAF Attention, Adaptive Fusion, UNet (via LoRA)
- LoRA Configuration:
  - Rank: 4
  - Alpha: 4
  - Applied to UNet cross-attention layers
- Epochs: 10
- Optimizer: AdamW (lr=1e-5)

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3090 (24GB) | RTX 4090 / A100 (32GB+) |
| **VRAM** | 16GB | 24GB+ |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 50GB | 100GB+ |
| **CUDA** | 11.8+ | 12.1+ |

### Inference Details

**Image Generation Pipeline:**

1. **Input Processing**
   - Sketch: (1, 256, 256) grayscale
   - Text: Tokenized to 77 tokens max
   - Region extraction: Automatic from sketch

2. **Stage 1: Structure Generation**
   - Noise scheduler: DDIM (50 steps)
   - Guidance scale: 7.5 (classifier-free guidance)
   - Output: Coarse image (3, 256, 256)

3. **Stage 2: Semantic Refinement**
   - Graph construction: Adjacency-based
   - Region-text attention: Multi-head (8 heads)
   - Adaptive fusion: Timestep-aware
   - Refinement steps: 30
   - Output: Refined image (3, 256, 256)

### Model Weights & Checkpoints

**Pre-trained Base Model:**
- Downloaded automatically from Hugging Face Hub
- Location: `~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/`

**Project Checkpoints:**
- Stage 1: `/workspace/checkpoints/stage1/epoch_18.pt`
- Stage 2: `/workspace/checkpoints/stage2/final.pt`
- Uploaded to: `DrRORAL/ragaf-diffusion-checkpoints` (Hugging Face)

### Key Hyperparameters

```python
# Model Architecture
pretrained_model_name = "runwayml/stable-diffusion-v1-5"
hidden_dim = 512
num_graph_layers = 2
num_attention_heads = 8

# Training
learning_rate_stage1 = 1e-4
learning_rate_stage2 = 1e-5
batch_size = 2
mixed_precision = "bf16"  # For RTX 5090, "fp16" for older GPUs

# LoRA
lora_rank = 4
lora_alpha = 4

# Inference
num_inference_steps = 50  # Stage 1
num_refinement_steps = 30  # Stage 2
guidance_scale = 7.5
```

### References

**Base Model:**
- **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
  - Rombach et al., CVPR 2022
  - CompVis & Stability AI

**Related Architectures:**
- **ControlNet**: [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
  - Zhang et al., ICCV 2023
  - Inspiration for sketch conditioning architecture

- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
  - Radford et al., ICML 2021
  - Used for text encoding

### Model Capabilities

**What This Model Can Do:**
- Generate photorealistic images from sketches
- Apply region-specific semantic details from text prompts
- Preserve sketch structure while adding rich textures
- Handle multi-object scenes with spatial coherence
- Adapt fusion weights based on generation progress

**What This Model Cannot Do:**
- Generate images without sketch input (not pure text-to-image)
- Process video or 3D content
- Real-time generation (requires ~10-30 seconds per image)
- Generate images larger than 512×512 without additional techniques

### Version Information

- **Stable Diffusion**: v1.5 (released August 2022)
- **PyTorch**: >=2.0.0
- **Diffusers**: >=0.21.0
- **Transformers**: >=4.30.0

### Quick Summary

**In simple terms:** This project uses **Stable Diffusion 1.5** (a popular text-to-image AI model from RunwayML/Stability AI) as its foundation, and adds custom components for sketch-based control and region-aware semantic generation.

**The AI model is**: A **diffusion model** that gradually removes noise to create images, guided by both sketch structure and text descriptions.

**Base Technology**: Latent Diffusion Models (LDM) with CLIP text conditioning.
