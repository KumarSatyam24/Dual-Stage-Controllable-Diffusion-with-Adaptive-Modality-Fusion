# RAGAF-Diffusion Development Notes

## Implementation Status

### ✅ Completed Components

1. **Data Processing**
   - ✅ Sketch extraction (Canny, XDoG, HED placeholder)
   - ✅ Region extraction via connected components
   - ✅ Region graph construction (adjacency, knn, radius, hybrid)
   - ✅ Automatic region merging and filtering

2. **Dataset Loaders**
   - ✅ Sketchy dataset loader with region extraction
   - ✅ MS COCO dataset loader with auto-sketch generation
   - ✅ Custom collate functions for batching
   - ✅ Data augmentation pipeline

3. **Model Architecture**
   - ✅ Stage 1: Sketch-guided diffusion base
   - ✅ Stage 2: Semantic refinement base
   - ✅ RAGAF attention module (graph attention + cross-attention)
   - ✅ Adaptive fusion module (timestep-aware)
   - ✅ Sketch encoder (ControlNet-style)

4. **Training Infrastructure**
   - ✅ Dual-stage training pipeline
   - ✅ Accelerate integration (distributed, mixed precision)
   - ✅ Gradient accumulation support
   - ✅ Checkpointing and resuming
   - ✅ Weights & Biases logging
   - ✅ Learning rate scheduling

5. **Inference**
   - ✅ Stage 1 pipeline
   - ✅ Stage 2 pipeline (structure defined)
   - ✅ Batch inference support
   - ✅ Visualization utilities

6. **Configuration & Utils**
   - ✅ Comprehensive config system
   - ✅ Utility functions (visualization, metrics)
   - ✅ Quick start script

### 🚧 To Be Implemented

1. **Model Refinements**
   - ⏳ Full ControlNet-style feature injection in Stage 1
   - ⏳ Proper region feature injection in Stage 2 UNet
   - ⏳ LoRA integration for efficient fine-tuning
   - ⏳ HED edge detection model (currently placeholder)

2. **Training Enhancements**
   - ⏳ Validation loop with metrics
   - ⏳ Progressive resolution training
   - ⏳ Curriculum learning strategy

3. **Evaluation**
   - ⏳ FID score computation
   - ⏳ CLIP score for text-image alignment
   - ⏳ Sketch fidelity metrics
   - ⏳ Region-text attention quality metrics

4. **Inference Features**
   - ⏳ Complete Stage 2 refinement diffusion loop
   - ⏳ Interactive region editing
   - ⏳ Multi-resolution generation
   - ⏳ Attention map visualization

## Known Issues & Limitations

### Current Limitations

1. **Stage 1 Sketch Conditioning**
   - Currently uses placeholder UNet forward
   - Need to implement proper sketch feature injection into UNet blocks
   - Zero-initialized convolutions defined but not yet integrated

2. **Stage 2 Feature Injection**
   - Region features computed but not properly injected into UNet
   - Need custom UNet wrapper or hook system for injection
   - Current implementation just uses standard text conditioning

3. **LoRA Integration**
   - LoRA flag exists but not implemented
   - Would significantly reduce trainable parameters
   - Need to integrate with `peft` library

4. **Batch Processing**
   - Region graphs are per-image, not batched
   - Training currently processes graphs sequentially
   - Need proper batching for multi-GPU efficiency

### Memory Considerations

**Minimum Requirements:**
- 16GB GPU for training (batch_size=2-4)
- 32GB GPU recommended (batch_size=8+)

**Memory Optimization Strategies:**
```python
# 1. Reduce batch size
batch_size = 2

# 2. Gradient accumulation
gradient_accumulation_steps = 4  # Effective batch size = 8

# 3. Mixed precision
mixed_precision = "fp16"

# 4. Freeze base UNet
freeze_stage1_unet = True

# 5. Don't preload graphs
preload_graphs = False
```

## Training Recommendations

### Stage 1: Sketch-Guided Diffusion

**Goal**: Learn to generate images that preserve sketch structure

**Recommended Settings:**
```python
stage1_epochs = 10-20
batch_size = 4
learning_rate = 1e-4
freeze_base_unet = True  # Only train sketch encoder
gradient_accumulation_steps = 2
```

**Expected Behavior:**
- Early epochs: Noisy outputs with rough sketch alignment
- Mid epochs: Clear structure preservation
- Late epochs: Fine details while maintaining structure

### Stage 2: Semantic Refinement

**Goal**: Refine with text while preserving sketch structure

**Recommended Settings:**
```python
stage2_epochs = 10-20
batch_size = 2  # More memory intensive
learning_rate = 5e-5  # Lower LR for refinement
gradient_accumulation_steps = 4
```

**Training Strategy:**
1. Load Stage 1 checkpoint as initialization
2. Train RAGAF attention first (freeze UNet)
3. Fine-tune full model with lower LR

## RunPod Cloud Training

### Setup on RunPod

```bash
# 1. Create RunPod instance (RTX 3090/4090)
# 2. Clone repository
git clone https://github.com/KumarSatyam24/RAGAF-Diffusion.git
cd RAGAF-Diffusion/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion

# 3. Install dependencies
pip install -r requirements.txt

# 4. Mount datasets to persistent volume
# Datasets should be in /workspace for persistence
export SKETCHY_ROOT=/workspace/sketchy
export COCO_ROOT=/workspace/coco

# 5. Train with W&B logging
python train.py \
    --stage both \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 20 \
    --mixed_precision fp16 \
    --checkpoint_dir /workspace/checkpoints \
    --use_wandb \
    --wandb_project ragaf-diffusion
```

### Monitoring on RunPod

```bash
# 1. Check GPU usage
nvidia-smi -l 1

# 2. Monitor W&B dashboard
# Go to https://wandb.ai/your-project

# 3. TensorBoard (if enabled)
tensorboard --logdir /workspace/checkpoints --host 0.0.0.0
```

## Debugging Tips

### Common Errors

**1. CUDA Out of Memory**
```python
# Solution 1: Reduce batch size
batch_size = 2

# Solution 2: Enable gradient checkpointing
use_gradient_checkpointing = True

# Solution 3: Clear cache periodically
torch.cuda.empty_cache()
```

**2. Region Extraction Returns Empty**
```python
# Check sketch quality
# Sketches should be high contrast, clear edges

# Adjust thresholds
min_region_area = 50  # Lower for more regions
max_num_regions = 100  # Higher limit
```

**3. Training Loss Not Decreasing**
```python
# Check learning rate
learning_rate = 1e-4  # May need adjustment

# Check gradient flow
# Enable gradient clipping
max_grad_norm = 1.0
```

## Research Directions

### Potential Improvements

1. **Multi-Scale RAGAF**
   - Apply RAGAF at multiple UNet scales
   - Coarse-to-fine region refinement

2. **Hierarchical Regions**
   - Build hierarchical region graph
   - Parent-child region relationships

3. **Contrastive Learning**
   - Region-text contrastive loss
   - Improve semantic alignment

4. **Progressive Generation**
   - Start with low resolution
   - Progressively increase resolution
   - Better quality and speed

5. **Interactive Editing**
   - Per-region text prompts
   - User-guided region importance

## Citation & References

### Related Work

- **Stable Diffusion**: Rombach et al., CVPR 2022
- **ControlNet**: Zhang et al., ICCV 2023
- **Graph Attention Networks**: Veličković et al., ICLR 2018
- **Sketchy Dataset**: Sangkloy et al., SIGGRAPH 2016

### Our Contributions

1. Region-aware graph attention for sketch-text fusion
2. Adaptive timestep-conditioned modality fusion
3. Dual-stage structure-preserving generation
4. Automatic region extraction without annotation

## Contact & Support

- **Issues**: Report on GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [Your research email]

---

Last Updated: January 30, 2026
