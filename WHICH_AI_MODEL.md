# Which AI Model Is This?

## Quick Answer

This project uses **Stable Diffusion v1.5** as its base AI model.

### Model Details

- **Model Name:** `runwayml/stable-diffusion-v1-5`
- **Type:** Latent Diffusion Model (Text-to-Image Generation)
- **Source:** [Hugging Face Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Created by:** RunwayML & Stability AI
- **Release Date:** August 2022
- **License:** CreativeML Open RAIL-M

## What Does This Mean?

**Stable Diffusion** is a powerful AI model that can generate images from text descriptions. It works by:
1. Starting with random noise
2. Gradually removing the noise over many steps
3. Using text guidance to shape what the final image looks like

This project **extends** Stable Diffusion v1.5 by adding:
- **Sketch control:** Ability to guide generation with hand-drawn sketches
- **Region awareness:** Different parts of the image can have different styles
- **Smart fusion:** Combines sketch structure with text details intelligently

## Complete Model Architecture

This project combines:

### 1. Pre-trained Components (From Stable Diffusion v1.5)
- **UNet Denoiser:** 860M parameters - Main image generation network
- **VAE (Variational Autoencoder):** 83M parameters - Compresses/decompresses images
- **CLIP Text Encoder:** 123M parameters - Understands text descriptions

### 2. Custom Components (New in RAGAF-Diffusion)
- **Sketch Encoder:** ~14M parameters - Processes sketch input
- **RAGAF Attention Module:** ~4M parameters - Connects regions with text
- **Adaptive Fusion:** <1M parameters - Balances sketch vs text influence

## Why Stable Diffusion v1.5?

- **Proven quality:** One of the most reliable text-to-image models
- **Well-documented:** Extensive community support and documentation
- **Efficient:** Can run on consumer GPUs (RTX 3090/4090)
- **Flexible:** Easy to extend with additional control mechanisms
- **Open source:** Available for research and development

## Where to Learn More

- **Full Technical Details:** See [AI_MODEL_INFO.md](AI_MODEL_INFO.md)
- **Model Card:** [Hugging Face Model Page](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Original Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## Summary

**In one sentence:** This is a sketch-to-image AI system built on top of **Stable Diffusion 1.5**, with custom additions for sketch control and region-aware text guidance.
