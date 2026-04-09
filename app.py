"""
RAGAF-Diffusion: Sketch + Text → Image Generator (Dual-Stage Diffusion)

A Streamlit interface for generating images from user-provided sketches and text prompts
using a dual-stage diffusion pipeline.

Stage 1: Coarse structure-preserving image generation guided by sketch input.
Stage 2: Semantic refinement with RAGAF fusion for region-aware text alignment.

Checkpoints:
- Stage 1: epoch_18 (final)
- Stage 2: epoch_6 (final)

Author: RAGAF-Diffusion Research Team
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import io

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import ModelConfig, InferenceConfig
from src.models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from src.models.stage2_refinement import Stage2SemanticRefinement, Stage2RefinementPipeline
from src.data.region_extraction import RegionExtractor
from src.data.region_graph import RegionGraphBuilder
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel


# =============================================================================
# Configuration
# =============================================================================
CHECKPOINT_DIR = "/workspace/checkpoints"
STAGE1_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage1/epoch_18.pt")
STAGE2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage2/epoch_6.pt")
MODEL_INPUT_SIZE = 512  # Model expects 512x512
SKETCH_SIZE = 512  # Canvas size


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Model Loading (Cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_models(device: str = "cuda") -> Dict:
    """
    Load Stage 1 and Stage 2 models from checkpoints.

    Args:
        device: Device to load models on

    Returns:
        Dictionary containing loaded models and pipelines
    """
    model_config = ModelConfig()
    device = get_device()

    st.info(f"Loading models on {device}...")

    loaded_models = {
        "stage1_model": None,
        "stage1_pipeline": None,
        "stage2_model": None,
        "stage2_pipeline": None,
        "vae": None,
        "tokenizer": None,
        "text_encoder": None,
        "region_extractor": None,
        "graph_builder": None,
    }

    # -------------------------------------------------------------------------
    # Load Stage 1
    # -------------------------------------------------------------------------
    if os.path.exists(STAGE1_CHECKPOINT):
        with st.spinner("Loading Stage 1 model..."):
            stage1_model = Stage1SketchGuidedDiffusion(
                pretrained_model_name=model_config.pretrained_model_name,
                freeze_base_unet=model_config.freeze_stage1_unet,
                use_lora=model_config.use_lora,
                lora_rank=model_config.lora_rank
            ).to(device)

            checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=device, weights_only=False)
            stage1_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            stage1_model.eval()

            stage1_pipeline = Stage1DiffusionPipeline(
                model=stage1_model,
                num_inference_steps=50,
                guidance_scale=7.5,
                device=device
            )

            loaded_models["stage1_model"] = stage1_model
            loaded_models["stage1_pipeline"] = stage1_pipeline
            st.success("✅ Stage 1 loaded successfully")
    else:
        st.warning(f"⚠️ Stage 1 checkpoint not found: {STAGE1_CHECKPOINT}")

    # -------------------------------------------------------------------------
    # Load Stage 2
    # -------------------------------------------------------------------------
    if os.path.exists(STAGE2_CHECKPOINT):
        with st.spinner("Loading Stage 2 model..."):
            from diffusers import UNet2DConditionModel

            unet = UNet2DConditionModel.from_pretrained(
                model_config.pretrained_model_name,
                subfolder="unet"
            )

            stage2_model = Stage2SemanticRefinement(
                unet=unet,
                node_feature_dim=model_config.node_feature_dim,
                text_dim=model_config.text_dim,
                hidden_dim=model_config.hidden_dim,
                num_graph_layers=model_config.num_graph_layers,
                num_attention_heads=model_config.num_attention_heads,
                fusion_method=model_config.fusion_method,
                use_region_adaptive_fusion=model_config.use_region_adaptive_fusion,
                residual_alpha=model_config.residual_alpha
            ).to(device)

            checkpoint = torch.load(STAGE2_CHECKPOINT, map_location=device, weights_only=False)
            stage2_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            stage2_model.eval()

            # Load VAE for Stage 2
            vae = AutoencoderKL.from_pretrained(
                model_config.pretrained_model_name,
                subfolder="vae"
            ).to(device)

            stage2_pipeline = Stage2RefinementPipeline(
                stage2_model=stage2_model,
                vae=vae,
                num_inference_steps=30,
                guidance_scale=7.5,
                device=device
            )

            # Load tokenizer and text encoder for Stage 2
            tokenizer = CLIPTokenizer.from_pretrained(
                model_config.pretrained_model_name,
                subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                model_config.pretrained_model_name,
                subfolder="text_encoder"
            ).to(device)

            # Region extraction
            region_extractor = RegionExtractor(
                min_region_area=100,
                max_num_regions=50
            )
            graph_builder = RegionGraphBuilder(
                graph_type="hybrid",
                image_size=(512, 512)
            )

            loaded_models["stage2_model"] = stage2_model
            loaded_models["stage2_pipeline"] = stage2_pipeline
            loaded_models["vae"] = vae
            loaded_models["tokenizer"] = tokenizer
            loaded_models["text_encoder"] = text_encoder
            loaded_models["region_extractor"] = region_extractor
            loaded_models["graph_builder"] = graph_builder

            st.success("✅ Stage 2 loaded successfully")
    else:
        st.warning(f"⚠️ Stage 2 checkpoint not found: {STAGE2_CHECKPOINT}")

    return loaded_models


# =============================================================================
# Preprocessing
# =============================================================================
def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    target_size: int = MODEL_INPUT_SIZE
) -> torch.Tensor:
    """
    Preprocess input image for model inference.

    Args:
        image: Input image (PIL Image or numpy array)
        target_size: Target size for resizing

    Returns:
        Preprocessed image tensor (1, 1, H, W) in range [0, 1]
    """
    if isinstance(image, np.ndarray):
        # Handle canvas output (RGBA or RGB)
        if image.shape[-1] == 4:  # RGBA
            # Convert RGBA to grayscale (use alpha as mask)
            alpha = image[:, :, 3] / 255.0
            gray = image[:, :, 0] * alpha + (1 - alpha) * 255
            image = Image.fromarray(gray.astype(np.uint8), mode='L')
        elif len(image.shape) == 3 and image.shape[-1] == 3:  # RGB
            image = Image.fromarray(image).convert('L')
        else:
            image = Image.fromarray(image).convert('L')
    elif isinstance(image, Image.Image):
        image = image.convert('L')

    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to tensor
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

    return image_tensor


def sketch_to_rgb(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Convert sketch to RGB format."""
    if isinstance(image, np.ndarray):
        if image.shape[-1] == 4:  # RGBA
            image = Image.fromarray(image).convert('RGB')
        elif len(image.shape) == 2:  # Grayscale
            image = Image.fromarray(image).convert('RGB')
        else:
            image = Image.fromarray(image).convert('RGB')
    else:
        image = image.convert('RGB')
    return image


# =============================================================================
# Inference Pipeline
# =============================================================================
def run_stage1(
    sketch_tensor: torch.Tensor,
    text_prompt: str,
    models: Dict,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5
) -> torch.Tensor:
    """
    Run Stage 1 inference.

    Args:
        sketch_tensor: Preprocessed sketch tensor
        text_prompt: Text prompt
        models: Dictionary of loaded models
        seed: Random seed
        guidance_scale: Guidance scale

    Returns:
        Stage 1 output image tensor
    """
    pipeline = models["stage1_pipeline"]

    # Temporarily update guidance scale
    original_scale = pipeline.guidance_scale
    pipeline.guidance_scale = guidance_scale

    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = pipeline.generate(
                    sketch=sketch_tensor,
                    text_prompt=text_prompt,
                    height=MODEL_INPUT_SIZE,
                    width=MODEL_INPUT_SIZE,
                    seed=seed
                )
    finally:
        pipeline.guidance_scale = original_scale

    return output


def run_stage2(
    stage1_output: torch.Tensor,
    sketch_np: np.ndarray,
    text_prompt: str,
    text_embeddings: torch.Tensor,
    models: Dict,
    strength: float = 0.5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Run Stage 2 inference.

    Args:
        stage1_output: Output from Stage 1
        sketch_np: Sketch as numpy array (for region extraction)
        text_prompt: Text prompt
        text_embeddings: Pre-computed text embeddings
        models: Dictionary of loaded models
        strength: Refinement strength
        seed: Random seed

    Returns:
        Stage 2 output image tensor
    """
    pipeline = models["stage2_pipeline"]
    region_extractor = models["region_extractor"]
    graph_builder = models["graph_builder"]

    # Extract regions from sketch
    regions = region_extractor.extract_regions(sketch_np)
    region_graph = graph_builder.build_graph(regions)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = pipeline.refine(
                stage1_image=stage1_output,
                region_graph=region_graph,
                text_prompt=text_prompt,
                text_embeddings=text_embeddings,
                strength=strength,
                seed=seed
            )

    return output


def encode_text(
    text_prompt: str,
    models: Dict,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Encode text prompt using CLIP text encoder.

    Args:
        text_prompt: Text prompt
        models: Dictionary containing tokenizer and text_encoder
        device: Device to run on

    Returns:
        Text embeddings tensor
    """
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]

    text_inputs = tokenizer(
        text_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(
            text_inputs.input_ids.to(device)
        )[0].squeeze(0)

    return text_embeddings


def run_pipeline(
    sketch_input: Union[Image.Image, np.ndarray],
    text_prompt: str,
    models: Dict,
    stage1_only: bool = False,
    refinement_strength: float = 0.5,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
) -> Dict[str, Union[torch.Tensor, Image.Image]]:
    """
    Run the complete inference pipeline.

    Args:
        sketch_input: Input sketch (PIL Image or numpy array)
        text_prompt: Text prompt
        models: Dictionary of loaded models
        stage1_only: Whether to skip Stage 2
        refinement_strength: Refinement strength for Stage 2
        guidance_scale: Guidance scale for Stage 1
        seed: Random seed

    Returns:
        Dictionary with results from each stage
    """
    device = get_device()
    results = {}

    # Preprocess sketch
    sketch_tensor = preprocess_image(sketch_input, MODEL_INPUT_SIZE)
    sketch_tensor = sketch_tensor.to(device)

    # Convert to PIL for display
    if isinstance(sketch_input, np.ndarray):
        if sketch_input.shape[-1] == 4:
            sketch_pil = Image.fromarray(sketch_input).convert('RGB')
        else:
            sketch_pil = Image.fromarray(sketch_input).convert('RGB')
    else:
        sketch_pil = sketch_input.convert('RGB')

    sketch_pil = sketch_pil.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    results["input_sketch"] = sketch_pil

    # Stage 1: Coarse generation
    if models.get("stage1_pipeline") is not None:
        st.info("🎨 Running Stage 1: Sketch-guided generation...")
        stage1_output = run_stage1(
            sketch_tensor=sketch_tensor,
            text_prompt=text_prompt,
            models=models,
            seed=seed,
            guidance_scale=guidance_scale
        )
        results["stage1_output"] = stage1_output

        # Convert to PIL for display
        stage1_img = stage1_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        stage1_img = (stage1_img * 255).astype(np.uint8)
        results["stage1_pil"] = Image.fromarray(stage1_img)

        st.success("✅ Stage 1 complete")
    else:
        st.error("❌ Stage 1 model not available")
        return results

    # Stage 2: Refinement (if not disabled)
    if not stage1_only and models.get("stage2_pipeline") is not None:
        st.info("✨ Running Stage 2: Semantic refinement...")

        # Encode text for Stage 2
        text_embeddings = encode_text(text_prompt, models, device)

        # Convert sketch tensor to numpy for region extraction
        sketch_np = (sketch_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

        stage2_output = run_stage2(
            stage1_output=stage1_output,
            sketch_np=sketch_np,
            text_prompt=text_prompt,
            text_embeddings=text_embeddings,
            models=models,
            strength=refinement_strength,
            seed=seed
        )
        results["stage2_output"] = stage2_output

        # Convert to PIL for display
        stage2_img = stage2_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        stage2_img = (stage2_img * 255).astype(np.uint8)
        results["stage2_pil"] = Image.fromarray(stage2_img)

        st.success("✅ Stage 2 complete")

    return results


# =============================================================================
# UI Components
# =============================================================================
def render_header():
    """Render the application header."""
    st.title("🎨 Sketch + Text → Image Generator")
    st.markdown("### Dual-Stage Controllable Diffusion with Adaptive Modality Fusion")
    st.markdown("""
    **Pipeline Overview:**
    1. **Stage 1** - Sketch-guided diffusion for coarse structure-preserving generation
    2. **Stage 2** - Semantic refinement with RAGAF attention for region-aware text alignment
    """)


def render_input_section():
    """Render the input section with sketch input options."""
    st.header("📥 Input")

    # Input method selection
    input_method = st.radio(
        "Choose sketch input method:",
        ["Upload Image", "Draw on Canvas"],
        horizontal=True
    )

    sketch_image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a sketch (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            help="Upload a grayscale or RGB sketch image. Will be converted to grayscale and resized to 512x512."
        )
        if uploaded_file is not None:
            sketch_image = Image.open(uploaded_file)
            st.image(sketch_image, caption="Uploaded Sketch", use_column_width=True)

    else:  # Draw on Canvas
        st.markdown("**Draw your sketch below:**")

        # Canvas settings
        stroke_width = st.slider("Stroke Width", 1, 25, 3)
        stroke_color = st.color_picker("Stroke Color", "#000000")
        bg_color = st.color_picker("Background Color", "#FFFFFF")

        # Create canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=SKETCH_SIZE,
            width=SKETCH_SIZE,
            drawing_mode="freedraw",
            key="sketch_canvas",
        )

        if canvas_result.image_data is not None:
            # Check if canvas has content (not just empty white background)
            canvas_data = canvas_result.image_data
            # Convert to grayscale and check if there's any drawing
            if len(canvas_data.shape) == 3:
                gray = np.mean(canvas_data[:, :, :3], axis=2)
                if np.std(gray) > 5:  # Has content
                    sketch_image = canvas_data
                    st.image(sketch_image, caption="Your Drawing", use_column_width=True)

    return sketch_image


def render_controls():
    """Render the control panel."""
    st.header("⚙️ Controls")

    # Text prompt
    text_prompt = st.text_area(
        "Text Prompt",
        value="A photo of a cat sitting on a couch",
        height=80,
        help="Describe what you want to generate. Be specific about objects, style, and details."
    )

    # Refinement strength
    refinement_strength = st.slider(
        "Refinement Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Higher values = more change from Stage 1 output (0.1-0.4 recommended for subtle refinement)"
    )

    # Guidance scale
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.5,
        help="Classifier-free guidance scale. Higher values = stronger prompt adherence (7.5 is typical)"
    )

    # Seed
    use_seed = st.checkbox("Use Fixed Seed", value=False)
    seed = None
    if use_seed:
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=2147483647,
            value=42,
            help="Fixed seed for reproducible generation"
        )

    # Stage options
    st.subheader("Pipeline Options")
    stage1_only = st.checkbox("Stage 1 Only (disable Stage 2)", value=False)
    show_stage2 = st.checkbox("Show Stage 2 Output", value=True)

    return {
        "text_prompt": text_prompt,
        "refinement_strength": refinement_strength,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "stage1_only": stage1_only,
        "show_stage2": show_stage2
    }


def render_output_section(results: Dict, show_stage2: bool = True):
    """Render the output section with results."""
    st.header("📤 Output")

    if not results:
        st.info("Generate an image to see results here")
        return

    # Determine number of columns
    num_images = 1  # Always show input
    if "stage1_pil" in results:
        num_images += 1
    if show_stage2 and "stage2_pil" in results:
        num_images += 1

    cols = st.columns(num_images)
    col_idx = 0

    # Input sketch
    with cols[col_idx]:
        st.subheader("📝 Input Sketch")
        if "input_sketch" in results:
            st.image(results["input_sketch"], use_column_width=True)
        col_idx += 1

    # Stage 1 output
    if "stage1_pil" in results and col_idx < len(cols):
        with cols[col_idx]:
            st.subheader("🎨 Stage 1 (Coarse)")
            st.image(results["stage1_pil"], use_column_width=True)
            col_idx += 1

    # Stage 2 output
    if show_stage2 and "stage2_pil" in results and col_idx < len(cols):
        with cols[col_idx]:
            st.subheader("✨ Stage 2 (Refined)")
            st.image(results["stage2_pil"], use_column_width=True)

    # Download button for final image
    if "stage2_pil" in results:
        final_img = results["stage2_pil"]
    elif "stage1_pil" in results:
        final_img = results["stage1_pil"]
    else:
        final_img = None

    if final_img is not None:
        # Convert to bytes for download
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Final Image",
            data=byte_im,
            file_name="generated_image.png",
            mime="image/png",
            use_container_width=True
        )


# =============================================================================
# Main Application
# =============================================================================
def main():
    """Main Streamlit application."""
    # Page config
    st.set_page_config(
        page_title="RAGAF-Diffusion Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    render_header()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Model Checkpoints:**
        - Stage 1: `epoch_18.pt` (final)
        - Stage 2: `epoch_6.pt` (final)

        **Device:**
        """)
        device = get_device()
        if device == "cuda":
            st.success(f"🚀 GPU ({torch.cuda.get_device_name(0)})")
        else:
            st.warning(f"⚠️ CPU (slow)")

        st.markdown("""
        **Tips:**
        - Clear, simple sketches work best
        - Be specific in your text prompts
        - Use lower refinement strength (0.2-0.4) for subtle improvements
        - Use fixed seed for reproducible results
        """)

        # Load models button
        st.header("🔧 Model Loading")
        if st.button("Load/Reload Models", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # Load models
    models = load_models()

    # Check model availability
    stage1_available = models.get("stage1_pipeline") is not None
    stage2_available = models.get("stage2_pipeline") is not None

    if not stage1_available and not stage2_available:
        st.error("❌ No models available. Please ensure checkpoints are available.")
        st.stop()

    # Layout: two columns for inputs and controls
    col_input, col_controls = st.columns([2, 1])

    with col_input:
        sketch_image = render_input_section()

    with col_controls:
        controls = render_controls()

    # Generation section
    st.markdown("---")
    generate_col, _ = st.columns([1, 3])
    with generate_col:
        generate_clicked = st.button(
            "🚀 Generate Image",
            use_container_width=True,
            type="primary",
            disabled=sketch_image is None
        )

    # Results section
    results = {}

    if generate_clicked and sketch_image is not None:
        if not controls["text_prompt"].strip():
            st.error("❌ Please enter a text prompt")
            st.stop()

        # Run pipeline
        with st.spinner("Generating image..."):
            try:
                results = run_pipeline(
                    sketch_input=sketch_image,
                    text_prompt=controls["text_prompt"],
                    models=models,
                    stage1_only=controls["stage1_only"],
                    refinement_strength=controls["refinement_strength"],
                    guidance_scale=controls["guidance_scale"],
                    seed=controls["seed"]
                )
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                import traceback
                st.expander("Error Details").code(traceback.format_exc())

    # Display results (persist across reruns using session state)
    if "last_results" not in st.session_state:
        st.session_state.last_results = {}

    if results:
        st.session_state.last_results = results

    render_output_section(
        st.session_state.last_results,
        show_stage2=controls["show_stage2"]
    )

    # Footer
    st.markdown("---")
    st.caption("RAGAF-Diffusion Research Team | Dual-Stage Controllable Diffusion with Adaptive Modality Fusion")


if __name__ == "__main__":
    main()
