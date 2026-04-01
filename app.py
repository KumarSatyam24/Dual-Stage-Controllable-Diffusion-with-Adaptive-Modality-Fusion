"""
Sketch + Text → Image Generator (Dual-Stage Diffusion)

A production-ready Streamlit application for interacting with a dual-stage
diffusion model with adaptive modality fusion.

Stage 1: Coarse structure generation from sketch
Stage 2: Semantic refinement with text guidance
"""

import streamlit as st
import torch
import torch.cuda.amp as amp
import numpy as np
from PIL import Image
import io
from pathlib import Path
import cv2
from streamlit_drawable_canvas import st_canvas
from typing import Optional, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# MODEL LOADING (Cached)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_models(checkpoint_path: Optional[str] = None):
    """
    Load Stage 1 and Stage 2 models once and cache them.

    This function is cached using @st.cache_resource to avoid reloading
    models on every Streamlit interaction.

    Args:
        checkpoint_path: Optional path to model checkpoint

    Returns:
        Tuple of (stage1_model, stage2_model, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Import model classes
        from src.models.stage1_diffusion import (
            Stage1SketchGuidedDiffusion,
            Stage1DiffusionPipeline
        )
        from src.models.stage2_refinement import (
            Stage2SemanticRefinement,
            Stage2RefinementPipeline
        )
        from src.data.region_graph import RegionGraph
        from diffusers import UNet2DConditionModel

        # Initialize Stage 1 model
        stage1_model = Stage1SketchGuidedDiffusion(
            pretrained_model_name="runwayml/stable-diffusion-v1-5",
            freeze_base_unet=True
        ).to(device)
        stage1_model.eval()

        # Initialize Stage 1 pipeline
        stage1_pipeline = Stage1DiffusionPipeline(
            model=stage1_model,
            num_inference_steps=50,
            guidance_scale=7.5,
            device=device
        )

        # Create a separate UNet for Stage 2 (Stage 2 modifies input channels)
        stage2_unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet"
        ).to(device)
        stage2_unet.eval()

        # Initialize Stage 2 model with its own UNet
        stage2_model = Stage2SemanticRefinement(
            unet=stage2_unet,
            node_feature_dim=6,
            text_dim=768,
            hidden_dim=512,
            num_graph_layers=2,
            num_attention_heads=8,
            fusion_method="learned",
            use_region_adaptive_fusion=True,
            num_timesteps=1000,
            use_residual=True,
            concatenate_stage1=True,
            residual_alpha=0.2
        ).to(device)
        stage2_model.eval()

        # Initialize Stage 2 pipeline (shares VAE with Stage 1)
        stage2_pipeline = Stage2RefinementPipeline(
            stage2_model=stage2_model,
            vae=stage1_model.vae,
            num_inference_steps=30,
            guidance_scale=7.5,
            device=device
        )

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'stage1_state_dict' in checkpoint:
                stage1_model.load_state_dict(checkpoint['stage1_state_dict'])
            if 'stage2_state_dict' in checkpoint:
                stage2_model.load_state_dict(checkpoint['stage2_state_dict'])
            st.success(f"✅ Loaded checkpoint from {checkpoint_path}")

        return stage1_pipeline, stage2_pipeline, device

    except Exception as e:
        st.warning(f"⚠️ Models not loaded: {e}")
        st.info("Running in demo mode - configure model paths in load_models()")
        return None, None, device


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess sketch/image for model inference.

    Args:
        image: Input image as numpy array (H, W, C) or PIL Image
        target_size: Target resolution (H, W)
        normalize: Whether to normalize to [-1, 1]

    Returns:
        Preprocessed image tensor (1, C, H, W)
    """
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        # Handle RGBA or grayscale
        if image.shape[-1] == 4:
            # Convert RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 3:
            # Ensure RGB (not BGR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image.astype('uint8'))

    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to target size
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Convert to grayscale for sketch (Stage 1 expects single channel)
    if len(img_array.shape) == 3:
        # Convert RGB to grayscale
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # Add channel dimension (1, H, W)
    img_gray = np.expand_dims(img_gray, axis=0)

    # Normalize to [-1, 1] if requested
    if normalize:
        img_gray = 2.0 * img_gray - 1.0

    # Convert to tensor (1, 1, H, W)
    tensor = torch.from_numpy(img_gray).unsqueeze(0)

    return tensor


def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Convert model output tensor to PIL Image.

    Args:
        tensor: Output tensor from model (1, 3, H, W) or (1, C, H, W)

    Returns:
        PIL Image
    """
    # Remove batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Move to CPU
    tensor = tensor.cpu()

    # Handle different channel dimensions
    if tensor.shape[0] == 1:
        # Grayscale to RGB
        tensor = tensor.repeat(3, 1, 1)

    # Clamp to valid range
    if tensor.min() < 0:
        # Assume [-1, 1] range
        tensor = (tensor + 1.0) / 2.0

    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Convert to numpy (C, H, W) -> (H, W, C)
    img_array = tensor.permute(1, 2, 0).numpy()

    # Convert to uint8
    img_array = (img_array * 255.0).astype(np.uint8)

    return Image.fromarray(img_array)


# =============================================================================
# REGION GRAPH CREATION (for Stage 2)
# =============================================================================

def create_region_graph_from_sketch(
    sketch: torch.Tensor,
    text_prompt: str
) -> Any:
    """
    Create a region graph from sketch for Stage 2.

    In a production app, this would use region detection algorithms.
    For this demo, we create a simple graph with a single region.

    Args:
        sketch: Preprocessed sketch tensor (1, 1, H, W)
        text_prompt: Text prompt

    Returns:
        RegionGraph object
    """
    try:
        from src.data.region_graph import RegionGraph

        # Extract sketch as numpy
        sketch_np = sketch.cpu().squeeze().numpy()

        # Normalize to [0, 1]
        if sketch_np.min() < 0:
            sketch_np = (sketch_np + 1.0) / 2.0

        # Create a simple region graph
        # In practice, you would use region segmentation
        H, W = sketch_np.shape

        # Create a single region mask covering the whole image
        region_mask = np.ones((H, W), dtype=np.float32)

        # Node features: [x_center, y_center, width, height, aspect_ratio, area_ratio]
        node_features = np.array([
            [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]  # Single region covering whole image
        ], dtype=np.float32)

        # Create region graph
        region_graph = RegionGraph(
            node_features=node_features,
            edge_index=np.array([[0], [0]]),  # Self-loop
            region_masks=[region_mask],
            edge_weights=np.array([1.0])
        )

        return region_graph

    except Exception as e:
        st.error(f"Error creating region graph: {e}")
        return None


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    stage1_pipeline: Any,
    stage2_pipeline: Any,
    sketch: Image.Image,
    prompt: str,
    device: torch.device,
    refinement_strength: float = 0.4,
    guidance_scale: float = 7.5,
    seed: int = 42,
    stage1_only: bool = False,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    Run the dual-stage inference pipeline.

    Args:
        stage1_pipeline: Stage 1 pipeline
        stage2_pipeline: Stage 2 pipeline
        sketch: Input sketch (PIL Image)
        prompt: Text prompt
        device: torch device
        refinement_strength: Strength of Stage 2 refinement (0.1-1.0)
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        stage1_only: If True, skip Stage 2
        target_size: Model input/output size

    Returns:
        Tuple of (stage1_output, stage2_output or None)
    """
    # Set random seed
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Preprocess sketch
    sketch_tensor = preprocess_image(sketch, target_size=target_size, normalize=True)
    sketch_tensor = sketch_tensor.to(device)

    # Stage 1: Coarse Generation
    with torch.no_grad():
        with st.spinner("🎨 Stage 1: Generating coarse structure..."):
            if device.type == 'cuda':
                with amp.autocast():
                    stage1_output = stage1_pipeline.generate(
                        sketch=sketch_tensor,
                        text_prompt=prompt,
                        height=target_size[0],
                        width=target_size[1],
                        seed=seed
                    )
            else:
                stage1_output = stage1_pipeline.generate(
                    sketch=sketch_tensor,
                    text_prompt=prompt,
                    height=target_size[0],
                    width=target_size[1],
                    seed=seed
                )

    # Convert Stage 1 output to PIL
    stage1_image = postprocess_tensor(stage1_output)

    # Skip Stage 2 if requested
    if stage1_only:
        return stage1_image, None

    # Stage 2: Refinement
    with torch.no_grad():
        with st.spinner("✨ Stage 2: Refining with semantic details..."):
            # Create region graph
            region_graph = create_region_graph_from_sketch(sketch_tensor, prompt)

            # Encode text using Stage 1's text encoder
            text_embeddings = stage1_pipeline.model.encode_text([prompt])
            text_embeddings = text_embeddings.to(device)

            if device.type == 'cuda':
                with amp.autocast():
                    stage2_output = stage2_pipeline.refine(
                        stage1_image=stage1_output,
                        region_graph=region_graph,
                        text_prompt=prompt,
                        text_embeddings=text_embeddings,
                        strength=refinement_strength,
                        seed=seed
                    )
            else:
                stage2_output = stage2_pipeline.refine(
                    stage1_image=stage1_output,
                    region_graph=region_graph,
                    text_prompt=prompt,
                    text_embeddings=text_embeddings,
                    strength=refinement_strength,
                    seed=seed
                )

    # Convert Stage 2 output to PIL
    stage2_image = postprocess_tensor(stage2_output)

    return stage1_image, stage2_image


# =============================================================================
# UI COMPONENTS
# =============================================================================

def create_download_button(image: Image.Image, filename: str = "generated_image.png"):
    """
    Create a download button for an image.

    Args:
        image: PIL Image to download
        filename: Download filename
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label=f"📥 Download {filename}",
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )


def render_sketch_input() -> Tuple[Optional[Image.Image], str]:
    """
    Render the sketch input section.

    Returns:
        Tuple of (sketch_image, input_method)
    """
    st.markdown("### 📝 Sketch Input")

    input_method = st.radio(
        "Choose input method:",
        ["🖼️ Upload Image", "✏️ Draw Sketch"],
        horizontal=True,
        key="input_method"
    )

    sketch = None

    if input_method == "🖼️ Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a sketch or image",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Upload a sketch or image file"
        )

        if uploaded_file is not None:
            sketch = Image.open(uploaded_file)
            st.image(sketch, caption="Uploaded Image", use_container_width=True)

    else:  # Draw Sketch
        st.markdown("**Draw your sketch on the canvas:**")

        # Drawing controls
        col1, col2 = st.columns(2)
        with col1:
            stroke_width = st.slider("Stroke width", 1, 10, 3)
        with col2:
            stroke_color = st.color_picker("Stroke color", "#000000")

        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#ffffff",
            background_image=None,
            update_streamlit=True,
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="sketch_canvas",
        )

        if canvas_result.image_data is not None:
            # Convert canvas to PIL Image
            sketch = Image.fromarray(canvas_result.image_data.astype('uint8'))
            st.image(sketch, caption="Your Sketch", use_container_width=True)

    return sketch, input_method


def render_text_input() -> str:
    """
    Render the text prompt input section.

    Returns:
        Text prompt string
    """
    st.markdown("### 💬 Text Prompt")

    prompt = st.text_area(
        "Describe what you want to generate:",
        placeholder="e.g., A red car on a street with buildings in the background",
        height=100,
        help="Be descriptive for better results"
    )

    # Quick prompts
    st.markdown("**Quick prompts:**")
    quick_prompts = [
        "A cat sitting on a couch",
        "A sunset over mountains",
        "A modern building with glass windows",
        "A tree in a field",
    ]

    cols = st.columns(len(quick_prompts))
    for i, qp in enumerate(quick_prompts):
        with cols[i]:
            if st.button(qp, key=f"qp_{i}", use_container_width=True):
                st.session_state['quick_prompt'] = qp
                st.rerun()

    # Check if quick prompt was selected
    if 'quick_prompt' in st.session_state:
        prompt = st.session_state['quick_prompt']
        del st.session_state['quick_prompt']

    return prompt


def render_advanced_controls() -> Dict[str, Any]:
    """
    Render advanced control options in the sidebar.

    Returns:
        Dictionary of control values
    """
    st.sidebar.markdown("## ⚙️ Advanced Settings")

    # Model settings
    resolution = st.sidebar.selectbox(
        "Resolution",
        [256, 512, 768],
        index=1,
        help="Higher resolution = more detail but slower"
    )

    # Guidance settings
    guidance_scale = st.sidebar.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.5,
        help="Higher = more faithful to text, Lower = more creative"
    )

    # Refinement settings
    refinement_strength = st.sidebar.slider(
        "Refinement Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Higher = more refinement details"
    )

    # Seed settings
    use_random_seed = st.sidebar.checkbox(
        "Random seed",
        value=False,
        help="Use random seed for each generation"
    )

    if use_random_seed:
        seed = np.random.randint(0, 2147483647)
    else:
        seed = st.sidebar.number_input(
            "Seed",
            min_value=0,
            max_value=2147483647,
            value=42,
            step=1
        )

    # Ablation study toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔬 Research Mode")

    stage1_only = st.sidebar.checkbox(
        "Stage 1 Only (Ablation)",
        value=False,
        help="Skip Stage 2 refinement for ablation study"
    )

    show_stage1 = st.sidebar.checkbox(
        "Show Stage 1 Output",
        value=True,
        help="Display intermediate Stage 1 result"
    )

    return {
        'resolution': resolution,
        'guidance_scale': guidance_scale,
        'refinement_strength': refinement_strength,
        'seed': seed,
        'stage1_only': stage1_only,
        'show_stage1': show_stage1
    }


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="Dual-Stage Diffusion Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("🎨 Sketch + Text → Image Generator")
    st.markdown("### Dual-Stage Controllable Diffusion with Adaptive Modality Fusion")
    st.markdown("---")

    # Load models (cached)
    with st.spinner("🚀 Loading models..."):
        stage1_pipeline, stage2_pipeline, device = load_models()

    # Device info
    device_info = f"**Device:** {device.type.upper()}"
    if device.type == 'cuda':
        device_info += f" ({torch.cuda.get_device_name(0)})"
    st.sidebar.info(device_info)

    # Check if models loaded
    models_ready = stage1_pipeline is not None and stage2_pipeline is not None

    if not models_ready:
        st.error("""
        ⚠️ **Models not loaded**

        Please ensure:
        1. Model checkpoint files exist
        2. Required dependencies are installed
        3. GPU is available (recommended)

        Configure model loading in the `load_models()` function.
        """)

    # Render advanced controls
    controls = render_advanced_controls()

    # Main layout: Input (left) | Output (right)
    col_input, col_output = st.columns([1, 1])

    # ==========================================================================
    # LEFT COLUMN: INPUT
    # ==========================================================================
    with col_input:
        # Sketch input
        sketch, input_method = render_sketch_input()

        st.markdown("---")

        # Text input
        prompt = render_text_input()

        # Generation button
        st.markdown("---")
        generate_clicked = st.button(
            "🚀 Generate Image",
            type="primary",
            use_container_width=True,
            disabled=not models_ready
        )

    # ==========================================================================
    # RIGHT COLUMN: OUTPUT
    # ==========================================================================
    with col_output:
        st.markdown("### 🖼️ Generated Results")

        if generate_clicked:
            # Validate inputs
            if sketch is None:
                st.error("❌ Please provide a sketch (upload or draw)")
            elif not prompt or not prompt.strip():
                st.error("❌ Please enter a text prompt")
            else:
                # Run generation
                try:
                    target_size = (controls['resolution'], controls['resolution'])

                    stage1_image, stage2_image = run_inference(
                        stage1_pipeline=stage1_pipeline,
                        stage2_pipeline=stage2_pipeline,
                        sketch=sketch,
                        prompt=prompt,
                        device=device,
                        refinement_strength=controls['refinement_strength'],
                        guidance_scale=controls['guidance_scale'],
                        seed=controls['seed'],
                        stage1_only=controls['stage1_only'],
                        target_size=target_size
                    )

                    # Store in session state
                    st.session_state['stage1_image'] = stage1_image
                    st.session_state['stage2_image'] = stage2_image
                    st.session_state['last_prompt'] = prompt

                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                    st.exception(e)

        # Display results
        if 'stage1_image' in st.session_state:
            stage1_image = st.session_state['stage1_image']
            stage2_image = st.session_state['stage2_image']

            # Show Stage 1 output if requested
            if controls['show_stage1'] and not controls['stage1_only']:
                st.markdown("**Stage 1 Output (Coarse):**")
                st.image(stage1_image, use_container_width=True)

                # Download button for Stage 1
                col_dl1, _ = st.columns([1, 3])
                with col_dl1:
                    create_download_button(stage1_image, "stage1_coarse.png")

                st.markdown("---")

            # Show final output
            if stage2_image is not None:
                st.markdown("**Final Output (Stage 2 Refined):**")
                display_image = stage2_image
                download_name = "stage2_refined.png"
            else:
                st.markdown("**Generated Output (Stage 1 Only):**")
                display_image = stage1_image
                download_name = "stage1_output.png"

            st.image(display_image, use_container_width=True)

            # Download button for final output
            col_dl2, _ = st.columns([1, 3])
            with col_dl2:
                create_download_button(display_image, download_name)

            st.success("✅ Generation complete!")

    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.85em;'>
        <p>Powered by <strong>Dual-Stage Controllable Diffusion with Adaptive Modality Fusion</strong></p>
        <p>Stage 1: Structure Generation | Stage 2: Semantic Refinement with RAGAF</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
