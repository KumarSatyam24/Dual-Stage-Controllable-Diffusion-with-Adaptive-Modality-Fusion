"""
Inference Script for RAGAF-Diffusion

Generate images from sketches and text prompts using trained RAGAF-Diffusion model.

Supports:
- Single-stage or dual-stage inference
- Batch processing
- Visualization of regions and attention maps
- Saving intermediate outputs

Author: RAGAF-Diffusion Research Team
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from configs.config import ModelConfig, InferenceConfig
from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from models.stage2_refinement import Stage2SemanticRefinement, Stage2RefinementPipeline
from data.sketch_extraction import SketchExtractor
from data.region_extraction import RegionExtractor
from data.region_graph import RegionGraphBuilder


class RAGAFDiffusionInference:
    """
    Inference pipeline for RAGAF-Diffusion.
    
    Handles end-to-end generation from sketch and text to final image.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        device: str = "cuda"
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_config: Model configuration
            inference_config: Inference configuration
            device: Device to run on
        """
        self.model_config = model_config
        self.inference_config = inference_config
        self.device = device
        
        # Load models
        self.load_models()
        
        # Setup extractors
        self.sketch_extractor = SketchExtractor(method="canny", invert=True)
        self.region_extractor = RegionExtractor(min_region_area=100, max_num_regions=50)
        self.graph_builder = RegionGraphBuilder(
            graph_type="hybrid",
            image_size=(512, 512)
        )
        
        # Create output directory
        Path(inference_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Inference pipeline ready on {device}")
    
    def load_models(self):
        """Load trained models from checkpoints."""
        print("Loading models...")
        
        # Stage 1
        if Path(self.inference_config.stage1_checkpoint).exists():
            print(f"Loading Stage 1 from: {self.inference_config.stage1_checkpoint}")
            
            self.stage1_model = Stage1SketchGuidedDiffusion(
                pretrained_model_name=self.model_config.pretrained_model_name
            ).to(self.device)
            
            checkpoint = torch.load(
                self.inference_config.stage1_checkpoint,
                map_location=self.device
            )
            self.stage1_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.stage1_model.eval()
            
            # Create Stage 1 pipeline
            self.stage1_pipeline = Stage1DiffusionPipeline(
                model=self.stage1_model,
                num_inference_steps=self.inference_config.num_inference_steps,
                guidance_scale=self.inference_config.guidance_scale,
                device=self.device
            )
        else:
            print(f"Warning: Stage 1 checkpoint not found at {self.inference_config.stage1_checkpoint}")
            self.stage1_model = None
            self.stage1_pipeline = None
        
        # Stage 2
        if Path(self.inference_config.stage2_checkpoint).exists():
            print(f"Loading Stage 2 from: {self.inference_config.stage2_checkpoint}")
            
            from diffusers import UNet2DConditionModel
            unet = UNet2DConditionModel.from_pretrained(
                self.model_config.pretrained_model_name,
                subfolder="unet"
            )
            
            self.stage2_model = Stage2SemanticRefinement(
                unet=unet,
                node_feature_dim=self.model_config.node_feature_dim,
                text_dim=self.model_config.text_dim,
                hidden_dim=self.model_config.hidden_dim
            ).to(self.device)
            
            checkpoint = torch.load(
                self.inference_config.stage2_checkpoint,
                map_location=self.device
            )
            self.stage2_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.stage2_model.eval()
            
            # Load VAE for Stage 2
            vae = AutoencoderKL.from_pretrained(
                self.model_config.pretrained_model_name,
                subfolder="vae"
            ).to(self.device)
            
            # Create Stage 2 pipeline
            self.stage2_pipeline = Stage2RefinementPipeline(
                stage2_model=self.stage2_model,
                vae=vae,
                num_inference_steps=self.inference_config.num_refinement_steps,
                guidance_scale=self.inference_config.guidance_scale,
                device=self.device
            )
        else:
            print(f"Warning: Stage 2 checkpoint not found at {self.inference_config.stage2_checkpoint}")
            self.stage2_model = None
            self.stage2_pipeline = None
        
        print("Models loaded successfully")
    
    @torch.no_grad()
    def generate(
        self,
        sketch_path: str,
        text_prompt: str,
        output_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Image.Image]:
        """
        Generate image from sketch and text prompt.
        
        Args:
            sketch_path: Path to sketch image
            text_prompt: Text prompt
            output_name: Output filename (without extension)
            seed: Random seed
        
        Returns:
            Dict with generated images at different stages
        """
        if output_name is None:
            output_name = Path(sketch_path).stem
        
        print(f"\nGenerating image for: {output_name}")
        print(f"  Sketch: {sketch_path}")
        print(f"  Prompt: {text_prompt}")
        
        # Load sketch
        sketch_pil = Image.open(sketch_path).convert("L")
        sketch_pil = sketch_pil.resize((512, 512))
        
        # Convert to tensor
        sketch_tensor = torch.from_numpy(np.array(sketch_pil)).float() / 255.0
        sketch_tensor = sketch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        
        results = {}
        
        # Extract regions
        if self.inference_config.visualize_regions:
            sketch_np = np.array(sketch_pil)
            regions = self.region_extractor.extract_regions(sketch_np)
            region_vis = self.region_extractor.visualize_regions(sketch_np, regions)
            results["regions"] = Image.fromarray(region_vis)
            
            # Build graph
            region_graph = self.graph_builder.build_graph(regions)
            print(f"  Extracted {region_graph.num_nodes} regions")
        
        # Stage 1: Sketch-guided generation
        if self.stage1_pipeline is not None:
            print("  Running Stage 1 (Sketch-guided generation)...")
            
            stage1_output = self.stage1_pipeline.generate(
                sketch=sketch_tensor,
                text_prompt=text_prompt,
                height=512,
                width=512,
                seed=seed
            )
            
            # Convert to PIL
            stage1_img = stage1_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
            stage1_img = (stage1_img * 255).astype(np.uint8)
            stage1_pil = Image.fromarray(stage1_img)
            
            results["stage1"] = stage1_pil
            print("  Stage 1 complete")
        else:
            print("  Skipping Stage 1 (no checkpoint)")
            stage1_pil = None
            stage1_output = None
        
        # Stage 2: Semantic refinement
        if self.stage2_pipeline is not None and stage1_output is not None:
            print("  Running Stage 2 (Semantic refinement)...")
            
            # Get text embeddings
            tokenizer = CLIPTokenizer.from_pretrained(
                self.model_config.pretrained_model_name,
                subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_config.pretrained_model_name,
                subfolder="text_encoder"
            ).to(self.device)
            
            text_inputs = tokenizer(
                text_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = text_encoder(text_inputs.input_ids.to(self.device))[0].squeeze(0)
            
            stage2_output = self.stage2_pipeline.refine(
                stage1_image=stage1_output,
                region_graph=region_graph,
                text_prompt=text_prompt,
                text_embeddings=text_embeddings,
                strength=self.inference_config.refinement_strength,
                seed=seed
            )
            
            # Convert to PIL
            stage2_img = stage2_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
            stage2_img = (stage2_img * 255).astype(np.uint8)
            stage2_pil = Image.fromarray(stage2_img)
            
            results["stage2"] = stage2_pil
            print("  Stage 2 complete")
        else:
            print("  Skipping Stage 2")
        
        # Save results
        self.save_results(results, sketch_pil, text_prompt, output_name)
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Image.Image],
        sketch: Image.Image,
        text_prompt: str,
        output_name: str
    ):
        """Save generation results."""
        output_dir = Path(self.inference_config.output_dir) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sketch
        sketch.save(output_dir / "sketch.png")
        
        # Save generated images
        if "stage1" in results:
            results["stage1"].save(output_dir / "stage1_output.png")
        
        if "stage2" in results:
            results["stage2"].save(output_dir / "stage2_output.png")
        
        if "regions" in results:
            results["regions"].save(output_dir / "regions.png")
        
        # Create comparison grid
        self.create_comparison_grid(results, sketch, text_prompt, output_dir)
        
        # Save prompt
        with open(output_dir / "prompt.txt", "w") as f:
            f.write(text_prompt)
        
        print(f"  Results saved to: {output_dir}")
    
    def create_comparison_grid(
        self,
        results: Dict[str, Image.Image],
        sketch: Image.Image,
        text_prompt: str,
        output_dir: Path
    ):
        """Create a comparison grid of results."""
        # Determine number of images
        images = [sketch]
        titles = ["Input Sketch"]
        
        if "regions" in results:
            images.append(results["regions"])
            titles.append("Extracted Regions")
        
        if "stage1" in results:
            images.append(results["stage1"])
            titles.append("Stage 1: Sketch-Guided")
        
        if "stage2" in results:
            images.append(results["stage2"])
            titles.append("Stage 2: Refined")
        
        # Create figure
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        
        if len(images) == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        
        plt.suptitle(f'"{text_prompt}"', fontsize=14, y=0.95)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAGAF-Diffusion Inference")
    parser.add_argument("--sketch", type=str, required=True, help="Path to sketch image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default=None, help="Output name")
    parser.add_argument("--stage1_checkpoint", type=str, default="./checkpoints/stage1/final.pt")
    parser.add_argument("--stage2_checkpoint", type=str, default="./checkpoints/stage2/final.pt")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Create configs
    model_config = ModelConfig()
    inference_config = InferenceConfig(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Create pipeline
    pipeline = RAGAFDiffusionInference(
        model_config=model_config,
        inference_config=inference_config,
        device=args.device
    )
    
    # Generate
    results = pipeline.generate(
        sketch_path=args.sketch,
        text_prompt=args.prompt,
        output_name=args.output,
        seed=args.seed
    )
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
