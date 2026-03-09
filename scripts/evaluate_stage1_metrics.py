"""
Stage 1 Model Evaluation Metrics
Comprehensive evaluation for sketch-guided diffusion model

Metrics Included:
1. Sketch Fidelity - How well the output follows the sketch structure
2. Image Quality - FID, IS scores
3. Semantic Accuracy - CLIP similarity between prompt and output
4. Edge Consistency - Edge map comparison
5. Human Evaluation Proxy - Perceptual metrics
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg
from tqdm import tqdm
import json

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("⚠️  CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("⚠️  LPIPS not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False


class SketchFidelityMetrics:
    """Measure how well generated images follow the input sketch"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def extract_edges(self, image: np.ndarray) -> np.ndarray:
        """Extract edge map from image using Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def edge_consistency_score(self, sketch: np.ndarray, generated: np.ndarray) -> float:
        """
        Compute edge consistency between sketch and generated image
        
        Returns:
            Score between 0-1, where 1 means perfect edge alignment
        """
        # Extract edges from generated image
        gen_edges = self.extract_edges(generated)
        
        # Normalize sketch to 0-255 range
        if sketch.max() <= 1.0:
            sketch = (sketch * 255).astype(np.uint8)
        else:
            sketch = sketch.astype(np.uint8)
            
        # Resize if needed
        if sketch.shape != gen_edges.shape:
            sketch = cv2.resize(sketch, (gen_edges.shape[1], gen_edges.shape[0]))
        
        # Compute intersection over union (IoU) of edge maps
        intersection = np.logical_and(sketch > 127, gen_edges > 127).sum()
        union = np.logical_or(sketch > 127, gen_edges > 127).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    
    def structural_similarity(self, sketch: np.ndarray, generated: np.ndarray) -> float:
        """
        Compute SSIM-like metric for sketch-image similarity
        """
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale if needed
        if len(generated.shape) == 3:
            generated_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
        else:
            generated_gray = generated
            
        if len(sketch.shape) == 3:
            sketch_gray = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
        else:
            sketch_gray = sketch
        
        # Resize if needed
        if sketch_gray.shape != generated_gray.shape:
            sketch_gray = cv2.resize(sketch_gray, (generated_gray.shape[1], generated_gray.shape[0]))
        
        score = ssim(sketch_gray, generated_gray, data_range=255)
        return float(score)


class ImageQualityMetrics:
    """Measure image quality using standard metrics"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = None
        self.lpips_model = None
        
    def load_inception(self):
        """Load Inception V3 model for FID/IS computation"""
        if self.inception_model is None:
            self.inception_model = inception_v3(pretrained=True, transform_input=False)
            self.inception_model = self.inception_model.to(self.device)
            self.inception_model.eval()
    
    def load_lpips(self):
        """Load LPIPS model for perceptual similarity"""
        if LPIPS_AVAILABLE and self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
    
    def calculate_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """
        Calculate Fréchet Inception Distance (FID)
        
        Lower is better - measures distribution distance between real and generated
        """
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)
    
    def get_inception_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from Inception model"""
        self.load_inception()
        
        with torch.no_grad():
            # Resize to 299x299 for Inception
            if images.shape[-1] != 299:
                images = torch.nn.functional.interpolate(
                    images, size=(299, 299), mode='bilinear', align_corners=False
                )
            
            # Get features from pool layer
            features = self.inception_model(images)
            
        return features.cpu().numpy()
    
    def calculate_inception_score(self, images: torch.Tensor, splits=10) -> Tuple[float, float]:
        """
        Calculate Inception Score (IS)
        
        Returns:
            (mean, std) - Higher is better
        """
        self.load_inception()
        
        with torch.no_grad():
            # Get predictions
            preds = []
            for i in range(0, len(images), 32):
                batch = images[i:i+32]
                if batch.shape[-1] != 299:
                    batch = torch.nn.functional.interpolate(
                        batch, size=(299, 299), mode='bilinear', align_corners=False
                    )
                pred = self.inception_model(batch)
                preds.append(pred.cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
        
        # Calculate IS
        split_scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k+1) * (len(preds) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
            split_scores.append(np.exp(np.mean(scores)))
        
        return float(np.mean(split_scores)), float(np.std(split_scores))


class SemanticAccuracyMetrics:
    """Measure semantic alignment between text prompt and generated image"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        
    def load_clip(self):
        """Load CLIP model"""
        if CLIP_AVAILABLE and self.clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
    
    def compute_clip_score(self, images: List[Image.Image], texts: List[str]) -> float:
        """
        Compute CLIP score - measures text-image alignment
        
        Returns:
            Score between 0-100, higher is better
        """
        if not CLIP_AVAILABLE:
            print("⚠️  CLIP not available, skipping CLIP score")
            return 0.0
        
        self.load_clip()
        
        # Preprocess images
        image_tensors = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        
        # Tokenize text
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Get features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensors)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features * text_features).sum(dim=-1)
        
        return float(similarity.mean().cpu().numpy() * 100)
    
    def compute_category_accuracy(self, images: List[Image.Image], 
                                  true_categories: List[str],
                                  candidate_categories: List[str]) -> Dict[str, float]:
        """
        Compute classification accuracy using CLIP
        
        This is like "true positive" rate - does CLIP classify the image 
        as the correct category?
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not CLIP_AVAILABLE:
            print("⚠️  CLIP not available, skipping category accuracy")
            return {"accuracy": 0.0, "top1": 0.0, "top5": 0.0}
        
        self.load_clip()
        
        correct_top1 = 0
        correct_top5 = 0
        
        for img, true_cat in zip(images, true_categories):
            # Preprocess image
            image_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            
            # Create text prompts for all categories
            text_prompts = [f"a photo of a {cat}" for cat in candidate_categories]
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (image_features @ text_features.T).squeeze(0)
                
            # Get top predictions
            top5_indices = similarity.topk(5).indices.cpu().numpy()
            top1_pred = candidate_categories[top5_indices[0]]
            top5_preds = [candidate_categories[i] for i in top5_indices]
            
            # Check if correct
            if top1_pred == true_cat:
                correct_top1 += 1
                correct_top5 += 1
            elif true_cat in top5_preds:
                correct_top5 += 1
        
        n = len(images)
        return {
            "top1_accuracy": correct_top1 / n if n > 0 else 0.0,
            "top5_accuracy": correct_top5 / n if n > 0 else 0.0,
            "total_samples": n
        }


class Stage1Evaluator:
    """Complete evaluation pipeline for Stage 1 model"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.sketch_metrics = SketchFidelityMetrics(device)
        self.quality_metrics = ImageQualityMetrics(device)
        self.semantic_metrics = SemanticAccuracyMetrics(device)
    
    def evaluate_single(self, sketch: np.ndarray, generated: Image.Image, 
                       prompt: str) -> Dict[str, float]:
        """
        Evaluate a single generated image
        
        Args:
            sketch: Input sketch (numpy array, 0-255)
            generated: Generated image (PIL Image)
            prompt: Text prompt used
            
        Returns:
            Dictionary of metrics
        """
        # Convert PIL to numpy
        generated_np = np.array(generated)
        
        # Sketch fidelity
        edge_score = self.sketch_metrics.edge_consistency_score(sketch, generated_np)
        struct_score = self.sketch_metrics.structural_similarity(sketch, generated_np)
        
        # Semantic alignment
        clip_score = self.semantic_metrics.compute_clip_score([generated], [prompt])
        
        return {
            "edge_consistency": edge_score,
            "structural_similarity": struct_score,
            "clip_score": clip_score,
        }
    
    def evaluate_dataset(self, 
                        sketch_paths: List[Path],
                        generated_paths: List[Path],
                        prompts: List[str],
                        categories: List[str] = None) -> Dict[str, float]:
        """
        Evaluate entire dataset
        
        Args:
            sketch_paths: Paths to input sketches
            generated_paths: Paths to generated images
            prompts: Text prompts used
            categories: Category labels for each image
            
        Returns:
            Comprehensive metrics dictionary
        """
        print("🔍 Evaluating Stage 1 Model Performance...")
        print(f"   Total samples: {len(generated_paths)}")
        print()
        
        # Initialize accumulators
        edge_scores = []
        struct_scores = []
        generated_images = []
        
        # Process each sample
        print("📊 Computing sketch fidelity metrics...")
        for sketch_path, gen_path in tqdm(zip(sketch_paths, generated_paths), total=len(sketch_paths)):
            # Load images
            sketch = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
            generated = Image.open(gen_path).convert('RGB')
            generated_np = np.array(generated)
            
            # Compute metrics
            edge_score = self.sketch_metrics.edge_consistency_score(sketch, generated_np)
            struct_score = self.sketch_metrics.structural_similarity(sketch, generated_np)
            
            edge_scores.append(edge_score)
            struct_scores.append(struct_score)
            generated_images.append(generated)
        
        # Semantic metrics
        print("📊 Computing semantic alignment metrics...")
        clip_score = self.semantic_metrics.compute_clip_score(generated_images, prompts)
        
        # Category accuracy (if categories provided)
        category_acc = {}
        if categories is not None and CLIP_AVAILABLE:
            print("📊 Computing category accuracy...")
            unique_categories = list(set(categories))
            category_acc = self.semantic_metrics.compute_category_accuracy(
                generated_images, categories, unique_categories
            )
        
        # Compile results
        results = {
            "sketch_fidelity": {
                "edge_consistency_mean": float(np.mean(edge_scores)),
                "edge_consistency_std": float(np.std(edge_scores)),
                "structural_similarity_mean": float(np.mean(struct_scores)),
                "structural_similarity_std": float(np.std(struct_scores)),
            },
            "semantic_alignment": {
                "clip_score": clip_score,
            },
            "image_quality": {
                "samples_evaluated": len(generated_images),
            },
            "overall_score": self._compute_overall_score(
                float(np.mean(edge_scores)),
                float(np.mean(struct_scores)),
                clip_score,
                category_acc.get("top1_accuracy", 0.0)
            )
        }
        
        if category_acc:
            results["category_accuracy"] = category_acc
        
        return results
    
    def _compute_overall_score(self, edge_score: float, struct_score: float, 
                               clip_score: float, category_acc: float) -> float:
        """
        Compute weighted overall score (0-100)
        
        This is like an "accuracy" metric for the generative model
        """
        # Normalize scores to 0-100 range
        edge_normalized = edge_score * 100  # Already 0-1
        struct_normalized = (struct_score + 1) * 50  # SSIM is -1 to 1
        clip_normalized = clip_score  # Already 0-100
        category_normalized = category_acc * 100  # 0-1 to 0-100
        
        # Weighted average
        weights = {
            "edge": 0.3,      # 30% - sketch fidelity
            "structure": 0.2,  # 20% - structural similarity
            "clip": 0.3,       # 30% - semantic alignment
            "category": 0.2,   # 20% - category accuracy
        }
        
        overall = (
            weights["edge"] * edge_normalized +
            weights["structure"] * struct_normalized +
            weights["clip"] * clip_normalized +
            weights["category"] * category_normalized
        )
        
        return float(overall)
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print()
        print("=" * 80)
        print("📊 STAGE 1 MODEL EVALUATION RESULTS")
        print("=" * 80)
        print()
        
        # Overall score
        print(f"🎯 OVERALL ACCURACY SCORE: {results['overall_score']:.2f}/100")
        print()
        
        # Sketch fidelity
        print("📐 Sketch Fidelity Metrics:")
        sf = results['sketch_fidelity']
        print(f"   Edge Consistency:       {sf['edge_consistency_mean']:.4f} ± {sf['edge_consistency_std']:.4f}")
        print(f"   Structural Similarity:  {sf['structural_similarity_mean']:.4f} ± {sf['structural_similarity_std']:.4f}")
        print()
        
        # Semantic alignment
        print("🎨 Semantic Alignment:")
        sa = results['semantic_alignment']
        print(f"   CLIP Score:             {sa['clip_score']:.2f}/100")
        print()
        
        # Category accuracy
        if 'category_accuracy' in results:
            ca = results['category_accuracy']
            print("✅ Category Accuracy (True Positive Rate):")
            print(f"   Top-1 Accuracy:         {ca['top1_accuracy']*100:.2f}%")
            print(f"   Top-5 Accuracy:         {ca['top5_accuracy']*100:.2f}%")
            print(f"   Total Samples:          {ca['total_samples']}")
            print()
        
        # Interpretation
        print("📈 Interpretation:")
        score = results['overall_score']
        if score >= 80:
            print("   🌟 EXCELLENT - Model performs very well!")
        elif score >= 70:
            print("   ✅ GOOD - Model performs well with minor issues")
        elif score >= 60:
            print("   ⚠️  FAIR - Model works but needs improvement")
        else:
            print("   ❌ POOR - Model needs significant improvement")
        print()
        
        print("=" * 80)
    
    def save_results(self, results: Dict, output_path: Path):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Model")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory containing generated images")
    parser.add_argument("--sketch_dir", type=str, required=True,
                       help="Directory containing input sketches")
    parser.add_argument("--results_json", type=str, default="evaluation_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Stage1Evaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Collect files
    output_dir = Path(args.output_dir)
    sketch_dir = Path(args.sketch_dir)
    
    generated_paths = sorted(output_dir.glob("*.png"))
    sketch_paths = sorted(sketch_dir.glob("*.png"))
    
    # Create prompts (you'll need to customize this)
    prompts = [f"a {p.stem}" for p in generated_paths]
    
    # Run evaluation
    results = evaluator.evaluate_dataset(sketch_paths, generated_paths, prompts)
    
    # Print and save
    evaluator.print_results(results)
    evaluator.save_results(results, Path(args.results_json))


if __name__ == "__main__":
    main()
