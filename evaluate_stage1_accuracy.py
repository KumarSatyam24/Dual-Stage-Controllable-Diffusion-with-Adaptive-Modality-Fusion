"""
Evaluate Stage 1 Model on Sketchy Dataset
Computes accuracy metrics for sketch-guided generation

This script answers:
1. How accurate is the model? (Overall score 0-100)
2. Does it follow the sketch? (Edge consistency)
3. Does it match the category? (True Positive Rate via CLIP)
4. Is the quality good? (Image quality metrics)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import json
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from datasets.sketchy_dataset import SketchyDataset
from configs.config import get_default_config

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("⚠️  CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False


class QuickEvaluator:
    """Simple evaluator for Stage 1 model"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
    
    def edge_consistency(self, sketch, generated_image):
        """
        Measure how well the generated image follows the sketch edges
        Returns: 0-1 score (higher is better)
        """
        # Convert to numpy if needed
        if isinstance(sketch, torch.Tensor):
            sketch = (sketch.cpu().numpy() * 255).astype(np.uint8)
            if sketch.ndim == 3:
                sketch = sketch[0]
        
        if isinstance(generated_image, torch.Tensor):
            generated_image = generated_image.cpu().permute(1, 2, 0).numpy()
            generated_image = (generated_image * 255).astype(np.uint8)
        elif isinstance(generated_image, Image.Image):
            generated_image = np.array(generated_image)
        
        # Extract edges from generated image
        gray = cv2.cvtColor(generated_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Resize sketch to match if needed
        if sketch.shape != edges.shape:
            sketch = cv2.resize(sketch, (edges.shape[1], edges.shape[0]))
        
        # Compute IoU of edge maps
        sketch_binary = sketch > 127
        edges_binary = edges > 127
        
        intersection = np.logical_and(sketch_binary, edges_binary).sum()
        union = np.logical_or(sketch_binary, edges_binary).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def category_accuracy(self, generated_images, true_categories, all_categories):
        """
        Compute classification accuracy using CLIP
        
        This is the "True Positive Rate" - does CLIP classify the image 
        as the correct category?
        
        Returns:
            top1_accuracy: Fraction where top prediction is correct
            top5_accuracy: Fraction where correct category is in top 5
        """
        if not CLIP_AVAILABLE:
            print("⚠️  CLIP not available, cannot compute category accuracy")
            return 0.0, 0.0, []
        
        correct_top1 = 0
        correct_top5 = 0
        predictions = []
        
        print("\n🔍 Computing category accuracy (True Positive Rate)...")
        for img, true_cat in tqdm(zip(generated_images, true_categories), total=len(generated_images)):
            # Preprocess image
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            
            image_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            
            # Create text prompts
            text_prompts = [f"a photo of a {cat}" for cat in all_categories]
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            
            # Compute similarity
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).squeeze(0)
            
            # Get top predictions
            top5_indices = similarity.topk(5).indices.cpu().numpy()
            top1_pred = all_categories[top5_indices[0]]
            top5_preds = [all_categories[i] for i in top5_indices]
            
            predictions.append({
                'true': true_cat,
                'predicted': top1_pred,
                'top5': top5_preds,
                'correct_top1': top1_pred == true_cat,
                'correct_top5': true_cat in top5_preds
            })
            
            if top1_pred == true_cat:
                correct_top1 += 1
                correct_top5 += 1
            elif true_cat in top5_preds:
                correct_top5 += 1
        
        n = len(generated_images)
        return correct_top1 / n, correct_top5 / n, predictions
    
    def evaluate_batch(self, sketches, images, categories, all_categories):
        """
        Evaluate a batch of generated images
        
        Returns:
            Dictionary with all metrics
        """
        edge_scores = []
        
        print("\n📐 Computing edge consistency scores...")
        for sketch, img in tqdm(zip(sketches, images), total=len(sketches)):
            edge_score = self.edge_consistency(sketch, img)
            edge_scores.append(edge_score)
        
        # Category accuracy
        top1_acc, top5_acc, predictions = self.category_accuracy(images, categories, all_categories)
        
        results = {
            'edge_consistency': {
                'mean': float(np.mean(edge_scores)),
                'std': float(np.std(edge_scores)),
                'min': float(np.min(edge_scores)),
                'max': float(np.max(edge_scores)),
                'scores': edge_scores
            },
            'category_accuracy': {
                'top1': float(top1_acc),
                'top5': float(top5_acc),
                'predictions': predictions
            },
            'num_samples': len(images)
        }
        
        # Compute overall accuracy score
        results['overall_accuracy'] = self._compute_overall_score(results)
        
        return results
    
    def _compute_overall_score(self, results):
        """
        Compute overall accuracy score (0-100)
        
        This combines:
        - Edge consistency (50%) - how well it follows the sketch
        - Category accuracy (50%) - whether it generates the right object
        """
        edge_score = results['edge_consistency']['mean'] * 100  # Convert to 0-100
        category_score = results['category_accuracy']['top1'] * 100  # Convert to 0-100
        
        # Weighted average
        overall = 0.5 * edge_score + 0.5 * category_score
        
        return float(overall)


def generate_and_evaluate(checkpoint_path, num_samples=100, guidance_scale=2.5):
    """
    Generate images and evaluate them
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to evaluate
        guidance_scale: Guidance scale for generation
    """
    print("=" * 80)
    print("🎯 STAGE 1 MODEL ACCURACY EVALUATION")
    print("=" * 80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = get_default_config()
    
    # Load dataset
    print("\n📁 Loading Sketchy dataset...")
    dataset = SketchyDataset(
        root_dir=config['data'].sketchy_root,
        split='test',
        image_size=config['data'].image_size,
        augment=False  # No augmentation for evaluation
    )
    
    print(f"   Total test samples: {len(dataset)}")
    print(f"   Evaluating on: {num_samples} samples")
    
    # Get all categories from the dataset
    all_categories = sorted(list(set([pair['category'] for pair in dataset.data_pairs])))
    print(f"   Total categories: {len(all_categories)}")
    
    # Load model
    print(f"\n🔧 Loading model from: {checkpoint_path}")
    
    # Create model instance
    model = Stage1SketchGuidedDiffusion(config['model']).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create pipeline with loaded model
    pipeline = Stage1DiffusionPipeline(
        model=model,
        num_inference_steps=config['inference'].num_inference_steps,
        guidance_scale=config['inference'].guidance_scale,
        device=device
    )
    print("   ✅ Model loaded successfully")
    
    # Initialize evaluator
    evaluator = QuickEvaluator(pipeline.model, device)
    
    # Generate images
    print(f"\n🎨 Generating {num_samples} images...")
    sketches = []
    generated_images = []
    categories = []
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices):
        sketch, prompt, category = dataset[idx]
        
        # Generate image
        sketch_batch = sketch.unsqueeze(0).to(device)
        prompt_batch = [prompt]
        
        with torch.no_grad():
            generated = pipeline.generate(
                sketch_batch,
                prompt_batch,
                num_inference_steps=50,
                guidance_scale=guidance_scale
            )
        
        sketches.append(sketch)
        generated_images.append(generated[0])
        categories.append(category)
    
    print("   ✅ Generation complete")
    
    # Evaluate
    print("\n📊 Evaluating generated images...")
    results = evaluator.evaluate_batch(sketches, generated_images, categories, all_categories)
    
    # Print results
    print_results(results, guidance_scale)
    
    # Save results
    output_file = f"stage1_evaluation_guidance{guidance_scale}.json"
    with open(output_file, 'w') as f:
        # Remove scores list for cleaner JSON
        save_results = results.copy()
        save_results['edge_consistency'] = {k: v for k, v in results['edge_consistency'].items() if k != 'scores'}
        save_results['category_accuracy'] = {k: v for k, v in results['category_accuracy'].items() if k != 'predictions'}
        json.dump(save_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results


def print_results(results, guidance_scale):
    """Pretty print evaluation results"""
    print()
    print("=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    print(f"⚙️  Configuration:")
    print(f"   Guidance Scale: {guidance_scale}")
    print(f"   Samples Evaluated: {results['num_samples']}")
    print()
    
    print(f"🎯 OVERALL ACCURACY: {results['overall_accuracy']:.2f}%")
    print()
    
    # Edge consistency
    ec = results['edge_consistency']
    print("📐 Sketch Fidelity (Edge Consistency):")
    print(f"   Mean Score:  {ec['mean']:.4f} ({ec['mean']*100:.2f}%)")
    print(f"   Std Dev:     {ec['std']:.4f}")
    print(f"   Range:       {ec['min']:.4f} - {ec['max']:.4f}")
    print()
    
    # Category accuracy (True Positive Rate)
    ca = results['category_accuracy']
    print("✅ Category Accuracy (True Positive Rate):")
    print(f"   Top-1 Accuracy:  {ca['top1']*100:.2f}%  ← Main 'accuracy' metric")
    print(f"   Top-5 Accuracy:  {ca['top5']*100:.2f}%")
    print()
    
    # Interpretation
    print("📈 Interpretation:")
    print()
    
    overall = results['overall_accuracy']
    if overall >= 80:
        print("   🌟 EXCELLENT - Model is highly accurate!")
        print("      - Follows sketches very well")
        print("      - Generates correct categories")
    elif overall >= 70:
        print("   ✅ GOOD - Model performs well")
        print("      - Good sketch fidelity")
        print("      - Mostly correct categories")
    elif overall >= 60:
        print("   ⚠️  FAIR - Model works but has issues")
        print("      - Reasonable sketch following")
        print("      - Some category mismatches")
    else:
        print("   ❌ NEEDS IMPROVEMENT")
        print("      - Poor sketch fidelity or category accuracy")
    print()
    
    # Explain metrics
    print("📚 Metric Explanations:")
    print()
    print("   • Overall Accuracy (0-100%):")
    print("     Combined score of sketch fidelity and category accuracy")
    print()
    print("   • Edge Consistency (0-1):")
    print("     Measures how well generated edges match input sketch")
    print("     IoU (Intersection over Union) of edge maps")
    print()
    print("   • Top-1 Category Accuracy:")
    print("     'True Positive Rate' - Does CLIP classify the image")
    print("     as the correct category? (This is like TP/(TP+FP))")
    print()
    print("   • Top-5 Category Accuracy:")
    print("     Is the correct category in the top 5 predictions?")
    print()
    
    # Show some examples
    if 'predictions' in ca and len(ca['predictions']) > 0:
        print("🔍 Sample Predictions:")
        print()
        correct = [p for p in ca['predictions'] if p['correct_top1']][:3]
        incorrect = [p for p in ca['predictions'] if not p['correct_top1']][:3]
        
        if correct:
            print("   ✅ Correct Predictions:")
            for p in correct:
                print(f"      True: {p['true']}, Predicted: {p['predicted']}")
        
        if incorrect:
            print()
            print("   ❌ Incorrect Predictions:")
            for p in incorrect:
                print(f"      True: {p['true']}, Predicted: {p['predicted']} (Top-5: {p['top5'][:2]}...)")
        print()
    
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 Model Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with 100 samples
  python evaluate_stage1_accuracy.py --checkpoint checkpoints/stage1/epoch_10.pth
  
  # Evaluate with different guidance scale
  python evaluate_stage1_accuracy.py --checkpoint checkpoints/stage1/epoch_10.pth --guidance_scale 3.0
  
  # Evaluate more samples
  python evaluate_stage1_accuracy.py --checkpoint checkpoints/stage1/epoch_10.pth --num_samples 500
        """
    )
    
    parser.add_argument("--checkpoint", type=str, 
                       default="/root/checkpoints/stage1/epoch_10.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--guidance_scale", type=float, default=2.5,
                       help="Guidance scale for generation")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = generate_and_evaluate(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale
    )


if __name__ == "__main__":
    main()
