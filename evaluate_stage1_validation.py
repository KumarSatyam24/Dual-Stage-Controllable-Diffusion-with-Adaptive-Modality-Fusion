"""
Stage 1 Validation Metrics Evaluation

Compares generated images with ground truth photos from the Sketchy dataset.
Computes standard image generation metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Frechet Inception Distance)
- IS (Inception Score)

Usage:
    python evaluate_stage1_validation.py --checkpoint /path/to/checkpoint.pt --num_samples 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import json
import argparse
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from datasets.sketchy_dataset import SketchyDataset
from configs.config import get_default_config

# Try to import LPIPS (perceptual loss)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS not available. Install with: pip install lpips")


class ValidationMetrics:
    """
    Computes validation metrics comparing generated images with ground truth.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize LPIPS if available
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None
        
        # Initialize Inception model for FID/IS
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        
        # Remove final classification layer for feature extraction
        self.inception_model.fc = nn.Identity()
    
    def preprocess_image(self, img):
        """Convert image to numpy array in [0, 255] range"""
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            if img.ndim == 4:
                img = img[0]  # Remove batch dimension
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = np.clip((img + 1) / 2 * 255, 0, 255).astype(np.uint8)
        elif isinstance(img, Image.Image):
            img = np.array(img)
        return img
    
    def compute_psnr(self, generated, ground_truth):
        """
        Compute PSNR (Peak Signal-to-Noise Ratio)
        Higher is better (typically 20-50 dB)
        """
        gen = self.preprocess_image(generated)
        gt = self.preprocess_image(ground_truth)
        
        # Convert to grayscale if needed for consistent comparison
        if gen.ndim == 3 and gen.shape[2] == 3:
            gen = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)
        if gt.ndim == 3 and gt.shape[2] == 3:
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        
        return psnr(gt, gen, data_range=255)
    
    def compute_ssim(self, generated, ground_truth):
        """
        Compute SSIM (Structural Similarity Index)
        Range: [-1, 1], higher is better (typically 0.5-0.95)
        """
        gen = self.preprocess_image(generated)
        gt = self.preprocess_image(ground_truth)
        
        # Convert to grayscale if needed
        if gen.ndim == 3 and gen.shape[2] == 3:
            gen = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)
        if gt.ndim == 3 and gt.shape[2] == 3:
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        
        return ssim(gt, gen, data_range=255)
    
    def compute_lpips(self, generated, ground_truth):
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity)
        Range: [0, 1], lower is better (typically 0.1-0.5)
        Measures perceptual similarity using deep features
        """
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return None
        
        # Convert to tensors in [-1, 1] range
        if not isinstance(generated, torch.Tensor):
            gen = torch.from_numpy(self.preprocess_image(generated)).float() / 127.5 - 1
            gen = gen.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
        else:
            gen = generated
        
        if not isinstance(ground_truth, torch.Tensor):
            gt = torch.from_numpy(self.preprocess_image(ground_truth)).float() / 127.5 - 1
            gt = gt.permute(2, 0, 1).unsqueeze(0)
        else:
            gt = ground_truth
        
        gen = gen.to(self.device)
        gt = gt.to(self.device)
        
        with torch.no_grad():
            distance = self.lpips_model(gen, gt)
        
        return distance.item()
    
    def compute_mse(self, generated, ground_truth):
        """
        Compute MSE (Mean Squared Error)
        Lower is better
        """
        gen = self.preprocess_image(generated).astype(np.float32)
        gt = self.preprocess_image(ground_truth).astype(np.float32)
        
        mse = np.mean((gen - gt) ** 2)
        return mse
    
    def compute_mae(self, generated, ground_truth):
        """
        Compute MAE (Mean Absolute Error)
        Lower is better
        """
        gen = self.preprocess_image(generated).astype(np.float32)
        gt = self.preprocess_image(ground_truth).astype(np.float32)
        
        mae = np.mean(np.abs(gen - gt))
        return mae
    
    @torch.no_grad()
    def get_inception_features(self, images):
        """
        Extract features from Inception network
        
        Args:
            images: List of images (torch tensors or PIL Images)
        
        Returns:
            Features array of shape (N, 2048)
        """
        features_list = []
        
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        for img in images:
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            elif not isinstance(img, torch.Tensor):
                img = torch.from_numpy(self.preprocess_image(img)).float() / 255.0
                img = img.permute(2, 0, 1)
            
            # Normalize
            img = transform(img.unsqueeze(0)).to(self.device)
            
            # Get features
            features = self.inception_model(img)
            features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def compute_fid(self, generated_images, real_images):
        """
        Compute FID (Frechet Inception Distance)
        Lower is better (typically 1-100)
        Measures quality and diversity of generated images
        """
        # Get inception features
        gen_features = self.get_inception_features(generated_images)
        real_features = self.get_inception_features(real_images)
        
        # Compute mean and covariance
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        # Compute FID
        diff = mu_gen - mu_real
        covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2 * covmean)
        
        return fid
    
    def compute_inception_score(self, images, splits=10):
        """
        Compute IS (Inception Score)
        Higher is better (typically 2-10)
        Measures quality and diversity
        """
        preds_list = []
        
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Get predictions
        for img in images:
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            elif not isinstance(img, torch.Tensor):
                img = torch.from_numpy(self.preprocess_image(img)).float() / 255.0
                img = img.permute(2, 0, 1)
            
            img = transform(img.unsqueeze(0)).to(self.device)
            
            with torch.no_grad():
                pred = self.inception_model(img)
                pred = F.softmax(pred, dim=1)
            
            preds_list.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds_list, axis=0)
        
        # Compute IS
        split_scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl = np.mean(np.sum(kl, axis=1))
            split_scores.append(np.exp(kl))
        
        return np.mean(split_scores), np.std(split_scores)


def evaluate_model(
    checkpoint_path,
    dataset,
    pipeline,
    metrics,
    config,
    num_samples=100,
    output_dir='validation_results'
):
    """
    Evaluate model on validation set
    
    Args:
        checkpoint_path: Path to model checkpoint
        dataset: Sketchy dataset (test split)
        pipeline: Inference pipeline
        metrics: ValidationMetrics instance
        config: Configuration dict
        num_samples: Number of samples to evaluate
        output_dir: Directory to save results
    
    Returns:
        Dictionary of metric values
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Storage for all metrics
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    mse_scores = []
    mae_scores = []
    
    generated_images = []
    real_images = []
    
    print(f"\n🎨 Generating and evaluating {num_samples} images...")
    
    # Sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        try:
            # Get data from dataset
            data = dataset[idx]
            sketch = data['sketch'].unsqueeze(0)  # Add batch dimension
            ground_truth = data['photo']
            prompt = data['text_prompt']
            category = data['category']
            
            # Use the configured image size from dataset
            img_size = config['data'].image_size
            
            # Generate image with proper dimensions
            with torch.no_grad():
                generated = pipeline.generate(
                    sketch=sketch,
                    text_prompt=prompt,
                    height=img_size,
                    width=img_size,
                    seed=42 + i  # Reproducible
                )
            
            # Convert to numpy/PIL for metrics
            gen_img = generated[0].cpu()  # Remove batch dimension
            gt_img = ground_truth
            
            # Compute per-image metrics
            psnr_val = metrics.compute_psnr(gen_img, gt_img)
            ssim_val = metrics.compute_ssim(gen_img, gt_img)
            mse_val = metrics.compute_mse(gen_img, gt_img)
            mae_val = metrics.compute_mae(gen_img, gt_img)
            
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            mse_scores.append(mse_val)
            mae_scores.append(mae_val)
            
            if LPIPS_AVAILABLE:
                lpips_val = metrics.compute_lpips(gen_img, gt_img)
                lpips_scores.append(lpips_val)
            
            # Store for FID/IS computation
            generated_images.append(gen_img)
            real_images.append(gt_img)
            
            # Save comparison every 10 images
            if i % 10 == 0:
                save_comparison(
                    sketch[0], gen_img, gt_img, prompt, category,
                    output_dir / f"comparison_{i:04d}.png",
                    psnr_val, ssim_val
                )
        
        except Exception as e:
            print(f"\n⚠️  Error processing sample {i}: {e}")
            continue
    
    print("\n📊 Computing aggregate metrics...")
    
    # Check if we have any successful generations
    if len(generated_images) == 0:
        print("\n❌ No images were successfully generated!")
        print("   Please check the errors above and try again.")
        return None
    
    print(f"   Successfully generated {len(generated_images)} images")
    
    # Compute FID and IS (only if we have enough samples)
    if len(generated_images) >= 2:
        fid_score = metrics.compute_fid(generated_images, real_images)
        is_mean, is_std = metrics.compute_inception_score(generated_images)
    else:
        print("   ⚠️  Too few samples for FID/IS, skipping...")
        fid_score = None
        is_mean = None
        is_std = None
    
    # Compile results
    results = {
        'num_samples': len(psnr_scores),
        'psnr': {
            'mean': float(np.mean(psnr_scores)),
            'std': float(np.std(psnr_scores)),
            'min': float(np.min(psnr_scores)),
            'max': float(np.max(psnr_scores))
        },
        'ssim': {
            'mean': float(np.mean(ssim_scores)),
            'std': float(np.std(ssim_scores)),
            'min': float(np.min(ssim_scores)),
            'max': float(np.max(ssim_scores))
        },
        'mse': {
            'mean': float(np.mean(mse_scores)),
            'std': float(np.std(mse_scores)),
            'min': float(np.min(mse_scores)),
            'max': float(np.max(mse_scores))
        },
        'mae': {
            'mean': float(np.mean(mae_scores)),
            'std': float(np.std(mae_scores)),
            'min': float(np.min(mae_scores)),
            'max': float(np.max(mae_scores))
        }
    }
    
    # Add FID and IS if computed
    if fid_score is not None:
        results['fid'] = float(fid_score)
    
    if is_mean is not None:
        results['inception_score'] = {
            'mean': float(is_mean),
            'std': float(is_std)
        }
    
    if LPIPS_AVAILABLE and lpips_scores:
        results['lpips'] = {
            'mean': float(np.mean(lpips_scores)),
            'std': float(np.std(lpips_scores)),
            'min': float(np.min(lpips_scores)),
            'max': float(np.max(lpips_scores))
        }
    
    return results


def save_comparison(sketch, generated, ground_truth, prompt, category, save_path, psnr_val, ssim_val):
    """Save comparison of sketch, generated, and ground truth"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sketch
    if isinstance(sketch, torch.Tensor):
        sketch_np = sketch.cpu().numpy()
        if sketch_np.ndim == 3:
            sketch_np = sketch_np[0]
    axes[0].imshow(sketch_np, cmap='gray')
    axes[0].set_title('Input Sketch')
    axes[0].axis('off')
    
    # Generated
    if isinstance(generated, torch.Tensor):
        gen_np = generated.cpu().permute(1, 2, 0).numpy()
        gen_np = np.clip((gen_np + 1) / 2, 0, 1)
    else:
        gen_np = np.array(generated) / 255.0
    axes[1].imshow(gen_np)
    axes[1].set_title(f'Generated\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}')
    axes[1].axis('off')
    
    # Ground truth
    if isinstance(ground_truth, torch.Tensor):
        gt_np = ground_truth.cpu().permute(1, 2, 0).numpy()
        gt_np = np.clip((gt_np + 1) / 2, 0, 1)
    else:
        gt_np = np.array(ground_truth) / 255.0
    axes[2].imshow(gt_np)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.suptitle(f'{category}: {prompt}', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_results(results):
    """Pretty print evaluation results"""
    print("\n" + "="*80)
    print("📊 VALIDATION METRICS RESULTS")
    print("="*80)
    print(f"\n📝 Evaluated {results['num_samples']} samples\n")
    
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ PIXEL-LEVEL METRICS                                                  │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ PSNR (Peak Signal-to-Noise Ratio)                                   │")
    print(f"│   Mean: {results['psnr']['mean']:7.3f} dB  (Higher is better)      │")
    print(f"│   Std:  {results['psnr']['std']:7.3f} dB                             │")
    print(f"│   Range: [{results['psnr']['min']:.2f}, {results['psnr']['max']:.2f}] dB              │")
    print(f"│                                                                      │")
    print(f"│ SSIM (Structural Similarity Index)                                   │")
    print(f"│   Mean: {results['ssim']['mean']:7.4f}     (Range: -1 to 1, higher is better) │")
    print(f"│   Std:  {results['ssim']['std']:7.4f}                                │")
    print(f"│   Range: [{results['ssim']['min']:.3f}, {results['ssim']['max']:.3f}]                 │")
    print(f"│                                                                      │")
    print(f"│ MSE (Mean Squared Error)                                             │")
    print(f"│   Mean: {results['mse']['mean']:10.2f}  (Lower is better)           │")
    print(f"│   Std:  {results['mse']['std']:10.2f}                                │")
    print(f"│                                                                      │")
    print(f"│ MAE (Mean Absolute Error)                                            │")
    print(f"│   Mean: {results['mae']['mean']:10.2f}  (Lower is better)           │")
    print(f"│   Std:  {results['mae']['std']:10.2f}                                │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    if 'lpips' in results:
        print("\n┌─────────────────────────────────────────────────────────────────────┐")
        print("│ PERCEPTUAL METRICS                                                   │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        print(f"│ LPIPS (Learned Perceptual Similarity)                               │")
        print(f"│   Mean: {results['lpips']['mean']:7.4f}     (Lower is better)       │")
        print(f"│   Std:  {results['lpips']['std']:7.4f}                               │")
        print(f"│   Range: [{results['lpips']['min']:.3f}, {results['lpips']['max']:.3f}]           │")
        print("└─────────────────────────────────────────────────────────────────────┘")
    
    if 'fid' in results or 'inception_score' in results:
        print("\n┌─────────────────────────────────────────────────────────────────────┐")
        print("│ DISTRIBUTION METRICS                                                 │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        
        if 'fid' in results:
            print(f"│ FID (Frechet Inception Distance)                                     │")
            print(f"│   Score: {results['fid']:8.3f}    (Lower is better, <50 is good)   │")
            print(f"│                                                                      │")
        
        if 'inception_score' in results:
            print(f"│ IS (Inception Score)                                                 │")
            print(f"│   Mean: {results['inception_score']['mean']:7.3f}     (Higher is better) │")
            print(f"│   Std:  {results['inception_score']['std']:7.3f}                     │")
        
        print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Stage 1 model with validation metrics')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--guidance_scale', type=float, default=2.5, help='Guidance scale for generation')
    parser.add_argument('--output_dir', type=str, default='validation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 STAGE 1 VALIDATION METRICS EVALUATION")
    print("="*80)
    print(f"\nDevice: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Guidance scale: {args.guidance_scale}")
    
    # Load config
    config = get_default_config()
    
    # Load dataset
    print("\n📁 Loading Sketchy dataset (test split)...")
    dataset = SketchyDataset(
        root_dir=config['data'].sketchy_root,
        split='test',
        image_size=config['data'].image_size,
        augment=False
    )
    print(f"   Total test samples: {len(dataset)}")
    
    # Create model and pipeline
    print(f"\n🔧 Loading model from: {args.checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create model with proper parameters
    model_config = config['model']
    model = Stage1SketchGuidedDiffusion(
        pretrained_model_name=model_config.pretrained_model_name,
        sketch_encoder_channels=model_config.sketch_encoder_channels,
        freeze_base_unet=model_config.freeze_stage1_unet,
        use_lora=model_config.use_lora,
        lora_rank=model_config.lora_rank
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    pipeline = Stage1DiffusionPipeline(
        model=model,
        num_inference_steps=config['inference'].num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=device
    )
    print("   ✅ Model loaded successfully")
    
    # Initialize metrics
    print("\n🔧 Initializing metrics...")
    metrics = ValidationMetrics(device=device)
    print("   ✅ Metrics initialized")
    
    # Run evaluation
    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset=dataset,
        pipeline=pipeline,
        metrics=metrics,
        config=config,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    # Print results (only if we got results)
    if results is not None:
        print_results(results)
        
        # Save results to JSON
        output_file = Path(args.output_dir) / 'validation_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"📁 Comparison images saved to: {args.output_dir}/")
    else:
        print("\n❌ Evaluation failed - no results to save")
        sys.exit(1)


if __name__ == '__main__':
    main()
