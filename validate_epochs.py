"""
Stage 1 Validation Pipeline - Epoch-by-Epoch Evaluation

This script evaluates Stage-1 sketch-guided diffusion model across training epochs.
Downloads checkpoints from HuggingFace, generates images from validation sketches,
computes metrics (FID, CLIP, LPIPS, SSIM, PSNR), and creates visualizations.

Author: Dual-Stage Controllable Diffusion Research Team
Date: March 2026
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Computer vision metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

# Deep learning metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS not available. Install: pip install lpips")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not available. Install: pip install git+https://github.com/openai/CLIP.git")

# HuggingFace
from huggingface_hub import hf_hub_download, list_repo_files

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.stage1_diffusion import Stage1SketchGuidedDiffusion, Stage1DiffusionPipeline
from datasets.sketchy_dataset import SketchyDataset
from configs.config import get_default_config


class EpochValidator:
    """
    Validates Stage-1 model across multiple training epochs.
    Downloads checkpoints from HuggingFace and evaluates each.
    """
    
    def __init__(
        self,
        hf_repo_id: str,
        dataset_root: str,
        output_dir: str = "validation_results",
        device: str = "cuda",
        guidance_scale: float = 2.5,
        num_inference_steps: int = 50
    ):
        """
        Initialize validator.
        
        Args:
            hf_repo_id: HuggingFace repository ID (e.g., "username/model-name")
            dataset_root: Path to Sketchy dataset
            output_dir: Output directory for results
            device: Device to use
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of diffusion steps
        """
        self.hf_repo_id = hf_repo_id
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics
        self._init_metrics()
        
        # Load config
        self.config = get_default_config()
        
        # Load validation dataset
        print(f"\n📁 Loading validation dataset from: {dataset_root}")
        self.val_dataset = SketchyDataset(
            root_dir=dataset_root,
            split='test',  # Use test split for validation
            image_size=self.config['data'].image_size,
            augment=False
        )
        print(f"   Validation samples: {len(self.val_dataset)}")
        
        # Storage for metrics across epochs
        self.epoch_metrics = {}
    
    def _init_metrics(self):
        """Initialize metric computation models"""
        print("\n🔧 Initializing metrics...")
        
        # LPIPS for perceptual similarity
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            print("   ✅ LPIPS initialized")
        else:
            self.lpips_model = None
        
        # CLIP for semantic similarity
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            print("   ✅ CLIP initialized")
        else:
            self.clip_model = None
        
        # Inception for FID
        from torchvision.models import inception_v3
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        self.inception_model.fc = nn.Identity()  # Remove classification layer
        print("   ✅ Inception initialized")
    
    def get_available_epochs(self) -> List[int]:
        """
        Get list of available epoch checkpoints from HuggingFace.
        
        Returns:
            List of epoch numbers
        """
        print(f"\n📦 Checking HuggingFace repo: {self.hf_repo_id}")
        try:
            files = list_repo_files(self.hf_repo_id)
            
            # Find checkpoint files in stage1 subfolder (e.g., stage1/epoch_1.pt, stage1/epoch_2.pt, etc.)
            epochs = []
            for file in files:
                # Check for stage1/epoch_X.pt pattern
                if file.startswith("stage1/epoch_") and file.endswith(".pt"):
                    try:
                        epoch_num = int(file.replace("stage1/epoch_", "").replace(".pt", ""))
                        epochs.append(epoch_num)
                    except ValueError:
                        continue
                # Also check for epoch_X.pt at root (fallback)
                elif file.startswith("epoch_") and file.endswith(".pt"):
                    try:
                        epoch_num = int(file.replace("epoch_", "").replace(".pt", ""))
                        epochs.append(epoch_num)
                    except ValueError:
                        continue
            
            # Also check for final.pt in stage1 subfolder or root
            if "stage1/final.pt" in files or "final.pt" in files:
                epochs.append("final")
            
            epochs = sorted([e for e in epochs if isinstance(e, int)]) + \
                     ([epochs[-1]] if "final" in epochs else [])
            
            print(f"   Found {len(epochs)} checkpoints: {epochs}")
            return epochs
            
        except Exception as e:
            print(f"   ❌ Error accessing HuggingFace repo: {e}")
            print(f"   💡 Using local checkpoints instead")
            return self._get_local_epochs()
    
    def _get_local_epochs(self) -> List[int]:
        """Get epochs from local checkpoint directory"""
        local_dir = Path("/root/checkpoints/stage1")
        if not local_dir.exists():
            return []
        
        epochs = []
        for file in local_dir.glob("epoch_*.pt"):
            try:
                epoch_num = int(file.stem.replace("epoch_", ""))
                epochs.append(epoch_num)
            except ValueError:
                continue
        
        if (local_dir / "final.pt").exists():
            # Get the max epoch number for final
            epochs.append(max(epochs) if epochs else 10)
        
        return sorted(epochs)
    
    def download_checkpoint(self, epoch: int) -> str:
        """
        Download checkpoint for specific epoch from HuggingFace.
        
        Args:
            epoch: Epoch number or "final"
            
        Returns:
            Path to downloaded checkpoint
        """
        filename = f"epoch_{epoch}.pt" if isinstance(epoch, int) else "final.pt"
        
        # Check local first
        local_path = Path(f"/root/checkpoints/stage1/{filename}")
        if local_path.exists():
            print(f"   Using local checkpoint: {local_path}")
            return str(local_path)
        
        # Download from HuggingFace (try stage1 subfolder first, then root)
        try:
            print(f"   Downloading {filename} from HuggingFace...")
            
            # Try stage1 subfolder first
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=f"stage1/{filename}",
                    cache_dir="/root/.cache/huggingface"
                )
                print(f"   ✅ Downloaded from stage1/ to: {checkpoint_path}")
                return checkpoint_path
            except:
                # Fallback to root directory
                checkpoint_path = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=filename,
                    cache_dir="/root/.cache/huggingface"
                )
                print(f"   ✅ Downloaded from root to: {checkpoint_path}")
                return checkpoint_path
                
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            return None
    
    def load_model(self, checkpoint_path: str) -> Tuple[Stage1DiffusionPipeline, dict]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (pipeline, checkpoint_info)
        """
        print(f"\n🔧 Loading model from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model_config = self.config['model']
        model = Stage1SketchGuidedDiffusion(
            pretrained_model_name=model_config.pretrained_model_name,
            sketch_encoder_channels=model_config.sketch_encoder_channels,
            freeze_base_unet=model_config.freeze_stage1_unet,
            use_lora=model_config.use_lora,
            lora_rank=model_config.lora_rank
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create pipeline
        pipeline = Stage1DiffusionPipeline(
            model=model,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            device=self.device
        )
        
        # Extract checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'global_step': checkpoint.get('global_step', 'unknown'),
            'train_loss': checkpoint.get('train_loss', None)
        }
        
        print(f"   ✅ Model loaded (Epoch: {checkpoint_info['epoch']}, "
              f"Step: {checkpoint_info['global_step']})")
        
        return pipeline, checkpoint_info
    
    def validate_epoch(
        self,
        epoch: int,
        pipeline: Stage1DiffusionPipeline,
        num_samples: int = 50,
        save_images: bool = True
    ) -> Dict:
        """
        Validate model for one epoch.
        
        Args:
            epoch: Epoch number
            pipeline: Inference pipeline
            num_samples: Number of validation samples
            save_images: Whether to save comparison images
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n🎨 Validating Epoch {epoch} ({num_samples} samples)...")
        
        # Create epoch output directory
        epoch_dir = self.output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for metrics
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        clip_scores = []
        generated_images = []
        real_images = []
        
        # Sample indices
        indices = np.random.RandomState(42).choice(
            len(self.val_dataset), 
            min(num_samples, len(self.val_dataset)), 
            replace=False
        )
        
        # Generate and evaluate
        for i, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch}")):
            try:
                # Get data
                data = self.val_dataset[idx]
                sketch = data['sketch'].unsqueeze(0)
                ground_truth = data['photo']
                prompt = data['text_prompt']
                category = data['category']
                
                # Generate
                with torch.no_grad():
                    generated = pipeline.generate(
                        sketch=sketch,
                        text_prompt=prompt,
                        height=self.config['data'].image_size,
                        width=self.config['data'].image_size,
                        seed=42 + i
                    )
                
                gen_img = generated[0].cpu()
                gt_img = ground_truth
                
                # Compute metrics
                psnr_val = self._compute_psnr(gen_img, gt_img)
                ssim_val = self._compute_ssim(gen_img, gt_img)
                
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
                
                if self.lpips_model is not None:
                    lpips_val = self._compute_lpips(gen_img, gt_img)
                    lpips_scores.append(lpips_val)
                
                if self.clip_model is not None:
                    clip_val = self._compute_clip_similarity(gen_img, prompt)
                    clip_scores.append(clip_val)
                
                # Store for FID
                generated_images.append(gen_img)
                real_images.append(gt_img)
                
                # Save comparison image
                if save_images and i % 5 == 0:  # Save every 5th image
                    self._save_comparison(
                        sketch[0], gen_img, gt_img, prompt, category,
                        epoch_dir / f"sample_{i:04d}.png",
                        psnr_val, ssim_val
                    )
                
            except Exception as e:
                print(f"   ⚠️  Error on sample {i}: {e}")
                continue
        
        # Compute aggregate metrics
        print(f"\n   Computing aggregate metrics...")
        metrics = {
            'epoch': epoch,
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
            }
        }
        
        if lpips_scores:
            metrics['lpips'] = {
                'mean': float(np.mean(lpips_scores)),
                'std': float(np.std(lpips_scores))
            }
        
        if clip_scores:
            metrics['clip_similarity'] = {
                'mean': float(np.mean(clip_scores)),
                'std': float(np.std(clip_scores))
            }
        
        # Compute FID if enough samples
        if len(generated_images) >= 10:
            fid_score = self._compute_fid(generated_images, real_images)
            metrics['fid'] = float(fid_score)
        
        # Save metrics
        with open(epoch_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _compute_psnr(self, generated, ground_truth):
        """Compute PSNR"""
        gen = self._to_numpy(generated)
        gt = self._to_numpy(ground_truth)
        return psnr(gt, gen, data_range=255)
    
    def _compute_ssim(self, generated, ground_truth):
        """Compute SSIM"""
        gen = self._to_numpy(generated)
        gt = self._to_numpy(ground_truth)
        
        # Convert to grayscale
        if gen.ndim == 3 and gen.shape[2] == 3:
            gen = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)
        if gt.ndim == 3 and gt.shape[2] == 3:
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        
        return ssim(gt, gen, data_range=255)
    
    def _compute_lpips(self, generated, ground_truth):
        """Compute LPIPS"""
        gen = self._to_tensor(generated).unsqueeze(0).to(self.device)
        gt = self._to_tensor(ground_truth).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            distance = self.lpips_model(gen, gt)
        
        return distance.item()
    
    def _compute_clip_similarity(self, generated, text_prompt):
        """Compute CLIP similarity between generated image and text prompt"""
        # Preprocess image
        gen_pil = self._to_pil(generated)
        image_input = self.clip_preprocess(gen_pil).unsqueeze(0).to(self.device)
        
        # Tokenize text
        text_input = clip.tokenize([text_prompt]).to(self.device)
        
        # Compute features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    def _compute_fid(self, generated_images, real_images):
        """Compute FID score"""
        from scipy import linalg
        
        gen_features = self._get_inception_features(generated_images)
        real_features = self._get_inception_features(real_images)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        diff = mu_gen - mu_real
        covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2 * covmean)
        return fid
    
    @torch.no_grad()
    def _get_inception_features(self, images):
        """Extract Inception features"""
        features_list = []
        
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        for img in images:
            img_tensor = self._to_tensor(img).unsqueeze(0).to(self.device)
            img_tensor = transform(img_tensor)
            features = self.inception_model(img_tensor)
            features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def _to_numpy(self, img):
        """Convert to numpy [0, 255]"""
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            img = np.clip((img + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return img
    
    def _to_tensor(self, img):
        """Convert to tensor [-1, 1]"""
        if isinstance(img, torch.Tensor):
            return img
        img_np = self._to_numpy(img)
        img_tensor = torch.from_numpy(img_np).float() / 127.5 - 1
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor
    
    def _to_pil(self, img):
        """Convert to PIL Image"""
        img_np = self._to_numpy(img)
        return Image.fromarray(img_np)
    
    def _save_comparison(self, sketch, generated, ground_truth, prompt, category, 
                        save_path, psnr_val, ssim_val):
        """Save comparison image"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sketch
        if isinstance(sketch, torch.Tensor):
            sketch_np = sketch.cpu().numpy()
            if sketch_np.ndim == 3:
                sketch_np = sketch_np[0]
        axes[0].imshow(sketch_np, cmap='gray')
        axes[0].set_title('Input Sketch', fontsize=12)
        axes[0].axis('off')
        
        # Generated
        gen_np = self._to_numpy(generated)
        axes[1].imshow(gen_np)
        axes[1].set_title(f'Generated\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.3f}', 
                         fontsize=11)
        axes[1].axis('off')
        
        # Ground truth
        gt_np = self._to_numpy(ground_truth)
        axes[2].imshow(gt_np)
        axes[2].set_title('Ground Truth', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f'{category}: {prompt}', fontsize=10, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_across_epochs(self):
        """
        Create plots showing metrics evolution across epochs.
        """
        print("\n📊 Creating metric plots...")
        
        if not self.epoch_metrics:
            print("   ⚠️  No epoch metrics available")
            return
        
        # Extract data
        epochs = sorted(self.epoch_metrics.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Stage-1 Validation Metrics Across Epochs', fontsize=16, y=0.995)
        
        # PSNR
        psnr_means = [self.epoch_metrics[e]['psnr']['mean'] for e in epochs]
        psnr_stds = [self.epoch_metrics[e]['psnr']['std'] for e in epochs]
        axes[0, 0].errorbar(epochs, psnr_means, yerr=psnr_stds, marker='o', capsize=5)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('PSNR (Higher is Better)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM
        ssim_means = [self.epoch_metrics[e]['ssim']['mean'] for e in epochs]
        ssim_stds = [self.epoch_metrics[e]['ssim']['std'] for e in epochs]
        axes[0, 1].errorbar(epochs, ssim_means, yerr=ssim_stds, marker='o', capsize=5, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('SSIM (Higher is Better)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # LPIPS
        if 'lpips' in self.epoch_metrics[epochs[0]]:
            lpips_means = [self.epoch_metrics[e]['lpips']['mean'] for e in epochs]
            lpips_stds = [self.epoch_metrics[e]['lpips']['std'] for e in epochs]
            axes[0, 2].errorbar(epochs, lpips_means, yerr=lpips_stds, marker='o', capsize=5, color='red')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('LPIPS')
            axes[0, 2].set_title('LPIPS (Lower is Better)')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'LPIPS not available', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
        
        # FID
        if 'fid' in self.epoch_metrics[epochs[0]]:
            fid_scores = [self.epoch_metrics[e]['fid'] for e in epochs]
            axes[1, 0].plot(epochs, fid_scores, marker='o', color='purple')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('FID Score')
            axes[1, 0].set_title('FID (Lower is Better)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'FID not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # CLIP Similarity
        if 'clip_similarity' in self.epoch_metrics[epochs[0]]:
            clip_means = [self.epoch_metrics[e]['clip_similarity']['mean'] for e in epochs]
            clip_stds = [self.epoch_metrics[e]['clip_similarity']['std'] for e in epochs]
            axes[1, 1].errorbar(epochs, clip_means, yerr=clip_stds, marker='o', capsize=5, color='orange')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('CLIP Similarity')
            axes[1, 1].set_title('CLIP Text-Image Similarity (Higher is Better)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'CLIP not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        # Summary table
        axes[1, 2].axis('off')
        summary_text = "Best Performance:\n\n"
        
        best_psnr_epoch = epochs[np.argmax(psnr_means)]
        summary_text += f"Best PSNR: Epoch {best_psnr_epoch}\n"
        summary_text += f"  {max(psnr_means):.2f} dB\n\n"
        
        best_ssim_epoch = epochs[np.argmax(ssim_means)]
        summary_text += f"Best SSIM: Epoch {best_ssim_epoch}\n"
        summary_text += f"  {max(ssim_means):.4f}\n\n"
        
        if 'fid' in self.epoch_metrics[epochs[0]]:
            best_fid_epoch = epochs[np.argmin(fid_scores)]
            summary_text += f"Best FID: Epoch {best_fid_epoch}\n"
            summary_text += f"  {min(fid_scores):.2f}"
        
        axes[1, 2].text(0.1, 0.5, summary_text, 
                       transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = self.output_dir / "metrics_across_epochs.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved plot to: {plot_path}")
    
    def generate_summary_report(self):
        """Generate HTML summary report"""
        print("\n📝 Generating summary report...")
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stage-1 Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .best { background-color: #90EE90 !important; font-weight: bold; }
                .metric-card { background: white; padding: 20px; margin: 20px 0; 
                              border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .images { display: flex; flex-wrap: wrap; gap: 10px; }
                .images img { max-width: 300px; border: 1px solid #ddd; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>🎨 Stage-1 Sketch-Guided Diffusion - Validation Report</h1>
            <div class="metric-card">
                <h2>📊 Metrics Across Epochs</h2>
                <table>
                    <tr>
                        <th>Epoch</th>
                        <th>PSNR (dB)</th>
                        <th>SSIM</th>
                        <th>LPIPS</th>
                        <th>FID</th>
                        <th>CLIP Sim</th>
                        <th>Samples</th>
                    </tr>
        """
        
        # Find best metrics
        epochs = sorted(self.epoch_metrics.keys())
        psnr_values = [self.epoch_metrics[e]['psnr']['mean'] for e in epochs]
        ssim_values = [self.epoch_metrics[e]['ssim']['mean'] for e in epochs]
        
        best_psnr_idx = np.argmax(psnr_values)
        best_ssim_idx = np.argmax(ssim_values)
        
        for i, epoch in enumerate(epochs):
            m = self.epoch_metrics[epoch]
            row_class_psnr = ' class="best"' if i == best_psnr_idx else ''
            row_class_ssim = ' class="best"' if i == best_ssim_idx else ''
            
            html += f"""
                    <tr>
                        <td><strong>Epoch {epoch}</strong></td>
                        <td{row_class_psnr}>{m['psnr']['mean']:.2f} ± {m['psnr']['std']:.2f}</td>
                        <td{row_class_ssim}>{m['ssim']['mean']:.4f} ± {m['ssim']['std']:.4f}</td>
                        <td>{m.get('lpips', {}).get('mean', 'N/A')}</td>
                        <td>{m.get('fid', 'N/A')}</td>
                        <td>{m.get('clip_similarity', {}).get('mean', 'N/A')}</td>
                        <td>{m['num_samples']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="metric-card">
                <h2>📈 Visualization</h2>
                <img src="metrics_across_epochs.png" style="max-width: 100%;">
            </div>
            
            <div class="metric-card">
                <h2>🎯 Recommendations</h2>
                <ul>
        """
        
        # Add recommendations
        final_epoch = epochs[-1]
        final_ssim = self.epoch_metrics[final_epoch]['ssim']['mean']
        
        if final_ssim < 0.5:
            html += "<li>⚠️ SSIM is low (&lt;0.5). Consider training for more epochs.</li>"
        elif final_ssim > 0.7:
            html += "<li>✅ SSIM is good (&gt;0.7). Model is learning structure well.</li>"
        
        if 'fid' in self.epoch_metrics[final_epoch]:
            final_fid = self.epoch_metrics[final_epoch]['fid']
            if final_fid < 50:
                html += "<li>✅ FID is good (&lt;50). Image quality is high.</li>"
            elif final_fid > 100:
                html += "<li>⚠️ FID is high (&gt;100). May need more training or tuning.</li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_path = self.output_dir / "validation_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"   ✅ Saved report to: {report_path}")
    
    def run_full_validation(
        self,
        epochs: Optional[List[int]] = None,
        num_samples: int = 50
    ):
        """
        Run complete validation pipeline across all epochs.
        
        Args:
            epochs: List of epochs to validate (None = auto-detect)
            num_samples: Number of samples per epoch
        """
        print("="*80)
        print("🎯 STAGE-1 EPOCH-BY-EPOCH VALIDATION")
        print("="*80)
        
        # Get epochs
        if epochs is None:
            epochs = self.get_available_epochs()
            if not epochs:
                print("\n❌ No checkpoints found!")
                return
        
        print(f"\n📋 Will validate {len(epochs)} epochs: {epochs}")
        
        # Validate each epoch
        for epoch in epochs:
            try:
                # Download checkpoint
                checkpoint_path = self.download_checkpoint(epoch)
                if checkpoint_path is None:
                    print(f"\n⚠️  Skipping epoch {epoch} (checkpoint not found)")
                    continue
                
                # Load model
                pipeline, checkpoint_info = self.load_model(checkpoint_path)
                
                # Validate
                metrics = self.validate_epoch(
                    epoch=epoch,
                    pipeline=pipeline,
                    num_samples=num_samples,
                    save_images=True
                )
                
                # Store metrics
                self.epoch_metrics[epoch] = metrics
                
                # Print summary
                print(f"\n   📊 Epoch {epoch} Results:")
                print(f"      PSNR: {metrics['psnr']['mean']:.2f} dB")
                print(f"      SSIM: {metrics['ssim']['mean']:.4f}")
                if 'fid' in metrics:
                    print(f"      FID: {metrics['fid']:.2f}")
                
                # Clean up
                del pipeline
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n❌ Error validating epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create visualizations
        self.plot_metrics_across_epochs()
        
        # Generate report
        self.generate_summary_report()
        
        # Save all metrics
        summary_path = self.output_dir / "all_epochs_metrics.json"
        with open(summary_path, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ VALIDATION COMPLETE!")
        print("="*80)
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"📊 Metrics plot: {self.output_dir / 'metrics_across_epochs.png'}")
        print(f"📝 HTML report: {self.output_dir / 'validation_report.html'}")
        print(f"📄 JSON metrics: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate Stage-1 model across training epochs'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default='DrRORAL/ragaf-diffusion-checkpoints',
        help='HuggingFace repository ID'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/workspace/sketchy',
        help='Path to Sketchy dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='validation_results',
        help='Output directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        default=None,
        help='Specific epochs to validate (default: all available)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='Number of validation samples per epoch'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=2.5,
        help='Guidance scale for generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = EpochValidator(
        hf_repo_id=args.hf_repo,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        device=args.device,
        guidance_scale=args.guidance_scale
    )
    
    # Run validation
    validator.run_full_validation(
        epochs=args.epochs,
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    main()
