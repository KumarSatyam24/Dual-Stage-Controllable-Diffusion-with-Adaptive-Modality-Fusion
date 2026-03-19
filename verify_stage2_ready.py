#!/usr/bin/env python
"""
Verify Stage 2 Training Setup - Complete System Check

This script verifies all components are ready for Stage 2 training.
Run this before starting training to catch any issues early.
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check(condition, message):
    """Print check result."""
    if condition:
        print(f"{GREEN}✅{RESET} {message}")
        return True
    else:
        print(f"{RED}❌{RESET} {message}")
        return False

def warning(message):
    """Print warning."""
    print(f"{YELLOW}⚠️{RESET}  {message}")

def info(message):
    """Print info."""
    print(f"{BLUE}ℹ️{RESET}  {message}")

def main():
    print(f"\n{BLUE}╔════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║      Stage 2 Training Setup Verification (Complete)        ║{RESET}")
    print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{RESET}\n")

    passed = 0
    total = 0

    # ====================================================================
    # 1. Environment Setup
    # ====================================================================
    print(f"{BLUE}[1] Environment Setup{RESET}")
    print("-" * 60)

    total += 1
    if check(
        os.path.exists("/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"),
        "Project directory exists"
    ):
        passed += 1

    total += 1
    if check(
        os.path.exists("/root/checkpoints/stage1_with_ssim/epoch_18.pt"),
        "Stage 1 checkpoint (epoch_18.pt) exists"
    ):
        ckpt_size = os.path.getsize("/root/checkpoints/stage1_with_ssim/epoch_18.pt") / (1024**3)
        info(f"Checkpoint size: {ckpt_size:.1f} GB")
        passed += 1
    else:
        warning("Download Stage 1 checkpoint from HuggingFace")

    total += 1
    try:
        python_version = subprocess.check_output(["python", "--version"], text=True).strip()
        if check(True, f"Python available ({python_version})"):
            passed += 1
    except:
        check(False, "Python not available")

    total += 1
    try:
        gpu_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().split('\n')[0]
        if check(True, f"GPU available: {gpu_output}"):
            passed += 1
    except:
        check(False, "CUDA/GPU not available")

    total += 1
    if check(
        os.path.exists("/workspace/sketchy"),
        "Sketchy dataset available (/workspace/sketchy)"
    ):
        num_samples = len(list(Path("/workspace/sketchy").glob("**/*.jpg")))
        info(f"Dataset contains ~{num_samples} images")
        passed += 1

    print()

    # ====================================================================
    # 2. Python Dependencies
    # ====================================================================
    print(f"{BLUE}[2] Python Dependencies{RESET}")
    print("-" * 60)

    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("diffusers", "Diffusers"),
        ("accelerate", "Accelerate"),
        ("lpips", "LPIPS"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]

    for package, name in required_packages:
        total += 1
        try:
            __import__(package)
            if check(True, f"{name} ({package})"):
                passed += 1
        except ImportError:
            check(False, f"{name} ({package}) - install with: pip install {package}")

    print()

    # ====================================================================
    # 3. Project Structure
    # ====================================================================
    print(f"{BLUE}[3] Project Structure{RESET}")
    print("-" * 60)

    required_files = [
        ("src/configs/config.py", "Main configuration"),
        ("src/models/stage1_diffusion.py", "Stage 1 model"),
        ("src/models/stage2_refinement.py", "Stage 2 model (with feature projection)"),
        ("src/models/ragaf_attention.py", "RAGAF attention module"),
        ("src/models/adaptive_fusion.py", "Adaptive fusion module"),
        ("scripts/training/train.py", "Training script"),
        ("scripts/inference/inference.py", "Inference script"),
        ("datasets/sketchy_dataset.py", "Sketchy dataset loader"),
    ]

    base_dir = "/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
    for file_path, description in required_files:
        total += 1
        full_path = os.path.join(base_dir, file_path)
        if check(os.path.exists(full_path), f"{description}"):
            passed += 1
        else:
            warning(f"Missing: {file_path}")

    print()

    # ====================================================================
    # 4. Stage 2 Model Verification
    # ====================================================================
    print(f"{BLUE}[4] Stage 2 Model Verification{RESET}")
    print("-" * 60)

    sys.path.insert(0, base_dir)

    try:
        from src.models.stage2_refinement import Stage2SemanticRefinement
        from diffusers import UNet2DConditionModel
        import torch

        total += 1
        info("Loading Stage 2 model components...")
        
        # Load UNet
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet"
        )
        
        # Create Stage 2 model
        stage2_model = Stage2SemanticRefinement(
            unet=unet,
            node_feature_dim=6,
            text_dim=768,
            hidden_dim=512,
            num_graph_layers=2,
            num_attention_heads=8,
            fusion_method="learned",
            use_region_adaptive_fusion=True
        )
        
        # Verify feature projection
        total += 1
        if check(
            hasattr(stage2_model, 'feature_projection'),
            "Feature projection layer exists"
        ):
            passed += 1
        
        # Count parameters
        total_params = sum(p.numel() for p in stage2_model.parameters())
        trainable_params = sum(p.numel() for p in stage2_model.parameters() if p.requires_grad)
        
        total += 1
        if check(True, f"Stage 2 model initialized"):
            info(f"Total parameters: {total_params/1e6:.1f}M")
            info(f"Trainable parameters: {trainable_params/1e6:.1f}M")
            passed += 1
        
    except Exception as e:
        check(False, f"Stage 2 model verification failed: {e}")

    print()

    # ====================================================================
    # 5. HuggingFace & Git
    # ====================================================================
    print(f"{BLUE}[5] HuggingFace & Git Integration{RESET}")
    print("-" * 60)

    total += 1
    try:
        from huggingface_hub import whoami
        user = whoami()
        if check(True, f"HuggingFace authenticated as: {user['name']}"):
            passed += 1
    except:
        check(False, "HuggingFace not authenticated - run: huggingface-cli login")

    total += 1
    os.chdir(base_dir)
    try:
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True
        ).strip()
        if check(True, "Git remote configured"):
            info(f"Remote: {remote_url}")
            passed += 1
    except:
        check(False, "Git remote not configured")

    print()

    # ====================================================================
    # 6. Disk Space & Resources
    # ====================================================================
    print(f"{BLUE}[6] Disk Space & Resources{RESET}")
    print("-" * 60)

    import shutil
    
    total += 1
    free_bytes = shutil.disk_usage("/root").free
    free_gb = free_bytes / (1024**3)
    if check(free_gb > 20, f"Sufficient disk space ({free_gb:.1f} GB free)"):
        passed += 1
    else:
        warning(f"Low disk space: {free_gb:.1f} GB free (need at least 20 GB)")

    total += 1
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_gb = mem_bytes / (1024**3)
        if check(mem_gb >= 32, f"Sufficient RAM ({mem_gb:.1f} GB)"):
            passed += 1
        else:
            warning(f"Limited RAM: {mem_gb:.1f} GB (recommended: 32+ GB)")
    except:
        pass

    print()

    # ====================================================================
    # Summary
    # ====================================================================
    print(f"{BLUE}╔════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║                        Summary                             ║{RESET}")
    print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{RESET}\n")

    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Tests passed: {passed}/{total} ({percentage:.0f}%)\n")

    if percentage == 100:
        print(f"{GREEN}╔════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{GREEN}║           ✅ ALL SYSTEMS GO - READY TO TRAIN ✅             ║{RESET}")
        print(f"{GREEN}╚════════════════════════════════════════════════════════════╝{RESET}\n")
        print(f"{GREEN}Next step: bash start_stage2_training.sh{RESET}\n")
        return 0
    elif percentage >= 80:
        print(f"{YELLOW}⚠️  Some checks failed, but you may still be able to train.{RESET}")
        print(f"{YELLOW}Review the failures above and fix any critical issues.{RESET}\n")
        return 1
    else:
        print(f"{RED}❌ Critical issues found. Please fix before training.{RESET}\n")
        return 2

if __name__ == "__main__":
    sys.exit(main())
