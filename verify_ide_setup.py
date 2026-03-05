#!/usr/bin/env python3
"""
IDE Setup Verification Script for RAGAF-Diffusion
Tests all imports and basic functionality
"""

import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"{GREEN}✓{NC} {description}")
        return True
    except ImportError as e:
        print(f"{RED}✗{NC} {description}: {e}")
        return False

def main():
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}RAGAF-Diffusion IDE Setup Verification{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")
    
    passed = 0
    failed = 0
    
    # Test Python version
    print(f"{BLUE}Python Environment:{NC}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  Python path: {sys.executable}\n")
    
    # Test core dependencies
    print(f"{BLUE}Core Dependencies:{NC}")
    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "HuggingFace Transformers"),
        ("diffusers", "HuggingFace Diffusers"),
        ("accelerate", "Accelerate"),
    ]
    
    for module, desc in tests:
        if test_import(module, desc):
            passed += 1
        else:
            failed += 1
    
    # Test PyTorch CUDA
    try:
        import torch
        print(f"\n{BLUE}PyTorch Configuration:{NC}")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Device Count: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.current_device()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"{RED}Error checking PyTorch: {e}{NC}")
    
    # Test data processing libraries
    print(f"\n{BLUE}Data Processing Libraries:{NC}")
    tests = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
        ("skimage", "scikit-image"),
        ("PIL", "Pillow"),
        ("networkx", "NetworkX"),
        ("pycocotools", "pycocotools"),
    ]
    
    for module, desc in tests:
        if test_import(module, desc):
            passed += 1
        else:
            failed += 1
    
    # Test utilities
    print(f"\n{BLUE}Utility Libraries:{NC}")
    tests = [
        ("matplotlib", "Matplotlib"),
        ("tensorboard", "TensorBoard"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
        ("einops", "einops"),
    ]
    
    for module, desc in tests:
        if test_import(module, desc):
            passed += 1
        else:
            failed += 1
    
    # Test project modules
    print(f"\n{BLUE}Project Modules:{NC}")
    project_tests = [
        ("configs.config", "Configuration"),
        ("data.sketch_extraction", "Sketch Extraction"),
        ("data.region_extraction", "Region Extraction"),
        ("data.region_graph", "Region Graph"),
        ("models.ragaf_attention", "RAGAF Attention"),
        ("models.adaptive_fusion", "Adaptive Fusion"),
        ("models.stage1_diffusion", "Stage 1 Diffusion"),
        ("models.stage2_refinement", "Stage 2 Refinement"),
        ("datasets.coco_dataset", "COCO Dataset"),
        ("datasets.sketchy_dataset", "Sketchy Dataset"),
        ("utils.common", "Common Utils"),
    ]
    
    for module, desc in project_tests:
        if test_import(module, desc):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}Summary:{NC}")
    print(f"  {GREEN}Passed: {passed}{NC}")
    print(f"  {RED}Failed: {failed}{NC}")
    
    if failed == 0:
        print(f"\n{GREEN}✓ IDE setup is complete! All modules are working.{NC}")
        print(f"\n{BLUE}Next steps:{NC}")
        print(f"  1. Configure datasets: Set SKETCHY_ROOT and COCO_ROOT environment variables")
        print(f"  2. Generate config: python configs/config.py")
        print(f"  3. Verify dataset: python verify_dataset.py")
        print(f"  4. Run tests: python test_stage1.py")
        print(f"  5. Start training: python train.py --stage 1 --config default_config.yaml")
        return 0
    else:
        print(f"\n{YELLOW}⚠ Some modules failed to import. Please install missing dependencies.{NC}")
        print(f"\nTo install missing packages:")
        print(f"  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
