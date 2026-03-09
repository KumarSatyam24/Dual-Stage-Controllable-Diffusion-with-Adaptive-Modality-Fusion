#!/bin/bash
# Quick Start Script for RAGAF-Diffusion
# This script demonstrates a complete workflow from setup to inference

set -e  # Exit on error

echo "=========================================="
echo "RAGAF-Diffusion Quick Start"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Environment Check
echo -e "${BLUE}Step 1: Checking environment...${NC}"
python --version
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No NVIDIA GPU detected. Training will be slow on CPU."
fi
echo ""

# Step 2: Test Data Processing
echo -e "${BLUE}Step 2: Testing data processing modules...${NC}"
echo "Testing sketch extraction..."
python data/sketch_extraction.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sketch extraction working${NC}"
else
    echo "✗ Sketch extraction failed"
fi

echo "Testing region extraction..."
python data/region_extraction.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Region extraction working${NC}"
else
    echo "✗ Region extraction failed"
fi

echo "Testing region graph construction..."
python data/region_graph.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Region graph construction working${NC}"
else
    echo "✗ Region graph construction failed"
fi
echo ""

# Step 3: Test Model Components
echo -e "${BLUE}Step 3: Testing model components...${NC}"
echo "Testing RAGAF attention module..."
python models/ragaf_attention.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ RAGAF attention module working${NC}"
else
    echo "✗ RAGAF attention module failed"
fi

echo "Testing adaptive fusion module..."
python models/adaptive_fusion.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Adaptive fusion module working${NC}"
else
    echo "✗ Adaptive fusion module failed"
fi
echo ""

# Step 4: Generate Default Config
echo -e "${BLUE}Step 4: Generating default configuration...${NC}"
python configs/config.py
if [ -f "default_config.yaml" ]; then
    echo -e "${GREEN}✓ Configuration file created: default_config.yaml${NC}"
else
    echo "✗ Configuration generation failed"
fi
echo ""

# Step 5: Dataset Check
echo -e "${BLUE}Step 5: Checking dataset availability...${NC}"
if [ -z "$SKETCHY_ROOT" ]; then
    echo "⚠ SKETCHY_ROOT not set. Please download Sketchy dataset and set:"
    echo "  export SKETCHY_ROOT=/path/to/sketchy"
else
    echo -e "${GREEN}✓ SKETCHY_ROOT set: $SKETCHY_ROOT${NC}"
    if [ -d "$SKETCHY_ROOT" ]; then
        echo -e "${GREEN}✓ Sketchy dataset directory exists${NC}"
    else
        echo "✗ Sketchy dataset directory not found"
    fi
fi

if [ -z "$COCO_ROOT" ]; then
    echo "⚠ COCO_ROOT not set. For COCO experiments, set:"
    echo "  export COCO_ROOT=/path/to/coco"
else
    echo -e "${GREEN}✓ COCO_ROOT set: $COCO_ROOT${NC}"
    if [ -d "$COCO_ROOT" ]; then
        echo -e "${GREEN}✓ COCO dataset directory exists${NC}"
    else
        echo "✗ COCO dataset directory not found"
    fi
fi
echo ""

# Step 6: Create Example Directories
echo -e "${BLUE}Step 6: Creating directory structure...${NC}"
mkdir -p examples
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs
echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""

# Step 7: Quick Test (if datasets available)
echo -e "${BLUE}Step 7: Dataset loading test...${NC}"
if [ -n "$SKETCHY_ROOT" ] && [ -d "$SKETCHY_ROOT" ]; then
    echo "Testing Sketchy dataset loader..."
    python -c "
from datasets.sketchy_dataset import SketchyDataset
import os
try:
    dataset = SketchyDataset(
        root_dir=os.environ.get('SKETCHY_ROOT'),
        split='train',
        image_size=512,
        augment=False
    )
    print(f'✓ Loaded {len(dataset)} samples from Sketchy dataset')
except Exception as e:
    print(f'✗ Failed to load Sketchy dataset: {e}')
" 2>/dev/null
else
    echo "⚠ Skipping Sketchy dataset test (SKETCHY_ROOT not set or not found)"
fi

if [ -n "$COCO_ROOT" ] && [ -d "$COCO_ROOT" ]; then
    echo "Testing COCO dataset loader..."
    python -c "
from datasets.coco_dataset import COCODataset
import os
try:
    dataset = COCODataset(
        root_dir=os.environ.get('COCO_ROOT'),
        split='val',  # Use val split for quick test
        image_size=512,
        max_samples=10,  # Just 10 samples for test
        cache_sketches=False
    )
    print(f'✓ Loaded {len(dataset)} samples from COCO dataset')
except Exception as e:
    print(f'✗ Failed to load COCO dataset: {e}')
" 2>/dev/null
else
    echo "⚠ Skipping COCO dataset test (COCO_ROOT not set or not found)"
fi
echo ""

# Final Summary
echo "=========================================="
echo -e "${GREEN}Quick Start Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download datasets if not already done:"
echo "   - Sketchy: https://sketchy.eye.gatech.edu/"
echo "   - MS COCO: https://cocodataset.org/"
echo ""
echo "2. Set environment variables:"
echo "   export SKETCHY_ROOT=/path/to/sketchy"
echo "   export COCO_ROOT=/path/to/coco"
echo ""
echo "3. Start training:"
echo "   python train.py --stage both --batch_size 4 --epochs 10"
echo ""
echo "4. Run inference (after training):"
echo "   python inference.py --sketch examples/sketch.png --prompt \"your prompt\""
echo ""
echo "For more information, see README.md"
echo ""
