#!/bin/bash

# ============================================================================
# Stage 2 Training Startup Script for RAGAF-Diffusion
# 
# This script sets up environment and starts Stage 2 semantic refinement training
# using the Stage 1 checkpoint (epoch_18.pt) as the base.
#
# Usage:
#   bash start_stage2_training.sh                    # Default: Stage 2, 10 epochs
#   bash start_stage2_training.sh stage2 5 4         # Stage 2, 5 epochs, batch 4
#   bash start_stage2_training.sh both 10 8          # Dual-stage, 10 epochs, batch 8
#
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RAGAF-Diffusion Stage 2 Training Startup           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# Configuration
# ============================================================================

PROJECT_DIR="/root/Dual-Stage-Controllable-Diffusion-with-Adaptive-Modality-Fusion"
CHECKPOINT_DIR="/root/checkpoints"
STAGE1_CHECKPOINT="${CHECKPOINT_DIR}/stage1_with_ssim/epoch_18.pt"
LOGS_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse arguments
STAGE="${1:-stage2}"              # Default: stage2
EPOCHS="${2:-10}"                 # Default: 10 epochs
BATCH_SIZE="${3:-4}"              # Default: batch size 4
LEARNING_RATE="${4:-1e-4}"        # Default: learning rate

echo -e "${GREEN}Configuration:${NC}"
echo "  Project dir:        ${PROJECT_DIR}"
echo "  Training stage:     ${STAGE}"
echo "  Epochs:             ${EPOCHS}"
echo "  Batch size:         ${BATCH_SIZE}"
echo "  Learning rate:      ${LEARNING_RATE}"
echo "  Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"
echo ""

# ============================================================================
# Validation
# ============================================================================

echo -e "${YELLOW}[1/6] Validating environment...${NC}"

# Check project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo -e "${RED}❌ Project directory not found: ${PROJECT_DIR}${NC}"
    exit 1
fi
cd "${PROJECT_DIR}"
echo -e "${GREEN}✅ Project directory found${NC}"

# Check Stage 1 checkpoint exists
if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
    echo -e "${RED}❌ Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}${NC}"
    echo "   Please download from HuggingFace: https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints"
    exit 1
fi
CKPT_SIZE=$(du -sh "${STAGE1_CHECKPOINT}" | cut -f1)
echo -e "${GREEN}✅ Stage 1 checkpoint found (${CKPT_SIZE})${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✅ Python ${PYTHON_VERSION} available${NC}"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ CUDA not available (nvidia-smi not found)${NC}"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo -e "${GREEN}✅ GPU available: ${GPU_NAME} (${GPU_VRAM})${NC}"

# Check datasets
if [ ! -d "/workspace/sketchy" ]; then
    echo -e "${RED}❌ Sketchy dataset not found at /workspace/sketchy${NC}"
    echo "   Training cannot proceed without data. Please ensure dataset is available."
    exit 1
fi
TRAIN_SAMPLES=$(find /workspace/sketchy -name "*.jpg" 2>/dev/null | wc -l)
echo -e "${GREEN}✅ Sketchy dataset found (~${TRAIN_SAMPLES} images)${NC}"

# Check required Python packages
echo -e "${YELLOW}[2/6] Checking Python dependencies...${NC}"
python -c "
import torch
import transformers
import diffusers
import accelerate
import lpips
print('✅ All required packages available')
" || {
    echo -e "${RED}❌ Missing required Python packages${NC}"
    echo "   Run: pip install -r requirements.txt"
    exit 1
}

# ============================================================================
# Directory Setup
# ============================================================================

echo -e "${YELLOW}[3/6] Setting up directories...${NC}"

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}/stage2"
echo -e "${GREEN}✅ Checkpoint directory ready: ${CHECKPOINT_DIR}/stage2${NC}"

# Create logs directory
mkdir -p "${LOGS_DIR}"
echo -e "${GREEN}✅ Logs directory ready: ${LOGS_DIR}${NC}"

# ============================================================================
# Verify Stage 2 Model
# ============================================================================

echo -e "${YELLOW}[4/6] Verifying Stage 2 model...${NC}"

python verify_stage2_model.py
if [ $? -ne 0 ]; then
    exit 1
fi

# ============================================================================
# Git Status
# ============================================================================

echo -e "${YELLOW}[5/6] Checking Git status...${NC}"

if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    REMOTE_URL=$(git config --get remote.origin.url)
    echo -e "${GREEN}✅ Git repository ready${NC}"
    echo "   Branch: ${CURRENT_BRANCH}"
    echo "   Remote: ${REMOTE_URL}"
else
    echo -e "${YELLOW}⚠️  Not a git repository (but this is OK)${NC}"
fi

# ============================================================================
# Start Training
# ============================================================================

echo -e "${YELLOW}[6/6] Starting training...${NC}"
echo ""

LOG_FILE="${LOGS_DIR}/stage2_training_${TIMESTAMP}.log"
PID_FILE="${LOGS_DIR}/stage2_training_${TIMESTAMP}.pid"

# Build training command
TRAIN_CMD="python scripts/training/train.py \
  --stage ${STAGE} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --epochs ${EPOCHS} \
  --checkpoint_dir ${CHECKPOINT_DIR}"

echo -e "${BLUE}Training Command:${NC}"
echo "${TRAIN_CMD}"
echo ""

# Run training in background
nohup ${TRAIN_CMD} > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo ${TRAIN_PID} > "${PID_FILE}"

# ============================================================================
# Post-Start Feedback
# ============================================================================

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Training Started! ✅                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Process ID: ${TRAIN_PID}"
echo "Log file:   ${LOG_FILE}"
echo ""
echo -e "${BLUE}Monitor Training:${NC}"
echo "  • Watch logs:        tail -f ${LOG_FILE}"
echo "  • GPU usage:         watch -n 1 nvidia-smi"
echo "  • Stop training:     kill ${TRAIN_PID}"
echo ""
echo -e "${BLUE}Expected Timeline:${NC}"
echo "  • Stage 2 (10 epochs):  ~7.5 hours"
echo "  • Dual-stage (10 epochs each): ~15 hours"
echo ""
echo -e "${BLUE}Checkpoint Location:${NC}"
echo "  • Local: ${CHECKPOINT_DIR}/stage2/"
echo "  • Hub:   https://huggingface.co/DrRORAL/ragaf-diffusion-checkpoints"
echo ""

# Show first 20 lines of log after 5 seconds
sleep 5
if [ -f "${LOG_FILE}" ]; then
    echo -e "${YELLOW}Training Log (first 20 lines):${NC}"
    head -20 "${LOG_FILE}"
    echo "  ..."
    echo "  (More logs available in: ${LOG_FILE})"
    echo ""
fi

# Final status
sleep 2
if ps -p ${TRAIN_PID} > /dev/null; then
    echo -e "${GREEN}✅ Training process is running (PID: ${TRAIN_PID})${NC}"
else
    echo -e "${RED}❌ Training process failed to start. Check logs:${NC}"
    echo "   ${LOG_FILE}"
    exit 1
fi

echo ""
echo -e "${BLUE}💡 Tip: To view real-time training progress, run:${NC}"
echo "   tail -f ${LOG_FILE}"
echo ""

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Device: ${DEVICE}"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Log file: ${LOG_FILE}"
echo ""

# Start training
echo -e "${BLUE}Starting training...${NC}"
echo ""

cd "${PROJECT_DIR}"

# Set PYTHONPATH for imports
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Run training with nohup for background execution
nohup python -u scripts/training/train.py \
    --stage stage2 \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --checkpoint_dir ${CHECKPOINT_DIR}/stage2 \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!

echo -e "${GREEN}✅ Training started with PID: ${TRAIN_PID}${NC}"
echo -e "${GREEN}✅ Logs: ${LOG_FILE}${NC}"
echo ""
echo -e "${YELLOW}Monitor training with:${NC}"
echo "  tail -f ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Or check GPU usage with:${NC}"
echo "  watch -n 2 'nvidia-smi'"
echo ""

# Save PID for reference
echo ${TRAIN_PID} > "${PROJECT_DIR}/.stage2_training_pid"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Stage 2 Training Started!${NC}"
echo -e "${GREEN}========================================${NC}"
