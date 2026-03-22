#!/bin/bash

# ============================================================================
# Stage 2 Training Startup Script for RAGAF-Diffusion
# 
# This script sets up environment and starts Stage 2 semantic refinement training
# using the Stage 1 checkpoint (epoch_18.pt) as the base.
#
# Usage:
#   bash start_stage2_training.sh                    # Default: Stage 2, 5 epochs
#   bash start_stage2_training.sh stage2 5 4         # Stage 2, 5 epochs, batch 4
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

PROJECT_DIR=$(pwd)
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
STAGE1_CHECKPOINT="${CHECKPOINT_DIR}/stage1_with_ssim/epoch_18.pt"
LOGS_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse arguments
STAGE="${1:-stage2}"              # Default: stage2
EPOCHS="${2:-5}"                  # Default: 5 epochs (to match TrainingConfig)
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

cd "${PROJECT_DIR}"
echo -e "${GREEN}✅ Project directory found${NC}"

# Check Stage 1 checkpoint exists
if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
    echo -e "${RED}❌ Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}${NC}"
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

# Check datasets
if [ ! -d "${PROJECT_DIR}/sketchy" ]; then
    echo -e "${RED}❌ Sketchy dataset not found at ${PROJECT_DIR}/sketchy${NC}"
    exit 1
fi
TRAIN_SAMPLES=$(find "${PROJECT_DIR}/sketchy" -name "*.jpg" 2>/dev/null | wc -l)
echo -e "${GREEN}✅ Sketchy dataset found (~${TRAIN_SAMPLES} images)${NC}"

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
# Start Training
# ============================================================================

echo -e "${YELLOW}[6/6] Starting training...${NC}"
echo ""

LOG_FILE="${LOGS_DIR}/stage2_training_${TIMESTAMP}.log"
PID_FILE="${LOGS_DIR}/stage2_training_${TIMESTAMP}.pid"

# Set PYTHONPATH for imports
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Build training command
TRAIN_CMD="python -u scripts/training/train.py \
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
echo ${TRAIN_PID} > "${PROJECT_DIR}/.stage2_training_pid"

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
echo ""

# Final status check
sleep 2
if ps -p ${TRAIN_PID} > /dev/null; then
    echo -e "${GREEN}✅ Training process is running (PID: ${TRAIN_PID})${NC}"
else
    echo -e "${RED}❌ Training process failed to start. Check logs:${NC}"
    echo "   ${LOG_FILE}"
    exit 1
fi
