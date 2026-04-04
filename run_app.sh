#!/bin/bash
# RAGAF-Diffusion Streamlit App Launcher
# Usage: ./run_app.sh [port]

PORT=${1:-8501}

echo "========================================"
echo "  RAGAF-Diffusion Streamlit App"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check for checkpoints
if [ -z "$STAGE1_CHECKPOINT" ]; then
    export STAGE1_CHECKPOINT="/workspace/checkpoints/stage1/epoch_18.pt"
fi

if [ -z "$STAGE2_CHECKPOINT" ]; then
    export STAGE2_CHECKPOINT="/workspace/checkpoints/stage2/epoch_6.pt"
fi

echo "Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "Stage 2 checkpoint: $STAGE2_CHECKPOINT"
echo ""

# Check if files exist
if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "Warning: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    echo "Please set STAGE1_CHECKPOINT environment variable to the correct path"
fi

if [ ! -f "$STAGE2_CHECKPOINT" ]; then
    echo "Warning: Stage 2 checkpoint not found at $STAGE2_CHECKPOINT"
    echo "Please set STAGE2_CHECKPOINT environment variable to the correct path"
fi

echo ""
echo "Starting Streamlit on port $PORT..."
echo "Open http://localhost:$PORT in your browser"
echo "========================================"

streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
