#!/bin/bash
# Quick Validation Script for Stage 1 Model

echo "=========================================="
echo "🎯 Stage 1 Validation Metrics Evaluation"
echo "=========================================="
echo ""

# Default values
CHECKPOINT="/root/checkpoints/stage1/final.pt"
NUM_SAMPLES=50
GUIDANCE_SCALE=2.5
OUTPUT_DIR="validation_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            NUM_SAMPLES=10
            shift
            ;;
        --full)
            NUM_SAMPLES=200
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--checkpoint PATH] [--samples N] [--guidance SCALE] [--output DIR] [--quick|--full]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Samples: $NUM_SAMPLES"
echo "  Guidance Scale: $GUIDANCE_SCALE"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation
python3 evaluate_stage1_validation.py \
    --checkpoint "$CHECKPOINT" \
    --num_samples "$NUM_SAMPLES" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Evaluation Complete!"
    echo "=========================================="
    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR/"
    echo "📊 Metrics JSON: $OUTPUT_DIR/validation_metrics.json"
    echo "🖼️  Comparisons: $OUTPUT_DIR/comparison_*.png"
    echo ""
else
    echo ""
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
    echo ""
fi

exit $EXIT_CODE
