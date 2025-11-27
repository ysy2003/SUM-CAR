#!/bin/bash
# Evaluate Qwen3-8B baseline (no memory) on all three tasks

echo "========================================="
echo "Qwen3-8B Baseline Evaluation"
echo "========================================="
echo ""
echo "Evaluating base Qwen3-8B model without memory"
echo ""

cd "$(dirname "$0")"

# Parse arguments
MAX_SAMPLES=100
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Model: Qwen/Qwen3-8B"
echo "  Max samples: $MAX_SAMPLES"
echo "  Thinking mode: ENABLED (always on)"
echo ""

# Run evaluation with thinking mode
OUT_FILE="qwen3_8b_thinking.json"
python eval_qwen3.py \
    --out "$OUT_FILE" \
    --max_samples "$MAX_SAMPLES"

echo ""
echo "========================================="
echo "Qwen3-8B Baseline Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved to: $OUT_FILE"
