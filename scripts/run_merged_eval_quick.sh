#!/bin/bash

echo "========================================"
echo "Merged Model Quick Evaluation"
echo "========================================"
echo ""
echo "Testing merged memory model on {} samples per task"
echo "to verify the code is working correctly"
echo ""

cd "$(dirname "$0")/.."

# Parse command line arguments
USE_COT=true
MERGED_DIR="out/merged"
MAX_SAMPLES=100
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_cot)
            USE_COT=true
            shift
            ;;
        --merged_dir)
            MERGED_DIR="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check if merged model exists
if [ ! -f "$MERGED_DIR/memory.pt" ]; then
    echo "❌ Error: Merged model not found at $MERGED_DIR/memory.pt"
    echo "   Please run merge first: bash run_merge.sh"
    exit 1
fi

echo "✓ Merged model found at $MERGED_DIR"
echo ""

# Run quick evaluation
if [ "$USE_COT" = true ]; then
    echo "Using Chain-of-Thought prompting"
    OUT_FILE="${MERGED_DIR##*/}_results_quick_cot.json"
    python scripts/eval_merged.py \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --merged_dir "$MERGED_DIR" \
        --out "scripts/$OUT_FILE" \
        --k_top 8 \
        --alpha 1.0 \
        --max_samples "$MAX_SAMPLES" \
        --use_cot
else
    echo "Using normal prompting"
    OUT_FILE="${MERGED_DIR##*/}_results_quick.json"
    python scripts/eval_merged.py \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --merged_dir "$MERGED_DIR" \
        --out "scripts/$OUT_FILE" \
        --k_top 8 \
        --alpha 1.0 \
        --max_samples "$MAX_SAMPLES"
fi

echo ""
echo "========================================"
echo "Quick Evaluation Complete!"
echo "========================================"
