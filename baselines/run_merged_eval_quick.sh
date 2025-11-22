#!/bin/bash

echo "========================================"
echo "Merged Model Quick Evaluation"
echo "========================================"
echo ""
echo "Testing merged memory model on 1 sample per task"
echo "to verify the code is working correctly"
echo ""

cd "$(dirname "$0")/.."

# Parse command line arguments
USE_COT=false
MERGED_DIR="out/merged"
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
    python baselines/eval_merged_quick.py \
        --base_model gpt2 \
        --merged_dir "$MERGED_DIR" \
        --out "baselines/$OUT_FILE" \
        --k_top 4 \
        --alpha 1.0 \
        --max_samples 1 \
        --use_cot
else
    echo "Using normal prompting"
    OUT_FILE="${MERGED_DIR##*/}_results_quick.json"
    python baselines/eval_merged_quick.py \
        --base_model gpt2 \
        --merged_dir "$MERGED_DIR" \
        --out "baselines/$OUT_FILE" \
        --k_top 4 \
        --alpha 1.0 \
        --max_samples 1
fi

echo ""
echo "========================================"
echo "Quick Evaluation Complete!"
echo "========================================"
