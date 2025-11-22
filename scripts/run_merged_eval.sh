#!/bin/bash

echo "========================================"
echo "Merged Model Full Evaluation"
echo "========================================"
echo ""
echo "Testing merged memory model on full test sets"
echo ""

cd "$(dirname "$0")/.."

# Parse command line arguments
USE_COT=false
MERGED_DIR="out/merged"
MAX_SAMPLES=""
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

# Run full evaluation
if [ "$USE_COT" = true ]; then
    echo "Using Chain-of-Thought prompting"
    OUT_FILE="${MERGED_DIR##*/}_results_cot.json"
    if [ -n "$MAX_SAMPLES" ]; then
        python scripts/eval_merged.py \
            --base_model gpt2 \
            --merged_dir "$MERGED_DIR" \
            --out "baselines/$OUT_FILE" \
            --k_top 4 \
            --alpha 1.0 \
            --max_samples "$MAX_SAMPLES" \
            --use_cot
    else
        python scripts/eval_merged.py \
            --base_model gpt2 \
            --merged_dir "$MERGED_DIR" \
            --out "baselines/$OUT_FILE" \
            --k_top 4 \
            --alpha 1.0 \
            --max_samples 99999 \
            --use_cot
    fi
else
    echo "Using normal prompting"
    OUT_FILE="${MERGED_DIR##*/}_results.json"
    if [ -n "$MAX_SAMPLES" ]; then
        python scripts/eval_merged.py \
            --base_model gpt2 \
            --merged_dir "$MERGED_DIR" \
            --out "baselines/$OUT_FILE" \
            --k_top 4 \
            --alpha 1.0 \
            --max_samples "$MAX_SAMPLES"
    else
        python scripts/eval_merged.py \
            --base_model gpt2 \
            --merged_dir "$MERGED_DIR" \
            --out "baselines/$OUT_FILE" \
            --k_top 4 \
            --alpha 1.0 \
            --max_samples 99999
    fi
fi

echo ""
echo "========================================"
echo "Full Evaluation Complete!"
echo "========================================"
