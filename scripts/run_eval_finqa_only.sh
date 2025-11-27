#!/bin/bash
# Evaluate finqa-only memory on all three tasks

echo "========================================"
echo "FinQA-Only Memory Evaluation"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

# Parse arguments
USE_COT=false
MAX_SAMPLES=100
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_cot)
            USE_COT=true
            shift
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

# Check if finqa-only model exists
if [ ! -f "out/finqa_only/memory.pt" ]; then
    echo "❌ Error: FinQA-only model not found"
    echo "   Please run: bash run_merge_finqa_only.sh"
    exit 1
fi

echo "✓ FinQA-only model found"
echo ""

# Run evaluation
if [ "$USE_COT" = true ]; then
    echo "Using Chain-of-Thought prompting"
    OUT_FILE="finqa_only_results_cot.json"
    python scripts/eval_merged.py \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --merged_dir out/finqa_only \
        --out "baselines/$OUT_FILE" \
        --k_top 8 \
        --alpha 1.0 \
        --max_samples "$MAX_SAMPLES" \
        --use_cot
else
    echo "Using normal prompting"
    OUT_FILE="finqa_only_results.json"
    python scripts/eval_merged.py \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --merged_dir out/finqa_only \
        --out "baselines/$OUT_FILE" \
        --k_top 8 \
        --alpha 1.0 \
        --max_samples "$MAX_SAMPLES"
fi

echo ""
echo "========================================"
echo "FinQA-Only Evaluation Complete!"
echo "========================================"
