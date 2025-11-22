#!/bin/bash

cd "$(dirname "$0")/.."

# Parse command line arguments first
USE_COT=false
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MAX_SAMPLES=5
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_cot)
            USE_COT=true
            shift
            ;;
        --base_model)
            BASE_MODEL="$2"
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

# Now print with actual values
echo "========================================"
echo "Base Model Quick Evaluation"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Max samples per task: $MAX_SAMPLES"
echo "  Chain-of-Thought: $USE_COT"
echo ""

# Run quick evaluation with limited samples per task
if [ "$USE_COT" = true ]; then
    echo "Using Chain-of-Thought prompting"
    python baselines/eval_base_model.py \
        --base_model "$BASE_MODEL" \
        --out baselines/llama3_8b_instruct_results_quick_cot.json \
        --max_samples $MAX_SAMPLES \
        --use_cot
else
    echo "Using normal prompting"
    python baselines/eval_base_model.py \
        --base_model "$BASE_MODEL" \
        --out baselines/llama3_8b_instruct_results_quick.json \
        --max_samples $MAX_SAMPLES
fi

echo ""
echo "========================================"
echo "Quick Evaluation Complete!"
echo "========================================"
