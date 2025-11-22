#!/bin/bash

cd "$(dirname "$0")/.."

# Parse command line arguments first
USE_COT=false
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
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
        *)
            shift
            ;;
    esac
done

# Now print with actual values
echo "========================================"
echo "Base Model Evaluation (Full Dataset)"
echo "========================================"
echo ""
echo "Testing base model (without memory)"
echo "on full GSM8K, HumanEval, and FinQA datasets"
echo ""
echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Chain-of-Thought: $USE_COT"
echo ""

# Run full evaluation
if [ "$USE_COT" = true ]; then
    echo "Using Chain-of-Thought prompting"
    python baselines/eval_base_model.py \
        --base_model "$BASE_MODEL" \
        --out baselines/llama3_8b_instruct_results_cot.json \
        --use_cot
else
    echo "Using normal prompting"
    python baselines/eval_base_model.py \
        --base_model "$BASE_MODEL" \
        --out baselines/llama3_8b_instruct_results.json
fi

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
