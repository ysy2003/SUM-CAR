#!/bin/bash

echo "========================================"
echo "Base Model Evaluation"
echo "========================================"
echo ""
echo "Testing GPT-2 base model (without memory)"
echo "on GSM8K, HumanEval, and FinQA"
echo ""

cd "$(dirname "$0")/.."

# Parse command line arguments
USE_COT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_cot)
            USE_COT=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Run full evaluation
if [ "$USE_COT" = true ]; then
    echo "Using Chain-of-Thought prompting"
    python baselines/eval_base_model.py \
        --base_model gpt2 \
        --out baselines/base_model_results_cot.json \
        --use_cot
else
    echo "Using normal prompting"
    python baselines/eval_base_model.py \
        --base_model gpt2 \
        --out baselines/base_model_results.json
fi

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
