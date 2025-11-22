#!/bin/bash

echo "========================================"
echo "Base Model Quick Evaluation"
echo "========================================"
echo ""
echo "Testing GPT-2 base model (without memory)"
echo "on 2 samples per task for quick verification"
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

# Run quick evaluation with 2 samples per task
if [ "$USE_COT" = true ]; then
    echo "Using Chain-of-Thought prompting"
    python baselines/eval_base_model.py \
        --base_model gpt2 \
        --out baselines/base_model_results_quick_cot.json \
        --max_samples 2 \
        --use_cot
else
    echo "Using normal prompting"
    python baselines/eval_base_model.py \
        --base_model gpt2 \
        --out baselines/base_model_results_quick.json \
        --max_samples 2
fi

echo ""
echo "========================================"
echo "Quick Evaluation Complete!"
echo "========================================"
