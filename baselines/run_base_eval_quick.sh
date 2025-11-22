#!/bin/bash

echo "========================================"
echo "Base Model Quick Evaluation (1 sample)"
echo "========================================"
echo ""
echo "Testing GPT-2 base model on 1 sample per task"
echo ""

cd "$(dirname "$0")/.."

# Run quick evaluation with 1 sample per task
python baselines/eval_base_model.py \
    --base_model gpt2 \
    --out baselines/base_model_results_quick.json \
    --max_samples 1

echo ""
echo "========================================"
echo "Quick Evaluation Complete!"
echo "========================================"
