#!/bin/bash

echo "========================================"
echo "Base Model Quick Evaluation (100 samples)"
echo "========================================"
echo ""
echo "Testing GPT-2 base model on small subset"
echo ""

cd "$(dirname "$0")/.."

# Run quick evaluation with 100 samples per task
python baselines/eval_base_model.py \
    --base_model gpt2 \
    --out baselines/base_model_results_quick.json \
    --max_samples 100

echo ""
echo "========================================"
echo "Quick Evaluation Complete!"
echo "========================================"
