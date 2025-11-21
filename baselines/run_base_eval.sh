#!/bin/bash

echo "========================================"
echo "Base Model Evaluation"
echo "========================================"
echo ""
echo "Testing GPT-2 base model (without memory)"
echo "on GSM8K, HumanEval, and FinQA"
echo ""

cd "$(dirname "$0")/.."

# Run full evaluation
python baselines/eval_base_model.py \
    --base_model gpt2 \
    --out baselines/base_model_results.json

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
