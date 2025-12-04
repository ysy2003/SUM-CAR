#!/bin/bash
# Train all tasks on Qwen3-8B with thinking mode
# Outputs to out_qwen3/ to avoid overwriting Llama outputs

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# Set PYTHONPATH so Python can find the sumcar module
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== Starting Multi-Task Training on Qwen3-8B ==="
echo "Tasks: Math (GSM8K), Code (CodeXGLUE), FinQA"
echo "Base Model: Qwen/Qwen3-8B (with thinking mode)"
echo ""

# 1. Math (GSM8K)
echo "========================================="
echo "Task 1/3: Training Math (GSM8K) with CoT"
echo "========================================="
python -m sumcar.cli.train_task \
    --config Qwen3finetune/configs/train_math_cot.yaml
echo "✓ Math training completed. Outputs in: out_qwen3/math_cot/"
echo ""

# 2. Code (CodeXGLUE)
echo "========================================="
echo "Task 2/3: Training Code (CodeXGLUE)"
echo "========================================="
python -m sumcar.cli.train_task \
    --config Qwen3finetune/configs/train_code.yaml
echo "✓ Code training completed. Outputs in: out_qwen3/code/"
echo ""

# 3. FinQA
echo "========================================="
echo "Task 3/3: Training FinQA with CoT"
echo "========================================="
python -m sumcar.cli.train_task \
    --config Qwen3finetune/configs/train_finqa_cot.yaml
echo "✓ FinQA training completed. Outputs in: out_qwen3/finqa_cot/"
echo ""

echo "=== All Training Completed ==="
echo ""
echo "Results summary:"
echo "  - Math (CoT):  out_qwen3/math_cot/"
echo "  - Code:        out_qwen3/code/"
echo "  - FinQA (CoT): out_qwen3/finqa_cot/"
echo ""
echo "Check training logs:"
echo "  - out_qwen3/math_cot/math_cot/training.log"
echo "  - out_qwen3/code/code/training.log"
echo "  - out_qwen3/finqa_cot/finqa_cot/training.log"
echo ""
echo "Check meta.json files for loss history and TF-IDF statistics:"
echo "  - out_qwen3/patch_math_cot_meta.json"
echo "  - out_qwen3/patch_code_meta.json"
echo "  - out_qwen3/patch_finqa_cot_meta.json"
echo ""
echo "Next step: Run merge script"
echo "  bash Qwen3finetune/run_merge_qwen3.sh"
