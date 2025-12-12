#!/bin/bash

set -e

# Set PYTHONPATH so Python can find the sumcar module
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== Starting Multi-Task Training ==="
echo "Tasks: Math (GSM8K), Code (CodeXGLUE), FinQA"
echo ""

# 1. Math (GSM8K)
echo "========================================="
echo "Task 1/3: Training Math (GSM8K) with CoT"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_math_cot.yaml
echo "✓ Math training completed. Outputs in: out_fp16/math_cot/"
echo ""

# 2. Code (CodeXGLUE)
echo "========================================="
echo "Task 2/3: Training Code (CodeXGLUE)"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_code.yaml
echo "✓ Code training completed. Outputs in: out_fp16/code/"
echo ""

# 3. FinQA
echo "========================================="
echo "Task 3/3: Training FinQA with CoT"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_finqa_cot.yaml
echo "✓ FinQA training completed. Outputs in: out_fp16/finqa_cot/"
echo ""

echo "=== All Training Completed ==="
echo ""
echo "Results summary (FP16):"
echo "  - Math (CoT):  out_fp16/math_cot/ (batch_size=4)"
echo "  - Code:        out_fp16/code/ (batch_size=8)"
echo "  - FinQA (CoT): out_fp16/finqa_cot/ (batch_size=2)"
echo ""
echo "Check training logs:"
echo "  - out_fp16/math_cot/math_cot/training.log"
echo "  - out_fp16/code/code/training.log"
echo "  - out_fp16/finqa_cot/finqa_cot/training.log"
echo ""
echo "Check patches and metadata:"
echo "  - out_fp16/patch_math_cot.json"
echo "  - out_fp16/patch_math_cot_meta.json"
echo "  - out_fp16/patch_codexglue.json"
echo "  - out_fp16/patch_codexglue_meta.json"
echo "  - out_fp16/patch_finqa_cot.json"
echo "  - out_fp16/patch_finqa_cot_meta.json"
