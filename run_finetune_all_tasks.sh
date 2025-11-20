#!/bin/bash
# 运行所有任务的 finetune
# 确保输出目录不会互相覆盖

set -e

echo "=== Starting Multi-Task Training ==="
echo "Tasks: Math (GSM8K), Code (CodeXGLUE), FinQA"
echo ""

# 1. Math (GSM8K)
echo "========================================="
echo "Task 1/3: Training Math (GSM8K)"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_math.yaml \
    --use_xla True
echo "✓ Math training completed. Outputs in: out/math/ and patches/math/"
echo ""

# 2. Code (CodeXGLUE)
echo "========================================="
echo "Task 2/3: Training Code (CodeXGLUE)"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_code.yaml \
    --use_xla True
echo "✓ Code training completed. Outputs in: out/code/ and patches/code/"
echo ""

# 3. FinQA
echo "========================================="
echo "Task 3/3: Training FinQA"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_finqa.yaml \
    --use_xla True
echo "✓ FinQA training completed. Outputs in: out/finqa/ and patches/finqa/"
echo ""

echo "=== All Training Completed ==="
echo ""
echo "Results summary:"
echo "  - Math patches:  out/math/, patches/math/"
echo "  - Code patches:  out/code/, patches/code/"
echo "  - FinQA patches: out/finqa/, patches/finqa/"
echo ""
echo "Check meta.json files for TF-IDF statistics:"
echo "  - out/math/patch_math_meta.json"
echo "  - out/code/patch_code_meta.json"
echo "  - out/finqa/patch_finqa_meta.json"
