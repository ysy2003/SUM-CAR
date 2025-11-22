#!/bin/bash
# 运行所有任务的 finetune
# 确保输出目录不会互相覆盖

set -e

echo "=== Starting Multi-Task Training ==="
echo "Tasks: Math (GSM8K), Code (CodeXGLUE), FinQA"
echo ""

# 1. Math (GSM8K)
echo "========================================="
echo "Task 1/3: Training Math (GSM8K) with CoT"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_math_cot.yaml \
    --use_xla True
echo "✓ Math training completed. Outputs in: out/math_cot/"
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
echo "Task 3/3: Training FinQA with CoT"
echo "========================================="
python -m sumcar.cli.train_task \
    --config configs/train_finqa_cot.yaml \
    --use_xla True
echo "✓ FinQA training completed. Outputs in: out/finqa_cot/"
echo ""

echo "=== All Training Completed ==="
echo ""
echo "Results summary:"
echo "  - Math (CoT):  out/math_cot/ (batch_size=4)"
echo "  - Code:        out/code/ (batch_size=8)"
echo "  - FinQA (CoT): out/finqa_cot/ (batch_size=2)"
echo ""
echo "Check training logs:"
echo "  - out/math_cot/math_cot/training.log"
echo "  - out/code/code/training.log"
echo "  - out/finqa_cot/finqa_cot/training.log"
echo ""
echo "Check meta.json files for loss history and TF-IDF statistics:"
echo "  - out/patch_math_cot_meta.json"
echo "  - out/patch_code_meta.json"
echo "  - out/patch_finqa_cot_meta.json"
