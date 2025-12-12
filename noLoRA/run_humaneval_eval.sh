#!/bin/bash
# Re-run HumanEval evaluations for baseline, math_only, and finance_only

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=============================================="
echo "       HumanEval Evaluation Pipeline"
echo "=============================================="
echo ""

# Parse arguments
MODE="full"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Mode: $MODE"
echo ""

# 1. Baseline
echo "=============================================="
echo "1. BASELINE on HumanEval"
echo "=============================================="
python noLoRA/eval_humaneval_baseline.py \
    --out noLoRA/eval_full/baseline_humaneval.json \
    --mode "$MODE"

echo ""

# 2. math_only
echo "=============================================="
echo "2. MATH_ONLY on HumanEval"
echo "=============================================="
python noLoRA/math_only/eval_humaneval_cross.py \
    --merged_dir noLoRA/math_only/acc_72%/merged \
    --out noLoRA/eval_full/math_only_humaneval.json \
    --mode "$MODE"

echo ""

# 3. finance_only
echo "=============================================="
echo "3. FINANCE_ONLY on HumanEval"
echo "=============================================="
python noLoRA/math_only/eval_humaneval_cross.py \
    --merged_dir noLoRA/finance_only/merged \
    --out noLoRA/eval_full/finance_only_humaneval.json \
    --mode "$MODE"

echo ""
echo "=============================================="
echo "              COMPLETE"
echo "=============================================="
echo "Results saved to noLoRA/eval_full/"
