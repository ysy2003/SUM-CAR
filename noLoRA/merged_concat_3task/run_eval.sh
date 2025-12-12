#!/bin/bash
# Evaluate merged_concat_3task on all task test sets

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Parse arguments
EVAL_MODE="full"  # "full" or number of samples
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

MERGED_DIR="noLoRA/merged_concat_3task"
EVAL_DIR="$MERGED_DIR/eval"

mkdir -p "$EVAL_DIR"

echo "=============================================="
echo "  Evaluating merged_concat_3task"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Merged dir: $MERGED_DIR"
echo "  - Eval mode: $EVAL_MODE"
echo "  - Output dir: $EVAL_DIR"
echo ""

# GSM8K
echo "=============================================="
echo "[1/3] Evaluating on GSM8K..."
echo "=============================================="
python noLoRA/math_only/eval_math_only.py \
    --merged_dir "$MERGED_DIR" \
    --out "$EVAL_DIR/gsm8k_results.json" \
    --mode "$EVAL_MODE"

# HumanEval
echo ""
echo "=============================================="
echo "[2/3] Evaluating on HumanEval..."
echo "=============================================="
python noLoRA/code_only/eval_humaneval.py \
    --merged_dir "$MERGED_DIR" \
    --out "$EVAL_DIR/humaneval_results.json" \
    --mode "$EVAL_MODE"

# FinQA
echo ""
echo "=============================================="
echo "[3/3] Evaluating on FinQA..."
echo "=============================================="
python noLoRA/math_only/eval_finqa_cross.py \
    --merged_dir "$MERGED_DIR" \
    --out "$EVAL_DIR/finqa_results.json" \
    --mode "$EVAL_MODE"

echo ""
echo "=============================================="
echo "  Evaluation Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - GSM8K: $EVAL_DIR/gsm8k_results.json"
echo "  - HumanEval: $EVAL_DIR/humaneval_results.json"
echo "  - FinQA: $EVAL_DIR/finqa_results.json"
