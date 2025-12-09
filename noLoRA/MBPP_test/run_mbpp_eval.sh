#!/bin/bash
# Evaluate baseline and code_only on MBPP test set

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

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

echo "=============================================="
echo "       MBPP Test Set Evaluation"
echo "=============================================="
echo "Mode: $MODE"
echo "Test set: MBPP sanitized test (257 samples)"
echo ""

# 1. Baseline
echo "=============================================="
echo "1. BASELINE on MBPP"
echo "=============================================="
python noLoRA/MBPP_test/eval_mbpp_baseline.py \
    --out noLoRA/MBPP_test/baseline_mbpp.json \
    --mode "$MODE"

echo ""

# 2. code_only
echo "=============================================="
echo "2. CODE_ONLY on MBPP"
echo "=============================================="

CODE_MERGED="noLoRA/code_only/merged"

if [ ! -f "$CODE_MERGED/memory.pt" ]; then
    echo "ERROR: code_only model not found at $CODE_MERGED/memory.pt"
    echo "Skipping code_only evaluation"
else
    python noLoRA/MBPP_test/eval_mbpp_code_only.py \
        --merged_dir "$CODE_MERGED" \
        --out noLoRA/MBPP_test/code_only_mbpp.json \
        --mode "$MODE"
fi

echo ""
echo "=============================================="
echo "              COMPLETE"
echo "=============================================="
echo "Results saved to noLoRA/MBPP_test/"
