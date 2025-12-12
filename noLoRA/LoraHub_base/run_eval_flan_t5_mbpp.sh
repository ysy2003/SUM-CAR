#!/bin/bash

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
echo "   FLAN-T5 MBPP Test Set Evaluation"
echo "=============================================="
echo "Model: google/flan-t5-large"
echo "Mode: $MODE"
echo "Test set: MBPP sanitized test (257 samples)"
echo ""

# Run evaluation
python noLoRA/LoraHub_base/eval_flan_t5_mbpp.py \
    --model_name google/flan-t5-large \
    --out noLoRA/LoraHub_base/flan_t5_mbpp_results.json \
    --mode "$MODE" \
    --use_fp16

echo ""
echo "=============================================="
echo "              COMPLETE"
echo "=============================================="
echo "Results saved to noLoRA/LoraHub_base/flan_t5_mbpp_results.json"
