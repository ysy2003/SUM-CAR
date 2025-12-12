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
echo "   FLAN-T5 Evaluation on 3 Tasks"
echo "=============================================="
echo "Model: google/flan-t5-large"
echo "Mode: $MODE"
echo ""

# Determine max_samples based on mode
if [ "$MODE" == "full" ]; then
    MAX_SAMPLES_ARG=""
    echo "Running full evaluation (all samples)"
elif [ "$MODE" == "10" ]; then
    MAX_SAMPLES_ARG="--max_samples 10"
    echo "Running quick test (10 samples per task)"
elif [ "$MODE" == "100" ]; then
    MAX_SAMPLES_ARG="--max_samples 100"
    echo "Running medium test (100 samples per task)"
else
    MAX_SAMPLES_ARG="--max_samples $MODE"
    echo "Running with $MODE samples per task"
fi

echo ""

# Run evaluation
python noLoRA/LoraHub_base/eval_flan_t5.py \
    --base_model google/flan-t5-large \
    --out noLoRA/LoraHub_base/flan_t5_results.json \
    --use_cot \
    --use_fp16 \
    $MAX_SAMPLES_ARG

echo ""
echo "=============================================="
echo "              COMPLETE"
echo "=============================================="
echo "Results saved to noLoRA/LoraHub_base/flan_t5_results.json"
