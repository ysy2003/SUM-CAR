#!/bin/bash

# Composite FinQA evaluation with Llama-3-8B-Instruct baseline

set -e
cd "$(dirname "$0")"

# Model
MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

# Output paths
GROUND_TRUTH_FILE="noLoRA/composite_eval/finqa_composite_dev_cleaned.jsonl"
GENERATIONS_FILE="noLoRA/composite_eval/llama3_generations.jsonl"

# Parse arguments
MODE="full"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "=============================================="
echo "  Composite FinQA Evaluation"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Mode: $MODE"
echo "Ground truth: $GROUND_TRUTH_FILE"
echo "Generations: $GENERATIONS_FILE"
echo ""

python scripts/run_composite_eval.py \
    --model_name_or_path "$MODEL_PATH" \
    --ground_truth_file "$GROUND_TRUTH_FILE" \
    --generations_file "$GENERATIONS_FILE" \
    --mode "$MODE"