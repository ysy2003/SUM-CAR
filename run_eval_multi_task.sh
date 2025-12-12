#!/bin/bash
# Evaluate multi-task finetuned model

set -e

echo "========================================"
echo "Multi-Task Baseline Evaluation"
echo "========================================"
echo ""

# Check if model exists
if [ ! -d "noLoRA/multi_task/finetuned_model" ]; then
    echo "❌ Error: noLoRA/multi_task/finetuned_model directory not found"
    echo "   Please run training first: bash run_multi-task_finetune.sh"
    exit 1
fi

echo "✓ Finetuned model found at: noLoRA/multi_task/finetuned_model/"
echo ""

# Parse arguments (default to quick eval with 100 samples)
MAX_SAMPLES=100
OUTPUT="noLoRA/multi_task/eval/multi-task_results.json"
SAVE_INTERMEDIATE=false
USE_FP16=false  # Aligned with training (FP32)

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            MAX_SAMPLES=""
            OUTPUT="noLoRA/multi_task/eval/multi-task_results_full.json"
            SAVE_INTERMEDIATE=true
            shift
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --no_fp16)
            USE_FP16=false
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Model: noLoRA/multi_task/finetuned_model/"
if [ -z "$MAX_SAMPLES" ]; then
    echo "  Mode: Full evaluation (all test samples)"
    echo "  Checkpoints: Will save after 100 samples and at end"
else
    echo "  Mode: Quick evaluation"
    echo "  Max samples per task: $MAX_SAMPLES"
fi
echo "  Precision: $([ "$USE_FP16" = true ] && echo "FP16" || echo "FP32")"
echo "  Chain-of-Thought: ON (matches training)"
echo "  Output: $OUTPUT"
if [ "$SAVE_INTERMEDIATE" = true ]; then
    echo "  Checkpoint: ${OUTPUT%.json}_checkpoint_100.json"
fi
echo ""

# Create output directory
mkdir -p noLoRA/multi_task/eval

# Run evaluation (FP32 aligned with training)
if [ -z "$MAX_SAMPLES" ]; then
    echo "Running full evaluation with checkpoints (this may take 30-60 minutes)..."
    python baselines/eval_base_model.py \
        --base_model noLoRA/multi_task/finetuned_model \
        --out "$OUTPUT" \
        --use_cot \
        --save_intermediate
else
    echo "Running quick evaluation..."
    python baselines/eval_base_model.py \
        --base_model noLoRA/multi_task/finetuned_model \
        --out "$OUTPUT" \
        --max_samples $MAX_SAMPLES \
        --use_cot
fi

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT"
echo ""
echo "To run full evaluation (all test samples):"
echo "  bash run_eval_multi_task.sh --full"
