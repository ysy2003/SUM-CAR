#!/bin/bash
# Evaluate multi-task finetuned model

set -e

echo "========================================"
echo "Multi-Task Baseline Evaluation"
echo "========================================"
echo ""

# Check if model exists
if [ ! -d "multi-task/finetuned_model_CoT" ]; then
    echo "❌ Error: multi-task/finetuned_model_CoT directory not found"
    echo "   Please run training first: bash run_multi-task_finetune.sh"
    exit 1
fi

echo "✓ Finetuned model found at: multi-task/finetuned_model_CoT/"
echo ""

# Parse arguments (default to quick eval with 100 samples)
MAX_SAMPLES=100
OUTPUT="eval/multi-task_results.json"
SAVE_INTERMEDIATE=false
USE_FP16=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            MAX_SAMPLES=""
            OUTPUT="eval/multi-task_results_full.json"
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
echo "  Model: multi-task/finetuned_model_CoT/"
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

# Run evaluation
if [ -z "$MAX_SAMPLES" ]; then
    echo "Running full evaluation with checkpoints (this may take 30-60 minutes)..."
    if [ "$USE_FP16" = true ]; then
        python baselines/eval_base_model.py \
            --base_model multi-task/finetuned_model_CoT \
            --out "$OUTPUT" \
            --use_fp16 \
            --use_cot \
            --save_intermediate
    else
        python baselines/eval_base_model.py \
            --base_model multi-task/finetuned_model_CoT \
            --out "$OUTPUT" \
            --use_cot \
            --save_intermediate
    fi
else
    echo "Running quick evaluation..."
    if [ "$USE_FP16" = true ]; then
        python baselines/eval_base_model.py \
            --base_model multi-task/finetuned_model_CoT \
            --out "$OUTPUT" \
            --max_samples $MAX_SAMPLES \
            --use_fp16 \
            --use_cot
    else
        python baselines/eval_base_model.py \
            --base_model multi-task/finetuned_model_CoT \
            --out "$OUTPUT" \
            --max_samples $MAX_SAMPLES \
            --use_cot
    fi
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
