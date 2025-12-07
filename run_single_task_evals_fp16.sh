#!/bin/bash
# Evaluate single-task models (FP16) to compare with merged model
# This shows performance when fine-tuned on only ONE task
# Supports resuming from checkpoint_100 if it exists

set -e
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "========================================="
echo "Single-Task Evaluation (FP16)"
echo "========================================="
echo ""
echo "This evaluates models fine-tuned on only ONE task:"
echo "  - Math-only: fine-tuned only on GSM8K"
echo "  - Code-only: fine-tuned only on CodeXGLUE"
echo "  - FinQA-only: fine-tuned only on FinQA"
echo ""

# Create output directory
mkdir -p eval/single_task/sumcar

# Parse arguments
MAX_SAMPLES=99999
CHECKPOINT_AT=100
USE_COT="--use_cot"
EVAL_MODE="full"  # "100" for first 100 only, "full" for full dataset
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --checkpoint_at)
            CHECKPOINT_AT="$2"
            shift 2
            ;;
        --no_cot)
            USE_COT=""
            shift
            ;;
        --mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Eval mode: $EVAL_MODE (use --mode 100 for first 100 only)"
echo "  Checkpoint at: $CHECKPOINT_AT"
echo "  Use CoT: $([ -n "$USE_COT" ] && echo 'Yes' || echo 'No')"
echo "  Output dir: eval/single_task/sumcar/"
echo ""

# =========================================
# Step 1: Create single-task memory models
# =========================================
echo "========================================="
echo "Step 1: Creating Single-Task Memory Models"
echo "========================================="
echo ""

BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SLOTS=8192
K_TOP=8
ALPHA=1.0

# Math-only
MATH_PATCH="out_fp16/patch_gsm8k.json"
MATH_OUT="out_fp16/math_only"
if [ ! -f "$MATH_OUT/memory.pt" ]; then
    if [ -f "$MATH_PATCH" ]; then
        echo "Creating math-only memory..."
        python -m sumcar.cli.merge_patches \
            --base_model "$BASE_MODEL" \
            --patches "$MATH_PATCH" \
            --out "$MATH_OUT" \
            --num_slots $NUM_SLOTS \
            --k_top $K_TOP \
            --alpha $ALPHA \
            --use_tfidf_scoring True \
            --use_capacity_budgeting True \
            --use_fp16 True
        echo "+ Math-only memory created"
    else
        echo "x Math patch not found: $MATH_PATCH"
    fi
else
    echo "+ Math-only memory exists"
fi

# Code-only
CODE_PATCH="out_fp16/patch_codexglue.json"
CODE_OUT="out_fp16/code_only"
if [ ! -f "$CODE_OUT/memory.pt" ]; then
    if [ -f "$CODE_PATCH" ]; then
        echo "Creating code-only memory..."
        python -m sumcar.cli.merge_patches \
            --base_model "$BASE_MODEL" \
            --patches "$CODE_PATCH" \
            --out "$CODE_OUT" \
            --num_slots $NUM_SLOTS \
            --k_top $K_TOP \
            --alpha $ALPHA \
            --use_tfidf_scoring True \
            --use_capacity_budgeting True \
            --use_fp16 True
        echo "+ Code-only memory created"
    else
        echo "x Code patch not found: $CODE_PATCH"
    fi
else
    echo "+ Code-only memory exists"
fi

# FinQA-only
FINQA_PATCH="out_fp16/patch_finqa.json"
FINQA_OUT="out_fp16/finqa_only"
if [ ! -f "$FINQA_OUT/memory.pt" ]; then
    if [ -f "$FINQA_PATCH" ]; then
        echo "Creating finqa-only memory..."
        python -m sumcar.cli.merge_patches \
            --base_model "$BASE_MODEL" \
            --patches "$FINQA_PATCH" \
            --out "$FINQA_OUT" \
            --num_slots $NUM_SLOTS \
            --k_top $K_TOP \
            --alpha $ALPHA \
            --use_tfidf_scoring True \
            --use_capacity_budgeting True \
            --use_fp16 True
        echo "+ FinQA-only memory created"
    else
        echo "x FinQA patch not found: $FINQA_PATCH"
    fi
else
    echo "+ FinQA-only memory exists"
fi

echo ""

# =========================================
# Step 2: Evaluate single-task models
# =========================================
echo "========================================="
echo "Step 2: Evaluating Single-Task Models"
echo "========================================="
echo ""

# Function to run evaluation with checkpoint/resume support
run_eval() {
    local MODEL_NAME=$1
    local MERGED_DIR=$2
    local OUT_FILE=$3

    local CHECKPOINT_FILE="${OUT_FILE%.json}_checkpoint_${CHECKPOINT_AT}.json"

    if [ "$EVAL_MODE" = "100" ]; then
        # Only run first 100 samples
        echo "  Running first $CHECKPOINT_AT samples only..."
        python scripts/eval_merged.py \
            --base_model "$BASE_MODEL" \
            --merged_dir "$MERGED_DIR" \
            --out "$CHECKPOINT_FILE" \
            --k_top $K_TOP \
            --alpha $ALPHA \
            --max_samples "$CHECKPOINT_AT" \
            --use_fp16 True \
            $USE_COT
        echo "  + Saved to: $CHECKPOINT_FILE"
    else
        # Full evaluation
        if [ -f "$CHECKPOINT_FILE" ]; then
            # Checkpoint exists - resume from checkpoint_at+1
            echo "  + Found checkpoint: $CHECKPOINT_FILE"
            echo "  Resuming from sample $((CHECKPOINT_AT + 1))..."
            python scripts/eval_merged.py \
                --base_model "$BASE_MODEL" \
                --merged_dir "$MERGED_DIR" \
                --out "$OUT_FILE" \
                --k_top $K_TOP \
                --alpha $ALPHA \
                --max_samples 99999 \
                --use_fp16 True \
                --checkpoint_at "$CHECKPOINT_AT" \
                --resume True \
                $USE_COT
        else
            # No checkpoint - run full with checkpoint saving
            echo "  No checkpoint found, running full evaluation..."
            echo "  Will save checkpoint at $CHECKPOINT_AT samples"
            python scripts/eval_merged.py \
                --base_model "$BASE_MODEL" \
                --merged_dir "$MERGED_DIR" \
                --out "$OUT_FILE" \
                --k_top $K_TOP \
                --alpha $ALPHA \
                --max_samples 99999 \
                --use_fp16 True \
                --checkpoint_at "$CHECKPOINT_AT" \
                $USE_COT
        fi
        echo "  + Saved to: $OUT_FILE"
    fi
}

# Evaluate math-only
if [ -f "$MATH_OUT/memory.pt" ]; then
    echo "[1/3] Evaluating math-only model on all tasks..."
    if [ -n "$USE_COT" ]; then
        OUT_FILE="eval/single_task/sumcar/math_only_results_cot.json"
    else
        OUT_FILE="eval/single_task/sumcar/math_only_results.json"
    fi
    run_eval "math_only" "$MATH_OUT" "$OUT_FILE"
    echo ""
else
    echo "[1/3] Skipping math-only (model not found)"
fi

# Evaluate code-only
if [ -f "$CODE_OUT/memory.pt" ]; then
    echo "[2/3] Evaluating code-only model on all tasks..."
    if [ -n "$USE_COT" ]; then
        OUT_FILE="eval/single_task/sumcar/code_only_results_cot.json"
    else
        OUT_FILE="eval/single_task/sumcar/code_only_results.json"
    fi
    run_eval "code_only" "$CODE_OUT" "$OUT_FILE"
    echo ""
else
    echo "[2/3] Skipping code-only (model not found)"
fi

# Evaluate finqa-only
if [ -f "$FINQA_OUT/memory.pt" ]; then
    echo "[3/3] Evaluating finqa-only model on all tasks..."
    if [ -n "$USE_COT" ]; then
        OUT_FILE="eval/single_task/sumcar/finqa_only_results_cot.json"
    else
        OUT_FILE="eval/single_task/sumcar/finqa_only_results.json"
    fi
    run_eval "finqa_only" "$FINQA_OUT" "$OUT_FILE"
    echo ""
else
    echo "[3/3] Skipping finqa-only (model not found)"
fi

echo ""
echo "========================================="
echo "Single-Task Evaluations Complete!"
echo "========================================="
echo ""
echo "Results saved in eval/single_task/sumcar/:"
if [ "$EVAL_MODE" = "100" ]; then
    echo "  - math_only_results_cot_checkpoint_100.json"
    echo "  - code_only_results_cot_checkpoint_100.json"
    echo "  - finqa_only_results_cot_checkpoint_100.json"
else
    echo "  - math_only_results_cot.json (full)"
    echo "  - code_only_results_cot.json (full)"
    echo "  - finqa_only_results_cot.json (full)"
fi
echo ""
echo "Compare with:"
echo "  - eval/sumcar/full_results_cot.json (merged model)"
