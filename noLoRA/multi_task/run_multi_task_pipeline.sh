#!/bin/bash
# Multi-task pipeline: fine-tune on multi-task data, then evaluate
# All outputs saved to noLoRA/multi_task/

set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Parse arguments
MAX_SAMPLES=""  # empty = use all samples
RATIO=""        # empty = no ratio limiting
EVAL_MODE="full"  # "full" or number of samples for eval
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --eval_mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Output directories
OUTPUT_DIR="noLoRA/multi_task"
DATASET_FILE="$OUTPUT_DIR/multi_task_dataset.json"
MODEL_DIR="$OUTPUT_DIR/finetuned_model"
EVAL_DIR="$OUTPUT_DIR/eval"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_DIR"

echo "=============================================="
echo "  Multi-Task Pipeline (Fine-tune + Eval)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Base model: Llama-3-8B-Instruct"
echo "  - Train samples: ${MAX_SAMPLES:-all (default)}"
echo "  - Ratio: ${RATIO:-none (use all data)}"
echo "  - Eval mode: $EVAL_MODE"
echo "  - Precision: FP16"
echo "  - Output dir: $OUTPUT_DIR"
echo ""

# =============================================
# Step 1: Prepare Dataset
# =============================================
echo "=============================================="
echo "Step 1: Preparing Multi-Task Dataset"
echo "=============================================="
echo ""

if [ -f "$DATASET_FILE" ]; then
    echo "+ Dataset already exists: $DATASET_FILE"
else
    # Build command with optional args
    PREPARE_CMD="python scripts/prepare_multi_task.py --out $DATASET_FILE"
    [ -n "$MAX_SAMPLES" ] && PREPARE_CMD="$PREPARE_CMD --max_samples $MAX_SAMPLES"
    [ -n "$RATIO" ] && PREPARE_CMD="$PREPARE_CMD --ratio $RATIO"
    eval $PREPARE_CMD
    echo ""
    echo "+ Dataset created: $DATASET_FILE"
fi
echo ""

# =============================================
# Step 2: Fine-tune
# =============================================
echo "=============================================="
echo "Step 2: Fine-Tuning"
echo "=============================================="
echo ""

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "+ Model already exists: $MODEL_DIR"
else
    echo "  Epochs: 1"
    echo "  Batch size: 1 (gradient accumulation: 8, effective: 8)"
    echo "  Learning rate: 5e-5"
    echo "  Max length: 1024"
    echo "  Precision: FP16"
    echo ""

    python scripts/multi-task_finetune.py \
        --data_file "$DATASET_FILE" \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --output_dir "$MODEL_DIR" \
        --epochs 1 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --lr 5e-5 \
        --max_length 1024 \
        --use_fp16

    echo ""
    echo "+ Model saved to: $MODEL_DIR"
fi
echo ""

# =============================================
# Step 3: Evaluate (using noLoRA baseline evals)
# =============================================
echo "=============================================="
echo "Step 3: Evaluating on All Tasks"
echo "=============================================="
echo ""

# GSM8K
echo "[1/3] Evaluating on GSM8K..."
python noLoRA/eval_gsm8k_baseline.py \
    --model_name "$MODEL_DIR" \
    --out "$EVAL_DIR/gsm8k_results.json" \
    --mode "$EVAL_MODE" \
    --use_fp16

# HumanEval
echo ""
echo "[2/3] Evaluating on HumanEval..."
python noLoRA/eval_humaneval_baseline.py \
    --model_name "$MODEL_DIR" \
    --out "$EVAL_DIR/humaneval_results.json" \
    --mode "$EVAL_MODE" \
    --use_fp16

# FinQA
echo ""
echo "[3/3] Evaluating on FinQA..."
python noLoRA/eval_finqa_baseline.py \
    --model_name "$MODEL_DIR" \
    --out "$EVAL_DIR/finqa_results.json" \
    --mode "$EVAL_MODE" \
    --use_fp16

echo ""
echo "+ Evaluation saved to: $EVAL_DIR/"
echo ""

# =============================================
# Summary
# =============================================
echo "=============================================="
echo "  Pipeline Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - Dataset: $DATASET_FILE"
echo "  - Model: $MODEL_DIR"
echo "  - GSM8K eval: $EVAL_DIR/gsm8k_results.json"
echo "  - HumanEval eval: $EVAL_DIR/humaneval_results.json"
echo "  - FinQA eval: $EVAL_DIR/finqa_results.json"
echo ""
