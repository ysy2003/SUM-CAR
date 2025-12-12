#!/bin/bash
# Train FinQA task with memory-only (no LoRA, frozen base model)
# Then merge and evaluate on FinQA dev/test set

set -e
# Change to SUM-CAR root directory (parent of noLoRA)
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Parse arguments
EVAL_MODE="full"  # "100" for first 100 only, "full" for full dataset
MAX_EXAMPLES=""   # empty = use config value (null = full dataset)
EPOCHS=""         # empty = use config value
SPLIT="dev"       # FinQA split: dev or test
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        --max_examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================="
echo "FinQA Memory-Only Training Pipeline"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Base model: Llama-3-8B-Instruct (FROZEN)"
echo "  - LoRA: DISABLED"
echo "  - Only training: KV Memory layer"
echo "  - Task: FinQA (finance)"
echo "  - Precision: FP32"
echo "  - Eval mode: $EVAL_MODE"
echo "  - Eval split: $SPLIT"
echo "  - Max examples: ${MAX_EXAMPLES:-config (null=full)}"
echo "  - Epochs: ${EPOCHS:-config}"
echo ""

# Create output directories
mkdir -p noLoRA
mkdir -p noLoRA/eval

# =========================================
# Step 1: Train memory layer on FinQA task
# =========================================
echo "========================================="
echo "Step 1: Training Memory Layer (FinQA)"
echo "========================================="
echo ""

if [ ! -f "noLoRA/patch_finqa.json" ]; then
    echo "Training memory layer on FinQA..."
    TRAIN_CMD="python -m sumcar.cli.train_task --task finqa --config noLoRA/train_finqa_memory_only.yaml"
    [ -n "$MAX_EXAMPLES" ] && TRAIN_CMD="$TRAIN_CMD --max_examples $MAX_EXAMPLES"
    [ -n "$EPOCHS" ] && TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
    eval $TRAIN_CMD
    echo ""
    echo "+ Training complete. Patch saved to: noLoRA/patch_finqa.json"
else
    echo "+ Patch already exists: noLoRA/patch_finqa.json"
fi

echo ""

# =========================================
# Step 2: Merge KV memory (single task)
# =========================================
echo "========================================="
echo "Step 2: Merging KV Memory (FinQA)"
echo "========================================="
echo ""

MERGED_DIR="noLoRA/merged_finqa"

if [ ! -f "$MERGED_DIR/memory.pt" ]; then
    echo "Merging FinQA memory patch..."
    python -m sumcar.cli.merge_patches \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --patches noLoRA/patch_finqa.json \
        --out "$MERGED_DIR" \
        --num_slots 8192 \
        --k_top 64 \
        --alpha 1.0 \
        --use_tfidf_scoring True \
        --use_capacity_budgeting True \
        --use_fp16 False
    echo ""
    echo "+ Merged memory saved to: $MERGED_DIR/memory.pt"
else
    echo "+ Merged memory already exists: $MERGED_DIR/memory.pt"
fi

echo ""

# =========================================
# Step 3: Evaluate on FinQA dev/test set
# =========================================
echo "========================================="
echo "Step 3: Evaluating on FinQA"
echo "========================================="
echo ""

OUT_FILE="noLoRA/eval/finqa_${SPLIT}_results.json"

echo "Evaluating on FinQA ($SPLIT) set (mode: $EVAL_MODE)..."
python noLoRA/eval_finqa_only.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --merged_dir "$MERGED_DIR" \
    --out "$OUT_FILE" \
    --k_top 64 \
    --alpha 1.0 \
    --use_fp16 False \
    --use_cot True \
    --mode "$EVAL_MODE" \
    --memory_position middle \
    --split "$SPLIT"

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Patch: noLoRA/patch_finqa.json"
echo "  - Merged memory: $MERGED_DIR/memory.pt"
echo "  - Evaluation: $OUT_FILE"
