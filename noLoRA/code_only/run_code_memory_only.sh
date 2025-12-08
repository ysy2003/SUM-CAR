#!/bin/bash
# Train code task with memory-only (no LoRA, frozen base model)
# Train on MBPP, evaluate on HumanEval with pass@1

set -e
# Change to SUM-CAR root directory (parent of noLoRA/code_only)
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Parse arguments
EVAL_MODE="full"  # "full" for all 164 HumanEval problems, or a number
MAX_EXAMPLES=""   # empty = use config value (null = full dataset)
EPOCHS=""         # empty = use config value
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
        *)
            shift
            ;;
    esac
done

echo "========================================="
echo "Code Memory-Only Training Pipeline"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Base model: Llama-3-8B-Instruct (FROZEN)"
echo "  - LoRA: DISABLED"
echo "  - Only training: KV Memory layer"
echo "  - Train task: MBPP (~400 samples)"
echo "  - Eval task: HumanEval (164 problems)"
echo "  - Metric: pass@1"
echo "  - Precision: FP32"
echo "  - Eval mode: $EVAL_MODE"
echo "  - Epochs: ${EPOCHS:-config}"
echo ""

# Create output directories
mkdir -p noLoRA/code_only
mkdir -p noLoRA/code_only/eval

# =========================================
# Step 1: Train memory layer on MBPP
# =========================================
echo "========================================="
echo "Step 1: Training Memory Layer on MBPP"
echo "========================================="
echo ""

if [ ! -f "noLoRA/code_only/patch_mbpp.json" ]; then
    echo "Training memory layer on MBPP..."
    TRAIN_CMD="python -m sumcar.cli.train_task --task mbpp --config noLoRA/code_only/train_code_memory_only.yaml"
    [ -n "$MAX_EXAMPLES" ] && TRAIN_CMD="$TRAIN_CMD --max_examples $MAX_EXAMPLES"
    [ -n "$EPOCHS" ] && TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
    eval $TRAIN_CMD
    echo ""
    echo "+ Training complete. Patch saved to: noLoRA/code_only/patch_mbpp.json"
else
    echo "+ Patch already exists: noLoRA/code_only/patch_mbpp.json"
fi

echo ""

# =========================================
# Step 2: Merge KV memory (single task)
# =========================================
echo "========================================="
echo "Step 2: Merging KV Memory"
echo "========================================="
echo ""

MERGED_DIR="noLoRA/code_only/merged"

if [ ! -f "$MERGED_DIR/memory.pt" ]; then
    echo "Merging memory patch..."
    python -m sumcar.cli.merge_patches \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --patches noLoRA/code_only/patch_mbpp.json \
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
# Step 3: Evaluate on HumanEval (pass@1)
# =========================================
echo "========================================="
echo "Step 3: Evaluating on HumanEval (pass@1)"
echo "========================================="
echo ""

OUT_FILE="noLoRA/code_only/eval/humaneval_results.json"

echo "Evaluating on HumanEval ($EVAL_MODE)..."
python noLoRA/code_only/eval_humaneval.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --merged_dir "$MERGED_DIR" \
    --out "$OUT_FILE" \
    --k_top 64 \
    --alpha 1.0 \
    --use_fp16 False \
    --mode "$EVAL_MODE" \
    --memory_position middle

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Patch: noLoRA/code_only/patch_mbpp.json"
echo "  - Merged memory: $MERGED_DIR/memory.pt"
echo "  - Evaluation: $OUT_FILE"
