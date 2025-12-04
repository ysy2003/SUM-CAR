#!/bin/bash
# Evaluate merged FP16 model on all tasks with checkpointing

set -e

echo "========================================="
echo "Merged Model Evaluation (FP16)"
echo "========================================="
echo ""

cd "$(dirname "$0")"

# Create eval output directory
mkdir -p eval/sumcar

# Default settings
USE_COT="--use_cot"
MERGED_DIR="out_fp16/merged"
CHECKPOINT_AT=100

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no_cot)
            USE_COT=""
            shift
            ;;
        --merged_dir)
            MERGED_DIR="$2"
            shift 2
            ;;
        --checkpoint_at)
            CHECKPOINT_AT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Merged dir: $MERGED_DIR"
echo "  Checkpoint at: $CHECKPOINT_AT samples per task"
echo "  Use CoT: $([ -n "$USE_COT" ] && echo 'Yes' || echo 'No')"
echo "  Precision: FP16"
echo "  Output dir: eval/sumcar/"
echo ""

# Check if merged model exists
if [ ! -f "$MERGED_DIR/memory.pt" ]; then
    echo "Error: Merged model not found at $MERGED_DIR/memory.pt"
    echo "Please run merge first: bash run_merge_fp16.sh"
    exit 1
fi

echo "+ Merged model found at $MERGED_DIR"
echo ""

# Determine output file name
if [ -n "$USE_COT" ]; then
    OUT_FILE="eval/sumcar/full_results_cot.json"
else
    OUT_FILE="eval/sumcar/full_results.json"
fi

echo "Running full evaluation with checkpoint at $CHECKPOINT_AT samples..."
echo "  - Checkpoint will be saved to: ${OUT_FILE%.json}_checkpoint_${CHECKPOINT_AT}.json"
echo "  - Final results will be saved to: $OUT_FILE"
echo ""

python scripts/eval_merged.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --merged_dir "$MERGED_DIR" \
    --out "$OUT_FILE" \
    --k_top 8 \
    --alpha 1.0 \
    --max_samples 99999 \
    --use_fp16 True \
    --checkpoint_at "$CHECKPOINT_AT" \
    $USE_COT

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Results saved to:"
echo "  - Checkpoint ($CHECKPOINT_AT samples): ${OUT_FILE%.json}_checkpoint_${CHECKPOINT_AT}.json"
echo "  - Full results: $OUT_FILE"
