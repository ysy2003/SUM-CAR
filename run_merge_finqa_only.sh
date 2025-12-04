#!/bin/bash
# Merge only FinQA patch for single-task evaluation

set -e
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== Merging FinQA-only Memory ==="

BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SLOTS=8192
K_TOP=8
ALPHA=1.0
OUT_DIR="out/finqa_only"

FINQA_PATCH="out/finqa_cot/patch_finqa.json"

if [ ! -f "$FINQA_PATCH" ]; then
    echo "Error: FinQA patch not found: $FINQA_PATCH"
    exit 1
fi

echo "Creating finqa-only memory at $OUT_DIR"

python -m sumcar.cli.merge_patches \
    --base_model "$BASE_MODEL" \
    --patches "$FINQA_PATCH" \
    --out "$OUT_DIR" \
    --num_slots $NUM_SLOTS \
    --k_top $K_TOP \
    --alpha $ALPHA \
    --use_tfidf_scoring True \
    --use_capacity_budgeting True \
    --verbose True \
    --use_fp16 True

echo ""
echo "FinQA-only memory created: $OUT_DIR/memory.pt"
