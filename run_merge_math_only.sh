#!/bin/bash
# Merge only Math patch for single-task evaluation

set -e
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== Merging Math-only Memory ==="

BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SLOTS=8192
K_TOP=8
ALPHA=1.0
OUT_DIR="out/math_only"

MATH_PATCH="out/math_cot/patch_gsm8k.json"

if [ ! -f "$MATH_PATCH" ]; then
    echo "Error: Math patch not found: $MATH_PATCH"
    exit 1
fi

echo "Creating math-only memory at $OUT_DIR"

python -m sumcar.cli.merge_patches \
    --base_model "$BASE_MODEL" \
    --patches "$MATH_PATCH" \
    --out "$OUT_DIR" \
    --num_slots $NUM_SLOTS \
    --k_top $K_TOP \
    --alpha $ALPHA \
    --use_tfidf_scoring True \
    --use_capacity_budgeting True \
    --verbose True

echo ""
echo "Math-only memory created: $OUT_DIR/memory.pt"
