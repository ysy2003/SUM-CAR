#!/bin/bash
# Merge only Code patch for single-task evaluation

set -e
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== Merging Code-only Memory ==="

BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SLOTS=8192
K_TOP=8
ALPHA=1.0
OUT_DIR="out/code_only"

CODE_PATCH="out/code/patch_codexglue.json"

if [ ! -f "$CODE_PATCH" ]; then
    echo "Error: Code patch not found: $CODE_PATCH"
    exit 1
fi

echo "Creating code-only memory at $OUT_DIR"

python -m sumcar.cli.merge_patches \
    --base_model "$BASE_MODEL" \
    --patches "$CODE_PATCH" \
    --out "$OUT_DIR" \
    --num_slots $NUM_SLOTS \
    --k_top $K_TOP \
    --alpha $ALPHA \
    --use_tfidf_scoring True \
    --use_capacity_budgeting True \
    --verbose True

echo ""
echo "Code-only memory created: $OUT_DIR/memory.pt"
