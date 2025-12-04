#!/bin/bash
# Merge Math, Code, FinQA patches (FP16 training outputs)

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== SUM-CAR Patch Merging (FP16) ==="
echo "Tasks: Math (GSM8K), Code (CodeXGLUE), FinQA"
echo ""

# FP16 patch files
MATH_PATCH="out_fp16/patch_gsm8k.json"
CODE_PATCH="out_fp16/patch_codexglue.json"
FINQA_PATCH="out_fp16/patch_finqa.json"

echo "Checking patch files..."
missing=0
if [ ! -f "$MATH_PATCH" ]; then
    echo "  x Math patch not found: $MATH_PATCH"
    missing=1
else
    echo "  + Math patch found"
fi

if [ ! -f "$CODE_PATCH" ]; then
    echo "  x Code patch not found: $CODE_PATCH"
    missing=1
else
    echo "  + Code patch found"
fi

if [ ! -f "$FINQA_PATCH" ]; then
    echo "  x FinQA patch not found: $FINQA_PATCH"
    missing=1
else
    echo "  + FinQA patch found"
fi

if [ $missing -eq 1 ]; then
    echo ""
    echo "Error: Some patch files are missing. Please run training first:"
    echo "  bash run_finetune_all_tasks.sh"
    exit 1
fi

echo ""
echo "========================================="
echo "Merging patches with TF-IDF scoring..."
echo "========================================="

# Merge config
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SLOTS=8192
K_TOP=8
ALPHA=1.0
OUT_DIR="out_fp16/merged"

echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Initial slots: $NUM_SLOTS"
echo "  K-top: $K_TOP"
echo "  Alpha: $ALPHA"
echo "  Output: $OUT_DIR"
echo "  Precision: FP16"
echo ""

# Run merge
python -m sumcar.cli.merge_patches \
    --base_model "$BASE_MODEL" \
    --patches "$MATH_PATCH,$CODE_PATCH,$FINQA_PATCH" \
    --out "$OUT_DIR" \
    --num_slots $NUM_SLOTS \
    --k_top $K_TOP \
    --alpha $ALPHA \
    --use_tfidf_scoring True \
    --use_capacity_budgeting True \
    --verbose True \
    --use_fp16 True

echo ""
echo "========================================="
echo "Merge completed!"
echo "========================================="
echo ""
echo "Output files:"
echo "  - Merged memory: $OUT_DIR/memory.pt"
echo "  - Remap table: $OUT_DIR/remap.json"
echo "  - Metadata: $OUT_DIR/patch_meta.json"
