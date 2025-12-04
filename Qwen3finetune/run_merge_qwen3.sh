#!/bin/bash
# Merge Math, Code, FinQA patches from Qwen3-8B training
# Uses TF-IDF driven conflict resolution and capacity budgeting
# Outputs to out_qwen3/merged/ to avoid overwriting Llama merged outputs

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# Set PYTHONPATH so Python can find the sumcar module
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== SUM-CAR Patch Merging (Qwen3-8B) ==="
echo "Tasks: Math (GSM8K), Code (CodeXGLUE), FinQA"
echo ""

# Check patch files exist
MATH_PATCH="out_qwen3/math_cot/patch_gsm8k.json"
CODE_PATCH="out_qwen3/code/patch_codexglue.json"
FINQA_PATCH="out_qwen3/finqa_cot/patch_finqa.json"

echo "Checking patch files..."
missing=0
if [ ! -f "$MATH_PATCH" ]; then
    echo "  ✗ Math patch not found: $MATH_PATCH"
    missing=1
else
    echo "  ✓ Math patch found"
fi

if [ ! -f "$CODE_PATCH" ]; then
    echo "  ✗ Code patch not found: $CODE_PATCH"
    missing=1
else
    echo "  ✓ Code patch found"
fi

if [ ! -f "$FINQA_PATCH" ]; then
    echo "  ✗ FinQA patch not found: $FINQA_PATCH"
    missing=1
else
    echo "  ✓ FinQA patch found"
fi

if [ $missing -eq 1 ]; then
    echo ""
    echo "Error: Some patch files are missing. Please run training first:"
    echo "  bash Qwen3finetune/run_finetune_all_tasks_qwen3.sh"
    exit 1
fi

echo ""
echo "========================================="
echo "Merging patches with TF-IDF scoring..."
echo "========================================="

# Merge parameters
BASE_MODEL="Qwen/Qwen3-8B"
NUM_SLOTS=8192      # Initial slot count (auto-expands as needed)
K_TOP=8             # Retrieve top-k slots per query (match training config)
ALPHA=1.0           # Memory contribution scaling factor
OUT_DIR="out_qwen3/merged"

echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Initial slots: $NUM_SLOTS"
echo "  K-top: $K_TOP"
echo "  Alpha: $ALPHA"
echo "  Output: $OUT_DIR"
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
    --verbose True

echo ""
echo "========================================="
echo "Merge completed!"
echo "========================================="
echo ""

# Display merge results
if [ -f "$OUT_DIR/patch_meta.json" ]; then
    echo "Merged patch statistics:"
    python -c "
import json
with open('$OUT_DIR/patch_meta.json', 'r') as f:
    meta = json.load(f)
print(f'  Total slots: {meta[\"total_slots\"]}')
for task in ['math', 'code', 'finqa']:
    if task in meta:
        print(f'  {task.upper()}: {meta[task][\"n_slots\"]} slots')
    elif f'gsm8k' in meta:
        print(f'  MATH (GSM8K): {meta[\"gsm8k\"][\"n_slots\"]} slots')
    elif f'codexglue_refine' in meta:
        print(f'  CODE: {meta[\"codexglue_refine\"][\"n_slots\"]} slots')
    elif f'finqa_rc' in meta:
        print(f'  FINQA: {meta[\"finqa_rc\"][\"n_slots\"]} slots')
"
else
    echo "Warning: patch_meta.json not found"
fi

echo ""
echo "Output files:"
echo "  - Merged memory: $OUT_DIR/memory.pt"
echo "  - Remap table: $OUT_DIR/remap.json"
echo "  - Metadata: $OUT_DIR/patch_meta.json"
echo ""
echo "Next step: Evaluate the merged Qwen3 model"
echo "  bash scripts/run_merged_eval_qwen3.sh"
