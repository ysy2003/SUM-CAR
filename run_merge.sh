#!/bin/bash
# 合并 Math, Code, FinQA 三个任务的 patches
# 使用 TF-IDF 驱动的冲突解决和容量配额策略

set -e

echo "=== SUM-CAR Patch Merging ==="
echo "Tasks: Math (GSM8K), Code (CodeXGLUE), FinQA"
echo ""

# 检查 patch 文件是否存在
MATH_PATCH="out/math/patch_gsm8k.json"
CODE_PATCH="out/code/patch_codexglue.json"
FINQA_PATCH="out/finqa/patch_finqa.json"

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
    echo "  ./run_all_tasks.sh"
    exit 1
fi

echo ""
echo "========================================="
echo "Merging patches with TF-IDF scoring..."
echo "========================================="

# 合并参数
BASE_MODEL="gpt2"
NUM_SLOTS=8192      # 初始槽位数（会根据需要自动扩展）
K_TOP=4             # 每次检索 top-k 个槽位
ALPHA=1.0           # 记忆贡献的缩放因子
OUT_DIR="out/merged"

echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Initial slots: $NUM_SLOTS"
echo "  K-top: $K_TOP"
echo "  Alpha: $ALPHA"
echo "  Output: $OUT_DIR"
echo ""

# 运行合并
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

# 显示合并结果
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
echo "Usage:"
echo "  Load the merged memory in your model and use remap.json"
echo "  to route queries to the correct task-specific slots."
