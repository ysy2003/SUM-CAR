#!/bin/bash
# Full evaluation pipeline:
# 1. Baseline (Llama-3-8B-Instruct) on 3 tasks
# 2. code_only finetuned model on 3 tasks
# 3. math_only finetuned model on 3 tasks
# 4. finance_only finetuned model on 3 tasks

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Parse arguments
MODE="full"  # "full" or number of samples
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "=============================================="
echo "       Full Evaluation Pipeline"
echo "=============================================="
echo ""
echo "Mode: $MODE"
echo "Tasks: GSM8K (math), FinQA (finance), HumanEval (code)"
echo "Models: Baseline, code_only, math_only, finance_only"
echo ""

# Create output directory
mkdir -p noLoRA/eval_full

# =============================================
# 1. BASELINE (Llama-3-8B-Instruct, no memory)
# =============================================
echo "=============================================="
echo "1. BASELINE (Llama-3-8B-Instruct)"
echo "=============================================="
echo ""

echo "[1/3] Baseline on GSM8K..."
python noLoRA/eval_gsm8k_baseline.py \
    --out noLoRA/eval_full/baseline_gsm8k.json \
    --mode "$MODE"

echo ""
echo "[2/3] Baseline on FinQA..."
python noLoRA/eval_finqa_baseline.py \
    --out noLoRA/eval_full/baseline_finqa.json \
    --mode "$MODE"

echo ""
echo "[3/3] Baseline on HumanEval..."
python noLoRA/eval_humaneval_baseline.py \
    --out noLoRA/eval_full/baseline_humaneval.json \
    --mode "$MODE"

echo ""
echo "+ Baseline evaluation complete!"
echo ""

# =============================================
# 2. CODE_ONLY (MBPP-trained memory)
# =============================================
echo "=============================================="
echo "2. CODE_ONLY (MBPP-trained memory)"
echo "=============================================="
echo ""

CODE_MERGED="noLoRA/code_only/merged"

if [ ! -f "$CODE_MERGED/memory.pt" ]; then
    echo "ERROR: code_only model not found at $CODE_MERGED/memory.pt"
    echo "Run: bash noLoRA/code_only/run_code_memory_only.sh"
    exit 1
fi

echo "[1/3] code_only on GSM8K..."
python noLoRA/code_only/eval_gsm8k_cross.py \
    --merged_dir "$CODE_MERGED" \
    --out noLoRA/eval_full/code_only_gsm8k.json \
    --mode "$MODE"

echo ""
echo "[2/3] code_only on FinQA..."
python noLoRA/code_only/eval_finqa_cross.py \
    --merged_dir "$CODE_MERGED" \
    --out noLoRA/eval_full/code_only_finqa.json \
    --mode "$MODE"

echo ""
echo "[3/3] code_only on HumanEval..."
python noLoRA/code_only/eval_humaneval.py \
    --merged_dir "$CODE_MERGED" \
    --out noLoRA/eval_full/code_only_humaneval.json \
    --mode "$MODE"

echo ""
echo "+ code_only evaluation complete!"
echo ""

# =============================================
# 3. MATH_ONLY (GSM8K-trained memory)
# =============================================
echo "=============================================="
echo "3. MATH_ONLY (GSM8K-trained memory)"
echo "=============================================="
echo ""

MATH_MERGED="noLoRA/math_only/acc_72%/merged"

if [ ! -f "$MATH_MERGED/memory.pt" ]; then
    echo "ERROR: math_only model not found at $MATH_MERGED/memory.pt"
    exit 1
fi

echo "[1/3] math_only on GSM8K..."
python noLoRA/math_only/eval_math_only.py \
    --merged_dir "$MATH_MERGED" \
    --out noLoRA/eval_full/math_only_gsm8k.json \
    --mode "$MODE"

echo ""
echo "[2/3] math_only on FinQA..."
python noLoRA/math_only/eval_finqa_cross.py \
    --merged_dir "$MATH_MERGED" \
    --out noLoRA/eval_full/math_only_finqa.json \
    --mode "$MODE"

echo ""
echo "[3/3] math_only on HumanEval..."
python noLoRA/math_only/eval_humaneval_cross.py \
    --merged_dir "$MATH_MERGED" \
    --out noLoRA/eval_full/math_only_humaneval.json \
    --mode "$MODE"

echo ""
echo "+ math_only evaluation complete!"
echo ""

# =============================================
# 4. FINANCE_ONLY (FinQA-trained memory)
# =============================================
echo "=============================================="
echo "4. FINANCE_ONLY (FinQA-trained memory)"
echo "=============================================="
echo ""

FINANCE_MERGED="noLoRA/finance_only/merged"

if [ ! -f "$FINANCE_MERGED/memory.pt" ]; then
    echo "ERROR: finance_only model not found at $FINANCE_MERGED/memory.pt"
    exit 1
fi

echo "[1/3] finance_only on GSM8K..."
python noLoRA/code_only/eval_gsm8k_cross.py \
    --merged_dir "$FINANCE_MERGED" \
    --out noLoRA/eval_full/finance_only_gsm8k.json \
    --mode "$MODE"

echo ""
echo "[2/3] finance_only on FinQA..."
python noLoRA/math_only/eval_finqa_cross.py \
    --merged_dir "$FINANCE_MERGED" \
    --out noLoRA/eval_full/finance_only_finqa.json \
    --mode "$MODE"

echo ""
echo "[3/3] finance_only on HumanEval..."
python noLoRA/code_only/eval_humaneval.py \
    --merged_dir "$FINANCE_MERGED" \
    --out noLoRA/eval_full/finance_only_humaneval.json \
    --mode "$MODE"

echo ""
echo "+ finance_only evaluation complete!"
echo ""

# =============================================
# SUMMARY
# =============================================
echo "=============================================="
echo "              EVALUATION COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to noLoRA/eval_full/:"
echo ""
echo "Baseline:"
echo "  - baseline_gsm8k.json"
echo "  - baseline_finqa.json"
echo "  - baseline_humaneval.json"
echo ""
echo "code_only:"
echo "  - code_only_gsm8k.json"
echo "  - code_only_finqa.json"
echo "  - code_only_humaneval.json"
echo ""
echo "math_only:"
echo "  - math_only_gsm8k.json"
echo "  - math_only_finqa.json"
echo "  - math_only_humaneval.json"
echo ""
echo "finance_only:"
echo "  - finance_only_gsm8k.json"
echo "  - finance_only_finqa.json"
echo "  - finance_only_humaneval.json"
