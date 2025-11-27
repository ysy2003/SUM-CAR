#!/bin/bash
# Run all single-task evaluations for comparison
# This creates a comprehensive comparison table

set -e

echo "========================================="
echo "SUM-CAR Single-Task Memory Evaluation"
echo "========================================="
echo ""
echo "This will:"
echo "  1. Merge individual patches to memory.pt"
echo "  2. Evaluate each on all 3 tasks (100 samples)"
echo "  3. Compare with baseline and merged results"
echo ""

# Parse arguments
MAX_SAMPLES=100
USE_COT="--use_cot"
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --no_cot)
            USE_COT=""
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Configuration:"
echo "  Max samples: $MAX_SAMPLES"
echo "  Use CoT: $([ -n "$USE_COT" ] && echo 'Yes' || echo 'No')"
echo ""

# Step 1: Create single-task memory models
echo "========================================="
echo "Step 1: Creating Single-Task Models"
echo "========================================="
echo ""

if [ ! -f "out/math_only/memory.pt" ]; then
    echo "Creating math-only memory..."
    bash run_merge_math_only.sh
else
    echo "✓ Math-only memory exists"
fi

if [ ! -f "out/code_only/memory.pt" ]; then
    echo "Creating code-only memory..."
    bash run_merge_code_only.sh
else
    echo "✓ Code-only memory exists"
fi

if [ ! -f "out/finqa_only/memory.pt" ]; then
    echo "Creating finqa-only memory..."
    bash run_merge_finqa_only.sh
else
    echo "✓ FinQA-only memory exists"
fi

echo ""
echo "========================================="
echo "Step 2: Evaluating Single-Task Models"
echo "========================================="
echo ""

# Evaluate math-only
echo "[1/3] Evaluating math-only memory on all tasks..."
bash scripts/run_eval_math_only.sh $USE_COT --max_samples $MAX_SAMPLES

echo ""
echo "[2/3] Evaluating code-only memory on all tasks..."
bash scripts/run_eval_code_only.sh $USE_COT --max_samples $MAX_SAMPLES

echo ""
echo "[3/3] Evaluating finqa-only memory on all tasks..."
bash scripts/run_eval_finqa_only.sh $USE_COT --max_samples $MAX_SAMPLES

echo ""
echo "========================================="
echo "All Evaluations Complete!"
echo "========================================="
echo ""
echo "Results saved in baselines/:"
echo "  - math_only_results_cot.json"
echo "  - code_only_results_cot.json"
echo "  - finqa_only_results_cot.json"
echo ""
echo "Compare with:"
echo "  - baselines/base_llama3_instruct_cot.json (baseline)"
echo "  - baselines/merged_results_cot.json (merged model)"
