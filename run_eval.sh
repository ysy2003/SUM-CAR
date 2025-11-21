#!/bin/bash

echo "=== SUM-CAR Evaluation ==="
echo "Tasks: GSM8K (Math), HumanEval (Code), FinQA"
echo ""

# Check if merged model exists
if [ ! -f "out/merged/memory.pt" ]; then
    echo "❌ Error: Merged model not found at out/merged/memory.pt"
    echo "   Please run merge first: bash run_merge.sh"
    exit 1
fi

echo "✓ Merged model found"
echo ""

echo "========================================="
echo "Evaluating merged model on three tasks..."
echo "========================================="
echo ""
echo "Note: This may take a while (10-30 minutes depending on hardware)"
echo "Output will be saved to out/eval_results.json"
echo "Log will be saved to out/eval.log"
echo ""

# Run evaluation with log
python -m sumcar.cli.eval_single \
    --base_model gpt2 \
    --merged out/merged \
    --out out/eval_results.json \
    --max_new_tokens 128 \
    --k_top 4 \
    --alpha 1.0 \
    2>&1 | tee out/eval.log

echo ""
echo "========================================="
echo "Evaluation completed!"
echo "========================================="
echo ""

# Display results if jq is available
if command -v jq &> /dev/null; then
    echo "Results:"
    echo "--------"
    jq '.' out/eval_results.json
else
    echo "Results saved to: out/eval_results.json"
    echo ""
    echo "Preview:"
    head -20 out/eval_results.json
fi
