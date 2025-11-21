#!/bin/bash

echo "========================================"
echo "Merged Model Quick Evaluation"
echo "========================================"
echo ""
echo "Testing merged memory model on 20 samples per task"
echo "to verify the code is working correctly"
echo ""

cd "$(dirname "$0")/.."

# Check if merged model exists
if [ ! -f "out/merged/memory.pt" ]; then
    echo "❌ Error: Merged model not found at out/merged/memory.pt"
    echo "   Please run merge first: bash run_merge.sh"
    exit 1
fi

echo "✓ Merged model found"
echo ""

# Run quick evaluation
python baselines/eval_merged_quick.py \
    --base_model gpt2 \
    --merged_dir out/merged \
    --out baselines/merged_model_results_quick.json \
    --k_top 4 \
    --alpha 1.0 \
    --max_samples 20

echo ""
echo "========================================"
echo "Quick Evaluation Complete!"
echo "========================================"
