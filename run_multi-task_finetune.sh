#!/bin/bash

set -e  # Exit on error

# Set PYTHONPATH so Python can find the sumcar module
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create output directory
mkdir -p out/multi_task

echo "=== Preparing Multi-Task Dataset ==="
# Generate multi-task dataset with 7:2:1 ratio (math:code:finance)
# Uses maximum available data while maintaining the ratio
# NOTE: This now includes answers in the 'tests' field (fixed)
python scripts/prepare_multi_task.py --out out/multi_task/multi_task_dataset.json --ratio 7:2:1
echo "✓ Dataset created: out/multi_task/multi_task_dataset.json"
echo ""

echo "=== Starting Multi-Task Fine-Tuning ==="
echo "Configuration:"
echo "  Model: meta-llama/Meta-Llama-3-8B-Instruct"
echo "  Output: multi-task/finetuned_model_CoT"
echo "  Epochs: 3"
echo "  Batch size: 4 (gradient accumulation: 2, effective: 8)"
echo "  Learning rate: 5e-5"
echo "  Mixed precision: FP16"
echo "  CUDA: Auto-detected"
echo "  CoT prompts: Yes (included in dataset)"
echo ""

# Run fine-tuning with memory-efficient settings
# batch_size=4 + gradient_accumulation_steps=2 gives effective batch size of 8
# fp16 reduces memory usage by ~50%
python scripts/multi-task_finetune.py \
    --data_file out/multi_task/multi_task_dataset.json \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir multi-task/finetuned_model_CoT \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr 5e-5 \
    --use_fp16

echo ""
echo "✓ Fine-tuning completed. Model saved to: multi-task/finetuned_model_CoT/"
echo ""
echo "Next steps:"
echo "  1. Evaluate: bash run_eval_multi_task.sh --full"
echo "  2. Compare with baseline: cat eval/multi-task_noCoT_full.json"