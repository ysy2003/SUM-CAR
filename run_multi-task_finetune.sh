#!/bin/bash

set -e  # Exit on error

# Set PYTHONPATH so Python can find the sumcar module
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create output directory
mkdir -p out/multi_task

echo "=== Preparing Multi-Task Dataset ==="
# Generate multi-task dataset with 7:2:1 ratio (math:code:finance)
# Uses maximum available data while maintaining the ratio
python scripts/prepare_multi_task.py --out out/multi_task/multi_task_dataset.json --ratio 7:2:1
echo "✓ Dataset created: out/multi_task/multi_task_dataset.json"
echo ""

echo "=== Starting Multi-Task Fine-Tuning ==="
# Run fine-tuning with memory-efficient settings
# batch_size=4 + gradient_accumulation_steps=2 gives effective batch size of 8
# fp16 reduces memory usage by ~50%
python scripts/multi-task_finetune.py \
    --data_file out/multi_task/multi_task_dataset.json \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir finetuned_model \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr 5e-5
echo ""
echo "✓ Fine-tuning completed. Model saved to: finetuned_model/"