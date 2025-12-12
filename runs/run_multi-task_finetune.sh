#!/bin/bash

# Multi-task fine-tuning: train on math + code + finance data

set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Parse arguments
MODE="full"
RATIO="7:2:1"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Output directories
OUTPUT_DIR="noLoRA/multi_task"
DATASET_FILE="$OUTPUT_DIR/multi_task_dataset.json"
MODEL_DIR="$OUTPUT_DIR/finetuned_model"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Multi-Task Fine-Tuning"
echo "=============================================="
echo "Mode: $MODE"
echo "Ratio (math:code:finance): $RATIO"
echo "Output: $MODEL_DIR"
echo ""

# Prepare dataset
echo "[1/2] Preparing Multi-Task Dataset..."
python scripts/prepare_multi_task.py \
    --out "$DATASET_FILE" \
    --ratio "$RATIO" \
    --mode "$MODE"
echo "+ Dataset created: $DATASET_FILE"
echo ""

# Fine-tune (aligned with noLoRA config)
echo "[2/2] Starting Fine-Tuning..."
echo "  Model: meta-llama/Meta-Llama-3-8B-Instruct"
echo "  Epochs: 1"
echo "  Batch size: 1 (gradient accumulation: 8, effective: 8)"
echo "  Learning rate: 5e-5"
echo "  Max length: 1024"
echo "  Precision: FP32"
echo ""

python scripts/multi-task_finetune.py \
    --data_file "$DATASET_FILE" \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir "$MODEL_DIR" \
    --epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr 5e-5 \
    --max_length 1024

echo ""
echo "=============================================="
echo "  Fine-tuning Complete"
echo "=============================================="
echo "Model saved to: $MODEL_DIR"