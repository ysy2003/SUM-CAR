#!/bin/bash

# Generate composite dataset
python scripts/prepare_composite.py --out composite_dataset.json --max_samples 100

# Run fine-tuning
python scripts/multi-task_finetune.py \
    --data_file composite_dataset.json \
    --base_model gpt2 \
    --output_dir finetuned_model \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5