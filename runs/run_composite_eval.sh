#!/bin/bash


GROUND_TRUTH_FILE="out/composite_eval/finqa_composite_dev_cleaned.jsonl"
GENERATIONS_FILE="out/composite_eval/generations.jsonl"


python scripts/run_composite_eval.py \
    --model_name_or_path "$MODEL_PATH" \
    --ground_truth_file "$GROUND_TRUTH_FILE" \
    --generations_file "$GENERATIONS_FILE" \
    --max_samples 200