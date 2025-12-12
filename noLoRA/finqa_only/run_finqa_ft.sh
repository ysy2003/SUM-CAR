#!/usr/bin/env bash
set -euo pipefail

# 从仓库根目录执行：bash run_gsm8k_finqa.sh

BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
MERGED_DIR="noLoRA/merged_finqa"
OUT_DIR="noLoRA/eval"
OUT_FILE_100="${OUT_DIR}/gsm8k_finqa_cot_100.json"

mkdir -p "${OUT_DIR}"

echo "=== Running GSM8K eval (first 100 samples, CoT) with FinQA memory ==="

python noLoRA/eval_math_only.py \
  --base_model="${BASE_MODEL}" \
  --merged_dir="${MERGED_DIR}" \
  --out="${OUT_FILE_100}" \
  --k_top=64 \
  --alpha=1.0 \
  --use_cot=True \
  --use_fp16=False \
  --mode=100 \
  --memory_position=middle
