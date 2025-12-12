# SUM-CAR

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt


# Train per-task patches 
# FinQA task
!bash noLoRA/finqa_only/run_finqa_memory_only.sh

# GSM8K task
!bash noLoRA/math_only/run_math_memory_only.sh

# Code task
!bash noLoRA/code_only/run_code_memory_only.sh


# Merge patches (for example, merging FinQA, GSM8K, and Code patches)
!PYTHONPATH=/content/drive/MyDrive/SUM-CAR/src python -m sumcar.cli.merge_patches \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --patches patch_finqa.json,patch_gsm8k.json,patch_mbpp.json \
  --out noLoRA/merged_math_finqa_code \
  --num_slots 65536 \
  --k_top 64 \
  --alpha 1.0 \
  --use_tfidf_scoring True \
  --use_capacity_budgeting True \
  --use_fp16 False \
  --verbose True \
  --max_slots_per_task 4096


# Evaluation
# Code (HumanEval) task evaluation
!python /content/drive/MyDrive/SUM-CAR/noLoRA/code_only/eval_humaneval.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --merged_dir merged_selection_3task \
  --out noLoRA/eval/merged_selection_3task_code_eval.json \
  --k_top 64 --alpha 1.0 \
  --use_fp16 False --mode full --memory_position middle

# Code (MBPP) task evaluation
!python /content/drive/MyDrive/SUM-CAR/noLoRA/eval_mbpp_finetune_only.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --merged_dir noLoRA/merged_selection_3task \
  --out noLoRA/eval/merged_selection_3task_mbpp_eval.json \
  --k_top 64 --alpha 1.0 \
  --use_fp16 False --memory_position middle

# FinQA task evaluation
!python /content/drive/MyDrive/SUM-CAR/noLoRA/eval_finqa_only.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --merged_dir noLoRA/merged_selection_3task \
  --out noLoRA/eval/merged_selection_3task_finqa_eval.json \
  --k_top 64 --alpha 1.0 --use_cot True \
  --use_fp16 False --memory_position middle


# Math (GSM8K) task evaluation
!python /content/drive/MyDrive/SUM-CAR/noLoRA/eval_math_only.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --merged_dir noLoRA/merged_selection_3task \
  --out noLoRA/eval/merged_selection_3task_math_eval.json \
  --k_top 64 --alpha 1.0 --use_cot True \
  --use_fp16 False --mode full --memory_position middle

## Full Evaluation (all models on all tasks)
# Evaluates baseline(Meta-Llama-3-8B-Instruct), code_only, math_only, finance_only on GSM8K, FinQA, HumanEval
!bash noLoRA/run_all_eval.sh --mode full


# Evaluation metrics
runs/main_runs_2.ipynb
runs/metrics.ipynb
!python3 src/sumcar/metrics/toggle_math_slots_offline.py

## Multitask
!bash noLoRA/multi_task/run_multi_task_pipeline.sh

## composite eval
!bash runs/run_composite_eval.sh --model  noLoRA/merged_concat_3task --mode full
