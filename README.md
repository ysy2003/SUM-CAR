# SUM-CAR

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt


# Train per-task patches 
!bash run_finqa_memory_only.sh 
!bash run_gsm8k_finqa_ft.sh


# Merge patches
!PYTHONPATH=/content/drive/MyDrive/SUM-CAR/src python -m sumcar.cli.merge_patches \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --patches noLoRA/patch_finqa.json,noLoRA_math/patch_gsm8k.json,noLoRA_code/patch_code.json \
  --out noLoRA/merged_math_finqa_code \
  --num_slots 65536 \
  --k_top 8 \
  --alpha 1.0 \
  --use_tfidf_scoring True \
  --use_capacity_budgeting True \
  --use_fp16 False \
  --verbose True \
  --max_slots_per_task 4096


# Evaluate single-task
!python /content/drive/MyDrive/SUM-CAR/noLoRA/code_only/eval_humaneval.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --merged_dir noLoRA/merged_selection_3task \
  --out noLoRA/eval/merged_selection_3task_code_eval.json \
  --k_top 8 --alpha 1.0 --use_cot True \
  --use_fp16 False --mode 100 --memory_position middle


# Evaluate composites
python scripts/prepare_composite.py --out data/composite.jsonl
python -m sumcar.cli.eval_composite --merged out/merged --composite data/composite.jsonl --out out/eval_composite.json
