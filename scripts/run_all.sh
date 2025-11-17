#!/usr/bin/env bash
set -euo pipefail

# 0) Activate venv and install deps (assumes you already ran this once)
# python -m venv .venv && source .venv/bin/activate
# pip install -U pip && pip install -r requirements.txt

# 1) Train patches
python -m sumcar.cli.train_task --task math  --config configs/train_math.yaml
python -m sumcar.cli.train_task --task code  --config configs/train_code.yaml
python -m sumcar.cli.train_task --task finqa --config configs/train_finqa.yaml

# 2) Evaluate each patch standalone (to produce S_patch for retention)
python -m sumcar.cli.eval_single --patch out/patch_math.json  --out out/eval_patch_math.json
python -m sumcar.cli.eval_single --patch out/patch_code.json  --out out/eval_patch_code.json
python -m sumcar.cli.eval_single --patch out/patch_finqa.json --out out/eval_patch_finqa.json

# Merge evals into one per_task json
python - <<'PY'
import json
S = {
  'gsm8k': json.load(open('out/eval_patch_math.json'))['gsm8k'],
  'humaneval': json.load(open('out/eval_patch_code.json'))['humaneval'],
  'finqa': json.load(open('out/eval_patch_finqa.json'))['finqa']
}
json.dump(S, open('out/per_task_scores.json', 'w'), indent=2)
print('wrote out/per_task_scores.json')
PY

# 3) Merge patches
python -m sumcar.cli.merge_patches \
  --base_model gpt2 \
  --patches out/patch_math.json out/patch_code.json out/patch_finqa.json \
  --out out/merged

# 4) Evaluate merged on single-task suites
python -m sumcar.cli.eval_single --merged out/merged --out out/eval_single.json

# 5) Prepare and evaluate composite prompts
mkdir -p data
python scripts/prepare_composite.py --out data/composite.jsonl
python -m sumcar.cli.eval_composite --merged out/merged --composite data/composite.jsonl --out out/eval_composite.json

# 6) Compute four metrics
python -m sumcar.cli.compute_metrics \
  --per_task out/per_task_scores.json \
  --merged out/eval_single.json \
  --composite out/eval_composite.json \
  --patch_meta out/merged/patch_meta.json \
  --out out/metrics.json

echo "All done. See out/metrics.json"
