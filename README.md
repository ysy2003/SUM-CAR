# SUM-CAR: Memory-Enhanced FFN with Task-Specific Adaptation

# SUM-CAR (Sparse Union Merge with Conflict-Aware Remapping)


This repo contains a minimal but complete scaffold to reproduce your proposal:
- Per-task sparse memory finetuning to produce **skill patches** for Math (GSM8K), Code (CodeXGLUE/MBPP), Finance (FinQA RC subset)
- **SUM-CAR** merging with conflict-aware slot **remapping** instead of parameter averaging
- Single-task + Composite evaluation with four metrics: **Retention**, **Joint Accuracy**, **Slot Growth per Task**, **Reversibility**


## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt


# Train per-task patches (examples)
python -m sumcar.cli.train_task --task math --config configs/train_math.yaml
python -m sumcar.cli.train_task --task code --config configs/train_code.yaml
python -m sumcar.cli.train_task --task finqa --config configs/train_finqa.yaml


# Merge patches
python -m sumcar.cli.merge_patches \
--base_run out/base \
--patches out/patch_math.json out/patch_code.json out/patch_finqa.json \
--out out/merged


# Evaluate single-task
python -m sumcar.cli.eval_single --merged out/merged --config configs/eval.yaml --out out/eval_single.json


# Evaluate composites
python scripts/prepare_composite.py --out data/composite.jsonl
python -m sumcar.cli.eval_composite --merged out/merged --composite data/composite.jsonl --out out/eval_composite.json


# Compute 4 metrics (Retention / JointAcc / SlotGrowth / Reversibility)
python -m sumcar.cli.compute_metrics \
--per_task out/per_task_scores.json \
--merged out/eval_single.json \
--composite out/eval_composite.json \
--patch_meta out/patch_meta.json \
--out out/metrics.json
```


Adjust configs for model size, memory slots and top-`t` unfreezing.