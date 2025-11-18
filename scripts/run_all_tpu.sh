#!/bin/bash
# 在 TPU v6e 上运行完整的 SUM-CAR 训练和评估流程

set -e

echo "=== SUM-CAR TPU v6e 完整训练流程 ==="
echo "基础模型: meta-llama/Meta-Llama-3-8B-Instruct"
echo "训练方法: KV Memory + 稀疏微调 + 特异度追踪"
echo ""

# 设置 TPU 环境变量
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1

# 创建必要目录
mkdir -p out data

echo "====== 阶段 1: 训练任务特定 patches ======"
echo ""

# 训练 Math 任务
echo ">>> [1/3] 训练 Math 任务 (GSM8K)..."
python -m sumcar.cli.train_task train \
  --config configs/train_math.yaml \
  --use_xla True
echo "✓ Math patch 完成"
echo ""

# 训练 Code 任务
echo ">>> [2/3] 训练 Code 任务 (CodeXGLUE)..."
python -m sumcar.cli.train_task train \
  --config configs/train_code.yaml \
  --use_xla True
echo "✓ Code patch 完成"
echo ""

# 训练 Finance 任务
echo ">>> [3/3] 训练 Finance 任务 (FinQA)..."
python -m sumcar.cli.train_task train \
  --config configs/train_finqa.yaml \
  --use_xla True
echo "✓ Finance patch 完成"
echo ""

echo "====== 阶段 2: 合并 patches ======"
python -m sumcar.cli.merge_patches \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --patches out/patch_math.json out/patch_code.json out/patch_finqa.json \
  --out out/merged
echo "✓ Patches 合并完成"
echo ""

echo "====== 阶段 3: 评估 ======"
# 单任务评估
python -m sumcar.cli.eval_single --merged out/merged --out out/eval_single.json
echo "✓ 单任务评估完成"

# 组合任务评估
python scripts/prepare_composite.py --out data/composite.jsonl
python -m sumcar.cli.eval_composite --merged out/merged --composite data/composite.jsonl --out out/eval_composite.json
echo "✓ 组合任务评估完成"
echo ""

echo "====== 阶段 4: 计算指标 ======"
python -m sumcar.cli.compute_metrics \
  --per_task out/per_task_scores.json \
  --merged out/eval_single.json \
  --composite out/eval_composite.json \
  --patch_meta out/merged/patch_meta.json \
  --out out/metrics.json
echo "✓ 指标计算完成"
echo ""

echo "=== 所有流程完成! ==="
echo ""
echo "结果文件:"
echo "  - Patches: out/patch_*.json"
echo "  - 合并模型: out/merged/"
echo "  - 评估结果: out/eval_*.json"
echo "  - 最终指标: out/metrics.json"
