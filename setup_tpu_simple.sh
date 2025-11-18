#!/bin/bash
# 简化的 TPU 设置脚本 - 不需要 sudo，只安装 Python 包

set -e

echo "=== SUM-CAR TPU 简化安装 ==="
echo "跳过系统包，只安装 Python 依赖..."
echo ""

# 安装 PyTorch（CPU 版本）
echo "1. 安装 PyTorch (CPU)..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

# 安装 PyTorch XLA (用于 TPU)
echo "2. 安装 PyTorch XLA for TPU v3..."
pip install torch-xla -f https://storage.googleapis.com/libtpu-releases/index.html --quiet

# 安装项目依赖
echo "3. 安装项目依赖..."
pip install transformers accelerate datasets --quiet
pip install pyyaml fire tqdm python-dotenv evaluate --quiet

# 安装项目本身
echo "4. 安装 sumcar 包..."
pip install -e . --quiet

# 验证 TPU 可用性
echo "5. 验证 TPU 设置..."
python3 -c "
import torch
import torch_xla.core.xla_model as xm
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ XLA devices: {xm.xrt_world_size()}')
print(f'✓ Current device: {xm.xla_device()}')
print('✓ TPU setup successful!')
"

echo ""
echo "=== 安装完成! ==="
echo ""
echo "开始训练："
echo "  python -m sumcar.cli.train_task train --config configs/train_math.yaml --use_xla True"
