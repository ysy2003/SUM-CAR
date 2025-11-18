#!/bin/bash
# GCP TPU v6e 设置脚本 - 用于在 TPU VM 上安装依赖和配置环境

set -e

echo "=== SUM-CAR TPU v6e Setup Script ==="
echo "开始在 TPU VM 上设置环境..."

# 更新系统包
echo "1. 更新系统包..."
sudo apt-get update
sudo apt-get install -y python3-pip git

# 安装 PyTorch（CPU 版本）
echo "2. 安装 PyTorch (CPU)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装 PyTorch XLA (用于 TPU)
echo "3. 安装 PyTorch XLA for TPU v6e..."
pip install torch-xla -f https://storage.googleapis.com/libtpu-releases/index.html

# 安装项目依赖
echo "4. 安装项目依赖..."
pip install transformers accelerate datasets peft safetensors
pip install pyyaml fire tqdm python-dotenv evaluate

# 安装项目本身
echo "5. 安装 sumcar 包..."
pip install -e .

# 验证 TPU 可用性
echo "6. 验证 TPU 设置..."
python3 -c "
import torch
import torch_xla.core.xla_model as xm
print(f'PyTorch version: {torch.__version__}')
print(f'XLA devices: {xm.xrt_world_size()}')
print(f'Current device: {xm.xla_device()}')
print('TPU setup successful!')
"

echo ""
echo "=== 设置完成! ==="
echo ""
echo "建议设置 GCS checkpoint 目录（preemptible 安全）："
echo "  export CKPT_DIR=\"gs://<your-bucket>/sumcar_kv/\""
echo ""
echo "运行训练（8核 TPU）："
echo "  python -m torch_xla.distributed.xla_multiprocessing \\"
echo "    python -m sumcar.cli.train_task train_xla \\"
echo "    --config configs/train_math.yaml \\"
echo "    --num_processes 8"

