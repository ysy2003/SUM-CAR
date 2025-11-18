#!/usr/bin/env python3
"""
在 Google Cloud TPU 上运行 SUM-CAR 训练的包装脚本
使用 PyTorch XLA 进行 TPU 加速
"""

import os
import sys
import yaml
import argparse
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def setup_tpu_environment():
    """设置 TPU 环境变量"""
    # 确保使用 XLA
    os.environ['XLA_USE_BF16'] = '1'  # 使用 bfloat16 以提高性能
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
    
    print(f"TPU 设备数量: {xm.xrt_world_size()}")
    print(f"当前设备: {xm.get_ordinal()}")
    
def main():
    parser = argparse.ArgumentParser(description='在 TPU 上训练 SUM-CAR 模型')
    parser.add_argument('--task', type=str, required=True, 
                       choices=['math', 'code', 'finqa'],
                       help='训练任务类型')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (YAML)')
    parser.add_argument('--tpu-cores', type=int, default=8,
                       help='使用的 TPU 核心数 (默认: 8)')
    
    args = parser.parse_args()
    
    # 设置 TPU 环境
    setup_tpu_environment()
    
    # 读取配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n=== 开始在 TPU 上训练 {args.task} 任务 ===")
    print(f"配置文件: {args.config}")
    print(f"基础模型: {config.get('base_model', 'N/A')}")
    print(f"TPU 核心数: {args.tpu_cores}\n")
    
    # 导入并运行训练脚本
    from sumcar.cli.train_task import main as train_main
    
    # 设置 accelerate 使用 TPU
    os.environ['ACCELERATE_USE_TPU'] = '1'
    
    # 调用原始训练函数
    try:
        train_main(config_path=args.config)
        print(f"\n=== {args.task} 任务训练完成! ===")
    except Exception as e:
        print(f"\n错误: 训练失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
