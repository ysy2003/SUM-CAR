"""
使用 Tinker 合并多个 LoRA patch
对应原来的 merge_patches.py，但处理的是 Tinker LoRA 模型
"""
import os
import json
import fire
from dotenv import load_dotenv

import tinker
from ..utils.io import ensure_dir, dump_json, load_json
from ..utils.logger import Logger

load_dotenv()


def main(
    patches: list = None,
    out: str = 'out/merged_tinker',
    merge_strategy: str = 'weighted',
):
    """
    合并多个 Tinker LoRA patch
    
    Args:
        patches: tinker_patch_*.json 文件路径列表
        out: 输出目录
        merge_strategy: 合并策略 ('weighted', 'average', etc.)
    
    注意：
        - Tinker 的 LoRA 合并可能需要在云端进行
        - 这里主要记录 meta 信息和准备评估所需的配置
    """
    assert patches and len(patches) > 0, "Provide --patches a list of tinker_patch_*.json"
    
    logger = Logger('[merge-tinker]')
    ensure_dir(out)
    
    # 加载所有 patch 信息
    patch_list = []
    for p_path in patches:
        p = load_json(p_path)
        patch_list.append(p)
        logger.log(f"Loaded patch: {p['task']} (model_path={p['model_path']})")
    
    # 验证基础模型一致
    base_models = set(p['base_model'] for p in patch_list)
    if len(base_models) > 1:
        raise ValueError(f"Base models not consistent: {base_models}")
    base_model = patch_list[0]['base_model']
    
    # 创建合并信息
    merged_info = {
        'base_model': base_model,
        'merge_strategy': merge_strategy,
        'patches': [
            {
                'task': p['task'],
                'model_path': p['model_path'],
                'lora_rank': p['lora_rank'],
                'num_examples': p['num_examples'],
            }
            for p in patch_list
        ],
        'num_tasks': len(patch_list),
    }
    
    # 对于 Tinker LoRA，合并可以：
    # 方案1: 使用多个 sampling_client，在推理时根据任务路由
    # 方案2: 如果 Tinker 支持，可以合并多个 LoRA adapter
    # 方案3: 分别评估每个任务的 LoRA，然后汇总结果
    
    logger.log("Note: Tinker LoRA models are stored separately.")
    logger.log("For evaluation, you can:")
    logger.log("  1. Use task-specific LoRA models for each evaluation")
    logger.log("  2. Or implement a router to select appropriate LoRA per query")
    
    # 保存合并元信息
    merged_path = os.path.join(out, 'merged_info.json')
    dump_json(merged_info, merged_path)
    logger.log(f"Merged info saved to: {merged_path}")
    
    # 创建评估配置
    eval_config = {
        'base_model': base_model,
        'tasks': {
            p['task']: {
                'model_path': p['model_path'],
                'lora_rank': p['lora_rank'],
            }
            for p in patch_list
        }
    }
    
    eval_config_path = os.path.join(out, 'eval_config.json')
    dump_json(eval_config, eval_config_path)
    logger.log(f"Evaluation config saved to: {eval_config_path}")


if __name__ == '__main__':
    fire.Fire(main)
