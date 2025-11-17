"""
使用 Tinker + LoRA 在 GPU 上训练任务特定的 patch
改造自原始的 train_task.py，使用 Tinker API 替代本地训练
"""
import os
import yaml
import math
import fire
import torch
from dotenv import load_dotenv

import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.data import conversation_to_datum

from datasets import Dataset
from ..config import MemCfg, TrainCfg
from ..utils.logger import Logger
from ..utils.io import ensure_dir, dump_json
from ..utils.seed import set_all
from ..data import gsm8k as gsm
from ..data import code_codexglue as cglue
from ..data import mbpp as mbpp
from ..data import finqa_rc as finqa

load_dotenv()

LOADERS = {
    'gsm8k': lambda split: gsm.load(split),
    'codexglue_refine': lambda split: cglue.load(split),
    'mbpp': lambda split: mbpp.load(split),
    'finqa_rc': lambda split: finqa.load(split),
}


def dataset_to_messages(ds: Dataset) -> list:
    """
    将 SUM-CAR 的数据集格式转换为 Tinker 所需的 messages 格式
    
    输入格式: {'prompt': ..., 'target': ...}
    输出格式: [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]
    """
    messages_list = []
    for row in ds:
        messages = [
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['target']}
        ]
        messages_list.append(messages)
    return messages_list


def to_batch_datums(msg_batch, renderer, max_len=2048):
    """将 messages batch 转换为 Tinker 训练所需的 datums"""
    batch = []
    for messages in msg_batch:
        d = conversation_to_datum(
            messages,
            renderer,
            max_len,
            train_on=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        batch.append(d)
    return batch


def train_lora_with_tinker(
    task: str,
    messages_list: list,
    base_model: str,
    lora_rank: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    max_length: int,
    logger: Logger,
    renderer,
):
    """
    使用 Tinker API 进行 LoRA 微调
    
    返回:
        sampling_client: 微调后的模型客户端
        training_stats: 训练统计信息
    """
    service_client = tinker.ServiceClient()
    
    # 创建 LoRA 训练客户端
    logger.log(f"Creating LoRA training client (rank={lora_rank})...")
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank,
    )
    
    # 计算训练步数
    max_steps = math.ceil(len(messages_list) / batch_size) * epochs
    logger.log(f"Starting training: {len(messages_list)} examples, {max_steps} steps, {epochs} epochs")
    
    training_stats = {
        'total_examples': len(messages_list),
        'batch_size': batch_size,
        'epochs': epochs,
        'total_steps': max_steps,
        'losses': []
    }
    
    step = 0
    for epoch in range(epochs):
        logger.log(f"=== Epoch {epoch + 1}/{epochs} ===")
        
        # 每个 epoch 打乱数据
        import random
        shuffled = messages_list.copy()
        random.shuffle(shuffled)
        
        for batch_start in range(0, len(shuffled), batch_size):
            batch_end = min(batch_start + batch_size, len(shuffled))
            msg_batch = shuffled[batch_start:batch_end]
            
            # 转换为 datums
            batch_datums = to_batch_datums(msg_batch, renderer, max_length)
            
            # 前向 + 反向
            fwd_bwd_future = training_client.forward_backward(
                batch_datums,
                loss_fn="cross_entropy",
            )
            
            # 学习率线性衰减
            current_lr = learning_rate * max(0.0, 1.0 - step / max_steps)
            
            # 优化器 step
            optim_future = training_client.optim_step(
                tinker.AdamParams(
                    learning_rate=current_lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            )
            
            # 等待结果
            fwd_bwd_result = fwd_bwd_future.result()
            _ = optim_future.result()
            
            loss = fwd_bwd_result.loss
            training_stats['losses'].append(loss)
            
            if step % 20 == 0:
                logger.log(f"[train] step={step}/{max_steps}, lr={current_lr:.2e}, loss={loss:.4f}")
            
            # 定期保存 checkpoint
            if step > 0 and step % 200 == 0:
                training_client.save_state(name=f"{task}_step_{step:06d}")
                logger.log(f"Checkpoint saved at step {step}")
            
            step += 1
    
    # 训练完成，保存最终权重
    logger.log("Training complete! Saving LoRA weights...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"sumcar_lora_{task}"
    )
    
    return sampling_client, training_stats


def main(task: str, config: str, lora_rank: int = 32):
    """
    使用 Tinker + LoRA 训练任务特定的 patch
    
    Args:
        task: 任务名称 (e.g., 'math' / 'code' / 'finqa')
        config: YAML 配置文件路径
        lora_rank: LoRA 秩 (默认 32，显存不足可用 8/16)
    
    Outputs:
        out/tinker_patch_{task}.json - LoRA 模型信息和统计
        out/tinker_patch_{task}_meta.json - 元数据
    """
    cfg = yaml.safe_load(open(config, 'r'))
    base_model = cfg['base_model']
    mem_cfg = cfg['mem']
    train_cfg = cfg['train']
    
    set_all(train_cfg.get('seed', 42))
    logger = Logger(f"[train-tinker:{task}]")
    
    # 1) 加载数据集
    ds_name = train_cfg['dataset']
    if ds_name not in LOADERS:
        raise ValueError(f"Unknown dataset key: {ds_name}")
    
    logger.log(f"Loading dataset: {ds_name}")
    ds = LOADERS[ds_name]('train')
    
    # 限制数据量（可选，用于快速测试）
    max_examples = train_cfg.get('max_examples', None)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
        logger.log(f"Limited to {len(ds)} examples for testing")
    
    # 2) 转换为 messages 格式
    logger.log("Converting dataset to messages format...")
    messages_list = dataset_to_messages(ds)
    
    # 3) 准备 tokenizer 和 renderer
    logger.log(f"Loading tokenizer and renderer for {base_model}...")
    tokenizer = get_tokenizer(base_model)
    renderer_name = model_info.get_recommended_renderer_name(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.log(f"Using renderer: {renderer_name}")
    
    # 4) 使用 Tinker 进行 LoRA 训练
    sampling_client, training_stats = train_lora_with_tinker(
        task=task,
        messages_list=messages_list,
        base_model=base_model,
        lora_rank=lora_rank,
        batch_size=train_cfg['batch_size'],
        learning_rate=train_cfg['lr'],
        epochs=train_cfg['epochs'],
        max_length=train_cfg['max_length'],
        logger=logger,
        renderer=renderer,
    )
    
    # 5) 导出 patch 信息
    save_dir = train_cfg['save_dir']
    ensure_dir(save_dir)
    ensure_dir('out')
    
    # 保存 LoRA 模型信息
    patch_info = {
        'task': task,
        'model_type': 'tinker_lora',
        'base_model': base_model,
        'lora_rank': lora_rank,
        'model_path': sampling_client.model_path,
        'training_stats': training_stats,
        'dataset': ds_name,
        'num_examples': len(messages_list),
    }
    
    meta_info = {
        'task': task,
        'model_type': 'tinker_lora',
        'base_model': base_model,
        'lora_rank': lora_rank,
        'num_examples': len(messages_list),
        'final_loss': training_stats['losses'][-1] if training_stats['losses'] else None,
        'avg_loss': sum(training_stats['losses']) / len(training_stats['losses']) if training_stats['losses'] else None,
    }
    
    patch_path = os.path.join('out', f'tinker_patch_{task}.json')
    meta_path = os.path.join('out', f'tinker_patch_{task}_meta.json')
    
    dump_json(patch_info, patch_path)
    dump_json(meta_info, meta_path)
    
    logger.log(f"Patch info saved to: {patch_path}")
    logger.log(f"Meta info saved to: {meta_path}")
    logger.log(f"Model path: {sampling_client.model_path}")
    
    # 可选：下载 checkpoint
    download_checkpoint = train_cfg.get('download_checkpoint', False)
    if download_checkpoint:
        logger.log("Downloading checkpoint archive...")
        rest_client = tinker.ServiceClient().create_rest_client()
        future = rest_client.download_checkpoint_archive_from_tinker_path(
            sampling_client.model_path
        )
        checkpoint_path = os.path.join(save_dir, f'{task}_checkpoint.tar.gz')
        with open(checkpoint_path, 'wb') as f:
            f.write(future.result())
        logger.log(f"Checkpoint downloaded to: {checkpoint_path}")


if __name__ == '__main__':
    fire.Fire(main)
