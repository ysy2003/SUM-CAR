import os, yaml 
import fire 
import math
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer 
from datasets import Dataset 
from dotenv import load_dotenv

# Tinker imports
import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.data import conversation_to_datum

from ..config import MemCfg, TrainCfg 
from ..utils.logger import Logger 
from ..utils.io import ensure_dir, dump_json 
from ..utils.seed import set_all 
from ..data.collator import CLMCollator 
from ..data import gsm8k as gsm 
from ..data import code_codexglue as cglue 
from ..data import mbpp as mbpp 
from ..data import finqa_rc as finqa
from ..memory.sparse_finetune import SparseFinetuner

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
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        batch.append(d)
    return batch


def main(task: str, config: str, use_tinker: bool = True, lora_rank: int = 32):
    """Train a per-task sparse-memory skill patch using Tinker + LoRA.

    Args:
        task: short name for this patch (e.g., 'math' / 'code' / 'finqa').
        config: path to YAML config (contains base_model, mem, train).
        use_tinker: if True, use Tinker + LoRA; if False, use original local training.
        lora_rank: LoRA rank for Tinker training (default 32).
    Outputs:
        out/patch_{task}.json         — serialized patch (LoRA info or KV memory)
        out/patch_{task}_meta.json    — metadata (task, stats)
    """
    cfg = yaml.safe_load(open(config, 'r'))
    base_model = cfg['base_model']
    mem_cfg = cfg['mem']
    train_cfg = cfg['train']

    set_all(train_cfg.get('seed', 42))
    logger = Logger(f"[train:{task}]")

    # 1) Load dataset
    ds_name = train_cfg['dataset']
    if ds_name not in LOADERS:
        raise ValueError(f"Unknown dataset key: {ds_name}")
    logger.log('loading dataset:', ds_name)
    ds = LOADERS[ds_name]('train')
    
    # 限制数据量（可选，用于快速测试）
    max_examples = train_cfg.get('max_examples', None)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
        logger.log(f'limited to {len(ds)} examples for testing')

    if use_tinker:
        # ============ Tinker + LoRA 训练路径 ============
        logger.log('Using Tinker + LoRA training mode')
        
        # 转换数据集为 messages 格式
        logger.log('Converting dataset to messages format...')
        messages_list = dataset_to_messages(ds)
        
        # 准备 tokenizer 和 renderer
        logger.log(f'Loading tokenizer and renderer for {base_model}...')
        tokenizer = get_tokenizer(base_model)
        renderer_name = model_info.get_recommended_renderer_name(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        logger.log(f'Using renderer: {renderer_name}')
        
        # 创建 Tinker service client
        service_client = tinker.ServiceClient()
        
        # 创建 LoRA 训练客户端
        logger.log(f'Creating LoRA training client (rank={lora_rank})...')
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=lora_rank,
        )
        
        # 训练参数
        batch_size = train_cfg['batch_size']
        learning_rate = train_cfg['lr']
        epochs = train_cfg['epochs']
        max_length = train_cfg['max_length']
        
        # 计算训练步数
        max_steps = math.ceil(len(messages_list) / batch_size) * epochs
        logger.log(f'Starting training: {len(messages_list)} examples, {max_steps} steps, {epochs} epochs')
        
        training_stats = {
            'total_examples': len(messages_list),
            'batch_size': batch_size,
            'epochs': epochs,
            'total_steps': max_steps,
            'losses': []
        }
        
        step = 0
        for epoch in range(epochs):
            logger.log(f'=== Epoch {epoch + 1}/{epochs} ===')
            
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
                    logger.log(f'[train] step={step}/{max_steps}, lr={current_lr:.2e}, loss={loss:.4f}')
                
                # 定期保存 checkpoint
                if step > 0 and step % 200 == 0:
                    training_client.save_state(name=f'{task}_step_{step:06d}')
                    logger.log(f'Checkpoint saved at step {step}')
                
                step += 1
        
        # 训练完成，保存最终权重
        logger.log('Training complete! Saving LoRA weights...')
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f'sumcar_lora_{task}'
        )
        
        # 导出 patch 信息
        save_dir = train_cfg['save_dir']
        ensure_dir(save_dir)
        ensure_dir('out')
        
        # 保存 LoRA 模型信息
        patch = {
            'task': task,
            'model_type': 'tinker_lora',
            'base_model': base_model,
            'lora_rank': lora_rank,
            'model_path': sampling_client.model_path,
            'training_stats': training_stats,
            'dataset': ds_name,
            'num_examples': len(messages_list),
        }
        
        meta = {
            'task': task,
            'model_type': 'tinker_lora',
            'base_model': base_model,
            'lora_rank': lora_rank,
            'num_examples': len(messages_list),
            'final_loss': training_stats['losses'][-1] if training_stats['losses'] else None,
            'avg_loss': sum(training_stats['losses']) / len(training_stats['losses']) if training_stats['losses'] else None,
        }
        
        dump_json(patch, os.path.join('out', f'patch_{task}.json'))
        dump_json(meta, os.path.join('out', f'patch_{task}_meta.json'))
        
        logger.log(f'Patch saved to out/patch_{task}.json')
        logger.log(f'Model path: {sampling_client.model_path}')
        
        # 可选：下载 checkpoint
        download_checkpoint = train_cfg.get('download_checkpoint', False)
        if download_checkpoint:
            logger.log('Downloading checkpoint archive...')
            rest_client = service_client.create_rest_client()
            future = rest_client.download_checkpoint_archive_from_tinker_path(
                sampling_client.model_path
            )
            checkpoint_path = os.path.join(save_dir, f'{task}_checkpoint.tar.gz')
            with open(checkpoint_path, 'wb') as f:
                f.write(future.result())
            logger.log(f'Checkpoint downloaded to: {checkpoint_path}')
    
    else:
        # ============ 原始本地训练路径 ============
        logger.log('Using original local training mode')
        
        # 2) Tokenizer & collator
        tok = AutoTokenizer.from_pretrained(base_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        collate = CLMCollator(tok, max_length=train_cfg['max_length'])
        dl = DataLoader(ds, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate)

        # 3) Build trainer (model + memory)
        ft = SparseFinetuner(base_model, mem_cfg, train_cfg, tokenizer=tok, logger=logger)

        # 4) Phase I: probe slot access (all frozen, just logging)
        ft.mem.freeze_all()
        ft.probe(dl, steps=train_cfg['probe_steps'])

        # 5) Phase II: choose top-t most-accessed slots, unfreeze and finetune
        top_t = train_cfg['top_t']
        slot_ids = ft.mem.top_slots(top_t)
        logger.log('unfreezing top-t slots:', len(slot_ids))
        ft.mem.unfreeze_slots(slot_ids)
        ft.train(dl, epochs=train_cfg['epochs'])

        # 6) Export patch
        save_dir = train_cfg['save_dir']
        patch = ft.build_patch(task, top_t, save_dir)
        ensure_dir('out')
        dump_json(patch, os.path.join('out', f'patch_{task}.json'))
        dump_json({'task': task, 'slot_ids': slot_ids, 'top_t': top_t}, os.path.join('out', f'patch_{task}_meta.json'))
        logger.log('patch saved to out/', f'patch_{task}.json')

if __name__ == '__main__':
    fire.Fire(main)

