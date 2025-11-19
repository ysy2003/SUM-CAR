import os, yaml 
import fire 
import math
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer 
from datasets import Dataset 
from dotenv import load_dotenv

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


def main(task: str = None, config: str = None, config_path: str = None, use_xla: bool = False):
    """Train a per-task sparse-memory skill patch using KV memory.

    Args:
        task: short name for this patch (e.g., 'math' / 'code' / 'finqa').
        config: path to YAML config (contains base_model, mem, train).
        config_path: alternative name for config parameter
        use_xla: if True, use XLA/TPU training
    Outputs:
        out/patch_{task}.json         — serialized patch (KV memory)
        out/patch_{task}_meta.json    — metadata (task, stats)
    """
    # 处理参数兼容性
    if config_path and not config:
        config = config_path
    
    cfg = yaml.safe_load(open(config, 'r'))
    base_model = cfg['base_model']
    mem_cfg = cfg['mem']
    train_cfg = cfg['train']
    
    # 打印配置以验证参数（防止 YAML 重复键问题）
    print(f"[cfg] num_slots={mem_cfg['num_slots']}, k_top={mem_cfg['k_top']}, "
          f"probe_steps={train_cfg.get('probe_steps', 1000)}, top_t={train_cfg['top_t']}, "
          f"batch_size={train_cfg['batch_size']}, max_length={train_cfg['max_length']}")
    
    # 从 config 推断 task（如果未提供）
    if not task:
        task = train_cfg.get('dataset', 'unknown').split('_')[0]

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

    # ============ KV Memory 训练路径（支持 XLA/TPU）============
    logger.log(f'Using KV memory training mode (XLA={use_xla})')
    
    # 2) Tokenizer & collator
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    collate = CLMCollator(tok, max_length=train_cfg['max_length'])
    dl = DataLoader(ds, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate)

    # 3) Build trainer (model + memory) with XLA support
    ft = SparseFinetuner(base_model, mem_cfg, train_cfg, tokenizer=tok, logger=logger, use_xla=use_xla)

    # 4) Phase I: probe slot access (all frozen, just logging)
    ft.mem.freeze_all()
    probe_steps = train_cfg.get('probe_steps', 1000)
    logger.log(f'Phase I: Probing {probe_steps} steps...')
    ft.probe(dl, steps=probe_steps)

    # 5) Phase II: choose top-t most-accessed slots, unfreeze and finetune
    top_t = train_cfg['top_t']
    slot_ids = ft.mem.top_slots(top_t)
    logger.log('unfreezing top-t slots:', len(slot_ids))
    ft.mem.unfreeze_slots(slot_ids)
    
    # Train with refresh_every parameter
    refresh_every = train_cfg.get('refresh_every', 200)
    logger.log(f'Phase II: Training {train_cfg["epochs"]} epochs with refresh_every={refresh_every}...')
    ft.train(dl, epochs=train_cfg['epochs'], refresh_every=refresh_every)

    # 6) Export patch
    save_dir = train_cfg['save_dir']
    patch = ft.build_patch(task, top_t, save_dir)
    ensure_dir('out')
    dump_json(patch, os.path.join('out', f'patch_{task}.json'))
    
    # 保存元数据（包含 specificity 信息）
    meta = {
        'task': task,
        'slot_ids': slot_ids,
        'top_t': top_t,
        'num_slots': mem_cfg['num_slots'],
        'use_xla': use_xla
    }
    dump_json(meta, os.path.join('out', f'patch_{task}_meta.json'))
    logger.log('patch saved to out/', f'patch_{task}.json')


if __name__ == '__main__':
    fire.Fire(main)

