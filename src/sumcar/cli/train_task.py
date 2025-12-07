import os, yaml, hashlib, time
import fire 
import math
import torch
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer, __version__ as transformers_version
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
from ..utils.checkpoint import CheckpointManager

load_dotenv()

LOADERS = {
    'gsm8k': lambda split, use_cot=False: gsm.load(split, use_cot=use_cot),
    'codexglue_refine': lambda split, use_cot=False: cglue.load(split),  # Code doesn't need CoT
    'mbpp': lambda split, use_cot=False: mbpp.load(split),
    'finqa_rc': lambda split, use_cot=False: finqa.load(split, use_cot=use_cot),
}


def dataset_to_messages(ds: Dataset) -> list:
    """
    Convert SUM-CAR dataset format to Tinker required messages format
    Input format: {'prompt': ..., 'target': ...}
    Output format: [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]
    """
    messages_list = []
    for row in ds:
        messages = [
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['target']}
        ]
        messages_list.append(messages)
    return messages_list


def main(task: str = None, config: str = None, config_path: str = None, max_examples: int = None, epochs: int = None):
    """Train a per-task sparse-memory skill patch using KV memory.

    Args:
        task: short name for this patch (e.g., 'math' / 'code' / 'finqa').
        config: path to YAML config (contains base_model, mem, train).
        config_path: alternative name for config parameter
        max_examples: override max_examples from config (CLI override)
        epochs: override epochs from config (CLI override)
    Outputs:
        out/patch_{task}.json         — serialized patch (KV memory)
        out/patch_{task}_meta.json    — metadata (task, stats)
    """
    # Handle parameter compatibility
    if config_path and not config:
        config = config_path

    cfg = yaml.safe_load(open(config, 'r'))
    base_model = cfg['base_model']
    mem_cfg = cfg['mem']
    train_cfg = cfg['train']

    # CLI overrides
    if max_examples is not None:
        train_cfg['max_examples'] = max_examples
        print(f"[CLI override] max_examples={max_examples}")
    if epochs is not None:
        train_cfg['epochs'] = epochs
        print(f"[CLI override] epochs={epochs}")

    # Print config to verify parameters (prevent YAML duplicate key issues)
    print(f"[cfg] num_slots={mem_cfg['num_slots']}, k_top={mem_cfg['k_top']}, "
          f"probe_steps={train_cfg.get('probe_steps', 1000)}, top_t={train_cfg['top_t']}, "
          f"batch_size={train_cfg['batch_size']}, max_length={train_cfg['max_length']}")

    # Infer task from config (if not provided)
    if not task:
        task = train_cfg.get('dataset', 'unknown').split('_')[0]

    set_all(train_cfg.get('seed', 42))
    logger = Logger(f"[train:{task}]")

    # 1) Load dataset
    ds_name = train_cfg['dataset']
    if ds_name not in LOADERS:
        raise ValueError(f"Unknown dataset key: {ds_name}")

    # Check if using CoT
    use_cot = train_cfg.get('use_cot', False)
    logger.log(f'Loading dataset: {ds_name} (use_cot={use_cot})')
    ds = LOADERS[ds_name]('train', use_cot=use_cot)

    # Limit data size (optional, for quick testing)
    max_ex = train_cfg.get('max_examples', None)
    if max_ex:
        ds = ds.select(range(min(max_ex, len(ds))))
        logger.log(f'limited to {len(ds)} examples for testing')

    # ============ KV Memory training path (GPU/CUDA support) ============
    logger.log(f'Using KV memory training mode')

    # 2) Tokenizer & collator
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    collate = CLMCollator(tok, max_length=train_cfg['max_length'])
    dl = DataLoader(ds, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate)

    # 3) Build trainer (model + memory) with GPU support
    ft = SparseFinetuner(base_model, mem_cfg, train_cfg, tokenizer=tok, logger=logger)

    # Create checkpoint manager
    # Use configurable checkpoint base directory (defaults to current directory)
    ckpt_base_dir = train_cfg.get('checkpoint_base_dir', '.')
    ckpt_manager = CheckpointManager(
        base_dir=ckpt_base_dir,
        task=task,
        base_model_id=base_model
    )
    ft.ckpt_manager = ckpt_manager  # Inject into finetuner
    logger.log(f'Checkpoint dirs created: {ckpt_base_dir}/runs/{task}/ckpts, {ckpt_base_dir}/patches/, {ckpt_base_dir}/merges/')

    # 4) Phase I: probe slot access (all frozen, just logging)
    probe_steps = train_cfg.get('probe_steps', 1000)
    top_t = train_cfg['top_t']

    if probe_steps > 0:
        ft.mem.freeze_all()
        logger.log(f'Phase I: Probing {probe_steps} steps...')
        ft.probe(dl, steps=probe_steps)

        # 5) Phase II: choose top-t most-accessed slots, unfreeze and finetune
        slot_ids = ft.mem.top_slots(top_t)
        logger.log('unfreezing top-t slots:', len(slot_ids))
        ft.mem.unfreeze_slots(slot_ids)
    else:
        # Skip probing, make all slots trainable
        logger.log(f'Skipping probe phase, making all {top_t} slots trainable')
        slot_ids = list(range(top_t))
        ft.mem.unfreeze_slots(slot_ids)

    # Train with refresh_every parameter
    refresh_every = train_cfg.get('refresh_every', 200)
    logger.log(f'Phase II: Training {train_cfg["epochs"]} epochs with refresh_every={refresh_every}...')
    loss_history = ft.train(dl, epochs=train_cfg['epochs'], refresh_every=refresh_every)

    # 6) Collect training statistics
    import transformers
    train_stats = {
        # Environment and model info
        'base_model_id': base_model,
        'base_model_hash': hashlib.md5(base_model.encode()).hexdigest()[:8],
        'transformers_version': transformers.__version__,
        'torch_version': torch.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),

        # Training config
        'seed': train_cfg.get('seed', 42),
        'optimizer': 'AdamW',
        'lr_kv': train_cfg.get('lr_kv', train_cfg['lr']),
        'lr_gate': train_cfg.get('lr_gate', train_cfg['lr'] * 0.1),
        'scheduler': 'linear_warmup',
        'batch_size': train_cfg['batch_size'],
        'max_length': train_cfg['max_length'],
        'epochs': train_cfg['epochs'],

        # Data statistics
        'dataset': ds_name,
        'num_examples': len(ds),
        'total_steps': len(dl) * train_cfg['epochs'],
        'tokens_total': len(ds) * train_cfg['max_length'] * train_cfg['epochs'],

        # Memory config
        'mem_config': {
            'num_slots': mem_cfg['num_slots'],
            'k_top': mem_cfg['k_top'],
            'd_model': ft.mem.d_model,
            'alpha': mem_cfg.get('alpha', 1.0),
            'tau': mem_cfg.get('tau', 10.0),
            'use_gate': mem_cfg.get('use_gate', True),
        },

        # Sparse training config
        'sparse_config': {
            'top_t': top_t,
            'probe_steps': train_cfg.get('probe_steps', 1000),
            'refresh_every': train_cfg.get('refresh_every', 200),
            'specificity_method': 'tf-idf',
        },
    }

    # 7) Export patch (pass loss history)
    save_dir = train_cfg['save_dir']
    patch = ft.build_patch(task, top_t, save_dir, train_stats, loss_history=loss_history)

    # Use configurable output directory (defaults to 'out')
    patch_output_dir = train_cfg.get('patch_output_dir', 'out')
    ensure_dir(patch_output_dir)

    # Save patch and meta
    patch_path = os.path.join(patch_output_dir, f'patch_{task}.json')
    meta_path = os.path.join(patch_output_dir, f'patch_{task}_meta.json')
    dump_json(patch, patch_path)
    dump_json(train_stats, meta_path)
    logger.log(f'patch saved to {patch_output_dir}/', f'patch_{task}.json')

    # Save training log to file
    log_dir = os.path.join(save_dir, task)
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, 'training.log')
    
    with open(log_path, 'w') as f:
        f.write(f"Task: {task}\n")
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"Base Model: {base_model}\n")
        f.write(f"Training started: {train_stats['timestamp']}\n")
        f.write(f"\n{'='*60}\n")
        f.write("Configuration\n")
        f.write(f"{'='*60}\n")
        f.write(f"Batch size: {train_cfg['batch_size']}\n")
        f.write(f"Max length: {train_cfg['max_length']}\n")
        f.write(f"Epochs: {train_cfg['epochs']}\n")
        f.write(f"Learning rate (KV): {train_stats['lr_kv']}\n")
        f.write(f"Learning rate (gate): {train_stats['lr_gate']}\n")
        f.write(f"Memory slots: {mem_cfg['num_slots']}\n")
        f.write(f"Top-t: {top_t}\n")
        f.write(f"k_top: {mem_cfg['k_top']}\n")
        f.write(f"Use CoT: {use_cot}\n")
        f.write(f"\n{'='*60}\n")
        f.write("Loss History\n")
        f.write(f"{'='*60}\n")
        
        if loss_history:
            f.write("Step\tEpoch\tLoss\n")
            for record in loss_history:
                f.write(f"{record['step']}\t{record['epoch']}\t{record['loss']:.6f}\n")
            
            # Loss statistics
            losses = [x['loss'] for x in loss_history]
            f.write(f"\n{'='*60}\n")
            f.write("Loss Statistics\n")
            f.write(f"{'='*60}\n")
            f.write(f"Initial loss: {losses[0]:.6f}\n")
            f.write(f"Final loss: {losses[-1]:.6f}\n")
            f.write(f"Min loss: {min(losses):.6f}\n")
            f.write(f"Max loss: {max(losses):.6f}\n")
            f.write(f"Mean loss: {sum(losses)/len(losses):.6f}\n")
            f.write(f"Loss reduction: {losses[0] - losses[-1]:.6f}\n")
            f.write(f"Loss reduction %: {100*(losses[0] - losses[-1])/losses[0]:.2f}%\n")
    
    logger.log(f'Training log saved to {log_path}')


if __name__ == '__main__':
    fire.Fire(main)

