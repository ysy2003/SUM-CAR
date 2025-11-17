import os, yaml 
import fire 
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer 
from datasets import Dataset 
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

LOADERS = {
    'gsm8k': lambda split: gsm.load(split),
    'codexglue_refine': lambda split: cglue.load(split),
    'mbpp': lambda split: mbpp.load(split),
    'finqa_rc': lambda split: finqa.load(split),
}

def main(task: str, config: str):
    """Train a per-task sparse-memory skill patch.

    Args:
        task: short name for this patch (e.g., 'math' / 'code' / 'finqa').
        config: path to YAML config (contains base_model, mem, train).
    Outputs:
        out/patch_{task}.json         — serialized patch (slot_ids, keys, vals, access_counts)
        out/patch_{task}_meta.json    — metadata (task, slot_ids, stats)
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

