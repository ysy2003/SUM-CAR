"""
稀疏微调器
用于任务特定的记忆槽微调和 patch 生成
"""
import math
import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..models.base_model import MemoryAugmentedCausalLM
from .kv_memory import KVMemoryLayer
from .metrics import choose_top_t_from_counts
from ..utils.io import ensure_dir, dump_json
from ..utils.logger import Logger


class SparseFinetuner:
    """
    稀疏微调器
    
    功能：
    1. 在任务数据上探测（probe）记忆槽使用情况
    2. 选择 top-t 槽位进行微调
    3. 导出任务特定的 patch
    """
    
    def __init__(self, base_model: str, mem_cfg: Dict, train_cfg: Dict, tokenizer=None, logger=None):
        """
        参数:
            base_model: 基础模型名称或路径
            mem_cfg: 记忆配置字典 {'num_slots', 'k_top', 'alpha'}
            train_cfg: 训练配置字典 {'lr', ...}
            tokenizer: 分词器（可选）
            logger: 日志器（可选）
        """
        self.logger = logger or Logger('[train]')
        self.tok = tokenizer or AutoTokenizer.from_pretrained(base_model)
        
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        
        # 推断模型维度
        d_model = self._infer_d_model(base_model)
        
        # 创建记忆层
        self.mem = KVMemoryLayer(
            d_model=d_model,
            num_slots=mem_cfg['num_slots'],
            k_top=mem_cfg['k_top'],
            alpha=mem_cfg.get('alpha', 1.0)
        )
        
        # 创建增强模型
        self.model = MemoryAugmentedCausalLM(base_model, self.mem)
        self.cfg = train_cfg
    
    def _infer_d_model(self, base_model: str) -> int:
        """
        推断模型的隐藏维度
        
        参数:
            base_model: 模型名称或路径
        
        返回:
            隐藏维度大小
        """
        from transformers import AutoModelForCausalLM
        tmp = AutoModelForCausalLM.from_pretrained(base_model)
        return tmp.get_input_embeddings().weight.shape[1]
    
    def probe(self, dl: DataLoader, steps: int = 1000, device: str = None):
        """
        探测阶段：收集槽位访问统计
        
        参数:
            dl: 数据加载器
            steps: 最大探测步数
            device: 设备（默认自动选择）
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(device)
        self.model.eval()
        self.mem.enable_access_logging()
        
        cnt = 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)
                cnt += 1
                if cnt >= steps:
                    break
        
        self.mem.disable_access_logging()
        total_accesses = int(self.mem.acc_counts.sum().item())
        self.logger.log(f'Probe done; counted {total_accesses} accesses')
    
    def train(self, dl: DataLoader, epochs: int = 1, device: str = None):
        """
        训练阶段：微调选中的槽位
        
        参数:
            dl: 数据加载器
            epochs: 训练轮数
            device: 设备（默认自动选择）
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(device)
        self.model.train()
        
        # 优化器和调度器
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg['lr']
        )
        steps_per_epoch = len(dl)
        sch = get_linear_schedule_with_warmup(opt, 0, steps_per_epoch * epochs)
        
        for ep in range(epochs):
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                
                loss.backward()
                opt.step()
                sch.step()
                opt.zero_grad()
            
            self.logger.log(f'Epoch {ep + 1}/{epochs} done')
    
    def build_patch(self, task: str, top_t: int, out_dir: str) -> Dict:
        """
        构建并保存任务 patch
        
        参数:
            task: 任务名称
            top_t: 选择的槽位数量
            out_dir: 输出目录
        
        返回:
            patch 字典
        """
        # 选择 top-t 槽位
        slot_ids = choose_top_t_from_counts(self.mem.acc_counts, top_t)
        
        # 导出 patch
        patch = self.mem.get_patch(slot_ids)
        patch['task'] = task
        
        # 元信息
        meta = {
            'task': task,
            'top_t': top_t,
            'num_slots': self.mem.num_slots,
            'access_total': int(self.mem.acc_counts.sum().item())
        }
        
        # 保存
        ensure_dir(out_dir)
        dump_json(patch, os.path.join(out_dir, f'patch_{task}.json'))
        dump_json(meta, os.path.join(out_dir, f'patch_{task}_meta.json'))
        
        return patch
