"""
稀疏微调器
用于任务特定的记忆槽微调和 patch 生成
支持 TPU/XLA 训练
"""
import math
import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from ..models.base_model import MemoryAugmentedCausalLM
from .kv_memory import KVMemoryLayer
from .metrics import choose_top_t_from_counts, SpecificityTracker, mask_kv_grads
from ..utils.io import ensure_dir, dump_json
from ..utils.logger import Logger


class SparseFinetuner:
    """
    稀疏微调器
    
    功能：
    1. 在任务数据上探测（probe）记忆槽使用情况
    2. 选择 top-t 槽位进行微调
    3. 导出任务特定的 patch
    4. 支持 TPU/XLA 训练（通过 use_xla 标志）
    """
    
    def __init__(self, base_model: str, mem_cfg: Dict, train_cfg: Dict, tokenizer=None, logger=None, use_xla: bool = False):
        """
        参数:
            base_model: 基础模型名称或路径
            mem_cfg: 记忆配置字典 {'num_slots', 'k_top', 'alpha', 'tau'}
            train_cfg: 训练配置字典 {'lr', ...}
            tokenizer: 分词器（可选）
            logger: 日志器（可选）
            use_xla: 是否使用 XLA（TPU 训练）
        """
        self.logger = logger or Logger('[train]')
        self.tok = tokenizer or AutoTokenizer.from_pretrained(base_model)
        self.use_xla = use_xla
        
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        
        # 推断模型维度
        d_model = self._infer_d_model(base_model)
        
        # 创建记忆层（增强版，带门控）
        self.mem = KVMemoryLayer(
            d_model=d_model,
            num_slots=mem_cfg['num_slots'],
            k_top=mem_cfg['k_top'],
            alpha=mem_cfg.get('alpha', 1.0),
            tau=mem_cfg.get('tau', 10.0),
            use_gate=mem_cfg.get('use_gate', True),
            normalize_retrieval=mem_cfg.get('normalize_retrieval', True)
        )
        
        # 创建增强模型
        self.model = MemoryAugmentedCausalLM(base_model, self.mem)
        self.cfg = train_cfg
        
        # 特异度追踪器
        self.spec_tracker = SpecificityTracker(M=mem_cfg['num_slots'])
    
    def _infer_d_model(self, base_model: str) -> int:
        """
        推断模型的隐藏维度
        
        参数:
            base_model: 模型名称或路径
        
        返回:
            隐藏维度大小
        """
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
            if self.use_xla:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(device)
        self.model.eval()
        self.mem.enable_access_logging()
        
        cnt = 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)
                
                # 更新特异度追踪
                hits = self.mem.pop_last_hits()
                if hits is not None:
                    self.spec_tracker.update_from_hits(hits)
                
                cnt += 1
                if cnt >= steps:
                    break
        
        self.mem.disable_access_logging()
        total_accesses = int(self.mem.acc_counts.sum().item())
        self.logger.log(f'Probe done; counted {total_accesses} accesses')
    
    def train(self, dl: DataLoader, epochs: int = 1, device: str = None, refresh_every: int = 200):
        """
        训练阶段：微调选中的槽位
        
        参数:
            dl: 数据加载器
            epochs: 训练轮数
            device: 设备（默认自动选择）
            refresh_every: 每隔多少步刷新 top-t 槽位
        """
        if device is None:
            if self.use_xla:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(device)
        self.model.train()
        
        # 优化器：分离 KV 参数和门控参数
        kv_params = [self.mem.keys, self.mem.vals]
        gate_params = list(self.mem.W_q.parameters())
        if self.mem.use_gate:
            gate_params.extend(list(self.mem.gate.parameters()))
        
        opt = torch.optim.AdamW([
            {'params': kv_params, 'lr': self.cfg.get('lr_kv', self.cfg['lr'])},
            {'params': gate_params, 'lr': self.cfg.get('lr_gate', self.cfg['lr'] * 0.1)},
        ])
        
        steps_per_epoch = len(dl)
        total_steps = steps_per_epoch * epochs
        sch = get_linear_schedule_with_warmup(opt, 0, total_steps)
        
        step = 0
        for ep in range(epochs):
            for batch in dl:
                step += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                
                loss.backward()
                
                # 稀疏梯度 mask
                mask_kv_grads(self.mem, device)
                
                # XLA 优化器步骤
                if self.use_xla:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(opt, barrier=True)
                else:
                    opt.step()
                
                sch.step()
                opt.zero_grad(set_to_none=True)
                
                # 更新特异度
                hits = self.mem.pop_last_hits()
                if hits is not None:
                    self.spec_tracker.update_from_hits(hits)
                
                # 定期刷新 top-t 可训练槽位
                if step % refresh_every == 0:
                    top_t = self.cfg.get('top_t', 2048)
                    top_slots = self.spec_tracker.top_t(top_t).to(device)
                    self.mem.set_trainable_slots(top_slots.tolist())
                    self.logger.log(f'[step {step}] refresh Top-t={len(top_slots)}; loss={loss.item():.4f}')
            
            self.logger.log(f'Epoch {ep + 1}/{epochs} done')
    
    def build_patch(self, task: str, top_t: int, out_dir: str) -> Dict:
        """
        构建并保存任务 patch（使用特异度分数）
        
        参数:
            task: 任务名称
            top_t: 选择的槽位数量
            out_dir: 输出目录
        
        返回:
            patch 字典
        """
        # 使用特异度追踪器选择 top-t 槽位
        slot_ids = self.spec_tracker.top_t(top_t).tolist()
        
        # 导出 patch
        patch = self.mem.get_patch(slot_ids)
        patch['task'] = task
        patch['specificity'] = self.spec_tracker.specificity()[slot_ids].tolist()
        
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
