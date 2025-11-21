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
from ..utils.checkpoint import CheckpointManager, TrainingState


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
            normalize_retrieval=mem_cfg.get('normalize_retrieval', True),
            track_hits=mem_cfg.get('track_hits', None),
            hits_source=mem_cfg.get('hits_source', 'topk'),
            track_interval=mem_cfg.get('track_interval', 200)
        )
        
        # 创建增强模型
        self.model = MemoryAugmentedCausalLM(base_model, self.mem)
        self.cfg = train_cfg
        
        # 特异度追踪器（传入任务名称）
        task_name = train_cfg.get('dataset', 'unknown')
        self.spec_tracker = SpecificityTracker(
            M=mem_cfg['num_slots'],
            task_name=task_name
        )
    
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
                
                # 更新特异度追踪（低频统计，避免 XLA OOM）
                if cnt % 50 == 0:
                    if self.use_xla:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()  # 结算 TPU 计算
                    
                    # 触发轻量级统计（仅基于 k_top，CPU 侧）
                    self.mem.maybe_collect_hits_light()
                    hits = self.mem.pop_last_hits()
                    if hits is not None:
                        self.spec_tracker.update_from_hits(hits)
                    
                    if self.use_xla:
                        xm.mark_step()  # CPU 统计不进 XLA 图
                
                cnt += 1
                if cnt >= steps:
                    break
        
        self.mem.disable_access_logging()
        total_accesses = int(self.mem.acc_counts.sum().item())
        self.logger.log(f'Probe done; counted {total_accesses} accesses')
        
        # 将 mem.acc_counts 同步到 spec_tracker（因为 track_hits 可能被禁用）
        if total_accesses > 0:
            # 找出所有被访问的槽位
            accessed_slots = (self.mem.acc_counts > 0).nonzero(as_tuple=True)[0]
            if len(accessed_slots) > 0:
                # 模拟 hits 格式：重复每个槽位ID access_count 次
                # 但为了效率，我们直接更新 tf_counts
                for slot_id in accessed_slots:
                    count = int(self.mem.acc_counts[slot_id].item())
                    self.spec_tracker.tf_counts[slot_id] = count
                self.logger.log(f'Synced {len(accessed_slots)} accessed slots to spec_tracker')
    
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
                
                # 更新特异度（每 refresh_every 步统计一次）
                if step % refresh_every == 0:
                    if self.use_xla:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()  # 结算 TPU 计算
                    
                    # 触发轻量级统计（仅基于 k_top，CPU 侧）
                    self.mem.maybe_collect_hits_light()
                    hits = self.mem.pop_last_hits()
                    if hits is not None:
                        self.spec_tracker.update_from_hits(hits)
                    else:
                        # 如果 hits 不可用（track_hits=false），从 acc_counts 同步
                        accessed_slots = (self.mem.acc_counts > 0).nonzero(as_tuple=True)[0]
                        for slot_id in accessed_slots:
                            count = int(self.mem.acc_counts[slot_id].item())
                            self.spec_tracker.tf_counts[slot_id] = count
                    
                    if self.use_xla:
                        xm.mark_step()  # CPU 统计不进 XLA 图
                    
                    # 刷新 top-t 可训练槽位
                    top_t = self.cfg.get('top_t', 2048)
                    top_slots = self.spec_tracker.top_t(top_t).to(device)
                    self.mem.set_trainable_slots(top_slots.tolist())
                    self.logger.log(f'[step {step}] refresh Top-t={len(top_slots)}; loss={loss.item():.4f}')
            
            self.logger.log(f'Epoch {ep + 1}/{epochs} done')
    
    def build_patch(self, task: str, top_t: int, out_dir: str, train_stats: Dict = None, use_ckpt_manager: bool = True) -> Dict:
        """
        构建并保存任务 patch（使用特异度分数）
        
        参数:
            task: 任务名称
            top_t: 选择的槽位数量
            out_dir: 输出目录
            train_stats: 训练统计信息（可选）
            use_ckpt_manager: 是否使用新的 checkpoint 管理器
        
        返回:
            patch 字典
        """
        # 使用特异度追踪器选择 top-t 槽位
        slot_ids = self.spec_tracker.top_t(top_t, total_tasks=1).tolist()
        
        # 计算 specificity 分数（TF-IDF，归一化到 [0,1]）
        specificity_all = self.spec_tracker.specificity(total_tasks=1, normalize=True)
        specificity = specificity_all[slot_ids].tolist()
        
        # 获取统计信息
        spec_stats = self.spec_tracker.get_stats()
        
        # 槽位访问统计
        access_counts = self.spec_tracker.get_access_counts()
        access_counts_list = [int(access_counts[sid]) for sid in slot_ids]
        
        # 获取 K/V 张量
        keys = self.mem.keys[slot_ids]
        values = self.mem.vals[slot_ids]
        
        # 如果使用新的 checkpoint 管理器
        if use_ckpt_manager and hasattr(self, 'ckpt_manager'):
            # 记忆配置
            memory_config = {
                'num_slots': self.mem.num_slots,
                'd_model': self.mem.d_model,
                'k_top': self.mem.k_top,
                'alpha': self.mem.alpha,
                'tau': self.mem.tau,
                'use_gate': self.mem.use_gate,
            }
            
            # 统计信息
            stats = {
                'access_total': spec_stats['total_accesses'],
                'unique_slots_accessed': spec_stats['unique_slots_accessed'],
                'specificity_stats': {
                    'max': spec_stats['spec_max'],
                    'min': spec_stats['spec_min'],
                    'mean': spec_stats['spec_mean'],
                    'std': spec_stats['spec_std'],
                    'top_t_min': float(specificity_all[slot_ids].min().item()),
                },
                'access_stats': {
                    'total': spec_stats['total_accesses'],
                    'max': spec_stats['max_access'],
                    'unique_accessed': spec_stats['unique_slots_accessed'],
                    'top_t_total': sum(access_counts_list),
                },
                # IDF 信息（用于重现实验）
                'idf_df_counts': self.spec_tracker.idf_df_counts,
            }
            
            # 使用 checkpoint 管理器保存
            self.ckpt_manager.save_patch(
                slot_ids=slot_ids,
                keys=keys,
                values=values,
                specificity=specificity,
                access_counts=access_counts_list,
                memory_config=memory_config,
                train_meta=train_stats or {},
                stats=stats
            )
        
        # 同时保存旧格式（兼容性）
        patch = self.mem.get_patch(slot_ids)
        patch['task'] = task
        patch['specificity'] = specificity
        patch['access_counts'] = access_counts_list
        
        # 元信息
        meta = {
            'task': task,
            'top_t': top_t,
            'num_slots': self.mem.num_slots,
            
            # TF-IDF 统计
            'specificity_stats': {
                'max': spec_stats['spec_max'],
                'min': spec_stats['spec_min'],
                'mean': spec_stats['spec_mean'],
                'std': spec_stats['spec_std'],
                'top_t_min': float(specificity_all[slot_ids].min().item()),
            },
            
            # 访问统计
            'access_stats': {
                'total': spec_stats['total_accesses'],
                'max': spec_stats['max_access'],
                'unique_accessed': spec_stats['unique_slots_accessed'],
                'top_t_total': sum(access_counts_list),
            },
            
            # IDF 信息（用于重现实验）
            'idf_df_counts': self.spec_tracker.idf_df_counts,
        }
        
        # 合并训练统计
        if train_stats:
            meta.update(train_stats)
        
        # 保存旧格式
        ensure_dir(out_dir)
        dump_json(patch, os.path.join(out_dir, f'patch_{task}.json'))
        dump_json(meta, os.path.join(out_dir, f'patch_{task}_meta.json'))
        
        return patch
