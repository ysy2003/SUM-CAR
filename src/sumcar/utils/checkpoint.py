"""
SUM-CAR Checkpoint Manager
管理三种 checkpoint：训练恢复用、补丁（patch）、合并产物
"""
import os
import json
import torch
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class TrainingState:
    """训练状态（用于恢复训练）"""
    global_step: int
    epoch: int
    samples_seen: int
    tokens_seen: int
    best_dev_score: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class CheckpointManager:
    """统一的 Checkpoint 管理器"""
    
    def __init__(self, base_dir: str, task: str, base_model_id: str):
        self.base_dir = Path(base_dir)
        self.task = task
        self.base_model_id = base_model_id
        self.base_hash = hashlib.md5(base_model_id.encode()).hexdigest()[:8]
        
        # 创建目录结构
        self.ckpt_dir = self.base_dir / "runs" / task / "ckpts"
        self.patch_dir = self.base_dir / "patches"
        self.merge_dir = self.base_dir / "merges"
        
        for d in [self.ckpt_dir, self.patch_dir, self.merge_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    # ==================== 1. 训练恢复用 Checkpoint ====================
    
    def save_training_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        training_state: TrainingState,
        config: Dict,
        scaler: Optional[Any] = None,
        is_best: bool = False,
        is_milestone: bool = False,
        milestone_name: str = None
    ):
        """
        保存完整的训练 checkpoint（可恢复训练）
        
        参数:
            model: 完整模型（包含基座和记忆层）
            optimizer: 优化器
            scheduler: 学习率调度器
            training_state: 训练状态
            config: 训练配置
            scaler: 混合精度缩放器
            is_best: 是否是最佳 checkpoint
            is_milestone: 是否是里程碑
            milestone_name: 里程碑名称
        """
        step = training_state.global_step
        
        # 确定保存路径
        if is_milestone and milestone_name:
            ckpt_path = self.ckpt_dir / f"milestone-{milestone_name}"
        elif is_best:
            ckpt_path = self.ckpt_dir / "best"
        else:
            ckpt_path = self.ckpt_dir / f"step-{step:07d}"
        
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 模型权重
        torch.save(model.state_dict(), ckpt_path / "model.pt")
        
        # 2. 优化器状态
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
        }, ckpt_path / "optimizer.pt")
        
        # 3. 混合精度状态
        if scaler:
            torch.save({'scaler': scaler.state_dict()}, ckpt_path / "scaler.pt")
        
        # 4. 随机数状态
        torch.save({
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, ckpt_path / "rng.pt")
        
        # 5. 训练状态和配置
        metadata = {
            'training_state': training_state.to_dict(),
            'config': config,
            'base_model_id': self.base_model_id,
            'base_hash': self.base_hash,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'torch_version': torch.__version__,
            'is_best': is_best,
            'is_milestone': is_milestone,
        }
        
        with open(ckpt_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 创建符号链接到 last
        last_link = self.ckpt_dir / "last"
        if last_link.exists():
            last_link.unlink()
        last_link.symlink_to(ckpt_path.name)
        
        # 清理旧 checkpoint（保留策略：last, last-1, best, milestone）
        self._cleanup_old_checkpoints()
        
        return ckpt_path
    
    def load_training_checkpoint(self, checkpoint_path: str = "last"):
        """
        加载训练 checkpoint
        
        参数:
            checkpoint_path: "last", "best", 或具体路径
        
        返回:
            包含所有状态的字典
        """
        if checkpoint_path in ["last", "best"]:
            ckpt_path = self.ckpt_dir / checkpoint_path
        else:
            ckpt_path = Path(checkpoint_path)
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        # 如果是符号链接，解析真实路径
        if ckpt_path.is_symlink():
            ckpt_path = ckpt_path.resolve()
        
        # 加载所有状态
        state = {
            'model': torch.load(ckpt_path / "model.pt", map_location='cpu'),
            'optimizer': torch.load(ckpt_path / "optimizer.pt", map_location='cpu'),
            'metadata': json.load(open(ckpt_path / "metadata.json")),
        }
        
        # 加载可选状态
        if (ckpt_path / "scaler.pt").exists():
            state['scaler'] = torch.load(ckpt_path / "scaler.pt", map_location='cpu')
        
        if (ckpt_path / "rng.pt").exists():
            state['rng'] = torch.load(ckpt_path / "rng.pt", map_location='cpu')
        
        return state
    
    def _cleanup_old_checkpoints(self, keep_last_n: int = 2):
        """清理旧的训练 checkpoint，保留最近 N 个 + best + milestones"""
        # 获取所有 step checkpoint
        step_ckpts = sorted([
            d for d in self.ckpt_dir.iterdir()
            if d.is_dir() and d.name.startswith('step-')
        ], key=lambda x: int(x.name.split('-')[1]))
        
        # 保留最近 N 个
        if len(step_ckpts) > keep_last_n:
            for old_ckpt in step_ckpts[:-keep_last_n]:
                import shutil
                shutil.rmtree(old_ckpt)
    
    # ==================== 2. 补丁（Patch）Checkpoint ====================
    
    def save_patch(
        self,
        slot_ids: List[int],
        keys: torch.Tensor,
        values: torch.Tensor,
        specificity: List[float],
        access_counts: List[int],
        memory_config: Dict,
        train_meta: Dict,
        stats: Dict
    ):
        """
        保存补丁 checkpoint（用于合并和推理）
        
        参数:
            slot_ids: 更新的槽位索引
            keys: K 向量 [t, d_k]
            values: V 向量 [t, d_v]
            specificity: 特异度分数
            access_counts: 访问计数
            memory_config: 记忆配置
            train_meta: 训练元信息
            stats: 统计信息
        """
        # 文件名：patch_<task>_<basehash>_<t>slots
        t = len(slot_ids)
        filename = f"patch_{self.task}_{self.base_hash}_{t}slots"
        
        # 1. 保存 K/V 张量（safetensors 格式更好，这里用 pt）
        patch_tensors = {
            'keys': keys.cpu(),
            'values': values.cpu(),
            'slot_ids': torch.tensor(slot_ids, dtype=torch.int32),
        }
        torch.save(patch_tensors, self.patch_dir / f"{filename}.pt")
        
        # 2. 保存元信息
        patch_meta = {
            'task': self.task,
            'slot_ids': slot_ids,
            'specificity': specificity,
            'access_counts': access_counts,
            
            # 记忆配置
            'memory_config': memory_config,
            
            # 基座引用
            'base_ref': {
                'base_model_id': self.base_model_id,
                'base_hash': self.base_hash,
            },
            
            # 训练元信息
            'train_meta': train_meta,
            
            # 统计信息
            'stats': stats,
            
            # 版本信息
            'patch_version': '1.0',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(self.patch_dir / f"{filename}.meta.json", 'w') as f:
            json.dump(patch_meta, f, indent=2)
        
        print(f"✅ Patch saved:")
        print(f"   Tensors: {self.patch_dir / f'{filename}.pt'}")
        print(f"   Metadata: {self.patch_dir / f'{filename}.meta.json'}")
        
        return self.patch_dir / f"{filename}.pt"
    
    def load_patch(self, patch_path: str):
        """
        加载补丁
        
        返回:
            包含张量和元数据的字典
        """
        patch_path = Path(patch_path)
        meta_path = patch_path.with_suffix('.meta.json')
        
        # 加载张量
        tensors = torch.load(patch_path, map_location='cpu')
        
        # 加载元数据
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'tensors': tensors,
            'metadata': metadata,
        }
    
    # ==================== 3. 合并产物 Checkpoint ====================
    
    def save_merge(
        self,
        merged_memory: torch.Tensor,
        merge_manifest: Dict,
        merge_report: List[Dict],
        remap_index: Dict,
        merge_name: str
    ):
        """
        保存合并产物（可逆）
        
        参数:
            merged_memory: 合并后的记忆张量
            merge_manifest: 合并清单
            merge_report: 逐槽合并报告
            remap_index: 重映射索引
            merge_name: 合并名称（如 "code+math+fin"）
        """
        merge_path = self.merge_dir / merge_name
        merge_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存合并后的记忆权重
        torch.save({
            'keys': merged_memory['keys'].cpu(),
            'values': merged_memory['vals'].cpu(),
        }, merge_path / "merged_memory.pt")
        
        # 2. 保存合并清单
        with open(merge_path / "merge_manifest.json", 'w') as f:
            json.dump(merge_manifest, f, indent=2)
        
        # 3. 保存合并报告（JSONL 格式，逐行记录）
        with open(merge_path / "merge_report.jsonl", 'w') as f:
            for record in merge_report:
                f.write(json.dumps(record) + '\n')
        
        # 4. 保存重映射索引
        with open(merge_path / "remap_index.json", 'w') as f:
            json.dump(remap_index, f, indent=2)
        
        print(f"✅ Merge saved:")
        print(f"   Path: {merge_path}")
        print(f"   Files: merged_memory.pt, manifest, report, remap_index")
        
        return merge_path
    
    def load_merge(self, merge_name: str):
        """
        加载合并产物
        
        返回:
            包含所有合并信息的字典
        """
        merge_path = self.merge_dir / merge_name
        
        if not merge_path.exists():
            raise FileNotFoundError(f"Merge not found: {merge_path}")
        
        # 加载所有组件
        merge = {
            'memory': torch.load(merge_path / "merged_memory.pt", map_location='cpu'),
            'manifest': json.load(open(merge_path / "merge_manifest.json")),
            'remap_index': json.load(open(merge_path / "remap_index.json")),
            'report': [],
        }
        
        # 加载报告（JSONL）
        with open(merge_path / "merge_report.jsonl", 'r') as f:
            for line in f:
                merge['report'].append(json.loads(line))
        
        return merge
