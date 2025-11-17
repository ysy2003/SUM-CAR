"""
KV Memory Layer for SUM-CAR
基于 key-value 检索的记忆层，支持 top-k 路由和稀疏更新
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class KVMemoryLayer(nn.Module):
    """
    Key-Value Memory Layer
    
    特性：
    - Top-k 检索机制
    - 访问计数统计
    - Patch 导出/应用
    - 动态槽位扩展
    - 稀疏训练支持
    """
    
    def __init__(
        self,
        d_model: int,
        num_slots: int = 200000,
        k_top: int = 32,
        alpha: float = 1.0,
        log_access: bool = True
    ):
        """
        参数:
            d_model: 模型维度
            num_slots: 记忆槽位数量
            k_top: Top-k 检索数量
            alpha: 输出缩放因子
            log_access: 是否记录访问统计
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.k_top = k_top
        self.alpha = alpha
        self._log_access = log_access
        
        # Keys 和 Values
        self.keys = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.vals = nn.Parameter(torch.zeros(num_slots, d_model))
        
        # 访问计数（不参与梯度）
        self.register_buffer('acc_counts', torch.zeros(num_slots, dtype=torch.long))
        
        # 可训练掩码（用于稀疏训练）
        self.register_buffer('_trainable_mask', torch.zeros(num_slots, dtype=torch.bool))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [B, L, D] 输入张量
        
        返回:
            [B, L, D] 输出张量
        """
        # x: [B, L, D]
        B, L, D = x.shape
        q = x.reshape(B * L, D)  # [BL, D]
        
        # Top-k 检索
        scores = torch.matmul(q, self.keys.t())  # [BL, S]
        topv, topi = torch.topk(scores, k=min(self.k_top, self.num_slots), dim=-1)
        
        # 注意力权重
        attn = F.softmax(topv, dim=-1)  # [BL, K]
        
        # 获取选中的 values
        sel_vals = self.vals[topi]  # [BL, K, D]
        
        # 加权求和
        out = torch.sum(attn.unsqueeze(-1) * sel_vals, dim=1)  # [BL, D]
        
        # 记录访问统计
        if self._log_access:
            with torch.no_grad():
                flat = topi.reshape(-1)
                binc = torch.bincount(flat, minlength=self.num_slots)
                self.acc_counts += binc.to(self.acc_counts.device)
        
        out = out.reshape(B, L, D)
        return self.alpha * out
    
    def get_patch(self, slot_ids: List[int]) -> Dict:
        """
        导出 patch
        
        参数:
            slot_ids: 要导出的槽位 ID 列表
        
        返回:
            包含 slot_ids, keys, vals, access_counts 的字典
        """
        with torch.no_grad():
            return {
                'slot_ids': [int(i) for i in slot_ids],
                'keys': self.keys[slot_ids].cpu().tolist(),
                'vals': self.vals[slot_ids].cpu().tolist(),
                'access_counts': [int(self.acc_counts[i]) for i in slot_ids]
            }
    
    @torch.no_grad()
    def apply_patch(self, patch: Dict):
        """
        应用 patch
        
        参数:
            patch: 包含 slot_ids, keys, vals 的字典
        """
        for sid, k, v in zip(patch['slot_ids'], patch['keys'], patch['vals']):
            self.keys[sid] = torch.tensor(k, device=self.keys.device, dtype=self.keys.dtype)
            self.vals[sid] = torch.tensor(v, device=self.vals.device, dtype=self.vals.dtype)
    
    @torch.no_grad()
    def expand_slots(self, add_n: int):
        """
        动态扩展槽位
        
        参数:
            add_n: 要添加的槽位数量
        """
        if add_n <= 0:
            return
        
        device, dtype = self.keys.device, self.keys.dtype
        
        # 创建新的 keys 和 vals
        new_k = torch.randn(add_n, self.d_model, device=device, dtype=dtype) * 0.02
        new_v = torch.zeros(add_n, self.d_model, device=device, dtype=dtype)
        
        # 拼接
        self.keys = nn.Parameter(torch.cat([self.keys, new_k], dim=0))
        self.vals = nn.Parameter(torch.cat([self.vals, new_v], dim=0))
        
        # 扩展访问计数和掩码
        self.acc_counts = torch.cat([
            self.acc_counts,
            torch.zeros(add_n, dtype=torch.long, device=device)
        ], dim=0)
        
        self._trainable_mask = torch.cat([
            self._trainable_mask,
            torch.zeros(add_n, dtype=torch.bool, device=device)
        ])
        
        self.num_slots += add_n
    
    def set_trainable_slots(self, slot_ids: List[int]):
        """
        设置可训练的槽位（用于稀疏训练）
        
        参数:
            slot_ids: 可训练的槽位 ID 列表
        """
        self._trainable_mask.zero_()
        self._trainable_mask[slot_ids] = True
    
    def get_trainable_slots(self) -> List[int]:
        """
        获取当前可训练的槽位 ID
        
        返回:
            槽位 ID 列表
        """
        return torch.where(self._trainable_mask)[0].tolist()
    
    def reset_access_counts(self):
        """重置访问计数"""
        self.acc_counts.zero_()
    
    def get_access_counts(self) -> torch.Tensor:
        """
        获取访问计数
        
        返回:
            [num_slots] 访问计数张量
        """
        return self.acc_counts.clone()
    
    def enable_access_logging(self):
        """启用访问统计"""
        self._log_access = True
    
    def disable_access_logging(self):
        """禁用访问统计"""
        self._log_access = False
    
    def extra_repr(self) -> str:
        """额外的表示信息"""
        return (f'd_model={self.d_model}, num_slots={self.num_slots}, '
                f'k_top={self.k_top}, alpha={self.alpha}')
