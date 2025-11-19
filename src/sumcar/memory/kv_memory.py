"""
KV Memory Layer for SUM-CAR
基于 key-value 检索的记忆层，支持 top-k 路由和稀疏更新
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


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
        log_access: bool = True,
        tau: float = 10.0,
        use_gate: bool = True,
        normalize_retrieval: bool = True
    ):
        """
        参数:
            d_model: 模型维度
            num_slots: 记忆槽位数量
            k_top: Top-k 检索数量
            alpha: 输出缩放因子
            log_access: 是否记录访问统计
            tau: 温度参数（用于 softmax）
            use_gate: 是否使用门控机制
            normalize_retrieval: 是否归一化 query 和 key
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        self.k_top = k_top
        self.alpha = alpha
        self._log_access = log_access
        self.tau = tau
        self.normalize_retrieval = normalize_retrieval
        
        # Keys 和 Values
        self.keys = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.vals = nn.Parameter(torch.zeros(num_slots, d_model))
        
        # Query 投影（用于更好的检索）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.W_q.weight)  # 初始化为单位矩阵
        
        # 门控机制（标量门控）
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Linear(d_model, 1, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)  # 初始门控很小
        
        # 访问计数（不参与梯度）
        self.register_buffer('acc_counts', torch.zeros(num_slots, dtype=torch.long))
        
        # 可训练掩码（用于稀疏训练）
        self.register_buffer('_trainable_mask', torch.zeros(num_slots, dtype=torch.bool))
        
        # 最近命中的槽位（用于特异度追踪）
        self._last_hits = None
    
    def forward(self, x, record_hits: bool = True):
        """
        前向传播
        
        参数:
            x: [B, L, D] 输入张量
            record_hits: 是否记录命中（用于特异度追踪）
        
        返回:
            [B, L, D] 输出张量
        """
        # x: [B, L, D]
        B, L, D = x.shape
        
        # Query 投影
        q = self.W_q(x)  # [B, L, D]
        
        # 归一化（如果启用）
        if self.normalize_retrieval:
            q = F.normalize(q, dim=-1)
            keys = F.normalize(self.keys, dim=-1)
        else:
            keys = self.keys
        
        # 计算相似度 [B, L, M]
        scores = torch.einsum('bld,md->blm', q, keys) / (D ** 0.5)
        
        # Top-k 检索（使用 int32 索引降低内存）
        topv, topi_i64 = torch.topk(scores, k=min(self.k_top, self.num_slots), dim=-1)  # [B, L, K]
        topi = topi_i64.to(torch.int32)  # 减少 XLA 内存占用
        
        # 注意力权重（带温度）
        attn = F.softmax(topv / self.tau, dim=-1)  # [B, L, K]
        
        # 获取选中的 values
        sel_vals = self.vals[topi]  # [B, L, K, D]
        
        # 加权求和
        out = torch.einsum('blk,blkd->bld', attn, sel_vals)  # [B, L, D]
        
        # 门控融合
        if self.use_gate:
            g = torch.sigmoid(self.gate(x))  # [B, L, 1]
            out = g * out
        
        # 记录访问统计
        if self._log_access:
            with torch.no_grad():
                flat = topi.reshape(-1).to(torch.int64)  # bincount 需要 int64
                binc = torch.bincount(flat, minlength=self.num_slots)
                self.acc_counts += binc.to(self.acc_counts.device)
        
        # 记录最近命中（用于特异度追踪）
        if self.training and record_hits:
            with torch.no_grad():
                # 1) 释放可能的大中间量，避免引用保留在内存图
                import gc
                for name in ["scores", "topv", "attn", "sel_vals", "out", "q", "keys"]:
                    if name in locals():
                        try:
                            del locals()[name]
                        except:
                            pass
                gc.collect()
                
                # 2) 绝对不要在 TPU 上改 dtype！只 reshape
                flat = topi.detach().reshape(-1)
                
                # 3) 分块拷到 CPU（避免一次性占用 HBM）
                CHUNK = 262_144  # 256k 索引一块
                host_parts = []
                for sub in torch.split(flat, CHUNK):
                    # 只做 .cpu()，不做类型转换
                    host_parts.append(sub.cpu())
                
                # 4) 在 CPU 上再做 dtype 转换 + 直方图统计（等价 unique）
                hit_ids = torch.cat(host_parts, dim=0).to(torch.int64)  # 只在 CPU 上转 dtype
                
                # 用 bincount 得到每个 slot 是否被命中，再取非零索引
                counts = torch.bincount(hit_ids, minlength=self.num_slots)  # CPU
                unique_ids = torch.nonzero(counts, as_tuple=False).reshape(-1).to(torch.int64)
                
                # 保持在 CPU，后续用到时再 .to(device)
                self._last_hits = unique_ids
        
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
    
    def pop_last_hits(self):
        """获取并清除最近命中的槽位"""
        hits = self._last_hits
        self._last_hits = None
        return hits
    
    def freeze_all(self):
        """冻结所有参数"""
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze_slots(self, slot_ids: List[int]):
        """解冻指定槽位（用于稀疏训练）"""
        self.set_trainable_slots(slot_ids)
        # 注意：keys 和 vals 总是可训练的，通过梯度 mask 来控制
        for p in self.parameters():
            p.requires_grad = True
    
    def top_slots(self, t: int) -> List[int]:
        """获取访问频率最高的 t 个槽位"""
        return torch.topk(self.acc_counts, k=min(t, self.num_slots)).indices.tolist()
    
    def extra_repr(self) -> str:
        """额外的表示信息"""
        return (f'd_model={self.d_model}, num_slots={self.num_slots}, '
                f'k_top={self.k_top}, alpha={self.alpha}, tau={self.tau}')
