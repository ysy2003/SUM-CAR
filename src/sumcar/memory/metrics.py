from typing import Dict, List
import torch


def slot_growth(prev_slots: int, new_slots: int) -> float:
    return (new_slots - prev_slots) / max(prev_slots, 1)


def choose_top_t_from_counts(counts, t: int) -> List[int]:
    # counts: torch tensor or list
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts)
    _, idx = torch.topk(counts, k=min(t, counts.numel()))
    return idx.cpu().tolist()


class SpecificityTracker:
    """特异度追踪器（用于 TF-IDF 风格的槽位选择）"""
    
    def __init__(self, M: int):
        self.M = M
        self.tf = torch.zeros(M, dtype=torch.long)   # 当前任务命中次数
        self.bg = torch.ones(M, dtype=torch.long)    # 背景/先验（避免除0）
    
    @torch.no_grad()
    def update_from_hits(self, hit_ids: torch.Tensor):
        """更新任务特定的访问频率"""
        if hit_ids is None or len(hit_ids) == 0:
            return
        self.tf.index_add_(0, hit_ids.cpu(), torch.ones_like(hit_ids.cpu(), dtype=torch.long))
    
    @torch.no_grad()
    def specificity(self):
        """计算特异度分数（TF-IDF 风格）"""
        N_all = (self.tf + self.bg).sum().item()
        score = self.tf.float() * torch.log((torch.tensor(N_all + 1.0)) / (self.bg.float() + 1.0))
        return score
    
    @torch.no_grad()
    def top_t(self, t: int):
        """获取特异度最高的 t 个槽位"""
        score = self.specificity()
        return torch.topk(score, k=min(t, self.M)).indices


def mask_kv_grads(kv_layer, device=None):
    """稀疏梯度 mask：只让 trainable_mask 中的槽位更新"""
    if kv_layer.keys.grad is None or kv_layer.vals.grad is None:
        return
    
    mask = kv_layer._trainable_mask
    if device is not None:
        mask = mask.to(device)
    
    # 把不在 trainable_mask 的槽位梯度置零
    kv_layer.keys.grad[~mask] = 0
    kv_layer.vals.grad[~mask] = 0