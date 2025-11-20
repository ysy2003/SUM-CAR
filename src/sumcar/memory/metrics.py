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
    """
    特异度追踪器（TF-IDF 风格）
    
    把"槽位=term，任务=doc"的类比：
    - tf_t(s) = access_counts_t(s) / sum_s access_counts_t(s)
    - df(s) = 出现于多少个任务（设阈值=该任务 max access 的 1% 以滤噪）
    - idf(s) = log((1+T)/(1+df(s))) + 1（平滑）
    - spec_tfidf(s) = tf_t(s) * idf(s) → 线性归一化到 [0,1]
    """
    
    def __init__(self, M: int, task_name: str = "task", idf_df_counts: Dict[int, int] = None):
        """
        参数:
            M: 槽位总数
            task_name: 任务名称（用于标识）
            idf_df_counts: 全局 df 计数（槽位 -> 出现任务数）用于 IDF 计算
        """
        self.M = M
        self.task_name = task_name
        # 当前任务的访问计数（TF）
        self.tf_counts = torch.zeros(M, dtype=torch.int32, device='cpu')
        # 全局 DF 计数（跨任务）
        self.idf_df_counts = idf_df_counts if idf_df_counts is not None else {}
    
    @torch.no_grad()
    def update_from_hits(self, hit_ids: torch.Tensor, chunk: int = 500_000):
        """更新任务特定的访问频率（优化：CPU 端 bincount，避免 TPU HBM 分配）"""
        if hit_ids is None or len(hit_ids) == 0:
            return
        
        # 1) 拉到 CPU + 展平 + 转 int64 以便 bincount
        hit_ids = hit_ids.detach().to('cpu').reshape(-1).to(torch.int64)
        
        # 2) 分块统计，避免一次性占用太多 host RAM
        n = hit_ids.numel()
        for s in range(0, n, chunk):
            sub = hit_ids[s:s+chunk]
            # bincount 的长度固定为 M，不会膨胀
            counts = torch.bincount(sub, minlength=self.M)  # int64
            # 3) 累加到 CPU 侧的 int32 计数器
            self.tf_counts.add_(counts[:self.M].to(torch.int32))  # in-place on CPU
    
    @torch.no_grad()
    def update_df_counts(self, threshold_ratio: float = 0.01):
        """
        更新全局 DF 计数（标记当前任务中显著访问的槽位）
        
        参数:
            threshold_ratio: 阈值比例（相对于 max access）
        """
        max_access = self.tf_counts.max().item()
        threshold = max(1, int(max_access * threshold_ratio))
        
        # 标记超过阈值的槽位
        for slot_id in range(self.M):
            if self.tf_counts[slot_id] >= threshold:
                self.idf_df_counts[slot_id] = self.idf_df_counts.get(slot_id, 0) + 1
    
    @torch.no_grad()
    def specificity(self, total_tasks: int = 1, normalize: bool = True):
        """
        计算 TF-IDF 特异度分数
        
        参数:
            total_tasks: 总任务数（用于 IDF 计算）
            normalize: 是否归一化到 [0, 1]
        
        返回:
            特异度分数张量 [M]
        """
        # TF: 归一化的访问频率
        total_accesses = self.tf_counts.sum().item()
        if total_accesses == 0:
            return torch.zeros(self.M, dtype=torch.float32, device='cpu')
        
        tf = self.tf_counts.float() / total_accesses
        
        # IDF: log((1+T)/(1+df(s))) + 1
        idf = torch.ones(self.M, dtype=torch.float32, device='cpu')
        for slot_id in range(self.M):
            df = self.idf_df_counts.get(slot_id, 0)
            idf[slot_id] = torch.log(torch.tensor((1.0 + total_tasks) / (1.0 + df))) + 1.0
        
        # TF-IDF
        tfidf = tf * idf
        
        # 线性归一化到 [0, 1]
        if normalize and tfidf.max() > 0:
            tfidf = tfidf / tfidf.max()
        
        return tfidf
    
    @torch.no_grad()
    def top_t(self, t: int, total_tasks: int = 1):
        """获取特异度最高的 t 个槽位"""
        score = self.specificity(total_tasks=total_tasks)
        return torch.topk(score, k=min(t, self.M)).indices
    
    def get_access_counts(self):
        """获取访问计数"""
        return self.tf_counts.clone()
    
    def get_stats(self):
        """获取统计信息"""
        spec = self.specificity(total_tasks=1, normalize=True)
        return {
            'total_accesses': int(self.tf_counts.sum().item()),
            'unique_slots_accessed': int((self.tf_counts > 0).sum().item()),
            'max_access': int(self.tf_counts.max().item()),
            'spec_mean': float(spec.mean().item()),
            'spec_std': float(spec.std().item()),
            'spec_max': float(spec.max().item()),
            'spec_min': float(spec.min().item()),
        }


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