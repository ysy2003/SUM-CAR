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
    
    def __init__(self, M: int, task_name: str = "task", idf_df_counts: Dict[int, int] = None):

        self.M = M
        self.task_name = task_name
        self.tf_counts = torch.zeros(M, dtype=torch.int32, device='cpu')
        self.idf_df_counts = idf_df_counts if idf_df_counts is not None else {}
    
    @torch.no_grad()
    def update_from_hits(self, hit_ids: torch.Tensor, chunk: int = 500_000):
        if hit_ids is None or len(hit_ids) == 0:
            return
        hit_ids = hit_ids.detach().to('cpu').reshape(-1).to(torch.int64)
        n = hit_ids.numel()
        for s in range(0, n, chunk):
            sub = hit_ids[s:s+chunk]
            counts = torch.bincount(sub, minlength=self.M)  # int64
            self.tf_counts.add_(counts[:self.M].to(torch.int32))  # in-place on CPU
    
    @torch.no_grad()
    def update_df_counts(self, threshold_ratio: float = 0.01):
        max_access = self.tf_counts.max().item()
        threshold = max(1, int(max_access * threshold_ratio))
        for slot_id in range(self.M):
            if self.tf_counts[slot_id] >= threshold:
                self.idf_df_counts[slot_id] = self.idf_df_counts.get(slot_id, 0) + 1
    
    @torch.no_grad()
    def specificity(self, total_tasks: int = 1, normalize: bool = True):
        # TF
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
        if normalize and tfidf.max() > 0:
            tfidf = tfidf / tfidf.max()
        
        return tfidf
    
    @torch.no_grad()
    def top_t(self, t: int, total_tasks: int = 1):
        score = self.specificity(total_tasks=total_tasks)
        return torch.topk(score, k=min(t, self.M)).indices
    
    def get_access_counts(self):
        return self.tf_counts.clone()
    
    def get_stats(self):
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
    if kv_layer.keys.grad is None or kv_layer.vals.grad is None:
        return
    
    mask = kv_layer._trainable_mask
    if device is not None:
        mask = mask.to(device)
    kv_layer.keys.grad[~mask] = 0
    kv_layer.vals.grad[~mask] = 0