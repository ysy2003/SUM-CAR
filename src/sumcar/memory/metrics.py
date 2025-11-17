from typing import Dict, List


def slot_growth(prev_slots: int, new_slots: int) -> float:
    return (new_slots - prev_slots) / max(prev_slots, 1)


def choose_top_t_from_counts(counts, t: int) -> List[int]:
    # counts: torch tensor or list
    import torch
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts)
    _, idx = torch.topk(counts, k=min(t, counts.numel()))
    return idx.cpu().tolist()