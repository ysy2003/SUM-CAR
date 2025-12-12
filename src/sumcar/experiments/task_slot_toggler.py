# src/sumcar/experiments/task_slot_toggler.py
"""
Utilities for toggling (zero / restore) task-specific memory slots in KVMemoryLayer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch

from ..memory.kv_memory import KVMemoryLayer


def load_task_global_slots(
    task: str,
    patch_meta_path: str,
    remap_path: str,
) -> List[int]:
    patch_meta = json.loads(Path(patch_meta_path).read_text())
    remap_obj = json.loads(Path(remap_path).read_text())
    remap = remap_obj["remap"]

    if task not in patch_meta:
        raise KeyError(f"Task {task!r} not found in patch_meta.json")

    local_ids = patch_meta[task]["slot_ids"]  # e.g. 0..4095
    global_ids: List[int] = []

    for s in local_ids:
        key = f"{task}:{s}"
        if key not in remap:
            continue
        g = remap[key]
        global_ids.append(int(g))

    global_ids = sorted(set(global_ids))
    return global_ids


class TaskSlotToggler:

    def __init__(self, memory_layer: KVMemoryLayer, slot_ids: List[int]):
        self.memory = memory_layer
        device = memory_layer.keys.device

        # [n_slots_task]
        self.slot_idx = torch.as_tensor(slot_ids, dtype=torch.long, device=device)

        self._saved_keys: torch.Tensor | None = None
        self._saved_vals: torch.Tensor | None = None

    @torch.no_grad()
    def zero_slots(self) -> None:
        self._saved_keys = self.memory.keys[self.slot_idx].clone()
        self._saved_vals = self.memory.vals[self.slot_idx].clone()

        self.memory.keys[self.slot_idx].zero_()
        self.memory.vals[self.slot_idx].zero_()

    @torch.no_grad()
    def restore_slots(self) -> None:
        assert self._saved_keys is not None and self._saved_vals is not None, \
            "restore_slots called before zero_slots()"

        self.memory.keys[self.slot_idx].copy_(self._saved_keys)
        self.memory.vals[self.slot_idx].copy_(self._saved_vals)
        self._saved_keys = None
        self._saved_vals = None
