"""Generate metric inputs directly from SUM-CAR resources."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from transformers import AutoTokenizer

from ..eval.quick import eval_finqa, eval_gsm8k, eval_humaneval
from ..memory.kv_memory import KVMemoryLayer
from ..models.base_model import MemoryAugmentedCausalLM

_TASK_EVAL_FNS = {
    "gsm8k": eval_gsm8k,
    "humaneval": eval_humaneval,
    "finqa": eval_finqa,
}

_TASK_NORMALIZATION = {
    "math": "gsm8k",
    "gsm8k": "gsm8k",
    "code": "humaneval",
    "codexglue": "humaneval",
    "humaneval": "humaneval",
    "finance": "finqa",
    "finqa": "finqa",
}

_DEFAULT_PATCH_PATTERNS = {
    "gsm8k": "/content/drive/MyDrive/SUM-CAR/noLoRA/memory_math.pt",
    "humaneval": "/content/drive/MyDrive/SUM-CAR/noLoRA/memory_code.pt",
    "finqa": "/content/drive/MyDrive/SUM-CAR/noLoRA/merged_finqa/memory.pt",
}


@dataclass
class AutoMetricsConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    k_top: int = 8
    alpha: float = 1.0
    use_fp16: bool = True
    max_samples: int = 100
    use_cot: bool = False


class MetricInputGenerator:
    """Generate all metrics inputs (single/merged/restored/routing) automatically."""

    def __init__(
        self,
        patch_map: Mapping[str, str],
        merged_memory: str,
        hits_csv: Optional[str],
        remap_csv: Optional[str],
        tasks: Iterable[str],
        config: AutoMetricsConfig,
    ) -> None:
        self.patch_map = dict(patch_map)
        self.merged_memory = merged_memory
        self.hits_csv = hits_csv
        self.remap_csv = remap_csv
        self.tasks = list(tasks)
        self.config = config

        missing = [t for t in self.tasks if t not in self.patch_map]
        if missing:
            raise ValueError(
                f"Missing patch tensors for tasks: {', '.join(missing)}. Provide them via --patches."
            )

        unsupported = [t for t in self.tasks if t not in _TASK_EVAL_FNS]
        if unsupported:
            raise ValueError(
                "Unsupported tasks for auto-eval: "
                + ", ".join(unsupported)
                + ". Available tasks: "
                + ", ".join(sorted(_TASK_EVAL_FNS))
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: Optional[MemoryAugmentedCausalLM] = None
        self._merged_state: Optional[dict] = None
        self._owner_map: dict[int, str] = {}
        self._single_cache: Optional[Dict[str, dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> dict:
        single_scores = self._compute_single_task_scores()
        merged_scores = self._evaluate_state(self._load_merged_state(), full_suite=True)
        restored_scores = self._compute_restored_scores()
        routing_inputs = self._compute_routing_inputs()

        return {
            "single": single_scores,
            "merged": merged_scores,
            "restored": restored_scores,
            "routing": routing_inputs,
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _compute_single_task_scores(self) -> Dict[str, dict]:
        if self._single_cache is not None:
            return self._single_cache

        results: Dict[str, dict] = {}
        for task, path in self.patch_map.items():
            if task not in self.tasks:
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(f"Patch tensor not found for task '{task}': {path}")
            state = self._load_patch_state(path)
            results[task] = self._evaluate_state(state, limit_to_task=task)
        self._single_cache = results
        return results

    def _compute_restored_scores(self) -> Dict[str, dict]:
        owners = self._load_owner_map()
        if not owners:
            return self._compute_single_task_scores()

        merged_state = self._load_merged_state()
        results: Dict[str, dict] = {}
        for task in self.tasks:
            filtered = self._filter_state_by_owner(merged_state, owners, task)
            results[task] = self._evaluate_state(filtered, limit_to_task=task)
        return results

    def _evaluate_state(
        self,
        state: dict,
        *,
        limit_to_task: Optional[str] = None,
        full_suite: bool = False,
    ) -> Dict[str, dict]:
        model = self._swap_memory(state)
        if limit_to_task:
            eval_tasks = [limit_to_task]
        elif full_suite:
            eval_tasks = list(self.tasks)
        else:
            eval_tasks = list(self.tasks)
        outputs: Dict[str, dict] = {}
        for task in eval_tasks:
            eval_fn = _TASK_EVAL_FNS.get(task)
            if eval_fn is None:
                continue
            kwargs = {"max_samples": self.config.max_samples}
            if task in ("gsm8k", "finqa"):
                kwargs["use_cot"] = self.config.use_cot
            outputs[task] = eval_fn(model, self.tokenizer, **kwargs)
        return outputs

    def _swap_memory(self, state: dict) -> MemoryAugmentedCausalLM:
        mem = self._state_to_memory_layer(state)
        if self.model is None:
            self.model = MemoryAugmentedCausalLM(
                self.config.base_model,
                mem,
                use_fp16=self.config.use_fp16,
            ).to(self.device)
        else:
            self.model.mem = mem
        self.model.eval()
        return self.model

    def _state_to_memory_layer(self, state: dict) -> KVMemoryLayer:
        keys = state["keys"].to(self.device)
        vals = state["vals"].to(self.device)
        mem = KVMemoryLayer(
            d_model=keys.shape[1],
            num_slots=keys.shape[0],
            k_top=self.config.k_top,
            alpha=self.config.alpha,
        ).to(self.device)
        if self.config.use_fp16:
            mem = mem.half()
        with torch.no_grad():
            mem.keys.data.copy_(keys.to(mem.keys.dtype))
            mem.vals.data.copy_(vals.to(mem.vals.dtype))
        return mem

    # ------------------------------------------------------------------
    # State loading utilities
    # ------------------------------------------------------------------

    def _load_patch_state(self, path: str) -> dict:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, (tuple, list)) and len(data) == 2:
            keys, vals = data
            return {"keys": keys, "vals": vals}
        if isinstance(data, Mapping):
            keys = None
            if "keys" in data:
                keys = data["keys"]
            elif "k" in data:
                keys = data["k"]

            if keys is None:
                raise ValueError(
                    f"Patch {path} does not contain 'keys' or 'k' tensor; "
                    f"available keys: {list(data.keys())}"
                )

            vals = None
            if "vals" in data:
                vals = data["vals"]
            elif "values" in data:
                vals = data["values"]
            elif "v" in data:
                vals = data["v"]

            if vals is None:
                raise ValueError(
                    f"Patch {path} does not contain 'vals'/'values'/'v' tensor; "
                    f"available keys: {list(data.keys())}"
                )

            return {"keys": keys, "vals": vals}

        raise TypeError(
            f"Unexpected patch format in {path}: type={type(data)}, "
            "expected (keys, vals) tuple or dict with 'keys'/'vals'."
        )



    def _load_merged_state(self) -> dict:
        if self._merged_state is None:
            if not os.path.exists(self.merged_memory):
                raise FileNotFoundError(f"Merged memory not found: {self.merged_memory}")
            data = torch.load(self.merged_memory, map_location="cpu")

            if isinstance(data, Mapping) and "keys" in data:
                keys = data["keys"]
                vals = data.get("vals")
                if vals is None:
                    vals = data.get("values")
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                keys, vals = data
            else:
                raise TypeError(
                    f"Unexpected merged memory format in {self.merged_memory}: "
                    f"type={type(data)}, keys={getattr(data, 'keys', lambda: [])()}"
                )

            if vals is None:
                raise ValueError(
                    f"Merged memory {self.merged_memory} does not contain 'vals' or 'values'."
                )

            self._merged_state = {"keys": keys, "vals": vals}

        return {
            "keys": self._merged_state["keys"].clone(),
            "vals": self._merged_state["vals"].clone(),
        }

    def _filter_state_by_owner(
        self,
        state: dict,
        owner_map: Mapping[int, str],
        keep_task: str,
    ) -> dict:
        keys = state["keys"].clone()
        vals = state["vals"].clone()

        for idx in range(keys.shape[0]):
            if owner_map.get(idx) != keep_task:
                keys[idx].zero_()
                vals[idx].zero_()

        return {"keys": keys, "vals": vals}



    # ------------------------------------------------------------------
    # Routing stats
    # ------------------------------------------------------------------
    def _compute_routing_inputs(self) -> dict:
        owner_map = self._load_owner_map()
        if not self.hits_csv or not os.path.exists(self.hits_csv):
            return {
                "home_hits": {},
                "cross_hits": {},
                "util": {},
                "conflict": {},
            }

        from collections import Counter, defaultdict

        home_hits = Counter()
        cross_hits = Counter()
        util_slots = defaultdict(set)
        conflict_slots = defaultdict(set)

        with open(self.hits_csv, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                task = row.get("task") or ""
                if not task:
                    continue
                try:
                    slot_id = int(row.get("slot_id", -1))
                except ValueError:
                    continue
                if slot_id < 0:
                    continue
                weight = float(row.get("weight", 1.0))

                owner = owner_map.get(slot_id)
                if owner_map and owner == task:
                    home_hits[task] += weight
                    util_slots[task].add(slot_id)
                elif owner_map and owner and owner != task:
                    cross_hits[task] += weight
                    conflict_slots[task].add(slot_id)
                else:
                    # Without owner info treat as home hit
                    home_hits[task] += weight
                    util_slots[task].add(slot_id)

        util_counts = {task: len(util_slots.get(task, set())) for task in self.tasks}
        conflict_counts = {task: len(conflict_slots.get(task, set())) for task in self.tasks}

        # Ensure presence even if zero
        for task in self.tasks:
            home_hits.setdefault(task, 0.0)
            cross_hits.setdefault(task, 0.0)

        return {
            "home_hits": dict(home_hits),
            "cross_hits": dict(cross_hits),
            "util": util_counts,
            "conflict": conflict_counts,
        }


def discover_patches(explicit: Optional[str]) -> Dict[str, str]:
    """Parse ``task=path`` pairs or fall back to default glob patterns."""
    patch_map: Dict[str, str] = {}

    if explicit:
        for part in explicit.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid --patch entry '{part}', expected task=path")
            task, path = part.split("=", 1)
            canon = normalize_task_name(task.strip())
            patch_map[canon] = path.strip()

    for task, pattern in _DEFAULT_PATCH_PATTERNS.items():
        if task in patch_map:
            continue
        matches = sorted(glob(pattern))
        if matches:
            patch_map[task] = matches[0]

    return patch_map


def normalize_task_name(name: str) -> str:
    return _TASK_NORMALIZATION.get(name.lower(), name.lower())


def load_metric_map(metric_map: Optional[str]) -> Dict[str, str]:
    if not metric_map:
        return {}
    if os.path.exists(metric_map):
        with open(metric_map, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(metric_map)
