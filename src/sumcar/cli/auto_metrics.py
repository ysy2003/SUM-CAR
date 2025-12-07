"""CLI to auto-generate metric inputs and compute SUM-CAR evaluation tables."""
from __future__ import annotations

import json
import os
from typing import Optional

import fire

from ..metrics import (
    compute_per_task_scores,
    compute_retention_metrics,
    compute_reversible_metrics,
    compute_routing_diagnostics,
)
from ..metrics.auto_inputs import (
    AutoMetricsConfig,
    MetricInputGenerator,
    discover_patches,
    load_metric_map,
    normalize_task_name,
)


def _parse_tasks(raw: Optional[str], fallback: list[str]) -> list[str]:
    if raw:
        tasks = [normalize_task_name(part.strip()) for part in raw.split(",") if part.strip()]
        if tasks:
            return tasks
    return [normalize_task_name(t) for t in fallback]


def _dump_inputs(dump_dir: Optional[str], inputs: dict) -> Optional[dict]:
    if not dump_dir:
        return None
    os.makedirs(dump_dir, exist_ok=True)
    paths = {}
    for key, value in inputs.items():
        path = os.path.join(dump_dir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
        paths[key] = path
    return paths


def main(
    *,
    out: str = "out/metrics_v2.json",
    tasks: str = "gsm8k,humaneval,finqa",
    patches: str = "",
    merged_memory: str = "out/merged/memory.pt",
    hits_csv: str = "out/logs/memory_hits.csv",
    remap_csv: str = "out/logs/remap_events.csv",
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    k_top: int = 8,
    alpha: float = 1.0,
    max_samples: int = 100,
    use_cot: bool = False,
    use_fp16: bool = True,
    metric_map: str | None = None,
    dump_dir: str = "out/metric_inputs",
) -> None:
    """Auto-run evaluations + metrics computation.

    Args mirror defaults in the repo layout; override them if your files live elsewhere.
    """

    patch_map = discover_patches(patches)
    if not patch_map:
        raise ValueError("No patch tensors found. Provide them via --patches task=path or train patches first.")

    task_list = _parse_tasks(tasks, list(patch_map.keys()))
    cfg = AutoMetricsConfig(
        base_model=base_model,
        k_top=k_top,
        alpha=alpha,
        use_fp16=use_fp16,
        max_samples=max_samples,
        use_cot=use_cot,
    )

    generator = MetricInputGenerator(
        patch_map=patch_map,
        merged_memory=merged_memory,
        hits_csv=hits_csv,
        remap_csv=remap_csv,
        tasks=task_list,
        config=cfg,
    )
    inputs = generator.run()

    dump_paths = _dump_inputs(dump_dir, inputs)
    metric_keys = load_metric_map(metric_map)

    per_task = compute_per_task_scores(
        inputs["single"],
        inputs["merged"],
        metric_keys=metric_keys,
        tasks=task_list,
    )
    retention = compute_retention_metrics(
        inputs["single"],
        inputs["merged"],
        metric_keys=metric_keys,
        tasks=task_list,
    )
    reversible = compute_reversible_metrics(
        inputs["merged"],
        inputs["restored"],
        metric_keys=metric_keys,
        tasks=task_list,
    )

    routing_counts = inputs.get("routing", {})
    if routing_counts.get("home_hits"):
        routing = compute_routing_diagnostics(
            routing_counts.get("home_hits", {}),
            routing_counts.get("cross_hits", {}),
            util=routing_counts.get("util"),
            conflict=routing_counts.get("conflict"),
            tasks=task_list,
        )
    else:
        routing = {}

    payload = {
        "per_task": per_task,
        "retention": retention,
        "reversible": reversible,
        "routing": routing,
        "input_files": dump_paths,
        "tasks": task_list,
        "patch_map": patch_map,
        "config": {
            "base_model": base_model,
            "k_top": k_top,
            "alpha": alpha,
            "max_samples": max_samples,
            "use_cot": use_cot,
        },
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[auto_metrics] Metrics saved to {out}")


if __name__ == "__main__":
    fire.Fire(main)
