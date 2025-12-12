#!/usr/bin/env python
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import json
from src.sumcar.metrics import (
    compute_per_task_scores,
    compute_retention_metrics,
    compute_reversible_metrics,
    compute_routing_diagnostics,
)
from src.sumcar.metrics.auto_inputs import load_metric_map

with open("analysis/single_scores.json") as f:
    single_scores = json.load(f)
with open("analysis/merged_scores.json") as f:
    merged_scores = json.load(f)
with open("analysis/restored_scores.json") as f:
    restored_scores = json.load(f)
with open("analysis/routing_inputs.json") as f:
    routing_inputs = json.load(f)

tasks = ["gsm8k", "finqa", "humaneval"]


metric_keys = {
    "gsm8k": "accuracy",
    "finqa": "accuracy",
    "humaneval": "pass@1",
}


per_task = compute_per_task_scores(
    single_scores,
    merged_scores,
    metric_keys=metric_keys,
    tasks=tasks,
    round_to=4,
)

retention = compute_retention_metrics(
    single_scores,
    merged_scores,
    metric_keys=metric_keys,
    tasks=tasks,
    round_to=4,
)

reversible = compute_reversible_metrics(
    merged_scores,
    restored_scores,
    metric_keys=metric_keys,
    tasks=tasks,
    round_to=4,
)

routing = compute_routing_diagnostics(
    routing_inputs["home_hits"],
    routing_inputs["cross_hits"],
    util=routing_inputs.get("util"),
    conflict=routing_inputs.get("conflict"),
    tasks=tasks,
    round_to=4,
)

all_metrics = {
    "per_task": per_task,
    "retention": retention,
    "reversible": reversible,
    "routing": routing,
}

with open("analysis/sumcar_metrics_all.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("Saved to analysis/sumcar_metrics_all.json")

