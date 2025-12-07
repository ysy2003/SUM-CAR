"""Per-task evaluation helpers.

Implements the metric family defined as::

	S_single(T): best single-task score for task T
	S_merged(T): merged-model score for task T

The function below keeps the implementation generic so it can be reused across
Code/Math/Finance or any other task trio.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ._utils import extract_score, resolve_tasks, round_value

__all__ = ["compute_per_task_scores"]


def compute_per_task_scores(
	single_scores: Mapping[str, Any],
	merged_scores: Mapping[str, Any],
	*,
	metric_keys: Optional[Mapping[str, str]] = None,
	tasks: Optional[Iterable[str]] = None,
	round_to: Optional[int] = 4,
) -> dict[str, dict[str, float]]:
	"""Compute ``S_single`` and ``S_merged`` for every task.

	Args:
		single_scores: Mapping of task -> standalone evaluation payload.
		merged_scores: Mapping of task -> merged-model evaluation payload.
		metric_keys: Optional mapping of task -> metric key to extract
			(use ``"__default__"`` for a fallback key shared by all tasks).
		tasks: Optional ordered iterable of tasks to evaluate. If ``None`` the
			function will infer the ordering from *single_scores*.
		round_to: Optional number of decimal places for presentation.

	Returns:
		``dict`` keyed by task with ``{"single": float, "merged": float}``.
	"""

	resolved_tasks = resolve_tasks(single_scores, (merged_scores,), tasks)
	results: dict[str, dict[str, float]] = {}
	for task in resolved_tasks:
		single_value = extract_score(task, single_scores, metric_keys)
		merged_value = extract_score(task, merged_scores, metric_keys)
		results[task] = {
			"single": round_value(single_value, round_to),
			"merged": round_value(merged_value, round_to),
		}
	return results
