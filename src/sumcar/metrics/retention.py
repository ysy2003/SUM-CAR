"""Retention metrics utilities."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ._utils import extract_score, macro_average, resolve_tasks, round_value

__all__ = ["compute_retention_metrics"]


def compute_retention_metrics(
	single_scores: Mapping[str, Any],
	merged_scores: Mapping[str, Any],
	*,
	metric_keys: Optional[Mapping[str, str]] = None,
	tasks: Optional[Iterable[str]] = None,
	round_to: Optional[int] = 4,
	eps: float = 1e-8,
) -> dict[str, Any]:
	"""Compute retention ratio ``R_T`` and drop ``Δ_drop`` per task.

	``R_T = S_merged(T) / S_single(T)``
	``Δ_drop(T) = S_single(T) - S_merged(T)``
	"""

	resolved_tasks = resolve_tasks(single_scores, (merged_scores,), tasks)
	ratio_display: dict[str, float] = {}
	drop_display: dict[str, float] = {}
	ratio_raw: dict[str, float] = {}
	drop_raw: dict[str, float] = {}

	for task in resolved_tasks:
		single_value = extract_score(task, single_scores, metric_keys)
		merged_value = extract_score(task, merged_scores, metric_keys)
		ratio_value = merged_value / max(single_value, eps)
		drop_value = single_value - merged_value

		ratio_raw[task] = ratio_value
		drop_raw[task] = drop_value
		ratio_display[task] = round_value(ratio_value, round_to)
		drop_display[task] = round_value(drop_value, round_to)

	return {
		"ratio": ratio_display,
		"drop": drop_display,
		"macro_ratio": round_value(macro_average(ratio_raw), round_to),
		"macro_drop": round_value(macro_average(drop_raw), round_to),
	}
