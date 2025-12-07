"""Reversible metric helpers."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ._utils import extract_score, macro_average, resolve_tasks, round_value

__all__ = ["compute_reversible_metrics"]


def compute_reversible_metrics(
	merged_scores: Mapping[str, Any],
	restored_scores: Mapping[str, Any],
	*,
	metric_keys: Optional[Mapping[str, str]] = None,
	tasks: Optional[Iterable[str]] = None,
	round_to: Optional[int] = 4,
) -> dict[str, Any]:
	"""Compute ``S_restore`` and ``Δ_restore`` for every task.

	``Δ_restore(¬T) = S_restore(T|¬T) - S_merged(T)``.
	"""

	resolved_tasks = resolve_tasks(merged_scores, (restored_scores,), tasks)
	restore_display: dict[str, float] = {}
	delta_display: dict[str, float] = {}
	restore_raw: dict[str, float] = {}
	delta_raw: dict[str, float] = {}

	for task in resolved_tasks:
		merged_value = extract_score(task, merged_scores, metric_keys)
		restore_value = extract_score(task, restored_scores, metric_keys)
		delta_value = restore_value - merged_value

		restore_raw[task] = restore_value
		delta_raw[task] = delta_value
		restore_display[task] = round_value(restore_value, round_to)
		delta_display[task] = round_value(delta_value, round_to)

	return {
		"restore": restore_display,
		"delta": delta_display,
		"macro_restore": round_value(macro_average(restore_raw), round_to),
		"macro_delta": round_value(macro_average(delta_raw), round_to),
	}
