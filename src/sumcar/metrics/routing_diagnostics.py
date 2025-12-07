"""Routing diagnostic utilities."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ._utils import macro_average, resolve_tasks, round_value

__all__ = ["compute_routing_diagnostics"]


def compute_routing_diagnostics(
	home_hits: Mapping[str, Any],
	cross_hits: Mapping[str, Any],
	*,
	util: Optional[Mapping[str, Any]] = None,
	conflict: Optional[Mapping[str, Any]] = None,
	tasks: Optional[Iterable[str]] = None,
	round_to: Optional[int] = 4,
	eps: float = 1e-8,
) -> dict[str, Any]:
	"""Compute home-hit, cross-hit, and util/conflict diagnostics per task."""

	resolved_tasks = resolve_tasks(home_hits, (cross_hits,), tasks)
	home_display: dict[str, float] = {}
	cross_display: dict[str, float] = {}
	home_raw: dict[str, float] = {}
	cross_raw: dict[str, float] = {}

	util_display: dict[str, float] = {}
	util_raw: dict[str, float] = {}
	include_util = util is not None and conflict is not None

	for task in resolved_tasks:
		home_value = float(home_hits.get(task, 0.0))
		cross_value = float(cross_hits.get(task, 0.0))
		total = home_value + cross_value
		if total <= eps:
			home_rate = 0.0
			cross_rate = 0.0
		else:
			home_rate = home_value / total
			cross_rate = cross_value / total

		home_raw[task] = home_rate
		cross_raw[task] = cross_rate
		home_display[task] = round_value(home_rate, round_to)
		cross_display[task] = round_value(cross_rate, round_to)

		if include_util:
			if task not in util or task not in conflict:
				raise KeyError(f'missing util/conflict stats for task "{task}"')
			conflict_value = max(float(conflict[task]), 1.0)
			util_value = float(util[task])
			ratio = util_value / conflict_value
			util_raw[task] = ratio
			util_display[task] = round_value(ratio, round_to)

	result: dict[str, Any] = {
		"home_hit": home_display,
		"cross_hit": cross_display,
		"macro_home_hit": round_value(macro_average(home_raw), round_to),
		"macro_cross_hit": round_value(macro_average(cross_raw), round_to),
	}

	if include_util:
		result["util_conflict"] = util_display
		result["macro_util_conflict"] = round_value(
			macro_average(util_raw), round_to
		)

	return result
