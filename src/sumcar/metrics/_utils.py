"""Shared helper utilities for SUM-CAR metric calculations."""
from __future__ import annotations

from numbers import Number
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "resolve_tasks",
    "extract_score",
    "round_value",
    "macro_average",
]

_PREFERRED_SCALAR_KEYS: Sequence[str] = (
    "accuracy",
    "acc",
    "em",
    "pass@1",
    "score",
    "value",
)


def resolve_tasks(
    primary: Mapping[str, Any],
    others: Iterable[Optional[Mapping[str, Any]]] | None = None,
    tasks: Optional[Iterable[str]] = None,
) -> list[str]:
    """Return the ordered list of tasks that should be evaluated.

    Args:
        primary: Mapping that must contain every task.
        others: Additional mappings that should share the same task keys.
        tasks: Optional explicit ordering of tasks to use.

    Raises:
        ValueError: If no tasks can be resolved.
        KeyError: If any required mapping is missing a task key.
    """
    if primary is None:
        raise ValueError("primary mapping is required")

    if tasks is None:
        resolved = list(primary.keys())
    else:
        resolved = list(dict.fromkeys(tasks))

    if not resolved:
        raise ValueError("no tasks specified for metric computation")

    compare_against = tuple(others or ())
    for task in resolved:
        if task not in primary:
            raise KeyError(f'task "{task}" missing from primary scores')
        for other in compare_against:
            if other is None:
                continue
            if task not in other:
                raise KeyError(f'task "{task}" missing from supplied mapping')
    return resolved


def extract_score(
    task: str,
    source: Mapping[str, Any],
    metric_keys: Optional[Mapping[str, str]] = None,
    preferred_keys: Sequence[str] = _PREFERRED_SCALAR_KEYS,
) -> float:
    """Extract a numeric score for *task* from *source*.

    The value can either be a bare float/int or a nested mapping that includes a
    scalar metric such as ``accuracy`` or ``pass@1``. When *metric_keys* is
    provided it is treated as a mapping from task name (or ``"__default__"``)
    to the metric key that should be used.
    """
    if task not in source:
        raise KeyError(f'task "{task}" not found in source mapping')

    value = source[task]
    if isinstance(value, Number):
        return float(value)

    if isinstance(value, Mapping):
        key = None
        if metric_keys:
            key = metric_keys.get(task) or metric_keys.get("__default__")
        if key:
            if key not in value:
                raise KeyError(
                    f'metric key "{key}" not found for task "{task}"'
                )
            metric_value = value[key]
            if not isinstance(metric_value, Number):
                raise TypeError(
                    f'expected numeric metric for key "{key}" (task "{task}")'
                )
            return float(metric_value)

        for candidate_key in preferred_keys:
            if candidate_key in value and isinstance(value[candidate_key], Number):
                return float(value[candidate_key])

        for metric_value in value.values():
            if isinstance(metric_value, Number):
                return float(metric_value)

    raise TypeError(
        f'unable to extract numeric score for task "{task}" from provided data'
    )


def round_value(value: float, decimals: Optional[int]) -> float:
    """Round *value* to ``decimals`` places when requested."""
    if decimals is None:
        return value
    return round(value, decimals)


def macro_average(values: Mapping[str, float]) -> float:
    """Compute the arithmetic mean over the provided mapping values."""
    if not values:
        return 0.0
    return sum(values.values()) / len(values)
