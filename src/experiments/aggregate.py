"""Deterministic aggregation over persisted experiment results."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.plan import ExperimentPlan
from experiments.results import ResultCorruptionError, ResultStatus, ResultStore


AGGREGATE_SCHEMA = "cwe.experiment-aggregate.v1"


@dataclass(frozen=True)
class AggregateReport:
    metric: str
    overview: dict[str, int]
    groups: tuple[dict[str, Any], ...]
    macro_by_model_role: tuple[dict[str, Any], ...]
    schema_version: str = AGGREGATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "overview": self.overview,
            "groups": list(self.groups),
            "macro_by_model_role": list(self.macro_by_model_role),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def aggregate_results(
    plan: ExperimentPlan,
    store: ResultStore,
    *,
    metric: str = "overall_score",
) -> AggregateReport:
    """Aggregate one scalar metric without consulting any LLM judge."""

    raw_groups: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
    overview = {
        "planned": len(plan.runs),
        "succeeded": 0,
        "failed": 0,
        "pending": 0,
        "invalid": 0,
        "truncated": 0,
    }

    for run in plan.runs:
        key = (run.suite, run.model_id, run.evaluated_role, run.task_family)
        group = raw_groups.setdefault(key, {
            "suite": run.suite,
            "model_id": run.model_id,
            "evaluated_role": run.evaluated_role,
            "task_family": run.task_family,
            "planned": 0,
            "succeeded": 0,
            "failed": 0,
            "pending": 0,
            "invalid": 0,
            "completed": 0,
            "task_successes": 0,
            "truncated": 0,
            "failure_modes": {},
            "stop_reasons": {},
            "values": [],
        })
        group["planned"] += 1
        try:
            result = store.read(run)
        except ResultCorruptionError:
            overview["invalid"] += 1
            group["invalid"] += 1
            continue
        if result is None:
            overview["pending"] += 1
            group["pending"] += 1
            continue
        if result.status == ResultStatus.FAILED:
            overview["failed"] += 1
            group["failed"] += 1
            continue

        overview["succeeded"] += 1
        group["succeeded"] += 1
        if result.truncated:
            overview["truncated"] += 1
            group["truncated"] += 1
        if result.stop_reason:
            reasons: dict[str, int] = group["stop_reasons"]
            reasons[result.stop_reason] = reasons.get(result.stop_reason, 0) + 1
        if result.completed is True:
            group["completed"] += 1
        task_success = result.metrics.get("task_success")
        if task_success is True or (
            task_success is None
            and result.completed is True
            and str(result.failure_mode or "").lower() == "ok"
        ):
            group["task_successes"] += 1
        if result.failure_mode:
            modes: dict[str, int] = group["failure_modes"]
            modes[result.failure_mode] = modes.get(result.failure_mode, 0) + 1
        value = result.metrics.get(metric)
        if value is not None and not isinstance(value, bool):
            group["values"].append(float(value))

    groups: list[dict[str, Any]] = []
    for key in sorted(raw_groups, key=_sortable_key):
        raw = raw_groups[key]
        values = raw.pop("values")
        raw["metric_count"] = len(values)
        raw["metric_mean"] = statistics.fmean(values) if values else None
        raw["completion_rate"] = (
            raw["completed"] / raw["succeeded"] if raw["succeeded"] else None
        )
        raw["task_success_rate"] = (
            raw["task_successes"] / raw["succeeded"] if raw["succeeded"] else None
        )
        raw["truncation_rate"] = (
            raw["truncated"] / raw["succeeded"] if raw["succeeded"] else None
        )
        raw["failure_modes"] = dict(sorted(raw["failure_modes"].items()))
        raw["stop_reasons"] = dict(sorted(raw["stop_reasons"].items()))
        groups.append(raw)

    macro_buckets: dict[tuple[str, str, str], list[float]] = {}
    for group in groups:
        mean = group["metric_mean"]
        if mean is None or group["task_family"] is None:
            continue
        macro_key = (group["suite"], group["model_id"], group["evaluated_role"])
        macro_buckets.setdefault(macro_key, []).append(float(mean))
    macros = tuple(
        {
            "suite": key[0],
            "model_id": key[1],
            "evaluated_role": key[2],
            "family_count": len(values),
            "macro_mean": statistics.fmean(values),
        }
        for key, values in sorted(macro_buckets.items())
    )
    return AggregateReport(
        metric=metric,
        overview=overview,
        groups=tuple(groups),
        macro_by_model_role=macros,
    )


def _sortable_key(value: tuple[str, str, str, str | None]) -> tuple[str, str, str, str]:
    return value[0], value[1], value[2], value[3] or ""


__all__ = ["AGGREGATE_SCHEMA", "AggregateReport", "aggregate_results"]
