"""Deterministic aggregation for the historical benchmark pilot.

The historical headline is intentionally conservative: infrastructure failures,
missing files, corrupt records, and role-inapplicable cells are never converted
to zero scores. The resulting headline is a pilot diagnostic only, even when
all 200 tasks are present. Verified publication requires World replay and
deterministic rescore. Partial aggregates remain useful for progress monitoring through
their explicit coverage and observed means.

The module accepts either a mapping keyed by ``RunSpecV2.run_key`` or a
directory of JSON result records.  It does not depend on an LLM judge and its
output order is independent of mapping, file-system, and plan iteration order.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from episode.capability_benchmark import TASK_REGISTRY_V2
from experiments.benchmark_plan import (
    EXPECTED_BUYER_TASKS_V2,
    EXPECTED_FAMILY_TASKS_V2,
    EXPECTED_MERCHANT_TASKS_V2,
    EXPECTED_TASKS_V2,
    MAIN_MODELS_V2,
    ExperimentPlanV2,
    RunSpecV2,
)


AGGREGATE_SCHEMA_V2 = "cwe.experiment-aggregate.v2"
DEFAULT_METRIC_V2 = "overall_score"
DIRECT_SIMULATION_PILOT_CLASSIFICATION = "direct_simulation_pilot"
FAMILIES_V2: tuple[str, ...] = tuple(f"T{number}" for number in range(1, 11))
ROLES_V2: tuple[str, str] = ("buyer", "merchant")

ResultRecord: TypeAlias = Mapping[str, Any]
ResultMapping: TypeAlias = Mapping[str, ResultRecord | None]
ResultSource: TypeAlias = ResultMapping | str | Path
OutcomeStatus: TypeAlias = Literal["succeeded", "failed", "pending", "invalid"]


@dataclass(frozen=True)
class _Outcome:
    status: OutcomeStatus
    score: float | None = None
    strict_success: bool | None = None
    security_violation: bool = False
    privacy_violation: bool = False


@dataclass(frozen=True)
class _LoadedResults:
    records: dict[str, ResultRecord]
    invalid_keys: frozenset[str]
    unplanned_records: int


@dataclass(frozen=True)
class AggregateReportV2:
    """Serializable v2 main-benchmark aggregate with no sampling statistics."""

    metric: str
    overview: dict[str, Any]
    models: tuple[dict[str, Any], ...]
    model_families: tuple[dict[str, Any], ...]
    family_roles: tuple[dict[str, Any], ...]
    schema_version: str = AGGREGATE_SCHEMA_V2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_classification": DIRECT_SIMULATION_PILOT_CLASSIFICATION,
            "formal_publication_eligible": False,
            "world_replay_claimed": False,
            "metric": self.metric,
            "overview": self.overview,
            "models": list(self.models),
            "model_families": list(self.model_families),
            "family_roles": list(self.family_roles),
        }

    def write(self, path: str | Path) -> Path:
        """Write canonical JSON; identical inputs produce identical bytes."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return target


def validate_complete_main_benchmark_plan(plan: ExperimentPlanV2) -> None:
    """Require the original six-model by 200-task compatibility matrix."""

    main_runs = tuple(run for run in plan.runs if run.suite == "main")
    expected_task_ids = set(TASK_REGISTRY_V2)
    counts = Counter(run.model_id for run in main_runs)
    if set(counts) != set(MAIN_MODELS_V2):
        raise ValueError(
            "complete v2 main plan must contain the original six-model panel"
        )

    for model_id in MAIN_MODELS_V2:
        selected = tuple(run for run in main_runs if run.model_id == model_id)
        if len(selected) != EXPECTED_TASKS_V2:
            raise ValueError(
                f"complete v2 main plan requires {EXPECTED_TASKS_V2} tasks for "
                f"{model_id!r}, got {len(selected)}"
            )
        task_ids = {run.task_id for run in selected}
        if task_ids != expected_task_ids:
            missing = sorted(expected_task_ids - task_ids)
            extra = sorted(task_ids - expected_task_ids)
            raise ValueError(
                f"complete v2 main plan task set drift for {model_id!r}: "
                f"missing={missing!r}, extra={extra!r}"
            )

    expected_runs = len(MAIN_MODELS_V2) * EXPECTED_TASKS_V2
    if len(main_runs) != expected_runs:
        raise ValueError(
            f"complete v2 main plan requires {expected_runs} runs, got {len(main_runs)}"
        )


def aggregate_benchmark_results(
    plan: ExperimentPlanV2,
    results: ResultSource,
    *,
    metric: str = DEFAULT_METRIC_V2,
    require_complete_main: bool = False,
) -> AggregateReportV2:
    """Aggregate deterministic main results without imputing unavailable scores.

    ``results`` can be a mapping of ``run_key -> result`` or a directory.  A
    directory may contain one record per JSON file (including the conventional
    ``<run-key>/result.json`` layout) or JSON objects that bundle such a mapping.
    Non-main runs in an ``all`` plan are explicitly ignored by this benchmark
    aggregate; self-play and 3x3 markets require their own market metrics.

    Succeeded records require a finite scalar metric in ``[0, 1]``.  Failed
    records, absent records, and invalid records are counted separately and
    excluded from every score denominator.  Consequently, only a genuine
    successful score of ``0.0`` enters the ``zero`` distribution bucket.
    """

    if not metric:
        raise ValueError("v2 aggregate metric must be non-empty")

    main_runs = tuple(
        sorted(
            (run for run in plan.runs if run.suite == "main"),
            key=lambda run: (_model_order(run.model_id), run.task_id, run.run_key),
        )
    )
    if not main_runs:
        raise ValueError("v2 benchmark aggregation requires at least one main run")
    if require_complete_main or len(main_runs) == len(MAIN_MODELS_V2) * EXPECTED_TASKS_V2:
        validate_complete_main_benchmark_plan(plan)

    expected_by_family_role = _expected_family_role_counts()
    planned_keys = frozenset(run.run_key for run in main_runs)
    loaded = _load_results(results, planned_keys)
    outcomes: dict[str, _Outcome] = {}
    for run in main_runs:
        if run.run_key in loaded.invalid_keys:
            outcomes[run.run_key] = _Outcome("invalid")
            continue
        record = loaded.records.get(run.run_key)
        outcomes[run.run_key] = (
            _Outcome("pending")
            if record is None
            else _parse_outcome(run, record, metric=metric)
        )

    overview_scope = _summarize_scope(
        main_runs,
        outcomes,
        expected=len(main_runs),
    )
    complete_plan = _is_complete_benchmark_plan(main_runs)
    overview: dict[str, Any] = {
        "planned": len(main_runs),
        "succeeded": overview_scope["succeeded"],
        "failed": overview_scope["failed"],
        "pending": overview_scope["pending"],
        "invalid": overview_scope["invalid"],
        "scored": overview_scope["score_count"],
        "complete_main_plan": complete_plan,
        "non_main_plan_runs_ignored": len(plan.runs) - len(main_runs),
        "unplanned_result_records_ignored": loaded.unplanned_records,
    }

    model_ids = tuple(
        sorted({run.model_id for run in main_runs}, key=_model_order)
    )
    model_family_rows: list[dict[str, Any]] = []
    family_role_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for model_id in model_ids:
        model_runs = tuple(run for run in main_runs if run.model_id == model_id)
        family_rows_for_model: list[dict[str, Any]] = []
        for family in FAMILIES_V2:
            family_runs = tuple(run for run in model_runs if run.task_family == family)
            family_summary = _summarize_scope(
                family_runs,
                outcomes,
                expected=EXPECTED_FAMILY_TASKS_V2,
            )
            family_row = {
                "model_id": model_id,
                "task_family": family,
                **family_summary,
            }
            family_rows_for_model.append(family_row)
            model_family_rows.append(family_row)

            for role in ROLES_V2:
                role_runs = tuple(
                    run for run in family_runs if run.evaluated_role == role
                )
                expected = expected_by_family_role[(family, role)]
                family_role_rows.append({
                    "model_id": model_id,
                    "task_family": family,
                    "evaluated_role": role,
                    "availability": "applicable" if expected else "not_applicable",
                    **_summarize_scope(role_runs, outcomes, expected=expected),
                })

        model_summary = _summarize_scope(
            model_runs,
            outcomes,
            expected=EXPECTED_TASKS_V2,
        )
        buyer_runs = tuple(
            run for run in model_runs if run.evaluated_role == "buyer"
        )
        merchant_runs = tuple(
            run for run in model_runs if run.evaluated_role == "merchant"
        )
        buyer_summary = _summarize_scope(
            buyer_runs,
            outcomes,
            expected=EXPECTED_BUYER_TASKS_V2,
        )
        merchant_summary = _summarize_scope(
            merchant_runs,
            outcomes,
            expected=EXPECTED_MERCHANT_TASKS_V2,
        )

        observed_family_means = [
            float(row["score_mean"])
            for row in family_rows_for_model
            if row["score_mean"] is not None
        ]
        complete_family_means = [
            float(row["complete_score_mean"])
            for row in family_rows_for_model
            if row["complete_score_mean"] is not None
        ]
        plan_complete = _is_complete_model_plan(model_runs)
        result_complete = plan_complete and model_summary["score_count"] == EXPECTED_TASKS_V2
        model_rows.append({
            "model_id": model_id,
            "plan_complete": plan_complete,
            "result_complete": result_complete,
            **model_summary,
            "headline_score": (
                statistics.fmean(complete_family_means)
                if result_complete and len(complete_family_means) == len(FAMILIES_V2)
                else None
            ),
            "observed_headline_score": (
                statistics.fmean(observed_family_means)
                if observed_family_means
                else None
            ),
            "observed_family_count": len(observed_family_means),
            "buyer": buyer_summary,
            "merchant": merchant_summary,
        })

    return AggregateReportV2(
        metric=metric,
        overview=overview,
        models=tuple(model_rows),
        model_families=tuple(model_family_rows),
        family_roles=tuple(family_role_rows),
    )


def _parse_outcome(run: RunSpecV2, record: ResultRecord, *, metric: str) -> _Outcome:
    if _contains_legacy_axis(record):
        return _Outcome("invalid")
    stored_key = record.get("run_key")
    if stored_key is not None and str(stored_key) != run.run_key:
        return _Outcome("invalid")

    raw_status = record.get("status")
    status_value = getattr(raw_status, "value", raw_status)
    status = str(status_value).lower() if status_value is not None else ""
    if status == "failed":
        return _Outcome("failed")
    if status == "pending":
        return _Outcome("pending")
    if status == "invalid":
        return _Outcome("invalid")
    if status != "succeeded":
        # A planned v2 run is role-applicable by construction.  An N/A result
        # therefore signals plan/result drift and is invalid, never a zero.
        return _Outcome("invalid")

    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return _Outcome("invalid")
    raw_score = record.get(metric, metrics.get(metric))
    if (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(float(raw_score))
        or not 0.0 <= float(raw_score) <= 1.0
    ):
        return _Outcome("invalid")

    if "strict_success" in record:
        strict_success = record["strict_success"]
    elif "strict_success" in metrics:
        strict_success = metrics["strict_success"]
    elif "task_success" in metrics:
        strict_success = metrics["task_success"]
    else:
        strict_success = (
            record.get("completed") is True
            and str(record.get("failure_mode") or "").lower() == "ok"
        )
    if not isinstance(strict_success, bool):
        return _Outcome("invalid")
    flags: dict[str, bool] = {}
    for name, aliases in {
        "security_violation": ("security_violation", "model_safety_violation"),
        "privacy_violation": ("privacy_violation", "model_privacy_violation"),
    }.items():
        present = [metrics[alias] for alias in aliases if alias in metrics]
        if any(not isinstance(value, bool) for value in present):
            return _Outcome("invalid")
        flags[name] = any(present)
    return _Outcome(
        "succeeded",
        float(raw_score),
        strict_success,
        security_violation=flags["security_violation"],
        privacy_violation=flags["privacy_violation"],
    )


def _summarize_scope(
    runs: Sequence[RunSpecV2],
    outcomes: Mapping[str, _Outcome],
    *,
    expected: int,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    scores: list[float] = []
    strict_successes = 0
    security_violations = 0
    privacy_violations = 0
    distribution = {"zero": 0, "partial": 0, "full": 0}

    for run in sorted(runs, key=lambda item: (item.task_id, item.run_key)):
        outcome = outcomes[run.run_key]
        status_counts[outcome.status] += 1
        if outcome.status != "succeeded":
            continue
        if outcome.score is None or outcome.strict_success is None:  # pragma: no cover
            raise AssertionError("a succeeded v2 aggregate outcome must be scoreable")
        scores.append(outcome.score)
        strict_successes += int(outcome.strict_success)
        security_violations += int(outcome.security_violation)
        privacy_violations += int(outcome.privacy_violation)
        if outcome.score == 0.0:
            distribution["zero"] += 1
        elif outcome.score == 1.0:
            distribution["full"] += 1
        else:
            distribution["partial"] += 1

    score_mean = statistics.fmean(scores) if scores else None
    complete = len(runs) == expected and len(scores) == expected
    return {
        "expected_tasks": expected,
        "planned": len(runs),
        "succeeded": status_counts["succeeded"],
        "failed": status_counts["failed"],
        "pending": status_counts["pending"],
        "invalid": status_counts["invalid"],
        "score_count": len(scores),
        "coverage": (len(scores) / expected if expected else None),
        "score_mean": score_mean,
        "complete_score_mean": score_mean if complete else None,
        "strict_successes": strict_successes,
        "strict_success_rate": (
            strict_successes / len(scores) if scores else None
        ),
        "security_violations": security_violations,
        "privacy_violations": privacy_violations,
        "score_distribution": {
            **distribution,
            "total": len(scores),
        },
    }


def _expected_family_role_counts() -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    for task in TASK_REGISTRY_V2.values():
        family = str(getattr(task.family, "value", task.family))
        counts[(family, str(task.evaluated_role))] += 1

    if any(
        sum(counts[(family, role)] for role in ROLES_V2)
        != EXPECTED_FAMILY_TASKS_V2
        for family in FAMILIES_V2
    ):
        raise ValueError("v2 task registry family/role denominators have drifted")
    if sum(counts[(family, "buyer")] for family in FAMILIES_V2) != EXPECTED_BUYER_TASKS_V2:
        raise ValueError("v2 task registry Buyer denominator has drifted")
    if (
        sum(counts[(family, "merchant")] for family in FAMILIES_V2)
        != EXPECTED_MERCHANT_TASKS_V2
    ):
        raise ValueError("v2 task registry Merchant denominator has drifted")
    return {
        (family, role): counts[(family, role)]
        for family in FAMILIES_V2
        for role in ROLES_V2
    }


def _is_complete_model_plan(runs: Sequence[RunSpecV2]) -> bool:
    return (
        len(runs) == EXPECTED_TASKS_V2
        and {run.task_id for run in runs} == set(TASK_REGISTRY_V2)
    )


def _is_complete_benchmark_plan(runs: Sequence[RunSpecV2]) -> bool:
    if len(runs) != len(MAIN_MODELS_V2) * EXPECTED_TASKS_V2:
        return False
    return all(
        _is_complete_model_plan(
            tuple(run for run in runs if run.model_id == model_id)
        )
        for model_id in MAIN_MODELS_V2
    )


def _model_order(model_id: str) -> tuple[int, str]:
    try:
        return MAIN_MODELS_V2.index(model_id), model_id
    except ValueError:
        return len(MAIN_MODELS_V2), model_id


def _contains_legacy_axis(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "seed" in value or "rollout" in value:
            return True
        return any(_contains_legacy_axis(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_legacy_axis(item) for item in value)
    return False


def _load_results(source: ResultSource, planned_keys: frozenset[str]) -> _LoadedResults:
    if isinstance(source, (str, Path)):
        return _load_result_directory(Path(source), planned_keys)
    if not isinstance(source, Mapping):
        raise TypeError("v2 aggregate results must be a mapping or JSON directory")

    records: dict[str, ResultRecord] = {}
    invalid_keys: set[str] = set()
    unplanned = 0
    for raw_key in sorted(source, key=str):
        key = str(raw_key)
        value = source[raw_key]
        if key not in planned_keys:
            unplanned += 1
            continue
        if value is None:
            continue
        if not isinstance(value, Mapping):
            invalid_keys.add(key)
            continue
        records[key] = value
    return _LoadedResults(records, frozenset(invalid_keys), unplanned)


def _load_result_directory(
    root: Path,
    planned_keys: frozenset[str],
) -> _LoadedResults:
    if not root.is_dir():
        raise ValueError(f"v2 aggregate result root is not a directory: {root}")

    records: dict[str, ResultRecord] = {}
    invalid_keys: set[str] = set()
    unplanned = 0
    paths = sorted(root.rglob("*.json"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        inferred_key = _infer_run_key(path, planned_keys)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            if inferred_key is None:
                unplanned += 1
            else:
                invalid_keys.add(inferred_key)
            continue
        if not isinstance(raw, Mapping):
            if inferred_key is None:
                unplanned += 1
            else:
                invalid_keys.add(inferred_key)
            continue

        # A JSON file can bundle the simplified run-key mapping used by the
        # in-memory API.  Otherwise it is treated as one result record.
        bundled_keys = set(map(str, raw)) & planned_keys
        if "run_key" not in raw and inferred_key is None and bundled_keys:
            for key in sorted(bundled_keys):
                value = raw[key]
                if value is None:
                    continue
                if not isinstance(value, Mapping) or key in records:
                    invalid_keys.add(key)
                    records.pop(key, None)
                elif key not in invalid_keys:
                    records[key] = value
            unplanned += len(set(map(str, raw)) - planned_keys)
            continue

        stored_key = raw.get("run_key")
        result_key: str | None = (
            str(stored_key) if stored_key is not None else inferred_key
        )
        if result_key not in planned_keys:
            unplanned += 1
            continue
        if (
            inferred_key is not None
            and stored_key is not None
            and result_key != inferred_key
        ):
            invalid_keys.add(inferred_key)
            if result_key in planned_keys:
                invalid_keys.add(result_key)
            continue
        if result_key is None:  # narrowed by membership above; keeps typing explicit
            raise AssertionError("planned result key unexpectedly resolved to null")
        if result_key in records or result_key in invalid_keys:
            invalid_keys.add(result_key)
            records.pop(result_key, None)
            continue
        records[result_key] = raw

    return _LoadedResults(records, frozenset(invalid_keys), unplanned)


def _infer_run_key(path: Path, planned_keys: frozenset[str]) -> str | None:
    if path.stem in planned_keys:
        return path.stem
    if path.parent.name in planned_keys:
        return path.parent.name
    return None


__all__ = [
    "AGGREGATE_SCHEMA_V2",
    "AggregateReportV2",
    "DEFAULT_METRIC_V2",
    "FAMILIES_V2",
    "ResultMapping",
    "ResultSource",
    "aggregate_benchmark_results",
    "validate_complete_main_benchmark_plan",
]
