"""Deterministic many-to-many market artifacts and aggregation.

This is intentionally separate from the legacy single-buyer Episode score.
Market metrics consume explicit structured populations, private valuation/floor
answer keys, completed transactions, exposure events, and protocol/security
checks.  The resulting artifact is self-verifying: normalized metric inputs and
execution provenance are hashed, and the metric bundle can be recomputed
without any model or judge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from episode.termination import SCOREABLE_STOP_REASONS
from evals.market_metrics import (
    AllocationOracle,
    BuyerValuation,
    Exposure,
    MarketMetrics,
    MarketTransaction,
    MerchantFloor,
    PrivacyEvent,
    ProtocolEvent,
    compute_market_metrics,
)
from evals.serialize import to_canonical
from experiments.benchmark_plan import RUN_SCHEMA_V2, RunSpecV2


LEGACY_MARKET_ARTIFACT_SCHEMA = "cwe.market-artifact.v1"
MARKET_ARTIFACT_SCHEMA = "cwe.market-artifact.v2"
SEED_FREE_MARKET_ARTIFACT_SCHEMA = "cwe.market-artifact.v3"
LEGACY_MARKET_AGGREGATE_SCHEMA = "cwe.market-aggregate.v1"
MARKET_AGGREGATE_SCHEMA = "cwe.market-aggregate.v2"
MARKET_STOP_REASONS = SCOREABLE_STOP_REASONS
_METRIC_NAMES = (
    "trade_rate",
    "consumer_surplus",
    "producer_surplus",
    "social_welfare",
    "allocative_efficiency",
    "inventory_correctness",
    "exposure_fairness",
    "privacy_leakage_rate",
    "protocol_violation_rate",
)
_FORBIDDEN_V3_AXES = frozenset({"seed", "rollout"})
_V3_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run",
        "run_key",
        "task_id",
        "identity_digest",
        "variant_id",
        "task_family",
        "population",
        "execution",
        "execution_digest",
        "inputs",
        "input_digest",
        "metrics",
        "evaluation",
    }
)


class MarketArtifactError(ValueError):
    """A persisted market artifact cannot be deterministically verified."""


def build_market_artifact(
    *,
    market_id: str,
    variant_id: str,
    task_family: str,
    seed: int,
    buyer_ids: Sequence[str],
    merchant_ids: Sequence[str],
    valuations: Sequence[BuyerValuation],
    merchant_floors: Sequence[MerchantFloor],
    transactions: Sequence[MarketTransaction] = (),
    exposures: Sequence[Exposure] = (),
    privacy_events: Sequence[PrivacyEvent] = (),
    protocol_events: Sequence[ProtocolEvent] = (),
    allocation_oracle: AllocationOracle | None = None,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute and package one model-free market result."""

    if not market_id or not variant_id or not task_family:
        raise MarketArtifactError("market_id, variant_id, and task_family are required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MarketArtifactError("seed must be a non-negative integer")
    inputs = _input_payload(
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
        valuations=valuations,
        merchant_floors=merchant_floors,
        transactions=transactions,
        exposures=exposures,
        privacy_events=privacy_events,
        protocol_events=protocol_events,
        allocation_oracle=allocation_oracle,
    )
    metrics = compute_market_metrics(
        valuations=valuations,
        merchant_floors=merchant_floors,
        transactions=transactions,
        exposures=exposures,
        privacy_events=privacy_events,
        protocol_events=protocol_events,
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
        allocation_oracle=allocation_oracle,
    )
    execution_payload = _normalize_execution(execution)
    return {
        "schema_version": MARKET_ARTIFACT_SCHEMA,
        "market_id": market_id,
        "variant_id": variant_id,
        "task_family": task_family,
        "seed": seed,
        "population": {
            "buyers": len(buyer_ids),
            "merchants": len(merchant_ids),
            "buyer_ids": list(buyer_ids),
            "merchant_ids": list(merchant_ids),
        },
        "execution": execution_payload,
        "execution_digest": _digest(execution_payload),
        "inputs": inputs,
        "input_digest": _digest(inputs),
        "metrics": market_metrics_to_dict(metrics),
        "evaluation": {
            "judge": "deterministic_python",
            "llm_judge": False,
            "legacy_episode_score_modified": False,
        },
    }


def convert_market_artifact_to_v3(
    artifact: Mapping[str, Any],
    *,
    run: RunSpecV2,
) -> dict[str, Any]:
    """Bind a verified v1/v2 market result to a seed-free v2 run identity.

    The deterministic inputs, their digest, and the recomputed metric bundle
    are preserved exactly.  Legacy identity axes are removed only from
    execution provenance, which is normalized and re-digested under the v2
    ``run_key``/``task_id`` identity.  Inputs or population data containing a
    legacy axis are rejected because removing either would invalidate the
    evidence the source ``input_digest`` authenticates.
    """

    source_schema = artifact.get("schema_version")
    if source_schema not in {
        LEGACY_MARKET_ARTIFACT_SCHEMA,
        MARKET_ARTIFACT_SCHEMA,
    }:
        raise MarketArtifactError(
            "market artifact v3 conversion requires a verified v1 or v2 source"
        )
    verify_market_artifact(artifact)
    if not isinstance(run, RunSpecV2):
        raise MarketArtifactError("market artifact v3 requires a RunSpecV2 identity")
    if run.suite != "many_to_many":
        raise MarketArtifactError("market artifact v3 requires a many_to_many RunSpecV2 identity")

    variant_id = str(artifact.get("variant_id", ""))
    task_family = str(artifact.get("task_family", ""))
    task = _task_for_v3_run(run)
    if variant_id != str(task.template_variant_id):
        raise MarketArtifactError(
            "source market variant does not match the v2 task identity: "
            f"source={variant_id!r}, task={task.template_variant_id!r}"
        )
    if task_family != run.task_family:
        raise MarketArtifactError("source market task_family does not match the v2 task identity")

    population_raw = artifact.get("population")
    inputs_raw = artifact.get("inputs")
    metrics_raw = artifact.get("metrics")
    execution_raw = artifact.get("execution")
    if not isinstance(population_raw, Mapping):  # verified source, defensive
        raise MarketArtifactError("market artifact population must be an object")
    if not isinstance(inputs_raw, Mapping):  # verified source, defensive
        raise MarketArtifactError("market artifact inputs must be an object")
    if not isinstance(metrics_raw, Mapping):  # verified source, defensive
        raise MarketArtifactError("market artifact metrics must be an object")
    if not isinstance(execution_raw, Mapping):  # verified source, defensive
        raise MarketArtifactError("market artifact execution must be an object")
    _assert_no_legacy_axes(population_raw, path="population")
    _assert_no_legacy_axes(inputs_raw, path="inputs")
    if (
        population_raw.get("buyers") != run.buyers
        or population_raw.get("merchants") != run.merchants
    ):
        raise MarketArtifactError("source market population does not match the v2 run identity")

    execution_without_axes = _remove_legacy_axes(execution_raw)
    if not isinstance(execution_without_axes, dict):  # pragma: no cover
        raise MarketArtifactError("execution metadata must normalize to an object")
    expected_execution_identity: dict[str, Any] = {
        "suite": run.suite,
        "model_id": run.model_id,
        "task_id": run.task_id,
        "task_version": run.task_version,
        "evaluated_role": run.evaluated_role,
        "buyers": run.buyers,
        "merchants": run.merchants,
    }
    for key, expected in expected_execution_identity.items():
        if key in execution_without_axes and execution_without_axes[key] != expected:
            raise MarketArtifactError(f"source execution {key} does not match the v2 run identity")
    execution_without_axes.update(expected_execution_identity)
    # Legacy run keys are derived from seed/rollout axes and therefore cannot
    # authenticate a v3 result.  Replace them with the complete v2 identity.
    execution_without_axes["run_key"] = run.run_key
    execution = _normalize_execution(execution_without_axes)

    run_payload = run.to_dict()
    population = to_canonical(dict(population_raw))
    inputs = to_canonical(dict(inputs_raw))
    metrics = to_canonical(dict(metrics_raw))
    if not isinstance(population, dict) or not isinstance(inputs, dict):
        raise MarketArtifactError("market population and inputs must normalize to objects")
    if not isinstance(metrics, dict):
        raise MarketArtifactError("market metrics must normalize to an object")
    converted = {
        "schema_version": SEED_FREE_MARKET_ARTIFACT_SCHEMA,
        "run": run_payload,
        "run_key": run.run_key,
        "task_id": run.task_id,
        "identity_digest": _digest(run_payload),
        "variant_id": variant_id,
        "task_family": task_family,
        "population": population,
        "execution": execution,
        "execution_digest": _digest(execution),
        "inputs": inputs,
        # Preserve the exact source digest; verification below proves the
        # canonical input evidence itself was not changed by conversion.
        "input_digest": artifact.get("input_digest"),
        "metrics": metrics,
        "evaluation": {
            "judge": "deterministic_python",
            "llm_judge": False,
            "legacy_episode_score_modified": False,
            "run_identity_schema": RUN_SCHEMA_V2,
            "source_schema_version": source_schema,
        },
    }
    _assert_no_legacy_axes(converted)
    verify_market_artifact(converted)
    return converted


def market_metrics_to_dict(metrics: MarketMetrics) -> dict[str, dict[str, Any]]:
    """Serialize exact Decimal values without introducing binary floats."""

    rows: dict[str, dict[str, Any]] = {}
    for name in _METRIC_NAMES:
        metric = getattr(metrics, name)
        rows[name] = {
            "value": _number(metric.value),
            "unit": metric.unit,
            "numerator": _number(metric.numerator),
            "denominator": _number(metric.denominator),
            "applicable": metric.is_applicable,
            "reason": metric.reason,
        }
    return rows


def verify_market_artifact(artifact: Mapping[str, Any]) -> None:
    """Recompute metrics and reject input, execution, digest, or result drift.

    Version 1 remains readable so frozen reports do not become unusable.  Its
    metric inputs are still verified exactly, but it predates authenticated
    execution provenance.  Version 2 additionally binds normalized execution
    metadata and enforces its completion/truncation contract.  Version 3 also
    binds a canonical ``RunSpecV2`` identity and rejects legacy identity axes
    at every persisted nesting level.
    """

    schema = artifact.get("schema_version")
    if schema not in {
        LEGACY_MARKET_ARTIFACT_SCHEMA,
        MARKET_ARTIFACT_SCHEMA,
        SEED_FREE_MARKET_ARTIFACT_SCHEMA,
    }:
        raise MarketArtifactError(f"unsupported market artifact schema: {schema!r}")
    run_v2: RunSpecV2 | None = None
    if schema == SEED_FREE_MARKET_ARTIFACT_SCHEMA:
        _assert_no_legacy_axes(artifact)
        run_v2 = _verify_v3_identity(artifact)
    execution = artifact.get("execution")
    if not isinstance(execution, dict):
        raise MarketArtifactError("market artifact execution must be an object")
    if schema in {MARKET_ARTIFACT_SCHEMA, SEED_FREE_MARKET_ARTIFACT_SCHEMA}:
        if execution != to_canonical(execution):
            raise MarketArtifactError("market artifact execution is not normalized")
        if artifact.get("execution_digest") != _digest(execution):
            raise MarketArtifactError("market artifact execution_digest mismatch")
        _verify_execution(execution)
    if run_v2 is not None:
        _verify_v3_execution_identity(execution, run_v2)
    inputs = artifact.get("inputs")
    if not isinstance(inputs, dict):
        raise MarketArtifactError("market artifact inputs must be an object")
    if schema == SEED_FREE_MARKET_ARTIFACT_SCHEMA and inputs != to_canonical(inputs):
        raise MarketArtifactError("market artifact inputs are not normalized")
    if artifact.get("input_digest") != _digest(inputs):
        raise MarketArtifactError("market artifact input_digest mismatch")
    population = artifact.get("population")
    if not isinstance(population, dict):
        raise MarketArtifactError("market artifact population must be an object")
    parsed = _parse_inputs(inputs)
    buyer_ids = _string_tuple(population.get("buyer_ids"), "population.buyer_ids")
    merchant_ids = _string_tuple(population.get("merchant_ids"), "population.merchant_ids")
    input_buyer_ids = _string_tuple(inputs.get("buyer_ids"), "inputs.buyer_ids")
    input_merchant_ids = _string_tuple(inputs.get("merchant_ids"), "inputs.merchant_ids")
    if input_buyer_ids != buyer_ids or input_merchant_ids != merchant_ids:
        raise MarketArtifactError("population identities do not match normalized inputs")
    if population.get("buyers") != len(buyer_ids):
        raise MarketArtifactError("population buyer count does not match buyer_ids")
    if population.get("merchants") != len(merchant_ids):
        raise MarketArtifactError("population merchant count does not match merchant_ids")
    if run_v2 is not None and (
        len(buyer_ids) != run_v2.buyers or len(merchant_ids) != run_v2.merchants
    ):
        raise MarketArtifactError("market artifact population does not match v2 run identity")
    recomputed = compute_market_metrics(
        **parsed,
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
    )
    if artifact.get("metrics") != market_metrics_to_dict(recomputed):
        raise MarketArtifactError("market artifact metrics do not match deterministic inputs")


def _verify_v3_identity(artifact: Mapping[str, Any]) -> RunSpecV2:
    keys = frozenset(str(key) for key in artifact)
    if keys != _V3_TOP_LEVEL_KEYS:
        missing = sorted(_V3_TOP_LEVEL_KEYS - keys)
        extra = sorted(keys - _V3_TOP_LEVEL_KEYS)
        raise MarketArtifactError(
            f"market artifact v3 has invalid top-level fields: missing={missing!r}, extra={extra!r}"
        )
    run_raw = artifact.get("run")
    if not isinstance(run_raw, dict):
        raise MarketArtifactError("market artifact v3 run must be an object")
    if artifact.get("identity_digest") != _digest(run_raw):
        raise MarketArtifactError("market artifact identity_digest mismatch")
    try:
        run = RunSpecV2.from_dict(run_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise MarketArtifactError(f"invalid market artifact v3 run identity: {exc}") from exc
    if run.suite != "many_to_many":
        raise MarketArtifactError("market artifact v3 run identity must use the many_to_many suite")
    if run_raw != run.to_dict():
        raise MarketArtifactError("market artifact v3 run identity is not canonical")
    if artifact.get("run_key") != run.run_key:
        raise MarketArtifactError("market artifact run_key does not match v2 identity")
    if artifact.get("task_id") != run.task_id:
        raise MarketArtifactError("market artifact task_id does not match v2 identity")
    if artifact.get("task_family") != run.task_family:
        raise MarketArtifactError("market artifact task_family does not match v2 identity")
    task = _task_for_v3_run(run)
    if artifact.get("variant_id") != str(task.template_variant_id):
        raise MarketArtifactError("market artifact variant_id does not match v2 task")

    evaluation = artifact.get("evaluation")
    if not isinstance(evaluation, dict):
        raise MarketArtifactError("market artifact v3 evaluation must be an object")
    source_schema = evaluation.get("source_schema_version")
    expected_evaluation = {
        "judge": "deterministic_python",
        "llm_judge": False,
        "legacy_episode_score_modified": False,
        "run_identity_schema": RUN_SCHEMA_V2,
        "source_schema_version": source_schema,
    }
    if (
        source_schema
        not in {
            LEGACY_MARKET_ARTIFACT_SCHEMA,
            MARKET_ARTIFACT_SCHEMA,
        }
        or evaluation != expected_evaluation
    ):
        raise MarketArtifactError("market artifact v3 evaluation metadata is invalid")
    return run


def _verify_v3_execution_identity(
    execution: Mapping[str, Any],
    run: RunSpecV2,
) -> None:
    expected = {
        "run_key": run.run_key,
        "suite": run.suite,
        "model_id": run.model_id,
        "task_id": run.task_id,
        "task_version": run.task_version,
        "evaluated_role": run.evaluated_role,
        "buyers": run.buyers,
        "merchants": run.merchants,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise MarketArtifactError(f"market artifact execution {key} does not match v2 identity")


def _task_for_v3_run(run: RunSpecV2) -> Any:
    # Local import keeps the long-lived v1/v2 artifact reader independent of
    # benchmark-v2 registry initialization unless a v3 artifact is requested.
    from episode.capability_benchmark import get_task_v2

    return get_task_v2(run.task_id)


def write_market_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Verify and write canonical, byte-stable JSON."""

    verify_market_artifact(artifact)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_market_artifact(path: str | Path) -> dict[str, Any]:
    """Load one artifact and verify it before returning any metric."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketArtifactError(f"cannot read market artifact {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MarketArtifactError(f"market artifact root must be an object: {source}")
    verify_market_artifact(raw)
    return raw


def aggregate_market_artifacts(
    artifacts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate verified market metrics by variant and over the full suite."""

    rows = [dict(artifact) for artifact in artifacts]
    if not rows:
        raise MarketArtifactError("cannot aggregate an empty market artifact set")
    seen: set[str] = set()
    for artifact in rows:
        verify_market_artifact(artifact)
        market_id = _market_identity(artifact)
        if not market_id or market_id in seen:
            raise MarketArtifactError(f"missing or duplicate market_id: {market_id!r}")
        seen.add(market_id)

    groups: list[dict[str, Any]] = []
    group_keys = sorted({(str(row["variant_id"]), str(row["task_family"])) for row in rows})
    for variant_id, task_family in group_keys:
        selected = [
            row
            for row in rows
            if row["variant_id"] == variant_id and row["task_family"] == task_family
        ]
        groups.append(
            {
                "variant_id": variant_id,
                "task_family": task_family,
                "markets": len(selected),
                **_aggregate_scope(selected),
            }
        )
    overall = _aggregate_scope(rows)
    legacy_v1_markets = overall["legacy_v1_markets"]
    return {
        "schema_version": MARKET_AGGREGATE_SCHEMA,
        "markets": len(rows),
        "completed_markets": overall["completed_markets"],
        "truncated_markets": overall["truncated_markets"],
        "legacy_v1_markets": legacy_v1_markets,
        "unknown_completion_markets": overall["unknown_completion_markets"],
        "market_ids": sorted(seen),
        "population_shapes": sorted(
            {f"{row['population']['buyers']}x{row['population']['merchants']}" for row in rows}
        ),
        "groups": groups,
        "overall": overall,
        "evaluation": {
            "judge": "deterministic_python",
            "llm_judge": False,
            "verified_inputs": True,
            "verified_execution_provenance": legacy_v1_markets == 0,
            "metric_views": {
                "metrics": "fixed_horizon_including_truncated",
                "fixed_horizon_metrics": "fixed_horizon_including_truncated",
                "complete_only_metrics": "naturally_completed_only",
            },
        },
    }


def _aggregate_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if _execution_state(row)[0]]
    truncated = [row for row in rows if _execution_state(row)[1]]
    legacy = [row for row in rows if row.get("schema_version") == LEGACY_MARKET_ARTIFACT_SCHEMA]
    completion_unknown = [row for row in rows if _execution_state(row)[2]]
    fixed_horizon = _aggregate_metrics(rows)
    return {
        "completed_markets": len(completed),
        "truncated_markets": len(truncated),
        "legacy_v1_markets": len(legacy),
        "unknown_completion_markets": len(completion_unknown),
        # ``metrics`` remains the compatibility view used by existing reports.
        # It intentionally includes deterministic partial state at a frozen
        # horizon; consumers wanting natural completions use the explicit view.
        "metrics": fixed_horizon,
        "fixed_horizon_metrics": fixed_horizon,
        "complete_only_metrics": _aggregate_metrics(completed, unit_rows=rows),
    }


def _aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    unit_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for metric_name in _METRIC_NAMES:
        values: list[Decimal] = []
        unit: str | None = None
        for row in unit_rows if unit_rows is not None else rows:
            metric = row["metrics"][metric_name]
            current_unit = str(metric["unit"])
            if unit is None:
                unit = current_unit
            elif unit != current_unit:
                raise MarketArtifactError(
                    f"metric {metric_name} has inconsistent units in aggregate"
                )
        for row in rows:
            metric = row["metrics"][metric_name]
            value = metric.get("value")
            if value is not None:
                values.append(Decimal(str(value)))
        with localcontext() as context:
            context.prec = 28
            mean = sum(values, Decimal(0)) / len(values) if values else None
        aggregated[metric_name] = {
            "unit": unit,
            "applicable_markets": len(values),
            "total_markets": len(rows),
            "macro_mean": _number(mean),
        }
    return aggregated


def _normalize_execution(execution: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = to_canonical(dict(execution or {}))
    if not isinstance(payload, dict):  # pragma: no cover - dict input guard
        raise MarketArtifactError("execution metadata must normalize to an object")

    has_completed = "completed" in payload
    has_truncated = "truncated" in payload
    if not has_completed and not has_truncated:
        payload["completed"] = True
        payload["truncated"] = False
    elif not has_completed:
        truncated = payload["truncated"]
        if not isinstance(truncated, bool):
            raise MarketArtifactError("execution.truncated must be a boolean")
        payload["completed"] = not truncated
    elif not has_truncated:
        completed = payload["completed"]
        if not isinstance(completed, bool):
            raise MarketArtifactError("execution.completed must be a boolean")
        payload["truncated"] = not completed
    payload.setdefault("stop_reason", None)
    _verify_execution(payload)
    return payload


def _verify_execution(execution: Mapping[str, Any]) -> None:
    completed = execution.get("completed")
    truncated = execution.get("truncated")
    stop_reason = execution.get("stop_reason")
    if not isinstance(completed, bool):
        raise MarketArtifactError("execution.completed must be a boolean")
    if not isinstance(truncated, bool):
        raise MarketArtifactError("execution.truncated must be a boolean")
    if completed == truncated:
        raise MarketArtifactError("execution must be exactly one of completed or truncated")
    if completed:
        if stop_reason is not None:
            raise MarketArtifactError("completed execution must have stop_reason=null")
    elif stop_reason not in MARKET_STOP_REASONS:
        raise MarketArtifactError(
            f"truncated execution stop_reason must be one of {sorted(MARKET_STOP_REASONS)}"
        )

    call_count = _optional_nonnegative_int(
        execution.get("model_call_count"),
        "execution.model_call_count",
    )
    call_limit = _optional_positive_int(
        execution.get("model_call_limit"),
        "execution.model_call_limit",
    )
    calls_used = _optional_nonnegative_int(
        execution.get("model_calls_used"),
        "execution.model_calls_used",
    )
    if (call_limit is None) != (calls_used is None):
        raise MarketArtifactError(
            "execution.model_call_limit and model_calls_used must both be set or null"
        )
    if call_limit is not None and calls_used is not None:
        if calls_used > call_limit:
            raise MarketArtifactError("execution.model_calls_used cannot exceed model_call_limit")
        if call_count is not None and call_count > calls_used:
            raise MarketArtifactError("execution.model_call_count cannot exceed model_calls_used")
    if stop_reason == "resource_limit":
        if call_limit is None or calls_used is None:
            raise MarketArtifactError(
                "resource_limit requires model_call_limit and model_calls_used"
            )
        if calls_used != call_limit:
            raise MarketArtifactError(
                "resource_limit requires model_calls_used == model_call_limit"
            )


def _execution_state(artifact: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    execution = artifact.get("execution")
    if not isinstance(execution, Mapping):  # verified before aggregation
        raise MarketArtifactError("market artifact execution must be an object")
    if artifact.get("schema_version") in {
        MARKET_ARTIFACT_SCHEMA,
        SEED_FREE_MARKET_ARTIFACT_SCHEMA,
    }:
        return (
            bool(execution["completed"]),
            bool(execution["truncated"]),
            False,
        )

    # V1 predates authenticated execution completion provenance.  An explicit
    # truncation marker is still useful, but absence of one must remain unknown:
    # at least one frozen provider pilot ended early without such a marker.
    truncated = (
        execution.get("truncated") is True or execution.get("stop_reason") == "resource_limit"
    )
    return False, truncated, not truncated


def _market_identity(artifact: Mapping[str, Any]) -> str:
    if artifact.get("schema_version") == SEED_FREE_MARKET_ARTIFACT_SCHEMA:
        return str(artifact.get("run_key", ""))
    return str(artifact.get("market_id", ""))


def _optional_nonnegative_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MarketArtifactError(f"{field} must be a non-negative integer or null")
    return int(value)


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MarketArtifactError(f"{field} must be a positive integer or null")
    return int(value)


def _input_payload(
    *,
    buyer_ids: Sequence[str],
    merchant_ids: Sequence[str],
    valuations: Sequence[BuyerValuation],
    merchant_floors: Sequence[MerchantFloor],
    transactions: Sequence[MarketTransaction],
    exposures: Sequence[Exposure],
    privacy_events: Sequence[PrivacyEvent],
    protocol_events: Sequence[ProtocolEvent],
    allocation_oracle: AllocationOracle | None,
) -> dict[str, Any]:
    return {
        "buyer_ids": list(buyer_ids),
        "merchant_ids": list(merchant_ids),
        "valuations": to_canonical([asdict(item) for item in valuations]),
        "merchant_floors": to_canonical([asdict(item) for item in merchant_floors]),
        "transactions": to_canonical([asdict(item) for item in transactions]),
        "exposures": to_canonical([asdict(item) for item in exposures]),
        "privacy_events": to_canonical([asdict(item) for item in privacy_events]),
        "protocol_events": to_canonical([asdict(item) for item in protocol_events]),
        "allocation_oracle": (
            None if allocation_oracle is None else to_canonical(asdict(allocation_oracle))
        ),
    }


def _parse_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        valuations = tuple(BuyerValuation(**row) for row in inputs["valuations"])
        merchant_floors = tuple(MerchantFloor(**row) for row in inputs["merchant_floors"])
        transactions = tuple(MarketTransaction(**row) for row in inputs["transactions"])
        exposures = tuple(
            Exposure(
                exposure_id=row["exposure_id"],
                buyer_id=row["buyer_id"],
                merchant_id=row["merchant_id"],
                listing_id=row.get("listing_id"),
                weight=Decimal(str(row.get("weight", "1"))),
            )
            for row in inputs["exposures"]
        )
        privacy_events = tuple(PrivacyEvent(**row) for row in inputs["privacy_events"])
        protocol_events = tuple(ProtocolEvent(**row) for row in inputs["protocol_events"])
        oracle_raw = inputs.get("allocation_oracle")
        allocation_oracle = None if oracle_raw is None else AllocationOracle(**oracle_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise MarketArtifactError(f"invalid normalized market inputs: {exc}") from exc
    return {
        "valuations": valuations,
        "merchant_floors": merchant_floors,
        "transactions": transactions,
        "exposures": exposures,
        "privacy_events": privacy_events,
        "protocol_events": protocol_events,
        "allocation_oracle": allocation_oracle,
    }


def _string_tuple(raw: Any, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise MarketArtifactError(f"{field} must be a list of strings")
    return tuple(raw)


def _assert_no_legacy_axes(value: Any, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.casefold() in _FORBIDDEN_V3_AXES:
                raise MarketArtifactError(
                    f"market artifact v3 contains forbidden legacy axis at {path}.{key}"
                )
            _assert_no_legacy_axes(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_legacy_axes(item, path=f"{path}[{index}]")


def _remove_legacy_axes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _remove_legacy_axes(item)
            for key, item in value.items()
            if str(key).casefold() not in _FORBIDDEN_V3_AXES
        }
    if isinstance(value, (list, tuple)):
        return [_remove_legacy_axes(item) for item in value]
    return value


def _number(value: Decimal | int | None) -> str | int | None:
    if isinstance(value, Decimal):
        return str(value)
    return value


def _digest(payload: Mapping[str, Any]) -> str:
    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


__all__ = [
    "LEGACY_MARKET_AGGREGATE_SCHEMA",
    "LEGACY_MARKET_ARTIFACT_SCHEMA",
    "MARKET_AGGREGATE_SCHEMA",
    "MARKET_ARTIFACT_SCHEMA",
    "MARKET_STOP_REASONS",
    "SEED_FREE_MARKET_ARTIFACT_SCHEMA",
    "MarketArtifactError",
    "aggregate_market_artifacts",
    "build_market_artifact",
    "convert_market_artifact_to_v3",
    "load_market_artifact",
    "market_metrics_to_dict",
    "verify_market_artifact",
    "write_market_artifact",
]
