"""Deterministic execution seam for registered scenario extensions.

World events run after declarative world seeding and before the episode's
authoritative initial snapshot. Metrics and oracle primitives run after the
final snapshot exists. Both launch paths use these helpers and write the same
``extensions.json`` evidence artifact; extension results never alter the
benchmark's deterministic headline score.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from episode.extensions import MARKET_METRICS, ORACLE_PRIMITIVES, WORLD_EVENT_HANDLERS

if TYPE_CHECKING:
    from episode.types import ScenarioSpec


class ExtensionExecutionError(RuntimeError):
    """A declared extension cannot be resolved, invoked, or serialized."""


@dataclass(frozen=True, slots=True)
class WorldEventExecution:
    """One completed registered world event."""

    event_id: str
    logical_time: int
    handler: str
    version: str
    payload: Mapping[str, Any]
    result: Any


@dataclass(frozen=True, slots=True)
class ExtensionEvaluationContext:
    """Read-only context supplied to registered metrics and oracle primitives."""

    scenario: "ScenarioSpec"
    initial_snapshot: Any
    final_snapshot: Any
    world_events: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExtensionEvaluationExecution:
    """One completed registered metric or oracle invocation."""

    evaluation_id: str
    kind: str
    name: str
    version: str
    arguments: Mapping[str, Any]
    result: Any


def apply_registered_world_events(
    world: Any,
    scenario: "ScenarioSpec",
) -> tuple[WorldEventExecution, ...]:
    """Apply declared events in stable logical-time/event-id order."""
    seen: set[str] = set()
    executions: list[WorldEventExecution] = []
    for event in sorted(
        scenario.world_events,
        key=lambda item: (item.logical_time, item.event_id),
    ):
        if not event.event_id or event.event_id in seen:
            raise ExtensionExecutionError(
                f"invalid or duplicate world event id {event.event_id!r}"
            )
        if event.logical_time < 0:
            raise ExtensionExecutionError(
                f"world event {event.event_id!r} has negative logical_time"
            )
        try:
            record = WORLD_EVENT_HANDLERS.record(event.handler)
        except KeyError as exc:
            raise ExtensionExecutionError(
                f"world event {event.event_id!r} uses unknown handler {event.handler!r}"
            ) from exc
        payload = MappingProxyType(dict(event.payload))
        try:
            result = record.component(world, payload)
        except Exception as exc:  # noqa: BLE001 - preserve extension identity in the error
            raise ExtensionExecutionError(
                f"world event {event.event_id!r} ({event.handler}) failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        executions.append(WorldEventExecution(
            event_id=event.event_id,
            logical_time=event.logical_time,
            handler=event.handler,
            version=record.version,
            payload=payload,
            result=result,
        ))
        seen.add(event.event_id)
    return tuple(executions)


def evaluate_registered_extensions(
    scenario: "ScenarioSpec",
    *,
    initial_snapshot: Any,
    final_snapshot: Any,
    world_events: tuple[WorldEventExecution, ...],
) -> tuple[ExtensionEvaluationExecution, ...]:
    """Evaluate declared metrics/oracles by stable evaluation id."""
    event_results = MappingProxyType({event.event_id: event.result for event in world_events})
    context = ExtensionEvaluationContext(
        scenario=scenario,
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
        world_events=event_results,
    )
    seen: set[str] = set()
    executions: list[ExtensionEvaluationExecution] = []
    for evaluation in sorted(
        scenario.extension_evaluations,
        key=lambda item: item.evaluation_id,
    ):
        if not evaluation.evaluation_id or evaluation.evaluation_id in seen:
            raise ExtensionExecutionError(
                f"invalid or duplicate extension evaluation id {evaluation.evaluation_id!r}"
            )
        registry = (
            MARKET_METRICS
            if evaluation.kind == "market_metric"
            else ORACLE_PRIMITIVES
            if evaluation.kind == "oracle_primitive"
            else None
        )
        if registry is None:
            raise ExtensionExecutionError(
                f"extension evaluation {evaluation.evaluation_id!r} has unknown kind "
                f"{evaluation.kind!r}"
            )
        try:
            record = registry.record(evaluation.name)
        except KeyError as exc:
            raise ExtensionExecutionError(
                f"extension evaluation {evaluation.evaluation_id!r} uses unknown "
                f"{evaluation.kind} {evaluation.name!r}"
            ) from exc
        arguments = dict(evaluation.arguments)
        if "context" in arguments:
            raise ExtensionExecutionError(
                f"extension evaluation {evaluation.evaluation_id!r} arguments reserve "
                "the key 'context'"
            )
        try:
            result = record.component(context=context, **arguments)
        except Exception as exc:  # noqa: BLE001 - preserve extension identity in the error
            raise ExtensionExecutionError(
                f"extension evaluation {evaluation.evaluation_id!r} "
                f"({evaluation.name}) failed: {type(exc).__name__}: {exc}"
            ) from exc
        executions.append(ExtensionEvaluationExecution(
            evaluation_id=evaluation.evaluation_id,
            kind=evaluation.kind,
            name=evaluation.name,
            version=record.version,
            arguments=MappingProxyType(arguments),
            result=result,
        ))
        seen.add(evaluation.evaluation_id)
    return tuple(executions)


def write_extension_artifact(
    path: str | Path,
    *,
    scenario: "ScenarioSpec",
    world_events: tuple[WorldEventExecution, ...],
    evaluations: tuple[ExtensionEvaluationExecution, ...],
) -> None:
    """Write canonical extension evidence when the scenario declares extensions."""
    if not scenario.world_events and not scenario.extension_evaluations:
        return
    payload = {
        "schema_version": "commerceworld.extensions.v1",
        "scenario_id": scenario.scenario_id,
        "world_events": list(world_events),
        "evaluations": list(evaluations),
    }
    try:
        serializable = _jsonable(payload)
        body = json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ExtensionExecutionError(
            f"scenario {scenario.scenario_id!r} extension result is not serializable: {exc}"
        ) from exc
    Path(path).write_text(body + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Convert common deterministic value objects to JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable({field.name: getattr(value, field.name) for field in fields(value)})
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    raise TypeError(f"unsupported value type {type(value).__name__}")


__all__ = [
    "ExtensionEvaluationContext",
    "ExtensionEvaluationExecution",
    "ExtensionExecutionError",
    "WorldEventExecution",
    "apply_registered_world_events",
    "evaluate_registered_extensions",
    "write_extension_artifact",
]
