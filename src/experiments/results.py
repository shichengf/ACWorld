"""Crash-safe result inventory and explicit resume semantics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from episode.termination import RESOURCE_LIMIT_STOP_REASON, SCOREABLE_STOP_REASONS
from experiments.plan import ExperimentPlan, RunSpec


RESULT_SCHEMA = "cwe.experiment-result.v1"
ATTEMPT_SCHEMA = "cwe.experiment-attempt.v1"
ATTEMPT_STATE_SCHEMA = "cwe.experiment-attempt-state.v1"
INVENTORY_SCHEMA = "cwe.experiment-inventory.v1"


def _validate_result_state(
    *,
    status: "ResultStatus",
    completed: bool | None,
    stop_reason: str | None,
    truncated: bool,
    metrics: Mapping[str, Any],
    label: str,
) -> None:
    """Validate the frozen contract for terminal and scoreable partial results."""

    if completed is not None and not isinstance(completed, bool):
        raise ValueError(f"{label} completed must be a boolean or null")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise ValueError(f"{label} stop_reason must be a string or null")
    if stop_reason is not None and stop_reason not in SCOREABLE_STOP_REASONS:
        allowed = ", ".join(sorted(SCOREABLE_STOP_REASONS))
        raise ValueError(f"{label} stop_reason must be one of: {allowed}")

    scoreable_abort = stop_reason in SCOREABLE_STOP_REASONS
    if truncated is not scoreable_abort:
        raise ValueError(
            f"{label} truncated must be true exactly for a scoreable stop_reason"
        )
    if scoreable_abort and status != ResultStatus.SUCCEEDED:
        raise ValueError(f"{label} scoreable stop_reason requires succeeded status")
    if scoreable_abort and completed is not False:
        raise ValueError(f"{label} scoreable stop_reason requires completed=false")

    resource_limited = metrics.get("resource_limited", False)
    if not isinstance(resource_limited, bool):
        raise ValueError(f"{label} metric 'resource_limited' must be a boolean")
    expected_resource_limited = stop_reason == RESOURCE_LIMIT_STOP_REASON
    if resource_limited is not expected_resource_limited:
        raise ValueError(
            f"{label} resource_limited must be true exactly for resource_limit"
        )
    if expected_resource_limited:
        limit = metrics.get("model_call_limit")
        used = metrics.get("model_call_count")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or isinstance(used, bool)
            or not isinstance(used, int)
            or used != limit
        ):
            raise ValueError(
                f"{label} resource_limit requires a positive model_call_limit "
                "equal to model_call_count"
            )


class ResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ResultCorruptionError(RuntimeError):
    """A persisted result cannot be trusted for resume decisions."""


@dataclass(frozen=True)
class ResultAttempt:
    """One superseded execution attempt retained inside the latest result.

    The current attempt remains at the top level of :class:`RunResult`; this
    history contains only prior attempts, so ordinary one-shot runs do not
    duplicate their observations.  The observation records are already
    sanitised by ``RecordingChannel`` and contain hashes and lengths rather
    than raw prompts or model output.
    """

    attempt_index: int
    run_key: str
    status: ResultStatus
    metrics: dict[str, bool | float | int | None] = field(default_factory=dict)
    completed: bool | None = None
    failure_mode: str | None = None
    stop_reason: str | None = None
    truncated: bool = False
    error: str | None = None
    observations: tuple[dict[str, Any], ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    contract_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    schema_version: str = ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ATTEMPT_SCHEMA:
            raise ValueError(f"unsupported attempt schema: {self.schema_version!r}")
        if isinstance(self.attempt_index, bool) or self.attempt_index <= 0:
            raise ValueError("attempt_index must be a positive integer")
        if not self.run_key:
            raise ValueError("attempt run_key must be non-empty")
        if self.status == ResultStatus.SUCCEEDED and self.error:
            raise ValueError("a succeeded attempt cannot contain an error")
        if not isinstance(self.metrics, dict):
            raise ValueError("attempt metrics must be a dictionary")
        if not isinstance(self.truncated, bool):
            raise ValueError("attempt truncated must be a boolean")
        _validate_result_state(
            status=self.status,
            completed=self.completed,
            stop_reason=self.stop_reason,
            truncated=self.truncated,
            metrics=self.metrics,
            label="attempt",
        )
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not (
                value is None or isinstance(value, (bool, int, float))
            ):
                raise ValueError(
                    f"attempt metric {name!r} must be bool, int, float, or null"
                )

    @classmethod
    def from_result(cls, result: "RunResult", *, attempt_index: int) -> "ResultAttempt":
        return cls(
            attempt_index=attempt_index,
            run_key=result.run_key,
            status=result.status,
            metrics=dict(result.metrics),
            completed=result.completed,
            failure_mode=result.failure_mode,
            stop_reason=result.stop_reason,
            truncated=result.truncated,
            error=result.error,
            observations=tuple(dict(item) for item in result.observations),
            artifacts=dict(result.artifacts),
            contract_id=result.contract_id,
            started_at=result.started_at,
            finished_at=result.finished_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_index": self.attempt_index,
            "run_key": self.run_key,
            "status": self.status.value,
            "metrics": self.metrics,
            "completed": self.completed,
            "failure_mode": self.failure_mode,
            "stop_reason": self.stop_reason,
            "truncated": self.truncated,
            "error": self.error,
            "observations": list(self.observations),
            "artifacts": self.artifacts,
            "contract_id": self.contract_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ResultAttempt":
        if str(raw.get("schema_version", "")) != ATTEMPT_SCHEMA:
            raise ResultCorruptionError(
                f"unsupported attempt schema: {raw.get('schema_version')!r}"
            )
        try:
            metrics_raw = raw.get("metrics", {})
            observations_raw = raw.get("observations", [])
            artifacts_raw = raw.get("artifacts", {})
            truncated_raw = raw.get("truncated", False)
            attempt_index_raw = raw.get("attempt_index")
            if not isinstance(metrics_raw, dict) or not isinstance(observations_raw, list):
                raise ValueError("attempt metrics must be an object and observations a list")
            if not isinstance(artifacts_raw, dict):
                raise ValueError("attempt artifacts must be an object")
            if not isinstance(truncated_raw, bool):
                raise ValueError("attempt truncated must be a boolean")
            if isinstance(attempt_index_raw, bool) or not isinstance(
                attempt_index_raw, int
            ):
                raise ValueError("attempt_index must be an integer")
            return cls(
                attempt_index=attempt_index_raw,
                run_key=str(raw["run_key"]),
                status=ResultStatus(str(raw["status"])),
                metrics={str(key): value for key, value in metrics_raw.items()},
                completed=raw.get("completed"),
                failure_mode=(
                    str(raw["failure_mode"])
                    if raw.get("failure_mode") is not None
                    else None
                ),
                stop_reason=(
                    str(raw["stop_reason"])
                    if raw.get("stop_reason") is not None
                    else None
                ),
                truncated=truncated_raw,
                error=_safe_persisted_error(
                    str(raw["error"]) if raw.get("error") is not None else None
                ),
                observations=_sanitize_observations(observations_raw),
                artifacts=_prefix_artifacts(artifacts_raw, None),
                contract_id=(
                    str(raw["contract_id"])
                    if raw.get("contract_id") is not None
                    else None
                ),
                started_at=(
                    str(raw["started_at"])
                    if raw.get("started_at") is not None
                    else None
                ),
                finished_at=(
                    str(raw["finished_at"])
                    if raw.get("finished_at") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultCorruptionError(f"invalid experiment attempt: {exc}") from exc


@dataclass(frozen=True)
class RunResult:
    run: RunSpec
    status: ResultStatus
    metrics: dict[str, bool | float | int | None] = field(default_factory=dict)
    completed: bool | None = None
    failure_mode: str | None = None
    stop_reason: str | None = None
    truncated: bool = False
    error: str | None = None
    observations: tuple[dict[str, Any], ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    contract_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    attempt_history: tuple[ResultAttempt, ...] = ()
    schema_version: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA:
            raise ValueError(f"unsupported result schema: {self.schema_version!r}")
        if self.status == ResultStatus.SUCCEEDED and self.error:
            raise ValueError("a succeeded result cannot contain an error")
        if not isinstance(self.metrics, dict):
            raise ValueError("result metrics must be a dictionary")
        if not isinstance(self.truncated, bool):
            raise ValueError("result truncated must be a boolean")
        _validate_result_state(
            status=self.status,
            completed=self.completed,
            stop_reason=self.stop_reason,
            truncated=self.truncated,
            metrics=self.metrics,
            label="result",
        )
        if self.contract_id is not None and not self.contract_id.strip():
            raise ValueError("contract_id must be a non-empty string or null")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not (
                value is None or isinstance(value, (bool, int, float))
            ):
                raise ValueError(f"metric {name!r} must be bool, int, float, or null")
        for expected_index, attempt in enumerate(self.attempt_history, start=1):
            if attempt.attempt_index != expected_index:
                raise ValueError("attempt_history indices must be contiguous from one")
            if attempt.run_key != self.run_key:
                raise ValueError("attempt_history run_key must match the current result")
            if attempt.contract_id != self.contract_id:
                raise ValueError("attempt_history contract_id must match the current result")

    @property
    def run_key(self) -> str:
        return self.run.run_key

    @property
    def attempt_count(self) -> int:
        return len(self.attempt_history) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_key": self.run_key,
            "run": self.run.to_dict(),
            "status": self.status.value,
            "metrics": self.metrics,
            "completed": self.completed,
            "failure_mode": self.failure_mode,
            "stop_reason": self.stop_reason,
            "truncated": self.truncated,
            "error": self.error,
            "observations": list(self.observations),
            "artifacts": self.artifacts,
            "contract_id": self.contract_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempt_history": [attempt.to_dict() for attempt in self.attempt_history],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunResult":
        if str(raw.get("schema_version", "")) != RESULT_SCHEMA:
            raise ResultCorruptionError(f"unsupported result schema: {raw.get('schema_version')!r}")
        try:
            run = RunSpec.from_dict(raw["run"])
            status = ResultStatus(str(raw["status"]))
            metrics_raw = raw.get("metrics", {})
            observations_raw = raw.get("observations", [])
            artifacts_raw = raw.get("artifacts", {})
            attempt_history_raw = raw.get("attempt_history", [])
            truncated_raw = raw.get("truncated", False)
            if not isinstance(metrics_raw, dict) or not isinstance(observations_raw, list):
                raise ValueError("metrics must be an object and observations must be a list")
            if not isinstance(artifacts_raw, dict):
                raise ValueError("artifacts must be an object")
            if not isinstance(attempt_history_raw, list):
                raise ValueError("attempt_history must be a list")
            if not isinstance(truncated_raw, bool):
                raise ValueError("truncated must be a boolean")
            result = cls(
                run=run,
                status=status,
                metrics={str(key): value for key, value in metrics_raw.items()},
                completed=raw.get("completed"),
                failure_mode=(str(raw["failure_mode"])
                              if raw.get("failure_mode") is not None else None),
                stop_reason=(str(raw["stop_reason"])
                             if raw.get("stop_reason") is not None else None),
                truncated=truncated_raw,
                error=_safe_persisted_error(
                    str(raw["error"]) if raw.get("error") is not None else None
                ),
                observations=_sanitize_observations(observations_raw),
                artifacts=_prefix_artifacts(artifacts_raw, None),
                contract_id=(
                    str(raw["contract_id"])
                    if raw.get("contract_id") is not None
                    else None
                ),
                started_at=(str(raw["started_at"])
                            if raw.get("started_at") is not None else None),
                finished_at=(str(raw["finished_at"])
                             if raw.get("finished_at") is not None else None),
                attempt_history=tuple(
                    ResultAttempt.from_dict(dict(item))
                    for item in attempt_history_raw
                ),
            )
        except (KeyError, TypeError, ValueError, ResultCorruptionError) as exc:
            raise ResultCorruptionError(f"invalid experiment result: {exc}") from exc
        if str(raw.get("run_key", "")) != result.run_key:
            raise ResultCorruptionError("result run_key does not match its run identity")
        return result


@dataclass(frozen=True)
class InventoryEntry:
    run_key: str
    status: str
    result_path: str
    error: str | None = None
    attempt_count: int = 0
    truncated: bool = False
    resource_limited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "status": self.status,
            "result_path": self.result_path,
            "error": self.error,
            "attempt_count": self.attempt_count,
            "truncated": self.truncated,
            "resource_limited": self.resource_limited,
        }


@dataclass(frozen=True)
class ResultInventory:
    manifest_id: str
    entries: tuple[InventoryEntry, ...]
    contract_id: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        values = {"pending": 0, "succeeded": 0, "failed": 0, "invalid": 0}
        for entry in self.entries:
            values[entry.status] = values.get(entry.status, 0) + 1
        return values

    @property
    def outcome_counts(self) -> dict[str, int]:
        return {
            "truncated": sum(entry.truncated for entry in self.entries),
            "resource_limited": sum(
                entry.resource_limited for entry in self.entries
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INVENTORY_SCHEMA,
            "manifest_id": self.manifest_id,
            "contract_id": self.contract_id,
            "planned_runs": len(self.entries),
            "counts": self.counts,
            "outcome_counts": self.outcome_counts,
            "entries": [entry.to_dict() for entry in self.entries],
        }


class ResultStore:
    """One ``result.json`` per run, with atomic replacement and validation."""

    def __init__(self, root: str | Path, *, contract_id: str | None = None) -> None:
        self.root = Path(root)
        if contract_id is not None and not contract_id.strip():
            raise ValueError("contract_id must be a non-empty string or null")
        self.contract_id = contract_id

    def run_dir(self, run: RunSpec) -> Path:
        return self.root / "runs" / run.run_key

    def result_path(self, run: RunSpec) -> Path:
        return self.run_dir(run) / "result.json"

    def attempt_dir(self, run: RunSpec, attempt_index: int) -> Path:
        if isinstance(attempt_index, bool) or attempt_index <= 0:
            raise ValueError("attempt_index must be a positive integer")
        return self.run_dir(run) / "attempts" / f"{attempt_index:04d}"

    def next_attempt_index(self, run: RunSpec) -> int:
        existing = self.read(run)
        recorded_next = 1 if existing is None else existing.attempt_count + 1
        attempts_root = self.run_dir(run) / "attempts"
        occupied = {
            int(path.name)
            for path in attempts_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        } if attempts_root.is_dir() else set()
        return max(recorded_next, max(occupied, default=0) + 1)

    def begin_attempt(
        self,
        run: RunSpec,
        *,
        started_at: str,
    ) -> tuple[Path, str]:
        """Reserve a never-overwritten artifact slot and persist a start marker."""

        physical_index = self.next_attempt_index(run)
        while True:
            attempt_dir = self.attempt_dir(run, physical_index)
            try:
                attempt_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                physical_index += 1
                continue
            break
        artifact_prefix = attempt_dir.relative_to(self.run_dir(run)).as_posix()
        self._write_attempt_state(
            attempt_dir,
            {
                "schema_version": ATTEMPT_STATE_SCHEMA,
                "run_key": run.run_key,
                # Attempt-state indices include interrupted/orphaned executions,
                # unlike result history, which contains terminal records only.
                "attempt_index": physical_index,
                "artifact_slot": physical_index,
                "status": "started",
                "contract_id": self.contract_id,
                "started_at": started_at,
                "finished_at": None,
                "error": None,
            },
        )
        return attempt_dir, artifact_prefix

    def finish_attempt(
        self,
        attempt_dir: Path,
        *,
        result: RunResult,
    ) -> Path:
        """Mark a reserved attempt terminal without persisting sensitive content."""

        marker = attempt_dir / "attempt.json"
        try:
            started = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResultCorruptionError(f"cannot read attempt marker {marker}: {exc}") from exc
        if not isinstance(started, dict) or started.get("run_key") != result.run_key:
            raise ResultCorruptionError(f"attempt marker identity mismatch: {marker}")
        terminal = dict(started)
        terminal.update({
            "status": result.status.value,
            "finished_at": result.finished_at,
            "error": _safe_persisted_error(result.error),
        })
        return self._write_attempt_state(attempt_dir, terminal)

    @staticmethod
    def _write_attempt_state(
        attempt_dir: Path,
        payload: Mapping[str, Any],
    ) -> Path:
        marker = attempt_dir / "attempt.json"
        _atomic_json_file(marker, payload)
        return marker

    def read(self, run: RunSpec) -> RunResult | None:
        path = self.result_path(run)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResultCorruptionError(f"cannot read {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ResultCorruptionError(f"result root is not an object: {path}")
        result = RunResult.from_dict(raw)
        if result.run != run:
            raise ResultCorruptionError(f"result identity does not match planned run: {path}")
        if self.contract_id is not None and result.contract_id != self.contract_id:
            raise ResultCorruptionError(
                "result contract_id does not match the active experiment contract: "
                f"stored={result.contract_id!r}, active={self.contract_id!r}"
            )
        return result

    def write(self, result: RunResult) -> Path:
        """Atomically replace a result without manufacturing an attempt."""

        normalized = _normalize_result(result)
        self._validate_contract(normalized)
        return self._write_result(normalized)

    def append_attempt(self, result: RunResult) -> Path:
        """Persist one runner attempt and retain the previously current result."""

        normalized = _normalize_result(result)
        self._validate_contract(normalized)
        existing = self.read(normalized.run)
        if existing is not None:
            history = (
                *existing.attempt_history,
                ResultAttempt.from_result(
                    existing,
                    attempt_index=existing.attempt_count,
                ),
            )
            normalized = _normalize_result(
                replace(normalized, attempt_history=history)
            )
        return self._write_result(normalized)

    def _validate_contract(self, result: RunResult) -> None:
        if self.contract_id is not None and result.contract_id != self.contract_id:
            raise ResultCorruptionError(
                "refusing to write a result outside the active experiment contract: "
                f"result={result.contract_id!r}, active={self.contract_id!r}"
            )

    def _write_result(self, result: RunResult) -> Path:
        path = self.result_path(result.run)
        _atomic_json_file(path, result.to_dict())
        return path

    def inventory(self, plan: ExperimentPlan) -> ResultInventory:
        entries: list[InventoryEntry] = []
        for run in plan.runs:
            path = self.result_path(run)
            try:
                result = self.read(run)
            except ResultCorruptionError as exc:
                entries.append(InventoryEntry(
                    run_key=run.run_key,
                    status="invalid",
                    result_path=str(path),
                    error=str(exc),
                ))
                continue
            entries.append(InventoryEntry(
                run_key=run.run_key,
                status=(result.status.value if result is not None else "pending"),
                result_path=str(path),
                attempt_count=(result.attempt_count if result is not None else 0),
                truncated=(result.truncated if result is not None else False),
                resource_limited=(
                    result.stop_reason == RESOURCE_LIMIT_STOP_REASON
                    if result is not None else False
                ),
            ))
        return ResultInventory(
            manifest_id=plan.manifest_id,
            entries=tuple(entries),
            contract_id=self.contract_id,
        )

    def write_inventory(self, plan: ExperimentPlan, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.root / "inventory.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.inventory(plan).to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return target


RunExecutor = Callable[[RunSpec, Path], Mapping[str, Any]]


@dataclass(frozen=True)
class ExecutionSummary:
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    stopped_early: bool = False
    failure_limit_reached: bool = False
    truncated: int = 0
    resource_limited: int = 0


class ExperimentRunner:
    """Resume-aware coordinator; callers must explicitly inject an executor.

    Formal paid execution lives in :mod:`experiments.benchmark_executor`; this
    generic coordinator remains provider-agnostic and equally usable by offline
    non-benchmark studies.
    """

    def __init__(self, store: ResultStore) -> None:
        self.store = store

    def run(
        self,
        plan: ExperimentPlan,
        executor: RunExecutor,
        *,
        retry_failed: bool = False,
        max_failures: int | None = None,
    ) -> ExecutionSummary:
        if max_failures is not None and (
            isinstance(max_failures, bool)
            or not isinstance(max_failures, int)
            or max_failures <= 0
        ):
            raise ValueError("max_failures must be a positive integer or null")

        attempted = succeeded = failed = skipped = 0
        truncated = resource_limited = 0
        stopped_early = failure_limit_reached = False
        for run_index, run in enumerate(plan.runs):
            existing = self.store.read(run)
            if existing is not None and (
                existing.status == ResultStatus.SUCCEEDED
                or (existing.status == ResultStatus.FAILED and not retry_failed)
            ):
                skipped += 1
                continue

            attempted += 1
            started = _now()
            attempt_dir, artifact_prefix = self.store.begin_attempt(
                run,
                started_at=started,
            )
            try:
                payload = executor(run, attempt_dir)
                result = _successful_result(
                    run,
                    payload,
                    started,
                    contract_id=self.store.contract_id,
                    artifact_prefix=artifact_prefix,
                )
            except Exception as exc:  # noqa: BLE001 - persist failures for resumability
                failure_observations = getattr(exc, "observations", ())
                if not isinstance(failure_observations, (list, tuple)):
                    failure_observations = ()
                result = RunResult(
                    run=run,
                    status=ResultStatus.FAILED,
                    contract_id=self.store.contract_id,
                    error=_safe_exception_fingerprint(exc),
                    observations=_sanitize_observations(failure_observations),
                    artifacts={"attempt_dir": artifact_prefix},
                    started_at=started,
                    finished_at=_now(),
                )
            self.store.finish_attempt(attempt_dir, result=result)
            self.store.append_attempt(result)
            if result.status == ResultStatus.SUCCEEDED:
                succeeded += 1
                if result.truncated:
                    truncated += 1
                if result.stop_reason == RESOURCE_LIMIT_STOP_REASON:
                    resource_limited += 1
            else:
                failed += 1
                if max_failures is not None and failed >= max_failures:
                    failure_limit_reached = True
                    stopped_early = run_index + 1 < len(plan.runs)
                    break
        self.store.write_inventory(plan)
        return ExecutionSummary(
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            stopped_early=stopped_early,
            failure_limit_reached=failure_limit_reached,
            truncated=truncated,
            resource_limited=resource_limited,
        )


def _successful_result(
    run: RunSpec,
    payload: Mapping[str, Any],
    started: str,
    *,
    contract_id: str | None = None,
    artifact_prefix: str | None = None,
) -> RunResult:
    metrics_raw = payload.get("metrics", {})
    observations_raw = payload.get("observations", ())
    artifacts_raw = payload.get("artifacts", {})
    if not isinstance(metrics_raw, Mapping):
        raise ValueError("executor metrics must be a mapping")
    if not isinstance(observations_raw, (list, tuple)):
        raise ValueError("executor observations must be a list or tuple")
    if not isinstance(artifacts_raw, Mapping):
        raise ValueError("executor artifacts must be a mapping")
    completed = payload.get("completed") if "completed" in payload else None
    truncated = payload.get("truncated", False)
    if completed is not None and not isinstance(completed, bool):
        raise ValueError("executor completed must be a boolean or null")
    if not isinstance(truncated, bool):
        raise ValueError("executor truncated must be a boolean")
    return RunResult(
        run=run,
        status=ResultStatus.SUCCEEDED,
        metrics={str(key): value for key, value in metrics_raw.items()},
        completed=completed,
        failure_mode=(str(payload["failure_mode"])
                      if payload.get("failure_mode") is not None else None),
        stop_reason=(str(payload["stop_reason"])
                     if payload.get("stop_reason") is not None else None),
        truncated=truncated,
        observations=_sanitize_observations(observations_raw),
        artifacts=_prefix_artifacts(artifacts_raw, artifact_prefix),
        contract_id=contract_id,
        started_at=started,
        finished_at=_now(),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_OBSERVATION_FIELDS = frozenset({
    "actor_id",
    "role",
    "source",
    "model_id",
    "call_index",
    "latency_seconds",
    "system_chars",
    "user_chars",
    "output_chars",
    "system_prompt_sha256",
    "user_prompt_sha256",
    "output_sha256",
    "error_type",
    "error_code",
})
_NULLABLE_OBSERVATION_FIELDS = frozenset({
    "model_id",
    "output_chars",
    "output_sha256",
    "error_type",
    "error_code",
})
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,127}$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_ARTIFACT_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DROP = object()


def _sanitize_observations(rows: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(rows, (list, tuple)):
        return ()
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        projected: dict[str, Any] = {}
        for key in _OBSERVATION_FIELDS:
            if key not in row:
                continue
            value = _validated_observation_value(key, row[key])
            if value is not _DROP:
                projected[key] = value
        if projected:
            sanitized.append(projected)
    return tuple(sanitized)


def _validated_observation_value(name: str, value: Any) -> Any:
    if value is None:
        return None if name in _NULLABLE_OBSERVATION_FIELDS else _DROP
    if name in {"actor_id", "model_id"}:
        return (
            value
            if isinstance(value, str) and _IDENTIFIER_PATTERN.fullmatch(value)
            else _DROP
        )
    if name == "role":
        return value if value in {"buyer", "merchant", "platform"} else _DROP
    if name == "source":
        return value if value in {"model", "reference"} else _DROP
    if name in {"call_index", "system_chars", "user_chars", "output_chars"}:
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else _DROP
        )
    if name == "latency_seconds":
        return (
            value
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            else _DROP
        )
    if name in {
        "system_prompt_sha256",
        "user_prompt_sha256",
        "output_sha256",
    }:
        return (
            value
            if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
            else _DROP
        )
    if name == "error_type":
        return (
            value
            if isinstance(value, str) and _ERROR_TYPE_PATTERN.fullmatch(value)
            else _DROP
        )
    if name == "error_code":
        return value if value in SCOREABLE_STOP_REASONS else _DROP
    return _DROP


def _safe_exception_fingerprint(exc: Exception) -> str:
    message = str(exc)
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    error_type = type(exc).__name__
    if not _ERROR_TYPE_PATTERN.fullmatch(error_type):
        error_type = "Exception"
    return (
        f"{error_type}: message_chars={len(message)} "
        f"message_sha256={digest}"
    )


_SAFE_ERROR_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*: message_chars=\d+ message_sha256=[0-9a-f]{64}$"
)


def _safe_persisted_error(error: str | None) -> str | None:
    if error is None or _SAFE_ERROR_PATTERN.fullmatch(error):
        return error
    digest = hashlib.sha256(error.encode("utf-8")).hexdigest()
    return (
        f"PersistedError: message_chars={len(error)} "
        f"message_sha256={digest}"
    )


def _normalize_attempt(attempt: ResultAttempt) -> ResultAttempt:
    return replace(
        attempt,
        error=_safe_persisted_error(attempt.error),
        observations=_sanitize_observations(attempt.observations),
        artifacts=_prefix_artifacts(attempt.artifacts, None),
    )


def _normalize_result(result: RunResult) -> RunResult:
    return replace(
        result,
        error=_safe_persisted_error(result.error),
        observations=_sanitize_observations(result.observations),
        artifacts=_prefix_artifacts(result.artifacts, None),
        attempt_history=tuple(
            _normalize_attempt(attempt) for attempt in result.attempt_history
        ),
    )


def _prefix_artifacts(
    artifacts: Mapping[Any, Any],
    prefix: str | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in artifacts.items():
        if not isinstance(key, str) or not _ARTIFACT_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"executor artifact key {key!r} is invalid")
        if not isinstance(value, str):
            raise ValueError(f"executor artifact {key!r} path must be a string")
        if (
            not value
            or len(value) > 512
            or any(character in value for character in ("\\", "\x00", "\n", "\r"))
        ):
            raise ValueError(f"executor artifact {key!r} must be a safe relative path")
        relative = Path(value)
        if relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"executor artifact {key!r} must be a safe relative path")
        result[key] = (
            (Path(prefix) / relative).as_posix()
            if prefix is not None
            else relative.as_posix()
        )
    return result


def _atomic_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "ATTEMPT_SCHEMA",
    "ATTEMPT_STATE_SCHEMA",
    "ExecutionSummary",
    "ExperimentRunner",
    "INVENTORY_SCHEMA",
    "InventoryEntry",
    "RESULT_SCHEMA",
    "ResultCorruptionError",
    "ResultInventory",
    "ResultStatus",
    "ResultAttempt",
    "ResultStore",
    "RunExecutor",
    "RunResult",
    "SCOREABLE_STOP_REASONS",
]
