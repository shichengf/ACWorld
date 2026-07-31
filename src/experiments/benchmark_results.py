"""Crash safe result persistence for the ACWorld capability benchmark.

The v1 result inventory remains the compatibility surface for the historical
seeded benchmark.  This module intentionally uses a separate schema and the
``RunSpecV2`` identity.  A terminal record is atomically installed at
``runs/<run-key>/result.json``; executor artifacts live in never-reused attempt
directories so a retry cannot overwrite evidence from an earlier execution.

Only sanitized request/response metadata is persisted.  Raw prompts, raw model
outputs, provider exception messages, API keys, and legacy seed/rollout axes are
not part of the v2 result contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from episode.termination import RESOURCE_LIMIT_STOP_REASON, SCOREABLE_STOP_REASONS
from experiments.benchmark_plan import ExperimentPlanV2, RunSpecV2
from experiments.recording import PRE_PROVIDER_CALL_ERROR_CODE
from experiments.results import ResultCorruptionError, ResultStatus
from experiments.run_diagnostics import RunDiagnosticsV1


RESULT_SCHEMA_V2 = "cwe.experiment-result.v2"
ATTEMPT_STATE_SCHEMA_V2 = "cwe.experiment-attempt-state.v2"
ATTEMPT_LEDGER_ACCOUNTING_FIELD_V2 = "ledger_call_accounting"
ATTEMPT_LEDGER_ACCOUNTING_SCHEMA_V2 = "cwe.formal-attempt-call-accounting.v3"


def _validate_result_state(
    *,
    status: ResultStatus,
    completed: bool | None,
    stop_reason: str | None,
    truncated: bool,
    metrics: Mapping[str, Any],
) -> None:
    if completed is not None and not isinstance(completed, bool):
        raise ValueError("completed must be a boolean or null")
    if not isinstance(truncated, bool):
        raise ValueError("truncated must be a boolean")
    if stop_reason is not None and stop_reason not in SCOREABLE_STOP_REASONS:
        allowed = ", ".join(sorted(SCOREABLE_STOP_REASONS))
        raise ValueError(f"stop_reason must be one of: {allowed}")

    scoreable_stop = stop_reason in SCOREABLE_STOP_REASONS
    if truncated is not scoreable_stop:
        raise ValueError("truncated must be true exactly for a scoreable stop_reason")
    if scoreable_stop and status != ResultStatus.SUCCEEDED:
        raise ValueError("a scoreable stop_reason requires succeeded status")
    if scoreable_stop and completed is not False:
        raise ValueError("a scoreable stop_reason requires completed=false")

    resource_limited = metrics.get("resource_limited", False)
    if not isinstance(resource_limited, bool):
        raise ValueError("metric 'resource_limited' must be a boolean")
    expected_resource_limited = stop_reason == RESOURCE_LIMIT_STOP_REASON
    if resource_limited is not expected_resource_limited:
        raise ValueError("resource_limited must be true exactly for stop_reason=resource_limit")
    if expected_resource_limited:
        limit = metrics.get("model_call_limit")
        count = metrics.get("model_call_count")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count != limit
        ):
            raise ValueError(
                "resource_limit requires a positive model_call_limit equal to model_call_count"
            )


@dataclass(frozen=True)
class RunResultV2:
    """One current terminal result for an immutable v2 run identity."""

    run: RunSpecV2
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
    source_revision: str | None = None
    attempt_index: int = 1
    started_at: str | None = None
    finished_at: str | None = None
    schema_version: str = RESULT_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_V2:
            raise ValueError(f"unsupported v2 result schema: {self.schema_version!r}")
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or self.attempt_index <= 0
        ):
            raise ValueError("attempt_index must be a positive integer")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("error must be a string or null")
        if self.status == ResultStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a succeeded result cannot contain an error")
        if not isinstance(self.metrics, dict):
            raise ValueError("metrics must be a dictionary")
        _validate_metrics(self.metrics)
        _validate_result_state(
            status=self.status,
            completed=self.completed,
            stop_reason=self.stop_reason,
            truncated=self.truncated,
            metrics=self.metrics,
        )
        if self.failure_mode is not None:
            if not isinstance(self.failure_mode, str) or not _SAFE_LABEL_PATTERN.fullmatch(
                self.failure_mode
            ):
                raise ValueError("failure_mode must be a safe identifier or null")
        if self.contract_id is not None:
            _validate_contract_id(self.contract_id)
        _validate_source_revision(self.source_revision)
        _validate_timestamp(self.started_at, label="started_at")
        _validate_timestamp(self.finished_at, label="finished_at")

    @property
    def run_key(self) -> str:
        return str(self.run.run_key)

    def to_dict(self) -> dict[str, Any]:
        """Return a persistence-safe representation even for direct callers."""

        return {
            "schema_version": self.schema_version,
            "run_key": self.run_key,
            "run": self.run.to_dict(),
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "completed": self.completed,
            "failure_mode": self.failure_mode,
            "stop_reason": self.stop_reason,
            "truncated": self.truncated,
            "error": _safe_persisted_error(self.error),
            "observations": list(_sanitize_observations(self.observations)),
            "artifacts": _normalize_artifacts(self.artifacts),
            "contract_id": self.contract_id,
            "source_revision": self.source_revision,
            "attempt_index": self.attempt_index,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunResultV2":
        try:
            _validate_result_keys(raw)
            if str(raw.get("schema_version", "")) != RESULT_SCHEMA_V2:
                raise ValueError(f"unsupported v2 result schema: {raw.get('schema_version')!r}")
            run_raw = raw.get("run")
            metrics_raw = raw.get("metrics")
            observations_raw = raw.get("observations")
            artifacts_raw = raw.get("artifacts")
            truncated_raw = raw.get("truncated")
            attempt_index_raw = raw.get("attempt_index")
            if not isinstance(run_raw, dict):
                raise ValueError("run must be an object")
            if not isinstance(metrics_raw, dict):
                raise ValueError("metrics must be an object")
            if not isinstance(observations_raw, list):
                raise ValueError("observations must be a list")
            if not isinstance(artifacts_raw, dict):
                raise ValueError("artifacts must be an object")
            if not isinstance(truncated_raw, bool):
                raise ValueError("truncated must be a boolean")
            if isinstance(attempt_index_raw, bool) or not isinstance(attempt_index_raw, int):
                raise ValueError("attempt_index must be an integer")
            for nullable_text in (
                "failure_mode",
                "stop_reason",
                "contract_id",
                "source_revision",
                "started_at",
                "finished_at",
            ):
                value = raw.get(nullable_text)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{nullable_text} must be a string or null")

            observations = _sanitize_observations(observations_raw)
            # Persisted records must already be sanitized.  Silently accepting
            # raw fields here could make resume trust a file that still contains
            # model text on disk.
            if list(observations) != observations_raw:
                raise ValueError("persisted observations are not fully sanitized")

            error_raw = raw.get("error")
            if error_raw is not None and (
                not isinstance(error_raw, str) or not _SAFE_ERROR_PATTERN.fullmatch(error_raw)
            ):
                raise ValueError("error is not a safe exception fingerprint")

            result = cls(
                run=RunSpecV2.from_dict(run_raw),
                status=ResultStatus(str(raw["status"])),
                metrics={str(key): value for key, value in metrics_raw.items()},
                completed=raw.get("completed"),
                failure_mode=(
                    str(raw["failure_mode"]) if raw.get("failure_mode") is not None else None
                ),
                stop_reason=(
                    str(raw["stop_reason"]) if raw.get("stop_reason") is not None else None
                ),
                truncated=truncated_raw,
                error=error_raw,
                observations=observations,
                artifacts=_normalize_artifacts(artifacts_raw),
                contract_id=(
                    str(raw["contract_id"]) if raw.get("contract_id") is not None else None
                ),
                source_revision=(
                    str(raw["source_revision"])
                    if raw.get("source_revision") is not None
                    else None
                ),
                attempt_index=attempt_index_raw,
                started_at=(str(raw["started_at"]) if raw.get("started_at") is not None else None),
                finished_at=(
                    str(raw["finished_at"]) if raw.get("finished_at") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultCorruptionError(f"invalid v2 experiment result: {exc}") from exc

        if str(raw.get("run_key", "")) != result.run_key:
            raise ResultCorruptionError("v2 result run_key does not match its run identity")
        return result


class ResultStoreV2:
    """Atomic one-result-per-run store with strict identity and contract checks."""

    def __init__(
        self,
        root: str | Path,
        *,
        contract_id: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        self.root = Path(root)
        if contract_id is not None:
            _validate_contract_id(contract_id)
        _validate_source_revision(source_revision)
        self.contract_id = contract_id
        self.source_revision = source_revision

    def run_dir(self, run: RunSpecV2) -> Path:
        return self.root / "runs" / str(run.run_key)

    def result_path(self, run: RunSpecV2) -> Path:
        return self.run_dir(run) / "result.json"

    def attempt_dir(self, run: RunSpecV2, attempt_index: int) -> Path:
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index <= 0
        ):
            raise ValueError("attempt_index must be a positive integer")
        return self.run_dir(run) / "attempts" / f"{attempt_index:04d}"

    def validate_run_layout(self, run: RunSpecV2) -> None:
        """Reject filesystem shapes that cannot safely hold one run.

        Formal paid gates call :meth:`read` before provider access.  Keeping
        this validation on the store also protects non-formal writers from
        following a result/attempt symlink or discovering a blocking file only
        after execution has already started.
        """

        directories = (
            (self.root / "runs", "v2 runs root"),
            (self.run_dir(run), "v2 run root"),
            (self.run_dir(run) / "attempts", "v2 attempts root"),
        )
        for path, label in directories:
            if path.is_symlink():
                raise ResultCorruptionError(f"{label} must not be a symlink: {path}")
            if path.exists() and not path.is_dir():
                raise ResultCorruptionError(f"{label} is not a directory: {path}")
        result_path = self.result_path(run)
        if result_path.is_symlink():
            raise ResultCorruptionError(
                f"v2 result file must not be a symlink: {result_path}"
            )
        if result_path.exists() and not result_path.is_file():
            raise ResultCorruptionError(f"v2 result path is not a file: {result_path}")

    def read(self, run: RunSpecV2) -> RunResultV2 | None:
        self.validate_run_layout(run)
        path = self.result_path(run)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ResultCorruptionError(f"cannot read v2 result {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ResultCorruptionError(f"v2 result root is not an object: {path}")
        result = RunResultV2.from_dict(raw)
        if result.run != run:
            raise ResultCorruptionError(f"v2 result identity does not match planned run: {path}")
        self._validate_contract(result)
        return result

    def write(self, result: RunResultV2) -> Path:
        normalized = replace(
            result,
            error=_safe_persisted_error(result.error),
            observations=_sanitize_observations(result.observations),
            artifacts=_normalize_artifacts(result.artifacts),
            source_revision=(result.source_revision or self.source_revision),
        )
        self._validate_contract(normalized)
        if (
            self.source_revision is not None
            and normalized.source_revision != self.source_revision
        ):
            raise ResultCorruptionError(
                "v2 result source_revision does not match the active execution revision"
            )
        self.validate_run_layout(normalized.run)
        path = self.result_path(normalized.run)
        existing = self.read(normalized.run)
        if existing is not None:
            if existing.status == ResultStatus.SUCCEEDED:
                if existing == normalized:
                    return path
                raise ResultCorruptionError(
                    f"a succeeded result cannot be overwritten: {normalized.run_key}"
                )
            if normalized.attempt_index != existing.attempt_index + 1:
                raise ResultCorruptionError(
                    f"replacement result does not advance one contiguous attempt: "
                    f"{normalized.run_key}"
                )
        _atomic_json_file(path, normalized.to_dict())
        return path

    def next_attempt_index(self, run: RunSpecV2) -> int:
        self.validate_run_layout(run)
        existing = self.read(run)
        last_result = existing.attempt_index if existing is not None else 0
        attempts_root = self.run_dir(run) / "attempts"
        occupied: set[int] = set()
        if attempts_root.is_dir():
            occupied = {
                int(path.name)
                for path in attempts_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            }
        return max(last_result, max(occupied, default=0)) + 1

    def begin_attempt(
        self,
        run: RunSpecV2,
        *,
        started_at: str,
    ) -> tuple[int, Path, str]:
        """Reserve a never-overwritten artifact directory for one execution."""

        existing = self.read(run)
        if existing is not None and existing.status == ResultStatus.SUCCEEDED:
            raise ResultCorruptionError(
                f"a succeeded run cannot begin another attempt: {run.run_key}"
            )
        attempt_index = self.next_attempt_index(run)
        while True:
            attempt_dir = self.attempt_dir(run, attempt_index)
            try:
                attempt_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                attempt_index += 1
                continue
            break
        prefix = attempt_dir.relative_to(self.run_dir(run)).as_posix()
        _atomic_json_file(
            attempt_dir / "attempt.state",
            {
                "schema_version": ATTEMPT_STATE_SCHEMA_V2,
                "run_key": run.run_key,
                "contract_id": self.contract_id,
                "attempt_index": attempt_index,
                "status": "started",
                "started_at": started_at,
                "finished_at": None,
                "error": None,
                "call_accounting": None,
                "source_revision": self.source_revision,
            },
        )
        return attempt_index, attempt_dir, prefix

    def finish_attempt(
        self,
        attempt_dir: Path,
        result: RunResultV2,
        *,
        ledger_call_accounting: Mapping[str, Any] | None = None,
    ) -> None:
        # Deliberately not ``*.json``: the v2 aggregate discovers result JSON
        # recursively, so internal attempt state must not masquerade as a second
        # terminal record for the same run key.
        marker = attempt_dir / "attempt.state"
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ResultCorruptionError(f"cannot read v2 attempt marker {marker}: {exc}") from exc
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != ATTEMPT_STATE_SCHEMA_V2
            or raw.get("run_key") != result.run_key
            or raw.get("contract_id") != result.contract_id
            or raw.get("attempt_index") != result.attempt_index
            or raw.get("source_revision") != result.source_revision
        ):
            raise ResultCorruptionError(
                f"v2 attempt marker identity does not match result: {marker}"
            )
        call_accounting = _safe_call_accounting(result)
        if ledger_call_accounting is not None:
            # Provider consumption is external accounting evidence.  It must
            # override any value inferred from an unverified payload.
            call_accounting.update(
                _validated_ledger_call_accounting(ledger_call_accounting)
            )
        immutable_result = attempt_dir / "attempt.result"
        _atomic_json_file_exclusive(immutable_result, result.to_dict())
        terminal = dict(raw)
        terminal.update(
            {
                "status": result.status.value,
                "finished_at": result.finished_at,
                "error": result.error,
                "call_accounting": call_accounting,
            }
        )
        _atomic_json_file(marker, terminal)

    def _validate_contract(self, result: RunResultV2) -> None:
        if self.contract_id is not None and result.contract_id != self.contract_id:
            raise ResultCorruptionError(
                "v2 result contract_id does not match the active experiment "
                f"contract: stored={result.contract_id!r}, "
                f"active={self.contract_id!r}"
            )


RunExecutorV2 = Callable[[RunSpecV2, Path], Mapping[str, Any]]


@dataclass(frozen=True)
class ExecutionSummaryV2:
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    stopped_early: bool = False
    failure_limit_reached: bool = False
    truncated: int = 0
    resource_limited: int = 0
    fatal_failure_reached: bool = False
    consecutive_stop_limit_reached: bool = False


class ExperimentRunnerV2:
    """Provider-neutral, resume-aware coordinator for a v2 experiment plan."""

    def __init__(self, store: ResultStoreV2) -> None:
        self.store = store

    def run(
        self,
        plan: ExperimentPlanV2,
        executor: RunExecutorV2,
        *,
        retry_failed: bool = False,
        max_failures: int | None = None,
    ) -> ExecutionSummaryV2:
        if max_failures is not None and (
            isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures <= 0
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
            result = _execute_one_v2(self.store, run, executor)

            if result.status == ResultStatus.SUCCEEDED:
                succeeded += 1
                if result.truncated:
                    truncated += 1
                if result.stop_reason == RESOURCE_LIMIT_STOP_REASON:
                    resource_limited += 1
                continue

            failed += 1
            if max_failures is not None and failed >= max_failures:
                failure_limit_reached = True
                stopped_early = run_index + 1 < len(plan.runs)
                break

        return ExecutionSummaryV2(
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            stopped_early=stopped_early,
            failure_limit_reached=failure_limit_reached,
            truncated=truncated,
            resource_limited=resource_limited,
        )

    def run_parallel(
        self,
        plan: ExperimentPlanV2,
        executor: RunExecutorV2,
        *,
        max_workers: int = 4,
        retry_failed: bool = False,
        max_failures: int | None = None,
        fatal_error_types: tuple[str, ...] = (),
        retry_error_codes: tuple[str, ...] = (),
        max_retries: int = 0,
        max_attempts_per_run: int | None = None,
        consecutive_stop_predicate: Callable[[RunResultV2], bool] | None = None,
        consecutive_stop_limit: int | None = None,
        initial_failure_count: int = 0,
        initial_consecutive_stop_count: int = 0,
    ) -> ExecutionSummaryV2:
        """Run isolated episodes concurrently without sharing a World.

        Formal batches are deliberately model-homogeneous.  At most four
        distinct run keys may execute at once; each receives its own attempt
        directory and the executor constructs a fresh Episode/World.  No run is
        submitted twice.  Once the infrastructure failure threshold is seen,
        already-running episodes finish but no new episode is submitted.
        """

        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or not 1 <= max_workers <= 4
        ):
            raise ValueError("max_workers must be an integer in [1, 4]")
        if max_failures is not None and (
            isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures <= 0
        ):
            raise ValueError("max_failures must be a positive integer or null")
        if len({run.run_key for run in plan.runs}) != len(plan.runs):
            raise ValueError("parallel plan contains duplicate run keys")
        models = {run.model_id for run in plan.runs}
        if len(models) > 1:
            raise ValueError("parallel formal batch must contain exactly one model")
        if any(
            not isinstance(name, str) or not _ERROR_TYPE_PATTERN.fullmatch(name)
            for name in fatal_error_types
        ) or len(fatal_error_types) != len(set(fatal_error_types)):
            raise ValueError("fatal_error_types must be unique safe exception names")
        if any(
            not isinstance(code, str) or not _ERROR_CODE_PATTERN.fullmatch(code)
            for code in retry_error_codes
        ) or len(retry_error_codes) != len(set(retry_error_codes)):
            raise ValueError("retry_error_codes must be unique safe error codes")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 2
        ):
            raise ValueError("max_retries must be an integer in [0, 2]")
        if bool(retry_error_codes) is not (max_retries > 0):
            raise ValueError("retry_error_codes and max_retries must be enabled together")
        if max_attempts_per_run is not None and (
            isinstance(max_attempts_per_run, bool)
            or not isinstance(max_attempts_per_run, int)
            or max_attempts_per_run <= 0
        ):
            raise ValueError("max_attempts_per_run must be a positive integer or null")
        if (consecutive_stop_predicate is None) is not (
            consecutive_stop_limit is None
        ):
            raise ValueError(
                "consecutive stop predicate and limit must be enabled together"
            )
        if consecutive_stop_limit is not None and (
            isinstance(consecutive_stop_limit, bool)
            or not isinstance(consecutive_stop_limit, int)
            or consecutive_stop_limit <= 0
        ):
            raise ValueError("consecutive_stop_limit must be a positive integer")
        if consecutive_stop_predicate is not None and max_workers != 1:
            raise ValueError(
                "consecutive result stopping requires one frozen-order worker"
            )
        if (
            isinstance(initial_failure_count, bool)
            or not isinstance(initial_failure_count, int)
            or initial_failure_count < 0
        ):
            raise ValueError("initial_failure_count must be a nonnegative integer")
        if initial_failure_count and max_failures is None:
            raise ValueError(
                "initial_failure_count requires an active failure limit"
            )
        if (
            isinstance(initial_consecutive_stop_count, bool)
            or not isinstance(initial_consecutive_stop_count, int)
            or initial_consecutive_stop_count < 0
        ):
            raise ValueError(
                "initial_consecutive_stop_count must be a nonnegative integer"
            )
        if initial_consecutive_stop_count and consecutive_stop_predicate is None:
            raise ValueError(
                "initial consecutive count requires consecutive stopping"
            )

        skipped = 0
        runnable: list[RunSpecV2] = []
        for run in plan.runs:
            existing = self.store.read(run)
            if existing is not None and (
                existing.status == ResultStatus.SUCCEEDED
                or (existing.status == ResultStatus.FAILED and not retry_failed)
            ):
                skipped += 1
            else:
                runnable.append(run)

        succeeded = failed = truncated = resource_limited = 0
        submitted = 0
        failure_limit_reached = bool(
            max_failures is not None
            and initial_failure_count >= max_failures
        )
        consecutive_stop_limit_reached = bool(
            consecutive_stop_limit is not None
            and initial_consecutive_stop_count >= consecutive_stop_limit
        )
        stopped_early = bool(runnable) and (
            failure_limit_reached or consecutive_stop_limit_reached
        )
        fatal_failure_reached = False
        consecutive_stop_count = initial_consecutive_stop_count
        iterator = iter(runnable)
        futures: dict[Future[RunResultV2], RunSpecV2] = {}

        def submit_next(pool: ThreadPoolExecutor) -> bool:
            nonlocal submitted
            try:
                run = next(iterator)
            except StopIteration:
                return False
            future = pool.submit(
                _execute_with_retries_v2,
                self.store,
                run,
                executor,
                retry_error_codes=frozenset(retry_error_codes),
                max_retries=max_retries,
                max_attempts_per_run=max_attempts_per_run,
            )
            futures[future] = run
            submitted += 1
            return True

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="commerceworld-episode",
        ) as pool:
            if not failure_limit_reached and not consecutive_stop_limit_reached:
                for _ in range(min(max_workers, len(runnable))):
                    submit_next(pool)
            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    result = future.result()
                    if result.status == ResultStatus.SUCCEEDED:
                        succeeded += 1
                        if result.truncated:
                            truncated += 1
                        if result.stop_reason == RESOURCE_LIMIT_STOP_REASON:
                            resource_limited += 1
                        if consecutive_stop_predicate is not None:
                            if consecutive_stop_predicate(result):
                                consecutive_stop_count += 1
                            else:
                                consecutive_stop_count = 0
                            if consecutive_stop_count >= int(
                                consecutive_stop_limit
                            ):
                                consecutive_stop_limit_reached = True
                    else:
                        failed += 1
                        consecutive_stop_count = 0
                        if result.error is not None and any(
                            result.error.startswith(f"{name}:")
                            for name in fatal_error_types
                        ):
                            fatal_failure_reached = True
                if fatal_failure_reached:
                    stopped_early = submitted < len(runnable)
                    continue
                if consecutive_stop_limit_reached:
                    stopped_early = submitted < len(runnable)
                    continue
                if (
                    max_failures is not None
                    and initial_failure_count + failed >= max_failures
                ):
                    failure_limit_reached = True
                    stopped_early = submitted < len(runnable)
                    continue
                for _ in range(len(completed)):
                    if not submit_next(pool):
                        break

        return ExecutionSummaryV2(
            attempted=submitted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            stopped_early=stopped_early,
            failure_limit_reached=failure_limit_reached,
            truncated=truncated,
            resource_limited=resource_limited,
            fatal_failure_reached=fatal_failure_reached,
            consecutive_stop_limit_reached=consecutive_stop_limit_reached,
        )


def _execute_with_retries_v2(
    store: ResultStoreV2,
    run: RunSpecV2,
    executor: RunExecutorV2,
    *,
    retry_error_codes: frozenset[str],
    max_retries: int,
    max_attempts_per_run: int | None,
) -> RunResultV2:
    """Retry only explicitly classified transient failures for one run key."""

    retries = 0
    while True:
        result = _execute_one_v2(store, run, executor)
        if result.status != ResultStatus.FAILED or retries >= max_retries:
            return result
        if (
            max_attempts_per_run is not None
            and result.attempt_index >= max_attempts_per_run
        ):
            return result
        codes = {
            str(row.get("error_code"))
            for row in result.observations
            if row.get("error_code") is not None
        }
        if not (codes & retry_error_codes):
            return result
        retries += 1


def _execute_one_v2(
    store: ResultStoreV2,
    run: RunSpecV2,
    executor: RunExecutorV2,
) -> RunResultV2:
    """Execute and atomically persist one isolated attempt."""

    started_at = _now()
    attempt_index, attempt_dir, artifact_prefix = store.begin_attempt(
        run,
        started_at=started_at,
    )
    ledger_call_accounting: object = None
    try:
        payload = executor(run, attempt_dir)
        if isinstance(payload, Mapping):
            ledger_call_accounting = payload.get(
                ATTEMPT_LEDGER_ACCOUNTING_FIELD_V2
            )
        result = _successful_result(
            run,
            payload,
            attempt_index=attempt_index,
            started_at=started_at,
            contract_id=store.contract_id,
            source_revision=store.source_revision,
            artifact_prefix=artifact_prefix,
        )
    except Exception as exc:  # noqa: BLE001 - persist for explicit resume
        exception_accounting = getattr(
            exc,
            ATTEMPT_LEDGER_ACCOUNTING_FIELD_V2,
            None,
        )
        if exception_accounting is not None:
            ledger_call_accounting = exception_accounting
        observations = getattr(exc, "observations", ())
        if not isinstance(observations, (list, tuple)):
            observations = ()
        safe_observations = _sanitize_observations(observations)
        failed_artifacts = {"attempt_dir": artifact_prefix}
        diagnostics = RunDiagnosticsV1.from_infrastructure_exception(exc)
        if diagnostics is not None:
            diagnostics_path = attempt_dir / "run_diagnostics.v1.json"
            _atomic_json_file(diagnostics_path, diagnostics.to_dict())
            failed_artifacts["run_diagnostics_v1"] = (
                f"{artifact_prefix}/run_diagnostics.v1.json"
            )
        result = RunResultV2(
            run=run,
            status=ResultStatus.FAILED,
            metrics=_failed_call_metrics(safe_observations),
            completed=None,
            failure_mode="infrastructure",
            error=_safe_exception_fingerprint(exc),
            observations=safe_observations,
            artifacts=failed_artifacts,
            contract_id=store.contract_id,
            source_revision=store.source_revision,
            attempt_index=attempt_index,
            started_at=started_at,
            finished_at=_now(),
        )

    if ledger_call_accounting is not None and not isinstance(
        ledger_call_accounting,
        Mapping,
    ):
        raise ResultCorruptionError(
            "executor ledger call accounting must be a mapping"
        )

    # Preserve the attempt-specific result before advancing the root's current
    # terminal pointer.  A crash between these steps is fail-closed by resume's
    # orphan/incomplete-attempt audit and never destroys an earlier attempt.
    store.finish_attempt(
        attempt_dir,
        result,
        ledger_call_accounting=ledger_call_accounting,
    )
    store.write(result)
    return result


def _successful_result(
    run: RunSpecV2,
    payload: Mapping[str, Any],
    *,
    attempt_index: int,
    started_at: str,
    contract_id: str | None,
    source_revision: str | None,
    artifact_prefix: str,
) -> RunResultV2:
    if not isinstance(payload, Mapping):
        raise ValueError("executor payload must be a mapping")
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

    metrics = {str(key): value for key, value in metrics_raw.items()}
    _validate_metrics(metrics)
    return RunResultV2(
        run=run,
        status=ResultStatus.SUCCEEDED,
        metrics=metrics,
        completed=completed,
        failure_mode=(
            str(payload["failure_mode"]) if payload.get("failure_mode") is not None else None
        ),
        stop_reason=(
            str(payload["stop_reason"]) if payload.get("stop_reason") is not None else None
        ),
        truncated=truncated,
        observations=_sanitize_observations(observations_raw),
        artifacts=_prefix_artifacts(artifacts_raw, artifact_prefix),
        contract_id=contract_id,
        source_revision=source_revision,
        attempt_index=attempt_index,
        started_at=started_at,
        finished_at=_now(),
    )


def _validate_metrics(metrics: Mapping[Any, Any]) -> None:
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metric names must be non-empty strings")
        if value is not None and not isinstance(value, (bool, int, float)):
            raise ValueError(f"metric {key!r} must be bool, int, float, or null")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric {key!r} must be finite")


def _validated_ledger_call_accounting(
    value: Mapping[str, Any],
) -> dict[str, int | str]:
    required = {
        "ledger_accounting_schema",
        "provider_call_count",
        "provider_response_count",
        "provider_error_count",
        "provider_unfinished_count",
        "reservation_set_sha256",
    }
    if set(value) != required:
        raise ResultCorruptionError(
            "ledger call accounting fields do not match the formal schema"
        )
    counts: dict[str, int] = {}
    numeric_names = required - {
        "ledger_accounting_schema",
        "reservation_set_sha256",
    }
    for name in numeric_names:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ResultCorruptionError(
                f"ledger call accounting field {name!r} is invalid"
            )
        counts[name] = item
    if counts["provider_call_count"] != (
        counts["provider_response_count"]
        + counts["provider_error_count"]
        + counts["provider_unfinished_count"]
    ):
        raise ResultCorruptionError(
            "ledger provider outcome counts do not partition provider calls"
        )
    schema = value.get("ledger_accounting_schema")
    if schema != ATTEMPT_LEDGER_ACCOUNTING_SCHEMA_V2:
        raise ResultCorruptionError(
            "ledger call accounting schema is invalid"
        )
    digest = value.get("reservation_set_sha256")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ResultCorruptionError(
            "ledger reservation-set digest is invalid"
        )
    return {
        "ledger_accounting_schema": schema,
        **counts,
        "reservation_set_sha256": digest,
    }


def _safe_call_accounting(result: RunResultV2) -> dict[str, int | float]:
    """Project one attempt to numeric-only accounting safe for long-term audit."""

    metric_count = result.metrics.get("model_call_count")
    model_call_count = (
        metric_count
        if isinstance(metric_count, int)
        and not isinstance(metric_count, bool)
        and metric_count >= 0
        else len(result.observations)
    )
    model_rows = [
        row for row in result.observations if row.get("source") == "model"
    ]
    provider_call_count = (
        sum(
            row.get("error_code")
            not in {RESOURCE_LIMIT_STOP_REASON, PRE_PROVIDER_CALL_ERROR_CODE}
            for row in model_rows
        )
        if model_rows
        else int(model_call_count)
    )
    error_count = 0
    latency_seconds = 0.0
    input_chars = 0
    output_chars = 0
    for row in result.observations:
        error_count += int(row.get("error_type") is not None)
        latency = row.get("latency_seconds")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            latency_seconds += float(latency)
        for name in ("system_chars", "user_chars"):
            value = row.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                input_chars += value
        output = row.get("output_chars")
        if isinstance(output, int) and not isinstance(output, bool) and output >= 0:
            output_chars += output
    return {
        "model_call_count": int(model_call_count),
        "provider_call_count": int(provider_call_count),
        "error_count": error_count,
        "latency_seconds": latency_seconds,
        "input_chars": input_chars,
        "output_chars": output_chars,
    }


_OBSERVATION_FIELDS = frozenset(
    {
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
        "decision_id",
        "error_type",
        "error_code",
    }
)
_NULLABLE_OBSERVATION_FIELDS = frozenset(
    {
        "model_id",
        "output_chars",
        "output_sha256",
        "decision_id",
        "error_type",
        "error_code",
    }
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,127}$")
_SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_ERROR_CODE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SAFE_OBSERVATION_ERROR_CODES = frozenset(
    {
        *SCOREABLE_STOP_REASONS,
        RESOURCE_LIMIT_STOP_REASON,
        "network",
        "provider_process_interrupted",
        "provider_timeout",
        "provider_429",
        "provider_5xx",
        "provider_4xx",
        "malformed_provider_response",
        PRE_PROVIDER_CALL_ERROR_CODE,
    }
)
_ARTIFACT_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ERROR_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*: message_chars=\d+ message_sha256=[0-9a-f]{64}$"
)
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


def sanitize_observations_v2(rows: object) -> tuple[dict[str, Any], ...]:
    """Return the hashes-and-lengths-only observation projection used on disk."""

    return _sanitize_observations(rows)


def _failed_call_metrics(
    observations: tuple[dict[str, Any], ...],
) -> dict[str, bool | float | int | None]:
    """Derive retry accounting only from the sanitized model observations."""

    model_rows = [row for row in observations if row.get("source") == "model"]
    provider_rows = [
        row
        for row in model_rows
        if row.get("error_code")
        not in {RESOURCE_LIMIT_STOP_REASON, PRE_PROVIDER_CALL_ERROR_CODE}
    ]
    metrics: dict[str, bool | float | int | None] = {
        "model_call_count": len(provider_rows),
        "model_error_count": sum(
            row.get("error_type") is not None for row in provider_rows
        ),
        "model_latency_seconds": sum(
            float(row.get("latency_seconds", 0.0)) for row in provider_rows
        ),
        "model_input_chars": sum(
            int(row.get("system_chars", 0)) + int(row.get("user_chars", 0))
            for row in provider_rows
        ),
        "model_output_chars": sum(
            int(row["output_chars"])
            for row in provider_rows
            if isinstance(row.get("output_chars"), int)
            and not isinstance(row.get("output_chars"), bool)
        ),
        "resource_limited": False,
    }
    if len(model_rows) != len(provider_rows):
        metrics["model_step_attempt_count"] = len(model_rows)
    return metrics


def _validated_observation_value(name: str, value: Any) -> Any:
    if value is None:
        return None if name in _NULLABLE_OBSERVATION_FIELDS else _DROP
    if name in {"actor_id", "model_id"}:
        return value if isinstance(value, str) and _IDENTIFIER_PATTERN.fullmatch(value) else _DROP
    if name == "decision_id":
        return (
            value
            if isinstance(value, str)
            and 0 < len(value) <= 512
            and all(ord(character) >= 32 for character in value)
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
        return value if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) else _DROP
    if name == "error_type":
        return value if isinstance(value, str) and _ERROR_TYPE_PATTERN.fullmatch(value) else _DROP
    if name == "error_code":
        return value if value in _SAFE_OBSERVATION_ERROR_CODES else _DROP
    return _DROP


def _safe_exception_fingerprint(exc: Exception) -> str:
    message = str(exc)
    error_type = type(exc).__name__
    if not _ERROR_TYPE_PATTERN.fullmatch(error_type):
        error_type = "Exception"
    return (
        f"{error_type}: message_chars={len(message)} "
        f"message_sha256={hashlib.sha256(message.encode('utf-8')).hexdigest()}"
    )


def _safe_persisted_error(error: str | None) -> str | None:
    if error is None or _SAFE_ERROR_PATTERN.fullmatch(error):
        return error
    return (
        f"PersistedError: message_chars={len(error)} "
        f"message_sha256={hashlib.sha256(error.encode('utf-8')).hexdigest()}"
    )


def _normalize_artifacts(artifacts: Mapping[Any, Any]) -> dict[str, str]:
    return _prefix_artifacts(artifacts, None)


def _prefix_artifacts(
    artifacts: Mapping[Any, Any],
    prefix: str | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in artifacts.items():
        if not isinstance(key, str) or not _ARTIFACT_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"artifact key {key!r} is invalid")
        if not isinstance(value, str):
            raise ValueError(f"artifact {key!r} path must be a string")
        if (
            not value
            or len(value) > 512
            or any(character in value for character in ("\\", "\x00", "\n", "\r"))
        ):
            raise ValueError(f"artifact {key!r} must be a safe relative path")
        path = Path(value)
        if path == Path(".") or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"artifact {key!r} must be a safe relative path")
        normalized[key] = (
            (Path(prefix) / path).as_posix() if prefix is not None else path.as_posix()
        )
    return normalized


_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "run_key",
        "run",
        "status",
        "metrics",
        "completed",
        "failure_mode",
        "stop_reason",
        "truncated",
        "error",
        "observations",
        "artifacts",
        "contract_id",
        "source_revision",
        "attempt_index",
        "started_at",
        "finished_at",
    }
)


def _validate_result_keys(raw: Mapping[str, Any]) -> None:
    keys = set(raw)
    legacy = {"seed", "rollout"} & keys
    if legacy:
        raise ValueError(f"v2 result cannot contain legacy axes: {sorted(legacy)!r}")
    # ``source_revision`` was added to the same v2 persistence generation so
    # generic historical rows remain readable.  Formal v2.1 stores require it
    # through ``ResultStoreV2(source_revision=...)``.
    missing = (_RESULT_KEYS - {"source_revision"}) - keys
    extra = keys - _RESULT_KEYS
    if missing or extra:
        raise ValueError(
            f"v2 result fields do not match schema: missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}"
        )


def _validate_contract_id(contract_id: str) -> None:
    if (
        not isinstance(contract_id, str)
        or not contract_id.strip()
        or len(contract_id) > 256
        or any(character in contract_id for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError("contract_id must be a safe non-empty string")


def _validate_source_revision(value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or not _GIT_REVISION_PATTERN.fullmatch(value)
    ):
        raise ValueError("source_revision must be a full lowercase Git SHA or null")


def _validate_timestamp(value: str | None, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string or null")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 string or null") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
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


def _atomic_json_file_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Install canonical JSON without replacing an immutable attempt result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
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
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ResultCorruptionError(
                f"immutable v2 attempt result already exists: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "ATTEMPT_LEDGER_ACCOUNTING_FIELD_V2",
    "ATTEMPT_LEDGER_ACCOUNTING_SCHEMA_V2",
    "ATTEMPT_STATE_SCHEMA_V2",
    "ExecutionSummaryV2",
    "ExperimentRunnerV2",
    "RESULT_SCHEMA_V2",
    "ResultCorruptionError",
    "ResultStatus",
    "ResultStoreV2",
    "RunExecutorV2",
    "RunResultV2",
    "sanitize_observations_v2",
]
