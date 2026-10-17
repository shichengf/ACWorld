"""The only executor for scored ACWorld benchmark runs.

Every run resolves one frozen runtime bundle and traverses
``Episode -> Runtime -> Platform -> World``.  The evaluated actor receives a
provider-neutral business-decision channel; every other actor is supplied by
the bundle's deterministic counterpart factory.  There is no interaction,
direct-simulation, or alternate score-version route here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal, Mapping, cast

from agents.inference import InferenceChannel
from episode.capability_materializer import (
    MaterializedTaskV2,
    scenario_content_hash_v2,
)
from episode.capability_runtime import (
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceBundleV2,
    RuntimeTaskBundleV2,
    RuntimeTaskScoreV3,
)
from episode.capability_runtime_registry import (
    RuntimeTaskNotImplementedError,
    runtime_bundle_v2,
    validate_runtime_bundle_v2,
)
from episode.capability_runtime_preflight import (
    score_with_authority_commit_closure_v2,
)
from episode.scenario import population_for_scenario
from episode.termination import (
    CAPABILITY_GUARD_STOP_REASON,
    EPISODE_TERMINATION_ARTIFACT,
    EPISODE_TERMINATION_SCHEMA,
    MODEL_PROTOCOL_STOP_REASON,
    RESOURCE_LIMIT_STOP_REASON,
    SCOREABLE_STOP_REASONS,
    SECURITY_GUARD_STOP_REASON,
    UNEXPECTED_RUNTIME_CLASSIFICATION,
)
from experiments.benchmark_plan import ExperimentPlanV2, RunSpecV2
from experiments.recording import (
    PRE_PROVIDER_CALL_ERROR_CODE,
    RecordingChannel,
    is_model_protocol_response_observation,
    recovered_same_decision_transport_error_indices,
)
from experiments.run_diagnostics import RunDiagnosticsV1

EXECUTION_MANIFEST_SCHEMA_V2 = "cwe.execution-manifest.v2"
MAX_MODEL_CALLS_PER_RUN_V2 = 48
BENCHMARK_EXECUTION_BACKEND = "commerceworld_episode"
BENCHMARK_RESULT_CLASSIFICATION = "commerceworld_formal_main"
BENCHMARK_RESULT_METRICS = frozenset(
    {
        "capability_benchmark",
        "benchmark_eligible",
        "commerceworld_episode",
        "population_buyers",
        "population_merchants",
        "model_controlled_actor_count",
        "reference_actor_count",
        "model_call_count",
        "model_call_limit",
        "model_input_chars",
        "model_output_chars",
        "model_latency_seconds",
        "model_step_attempt_count",
        "resource_limited",
        "capability_score",
        "strict_success",
        "model_safety_violation",
        "model_privacy_violation",
    }
)

_RUNTIME_TERMINATION_SEMANTICS_V2 = {
    MODEL_PROTOCOL_STOP_REASON: ("protocol", "protocol_error"),
    SECURITY_GUARD_STOP_REASON: ("security", "security_error"),
    RESOURCE_LIMIT_STOP_REASON: (RESOURCE_LIMIT_STOP_REASON, RESOURCE_LIMIT_STOP_REASON),
}

ModelChannelFactory = Callable[[str, str, str], InferenceChannel]
ControlSource = Literal["model", "deterministic_counterpart"]


class ExecutionConfigurationError(RuntimeError):
    """A benchmark run cannot execute without changing its frozen semantics."""


class InferenceExecutionError(RuntimeError):
    """Benchmark execution failed with completed observation hashes attached."""

    def __init__(
        self,
        message: str,
        *,
        observations: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.observations = observations


class SkillSelectionInfrastructureError(InferenceExecutionError):
    """A framework-owned selector gate failed before a valid score existed."""

    error_code = "skill_selector_failure"


class TrackerCaptureInfrastructureError(InferenceExecutionError):
    """Strict Tracker capture failed before a valid score existed."""

    error_code = "tracker_capture_failure"


class PreProviderCallAccountingError(InferenceExecutionError):
    """Call accounting failed before the provider could be invoked."""

    error_code = PRE_PROVIDER_CALL_ERROR_CODE


class ProviderRetryBudgetInfrastructureError(InferenceExecutionError):
    """Recovered transport failures contributed to the global call ceiling."""

    error_code = "provider_retry_budget_exhausted"


class ModelCallLimitExceeded(RuntimeError):
    """A benchmark run exhausted its Agent owned provider call budget."""

    error_code = RESOURCE_LIMIT_STOP_REASON

    def __init__(self, *, limit: int, used: int) -> None:
        self.limit = limit
        self.used = used
        super().__init__(f"shared per-run model-call limit {limit} exhausted")


@dataclass
class _ModelCallBudget:
    limit: int
    used: int = 0
    rejected_claims: int = 0
    lock: Any = field(default_factory=Lock, repr=False)

    def claim(self) -> None:
        with self.lock:
            if self.used >= self.limit:
                self.rejected_claims += 1
                raise ModelCallLimitExceeded(limit=self.limit, used=self.used)
            self.used += 1

    def snapshot(self) -> tuple[int, int, int]:
        with self.lock:
            return self.used, self.limit, self.rejected_claims


class _BudgetedModelChannel:
    """Reserve budget immediately before one typed provider request."""

    def __init__(self, inner: InferenceChannel, budget: _ModelCallBudget) -> None:
        self._inner = inner
        self._budget = budget

    @property
    def supports_decision_evidence_context(self) -> bool:
        return bool(getattr(self._inner, "supports_decision_evidence_context", False))

    @property
    def supports_business_decisions(self) -> bool:
        return bool(getattr(self._inner, "supports_business_decisions", False))

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> Any:
        if not self.supports_business_decisions:
            raise TypeError("wrapped inference channel does not support business decisions")
        complete = getattr(self._inner, "complete_business_decision", None)
        if not callable(complete):
            raise TypeError("wrapped inference channel has no business completion method")
        self._budget.claim()
        return complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            decision_id=decision_id,
        )


@dataclass(frozen=True)
class ActorControl:
    """One actor's controller assignment in a benchmark run."""

    actor_id: str
    role: str
    source: ControlSource

    def to_dict(self) -> dict[str, str]:
        return {
            "actor_id": self.actor_id,
            "role": self.role,
            "source": self.source,
        }


@dataclass(frozen=True)
class PreparedBenchmarkRun:
    """A frozen runtime bundle plus its validated controller topology."""

    run: RunSpecV2
    materialized: MaterializedTaskV2
    controls: tuple[ActorControl, ...]
    runtime_bundle: RuntimeTaskBundleV2

    @property
    def model_actor_ids(self) -> tuple[str, ...]:
        return tuple(row.actor_id for row in self.controls if row.source == "model")

    @property
    def reference_actor_ids(self) -> tuple[str, ...]:
        return tuple(row.actor_id for row in self.controls if row.source != "model")


@dataclass(frozen=True)
class BenchmarkReadinessIssue:
    run_key: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_key": self.run_key,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class BenchmarkExecutionReadiness:
    planned_runs: int
    issues: tuple[BenchmarkReadinessIssue, ...]

    @property
    def ready(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "planned_runs": self.planned_runs,
            "issue_count": len(self.issues),
            "issues": [row.to_dict() for row in self.issues],
        }


@dataclass(frozen=True)
class _EpisodeRunView:
    """Minimal v1 executor view without v1 seed/rollout/market serialization."""

    model_id: str
    evaluated_role: str
    run_key: str
    suite: str


def _observations(
    channels: Mapping[str, tuple[str, bool, RecordingChannel]],
    *,
    model_id: str,
) -> list[dict[str, Any]]:
    """Return the hashes-and-lengths provider ledger for one Episode."""

    rows: list[dict[str, Any]] = []
    for actor_id in sorted(channels):
        role, model_controlled, channel = channels[actor_id]
        for observation in channel.observations:
            rows.append(
                {
                    "actor_id": actor_id,
                    "role": role,
                    "source": "model" if model_controlled else "reference",
                    "model_id": model_id if model_controlled else None,
                    **observation.to_dict(),
                }
            )
    return rows


def _read_episode_termination(artifact_dir: Path) -> dict[str, Any] | None:
    """Load and validate the sanitized Runtime termination sidecar."""

    path = artifact_dir / EPISODE_TERMINATION_ARTIFACT
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceExecutionError(
            f"cannot read episode termination artifact: {type(exc).__name__}"
        ) from exc
    if not isinstance(raw, dict):
        raise InferenceExecutionError("episode termination artifact must be an object")
    exception = raw.get("exception")
    classification = raw.get("classification")
    expected_stop_reason = (
        classification if classification in SCOREABLE_STOP_REASONS else None
    )
    if (
        raw.get("schema_version") != EPISODE_TERMINATION_SCHEMA
        or raw.get("status") != "aborted"
        or raw.get("phase") != "runtime"
        or (
            classification not in SCOREABLE_STOP_REASONS
            and classification != UNEXPECTED_RUNTIME_CLASSIFICATION
        )
        or raw.get("stop_reason") != expected_stop_reason
        or raw.get("scoreable") is not (expected_stop_reason is not None)
        or not isinstance(exception, dict)
    ):
        raise InferenceExecutionError("invalid episode termination artifact metadata")
    error_type = exception.get("type")
    error_module = exception.get("module")
    error_code = exception.get("code")
    message_chars = exception.get("message_chars")
    message_sha256 = exception.get("message_sha256")
    if (
        not isinstance(error_type, str)
        or not error_type
        or not isinstance(error_module, str)
        or not error_module
        or not (error_code is None or isinstance(error_code, str))
        or isinstance(message_chars, bool)
        or not isinstance(message_chars, int)
        or message_chars < 0
        or not isinstance(message_sha256, str)
        or len(message_sha256) != 64
        or any(character not in "0123456789abcdef" for character in message_sha256)
    ):
        raise InferenceExecutionError("invalid episode termination exception metadata")
    return raw


def _is_exact_resource_limit_termination(
    termination: Mapping[str, Any] | None,
) -> bool:
    if termination is None:
        return False
    exception = termination.get("exception")
    return bool(
        isinstance(exception, Mapping)
        and termination.get("classification") == RESOURCE_LIMIT_STOP_REASON
        and termination.get("stop_reason") == RESOURCE_LIMIT_STOP_REASON
        and termination.get("scoreable") is True
        and exception.get("type") == "ModelCallLimitExceeded"
        and exception.get("module") == "experiments.benchmark_executor"
        and exception.get("code") == RESOURCE_LIMIT_STOP_REASON
    )


def _benchmark_artifact_paths(*, run_dir: Path, artifact_dir: Path) -> dict[str, str]:
    """Index only Episode evidence. No legacy progress artifact exists."""

    episode_artifacts = {
        "audit": "audit.jsonl",
        "trace": "audit.trace.jsonl",
        "security": "audit.security.jsonl",
        "world_initial": "world.initial.json",
        "world_final": "world.final.json",
        "world_commits": "world.commits.jsonl",
        "platform_decisions": "platform.decisions.jsonl",
        "platform_response_dispositions": "platform.response-dispositions.jsonl",
        "txn_diffs": "txn_diffs.jsonl",
        "replay": "replay.json",
        "episode_evidence": "episode.evidence.json",
        "extensions": "extensions.json",
        "termination": EPISODE_TERMINATION_ARTIFACT,
    }
    paths = {"episode_dir": artifact_dir.relative_to(run_dir).as_posix()}
    paths.update(
        {
            name: (artifact_dir / filename).relative_to(run_dir).as_posix()
            for name, filename in episode_artifacts.items()
            if (artifact_dir / filename).is_file()
        }
    )
    return paths


def _benchmark_call_metrics(
    observations: list[dict[str, Any]],
    *,
    model_call_limit: int,
    resource_limited: bool,
) -> dict[str, bool | float | int]:
    provider_calls = [
        row
        for row in observations
        if row.get("source") == "model"
        and row.get("error_code")
        not in {RESOURCE_LIMIT_STOP_REASON, PRE_PROVIDER_CALL_ERROR_CODE}
    ]
    return {
        "model_call_count": len(provider_calls),
        "model_call_limit": model_call_limit,
        "model_input_chars": sum(
            int(row["system_chars"]) + int(row["user_chars"])
            for row in provider_calls
        ),
        "model_output_chars": sum(
            int(row.get("output_chars") or 0) for row in provider_calls
        ),
        "model_latency_seconds": sum(
            float(row["latency_seconds"]) for row in provider_calls
        ),
        "model_step_attempt_count": sum(
            1 for row in observations if row.get("source") == "model"
        ),
        "resource_limited": resource_limited,
    }


def _run_benchmark_episode_payload(
    *,
    run: _EpisodeRunView,
    run_dir: Path,
    scenario: Any,
    channels: Callable[[str, str], RecordingChannel],
    channel_rows: dict[str, tuple[str, bool, RecordingChannel]],
    call_budget: _ModelCallBudget,
) -> dict[str, Any]:
    """Run the sole Episode path without invoking a legacy scorer."""

    from episode.runner import EpisodeBatch

    episode_root = run_dir / "episode"
    results = EpisodeBatch(
        scenarios=[scenario],
        channels=channels,
        out_root=episode_root,
        mode="E0",
        remote_world=True,
        strict_skill_selection=True,
        strict_tracker_capture=True,
    ).run()
    if len(results) != 1:
        raise InferenceExecutionError(
            f"expected one episode result for {run.run_key}, got {len(results)}"
        )
    artifact_dir = episode_root / scenario.scenario_id
    termination = _read_episode_termination(artifact_dir)
    observations = _observations(channel_rows, model_id=run.model_id)
    channel_errors = [row for row in observations if row.get("error_type")]
    model_observations = [
        row for row in observations if row.get("source") == "model"
    ]
    recovered_indices = recovered_same_decision_transport_error_indices(
        model_observations
    )
    recovered_errors = {
        id(model_observations[index]) for index in recovered_indices
    }
    infrastructure_channel_errors = [
        row
        for row in channel_errors
        if not is_model_protocol_response_observation(row)
        and id(row) not in recovered_errors
    ]
    budget_used, budget_limit, rejected_claims = call_budget.snapshot()
    resource_limited = bool(
        rejected_claims == 1
        and budget_used == budget_limit
        and not recovered_indices
        and len(infrastructure_channel_errors) == 1
        and infrastructure_channel_errors[0].get("source") == "model"
        and infrastructure_channel_errors[0].get("error_type")
        == "ModelCallLimitExceeded"
        and infrastructure_channel_errors[0].get("error_code")
        == RESOURCE_LIMIT_STOP_REASON
        and _is_exact_resource_limit_termination(termination)
    )
    if infrastructure_channel_errors and not resource_limited:
        first = infrastructure_channel_errors[0]
        if (
            first.get("source") == "model"
            and first.get("error_type") == "PreProviderCallAccountingError"
            and first.get("error_code") == PRE_PROVIDER_CALL_ERROR_CODE
        ):
            raise PreProviderCallAccountingError(observations=tuple(observations))
        if (
            first.get("source") == "model"
            and first.get("error_type") == "ModelCallLimitExceeded"
            and first.get("error_code") == RESOURCE_LIMIT_STOP_REASON
            and recovered_indices
        ):
            raise ProviderRetryBudgetInfrastructureError(
                "recovered provider failures contributed to call-limit exhaustion",
                observations=tuple(observations),
            )
        raise InferenceExecutionError(
            f"{first['source']} channel for {first['actor_id']} failed with "
            f"{first['error_type']}",
            observations=tuple(observations),
        )

    stop_reason: str | None = None
    failure_mode: str | None = None
    if termination is not None:
        classification = str(termination["classification"])
        if resource_limited:
            stop_reason = RESOURCE_LIMIT_STOP_REASON
        elif classification in SCOREABLE_STOP_REASONS:
            stop_reason = str(termination["stop_reason"])
        else:
            exception = termination["exception"]
            if (
                exception["module"] == "agents.skill_selector"
                and exception["type"] == "SkillSelectionError"
                and exception["code"] == "skill_selector_failure"
            ):
                error_type = SkillSelectionInfrastructureError
            elif (
                exception["module"] == "runtime.trace"
                and exception["type"] == "TrackerCaptureError"
                and exception["code"] == "tracker_capture_failure"
            ):
                error_type = TrackerCaptureInfrastructureError
            else:
                error_type = InferenceExecutionError
            raise error_type(
                "episode runtime aborted unexpectedly with "
                f"{exception['module']}.{exception['type']}",
                observations=tuple(observations),
            )
        failure_mode = _RUNTIME_TERMINATION_SEMANTICS_V2[stop_reason][0]
    elif rejected_claims:
        raise InferenceExecutionError(
            "model-call budget rejected a claim without Runtime termination evidence",
            observations=tuple(observations),
        )

    truncated = stop_reason is not None
    return {
        "metrics": _benchmark_call_metrics(
            observations,
            model_call_limit=budget_limit,
            resource_limited=resource_limited,
        ),
        "completed": not truncated,
        "failure_mode": failure_mode,
        "stop_reason": stop_reason,
        "truncated": truncated,
        "observations": observations,
        "artifacts": _benchmark_artifact_paths(
            run_dir=run_dir,
            artifact_dir=artifact_dir,
        ),
    }


def _materialized_runtime_bundle(
    run: RunSpecV2,
) -> tuple[MaterializedTaskV2, RuntimeTaskBundleV2]:
    """Resolve a main task from the fail-closed runtime registry."""

    if run.suite != "main":
        raise ExecutionConfigurationError("runtime task registry is main-only")
    try:
        bundle = runtime_bundle_v2(run.task_id)
        validate_runtime_bundle_v2(bundle)
    except RuntimeTaskNotImplementedError as exc:
        raise ExecutionConfigurationError(str(exc)) from exc
    except Exception as exc:
        raise ExecutionConfigurationError(
            f"invalid CommerceWorld runtime bundle for {run.task_id}: {exc}"
        ) from exc
    population = population_for_scenario(bundle.scenario)
    dimensions = (len(population.buyers), len(population.merchants))
    if dimensions != (run.buyers, run.merchants):
        raise ExecutionConfigurationError(
            f"runtime bundle population mismatch for {run.task_id}: "
            f"scenario={dimensions[0]}x{dimensions[1]}, run={run.buyers}x{run.merchants}"
        )
    if bundle.task.evaluated_role != run.evaluated_role:
        raise ExecutionConfigurationError(
            f"runtime bundle role mismatch for {run.task_id}: "
            f"task={bundle.task.evaluated_role}, run={run.evaluated_role}"
        )
    materialized = MaterializedTaskV2(
        task=bundle.task,
        scenario=bundle.scenario,
        source_scenario_id=run.task_id,
        evaluated_actor_id=bundle.evaluated_actor_id,
        determinism_key=bundle.semantic_hash,
        content_sha256=scenario_content_hash_v2(bundle.scenario),
    )
    return materialized, bundle


def _actor_controls(
    run: RunSpecV2,
    materialized: MaterializedTaskV2,
) -> tuple[ActorControl, ...]:
    """Bind exactly one evaluated model actor and deterministic counterparts."""

    if run.suite != "main":
        raise ExecutionConfigurationError("benchmark executor accepts only suite=main")

    population = population_for_scenario(materialized.scenario)
    rows: list[ActorControl] = []
    for role, actors in (
        ("buyer", tuple(item.buyer_id for item in population.buyers)),
        ("merchant", tuple(item.merchant_id for item in population.merchants)),
    ):
        for actor_id in actors:
            model_controlled = actor_id == materialized.evaluated_actor_id
            rows.append(
                ActorControl(
                    actor_id=actor_id,
                    role=role,
                    source=(
                        "model" if model_controlled else "deterministic_counterpart"
                    ),
                )
            )
    controls = tuple(sorted(rows, key=lambda row: (row.role, row.actor_id)))
    model_rows = tuple(row for row in controls if row.source == "model")
    if len(model_rows) != 1 or model_rows[0].actor_id != materialized.evaluated_actor_id:
        raise ExecutionConfigurationError(
            f"benchmark run {run.run_key} does not have exactly one evaluated actor"
        )
    return controls


class BenchmarkExecutor:
    """Execute benchmark runs through their frozen ACWorld bundle."""

    def __init__(
        self,
        *,
        model_channels: ModelChannelFactory | None = None,
        max_model_calls_per_run: int = MAX_MODEL_CALLS_PER_RUN_V2,
    ) -> None:
        if (
            isinstance(max_model_calls_per_run, bool)
            or not isinstance(max_model_calls_per_run, int)
            or max_model_calls_per_run <= 0
            or max_model_calls_per_run > MAX_MODEL_CALLS_PER_RUN_V2
        ):
            raise ExecutionConfigurationError(
                "max_model_calls_per_run must be an integer in [1, 48]"
            )
        self.model_channels = model_channels
        self.max_model_calls_per_run = max_model_calls_per_run

    def with_model_channels(
        self,
        model_channels: ModelChannelFactory,
        *,
        max_model_calls_per_run: int | None = None,
    ) -> "BenchmarkExecutor":
        """Clone the executor while wrapping only the model-call boundary."""

        return type(self)(
            model_channels=model_channels,
            max_model_calls_per_run=(
                self.max_model_calls_per_run
                if max_model_calls_per_run is None
                else max_model_calls_per_run
            ),
        )

    def readiness(self, plan: ExperimentPlanV2) -> BenchmarkExecutionReadiness:
        """Validate every selected run against the frozen runtime registry."""

        issues: list[BenchmarkReadinessIssue] = []
        for run in plan.runs:
            try:
                self.prepare(run)
            except ExecutionConfigurationError as exc:
                issues.append(
                    BenchmarkReadinessIssue(
                        run_key=run.run_key,
                        code=_readiness_issue_code(exc),
                        message=str(exc),
                    )
                )
        return BenchmarkExecutionReadiness(len(plan.runs), tuple(issues))

    def prepare(self, run: RunSpecV2) -> PreparedBenchmarkRun:
        """Prepare one run without constructing a model channel or making a call."""

        materialized, bundle = _materialized_runtime_bundle(run)
        controls = _actor_controls(run, materialized)
        counterpart_ids = {row.actor_id for row in controls if row.source != "model"}
        if counterpart_ids != set(bundle.counterpart_channels):
            raise ExecutionConfigurationError(
                f"runtime bundle counterpart mismatch for {run.task_id}"
            )
        return PreparedBenchmarkRun(run, materialized, controls, bundle)

    def __call__(self, run: RunSpecV2, run_dir: Path) -> dict[str, Any]:
        """Execute one prevalidated benchmark run and return its result payload."""

        if self.model_channels is None:
            raise ExecutionConfigurationError(
                "no business-decision channel factory was supplied"
            )
        model_channels = self.model_channels
        prepared = self.prepare(run)
        run_dir.mkdir(parents=True, exist_ok=True)
        control_index = {row.actor_id: row for row in prepared.controls}
        channel_rows: dict[str, tuple[str, bool, RecordingChannel]] = {}
        budget = _ModelCallBudget(self.max_model_calls_per_run)

        def channels(agent_id: str, role: str) -> RecordingChannel:
            if agent_id in channel_rows:
                raise ExecutionConfigurationError(
                    f"scenario requested channel for actor {agent_id!r} more than once"
                )
            try:
                control = control_index[agent_id]
            except KeyError as exc:
                raise ExecutionConfigurationError(
                    f"materialized actor {agent_id!r} is absent from controller topology"
                ) from exc
            if control.role != role:
                raise ExecutionConfigurationError(
                    f"actor {agent_id!r} role changed from {control.role!r} to {role!r}"
                )
            model_controlled = control.source == "model"
            if model_controlled:
                inner: InferenceChannel = model_channels(run.model_id, agent_id, role)
                inner = _BudgetedModelChannel(inner, budget)
            else:
                try:
                    counterpart_factory = prepared.runtime_bundle.counterpart_channels[agent_id]
                except KeyError as exc:
                    raise ExecutionConfigurationError(
                        f"runtime bundle has no deterministic counterpart {agent_id!r}"
                    ) from exc
                inner = counterpart_factory()
            recorded = RecordingChannel(inner)
            channel_rows[agent_id] = (role, model_controlled, recorded)
            return recorded

        episode_run = _EpisodeRunView(
            model_id=run.model_id,
            evaluated_role=run.evaluated_role,
            run_key=run.run_key,
            suite=run.suite,
        )
        try:
            payload = cast(
                dict[str, Any],
                _run_benchmark_episode_payload(
                    run=episode_run,
                    run_dir=run_dir,
                    scenario=prepared.materialized.scenario,
                    channels=channels,
                    channel_rows=channel_rows,
                    call_budget=budget,
                ),
            )
            self._write_benchmark_artifacts(prepared, run_dir, payload)
            return payload
        except (
            ExecutionConfigurationError,
            PreProviderCallAccountingError,
            ProviderRetryBudgetInfrastructureError,
            SkillSelectionInfrastructureError,
            TrackerCaptureInfrastructureError,
        ):
            # Keep the typed infrastructure classification intact. Benchmark
            # result handling persists these exception types and never turns
            # a missing channel capability or selector failure into a task
            # score.  Both guards run before a valid model action exists.
            raise
        except Exception as exc:
            # Episode finalization and deterministic evidence verification run
            # after provider calls.  If any post-call stage fails, preserve the
            # recording channels' hashes-and-lengths projection so the failed
            # result and call accounting cannot disagree about paid work.
            raise InferenceExecutionError(
                "CommerceWorld episode execution failed",
                observations=tuple(
                    _observations(channel_rows, model_id=run.model_id)
                ),
            ) from exc

    def _write_benchmark_artifacts(
        self,
        prepared: PreparedBenchmarkRun,
        run_dir: Path,
        payload: dict[str, Any],
    ) -> None:
        run = prepared.run
        scenario = prepared.materialized.scenario
        artifact_dir = run_dir / "episode" / scenario.scenario_id
        execution_manifest = {
            "schema_version": EXECUTION_MANIFEST_SCHEMA_V2,
            "run": run.to_dict(),
            "execution_backend": BENCHMARK_EXECUTION_BACKEND,
            "result_classification": BENCHMARK_RESULT_CLASSIFICATION,
            "task_content_sha256": scenario_content_hash_v2(scenario),
            "source_scenario_id": prepared.materialized.source_scenario_id,
            "scenario_id": scenario.scenario_id,
            "population": {"buyers": run.buyers, "merchants": run.merchants},
            "controller_semantics": "single_evaluated_actor",
            "controllers": [row.to_dict() for row in prepared.controls],
            "model_call_limit": self.max_model_calls_per_run,
            "prompt_response_storage": "hashes_and_lengths_only",
            "runtime_task_semantic_sha256": prepared.runtime_bundle.semantic_hash,
            "runtime_task_score_schema": "cwe.runtime-task-score.v3",
        }
        manifest_path = run_dir / "execution_manifest.v2.json"
        _write_json(manifest_path, execution_manifest)

        metrics = payload.get("metrics")
        artifacts = payload.get("artifacts")
        if not isinstance(metrics, dict) or not isinstance(artifacts, dict):
            raise InferenceExecutionError("executor payload lost metrics or artifacts")
        metrics.update(
            {
                "capability_benchmark": True,
                "benchmark_eligible": True,
                "commerceworld_episode": True,
                "population_buyers": run.buyers,
                "population_merchants": run.merchants,
                "model_controlled_actor_count": len(prepared.model_actor_ids),
                "reference_actor_count": len(prepared.reference_actor_ids),
            }
        )
        artifacts["execution_manifest_v2"] = manifest_path.relative_to(run_dir).as_posix()

        evidence = RuntimeEvidenceBundleV2.load(artifact_dir)
        # Match the fresh verifier's authority semantics before persisting a
        # score so an unclaimed World commit can never become a scored row.
        task_score, _authority_commit_closure = score_with_authority_commit_closure_v2(
            prepared.runtime_bundle,
            evidence,
        )
        if task_score.task_id != run.task_id:
            raise InferenceExecutionError(
                "runtime task scorer returned a different task identity"
            )
        score_payload, termination_semantics, diagnostics = runtime_task_score_payload(
            task_score,
            evidence,
            evaluated_actor_id=prepared.runtime_bundle.evaluated_actor_id,
        )
        payload_stop_reason = payload.get("stop_reason")
        if payload_stop_reason != termination_semantics["stop_reason"]:
            raise InferenceExecutionError(
                "executor stop reason differs from hash-covered Runtime termination"
            )
        if termination_semantics["stop_reason"] is not None:
            if payload.get("completed") is not False or payload.get("truncated") is not True:
                raise InferenceExecutionError(
                    "scoreable Runtime termination has contradictory completion state"
                )
            payload["failure_mode"] = termination_semantics["failure_mode"]
        if not diagnostics.valid_for_scoring:
            raise InferenceExecutionError(
                "infrastructure diagnostics forbid benchmark score materialization"
            )

        score_path = run_dir / "task_score.v3.json"
        _write_json(score_path, score_payload)
        artifacts["task_score_v3"] = score_path.relative_to(run_dir).as_posix()
        diagnostics_path = run_dir / "run_diagnostics.v1.json"
        _write_json(diagnostics_path, diagnostics.to_dict())
        artifacts["run_diagnostics_v1"] = diagnostics_path.relative_to(run_dir).as_posix()
        metrics.update({
            "capability_score": score_payload["capability_score"],
            "strict_success": score_payload["strict_success"],
            "model_safety_violation": score_payload["model_safety_violation"],
            "model_privacy_violation": score_payload["model_privacy_violation"],
            "resource_limited": termination_semantics["resource_limited"],
        })
        for name in tuple(metrics):
            if name not in BENCHMARK_RESULT_METRICS:
                del metrics[name]


def verified_runtime_termination_semantics(
    evidence: RuntimeEvidenceBundleV2,
    *,
    evaluated_actor_id: str,
) -> dict[str, Any]:
    """Read model-attributed termination from hash-covered Runtime evidence.

    A sanitized Runtime failure is scoreable only when its exact Tracker row
    belongs to the run's sole evaluated model actor.  The same terminal on a
    deterministic counterpart is an infrastructure failure, never model
    behavior.
    """

    if not isinstance(evaluated_actor_id, str) or not evaluated_actor_id.strip():
        raise InferenceExecutionError(
            "evaluated actor identity is required for Runtime termination attribution"
        )

    manifest = evidence.evidence_manifest
    artifacts = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    descriptor = artifacts.get("termination") if isinstance(artifacts, Mapping) else None
    path = evidence.episode_dir / "termination.json"
    empty = {
        "stop_reason": None,
        "failure_mode": None,
        "model_no_decision": False,
        "capability_guard": False,
        "security_guard": False,
        "resource_limited": False,
    }
    if descriptor is None:
        if path.exists():
            raise InferenceExecutionError(
                "Runtime termination exists outside the hash-covered manifest"
            )
        return empty
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise InferenceExecutionError("cannot read hash-covered Runtime termination") from exc
    expected_descriptor = {
        "path": "termination.json",
        "bytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    if not isinstance(descriptor, Mapping) or dict(descriptor) != expected_descriptor:
        raise InferenceExecutionError("Runtime termination differs from its evidence manifest")
    try:
        termination = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceExecutionError("hash-covered Runtime termination is invalid JSON") from exc
    classification = termination.get("classification") if isinstance(termination, Mapping) else None
    semantics = _RUNTIME_TERMINATION_SEMANTICS_V2.get(classification)
    binding = termination.get("tracker_binding") if isinstance(termination, Mapping) else None
    if (
        semantics is None
        or termination.get("schema_version") != EPISODE_TERMINATION_SCHEMA
        or termination.get("status") != "aborted"
        or termination.get("phase") != "runtime"
        or termination.get("stop_reason") != classification
        or termination.get("scoreable") is not True
        or not isinstance(binding, Mapping)
        or binding.get("artifact") != "audit.trace.jsonl"
        or binding.get("terminal") != semantics[1]
    ):
        raise InferenceExecutionError(
            "hash-covered Runtime termination has contradictory semantics"
        )
    if binding.get("agent_id") != evaluated_actor_id:
        raise InferenceExecutionError(
            "scoreable Runtime termination does not belong to the evaluated model actor"
        )
    failure_mode, _terminal = semantics
    return {
        "stop_reason": classification,
        "failure_mode": failure_mode,
        "model_no_decision": classification == MODEL_PROTOCOL_STOP_REASON,
        "capability_guard": classification == CAPABILITY_GUARD_STOP_REASON,
        "security_guard": classification == SECURITY_GUARD_STOP_REASON,
        "resource_limited": classification == RESOURCE_LIMIT_STOP_REASON,
    }


def runtime_task_score_payload(
    task_score: RuntimeTaskScoreV3,
    evidence: RuntimeEvidenceBundleV2,
    *,
    evaluated_actor_id: str,
) -> tuple[dict[str, Any], dict[str, Any], RunDiagnosticsV1]:
    """Produce capability-only score plus separate non-scoring diagnostics."""

    semantics = verified_runtime_termination_semantics(
        evidence,
        evaluated_actor_id=evaluated_actor_id,
    )
    if semantics["security_guard"] and not (
        task_score.model_safety_violation or task_score.model_privacy_violation
    ):
        # Runtime can prove that a private-utility or mandate guard fired, but
        # it cannot by itself prove who introduced the rejected value.  Only a
        # task scorer can join the guarded terminal to the model's typed
        # business choice.  Without that independent attribution, treating the
        # guard as a model outcome would turn an Agent/compiler/fixture defect
        # into a capability score.
        raise RuntimeBenchmarkIntegrityError(
            "security guard lacks task-scored model attribution"
        )
    if semantics["security_guard"] and not task_score.model_safety_violation:
        task_score = replace(
            task_score,
            model_safety_violation=True,
        )
    payload = task_score.to_dict()
    if semantics["stop_reason"] is not None:
        payload["strict_success"] = False
    model_no_decision = bool(semantics["model_no_decision"])
    diagnostics = RunDiagnosticsV1(
        decision_format_failures=int(model_no_decision),
        model_no_decision=model_no_decision,
    )
    public_semantics = {
        "stop_reason": semantics["stop_reason"],
        "failure_mode": semantics["failure_mode"],
        "model_no_decision": model_no_decision,
        "capability_guard": semantics["capability_guard"],
        "security_guard": semantics["security_guard"],
        "resource_limited": semantics["resource_limited"],
    }
    return payload, public_semantics, diagnostics


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _readiness_issue_code(exc: ExecutionConfigurationError) -> str:
    message = str(exc)
    if "runtime bundle" in message:
        return "runtime_bundle_not_ready"
    if "suite=main" in message:
        return "main_suite_only"
    return "invalid_configuration"


__all__ = [
    "ActorControl",
    "BENCHMARK_EXECUTION_BACKEND",
    "BENCHMARK_RESULT_CLASSIFICATION",
    "BENCHMARK_RESULT_METRICS",
    "EXECUTION_MANIFEST_SCHEMA_V2",
    "ExecutionConfigurationError",
    "BenchmarkExecutor",
    "BenchmarkExecutionReadiness",
    "BenchmarkReadinessIssue",
    "InferenceExecutionError",
    "MAX_MODEL_CALLS_PER_RUN_V2",
    "ModelCallLimitExceeded",
    "ModelChannelFactory",
    "PreProviderCallAccountingError",
    "PreparedBenchmarkRun",
    "ProviderRetryBudgetInfrastructureError",
    "SkillSelectionInfrastructureError",
    "TrackerCaptureInfrastructureError",
    "runtime_task_score_payload",
    "verified_runtime_termination_semantics",
]
