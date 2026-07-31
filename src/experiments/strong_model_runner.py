"""Resumable OpenRouter runner for the ACWorld benchmark.

It reuses the current 200 tasks, normal Agent path, ACWorld runtime, and scorer.
Resume is the ordinary ``ResultStoreV2`` behavior: completed task results are
skipped on the next invocation and missing tasks continue.
"""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from episode.capability_runtime_registry import runtime_bundle_v2
from experiments.benchmark_executor import BenchmarkExecutor
from experiments.openrouter_runtime import (
    FATAL_RUN_ERRORS,
    openrouter_channel_factory,
    require_business_decision_executor,
)
from experiments.benchmark_plan import (
    SUPPORTED_BENCHMARK_MODELS_V2,
    STRONG_MODELS_V2,
    ExperimentPlanV2,
    RunSpecV2,
    build_local_reference_smoke_benchmark_plan,
    build_strong_model_benchmark_plan,
    benchmark_canary_task_ids,
)
from experiments.results import ResultStatus
from experiments.benchmark_results import (
    ExecutionSummaryV2,
    ExperimentRunnerV2,
    ResultStoreV2,
    RunResultV2,
)


OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
DEFAULT_OUTPUT_ROOT = Path("output/benchmark")
DEFAULT_MULTI_ITEM_OUTPUT_ROOT = Path("output/multi-item-rerun")
DEFAULT_MAX_WORKERS = 2
LIVE_SMOKE_TASK_ID = "CWV2-T01-07"
MULTI_ITEM_TASK_IDS = tuple(f"CWV2-T05-{index:02d}" for index in range(1, 21))
MULTI_ITEM_RERUN_MODELS = (
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-luna",
)


class StrongModelRunError(RuntimeError):
    """A benchmark run cannot safely continue."""


@dataclass(frozen=True)
class StrongModelCandidate:
    model_id: str
    label: str
    note: str


STRONG_MODEL_CANDIDATES: tuple[StrongModelCandidate, ...] = (
    StrongModelCandidate(
        model_id="qwen/qwen3.5-plus-20260420",
        label="Qwen3.5 Plus",
        note="Qwen model in the paper panel.",
    ),
    StrongModelCandidate(
        model_id="deepseek/deepseek-v4-pro",
        label="DeepSeek-V4-Pro",
        note="DeepSeek model in the paper panel.",
    ),
    StrongModelCandidate(
        model_id="mistralai/mistral-medium-3-5",
        label="Mistral Medium 3.5",
        note="Mistral model in the paper panel.",
    ),
    StrongModelCandidate(
        model_id="google/gemini-3.5-flash",
        label="Gemini 3.5 Flash",
        note="Google model in the paper panel.",
    ),
    StrongModelCandidate(
        model_id="anthropic/claude-sonnet-5",
        label="Claude Sonnet 5",
        note="Anthropic model in the paper panel.",
    ),
    StrongModelCandidate(
        model_id="openai/gpt-5.6-terra",
        label="GPT-5.6 Terra",
        note="OpenAI model in the paper panel.",
    ),
    StrongModelCandidate(
        model_id="openai/gpt-5.6-sol",
        label="GPT-5.6 Sol",
        note="OpenAI reasoning model.",
    ),
    StrongModelCandidate(
        model_id="openai/gpt-5.6-luna",
        label="GPT-5.6 Luna",
        note="OpenAI general-purpose model.",
    ),
    StrongModelCandidate(
        model_id="moonshotai/kimi-k3",
        label="Kimi K3",
        note="Moonshot AI model.",
    ),
    StrongModelCandidate(
        model_id="google/gemini-3.6-flash",
        label="Gemini 3.6 Flash",
        note="Google fast model.",
    ),
    StrongModelCandidate(
        model_id="anthropic/claude-opus-4.8",
        label="Claude Opus 4.8",
        note="Optional endpoint; not part of the paper panel.",
    ),
)

if (
    tuple(candidate.model_id for candidate in STRONG_MODEL_CANDIDATES)
    != SUPPORTED_BENCHMARK_MODELS_V2
):
    raise RuntimeError("model descriptions differ from the executable allowlist")


def safe_model_slug(model_id: str) -> str:
    """Convert one validated OpenRouter ID to a readable directory name."""

    if model_id not in SUPPORTED_BENCHMARK_MODELS_V2:
        raise StrongModelRunError(f"unsupported benchmark model: {model_id!r}")
    return model_id.replace("/", "__")


def check_openrouter_catalog(
    model_ids: tuple[str, ...],
    *,
    endpoint: str = OPENROUTER_MODELS_ENDPOINT,
    timeout: float = 30.0,
) -> dict[str, bool]:
    """Check exact model IDs through OpenRouter's public, non-inference catalog."""

    if not model_ids or any(
        model_id not in SUPPORTED_BENCHMARK_MODELS_V2 for model_id in model_ids
    ):
        raise StrongModelRunError("catalog check received an unsupported model selection")
    request = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": "ACWorld-Strong-Model-Runner/1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError) as exc:
        raise StrongModelRunError(
            f"cannot read the OpenRouter model catalog: {type(exc).__name__}"
        ) from exc
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise StrongModelRunError("OpenRouter model catalog has no data list")
    available = {
        str(row["id"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    result = {model_id: model_id in available for model_id in model_ids}
    missing = tuple(model_id for model_id, present in result.items() if not present)
    if missing:
        raise StrongModelRunError(f"OpenRouter catalog is missing exact model IDs: {missing!r}")
    return result


def read_api_key(path: str | Path) -> str:
    """Read one key without logging or copying it into the result directory."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise StrongModelRunError(f"API key file does not exist: {source}")
    key = source.read_text(encoding="utf-8").strip()
    if not key:
        raise StrongModelRunError("API key file is empty")
    if any(character.isspace() for character in key):
        raise StrongModelRunError("API key file must contain exactly one token")
    return key


def select_plan_tasks(
    plan: ExperimentPlanV2,
    task_ids: tuple[str, ...],
) -> ExperimentPlanV2:
    """Return a plan subset in the caller's fixed task order."""

    by_task = {run.task_id: run for run in plan.runs}
    missing = tuple(task_id for task_id in task_ids if task_id not in by_task)
    if missing:
        raise StrongModelRunError(f"plan is missing requested tasks: {missing!r}")
    return ExperimentPlanV2(
        runs=tuple(by_task[task_id] for task_id in task_ids),
        model_panel_id=plan.model_panel_id,
    )


def execute_resumable_plan(
    plan: ExperimentPlanV2,
    *,
    result_root: str | Path,
    executor: Callable[[RunSpecV2, Path], Mapping[str, Any]],
    max_workers: int,
    retry_failed: bool = False,
    max_failures: int | None = None,
) -> ExecutionSummaryV2:
    """Run missing rows and skip successful rows already present on disk."""

    if not 1 <= max_workers <= DEFAULT_MAX_WORKERS:
        raise StrongModelRunError("strong-model concurrency must be one or two")
    store = ResultStoreV2(result_root)
    return ExperimentRunnerV2(store).run_parallel(
        plan,
        executor,
        max_workers=max_workers,
        retry_failed=retry_failed,
        max_failures=max_failures,
        fatal_error_types=FATAL_RUN_ERRORS,
    )


def run_reference_smoke(
    output_root: str | Path,
) -> tuple[ExecutionSummaryV2, dict[str, Any]]:
    """Run one Buyer and one Merchant task with deterministic local channels."""

    plan = build_local_reference_smoke_benchmark_plan()

    def execute(run: RunSpecV2, attempt_dir: Path) -> Mapping[str, Any]:
        bundle = runtime_bundle_v2(run.task_id)
        executor = BenchmarkExecutor(
            model_channels=lambda _model, _actor, _role: bundle.ideal_channel()
        )
        return executor(run, attempt_dir)

    root = Path(output_root)
    summary = execute_resumable_plan(
        plan,
        result_root=root,
        executor=execute,
        max_workers=1,
        retry_failed=True,
        max_failures=1,
    )
    report = write_result_summary(plan, root)
    if report["succeeded"] != 2 or report["full"] != 2:
        raise StrongModelRunError("local Buyer/Merchant reference smoke did not fully pass")
    return summary, report


def run_openrouter_model(
    model_id: str,
    *,
    api_key: str,
    output_root: str | Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Run or resume one exact model over all 200 tasks."""

    if model_id not in SUPPORTED_BENCHMARK_MODELS_V2:
        raise StrongModelRunError(f"unsupported benchmark model: {model_id!r}")
    if not api_key:
        raise StrongModelRunError("OpenRouter API key is required")
    if not 1 <= max_workers <= DEFAULT_MAX_WORKERS:
        raise StrongModelRunError("strong-model concurrency must be one or two")

    plan = build_strong_model_benchmark_plan((model_id,))
    model_root = Path(output_root) / safe_model_slug(model_id)
    store = ResultStoreV2(model_root)

    factory = openrouter_channel_factory(
        api_key=api_key,
        allowed_model_ids=SUPPORTED_BENCHMARK_MODELS_V2,
    )
    base_executor = BenchmarkExecutor(model_channels=factory)
    readiness = base_executor.readiness(plan)
    if not readiness.ready:
        raise StrongModelRunError(
            f"benchmark readiness failed for {len(readiness.issues)} task(s)"
        )
    executor = require_business_decision_executor(base_executor)

    canary_ids = benchmark_canary_task_ids()
    canary = select_plan_tasks(plan, canary_ids)
    canary_fatal = _run_phase_with_one_retry(
        canary,
        result_root=model_root,
        executor=executor,
        max_workers=1,
    )
    report = write_result_summary(plan, model_root)
    if canary_fatal or _has_provider_4xx(canary, store):
        raise StrongModelRunError(
            _incomplete_message("20-task prefix", canary, store)
        )

    canary_set = frozenset(canary_ids)
    remainder = ExperimentPlanV2(
        runs=tuple(run for run in plan.runs if run.task_id not in canary_set),
        model_panel_id=plan.model_panel_id,
    )
    _run_phase_with_one_retry(
        remainder,
        result_root=model_root,
        executor=executor,
        max_workers=max_workers,
    )
    report = write_result_summary(plan, model_root)
    final_state = _plan_state(plan, store)
    if final_state["failed"] or final_state["missing"]:
        raise StrongModelRunError(
            _incomplete_message("model batch", plan, store)
        )
    return report


def run_openrouter_smoke(
    model_id: str,
    *,
    api_key: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run one real task through the same path used by the full benchmark."""

    if model_id not in SUPPORTED_BENCHMARK_MODELS_V2:
        raise StrongModelRunError(f"unsupported benchmark model: {model_id!r}")
    if not api_key:
        raise StrongModelRunError("OpenRouter API key is required")

    full_plan = build_strong_model_benchmark_plan((model_id,))
    plan = select_plan_tasks(full_plan, (LIVE_SMOKE_TASK_ID,))
    model_root = Path(output_root) / safe_model_slug(model_id)
    factory = openrouter_channel_factory(
        api_key=api_key,
        allowed_model_ids=SUPPORTED_BENCHMARK_MODELS_V2,
    )
    base_executor = BenchmarkExecutor(model_channels=factory)
    readiness = base_executor.readiness(plan)
    if not readiness.ready:
        raise StrongModelRunError("live smoke task is not ready")
    executor = require_business_decision_executor(base_executor)
    fatal = _run_phase_with_one_retry(
        plan,
        result_root=model_root,
        executor=executor,
        max_workers=1,
    )
    store = ResultStoreV2(model_root)
    state = _plan_state(plan, store)
    if fatal or state["failed"] or state["missing"]:
        raise StrongModelRunError(
            _incomplete_message("live smoke", plan, store)
        )
    return write_result_summary(plan, model_root)


def run_openrouter_multi_item(
    model_id: str,
    *,
    api_key: str,
    output_root: str | Path = DEFAULT_MULTI_ITEM_OUTPUT_ROOT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    one_task_smoke: bool = False,
) -> tuple[dict[str, Any], str]:
    """Run only the corrected T5 family and emit a copy-paste result line."""

    if model_id not in MULTI_ITEM_RERUN_MODELS:
        raise StrongModelRunError(
            "the corrected T5-only rerun is limited to Sol and Luna; "
            "Opus 4.8 requires its existing full 200-task run"
        )
    if not api_key:
        raise StrongModelRunError("OpenRouter API key is required")
    if not 1 <= max_workers <= DEFAULT_MAX_WORKERS:
        raise StrongModelRunError("strong-model concurrency must be one or two")

    full_plan = build_strong_model_benchmark_plan((model_id,))
    task_ids = MULTI_ITEM_TASK_IDS[:1] if one_task_smoke else MULTI_ITEM_TASK_IDS
    plan = select_plan_tasks(full_plan, task_ids)
    model_root = Path(output_root) / safe_model_slug(model_id)

    factory = openrouter_channel_factory(
        api_key=api_key,
        allowed_model_ids=STRONG_MODELS_V2,
    )
    base_executor = BenchmarkExecutor(model_channels=factory)
    readiness = base_executor.readiness(plan)
    if not readiness.ready:
        raise StrongModelRunError(
            f"benchmark readiness failed for {len(readiness.issues)} task(s)"
        )
    executor = require_business_decision_executor(base_executor)
    fatal = _run_phase_with_one_retry(
        plan,
        result_root=model_root,
        executor=executor,
        max_workers=max_workers,
    )
    store = ResultStoreV2(model_root)
    state = _plan_state(plan, store)
    if fatal or state["failed"] or state["missing"]:
        raise StrongModelRunError(
            _incomplete_message("Multi-item rerun", plan, store)
        )
    return write_multi_item_report(plan, model_root)


def _run_phase_with_one_retry(
    plan: ExperimentPlanV2,
    *,
    result_root: str | Path,
    executor: Callable[[RunSpecV2, Path], Mapping[str, Any]],
    max_workers: int,
) -> bool:
    """Finish a phase, then retry its infrastructure failures once.

    Successful tasks are never repeated.  A task-level failure does not block
    unrelated tasks, while a fatal validation failure still stops submission.
    """

    first = execute_resumable_plan(
        plan,
        result_root=result_root,
        executor=executor,
        max_workers=max_workers,
        retry_failed=True,
        max_failures=None,
    )
    if first.fatal_failure_reached:
        return True
    store = ResultStoreV2(result_root)
    if _plan_state(plan, store)["failed"]:
        second = execute_resumable_plan(
            plan,
            result_root=result_root,
            executor=executor,
            max_workers=max_workers,
            retry_failed=True,
            max_failures=None,
        )
        return second.fatal_failure_reached
    return False


def _has_provider_4xx(
    plan: ExperimentPlanV2,
    store: ResultStoreV2,
) -> bool:
    """Detect a bad credential/model request before promoting past the prefix."""

    return any(
        result is not None
        and result.status == ResultStatus.FAILED
        and any(
            observation.get("error_code") == "provider_4xx"
            for observation in result.observations
        )
        for run in plan.runs
        if (result := store.read(run)) is not None
    )


def _incomplete_message(
    label: str,
    plan: ExperimentPlanV2,
    store: ResultStoreV2,
) -> str:
    failed = tuple(
        run.task_id
        for run in plan.runs
        if (result := store.read(run)) is not None
        and result.status == ResultStatus.FAILED
    )
    missing = tuple(run.task_id for run in plan.runs if store.read(run) is None)
    return (
        f"{label} is incomplete after one automatic retry; "
        f"failed={list(failed)!r}, missing={list(missing)!r}. "
        "Completed tasks remain saved; rerun the same command to try only "
        "failed or missing tasks."
    )


def write_result_summary(
    plan: ExperimentPlanV2,
    result_root: str | Path,
) -> dict[str, Any]:
    """Write one plain CSV and one compact JSON summary for a plan."""

    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    store = ResultStoreV2(root)
    rows: list[dict[str, Any]] = []
    for run in plan.runs:
        result = store.read(run)
        rows.append(_result_row(run, result))

    succeeded = sum(row["status"] == ResultStatus.SUCCEEDED.value for row in rows)
    failed = sum(row["status"] == ResultStatus.FAILED.value for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    scores = [
        float(row["capability_score"])
        for row in rows
        if isinstance(row["capability_score"], (int, float))
        and not isinstance(row["capability_score"], bool)
    ]
    report = {
        "model_id": plan.runs[0].model_id if plan.runs else None,
        "planned": len(rows),
        "succeeded": succeeded,
        "failed": failed,
        "missing": missing,
        "full": sum(row["outcome"] == "full" for row in rows),
        "partial": sum(row["outcome"] == "partial" for row in rows),
        "zero": sum(row["outcome"] == "zero" for row in rows),
        "protocol_errors": sum(bool(row["protocol_error"]) for row in rows),
        "model_calls": sum(
            int(row["model_call_count"])
            for row in rows
            if isinstance(row["model_call_count"], int)
            and not isinstance(row["model_call_count"], bool)
        ),
        "mean_capability_score": (
            round(sum(scores) / len(scores), 8) if scores else None
        ),
    }
    _write_json(root / "summary.json", report)
    _write_csv(root / "results.csv", rows)
    return report


def write_multi_item_report(
    plan: ExperimentPlanV2,
    result_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Write the compact statistics needed to transfer one T5 rerun."""

    if not plan.runs or any(run.task_family != "T5" for run in plan.runs):
        raise StrongModelRunError("Multi-item report requires only T5 tasks")
    root = Path(result_root)
    summary = write_result_summary(plan, root)
    store = ResultStoreV2(root)
    role_scores: dict[str, list[float]] = {"buyer": [], "merchant": []}
    task_scores: dict[str, float] = {}
    for run in plan.runs:
        result = store.read(run)
        if result is None:
            continue
        value = result.metrics.get("capability_score")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        score = float(value)
        role_scores[run.evaluated_role].append(score)
        task_scores[run.task_id] = score

    def mean_pct(values: list[float]) -> float | None:
        return round(100.0 * sum(values) / len(values), 4) if values else None

    payload = {
        "schema": "ACWORLD_T5_RERUN_V1",
        "model_id": summary["model_id"],
        "completed": summary["succeeded"],
        "failed": summary["failed"],
        "missing": summary["missing"],
        "mean_pct": (
            round(100.0 * float(summary["mean_capability_score"]), 4)
            if summary["mean_capability_score"] is not None
            else None
        ),
        "buyer_mean_pct": mean_pct(role_scores["buyer"]),
        "merchant_mean_pct": mean_pct(role_scores["merchant"]),
        "full_partial_zero": [
            summary["full"],
            summary["partial"],
            summary["zero"],
        ],
        "protocol_errors": summary["protocol_errors"],
        "model_calls": summary["model_calls"],
        "task_scores": {
            task_id: task_scores[task_id]
            for task_id in sorted(task_scores)
        },
    }
    line = "ACWORLD_T5_RERUN_V1 " + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    _write_json(root / "multi-item-report.json", payload)
    (root / "multi-item-report.txt").write_text(line + "\n", encoding="utf-8")
    return payload, line


def _plan_state(
    plan: ExperimentPlanV2,
    store: ResultStoreV2,
) -> dict[str, int]:
    state = {"succeeded": 0, "failed": 0, "missing": 0}
    for run in plan.runs:
        result = store.read(run)
        key = "missing" if result is None else result.status.value
        state[key] += 1
    return state


def _result_row(
    run: RunSpecV2,
    result: RunResultV2 | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "model_id": run.model_id,
            "task_id": run.task_id,
            "task_family": run.task_family,
            "evaluated_role": run.evaluated_role,
            "status": "missing",
            "capability_score": None,
            "outcome": None,
            "strict_success": None,
            "protocol_error": None,
            "model_call_count": None,
            "stop_reason": None,
            "failure_mode": None,
            "error": None,
        }
    score = result.metrics.get("capability_score")
    numeric_score = (
        float(score)
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else None
    )
    if numeric_score is None:
        outcome = None
    elif numeric_score == 1.0:
        outcome = "full"
    elif numeric_score == 0.0:
        outcome = "zero"
    else:
        outcome = "partial"
    return {
        "model_id": run.model_id,
        "task_id": run.task_id,
        "task_family": run.task_family,
        "evaluated_role": run.evaluated_role,
        "status": result.status.value,
        "capability_score": numeric_score,
        "outcome": outcome,
        "strict_success": result.metrics.get("strict_success"),
        "protocol_error": result.failure_mode == "protocol",
        "model_call_count": result.metrics.get("model_call_count"),
        "stop_reason": result.stop_reason,
        "failure_mode": result.failure_mode,
        "error": result.error,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        tuple(rows[0])
        if rows
        else (
            "model_id",
            "task_id",
            "task_family",
            "evaluated_role",
            "status",
            "capability_score",
            "outcome",
            "strict_success",
            "protocol_error",
            "model_call_count",
            "stop_reason",
            "failure_mode",
            "error",
        )
    )
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MULTI_ITEM_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "MULTI_ITEM_RERUN_MODELS",
    "MULTI_ITEM_TASK_IDS",
    "OPENROUTER_MODELS_ENDPOINT",
    "STRONG_MODEL_CANDIDATES",
    "StrongModelCandidate",
    "StrongModelRunError",
    "check_openrouter_catalog",
    "execute_resumable_plan",
    "LIVE_SMOKE_TASK_ID",
    "read_api_key",
    "run_openrouter_model",
    "run_openrouter_smoke",
    "run_openrouter_multi_item",
    "run_reference_smoke",
    "safe_model_slug",
    "select_plan_tasks",
    "write_multi_item_report",
    "write_result_summary",
]
