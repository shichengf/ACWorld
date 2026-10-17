"""Reference and OpenRouter runners for the large-catalog suite."""

from __future__ import annotations

import csv
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from agents.inference import ChannelTransportError, OpenAIChannel
from experiments.benchmark_plan import PAPER_MODELS_V2
from large_catalog.database import CatalogDatabase
from large_catalog.models import LargeCatalogTask, TaskResult
from large_catalog.reference import ReferencePolicy
from large_catalog.runtime import LargeCatalogRuntimeError, ModelPolicy, run_episode
from large_catalog.scoring import score_run


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LARGE_CATALOG_MODELS = PAPER_MODELS_V2
CANARY_TASK_IDS = ("LC-A1-01", "LC-B1-01", "LC-D1-01")


class LargeCatalogRunError(RuntimeError):
    """A batch cannot continue safely."""


class CostTrackingOpenAIChannel(OpenAIChannel):
    """OpenAI-compatible channel that retains only aggregate reported cost."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reported_cost_usd = 0.0

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = super()._post_chat_completions(payload)
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            raw = usage.get("cost")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                self.reported_cost_usd += float(raw)
        return response


def run_reference_tasks(
    *,
    database_path: str | Path,
    tasks: Sequence[LargeCatalogTask],
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(output_root) / "reference"
    root.mkdir(parents=True, exist_ok=True)
    results: list[TaskResult] = []
    with CatalogDatabase(database_path) as database:
        for task in tasks:
            result_path = root / f"{task.task_id}.json"
            if result_path.is_file():
                results.append(_load_result(result_path))
                continue
            run = run_episode(task=task, database=database, policy=ReferencePolicy())
            result = score_run(task, run, model_id="reference")
            _write_json(result_path, result.to_dict())
            results.append(result)
    summary = summarize_results(results, tasks=tasks)
    _write_json(root / "summary.json", summary)
    return summary


def run_model_tasks(
    *,
    model_id: str,
    api_key: str,
    database_path: str | Path,
    tasks: Sequence[LargeCatalogTask],
    output_root: str | Path,
    max_workers: int = 2,
    max_cost_usd: float = 10.0,
    canary_only: bool = False,
) -> dict[str, Any]:
    if model_id not in LARGE_CATALOG_MODELS:
        raise LargeCatalogRunError(f"unsupported large-catalog model: {model_id}")
    if not api_key:
        raise LargeCatalogRunError("OpenRouter API key is empty")
    if max_workers not in {1, 2}:
        raise LargeCatalogRunError("max_workers must be one or two")
    if max_cost_usd <= 0:
        raise LargeCatalogRunError("max_cost_usd must be positive")
    by_id = {task.task_id: task for task in tasks}
    canaries = [by_id[task_id] for task_id in CANARY_TASK_IDS]
    selected = canaries if canary_only else [*canaries, *[t for t in tasks if t not in canaries]]
    root = Path(output_root) / _model_slug(model_id)
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    reported_cost = 0.0

    def execute(task: LargeCatalogTask) -> TaskResult:
        nonlocal reported_cost
        result_path = root / f"{task.task_id}.json"
        if result_path.is_file():
            return _load_result(result_path)
        with lock:
            if reported_cost >= max_cost_usd:
                raise LargeCatalogRunError(
                    f"reported cost reached the US${max_cost_usd:.2f} limit"
                )
        last_transport: Exception | None = None
        for attempt in range(2):
            channel = CostTrackingOpenAIChannel(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                model=model_id,
                timeout=240.0,
                business_response_format=True,
            )
            policy = ModelPolicy(channel)
            try:
                with CatalogDatabase(database_path) as database:
                    run = run_episode(task=task, database=database, policy=policy)
            except ChannelTransportError as exc:
                last_transport = exc
                with lock:
                    reported_cost += channel.reported_cost_usd
                if attempt == 0:
                    continue
                raise
            except LargeCatalogRuntimeError as exc:
                # Strict-JSON and typed-decision failures are model outcomes,
                # not infrastructure retries.
                run = {
                    "task_id": task.task_id,
                    "role": task.role,
                    "terminal": "model_protocol_error",
                    "final_intent": None,
                    "final_arguments": None,
                    "observations": [],
                    "effects": [],
                    "trace": [],
                    "model_calls": policy.calls,
                    "latency_seconds": 0.0,
                    "protocol_error": str(exc),
                }
            with lock:
                reported_cost += channel.reported_cost_usd
            result = score_run(task, run, model_id=model_id)
            _write_json(result_path, result.to_dict())
            return result
        raise LargeCatalogRunError(
            f"{task.task_id} failed after one transport retry: {type(last_transport).__name__}"
        )

    # Canaries run sequentially and must produce ordinary scoreable records.
    canary_results = [execute(task) for task in canaries]
    if any(result.terminal not in {"completed", "model_protocol_error", "platform_rejection"} for result in canary_results):
        raise LargeCatalogRunError("canary produced an unscoreable terminal")
    results: list[TaskResult] = list(canary_results)
    if not canary_only:
        remainder = [task for task in selected if task.task_id not in CANARY_TASK_IDS]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(execute, task): task for task in remainder}
            for future in as_completed(futures):
                results.append(future.result())
                with lock:
                    if reported_cost >= max_cost_usd:
                        for pending in futures:
                            pending.cancel()
                        raise LargeCatalogRunError(
                            f"reported cost reached the US${max_cost_usd:.2f} limit"
                        )
    results.sort(key=lambda row: row.task_id)
    summary = summarize_results(results, tasks=tasks)
    summary["reported_cost_usd"] = reported_cost
    _write_json(root / "summary.json", summary)
    _write_results_csv(root / "results.csv", results)
    return summary


def summarize_results(
    results: Sequence[TaskResult],
    *,
    tasks: Sequence[LargeCatalogTask] = (),
) -> dict[str, Any]:
    if not results:
        return {
            "runs": 0,
            "overall_mean": None,
            "full": 0,
            "partial": 0,
            "zero": 0,
        }
    by_role: dict[str, list[float]] = defaultdict(list)
    by_capability: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)
    task_lookup = {
        task.task_id: {"capability": task.capability, "family": task.family}
        for task in tasks
    }
    for result in results:
        by_role[result.role].append(result.score)
        capability = task_lookup.get(result.task_id, {}).get("capability")
        if capability:
            by_capability[str(capability)].append(result.score)
        stage_rows: dict[str, list[float]] = defaultdict(list)
        for reward in result.process_rewards:
            stage_rows[reward.stage].append(reward.points / reward.maximum)
        for stage, values in stage_rows.items():
            by_stage[stage].append(mean(values))
    return {
        "runs": len(results),
        "overall_mean": mean(result.score for result in results),
        "role_means": {role: mean(values) for role, values in sorted(by_role.items())},
        "capability_means": {
            key: mean(values) for key, values in sorted(by_capability.items())
        },
        "stage_means": {key: mean(values) for key, values in sorted(by_stage.items())},
        "full": sum(result.score == 1.0 for result in results),
        "partial": sum(0.0 < result.score < 1.0 for result in results),
        "zero": sum(result.score == 0.0 for result in results),
        "protocol_errors": sum(result.terminal == "model_protocol_error" for result in results),
        "model_calls": sum(result.model_calls for result in results),
        "latency_seconds": sum(result.latency_seconds for result in results),
    }


def write_combined_summary(
    *,
    tasks: Sequence[LargeCatalogTask],
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(output_root)
    rows = []
    for model_id in LARGE_CATALOG_MODELS:
        summary_path = root / _model_slug(model_id) / "summary.json"
        if summary_path.is_file():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            row["model_id"] = model_id
            rows.append(row)
    payload = {
        "suite_tasks": len(tasks),
        "models": rows,
    }
    _write_json(root / "summary.json", payload)
    return payload


def write_prompt_audit(
    tasks: Iterable[LargeCatalogTask],
    destination: str | Path,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["task_id", "role", "family", "capability", "english_prompt", "chinese_translation"]
        )
        for task in tasks:
            writer.writerow(
                [
                    task.task_id,
                    task.role,
                    task.family,
                    task.capability,
                    task.prompt,
                    task.prompt_zh,
                ]
            )


def _write_results_csv(path: Path, results: Sequence[TaskResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "model_id",
                "role",
                "score",
                "strict_success",
                "terminal",
                "model_calls",
                "latency_seconds",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.task_id,
                    row.model_id,
                    row.role,
                    f"{row.score:.6f}",
                    str(row.strict_success).lower(),
                    row.terminal,
                    row.model_calls,
                    f"{row.latency_seconds:.6f}",
                ]
            )


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_result(path: Path) -> TaskResult:
    from large_catalog.models import ProcessReward

    row = json.loads(path.read_text(encoding="utf-8"))
    return TaskResult(
        task_id=row["task_id"],
        model_id=row["model_id"],
        role=row["role"],
        score=float(row["score"]),
        strict_success=bool(row["strict_success"]),
        terminal=row["terminal"],
        process_rewards=tuple(ProcessReward(**item) for item in row["process_rewards"]),
        trace=tuple(row.get("trace", ())),
        model_calls=int(row.get("model_calls", 0)),
        latency_seconds=float(row.get("latency_seconds", 0.0)),
        error=row.get("error"),
    )
