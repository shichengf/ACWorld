"""Experiment planning for the 200-task ACWorld capability benchmark.

The v1 planner remains the compatibility surface for the historical S1--S40
benchmark.  This module deliberately has a separate schema: a v2 run is
identified by its suite, exact model slug, versioned task, evaluated role, and
population.  It has no public seed or rollout axis.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from episode.capability_benchmark import TASK_REGISTRY_V2, get_task_v2


PLAN_SCHEMA_V2 = "cwe.experiment-plan.v2"
RUN_SCHEMA_V2 = "cwe.experiment-run.v2"
MODEL_PANEL_ID_V2 = "acworld-qwen35-six-model-panel-v2"
STRONG_MODEL_PANEL_ID_V2 = "acworld-strong-model-panel-v1"

# Ordered from the intended inexpensive-first execution order. The order is
# retained for compatibility with the original six-model campaign.
MAIN_MODELS_V2: tuple[str, ...] = (
    "qwen/qwen3.5-plus-20260420",
    "deepseek/deepseek-v4-pro",
    "mistralai/mistral-medium-3-5",
    "google/gemini-3.5-flash",
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.6-terra",
)
# Additional endpoints use the same 200 tasks, Runtime path, Agent interface,
# and scorer as the original six-model campaign.
STRONG_MODELS_V2: tuple[str, ...] = (
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-luna",
    "anthropic/claude-opus-4.8",
)
LOCAL_STRONG_MODELS_V2: tuple[str, ...] = (
    "moonshotai/kimi-k3",
    "google/gemini-3.6-flash",
)
ALL_STRONG_MODELS_V2: tuple[str, ...] = (
    *STRONG_MODELS_V2,
    *LOCAL_STRONG_MODELS_V2,
)
# The ten models reported in the paper.  Keep this separate from the older
# "strong model" convenience group because Claude Opus 4.8 was prepared as an
# optional endpoint but is not part of the reported panel.
PAPER_MODELS_V2: tuple[str, ...] = (
    *MAIN_MODELS_V2,
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-luna",
    "moonshotai/kimi-k3",
    "google/gemini-3.6-flash",
)
SUPPORTED_BENCHMARK_MODELS_V2: tuple[str, ...] = (
    *PAPER_MODELS_V2,
    "anthropic/claude-opus-4.8",
)
LOCAL_REFERENCE_MODEL_V2 = "local/acworld-reference-smoke"
# Historical direct-simulation result records remain readable, but this slug is
# not emitted by the public planner and cannot enter a scored contract.
LEGACY_PILOT_MAIN_MODELS_V2: tuple[str, ...] = ("qwen/qwen3.7-plus",)
INTERACTION_MODELS_V2: tuple[str, ...] = (
    "qwen/qwen3.7-plus",
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.5-flash",
)
INTERACTION_POPULATION_V2: tuple[int, int] = (3, 3)

EXPECTED_TASKS_V2 = 200
EXPECTED_FAMILY_TASKS_V2 = 20
EXPECTED_BUYER_TASKS_V2 = 116
EXPECTED_MERCHANT_TASKS_V2 = 84

SuiteNameV2 = Literal["main", "self_play", "many_to_many", "all"]


def _task_family(task_id: str) -> str:
    family = get_task_v2(task_id).family
    return str(getattr(family, "value", family))


def _tasks_for(family: str, role: str) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                task
                for task in TASK_REGISTRY_V2.values()
                if str(getattr(task.family, "value", task.family)) == family
                and task.evaluated_role == role
            ),
            key=lambda task: task.task_id,
        )
    )


def _representative_task_id(family: str, role: str) -> str:
    """Choose the first (lowest-complexity) frozen task for an interaction lane."""

    candidates = _tasks_for(family, role)
    if not candidates:
        raise ValueError(f"v2 registry has no {family}/{role} task")
    return str(candidates[0].task_id)


def _representative_variant_task_id(
    family: str,
    role: str,
    template_variant_id: str,
) -> str:
    """Choose the lowest-complexity task bound to a reviewed interaction world."""

    candidates = tuple(
        task for task in _tasks_for(family, role) if task.template_variant_id == template_variant_id
    )
    if not candidates:
        raise ValueError(f"v2 registry has no {family}/{role} task for {template_variant_id}")
    return str(candidates[0].task_id)


def interaction_lanes_v2() -> tuple[tuple[str, str], ...]:
    """Return the four fixed 3x3 showcase lanes as ``(task_id, role)`` pairs."""

    return (
        (_representative_task_id("T4", "buyer"), "buyer"),
        (_representative_task_id("T4", "merchant"), "merchant"),
        (_representative_task_id("T6", "buyer"), "buyer"),
        # Reuse the reviewed S19 3x3 sponsored-ranking market and policy. T8's
        # first task is S18, whose historical scenario is only 1x3.
        (_representative_variant_task_id("T8", "buyer", "S19"), "buyer"),
    )


def self_play_task_id_v2() -> str:
    """Return the single fixed, low-complexity T4 self-play scenario."""

    return _representative_task_id("T4", "buyer")


def _validate_registry_v2() -> None:
    """Fail planning if the paper taxonomy and executable registry have drifted."""

    if len(TASK_REGISTRY_V2) != EXPECTED_TASKS_V2:
        raise ValueError(
            f"v2 registry must contain {EXPECTED_TASKS_V2} tasks, got {len(TASK_REGISTRY_V2)}"
        )

    family_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for key, task in TASK_REGISTRY_V2.items():
        if key != task.task_id:
            raise ValueError(f"v2 registry key {key!r} does not match task_id {task.task_id!r}")
        family_counts[str(getattr(task.family, "value", task.family))] += 1
        role_counts[str(task.evaluated_role)] += 1
        if task.buyers < 1 or task.merchants < 1:
            raise ValueError(f"task {task.task_id!r} has a non-positive population")

    expected_families = {f"T{number}" for number in range(1, 11)}
    if set(family_counts) != expected_families or any(
        family_counts[family] != EXPECTED_FAMILY_TASKS_V2 for family in expected_families
    ):
        raise ValueError(
            "v2 registry must contain exactly 20 tasks in each T1--T10 family: "
            f"got {dict(sorted(family_counts.items()))!r}"
        )
    expected_roles = {
        "buyer": EXPECTED_BUYER_TASKS_V2,
        "merchant": EXPECTED_MERCHANT_TASKS_V2,
    }
    if dict(role_counts) != expected_roles:
        raise ValueError(
            f"v2 registry must use the frozen 116/84 role split, got {dict(role_counts)!r}"
        )


@dataclass(frozen=True)
class RunSpecV2:
    """One immutable v2 run with no seed or rollout identity axis."""

    suite: str
    model_id: str
    task_id: str
    evaluated_role: str
    buyers: int
    merchants: int

    def __post_init__(self) -> None:
        if self.suite not in {"main", "self_play", "many_to_many"}:
            raise ValueError(f"unknown v2 experiment suite: {self.suite!r}")
        if self.buyers < 1 or self.merchants < 1:
            raise ValueError("v2 run population sizes must be positive")
        try:
            task = get_task_v2(self.task_id)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown v2 benchmark task: {self.task_id!r}") from exc

        if self.suite == "main":
            if self.model_id not in {
                *MAIN_MODELS_V2,
                *LEGACY_PILOT_MAIN_MODELS_V2,
                *SUPPORTED_BENCHMARK_MODELS_V2,
                LOCAL_REFERENCE_MODEL_V2,
            }:
                raise ValueError(f"model is outside the supported v2 panels: {self.model_id!r}")
            if self.evaluated_role != task.evaluated_role:
                raise ValueError(
                    f"main run role {self.evaluated_role!r} does not match "
                    f"task {self.task_id!r} role {task.evaluated_role!r}"
                )
            if (self.buyers, self.merchants) != (task.buyers, task.merchants):
                raise ValueError(
                    f"main run population does not match task {self.task_id!r}: "
                    f"expected {task.buyers}x{task.merchants}, "
                    f"got {self.buyers}x{self.merchants}"
                )
            return

        if self.model_id not in INTERACTION_MODELS_V2:
            raise ValueError(f"model is outside the frozen v2 interaction panel: {self.model_id!r}")
        if (self.buyers, self.merchants) != INTERACTION_POPULATION_V2:
            raise ValueError(
                "v2 interaction runs require the fixed 3x3 population, "
                f"got {self.buyers}x{self.merchants}"
            )
        if self.suite == "self_play":
            if self.evaluated_role != "self_play":
                raise ValueError("v2 self-play runs require evaluated_role='self_play'")
            if self.task_id != self_play_task_id_v2():
                raise ValueError(f"v2 self-play is frozen to task {self_play_task_id_v2()!r}")
            return

        lane = (self.task_id, self.evaluated_role)
        if lane not in interaction_lanes_v2():
            raise ValueError(f"run is outside the four frozen v2 3x3 lanes: {lane!r}")

    @property
    def task_version(self) -> str:
        return str(get_task_v2(self.task_id).task_version)

    @property
    def task_family(self) -> str:
        return _task_family(self.task_id)

    @property
    def population(self) -> tuple[int, int]:
        return self.buyers, self.merchants

    def configuration(self) -> dict[str, Any]:
        """Return the direct user-supplied axes for one benchmark run."""

        return {
            "schema_version": RUN_SCHEMA_V2,
            "suite": self.suite,
            "model_id": self.model_id,
            "task_id": self.task_id,
            "evaluated_role": self.evaluated_role,
            "buyers": self.buyers,
            "merchants": self.merchants,
        }

    def identity(self) -> dict[str, Any]:
        """Return the persisted result identity used by the stable run key."""

        return {
            **self.configuration(),
            "task_version": self.task_version,
        }

    @property
    def run_key(self) -> str:
        canonical = json.dumps(
            self.identity(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return f"run-v2-{hashlib.sha256(canonical).hexdigest()[:24]}"

    def to_plan_dict(self) -> dict[str, Any]:
        """Serialize only runnable configuration, without derived freeze metadata."""

        return self.configuration()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the strict identity embedded in persisted run results."""

        return {
            **self.identity(),
            "task_family": self.task_family,
            "run_key": self.run_key,
        }

    @classmethod
    def _from_configuration(cls, raw: dict[str, Any]) -> "RunSpecV2":
        if str(raw.get("schema_version", "")) != RUN_SCHEMA_V2:
            raise ValueError(f"unsupported v2 run schema: {raw.get('schema_version')!r}")
        legacy_axes = {"seed", "rollout"} & set(raw)
        if legacy_axes:
            raise ValueError(f"v2 run identity cannot contain legacy axes: {sorted(legacy_axes)!r}")
        return cls(
            suite=str(raw["suite"]),
            model_id=str(raw["model_id"]),
            task_id=str(raw["task_id"]),
            evaluated_role=str(raw["evaluated_role"]),
            buyers=int(raw["buyers"]),
            merchants=int(raw["merchants"]),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunSpecV2":
        """Load the strict identity stored inside a run result."""

        spec = cls._from_configuration(raw)
        stored_version = raw.get("task_version")
        if stored_version is None or str(stored_version) != spec.task_version:
            raise ValueError(
                f"task version does not match registry for {spec.task_id!r}: "
                f"stored={stored_version!r}, registry={spec.task_version!r}"
            )
        stored_family = raw.get("task_family")
        if stored_family is not None and str(stored_family) != spec.task_family:
            raise ValueError(
                f"task family does not match registry for {spec.task_id!r}: "
                f"stored={stored_family!r}, registry={spec.task_family!r}"
            )
        stored_key = raw.get("run_key")
        if stored_key is not None and str(stored_key) != spec.run_key:
            raise ValueError(
                f"v2 run key does not match identity: stored={stored_key!r}, "
                f"computed={spec.run_key!r}"
            )
        return spec

    @classmethod
    def from_plan_dict(cls, raw: dict[str, Any]) -> "RunSpecV2":
        """Load runnable configuration without requiring derived version fields."""

        spec = cls._from_configuration(raw)
        stored_family = raw.get("task_family")
        if stored_family is not None and str(stored_family) != spec.task_family:
            raise ValueError(
                f"task family does not match registry for {spec.task_id!r}: "
                f"stored={stored_family!r}, registry={spec.task_family!r}"
            )
        stored_key = raw.get("run_key")
        if stored_key is not None and str(stored_key) != spec.run_key:
            raise ValueError(
                f"v2 run key does not match configuration: stored={stored_key!r}, "
                f"computed={spec.run_key!r}"
            )
        return spec


@dataclass(frozen=True)
class ExperimentPlanV2:
    """A serializable v2 plan or a validated subset such as the canary."""

    runs: tuple[RunSpecV2, ...]
    model_panel_id: str = MODEL_PANEL_ID_V2
    schema_version: str = PLAN_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA_V2:
            raise ValueError(f"unsupported v2 plan schema: {self.schema_version!r}")
        if self.model_panel_id not in {
            MODEL_PANEL_ID_V2,
            STRONG_MODEL_PANEL_ID_V2,
        }:
            raise ValueError(f"unknown v2 model panel: {self.model_panel_id!r}")
        keys = [run.run_key for run in self.runs]
        if len(keys) != len(set(keys)):
            raise ValueError("v2 experiment plan contains duplicate run keys")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_panel_id": self.model_panel_id,
            "planned_runs": len(self.runs),
            "runs": [run.to_plan_dict() for run in self.runs],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentPlanV2":
        if str(raw.get("schema_version", "")) != PLAN_SCHEMA_V2:
            raise ValueError(f"unsupported v2 plan schema: {raw.get('schema_version')!r}")
        runs = raw.get("runs")
        if not isinstance(runs, list) or any(not isinstance(item, dict) for item in runs):
            raise ValueError("v2 experiment plan runs must be a list of objects")
        plan = cls(
            model_panel_id=str(raw.get("model_panel_id", "")),
            runs=tuple(RunSpecV2.from_plan_dict(item) for item in runs),
        )
        stored_count = raw.get("planned_runs")
        if stored_count is not None and int(stored_count) != len(plan.runs):
            raise ValueError("v2 experiment plan planned_runs does not match runs")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentPlanV2":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("v2 experiment plan root must be an object")
        return cls.from_dict(raw)


def _main_runs_v2() -> list[RunSpecV2]:
    return [
        RunSpecV2(
            suite="main",
            model_id=model_id,
            task_id=task.task_id,
            evaluated_role=task.evaluated_role,
            buyers=task.buyers,
            merchants=task.merchants,
        )
        for model_id in MAIN_MODELS_V2
        for task in sorted(TASK_REGISTRY_V2.values(), key=lambda item: item.task_id)
    ]


def _self_play_runs_v2() -> list[RunSpecV2]:
    buyers, merchants = INTERACTION_POPULATION_V2
    task_id = self_play_task_id_v2()
    return [
        RunSpecV2(
            suite="self_play",
            model_id=model_id,
            task_id=task_id,
            evaluated_role="self_play",
            buyers=buyers,
            merchants=merchants,
        )
        for model_id in INTERACTION_MODELS_V2
    ]


def _many_to_many_runs_v2() -> list[RunSpecV2]:
    buyers, merchants = INTERACTION_POPULATION_V2
    return [
        RunSpecV2(
            suite="many_to_many",
            model_id=model_id,
            task_id=task_id,
            evaluated_role=role,
            buyers=buyers,
            merchants=merchants,
        )
        for model_id in INTERACTION_MODELS_V2
        for task_id, role in interaction_lanes_v2()
    ]


def build_experiment_benchmark_plan(suite: SuiteNameV2 = "all") -> ExperimentPlanV2:
    """Build the original six-model matrix and optional interaction suites."""

    if suite not in {"main", "self_play", "many_to_many", "all"}:
        raise ValueError(f"unknown v2 experiment suite: {suite!r}")
    _validate_registry_v2()
    runs: list[RunSpecV2] = []
    if suite in {"main", "all"}:
        runs.extend(_main_runs_v2())
    if suite in {"self_play", "all"}:
        runs.extend(_self_play_runs_v2())
    if suite in {"many_to_many", "all"}:
        runs.extend(_many_to_many_runs_v2())

    expected = {
        "main": 1200,
        "self_play": 3,
        "many_to_many": 12,
        "all": 1215,
    }[suite]
    if len(runs) != expected:
        raise ValueError(f"v2 {suite} matrix must contain {expected} runs, got {len(runs)}")
    return ExperimentPlanV2(runs=tuple(runs))


def build_strong_model_benchmark_plan(
    model_ids: tuple[str, ...],
    *,
    task_ids: tuple[str, ...] | None = None,
) -> ExperimentPlanV2:
    """Build a supplemental model plan over the unchanged benchmark tasks."""

    _validate_registry_v2()
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("strong-model selection must be non-empty and unique")
    unsupported = tuple(
        model_id
        for model_id in model_ids
        if model_id not in SUPPORTED_BENCHMARK_MODELS_V2
    )
    if unsupported:
        raise ValueError(f"unsupported strong-model selection: {unsupported!r}")

    selected_task_ids = (
        tuple(sorted(TASK_REGISTRY_V2))
        if task_ids is None
        else task_ids
    )
    if not selected_task_ids or len(selected_task_ids) != len(set(selected_task_ids)):
        raise ValueError("strong-model task selection must be non-empty and unique")
    missing = tuple(task_id for task_id in selected_task_ids if task_id not in TASK_REGISTRY_V2)
    if missing:
        raise ValueError(f"unknown strong-model benchmark tasks: {missing!r}")

    runs = tuple(
        RunSpecV2(
            suite="main",
            model_id=model_id,
            task_id=task_id,
            evaluated_role=get_task_v2(task_id).evaluated_role,
            buyers=get_task_v2(task_id).buyers,
            merchants=get_task_v2(task_id).merchants,
        )
        for model_id in model_ids
        for task_id in selected_task_ids
    )
    return ExperimentPlanV2(
        runs=runs,
        model_panel_id=STRONG_MODEL_PANEL_ID_V2,
    )


def build_local_reference_smoke_benchmark_plan() -> ExperimentPlanV2:
    """Build one Buyer and one Merchant smoke run on hardened task families."""

    task_ids = ("CWV2-T04-01", "CWV2-T07-20")
    runs = tuple(
        RunSpecV2(
            suite="main",
            model_id=LOCAL_REFERENCE_MODEL_V2,
            task_id=task_id,
            evaluated_role=get_task_v2(task_id).evaluated_role,
            buyers=get_task_v2(task_id).buyers,
            merchants=get_task_v2(task_id).merchants,
        )
        for task_id in task_ids
    )
    return ExperimentPlanV2(
        runs=runs,
        model_panel_id=STRONG_MODEL_PANEL_ID_V2,
    )


def benchmark_canary_task_ids() -> tuple[str, ...]:
    """Freeze two representative tasks per family for a model canary.

    A mixed-role family contributes its lowest-complexity Buyer task and its
    highest-complexity Merchant task.  A single-role family contributes the
    overall easiest and hardest tasks.  The selection is deterministic and is
    derived only from the fixed 200-task registry.
    """

    selected: list[str] = []
    for family_number in range(1, 11):
        family = f"T{family_number}"
        rows = tuple(
            sorted(
                (
                    task
                    for task in TASK_REGISTRY_V2.values()
                    if _task_family(task.task_id) == family
                ),
                key=lambda task: (int(task.difficulty_rank), str(task.task_id)),
            )
        )
        buyer_rows = tuple(task for task in rows if task.evaluated_role == "buyer")
        merchant_rows = tuple(task for task in rows if task.evaluated_role == "merchant")
        if buyer_rows and merchant_rows:
            pair = (buyer_rows[0], merchant_rows[-1])
        else:
            pair = (rows[0], rows[-1])
        selected.extend(str(task.task_id) for task in pair)
    if len(selected) != 20 or len(set(selected)) != 20:
        raise ValueError("model canary must select 20 unique fixed tasks")
    return tuple(selected)


def select_model_canary_runs_v4(
    plan: ExperimentPlanV2,
    model_id: str,
) -> ExperimentPlanV2:
    """Select one exact 20-run model canary from the original core plan."""

    if model_id not in MAIN_MODELS_V2:
        raise ValueError("canary model is outside the frozen model panel")
    by_identity = {
        (run.model_id, run.task_id): run
        for run in plan.runs
        if run.suite == "main"
    }
    identities = tuple((model_id, task_id) for task_id in benchmark_canary_task_ids())
    missing = tuple(identity for identity in identities if identity not in by_identity)
    if missing:
        raise ValueError(f"source plan is missing model canary runs: {missing!r}")
    selected = tuple(by_identity[identity] for identity in identities)
    if len(selected) != 20:
        raise ValueError("model canary must contain exactly 20 main runs")
    return ExperimentPlanV2(model_panel_id=plan.model_panel_id, runs=selected)


__all__ = [
    "EXPECTED_BUYER_TASKS_V2",
    "EXPECTED_FAMILY_TASKS_V2",
    "EXPECTED_MERCHANT_TASKS_V2",
    "EXPECTED_TASKS_V2",
    "ExperimentPlanV2",
    "ALL_STRONG_MODELS_V2",
    "INTERACTION_MODELS_V2",
    "INTERACTION_POPULATION_V2",
    "LOCAL_REFERENCE_MODEL_V2",
    "LOCAL_STRONG_MODELS_V2",
    "MAIN_MODELS_V2",
    "MODEL_PANEL_ID_V2",
    "PAPER_MODELS_V2",
    "PLAN_SCHEMA_V2",
    "RUN_SCHEMA_V2",
    "RunSpecV2",
    "STRONG_MODEL_PANEL_ID_V2",
    "STRONG_MODELS_V2",
    "SUPPORTED_BENCHMARK_MODELS_V2",
    "SuiteNameV2",
    "build_experiment_benchmark_plan",
    "build_local_reference_smoke_benchmark_plan",
    "build_strong_model_benchmark_plan",
    "benchmark_canary_task_ids",
    "interaction_lanes_v2",
    "select_model_canary_runs_v4",
    "self_play_task_id_v2",
]
