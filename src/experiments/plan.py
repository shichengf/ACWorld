"""Deterministic, offline planning for the KDD 2027 experiment matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from episode.benchmark import VARIANT_REGISTRY
from experiments.manifest import ModelManifest


PLAN_SCHEMA = "cwe.experiment-plan.v1"
RUN_SCHEMA = "cwe.experiment-run.v1"
DEFAULT_SEEDS: tuple[int, ...] = (42, 1337, 2024)
COMPACT_SEED = 42
PLAN_PROFILES: tuple[str, ...] = ("full", "compact", "hybrid")
MAIN_VARIANTS: tuple[str, ...] = tuple(
    variant_id
    for variant_id, definition in VARIANT_REGISTRY.items()
    if definition.track.value == "agent"
)
SELF_PLAY_VARIANTS: tuple[str, ...] = (
    "S2",   # T1 discovery
    "S12",  # T2 grounding
    "S13",  # T3 preference/social
    "S3",   # T4 negotiation
    "S20",  # T5 cart
    "S22",  # T6 inventory
    "S36",  # T7 after-sales
    "S19",  # T8 governance
    "S28",  # T9 adversarial content
    "S38",  # T10 transaction integrity
)
MANY_TO_MANY_VARIANTS: tuple[str, ...] = ("S3", "S22", "S19")
HYBRID_MAIN_MODELS: tuple[str, ...] = (
    "openai/gpt-5.6-terra",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.5-plus-20260420",
    "mistralai/mistral-medium-3-5",
)
HYBRID_INTERACTION_MODELS: tuple[str, ...] = (
    "google/gemini-3.5-flash",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.5-plus-20260420",
)
HYBRID_MAIN_LANES: tuple[tuple[str, str], ...] = (
    ("S2", "buyer"),
    ("S12", "buyer"),
    ("S13", "buyer"),
    ("S3", "buyer"),
    ("S20", "buyer"),
    ("S22", "buyer"),
    ("S36", "buyer"),
    ("S19", "buyer"),
    ("S28", "buyer"),
    ("S38", "buyer"),
    ("S3", "merchant"),
    ("S40", "merchant"),
)
HYBRID_SELF_PLAY_VARIANTS: tuple[str, ...] = ("S3",)
HYBRID_MANY_TO_MANY_LANES: tuple[tuple[str, str], ...] = (
    ("S3", "buyer"),
    ("S3", "merchant"),
    ("S22", "buyer"),
    ("S19", "buyer"),
)
# Paid model-controlled markets use the compact 3x3 topology in every
# profile.  The separate deterministic scripted-market baseline remains 5x5.
MODEL_MANY_TO_MANY_POPULATION: tuple[int, int] = (3, 3)

# Population is an experiment-identity axis, not an executor default.  Most
# role-isolated benchmark instances use the legacy 1x1 compatibility shape,
# while S18 is intentionally one buyer comparing three independently routed
# merchants.  Keeping this mapping beside the preregistered matrix prevents a
# planner from producing a syntactically valid but unresolvable 1x1 S18 run.
ROLE_ISOLATED_POPULATIONS: dict[str, tuple[int, int]] = {
    "S18": (1, 3),
    "S19": (1, 3),
    "S34": (1, 4),
}

# A model run is meaningful only when the scenario gives that role an
# action-bearing decision.  Keeping N/A roles out of the paid matrix avoids
# reporting zero-call rows as agent evidence.  Most tasks evaluate the buyer;
# four negotiation/privacy variants exercise both sides, and five variants
# specifically evaluate a merchant response.
DUAL_ROLE_VARIANTS = frozenset({"S3", "S11", "S30", "S31"})
MERCHANT_ONLY_VARIANTS = frozenset({"S26", "S27", "S32", "S35", "S40"})
MAIN_VARIANT_ROLES: dict[str, tuple[str, ...]] = {
    variant_id: (
        ("merchant",)
        if variant_id in MERCHANT_ONLY_VARIANTS
        else ("buyer", "merchant")
        if variant_id in DUAL_ROLE_VARIANTS
        else ("buyer",)
    )
    for variant_id in MAIN_VARIANTS
}

SuiteName = Literal["main", "self_play", "many_to_many", "all"]
EvaluatedRole = Literal["buyer", "merchant", "self_play"]
PlanProfile = Literal["full", "compact", "hybrid"]


@dataclass(frozen=True)
class RunSpec:
    """One immutable experimental unit; every identity field enters ``run_key``."""

    suite: str
    model_id: str
    variant_id: str
    seed: int
    evaluated_role: str
    rollout: int = 0
    buyers: int = 1
    merchants: int = 1

    def __post_init__(self) -> None:
        if self.suite not in {"main", "self_play", "many_to_many"}:
            raise ValueError(f"unknown experiment suite: {self.suite!r}")
        if self.variant_id not in VARIANT_REGISTRY:
            raise ValueError(f"unknown benchmark variant: {self.variant_id!r}")
        if self.evaluated_role not in {"buyer", "merchant", "self_play"}:
            raise ValueError(f"unknown evaluated role: {self.evaluated_role!r}")
        if self.seed < 0 or self.rollout < 0:
            raise ValueError("seed and rollout must be non-negative")
        if self.buyers < 1 or self.merchants < 1:
            raise ValueError("population sizes must be positive")

    @property
    def task_family(self) -> str | None:
        family = VARIANT_REGISTRY[self.variant_id].task_family
        return family.value if family is not None else None

    def identity(self) -> dict[str, Any]:
        """Canonical identity payload used to derive the stable run key."""

        return {
            "schema_version": RUN_SCHEMA,
            "suite": self.suite,
            "model_id": self.model_id,
            "variant_id": self.variant_id,
            "seed": self.seed,
            "evaluated_role": self.evaluated_role,
            "rollout": self.rollout,
            "buyers": self.buyers,
            "merchants": self.merchants,
        }

    @property
    def run_key(self) -> str:
        canonical = json.dumps(
            self.identity(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return f"run-{hashlib.sha256(canonical).hexdigest()[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity(), "task_family": self.task_family, "run_key": self.run_key}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunSpec":
        if str(raw.get("schema_version", "")) != RUN_SCHEMA:
            raise ValueError(f"unsupported run schema: {raw.get('schema_version')!r}")
        spec = cls(
            suite=str(raw["suite"]),
            model_id=str(raw["model_id"]),
            variant_id=str(raw["variant_id"]),
            seed=int(raw["seed"]),
            evaluated_role=str(raw["evaluated_role"]),
            rollout=int(raw.get("rollout", 0)),
            buyers=int(raw.get("buyers", 1)),
            merchants=int(raw.get("merchants", 1)),
        )
        stored_key = raw.get("run_key")
        if stored_key is not None and str(stored_key) != spec.run_key:
            raise ValueError(
                f"run key does not match identity: stored={stored_key!r}, "
                f"computed={spec.run_key!r}"
            )
        return spec


@dataclass(frozen=True)
class ExperimentPlan:
    """A complete matrix that can be written, audited, and resumed later."""

    manifest_id: str
    runs: tuple[RunSpec, ...]
    schema_version: str = PLAN_SCHEMA
    profile: str = "full"

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA:
            raise ValueError(f"unsupported experiment plan schema: {self.schema_version!r}")
        if self.profile not in PLAN_PROFILES:
            raise ValueError(f"unsupported experiment plan profile: {self.profile!r}")
        keys = [run.run_key for run in self.runs]
        if len(keys) != len(set(keys)):
            raise ValueError("experiment plan contains duplicate run keys")
        if self.profile == "compact":
            self._validate_compact_runs()
        if self.profile == "hybrid":
            self._validate_hybrid_runs()

    def _validate_compact_runs(self) -> None:
        """Keep compact plans and their selected subsets inside frozen axes."""

        for run in self.runs:
            if run.suite == "main":
                valid = run.seed in DEFAULT_SEEDS and run.rollout == 0
            else:
                valid = run.seed == COMPACT_SEED and run.rollout == 0
            if not valid:
                raise ValueError(
                    "run is outside compact profile axes: "
                    f"suite={run.suite!r}, seed={run.seed}, rollout={run.rollout}"
                )

    def _validate_hybrid_runs(self) -> None:
        """Reject runs outside the frozen environment-first paid subset."""

        for run in self.runs:
            if run.suite == "main":
                buyers, merchants = ROLE_ISOLATED_POPULATIONS.get(
                    run.variant_id, (1, 1)
                )
                valid = (
                    run.model_id in HYBRID_MAIN_MODELS
                    and (run.variant_id, run.evaluated_role) in HYBRID_MAIN_LANES
                    and run.seed in DEFAULT_SEEDS
                    and run.rollout == 0
                    and (run.buyers, run.merchants) == (buyers, merchants)
                )
            elif run.suite == "self_play":
                buyers, merchants = ROLE_ISOLATED_POPULATIONS.get(
                    run.variant_id, (1, 1)
                )
                valid = (
                    run.model_id in HYBRID_INTERACTION_MODELS
                    and run.variant_id in HYBRID_SELF_PLAY_VARIANTS
                    and run.evaluated_role == "self_play"
                    and run.seed == COMPACT_SEED
                    and run.rollout == 0
                    and (run.buyers, run.merchants) == (buyers, merchants)
                )
            else:
                valid = (
                    run.model_id in HYBRID_INTERACTION_MODELS
                    and (run.variant_id, run.evaluated_role)
                    in HYBRID_MANY_TO_MANY_LANES
                    and run.seed == COMPACT_SEED
                    and run.rollout == 0
                    and (run.buyers, run.merchants)
                    == MODEL_MANY_TO_MANY_POPULATION
                )
            if not valid:
                raise ValueError(
                    "run is outside hybrid profile membership: "
                    f"suite={run.suite!r}, model={run.model_id!r}, "
                    f"variant={run.variant_id!r}, role={run.evaluated_role!r}, "
                    f"seed={run.seed}, rollout={run.rollout}, "
                    f"population={run.buyers}x{run.merchants}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "profile": self.profile,
            "planned_runs": len(self.runs),
            "runs": [run.to_dict() for run in self.runs],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentPlan":
        if str(raw.get("schema_version", "")) != PLAN_SCHEMA:
            raise ValueError(f"unsupported experiment plan schema: {raw.get('schema_version')!r}")
        runs = raw.get("runs")
        if not isinstance(runs, list):
            raise ValueError("experiment plan runs must be a list")
        plan = cls(
            manifest_id=str(raw.get("manifest_id", "")),
            runs=tuple(RunSpec.from_dict(item) for item in runs),
            # Plans written before profiles were introduced are the original
            # full matrix and remain loadable without migration.
            profile=str(raw.get("profile", "full")),
        )
        stored_count = raw.get("planned_runs")
        if stored_count is not None and int(stored_count) != len(plan.runs):
            raise ValueError("experiment plan planned_runs does not match runs")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentPlan":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("experiment plan root must be an object")
        return cls.from_dict(raw)


def build_experiment_plan(
    suite: SuiteName,
    manifest: ModelManifest,
    *,
    profile: PlanProfile = "full",
) -> ExperimentPlan:
    """Expand one paper experiment suite locally; no model calls are made."""

    if suite not in {"main", "self_play", "many_to_many", "all"}:
        raise ValueError(f"unknown experiment suite: {suite!r}")
    if profile not in PLAN_PROFILES:
        raise ValueError(f"unknown experiment profile: {profile!r}")
    if profile == "hybrid" and set(manifest.model_ids) != set(HYBRID_MAIN_MODELS):
        raise ValueError(
            "hybrid profile requires the frozen six-model manifest; "
            f"expected={list(HYBRID_MAIN_MODELS)!r}, got={list(manifest.model_ids)!r}"
        )
    runs: list[RunSpec] = []
    if suite in {"main", "all"}:
        runs.extend(_main_runs(manifest, profile=profile))
    if suite in {"self_play", "all"}:
        runs.extend(_self_play_runs(manifest, profile=profile))
    if suite in {"many_to_many", "all"}:
        runs.extend(_many_to_many_runs(manifest, profile=profile))
    return ExperimentPlan(
        manifest_id=manifest.manifest_id,
        runs=tuple(runs),
        profile=profile,
    )


def _main_runs(manifest: ModelManifest, *, profile: PlanProfile) -> list[RunSpec]:
    runs: list[RunSpec] = []
    if profile == "hybrid":
        for model_id in HYBRID_MAIN_MODELS:
            for variant_id, role in HYBRID_MAIN_LANES:
                buyers, merchants = ROLE_ISOLATED_POPULATIONS.get(variant_id, (1, 1))
                for seed in DEFAULT_SEEDS:
                    runs.append(RunSpec(
                        suite="main",
                        model_id=model_id,
                        variant_id=variant_id,
                        seed=seed,
                        evaluated_role=role,
                        buyers=buyers,
                        merchants=merchants,
                    ))
        return runs
    rollouts = range(1) if profile == "compact" else range(3)
    for model_id in manifest.model_ids:
        for variant_id in MAIN_VARIANTS:
            buyers, merchants = ROLE_ISOLATED_POPULATIONS.get(variant_id, (1, 1))
            for seed in DEFAULT_SEEDS:
                for role in MAIN_VARIANT_ROLES[variant_id]:
                    for rollout in rollouts:
                        runs.append(RunSpec(
                            suite="main",
                            model_id=model_id,
                            variant_id=variant_id,
                            seed=seed,
                            evaluated_role=role,
                            rollout=rollout,
                            buyers=buyers,
                            merchants=merchants,
                        ))
    return runs


def _self_play_runs(manifest: ModelManifest, *, profile: PlanProfile) -> list[RunSpec]:
    runs: list[RunSpec] = []
    compact_profile = profile in {"compact", "hybrid"}
    seeds = (COMPACT_SEED,) if compact_profile else DEFAULT_SEEDS
    model_ids = HYBRID_INTERACTION_MODELS if profile == "hybrid" else manifest.model_ids
    variants = HYBRID_SELF_PLAY_VARIANTS if profile == "hybrid" else SELF_PLAY_VARIANTS
    for model_id in model_ids:
        for variant_id in variants:
            buyers, merchants = ROLE_ISOLATED_POPULATIONS.get(variant_id, (1, 1))
            for seed in seeds:
                runs.append(RunSpec(
                    suite="self_play",
                    model_id=model_id,
                    variant_id=variant_id,
                    seed=seed,
                    evaluated_role="self_play",
                    buyers=buyers,
                    merchants=merchants,
                ))
    return runs


def _many_to_many_runs(
    manifest: ModelManifest,
    *,
    profile: PlanProfile,
) -> list[RunSpec]:
    compact_profile = profile in {"compact", "hybrid"}
    seeds = (COMPACT_SEED,) if compact_profile else DEFAULT_SEEDS
    model_ids = HYBRID_INTERACTION_MODELS if profile == "hybrid" else manifest.model_ids
    if profile == "hybrid":
        lanes = HYBRID_MANY_TO_MANY_LANES
    else:
        lanes = tuple(
            (variant_id, role)
            for variant_id in MANY_TO_MANY_VARIANTS
            for role in ("buyer", "merchant")
        )
    buyers, merchants = MODEL_MANY_TO_MANY_POPULATION
    return [
        RunSpec(
            suite="many_to_many", model_id=model_id, variant_id=variant_id, seed=seed,
            evaluated_role=role, buyers=buyers, merchants=merchants,
        )
        for model_id in model_ids
        for variant_id, role in lanes
        for seed in seeds
    ]


__all__ = [
    "COMPACT_SEED",
    "DEFAULT_SEEDS",
    "ExperimentPlan",
    "HYBRID_INTERACTION_MODELS",
    "HYBRID_MANY_TO_MANY_LANES",
    "HYBRID_MAIN_LANES",
    "HYBRID_MAIN_MODELS",
    "HYBRID_SELF_PLAY_VARIANTS",
    "MAIN_VARIANTS",
    "MAIN_VARIANT_ROLES",
    "DUAL_ROLE_VARIANTS",
    "MERCHANT_ONLY_VARIANTS",
    "MANY_TO_MANY_VARIANTS",
    "MODEL_MANY_TO_MANY_POPULATION",
    "PLAN_SCHEMA",
    "PLAN_PROFILES",
    "PlanProfile",
    "ROLE_ISOLATED_POPULATIONS",
    "RUN_SCHEMA",
    "RunSpec",
    "SELF_PLAY_VARIANTS",
    "build_experiment_plan",
]
