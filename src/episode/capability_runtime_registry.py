"""Fail-closed registry for ACWorld capability benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from episode.capability_benchmark import TASK_REGISTRY_V2
from episode.capability_runtime import RuntimeTaskBundleV2
from episode.capability_runtime_t1 import runtime_bundle_t1
from episode.capability_runtime_t2 import runtime_bundle_t2
from episode.capability_runtime_t3 import runtime_bundle_t3
from episode.capability_runtime_t4 import runtime_bundle_t4
from episode.capability_runtime_t5 import runtime_bundle_t5
from episode.capability_runtime_t6 import runtime_bundle_t6
from episode.capability_runtime_t7 import runtime_bundle_t7
from episode.capability_runtime_t8 import runtime_bundle_t8
from episode.capability_runtime_t9 import runtime_bundle_t9
from episode.capability_runtime_t10 import runtime_bundle_t10
from episode.scenario import kickoff_envelopes, population_for_scenario, seed_world
from world import World


class RuntimeTaskNotImplementedError(LookupError):
    """The fixed task has not yet been migrated into CommerceWorld."""


RuntimeBundleBuilderV2 = Callable[[str], RuntimeTaskBundleV2]


# A family enters this *formal* map only after its scenario, counterpart,
# ideal path, mutation, scorer, replay tests, and every required CommerceWorld
# core subsystem are present.  Migration modules may exist outside this map
# while their World/Platform persistence is still being implemented.  This
# fail-closed split prevents a structurally runnable scenario from entering a
# paid contract while it still carries benchmark-only state or depends on an
# in-process Platform dictionary.
_FAMILY_BUILDERS: Mapping[str, RuntimeBundleBuilderV2] = {
    "T1": runtime_bundle_t1,
    "T2": runtime_bundle_t2,
    "T3": runtime_bundle_t3,
    "T4": runtime_bundle_t4,
    "T5": runtime_bundle_t5,
    "T6": runtime_bundle_t6,
    "T7": runtime_bundle_t7,
    "T8": runtime_bundle_t8,
    "T9": runtime_bundle_t9,
    "T10": runtime_bundle_t10,
}


@dataclass(frozen=True)
class RuntimeBenchmarkReadinessV2:
    total_tasks: int
    ready_task_ids: tuple[str, ...]
    missing_task_ids: tuple[str, ...]
    invalid_tasks: tuple[tuple[str, str], ...]

    @property
    def formal_ready(self) -> bool:
        return (
            len(self.ready_task_ids) == self.total_tasks
            and not self.missing_task_ids
            and not self.invalid_tasks
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "cwe.runtime-benchmark-readiness.v2",
            "total_tasks": self.total_tasks,
            "ready_tasks": len(self.ready_task_ids),
            "missing_tasks": len(self.missing_task_ids),
            "invalid_task_count": len(self.invalid_tasks),
            "formal_ready": self.formal_ready,
            "ready_task_ids": list(self.ready_task_ids),
            "missing_task_ids": list(self.missing_task_ids),
            "invalid_tasks": [
                {"task_id": task_id, "issue": issue} for task_id, issue in self.invalid_tasks
            ],
        }


def runtime_bundle_v2(task_id: str) -> RuntimeTaskBundleV2:
    """Resolve one migrated task without any direct-simulator fallback."""

    try:
        definition = TASK_REGISTRY_V2[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown Benchmark v2 task {task_id!r}") from exc
    builder = _FAMILY_BUILDERS.get(definition.family.value)
    if builder is None:
        raise RuntimeTaskNotImplementedError(f"{task_id} has no CommerceWorld runtime task bundle")
    bundle = builder(task_id)
    if bundle.task != definition:
        raise ValueError(f"{task_id} runtime bundle changed its fixed task definition")
    return bundle


def validate_runtime_bundle_v2(bundle: RuntimeTaskBundleV2) -> None:
    """Run the free structural lower bound for one runtime task bundle."""

    # Keep the core-ownership boundary in readiness itself, not only in the
    # later paid-inference preflight.  A family that hides a missing World
    # subsystem in scenario state is therefore never reported as executable.
    from episode.capability_runtime_authority import (
        runtime_scenario_authority_violations_v2,
    )

    authority_violations = runtime_scenario_authority_violations_v2(bundle.scenario)
    if authority_violations:
        raise ValueError(
            "scenario violates CommerceWorld core ownership: " + "; ".join(authority_violations)
        )

    population = population_for_scenario(bundle.scenario)
    actor_ids = {
        *(buyer.buyer_id for buyer in population.buyers),
        *(merchant.merchant_id for merchant in population.merchants),
    }
    if bundle.evaluated_actor_id not in actor_ids:
        raise ValueError("evaluated actor is not present in the ScenarioSpec population")
    expected_counterparts = actor_ids - {bundle.evaluated_actor_id}
    if set(bundle.counterpart_channels) != expected_counterparts:
        raise ValueError("counterpart channel set does not exactly cover non-evaluated actors")
    if len(kickoff_envelopes(bundle.scenario)) == 0:
        raise ValueError("runtime task has no kickoff event")
    world = World()
    seed_world(world, bundle.scenario)
    snapshot = world.snapshot()
    if not snapshot.catalog:
        raise ValueError("runtime task initialized an empty authoritative catalog")


def runtime_readiness_v2() -> RuntimeBenchmarkReadinessV2:
    ready: list[str] = []
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    for task_id in sorted(TASK_REGISTRY_V2):
        try:
            bundle = runtime_bundle_v2(task_id)
        except RuntimeTaskNotImplementedError:
            missing.append(task_id)
            continue
        try:
            validate_runtime_bundle_v2(bundle)
        except Exception as exc:  # noqa: BLE001 - return every free readiness issue
            invalid.append((task_id, f"{type(exc).__name__}: {exc}"))
        else:
            ready.append(task_id)
    return RuntimeBenchmarkReadinessV2(
        total_tasks=len(TASK_REGISTRY_V2),
        ready_task_ids=tuple(ready),
        missing_task_ids=tuple(missing),
        invalid_tasks=tuple(invalid),
    )


__all__ = [
    "RuntimeBenchmarkReadinessV2",
    "RuntimeTaskNotImplementedError",
    "runtime_bundle_v2",
    "runtime_readiness_v2",
    "validate_runtime_bundle_v2",
]
