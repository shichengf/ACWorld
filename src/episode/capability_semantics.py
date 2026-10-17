"""Fail-closed semantic readiness inventory for capability benchmark tasks.

The 200-task registry is a reviewed design specification.  Reusing a legacy
scenario proves neither that a new capability is present nor that its declared
difficulty axis changes the decision.  This module records that distinction in
code so an executor cannot accidentally promote structurally materialized
compatibility fixtures into formal model-evaluation tasks.

The initial inventory comes from a task-by-task audit of the v1 scenario,
oracle, evaluated role, and reactive-policy topology.  A task becomes formally
ready only through future executable evidence; editing this classification is
not itself sufficient to pass any readiness gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2


SEMANTIC_INVENTORY_SCHEMA_V2 = "cwe.benchmark-semantic-inventory.v2"
BaseCapabilityStatusV2 = Literal[
    "strong_source_anchor",
    "weak_proxy",
    "requires_new_implementation",
]
ExecutionRouteV2 = Literal[
    "episode_anchor",
    "redesign",
    "direct_or_dedicated_episode",
]


class SemanticReadinessError(RuntimeError):
    """A structurally valid task lacks capability-specific benchmark evidence."""


@dataclass(frozen=True)
class TaskSemanticReadinessV2:
    task_id: str
    capability_id: str
    base_capability_status: BaseCapabilityStatusV2
    execution_route: ExecutionRouteV2
    difficulty_realized: bool
    capability_oracle_verified: bool
    ideal_trajectory_verified: bool
    targeted_mutation_verified: bool
    issues: tuple[str, ...]

    @property
    def formal_ready(self) -> bool:
        return (
            self.base_capability_status == "strong_source_anchor"
            and self.difficulty_realized
            and self.capability_oracle_verified
            and self.ideal_trajectory_verified
            and self.targeted_mutation_verified
            and not self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "capability_id": self.capability_id,
            "base_capability_status": self.base_capability_status,
            "execution_route": self.execution_route,
            "difficulty_realized": self.difficulty_realized,
            "capability_oracle_verified": self.capability_oracle_verified,
            "ideal_trajectory_verified": self.ideal_trajectory_verified,
            "targeted_mutation_verified": self.targeted_mutation_verified,
            "formal_ready": self.formal_ready,
            "issues": list(self.issues),
        }


def _ids(family: int, *ranges: tuple[int, int]) -> set[str]:
    return {
        f"CWV2-T{family:02d}-{ordinal:02d}"
        for start, end in ranges
        for ordinal in range(start, end + 1)
    }


# Strong means only that the base capability and evaluated role have a credible
# v1 anchor.  It does *not* mean the v2 difficulty axis, oracle, GT, or mutation
# has been implemented.  The audit totals are normative and covered by tests.
_STRONG_SOURCE_TASKS = frozenset().union(
    _ids(1, (4, 7), (13, 16)),
    _ids(2, (1, 2), (9, 12)),
    _ids(3, (1, 12)),
    _ids(4, (1, 7), (11, 17)),
    _ids(5, (1, 7)),
    _ids(6, (5, 6), (19, 20)),
    _ids(7, (1, 8)),
    _ids(8, (1, 8), (18, 20)),
    _ids(9, (1, 7), (11, 14)),
    _ids(10, (3, 4)),
)
_WEAK_PROXY_TASKS = frozenset().union(
    _ids(1, (1, 3)),
    _ids(4, (18, 19)),
    _ids(10, (1, 2)),
)


def _assessment(task: TaskDefinitionV2) -> TaskSemanticReadinessV2:
    if task.task_id in _STRONG_SOURCE_TASKS:
        status: BaseCapabilityStatusV2 = "strong_source_anchor"
        route: ExecutionRouteV2 = "episode_anchor"
        base_issue = "source_anchor_only_not_a_v2_task_implementation"
    elif task.task_id in _WEAK_PROXY_TASKS:
        status = "weak_proxy"
        route = "redesign"
        base_issue = "source_scenario_only_weakly_proxies_declared_capability"
    else:
        status = "requires_new_implementation"
        route = "direct_or_dedicated_episode"
        base_issue = "declared_capability_or_role_is_not_supported_by_source_oracle"
    return TaskSemanticReadinessV2(
        task_id=task.task_id,
        capability_id=task.capability_id,
        base_capability_status=status,
        execution_route=route,
        difficulty_realized=False,
        capability_oracle_verified=False,
        ideal_trajectory_verified=False,
        targeted_mutation_verified=False,
        issues=(
            base_issue,
            "difficulty_factor_not_yet_realized_and_verified",
            "capability_specific_oracle_not_yet_verified",
            "ideal_trajectory_not_yet_verified",
            "targeted_mutation_not_yet_verified",
        ),
    )


TASK_SEMANTIC_READINESS_V2: Mapping[str, TaskSemanticReadinessV2] = {
    task_id: _assessment(task)
    for task_id, task in TASK_REGISTRY_V2.items()
}


def semantic_readiness_summary_v2() -> dict[str, Any]:
    status_counts = Counter(
        row.base_capability_status for row in TASK_SEMANTIC_READINESS_V2.values()
    )
    route_counts = Counter(
        row.execution_route for row in TASK_SEMANTIC_READINESS_V2.values()
    )
    ready = sum(row.formal_ready for row in TASK_SEMANTIC_READINESS_V2.values())
    return {
        "schema_version": SEMANTIC_INVENTORY_SCHEMA_V2,
        "tasks": len(TASK_SEMANTIC_READINESS_V2),
        "base_capability_status": dict(sorted(status_counts.items())),
        "execution_routes": dict(sorted(route_counts.items())),
        "difficulty_realized": sum(
            row.difficulty_realized for row in TASK_SEMANTIC_READINESS_V2.values()
        ),
        "capability_oracles_verified": sum(
            row.capability_oracle_verified for row in TASK_SEMANTIC_READINESS_V2.values()
        ),
        "ideal_trajectories_verified": sum(
            row.ideal_trajectory_verified for row in TASK_SEMANTIC_READINESS_V2.values()
        ),
        "targeted_mutations_verified": sum(
            row.targeted_mutation_verified for row in TASK_SEMANTIC_READINESS_V2.values()
        ),
        "formal_ready": ready,
        "formal_pending": len(TASK_SEMANTIC_READINESS_V2) - ready,
    }


def require_formal_task_ready_v2(task_id: str) -> TaskSemanticReadinessV2:
    try:
        assessment = TASK_SEMANTIC_READINESS_V2[task_id]
    except KeyError as exc:
        raise SemanticReadinessError(f"unknown v2 task {task_id!r}") from exc
    if not assessment.formal_ready:
        raise SemanticReadinessError(
            f"v2 task {task_id} is structurally registered but not formally ready: "
            + ", ".join(assessment.issues)
        )
    return assessment


def validate_semantic_inventory_v2() -> None:
    if set(TASK_SEMANTIC_READINESS_V2) != set(TASK_REGISTRY_V2):
        raise ValueError("semantic inventory does not cover the canonical 200-task registry")
    summary = semantic_readiness_summary_v2()
    if summary["base_capability_status"] != {
        "requires_new_implementation": 110,
        "strong_source_anchor": 83,
        "weak_proxy": 7,
    }:
        raise ValueError("semantic audit classification counts have drifted")
    if summary["formal_ready"] != 0:
        raise ValueError(
            "initial semantic inventory must fail closed until executable evidence is bound"
        )


validate_semantic_inventory_v2()


__all__ = [
    "SEMANTIC_INVENTORY_SCHEMA_V2",
    "TASK_SEMANTIC_READINESS_V2",
    "SemanticReadinessError",
    "TaskSemanticReadinessV2",
    "require_formal_task_ready_v2",
    "semantic_readiness_summary_v2",
    "validate_semantic_inventory_v2",
]
