"""Paid inference gate for the ACWorld 200-task benchmark.

This module performs only deterministic local execution.  It never constructs
an OpenRouter channel.  A formal contract may be frozen only after all 200
tasks pass the real Episode path, exact replay, and task-specific rescore; the
lexicographically first task for every capability passes all of its targeted
mutations; every actual family/role pair passes one HTTP parity episode; and
the static direct-simulator import ban remains clean.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from episode.capability_benchmark import TASK_REGISTRY_V2
from episode.capability_runtime import (
    RuntimeEvidenceBundleV2,
    RuntimeTaskScoreV3,
    canonical_sha256,
    record_verified_operation_evidence_calls,
)
from episode.capability_runtime_authority import (
    runtime_evidence_core_ownership_violations_v2,
    runtime_scenario_authority_violations_v2,
)
from episode.capability_runtime_registry import (
    runtime_bundle_v2,
    runtime_readiness_v2,
)
from episode.http_launcher import run_http_episode
from episode.runner import EpisodeBatch
from episode.scenario import population_for_scenario
from episode.termination import (
    EPISODE_TERMINATION_SCHEMA,
    SCOREABLE_STOP_REASONS,
    load_verified_scoreable_termination,
)
from runtime.platform_protocol import (
    platform_decision_attribution,
    platform_decision_recognition,
)
from runtime.reference_skill_evidence import verify_reference_skill_evidence_v2
from runtime.tracker_evidence import (
    verified_model_business_choices,
    verified_model_local_finish,
    verify_all_active_actor_tracker_evidence,
    verified_tracker_emitted_msg_ids,
    verify_tracker_evidence,
)


RUNTIME_PREFLIGHT_SCHEMA_V4 = "cwe.runtime-benchmark-preflight.v4"
RUNTIME_PREFLIGHT_IDEAL_EXECUTIONS_V4 = 200
RUNTIME_PREFLIGHT_MUTATION_CAPABILITIES_V4 = 80
RUNTIME_PREFLIGHT_MUTATION_EXECUTIONS_V4 = 95
RUNTIME_PREFLIGHT_MUTATION_CHECK_PAIRS_V4 = 129
RUNTIME_PREFLIGHT_HTTP_EXECUTIONS_V4 = 18
RUNTIME_PREFLIGHT_EXECUTIONS_V4 = 313


@dataclass(frozen=True)
class _RuntimePreflightSelectionV4:
    mutation_task_ids: tuple[str, ...]
    mutation_capability_ids: tuple[str, ...]
    expected_changed_check_pairs: tuple[tuple[str, str], ...]
    http_task_ids: tuple[str, ...]
    http_family_role_pairs: tuple[tuple[str, str], ...]

    @property
    def mutation_execution_count(self) -> int:
        return len({mutation_id for mutation_id, _ in self.expected_changed_check_pairs})


def _preflight_selection_v4(task_ids: Iterable[str]) -> _RuntimePreflightSelectionV4:
    """Select the smallest deterministic coverage slice inside ``task_ids``.

    Ideal execution still covers every selected task.  Mutations cover the
    lexicographically first task for every represented capability, while HTTP
    parity covers the lexicographically first task for every represented
    ``(family, evaluated_role)`` pair.
    """

    selected = tuple(sorted(task_ids))
    if len(selected) != len(set(selected)):
        raise ValueError("runtime preflight task ids must be unique")
    unknown = tuple(task_id for task_id in selected if task_id not in TASK_REGISTRY_V2)
    if unknown:
        raise ValueError(f"runtime preflight task ids are unknown: {', '.join(unknown)}")

    first_by_capability: dict[str, str] = {}
    first_by_family_role: dict[tuple[str, str], str] = {}
    for task_id in selected:
        definition = TASK_REGISTRY_V2[task_id]
        first_by_capability.setdefault(definition.capability_id, task_id)
        first_by_family_role.setdefault(
            (definition.family.value, definition.evaluated_role), task_id
        )

    expected_pairs: list[tuple[str, str]] = []
    mutation_ids: list[str] = []
    for capability_id, task_id in sorted(first_by_capability.items()):
        mutations = runtime_bundle_v2(task_id).mutations
        if not mutations:
            raise ValueError(
                f"{task_id}: selected capability {capability_id!r} has no mutation"
            )
        expected_pairs.extend(
            (mutation.mutation_id, check_name)
            for mutation in mutations
            for check_name in mutation.expected_changed_checks
        )
        mutation_ids.extend(mutation.mutation_id for mutation in mutations)
    if len(mutation_ids) != len(set(mutation_ids)):
        raise ValueError("selected runtime mutations do not have globally unique ids")
    if len(expected_pairs) != len(set(expected_pairs)):
        raise ValueError("selected runtime mutation/check pairs are duplicated")

    selection = _RuntimePreflightSelectionV4(
        mutation_task_ids=tuple(sorted(first_by_capability.values())),
        mutation_capability_ids=tuple(sorted(first_by_capability)),
        expected_changed_check_pairs=tuple(sorted(expected_pairs)),
        http_task_ids=tuple(sorted(first_by_family_role.values())),
        http_family_role_pairs=tuple(sorted(first_by_family_role)),
    )
    if set(selected) == set(TASK_REGISTRY_V2) and (
        len(selected) != RUNTIME_PREFLIGHT_IDEAL_EXECUTIONS_V4
        or len(selection.mutation_capability_ids)
        != RUNTIME_PREFLIGHT_MUTATION_CAPABILITIES_V4
        or selection.mutation_execution_count
        != RUNTIME_PREFLIGHT_MUTATION_EXECUTIONS_V4
        or len(selection.expected_changed_check_pairs)
        != RUNTIME_PREFLIGHT_MUTATION_CHECK_PAIRS_V4
        or len(selection.http_family_role_pairs)
        != RUNTIME_PREFLIGHT_HTTP_EXECUTIONS_V4
        or (
            len(selected)
            + selection.mutation_execution_count
            + len(selection.http_task_ids)
        )
        != RUNTIME_PREFLIGHT_EXECUTIONS_V4
    ):
        raise ValueError("CommerceWorld-v2.1 runtime preflight coverage baseline drifted")
    return selection


@dataclass(frozen=True)
class RuntimePreflightEvidenceReferenceV2:
    """Durable, independently reopenable proof for one CommerceWorld episode.

    This generic attestation does not replace a task-specific scorer.  It binds
    that scorer's output to the exact hash-covered episode while separately
    recording the operations and state delta exercised by the real runtime.
    """

    run_kind: str
    episode_path: str
    episode_manifest_sha256: str
    task_score_sha256: str
    active_actor_ids: tuple[str, ...]
    verified_actor_count: int
    tracker_record_count: int
    tracker_emitted_envelope_count: int
    tracker_world_tool_result_count: int
    audit_envelope_count: int
    platform_decision_count: int
    recognized_platform_exchange_count: int
    accepted_platform_exchange_count: int
    world_commit_count: int
    world_transaction_count: int
    world_commit_ids: tuple[str, ...]
    action_kinds: tuple[str, ...]
    world_operation_kinds: tuple[str, ...]
    world_authority_action_kinds: tuple[str, ...]
    world_table_write_count: int
    authority_contract_ids: tuple[str, ...]
    authority_contract_call_count: int
    authority_claimed_commit_ids: tuple[str, ...]
    authority_unclaimed_commit_ids: tuple[str, ...]
    authority_duplicate_claim_commit_ids: tuple[str, ...]
    initial_state_sha256: str
    final_state_sha256: str
    state_changed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "run_kind": self.run_kind,
            "episode_path": self.episode_path,
            "episode_manifest_sha256": self.episode_manifest_sha256,
            "task_score_sha256": self.task_score_sha256,
            "task_specific_scorer_verified": True,
            "active_actor_ids": list(self.active_actor_ids),
            "verified_actor_count": self.verified_actor_count,
            "tracker_record_count": self.tracker_record_count,
            "tracker_emitted_envelope_count": self.tracker_emitted_envelope_count,
            "tracker_world_tool_result_count": self.tracker_world_tool_result_count,
            "audit_envelope_count": self.audit_envelope_count,
            "platform_decision_count": self.platform_decision_count,
            "recognized_platform_exchange_count": self.recognized_platform_exchange_count,
            "accepted_platform_exchange_count": self.accepted_platform_exchange_count,
            "world_commit_count": self.world_commit_count,
            "world_transaction_count": self.world_transaction_count,
            "world_commit_ids": list(self.world_commit_ids),
            "action_kinds": list(self.action_kinds),
            "world_operation_kinds": list(self.world_operation_kinds),
            "world_authority_action_kinds": list(self.world_authority_action_kinds),
            "world_table_write_count": self.world_table_write_count,
            "authority_contract_ids": list(self.authority_contract_ids),
            "authority_contract_call_count": self.authority_contract_call_count,
            "authority_claimed_commit_ids": list(self.authority_claimed_commit_ids),
            "authority_unclaimed_commit_ids": list(self.authority_unclaimed_commit_ids),
            "authority_duplicate_claim_commit_ids": list(self.authority_duplicate_claim_commit_ids),
            "authority_commit_closure_verified": (
                not self.authority_unclaimed_commit_ids
                and not self.authority_duplicate_claim_commit_ids
                and (self.authority_contract_call_count > 0 or self.world_commit_count == 0)
            ),
            "initial_state_sha256": self.initial_state_sha256,
            "final_state_sha256": self.final_state_sha256,
            "state_changed": self.state_changed,
        }


@dataclass(frozen=True)
class RuntimePreflightMutationResultV2:
    mutation_id: str
    mutation_kind: str
    expected_changed_checks: tuple[str, ...]
    changed_checks: tuple[str, ...]
    raw_capability_score: float
    capability_score: float
    model_safety_violation: bool
    model_privacy_violation: bool
    replay_verified: bool
    commerceworld_path_verified: bool
    tracker_verified: bool
    core_ownership_verified: bool
    scoreable_completion: bool
    targeted: bool
    evidence: RuntimePreflightEvidenceReferenceV2 | None = None

    @property
    def passed(self) -> bool:
        return (
            self.replay_verified
            and self.commerceworld_path_verified
            and self.tracker_verified
            and self.core_ownership_verified
            and self.scoreable_completion
            and self.targeted
            and self.evidence is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mutation_id": self.mutation_id,
            "mutation_kind": self.mutation_kind,
            "expected_changed_checks": list(self.expected_changed_checks),
            "changed_checks": list(self.changed_checks),
            "raw_capability_score": self.raw_capability_score,
            "capability_score": self.capability_score,
            "model_safety_violation": self.model_safety_violation,
            "model_privacy_violation": self.model_privacy_violation,
            "replay_verified": self.replay_verified,
            "commerceworld_path_verified": self.commerceworld_path_verified,
            "tracker_verified": self.tracker_verified,
            "core_ownership_verified": self.core_ownership_verified,
            "scoreable_completion": self.scoreable_completion,
            "targeted": self.targeted,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RuntimePreflightTaskResultV2:
    task_id: str
    ideal_score: float
    ideal_strict_success: bool
    replay_verified: bool
    commerceworld_path_verified: bool
    tracker_verified: bool
    ideal_clean_completion: bool
    mutation_count: int
    mutations_verified: int
    mutation_replays_verified: int
    mutation_commerceworld_paths_verified: int
    mutation_trackers_verified: int
    mutation_core_ownership_verified: int
    mutation_clean_completions: int
    mutation_scoreable_completions: int
    mutations: tuple[RuntimePreflightMutationResultV2, ...]
    http_parity: bool | None
    http_clean_completion: bool | None
    task_definition_sha256: str = ""
    runtime_semantic_sha256: str = ""
    scenario_sha256: str = ""
    evaluated_actor_id: str = ""
    evaluated_role: str = ""
    ideal_evidence: RuntimePreflightEvidenceReferenceV2 | None = None
    http_evidence: RuntimePreflightEvidenceReferenceV2 | None = None
    ideal_skill_path_verified: bool = False
    http_skill_path_verified: bool | None = None
    acceptable_set_branches: tuple[str, ...] = ()
    acceptable_set_branches_verified: bool = True

    @property
    def passed(self) -> bool:
        mutation_lane_verified = (
            self.mutation_count == 0
            and self.mutations_verified == 0
            and self.mutation_replays_verified == 0
            and self.mutation_commerceworld_paths_verified == 0
            and self.mutation_trackers_verified == 0
            and self.mutation_core_ownership_verified == 0
            and self.mutation_clean_completions == 0
            and self.mutation_scoreable_completions == 0
            and not self.mutations
            and not self.acceptable_set_branches
            and self.acceptable_set_branches_verified
        ) or (
            self.mutation_count > 0
            and self.mutations_verified == self.mutation_count
            and self.mutation_replays_verified == self.mutation_count
            and self.mutation_commerceworld_paths_verified == self.mutation_count
            and self.mutation_trackers_verified == self.mutation_count
            and self.mutation_core_ownership_verified == self.mutation_count
            and self.mutation_scoreable_completions == self.mutation_count
            and len(self.mutations) == self.mutation_count
            and all(mutation.passed for mutation in self.mutations)
            and self.acceptable_set_branches_verified
        )
        http_lane_verified = (
            self.http_parity is None
            and self.http_clean_completion is None
            and self.http_evidence is None
            and self.http_skill_path_verified is None
        ) or (
            self.http_parity is True
            and self.http_clean_completion is True
            and self.http_evidence is not None
            and self.http_skill_path_verified is True
        )
        return (
            self.ideal_score == 1.0
            and self.ideal_strict_success
            and self.replay_verified
            and self.commerceworld_path_verified
            and self.tracker_verified
            and self.ideal_clean_completion
            and mutation_lane_verified
            and http_lane_verified
            and bool(self.task_definition_sha256)
            and bool(self.runtime_semantic_sha256)
            and bool(self.scenario_sha256)
            and bool(self.evaluated_actor_id)
            and bool(self.evaluated_role)
            and self.ideal_evidence is not None
            and self.ideal_skill_path_verified
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "ideal_score": self.ideal_score,
            "ideal_strict_success": self.ideal_strict_success,
            "replay_verified": self.replay_verified,
            "commerceworld_path_verified": self.commerceworld_path_verified,
            "tracker_verified": self.tracker_verified,
            "ideal_clean_completion": self.ideal_clean_completion,
            "mutation_count": self.mutation_count,
            "mutations_verified": self.mutations_verified,
            "mutation_replays_verified": self.mutation_replays_verified,
            "mutation_commerceworld_paths_verified": (self.mutation_commerceworld_paths_verified),
            "mutation_trackers_verified": self.mutation_trackers_verified,
            "mutation_core_ownership_verified": (self.mutation_core_ownership_verified),
            "mutation_clean_completions": self.mutation_clean_completions,
            "mutation_scoreable_completions": self.mutation_scoreable_completions,
            "mutations": [mutation.to_dict() for mutation in self.mutations],
            "http_parity": self.http_parity,
            "http_clean_completion": self.http_clean_completion,
            "task_definition_sha256": self.task_definition_sha256,
            "runtime_semantic_sha256": self.runtime_semantic_sha256,
            "scenario_sha256": self.scenario_sha256,
            "evaluated_actor_id": self.evaluated_actor_id,
            "evaluated_role": self.evaluated_role,
            "ideal_evidence": (
                self.ideal_evidence.to_dict() if self.ideal_evidence is not None else None
            ),
            "http_evidence": (
                self.http_evidence.to_dict() if self.http_evidence is not None else None
            ),
            "ideal_skill_path_verified": self.ideal_skill_path_verified,
            "http_skill_path_verified": self.http_skill_path_verified,
            "acceptable_set_branches": list(self.acceptable_set_branches),
            "acceptable_set_branches_verified": self.acceptable_set_branches_verified,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RuntimePreflightReportV2:
    required_tasks: int
    tasks: tuple[RuntimePreflightTaskResultV2, ...]
    issues: tuple[str, ...]
    direct_import_violations: tuple[str, ...]
    git_revision: str = ""
    evidence_root: str = ""
    evidence_root_repository_relative: bool = False
    source_bindings: tuple[dict[str, object], ...] = ()

    def _coverage(self) -> dict[str, object]:
        required = _preflight_selection_v4(TASK_REGISTRY_V2)
        mutation_rows = tuple(row for row in self.tasks if row.mutation_count > 0)
        http_rows = tuple(row for row in self.tasks if row.http_evidence is not None)
        mutation_task_ids = tuple(sorted(row.task_id for row in mutation_rows))
        mutation_capabilities = tuple(
            sorted(TASK_REGISTRY_V2[row.task_id].capability_id for row in mutation_rows)
        )
        expected_changed_check_pairs = tuple(
            sorted(
                (mutation.mutation_id, check_name)
                for row in mutation_rows
                for mutation in row.mutations
                for check_name in mutation.expected_changed_checks
            )
        )
        changed_check_pairs = tuple(
            sorted(
                (mutation.mutation_id, check_name)
                for row in mutation_rows
                for mutation in row.mutations
                if mutation.passed
                for check_name in mutation.changed_checks
            )
        )
        acceptable_set_branches = tuple(
            sorted(
                f"{TASK_REGISTRY_V2[row.task_id].capability_id}::{branch}"
                for row in mutation_rows
                for branch in row.acceptable_set_branches
            )
        )
        covered_acceptable_set_branches = tuple(
            sorted(
                f"{TASK_REGISTRY_V2[row.task_id].capability_id}::{branch}"
                for row in mutation_rows
                if row.acceptable_set_branches_verified
                for branch in row.acceptable_set_branches
            )
        )
        http_task_ids = tuple(sorted(row.task_id for row in http_rows))
        http_family_role_pairs = tuple(
            sorted(
                (
                    TASK_REGISTRY_V2[row.task_id].family.value,
                    TASK_REGISTRY_V2[row.task_id].evaluated_role,
                )
                for row in http_rows
            )
        )
        return {
            "required": required,
            "mutation_task_ids": mutation_task_ids,
            "mutation_capabilities": mutation_capabilities,
            "expected_changed_check_pairs": expected_changed_check_pairs,
            "changed_check_pairs": changed_check_pairs,
            "acceptable_set_branches": acceptable_set_branches,
            "covered_acceptable_set_branches": covered_acceptable_set_branches,
            "http_task_ids": http_task_ids,
            "http_family_role_pairs": http_family_role_pairs,
        }

    @property
    def formal_ready(self) -> bool:
        coverage = self._coverage()
        required = coverage["required"]
        assert isinstance(required, _RuntimePreflightSelectionV4)
        return (
            self.required_tasks == len(TASK_REGISTRY_V2)
            and len(self.tasks) == len(TASK_REGISTRY_V2)
            and tuple(sorted(row.task_id for row in self.tasks))
            == tuple(sorted(TASK_REGISTRY_V2))
            and coverage["mutation_task_ids"] == required.mutation_task_ids
            and coverage["mutation_capabilities"] == required.mutation_capability_ids
            and len(required.mutation_capability_ids) >= 80
            and coverage["expected_changed_check_pairs"]
            == required.expected_changed_check_pairs
            and coverage["changed_check_pairs"] == required.expected_changed_check_pairs
            and bool(coverage["acceptable_set_branches"])
            and coverage["covered_acceptable_set_branches"]
            == coverage["acceptable_set_branches"]
            and coverage["http_task_ids"] == required.http_task_ids
            and coverage["http_family_role_pairs"] == required.http_family_role_pairs
            and all(row.passed for row in self.tasks)
            and not self.issues
            and not self.direct_import_violations
            and len(self.git_revision) == 40
            and bool(self.evidence_root)
            and self.evidence_root_repository_relative
            and bool(self.source_bindings)
        )

    def to_dict(self) -> dict[str, object]:
        coverage = self._coverage()
        required = coverage["required"]
        assert isinstance(required, _RuntimePreflightSelectionV4)
        references = tuple(
            reference
            for task in self.tasks
            for reference in (
                task.ideal_evidence,
                *(mutation.evidence for mutation in task.mutations),
                task.http_evidence,
            )
            if reference is not None
        )
        ideal_executions = sum(task.ideal_evidence is not None for task in self.tasks)
        mutation_executions = sum(task.mutation_count for task in self.tasks)
        http_executions = sum(task.http_evidence is not None for task in self.tasks)
        verified_executions = (
            sum(
                task.ideal_evidence is not None
                and task.replay_verified
                and task.commerceworld_path_verified
                and task.tracker_verified
                and task.ideal_clean_completion
                and task.ideal_skill_path_verified
                for task in self.tasks
            )
            + sum(mutation.passed for task in self.tasks for mutation in task.mutations)
            + sum(
                task.http_evidence is not None
                and task.http_parity is True
                and task.http_clean_completion is True
                and task.http_skill_path_verified is True
                for task in self.tasks
            )
        )
        required_executions = (
            len(TASK_REGISTRY_V2)
            + required.mutation_execution_count
            + len(required.http_task_ids)
        )

        def check_pair_payload(rows: object) -> list[dict[str, str]]:
            assert isinstance(rows, tuple)
            return [
                {"mutation_id": mutation_id, "check": check_name}
                for mutation_id, check_name in rows
            ]

        def family_role_payload(rows: object) -> list[dict[str, str]]:
            assert isinstance(rows, tuple)
            return [
                {"family": family, "evaluated_role": role}
                for family, role in rows
            ]

        return {
            "schema_version": RUNTIME_PREFLIGHT_SCHEMA_V4,
            "execution_backend": "commerceworld_episode",
            "required_tasks": self.required_tasks,
            "verified_tasks": sum(row.passed for row in self.tasks),
            "required_executions": required_executions,
            "verified_executions": verified_executions,
            "ideal_executions": ideal_executions,
            "required_mutation_executions": required.mutation_execution_count,
            "mutation_executions": mutation_executions,
            "required_http_executions": len(required.http_task_ids),
            "http_executions": http_executions,
            "required_mutation_task_ids": list(required.mutation_task_ids),
            "mutation_task_ids": list(coverage["mutation_task_ids"]),
            "required_mutation_capabilities": list(required.mutation_capability_ids),
            "mutation_capability_coverage": list(coverage["mutation_capabilities"]),
            "mutation_capability_coverage_verified": (
                coverage["mutation_capabilities"] == required.mutation_capability_ids
            ),
            "required_expected_changed_check_pairs": check_pair_payload(
                required.expected_changed_check_pairs
            ),
            "expected_changed_check_coverage": check_pair_payload(
                coverage["changed_check_pairs"]
            ),
            "expected_changed_check_coverage_verified": (
                coverage["expected_changed_check_pairs"]
                == required.expected_changed_check_pairs
                and coverage["changed_check_pairs"] == required.expected_changed_check_pairs
            ),
            "acceptable_set_branches": list(coverage["acceptable_set_branches"]),
            "acceptable_set_branch_coverage": list(
                coverage["covered_acceptable_set_branches"]
            ),
            "acceptable_set_branch_coverage_verified": (
                bool(coverage["acceptable_set_branches"])
                and coverage["covered_acceptable_set_branches"]
                == coverage["acceptable_set_branches"]
            ),
            "required_http_task_ids": list(required.http_task_ids),
            "http_task_ids": list(coverage["http_task_ids"]),
            "required_http_family_role_coverage": family_role_payload(
                required.http_family_role_pairs
            ),
            "http_family_role_coverage": family_role_payload(
                coverage["http_family_role_pairs"]
            ),
            "http_family_role_coverage_verified": (
                coverage["http_family_role_pairs"] == required.http_family_role_pairs
            ),
            "ideal_skill_paths_verified": sum(
                task.ideal_skill_path_verified for task in self.tasks
            ),
            "http_skill_paths_verified": sum(
                task.http_skill_path_verified is True for task in self.tasks
            ),
            "all_active_actor_tracker_executions_verified": len(references),
            "active_actor_trackers_verified": sum(
                reference.verified_actor_count for reference in references
            ),
            "mutation_replays_verified": sum(row.mutation_replays_verified for row in self.tasks),
            "mutation_commerceworld_paths_verified": sum(
                row.mutation_commerceworld_paths_verified for row in self.tasks
            ),
            "mutation_trackers_verified": sum(row.mutation_trackers_verified for row in self.tasks),
            "mutation_core_ownership_verified": sum(
                row.mutation_core_ownership_verified for row in self.tasks
            ),
            "issues": list(self.issues),
            "direct_import_violations": list(self.direct_import_violations),
            "git_revision": self.git_revision,
            "evidence_root": self.evidence_root,
            "evidence_root_repository_relative": self.evidence_root_repository_relative,
            "source_bindings": [dict(row) for row in self.source_bindings],
            "source_tree_sha256": canonical_sha256(self.source_bindings),
            "formal_ready": self.formal_ready,
            "tasks": [row.to_dict() for row in self.tasks],
        }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise ValueError("preflight cannot bind an invalid git revision")
    return revision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_preflight_source_bindings_v2(
    repo_root: str | Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Hash the complete source surface that can affect a formal episode.

    The preflight must become stale when any executable CommerceWorld component
    changes, not only when a family adapter or evidence verifier changes.  A
    narrower hand-maintained list previously omitted, among other files, the
    negotiation policy and actor terminal.  That allowed a candidate contract
    to pair current execution code with preflight evidence produced by older
    code.  Historical direct simulators remain deliberately excluded because
    the formal registry cannot import them and the contract inventory rejects
    them independently.
    """

    root = Path(repo_root).resolve() if repo_root is not None else _repository_root()
    patterns = (
        "src/episode/**/*.py",
        "src/agents/**/*.py",
        "src/evals/**/*.py",
        "src/runtime/**/*.py",
        "src/protocol/**/*.py",
        "src/world/**/*.py",
        "src/experiments/**/*.py",
        "src/agents/prompts/*.md",
        "skills/**/SKILL.md",
        "pyproject.toml",
        "uv.lock",
    )
    direct_markers = ("capability_benchmark_direct", "direct_executor")
    paths = {
        path.resolve()
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
        and not any(marker in path.name.casefold() for marker in direct_markers)
    }
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(f"preflight source binding is missing: {path}")
        raw = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return tuple(rows)


def _active_actor_tracker_verdicts(
    evidence: RuntimeEvidenceBundleV2,
    *,
    scenario: object,
    evaluated_actor_id: str,
    strict_ideal: bool,
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    """Verify every actor that actually entered the decision loop."""

    population = population_for_scenario(scenario)
    declared_actor_ids = {
        *(buyer.buyer_id for buyer in population.buyers),
        *(merchant.merchant_id for merchant in population.merchants),
    }
    closure = verify_all_active_actor_tracker_evidence(
        evidence,
        declared_actor_ids=declared_actor_ids,
        evaluated_actor_id=evaluated_actor_id,
        evaluated_actor_strict=strict_ideal,
    )
    if not closure.verified:
        raise ValueError("; ".join(closure.issues))
    return closure.actor_verdicts, closure.active_actor_ids


def verify_all_active_actor_trackers_v2(
    evidence: RuntimeEvidenceBundleV2,
    *,
    scenario: object,
    evaluated_actor_id: str,
    evaluated_actor_strict: bool,
) -> dict[str, object]:
    """Return a serializable all-active-actor Tracker closure verdict."""

    verdicts, active_actor_ids = _active_actor_tracker_verdicts(
        evidence,
        scenario=scenario,
        evaluated_actor_id=evaluated_actor_id,
        strict_ideal=evaluated_actor_strict,
    )
    return {
        "verified": all(verdict.verified for verdict in verdicts),
        "active_actor_ids": list(active_actor_ids),
        "verified_actor_count": sum(verdict.verified for verdict in verdicts),
        "record_count": sum(verdict.record_count for verdict in verdicts),
        "emitted_envelope_count": sum(verdict.emitted_envelope_count for verdict in verdicts),
        "world_tool_result_count": sum(verdict.world_tool_result_count for verdict in verdicts),
        "actors": [verdict.to_dict() for verdict in verdicts],
    }


def score_with_authority_commit_closure_v2(
    bundle: object,
    evidence: RuntimeEvidenceBundleV2,
):
    """Run the task scorer and prove registered contracts claim every commit."""

    with record_verified_operation_evidence_calls() as calls:
        score = getattr(bundle, "scorer")(evidence)
        from runtime.operation_evidence_completion import (
            complete_unclaimed_operation_evidence,
        )

        complete_unclaimed_operation_evidence(evidence, calls)
    return score, _authority_commit_attestation(evidence, calls)


def _authority_commit_attestation(
    evidence: RuntimeEvidenceBundleV2,
    authority_calls: Iterable[tuple[str, str, object]],
) -> dict[str, object]:
    commit_ids = tuple(str(row.get("commit_id", "")) for row in evidence.world_events)
    if any(not value for value in commit_ids) or len(commit_ids) != len(set(commit_ids)):
        raise ValueError("World commit journal has missing or duplicate commit ids")
    authority_calls = tuple(authority_calls)
    if not authority_calls:
        if commit_ids:
            raise ValueError(
                "scorer invoked no registered operation evidence contract for "
                "a nonempty World commit journal"
            )
        return {
            "verified": True,
            "contract_ids": [],
            "contract_options_sha256s": [],
            "contract_call_count": 0,
            "unique_contract_call_count": 0,
            "claimed_commit_ids": [],
            "unclaimed_commit_ids": [],
            "foreign_claim_commit_ids": [],
            "duplicate_claim_commit_ids": [],
            "world_commit_ids": [],
        }
    authority_contract_ids = tuple(call[0] for call in authority_calls)
    call_identities = tuple((call[0], call[1]) for call in authority_calls)
    repeated_call_identities = tuple(
        sorted(identity for identity in set(call_identities) if call_identities.count(identity) > 1)
    )
    if repeated_call_identities:
        raise ValueError(
            "scorer invoked the same registered operation evidence contract "
            "with identical options more than once: "
            + ", ".join(
                f"{contract_id} ({options_sha256})"
                for contract_id, options_sha256 in repeated_call_identities
            )
        )
    call_claims = tuple(
        (contract_id, options_sha256, frozenset(_extract_commit_ids(result)))
        for contract_id, options_sha256, result in authority_calls
    )
    claims_by_call = tuple(set(row[2]) for row in call_claims)
    claimed_commit_ids = tuple(
        sorted({commit_id for claims in claims_by_call for commit_id in claims})
    )
    duplicate_claims = tuple(
        sorted(
            commit_id
            for commit_id in claimed_commit_ids
            if sum(commit_id in claims for claims in claims_by_call) > 1
        )
    )
    foreign_claims = tuple(sorted(set(claimed_commit_ids) - set(commit_ids)))
    unclaimed_commits = tuple(sorted(set(commit_ids) - set(claimed_commit_ids)))
    if foreign_claims:
        raise ValueError(
            "registered operation contracts claimed foreign World commits: "
            + ", ".join(foreign_claims)
        )
    if duplicate_claims:
        raise ValueError(
            "World commits were claimed by multiple registered contract calls: "
            + ", ".join(duplicate_claims)
        )
    if unclaimed_commits:
        raise ValueError(
            "registered operation contracts do not claim every World commit: "
            + ", ".join(unclaimed_commits)
        )
    return {
        "verified": True,
        "contract_ids": list(authority_contract_ids),
        "contract_options_sha256s": [call[1] for call in authority_calls],
        "contract_call_count": len(authority_calls),
        "unique_contract_call_count": len(call_identities),
        "claimed_commit_ids": list(claimed_commit_ids),
        "unclaimed_commit_ids": list(unclaimed_commits),
        "foreign_claim_commit_ids": list(foreign_claims),
        "duplicate_claim_commit_ids": list(duplicate_claims),
        "world_commit_ids": list(commit_ids),
    }


def _evidence_reference(
    evidence: RuntimeEvidenceBundleV2,
    score: object,
    *,
    evidence_root: Path,
    run_kind: str,
    scenario: object,
    evaluated_actor_id: str,
    strict_ideal: bool,
    authority_calls: Iterable[tuple[str, str, object]],
) -> RuntimePreflightEvidenceReferenceV2:
    verdicts, active_actor_ids = _active_actor_tracker_verdicts(
        evidence,
        scenario=scenario,
        evaluated_actor_id=evaluated_actor_id,
        strict_ideal=strict_ideal,
    )
    failed = [
        f"{verdict.evaluated_actor_id}: {'; '.join(verdict.issues)}"
        for verdict in verdicts
        if not verdict.verified
    ]
    if failed:
        raise ValueError("active-actor Tracker verification failed: " + " | ".join(failed))
    try:
        episode_path = (
            evidence.episode_dir.resolve().relative_to(evidence_root.resolve()).as_posix()
        )
    except ValueError as exc:
        raise ValueError("episode evidence escapes the preflight evidence root") from exc
    manifest_path = evidence.episode_dir / "episode.evidence.json"
    manifest = evidence.evidence_manifest
    if not isinstance(manifest, Mapping):
        raise ValueError("episode evidence manifest is unavailable")
    replay = manifest.get("replay")
    streams = manifest.get("streams")
    if not isinstance(replay, Mapping) or not isinstance(streams, Mapping):
        raise ValueError("episode manifest has no replay or stream attestation")

    recognized = 0
    accepted = 0
    for decision in evidence.platform_decisions:
        is_recognized, _ = platform_decision_recognition(decision)
        recognized += int(is_recognized)
        accepted += int(is_recognized and decision.get("decision") == "accepted")
    commit_ids = tuple(str(row.get("commit_id", "")) for row in evidence.world_events)
    world_transaction_count = streams.get("world_transactions")
    if (
        isinstance(world_transaction_count, bool)
        or not isinstance(world_transaction_count, int)
        or world_transaction_count < 0
    ):
        raise ValueError("episode manifest has invalid World transaction count")
    if streams.get("world_commits") != len(evidence.world_events):
        raise ValueError("World commit journal count differs from manifest")
    authority = _authority_commit_attestation(evidence, authority_calls)
    authority_contract_ids = tuple(map(str, authority["contract_ids"]))
    claimed_commit_ids = tuple(map(str, authority["claimed_commit_ids"]))
    unclaimed_commits = tuple(map(str, authority["unclaimed_commit_ids"]))
    duplicate_claims = tuple(map(str, authority["duplicate_claim_commit_ids"]))

    return RuntimePreflightEvidenceReferenceV2(
        run_kind=run_kind,
        episode_path=episode_path,
        episode_manifest_sha256=_sha256_file(manifest_path),
        task_score_sha256=canonical_sha256(getattr(score, "to_dict")()),
        active_actor_ids=active_actor_ids,
        verified_actor_count=len(verdicts),
        tracker_record_count=sum(verdict.record_count for verdict in verdicts),
        tracker_emitted_envelope_count=sum(verdict.emitted_envelope_count for verdict in verdicts),
        tracker_world_tool_result_count=sum(
            verdict.world_tool_result_count for verdict in verdicts
        ),
        audit_envelope_count=len(evidence.envelopes),
        platform_decision_count=len(evidence.platform_decisions),
        recognized_platform_exchange_count=recognized,
        accepted_platform_exchange_count=accepted,
        world_commit_count=len(evidence.world_events),
        world_transaction_count=world_transaction_count,
        world_commit_ids=commit_ids,
        action_kinds=tuple(
            sorted(
                {
                    str(envelope["action"]["kind"])
                    for envelope in evidence.envelopes
                    if isinstance(envelope.get("action"), Mapping)
                    and isinstance(envelope["action"].get("kind"), str)
                }
            )
        ),
        world_operation_kinds=tuple(
            sorted(
                {
                    str(row.get("operation"))
                    for row in evidence.world_events
                    if isinstance(row.get("operation"), str) and row.get("operation")
                }
            )
        ),
        world_authority_action_kinds=tuple(
            sorted(
                {
                    str(row.get("authority_action"))
                    for row in evidence.world_events
                    if isinstance(row.get("authority_action"), str) and row.get("authority_action")
                }
            )
        ),
        world_table_write_count=sum(
            len(row.get("table_writes", ()))
            for row in evidence.world_events
            if isinstance(row.get("table_writes"), list)
        ),
        authority_contract_ids=authority_contract_ids,
        authority_contract_call_count=int(authority["contract_call_count"]),
        authority_claimed_commit_ids=claimed_commit_ids,
        authority_unclaimed_commit_ids=unclaimed_commits,
        authority_duplicate_claim_commit_ids=duplicate_claims,
        initial_state_sha256=evidence.initial_digest,
        final_state_sha256=evidence.final_digest,
        state_changed=evidence.initial_digest != evidence.final_digest,
    )


def _extract_commit_ids(value: object, *, _seen: set[int] | None = None) -> set[str]:
    """Recursively extract exact commit objects returned by authority contracts."""

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return set()
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return set()
    seen.add(identity)
    found: set[str] = set()
    if isinstance(value, Mapping):
        commit_id = value.get("commit_id")
        if isinstance(commit_id, str) and commit_id:
            found.add(commit_id)
        for nested in value.values():
            found.update(_extract_commit_ids(nested, _seen=seen))
        return found
    if isinstance(value, (tuple, list, set, frozenset)):
        for nested in value:
            found.update(_extract_commit_ids(nested, _seen=seen))
        return found
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            found.update(_extract_commit_ids(getattr(value, field.name), _seen=seen))
    return found


def verify_runtime_preflight_evidence_bindings_v2(
    raw: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> None:
    """Independently reopen, replay, rescore, and re-attest every report row."""

    root = Path(repo_root).resolve(strict=True)
    if raw.get("git_revision") != _git_revision(root):
        raise ValueError("runtime preflight git revision is stale")
    current_sources = runtime_preflight_source_bindings_v2(root)
    if raw.get("source_bindings") != [dict(row) for row in current_sources]:
        raise ValueError("runtime preflight source bindings are stale")
    if raw.get("source_tree_sha256") != canonical_sha256(current_sources):
        raise ValueError("runtime preflight source tree digest is stale")
    if raw.get("evidence_root_repository_relative") is not True:
        raise ValueError("formal preflight evidence root must be repository-relative")
    evidence_root_value = raw.get("evidence_root")
    if not isinstance(evidence_root_value, str) or not evidence_root_value:
        raise ValueError("runtime preflight has no durable evidence root")
    evidence_root = (root / evidence_root_value).resolve(strict=True)
    try:
        evidence_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("runtime preflight evidence root escapes repository") from exc

    task_rows = raw.get("tasks")
    if not isinstance(task_rows, list):
        raise ValueError("runtime preflight task rows are missing")
    task_ids = tuple(
        sorted(
            task_row.get("task_id")
            for task_row in task_rows
            if isinstance(task_row, Mapping) and isinstance(task_row.get("task_id"), str)
        )
    )
    if len(task_ids) != len(task_rows) or len(task_ids) != len(set(task_ids)):
        raise ValueError("runtime preflight task row identities are incomplete or duplicated")
    selection = _preflight_selection_v4(task_ids)
    mutation_task_ids = tuple(
        sorted(
            str(task_row["task_id"])
            for task_row in task_rows
            if isinstance(task_row, Mapping) and task_row.get("mutation_count") != 0
        )
    )
    if mutation_task_ids != selection.mutation_task_ids:
        raise ValueError("runtime preflight mutation task selection is not canonical")
    http_task_ids = tuple(
        sorted(
            str(task_row["task_id"])
            for task_row in task_rows
            if isinstance(task_row, Mapping) and task_row.get("http_evidence") is not None
        )
    )
    if http_task_ids and http_task_ids != selection.http_task_ids:
        raise ValueError("runtime preflight HTTP family-role selection is not canonical")
    if raw.get("mutation_task_ids") != list(mutation_task_ids):
        raise ValueError("runtime preflight mutation task coverage summary drifted")
    if raw.get("http_task_ids") != list(http_task_ids):
        raise ValueError("runtime preflight HTTP task coverage summary drifted")
    reopened_expected_check_pairs: list[tuple[str, str]] = []
    reopened_changed_check_pairs: list[tuple[str, str]] = []
    reopened_acceptable_set_branches: list[str] = []
    for task_row in task_rows:
        if not isinstance(task_row, Mapping):
            raise ValueError("runtime preflight task row is not an object")
        task_id = task_row.get("task_id")
        if not isinstance(task_id, str) or task_id not in TASK_REGISTRY_V2:
            raise ValueError("runtime preflight task identity is invalid")
        definition = TASK_REGISTRY_V2[task_id]
        bundle = runtime_bundle_v2(task_id)
        expected_task_binding = {
            "task_definition_sha256": definition.canonical_hash,
            "runtime_semantic_sha256": bundle.semantic_hash,
            "scenario_sha256": canonical_sha256(asdict(bundle.scenario)),
            "evaluated_actor_id": bundle.evaluated_actor_id,
            "evaluated_role": definition.evaluated_role,
        }
        if any(task_row.get(key) != value for key, value in expected_task_binding.items()):
            raise ValueError(f"{task_id}: runtime preflight semantic binding is stale")

        ideal_evidence, ideal_score = _reopen_and_reattest(
            task_row.get("ideal_evidence"),
            evidence_root=evidence_root,
            bundle=bundle,
            run_kind="ideal-in-process",
            strict_ideal=True,
        )
        ideal_skill_coverage = verify_reference_skill_evidence_v2(
            ideal_evidence,
            evaluated_actor_id=bundle.evaluated_actor_id,
        )
        if not ideal_skill_coverage.verified:
            detail = ideal_skill_coverage.issues[0] if ideal_skill_coverage.issues else "unknown"
            raise ValueError(f"{task_id}: reopened ideal reference bypasses skill path: {detail}")
        if task_row.get("ideal_skill_path_verified") is not True:
            raise ValueError(f"{task_id}: ideal reference skill attestation is missing")
        if (
            not _strict_replay_verified(ideal_evidence)
            or runtime_evidence_core_ownership_violations_v2(ideal_evidence)
            or _commerceworld_path_issue(
                ideal_evidence,
                evaluated_actor_id=bundle.evaluated_actor_id,
                strict_ideal=True,
            )
            is not None
            or (ideal_evidence.episode_dir / "termination.json").exists()
            or task_row.get("replay_verified") is not True
            or task_row.get("commerceworld_path_verified") is not True
            or task_row.get("tracker_verified") is not True
            or task_row.get("ideal_clean_completion") is not True
            or task_row.get("ideal_score") != ideal_score.capability_score
            or task_row.get("ideal_strict_success") is not ideal_score.strict_success
        ):
            raise ValueError(f"{task_id}: ideal score differs from reopened evidence")

        mutation_rows = task_row.get("mutations")
        expected_mutations = bundle.mutations if task_id in mutation_task_ids else ()
        if not isinstance(mutation_rows, list) or len(mutation_rows) != len(
            expected_mutations
        ):
            raise ValueError(f"{task_id}: mutation evidence rows are incomplete")
        for mutation_row, mutation in zip(
            mutation_rows,
            expected_mutations,
            strict=True,
        ):
            if not isinstance(mutation_row, Mapping):
                raise ValueError(f"{task_id}: mutation evidence row is invalid")
            mutation_evidence, mutation_score = _reopen_and_reattest(
                mutation_row.get("evidence"),
                evidence_root=evidence_root,
                bundle=bundle,
                run_kind=f"mutation:{mutation.mutation_id}",
                strict_ideal=False,
            )
            ideal_checks = {row.name: row.credit for row in ideal_score.checks}
            changed_checks = sorted(
                row.name
                for row in mutation_score.checks
                if row.credit != ideal_checks.get(row.name)
            )
            reopened_expected_check_pairs.extend(
                (mutation.mutation_id, check_name)
                for check_name in mutation.expected_changed_checks
            )
            reopened_changed_check_pairs.extend(
                (mutation.mutation_id, check_name) for check_name in changed_checks
            )
            if (
                not _strict_replay_verified(mutation_evidence)
                or runtime_evidence_core_ownership_violations_v2(mutation_evidence)
                or _commerceworld_path_issue(
                    mutation_evidence,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                    strict_ideal=False,
                )
                is not None
                or not _scoreable_completion(mutation_evidence)
                or not _mutation_tracker_has_no_forced_flush(
                    mutation_evidence,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                )
                or not _mutation_is_targeted(ideal_score, mutation_score, mutation)
                or mutation_row.get("changed_checks") != changed_checks
                or mutation_row.get("raw_capability_score") != mutation_score.raw_score
                or mutation_row.get("capability_score") != mutation_score.capability_score
                or mutation_row.get("model_safety_violation")
                is not mutation_score.model_safety_violation
                or mutation_row.get("model_privacy_violation")
                is not mutation_score.model_privacy_violation
                or mutation_row.get("replay_verified") is not True
                or mutation_row.get("commerceworld_path_verified") is not True
                or mutation_row.get("tracker_verified") is not True
                or mutation_row.get("core_ownership_verified") is not True
                or mutation_row.get("scoreable_completion") is not True
                or mutation_row.get("targeted") is not True
            ):
                raise ValueError(
                    f"{task_id}/{mutation.mutation_id}: score differs from reopened evidence"
                )

        expected_acceptable_branches = (
            _acceptable_set_branches_v4(ideal_score) if task_id in mutation_task_ids else ()
        )
        reopened_acceptable_set_branches.extend(
            f"{definition.capability_id}::{branch}"
            for branch in expected_acceptable_branches
        )
        targeted_checks = {
            check_name
            for mutation_row in mutation_rows
            if isinstance(mutation_row, Mapping) and mutation_row.get("passed") is True
            for check_name in mutation_row.get("changed_checks", ())
            if isinstance(check_name, str)
        }
        acceptable_branches_verified = all(
            branch.split("::", maxsplit=1)[0] in targeted_checks
            for branch in expected_acceptable_branches
        )
        if (
            task_row.get("acceptable_set_branches")
            != list(expected_acceptable_branches)
            or task_row.get("acceptable_set_branches_verified")
            is not acceptable_branches_verified
        ):
            raise ValueError(f"{task_id}: acceptable-set branch coverage drifted")

        if task_id not in http_task_ids:
            if any(
                task_row.get(key) is not None
                for key in (
                    "http_evidence",
                    "http_parity",
                    "http_clean_completion",
                    "http_skill_path_verified",
                )
            ):
                raise ValueError(f"{task_id}: non-selected HTTP fields are populated")
            continue

        http_evidence, http_score = _reopen_and_reattest(
            task_row.get("http_evidence"),
            evidence_root=evidence_root,
            bundle=bundle,
            run_kind="ideal-http",
            strict_ideal=True,
        )
        http_skill_coverage = verify_reference_skill_evidence_v2(
            http_evidence,
            evaluated_actor_id=bundle.evaluated_actor_id,
        )
        if not http_skill_coverage.verified:
            detail = http_skill_coverage.issues[0] if http_skill_coverage.issues else "unknown"
            raise ValueError(f"{task_id}: reopened HTTP reference bypasses skill path: {detail}")
        if task_row.get("http_skill_path_verified") is not True:
            raise ValueError(f"{task_id}: HTTP reference skill attestation is missing")
        if (
            not _strict_replay_verified(http_evidence)
            or runtime_evidence_core_ownership_violations_v2(http_evidence)
            or _commerceworld_path_issue(
                http_evidence,
                evaluated_actor_id=bundle.evaluated_actor_id,
                strict_ideal=True,
            )
            is not None
            or (http_evidence.episode_dir / "termination.json").exists()
            or http_evidence.final_digest != ideal_evidence.final_digest
            or http_score.to_dict() != ideal_score.to_dict()
            or task_row.get("http_parity") is not True
            or task_row.get("http_clean_completion") is not True
        ):
            raise ValueError(f"{task_id}: reopened HTTP score differs from in-process score")

    formal_selection = _preflight_selection_v4(TASK_REGISTRY_V2)
    mutation_capabilities = tuple(
        sorted(TASK_REGISTRY_V2[task_id].capability_id for task_id in mutation_task_ids)
    )
    http_family_roles = tuple(
        sorted(
            (
                TASK_REGISTRY_V2[task_id].family.value,
                TASK_REGISTRY_V2[task_id].evaluated_role,
            )
            for task_id in http_task_ids
        )
    )

    def check_pair_payload(rows: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
        return [
            {"mutation_id": mutation_id, "check": check_name}
            for mutation_id, check_name in sorted(rows)
        ]

    def family_role_payload(rows: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
        return [
            {"family": family, "evaluated_role": role} for family, role in sorted(rows)
        ]

    acceptable_set_branches = sorted(reopened_acceptable_set_branches)
    if (
        raw.get("required_mutation_task_ids")
        != list(formal_selection.mutation_task_ids)
        or raw.get("required_mutation_capabilities")
        != list(formal_selection.mutation_capability_ids)
        or raw.get("mutation_capability_coverage") != list(mutation_capabilities)
        or raw.get("mutation_capability_coverage_verified")
        is not (mutation_capabilities == formal_selection.mutation_capability_ids)
        or raw.get("required_expected_changed_check_pairs")
        != check_pair_payload(formal_selection.expected_changed_check_pairs)
        or raw.get("expected_changed_check_coverage")
        != check_pair_payload(reopened_changed_check_pairs)
        or raw.get("expected_changed_check_coverage_verified")
        is not (
            tuple(sorted(reopened_expected_check_pairs))
            == formal_selection.expected_changed_check_pairs
            and tuple(sorted(reopened_changed_check_pairs))
            == formal_selection.expected_changed_check_pairs
        )
        or raw.get("acceptable_set_branches") != acceptable_set_branches
        or raw.get("acceptable_set_branch_coverage") != acceptable_set_branches
        or raw.get("acceptable_set_branch_coverage_verified")
        is not bool(acceptable_set_branches)
        or raw.get("required_http_task_ids") != list(formal_selection.http_task_ids)
        or raw.get("required_http_family_role_coverage")
        != family_role_payload(formal_selection.http_family_role_pairs)
        or raw.get("http_family_role_coverage") != family_role_payload(http_family_roles)
        or raw.get("http_family_role_coverage_verified")
        is not (http_family_roles == formal_selection.http_family_role_pairs)
    ):
        raise ValueError("runtime preflight coverage summary differs from reopened evidence")


def _reopen_and_reattest(
    reference: object,
    *,
    evidence_root: Path,
    bundle: object,
    run_kind: str,
    strict_ideal: bool,
):
    if not isinstance(reference, Mapping):
        raise ValueError(f"{run_kind}: evidence reference is missing")
    relative = reference.get("episode_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{run_kind}: evidence path is missing")
    episode_dir = (evidence_root / relative).resolve(strict=True)
    try:
        episode_dir.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(f"{run_kind}: evidence path escapes its root") from exc
    evidence = RuntimeEvidenceBundleV2.load(episode_dir)
    with record_verified_operation_evidence_calls() as authority_calls:
        score = getattr(bundle, "scorer")(evidence)
    rebuilt = _evidence_reference(
        evidence,
        score,
        evidence_root=evidence_root,
        run_kind=run_kind,
        scenario=getattr(bundle, "scenario"),
        evaluated_actor_id=getattr(bundle, "evaluated_actor_id"),
        strict_ideal=strict_ideal,
        authority_calls=authority_calls,
    )
    if reference != rebuilt.to_dict():
        raise ValueError(f"{run_kind}: evidence reference differs from reopened artifacts")
    return evidence, score


def _channels(bundle: object, *, mutated: object | None = None):
    evaluated_actor_id = getattr(bundle, "evaluated_actor_id")
    evaluated = (
        getattr(mutated, "channel")() if mutated is not None else getattr(bundle, "ideal_channel")()
    )
    channels_by_actor = {
        evaluated_actor_id: evaluated,
        **{
            actor_id: factory()
            for actor_id, factory in getattr(bundle, "counterpart_channels").items()
        },
    }

    def factory(agent_id: str, role: str):
        del role
        return channels_by_actor[agent_id]

    return factory


def _run_in_process_with_authority(
    bundle: object,
    root: Path,
    *,
    mutation: object | None = None,
):
    scenario = getattr(bundle, "scenario")
    EpisodeBatch(
        scenarios=[scenario],
        channels=_channels(bundle, mutated=mutation),
        out_root=root,
        remote_world=True,
        strict_skill_selection=True,
        strict_tracker_capture=True,
    ).run()
    episode_dir = root / scenario.scenario_id
    evidence = RuntimeEvidenceBundleV2.load(episode_dir)
    with record_verified_operation_evidence_calls() as authority_calls:
        score = getattr(bundle, "scorer")(evidence)
    return evidence, score, tuple(authority_calls)


def _run_in_process(bundle: object, root: Path, *, mutation: object | None = None):
    """Compatibility reader returning the evidence and score pair."""

    evidence, score, _ = _run_in_process_with_authority(
        bundle,
        root,
        mutation=mutation,
    )
    return evidence, score


def _mutation_is_targeted(
    ideal: RuntimeTaskScoreV3,
    mutated: RuntimeTaskScoreV3,
    mutation: object,
) -> bool:
    ideal_checks = {row.name: row.credit for row in ideal.checks}
    changed_checks = {
        row.name for row in mutated.checks if row.credit != ideal_checks.get(row.name)
    }
    expected = set(getattr(mutation, "expected_changed_checks"))
    return (
        ideal.capability_score == 1.0
        and 0.0 <= mutated.capability_score < 1.0
        and bool(changed_checks)
        and changed_checks == expected
    )


def _acceptable_set_branches_v4(score: object) -> tuple[str, ...]:
    """Derive declared acceptable-set branches from scorer evidence.

    There is intentionally no hand-maintained family list.  A scorer declares
    an acceptable set by including an ``acceptable_*`` evidence field on the
    score-bearing check.  The preflight records the field's cardinality branch
    and later requires a targeted mutation to change that exact check.  An
    unclassifiable declaration fails closed instead of silently disappearing
    from coverage.
    """

    def cardinality(value: object, *, location: str) -> str:
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{location}: acceptable-set evidence has no cardinality")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"{location}: acceptable-set count is negative")
            count = value
        elif isinstance(value, (str, bytes, tuple, list, set, frozenset, Mapping)):
            count = len(value)
        else:
            raise ValueError(
                f"{location}: acceptable-set evidence type {type(value).__name__} is unsupported"
            )
        if count == 0:
            return "empty"
        if count == 1:
            return "single"
        return "multiple"

    def declarations(
        value: object,
        *,
        path: tuple[str, ...] = (),
    ) -> Iterable[tuple[str, object]]:
        if not isinstance(value, Mapping):
            return ()
        found: list[tuple[str, object]] = []
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_path = (*path, key)
            normalized = key.casefold().replace("-", "_")
            if normalized == "acceptable" or normalized.startswith("acceptable_"):
                found.append((".".join(nested_path), nested))
                continue
            if isinstance(nested, Mapping):
                found.extend(declarations(nested, path=nested_path))
        return tuple(found)

    branches: list[str] = []
    for check in getattr(score, "checks"):
        check_name = str(getattr(check, "name"))
        evidence = getattr(check, "evidence")
        for path, value in declarations(evidence):
            branches.append(
                f"{check_name}::{path}::{cardinality(value, location=f'{check_name}.{path}')}"
            )
    return tuple(sorted(set(branches)))


def _scoreable_completion(evidence: RuntimeEvidenceBundleV2) -> bool:
    """Accept a clean episode or an explicitly scoreable model/security stop."""

    path = evidence.episode_dir / "termination.json"
    if not path.exists():
        return True
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == EPISODE_TERMINATION_SCHEMA
        and value.get("status") == "aborted"
        and value.get("phase") == "runtime"
        and value.get("scoreable") is True
        and value.get("classification") in SCOREABLE_STOP_REASONS
        and value.get("stop_reason") == value.get("classification")
    )


def _strict_replay_verified(evidence: RuntimeEvidenceBundleV2) -> bool:
    """Require the independently regenerated replay result in strict mode."""

    manifest = evidence.evidence_manifest
    replay = manifest.get("replay") if isinstance(manifest, dict) else None
    return bool(
        isinstance(replay, dict)
        and replay.get("replay_ok") is True
        and replay.get("strict") is True
    )


def _mutation_tracker_has_no_forced_flush(
    evidence: RuntimeEvidenceBundleV2,
    *,
    evaluated_actor_id: str,
) -> bool:
    """Reject incomplete teardown records while allowing bound Agent guards."""

    return not any(
        row.get("agent_id") == evaluated_actor_id and row.get("terminal") == "forced_flush"
        for row in evidence.trace_rows
    )


def _commerceworld_path_issue(
    evidence: RuntimeEvidenceBundleV2,
    *,
    evaluated_actor_id: str,
    strict_ideal: bool = True,
) -> str | None:
    """Require an exact evaluated-actor CommerceWorld decision path.

    A replayable snapshot alone is not enough for the formal benchmark.  The
    Normally the evaluated policy must send at least one recognized typed
    request through Runtime to the real Platform, and the Platform decision
    journal must claim the exact audited request and response bytes.  Family
    scorers separately verify accepted task-specific Platform to World
    operation graphs.  A mutation may end in an exactly bound, scoreable Agent
    guard before an illegal action reaches Platform, so its Tracker is verified
    with ``strict_ideal=False``.  The no-exchange exceptions are one verified
    model-owned local ``finish`` or a scoreable termination bound to this exact
    evaluated actor; ideal runs, counterpart failures, and framework-only
    no-reply rows still fail closed.
    """

    manifest = evidence.evidence_manifest
    replay = manifest.get("replay") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("execution_backend") != "commerceworld_episode"
        or not isinstance(replay, dict)
        or replay.get("replay_ok") is not True
        or replay.get("strict") is not True
    ):
        return "episode manifest does not attest a replayed CommerceWorld backend"
    try:
        emitted_msg_ids = set(
            verified_tracker_emitted_msg_ids(
                evidence,
                evaluated_actor_id=evaluated_actor_id,
                strict_ideal=strict_ideal,
            )
        )
        model_choices = {
            row.emitted_msg_id: row
            for row in verified_model_business_choices(
                evidence,
                evaluated_actor_id=evaluated_actor_id,
                strict_ideal=strict_ideal,
            )
        }
        attempted = tuple(
            exchange
            for exchange in evidence.platform_exchanges(
                actor_id=evaluated_actor_id,
            )
            if exchange.request.get("msg_id") in emitted_msg_ids
        )
        attributions = tuple(
            platform_decision_attribution(
                exchange.decision,
                model_choice=model_choices.get(
                    str(exchange.request.get("msg_id", ""))
                ),
            )
            for exchange in attempted
        )
    except Exception as exc:  # exact-join failures are formal gate failures
        return f"Tracker or Platform exact exchange join failed ({type(exc).__name__})"
    invalid = tuple(row.reason for row in attributions if not row.valid_for_scoring)
    if invalid:
        return "evaluated actor has infrastructure-attributed Platform rejection: " + ", ".join(
            sorted(invalid)
        )
    if not attempted:
        if not strict_ideal:
            if verified_model_local_finish(
                evidence,
                evaluated_actor_id=evaluated_actor_id,
                strict_ideal=False,
            ):
                return None
            termination = load_verified_scoreable_termination(evidence.episode_dir)
            binding = (
                termination.get("tracker_binding")
                if isinstance(termination, Mapping)
                else None
            )
            if (
                isinstance(binding, Mapping)
                and binding.get("agent_id") == evaluated_actor_id
            ):
                return None
        return "evaluated actor Tracker has no recognized Runtime to Platform exchange"
    if any(
        not str(exchange.request.get("to", "")).startswith("platform:")
        or exchange.decision.get("decision") not in {"accepted", "rejected"}
        for exchange in attempted
    ):
        return "evaluated actor exchange is not bound to an exact Platform route"
    return None


def runtime_direct_import_violations_v2(
    source_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return imports that let a task adapter bypass the Episode execution path.

    Formal family modules may describe scenarios, construct actor policies, and
    read typed evidence.  They may not instantiate the authoritative World,
    DatabaseWorld, Platform service, or a direct benchmark executor themselves.
    Those objects are owned by :class:`EpisodeBatch`; keeping them out of task
    modules makes the required Runtime -> Platform -> World call path a static
    property in addition to the dynamic evidence checks below.
    """

    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    violations: list[str] = []
    # Scan the shared runtime boundary and registry as well as every family.
    # Otherwise a family could appear clean while a shared helper imported a
    # direct adapter on its behalf.
    for path in sorted(root.glob("capability_runtime*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            imported_names: tuple[str, ...] = ()
            if isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
                imported_names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            for module in modules:
                direct_bypass = "capability_benchmark_direct" in module or "direct_executor" in module
                execution_object_bypass = module in {
                    "agents.platform",
                    "agents.world_client",
                    "world.api",
                    "world.client",
                    "world.service",
                    "world.state",
                    "world.store",
                }
                world_reexport_bypass = module == "world" and bool(
                    {"World", "DatabaseWorld", "SQLiteWorldStore"} & set(imported_names)
                )
                registry_validation_world = (
                    path.name == "capability_runtime_registry.py"
                    and module == "world"
                    and set(imported_names) == {"World"}
                )
                if registry_validation_world:
                    # The fail-closed registry constructs a throwaway World
                    # only to prove ScenarioSpec initialization is valid.  It
                    # neither executes a policy nor produces score evidence.
                    continue
                if direct_bypass or execution_object_bypass or world_reexport_bypass:
                    violations.append(f"{path.name}:{getattr(node, 'lineno', 0)}:{module}")
    return tuple(violations)


def run_runtime_preflight_v2(
    out_root: str | Path,
    *,
    task_ids: Iterable[str] | None = None,
    verify_http: bool = False,
) -> RuntimePreflightReportV2:
    """Execute deterministic ideal/mutation gates for selected fixed tasks."""

    selected = tuple(sorted(task_ids or TASK_REGISTRY_V2))
    selection = _preflight_selection_v4(selected)
    mutation_task_ids = frozenset(selection.mutation_task_ids)
    http_task_ids = frozenset(selection.http_task_ids) if verify_http else frozenset()
    root = Path(out_root)
    rows: list[RuntimePreflightTaskResultV2] = []
    issues: list[str] = []
    readiness = runtime_readiness_v2()
    unavailable = set(selected) & (
        set(readiness.missing_task_ids) | {task_id for task_id, _ in readiness.invalid_tasks}
    )
    for task_id in sorted(unavailable):
        issues.append(f"{task_id}: runtime task bundle is not structurally ready")

    for task_id in selected:
        if task_id in unavailable:
            continue
        bundle = runtime_bundle_v2(task_id)
        authority_violations = runtime_scenario_authority_violations_v2(bundle.scenario)
        if authority_violations:
            issues.extend(
                f"{task_id}: scenario authority violation: {violation}"
                for violation in authority_violations
            )
            continue
        try:
            evidence, ideal, ideal_authority_calls = _run_in_process_with_authority(
                bundle,
                root / task_id / "ideal-in-process",
            )
            ideal_evidence_reference = _evidence_reference(
                evidence,
                ideal,
                evidence_root=root,
                run_kind="ideal-in-process",
                scenario=bundle.scenario,
                evaluated_actor_id=bundle.evaluated_actor_id,
                strict_ideal=True,
                authority_calls=ideal_authority_calls,
            )
            core_ownership_violations = runtime_evidence_core_ownership_violations_v2(evidence)
            issues.extend(
                f"{task_id}: CommerceWorld core ownership violation: {violation}"
                for violation in core_ownership_violations
            )
            replay_verified = _strict_replay_verified(evidence)
            path_issue = _commerceworld_path_issue(
                evidence,
                evaluated_actor_id=bundle.evaluated_actor_id,
            )
            commerceworld_path_verified = path_issue is None
            if path_issue is not None:
                issues.append(f"{task_id}: {path_issue}")
            tracker_verdict = verify_tracker_evidence(
                evidence,
                evaluated_actor_id=bundle.evaluated_actor_id,
                strict_ideal=True,
            )
            tracker_verified = tracker_verdict.verified
            issues.extend(
                f"{task_id}: Tracker evidence failed: {issue}" for issue in tracker_verdict.issues
            )
            ideal_skill_coverage = verify_reference_skill_evidence_v2(
                evidence,
                evaluated_actor_id=bundle.evaluated_actor_id,
            )
            ideal_skill_path_verified = ideal_skill_coverage.verified
            issues.extend(
                f"{task_id}: reference skill path failed: {issue}"
                for issue in ideal_skill_coverage.issues
            )
            ideal_clean = not (evidence.episode_dir / "termination.json").exists()
            if not ideal_clean:
                issues.append(f"{task_id}: ideal episode terminated early")
            verified_mutations = 0
            mutation_replays_verified = 0
            mutation_paths_verified = 0
            mutation_trackers_verified = 0
            mutation_core_ownership_verified = 0
            clean_mutations = 0
            scoreable_mutations = 0
            mutation_results: list[RuntimePreflightMutationResultV2] = []
            selected_mutations = bundle.mutations if task_id in mutation_task_ids else ()
            for mutation in selected_mutations:
                mutation_evidence, mutation_score, mutation_authority_calls = (
                    _run_in_process_with_authority(
                        bundle,
                        root / task_id / "mutations" / mutation.mutation_id,
                        mutation=mutation,
                    )
                )
                mutation_evidence_reference = _evidence_reference(
                    mutation_evidence,
                    mutation_score,
                    evidence_root=root,
                    run_kind=f"mutation:{mutation.mutation_id}",
                    scenario=bundle.scenario,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                    strict_ideal=False,
                    authority_calls=mutation_authority_calls,
                )
                mutation_core_violations = runtime_evidence_core_ownership_violations_v2(
                    mutation_evidence
                )
                issues.extend(
                    f"{task_id}/{mutation.mutation_id}: CommerceWorld core "
                    f"ownership violation: {violation}"
                    for violation in mutation_core_violations
                )
                mutation_core_verified = not mutation_core_violations
                if mutation_core_verified:
                    mutation_core_ownership_verified += 1

                mutation_replay_verified = _strict_replay_verified(mutation_evidence)
                if mutation_replay_verified:
                    mutation_replays_verified += 1
                else:
                    issues.append(
                        f"{task_id}/{mutation.mutation_id}: mutation failed strict replay"
                    )

                mutation_path_issue = _commerceworld_path_issue(
                    mutation_evidence,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                    strict_ideal=False,
                )
                mutation_path_verified = mutation_path_issue is None
                if mutation_path_verified:
                    mutation_paths_verified += 1
                else:
                    issues.append(f"{task_id}/{mutation.mutation_id}: {mutation_path_issue}")

                mutation_tracker_verdict = verify_tracker_evidence(
                    mutation_evidence,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                    strict_ideal=False,
                )
                mutation_tracker_verified = (
                    mutation_tracker_verdict.verified
                    and _mutation_tracker_has_no_forced_flush(
                        mutation_evidence,
                        evaluated_actor_id=bundle.evaluated_actor_id,
                    )
                )
                if mutation_tracker_verified:
                    mutation_trackers_verified += 1
                else:
                    issues.extend(
                        f"{task_id}/{mutation.mutation_id}: Tracker evidence failed: {issue}"
                        for issue in mutation_tracker_verdict.issues
                    )
                    if (
                        mutation_tracker_verdict.verified
                        and not _mutation_tracker_has_no_forced_flush(
                            mutation_evidence,
                            evaluated_actor_id=bundle.evaluated_actor_id,
                        )
                    ):
                        issues.append(
                            f"{task_id}/{mutation.mutation_id}: Tracker evidence "
                            "ended in forced_flush instead of a completed action or "
                            "scoreable guard"
                        )

                mutation_clean = not (mutation_evidence.episode_dir / "termination.json").exists()
                if mutation_clean:
                    clean_mutations += 1
                mutation_scoreable = _scoreable_completion(mutation_evidence)
                if mutation_scoreable:
                    scoreable_mutations += 1
                else:
                    issues.append(
                        f"{task_id}/{mutation.mutation_id}: mutation did not finish "
                        "cleanly or through a scoreable guard"
                    )
                mutation_targeted = _mutation_is_targeted(
                    ideal,
                    mutation_score,
                    mutation,
                )
                if (
                    mutation_scoreable
                    and mutation_targeted
                    and mutation_core_verified
                    and mutation_replay_verified
                    and mutation_path_verified
                    and mutation_tracker_verified
                ):
                    verified_mutations += 1
                if not mutation_targeted:
                    issues.append(
                        f"{task_id}/{mutation.mutation_id}: mutation is not a targeted non-full capability outcome"
                    )
                ideal_checks = {row.name: row.credit for row in ideal.checks}
                changed_checks = tuple(
                    sorted(
                        row.name
                        for row in mutation_score.checks
                        if row.credit != ideal_checks.get(row.name)
                    )
                )
                mutation_results.append(
                    RuntimePreflightMutationResultV2(
                        mutation_id=mutation.mutation_id,
                        mutation_kind=mutation.mutation_kind,
                        expected_changed_checks=tuple(sorted(mutation.expected_changed_checks)),
                        changed_checks=changed_checks,
                        raw_capability_score=mutation_score.raw_score,
                        capability_score=mutation_score.capability_score,
                        model_safety_violation=mutation_score.model_safety_violation,
                        model_privacy_violation=mutation_score.model_privacy_violation,
                        replay_verified=mutation_replay_verified,
                        commerceworld_path_verified=mutation_path_verified,
                        tracker_verified=mutation_tracker_verified,
                        core_ownership_verified=mutation_core_verified,
                        scoreable_completion=mutation_scoreable,
                        targeted=mutation_targeted,
                        evidence=mutation_evidence_reference,
                    )
                )

            acceptable_set_branches = (
                _acceptable_set_branches_v4(ideal) if task_id in mutation_task_ids else ()
            )
            targeted_checks = {
                check_name
                for mutation in mutation_results
                if mutation.passed
                for check_name in mutation.changed_checks
            }
            acceptable_set_branches_verified = all(
                branch.split("::", maxsplit=1)[0] in targeted_checks
                for branch in acceptable_set_branches
            )
            if not acceptable_set_branches_verified:
                issues.append(f"{task_id}: acceptable-set branch mutation coverage is incomplete")

            http_parity: bool | None = None
            http_clean: bool | None = None
            http_skill_path_verified: bool | None = None
            http_evidence_reference: RuntimePreflightEvidenceReferenceV2 | None = None
            if task_id in http_task_ids:
                http_root = root / task_id / "ideal-http"
                run_http_episode(
                    scenario=bundle.scenario,
                    channels=_channels(bundle),
                    out_dir=http_root,
                    strict_skill_selection=True,
                )
                http_evidence = RuntimeEvidenceBundleV2.load(http_root)
                http_core_violations = runtime_evidence_core_ownership_violations_v2(http_evidence)
                issues.extend(
                    f"{task_id}/http: CommerceWorld core ownership violation: {violation}"
                    for violation in http_core_violations
                )
                with record_verified_operation_evidence_calls() as http_authority_calls:
                    http_score = bundle.scorer(http_evidence)
                http_evidence_reference = _evidence_reference(
                    http_evidence,
                    http_score,
                    evidence_root=root,
                    run_kind="ideal-http",
                    scenario=bundle.scenario,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                    strict_ideal=True,
                    authority_calls=http_authority_calls,
                )
                http_clean = not (http_root / "termination.json").exists()
                http_path_issue = _commerceworld_path_issue(
                    http_evidence,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                )
                if http_path_issue is not None:
                    issues.append(f"{task_id}/http: {http_path_issue}")
                http_tracker_verdict = verify_tracker_evidence(
                    http_evidence,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                    strict_ideal=True,
                )
                http_tracker_verified = http_tracker_verdict.verified
                issues.extend(
                    f"{task_id}/http: Tracker evidence failed: {issue}"
                    for issue in http_tracker_verdict.issues
                )
                http_skill_coverage = verify_reference_skill_evidence_v2(
                    http_evidence,
                    evaluated_actor_id=bundle.evaluated_actor_id,
                )
                http_skill_path_verified = http_skill_coverage.verified
                issues.extend(
                    f"{task_id}/http: reference skill path failed: {issue}"
                    for issue in http_skill_coverage.issues
                )
                http_parity = (
                    http_score.to_dict() == ideal.to_dict()
                    and http_evidence.final_digest == evidence.final_digest
                    and http_path_issue is None
                    and http_tracker_verified
                    and http_skill_path_verified
                )
                if not http_parity:
                    issues.append(f"{task_id}: HTTP and in-process results differ")
                if not http_clean:
                    issues.append(f"{task_id}: HTTP ideal episode terminated early")
            row = RuntimePreflightTaskResultV2(
                task_id=task_id,
                ideal_score=ideal.capability_score,
                ideal_strict_success=ideal.strict_success,
                replay_verified=replay_verified,
                commerceworld_path_verified=commerceworld_path_verified,
                tracker_verified=tracker_verified,
                ideal_clean_completion=ideal_clean,
                mutation_count=len(selected_mutations),
                mutations_verified=verified_mutations,
                mutation_replays_verified=mutation_replays_verified,
                mutation_commerceworld_paths_verified=mutation_paths_verified,
                mutation_trackers_verified=mutation_trackers_verified,
                mutation_core_ownership_verified=(mutation_core_ownership_verified),
                mutation_clean_completions=clean_mutations,
                mutation_scoreable_completions=scoreable_mutations,
                mutations=tuple(mutation_results),
                http_parity=http_parity,
                http_clean_completion=http_clean,
                task_definition_sha256=TASK_REGISTRY_V2[task_id].canonical_hash,
                runtime_semantic_sha256=bundle.semantic_hash,
                scenario_sha256=canonical_sha256(asdict(bundle.scenario)),
                evaluated_actor_id=bundle.evaluated_actor_id,
                evaluated_role=TASK_REGISTRY_V2[task_id].evaluated_role,
                ideal_evidence=ideal_evidence_reference,
                http_evidence=http_evidence_reference,
                ideal_skill_path_verified=ideal_skill_path_verified,
                http_skill_path_verified=http_skill_path_verified,
                acceptable_set_branches=acceptable_set_branches,
                acceptable_set_branches_verified=acceptable_set_branches_verified,
            )
            rows.append(row)
            if not row.passed:
                issues.append(f"{task_id}: ideal, replay, or mutation gate failed")
        except Exception as exc:  # noqa: BLE001 - collect every local preflight defect
            issues.append(f"{task_id}: {type(exc).__name__}: {exc}")

    repo_root = _repository_root()
    resolved_root = root.resolve()
    try:
        evidence_root = resolved_root.relative_to(repo_root).as_posix()
        evidence_root_repository_relative = True
    except ValueError:
        evidence_root = resolved_root.as_posix()
        evidence_root_repository_relative = False
    return RuntimePreflightReportV2(
        required_tasks=len(selected),
        tasks=tuple(rows),
        issues=tuple(issues),
        direct_import_violations=runtime_direct_import_violations_v2(),
        git_revision=_git_revision(repo_root),
        evidence_root=evidence_root,
        evidence_root_repository_relative=evidence_root_repository_relative,
        source_bindings=runtime_preflight_source_bindings_v2(repo_root),
    )


__all__ = [
    "RUNTIME_PREFLIGHT_EXECUTIONS_V4",
    "RUNTIME_PREFLIGHT_HTTP_EXECUTIONS_V4",
    "RUNTIME_PREFLIGHT_IDEAL_EXECUTIONS_V4",
    "RUNTIME_PREFLIGHT_MUTATION_CAPABILITIES_V4",
    "RUNTIME_PREFLIGHT_MUTATION_CHECK_PAIRS_V4",
    "RUNTIME_PREFLIGHT_MUTATION_EXECUTIONS_V4",
    "RUNTIME_PREFLIGHT_SCHEMA_V4",
    "RuntimePreflightReportV2",
    "RuntimePreflightEvidenceReferenceV2",
    "RuntimePreflightMutationResultV2",
    "RuntimePreflightTaskResultV2",
    "run_runtime_preflight_v2",
    "runtime_direct_import_violations_v2",
    "runtime_preflight_source_bindings_v2",
]
