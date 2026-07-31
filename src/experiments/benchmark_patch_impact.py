"""Reproducible, hash-only impact evidence for narrow benchmark patches.

The first use of this schema is the B3 buyer cart-skill activation repair.  It
compares the pre-fix public mandate gate with the current gate across all 200
runtime tasks, then executes the 14 affected T5 Buyer tasks under both gates for
their ideal and targeted-mutation trajectories.  Provider payloads are never
persisted: only canonical request, final-World, and score digests leave the
temporary run directory.

This evidence does not rewrite or re-identify the original 1,200-run campaign.
It is a maintenance impact record for the source tree that follows it.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar, cast

from agents.memory import InMemoryStore
from agents.skill_selector import Gate, _SelectorContext
from agents.skill_selector_buyer import BuyerSkillSelector
from episode.capability_benchmark import TASK_REGISTRY_V2
from episode.capability_runtime import (
    RuntimeEvidenceBundleV2,
    canonical_sha256,
)
from episode.capability_runtime_registry import runtime_bundle_v2
from episode.runner import EpisodeBatch
from episode.scenario import kickoff_envelopes
from experiments.environment_study import write_json_artifact


BENCHMARK_PATCH_IMPACT_SCHEMA = "cwe.benchmark-patch-impact.v1"
B3_PATCH_ID = "B3-cart-planning-skill-activation"
B3_EXPECTED_TASK_IDS = tuple(f"CWV2-T05-{index:02d}" for index in range(1, 15))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VARIANTS = frozenset({"ideal", "targeted_mutation"})
_T = TypeVar("_T")


class BenchmarkPatchImpactError(ValueError):
    """The impact audit or its serialized evidence is inconsistent."""


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BenchmarkPatchImpactError(f"{field} must be a lowercase SHA-256")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    actual = frozenset(map(str, value))
    if actual != expected:
        raise BenchmarkPatchImpactError(
            f"{field} keys differ: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


@dataclass(frozen=True)
class PatchTrajectoryDigestV1:
    """One legacy/current trajectory comparison with no provider plaintext."""

    task_id: str
    variant: str
    legacy_provider_request_sha256: str
    current_provider_request_sha256: str
    legacy_decision_count: int
    current_decision_count: int
    legacy_final_world_sha256: str
    current_final_world_sha256: str
    legacy_score_sha256: str
    current_score_sha256: str

    def __post_init__(self) -> None:
        if self.task_id not in B3_EXPECTED_TASK_IDS:
            raise BenchmarkPatchImpactError("trajectory task_id is outside the B3 scope")
        if self.variant not in _VARIANTS:
            raise BenchmarkPatchImpactError("unknown patch trajectory variant")
        for name in (
            "legacy_provider_request_sha256",
            "current_provider_request_sha256",
            "legacy_final_world_sha256",
            "current_final_world_sha256",
            "legacy_score_sha256",
            "current_score_sha256",
        ):
            _require_sha256(getattr(self, name), field=name)
        for name in ("legacy_decision_count", "current_decision_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BenchmarkPatchImpactError(f"{name} must be a positive integer")

    @property
    def provider_requests_equal(self) -> bool:
        return (
            self.legacy_provider_request_sha256
            == self.current_provider_request_sha256
            and self.legacy_decision_count == self.current_decision_count
        )

    @property
    def final_world_equal(self) -> bool:
        return self.legacy_final_world_sha256 == self.current_final_world_sha256

    @property
    def score_equal(self) -> bool:
        return self.legacy_score_sha256 == self.current_score_sha256

    @property
    def semantically_equal(self) -> bool:
        return self.provider_requests_equal and self.final_world_equal and self.score_equal

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "variant": self.variant,
            "legacy_provider_request_sha256": self.legacy_provider_request_sha256,
            "current_provider_request_sha256": self.current_provider_request_sha256,
            "legacy_decision_count": self.legacy_decision_count,
            "current_decision_count": self.current_decision_count,
            "legacy_final_world_sha256": self.legacy_final_world_sha256,
            "current_final_world_sha256": self.current_final_world_sha256,
            "legacy_score_sha256": self.legacy_score_sha256,
            "current_score_sha256": self.current_score_sha256,
            "provider_requests_equal": self.provider_requests_equal,
            "final_world_equal": self.final_world_equal,
            "score_equal": self.score_equal,
            "semantically_equal": self.semantically_equal,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchTrajectoryDigestV1":
        _require_exact_keys(
            value,
            frozenset(
                {
                    "task_id",
                    "variant",
                    "legacy_provider_request_sha256",
                    "current_provider_request_sha256",
                    "legacy_decision_count",
                    "current_decision_count",
                    "legacy_final_world_sha256",
                    "current_final_world_sha256",
                    "legacy_score_sha256",
                    "current_score_sha256",
                    "provider_requests_equal",
                    "final_world_equal",
                    "score_equal",
                    "semantically_equal",
                }
            ),
            field="trajectory",
        )
        row = cls(
            task_id=str(value["task_id"]),
            variant=str(value["variant"]),
            legacy_provider_request_sha256=str(
                value["legacy_provider_request_sha256"]
            ),
            current_provider_request_sha256=str(
                value["current_provider_request_sha256"]
            ),
            legacy_decision_count=int(value["legacy_decision_count"]),
            current_decision_count=int(value["current_decision_count"]),
            legacy_final_world_sha256=str(value["legacy_final_world_sha256"]),
            current_final_world_sha256=str(value["current_final_world_sha256"]),
            legacy_score_sha256=str(value["legacy_score_sha256"]),
            current_score_sha256=str(value["current_score_sha256"]),
        )
        declared = {
            "provider_requests_equal": row.provider_requests_equal,
            "final_world_equal": row.final_world_equal,
            "score_equal": row.score_equal,
            "semantically_equal": row.semantically_equal,
        }
        for field, expected in declared.items():
            if value[field] is not expected:
                raise BenchmarkPatchImpactError(f"trajectory {field} is inconsistent")
        return row


@dataclass(frozen=True)
class BenchmarkPatchImpactV1:
    """Frozen maintenance evidence for one bounded source patch."""

    patch_id: str
    benchmark_suite_version: str
    task_count_scanned: int
    changed_activation_task_ids: tuple[str, ...]
    trajectories: tuple[PatchTrajectoryDigestV1, ...]
    baseline_source_revision: str
    repaired_source_revision: str

    def __post_init__(self) -> None:
        if self.patch_id != B3_PATCH_ID:
            raise BenchmarkPatchImpactError("unsupported patch_id")
        if self.benchmark_suite_version != "CommerceWorld-v2.1":
            raise BenchmarkPatchImpactError("unexpected benchmark suite version")
        if self.task_count_scanned != 200:
            raise BenchmarkPatchImpactError("B3 impact audit must scan exactly 200 tasks")
        if self.changed_activation_task_ids != B3_EXPECTED_TASK_IDS:
            raise BenchmarkPatchImpactError("B3 activation scope is not the 14 T5 Buyer tasks")
        if len(self.trajectories) != 28:
            raise BenchmarkPatchImpactError("B3 audit requires 14 ideal and 14 mutation rows")
        expected_rows = {
            (task_id, variant)
            for task_id in B3_EXPECTED_TASK_IDS
            for variant in sorted(_VARIANTS)
        }
        actual_rows = {(row.task_id, row.variant) for row in self.trajectories}
        if actual_rows != expected_rows:
            raise BenchmarkPatchImpactError("B3 trajectory matrix is incomplete or duplicated")
        if not all(row.semantically_equal for row in self.trajectories):
            raise BenchmarkPatchImpactError("B3 changed provider requests, World, or score")
        for name in ("baseline_source_revision", "repaired_source_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) < 7:
                raise BenchmarkPatchImpactError(f"{name} must identify a source revision")
        if self.baseline_source_revision == self.repaired_source_revision:
            raise BenchmarkPatchImpactError("baseline and repaired revisions must differ")

    @property
    def impact_id(self) -> str:
        payload = self.to_dict(include_impact_id=False)
        return cast(str, canonical_sha256(payload))

    def to_dict(self, *, include_impact_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": BENCHMARK_PATCH_IMPACT_SCHEMA,
            "patch_id": self.patch_id,
            "benchmark_suite_version": self.benchmark_suite_version,
            "task_count_scanned": self.task_count_scanned,
            "changed_activation_task_ids": list(self.changed_activation_task_ids),
            "trajectory_count": len(self.trajectories),
            "trajectories": [row.to_dict() for row in self.trajectories],
            "all_provider_requests_equal": all(
                row.provider_requests_equal for row in self.trajectories
            ),
            "all_final_world_equal": all(row.final_world_equal for row in self.trajectories),
            "all_scores_equal": all(row.score_equal for row in self.trajectories),
            "baseline_source_revision": self.baseline_source_revision,
            "repaired_source_revision": self.repaired_source_revision,
            "paid_rerun_required": False,
            "frozen_campaign_artifacts_modified": False,
        }
        if include_impact_id:
            value["impact_id"] = self.impact_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkPatchImpactV1":
        _require_exact_keys(
            value,
            frozenset(
                {
                    "schema_version",
                    "patch_id",
                    "benchmark_suite_version",
                    "task_count_scanned",
                    "changed_activation_task_ids",
                    "trajectory_count",
                    "trajectories",
                    "all_provider_requests_equal",
                    "all_final_world_equal",
                    "all_scores_equal",
                    "baseline_source_revision",
                    "repaired_source_revision",
                    "paid_rerun_required",
                    "frozen_campaign_artifacts_modified",
                    "impact_id",
                }
            ),
            field="benchmark patch impact",
        )
        if value["schema_version"] != BENCHMARK_PATCH_IMPACT_SCHEMA:
            raise BenchmarkPatchImpactError("unsupported impact schema")
        raw_tasks = value["changed_activation_task_ids"]
        raw_rows = value["trajectories"]
        if not isinstance(raw_tasks, list) or not isinstance(raw_rows, list):
            raise BenchmarkPatchImpactError("impact arrays must be JSON arrays")
        report = cls(
            patch_id=str(value["patch_id"]),
            benchmark_suite_version=str(value["benchmark_suite_version"]),
            task_count_scanned=int(value["task_count_scanned"]),
            changed_activation_task_ids=tuple(map(str, raw_tasks)),
            trajectories=tuple(
                PatchTrajectoryDigestV1.from_dict(row)
                for row in raw_rows
                if isinstance(row, Mapping)
            ),
            baseline_source_revision=str(value["baseline_source_revision"]),
            repaired_source_revision=str(value["repaired_source_revision"]),
        )
        expected = report.to_dict()
        if dict(value) != expected:
            raise BenchmarkPatchImpactError("impact derived fields or impact_id are inconsistent")
        return report


class _CapturingDecisionChannel:
    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.requests: list[dict[str, Any]] = []

    def complete_business_decision(self, **kwargs: Any) -> Any:
        prompt = kwargs.get("user_prompt")
        if not isinstance(prompt, str):
            raise BenchmarkPatchImpactError("business request prompt is not text")
        start = prompt.find("{")
        if start < 0:
            raise BenchmarkPatchImpactError("business request contains no JSON object")
        request = json.loads(prompt[start:])
        if not isinstance(request, dict):
            raise BenchmarkPatchImpactError("provider request must be an object")
        self.requests.append(copy.deepcopy(request))
        return self._delegate.complete_business_decision(**kwargs)


def _legacy_cart_checkout_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """The exact pre-B3 cart activation behavior, isolated for impact audit."""

    from agents import skill_selector_buyer as selector

    kind = ctx.env.action.get("kind", "")
    if kind in {"platform.cart_quote", "platform.cart_settlement"}:
        return True, f"cart lifecycle reply ({kind})"
    payload = selector._payload(ctx)
    if kind == "delegate.create_purchase_mandate" and selector._contains_mapping_shape(
        payload, frozenset({"market_id", "lines"})
    ):
        return True, "mandate contains a specified multi-line market cart request"
    return False, f"no cart workflow trigger (kind={kind!r})"


def _with_cart_gate(gate: Gate, callback: Callable[[], _T]) -> _T:
    from agents import skill_selector_buyer as selector

    previous = selector._BUYER_GATES["cart-checkout"]
    selector._BUYER_GATES["cart-checkout"] = gate
    try:
        return callback()
    finally:
        selector._BUYER_GATES["cart-checkout"] = previous


def scan_b3_activation_scope() -> tuple[str, ...]:
    """Return task ids whose buyer kickoff cart activation changes under B3."""

    from agents import skill_selector_buyer as selector

    current_gate: Gate = selector._BUYER_GATES["cart-checkout"]
    changed: list[str] = []
    for task_id in TASK_REGISTRY_V2:
        bundle = runtime_bundle_v2(task_id)
        buyer_inbounds = tuple(
            envelope
            for envelope in kickoff_envelopes(bundle.scenario)
            if str(envelope.to).startswith("buyer:")
        )

        def selections(gate: Gate) -> tuple[bool, ...]:
            return _with_cart_gate(
                gate,
                lambda: tuple(
                    "cart-checkout"
                    in BuyerSkillSelector().select(envelope, memory=InMemoryStore())
                    for envelope in buyer_inbounds
                ),
            )

        if selections(_legacy_cart_checkout_gate) != selections(current_gate):
            changed.append(task_id)
    return tuple(changed)


def _run_t5_trajectory(
    *,
    task_id: str,
    variant: str,
    gate: Gate,
    out_root: Path,
) -> tuple[str, int, str, str]:
    bundle = runtime_bundle_v2(task_id)
    channel = (
        bundle.ideal_channel()
        if variant == "ideal"
        else bundle.mutations[0].channel()
    )
    capturing = _CapturingDecisionChannel(channel)
    channels_by_actor = {
        bundle.evaluated_actor_id: capturing,
        **{
            actor_id: factory()
            for actor_id, factory in bundle.counterpart_channels.items()
        },
    }

    def execute() -> tuple[str, int, str, str]:
        EpisodeBatch(
            scenarios=[bundle.scenario],
            channels=lambda agent_id, role: channels_by_actor[agent_id],
            out_root=out_root,
            remote_world=True,
            strict_tracker_capture=True,
        ).run()
        episode_dir = out_root / bundle.scenario.scenario_id
        if (episode_dir / "termination.json").exists():
            raise BenchmarkPatchImpactError(
                f"{task_id}/{variant} terminated under impact audit"
            )
        evidence = RuntimeEvidenceBundleV2.load(episode_dir)
        score = bundle.scorer(evidence)
        return (
            canonical_sha256(capturing.requests),
            len(capturing.requests),
            evidence.final_digest,
            canonical_sha256(score.to_dict()),
        )

    return _with_cart_gate(gate, execute)


def audit_b3_patch_impact(
    *,
    baseline_source_revision: str,
    repaired_source_revision: str,
) -> BenchmarkPatchImpactV1:
    """Run the bounded B3 activation and trajectory equivalence audit."""

    from agents import skill_selector_buyer as selector

    changed = scan_b3_activation_scope()
    if changed != B3_EXPECTED_TASK_IDS:
        raise BenchmarkPatchImpactError(
            f"B3 activation scope changed: expected {B3_EXPECTED_TASK_IDS!r}, got {changed!r}"
        )
    current_gate: Gate = selector._BUYER_GATES["cart-checkout"]
    rows: list[PatchTrajectoryDigestV1] = []
    with tempfile.TemporaryDirectory(prefix="cwenv-b3-impact-") as temp:
        root = Path(temp)
        for task_id in changed:
            for variant in ("ideal", "targeted_mutation"):
                legacy = _run_t5_trajectory(
                    task_id=task_id,
                    variant=variant,
                    gate=_legacy_cart_checkout_gate,
                    out_root=root / "legacy" / task_id / variant,
                )
                current = _run_t5_trajectory(
                    task_id=task_id,
                    variant=variant,
                    gate=current_gate,
                    out_root=root / "current" / task_id / variant,
                )
                rows.append(
                    PatchTrajectoryDigestV1(
                        task_id=task_id,
                        variant=variant,
                        legacy_provider_request_sha256=legacy[0],
                        current_provider_request_sha256=current[0],
                        legacy_decision_count=legacy[1],
                        current_decision_count=current[1],
                        legacy_final_world_sha256=legacy[2],
                        current_final_world_sha256=current[2],
                        legacy_score_sha256=legacy[3],
                        current_score_sha256=current[3],
                    )
                )
    return BenchmarkPatchImpactV1(
        patch_id=B3_PATCH_ID,
        benchmark_suite_version="CommerceWorld-v2.1",
        task_count_scanned=len(TASK_REGISTRY_V2),
        changed_activation_task_ids=changed,
        trajectories=tuple(rows),
        baseline_source_revision=baseline_source_revision,
        repaired_source_revision=repaired_source_revision,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.benchmark_patch_impact")
    parser.add_argument("--baseline-source-revision", required=True)
    parser.add_argument("--repaired-source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_b3_patch_impact(
        baseline_source_revision=args.baseline_source_revision,
        repaired_source_revision=args.repaired_source_revision,
    )
    write_json_artifact(report.to_dict(), args.output)
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "B3_EXPECTED_TASK_IDS",
    "B3_PATCH_ID",
    "BENCHMARK_PATCH_IMPACT_SCHEMA",
    "BenchmarkPatchImpactError",
    "BenchmarkPatchImpactV1",
    "PatchTrajectoryDigestV1",
    "audit_b3_patch_impact",
    "scan_b3_activation_scope",
]
