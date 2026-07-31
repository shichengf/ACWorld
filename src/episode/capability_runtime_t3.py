"""ACWorld T3 context and preference reasoning tasks.

All twenty problem instances are constructed here from the immutable v2 task
registry and its difficulty factors.  Every benchmark action is executed by a
real ``ScenarioSpec`` through Episode -> Runtime -> Platform -> World, and
every scored observation is recovered from a verified
``RuntimeEvidenceBundleV2``.

Candidate products are seeded as World catalog listings.  Trust and
authorization records use the dedicated World EvidenceRecord table, while
principal-authorized preference and constraint changes use the dedicated
MandateAuthority and MandateRevision tables.  Social tasks additionally seed
the CommerceWorld friendship/review tables.  The buyer must retrieve those core rows
with real World tools before its audited Platform selection can receive
credit.  Its reasoning conclusion is scored only from the Runtime-accepted
actor decision record emitted after Platform returns the authoritative match
certificate.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from agents.agent_phase import public_reference_alias_v1
from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1
from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2
from episode.capability_runtime import (
    COMMERCEWORLD_EPISODE_BACKEND_V2,
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceBundleV2,
    RuntimeEvidenceError,
    RuntimeMutationV2,
    RuntimeRubricCheckV2,
    RuntimeTaskBundleV2,
    RuntimeTaskScoreV3,
    canonical_sha256,
    renormalize_capability_checks_v2,
    require_runtime_benchmark_integrity_v2,
    score_checks,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.t3_provider_grounding import grounded_t3_facts_v1
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from protocol.evidence_records import (
    MandateRevision,
    MandateRevisionAuthority,
    build_evidence_record,
    build_mandate_revision,
    evidence_record_to_dict,
    mandate_revision_to_dict,
)
from runtime.authority_operation_evidence import (
    SEARCH_SESSION_EVIDENCE_CONTRACT,
    VerifiedSearchSessionEvidence,
)
from runtime.match_certificate_evidence import (
    MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
    VerifiedMatchCertificateEvidence,
)
from runtime.supply_fulfillment_evidence import (
    SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
    VerifiedSupplyFulfillmentEvidence,
    verified_settlement_order,
)
from runtime.tracker_evidence import (
    TrackerEvidenceError,
    VerifiedModelBusinessChoice,
    VerifiedModelWorldRead,
    tracker_row_has_usable_completed_steps,
    verified_model_business_choices,
    verified_model_world_reads,
)
from world.evidence_contracts import mandate_authority_to_wire


T3_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t3.v2"
T3_RUNTIME_DATA_SCHEMA_V2 = "cwe.runtime-task-data.t3.v2"
_BUYER_ID = "buyer:t3-benchmark"
_MERCHANT_ID = "merchant:t3-benchmark"
_CANDIDATE_CATEGORY = "decisionproduct"
_EVIDENCE_ISSUER = "platform:evidence"
_TRUST_THRESHOLD_BPS = 5_000
_DECISION_DETAIL_FIELDS = (
    "considered_listing_ids",
    "used_signal_ids",
    "applied_update_ids",
    "governing_instruction_ids",
    "discarded_input_ids",
)
_T3_TASK_IDS = tuple(
    task_id for task_id, definition in TASK_REGISTRY_V2.items() if definition.family.value == "T3"
)


Scalar = str | int | bool
FeaturePairs = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _T3AuthorityPath:
    linked_response_ids: frozenset[str]
    linked_certificate_ids: frozenset[str]
    linked_order_ids: frozenset[str]
    linked_sku_ids: frozenset[str]
    preclaimed_commit_ids: tuple[str, ...]
    search_prefix_verified: bool
    search_detail: Mapping[str, Any]


@dataclass(frozen=True)
class _T3TerminalOutcome:
    credit: float
    linked_response_ids: frozenset[str]
    detail: Mapping[str, Any]


def _pairs(values: FeaturePairs) -> dict[str, int]:
    return dict(values)


@dataclass(frozen=True)
class T3Constraint:
    field: str
    operator: str
    value: Scalar

    def to_dict(self) -> dict[str, Scalar]:
        return {"field": self.field, "operator": self.operator, "value": self.value}

    def accepts(self, listing: "T3Listing") -> bool:
        attributes = dict(listing.attributes)
        if self.field not in attributes:
            return False
        observed = attributes[self.field]
        if self.operator == "eq":
            return observed == self.value
        if self.operator == "ne":
            return observed != self.value
        if self.operator == "le":
            return (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and isinstance(self.value, int)
                and not isinstance(self.value, bool)
                and observed <= self.value
            )
        if self.operator == "ge":
            return (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and isinstance(self.value, int)
                and not isinstance(self.value, bool)
                and observed >= self.value
            )
        raise ValueError(f"unsupported T3 constraint operator {self.operator!r}")


@dataclass(frozen=True)
class T3Listing:
    listing_id: str
    attributes: tuple[tuple[str, Scalar], ...]
    preference_values: FeaturePairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "attributes": dict(self.attributes),
            "preference_values": _pairs(self.preference_values),
        }


@dataclass(frozen=True)
class T3SocialSignal:
    signal_id: str
    advisor_id: str
    listing_id: str
    rating: int
    trust_record_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "advisor_id": self.advisor_id,
            "listing_id": self.listing_id,
            "rating": self.rating,
            "trust_record_id": self.trust_record_id,
        }


@dataclass(frozen=True)
class T3PreferenceUpdate:
    update_id: str
    issuer_id: str
    revision: int
    preference_weights: FeaturePairs
    credential_record_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "issuer_id": self.issuer_id,
            "revision": self.revision,
            "preference_weights": _pairs(self.preference_weights),
            "credential_record_id": self.credential_record_id,
        }


@dataclass(frozen=True)
class T3AuthorityInstruction:
    instruction_id: str
    issuer_id: str
    constraint: T3Constraint
    credential_record_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "issuer_id": self.issuer_id,
            "constraint": self.constraint.to_dict(),
            "credential_record_id": self.credential_record_id,
        }


@dataclass(frozen=True)
class T3CorpusRecord:
    record_id: str
    record_kind: str
    subject_id: str
    attributes: tuple[tuple[str, Scalar], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_kind": self.record_kind,
            "subject_id": self.subject_id,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class T3HiddenCorpus:
    records: tuple[T3CorpusRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}

    def record(self, record_id: str) -> T3CorpusRecord:
        for record in self.records:
            if record.record_id == record_id:
                return record
        raise KeyError(record_id)


@dataclass(frozen=True)
class T3PublicTask:
    directive: str
    hard_constraints: tuple[T3Constraint, ...]
    initial_preference_weights: FeaturePairs
    listings: tuple[T3Listing, ...]
    social_signals: tuple[T3SocialSignal, ...] = ()
    preference_updates: tuple[T3PreferenceUpdate, ...] = ()
    authority_instructions: tuple[T3AuthorityInstruction, ...] = ()
    available_record_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "directive": self.directive,
            "hard_constraints": [row.to_dict() for row in self.hard_constraints],
            "initial_preference_weights": _pairs(self.initial_preference_weights),
            "listings": [row.to_dict() for row in self.listings],
            "social_signals": [row.to_dict() for row in self.social_signals],
            "preference_updates": [row.to_dict() for row in self.preference_updates],
            "authority_instructions": [row.to_dict() for row in self.authority_instructions],
            "available_record_ids": list(self.available_record_ids),
        }


@dataclass(frozen=True)
class T3ExpectedContract:
    oracle_id: str
    selected_listing_id: str
    feasible_listing_ids: tuple[str, ...]
    relevant_record_ids: tuple[str, ...]
    used_signal_ids: tuple[str, ...]
    applied_update_ids: tuple[str, ...]
    governing_instruction_ids: tuple[str, ...]
    discarded_input_ids: tuple[str, ...]
    effective_preference_weights: FeaturePairs
    effective_listing_scores: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "selected_listing_id": self.selected_listing_id,
            "feasible_listing_ids": list(self.feasible_listing_ids),
            "relevant_record_ids": list(self.relevant_record_ids),
            "used_signal_ids": list(self.used_signal_ids),
            "applied_update_ids": list(self.applied_update_ids),
            "governing_instruction_ids": list(self.governing_instruction_ids),
            "discarded_input_ids": list(self.discarded_input_ids),
            "effective_preference_weights": _pairs(self.effective_preference_weights),
            "effective_listing_scores": dict(self.effective_listing_scores),
        }


@dataclass(frozen=True)
class T3RuntimeCase:
    task: TaskDefinitionV2
    public_task: T3PublicTask
    hidden_corpus: T3HiddenCorpus
    expected: T3ExpectedContract
    schema_version: str = T3_RUNTIME_DATA_SCHEMA_V2

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "registry_binding": self.task.to_dict(),
                "public_task": self.public_task.to_dict(),
                "authoritative_corpus": self.hidden_corpus.to_dict(),
                "expected_contract": self.expected.to_dict(),
            }
        )


def _attributes(**values: Scalar) -> tuple[tuple[str, Scalar], ...]:
    return tuple(sorted(values.items()))


def _features(**values: int) -> FeaturePairs:
    return tuple(sorted(values.items()))


def _factor(task: TaskDefinitionV2, name: str) -> int:
    factors = dict(task.difficulty_factors)
    if set(factors) != {"difficulty_level", name}:
        raise ValueError(f"{task.task_id} has unexpected T3 factors {sorted(factors)!r}")
    value = factors[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{task.task_id} factor {name!r} must be an integer")
    return value


def _base_constraint() -> tuple[T3Constraint, ...]:
    return (T3Constraint("price_cents", "le", 12_000),)


def _weighted_problem(
    task: TaskDefinitionV2,
) -> tuple[T3PublicTask, T3HiddenCorpus]:
    count = _factor(task, "soft_preference_count")
    names = (
        "quality",
        "comfort",
        "durability",
        "efficiency",
        "support",
        "portability",
    )[:count]
    weights = tuple((name, index + 1) for index, name in enumerate(names))
    listings = [
        T3Listing(
            "listing-01",
            _attributes(price_cents=10_000),
            tuple((name, 85) for name in names),
        )
    ]
    for index, featured in enumerate(names, start=2):
        values = tuple((name, 100 if name == featured else 45) for name in names)
        listings.append(
            T3Listing(
                f"listing-{index:02d}",
                _attributes(price_cents=9_000 + index),
                values,
            )
        )
    listings.append(
        T3Listing(
            f"listing-{len(listings) + 1:02d}",
            _attributes(price_cents=8_500),
            tuple((name, 30) for name in names),
        )
    )
    return (
        T3PublicTask(
            directive=("Choose the feasible listing with the highest weighted preference value."),
            hard_constraints=_base_constraint(),
            initial_preference_weights=weights,
            listings=tuple(listings),
        ),
        T3HiddenCorpus(()),
    )


def _hard_over_soft_problem(
    task: TaskDefinitionV2,
) -> tuple[T3PublicTask, T3HiddenCorpus]:
    count = _factor(task, "soft_conflict_count")
    constraints = list(_base_constraint())
    constraints.extend(
        T3Constraint(f"requirement_{index}", "eq", True) for index in range(1, count + 1)
    )
    safe = {f"requirement_{index}": True for index in range(1, count + 1)}
    listings = [
        T3Listing(
            "listing-01",
            _attributes(price_cents=10_000, **safe),
            _features(desirability=70),
        ),
        T3Listing(
            "listing-02",
            _attributes(price_cents=9_500, **safe),
            _features(desirability=50),
        ),
    ]
    for index in range(1, count + 1):
        attrs = dict(safe)
        attrs[f"requirement_{index}"] = False
        listings.append(
            T3Listing(
                f"listing-{index + 2:02d}",
                _attributes(price_cents=8_000 + index, **attrs),
                _features(desirability=101 - index),
            )
        )
    return (
        T3PublicTask(
            directive=("Filter by every hard mandate condition before comparing preference value."),
            hard_constraints=tuple(constraints),
            initial_preference_weights=_features(desirability=1),
            listings=tuple(listings),
        ),
        T3HiddenCorpus(()),
    )


def _social_problem(
    task: TaskDefinitionV2,
) -> tuple[T3PublicTask, T3HiddenCorpus]:
    count = _factor(task, "advisor_count")
    listings = (
        T3Listing("listing-01", _attributes(price_cents=10_000), _features(fit=50)),
        T3Listing("listing-02", _attributes(price_cents=10_000), _features(fit=60)),
        T3Listing("listing-03", _attributes(price_cents=9_000), _features(fit=30)),
    )
    signals: list[T3SocialSignal] = []
    records: list[T3CorpusRecord] = []
    for index in range(1, count + 1):
        advisor = f"advisor-{index:02d}"
        record_id = f"record-{index:02d}"
        signals.append(T3SocialSignal(f"signal-{index:02d}", advisor, "listing-01", 5, record_id))
        records.append(
            T3CorpusRecord(
                record_id,
                "source_trust",
                advisor,
                _attributes(verified=True, trust_bps=8_000 + index * 100),
            )
        )
    return (
        T3PublicTask(
            directive=(
                "Combine product fit with verified social evidence and choose one feasible listing."
            ),
            hard_constraints=_base_constraint(),
            initial_preference_weights=_features(fit=1),
            listings=listings,
            social_signals=tuple(signals),
            available_record_ids=tuple(record.record_id for record in records),
        ),
        T3HiddenCorpus(tuple(records)),
    )


def _noisy_problem(
    task: TaskDefinitionV2,
) -> tuple[T3PublicTask, T3HiddenCorpus]:
    count = _factor(task, "unreliable_advisor_count")
    listings = (
        T3Listing("listing-01", _attributes(price_cents=10_000), _features(fit=50)),
        T3Listing("listing-02", _attributes(price_cents=10_000), _features(fit=70)),
        T3Listing("listing-03", _attributes(price_cents=9_000), _features(fit=20)),
    )
    signals = [T3SocialSignal("signal-01", "advisor-01", "listing-01", 5, "record-01")]
    records = [
        T3CorpusRecord(
            "record-01",
            "source_trust",
            "advisor-01",
            _attributes(verified=True, trust_bps=9_000),
        )
    ]
    for index in range(1, count + 1):
        ordinal = index + 1
        advisor = f"advisor-{ordinal:02d}"
        record_id = f"record-{ordinal:02d}"
        signals.append(T3SocialSignal(f"signal-{ordinal:02d}", advisor, "listing-02", 5, record_id))
        records.append(
            T3CorpusRecord(
                record_id,
                "source_trust",
                advisor,
                _attributes(verified=False, trust_bps=500),
            )
        )
    return (
        T3PublicTask(
            directive=(
                "Resolve conflicting social claims using authoritative source-trust records."
            ),
            hard_constraints=_base_constraint(),
            initial_preference_weights=_features(fit=1),
            listings=listings,
            social_signals=tuple(signals),
            available_record_ids=tuple(record.record_id for record in records),
        ),
        T3HiddenCorpus(tuple(records)),
    )


def _update_problem(
    task: TaskDefinitionV2,
) -> tuple[T3PublicTask, T3HiddenCorpus]:
    count = _factor(task, "preference_revision_count")
    listings = (
        T3Listing(
            "listing-01",
            _attributes(price_cents=10_000),
            _features(portability=0, quality=100),
        ),
        T3Listing(
            "listing-02",
            _attributes(price_cents=10_000),
            _features(portability=100, quality=0),
        ),
        T3Listing(
            "listing-03",
            _attributes(price_cents=9_000),
            _features(portability=45, quality=45),
        ),
    )
    updates: list[T3PreferenceUpdate] = []
    records: list[T3CorpusRecord] = []
    for revision in range(1, count + 1):
        weights = (
            _features(portability=6 + revision, quality=1)
            if revision % 2
            else _features(portability=1, quality=6 + revision)
        )
        record_id = f"credential-{revision:02d}"
        updates.append(
            T3PreferenceUpdate(
                f"update-{revision:02d}",
                "principal",
                revision,
                weights,
                record_id,
            )
        )
        records.append(
            T3CorpusRecord(
                record_id,
                "authorization",
                "principal",
                _attributes(authorized=True, scope="preferences"),
            )
        )
    final_weights = _pairs(updates[-1].preference_weights)
    lure_weights = (
        _features(portability=1, quality=100)
        if final_weights["portability"] > final_weights["quality"]
        else _features(portability=100, quality=1)
    )
    updates.append(
        T3PreferenceUpdate("update-x", "merchant:m1", count + 1, lure_weights, "credential-x")
    )
    records.append(
        T3CorpusRecord(
            "credential-x",
            "authorization",
            "merchant:m1",
            _attributes(authorized=False, scope="none"),
        )
    )
    return (
        T3PublicTask(
            directive=("Apply revisions only when the issuer is authorized for buyer preferences."),
            hard_constraints=_base_constraint(),
            initial_preference_weights=_features(portability=1, quality=5),
            listings=listings,
            preference_updates=tuple(updates),
            available_record_ids=tuple(record.record_id for record in records),
        ),
        T3HiddenCorpus(tuple(records)),
    )


def _authority_problem(
    task: TaskDefinitionV2,
) -> tuple[T3PublicTask, T3HiddenCorpus]:
    count = _factor(task, "authority_layer_count")
    listings = (
        T3Listing(
            "listing-01",
            _attributes(price_cents=10_000, region="north"),
            _features(appeal=50),
        ),
        T3Listing(
            "listing-02",
            _attributes(price_cents=10_000, region="south"),
            _features(appeal=100),
        ),
        T3Listing(
            "listing-03",
            _attributes(price_cents=9_000, region="north"),
            _features(appeal=30),
        ),
    )
    instructions: list[T3AuthorityInstruction] = []
    records: list[T3CorpusRecord] = []
    issuers = ("principal", "delegate", "profile", "advisor")
    for rank in range(1, count + 1):
        issuer = issuers[rank - 1]
        record_id = f"authority-{rank:02d}"
        region = "north" if rank == 1 else "south"
        instructions.append(
            T3AuthorityInstruction(
                f"instruction-{rank:02d}",
                issuer,
                T3Constraint("region", "eq", region),
                record_id,
            )
        )
        records.append(
            T3CorpusRecord(
                record_id,
                "authority",
                issuer,
                _attributes(
                    authorized=True,
                    authority_rank=rank,
                    scope="listing_region",
                ),
            )
        )
    return (
        T3PublicTask(
            directive=(
                "Resolve conflicting instructions through authoritative rank "
                "records, then preserve the governing mandate."
            ),
            hard_constraints=_base_constraint(),
            initial_preference_weights=_features(appeal=1),
            listings=listings,
            authority_instructions=tuple(instructions),
            available_record_ids=tuple(record.record_id for record in records),
        ),
        T3HiddenCorpus(tuple(records)),
    )


_TASK_BUILDERS: Mapping[
    str,
    Callable[[TaskDefinitionV2], tuple[T3PublicTask, T3HiddenCorpus]],
] = {
    "t3.weighted_soft_preferences": _weighted_problem,
    "t3.hard_over_soft": _hard_over_soft_problem,
    "t3.social_evidence": _social_problem,
    "t3.noisy_social_signals": _noisy_problem,
    "t3.preference_update": _update_problem,
    "t3.mandate_consistency": _authority_problem,
}


def _record_values(corpus: T3HiddenCorpus, record_id: str) -> dict[str, Scalar]:
    return dict(corpus.record(record_id).attributes)


def _effective_contract(
    task: TaskDefinitionV2,
    public: T3PublicTask,
    corpus: T3HiddenCorpus,
) -> T3ExpectedContract:
    update_order = {
        row.update_id: f"preference-step-{index:02d}"
        for index, row in enumerate(public.preference_updates, start=1)
    }
    instruction_order = {
        row.instruction_id: f"authority-step-{index:02d}"
        for index, row in enumerate(public.authority_instructions, start=1)
    }
    # The published business reference is the stable tie-break.  It must not
    # depend on the presentation position: otherwise merely reordering the
    # same candidates can silently change the oracle.
    listing_order = {row.listing_id: row.listing_id for row in public.listings}
    authorized_updates = tuple(
        sorted(
            (
                update
                for update in public.preference_updates
                if _record_values(corpus, update.credential_record_id).get("authorized") is True
                and _record_values(corpus, update.credential_record_id).get("scope")
                == "preferences"
            ),
            key=lambda update: (update.revision, update_order[update.update_id]),
        )
    )
    weights = (
        authorized_updates[-1].preference_weights
        if authorized_updates
        else public.initial_preference_weights
    )

    by_field: dict[str, list[tuple[int, T3AuthorityInstruction]]] = {}
    for instruction in public.authority_instructions:
        attrs = _record_values(corpus, instruction.credential_record_id)
        rank = attrs.get("authority_rank")
        if attrs.get("authorized") is True and isinstance(rank, int) and not isinstance(rank, bool):
            by_field.setdefault(instruction.constraint.field, []).append((rank, instruction))
    governing = tuple(
        min(
            rows,
            key=lambda row: (row[0], instruction_order[row[1].instruction_id]),
        )[1]
        for _, rows in sorted(by_field.items())
    )
    constraints = (
        *public.hard_constraints,
        *(instruction.constraint for instruction in governing),
    )
    feasible = tuple(
        listing.listing_id
        for listing in public.listings
        if all(rule.accepts(listing) for rule in constraints)
    )
    if not feasible:
        raise ValueError("T3 runtime task has no feasible listing")

    trusted = tuple(
        signal
        for signal in public.social_signals
        if _record_values(corpus, signal.trust_record_id).get("verified") is True
        and isinstance(_record_values(corpus, signal.trust_record_id).get("trust_bps"), int)
        and not isinstance(_record_values(corpus, signal.trust_record_id).get("trust_bps"), bool)
        and int(_record_values(corpus, signal.trust_record_id)["trust_bps"]) >= _TRUST_THRESHOLD_BPS
    )
    scores: dict[str, int] = {}
    for listing in public.listings:
        values = _pairs(listing.preference_values)
        score = sum(weight * values.get(name, 0) for name, weight in weights)
        for signal in trusted:
            if signal.listing_id == listing.listing_id:
                trust = _record_values(corpus, signal.trust_record_id)["trust_bps"]
                assert isinstance(trust, int) and not isinstance(trust, bool)
                score += signal.rating * trust // 1_000
        scores[listing.listing_id] = score
    selected = min(
        feasible,
        key=lambda listing_id: (-scores[listing_id], listing_order[listing_id]),
    )

    used_signal_ids = tuple(signal.signal_id for signal in trusted)
    applied_update_ids = tuple(update.update_id for update in authorized_updates)
    governing_ids = tuple(row.instruction_id for row in governing)
    discarded = (
        *(row.signal_id for row in public.social_signals if row.signal_id not in used_signal_ids),
        *(
            row.update_id
            for row in public.preference_updates
            if row.update_id not in applied_update_ids
        ),
        *(
            row.instruction_id
            for row in public.authority_instructions
            if row.instruction_id not in governing_ids
        ),
    )
    return T3ExpectedContract(
        oracle_id=task.capability_id,
        selected_listing_id=selected,
        feasible_listing_ids=feasible,
        relevant_record_ids=public.available_record_ids,
        used_signal_ids=used_signal_ids,
        applied_update_ids=applied_update_ids,
        governing_instruction_ids=governing_ids,
        discarded_input_ids=discarded,
        effective_preference_weights=weights,
        effective_listing_scores=tuple(
            (listing.listing_id, scores[listing.listing_id]) for listing in public.listings
        ),
    )


def runtime_case_t3(task_id: str) -> T3RuntimeCase:
    try:
        task = TASK_REGISTRY_V2[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown Benchmark v2 task {task_id!r}") from exc
    if task.family.value != "T3" or task.evaluated_role != "buyer":
        raise ValueError(f"{task_id} is not a buyer-side T3 task")
    try:
        public, corpus = _TASK_BUILDERS[task.capability_id](task)
    except KeyError as exc:
        raise ValueError(f"no runtime T3 builder for {task.capability_id!r}") from exc
    # Balance the location of the correct candidate without changing any
    # candidate facts or the public objective.  Position is intentionally a
    # presentation property, never an oracle input.
    baseline = _effective_contract(task, public, corpus)
    task_ordinal = int(task.task_id.rsplit("-", 1)[1])
    target_position = (task_ordinal - 1) % min(3, len(public.listings))
    target_listing_id = f"listing-{target_position + 1:02d}"
    if baseline.selected_listing_id != target_listing_id:
        if target_listing_id not in {row.listing_id for row in public.listings}:
            raise ValueError(f"{task.task_id}: balanced answer identity is unavailable")
        identity_map = {
            baseline.selected_listing_id: target_listing_id,
            target_listing_id: baseline.selected_listing_id,
        }
        public = replace(
            public,
            listings=tuple(
                replace(
                    row,
                    listing_id=identity_map.get(row.listing_id, row.listing_id),
                )
                for row in public.listings
            ),
            social_signals=tuple(
                replace(
                    row,
                    listing_id=identity_map.get(row.listing_id, row.listing_id),
                )
                for row in public.social_signals
            ),
        )
        baseline = _effective_contract(task, public, corpus)
        if baseline.selected_listing_id != target_listing_id:
            raise ValueError(f"{task.task_id}: answer identity balancing changed the oracle")
    listings = list(public.listings)
    selected = next(row for row in listings if row.listing_id == baseline.selected_listing_id)
    listings.remove(selected)
    listings.insert(target_position, selected)
    public = replace(public, listings=tuple(listings))
    return T3RuntimeCase(
        task=task,
        public_task=public,
        hidden_corpus=corpus,
        expected=_effective_contract(task, public, corpus),
    )


def _compatibility_seed(case: T3RuntimeCase) -> int:
    return int(case.content_sha256[:8], 16) % 2_147_483_646 + 1


def _constraint_payload(rule: T3Constraint) -> dict[str, Any]:
    return {
        "field": rule.field,
        "operator": rule.operator,
        "value": rule.value,
    }


def _visible_constraint_payload(rule: T3Constraint) -> dict[str, Any]:
    """Avoid copying the signed private budget into a public prompt block."""

    if rule.field == "price_cents":
        return {
            "field": "private_budget_ceiling",
            "operator": rule.operator,
            "description": "price must respect the signed private budget",
        }
    return _constraint_payload(rule)


def _candidate_attributes(listing: T3Listing) -> dict[str, Any]:
    attributes = dict(listing.attributes)
    return {
        "listing_id": listing.listing_id,
        "hard_attributes": attributes,
        "preference_values": dict(listing.preference_values),
        "price_cents": int(attributes["price_cents"]),
        "shipping_days": 1,
        **attributes,
    }


def _candidate_product_id(case: T3RuntimeCase, listing: T3Listing) -> str:
    """Return the frozen internal catalog identity used by World validation."""

    return f"benchmark-product:{case.task.task_id}:{listing.listing_id}"


def _candidate_row(case: T3RuntimeCase, listing: T3Listing) -> dict[str, Any]:
    price_cents = int(dict(listing.attributes)["price_cents"])
    return {
        "sku_id": listing.listing_id,
        "product_id": _candidate_product_id(case, listing),
        "merchant_id": _MERCHANT_ID,
        "category": _CANDIDATE_CATEGORY,
        "name": f"Catalog item {listing.listing_id}",
        "list_price": str(Decimal(price_cents) / Decimal(100)),
        "inventory": 3,
        "attributes": _candidate_attributes(listing),
    }


def _evidence_rows(case: T3RuntimeCase) -> list[dict[str, Any]]:
    """Build validated initial World evidence, never catalog pseudo-products."""

    return [
        evidence_record_to_dict(
            build_evidence_record(
                record_id=record.record_id,
                kind=record.record_kind,
                subject_id=record.subject_id,
                issuer_id=_EVIDENCE_ISSUER,
                facts=dict(record.attributes),
                trust={
                    "registry": "commerceworld",
                    "task_id": case.task.task_id,
                },
                version=1,
                owner_id=_BUYER_ID,
                read_acl=(_BUYER_ID,),
                issued_at_tick=0,
            )
        )
        for record in case.hidden_corpus.records
    ]


def _mandate_state(
    case: T3RuntimeCase,
) -> tuple[MandateRevisionAuthority | None, tuple[MandateRevision, ...]]:
    """Materialize only principal-authorized changes as World mandate state.

    Public candidate updates and instructions remain task inputs.  World
    records which ones actually carry principal authority.  An untrusted
    merchant proposal therefore never becomes a MandateRevision merely
    because the scenario or reference policy knows its expected disposition.
    """

    capability = case.task.capability_id
    if capability == "t3.preference_update":
        authorized = tuple(
            update
            for update in case.public_task.preference_updates
            if update.issuer_id == "principal"
        )
        authority = MandateRevisionAuthority(
            principal_id="principal",
            buyer_id=_BUYER_ID,
            mandate_id=case.task.task_id,
            allowed_fields=("preferences",),
        )
        revisions: list[MandateRevision] = []
        previous_digest: str | None = None
        for revision_number, update in enumerate(authorized, start=1):
            revision = build_mandate_revision(
                principal_id=authority.principal_id,
                buyer_id=_BUYER_ID,
                mandate_id=case.task.task_id,
                revision=revision_number,
                previous_digest=previous_digest,
                changes={
                    "preferences": {
                        "source_update_id": update.update_id,
                        "weights": dict(update.preference_weights),
                    }
                },
                authorized_fields=("preferences",),
                logical_tick=revision_number,
            )
            revisions.append(revision)
            previous_digest = revision.revision_digest
        return authority, tuple(revisions)

    if capability == "t3.mandate_consistency":
        principal_instruction = next(
            instruction
            for instruction in case.public_task.authority_instructions
            if instruction.issuer_id == "principal"
        )
        authority = MandateRevisionAuthority(
            principal_id="principal",
            buyer_id=_BUYER_ID,
            mandate_id=case.task.task_id,
            allowed_fields=("hard_constraints",),
        )
        revision = build_mandate_revision(
            principal_id=authority.principal_id,
            buyer_id=_BUYER_ID,
            mandate_id=case.task.task_id,
            revision=1,
            previous_digest=None,
            changes={
                "hard_constraints": {
                    "source_instruction_id": principal_instruction.instruction_id,
                    "constraint": principal_instruction.constraint.to_dict(),
                }
            },
            authorized_fields=("hard_constraints",),
            logical_tick=1,
        )
        return authority, (revision,)

    return None, ()


def _budget_for(case: T3RuntimeCase) -> int:
    caps = [
        int(rule.value)
        for rule in case.public_task.hard_constraints
        if rule.field == "price_cents"
        and rule.operator == "le"
        and isinstance(rule.value, int)
        and not isinstance(rule.value, bool)
    ]
    if caps:
        return min(caps)
    return max(
        int(dict(listing.attributes)["price_cents"]) for listing in case.public_task.listings
    )


def _benchmark_contract(case: T3RuntimeCase) -> dict[str, Any]:
    public = case.public_task
    required_read_intents: list[dict[str, Any]] = [
        {
            "intent": "observe_listing",
            "arguments": {"sku_ref": "one candidate_listing_ref"},
            "repeat": len(public.listings),
            "completion": "one observed listing for every candidate_listing_ref",
        }
    ]
    if public.available_record_ids:
        required_read_intents.append(
            {
                "intent": "observe_evidence_record",
                "arguments": {"record_ref": "one available_record_ref"},
                "repeat": len(public.available_record_ids),
                "completion": "one observed evidence record for every available_record_ref",
            }
        )
    if public.preference_updates or public.authority_instructions:
        required_read_intents.append(
            {
                "intent": "observe_mandate_revisions",
                "arguments": {},
                "repeat": 1,
                "completion": "the current mandate revision history is observed",
            }
        )
    reviewed_listing_ids = tuple(
        dict.fromkeys(row.listing_id for row in public.social_signals)
    )
    if reviewed_listing_ids:
        required_read_intents.append(
            {
                "intent": "observe_review_evidence",
                "arguments": {"sku_ref": "one listing referenced by a social signal"},
                "repeat": len(reviewed_listing_ids),
                "completion": "review evidence is observed for every referenced listing",
            }
        )
    social_grounding = (
        {
            "social_grounding": (
                "choose observe_review_evidence for every listing referenced by a "
                "social signal before selecting an offer"
            )
        }
        if public.social_signals
        else {}
    )
    return {
        "schema_version": T3_RUNTIME_SCHEMA_V2,
        "task_id": case.task.task_id,
        "directive": public.directive,
        "candidate_search": {
            "scope": "complete_candidate_set",
            "filters": "forbidden",
        },
        "hard_constraints": [_visible_constraint_payload(rule) for rule in public.hard_constraints],
        "initial_preference_weights": dict(public.initial_preference_weights),
        "decision_policy": {
            "rule_set": "preference-selection-v1",
            "feasibility_rule": "all published hard constraints must pass",
            "preference_score": "sum(weight * feature_value); missing features contribute 0",
            "normalization": "integer feature values and weights are used as published; no rescaling",
            "social_trust_rule": (
                "use a signal only when its source record has verified=true and "
                "trust_bps at least social_trust_minimum_bps"
            ),
            "social_trust_minimum_bps": _TRUST_THRESHOLD_BPS,
            "social_bonus": "rating * trust_bps // 1000",
            "preference_update_rule": (
                "apply authorized scope=preferences updates in ascending revision and "
                "stable_update_key order; the last applied update supplies the weights"
            ),
            "authority_rule": (
                "for each constrained field use the authorized instruction with the "
                "lowest authority_rank, breaking equal ranks by stable_instruction_key"
            ),
            "selection_rule": (
                "maximize preference score among feasible candidates, breaking equal "
                "scores by ascending stable_tie_break_key"
            ),
        },
        "model_action_protocol": {
            "interface": "choose exactly one allowed business intent per model turn",
            "response_mode": "strict_json_only",
            "required_before_accept_ranked_offer": required_read_intents,
            "instruction": (
                "Search the complete candidate set without filters. Complete every required "
                "read above and confirm its observed_business_facts before choosing "
                "accept_ranked_offer. Apply hard constraints only after grounding every "
                "candidate. Use the model-visible intent names and choose only from the "
                "allowed business intents shown for the current turn. Your entire response "
                "must be exactly the single JSON object required by response_contract, with "
                "no analysis, prose, or Markdown before or after it."
            ),
        },
        # This compatibility fallback is intentionally task-local.  Some
        # providers still wrap one otherwise valid JSON response in prose even
        # under a strict JSON-only instruction.  The runtime accepts only one
        # unambiguous fenced object and preserves the raw response digest.
        "response_extraction_mode": "single_fenced_json_fallback",
        "provider_visibility": {
            phase_id: {
                "exclude_fields": [
                    "candidate_facts",
                    "source_records",
                    "social_signals",
                    "preference_updates",
                    "authority_instructions",
                ]
            }
            for phase_id in ("buyer_discovery", "buyer_selection")
        },
        # These are public business inputs, not server answers.  They duplicate
        # the values that the actor grounds through World reads so a later
        # stateless settlement turn remains independently solvable.
        "candidate_facts": [
            {
                **row.to_dict(),
                "stable_tie_break_key": row.listing_id,
            }
            for row in public.listings
        ],
        "source_records": [row.to_dict() for row in case.hidden_corpus.records],
        "candidate_listing_ids": [row.listing_id for row in public.listings],
        "social_signals": [row.to_dict() for row in public.social_signals],
        "social_signal_ids": [row.signal_id for row in public.social_signals],
        "social_signal_targets": [
            {
                "signal_id": row.signal_id,
                "listing_id": row.listing_id,
            }
            for row in public.social_signals
        ],
        "preference_updates": [
            {
                **row.to_dict(),
                "update_sequence": row.revision,
                "stable_update_key": f"preference-step-{index:02d}",
            }
            for index, row in enumerate(public.preference_updates, start=1)
        ],
        "preference_update_ids": [row.update_id for row in public.preference_updates],
        "authority_instructions": [
            {
                **row.to_dict(),
                "stable_instruction_key": f"authority-step-{index:02d}",
            }
            for index, row in enumerate(public.authority_instructions, start=1)
        ],
        "authority_instruction_ids": [row.instruction_id for row in public.authority_instructions],
        "available_record_ids": list(public.available_record_ids),
        "execution_contract": {
            "candidate_discovery": "commerce.search via platform:aggregator",
            "candidate_grounding": (
                "choose observe_listing once for every candidate_listing_ref"
            ),
            "record_grounding": (
                "choose observe_evidence_record once for every available_record_ref"
            ),
            "mandate_grounding": (
                "choose observe_mandate_revisions when preference updates or authority "
                "instructions are present"
            ),
            **social_grounding,
            "terminal_selection": "commerce.accept_offer to platform:aggregator",
            "terminal_transaction": ("settle the certificate-bound order through platform:psp"),
            "decision_evidence": {
                "trigger": "authoritative platform.settlement_receipt response",
                "endpoint": "runtime:evidence",
                "action_kind": "commerce.submit_decision_record",
                "compact_payload_fields": ["outcome", "summary", "details"],
                "details_fields": list(_DECISION_DETAIL_FIELDS),
            },
        },
    }


def _decision_payload_schema() -> dict[str, Any]:
    """Describe the public T3 conclusion shape without answer values."""

    descriptions = {
        "considered_listing_ids": (
            "Every candidate listing id compared using authoritative World facts."
        ),
        "used_signal_ids": ("Verified social signal ids that affected the final decision."),
        "applied_update_ids": (
            "Principal-authorized preference update ids applied to the decision."
        ),
        "governing_instruction_ids": (
            "Applicable authority instruction ids that governed the decision."
        ),
        "discarded_input_ids": (
            "Untrusted, unauthorized, stale, or superseded input ids that were discarded."
        ),
    }
    properties = {
        field: {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
            "description": description,
        }
        for field, description in descriptions.items()
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(_DECISION_DETAIL_FIELDS),
        "additionalProperties": False,
    }


def _public_task_context(case: T3RuntimeCase) -> dict[str, Any]:
    """Return the value-free phase contract consumed by the local Agent."""

    return {
        "schema_version": T3_RUNTIME_SCHEMA_V2,
        "task_id": case.task.task_id,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": [
                {
                    "phase_id": "buyer_discovery",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["delegate.create_purchase_mandate"],
                        "inbound_sender_roles": ["consumer"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.search",
                            "destination": "platform:aggregator",
                        }
                    ],
                    "world_reads": "deny",
                    "finish": "forbid",
                },
                {
                    "phase_id": "buyer_selection",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["platform.rank_offers"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.search",
                            "destination": "platform:aggregator",
                        },
                        {
                            "action_kind": "commerce.accept_offer",
                            "destination": "platform:aggregator",
                        },
                        {
                            "action_kind": "delegate.reject_purchase",
                            "destination": "@inbound_sender",
                        },
                    ],
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
                    "pre_action_read_gate": {
                        "action_kind": "commerce.accept_offer",
                        "exclusive_until_complete": True,
                        "requirements": [
                            {
                                "tool": "world.get_listing",
                                "argument_field": "sku_id",
                                "required_values": [
                                    row.listing_id for row in case.public_task.listings
                                ],
                            },
                            *(
                                [
                                    {
                                        "tool": "world.get_evidence_record",
                                        "argument_field": "record_id",
                                        "required_values": list(
                                            case.public_task.available_record_ids
                                        ),
                                    }
                                ]
                                if case.public_task.available_record_ids
                                else []
                            ),
                            *(
                                [
                                    {
                                        "tool": "world.get_mandate_revisions",
                                        "minimum_successes": 1,
                                    }
                                ]
                                if (
                                    case.public_task.preference_updates
                                    or case.public_task.authority_instructions
                                )
                                else []
                            ),
                            *(
                                [
                                    {
                                        "tool": "world.get_review_evidence",
                                        "argument_field": "sku_id",
                                        "required_values": list(
                                            dict.fromkeys(
                                                row.listing_id
                                                for row in case.public_task.social_signals
                                            )
                                        ),
                                    }
                                ]
                                if case.public_task.social_signals
                                else []
                            ),
                        ],
                    },
                },
                {
                    "phase_id": "buyer_decision_evidence",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["platform.settlement_receipt"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "forbid",
                    "result": {
                        "action_kind": "commerce.submit_decision_record",
                        "destination": "runtime:evidence",
                        "result_format": "direct_details",
                        "payload_schema": _decision_payload_schema(),
                    },
                },
            ],
        },
    }


def _social_state(case: T3RuntimeCase) -> dict[str, Any]:
    signals = case.public_task.social_signals
    if not signals:
        return {}
    advisors = tuple(sorted({row.advisor_id for row in signals}))
    return {
        "friendships": [{"buyer_id": _BUYER_ID, "friends": list(advisors)}],
        "reviews": [
            {
                "review_id": row.signal_id,
                "reviewer_id": row.advisor_id,
                "sku_id": row.listing_id,
                "merchant_id": _MERCHANT_ID,
                "rating": row.rating,
                "text": "Customer rating for the listed product.",
            }
            for row in signals
        ],
    }


def scenario_for_t3(task_id: str) -> ScenarioSpec:
    """Build one fixed T3 task as a real CommerceWorld scenario."""

    case = runtime_case_t3(task_id)
    candidates = [_candidate_row(case, row) for row in case.public_task.listings]
    evidence_records = _evidence_rows(case)
    mandate_authority, mandate_revisions = _mandate_state(case)
    catalog = list(candidates)
    budget = _budget_for(case)
    mandate = {
        "mandate_id": task_id,
        "goal": case.public_task.directive,
        "quantity": 1,
        "return_after_purchase": False,
        "hard_constraints": {
            "budget": budget,
            "delivery_days": 365,
            "must_have": [],
        },
        "soft_constraints": [
            {"feature": name, "importance": weight}
            for name, weight in case.public_task.initial_preference_weights
        ],
        "soft_preferences": {"style": [], "avoid": []},
        "authority": {
            "can_buy_without_confirmation": True,
            "must_not_share_with_merchant": ["budget"],
        },
        "intent_expiry": "2099-12-31T00:00:00Z",
        "benchmark_contract": _benchmark_contract(case),
        "task_context": _public_task_context(case),
    }
    population = PopulationSpec(
        buyers=(
            BuyerSpec(
                buyer_id=_BUYER_ID,
                persona={"name": "T3 benchmark buyer", "task_family": "T3"},
                mandate=mandate,
            ),
        ),
        merchants=(
            MerchantSpec(
                merchant_id=_MERCHANT_ID,
                persona={"name": "Deterministic T3 catalog merchant"},
                policy={
                    "floor_price": max(
                        1,
                        min(
                            int(dict(row.attributes)["price_cents"])
                            for row in case.public_task.listings
                        )
                        // 2,
                    ),
                    "margin_target_bps": 1_500,
                    "max_negotiation_rounds": 3,
                    "refund_policy": "30_day_return",
                    "claim_aggressiveness": "neutral",
                },
                catalog_scope=tuple(str(row["sku_id"]) for row in candidates),
            ),
        ),
        matching={"top_k": len(candidates)},
        execution={"max_transactions_per_buyer": 1},
    )
    expected = case.expected.to_dict()
    return ScenarioSpec(
        scenario_id=task_id.casefold().replace("-", "_") + "__runtime",
        seed=_compatibility_seed(case),
        initial_state={
            "catalog": catalog,
            "logical_time": max(
                (revision.logical_tick for revision in mandate_revisions),
                default=0,
            ),
            "evidence_records": evidence_records,
            "mandate_authorities": (
                [] if mandate_authority is None else [mandate_authority_to_wire(mandate_authority)]
            ),
            "mandate_revisions": [
                mandate_revision_to_dict(revision) for revision in mandate_revisions
            ],
            **_social_state(case),
        },
        buyer_goal={},
        merchant_policy={},
        allowed_actions=("search", "accept_offer", "reject_offer", "settle"),
        success_oracle={
            "schema_version": T3_RUNTIME_SCHEMA_V2,
            "task_id": task_id,
            "task_content_sha256": case.content_sha256,
            "expected_contract": expected,
            "terminal_outcome": {
                "kind": "settled_order",
                "order_id": (f"ord-{task_id}-agg:{case.expected.selected_listing_id}"),
                "sku_id": case.expected.selected_listing_id,
                "buyer_id": _BUYER_ID,
                "merchant_id": _MERCHANT_ID,
                "qty": 1,
            },
        },
        platform_policy=None,
        population=population,
    )


def _decision_details(case: T3RuntimeCase) -> dict[str, Any]:
    expected = case.expected
    return {
        "considered_listing_ids": [row.listing_id for row in case.public_task.listings],
        "used_signal_ids": list(expected.used_signal_ids),
        "applied_update_ids": list(expected.applied_update_ids),
        "governing_instruction_ids": list(expected.governing_instruction_ids),
        "discarded_input_ids": list(expected.discarded_input_ids),
    }


def _business_request(user_prompt: str) -> dict[str, Any]:
    """Parse the sole provider-facing T3 business request."""

    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("T3 business prompt has no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict) or value.get("schema_version") != (
        "cwe.llm-decision-request.v1"
    ):
        raise ValueError("T3 business prompt has the wrong request schema")
    return value


def _business_response(
    request: Mapping[str, Any],
    intent: str,
    arguments: Mapping[str, Any],
) -> BusinessDecisionResponseV1:
    rows = request.get("allowed_intents")
    available = {str(row.get("intent")) for row in rows or () if isinstance(row, Mapping)}
    if intent not in available:
        raise ValueError(f"T3 business intent {intent!r} is unavailable")
    content = json.dumps(
        {
            "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "intent": intent,
            "arguments": copy.deepcopy(dict(arguments)),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return BusinessDecisionResponseV1(
        content=content,
        response_chars=len(content),
        response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _task_business_facts(request: Mapping[str, Any]) -> dict[str, Any]:
    observations = request.get("observations")
    for observation in observations or ():
        if not isinstance(observation, Mapping):
            continue
        persistent = observation.get("persistent_task_business_facts")
        benchmark = persistent.get("benchmark") if isinstance(persistent, Mapping) else None
        if isinstance(benchmark, Mapping):
            return copy.deepcopy(dict(benchmark))
    raise ValueError("T3 business request has no persistent benchmark facts")


def _ideal_rows(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    """Validate one public array used by the runtime-local ideal policy."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"T3 public {label} must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"T3 public {label} contains a non-object row")
    return tuple(value)


def _ideal_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"T3 public {label} must be non-empty text")
    return value


def _ideal_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"T3 public {label} must be an integer")
    return value


def _ideal_indexed_rows(
    value: Any,
    *,
    key: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in _ideal_rows(value, label=label):
        identity = _ideal_text(row.get(key), label=f"{label} {key}")
        if identity in output:
            raise ValueError(f"T3 public {label} contains duplicate {identity!r}")
        output[identity] = row
    return output


def _ideal_constraint_accepts(
    constraint: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> bool:
    field = _ideal_text(constraint.get("field"), label="constraint field")
    operator = _ideal_text(constraint.get("operator"), label="constraint operator")
    if field == "private_budget_ceiling":
        # Platform ranking is already bound to the actor's signed private
        # budget.  The numeric secret is intentionally absent from the model
        # request and cannot distinguish the candidates supplied here.
        return True
    if field not in attributes or "value" not in constraint:
        raise ValueError(f"T3 public constraint {field!r} has no visible comparison value")
    observed = attributes[field]
    expected = constraint["value"]
    if operator == "eq":
        return observed == expected
    if operator == "ne":
        return observed != expected
    if operator == "le":
        return observed <= expected
    if operator == "ge":
        return observed >= expected
    raise ValueError(f"unsupported T3 public constraint operator {operator!r}")


def _ideal_t3_choice(
    request: Mapping[str, Any],
) -> tuple[str, dict[str, list[str]]]:
    """Derive T3's optimum solely from facts in this provider request.

    This is deliberately a runtime-local implementation.  The independent
    provider-view solver remains an audit oracle and is never called by the
    ideal channel that produces benchmark evidence.
    """

    facts = grounded_t3_facts_v1(request, _task_business_facts(request))
    policy = facts.get("decision_policy")
    if not isinstance(policy, Mapping) or policy.get("rule_set") != ("preference-selection-v1"):
        raise ValueError("T3 public decision policy is missing")

    candidates = _ideal_rows(facts.get("candidate_facts"), label="candidate_facts")
    records = _ideal_indexed_rows(
        facts.get("source_records", ()),
        key="record_ref",
        label="source_records",
    )
    initial_weights = facts.get("initial_preference_weights")
    if not isinstance(initial_weights, Mapping):
        raise ValueError("T3 public initial_preference_weights are missing")
    weights = {
        str(name): _ideal_integer(value, label=f"weight {name}")
        for name, value in initial_weights.items()
    }

    update_rows = _ideal_rows(
        facts.get("preference_updates", ()),
        label="preference_updates",
    )
    authorized_updates: list[Mapping[str, Any]] = []
    for update in update_rows:
        if update.get("authorized") is True:
            authorized_updates.append(update)
            continue
        credential_ref = _ideal_text(
            update.get("credential_record_ref"),
            label="preference update credential_record_ref",
        )
        credential = records.get(credential_ref)
        attributes = credential.get("attributes") if isinstance(credential, Mapping) else None
        if not isinstance(attributes, Mapping):
            raise ValueError(f"T3 public preference credential {credential_ref!r} is unavailable")
        if attributes.get("authorized") is True and attributes.get("scope") == "preferences":
            authorized_updates.append(update)
    authorized_updates.sort(
        key=lambda row: (
            _ideal_integer(row.get("update_sequence"), label="preference update sequence"),
            _ideal_text(row.get("stable_update_key"), label="stable_update_key"),
        )
    )
    if authorized_updates:
        final_weights = authorized_updates[-1].get("preference_weights")
        if not isinstance(final_weights, Mapping):
            raise ValueError("T3 public authorized update has no weights")
        weights = {
            str(name): _ideal_integer(value, label=f"updated weight {name}")
            for name, value in final_weights.items()
        }

    instruction_rows = _ideal_rows(
        facts.get("authority_instructions", ()),
        label="authority_instructions",
    )
    governing_by_field: dict[str, tuple[int, str, Mapping[str, Any]]] = {}
    for instruction in instruction_rows:
        if instruction.get("authorized") is True:
            rank = 0
        else:
            credential_ref = _ideal_text(
                instruction.get("credential_record_ref"),
                label="authority credential_record_ref",
            )
            credential = records.get(credential_ref)
            attributes = credential.get("attributes") if isinstance(credential, Mapping) else None
            if not isinstance(attributes, Mapping):
                raise ValueError(
                    f"T3 public authority credential {credential_ref!r} is unavailable"
                )
            if attributes.get("authorized") is not True:
                continue
            rank = _ideal_integer(attributes.get("authority_rank"), label="authority rank")
        constraint = instruction.get("constraint")
        if not isinstance(constraint, Mapping):
            raise ValueError("T3 public authority instruction has no constraint")
        field = _ideal_text(constraint.get("field"), label="authority constraint field")
        candidate = (
            rank,
            _ideal_text(
                instruction.get("stable_instruction_key"),
                label="stable_instruction_key",
            ),
            instruction,
        )
        current = governing_by_field.get(field)
        if current is None or candidate[:2] < current[:2]:
            governing_by_field[field] = candidate

    hard_constraints = list(
        _ideal_rows(facts.get("hard_constraints", ()), label="hard_constraints")
    )
    hard_constraints.extend(row[2]["constraint"] for _, row in sorted(governing_by_field.items()))

    social_signals = _ideal_rows(
        facts.get("social_signals", ()),
        label="social_signals",
    )
    trust_threshold = _ideal_integer(
        policy.get("social_trust_minimum_bps"),
        label="social_trust_minimum_bps",
    )
    trusted_signals: list[Mapping[str, Any]] = []
    for signal in social_signals:
        record_ref = _ideal_text(signal.get("trust_record_ref"), label="trust_record_ref")
        record = records.get(record_ref)
        attributes = record.get("attributes") if isinstance(record, Mapping) else None
        if not isinstance(attributes, Mapping):
            raise ValueError(f"T3 public trust record {record_ref!r} is unavailable")
        trust_bps = attributes.get("trust_bps")
        if (
            attributes.get("verified") is True
            and isinstance(trust_bps, int)
            and not isinstance(trust_bps, bool)
            and trust_bps >= trust_threshold
        ):
            trusted_signals.append(signal)

    candidate_by_ref: dict[str, Mapping[str, Any]] = {}
    scores: dict[str, int] = {}
    feasible_refs: list[str] = []
    for candidate in candidates:
        listing_ref = _ideal_text(candidate.get("listing_ref"), label="candidate listing_ref")
        if listing_ref in candidate_by_ref:
            raise ValueError("T3 public candidate_facts contain a duplicate listing_ref")
        candidate_by_ref[listing_ref] = candidate
        hard_attributes = candidate.get("hard_attributes", candidate.get("attributes"))
        preference_values = candidate.get("preference_values")
        if not isinstance(hard_attributes, Mapping) or not isinstance(preference_values, Mapping):
            raise ValueError("T3 public candidate facts are incomplete")
        if all(
            _ideal_constraint_accepts(constraint, hard_attributes)
            for constraint in hard_constraints
        ):
            feasible_refs.append(listing_ref)
        score = sum(
            weight
            * _ideal_integer(
                preference_values.get(name, 0),
                label=f"feature {name}",
            )
            for name, weight in weights.items()
        )
        for signal in trusted_signals:
            if signal.get("listing_ref") != listing_ref:
                continue
            record = records[_ideal_text(signal.get("trust_record_ref"), label="trust_record_ref")]
            attributes = record["attributes"]
            score += (
                _ideal_integer(signal.get("rating"), label="social rating")
                * _ideal_integer(attributes.get("trust_bps"), label="trust_bps")
                // 1_000
            )
        scores[listing_ref] = score
    if not feasible_refs:
        raise ValueError("T3 public facts contain no feasible candidate")
    selected_ref = min(
        feasible_refs,
        key=lambda ref: (
            -scores[ref],
            _ideal_text(
                candidate_by_ref[ref].get("stable_tie_break_key"),
                label="stable_tie_break_key",
            ),
        ),
    )

    used_signal_refs = [
        _ideal_text(row.get("signal_ref"), label="signal_ref") for row in trusted_signals
    ]
    applied_update_refs = [
        _ideal_text(row.get("update_ref"), label="update_ref") for row in authorized_updates
    ]
    governing_instruction_refs = [
        _ideal_text(row[2].get("instruction_ref"), label="instruction_ref")
        for _, row in sorted(governing_by_field.items())
    ]
    used_refs = set((*used_signal_refs, *applied_update_refs, *governing_instruction_refs))
    all_context_refs = [
        *(
            _ideal_text(value, label="signal_ref")
            for value in facts.get(
                "social_signal_refs",
                [row.get("signal_ref") for row in social_signals],
            )
        ),
        *(
            _ideal_text(value, label="update_ref")
            for value in facts.get(
                "preference_update_refs",
                [row.get("update_ref") for row in update_rows],
            )
        ),
        *(
            _ideal_text(value, label="instruction_ref")
            for value in facts.get(
                "authority_instruction_refs",
                [row.get("instruction_ref") for row in instruction_rows],
            )
        ),
    ]
    return selected_ref, {
        "considered_listing_refs": [
            _ideal_text(row.get("listing_ref"), label="candidate listing_ref") for row in candidates
        ],
        "used_signal_refs": used_signal_refs,
        "applied_update_refs": applied_update_refs,
        "governing_instruction_refs": governing_instruction_refs,
        "discarded_input_refs": [ref for ref in all_context_refs if ref not in used_refs],
    }


def _ideal_business_decision_t3(
    request: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return the runtime ideal decision using only this current request."""

    phase = _ideal_text(request.get("phase"), label="phase")
    if phase == "buyer_discovery":
        return "search", {"query": ""}

    selected_ref, details = _ideal_t3_choice(request)
    if phase == "buyer_selection":
        selected = [row for row in _ranked_offers(request) if row.get("sku_ref") == selected_ref]
        if len(selected) != 1:
            raise ValueError("T3 public optimum has no unique ranked offer")
        offer_ref = _ideal_text(selected[0].get("offer_ref"), label="selected offer_ref")
        return (
            "accept_ranked_offer",
            {
                "offer_ref": offer_ref,
                "reason": "Highest feasible score under the published decision policy.",
            },
        )
    if phase == "buyer_decision_evidence":
        return (
            "submit_decision_record",
            {
                "outcome": "completed",
                "summary": "Recorded the provider-visible grounded selection.",
                "payload": details,
            },
        )
    raise ValueError(f"unsupported T3 business phase {phase!r}")


def _reference_aliases(
    case: T3RuntimeCase,
    task_facts: Mapping[str, Any],
) -> dict[str, str]:
    """Bind task-owned business identities to the Agent-issued public refs."""

    output: dict[str, str] = {}

    def bind(internal_values: Sequence[str], public_name: str) -> None:
        raw_refs = task_facts.get(public_name, [])
        if not isinstance(raw_refs, list) or len(raw_refs) != len(internal_values):
            raise ValueError(f"T3 public {public_name} do not match frozen task facts")
        for internal, public in zip(internal_values, raw_refs, strict=True):
            if not isinstance(public, str) or not public:
                raise ValueError(f"T3 public {public_name} contain an invalid ref")
            previous = output.get(str(internal))
            if previous is not None and previous != public:
                raise ValueError("T3 Agent issued two refs for one business identity")
            output[str(internal)] = public

    public = case.public_task
    bind(tuple(row.listing_id for row in public.listings), "candidate_listing_refs")
    bind(tuple(row.signal_id for row in public.social_signals), "social_signal_refs")
    bind(
        tuple(row.update_id for row in public.preference_updates),
        "preference_update_refs",
    )
    bind(
        tuple(row.instruction_id for row in public.authority_instructions),
        "authority_instruction_refs",
    )
    bind(public.available_record_ids, "available_record_refs")
    return output


def _publicize_business_value(value: Any, aliases: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return aliases.get(value, value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_name, item in value.items():
            name = str(raw_name)
            public_name = aliases.get(name, name)
            if public_name == name:
                if name.endswith("_ids"):
                    public_name = name[:-4] + "_refs"
                elif name.endswith("_id"):
                    public_name = name[:-3] + "_ref"
            if public_name in output:
                raise ValueError("T3 public reference projection has a key collision")
            output[public_name] = _publicize_business_value(item, aliases)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_publicize_business_value(item, aliases) for item in value]
    return copy.deepcopy(value)


def _ranked_offers(request: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    observations = request.get("observations")
    for observation in observations or ():
        if not isinstance(observation, Mapping) or observation.get("event") != "rank_offers":
            continue
        facts = observation.get("facts")
        rows = facts.get("candidates") if isinstance(facts, Mapping) else None
        if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
            return tuple(copy.deepcopy(dict(row)) for row in rows)
    return ()


class _T3BusinessChannel:
    """Typed ideal/mutation policy over public T3 observations and refs."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(
        self,
        case: T3RuntimeCase,
        *,
        listing_id: str | None = None,
        detail_overrides: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.case = case
        # ``None`` is the ideal policy: its semantic choice is solved from the
        # provider request.  A concrete id is reserved for targeted mutation.
        self._listing_id = listing_id
        self._detail_overrides = {
            str(name): list(values) for name, values in (detail_overrides or {}).items()
        }
        self._active_decision_id: str | None = None
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._terminal: tuple[str, dict[str, Any]] | None = None

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T3 business decision requires an Agent decision id")
        request = _business_request(user_prompt)
        if decision_id != self._active_decision_id:
            self._active_decision_id = decision_id
            self._pending = []
            self._terminal = None
            self._prepare_turn(request)
        if self._pending:
            intent, arguments = self._pending.pop(0)
            return _business_response(request, intent, arguments)
        if self._terminal is None:
            self._terminal = _ideal_business_decision_t3(request)
        intent, arguments = self._terminal
        return _business_response(request, intent, arguments)

    def _prepare_turn(self, request: Mapping[str, Any]) -> None:
        phase = str(request.get("phase", ""))
        if phase == "buyer_discovery":
            self._terminal = _ideal_business_decision_t3(request)
            return
        if phase == "buyer_selection":
            self._prepare_selection(request)
            return
        if phase == "buyer_decision_evidence":
            self._terminal = self._report_choice(request)
            return
        raise ValueError(f"unexpected T3 business phase {phase!r}")

    def _prepare_selection(self, request: Mapping[str, Any]) -> None:
        facts = _task_business_facts(request)
        self._pending.extend(
            ("observe_listing", {"sku_ref": ref}) for ref in facts.get("candidate_listing_refs", ())
        )
        self._pending.extend(
            ("observe_evidence_record", {"record_ref": ref})
            for ref in facts.get("available_record_refs", ())
        )
        if facts.get("preference_update_refs") or facts.get("authority_instruction_refs"):
            self._pending.append(("observe_mandate_revisions", {}))
        reviewed_listing_ids = tuple(
            dict.fromkeys(
                str(row["listing_ref"])
                for row in facts.get("social_signal_targets", ())
                if isinstance(row, Mapping) and isinstance(row.get("listing_ref"), str)
            )
        )
        self._pending.extend(
            ("observe_review_evidence", {"sku_ref": listing_id})
            for listing_id in reviewed_listing_ids
        )

        if self._listing_id is None:
            # The current request is refreshed after each Agent-owned read.
            # Defer the actual choice until all observations are present.
            return
        aliases = _reference_aliases(self.case, facts)
        selected_ref = aliases[self._listing_id]
        selected = next(
            (row for row in _ranked_offers(request) if row.get("sku_ref") == selected_ref),
            None,
        )
        offer_ref = selected.get("offer_ref") if isinstance(selected, Mapping) else None
        if not isinstance(offer_ref, str) or not offer_ref:
            raise ValueError("T3 selected listing has no ranked public offer ref")
        self._terminal = (
            "accept_ranked_offer",
            {
                "offer_ref": offer_ref,
                "reason": "Choose the selected listing after grounding all relevant facts.",
            },
        )

    def _report_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if self._listing_id is None and not self._detail_overrides:
            return _ideal_business_decision_t3(request)
        details = _decision_details(self.case)
        details.update(copy.deepcopy(self._detail_overrides))
        aliases = _reference_aliases(self.case, _task_business_facts(request))
        return (
            "submit_decision_record",
            {
                "outcome": "completed",
                "summary": "Recorded the grounded T3 decision after settlement.",
                "payload": _publicize_business_value(details, aliases),
            },
        )


def _raw_preference_choice(
    case: T3RuntimeCase,
    weights: Mapping[str, int],
) -> str:
    scores = {
        listing.listing_id: sum(
            weights.get(name, 0) * value for name, value in listing.preference_values
        )
        for listing in case.public_task.listings
    }
    return min(scores, key=lambda listing_id: (-scores[listing_id], listing_id))


def _mutation_plan(
    case: T3RuntimeCase,
) -> tuple[str, dict[str, list[str]]]:
    """Return the one semantic defect targeted by this task capability."""

    capability = case.task.capability_id
    expected = case.expected
    if capability == "t3.weighted_soft_preferences":
        selected = next(
            listing_id
            for listing_id in expected.feasible_listing_ids
            if listing_id != expected.selected_listing_id
        )
        return selected, {}
    if capability == "t3.hard_over_soft":
        selected = next(
            row.listing_id
            for row in case.public_task.listings
            if row.listing_id not in expected.feasible_listing_ids
        )
        return selected, {}
    if capability in {"t3.social_evidence", "t3.noisy_social_signals"}:
        return (
            _raw_preference_choice(
                case,
                dict(case.public_task.initial_preference_weights),
            ),
            {},
        )
    if capability == "t3.preference_update":
        unauthorized = next(
            row
            for row in case.public_task.preference_updates
            if row.update_id in expected.discarded_input_ids
        )
        return (
            _raw_preference_choice(case, dict(unauthorized.preference_weights)),
            {
                "applied_update_ids": [
                    *expected.applied_update_ids[:-1],
                    unauthorized.update_id,
                ],
                "discarded_input_ids": [],
            },
        )
    if capability == "t3.mandate_consistency":
        lower = case.public_task.authority_instructions[-1]
        return (
            _raw_preference_choice(
                case,
                dict(case.public_task.initial_preference_weights),
            ),
            {
                "governing_instruction_ids": [lower.instruction_id],
                "discarded_input_ids": list(expected.governing_instruction_ids),
            },
        )
    raise ValueError(f"unsupported T3 mutation capability {capability!r}")


class _MutationT3Channel(_T3BusinessChannel):
    """Change only the capability-targeted terminal decision and claims."""

    def __init__(self, case: T3RuntimeCase) -> None:
        listing_id, claim_overrides = _mutation_plan(case)
        if listing_id == case.expected.selected_listing_id:
            raise ValueError(f"{case.task.task_id} T3 mutation did not change selection")
        super().__init__(
            case,
            listing_id=listing_id,
            detail_overrides=claim_overrides,
        )


def _mutation_targets(case: T3RuntimeCase) -> tuple[str, ...]:
    targets = {
        "t3.weighted_soft_preferences": ("weighted_preference_utility_optimal",),
        "t3.hard_over_soft": (
            "hard_constraints_dominate_soft_value",
            "best_feasible_listing",
        ),
        "t3.social_evidence": ("context_aware_listing_optimal",),
        "t3.noisy_social_signals": ("trust_aware_listing_optimal",),
        "t3.preference_update": (
            "authorized_revisions_applied_in_order",
            "unauthorized_revision_rejected",
            "updated_preference_utility_optimal",
        ),
        "t3.mandate_consistency": (
            "highest_authority_instruction_governs",
            "lower_conflicting_instructions_rejected",
            "governing_mandate_preserved",
            "best_listing_within_mandate",
        ),
    }
    try:
        return targets[case.task.capability_id]
    except KeyError as exc:
        raise ValueError(f"unsupported T3 mutation capability {case.task.capability_id!r}") from exc


class _NoReplyT3Channel:
    """Typed inert counterpart; normal T3 terminal turns bypass it."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt, decision_id
        request = _business_request(user_prompt)
        return _business_response(
            request,
            "finish",
            {"reason": "No counterpart business action is required."},
        )


def _no_reply_channel() -> _NoReplyT3Channel:
    return _NoReplyT3Channel()


def _catalog_rows(evidence: RuntimeEvidenceBundleV2) -> tuple[dict[str, Any], ...]:
    rows = evidence.initial_world["tables"].get("catalog")
    if not isinstance(rows, list):
        return ()
    return tuple(dict(row) for row in rows if isinstance(row, dict))


def _world_contract_matches(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
) -> bool:
    expected_candidates = {row.listing_id: row for row in case.public_task.listings}
    catalog_rows = _catalog_rows(evidence)
    if len(catalog_rows) != len(expected_candidates):
        return False
    candidate_rows: dict[str, Mapping[str, Any]] = {}
    for row in catalog_rows:
        attrs = row.get("attributes")
        if not isinstance(attrs, dict):
            return False
        sku_id = str(row.get("sku_id", ""))
        listing = expected_candidates.get(sku_id)
        if listing is None:
            return False
        if (
            row.get("product_id") != _candidate_product_id(case, listing)
            or row.get("merchant_id") != _MERCHANT_ID
            or row.get("category") != _CANDIDATE_CATEGORY
        ):
            return False
        candidate_rows[sku_id] = attrs

    if set(candidate_rows) != set(expected_candidates):
        return False
    for listing_id, listing in expected_candidates.items():
        expected = _candidate_attributes(listing)
        observed = candidate_rows[listing_id]
        if any(observed.get(key) != value for key, value in expected.items()):
            return False

    tables = evidence.initial_world.get("tables", {})
    observed_evidence = tables.get("evidence_records", [])
    expected_evidence = _evidence_rows(case)
    if not isinstance(observed_evidence, list) or {
        str(row.get("record_id", "")): row for row in observed_evidence if isinstance(row, Mapping)
    } != {str(row["record_id"]): row for row in expected_evidence}:
        return False

    authority, revisions = _mandate_state(case)
    expected_authorities = [] if authority is None else [mandate_authority_to_wire(authority)]
    if tables.get("mandate_authorities", []) != expected_authorities:
        return False
    expected_revisions = [mandate_revision_to_dict(revision) for revision in revisions]
    if tables.get("mandate_revisions", []) != expected_revisions:
        return False
    return True


def _selected_listing_id(evidence: RuntimeEvidenceBundleV2) -> str | None:
    accepts = evidence.accepted_platform_exchanges(
        kind="commerce.accept_offer",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.create_match_certificate",
    )
    for exchange in reversed(accepts):
        envelope = exchange.request
        payload = envelope["action"].get("payload") or {}
        offer_id = str(payload.get("offer_id", ""))
        if offer_id.startswith("agg:"):
            return offer_id.removeprefix("agg:") or None
        if payload.get("sku_id"):
            return str(payload["sku_id"])
    return None


def _ranked_listing_ids(evidence: RuntimeEvidenceBundleV2) -> tuple[str, ...]:
    output: list[str] = []
    for exchange in evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    ):
        for envelope in exchange.responses:
            if envelope["action"].get("kind") != "platform.rank_offers":
                continue
            payload = envelope["action"].get("payload") or {}
            candidates = payload.get("candidates") or ()
            for row in candidates:
                if isinstance(row, dict) and row.get("sku_id"):
                    value = str(row["sku_id"])
                    if value not in output:
                        output.append(value)
    return tuple(output)


def _world_linked_certificate_responses(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
) -> _T3AuthorityPath:
    """Verify the search prefix and any completed selection authority graph."""

    verified: VerifiedMatchCertificateEvidence | None = None
    try:
        candidate = evidence.verified_operation_evidence(
            MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
            options={
                "expected_buyer_id": _BUYER_ID,
                "allow_rejected": False,
            },
        )
        if isinstance(candidate, VerifiedMatchCertificateEvidence):
            verified = candidate
    except RuntimeEvidenceError:
        pass
    matched_session_ids = tuple(
        row.session.session_id for row in (() if verified is None else verified.certificates)
    )
    search_verified: VerifiedSearchSessionEvidence | None = None
    try:
        candidate = evidence.verified_operation_evidence(
            SEARCH_SESSION_EVIDENCE_CONTRACT,
            options={"exclude_session_ids": matched_session_ids},
        )
        if isinstance(candidate, VerifiedSearchSessionEvidence):
            search_verified = candidate
    except RuntimeEvidenceError:
        pass

    unmatched_sessions = () if search_verified is None else search_verified.sessions
    matched_sessions = () if verified is None else verified.certificates
    session_rows: tuple[Any, ...] = (*unmatched_sessions, *matched_sessions)

    def session_value(row: Any, field: str) -> Any:
        session = row.session
        if isinstance(session, Mapping):
            return session.get(field)
        return getattr(session, field, None)

    def session_offers(row: Any) -> tuple[Any, ...]:
        value = session_value(row, "offers")
        return tuple(value) if isinstance(value, (list, tuple)) else ()

    def offer_value(offer: Any, field: str) -> Any:
        if isinstance(offer, Mapping):
            return offer.get(field)
        return getattr(offer, field, None)

    all_session_ids = () if search_verified is None else search_verified.all_session_ids
    expected_skus = {row.listing_id for row in case.public_task.listings}
    observed_session_skus = {
        str(offer_value(offer, "sku_id"))
        for offer in (() if not session_rows else session_offers(session_rows[0]))
    }
    search_prefix_verified = bool(
        search_verified is not None
        and len(all_session_ids) == 1
        and len(session_rows) == 1
        and session_value(session_rows[0], "session_id") == all_session_ids[0]
        and session_value(session_rows[0], "buyer_id") == _BUYER_ID
        and session_value(session_rows[0], "mandate_id") == case.task.task_id
        # A model-authored public filter may legitimately narrow the recall
        # set, including to zero offers.  That is a capability outcome scored
        # by ``complete_candidate_comparison`` below, not a corrupt World
        # session.  Integrity requires only that the authenticated session
        # cannot invent a SKU outside this task's authoritative catalog.
        and observed_session_skus <= expected_skus
        and all(
            offer_value(offer, "buyer_id") == _BUYER_ID
            and offer_value(offer, "mandate_id") == case.task.task_id
            for offer in session_offers(session_rows[0])
        )
    )
    search_commit_ids = tuple(
        str(row.commit.get("commit_id")) for row in unmatched_sessions
    ) + tuple(str(row.search_commit.get("commit_id")) for row in matched_sessions)
    preclaimed_commits = [row.commit for row in unmatched_sessions]
    for row in matched_sessions:
        preclaimed_commits.extend((row.search_commit, row.commit))
    preclaimed_commits.sort(key=lambda row: int(row.get("sequence", -1)))
    preclaimed_commit_ids = tuple(str(row["commit_id"]) for row in preclaimed_commits)

    match_valid = verified is not None and not any(
        row.session.mandate_id != case.task.task_id
        or row.acceptance.mandate_id != case.task.task_id
        or row.certificate.mandate_id != case.task.task_id
        for row in verified.certificates
    )
    linked_response_ids = frozenset(
        str(response["msg_id"])
        for row in (() if not match_valid or verified is None else verified.certificates)
        for response in row.responses
    )
    linked_certificate_ids = frozenset(
        () if not match_valid or verified is None else verified.certificate_ids
    )
    linked_order_ids = frozenset(() if not match_valid or verified is None else verified.order_ids)
    linked_sku_ids = frozenset(() if not match_valid or verified is None else verified.sku_ids)
    return _T3AuthorityPath(
        linked_response_ids=linked_response_ids,
        linked_certificate_ids=linked_certificate_ids,
        linked_order_ids=linked_order_ids,
        linked_sku_ids=linked_sku_ids,
        preclaimed_commit_ids=preclaimed_commit_ids,
        search_prefix_verified=search_prefix_verified,
        search_detail={
            "contract_id": SEARCH_SESSION_EVIDENCE_CONTRACT,
            "contract_verified": search_verified is not None,
            "all_session_ids": list(all_session_ids),
            "excluded_session_ids": (
                [] if search_verified is None else list(search_verified.excluded_session_ids)
            ),
            "search_commit_ids": list(search_commit_ids),
            "task_candidate_skus": sorted(expected_skus),
        },
    )


def _table_rows(
    evidence: RuntimeEvidenceBundleV2,
    table: str,
    *,
    initial: bool = False,
) -> tuple[dict[str, Any], ...]:
    snapshot = evidence.initial_world if initial else evidence.final_world
    rows = snapshot.get("tables", {}).get(table, [])
    if not isinstance(rows, list):
        return ()
    return tuple(dict(row) for row in rows if isinstance(row, Mapping))


def _inventory_rows(
    evidence: RuntimeEvidenceBundleV2,
    *,
    initial: bool = False,
) -> dict[str, dict[str, Any]]:
    snapshot = evidence.initial_world if initial else evidence.final_world
    rows = snapshot.get("tables", {}).get("inventory", {})
    if not isinstance(rows, Mapping):
        return {}
    return {str(sku_id): dict(row) for sku_id, row in rows.items() if isinstance(row, Mapping)}


def _terminal_outcome(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
    *,
    authority: _T3AuthorityPath,
) -> _T3TerminalOutcome:
    """Verify one certificate-bound PSP settlement and its exact World result."""

    accepted_settlements = evidence.accepted_platform_exchanges(
        kind="platform.settle_payment",
        actor_id=_BUYER_ID,
        endpoint="platform:psp",
    )
    settlement_commits = tuple(
        row for row in evidence.world_events if row.get("operation") == "settle"
    )
    verified: VerifiedSupplyFulfillmentEvidence | None = None
    verification_error: str | None = None
    try:
        candidate = evidence.verified_operation_evidence(
            SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            options={
                "allow_rejected": True,
                "evaluated_actor_id": _BUYER_ID,
                "preclaimed_commit_ids": authority.preclaimed_commit_ids,
            },
        )
        if isinstance(candidate, VerifiedSupplyFulfillmentEvidence):
            verified = candidate
    except RuntimeEvidenceError as exc:
        verification_error = str(exc)
    if (accepted_settlements or settlement_commits) and verified is None:
        raise RuntimeBenchmarkIntegrityError(
            "T3 accepted settlement is not bound to authoritative World evidence"
        )

    operations = () if verified is None else verified.operations
    settlements = (
        ()
        if verified is None
        else verified.operations_for(
            action_kind="platform.settle_payment",
            actor_id=_BUYER_ID,
        )
    )
    rejected = () if verified is None else verified.rejected_exchanges
    response_ids = frozenset(
        str(row.response["msg_id"])
        for row in settlements
        if isinstance(row.response, Mapping) and isinstance(row.response.get("msg_id"), str)
    )
    detail: dict[str, Any] = {
        "contract_id": SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
        "contract_verified": verified is not None,
        "verification_error": verification_error,
        "operation_count": len(operations),
        "settlement_count": len(settlements),
        "rejected_transaction_count": len(rejected),
        "world_linked_order_ids": sorted(authority.linked_order_ids),
        "world_linked_sku_ids": sorted(authority.linked_sku_ids),
        "settlement_response_ids": sorted(response_ids),
    }
    if (
        verified is None
        or len(operations) != 1
        or len(settlements) != 1
        or rejected
        or len(authority.linked_certificate_ids) != 1
        or len(authority.linked_order_ids) != 1
        or len(authority.linked_sku_ids) != 1
    ):
        return _T3TerminalOutcome(0.0, response_ids, detail)

    settlement = settlements[0]
    request_action = settlement.exchange.request.get("action")
    payload = request_action.get("payload") if isinstance(request_action, Mapping) else None
    if not isinstance(payload, Mapping):
        return _T3TerminalOutcome(0.0, response_ids, detail)

    certificate_id = next(iter(authority.linked_certificate_ids))
    order_id = next(iter(authority.linked_order_ids))
    sku_id = next(iter(authority.linked_sku_ids))
    settled_order = verified_settlement_order(settlement)
    exact_request = bool(
        payload.get("cert_id") == certificate_id
        and settled_order.get("order_id") == order_id
        and settled_order.get("buyer_id") == _BUYER_ID
        and settled_order.get("merchant_id") == _MERCHANT_ID
        and settled_order.get("sku_id") == sku_id
        and settled_order.get("qty") == 1
    )

    orders = _table_rows(evidence, "orders")
    ledger = _table_rows(evidence, "ledger")
    timelines = _table_rows(evidence, "order_timelines")
    payment_states = _table_rows(evidence, "payment_states")
    fulfillments = _table_rows(evidence, "fulfillments")
    order = orders[0] if len(orders) == 1 else {}
    receipt = ledger[0] if len(ledger) == 1 else {}
    timeline = timelines[0] if len(timelines) == 1 else {}
    payment = payment_states[0] if len(payment_states) == 1 else {}
    exact_tables = bool(
        len(orders) == len(ledger) == len(timelines) == len(payment_states) == 1
        and not fulfillments
        and order.get("order_id") == order_id
        and order.get("buyer_id") == _BUYER_ID
        and order.get("merchant_id") == _MERCHANT_ID
        and order.get("sku_id") == sku_id
        and order.get("qty") == 1
        and order.get("state") == "settled"
        and receipt.get("order_id") == order_id
        and receipt.get("buyer_id") == _BUYER_ID
        and receipt.get("merchant_id") == _MERCHANT_ID
        and receipt.get("sku_id") == sku_id
        and receipt.get("qty") == 1
        and receipt.get("effect") == "charge"
        and timeline.get("order_id") == order_id
        and isinstance(timeline.get("settled_at_tick"), int)
        and payment.get("order_id") == order_id
        and payment.get("owner_id") == _BUYER_ID
        and payment.get("merchant_id") == _MERCHANT_ID
        and payment.get("sku_id") == sku_id
        and payment.get("state") == "captured"
    )

    initial_inventory = _inventory_rows(evidence, initial=True)
    final_inventory = _inventory_rows(evidence)
    initial_selected = initial_inventory.get(sku_id, {})
    final_selected = final_inventory.get(sku_id, {})
    other_inventory_unchanged = all(
        final_inventory.get(other_sku) == row
        for other_sku, row in initial_inventory.items()
        if other_sku != sku_id
    )
    exact_inventory = bool(
        set(final_inventory) == set(initial_inventory)
        and initial_selected
        and final_selected
        and int(final_selected.get("qty_available", -1))
        == int(initial_selected.get("qty_available", -2))
        and int(final_selected.get("qty_reserved", -1))
        == int(initial_selected.get("qty_reserved", -1)) + 1
        and int(final_selected.get("version", -1)) == int(initial_selected.get("version", -1)) + 1
        and other_inventory_unchanged
    )
    exact = bool(exact_request and exact_tables and exact_inventory)
    detail.update(
        {
            "certificate_id": certificate_id,
            "order_id": order_id,
            "sku_id": sku_id,
            "expected_task_sku_id": case.expected.selected_listing_id,
            "exact_certificate_request_binding": exact_request,
            "exact_transaction_tables": exact_tables,
            "exact_inventory_reservation": exact_inventory,
        }
    )
    if not exact:
        raise RuntimeBenchmarkIntegrityError(
            "T3 verified settlement was not faithfully committed by World"
        )
    return _T3TerminalOutcome(1.0, response_ids, detail)


def _response_by_request(
    evidence: RuntimeEvidenceBundleV2,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for envelope in evidence.actions(kind="world.response"):
        if envelope.get("from") != "world" or envelope.get("to") != _BUYER_ID:
            continue
        request_id = envelope.get("in_reply_to")
        payload = envelope["action"].get("payload")
        if isinstance(request_id, str) and isinstance(payload, dict):
            output[request_id] = payload
    return output


def _trace_tool_results(
    evidence: RuntimeEvidenceBundleV2,
    *,
    tool: str,
) -> tuple[Mapping[str, Any], ...]:
    """Return framework-produced tool result slots from the signed trace.

    In-process episodes execute World tools synchronously, so no VCP
    ``world.response`` is needed.  Split-process episodes use the audited
    request/response join below.  Both paths write the actual framework tool
    result into ``audit.trace.jsonl``; model prose cannot manufacture a slot.
    """

    output: list[Mapping[str, Any]] = []
    for trace in evidence.trace_rows:
        if trace.get("agent_id") != _BUYER_ID or not tracker_row_has_usable_completed_steps(trace):
            continue
        for step in trace.get("steps", ()):
            if not isinstance(step, dict) or step.get("kind") != "tool_call":
                continue
            data = step.get("data")
            if not isinstance(data, dict):
                continue
            for result in data.get("results", ()):
                if isinstance(result, dict) and result.get("tool") == tool:
                    output.append(result)
    return tuple(output)


def _attested_catalog_reads(evidence: RuntimeEvidenceBundleV2) -> frozenset[str]:
    responses = _response_by_request(evidence)
    output: set[str] = set()
    for request in evidence.actions(kind="world.read_catalog", actor_id=_BUYER_ID):
        request_payload = request["action"].get("payload") or {}
        sku_id = str(request_payload.get("sku_id", ""))
        response = responses.get(str(request.get("msg_id", "")))
        if sku_id and response is not None and str(response.get("sku_id", "")) == sku_id:
            output.add(sku_id)
    for slot in _trace_tool_results(evidence, tool="world.get_listing"):
        args = slot.get("args")
        result = slot.get("result")
        sku_id = str(args.get("sku_id", "")) if isinstance(args, dict) else ""
        if sku_id and isinstance(result, dict) and str(result.get("sku_id", "")) == sku_id:
            output.add(sku_id)
    return frozenset(output)


def _attested_evidence_reads(
    evidence: RuntimeEvidenceBundleV2,
) -> frozenset[tuple[str, str]]:
    """Return exact record-id/digest pairs produced by authorized World reads."""

    responses = _response_by_request(evidence)
    output: set[tuple[str, str]] = set()
    for request in evidence.actions(kind="world.read_evidence_record", actor_id=_BUYER_ID):
        request_payload = request["action"].get("payload") or {}
        record_id = str(request_payload.get("record_id", ""))
        response = responses.get(str(request.get("msg_id", "")))
        if (
            record_id
            and isinstance(response, Mapping)
            and str(response.get("record_id", "")) == record_id
            and isinstance(response.get("record_digest"), str)
        ):
            output.add((record_id, str(response["record_digest"])))
    for slot in _trace_tool_results(evidence, tool="world.get_evidence_record"):
        args = slot.get("args")
        result = slot.get("result")
        record_id = str(args.get("record_id", "")) if isinstance(args, dict) else ""
        if (
            record_id
            and isinstance(result, Mapping)
            and str(result.get("record_id", "")) == record_id
            and isinstance(result.get("record_digest"), str)
        ):
            output.add((record_id, str(result["record_digest"])))
    return frozenset(output)


def _attested_mandate_revision_digests(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
) -> frozenset[str]:
    """Return revision digests from exact buyer-authorized World reads."""

    output: set[str] = set()

    def collect(value: Any) -> None:
        if not isinstance(value, list):
            return
        for row in value:
            if (
                isinstance(row, Mapping)
                and row.get("mandate_id") == case.task.task_id
                and isinstance(row.get("revision_digest"), str)
            ):
                output.add(str(row["revision_digest"]))

    responses = _response_by_request(evidence)
    for request in evidence.actions(kind="world.read_mandate_revisions", actor_id=_BUYER_ID):
        payload = request["action"].get("payload") or {}
        if payload.get("mandate_id") != case.task.task_id:
            continue
        collect(responses.get(str(request.get("msg_id", ""))))
    for slot in _trace_tool_results(evidence, tool="world.get_mandate_revisions"):
        args = slot.get("args")
        if not isinstance(args, Mapping) or args.get("mandate_id") != case.task.task_id:
            continue
        collect(slot.get("result"))
    return frozenset(output)


def _attested_review_ids(evidence: RuntimeEvidenceBundleV2) -> frozenset[str]:
    responses = _response_by_request(evidence)
    output: set[str] = set()
    for request in evidence.actions(kind="world.read_review_evidence", actor_id=_BUYER_ID):
        response = responses.get(str(request.get("msg_id", "")))
        if response is None:
            continue
        review_ids = response.get("review_ids") or ()
        if isinstance(review_ids, list):
            output.update(str(value) for value in review_ids)
    for slot in _trace_tool_results(evidence, tool="world.get_review_evidence"):
        result = slot.get("result")
        review_ids = result.get("review_ids") if isinstance(result, dict) else None
        if isinstance(review_ids, list):
            output.update(str(value) for value in review_ids)
    return frozenset(output)


def _claim_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not all(isinstance(row, str) for row in value):
        return None
    if len(value) != len(set(value)):
        return None
    return tuple(value)


def _observed_claims(
    task_id: str,
    evidence: RuntimeEvidenceBundleV2,
    *,
    allowed_parent_response_ids: frozenset[str],
) -> tuple[dict[str, tuple[str, ...]], bool]:
    accepted = evidence.accepted_actor_results(
        action_kind="commerce.submit_decision_record",
        actor_id=_BUYER_ID,
        task_id=task_id,
    )
    if len(accepted) != 1:
        return {}, False
    record = accepted[0]
    if record.get("in_reply_to") not in allowed_parent_response_ids:
        return {}, False
    report = record.get("report")
    content = report.get("content") if isinstance(report, dict) else None
    details = content.get("details") if isinstance(content, dict) else None
    if (
        not isinstance(content, dict)
        or content.get("outcome") != "completed"
        or not isinstance(details, dict)
        or set(details) != set(_DECISION_DETAIL_FIELDS)
    ):
        return {}, False
    output: dict[str, tuple[str, ...]] = {}
    for key in _DECISION_DETAIL_FIELDS:
        rows = _claim_tuple(details.get(key))
        if rows is None:
            return output, False
        output[key] = rows
    return output, True


def _decision_record_has_terminal_parent(
    task_id: str,
    evidence: RuntimeEvidenceBundleV2,
    *,
    terminal_response_ids: frozenset[str],
) -> bool:
    """Bind strict completion evidence to the settlement receipt, not the match."""

    accepted = evidence.accepted_actor_results(
        action_kind="commerce.submit_decision_record",
        actor_id=_BUYER_ID,
        task_id=task_id,
    )
    return bool(len(accepted) == 1 and accepted[0].get("in_reply_to") in terminal_response_ids)


def _platform_path_verified(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
    *,
    linked_certificate_evidence: _T3AuthorityPath,
) -> bool:
    searches = evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    )
    accepts = evidence.accepted_platform_exchanges(
        kind="commerce.accept_offer",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.create_match_certificate",
    )
    return bool(
        searches
        and len(accepts) == 1
        and len(linked_certificate_evidence.linked_response_ids) == 1
        and len(linked_certificate_evidence.linked_certificate_ids) == 1
    )


def _environment_path_observation(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
    *,
    linked_certificate_evidence: _T3AuthorityPath,
    terminal_outcome: _T3TerminalOutcome,
) -> dict[str, Any]:
    search_exchanges = evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    )
    accept_exchanges = evidence.accepted_platform_exchanges(
        kind="commerce.accept_offer",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.create_match_certificate",
    )
    searches = tuple(row.request for row in search_exchanges)
    ranks = tuple(
        response
        for row in search_exchanges
        for response in row.responses
        if response["action"].get("kind") == "platform.rank_offers"
    )
    accepts = tuple(row.request for row in accept_exchanges)
    manifest = evidence.evidence_manifest or {}
    manifest_valid = manifest.get("execution_backend") == COMMERCEWORLD_EPISODE_BACKEND_V2
    path_valid = bool(searches and ranks and accepts) and all(
        row.get("to") == "platform:aggregator" for row in (*searches, *accepts)
    )
    world_bound = _world_contract_matches(case, evidence)
    platform_bound = _platform_path_verified(
        case,
        evidence,
        linked_certificate_evidence=linked_certificate_evidence,
    )
    search_stage_valid = bool(
        manifest_valid
        and world_bound
        and linked_certificate_evidence.search_prefix_verified
        and len(searches) == 1
        and len(ranks) == 1
    )
    return {
        "manifest_verified": manifest_valid,
        "platform_route_path_verified": path_valid,
        "world_contract_bound": world_bound,
        "platform_decisions_verified": platform_bound,
        "search_authority_verified": search_stage_valid,
        "search_authority": dict(linked_certificate_evidence.search_detail),
        "world_linked_match_certificate_ids": sorted(
            linked_certificate_evidence.linked_certificate_ids
        ),
        "terminal_outcome_verified": terminal_outcome.credit == 1.0,
        "terminal_outcome": dict(terminal_outcome.detail),
        "behavioral_evidence_only": True,
        "buyer_search_count": len(searches),
        "platform_rank_count": len(ranks),
        "buyer_accept_count": len(accepts),
    }


def _check(
    name: str,
    weight: float,
    passed: bool,
    evidence: Mapping[str, Any],
) -> RuntimeRubricCheckV2:
    return RuntimeRubricCheckV2(
        name=name,
        weight=weight,
        credit=1.0 if passed else 0.0,
        evidence=dict(evidence),
    )


_T3_SCORE_BEARING_MODEL_ROUTES: Mapping[str, tuple[str, str]] = {
    "commerce.search": ("search", "platform:aggregator"),
    "commerce.accept_offer": ("accept_ranked_offer", "platform:aggregator"),
    "commerce.submit_decision_record": (
        "submit_decision_record",
        "runtime:evidence",
    ),
}
_T3_SCORE_BEARING_MODEL_INTENTS = frozenset(
    {intent for intent, _destination in _T3_SCORE_BEARING_MODEL_ROUTES.values()}
)

_T3_MODEL_WORLD_READ_TOOLS = frozenset(
    {
        "world.get_listing",
        "world.get_evidence_record",
        "world.get_mandate_revisions",
        "world.get_review_evidence",
    }
)


def _model_text_evidence_matches(value: Any, wire_text: Any) -> bool:
    """Compare scorer-safe model text evidence with its emitted wire text."""

    if not isinstance(wire_text, str):
        return False
    if isinstance(value, str):
        return value == wire_text
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"text_chars", "text_sha256"}
        and value.get("text_chars") == len(wire_text)
        and value.get("text_sha256") == hashlib.sha256(wire_text.encode("utf-8")).hexdigest()
    )


def _model_safe_value_matches_wire(value: Any, wire_value: Any) -> bool:
    """Compare one recursively sanitized model value with emitted business data."""

    if isinstance(value, Mapping) and set(value) == {
        "text_chars",
        "text_sha256",
    }:
        return _model_text_evidence_matches(value, wire_value)
    if isinstance(value, Mapping):
        return bool(
            isinstance(wire_value, Mapping)
            and set(value) == set(wire_value)
            and all(
                _model_safe_value_matches_wire(item, wire_value[name])
                for name, item in value.items()
            )
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return bool(
            isinstance(wire_value, Sequence)
            and not isinstance(wire_value, (str, bytes, bytearray))
            and len(value) == len(wire_value)
            and all(
                _model_safe_value_matches_wire(item, observed)
                for item, observed in zip(value, wire_value, strict=True)
            )
        )
    return type(value) is type(wire_value) and value == wire_value


def _t3_public_reference_aliases(case: T3RuntimeCase) -> dict[str, str]:
    public = case.public_task
    identities = (
        *(row.listing_id for row in public.listings),
        *(row.signal_id for row in public.social_signals),
        *(row.update_id for row in public.preference_updates),
        *(row.instruction_id for row in public.authority_instructions),
        *public.available_record_ids,
    )
    return {identity: public_reference_alias_v1(identity) for identity in dict.fromkeys(identities)}


def _t3_model_choice_matches_wire(
    case: T3RuntimeCase,
    choice: VerifiedModelBusinessChoice,
    envelope: Mapping[str, Any],
) -> bool:
    """Validate public model arguments against their Agent-compiled envelope."""

    action = envelope.get("action")
    if not isinstance(action, Mapping):
        return False
    action_kind = action.get("kind")
    payload = action.get("payload")
    if not isinstance(action_kind, str) or not isinstance(payload, Mapping):
        return False
    route = _T3_SCORE_BEARING_MODEL_ROUTES.get(action_kind)
    if route is None:
        return False
    expected_intent, expected_destination = route
    if (
        choice.emitted_msg_id != envelope.get("msg_id")
        or choice.intent != expected_intent
        or choice.action_kind != action_kind
        or choice.destination != expected_destination
        or envelope.get("to") != expected_destination
    ):
        return False

    arguments = choice.arguments
    if action_kind == "commerce.search":
        if set(arguments) - {"query", "filters"} or "query" not in arguments:
            return False
        if not _model_text_evidence_matches(arguments["query"], payload.get("query")):
            return False
        model_has_filters = "filters" in arguments
        wire_has_filters = "filters" in payload
        if model_has_filters != wire_has_filters:
            return False
        if model_has_filters and not _model_safe_value_matches_wire(
            arguments["filters"], payload["filters"]
        ):
            return False
        return True

    if action_kind == "commerce.accept_offer":
        if set(arguments) - {"offer_ref", "reason"} or "offer_ref" not in arguments:
            return False
        offer_id = payload.get("offer_id")
        return bool(
            isinstance(offer_id, str)
            and arguments["offer_ref"] == public_reference_alias_v1(offer_id)
        )

    if set(arguments) != {"outcome", "summary", "payload"}:
        return False
    if set(payload) != {"outcome", "summary", "details"}:
        return False
    public_details = _publicize_business_value(
        payload["details"],
        _t3_public_reference_aliases(case),
    )
    return bool(
        arguments["outcome"] == payload["outcome"]
        and _model_text_evidence_matches(arguments["summary"], payload["summary"])
        and _model_safe_value_matches_wire(arguments["payload"], public_details)
    )


def _require_t3_model_compilation_integrity(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Exact-join every score-bearing T3 wire action to its model choice."""

    try:
        choices = verified_model_business_choices(
            evidence,
            evaluated_actor_id=_BUYER_ID,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T3 model business-choice provenance is invalid"
        ) from exc
    choice_by_msg_id = {
        row.emitted_msg_id: row
        for row in choices
        if row.intent in _T3_SCORE_BEARING_MODEL_INTENTS
        or row.action_kind in _T3_SCORE_BEARING_MODEL_ROUTES
    }
    actions = tuple(
        row
        for row in evidence.envelopes
        if row.get("from") == _BUYER_ID
        and (row.get("action") or {}).get("kind") in _T3_SCORE_BEARING_MODEL_ROUTES
    )
    action_by_msg_id = {
        str(row.get("msg_id")): row for row in actions if isinstance(row.get("msg_id"), str)
    }
    if len(action_by_msg_id) != len(actions) or set(action_by_msg_id) != set(choice_by_msg_id):
        raise RuntimeBenchmarkIntegrityError(
            "T3 score-bearing wire actions do not exactly match model choices"
        )
    mismatched = sorted(
        msg_id
        for msg_id, action in action_by_msg_id.items()
        if not _t3_model_choice_matches_wire(case, choice_by_msg_id[msg_id], action)
    )
    if mismatched:
        raise RuntimeBenchmarkIntegrityError(
            "T3 Agent compiler drifted from public model arguments for message(s): "
            + ", ".join(mismatched)
        )


def _t3_model_world_read_matches_compilation(
    case: T3RuntimeCase,
    read: VerifiedModelWorldRead,
) -> bool:
    if read.intent == "observe_listing":
        internal = read.args.get("sku_id")
        return bool(
            read.tool == "world.get_listing"
            and set(read.arguments) == {"sku_ref"}
            and set(read.args) == {"sku_id"}
            and isinstance(internal, str)
            and read.arguments.get("sku_ref") == public_reference_alias_v1(internal)
        )
    if read.intent == "observe_evidence_record":
        internal = read.args.get("record_id")
        return bool(
            read.tool == "world.get_evidence_record"
            and set(read.arguments) == {"record_ref"}
            and set(read.args) == {"record_id"}
            and isinstance(internal, str)
            and read.arguments.get("record_ref") == public_reference_alias_v1(internal)
        )
    if read.intent == "observe_review_evidence":
        internal = read.args.get("sku_id")
        model_keys = set(read.arguments)
        compiled_keys = set(read.args)
        has_model_merchant = "merchant_ref" in read.arguments
        has_compiled_merchant = "merchant_id" in read.args
        return bool(
            read.tool == "world.get_review_evidence"
            and model_keys in ({"sku_ref"}, {"sku_ref", "merchant_ref"})
            and compiled_keys in ({"sku_id"}, {"sku_id", "merchant_id"})
            and has_model_merchant is has_compiled_merchant
            and isinstance(internal, str)
            and read.arguments.get("sku_ref") == public_reference_alias_v1(internal)
            and (
                not has_compiled_merchant
                or (
                    read.args.get("merchant_id") == _MERCHANT_ID
                    and read.arguments.get("merchant_ref")
                    == public_reference_alias_v1(_MERCHANT_ID)
                )
            )
        )
    if read.intent == "observe_mandate_revisions":
        return bool(
            read.tool == "world.get_mandate_revisions"
            and not read.arguments
            and read.args == {"mandate_id": case.task.task_id}
        )
    return False


def _require_t3_model_world_read_integrity(
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Prove public T3 observation refs reached the intended World objects."""

    try:
        reads = verified_model_world_reads(
            evidence,
            evaluated_actor_id=_BUYER_ID,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError("T3 model World-read provenance is invalid") from exc
    relevant_intents = frozenset(
        {
            "observe_listing",
            "observe_evidence_record",
            "observe_mandate_revisions",
            "observe_review_evidence",
        }
    )
    if any(
        not _t3_model_world_read_matches_compilation(case, read)
        for read in reads
        if read.intent in relevant_intents or read.tool in _T3_MODEL_WORLD_READ_TOOLS
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T3 Agent compiler mapped a public read reference to the wrong World object"
        )


def _score_capability(
    definition: TaskDefinitionV2,
    case: T3RuntimeCase,
    evidence: RuntimeEvidenceBundleV2,
) -> RuntimeTaskScoreV3:
    expected = case.expected
    _require_t3_model_world_read_integrity(case, evidence)
    _require_t3_model_compilation_integrity(case, evidence)
    linked_certificate_evidence = _world_linked_certificate_responses(case, evidence)
    accepted_searches = evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    )
    if accepted_searches and not linked_certificate_evidence.search_prefix_verified:
        raise RuntimeBenchmarkIntegrityError(
            "T3 accepted search is not bound to an authoritative World session"
        )
    accepted_matches = evidence.accepted_platform_exchanges(
        kind="commerce.accept_offer",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.create_match_certificate",
    )
    if accepted_matches and not linked_certificate_evidence.linked_certificate_ids:
        raise RuntimeBenchmarkIntegrityError(
            "T3 accepted offer is not bound to an authoritative World match certificate"
        )
    terminal_outcome = _terminal_outcome(
        case,
        evidence,
        authority=linked_certificate_evidence,
    )
    if accepted_matches and terminal_outcome.credit != 1.0:
        raise RuntimeBenchmarkIntegrityError(
            "T3 Agent continuation did not faithfully complete the selected offer"
        )
    claims, claims_valid = _observed_claims(
        case.task.task_id,
        evidence,
        allowed_parent_response_ids=terminal_outcome.linked_response_ids,
    )
    product_ids = tuple(row.listing_id for row in case.public_task.listings)
    product_set = set(product_ids)
    record_set = set(expected.relevant_record_ids)
    selected = _selected_listing_id(evidence)
    ranked = _ranked_listing_ids(evidence)
    attested_reads = _attested_catalog_reads(evidence)
    grounded_products = product_set & set(attested_reads)
    expected_record_digests = {
        str(row["record_id"]): str(row["record_digest"]) for row in _evidence_rows(case)
    }
    attested_evidence = _attested_evidence_reads(evidence)
    inspected_records = {
        record_id
        for record_id, digest in attested_evidence
        if expected_record_digests.get(record_id) == digest
    }
    mandate_revision_digests = _attested_mandate_revision_digests(case, evidence)
    expected_mandate_digests = {revision.revision_digest for revision in _mandate_state(case)[1]}
    mandate_history_grounded = mandate_revision_digests == expected_mandate_digests
    review_ids = _attested_review_ids(evidence)
    environment_path = _environment_path_observation(
        case,
        evidence,
        linked_certificate_evidence=linked_certificate_evidence,
        terminal_outcome=terminal_outcome,
    )
    if environment_path.get("world_contract_bound") is not True:
        raise RuntimeBenchmarkIntegrityError(
            "T3 task snapshot is not bound to the expected World contract"
        )

    # World reads and the selected offer prove what the model observed and did;
    # the terminal decision record proves which signals, updates, and authority
    # it says it actually used.  Both are model-authored business decisions and
    # both must agree.  Platform/World closure remains a validity prerequisite,
    # never a substitute for a missing or contradictory model conclusion.
    considered = claims.get("considered_listing_ids", ()) if claims_valid else ()
    coverage = (
        set(ranked) == product_set
        and grounded_products == product_set
        and set(considered) == product_set
        and len(considered) == len(product_ids)
    )
    retrieval = inspected_records == record_set
    feasible = (
        selected in set(expected.feasible_listing_ids)
        and selected in grounded_products
        and selected in set(ranked)
    )
    optimal = selected == expected.selected_listing_id
    used_claims = claims.get("used_signal_ids", ()) if claims_valid else ()
    applied_claims = claims.get("applied_update_ids", ()) if claims_valid else ()
    governing_claims = claims.get("governing_instruction_ids", ()) if claims_valid else ()
    discarded_claims = claims.get("discarded_input_ids", ()) if claims_valid else ()
    used_matches = bool(
        claims_valid
        and set(used_claims) == set(expected.used_signal_ids)
        and set(used_claims) <= set(review_ids)
    )
    applied_matches = bool(
        claims_valid
        and tuple(applied_claims) == tuple(expected.applied_update_ids)
        and mandate_history_grounded
    )
    governing_matches = bool(
        claims_valid
        and tuple(governing_claims) == tuple(expected.governing_instruction_ids)
        and mandate_history_grounded
    )
    discarded_matches = bool(
        claims_valid
        and set(discarded_claims) == set(expected.discarded_input_ids)
        and (
            set(discarded_claims) <= set(review_ids)
            if case.task.capability_id
            in {
                "t3.social_evidence",
                "t3.noisy_social_signals",
            }
            else mandate_history_grounded
        )
    )

    common = {
        "selected_listing_id": selected,
        "expected_listing_id": expected.selected_listing_id,
        "ranked_listing_ids": list(ranked),
        "grounded_listing_ids": sorted(grounded_products),
        "considered_listing_ids": list(considered),
        "inspected_record_ids": sorted(inspected_records),
        "expected_record_ids": list(expected.relevant_record_ids),
        "mandate_revision_digests": sorted(mandate_revision_digests),
        "expected_mandate_revision_digests": sorted(expected_mandate_digests),
        "decision_record_valid": claims_valid,
        "decision_record_claims": {name: list(values) for name, values in sorted(claims.items())},
    }
    capability = definition.capability_id
    if capability == "t3.weighted_soft_preferences":
        no_context = (
            bool(ranked)
            and selected is not None
            and not any((inspected_records, review_ids, mandate_revision_digests))
        )
        checks = (
            _check("complete_candidate_comparison", 0.15, coverage, common),
            _check("hard_constraints_preserved", 0.20, feasible, common),
            _check("weighted_preference_utility_optimal", 0.30, optimal, common),
            _check(
                "no_spurious_context",
                0.10,
                no_context,
                {
                    "review_ids": sorted(review_ids),
                    "inspected_record_ids": sorted(inspected_records),
                },
            ),
        )
    elif capability == "t3.hard_over_soft":
        no_context = (
            bool(ranked)
            and selected is not None
            and not any((inspected_records, review_ids, mandate_revision_digests))
        )
        checks = (
            _check("complete_candidate_comparison", 0.15, coverage, common),
            _check("hard_constraints_dominate_soft_value", 0.30, feasible, common),
            _check("best_feasible_listing", 0.20, optimal, common),
            _check(
                "no_spurious_context",
                0.10,
                no_context,
                {"review_ids": sorted(review_ids)},
            ),
        )
    elif capability == "t3.social_evidence":
        social_ids = {row.signal_id for row in case.public_task.social_signals}
        social_used = used_matches and discarded_matches and social_ids <= set(review_ids)
        checks = (
            _check("complete_candidate_comparison", 0.10, coverage, common),
            _check("source_trust_records_inspected", 0.15, retrieval, common),
            _check(
                "verified_social_evidence_used",
                0.20,
                social_used,
                {
                    "observed_review_ids": sorted(review_ids),
                    "claimed_used_signal_ids": list(used_claims),
                    "claimed_discarded_input_ids": list(discarded_claims),
                    "expected_used_signal_ids": list(expected.used_signal_ids),
                    "expected_discarded_input_ids": list(expected.discarded_input_ids),
                },
            ),
            _check("hard_constraints_preserved", 0.10, feasible, common),
            _check("context_aware_listing_optimal", 0.20, optimal, common),
        )
    elif capability == "t3.noisy_social_signals":
        trusted = set(expected.used_signal_ids)
        untrusted = {row.signal_id for row in case.public_task.social_signals} - trusted
        checks = (
            _check("complete_candidate_comparison", 0.10, coverage, common),
            _check("all_source_trust_records_inspected", 0.15, retrieval, common),
            _check(
                "trusted_signals_used",
                0.15,
                used_matches and trusted <= set(review_ids),
                {
                    "expected_used_signal_ids": list(expected.used_signal_ids),
                    "claimed_used_signal_ids": list(used_claims),
                    "observed_review_ids": sorted(review_ids),
                },
            ),
            _check(
                "unreliable_signals_discarded",
                0.15,
                discarded_matches and untrusted <= set(review_ids),
                {
                    "observed_review_ids": sorted(review_ids),
                    "claimed_discarded_input_ids": list(discarded_claims),
                    "expected_untrusted_signal_ids": sorted(untrusted),
                },
            ),
            _check("hard_constraints_preserved", 0.10, feasible, common),
            _check("trust_aware_listing_optimal", 0.10, optimal, common),
        )
    elif capability == "t3.preference_update":
        checks = (
            _check("complete_candidate_comparison", 0.10, coverage, common),
            _check("authorization_records_inspected", 0.10, retrieval, common),
            _check(
                "authorized_revisions_applied_in_order",
                0.20,
                applied_matches and mandate_history_grounded,
                {
                    "expected_update_ids": list(expected.applied_update_ids),
                    "claimed_update_ids": list(applied_claims),
                    "mandate_history_grounded": mandate_history_grounded,
                },
            ),
            _check(
                "unauthorized_revision_rejected",
                0.15,
                discarded_matches,
                {
                    "expected_discarded_ids": list(expected.discarded_input_ids),
                    "claimed_discarded_ids": list(discarded_claims),
                    "mandate_history_grounded": mandate_history_grounded,
                },
            ),
            _check("hard_constraints_preserved", 0.10, feasible, common),
            _check("updated_preference_utility_optimal", 0.10, optimal, common),
        )
    elif capability == "t3.mandate_consistency":
        checks = (
            _check("complete_candidate_comparison", 0.10, coverage, common),
            _check("authority_records_inspected", 0.10, retrieval, common),
            _check(
                "highest_authority_instruction_governs",
                0.20,
                governing_matches and mandate_history_grounded,
                {
                    "expected_governing_ids": list(expected.governing_instruction_ids),
                    "claimed_governing_ids": list(governing_claims),
                    "mandate_history_grounded": mandate_history_grounded,
                },
            ),
            _check(
                "lower_conflicting_instructions_rejected",
                0.15,
                discarded_matches,
                {
                    "expected_discarded_ids": list(expected.discarded_input_ids),
                    "claimed_discarded_ids": list(discarded_claims),
                    "mandate_history_grounded": mandate_history_grounded,
                },
            ),
            _check("governing_mandate_preserved", 0.15, feasible, common),
            _check("best_listing_within_mandate", 0.05, optimal, common),
        )
    else:
        raise ValueError(f"unsupported T3 capability {capability!r}")

    checks = renormalize_capability_checks_v2(checks)
    issues = () if all(row.credit == 1.0 for row in checks) else ("t3_runtime_contract_incomplete",)
    return score_checks(definition, checks, issues=issues)


def score_t3_runtime(
    task_id: str,
    evidence: RuntimeEvidenceBundleV2,
) -> RuntimeTaskScoreV3:
    """Score T3 only from verified Runtime, Platform, and World evidence."""

    require_runtime_benchmark_integrity_v2(evidence)
    require_frozen_scenario_fixture_v2(
        evidence,
        scenario_for_t3(task_id),
        family="T3",
    )
    case = runtime_case_t3(task_id)
    return _score_capability(TASK_REGISTRY_V2[task_id], case, evidence)


def runtime_bundle_t3(task_id: str) -> RuntimeTaskBundleV2:
    case = runtime_case_t3(task_id)
    scenario = scenario_for_t3(task_id)
    semantic_hash = canonical_sha256(
        {
            "task_content_sha256": case.content_sha256,
            "model_visible_benchmark_contract": _benchmark_contract(case),
            "model_visible_task_context": _public_task_context(case),
            "scenario_oracle": scenario.success_oracle,
            "scenario_state": scenario.initial_state,
        }
    )
    return RuntimeTaskBundleV2(
        task=case.task,
        scenario=scenario,
        evaluated_actor_id=_BUYER_ID,
        ideal_channel=lambda: _T3BusinessChannel(case),
        counterpart_channels={_MERCHANT_ID: _no_reply_channel},
        scorer=lambda evidence: score_t3_runtime(task_id, evidence),
        semantic_hash=semantic_hash,
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _MutationT3Channel(case),
                expected_changed_checks=_mutation_targets(case),
            ),
        ),
    )


def runtime_bundles_t3() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t3(task_id) for task_id in sorted(_T3_TASK_IDS))


__all__ = [
    "T3_RUNTIME_DATA_SCHEMA_V2",
    "T3_RUNTIME_SCHEMA_V2",
    "runtime_bundle_t3",
    "runtime_bundles_t3",
    "runtime_case_t3",
    "scenario_for_t3",
    "score_t3_runtime",
]
