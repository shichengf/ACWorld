"""Strict passive contracts for marketplace evidence and governance state.

This module defines wire-stable value objects only.  It does not read or write
World state, route VCP actions, rank listings, or implement benchmark policy.
Future World/Platform integrations are expected to persist the canonical wire
forms and pass trusted authority/owner identities into the validators.

All monetary values, scores, revisions, and logical times are integers.  This
keeps canonical JSON deterministic and prevents NaN/Infinity from entering an
authoritative evidence stream.  Versioned records bind their predecessor by
digest and carry append-only provenance digests.  Exact idempotent retries can
therefore return the persisted object byte-for-byte; reuse of a key for changed
content is rejected.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from collections.abc import Iterable
from typing import Any, Mapping

from protocol.errors import IdempotencyConflict, SchemaError


PLACEMENT_SCHEMA = "cwe.market-placement.v1"
CAMPAIGN_SCHEMA = "cwe.market-campaign.v1"
REVIEW_EVIDENCE_SCHEMA = "cwe.review-evidence.v1"
REVIEW_AGGREGATE_SCHEMA = "cwe.review-aggregate.v1"
MARKET_SIGNAL_SCHEMA = "cwe.market-signal.v1"
GOVERNANCE_CASE_SCHEMA = "cwe.governance-case.v1"
REPUTATION_EVENT_SCHEMA = "cwe.reputation-event.v1"
REMEDIATION_STEP_SCHEMA = "cwe.remediation-step.v1"
REMEDIATION_PLAN_SCHEMA = "cwe.remediation-plan.v1"
RANKING_CONTEXT_SCHEMA = "cwe.ranking-context.v1"

PLACEMENT_KINDS = frozenset({"organic", "sponsored"})
DISCLOSURE_STATUSES = frozenset({"not_required", "pending", "disclosed"})
CAMPAIGN_STATUSES = frozenset({"draft", "active", "paused", "closed"})
MARKET_SIGNAL_KINDS = frozenset(
    {
        "coordinated_pricing",
        "shared_coordination_language",
        "review_manipulation",
        "sponsorship_disclosure",
        "fulfillment_outcome",
        "compliance_audit",
    }
)
GOVERNANCE_CASE_KINDS = frozenset(
    {"competition", "review_integrity", "advertising_disclosure", "reputation_integrity"}
)
GOVERNANCE_CASE_STATUSES = frozenset({"open", "under_review", "resolved", "dismissed"})
REPUTATION_EVENT_KINDS = frozenset(
    {
        "fulfilled_order",
        "disputed_order",
        "refund",
        "remediation_verified",
        "compliance_violation",
    }
)
REMEDIATION_ACTION_KINDS = frozenset(
    {
        "buyer_refund",
        "defect_correction",
        "buyer_notification",
        "independent_audit",
        "ongoing_monitoring",
    }
)
REMEDIATION_STEP_STATUSES = frozenset({"pending", "completed", "verified", "rejected"})
REMEDIATION_PLAN_STATUSES = frozenset({"draft", "active", "completed", "terminated"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MarketGovernanceValidationError(SchemaError):
    """A well-shaped governance record violated a binding or history rule."""


@dataclass(frozen=True, slots=True)
class Placement:
    placement_id: str
    campaign_id: str | None
    owner_merchant_id: str
    sku_id: str
    placement_kind: str
    disclosure_status: str
    disclosure_text: str | None
    bid_cents: int
    fee_cents: int
    currency: str
    starts_at_tick: int
    ends_at_tick: int
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    placement_digest: str = ""


@dataclass(frozen=True, slots=True)
class Campaign:
    campaign_id: str
    owner_merchant_id: str
    status: str
    budget_cents: int
    currency: str
    starts_at_tick: int
    ends_at_tick: int
    placements: tuple[Placement, ...]
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    campaign_digest: str = ""


@dataclass(frozen=True, slots=True)
class ReviewEvidence:
    review_id: str
    sku_id: str
    merchant_id: str
    reviewer_id: str
    rating: int
    verified_purchase: bool
    order_id: str | None
    submitted_at_tick: int
    account_created_at_tick: int
    burst_group_id: str | None
    source_ref: str
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    evidence_digest: str = ""


@dataclass(frozen=True, slots=True)
class ReviewAggregate:
    aggregate_id: str
    sku_id: str
    merchant_id: str
    aggregation_policy_id: str
    aggregation_policy_version: int
    evidence_digests: tuple[str, ...]
    review_count: int
    verified_review_count: int
    rating_sum: int
    verified_rating_sum: int
    computed_at_tick: int
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    aggregate_digest: str = ""

    @property
    def mean_rating_milli(self) -> int:
        return 0 if self.review_count == 0 else self.rating_sum * 1_000 // self.review_count

    @property
    def verified_mean_rating_milli(self) -> int:
        return (
            0
            if self.verified_review_count == 0
            else self.verified_rating_sum * 1_000 // self.verified_review_count
        )


@dataclass(frozen=True, slots=True)
class MarketSignal:
    signal_id: str
    signal_kind: str
    subject_merchant_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    confidence_bps: int
    observed_at_tick: int
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    signal_digest: str = ""


@dataclass(frozen=True, slots=True)
class GovernanceCase:
    case_id: str
    case_kind: str
    subject_merchant_ids: tuple[str, ...]
    signal_digests: tuple[str, ...]
    status: str
    resolution_code: str | None
    opened_at_tick: int
    updated_at_tick: int
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    case_digest: str = ""


@dataclass(frozen=True, slots=True)
class ReputationEvent:
    event_id: str
    merchant_id: str
    event_kind: str
    source_ref: str
    outcome_bps: int
    occurred_at_tick: int
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    event_digest: str = ""


@dataclass(frozen=True, slots=True)
class RemediationStep:
    step_id: str
    plan_id: str
    owner_merchant_id: str
    sequence_no: int
    action_kind: str
    prerequisite_step_ids: tuple[str, ...]
    status: str
    evidence_refs: tuple[str, ...]
    completed_at_tick: int | None
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    step_digest: str = ""


@dataclass(frozen=True, slots=True)
class RemediationPlan:
    plan_id: str
    governance_case_id: str
    owner_merchant_id: str
    status: str
    created_at_tick: int
    updated_at_tick: int
    steps: tuple[RemediationStep, ...]
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    plan_digest: str = ""


@dataclass(frozen=True, slots=True)
class RankingContext:
    context_id: str
    request_id: str
    buyer_id: str
    mandate_id: str
    policy_id: str
    policy_version: int
    catalog_revision: int
    issued_at_tick: int
    candidate_sku_ids: tuple[str, ...]
    ranked_sku_ids: tuple[str, ...]
    placement_digests: tuple[str, ...]
    review_aggregate_digests: tuple[str, ...]
    governance_case_digests: tuple[str, ...]
    reputation_event_digests: tuple[str, ...]
    authored_by: str
    idempotency_key: str
    version: int = 1
    previous_digest: str | None = None
    provenance_digests: tuple[str, ...] = ()
    context_digest: str = ""


GovernanceRecord = (
    Placement
    | Campaign
    | ReviewEvidence
    | ReviewAggregate
    | MarketSignal
    | GovernanceCase
    | ReputationEvent
    | RemediationStep
    | RemediationPlan
    | RankingContext
)


def canonical_digest(value: Any) -> str:
    """SHA-256 over compact, sorted, NaN-free canonical JSON."""

    try:
        body = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(body).hexdigest()


def _placement_contract(value: Placement) -> dict[str, Any]:
    return {
        "schema_version": PLACEMENT_SCHEMA,
        "placement_id": value.placement_id,
        "campaign_id": value.campaign_id,
        "owner_merchant_id": value.owner_merchant_id,
        "sku_id": value.sku_id,
        "placement_kind": value.placement_kind,
        "disclosure_status": value.disclosure_status,
        "disclosure_text": value.disclosure_text,
        "bid_cents": value.bid_cents,
        "fee_cents": value.fee_cents,
        "currency": value.currency,
        "starts_at_tick": value.starts_at_tick,
        "ends_at_tick": value.ends_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _campaign_contract(value: Campaign) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": value.campaign_id,
        "owner_merchant_id": value.owner_merchant_id,
        "status": value.status,
        "budget_cents": value.budget_cents,
        "currency": value.currency,
        "starts_at_tick": value.starts_at_tick,
        "ends_at_tick": value.ends_at_tick,
        "placements": [placement_to_wire(item) for item in value.placements],
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _review_evidence_contract(value: ReviewEvidence) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_EVIDENCE_SCHEMA,
        "review_id": value.review_id,
        "sku_id": value.sku_id,
        "merchant_id": value.merchant_id,
        "reviewer_id": value.reviewer_id,
        "rating": value.rating,
        "verified_purchase": value.verified_purchase,
        "order_id": value.order_id,
        "submitted_at_tick": value.submitted_at_tick,
        "account_created_at_tick": value.account_created_at_tick,
        "burst_group_id": value.burst_group_id,
        "source_ref": value.source_ref,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _review_aggregate_contract(value: ReviewAggregate) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_AGGREGATE_SCHEMA,
        "aggregate_id": value.aggregate_id,
        "sku_id": value.sku_id,
        "merchant_id": value.merchant_id,
        "aggregation_policy_id": value.aggregation_policy_id,
        "aggregation_policy_version": value.aggregation_policy_version,
        "evidence_digests": list(value.evidence_digests),
        "review_count": value.review_count,
        "verified_review_count": value.verified_review_count,
        "rating_sum": value.rating_sum,
        "verified_rating_sum": value.verified_rating_sum,
        "computed_at_tick": value.computed_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _market_signal_contract(value: MarketSignal) -> dict[str, Any]:
    return {
        "schema_version": MARKET_SIGNAL_SCHEMA,
        "signal_id": value.signal_id,
        "signal_kind": value.signal_kind,
        "subject_merchant_ids": list(value.subject_merchant_ids),
        "source_refs": list(value.source_refs),
        "confidence_bps": value.confidence_bps,
        "observed_at_tick": value.observed_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _governance_case_contract(value: GovernanceCase) -> dict[str, Any]:
    return {
        "schema_version": GOVERNANCE_CASE_SCHEMA,
        "case_id": value.case_id,
        "case_kind": value.case_kind,
        "subject_merchant_ids": list(value.subject_merchant_ids),
        "signal_digests": list(value.signal_digests),
        "status": value.status,
        "resolution_code": value.resolution_code,
        "opened_at_tick": value.opened_at_tick,
        "updated_at_tick": value.updated_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _reputation_event_contract(value: ReputationEvent) -> dict[str, Any]:
    return {
        "schema_version": REPUTATION_EVENT_SCHEMA,
        "event_id": value.event_id,
        "merchant_id": value.merchant_id,
        "event_kind": value.event_kind,
        "source_ref": value.source_ref,
        "outcome_bps": value.outcome_bps,
        "occurred_at_tick": value.occurred_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _remediation_step_contract(value: RemediationStep) -> dict[str, Any]:
    return {
        "schema_version": REMEDIATION_STEP_SCHEMA,
        "step_id": value.step_id,
        "plan_id": value.plan_id,
        "owner_merchant_id": value.owner_merchant_id,
        "sequence_no": value.sequence_no,
        "action_kind": value.action_kind,
        "prerequisite_step_ids": list(value.prerequisite_step_ids),
        "status": value.status,
        "evidence_refs": list(value.evidence_refs),
        "completed_at_tick": value.completed_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _remediation_plan_contract(value: RemediationPlan) -> dict[str, Any]:
    return {
        "schema_version": REMEDIATION_PLAN_SCHEMA,
        "plan_id": value.plan_id,
        "governance_case_id": value.governance_case_id,
        "owner_merchant_id": value.owner_merchant_id,
        "status": value.status,
        "created_at_tick": value.created_at_tick,
        "updated_at_tick": value.updated_at_tick,
        "steps": [remediation_step_to_wire(item) for item in value.steps],
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def _ranking_context_contract(value: RankingContext) -> dict[str, Any]:
    return {
        "schema_version": RANKING_CONTEXT_SCHEMA,
        "context_id": value.context_id,
        "request_id": value.request_id,
        "buyer_id": value.buyer_id,
        "mandate_id": value.mandate_id,
        "policy_id": value.policy_id,
        "policy_version": value.policy_version,
        "catalog_revision": value.catalog_revision,
        "issued_at_tick": value.issued_at_tick,
        "candidate_sku_ids": list(value.candidate_sku_ids),
        "ranked_sku_ids": list(value.ranked_sku_ids),
        "placement_digests": list(value.placement_digests),
        "review_aggregate_digests": list(value.review_aggregate_digests),
        "governance_case_digests": list(value.governance_case_digests),
        "reputation_event_digests": list(value.reputation_event_digests),
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "version": value.version,
        "previous_digest": value.previous_digest,
        "provenance_digests": list(value.provenance_digests),
    }


def placement_digest(value: Placement) -> str:
    return canonical_digest(_placement_contract(value))


def campaign_digest(value: Campaign) -> str:
    return canonical_digest(_campaign_contract(value))


def review_evidence_digest(value: ReviewEvidence) -> str:
    return canonical_digest(_review_evidence_contract(value))


def review_aggregate_digest(value: ReviewAggregate) -> str:
    return canonical_digest(_review_aggregate_contract(value))


def market_signal_digest(value: MarketSignal) -> str:
    return canonical_digest(_market_signal_contract(value))


def governance_case_digest(value: GovernanceCase) -> str:
    return canonical_digest(_governance_case_contract(value))


def reputation_event_digest(value: ReputationEvent) -> str:
    return canonical_digest(_reputation_event_contract(value))


def remediation_step_digest(value: RemediationStep) -> str:
    return canonical_digest(_remediation_step_contract(value))


def remediation_plan_digest(value: RemediationPlan) -> str:
    return canonical_digest(_remediation_plan_contract(value))


def ranking_context_digest(value: RankingContext) -> str:
    return canonical_digest(_ranking_context_contract(value))


def seal_placement(value: Placement) -> Placement:
    unsigned = replace(value, placement_digest="")
    _validate_placement_fields(unsigned)
    sealed = replace(unsigned, placement_digest=placement_digest(unsigned))
    validate_placement(sealed)
    return sealed


def validate_placement(
    value: Placement,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_placement_fields(value)
    _validate_digest(value.placement_digest, placement_digest(value), "placement")
    _bind_authority_owner(
        value.authored_by,
        value.owner_merchant_id,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )


def seal_campaign(value: Campaign) -> Campaign:
    placements = tuple(seal_placement(item) for item in value.placements)
    provenance = _append_missing_digests(
        value.provenance_digests,
        tuple(item.placement_digest for item in placements),
    )
    unsigned = replace(
        value,
        placements=placements,
        provenance_digests=provenance,
        campaign_digest="",
    )
    _validate_campaign_fields(unsigned)
    sealed = replace(unsigned, campaign_digest=campaign_digest(unsigned))
    validate_campaign(sealed)
    return sealed


def validate_campaign(
    value: Campaign,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_campaign_fields(value)
    _validate_digest(value.campaign_digest, campaign_digest(value), "campaign")
    _bind_authority_owner(
        value.authored_by,
        value.owner_merchant_id,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )


def seal_review_evidence(value: ReviewEvidence) -> ReviewEvidence:
    unsigned = replace(value, evidence_digest="")
    _validate_review_evidence_fields(unsigned)
    sealed = replace(unsigned, evidence_digest=review_evidence_digest(unsigned))
    validate_review_evidence(sealed)
    return sealed


def validate_review_evidence(
    value: ReviewEvidence,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_review_evidence_fields(value)
    _validate_digest(value.evidence_digest, review_evidence_digest(value), "review evidence")
    _bind_authority_owner(
        value.authored_by,
        value.merchant_id,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )


def seal_review_aggregate(value: ReviewAggregate) -> ReviewAggregate:
    provenance = _append_missing_digests(value.provenance_digests, value.evidence_digests)
    unsigned = replace(value, provenance_digests=provenance, aggregate_digest="")
    _validate_review_aggregate_fields(unsigned)
    sealed = replace(unsigned, aggregate_digest=review_aggregate_digest(unsigned))
    validate_review_aggregate(sealed)
    return sealed


def validate_review_aggregate(
    value: ReviewAggregate,
    *,
    evidence: tuple[ReviewEvidence, ...] | None = None,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_review_aggregate_fields(value)
    _validate_digest(value.aggregate_digest, review_aggregate_digest(value), "review aggregate")
    _bind_authority_owner(
        value.authored_by,
        value.merchant_id,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    if evidence is not None:
        _validate_review_aggregate_evidence(value, evidence)


def seal_market_signal(value: MarketSignal) -> MarketSignal:
    unsigned = replace(value, signal_digest="")
    _validate_market_signal_fields(unsigned)
    sealed = replace(unsigned, signal_digest=market_signal_digest(unsigned))
    validate_market_signal(sealed)
    return sealed


def validate_market_signal(
    value: MarketSignal,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_market_signal_fields(value)
    _validate_digest(value.signal_digest, market_signal_digest(value), "market signal")
    _bind_authority_subjects(
        value.authored_by,
        value.subject_merchant_ids,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )


def seal_governance_case(value: GovernanceCase) -> GovernanceCase:
    provenance = _append_missing_digests(value.provenance_digests, value.signal_digests)
    unsigned = replace(value, provenance_digests=provenance, case_digest="")
    _validate_governance_case_fields(unsigned)
    sealed = replace(unsigned, case_digest=governance_case_digest(unsigned))
    validate_governance_case(sealed)
    return sealed


def validate_governance_case(
    value: GovernanceCase,
    *,
    signals: tuple[MarketSignal, ...] | None = None,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_governance_case_fields(value)
    _validate_digest(value.case_digest, governance_case_digest(value), "governance case")
    _bind_authority_subjects(
        value.authored_by,
        value.subject_merchant_ids,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    if signals is not None:
        _validate_case_signals(value, signals)


def seal_reputation_event(value: ReputationEvent) -> ReputationEvent:
    unsigned = replace(value, event_digest="")
    _validate_reputation_event_fields(unsigned)
    sealed = replace(unsigned, event_digest=reputation_event_digest(unsigned))
    validate_reputation_event(sealed)
    return sealed


def validate_reputation_event(
    value: ReputationEvent,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_reputation_event_fields(value)
    _validate_digest(value.event_digest, reputation_event_digest(value), "reputation event")
    _bind_authority_owner(
        value.authored_by,
        value.merchant_id,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )


def seal_remediation_step(value: RemediationStep) -> RemediationStep:
    unsigned = replace(value, step_digest="")
    _validate_remediation_step_fields(unsigned)
    sealed = replace(unsigned, step_digest=remediation_step_digest(unsigned))
    validate_remediation_step(sealed)
    return sealed


def validate_remediation_step(
    value: RemediationStep,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_remediation_step_fields(value)
    _validate_digest(value.step_digest, remediation_step_digest(value), "remediation step")
    _bind_authority_owner(
        value.authored_by,
        value.owner_merchant_id,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )


def seal_remediation_plan(value: RemediationPlan) -> RemediationPlan:
    steps = tuple(seal_remediation_step(item) for item in value.steps)
    provenance = _append_missing_digests(
        value.provenance_digests,
        tuple(item.step_digest for item in steps),
    )
    unsigned = replace(
        value,
        steps=steps,
        provenance_digests=provenance,
        plan_digest="",
    )
    _validate_remediation_plan_fields(unsigned)
    sealed = replace(unsigned, plan_digest=remediation_plan_digest(unsigned))
    validate_remediation_plan(sealed)
    return sealed


def validate_remediation_plan(
    value: RemediationPlan,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> None:
    _validate_remediation_plan_fields(value)
    _validate_digest(value.plan_digest, remediation_plan_digest(value), "remediation plan")
    _bind_authority_owner(
        value.authored_by,
        value.owner_merchant_id,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )


def seal_ranking_context(value: RankingContext, *, server_id: str) -> RankingContext:
    _require_text(server_id, "server_id")
    referenced = (
        value.placement_digests
        + value.review_aggregate_digests
        + value.governance_case_digests
        + value.reputation_event_digests
    )
    provenance = _append_missing_digests(value.provenance_digests, referenced)
    unsigned = replace(value, provenance_digests=provenance, context_digest="")
    _validate_ranking_context_fields(unsigned)
    if unsigned.authored_by != server_id:
        raise MarketGovernanceValidationError("ranking context server authority mismatch")
    sealed = replace(unsigned, context_digest=ranking_context_digest(unsigned))
    validate_ranking_context(sealed, server_id=server_id)
    return sealed


def validate_ranking_context(value: RankingContext, *, server_id: str) -> None:
    _require_text(server_id, "server_id")
    _validate_ranking_context_fields(value)
    _validate_digest(value.context_digest, ranking_context_digest(value), "ranking context")
    if value.authored_by != server_id:
        raise MarketGovernanceValidationError("ranking context server authority mismatch")


def _validate_placement_fields(value: Placement) -> None:
    for name in (
        "placement_id",
        "owner_merchant_id",
        "sku_id",
        "currency",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_optional_text(value.campaign_id, "campaign_id")
    _require_optional_text(value.disclosure_text, "disclosure_text")
    _require_enum(value.placement_kind, PLACEMENT_KINDS, "placement_kind")
    _require_enum(value.disclosure_status, DISCLOSURE_STATUSES, "disclosure_status")
    _require_nonnegative_int(value.bid_cents, "bid_cents")
    _require_nonnegative_int(value.fee_cents, "fee_cents")
    _validate_tick_range(value.starts_at_tick, value.ends_at_tick, "placement")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)
    if value.placement_kind == "organic":
        if value.campaign_id is not None or value.bid_cents != 0 or value.fee_cents != 0:
            raise SchemaError("organic placement cannot carry campaign, bid, or fee")
        if value.disclosure_status != "not_required" or value.disclosure_text is not None:
            raise SchemaError("organic placement disclosure must be not_required")
    else:
        if value.campaign_id is None:
            raise SchemaError("sponsored placement requires campaign_id")
        if value.disclosure_status == "not_required":
            raise SchemaError("sponsored placement requires disclosure")
        if value.disclosure_status == "disclosed" and value.disclosure_text is None:
            raise SchemaError("disclosed placement requires disclosure_text")
        if value.disclosure_status == "pending" and value.disclosure_text is not None:
            raise SchemaError("pending disclosure cannot carry disclosure_text")


def _validate_campaign_fields(value: Campaign) -> None:
    for name in (
        "campaign_id",
        "owner_merchant_id",
        "currency",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_enum(value.status, CAMPAIGN_STATUSES, "campaign status")
    _require_positive_int(value.budget_cents, "budget_cents")
    _validate_tick_range(value.starts_at_tick, value.ends_at_tick, "campaign")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)
    _require_tuple(value.placements, "placements")
    _require_unique((item.placement_id for item in value.placements), "placement_id")
    for item in value.placements:
        validate_placement(
            item,
            expected_authority=value.authored_by,
            expected_owner_id=value.owner_merchant_id,
        )
        if item.campaign_id != value.campaign_id:
            raise MarketGovernanceValidationError("placement campaign binding mismatch")
        if item.placement_kind != "sponsored":
            raise MarketGovernanceValidationError("campaign may contain only sponsored placements")
        if item.currency != value.currency:
            raise MarketGovernanceValidationError("placement currency binding mismatch")
        if item.starts_at_tick < value.starts_at_tick or item.ends_at_tick > value.ends_at_tick:
            raise MarketGovernanceValidationError("placement lies outside campaign lifetime")
        if item.placement_digest not in value.provenance_digests:
            raise MarketGovernanceValidationError("campaign provenance omits placement digest")
    if sum(item.fee_cents for item in value.placements) > value.budget_cents:
        raise MarketGovernanceValidationError("campaign fees exceed budget")


def _validate_review_evidence_fields(value: ReviewEvidence) -> None:
    for name in (
        "review_id",
        "sku_id",
        "merchant_id",
        "reviewer_id",
        "source_ref",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_optional_text(value.order_id, "order_id")
    _require_optional_text(value.burst_group_id, "burst_group_id")
    _require_int_range(value.rating, 1, 5, "rating")
    _require_bool(value.verified_purchase, "verified_purchase")
    _require_nonnegative_int(value.submitted_at_tick, "submitted_at_tick")
    _require_nonnegative_int(value.account_created_at_tick, "account_created_at_tick")
    if value.account_created_at_tick > value.submitted_at_tick:
        raise SchemaError("review account cannot be created after submission")
    if value.verified_purchase and value.order_id is None:
        raise SchemaError("verified review requires order_id")
    if not value.verified_purchase and value.order_id is not None:
        raise SchemaError("unverified review cannot claim order_id")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)


def _validate_review_aggregate_fields(value: ReviewAggregate) -> None:
    for name in (
        "aggregate_id",
        "sku_id",
        "merchant_id",
        "aggregation_policy_id",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_positive_int(value.aggregation_policy_version, "aggregation_policy_version")
    _validate_digest_tuple(value.evidence_digests, "evidence_digests")
    _require_nonnegative_int(value.review_count, "review_count")
    _require_nonnegative_int(value.verified_review_count, "verified_review_count")
    _require_nonnegative_int(value.rating_sum, "rating_sum")
    _require_nonnegative_int(value.verified_rating_sum, "verified_rating_sum")
    _require_nonnegative_int(value.computed_at_tick, "computed_at_tick")
    if value.review_count != len(value.evidence_digests):
        raise MarketGovernanceValidationError("review count does not match evidence digests")
    if value.verified_review_count > value.review_count:
        raise SchemaError("verified review count exceeds review count")
    if value.rating_sum < value.review_count or value.rating_sum > 5 * value.review_count:
        if value.review_count != 0 or value.rating_sum != 0:
            raise SchemaError("rating_sum is outside the 1..5 review range")
    if (
        value.verified_rating_sum < value.verified_review_count
        or value.verified_rating_sum > 5 * value.verified_review_count
    ):
        if value.verified_review_count != 0 or value.verified_rating_sum != 0:
            raise SchemaError("verified_rating_sum is outside the 1..5 review range")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)
    if not set(value.evidence_digests).issubset(value.provenance_digests):
        raise MarketGovernanceValidationError("aggregate provenance omits review evidence")


def _validate_review_aggregate_evidence(
    value: ReviewAggregate,
    evidence: tuple[ReviewEvidence, ...],
) -> None:
    _require_tuple(evidence, "evidence")
    for item in evidence:
        validate_review_evidence(item, expected_owner_id=value.merchant_id)
        if item.sku_id != value.sku_id:
            raise MarketGovernanceValidationError("review aggregate SKU binding mismatch")
    digests = tuple(item.evidence_digest for item in evidence)
    if digests != value.evidence_digests:
        raise MarketGovernanceValidationError("review aggregate evidence digest mismatch")
    verified = tuple(item for item in evidence if item.verified_purchase)
    expected = (
        len(evidence),
        len(verified),
        sum(item.rating for item in evidence),
        sum(item.rating for item in verified),
    )
    actual = (
        value.review_count,
        value.verified_review_count,
        value.rating_sum,
        value.verified_rating_sum,
    )
    if actual != expected:
        raise MarketGovernanceValidationError("review aggregate arithmetic mismatch")


def _validate_market_signal_fields(value: MarketSignal) -> None:
    for name in ("signal_id", "authored_by", "idempotency_key"):
        _require_text(getattr(value, name), name)
    _require_enum(value.signal_kind, MARKET_SIGNAL_KINDS, "signal_kind")
    _validate_text_tuple(value.subject_merchant_ids, "subject_merchant_ids", nonempty=True)
    _validate_text_tuple(value.source_refs, "source_refs", nonempty=True)
    _require_int_range(value.confidence_bps, 0, 10_000, "confidence_bps")
    _require_nonnegative_int(value.observed_at_tick, "observed_at_tick")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)


def _validate_governance_case_fields(value: GovernanceCase) -> None:
    for name in ("case_id", "authored_by", "idempotency_key"):
        _require_text(getattr(value, name), name)
    _require_enum(value.case_kind, GOVERNANCE_CASE_KINDS, "case_kind")
    _validate_text_tuple(value.subject_merchant_ids, "subject_merchant_ids", nonempty=True)
    _validate_digest_tuple(value.signal_digests, "signal_digests", nonempty=True)
    _require_enum(value.status, GOVERNANCE_CASE_STATUSES, "governance case status")
    _require_optional_text(value.resolution_code, "resolution_code")
    _require_nonnegative_int(value.opened_at_tick, "opened_at_tick")
    _require_nonnegative_int(value.updated_at_tick, "updated_at_tick")
    if value.updated_at_tick < value.opened_at_tick:
        raise SchemaError("governance case update predates opening")
    terminal = value.status in {"resolved", "dismissed"}
    if terminal != (value.resolution_code is not None):
        raise SchemaError("terminal governance case requires exactly one resolution_code")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)
    if not set(value.signal_digests).issubset(value.provenance_digests):
        raise MarketGovernanceValidationError("governance case provenance omits signal")


def _validate_case_signals(value: GovernanceCase, signals: tuple[MarketSignal, ...]) -> None:
    _require_tuple(signals, "signals")
    for item in signals:
        validate_market_signal(item)
    if tuple(item.signal_digest for item in signals) != value.signal_digests:
        raise MarketGovernanceValidationError("governance case signal digest mismatch")
    subjects: set[str] = set()
    for item in signals:
        subjects.update(item.subject_merchant_ids)
    if subjects != set(value.subject_merchant_ids):
        raise MarketGovernanceValidationError("governance case subject binding mismatch")


def _validate_reputation_event_fields(value: ReputationEvent) -> None:
    for name in (
        "event_id",
        "merchant_id",
        "source_ref",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_enum(value.event_kind, REPUTATION_EVENT_KINDS, "event_kind")
    _require_int_range(value.outcome_bps, 0, 10_000, "outcome_bps")
    _require_nonnegative_int(value.occurred_at_tick, "occurred_at_tick")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)


def _validate_remediation_step_fields(value: RemediationStep) -> None:
    for name in (
        "step_id",
        "plan_id",
        "owner_merchant_id",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_positive_int(value.sequence_no, "sequence_no")
    _require_enum(value.action_kind, REMEDIATION_ACTION_KINDS, "action_kind")
    _validate_text_tuple(value.prerequisite_step_ids, "prerequisite_step_ids")
    _require_enum(value.status, REMEDIATION_STEP_STATUSES, "remediation step status")
    _validate_text_tuple(value.evidence_refs, "evidence_refs")
    _require_optional_nonnegative_int(value.completed_at_tick, "completed_at_tick")
    if value.step_id in value.prerequisite_step_ids:
        raise SchemaError("remediation step cannot depend on itself")
    completed = value.status in {"completed", "verified"}
    if completed:
        if value.completed_at_tick is None or not value.evidence_refs or not value.provenance_digests:
            raise SchemaError("completed remediation step requires time and evidence")
    elif value.completed_at_tick is not None:
        raise SchemaError("non-completed remediation step cannot carry completion time")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)


def _validate_remediation_plan_fields(value: RemediationPlan) -> None:
    for name in (
        "plan_id",
        "governance_case_id",
        "owner_merchant_id",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_enum(value.status, REMEDIATION_PLAN_STATUSES, "remediation plan status")
    _require_nonnegative_int(value.created_at_tick, "created_at_tick")
    _require_nonnegative_int(value.updated_at_tick, "updated_at_tick")
    if value.updated_at_tick < value.created_at_tick:
        raise SchemaError("remediation plan update predates creation")
    _require_tuple(value.steps, "steps")
    if not value.steps:
        raise SchemaError("remediation plan requires at least one step")
    _require_unique((item.step_id for item in value.steps), "step_id")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)
    seen: set[str] = set()
    for expected_sequence, item in enumerate(value.steps, start=1):
        validate_remediation_step(
            item,
            expected_authority=value.authored_by,
            expected_owner_id=value.owner_merchant_id,
        )
        if item.plan_id != value.plan_id:
            raise MarketGovernanceValidationError("remediation step plan binding mismatch")
        if item.sequence_no != expected_sequence:
            raise MarketGovernanceValidationError("remediation step sequence must be contiguous")
        if not set(item.prerequisite_step_ids).issubset(seen):
            raise MarketGovernanceValidationError("remediation prerequisite must reference earlier step")
        if item.step_digest not in value.provenance_digests:
            raise MarketGovernanceValidationError("remediation plan provenance omits step")
        seen.add(item.step_id)
    if value.status == "completed" and any(item.status != "verified" for item in value.steps):
        raise MarketGovernanceValidationError("completed remediation plan requires verified steps")


def _validate_ranking_context_fields(value: RankingContext) -> None:
    for name in (
        "context_id",
        "request_id",
        "buyer_id",
        "mandate_id",
        "policy_id",
        "authored_by",
        "idempotency_key",
    ):
        _require_text(getattr(value, name), name)
    _require_positive_int(value.policy_version, "policy_version")
    _require_nonnegative_int(value.catalog_revision, "catalog_revision")
    _require_nonnegative_int(value.issued_at_tick, "issued_at_tick")
    _validate_text_tuple(value.candidate_sku_ids, "candidate_sku_ids", nonempty=True)
    _validate_text_tuple(value.ranked_sku_ids, "ranked_sku_ids", nonempty=True)
    if set(value.candidate_sku_ids) != set(value.ranked_sku_ids):
        raise MarketGovernanceValidationError("ranked SKUs must be an exact candidate permutation")
    referenced = (
        value.placement_digests,
        value.review_aggregate_digests,
        value.governance_case_digests,
        value.reputation_event_digests,
    )
    for name, digests in zip(
        (
            "placement_digests",
            "review_aggregate_digests",
            "governance_case_digests",
            "reputation_event_digests",
        ),
        referenced,
        strict=True,
    ):
        _validate_digest_tuple(digests, name)
    flattened = tuple(item for values in referenced for item in values)
    _require_unique(flattened, "ranking evidence digest")
    _validate_version_fields(value.version, value.previous_digest, value.provenance_digests)
    if not set(flattened).issubset(value.provenance_digests):
        raise MarketGovernanceValidationError("ranking context provenance omits evidence")


def validate_version_successor(
    previous: GovernanceRecord,
    current: GovernanceRecord,
    *,
    expected_authority: str,
    expected_owner_id: str | None = None,
) -> None:
    """Validate one append-only revision against its sealed predecessor."""

    _require_text(expected_authority, "expected_authority")
    if type(previous) is not type(current):
        raise MarketGovernanceValidationError("record type changed across versions")
    if isinstance(previous, ReputationEvent) and isinstance(current, ReputationEvent):
        validate_reputation_successor(
            previous,
            current,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return
    _validate_record(
        previous,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    _validate_record(
        current,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    if _record_id(previous) != _record_id(current):
        raise MarketGovernanceValidationError("record identity changed across versions")
    if _record_owner(previous) != _record_owner(current):
        raise MarketGovernanceValidationError("record owner changed across versions")
    if current.version != previous.version + 1:
        raise MarketGovernanceValidationError("record version is not contiguous")
    if current.previous_digest != _record_digest(previous):
        raise MarketGovernanceValidationError("record predecessor digest mismatch")
    if current.provenance_digests[: len(previous.provenance_digests)] != previous.provenance_digests:
        raise MarketGovernanceValidationError("record provenance is not append-only")
    if current.idempotency_key == previous.idempotency_key:
        raise IdempotencyConflict("new record version reused predecessor idempotency key")
    _validate_domain_successor(previous, current)


def validate_reputation_successor(
    previous: ReputationEvent,
    current: ReputationEvent,
    *,
    expected_authority: str,
    expected_owner_id: str | None = None,
) -> None:
    """Validate the next immutable event in one merchant reputation stream."""

    validate_reputation_event(
        previous,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    validate_reputation_event(
        current,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    if current.merchant_id != previous.merchant_id:
        raise MarketGovernanceValidationError("reputation stream merchant changed")
    if current.event_id == previous.event_id:
        raise MarketGovernanceValidationError("reputation event_id must be append-only")
    if current.version != previous.version + 1:
        raise MarketGovernanceValidationError("reputation stream version is not contiguous")
    if current.previous_digest != previous.event_digest:
        raise MarketGovernanceValidationError("reputation predecessor digest mismatch")
    if current.provenance_digests[: len(previous.provenance_digests)] != previous.provenance_digests:
        raise MarketGovernanceValidationError("reputation provenance is not append-only")
    if current.occurred_at_tick < previous.occurred_at_tick:
        raise MarketGovernanceValidationError("reputation event time moved backwards")
    if current.idempotency_key == previous.idempotency_key:
        raise IdempotencyConflict("new reputation event reused predecessor idempotency key")


def replay_exact_or_conflict(
    existing: GovernanceRecord,
    proposed: GovernanceRecord,
    *,
    expected_authority: str,
    expected_owner_id: str | None = None,
) -> GovernanceRecord:
    """Return a persisted exact retry, or reject changed content under its key."""

    if type(existing) is not type(proposed):
        raise IdempotencyConflict("idempotency key reused across record types")
    _validate_record(
        existing,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    _validate_record(
        proposed,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    if existing.idempotency_key != proposed.idempotency_key:
        raise IdempotencyConflict("idempotency lookup key mismatch")
    if _record_id(existing) != _record_id(proposed):
        raise IdempotencyConflict("idempotency key reused for another record")
    if _record_digest(existing) != _record_digest(proposed):
        raise IdempotencyConflict("record changed under an existing idempotency key")
    return existing


def _validate_domain_successor(previous: GovernanceRecord, current: GovernanceRecord) -> None:
    if isinstance(previous, Placement) and isinstance(current, Placement):
        _same_fields(
            previous,
            current,
            (
                "campaign_id",
                "owner_merchant_id",
                "sku_id",
                "placement_kind",
                "currency",
                "starts_at_tick",
                "ends_at_tick",
            ),
        )
        allowed = {
            "not_required": {"not_required"},
            "pending": {"pending", "disclosed"},
            "disclosed": {"disclosed"},
        }
        _require_transition(
            previous.disclosure_status,
            current.disclosure_status,
            allowed,
            "placement disclosure",
        )
        return
    if isinstance(previous, Campaign) and isinstance(current, Campaign):
        _same_fields(
            previous,
            current,
            ("owner_merchant_id", "currency", "starts_at_tick", "ends_at_tick"),
        )
        _require_transition(
            previous.status,
            current.status,
            {
                "draft": {"draft", "active", "closed"},
                "active": {"active", "paused", "closed"},
                "paused": {"paused", "active", "closed"},
                "closed": {"closed"},
            },
            "campaign",
        )
        _validate_child_successors(
            previous.placements,
            current.placements,
            expected_authority=previous.authored_by,
            expected_owner_id=previous.owner_merchant_id,
            label="campaign placement",
        )
        return
    if isinstance(previous, ReviewEvidence) and isinstance(current, ReviewEvidence):
        _same_fields(
            previous,
            current,
            (
                "sku_id",
                "merchant_id",
                "reviewer_id",
                "rating",
                "verified_purchase",
                "order_id",
                "submitted_at_tick",
                "account_created_at_tick",
                "burst_group_id",
                "source_ref",
            ),
        )
        return
    if isinstance(previous, ReviewAggregate) and isinstance(current, ReviewAggregate):
        _same_fields(
            previous,
            current,
            ("sku_id", "merchant_id", "aggregation_policy_id"),
        )
        if current.aggregation_policy_version < previous.aggregation_policy_version:
            raise MarketGovernanceValidationError("review policy version moved backwards")
        _require_prefix(previous.evidence_digests, current.evidence_digests, "review evidence")
        for field in (
            "review_count",
            "verified_review_count",
            "rating_sum",
            "verified_rating_sum",
            "computed_at_tick",
        ):
            if getattr(current, field) < getattr(previous, field):
                raise MarketGovernanceValidationError(f"{field} moved backwards")
        return
    if isinstance(previous, MarketSignal) and isinstance(current, MarketSignal):
        _same_fields(
            previous,
            current,
            ("signal_kind", "subject_merchant_ids", "observed_at_tick"),
        )
        _require_prefix(previous.source_refs, current.source_refs, "market signal sources")
        return
    if isinstance(previous, GovernanceCase) and isinstance(current, GovernanceCase):
        _same_fields(
            previous,
            current,
            ("case_kind", "subject_merchant_ids", "opened_at_tick"),
        )
        _require_prefix(previous.signal_digests, current.signal_digests, "governance signals")
        if current.updated_at_tick < previous.updated_at_tick:
            raise MarketGovernanceValidationError("governance update time moved backwards")
        _require_transition(
            previous.status,
            current.status,
            {
                "open": {"open", "under_review", "resolved", "dismissed"},
                "under_review": {"under_review", "resolved", "dismissed"},
                "resolved": {"resolved"},
                "dismissed": {"dismissed"},
            },
            "governance case",
        )
        return
    if isinstance(previous, RemediationStep) and isinstance(current, RemediationStep):
        _same_fields(
            previous,
            current,
            (
                "plan_id",
                "owner_merchant_id",
                "sequence_no",
                "action_kind",
                "prerequisite_step_ids",
            ),
        )
        _require_prefix(previous.evidence_refs, current.evidence_refs, "remediation evidence")
        _require_transition(
            previous.status,
            current.status,
            {
                "pending": {"pending", "completed", "rejected"},
                "completed": {"completed", "verified", "rejected"},
                "verified": {"verified"},
                "rejected": {"rejected"},
            },
            "remediation step",
        )
        return
    if isinstance(previous, RemediationPlan) and isinstance(current, RemediationPlan):
        _same_fields(
            previous,
            current,
            ("governance_case_id", "owner_merchant_id", "created_at_tick"),
        )
        if current.updated_at_tick < previous.updated_at_tick:
            raise MarketGovernanceValidationError("remediation update time moved backwards")
        _require_transition(
            previous.status,
            current.status,
            {
                "draft": {"draft", "active", "terminated"},
                "active": {"active", "completed", "terminated"},
                "completed": {"completed"},
                "terminated": {"terminated"},
            },
            "remediation plan",
        )
        _validate_child_successors(
            previous.steps,
            current.steps,
            expected_authority=previous.authored_by,
            expected_owner_id=previous.owner_merchant_id,
            label="remediation step",
        )
        return
    if isinstance(previous, RankingContext) and isinstance(current, RankingContext):
        _same_fields(
            previous,
            current,
            ("request_id", "buyer_id", "mandate_id", "policy_id", "candidate_sku_ids"),
        )
        if current.policy_version < previous.policy_version:
            raise MarketGovernanceValidationError("ranking policy version moved backwards")
        if current.catalog_revision < previous.catalog_revision:
            raise MarketGovernanceValidationError("ranking catalog revision moved backwards")
        if current.issued_at_tick < previous.issued_at_tick:
            raise MarketGovernanceValidationError("ranking issuance time moved backwards")
        for label, old, new in (
            ("placement evidence", previous.placement_digests, current.placement_digests),
            (
                "review aggregate evidence",
                previous.review_aggregate_digests,
                current.review_aggregate_digests,
            ),
            (
                "governance case evidence",
                previous.governance_case_digests,
                current.governance_case_digests,
            ),
            (
                "reputation evidence",
                previous.reputation_event_digests,
                current.reputation_event_digests,
            ),
        ):
            _require_prefix(old, new, label)
        return
    raise TypeError(f"unsupported governance successor type {type(previous).__name__}")


def _validate_child_successors(
    previous: tuple[GovernanceRecord, ...],
    current: tuple[GovernanceRecord, ...],
    *,
    expected_authority: str,
    expected_owner_id: str,
    label: str,
) -> None:
    if len(current) < len(previous):
        raise MarketGovernanceValidationError(f"{label} records were removed")
    for old, new in zip(previous, current, strict=False):
        if _record_id(old) != _record_id(new):
            raise MarketGovernanceValidationError(f"{label} order changed")
        if _record_digest(old) != _record_digest(new):
            validate_version_successor(
                old,
                new,
                expected_authority=expected_authority,
                expected_owner_id=expected_owner_id,
            )


def _validate_record(
    value: GovernanceRecord,
    *,
    expected_authority: str,
    expected_owner_id: str | None,
) -> None:
    if isinstance(value, Placement):
        validate_placement(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, Campaign):
        validate_campaign(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, ReviewEvidence):
        validate_review_evidence(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, ReviewAggregate):
        validate_review_aggregate(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, MarketSignal):
        validate_market_signal(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, GovernanceCase):
        validate_governance_case(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, ReputationEvent):
        validate_reputation_event(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, RemediationStep):
        validate_remediation_step(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, RemediationPlan):
        validate_remediation_plan(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
    elif isinstance(value, RankingContext):
        validate_ranking_context(value, server_id=expected_authority)
        if expected_owner_id is not None and value.buyer_id != expected_owner_id:
            raise MarketGovernanceValidationError("record owner binding mismatch")
    else:  # pragma: no cover - exhaustive union guard
        raise TypeError(f"unsupported governance record {type(value).__name__}")


def _record_id(value: GovernanceRecord) -> str:
    for field in (
        "placement_id",
        "campaign_id",
        "review_id",
        "aggregate_id",
        "signal_id",
        "case_id",
        "event_id",
        "step_id",
        "plan_id",
        "context_id",
    ):
        candidate = getattr(value, field, None)
        if isinstance(candidate, str):
            return candidate
    raise TypeError(f"record {type(value).__name__} has no identity")


def _record_owner(value: GovernanceRecord) -> str | tuple[str, ...]:
    if isinstance(value, RankingContext):
        return value.buyer_id
    if isinstance(value, (MarketSignal, GovernanceCase)):
        return value.subject_merchant_ids
    for field in ("owner_merchant_id", "merchant_id"):
        candidate = getattr(value, field, None)
        if isinstance(candidate, str):
            return candidate
    raise TypeError(f"record {type(value).__name__} has no owner")


def _record_digest(value: GovernanceRecord) -> str:
    for field in (
        "placement_digest",
        "campaign_digest",
        "evidence_digest",
        "aggregate_digest",
        "signal_digest",
        "case_digest",
        "event_digest",
        "step_digest",
        "plan_digest",
        "context_digest",
    ):
        candidate = getattr(value, field, None)
        if isinstance(candidate, str):
            return candidate
    raise TypeError(f"record {type(value).__name__} has no digest")


# Explicit wire key sets make schema changes reviewable.  They intentionally
# include nullable fields; omission and null are not equivalent on the wire.
_WIRE_KEYS: dict[str, frozenset[str]] = {
    "placement": frozenset(
        {
            "schema_version",
            "placement_id",
            "campaign_id",
            "owner_merchant_id",
            "sku_id",
            "placement_kind",
            "disclosure_status",
            "disclosure_text",
            "bid_cents",
            "fee_cents",
            "currency",
            "starts_at_tick",
            "ends_at_tick",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "placement_digest",
        }
    ),
    "campaign": frozenset(
        {
            "schema_version",
            "campaign_id",
            "owner_merchant_id",
            "status",
            "budget_cents",
            "currency",
            "starts_at_tick",
            "ends_at_tick",
            "placements",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "campaign_digest",
        }
    ),
    "review evidence": frozenset(
        {
            "schema_version",
            "review_id",
            "sku_id",
            "merchant_id",
            "reviewer_id",
            "rating",
            "verified_purchase",
            "order_id",
            "submitted_at_tick",
            "account_created_at_tick",
            "burst_group_id",
            "source_ref",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "evidence_digest",
        }
    ),
    "review aggregate": frozenset(
        {
            "schema_version",
            "aggregate_id",
            "sku_id",
            "merchant_id",
            "aggregation_policy_id",
            "aggregation_policy_version",
            "evidence_digests",
            "review_count",
            "verified_review_count",
            "rating_sum",
            "verified_rating_sum",
            "computed_at_tick",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "aggregate_digest",
        }
    ),
    "market signal": frozenset(
        {
            "schema_version",
            "signal_id",
            "signal_kind",
            "subject_merchant_ids",
            "source_refs",
            "confidence_bps",
            "observed_at_tick",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "signal_digest",
        }
    ),
    "governance case": frozenset(
        {
            "schema_version",
            "case_id",
            "case_kind",
            "subject_merchant_ids",
            "signal_digests",
            "status",
            "resolution_code",
            "opened_at_tick",
            "updated_at_tick",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "case_digest",
        }
    ),
    "reputation event": frozenset(
        {
            "schema_version",
            "event_id",
            "merchant_id",
            "event_kind",
            "source_ref",
            "outcome_bps",
            "occurred_at_tick",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "event_digest",
        }
    ),
    "remediation step": frozenset(
        {
            "schema_version",
            "step_id",
            "plan_id",
            "owner_merchant_id",
            "sequence_no",
            "action_kind",
            "prerequisite_step_ids",
            "status",
            "evidence_refs",
            "completed_at_tick",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "step_digest",
        }
    ),
    "remediation plan": frozenset(
        {
            "schema_version",
            "plan_id",
            "governance_case_id",
            "owner_merchant_id",
            "status",
            "created_at_tick",
            "updated_at_tick",
            "steps",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "plan_digest",
        }
    ),
    "ranking context": frozenset(
        {
            "schema_version",
            "context_id",
            "request_id",
            "buyer_id",
            "mandate_id",
            "policy_id",
            "policy_version",
            "catalog_revision",
            "issued_at_tick",
            "candidate_sku_ids",
            "ranked_sku_ids",
            "placement_digests",
            "review_aggregate_digests",
            "governance_case_digests",
            "reputation_event_digests",
            "authored_by",
            "idempotency_key",
            "version",
            "previous_digest",
            "provenance_digests",
            "context_digest",
        }
    ),
}


def placement_to_wire(value: Placement) -> dict[str, Any]:
    validate_placement(value)
    return {**_placement_contract(value), "placement_digest": value.placement_digest}


def campaign_to_wire(value: Campaign) -> dict[str, Any]:
    validate_campaign(value)
    return {**_campaign_contract(value), "campaign_digest": value.campaign_digest}


def review_evidence_to_wire(value: ReviewEvidence) -> dict[str, Any]:
    validate_review_evidence(value)
    return {**_review_evidence_contract(value), "evidence_digest": value.evidence_digest}


def review_aggregate_to_wire(value: ReviewAggregate) -> dict[str, Any]:
    validate_review_aggregate(value)
    return {**_review_aggregate_contract(value), "aggregate_digest": value.aggregate_digest}


def market_signal_to_wire(value: MarketSignal) -> dict[str, Any]:
    validate_market_signal(value)
    return {**_market_signal_contract(value), "signal_digest": value.signal_digest}


def governance_case_to_wire(value: GovernanceCase) -> dict[str, Any]:
    validate_governance_case(value)
    return {**_governance_case_contract(value), "case_digest": value.case_digest}


def reputation_event_to_wire(value: ReputationEvent) -> dict[str, Any]:
    validate_reputation_event(value)
    return {**_reputation_event_contract(value), "event_digest": value.event_digest}


def remediation_step_to_wire(value: RemediationStep) -> dict[str, Any]:
    validate_remediation_step(value)
    return {**_remediation_step_contract(value), "step_digest": value.step_digest}


def remediation_plan_to_wire(value: RemediationPlan) -> dict[str, Any]:
    validate_remediation_plan(value)
    return {**_remediation_plan_contract(value), "plan_digest": value.plan_digest}


def ranking_context_to_wire(value: RankingContext, *, server_id: str) -> dict[str, Any]:
    validate_ranking_context(value, server_id=server_id)
    return {**_ranking_context_contract(value), "context_digest": value.context_digest}


def coerce_placement(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> Placement:
    if isinstance(value, Placement):
        validate_placement(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "placement", PLACEMENT_SCHEMA)
    result = Placement(
        placement_id=_wire_text(row, "placement_id"),
        campaign_id=_wire_optional_text(row, "campaign_id"),
        owner_merchant_id=_wire_text(row, "owner_merchant_id"),
        sku_id=_wire_text(row, "sku_id"),
        placement_kind=_wire_text(row, "placement_kind"),
        disclosure_status=_wire_text(row, "disclosure_status"),
        disclosure_text=_wire_optional_text(row, "disclosure_text"),
        bid_cents=_wire_int(row, "bid_cents"),
        fee_cents=_wire_int(row, "fee_cents"),
        currency=_wire_text(row, "currency"),
        starts_at_tick=_wire_int(row, "starts_at_tick"),
        ends_at_tick=_wire_int(row, "ends_at_tick"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        placement_digest=_wire_text(row, "placement_digest"),
    )
    validate_placement(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_campaign(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> Campaign:
    if isinstance(value, Campaign):
        validate_campaign(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "campaign", CAMPAIGN_SCHEMA)
    result = Campaign(
        campaign_id=_wire_text(row, "campaign_id"),
        owner_merchant_id=_wire_text(row, "owner_merchant_id"),
        status=_wire_text(row, "status"),
        budget_cents=_wire_int(row, "budget_cents"),
        currency=_wire_text(row, "currency"),
        starts_at_tick=_wire_int(row, "starts_at_tick"),
        ends_at_tick=_wire_int(row, "ends_at_tick"),
        placements=tuple(coerce_placement(item) for item in _wire_list(row, "placements")),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        campaign_digest=_wire_text(row, "campaign_digest"),
    )
    validate_campaign(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_review_evidence(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> ReviewEvidence:
    if isinstance(value, ReviewEvidence):
        validate_review_evidence(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "review evidence", REVIEW_EVIDENCE_SCHEMA)
    result = ReviewEvidence(
        review_id=_wire_text(row, "review_id"),
        sku_id=_wire_text(row, "sku_id"),
        merchant_id=_wire_text(row, "merchant_id"),
        reviewer_id=_wire_text(row, "reviewer_id"),
        rating=_wire_int(row, "rating"),
        verified_purchase=_wire_bool(row, "verified_purchase"),
        order_id=_wire_optional_text(row, "order_id"),
        submitted_at_tick=_wire_int(row, "submitted_at_tick"),
        account_created_at_tick=_wire_int(row, "account_created_at_tick"),
        burst_group_id=_wire_optional_text(row, "burst_group_id"),
        source_ref=_wire_text(row, "source_ref"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        evidence_digest=_wire_text(row, "evidence_digest"),
    )
    validate_review_evidence(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_review_aggregate(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> ReviewAggregate:
    if isinstance(value, ReviewAggregate):
        validate_review_aggregate(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "review aggregate", REVIEW_AGGREGATE_SCHEMA)
    result = ReviewAggregate(
        aggregate_id=_wire_text(row, "aggregate_id"),
        sku_id=_wire_text(row, "sku_id"),
        merchant_id=_wire_text(row, "merchant_id"),
        aggregation_policy_id=_wire_text(row, "aggregation_policy_id"),
        aggregation_policy_version=_wire_int(row, "aggregation_policy_version"),
        evidence_digests=_wire_digest_tuple(row, "evidence_digests"),
        review_count=_wire_int(row, "review_count"),
        verified_review_count=_wire_int(row, "verified_review_count"),
        rating_sum=_wire_int(row, "rating_sum"),
        verified_rating_sum=_wire_int(row, "verified_rating_sum"),
        computed_at_tick=_wire_int(row, "computed_at_tick"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        aggregate_digest=_wire_text(row, "aggregate_digest"),
    )
    validate_review_aggregate(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_market_signal(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> MarketSignal:
    if isinstance(value, MarketSignal):
        validate_market_signal(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "market signal", MARKET_SIGNAL_SCHEMA)
    result = MarketSignal(
        signal_id=_wire_text(row, "signal_id"),
        signal_kind=_wire_text(row, "signal_kind"),
        subject_merchant_ids=_wire_text_tuple(row, "subject_merchant_ids"),
        source_refs=_wire_text_tuple(row, "source_refs"),
        confidence_bps=_wire_int(row, "confidence_bps"),
        observed_at_tick=_wire_int(row, "observed_at_tick"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        signal_digest=_wire_text(row, "signal_digest"),
    )
    validate_market_signal(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_governance_case(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> GovernanceCase:
    if isinstance(value, GovernanceCase):
        validate_governance_case(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "governance case", GOVERNANCE_CASE_SCHEMA)
    result = GovernanceCase(
        case_id=_wire_text(row, "case_id"),
        case_kind=_wire_text(row, "case_kind"),
        subject_merchant_ids=_wire_text_tuple(row, "subject_merchant_ids"),
        signal_digests=_wire_digest_tuple(row, "signal_digests"),
        status=_wire_text(row, "status"),
        resolution_code=_wire_optional_text(row, "resolution_code"),
        opened_at_tick=_wire_int(row, "opened_at_tick"),
        updated_at_tick=_wire_int(row, "updated_at_tick"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        case_digest=_wire_text(row, "case_digest"),
    )
    validate_governance_case(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_reputation_event(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> ReputationEvent:
    if isinstance(value, ReputationEvent):
        validate_reputation_event(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "reputation event", REPUTATION_EVENT_SCHEMA)
    result = ReputationEvent(
        event_id=_wire_text(row, "event_id"),
        merchant_id=_wire_text(row, "merchant_id"),
        event_kind=_wire_text(row, "event_kind"),
        source_ref=_wire_text(row, "source_ref"),
        outcome_bps=_wire_int(row, "outcome_bps"),
        occurred_at_tick=_wire_int(row, "occurred_at_tick"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        event_digest=_wire_text(row, "event_digest"),
    )
    validate_reputation_event(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_remediation_step(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> RemediationStep:
    if isinstance(value, RemediationStep):
        validate_remediation_step(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "remediation step", REMEDIATION_STEP_SCHEMA)
    result = RemediationStep(
        step_id=_wire_text(row, "step_id"),
        plan_id=_wire_text(row, "plan_id"),
        owner_merchant_id=_wire_text(row, "owner_merchant_id"),
        sequence_no=_wire_int(row, "sequence_no"),
        action_kind=_wire_text(row, "action_kind"),
        prerequisite_step_ids=_wire_text_tuple(row, "prerequisite_step_ids"),
        status=_wire_text(row, "status"),
        evidence_refs=_wire_text_tuple(row, "evidence_refs"),
        completed_at_tick=_wire_optional_int(row, "completed_at_tick"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        step_digest=_wire_text(row, "step_digest"),
    )
    validate_remediation_step(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_remediation_plan(
    value: Any,
    *,
    expected_authority: str | None = None,
    expected_owner_id: str | None = None,
) -> RemediationPlan:
    if isinstance(value, RemediationPlan):
        validate_remediation_plan(
            value,
            expected_authority=expected_authority,
            expected_owner_id=expected_owner_id,
        )
        return value
    row = _strict_row(value, "remediation plan", REMEDIATION_PLAN_SCHEMA)
    result = RemediationPlan(
        plan_id=_wire_text(row, "plan_id"),
        governance_case_id=_wire_text(row, "governance_case_id"),
        owner_merchant_id=_wire_text(row, "owner_merchant_id"),
        status=_wire_text(row, "status"),
        created_at_tick=_wire_int(row, "created_at_tick"),
        updated_at_tick=_wire_int(row, "updated_at_tick"),
        steps=tuple(coerce_remediation_step(item) for item in _wire_list(row, "steps")),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        plan_digest=_wire_text(row, "plan_digest"),
    )
    validate_remediation_plan(
        result,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner_id,
    )
    return result


def coerce_ranking_context(value: Any, *, server_id: str) -> RankingContext:
    if isinstance(value, RankingContext):
        validate_ranking_context(value, server_id=server_id)
        return value
    row = _strict_row(value, "ranking context", RANKING_CONTEXT_SCHEMA)
    result = RankingContext(
        context_id=_wire_text(row, "context_id"),
        request_id=_wire_text(row, "request_id"),
        buyer_id=_wire_text(row, "buyer_id"),
        mandate_id=_wire_text(row, "mandate_id"),
        policy_id=_wire_text(row, "policy_id"),
        policy_version=_wire_int(row, "policy_version"),
        catalog_revision=_wire_int(row, "catalog_revision"),
        issued_at_tick=_wire_int(row, "issued_at_tick"),
        candidate_sku_ids=_wire_text_tuple(row, "candidate_sku_ids"),
        ranked_sku_ids=_wire_text_tuple(row, "ranked_sku_ids"),
        placement_digests=_wire_digest_tuple(row, "placement_digests"),
        review_aggregate_digests=_wire_digest_tuple(row, "review_aggregate_digests"),
        governance_case_digests=_wire_digest_tuple(row, "governance_case_digests"),
        reputation_event_digests=_wire_digest_tuple(row, "reputation_event_digests"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        version=_wire_int(row, "version"),
        previous_digest=_wire_optional_text(row, "previous_digest"),
        provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
        context_digest=_wire_text(row, "context_digest"),
    )
    validate_ranking_context(result, server_id=server_id)
    return result


def _strict_row(value: Any, label: str, schema: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    actual = frozenset(value.keys())
    expected = _WIRE_KEYS[label]
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        raise SchemaError(f"{label} has invalid fields: {', '.join(details)}")
    if value.get("schema_version") != schema:
        raise SchemaError(f"unsupported schema_version for {schema}")
    return value


def _wire_text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    _require_text(value, key)
    return value


def _wire_optional_text(row: Mapping[str, Any], key: str) -> str | None:
    value = row[key]
    _require_optional_text(value, key)
    return value


def _wire_int(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{key} must be an integer")
    return value


def _wire_optional_int(row: Mapping[str, Any], key: str) -> int | None:
    value = row[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{key} must be an integer or null")
    return value


def _wire_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row[key]
    _require_bool(value, key)
    return value


def _wire_list(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row[key]
    if not isinstance(value, list):
        raise SchemaError(f"{key} must be an array")
    return value


def _wire_text_tuple(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = _wire_list(row, key)
    for value in values:
        _require_text(value, f"{key} item")
    return tuple(values)


def _wire_digest_tuple(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = _wire_text_tuple(row, key)
    for value in values:
        _require_digest(value, f"{key} item")
    return values


def _require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{label} must be a non-empty string")


def _require_optional_text(value: Any, label: str) -> None:
    if value is not None:
        _require_text(value, label)


def _require_bool(value: Any, label: str) -> None:
    if not isinstance(value, bool):
        raise SchemaError(f"{label} must be a boolean")


def _require_positive_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")


def _require_optional_nonnegative_int(value: Any, label: str) -> None:
    if value is not None:
        _require_nonnegative_int(value, label)


def _require_int_range(value: Any, minimum: int, maximum: int, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise SchemaError(f"{label} must be an integer in [{minimum}, {maximum}]")


def _require_enum(value: Any, allowed: frozenset[str], label: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise SchemaError(f"unknown {label} {value!r}")


def _require_tuple(value: Any, label: str) -> None:
    if not isinstance(value, tuple):
        raise SchemaError(f"{label} must be a tuple")


def _require_unique(values: Iterable[Any], label: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise SchemaError(f"duplicate {label} {value!r}")
        seen.add(value)


def _validate_text_tuple(
    value: tuple[str, ...],
    label: str,
    *,
    nonempty: bool = False,
) -> None:
    _require_tuple(value, label)
    if nonempty and not value:
        raise SchemaError(f"{label} must not be empty")
    for item in value:
        _require_text(item, f"{label} item")
    _require_unique(value, label)


def _require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SchemaError(f"{label} must be a lowercase SHA-256 hex digest")


def _validate_digest_tuple(
    value: tuple[str, ...],
    label: str,
    *,
    nonempty: bool = False,
) -> None:
    _require_tuple(value, label)
    if nonempty and not value:
        raise SchemaError(f"{label} must not be empty")
    for item in value:
        _require_digest(item, f"{label} item")
    _require_unique(value, label)


def _validate_digest(actual: str, expected: str, label: str) -> None:
    _require_digest(actual, f"{label}_digest")
    if actual != expected:
        raise MarketGovernanceValidationError(f"{label} digest mismatch")


def _validate_tick_range(start: Any, end: Any, label: str) -> None:
    _require_nonnegative_int(start, f"{label} starts_at_tick")
    _require_nonnegative_int(end, f"{label} ends_at_tick")
    if end <= start:
        raise SchemaError(f"{label} end must be after start")


def _validate_version_fields(
    version: Any,
    previous_digest: Any,
    provenance_digests: Any,
) -> None:
    _require_positive_int(version, "version")
    _require_optional_text(previous_digest, "previous_digest")
    if version == 1 and previous_digest is not None:
        raise SchemaError("version 1 record cannot have previous_digest")
    if version > 1:
        _require_digest(previous_digest, "previous_digest")
    _validate_digest_tuple(provenance_digests, "provenance_digests")


def _append_missing_digests(
    existing: tuple[str, ...],
    referenced: tuple[str, ...],
) -> tuple[str, ...]:
    _validate_digest_tuple(existing, "provenance_digests")
    _validate_digest_tuple(referenced, "referenced digests")
    result = list(existing)
    known = set(existing)
    for digest in referenced:
        if digest not in known:
            result.append(digest)
            known.add(digest)
    return tuple(result)


def _bind_authority_owner(
    authored_by: str,
    owner_id: str,
    *,
    expected_authority: str | None,
    expected_owner_id: str | None,
) -> None:
    if expected_authority is not None and authored_by != expected_authority:
        raise MarketGovernanceValidationError("record authority binding mismatch")
    if expected_owner_id is not None and owner_id != expected_owner_id:
        raise MarketGovernanceValidationError("record owner binding mismatch")


def _bind_authority_subjects(
    authored_by: str,
    subject_ids: tuple[str, ...],
    *,
    expected_authority: str | None,
    expected_owner_id: str | None,
) -> None:
    if expected_authority is not None and authored_by != expected_authority:
        raise MarketGovernanceValidationError("record authority binding mismatch")
    if expected_owner_id is not None and expected_owner_id not in subject_ids:
        raise MarketGovernanceValidationError("record owner is not a bound subject")


def _same_fields(previous: Any, current: Any, fields: tuple[str, ...]) -> None:
    for field in fields:
        if getattr(previous, field) != getattr(current, field):
            raise MarketGovernanceValidationError(f"{field} changed across versions")


def _require_transition(
    previous: str,
    current: str,
    allowed: Mapping[str, set[str]],
    label: str,
) -> None:
    if current not in allowed[previous]:
        raise MarketGovernanceValidationError(
            f"invalid {label} transition {previous!r} -> {current!r}"
        )


def _require_prefix(previous: tuple[Any, ...], current: tuple[Any, ...], label: str) -> None:
    if current[: len(previous)] != previous:
        raise MarketGovernanceValidationError(f"{label} is not append-only")


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_STATUSES",
    "DISCLOSURE_STATUSES",
    "GOVERNANCE_CASE_KINDS",
    "GOVERNANCE_CASE_SCHEMA",
    "GOVERNANCE_CASE_STATUSES",
    "MARKET_SIGNAL_KINDS",
    "MARKET_SIGNAL_SCHEMA",
    "PLACEMENT_KINDS",
    "PLACEMENT_SCHEMA",
    "RANKING_CONTEXT_SCHEMA",
    "REMEDIATION_ACTION_KINDS",
    "REMEDIATION_PLAN_SCHEMA",
    "REMEDIATION_PLAN_STATUSES",
    "REMEDIATION_STEP_SCHEMA",
    "REMEDIATION_STEP_STATUSES",
    "REPUTATION_EVENT_KINDS",
    "REPUTATION_EVENT_SCHEMA",
    "REVIEW_AGGREGATE_SCHEMA",
    "REVIEW_EVIDENCE_SCHEMA",
    "Campaign",
    "GovernanceCase",
    "GovernanceRecord",
    "MarketGovernanceValidationError",
    "MarketSignal",
    "Placement",
    "RankingContext",
    "RemediationPlan",
    "RemediationStep",
    "ReputationEvent",
    "ReviewAggregate",
    "ReviewEvidence",
    "campaign_digest",
    "campaign_to_wire",
    "canonical_digest",
    "coerce_campaign",
    "coerce_governance_case",
    "coerce_market_signal",
    "coerce_placement",
    "coerce_ranking_context",
    "coerce_remediation_plan",
    "coerce_remediation_step",
    "coerce_reputation_event",
    "coerce_review_aggregate",
    "coerce_review_evidence",
    "governance_case_digest",
    "governance_case_to_wire",
    "market_signal_digest",
    "market_signal_to_wire",
    "placement_digest",
    "placement_to_wire",
    "ranking_context_digest",
    "ranking_context_to_wire",
    "remediation_plan_digest",
    "remediation_plan_to_wire",
    "remediation_step_digest",
    "remediation_step_to_wire",
    "replay_exact_or_conflict",
    "reputation_event_digest",
    "reputation_event_to_wire",
    "review_aggregate_digest",
    "review_aggregate_to_wire",
    "review_evidence_digest",
    "review_evidence_to_wire",
    "seal_campaign",
    "seal_governance_case",
    "seal_market_signal",
    "seal_placement",
    "seal_ranking_context",
    "seal_remediation_plan",
    "seal_remediation_step",
    "seal_reputation_event",
    "seal_review_aggregate",
    "seal_review_evidence",
    "validate_campaign",
    "validate_governance_case",
    "validate_market_signal",
    "validate_placement",
    "validate_ranking_context",
    "validate_remediation_plan",
    "validate_remediation_step",
    "validate_reputation_event",
    "validate_reputation_successor",
    "validate_review_aggregate",
    "validate_review_evidence",
    "validate_version_successor",
]
