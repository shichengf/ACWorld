"""World-authoritative marketplace governance domain rules.

The value objects in :mod:`protocol.market_governance` are passive, replayable
contracts.  This module is the authority boundary that derives those records
from authenticated actors and trusted World rows.  Agents submit compact
intents only.  They cannot supply owners, verification flags, money, logical
time, versions, predecessor links, evidence digests, or sealed outcomes.

This module intentionally has no scenario, scorer, transport, or persistence
adapter.  World backends can persist the returned records and operation
fingerprints atomically in a later integration step.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Literal, TypeAlias, TypedDict, cast

from protocol.evidence_records import (
    EvidenceRecord,
    MandateRevision,
    validate_evidence_record,
    validate_mandate_revision,
)
from protocol.market_governance import (
    REMEDIATION_ACTION_KINDS,
    Campaign,
    GovernanceCase,
    MarketSignal,
    Placement,
    RankingContext,
    RemediationPlan,
    RemediationStep,
    ReputationEvent,
    ReviewAggregate,
    ReviewEvidence,
    seal_campaign,
    seal_governance_case,
    seal_market_signal,
    seal_placement,
    seal_ranking_context,
    seal_remediation_plan,
    seal_remediation_step,
    seal_reputation_event,
    seal_review_aggregate,
    seal_review_evidence,
    validate_campaign,
    validate_governance_case,
    validate_market_signal,
    validate_placement,
    validate_ranking_context,
    validate_remediation_plan,
    validate_reputation_event,
    validate_reputation_successor,
    validate_review_aggregate,
    validate_review_evidence,
    validate_version_successor,
)
from world.errors import IdempotencyConflict, WorldError, WriteNotAuthorized
from world.types import Listing, Order, Receipt


LISTING_GOVERNANCE_BINDING_SCHEMA = "cwe.world-listing-governance-binding.v1"
PURCHASE_GOVERNANCE_BINDING_SCHEMA = "cwe.world-purchase-governance-binding.v1"
ADS_CAMPAIGN_TERMS_SCHEMA = "cwe.world-ads-campaign-terms.v1"
REVIEW_ACCOUNT_BINDING_SCHEMA = "cwe.world-review-account-binding.v1"
MARKET_OBSERVATION_SCHEMA = "cwe.world-market-observation.v1"
GOVERNANCE_RESPONSE_ATTESTATION_SCHEMA = "cwe.world-governance-response-attestation.v1"
GOVERNANCE_RESOLUTION_DECISION_SCHEMA = "cwe.world-governance-resolution-decision.v1"
REPUTATION_POLICY_SCHEMA = "cwe.world-reputation-policy.v1"
REMEDIATION_BLUEPRINT_SCHEMA = "cwe.world-remediation-blueprint.v1"
GOVERNANCE_EVIDENCE_BINDING_SCHEMA = "cwe.world-governance-evidence-binding.v1"
GOVERNANCE_INTENT_SCHEMA = "cwe.world-market-governance-intent.v1"

AppendDisposition: TypeAlias = Literal["append", "idempotent"]


class MarketGovernanceCoreError(WorldError):
    """Base error for authoritative marketplace governance rules."""


class GovernanceIntentError(MarketGovernanceCoreError):
    """An actor intent is malformed or tries to self-report authority."""


class GovernanceBindingError(MarketGovernanceCoreError):
    """Trusted World rows do not form the claimed commerce binding."""


class GovernanceTransitionError(MarketGovernanceCoreError):
    """A requested governance transition is invalid for current state."""


@dataclass(frozen=True, slots=True)
class ListingGovernanceBinding:
    """World-derived identity for one merchant-owned catalog listing."""

    sku_id: str
    product_id: str | None
    merchant_id: str
    catalog_revision: int
    listing_digest: str
    schema_id: str = LISTING_GOVERNANCE_BINDING_SCHEMA


@dataclass(frozen=True, slots=True)
class PurchaseGovernanceBinding:
    """World-derived order and settlement evidence for review/reputation use."""

    order_id: str
    txn_id: str
    buyer_id: str
    merchant_id: str
    sku_id: str
    qty: int
    settled_at_tick: int
    order_digest: str
    receipt_digest: str
    listing_digest: str
    binding_digest: str
    schema_id: str = PURCHASE_GOVERNANCE_BINDING_SCHEMA


@dataclass(frozen=True, slots=True)
class AdsPlacementTerms:
    """One placement allocation issued by a trusted advertising service."""

    sku_id: str
    bid_cents: int
    fee_cents: int


@dataclass(frozen=True, slots=True)
class AdsCampaignTerms:
    """Trusted paid-placement terms.  These values never come from an agent."""

    campaign_id: str
    budget_cents: int
    currency: str
    starts_at_tick: int
    ends_at_tick: int
    placements: tuple[AdsPlacementTerms, ...]
    issued_by_id: str
    terms_digest: str
    schema_id: str = ADS_CAMPAIGN_TERMS_SCHEMA


@dataclass(frozen=True, slots=True)
class ReviewAccountBinding:
    """Trusted account metadata used to classify review evidence."""

    reviewer_id: str
    account_created_at_tick: int
    burst_group_id: str | None
    authority_id: str
    binding_digest: str
    schema_id: str = REVIEW_ACCOUNT_BINDING_SCHEMA


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """Persisted detector output ingested by the governance authority.

    ``detected_by_id`` is the issuer of the immutable EvidenceRecord, while
    ``ingested_by_id`` is the Platform governance service that turns the
    detector result into an authoritative MarketSignal.  Keeping both roles
    in the sealed observation prevents the Platform from being mistaken for
    the external detector and gives replay an exact evidence-record join.
    """

    signal_kind: str
    subject_listings: tuple[ListingGovernanceBinding, ...]
    source_refs: tuple[str, ...]
    confidence_bps: int
    source_record_id: str
    source_record_digest: str
    detected_by_id: str
    ingested_by_id: str
    observation_digest: str
    schema_id: str = MARKET_OBSERVATION_SCHEMA


@dataclass(frozen=True, slots=True)
class GovernanceResponseAttestation:
    """Platform-attested merchant response, without authority over case outcome."""

    response_id: str
    case_id: str
    case_digest: str
    subject_merchant_id: str
    response_kind: str
    signal_digests: tuple[str, ...]
    submitted_at_tick: int
    authored_by: str
    idempotency_key: str
    request_fingerprint: str
    provenance_digests: tuple[str, ...]
    attestation_digest: str
    schema_id: str = GOVERNANCE_RESPONSE_ATTESTATION_SCHEMA


@dataclass(frozen=True, slots=True)
class GovernanceResolutionDecision:
    """Trusted Platform policy decision applied separately from actor response."""

    decision_id: str
    case_id: str
    case_digest: str
    resolution_kind: str
    target_status: str
    resolution_code: str
    policy_id: str
    policy_version: int
    response_digests: tuple[str, ...]
    decided_at_tick: int
    authored_by: str
    idempotency_key: str
    decision_digest: str
    schema_id: str = GOVERNANCE_RESOLUTION_DECISION_SCHEMA


@dataclass(frozen=True, slots=True)
class ReputationPolicyRevision:
    """Trusted mapping from authoritative source kinds to reputation outcomes."""

    policy_id: str
    revision: int
    previous_digest: str | None
    effective_tick: int
    published_by_id: str
    fulfilled_order_bps: int
    disputed_order_bps: int
    refund_bps: int
    remediation_verified_bps: int
    compliance_violation_bps: int
    policy_digest: str
    schema_id: str = REPUTATION_POLICY_SCHEMA


@dataclass(frozen=True, slots=True)
class RemediationBlueprintStep:
    """One policy-defined remediation action and its dependencies."""

    action_kind: str
    prerequisite_sequence_nos: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RemediationBlueprint:
    """Trusted plan template selected by governance policy, not a merchant."""

    blueprint_id: str
    governance_case_kind: str
    steps: tuple[RemediationBlueprintStep, ...]
    issued_by_id: str
    blueprint_digest: str
    schema_id: str = REMEDIATION_BLUEPRINT_SCHEMA


@dataclass(frozen=True, slots=True)
class GovernanceEvidenceBinding:
    """World-derived evidence accepted for a remediation step."""

    source_ref: str
    merchant_id: str
    evidence_kind: str
    observed_at_tick: int
    verified_by_id: str
    source_digest: str
    evidence_digest: str
    schema_id: str = GOVERNANCE_EVIDENCE_BINDING_SCHEMA


class GovernanceIntent(TypedDict, total=False):
    """Normalized compact agent intent."""

    op: str
    campaign_id: str
    placement_id: str
    disclosure_text: str
    sku_id: str
    rating: int
    review_text: str
    case_id: str
    plan_id: str
    step_id: str


_INTENT_FIELDS: dict[str, frozenset[str]] = {
    "publish_campaign": frozenset({"op", "campaign_id"}),
    "activate_campaign": frozenset({"op", "campaign_id"}),
    "disclose_placement": frozenset({"op", "campaign_id", "placement_id", "disclosure_text"}),
    "submit_review": frozenset({"op", "sku_id", "rating"}),
    "reject_review_manipulation": frozenset({"op", "case_id"}),
    "reject_coordination": frozenset({"op", "case_id"}),
    "accept_remediation_plan": frozenset({"op", "plan_id"}),
    "complete_remediation_step": frozenset({"op", "plan_id", "step_id"}),
}

_OPTIONAL_INTENT_FIELDS: dict[str, frozenset[str]] = {
    "submit_review": frozenset({"review_text"}),
}

_FORBIDDEN_INTENT_FIELDS = frozenset(
    {
        "owner",
        "owner_id",
        "owner_merchant_id",
        "merchant",
        "merchant_id",
        "buyer_id",
        "reviewer_id",
        "actor_id",
        "authored_by",
        "verified_purchase",
        "order_id",
        "txn_id",
        "receipt_id",
        "source_ref",
        "source_refs",
        "status",
        "target_status",
        "resolution_code",
        "resolution_kind",
        "outcome",
        "outcome_bps",
        "confidence_bps",
        "budget",
        "budget_cents",
        "bid",
        "bid_cents",
        "fee",
        "fee_cents",
        "currency",
        "logical_tick",
        "server_tick",
        "starts_at_tick",
        "ends_at_tick",
        "submitted_at_tick",
        "occurred_at_tick",
        "completed_at_tick",
        "catalog_revision",
        "policy_version",
        "policy_revision",
        "version",
        "previous_digest",
        "digest",
        "record_digest",
        "listing_digest",
        "binding_digest",
        "evidence_digest",
        "provenance_digests",
        "idempotency_key",
    }
)


def normalize_governance_intent(value: Mapping[str, Any]) -> GovernanceIntent:
    """Validate an exact compact intent and reject self-reported authority."""

    if not isinstance(value, Mapping):
        raise GovernanceIntentError("governance intent must be an object")
    if any(not isinstance(key, str) or not key for key in value):
        raise GovernanceIntentError("governance intent field names must be strings")
    forbidden = sorted(set(value).intersection(_FORBIDDEN_INTENT_FIELDS))
    if forbidden:
        raise GovernanceIntentError(
            "governance authority fields are World-owned: " + ", ".join(forbidden)
        )
    _reject_nested_authority(value)
    op = value.get("op")
    if not isinstance(op, str) or op not in _INTENT_FIELDS:
        raise GovernanceIntentError("unsupported governance operation")
    expected = _INTENT_FIELDS[op]
    optional = _OPTIONAL_INTENT_FIELDS.get(op, frozenset())
    if not expected.issubset(value) or set(value) - expected - optional:
        raise GovernanceIntentError(
            f"{op} intent requires "
            + ", ".join(sorted(expected))
            + (
                "; optional fields: " + ", ".join(sorted(optional))
                if optional
                else ""
            )
        )
    normalized: dict[str, Any] = {"op": op}
    for name in (set(value) & (expected | optional)) - {"op"}:
        item = value[name]
        if name == "rating":
            normalized[name] = _int_range(item, 1, 5, "rating")
        else:
            normalized[name] = _text(item, name)
    _canonical_json(normalized)
    return cast(GovernanceIntent, normalized)


def governance_intent_fingerprint(
    intent: Mapping[str, Any],
    *,
    original_actor: str,
    trusted_context_digests: Iterable[str],
) -> str:
    """Bind one compact intent to the authenticated actor and trusted context."""

    normalized = normalize_governance_intent(intent)
    actor = _text(original_actor, "original_actor")
    digests = _canonical_digests(trusted_context_digests, "trusted_context_digests")
    return _digest(
        {
            "schema_id": GOVERNANCE_INTENT_SCHEMA,
            "original_actor": actor,
            "trusted_context_digests": list(digests),
            "intent": dict(normalized),
        }
    )


def authoritative_listing_digest(listing: Listing) -> str:
    """Digest all authoritative fields of one World listing."""

    if not isinstance(listing, Listing):
        raise GovernanceBindingError("listing must be a World Listing")
    return _digest(
        {
            "sku_id": str(listing.sku_id),
            "product_id": listing.product_id,
            "merchant_id": str(listing.merchant_id),
            "category": listing.category,
            "name": listing.name,
            "attributes": listing.attributes,
            "list_price": {
                "amount": _decimal_string(listing.list_price.amount),
                "currency": listing.list_price.currency,
            },
        }
    )


def derive_listing_governance_binding(
    listing: Listing,
    *,
    catalog_revision: int,
) -> ListingGovernanceBinding:
    """Derive listing owner and revision from an authoritative catalog row."""

    if not isinstance(listing, Listing):
        raise GovernanceBindingError("listing must be a World Listing")
    sku_id = _text(str(listing.sku_id), "listing.sku_id")
    merchant_id = _text(str(listing.merchant_id), "listing.merchant_id")
    if not merchant_id.startswith("merchant:"):
        raise GovernanceBindingError("listing owner must be a merchant actor")
    revision = _nonnegative_int(catalog_revision, "catalog_revision")
    return ListingGovernanceBinding(
        sku_id=sku_id,
        product_id=listing.product_id,
        merchant_id=merchant_id,
        catalog_revision=revision,
        listing_digest=authoritative_listing_digest(listing),
    )


def authoritative_order_digest(order: Order) -> str:
    if not isinstance(order, Order):
        raise GovernanceBindingError("order must be a World Order")
    return _digest(
        {
            "order_id": str(order.order_id),
            "buyer_id": str(order.buyer_id),
            "merchant_id": str(order.merchant_id),
            "sku_id": str(order.sku_id),
            "qty": order.qty,
            "agreed_price": {
                "amount": _decimal_string(order.agreed_price.amount),
                "currency": order.agreed_price.currency,
            },
            "state": order.state.value,
            "request_order": order.request_order,
        }
    )


def authoritative_receipt_digest(receipt: Receipt) -> str:
    if not isinstance(receipt, Receipt):
        raise GovernanceBindingError("receipt must be a World Receipt")
    return _digest(
        {
            "txn_id": str(receipt.txn_id),
            "ts": receipt.ts,
            "order_id": str(receipt.order_id),
            "buyer_id": str(receipt.buyer_id),
            "merchant_id": str(receipt.merchant_id),
            "sku_id": str(receipt.sku_id),
            "qty": receipt.qty,
            "price": {
                "amount": _decimal_string(receipt.price.amount),
                "currency": receipt.price.currency,
            },
            "idempotency_key": receipt.idempotency_key,
            "effect": receipt.effect,
        }
    )


def derive_purchase_governance_binding(
    *,
    order: Order,
    receipt: Receipt,
    listing: Listing,
    settled_at_tick: int,
    catalog_revision: int,
) -> PurchaseGovernanceBinding:
    """Bind an order and receipt to the listing without trusting review claims."""

    listing_binding = derive_listing_governance_binding(listing, catalog_revision=catalog_revision)
    if not isinstance(order, Order) or not isinstance(receipt, Receipt):
        raise GovernanceBindingError("purchase binding requires World Order and Receipt")
    identities = (
        str(order.order_id) == str(receipt.order_id),
        str(order.buyer_id) == str(receipt.buyer_id),
        str(order.merchant_id) == str(receipt.merchant_id),
        str(order.sku_id) == str(receipt.sku_id),
        str(order.merchant_id) == listing_binding.merchant_id,
        str(order.sku_id) == listing_binding.sku_id,
    )
    if not all(identities):
        raise GovernanceBindingError("order, receipt, and listing identity mismatch")
    if order.qty <= 0 or receipt.qty <= 0 or receipt.qty > order.qty:
        raise GovernanceBindingError("receipt quantity is invalid for order")
    if (
        receipt.price.amount != order.agreed_price.amount
        or receipt.price.currency != order.agreed_price.currency
    ):
        raise GovernanceBindingError("receipt price does not match order")
    tick = _nonnegative_int(settled_at_tick, "settled_at_tick")
    order_digest = authoritative_order_digest(order)
    receipt_digest = authoritative_receipt_digest(receipt)
    contract = {
        "schema_id": PURCHASE_GOVERNANCE_BINDING_SCHEMA,
        "order_id": str(order.order_id),
        "txn_id": str(receipt.txn_id),
        "buyer_id": str(order.buyer_id),
        "merchant_id": str(order.merchant_id),
        "sku_id": str(order.sku_id),
        "qty": receipt.qty,
        "settled_at_tick": tick,
        "order_digest": order_digest,
        "receipt_digest": receipt_digest,
        "listing_digest": listing_binding.listing_digest,
    }
    return PurchaseGovernanceBinding(
        order_id=str(order.order_id),
        txn_id=str(receipt.txn_id),
        buyer_id=str(order.buyer_id),
        merchant_id=str(order.merchant_id),
        sku_id=str(order.sku_id),
        qty=receipt.qty,
        settled_at_tick=tick,
        order_digest=order_digest,
        receipt_digest=receipt_digest,
        listing_digest=listing_binding.listing_digest,
        binding_digest=_digest(contract),
    )


def build_ads_campaign_terms(
    *,
    campaign_id: str,
    budget_cents: int,
    currency: str,
    starts_at_tick: int,
    ends_at_tick: int,
    placements: Iterable[AdsPlacementTerms],
    issued_by_id: str,
) -> AdsCampaignTerms:
    """Seal trusted campaign terms before any merchant can act on them."""

    normalized = tuple(sorted(placements, key=lambda item: item.sku_id))
    candidate = AdsCampaignTerms(
        campaign_id=_text(campaign_id, "campaign_id"),
        budget_cents=_positive_int(budget_cents, "budget_cents"),
        currency=_currency(currency),
        starts_at_tick=_nonnegative_int(starts_at_tick, "starts_at_tick"),
        ends_at_tick=_nonnegative_int(ends_at_tick, "ends_at_tick"),
        placements=normalized,
        issued_by_id=_text(issued_by_id, "issued_by_id"),
        terms_digest="",
    )
    digest = _digest(_campaign_terms_contract(candidate))
    sealed = replace(candidate, terms_digest=digest)
    validate_ads_campaign_terms(sealed)
    return sealed


def validate_ads_campaign_terms(
    terms: AdsCampaignTerms,
    *,
    expected_authority: str | None = None,
) -> None:
    if not isinstance(terms, AdsCampaignTerms):
        raise GovernanceBindingError("campaign terms have wrong type")
    _text(terms.campaign_id, "campaign_id")
    _positive_int(terms.budget_cents, "budget_cents")
    _currency(terms.currency)
    _nonnegative_int(terms.starts_at_tick, "starts_at_tick")
    _nonnegative_int(terms.ends_at_tick, "ends_at_tick")
    if terms.ends_at_tick <= terms.starts_at_tick:
        raise GovernanceBindingError("campaign lifetime must be positive")
    _text(terms.issued_by_id, "issued_by_id")
    if expected_authority is not None and terms.issued_by_id != expected_authority:
        raise WriteNotAuthorized("campaign terms authority mismatch")
    if not terms.placements:
        raise GovernanceBindingError("campaign requires at least one placement")
    if terms.placements != tuple(sorted(terms.placements, key=lambda item: item.sku_id)):
        raise GovernanceBindingError("campaign placements must be in canonical SKU order")
    seen: set[str] = set()
    total_fees = 0
    for placement in terms.placements:
        if not isinstance(placement, AdsPlacementTerms):
            raise GovernanceBindingError("placement terms have wrong type")
        sku_id = _text(placement.sku_id, "placement.sku_id")
        if sku_id in seen:
            raise GovernanceBindingError("campaign placement SKUs must be unique")
        seen.add(sku_id)
        _nonnegative_int(placement.bid_cents, "placement.bid_cents")
        total_fees += _nonnegative_int(placement.fee_cents, "placement.fee_cents")
    if total_fees > terms.budget_cents:
        raise GovernanceBindingError("campaign fees exceed trusted budget")
    if terms.terms_digest != _digest(_campaign_terms_contract(terms)):
        raise GovernanceBindingError("campaign terms digest mismatch")


def build_review_account_binding(
    *,
    reviewer_id: str,
    account_created_at_tick: int,
    burst_group_id: str | None,
    authority_id: str,
) -> ReviewAccountBinding:
    """Seal trusted account-age and burst metadata."""

    contract = {
        "schema_id": REVIEW_ACCOUNT_BINDING_SCHEMA,
        "reviewer_id": _text(reviewer_id, "reviewer_id"),
        "account_created_at_tick": _nonnegative_int(
            account_created_at_tick, "account_created_at_tick"
        ),
        "burst_group_id": _optional_text(burst_group_id, "burst_group_id"),
        "authority_id": _text(authority_id, "authority_id"),
    }
    return ReviewAccountBinding(**contract, binding_digest=_digest(contract))  # type: ignore[arg-type]


def validate_review_account_binding(
    binding: ReviewAccountBinding,
    *,
    expected_authority: str,
) -> None:
    if not isinstance(binding, ReviewAccountBinding):
        raise GovernanceBindingError("review account binding has wrong type")
    contract = {
        "schema_id": REVIEW_ACCOUNT_BINDING_SCHEMA,
        "reviewer_id": _text(binding.reviewer_id, "reviewer_id"),
        "account_created_at_tick": _nonnegative_int(
            binding.account_created_at_tick, "account_created_at_tick"
        ),
        "burst_group_id": _optional_text(binding.burst_group_id, "burst_group_id"),
        "authority_id": _text(binding.authority_id, "authority_id"),
    }
    if binding.authority_id != expected_authority:
        raise WriteNotAuthorized("review account authority mismatch")
    if binding.binding_digest != _digest(contract):
        raise GovernanceBindingError("review account binding digest mismatch")


def derive_campaign_publish(
    intent: Mapping[str, Any],
    *,
    terms: AdsCampaignTerms,
    listing_bindings: Sequence[ListingGovernanceBinding],
    original_actor: str,
    ads_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: Campaign | None = None,
) -> Campaign:
    """Create a draft paid campaign from trusted terms and catalog ownership."""

    normalized = _intent(intent, "publish_campaign")
    validate_ads_campaign_terms(terms, expected_authority=ads_authority_id)
    if normalized["campaign_id"] != terms.campaign_id:
        raise GovernanceBindingError("intent campaign does not match trusted terms")
    actor = _text(original_actor, "original_actor")
    tick = _nonnegative_int(server_tick, "server_tick")
    key = _text(idempotency_key, "idempotency_key")
    by_sku = _listing_binding_index(listing_bindings)
    term_skus = tuple(item.sku_id for item in terms.placements)
    if set(by_sku) != set(term_skus):
        raise GovernanceBindingError("campaign terms and listing bindings differ")
    owners = {row.merchant_id for row in by_sku.values()}
    if len(owners) != 1:
        raise GovernanceBindingError("one campaign cannot cross merchant owners")
    owner = next(iter(owners))
    if actor != owner:
        raise WriteNotAuthorized("only listing owner may publish campaign")
    if tick > terms.ends_at_tick:
        raise GovernanceTransitionError("expired campaign terms cannot be published")
    if current is not None and current.idempotency_key == key:
        _validate_campaign_exact_retry(
            current,
            terms=terms,
            listing_bindings=by_sku,
            expected_authority=ads_authority_id,
            expected_owner=owner,
            expected_status="draft",
        )
        return current
    if current is not None:
        raise GovernanceTransitionError("campaign is already published")
    placements = tuple(
        seal_placement(
            Placement(
                placement_id=_derived_id("placement", terms.campaign_id, allocation.sku_id),
                campaign_id=terms.campaign_id,
                owner_merchant_id=owner,
                sku_id=allocation.sku_id,
                placement_kind="sponsored",
                disclosure_status="pending",
                disclosure_text=None,
                bid_cents=allocation.bid_cents,
                fee_cents=allocation.fee_cents,
                currency=terms.currency,
                starts_at_tick=terms.starts_at_tick,
                ends_at_tick=terms.ends_at_tick,
                authored_by=ads_authority_id,
                idempotency_key=f"{key}:placement:{index}",
                provenance_digests=(
                    terms.terms_digest,
                    by_sku[allocation.sku_id].listing_digest,
                ),
            )
        )
        for index, allocation in enumerate(terms.placements, start=1)
    )
    candidate = seal_campaign(
        Campaign(
            campaign_id=terms.campaign_id,
            owner_merchant_id=owner,
            status="draft",
            budget_cents=terms.budget_cents,
            currency=terms.currency,
            starts_at_tick=terms.starts_at_tick,
            ends_at_tick=terms.ends_at_tick,
            placements=placements,
            authored_by=ads_authority_id,
            idempotency_key=key,
            provenance_digests=(terms.terms_digest,),
        )
    )
    validate_campaign(
        candidate,
        expected_authority=ads_authority_id,
        expected_owner_id=owner,
    )
    return candidate


def derive_campaign_activation(
    intent: Mapping[str, Any],
    *,
    current: Campaign,
    original_actor: str,
    ads_authority_id: str,
    server_tick: int,
    idempotency_key: str,
) -> Campaign:
    """Activate a disclosed campaign without accepting agent-supplied state."""

    normalized = _intent(intent, "activate_campaign")
    validate_campaign(
        current,
        expected_authority=ads_authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    if normalized["campaign_id"] != current.campaign_id:
        raise GovernanceBindingError("campaign intent binding mismatch")
    actor = _text(original_actor, "original_actor")
    if actor != current.owner_merchant_id:
        raise WriteNotAuthorized("only campaign owner may activate campaign")
    tick = _nonnegative_int(server_tick, "server_tick")
    if not current.starts_at_tick <= tick <= current.ends_at_tick:
        raise GovernanceTransitionError("campaign is outside its active window")
    key = _text(idempotency_key, "idempotency_key")
    if current.idempotency_key == key:
        if current.status != "active":
            raise IdempotencyConflict("campaign activation changed under existing key")
        return current
    if current.status not in {"draft", "paused"}:
        raise GovernanceTransitionError("campaign cannot transition to active")
    if any(item.disclosure_status != "disclosed" for item in current.placements):
        raise GovernanceTransitionError("all sponsored placements must be disclosed")
    candidate = seal_campaign(
        replace(
            current,
            status="active",
            idempotency_key=key,
            version=current.version + 1,
            previous_digest=current.campaign_digest,
            provenance_digests=_merge_digests(
                current.provenance_digests, (current.campaign_digest,)
            ),
            campaign_digest="",
        )
    )
    validate_version_successor(
        current,
        candidate,
        expected_authority=ads_authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    return candidate


def derive_placement_disclosure(
    intent: Mapping[str, Any],
    *,
    current: Campaign,
    listing_binding: ListingGovernanceBinding,
    original_actor: str,
    ads_authority_id: str,
    server_tick: int,
    idempotency_key: str,
) -> Campaign:
    """Version one placement and its campaign after an owner disclosure."""

    normalized = _intent(intent, "disclose_placement")
    validate_campaign(
        current,
        expected_authority=ads_authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    _validate_listing_binding(listing_binding)
    if normalized["campaign_id"] != current.campaign_id:
        raise GovernanceBindingError("campaign intent binding mismatch")
    actor = _text(original_actor, "original_actor")
    if actor != current.owner_merchant_id or actor != listing_binding.merchant_id:
        raise WriteNotAuthorized("only placement owner may disclose sponsorship")
    tick = _nonnegative_int(server_tick, "server_tick")
    if tick > current.ends_at_tick:
        raise GovernanceTransitionError("expired placement cannot be disclosed")
    key = _text(idempotency_key, "idempotency_key")
    placement_id = normalized["placement_id"]
    selected = next(
        (item for item in current.placements if item.placement_id == placement_id),
        None,
    )
    if selected is None:
        raise GovernanceBindingError("placement is not in campaign")
    if selected.sku_id != listing_binding.sku_id:
        raise GovernanceBindingError("placement does not bind supplied listing")
    disclosure_text = normalized["disclosure_text"]
    if current.idempotency_key == key:
        if selected.disclosure_status != "disclosed" or selected.disclosure_text != disclosure_text:
            raise IdempotencyConflict("placement disclosure changed under existing key")
        return current
    if selected.disclosure_status == "disclosed":
        if selected.disclosure_text == disclosure_text:
            raise GovernanceTransitionError("placement is already disclosed")
        raise GovernanceTransitionError("disclosed placement text is immutable")
    updated = seal_placement(
        replace(
            selected,
            disclosure_status="disclosed",
            disclosure_text=disclosure_text,
            authored_by=ads_authority_id,
            idempotency_key=f"{key}:placement",
            version=selected.version + 1,
            previous_digest=selected.placement_digest,
            provenance_digests=_merge_digests(
                selected.provenance_digests,
                (selected.placement_digest, listing_binding.listing_digest),
            ),
            placement_digest="",
        )
    )
    placements = tuple(
        updated if item.placement_id == placement_id else item for item in current.placements
    )
    candidate = seal_campaign(
        replace(
            current,
            placements=placements,
            idempotency_key=key,
            version=current.version + 1,
            previous_digest=current.campaign_digest,
            provenance_digests=_merge_digests(
                current.provenance_digests,
                (current.campaign_digest, listing_binding.listing_digest),
            ),
            campaign_digest="",
        )
    )
    validate_version_successor(
        current,
        candidate,
        expected_authority=ads_authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    return candidate


def derive_review_evidence(
    intent: Mapping[str, Any],
    *,
    listing_binding: ListingGovernanceBinding,
    account_binding: ReviewAccountBinding,
    purchase_binding: PurchaseGovernanceBinding | None,
    original_actor: str,
    review_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: ReviewEvidence | None = None,
    source_record: EvidenceRecord | None = None,
) -> ReviewEvidence:
    """Derive reviewer, owner, verified purchase, order, and time from World data."""

    normalized = _intent(intent, "submit_review")
    _validate_listing_binding(listing_binding)
    validate_review_account_binding(account_binding, expected_authority=review_authority_id)
    actor = _text(original_actor, "original_actor")
    if actor != account_binding.reviewer_id:
        raise WriteNotAuthorized("authenticated actor does not own review account")
    if normalized["sku_id"] != listing_binding.sku_id:
        raise GovernanceBindingError("review SKU does not match listing binding")
    tick = _nonnegative_int(server_tick, "server_tick")
    if account_binding.account_created_at_tick > tick:
        raise GovernanceBindingError("review account was created after submission")
    verified = purchase_binding is not None
    order_id: str | None = None
    source_ref: str
    provenance = [listing_binding.listing_digest, account_binding.binding_digest]
    if purchase_binding is not None:
        _validate_purchase_binding(purchase_binding)
        if (
            purchase_binding.buyer_id != actor
            or purchase_binding.sku_id != listing_binding.sku_id
            or purchase_binding.merchant_id != listing_binding.merchant_id
        ):
            raise GovernanceBindingError("purchase does not authorize this verified review")
        if purchase_binding.settled_at_tick > tick:
            raise GovernanceBindingError("review predates verified settlement")
        order_id = purchase_binding.order_id
        source_ref = f"receipt:{purchase_binding.txn_id}"
        provenance.append(purchase_binding.binding_digest)
    else:
        source_ref = _derived_id("review-submission", listing_binding.sku_id, actor)
    if source_record is not None:
        validate_evidence_record(source_record)
        if source_record.owner_id != actor:
            raise WriteNotAuthorized("review observation owner is not the reviewer")
        if (
            review_authority_id != source_record.owner_id
            and review_authority_id not in source_record.read_acl
        ):
            raise WriteNotAuthorized(
                "review authority cannot read review observation evidence"
            )
        source_ref = f"evidence-record:{source_record.record_id}"
        provenance.append(source_record.record_digest)
    key = _text(idempotency_key, "idempotency_key")
    review_id = _derived_id("review", listing_binding.sku_id, actor, key)
    if current is not None and current.idempotency_key == key:
        validate_review_evidence(
            current,
            expected_authority=review_authority_id,
            expected_owner_id=listing_binding.merchant_id,
        )
        expected = (
            review_id,
            listing_binding.sku_id,
            listing_binding.merchant_id,
            actor,
            normalized["rating"],
            verified,
            order_id,
            account_binding.account_created_at_tick,
            account_binding.burst_group_id,
            source_ref,
        )
        actual = (
            current.review_id,
            current.sku_id,
            current.merchant_id,
            current.reviewer_id,
            current.rating,
            current.verified_purchase,
            current.order_id,
            current.account_created_at_tick,
            current.burst_group_id,
            current.source_ref,
        )
        if actual != expected:
            raise IdempotencyConflict("review changed under existing key")
        return current
    candidate = seal_review_evidence(
        ReviewEvidence(
            review_id=review_id,
            sku_id=listing_binding.sku_id,
            merchant_id=listing_binding.merchant_id,
            reviewer_id=actor,
            rating=normalized["rating"],
            verified_purchase=verified,
            order_id=order_id,
            submitted_at_tick=tick,
            account_created_at_tick=account_binding.account_created_at_tick,
            burst_group_id=account_binding.burst_group_id,
            source_ref=source_ref,
            authored_by=review_authority_id,
            idempotency_key=key,
            provenance_digests=tuple(provenance),
        )
    )
    if current is not None:
        raise GovernanceTransitionError("review evidence is immutable")
    validate_review_evidence(
        candidate,
        expected_authority=review_authority_id,
        expected_owner_id=listing_binding.merchant_id,
    )
    return candidate


def derive_review_aggregate(
    *,
    listing_binding: ListingGovernanceBinding,
    evidence: Sequence[ReviewEvidence],
    aggregation_policy_id: str,
    aggregation_policy_version: int,
    review_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: ReviewAggregate | None = None,
) -> ReviewAggregate:
    """Compute review counts and scores solely from persisted review evidence."""

    _validate_listing_binding(listing_binding)
    rows = tuple(evidence)
    if len({item.review_id for item in rows}) != len(rows):
        raise GovernanceBindingError("review evidence contains duplicate ids")
    expected_order = tuple(sorted(rows, key=lambda item: (item.submitted_at_tick, item.review_id)))
    if rows != expected_order:
        raise GovernanceBindingError("review evidence must be in authoritative order")
    for item in rows:
        validate_review_evidence(
            item,
            expected_authority=review_authority_id,
            expected_owner_id=listing_binding.merchant_id,
        )
        if item.sku_id != listing_binding.sku_id:
            raise GovernanceBindingError("review aggregate crosses listing identity")
    if current is not None:
        validate_review_aggregate(
            current,
            expected_authority=review_authority_id,
            expected_owner_id=listing_binding.merchant_id,
        )
        old = current.evidence_digests
        new = tuple(item.evidence_digest for item in rows)
        if new[: len(old)] != old:
            raise GovernanceTransitionError("review evidence history is not append-only")
    key = _text(idempotency_key, "idempotency_key")
    tick = _nonnegative_int(server_tick, "server_tick")
    verified = tuple(item for item in rows if item.verified_purchase)
    if current is not None and current.idempotency_key == key:
        expected = (
            listing_binding.sku_id,
            listing_binding.merchant_id,
            aggregation_policy_id,
            aggregation_policy_version,
            tuple(item.evidence_digest for item in rows),
            len(rows),
            len(verified),
            sum(item.rating for item in rows),
            sum(item.rating for item in verified),
        )
        actual = (
            current.sku_id,
            current.merchant_id,
            current.aggregation_policy_id,
            current.aggregation_policy_version,
            current.evidence_digests,
            current.review_count,
            current.verified_review_count,
            current.rating_sum,
            current.verified_rating_sum,
        )
        if actual != expected:
            raise IdempotencyConflict("review aggregate changed under existing key")
        return current
    candidate = seal_review_aggregate(
        ReviewAggregate(
            aggregate_id=_derived_id("review-aggregate", listing_binding.sku_id),
            sku_id=listing_binding.sku_id,
            merchant_id=listing_binding.merchant_id,
            aggregation_policy_id=_text(aggregation_policy_id, "aggregation_policy_id"),
            aggregation_policy_version=_positive_int(
                aggregation_policy_version, "aggregation_policy_version"
            ),
            evidence_digests=tuple(item.evidence_digest for item in rows),
            review_count=len(rows),
            verified_review_count=len(verified),
            rating_sum=sum(item.rating for item in rows),
            verified_rating_sum=sum(item.rating for item in verified),
            computed_at_tick=tick,
            authored_by=review_authority_id,
            idempotency_key=key,
            version=1 if current is None else current.version + 1,
            previous_digest=None if current is None else current.aggregate_digest,
            provenance_digests=(
                (listing_binding.listing_digest,)
                if current is None
                else current.provenance_digests + (current.aggregate_digest,)
            ),
        )
    )
    validate_review_aggregate(
        candidate,
        evidence=rows,
        expected_authority=review_authority_id,
        expected_owner_id=listing_binding.merchant_id,
    )
    if current is None:
        return candidate
    validate_version_successor(
        current,
        candidate,
        expected_authority=review_authority_id,
        expected_owner_id=listing_binding.merchant_id,
    )
    return candidate


def build_market_observation(
    *,
    signal_kind: str,
    subject_listings: Iterable[ListingGovernanceBinding],
    source_refs: Iterable[str],
    confidence_bps: int,
    evidence_record: EvidenceRecord,
    governance_authority_id: str,
) -> MarketObservation:
    """Seal detector output and bind it to an exact persisted evidence row."""

    listings = tuple(
        sorted(subject_listings, key=lambda item: (item.merchant_id, item.sku_id))
    )
    if not listings:
        raise GovernanceBindingError("market observation requires subject listings")
    if len({item.sku_id for item in listings}) != len(listings):
        raise GovernanceBindingError("market observation listing subjects must be unique")
    for item in listings:
        _validate_listing_binding(item)
    sources = _canonical_texts(source_refs, "source_refs")
    if not sources:
        raise GovernanceBindingError("market observation requires source refs")
    kind = _text(signal_kind, "signal_kind")
    confidence = _basis_points(confidence_bps, "confidence_bps")
    authority = _text(governance_authority_id, "governance_authority_id")
    validate_evidence_record(evidence_record)
    if evidence_record.kind != "market_detector_observation":
        raise GovernanceBindingError("market observation evidence has wrong kind")
    if authority != evidence_record.owner_id and authority not in evidence_record.read_acl:
        raise WriteNotAuthorized("governance authority cannot read detector evidence")

    recorded_signal_kind = evidence_record.facts.get("signal_kind")
    recorded_skus = evidence_record.facts.get("subject_sku_ids")
    recorded_sources = evidence_record.facts.get("source_refs")
    recorded_confidence = evidence_record.trust.get("confidence_bps")
    if recorded_signal_kind != kind:
        raise GovernanceBindingError("market observation signal kind differs from evidence")
    if not isinstance(recorded_skus, tuple):
        raise GovernanceBindingError("market observation evidence has invalid subject SKUs")
    if not isinstance(recorded_sources, tuple):
        raise GovernanceBindingError("market observation evidence has invalid source refs")
    if _canonical_texts(recorded_skus, "subject_sku_ids") != tuple(
        sorted(item.sku_id for item in listings)
    ):
        raise GovernanceBindingError("market observation listings differ from evidence")
    if _canonical_texts(recorded_sources, "source_refs") != sources:
        raise GovernanceBindingError("market observation source refs differ from evidence")
    if recorded_confidence != confidence:
        raise GovernanceBindingError("market observation confidence differs from evidence")

    candidate = MarketObservation(
        signal_kind=kind,
        subject_listings=listings,
        source_refs=sources,
        confidence_bps=confidence,
        source_record_id=evidence_record.record_id,
        source_record_digest=evidence_record.record_digest,
        detected_by_id=evidence_record.issuer_id,
        ingested_by_id=authority,
        observation_digest="",
    )
    sealed = replace(
        candidate,
        observation_digest=_digest(_market_observation_contract(candidate)),
    )
    validate_market_observation(sealed)
    return sealed


def validate_market_observation(
    observation: MarketObservation,
    *,
    expected_authority: str | None = None,
) -> None:
    if not isinstance(observation, MarketObservation):
        raise GovernanceBindingError("market observation has wrong type")
    _text(observation.signal_kind, "signal_kind")
    if expected_authority is not None and observation.ingested_by_id != expected_authority:
        raise WriteNotAuthorized("market observation authority mismatch")
    _text(observation.source_record_id, "source_record_id")
    _require_digest(observation.source_record_digest, "source_record_digest")
    _text(observation.detected_by_id, "detected_by_id")
    _text(observation.ingested_by_id, "ingested_by_id")
    _basis_points(observation.confidence_bps, "confidence_bps")
    if not observation.subject_listings:
        raise GovernanceBindingError("market observation has no subjects")
    for item in observation.subject_listings:
        _validate_listing_binding(item)
    if (
        tuple(
            sorted(
                observation.subject_listings,
                key=lambda item: (item.merchant_id, item.sku_id),
            )
        )
        != observation.subject_listings
    ):
        raise GovernanceBindingError("market observation subjects are not canonical")
    _canonical_texts(observation.source_refs, "source_refs")
    if observation.observation_digest != _digest(_market_observation_contract(observation)):
        raise GovernanceBindingError("market observation digest mismatch")


def derive_market_signal(
    *,
    observation: MarketObservation,
    governance_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: MarketSignal | None = None,
) -> MarketSignal:
    """Turn trusted detector output into an immutable authoritative signal."""

    validate_market_observation(observation, expected_authority=governance_authority_id)
    subjects = tuple(sorted({item.merchant_id for item in observation.subject_listings}))
    tick = _nonnegative_int(server_tick, "server_tick")
    key = _text(idempotency_key, "idempotency_key")
    if current is not None and current.idempotency_key == key:
        validate_market_signal(current, expected_authority=governance_authority_id)
        expected = (
            _derived_id("market-signal", observation.observation_digest),
            observation.signal_kind,
            subjects,
            observation.source_refs,
            observation.confidence_bps,
            _merge_digests(
                (observation.observation_digest, observation.source_record_digest),
                (item.listing_digest for item in observation.subject_listings),
            ),
        )
        actual = (
            current.signal_id,
            current.signal_kind,
            current.subject_merchant_ids,
            current.source_refs,
            current.confidence_bps,
            current.provenance_digests,
        )
        if actual != expected:
            raise IdempotencyConflict("market signal changed under existing key")
        return current
    candidate = seal_market_signal(
        MarketSignal(
            signal_id=_derived_id("market-signal", observation.observation_digest),
            signal_kind=observation.signal_kind,
            subject_merchant_ids=subjects,
            source_refs=observation.source_refs,
            confidence_bps=observation.confidence_bps,
            observed_at_tick=tick,
            authored_by=governance_authority_id,
            idempotency_key=key,
            provenance_digests=_merge_digests(
                (observation.observation_digest, observation.source_record_digest),
                (item.listing_digest for item in observation.subject_listings),
            ),
        )
    )
    if current is not None:
        raise GovernanceTransitionError("market signal is immutable")
    validate_market_signal(candidate, expected_authority=governance_authority_id)
    return candidate


def derive_governance_case(
    *,
    case_kind: str,
    signals: Sequence[MarketSignal],
    governance_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: GovernanceCase | None = None,
) -> GovernanceCase:
    """Open a case whose subjects and provenance are derived from signals."""

    rows = tuple(sorted(signals, key=lambda item: (item.observed_at_tick, item.signal_id)))
    if not rows:
        raise GovernanceBindingError("governance case requires market signals")
    for item in rows:
        validate_market_signal(item, expected_authority=governance_authority_id)
    if len({item.signal_digest for item in rows}) != len(rows):
        raise GovernanceBindingError("governance case has duplicate signals")
    subjects = tuple(sorted({merchant for item in rows for merchant in item.subject_merchant_ids}))
    tick = _nonnegative_int(server_tick, "server_tick")
    key = _text(idempotency_key, "idempotency_key")
    case_id = _derived_id("governance-case", case_kind, *(item.signal_digest for item in rows))
    if current is not None and current.idempotency_key == key:
        validate_governance_case(
            current,
            signals=rows,
            expected_authority=governance_authority_id,
        )
        if (
            current.case_id != case_id
            or current.case_kind != case_kind
            or current.subject_merchant_ids != subjects
            or current.signal_digests != tuple(item.signal_digest for item in rows)
            or current.status != "open"
        ):
            raise IdempotencyConflict("governance case changed under existing key")
        return current
    candidate = seal_governance_case(
        GovernanceCase(
            case_id=case_id,
            case_kind=_text(case_kind, "case_kind"),
            subject_merchant_ids=subjects,
            signal_digests=tuple(item.signal_digest for item in rows),
            status="open",
            resolution_code=None,
            opened_at_tick=tick,
            updated_at_tick=tick,
            authored_by=governance_authority_id,
            idempotency_key=key,
        )
    )
    validate_governance_case(
        candidate,
        signals=rows,
        expected_authority=governance_authority_id,
    )
    if current is not None:
        raise GovernanceTransitionError("governance case is already open")
    return candidate


def derive_governance_response_attestation(
    intent: Mapping[str, Any],
    *,
    current: GovernanceCase,
    signals: Sequence[MarketSignal],
    original_actor: str,
    governance_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    existing: GovernanceResponseAttestation | None = None,
) -> GovernanceResponseAttestation:
    """Attest a subject's rejection without changing the governance case."""

    normalized = normalize_governance_intent(intent)
    op = normalized["op"]
    case_kind_by_op = {
        "reject_review_manipulation": "review_integrity",
        "reject_coordination": "competition",
    }
    if op not in case_kind_by_op:
        raise GovernanceIntentError("intent is not a governance response")
    rows = tuple(sorted(signals, key=lambda item: (item.observed_at_tick, item.signal_id)))
    validate_governance_case(
        current,
        signals=rows,
        expected_authority=governance_authority_id,
    )
    if normalized["case_id"] != current.case_id:
        raise GovernanceBindingError("case intent binding mismatch")
    if current.case_kind != case_kind_by_op[op]:
        raise GovernanceTransitionError("intent does not apply to this case kind")
    if current.status not in {"open", "under_review"}:
        raise GovernanceTransitionError("governance case is already terminal")
    actor = _text(original_actor, "original_actor")
    if actor not in current.subject_merchant_ids:
        raise WriteNotAuthorized("actor is not a subject of governance case")
    tick = _nonnegative_int(server_tick, "server_tick")
    if tick < current.updated_at_tick:
        raise GovernanceTransitionError("governance time moved backwards")
    key = _text(idempotency_key, "idempotency_key")
    signal_digests = tuple(item.signal_digest for item in rows)
    fingerprint = governance_intent_fingerprint(
        normalized,
        original_actor=actor,
        trusted_context_digests=(current.case_digest, *signal_digests),
    )
    contract = {
        "schema_id": GOVERNANCE_RESPONSE_ATTESTATION_SCHEMA,
        "response_id": _derived_id("governance-response", current.case_id, actor, key),
        "case_id": current.case_id,
        "case_digest": current.case_digest,
        "subject_merchant_id": actor,
        "response_kind": op,
        "signal_digests": list(signal_digests),
        "submitted_at_tick": tick,
        "authored_by": governance_authority_id,
        "idempotency_key": key,
        "request_fingerprint": fingerprint,
        "provenance_digests": [current.case_digest, *signal_digests],
    }
    candidate = GovernanceResponseAttestation(
        response_id=cast(str, contract["response_id"]),
        case_id=current.case_id,
        case_digest=current.case_digest,
        subject_merchant_id=actor,
        response_kind=op,
        signal_digests=signal_digests,
        submitted_at_tick=tick,
        authored_by=governance_authority_id,
        idempotency_key=key,
        request_fingerprint=fingerprint,
        provenance_digests=(current.case_digest, *signal_digests),
        attestation_digest=_digest(contract),
    )
    validate_governance_response_attestation(
        candidate,
        governance_case=current,
        signals=rows,
        expected_authority=governance_authority_id,
    )
    if existing is None:
        return candidate
    validate_governance_response_attestation(
        existing,
        governance_case=current,
        signals=rows,
        expected_authority=governance_authority_id,
    )
    if existing.idempotency_key != key or existing != candidate:
        raise IdempotencyConflict("governance response changed under existing key")
    return existing


def validate_governance_response_attestation(
    response: GovernanceResponseAttestation,
    *,
    governance_case: GovernanceCase,
    signals: Sequence[MarketSignal],
    expected_authority: str,
) -> None:
    if not isinstance(response, GovernanceResponseAttestation):
        raise GovernanceBindingError("governance response has wrong type")
    validate_governance_case(governance_case, signals=tuple(signals))
    for name in (
        "response_id",
        "case_id",
        "subject_merchant_id",
        "response_kind",
        "authored_by",
        "idempotency_key",
    ):
        _text(getattr(response, name), name)
    _require_digest(response.case_digest, "case_digest")
    _require_digest(response.attestation_digest, "attestation_digest")
    if response.authored_by != expected_authority:
        raise WriteNotAuthorized("governance response authority mismatch")
    if (
        response.case_id != governance_case.case_id
        or response.case_digest != governance_case.case_digest
    ):
        raise GovernanceBindingError("governance response case binding mismatch")
    if response.subject_merchant_id not in governance_case.subject_merchant_ids:
        raise GovernanceBindingError("governance response subject is outside case")
    if response.response_kind not in {
        "reject_review_manipulation",
        "reject_coordination",
    }:
        raise GovernanceBindingError("unsupported governance response kind")
    expected_case_kind = {
        "reject_review_manipulation": "review_integrity",
        "reject_coordination": "competition",
    }[response.response_kind]
    if governance_case.case_kind != expected_case_kind:
        raise GovernanceBindingError("governance response kind does not match case")
    signal_digests = tuple(item.signal_digest for item in signals)
    if response.signal_digests != signal_digests:
        raise GovernanceBindingError("governance response signal binding mismatch")
    _nonnegative_int(response.submitted_at_tick, "submitted_at_tick")
    _text(response.idempotency_key, "idempotency_key")
    _require_digest(response.request_fingerprint, "request_fingerprint")
    expected_provenance = _merge_digests((governance_case.case_digest,), signal_digests)
    if response.provenance_digests != expected_provenance:
        raise GovernanceBindingError("governance response provenance mismatch")
    expected_fingerprint = governance_intent_fingerprint(
        {"op": response.response_kind, "case_id": response.case_id},
        original_actor=response.subject_merchant_id,
        trusted_context_digests=(governance_case.case_digest, *signal_digests),
    )
    if response.request_fingerprint != expected_fingerprint:
        raise GovernanceBindingError("governance response fingerprint mismatch")
    contract = {
        "schema_id": GOVERNANCE_RESPONSE_ATTESTATION_SCHEMA,
        "response_id": response.response_id,
        "case_id": response.case_id,
        "case_digest": response.case_digest,
        "subject_merchant_id": response.subject_merchant_id,
        "response_kind": response.response_kind,
        "signal_digests": list(response.signal_digests),
        "submitted_at_tick": response.submitted_at_tick,
        "authored_by": response.authored_by,
        "idempotency_key": response.idempotency_key,
        "request_fingerprint": response.request_fingerprint,
        "provenance_digests": list(response.provenance_digests),
    }
    if response.attestation_digest != _digest(contract):
        raise GovernanceBindingError("governance response digest mismatch")


def derive_governance_resolution_decision(
    *,
    governance_case: GovernanceCase,
    signals: Sequence[MarketSignal],
    responses: Sequence[GovernanceResponseAttestation],
    resolution_kind: str,
    policy_id: str,
    policy_version: int,
    original_actor: str,
    governance_authority_id: str,
    server_tick: int,
    idempotency_key: str,
) -> GovernanceResolutionDecision:
    """Derive a case outcome only from an authenticated Platform policy action."""

    signal_rows = tuple(sorted(signals, key=lambda item: (item.observed_at_tick, item.signal_id)))
    validate_governance_case(governance_case, signals=signal_rows)
    if _text(original_actor, "original_actor") != governance_authority_id:
        raise WriteNotAuthorized("only governance authority may decide a case")
    outcome = {
        "violation_confirmed": ("resolved", "violation_confirmed"),
        "no_violation": ("dismissed", "no_violation"),
        "compliant_rejection_recorded": ("resolved", "compliant_rejection_recorded"),
    }.get(resolution_kind)
    if outcome is None:
        raise GovernanceTransitionError("unsupported governance resolution kind")
    rows = tuple(sorted(responses, key=lambda item: item.attestation_digest))
    if resolution_kind == "compliant_rejection_recorded" and not rows:
        raise GovernanceTransitionError("compliant rejection requires an attestation")
    for response in rows:
        validate_governance_response_attestation(
            response,
            governance_case=governance_case,
            signals=signal_rows,
            expected_authority=governance_authority_id,
        )
    tick = _nonnegative_int(server_tick, "server_tick")
    key = _text(idempotency_key, "idempotency_key")
    target_status, resolution_code = outcome
    contract = {
        "schema_id": GOVERNANCE_RESOLUTION_DECISION_SCHEMA,
        "decision_id": _derived_id("governance-decision", governance_case.case_id, key),
        "case_id": governance_case.case_id,
        "case_digest": governance_case.case_digest,
        "resolution_kind": resolution_kind,
        "target_status": target_status,
        "resolution_code": resolution_code,
        "policy_id": _text(policy_id, "policy_id"),
        "policy_version": _positive_int(policy_version, "policy_version"),
        "response_digests": [item.attestation_digest for item in rows],
        "decided_at_tick": tick,
        "authored_by": governance_authority_id,
        "idempotency_key": key,
    }
    decision = GovernanceResolutionDecision(
        decision_id=cast(str, contract["decision_id"]),
        case_id=governance_case.case_id,
        case_digest=governance_case.case_digest,
        resolution_kind=resolution_kind,
        target_status=target_status,
        resolution_code=resolution_code,
        policy_id=cast(str, contract["policy_id"]),
        policy_version=cast(int, contract["policy_version"]),
        response_digests=tuple(item.attestation_digest for item in rows),
        decided_at_tick=tick,
        authored_by=governance_authority_id,
        idempotency_key=key,
        decision_digest=_digest(contract),
    )
    validate_governance_resolution_decision(
        decision,
        governance_case=governance_case,
        expected_authority=governance_authority_id,
    )
    return decision


def validate_governance_resolution_decision(
    decision: GovernanceResolutionDecision,
    *,
    governance_case: GovernanceCase,
    expected_authority: str,
) -> None:
    if not isinstance(decision, GovernanceResolutionDecision):
        raise GovernanceBindingError("governance decision has wrong type")
    validate_governance_case(governance_case)
    for name in (
        "decision_id",
        "case_id",
        "resolution_kind",
        "target_status",
        "resolution_code",
        "policy_id",
        "authored_by",
        "idempotency_key",
    ):
        _text(getattr(decision, name), name)
    _require_digest(decision.case_digest, "case_digest")
    _require_digest(decision.decision_digest, "decision_digest")
    if decision.authored_by != expected_authority:
        raise WriteNotAuthorized("governance decision authority mismatch")
    if (
        decision.case_id != governance_case.case_id
        or decision.case_digest != governance_case.case_digest
    ):
        raise GovernanceBindingError("governance decision case binding mismatch")
    expected_outcome = {
        "violation_confirmed": ("resolved", "violation_confirmed"),
        "no_violation": ("dismissed", "no_violation"),
        "compliant_rejection_recorded": (
            "resolved",
            "compliant_rejection_recorded",
        ),
    }.get(decision.resolution_kind)
    if expected_outcome != (decision.target_status, decision.resolution_code):
        raise GovernanceBindingError("governance decision outcome is not canonical")
    _text(decision.policy_id, "policy_id")
    _positive_int(decision.policy_version, "policy_version")
    tick = _nonnegative_int(decision.decided_at_tick, "decided_at_tick")
    if tick < governance_case.updated_at_tick:
        raise GovernanceTransitionError("governance decision time moved backwards")
    _text(decision.idempotency_key, "idempotency_key")
    response_digests = _canonical_digests(decision.response_digests, "response_digests")
    if response_digests != decision.response_digests:
        raise GovernanceBindingError("governance response digests are not canonical")
    if decision.resolution_kind == "compliant_rejection_recorded" and not response_digests:
        raise GovernanceTransitionError("compliant rejection requires an attestation")
    contract = {
        "schema_id": GOVERNANCE_RESOLUTION_DECISION_SCHEMA,
        "decision_id": decision.decision_id,
        "case_id": decision.case_id,
        "case_digest": decision.case_digest,
        "resolution_kind": decision.resolution_kind,
        "target_status": decision.target_status,
        "resolution_code": decision.resolution_code,
        "policy_id": decision.policy_id,
        "policy_version": decision.policy_version,
        "response_digests": list(decision.response_digests),
        "decided_at_tick": decision.decided_at_tick,
        "authored_by": decision.authored_by,
        "idempotency_key": decision.idempotency_key,
    }
    if decision.decision_digest != _digest(contract):
        raise GovernanceBindingError("governance decision digest mismatch")


def derive_governance_case_resolution(
    *,
    current: GovernanceCase,
    signals: Sequence[MarketSignal],
    decision: GovernanceResolutionDecision,
    governance_authority_id: str,
) -> GovernanceCase:
    """Apply a separately authored Platform decision to a versioned case."""

    rows = tuple(sorted(signals, key=lambda item: (item.observed_at_tick, item.signal_id)))
    validate_governance_case(
        current,
        signals=rows,
        expected_authority=governance_authority_id,
    )
    validate_governance_resolution_decision(
        decision,
        governance_case=current,
        expected_authority=governance_authority_id,
    )
    if current.status not in {"open", "under_review"}:
        raise GovernanceTransitionError("governance case is already terminal")
    candidate = seal_governance_case(
        replace(
            current,
            status=decision.target_status,
            resolution_code=decision.resolution_code,
            updated_at_tick=decision.decided_at_tick,
            authored_by=governance_authority_id,
            idempotency_key=decision.idempotency_key,
            version=current.version + 1,
            previous_digest=current.case_digest,
            provenance_digests=_merge_digests(
                current.provenance_digests,
                (
                    current.case_digest,
                    decision.decision_digest,
                    *decision.response_digests,
                ),
            ),
            case_digest="",
        )
    )
    validate_version_successor(
        current,
        candidate,
        expected_authority=governance_authority_id,
    )
    return candidate


def build_reputation_policy_revision(
    *,
    policy_id: str,
    revision: int,
    previous_digest: str | None,
    effective_tick: int,
    published_by_id: str,
    fulfilled_order_bps: int,
    disputed_order_bps: int,
    refund_bps: int,
    remediation_verified_bps: int,
    compliance_violation_bps: int,
) -> ReputationPolicyRevision:
    """Seal a trusted policy that determines all reputation outcomes."""

    candidate = ReputationPolicyRevision(
        policy_id=_text(policy_id, "policy_id"),
        revision=_positive_int(revision, "revision"),
        previous_digest=previous_digest,
        effective_tick=_nonnegative_int(effective_tick, "effective_tick"),
        published_by_id=_text(published_by_id, "published_by_id"),
        fulfilled_order_bps=_basis_points(fulfilled_order_bps, "fulfilled_order_bps"),
        disputed_order_bps=_basis_points(disputed_order_bps, "disputed_order_bps"),
        refund_bps=_basis_points(refund_bps, "refund_bps"),
        remediation_verified_bps=_basis_points(
            remediation_verified_bps, "remediation_verified_bps"
        ),
        compliance_violation_bps=_basis_points(
            compliance_violation_bps, "compliance_violation_bps"
        ),
        policy_digest="",
    )
    sealed = replace(candidate, policy_digest=_digest(_reputation_policy_contract(candidate)))
    validate_reputation_policy(sealed)
    return sealed


def validate_reputation_policy(policy: ReputationPolicyRevision) -> None:
    if not isinstance(policy, ReputationPolicyRevision):
        raise GovernanceBindingError("reputation policy has wrong type")
    _text(policy.policy_id, "policy_id")
    _positive_int(policy.revision, "revision")
    _nonnegative_int(policy.effective_tick, "effective_tick")
    _text(policy.published_by_id, "published_by_id")
    for name in (
        "fulfilled_order_bps",
        "disputed_order_bps",
        "refund_bps",
        "remediation_verified_bps",
        "compliance_violation_bps",
    ):
        _basis_points(getattr(policy, name), name)
    if policy.revision == 1:
        if policy.previous_digest is not None:
            raise GovernanceBindingError("first reputation policy has no predecessor")
    else:
        _require_digest(policy.previous_digest, "previous_digest")
    if policy.policy_digest != _digest(_reputation_policy_contract(policy)):
        raise GovernanceBindingError("reputation policy digest mismatch")


def validate_reputation_policy_transition(
    current: ReputationPolicyRevision | None,
    candidate: ReputationPolicyRevision,
    *,
    original_actor: str,
    server_tick: int,
    trusted_publisher_ids: Iterable[str],
) -> AppendDisposition:
    """Validate publisher authority, logical time, and contiguous policy history."""

    validate_reputation_policy(candidate)
    actor = _text(original_actor, "original_actor")
    trusted = frozenset(_canonical_texts(trusted_publisher_ids, "trusted publishers"))
    if actor not in trusted or candidate.published_by_id != actor:
        raise WriteNotAuthorized("reputation policy publisher is not trusted")
    if candidate.effective_tick != _nonnegative_int(server_tick, "server_tick"):
        raise GovernanceTransitionError("policy effective tick must equal World time")
    if current is None:
        if candidate.revision != 1 or candidate.previous_digest is not None:
            raise GovernanceTransitionError("policy history must begin at revision 1")
        return "append"
    validate_reputation_policy(current)
    if candidate.policy_digest == current.policy_digest:
        return "idempotent"
    if candidate.policy_id != current.policy_id:
        raise GovernanceTransitionError("reputation policy identity changed")
    if (
        candidate.revision != current.revision + 1
        or candidate.previous_digest != current.policy_digest
    ):
        raise GovernanceTransitionError("reputation policy lineage is not contiguous")
    if candidate.effective_tick <= current.effective_tick:
        raise GovernanceTransitionError("reputation policy time must increase")
    return "append"


def derive_settlement_reputation_event(
    *,
    purchase_binding: PurchaseGovernanceBinding,
    listing_binding: ListingGovernanceBinding,
    policy: ReputationPolicyRevision,
    reputation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: ReputationEvent | None = None,
) -> ReputationEvent:
    """Append fulfilled-order reputation from a real World settlement."""

    _validate_purchase_binding(purchase_binding)
    _validate_listing_binding(listing_binding)
    if (
        purchase_binding.sku_id != listing_binding.sku_id
        or purchase_binding.merchant_id != listing_binding.merchant_id
        or purchase_binding.listing_digest != listing_binding.listing_digest
    ):
        raise GovernanceBindingError("settlement and listing binding mismatch")
    return _derive_reputation_event(
        event_kind="fulfilled_order",
        merchant_id=listing_binding.merchant_id,
        source_ref=f"settlement:{purchase_binding.txn_id}",
        source_digest=purchase_binding.binding_digest,
        policy=policy,
        reputation_authority_id=reputation_authority_id,
        server_tick=server_tick,
        idempotency_key=idempotency_key,
        current=current,
    )


def derive_governance_reputation_event(
    *,
    governance_case: GovernanceCase,
    listing_binding: ListingGovernanceBinding,
    policy: ReputationPolicyRevision,
    reputation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: ReputationEvent | None = None,
) -> ReputationEvent:
    """Append a violation event from a terminal governance case."""

    validate_governance_case(governance_case)
    _validate_listing_binding(listing_binding)
    if listing_binding.merchant_id not in governance_case.subject_merchant_ids:
        raise GovernanceBindingError("listing owner is not a governance case subject")
    if governance_case.status != "resolved":
        raise GovernanceTransitionError("reputation requires a resolved governance case")
    return _derive_reputation_event(
        event_kind="compliance_violation",
        merchant_id=listing_binding.merchant_id,
        source_ref=f"governance-case:{governance_case.case_id}",
        source_digest=governance_case.case_digest,
        policy=policy,
        reputation_authority_id=reputation_authority_id,
        server_tick=server_tick,
        idempotency_key=idempotency_key,
        current=current,
    )


def derive_remediation_reputation_event(
    *,
    remediation_plan: RemediationPlan,
    listing_binding: ListingGovernanceBinding,
    policy: ReputationPolicyRevision,
    reputation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: ReputationEvent | None = None,
) -> ReputationEvent:
    """Append remediation credit only after all steps are verified."""

    validate_remediation_plan(remediation_plan)
    _validate_listing_binding(listing_binding)
    if remediation_plan.owner_merchant_id != listing_binding.merchant_id:
        raise GovernanceBindingError("remediation owner and listing owner differ")
    if remediation_plan.status != "completed" or any(
        item.status != "verified" for item in remediation_plan.steps
    ):
        raise GovernanceTransitionError("reputation credit requires completed remediation")
    return _derive_reputation_event(
        event_kind="remediation_verified",
        merchant_id=listing_binding.merchant_id,
        source_ref=f"remediation-plan:{remediation_plan.plan_id}",
        source_digest=remediation_plan.plan_digest,
        policy=policy,
        reputation_authority_id=reputation_authority_id,
        server_tick=server_tick,
        idempotency_key=idempotency_key,
        current=current,
    )


def _derive_reputation_event(
    *,
    event_kind: str,
    merchant_id: str,
    source_ref: str,
    source_digest: str,
    policy: ReputationPolicyRevision,
    reputation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: ReputationEvent | None,
) -> ReputationEvent:
    validate_reputation_policy(policy)
    if policy.published_by_id != reputation_authority_id:
        raise WriteNotAuthorized("reputation policy authority mismatch")
    tick = _nonnegative_int(server_tick, "server_tick")
    if policy.effective_tick > tick:
        raise GovernanceTransitionError("reputation policy is not yet effective")
    _require_digest(source_digest, "source_digest")
    key = _text(idempotency_key, "idempotency_key")
    outcome_by_kind = {
        "fulfilled_order": policy.fulfilled_order_bps,
        "disputed_order": policy.disputed_order_bps,
        "refund": policy.refund_bps,
        "remediation_verified": policy.remediation_verified_bps,
        "compliance_violation": policy.compliance_violation_bps,
    }
    outcome = outcome_by_kind[event_kind]
    version = 1 if current is None else current.version + 1
    provenance = (
        _merge_digests((policy.policy_digest, source_digest))
        if current is None
        else _merge_digests(
            current.provenance_digests,
            (current.event_digest, policy.policy_digest, source_digest),
        )
    )
    candidate = seal_reputation_event(
        ReputationEvent(
            event_id=_derived_id("reputation-event", merchant_id, key),
            merchant_id=_text(merchant_id, "merchant_id"),
            event_kind=event_kind,
            source_ref=_text(source_ref, "source_ref"),
            outcome_bps=outcome,
            occurred_at_tick=tick,
            authored_by=reputation_authority_id,
            idempotency_key=key,
            version=version,
            previous_digest=None if current is None else current.event_digest,
            provenance_digests=provenance,
        )
    )
    validate_reputation_event(
        candidate,
        expected_authority=reputation_authority_id,
        expected_owner_id=merchant_id,
    )
    if current is None:
        return candidate
    if current.idempotency_key == key:
        if (
            current.merchant_id != merchant_id
            or current.event_kind != event_kind
            or current.source_ref != source_ref
            or current.outcome_bps != outcome
        ):
            raise IdempotencyConflict("reputation event changed under existing key")
        return current
    validate_reputation_successor(
        current,
        candidate,
        expected_authority=reputation_authority_id,
        expected_owner_id=merchant_id,
    )
    return candidate


def build_remediation_blueprint(
    *,
    blueprint_id: str,
    governance_case_kind: str,
    steps: Iterable[RemediationBlueprintStep],
    issued_by_id: str,
) -> RemediationBlueprint:
    """Seal a trusted, ordered remediation template."""

    rows = tuple(steps)
    candidate = RemediationBlueprint(
        blueprint_id=_text(blueprint_id, "blueprint_id"),
        governance_case_kind=_text(governance_case_kind, "governance_case_kind"),
        steps=rows,
        issued_by_id=_text(issued_by_id, "issued_by_id"),
        blueprint_digest="",
    )
    sealed = replace(
        candidate, blueprint_digest=_digest(_remediation_blueprint_contract(candidate))
    )
    validate_remediation_blueprint(sealed)
    return sealed


def validate_remediation_blueprint(
    blueprint: RemediationBlueprint,
    *,
    expected_authority: str | None = None,
) -> None:
    if not isinstance(blueprint, RemediationBlueprint):
        raise GovernanceBindingError("remediation blueprint has wrong type")
    _text(blueprint.blueprint_id, "blueprint_id")
    _text(blueprint.governance_case_kind, "governance_case_kind")
    _text(blueprint.issued_by_id, "issued_by_id")
    if expected_authority is not None and blueprint.issued_by_id != expected_authority:
        raise WriteNotAuthorized("remediation blueprint authority mismatch")
    if not blueprint.steps:
        raise GovernanceBindingError("remediation blueprint requires steps")
    for sequence_no, step in enumerate(blueprint.steps, start=1):
        if not isinstance(step, RemediationBlueprintStep):
            raise GovernanceBindingError("remediation blueprint step has wrong type")
        if step.action_kind not in REMEDIATION_ACTION_KINDS:
            raise GovernanceBindingError("unknown remediation action kind")
        prerequisites = tuple(sorted(set(step.prerequisite_sequence_nos)))
        if prerequisites != step.prerequisite_sequence_nos:
            raise GovernanceBindingError("remediation prerequisites must be canonical")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in prerequisites
        ):
            raise GovernanceBindingError("remediation prerequisite must be positive")
        if any(item >= sequence_no for item in prerequisites):
            raise GovernanceBindingError("remediation prerequisite must be earlier")
    if blueprint.blueprint_digest != _digest(_remediation_blueprint_contract(blueprint)):
        raise GovernanceBindingError("remediation blueprint digest mismatch")


def derive_purchase_governance_evidence(
    *,
    purchase_binding: PurchaseGovernanceBinding,
    listing_binding: ListingGovernanceBinding,
    evidence_kind: str,
    verified_by_id: str,
    server_tick: int,
) -> GovernanceEvidenceBinding:
    """Derive remediation evidence from an authoritative settlement binding."""

    _validate_purchase_binding(purchase_binding)
    _validate_listing_binding(listing_binding)
    if (
        purchase_binding.merchant_id != listing_binding.merchant_id
        or purchase_binding.sku_id != listing_binding.sku_id
        or purchase_binding.listing_digest != listing_binding.listing_digest
    ):
        raise GovernanceBindingError("purchase evidence and listing differ")
    return _seal_governance_evidence(
        source_ref=f"settlement:{purchase_binding.txn_id}",
        merchant_id=listing_binding.merchant_id,
        evidence_kind=evidence_kind,
        observed_at_tick=server_tick,
        verified_by_id=verified_by_id,
        source_digest=purchase_binding.binding_digest,
    )


def derive_case_governance_evidence(
    *,
    governance_case: GovernanceCase,
    listing_binding: ListingGovernanceBinding,
    evidence_kind: str,
    verified_by_id: str,
    server_tick: int,
) -> GovernanceEvidenceBinding:
    """Derive remediation evidence from an authoritative governance case."""

    validate_governance_case(governance_case)
    _validate_listing_binding(listing_binding)
    if listing_binding.merchant_id not in governance_case.subject_merchant_ids:
        raise GovernanceBindingError("case evidence owner is not a case subject")
    return _seal_governance_evidence(
        source_ref=f"governance-case:{governance_case.case_id}",
        merchant_id=listing_binding.merchant_id,
        evidence_kind=evidence_kind,
        observed_at_tick=server_tick,
        verified_by_id=verified_by_id,
        source_digest=governance_case.case_digest,
    )


def derive_record_governance_evidence(
    *,
    evidence_record: EvidenceRecord,
    merchant_id: str,
    expected_subject_id: str,
    evidence_kind: str,
    verified_by_id: str,
) -> GovernanceEvidenceBinding:
    """Bind a persisted external evidence record to one remediation subject.

    The merchant, step subject, kind, time, and digest are checked against the
    sealed World record.  An actor never supplies an evidence reference that
    the World merely echoes into a remediation plan.
    """

    validate_evidence_record(evidence_record)
    owner = _text(merchant_id, "merchant_id")
    subject = _text(expected_subject_id, "expected_subject_id")
    kind = _text(evidence_kind, "evidence_kind")
    if evidence_record.owner_id != owner:
        raise GovernanceBindingError("governance evidence crosses merchant owner")
    if evidence_record.subject_id != subject:
        raise GovernanceBindingError("governance evidence subject mismatch")
    recorded_kind = evidence_record.facts.get("evidence_kind")
    if recorded_kind != kind:
        raise GovernanceBindingError("governance evidence kind mismatch")
    return _seal_governance_evidence(
        source_ref=f"evidence-record:{evidence_record.record_id}",
        merchant_id=owner,
        evidence_kind=kind,
        observed_at_tick=evidence_record.issued_at_tick,
        verified_by_id=verified_by_id,
        source_digest=evidence_record.record_digest,
    )


def derive_remediation_plan(
    *,
    governance_case: GovernanceCase,
    listing_binding: ListingGovernanceBinding,
    blueprint: RemediationBlueprint,
    remediation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: RemediationPlan | None = None,
) -> RemediationPlan:
    """Create a draft plan whose owner and actions come from trusted state."""

    validate_governance_case(governance_case)
    _validate_listing_binding(listing_binding)
    validate_remediation_blueprint(blueprint, expected_authority=remediation_authority_id)
    if governance_case.status != "resolved":
        raise GovernanceTransitionError("remediation requires a resolved case")
    if governance_case.case_kind != blueprint.governance_case_kind:
        raise GovernanceBindingError("blueprint does not apply to governance case")
    owner = listing_binding.merchant_id
    if owner not in governance_case.subject_merchant_ids:
        raise GovernanceBindingError("remediation owner is not a case subject")
    tick = _nonnegative_int(server_tick, "server_tick")
    key = _text(idempotency_key, "idempotency_key")
    plan_id = _derived_id(
        "remediation-plan", governance_case.case_id, owner, blueprint.blueprint_id
    )
    step_ids = tuple(
        _derived_id("remediation-step", plan_id, str(index))
        for index in range(1, len(blueprint.steps) + 1)
    )
    if current is not None and current.idempotency_key == key:
        validate_remediation_plan(
            current,
            expected_authority=remediation_authority_id,
            expected_owner_id=owner,
        )
        if (
            current.plan_id != plan_id
            or current.governance_case_id != governance_case.case_id
            or current.owner_merchant_id != owner
            or current.status != "draft"
            or tuple(item.step_id for item in current.steps) != step_ids
            or tuple(item.action_kind for item in current.steps)
            != tuple(item.action_kind for item in blueprint.steps)
        ):
            raise IdempotencyConflict("remediation plan changed under existing key")
        return current
    steps = tuple(
        seal_remediation_step(
            RemediationStep(
                step_id=step_ids[index - 1],
                plan_id=plan_id,
                owner_merchant_id=owner,
                sequence_no=index,
                action_kind=spec.action_kind,
                prerequisite_step_ids=tuple(
                    step_ids[value - 1] for value in spec.prerequisite_sequence_nos
                ),
                status="pending",
                evidence_refs=(),
                completed_at_tick=None,
                authored_by=remediation_authority_id,
                idempotency_key=f"{key}:step:{index}",
                provenance_digests=(blueprint.blueprint_digest,),
            )
        )
        for index, spec in enumerate(blueprint.steps, start=1)
    )
    candidate = seal_remediation_plan(
        RemediationPlan(
            plan_id=plan_id,
            governance_case_id=governance_case.case_id,
            owner_merchant_id=owner,
            status="draft",
            created_at_tick=tick,
            updated_at_tick=tick,
            steps=steps,
            authored_by=remediation_authority_id,
            idempotency_key=key,
            provenance_digests=(
                governance_case.case_digest,
                blueprint.blueprint_digest,
                listing_binding.listing_digest,
            ),
        )
    )
    validate_remediation_plan(
        candidate,
        expected_authority=remediation_authority_id,
        expected_owner_id=owner,
    )
    if current is not None:
        raise GovernanceTransitionError("remediation plan is already created")
    return candidate


def derive_remediation_plan_activation(
    intent: Mapping[str, Any],
    *,
    current: RemediationPlan,
    original_actor: str,
    remediation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
) -> RemediationPlan:
    """Activate a plan on the owner's compact acceptance intent."""

    normalized = _intent(intent, "accept_remediation_plan")
    validate_remediation_plan(
        current,
        expected_authority=remediation_authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    if normalized["plan_id"] != current.plan_id:
        raise GovernanceBindingError("remediation plan intent binding mismatch")
    actor = _text(original_actor, "original_actor")
    if actor != current.owner_merchant_id:
        raise WriteNotAuthorized("only remediation owner may activate plan")
    tick = _nonnegative_int(server_tick, "server_tick")
    if tick < current.updated_at_tick:
        raise GovernanceTransitionError("remediation time moved backwards")
    key = _text(idempotency_key, "idempotency_key")
    if current.idempotency_key == key:
        if current.status != "active":
            raise IdempotencyConflict("plan activation changed under existing key")
        return current
    if current.status != "draft":
        raise GovernanceTransitionError("only draft remediation plan can activate")
    return _next_remediation_plan(
        current,
        status="active",
        steps=current.steps,
        tick=tick,
        key=key,
        authority_id=remediation_authority_id,
    )


def derive_remediation_step_completion(
    intent: Mapping[str, Any],
    *,
    current: RemediationPlan,
    evidence: Sequence[GovernanceEvidenceBinding],
    original_actor: str,
    remediation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
) -> RemediationPlan:
    """Complete one step using trusted evidence, never merchant-provided refs."""

    normalized = _intent(intent, "complete_remediation_step")
    validate_remediation_plan(
        current,
        expected_authority=remediation_authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    if normalized["plan_id"] != current.plan_id:
        raise GovernanceBindingError("remediation plan intent binding mismatch")
    actor = _text(original_actor, "original_actor")
    if actor != current.owner_merchant_id:
        raise WriteNotAuthorized("only remediation owner may complete a step")
    if current.status != "active":
        raise GovernanceTransitionError("step completion requires active plan")
    step_id = normalized["step_id"]
    selected = next((item for item in current.steps if item.step_id == step_id), None)
    if selected is None:
        raise GovernanceBindingError("step is not in remediation plan")
    rows = tuple(evidence)
    if not rows:
        raise GovernanceTransitionError(
            "step completion requires trusted evidence"
        )
    for item in rows:
        _validate_governance_evidence(item)
        if item.merchant_id != current.owner_merchant_id:
            raise GovernanceBindingError("remediation evidence crosses merchant owner")
    completed = {item.step_id for item in current.steps if item.status == "verified"}
    if not set(selected.prerequisite_step_ids).issubset(completed):
        raise GovernanceTransitionError("remediation prerequisites are not verified")
    tick = _nonnegative_int(server_tick, "server_tick")
    if any(item.observed_at_tick > tick for item in rows):
        raise GovernanceBindingError("remediation evidence is from the future")
    key = _text(idempotency_key, "idempotency_key")
    evidence_refs = tuple(sorted({item.source_ref for item in rows}))
    evidence_digests = tuple(sorted({item.evidence_digest for item in rows}))
    if current.idempotency_key == key:
        if selected.status != "completed" or selected.evidence_refs != evidence_refs:
            raise IdempotencyConflict("step completion changed under existing key")
        return current
    if selected.status != "pending":
        raise GovernanceTransitionError("only pending remediation step can complete")
    updated = seal_remediation_step(
        replace(
            selected,
            status="completed",
            evidence_refs=evidence_refs,
            completed_at_tick=tick,
            idempotency_key=f"{key}:step",
            version=selected.version + 1,
            previous_digest=selected.step_digest,
            provenance_digests=_merge_digests(
                selected.provenance_digests,
                (selected.step_digest, *evidence_digests),
            ),
            step_digest="",
        )
    )
    steps = tuple(updated if item.step_id == step_id else item for item in current.steps)
    return _next_remediation_plan(
        current,
        status="active",
        steps=steps,
        tick=tick,
        key=key,
        authority_id=remediation_authority_id,
    )


def derive_remediation_step_verification(
    *,
    current: RemediationPlan,
    step_id: str,
    original_actor: str,
    remediation_authority_id: str,
    server_tick: int,
    idempotency_key: str,
) -> RemediationPlan:
    """Verify a completed step as a trusted service transition."""

    validate_remediation_plan(
        current,
        expected_authority=remediation_authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    actor = _text(original_actor, "original_actor")
    if actor != remediation_authority_id:
        raise WriteNotAuthorized("only remediation authority may verify a step")
    selected = next((item for item in current.steps if item.step_id == step_id), None)
    if selected is None:
        raise GovernanceBindingError("step is not in remediation plan")
    tick = _nonnegative_int(server_tick, "server_tick")
    if tick < current.updated_at_tick:
        raise GovernanceTransitionError("remediation time moved backwards")
    key = _text(idempotency_key, "idempotency_key")
    if current.idempotency_key == key:
        if selected.status != "verified":
            raise IdempotencyConflict("step verification changed under existing key")
        return current
    if selected.status != "completed":
        raise GovernanceTransitionError("only completed remediation step can verify")
    updated = seal_remediation_step(
        replace(
            selected,
            status="verified",
            idempotency_key=f"{key}:step",
            version=selected.version + 1,
            previous_digest=selected.step_digest,
            provenance_digests=_merge_digests(selected.provenance_digests, (selected.step_digest,)),
            step_digest="",
        )
    )
    steps = tuple(updated if item.step_id == step_id else item for item in current.steps)
    status = "completed" if all(item.status == "verified" for item in steps) else "active"
    return _next_remediation_plan(
        current,
        status=status,
        steps=steps,
        tick=tick,
        key=key,
        authority_id=remediation_authority_id,
    )


def _next_remediation_plan(
    current: RemediationPlan,
    *,
    status: str,
    steps: tuple[RemediationStep, ...],
    tick: int,
    key: str,
    authority_id: str,
) -> RemediationPlan:
    candidate = seal_remediation_plan(
        replace(
            current,
            status=status,
            updated_at_tick=tick,
            steps=steps,
            authored_by=authority_id,
            idempotency_key=key,
            version=current.version + 1,
            previous_digest=current.plan_digest,
            provenance_digests=_merge_digests(current.provenance_digests, (current.plan_digest,)),
            plan_digest="",
        )
    )
    validate_version_successor(
        current,
        candidate,
        expected_authority=authority_id,
        expected_owner_id=current.owner_merchant_id,
    )
    return candidate


def derive_ranking_context(
    *,
    request_id: str,
    requester_id: str,
    mandate_revision: MandateRevision,
    candidate_bindings: Sequence[ListingGovernanceBinding],
    ranked_sku_ids: Sequence[str],
    placements: Sequence[Placement],
    review_aggregates: Sequence[ReviewAggregate],
    governance_cases: Sequence[GovernanceCase],
    reputation_events: Sequence[ReputationEvent],
    policy_id: str,
    policy_version: int,
    ranking_authority_id: str,
    server_tick: int,
    idempotency_key: str,
    current: RankingContext | None = None,
) -> RankingContext:
    """Seal each Platform ranking result as World-persistable exact evidence.

    The ranking algorithm remains a replaceable Platform policy.  This method
    verifies its candidate permutation and binds every evidence version to the
    World-owned listing, mandate, catalog revision, and logical time.
    """

    validate_mandate_revision(mandate_revision)
    buyer_id = _text(mandate_revision.buyer_id, "mandate buyer_id")
    if _text(requester_id, "requester_id") != buyer_id:
        raise WriteNotAuthorized("ranking requester does not own mandate")
    candidates = tuple(candidate_bindings)
    if not candidates:
        raise GovernanceBindingError("ranking context requires candidates")
    by_sku = _listing_binding_index(candidates)
    if len(by_sku) != len(candidates):
        raise GovernanceBindingError("ranking candidates contain duplicate SKUs")
    revisions = {item.catalog_revision for item in candidates}
    if len(revisions) != 1:
        raise GovernanceBindingError("ranking candidates cross catalog snapshots")
    candidate_skus = tuple(item.sku_id for item in candidates)
    ranked = tuple(_text(item, "ranked_sku_id") for item in ranked_sku_ids)
    if len(ranked) != len(set(ranked)) or set(ranked) != set(candidate_skus):
        raise GovernanceBindingError("ranked SKUs must exactly permute candidates")
    candidate_merchants = {item.merchant_id for item in candidates}

    placement_rows = tuple(sorted(placements, key=lambda item: item.placement_id))
    for placement in placement_rows:
        validate_placement(placement)
        listing = by_sku.get(placement.sku_id)
        if listing is None or listing.merchant_id != placement.owner_merchant_id:
            raise GovernanceBindingError("ranking placement is outside candidates")

    aggregate_rows = tuple(
        sorted(review_aggregates, key=lambda item: (item.sku_id, item.aggregate_id))
    )
    for aggregate in aggregate_rows:
        validate_review_aggregate(aggregate)
        listing = by_sku.get(aggregate.sku_id)
        if listing is None or listing.merchant_id != aggregate.merchant_id:
            raise GovernanceBindingError("ranking review aggregate is outside candidates")

    case_rows = tuple(sorted(governance_cases, key=lambda item: item.case_id))
    for governance_case in case_rows:
        validate_governance_case(governance_case)
        if not set(governance_case.subject_merchant_ids).issubset(candidate_merchants):
            raise GovernanceBindingError("ranking governance case is outside candidates")

    reputation_rows = tuple(
        sorted(
            reputation_events,
            key=lambda item: (item.merchant_id, item.version, item.event_id),
        )
    )
    for reputation_event in reputation_rows:
        validate_reputation_event(reputation_event)
        if reputation_event.merchant_id not in candidate_merchants:
            raise GovernanceBindingError("ranking reputation event is outside candidates")

    key = _text(idempotency_key, "idempotency_key")
    tick = _nonnegative_int(server_tick, "server_tick")
    context_id = _derived_id("ranking-context", request_id, buyer_id)
    candidate = seal_ranking_context(
        RankingContext(
            context_id=context_id,
            request_id=_text(request_id, "request_id"),
            buyer_id=buyer_id,
            mandate_id=_text(mandate_revision.mandate_id, "mandate_id"),
            policy_id=_text(policy_id, "policy_id"),
            policy_version=_positive_int(policy_version, "policy_version"),
            catalog_revision=next(iter(revisions)),
            issued_at_tick=tick,
            candidate_sku_ids=candidate_skus,
            ranked_sku_ids=ranked,
            placement_digests=tuple(item.placement_digest for item in placement_rows),
            review_aggregate_digests=tuple(item.aggregate_digest for item in aggregate_rows),
            governance_case_digests=tuple(item.case_digest for item in case_rows),
            reputation_event_digests=tuple(item.event_digest for item in reputation_rows),
            authored_by=ranking_authority_id,
            idempotency_key=key,
            version=1 if current is None else current.version + 1,
            previous_digest=None if current is None else current.context_digest,
            provenance_digests=(
                _merge_digests(
                    (
                        mandate_revision.revision_digest,
                        *(item.listing_digest for item in candidates),
                    )
                )
                if current is None
                else _merge_digests(
                    current.provenance_digests,
                    (
                        current.context_digest,
                        mandate_revision.revision_digest,
                        *(item.listing_digest for item in candidates),
                    ),
                )
            ),
        ),
        server_id=ranking_authority_id,
    )
    validate_ranking_context(candidate, server_id=ranking_authority_id)
    if current is None:
        return candidate
    if current.idempotency_key == key:
        if (
            current.request_id != candidate.request_id
            or current.buyer_id != candidate.buyer_id
            or current.mandate_id != candidate.mandate_id
            or current.policy_id != candidate.policy_id
            or current.policy_version != candidate.policy_version
            or current.catalog_revision != candidate.catalog_revision
            or current.candidate_sku_ids != candidate.candidate_sku_ids
            or current.ranked_sku_ids != candidate.ranked_sku_ids
            or current.placement_digests != candidate.placement_digests
            or current.review_aggregate_digests != candidate.review_aggregate_digests
            or current.governance_case_digests != candidate.governance_case_digests
            or current.reputation_event_digests != candidate.reputation_event_digests
        ):
            raise IdempotencyConflict("ranking context changed under existing key")
        return current
    validate_version_successor(
        current,
        candidate,
        expected_authority=ranking_authority_id,
        expected_owner_id=buyer_id,
    )
    return candidate


def _campaign_terms_contract(terms: AdsCampaignTerms) -> dict[str, Any]:
    return {
        "schema_id": ADS_CAMPAIGN_TERMS_SCHEMA,
        "campaign_id": terms.campaign_id,
        "budget_cents": terms.budget_cents,
        "currency": terms.currency,
        "starts_at_tick": terms.starts_at_tick,
        "ends_at_tick": terms.ends_at_tick,
        "placements": [
            {
                "sku_id": item.sku_id,
                "bid_cents": item.bid_cents,
                "fee_cents": item.fee_cents,
            }
            for item in terms.placements
        ],
        "issued_by_id": terms.issued_by_id,
    }


def _market_observation_contract(observation: MarketObservation) -> dict[str, Any]:
    return {
        "schema_id": MARKET_OBSERVATION_SCHEMA,
        "signal_kind": observation.signal_kind,
        "subject_listings": [
            {
                "sku_id": item.sku_id,
                "product_id": item.product_id,
                "merchant_id": item.merchant_id,
                "catalog_revision": item.catalog_revision,
                "listing_digest": item.listing_digest,
            }
            for item in observation.subject_listings
        ],
        "source_refs": list(observation.source_refs),
        "confidence_bps": observation.confidence_bps,
        "source_record_id": observation.source_record_id,
        "source_record_digest": observation.source_record_digest,
        "detected_by_id": observation.detected_by_id,
        "ingested_by_id": observation.ingested_by_id,
    }


def _reputation_policy_contract(policy: ReputationPolicyRevision) -> dict[str, Any]:
    return {
        "schema_id": REPUTATION_POLICY_SCHEMA,
        "policy_id": policy.policy_id,
        "revision": policy.revision,
        "previous_digest": policy.previous_digest,
        "effective_tick": policy.effective_tick,
        "published_by_id": policy.published_by_id,
        "fulfilled_order_bps": policy.fulfilled_order_bps,
        "disputed_order_bps": policy.disputed_order_bps,
        "refund_bps": policy.refund_bps,
        "remediation_verified_bps": policy.remediation_verified_bps,
        "compliance_violation_bps": policy.compliance_violation_bps,
    }


def _remediation_blueprint_contract(
    blueprint: RemediationBlueprint,
) -> dict[str, Any]:
    return {
        "schema_id": REMEDIATION_BLUEPRINT_SCHEMA,
        "blueprint_id": blueprint.blueprint_id,
        "governance_case_kind": blueprint.governance_case_kind,
        "steps": [
            {
                "action_kind": item.action_kind,
                "prerequisite_sequence_nos": list(item.prerequisite_sequence_nos),
            }
            for item in blueprint.steps
        ],
        "issued_by_id": blueprint.issued_by_id,
    }


def _seal_governance_evidence(
    *,
    source_ref: str,
    merchant_id: str,
    evidence_kind: str,
    observed_at_tick: int,
    verified_by_id: str,
    source_digest: str,
) -> GovernanceEvidenceBinding:
    normalized_source_ref = _text(source_ref, "source_ref")
    normalized_merchant_id = _text(merchant_id, "merchant_id")
    normalized_evidence_kind = _text(evidence_kind, "evidence_kind")
    normalized_tick = _nonnegative_int(observed_at_tick, "observed_at_tick")
    normalized_verifier = _text(verified_by_id, "verified_by_id")
    contract: dict[str, Any] = {
        "schema_id": GOVERNANCE_EVIDENCE_BINDING_SCHEMA,
        "source_ref": normalized_source_ref,
        "merchant_id": normalized_merchant_id,
        "evidence_kind": normalized_evidence_kind,
        "observed_at_tick": normalized_tick,
        "verified_by_id": normalized_verifier,
        "source_digest": source_digest,
    }
    _require_digest(source_digest, "source_digest")
    return GovernanceEvidenceBinding(
        source_ref=normalized_source_ref,
        merchant_id=normalized_merchant_id,
        evidence_kind=normalized_evidence_kind,
        observed_at_tick=normalized_tick,
        verified_by_id=normalized_verifier,
        source_digest=source_digest,
        evidence_digest=_digest(contract),
    )


def _validate_governance_evidence(binding: GovernanceEvidenceBinding) -> None:
    if not isinstance(binding, GovernanceEvidenceBinding):
        raise GovernanceBindingError("governance evidence has wrong type")
    contract = {
        "schema_id": GOVERNANCE_EVIDENCE_BINDING_SCHEMA,
        "source_ref": _text(binding.source_ref, "source_ref"),
        "merchant_id": _text(binding.merchant_id, "merchant_id"),
        "evidence_kind": _text(binding.evidence_kind, "evidence_kind"),
        "observed_at_tick": _nonnegative_int(binding.observed_at_tick, "observed_at_tick"),
        "verified_by_id": _text(binding.verified_by_id, "verified_by_id"),
        "source_digest": binding.source_digest,
    }
    _require_digest(binding.source_digest, "source_digest")
    if binding.evidence_digest != _digest(contract):
        raise GovernanceBindingError("governance evidence digest mismatch")


def _validate_listing_binding(binding: ListingGovernanceBinding) -> None:
    if not isinstance(binding, ListingGovernanceBinding):
        raise GovernanceBindingError("listing governance binding has wrong type")
    _text(binding.sku_id, "sku_id")
    _optional_text(binding.product_id, "product_id")
    merchant = _text(binding.merchant_id, "merchant_id")
    if not merchant.startswith("merchant:"):
        raise GovernanceBindingError("listing owner must be a merchant")
    _nonnegative_int(binding.catalog_revision, "catalog_revision")
    _require_digest(binding.listing_digest, "listing_digest")


def _validate_purchase_binding(binding: PurchaseGovernanceBinding) -> None:
    if not isinstance(binding, PurchaseGovernanceBinding):
        raise GovernanceBindingError("purchase governance binding has wrong type")
    contract = {
        "schema_id": PURCHASE_GOVERNANCE_BINDING_SCHEMA,
        "order_id": _text(binding.order_id, "order_id"),
        "txn_id": _text(binding.txn_id, "txn_id"),
        "buyer_id": _text(binding.buyer_id, "buyer_id"),
        "merchant_id": _text(binding.merchant_id, "merchant_id"),
        "sku_id": _text(binding.sku_id, "sku_id"),
        "qty": _positive_int(binding.qty, "qty"),
        "settled_at_tick": _nonnegative_int(binding.settled_at_tick, "settled_at_tick"),
        "order_digest": binding.order_digest,
        "receipt_digest": binding.receipt_digest,
        "listing_digest": binding.listing_digest,
    }
    for name in ("order_digest", "receipt_digest", "listing_digest"):
        _require_digest(cast(str, contract[name]), name)
    if binding.binding_digest != _digest(contract):
        raise GovernanceBindingError("purchase governance binding digest mismatch")


def validate_purchase_governance_binding(
    binding: PurchaseGovernanceBinding,
) -> None:
    """Validate one World-derived verified-purchase authority projection.

    This public wrapper lets Platform and local Agent authority adapters
    validate a binding returned by World without copying its canonical digest
    contract.  It does not create a binding or infer purchase authority from
    actor text; only :func:`derive_purchase_governance_binding` does that from
    authoritative Order, charge Receipt, and Listing rows.
    """

    _validate_purchase_binding(binding)


def _listing_binding_index(
    bindings: Sequence[ListingGovernanceBinding],
) -> dict[str, ListingGovernanceBinding]:
    rows: dict[str, ListingGovernanceBinding] = {}
    for item in bindings:
        _validate_listing_binding(item)
        if item.sku_id in rows:
            raise GovernanceBindingError("duplicate listing governance binding")
        rows[item.sku_id] = item
    return rows


def _validate_campaign_exact_retry(
    current: Campaign,
    *,
    terms: AdsCampaignTerms,
    listing_bindings: Mapping[str, ListingGovernanceBinding],
    expected_authority: str,
    expected_owner: str,
    expected_status: str,
) -> None:
    validate_campaign(
        current,
        expected_authority=expected_authority,
        expected_owner_id=expected_owner,
    )
    expected = {item.sku_id: (item.bid_cents, item.fee_cents) for item in terms.placements}
    actual = {item.sku_id: (item.bid_cents, item.fee_cents) for item in current.placements}
    if (
        current.campaign_id != terms.campaign_id
        or current.status != expected_status
        or current.budget_cents != terms.budget_cents
        or current.currency != terms.currency
        or current.starts_at_tick != terms.starts_at_tick
        or current.ends_at_tick != terms.ends_at_tick
        or expected != actual
        or set(listing_bindings) != set(actual)
    ):
        raise IdempotencyConflict("campaign changed under existing key")


def _intent(value: Mapping[str, Any], expected_op: str) -> GovernanceIntent:
    normalized = normalize_governance_intent(value)
    if normalized["op"] != expected_op:
        raise GovernanceIntentError(f"expected {expected_op} intent")
    return normalized


def _reject_nested_authority(value: Any, *, path: str = "intent") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in _FORBIDDEN_INTENT_FIELDS:
                raise GovernanceIntentError(
                    f"governance authority field is World-owned: {path}.{key}"
                )
            _reject_nested_authority(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nested_authority(item, path=f"{path}[{index}]")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise GovernanceBindingError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _derived_id(prefix: str, *parts: str) -> str:
    token = _digest({"prefix": prefix, "parts": list(parts)})[:24]
    return f"{prefix}:{token}"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceBindingError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GovernanceBindingError(f"{name} must be a nonnegative integer")
    return cast(int, value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GovernanceBindingError(f"{name} must be a positive integer")
    return cast(int, value)


def _int_range(value: Any, lower: int, upper: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lower or value > upper:
        raise GovernanceIntentError(f"{name} must be in {lower}..{upper}")
    return cast(int, value)


def _basis_points(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000:
        raise GovernanceBindingError(f"{name} must be in 0..10000")
    return cast(int, value)


def _currency(value: Any) -> str:
    text = _text(value, "currency")
    if len(text) != 3 or text.upper() != text or not text.isalpha():
        raise GovernanceBindingError("currency must be a three-letter uppercase code")
    return text


def _canonical_texts(values: Iterable[str], name: str) -> tuple[str, ...]:
    rows = tuple(sorted({_text(value, name) for value in values}))
    return rows


def _canonical_digests(values: Iterable[str], name: str) -> tuple[str, ...]:
    rows = tuple(sorted(set(values)))
    for value in rows:
        _require_digest(value, name)
    return rows


def _merge_digests(*groups: Iterable[str]) -> tuple[str, ...]:
    """Preserve first-seen order while removing overlapping provenance."""

    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            _require_digest(value, "provenance digest")
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return tuple(merged)


def _require_digest(value: str | None, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GovernanceBindingError(f"{name} must be a lowercase SHA-256 digest")


def _decimal_string(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise GovernanceBindingError("money amount must be a finite Decimal")
    return format(value, "f")


__all__ = [
    "AdsCampaignTerms",
    "AdsPlacementTerms",
    "GovernanceBindingError",
    "GovernanceEvidenceBinding",
    "GovernanceIntentError",
    "GovernanceResolutionDecision",
    "GovernanceResponseAttestation",
    "GovernanceTransitionError",
    "ListingGovernanceBinding",
    "MarketGovernanceCoreError",
    "MarketObservation",
    "PurchaseGovernanceBinding",
    "RemediationBlueprint",
    "RemediationBlueprintStep",
    "ReputationPolicyRevision",
    "ReviewAccountBinding",
    "authoritative_listing_digest",
    "authoritative_order_digest",
    "authoritative_receipt_digest",
    "build_ads_campaign_terms",
    "build_market_observation",
    "build_remediation_blueprint",
    "build_reputation_policy_revision",
    "build_review_account_binding",
    "derive_campaign_activation",
    "derive_campaign_publish",
    "derive_case_governance_evidence",
    "derive_governance_case",
    "derive_governance_case_resolution",
    "derive_governance_reputation_event",
    "derive_governance_resolution_decision",
    "derive_governance_response_attestation",
    "derive_listing_governance_binding",
    "derive_market_signal",
    "derive_placement_disclosure",
    "derive_purchase_governance_binding",
    "derive_purchase_governance_evidence",
    "derive_ranking_context",
    "derive_remediation_plan",
    "derive_remediation_plan_activation",
    "derive_remediation_reputation_event",
    "derive_remediation_step_completion",
    "derive_remediation_step_verification",
    "derive_review_aggregate",
    "derive_review_evidence",
    "derive_settlement_reputation_event",
    "governance_intent_fingerprint",
    "normalize_governance_intent",
    "validate_ads_campaign_terms",
    "validate_market_observation",
    "validate_purchase_governance_binding",
    "validate_governance_response_attestation",
    "validate_governance_resolution_decision",
    "validate_remediation_blueprint",
    "validate_reputation_policy",
    "validate_reputation_policy_transition",
    "validate_review_account_binding",
]
