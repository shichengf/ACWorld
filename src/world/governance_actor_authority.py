"""Owner-scoped World projections for high-level governance Agents.

The governance store already contains the facts needed to authorize campaign
publication and verified-purchase reviews.  This module turns those typed,
sealed facts into narrow wire projections.  It does not enumerate tables,
route envelopes, call a model, or commit state.  A World service endpoint can
therefore use these helpers in both in-process and HTTP topologies without
giving Platform direct access to World internals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from protocol.errors import SchemaError
from world.market_governance_core import (
    AdsCampaignTerms,
    GovernanceBindingError,
    PurchaseGovernanceBinding,
    validate_ads_campaign_terms,
    validate_purchase_governance_binding,
)
from world.market_governance_persistence import (
    GovernancePolicyEnvelope,
    MarketGovernancePersistenceError,
    canonical_digest,
    validate_policy_envelope,
)


CAMPAIGN_TERMS_AUTHORITY_SCHEMA = "cwe.owner-campaign-terms-authority.v1"
CAMPAIGN_TERMS_NOTICE_SCHEMA = "cwe.owner-campaign-terms-notice.v1"
VERIFIED_PURCHASE_AUTHORITY_SCHEMA = "cwe.verified-purchase-authority.v1"
VERIFIED_PURCHASE_REVIEW_REQUEST_SCHEMA = (
    "cwe.verified-purchase-review-request.v1"
)
MAX_REVIEW_TEXT_LENGTH = 4_000


class GovernanceActorAuthorityError(SchemaError):
    """A purported owner-scoped World authority is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class OwnerCampaignTermsAuthority:
    """One unpublished campaign policy visible to its listing owner."""

    campaign_id: str
    owner_merchant_id: str
    placement_sku_ids: tuple[str, ...]
    starts_at_tick: int
    ends_at_tick: int
    terms_digest: str
    policy_envelope_digest: str
    policy_logical_tick: int
    authority_digest: str = ""
    schema_version: str = CAMPAIGN_TERMS_AUTHORITY_SCHEMA


@dataclass(frozen=True, slots=True)
class VerifiedPurchaseAuthority:
    """One buyer-scoped review option copied from a sealed purchase binding."""

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
    schema_version: str = VERIFIED_PURCHASE_AUTHORITY_SCHEMA


def build_owner_campaign_terms_authority(
    policy: GovernancePolicyEnvelope,
    *,
    owner_merchant_id: str,
    current_tick: int,
) -> OwnerCampaignTermsAuthority:
    """Project one persisted ads policy after exact owner and time checks."""

    try:
        validate_policy_envelope(policy)
    except (SchemaError, MarketGovernancePersistenceError) as exc:
        raise GovernanceActorAuthorityError(
            f"invalid campaign policy envelope: {exc}"
        ) from exc
    owner = _merchant(owner_merchant_id, "owner_merchant_id")
    tick = _nonnegative_int(current_tick, "current_tick")
    if policy.kind != "ads_campaign_terms" or not isinstance(
        policy.payload, AdsCampaignTerms
    ):
        raise GovernanceActorAuthorityError(
            "campaign publication authority requires ads campaign terms"
        )
    terms = policy.payload
    try:
        validate_ads_campaign_terms(terms, expected_authority="platform:ads")
    except (SchemaError, GovernanceBindingError) as exc:
        raise GovernanceActorAuthorityError(
            f"invalid campaign terms: {exc}"
        ) from exc
    if policy.owner_ids != (owner,):
        raise GovernanceActorAuthorityError(
            "campaign policy is not exclusively owned by the requesting merchant"
        )
    if (
        policy.stable_id != terms.campaign_id
        or policy.semantic_digest != terms.terms_digest
    ):
        raise GovernanceActorAuthorityError(
            "campaign policy envelope and terms identity differ"
        )
    if policy.logical_tick > tick:
        raise GovernanceActorAuthorityError(
            "campaign policy authority is from the future"
        )
    if tick > terms.ends_at_tick:
        raise GovernanceActorAuthorityError(
            "campaign publication authority has expired"
        )
    authority = OwnerCampaignTermsAuthority(
        campaign_id=terms.campaign_id,
        owner_merchant_id=owner,
        placement_sku_ids=tuple(row.sku_id for row in terms.placements),
        starts_at_tick=terms.starts_at_tick,
        ends_at_tick=terms.ends_at_tick,
        terms_digest=terms.terms_digest,
        policy_envelope_digest=policy.envelope_digest,
        policy_logical_tick=policy.logical_tick,
    )
    authority = replace(
        authority,
        authority_digest=canonical_digest(
            _campaign_authority_payload(authority, include_digest=False)
        ),
    )
    validate_owner_campaign_terms_authority(authority)
    return authority


def validate_owner_campaign_terms_authority(
    authority: OwnerCampaignTermsAuthority,
) -> None:
    if not isinstance(authority, OwnerCampaignTermsAuthority):
        raise GovernanceActorAuthorityError(
            "campaign terms authority has the wrong type"
        )
    if authority.schema_version != CAMPAIGN_TERMS_AUTHORITY_SCHEMA:
        raise GovernanceActorAuthorityError(
            "unsupported campaign terms authority schema"
        )
    _text(authority.campaign_id, "campaign_id")
    _merchant(authority.owner_merchant_id, "owner_merchant_id")
    _canonical_texts(authority.placement_sku_ids, "placement_sku_ids")
    starts = _nonnegative_int(authority.starts_at_tick, "starts_at_tick")
    ends = _nonnegative_int(authority.ends_at_tick, "ends_at_tick")
    if ends <= starts:
        raise GovernanceActorAuthorityError(
            "campaign authority lifetime must be positive"
        )
    _digest(authority.terms_digest, "terms_digest")
    _digest(authority.policy_envelope_digest, "policy_envelope_digest")
    _nonnegative_int(authority.policy_logical_tick, "policy_logical_tick")
    expected = canonical_digest(
        _campaign_authority_payload(authority, include_digest=False)
    )
    if authority.authority_digest != expected:
        raise GovernanceActorAuthorityError(
            "campaign terms authority digest mismatch"
        )


def owner_campaign_terms_authority_to_dict(
    authority: OwnerCampaignTermsAuthority,
) -> dict[str, Any]:
    validate_owner_campaign_terms_authority(authority)
    return _campaign_authority_payload(authority, include_digest=True)


def coerce_owner_campaign_terms_authority(
    value: Any,
) -> OwnerCampaignTermsAuthority:
    if isinstance(value, OwnerCampaignTermsAuthority):
        validate_owner_campaign_terms_authority(value)
        return value
    fields = {
        "schema_version",
        "campaign_id",
        "owner_merchant_id",
        "placement_sku_ids",
        "starts_at_tick",
        "ends_at_tick",
        "terms_digest",
        "policy_envelope_digest",
        "policy_logical_tick",
        "authority_digest",
    }
    row = _exact_mapping(value, fields, "campaign terms authority")
    placements = row["placement_sku_ids"]
    if not isinstance(placements, list):
        raise GovernanceActorAuthorityError(
            "campaign placement_sku_ids must be an array"
        )
    authority = OwnerCampaignTermsAuthority(
        schema_version=row["schema_version"],
        campaign_id=row["campaign_id"],
        owner_merchant_id=row["owner_merchant_id"],
        placement_sku_ids=tuple(placements),
        starts_at_tick=row["starts_at_tick"],
        ends_at_tick=row["ends_at_tick"],
        terms_digest=row["terms_digest"],
        policy_envelope_digest=row["policy_envelope_digest"],
        policy_logical_tick=row["policy_logical_tick"],
        authority_digest=row["authority_digest"],
    )
    validate_owner_campaign_terms_authority(authority)
    return authority


def build_owner_campaign_terms_notice(
    policies: Iterable[GovernancePolicyEnvelope],
    *,
    owner_merchant_id: str,
    current_tick: int,
    existing_campaign_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a canonical notice for every current unpublished owner policy."""

    owner = _merchant(owner_merchant_id, "owner_merchant_id")
    tick = _nonnegative_int(current_tick, "current_tick")
    existing = frozenset(
        _text(value, "existing campaign id") for value in existing_campaign_ids
    )
    authorities = tuple(
        sorted(
            (
                build_owner_campaign_terms_authority(
                    row,
                    owner_merchant_id=owner,
                    current_tick=tick,
                )
                for row in policies
                if row.stable_id not in existing
            ),
            key=lambda row: row.campaign_id,
        )
    )
    if not authorities:
        raise GovernanceActorAuthorityError(
            "merchant has no current unpublished campaign terms"
        )
    if len({row.campaign_id for row in authorities}) != len(authorities):
        raise GovernanceActorAuthorityError(
            "campaign terms notice contains duplicate campaign identities"
        )
    body = {
        "schema_version": CAMPAIGN_TERMS_NOTICE_SCHEMA,
        "owner_merchant_id": owner,
        "world_tick": tick,
        "campaign_authorities": [
            owner_campaign_terms_authority_to_dict(row) for row in authorities
        ],
    }
    return {**body, "request_digest": canonical_digest(body)}


def coerce_owner_campaign_terms_notice(
    value: Any,
    *,
    owner_merchant_id: str,
    current_tick: int,
) -> tuple[OwnerCampaignTermsAuthority, ...]:
    """Validate an exact Platform-delivered owner campaign option set."""

    fields = {
        "schema_version",
        "owner_merchant_id",
        "world_tick",
        "campaign_authorities",
        "request_digest",
    }
    row = _exact_mapping(value, fields, "campaign terms notice")
    owner = _merchant(owner_merchant_id, "owner_merchant_id")
    tick = _nonnegative_int(current_tick, "current_tick")
    if (
        row["schema_version"] != CAMPAIGN_TERMS_NOTICE_SCHEMA
        or row["owner_merchant_id"] != owner
        or row["world_tick"] != tick
    ):
        raise GovernanceActorAuthorityError(
            "campaign terms notice actor, tick, or schema authority mismatch"
        )
    raw = row["campaign_authorities"]
    if not isinstance(raw, list) or not raw:
        raise GovernanceActorAuthorityError(
            "campaign terms notice requires a non-empty authority array"
        )
    authorities = tuple(coerce_owner_campaign_terms_authority(item) for item in raw)
    if any(item.owner_merchant_id != owner for item in authorities):
        raise GovernanceActorAuthorityError(
            "campaign terms notice crosses merchant owner"
        )
    if any(tick > item.ends_at_tick for item in authorities):
        raise GovernanceActorAuthorityError(
            "campaign terms notice contains an expired authority"
        )
    if len({item.campaign_id for item in authorities}) != len(authorities):
        raise GovernanceActorAuthorityError(
            "campaign terms notice contains duplicate campaign identities"
        )
    body = {name: row[name] for name in fields if name != "request_digest"}
    _digest(row["request_digest"], "request_digest")
    if row["request_digest"] != canonical_digest(body):
        raise GovernanceActorAuthorityError(
            "campaign terms notice request digest mismatch"
        )
    return authorities


def build_verified_purchase_authority(
    binding: PurchaseGovernanceBinding,
    *,
    buyer_id: str,
    current_tick: int,
) -> VerifiedPurchaseAuthority:
    """Copy one sealed purchase binding after buyer and time validation."""

    try:
        validate_purchase_governance_binding(binding)
    except (SchemaError, GovernanceBindingError) as exc:
        raise GovernanceActorAuthorityError(
            f"invalid verified-purchase binding: {exc}"
        ) from exc
    buyer = _buyer(buyer_id, "buyer_id")
    tick = _nonnegative_int(current_tick, "current_tick")
    if binding.buyer_id != buyer:
        raise GovernanceActorAuthorityError(
            "verified purchase crosses buyer authority"
        )
    if binding.settled_at_tick > tick:
        raise GovernanceActorAuthorityError(
            "verified purchase is from the future"
        )
    authority = VerifiedPurchaseAuthority(
        order_id=binding.order_id,
        txn_id=binding.txn_id,
        buyer_id=binding.buyer_id,
        merchant_id=binding.merchant_id,
        sku_id=binding.sku_id,
        qty=binding.qty,
        settled_at_tick=binding.settled_at_tick,
        order_digest=binding.order_digest,
        receipt_digest=binding.receipt_digest,
        listing_digest=binding.listing_digest,
        binding_digest=binding.binding_digest,
    )
    validate_verified_purchase_authority(authority)
    return authority


def validate_verified_purchase_authority(
    authority: VerifiedPurchaseAuthority,
) -> None:
    if not isinstance(authority, VerifiedPurchaseAuthority):
        raise GovernanceActorAuthorityError(
            "verified-purchase authority has the wrong type"
        )
    if authority.schema_version != VERIFIED_PURCHASE_AUTHORITY_SCHEMA:
        raise GovernanceActorAuthorityError(
            "unsupported verified-purchase authority schema"
        )
    binding = PurchaseGovernanceBinding(
        order_id=authority.order_id,
        txn_id=authority.txn_id,
        buyer_id=authority.buyer_id,
        merchant_id=authority.merchant_id,
        sku_id=authority.sku_id,
        qty=authority.qty,
        settled_at_tick=authority.settled_at_tick,
        order_digest=authority.order_digest,
        receipt_digest=authority.receipt_digest,
        listing_digest=authority.listing_digest,
        binding_digest=authority.binding_digest,
    )
    try:
        validate_purchase_governance_binding(binding)
    except (SchemaError, GovernanceBindingError) as exc:
        raise GovernanceActorAuthorityError(
            f"invalid verified-purchase authority: {exc}"
        ) from exc
    _buyer(authority.buyer_id, "buyer_id")
    _merchant(authority.merchant_id, "merchant_id")


def verified_purchase_authority_to_dict(
    authority: VerifiedPurchaseAuthority,
) -> dict[str, Any]:
    validate_verified_purchase_authority(authority)
    return {
        "schema_version": authority.schema_version,
        "order_id": authority.order_id,
        "txn_id": authority.txn_id,
        "buyer_id": authority.buyer_id,
        "merchant_id": authority.merchant_id,
        "sku_id": authority.sku_id,
        "qty": authority.qty,
        "settled_at_tick": authority.settled_at_tick,
        "order_digest": authority.order_digest,
        "receipt_digest": authority.receipt_digest,
        "listing_digest": authority.listing_digest,
        "binding_digest": authority.binding_digest,
    }


def coerce_verified_purchase_authority(value: Any) -> VerifiedPurchaseAuthority:
    fields = {
        "schema_version",
        "order_id",
        "txn_id",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "qty",
        "settled_at_tick",
        "order_digest",
        "receipt_digest",
        "listing_digest",
        "binding_digest",
    }
    row = _exact_mapping(value, fields, "verified-purchase authority")
    authority = VerifiedPurchaseAuthority(**row)
    validate_verified_purchase_authority(authority)
    return authority


def build_verified_purchase_review_request(
    bindings: Iterable[PurchaseGovernanceBinding],
    *,
    buyer_id: str,
    current_tick: int,
) -> dict[str, Any]:
    """Build the exact buyer-scoped option set for a review Agent turn."""

    buyer = _buyer(buyer_id, "buyer_id")
    tick = _nonnegative_int(current_tick, "current_tick")
    authorities = tuple(
        sorted(
            (
                build_verified_purchase_authority(
                    row,
                    buyer_id=buyer,
                    current_tick=tick,
                )
                for row in bindings
            ),
            key=lambda row: (row.sku_id, row.order_id, row.txn_id),
        )
    )
    if not authorities:
        raise GovernanceActorAuthorityError(
            "buyer has no verified purchase available for review"
        )
    if len({row.binding_digest for row in authorities}) != len(authorities):
        raise GovernanceActorAuthorityError(
            "verified-purchase request contains duplicate bindings"
        )
    body = {
        "schema_version": VERIFIED_PURCHASE_REVIEW_REQUEST_SCHEMA,
        "buyer_id": buyer,
        "world_tick": tick,
        "purchase_authorities": [
            verified_purchase_authority_to_dict(row) for row in authorities
        ],
    }
    return {**body, "request_digest": canonical_digest(body)}


def coerce_verified_purchase_review_request(
    value: Any,
    *,
    buyer_id: str,
    current_tick: int,
) -> tuple[VerifiedPurchaseAuthority, ...]:
    """Validate an exact Platform-delivered verified-purchase option set."""

    fields = {
        "schema_version",
        "buyer_id",
        "world_tick",
        "purchase_authorities",
        "request_digest",
    }
    row = _exact_mapping(value, fields, "verified-purchase review request")
    buyer = _buyer(buyer_id, "buyer_id")
    tick = _nonnegative_int(current_tick, "current_tick")
    if (
        row["schema_version"] != VERIFIED_PURCHASE_REVIEW_REQUEST_SCHEMA
        or row["buyer_id"] != buyer
        or row["world_tick"] != tick
    ):
        raise GovernanceActorAuthorityError(
            "verified-purchase request actor, tick, or schema authority mismatch"
        )
    raw = row["purchase_authorities"]
    if not isinstance(raw, list) or not raw:
        raise GovernanceActorAuthorityError(
            "verified-purchase request requires a non-empty authority array"
        )
    authorities = tuple(coerce_verified_purchase_authority(item) for item in raw)
    if any(item.buyer_id != buyer for item in authorities):
        raise GovernanceActorAuthorityError(
            "verified-purchase request crosses buyer authority"
        )
    if any(item.settled_at_tick > tick for item in authorities):
        raise GovernanceActorAuthorityError(
            "verified-purchase request contains a future settlement"
        )
    if len({item.binding_digest for item in authorities}) != len(authorities):
        raise GovernanceActorAuthorityError(
            "verified-purchase request contains duplicate bindings"
        )
    body = {name: row[name] for name in fields if name != "request_digest"}
    _digest(row["request_digest"], "request_digest")
    if row["request_digest"] != canonical_digest(body):
        raise GovernanceActorAuthorityError(
            "verified-purchase review request digest mismatch"
        )
    return authorities


def normalize_review_text(value: Any) -> str:
    """Normalize the optional actor-authored review text for exact replay."""

    if value is None:
        return ""
    if not isinstance(value, str):
        raise GovernanceActorAuthorityError("review_text must be text")
    text = value.strip()
    if len(text) > MAX_REVIEW_TEXT_LENGTH:
        raise GovernanceActorAuthorityError(
            f"review_text exceeds the {MAX_REVIEW_TEXT_LENGTH}-character limit"
        )
    return text


def _campaign_authority_payload(
    authority: OwnerCampaignTermsAuthority,
    *,
    include_digest: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": authority.schema_version,
        "campaign_id": authority.campaign_id,
        "owner_merchant_id": authority.owner_merchant_id,
        "placement_sku_ids": list(authority.placement_sku_ids),
        "starts_at_tick": authority.starts_at_tick,
        "ends_at_tick": authority.ends_at_tick,
        "terms_digest": authority.terms_digest,
        "policy_envelope_digest": authority.policy_envelope_digest,
        "policy_logical_tick": authority.policy_logical_tick,
    }
    if include_digest:
        payload["authority_digest"] = authority.authority_digest
    return payload


def _exact_mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        missing = [] if not isinstance(value, Mapping) else sorted(fields - set(value))
        unknown = [] if not isinstance(value, Mapping) else sorted(set(value) - fields)
        raise GovernanceActorAuthorityError(
            f"{label} fields are not exact: missing={missing!r}, unknown={unknown!r}"
        )
    return dict(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceActorAuthorityError(f"{label} must be non-empty text")
    return value


def _buyer(value: Any, label: str) -> str:
    actor = _text(value, label)
    if not actor.startswith("buyer:"):
        raise GovernanceActorAuthorityError(f"{label} must name a buyer")
    return actor


def _merchant(value: Any, label: str) -> str:
    actor = _text(value, label)
    if not actor.startswith("merchant:"):
        raise GovernanceActorAuthorityError(f"{label} must name a merchant")
    return actor


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GovernanceActorAuthorityError(
            f"{label} must be a nonnegative integer"
        )
    return value


def _digest(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise GovernanceActorAuthorityError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return digest


def _canonical_texts(value: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise GovernanceActorAuthorityError(f"{label} must be a non-empty tuple")
    rows = tuple(_text(row, label) for row in value)
    if rows != tuple(sorted(set(rows))):
        raise GovernanceActorAuthorityError(
            f"{label} must be sorted and duplicate-free"
        )
    return rows


__all__ = [
    "CAMPAIGN_TERMS_AUTHORITY_SCHEMA",
    "CAMPAIGN_TERMS_NOTICE_SCHEMA",
    "GovernanceActorAuthorityError",
    "MAX_REVIEW_TEXT_LENGTH",
    "OwnerCampaignTermsAuthority",
    "VERIFIED_PURCHASE_AUTHORITY_SCHEMA",
    "VERIFIED_PURCHASE_REVIEW_REQUEST_SCHEMA",
    "VerifiedPurchaseAuthority",
    "build_owner_campaign_terms_authority",
    "build_owner_campaign_terms_notice",
    "build_verified_purchase_authority",
    "build_verified_purchase_review_request",
    "coerce_owner_campaign_terms_authority",
    "coerce_owner_campaign_terms_notice",
    "coerce_verified_purchase_authority",
    "coerce_verified_purchase_review_request",
    "normalize_review_text",
    "owner_campaign_terms_authority_to_dict",
    "validate_owner_campaign_terms_authority",
    "validate_verified_purchase_authority",
    "verified_purchase_authority_to_dict",
]
