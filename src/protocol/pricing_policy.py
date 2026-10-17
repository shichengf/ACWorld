"""Immutable, replay-stable merchant pricing policy contracts.

This module is deliberately a passive protocol layer.  It defines the state
that CommerceWorld must persist before a cart quote can cite a pricing policy,
but it does not read listings, calculate a quote, register actions, or mutate
World.  A World integration is responsible for checking listing and product
ownership, supplying the trusted logical tick, and storing accepted revisions.

Money is represented only as integer minor units.  Percentage rates are
integer basis points, where 10,000 basis points equals 100 percent.  Quantity
tiers are contiguous and cover every positive quantity.  The tier containing
the total priced quantity sets the unit price for every unit.  Bundle
conditions are one-shot predicates.  Listing selectors count the exact
listing; product selectors aggregate units across listings for that product.
A fixed bundle discount subtracts ``discount_minor``.  A rate discount uses
``floor(cart_subtotal * discount_bps / 10_000)``.  ``best_only`` selects the
largest satisfied discount, breaking a tie by bundle id.  ``cumulative`` sums
all satisfied discounts in bundle-id order.  Either result is capped at the
cart subtotal.  Component rules are cumulative and use the undiscounted
subtotal and total priced quantity.  Rate components use integer floor
division.  These semantics make policy records unambiguous without putting a
scenario-specific quote calculator in this protocol module.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping, TypeAlias, cast

from protocol.errors import SchemaError


PRICING_POLICY_SCHEMA = "cwe.pricing-policy-revision.v1"
GENESIS_PRICING_POLICY_DIGEST = "0" * 64

SelectorKind: TypeAlias = Literal["listing", "product"]
BundleStacking: TypeAlias = Literal["best_only", "cumulative"]
ComponentKind: TypeAlias = Literal["shipping", "tax", "fee"]
PricingPolicyRetryDisposition: TypeAlias = Literal["append", "idempotent"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_SELECTOR_KINDS = frozenset({"listing", "product"})
_BUNDLE_STACKING = frozenset({"best_only", "cumulative"})
_COMPONENT_KINDS = frozenset({"shipping", "tax", "fee"})


class PricingPolicyContractError(SchemaError):
    """Base error for immutable pricing policy contracts."""


class PricingPolicySchemaError(PricingPolicyContractError):
    """A pricing policy or nested rule violates its exact schema."""


class PricingPolicyDigestMismatch(PricingPolicySchemaError):
    """A policy digest disagrees with its canonical semantic content."""


class PricingPolicyAuthorityError(PricingPolicyContractError):
    """A policy actor, owner, or logical tick is not authoritative."""


class PricingPolicyOrderingError(PricingPolicyContractError):
    """A revision number, predecessor, or logical time is out of order."""


class PricingPolicyStaleError(PricingPolicyContractError):
    """No policy revision is active at the requested World tick."""


class PricingPolicyIdentityError(PricingPolicyContractError):
    """Two revisions share neither stable identity nor actor retry scope."""


class PricingPolicyIdempotencyConflict(PricingPolicyContractError):
    """A revision identity or actor idempotency key has conflicting content."""


@dataclass(frozen=True, slots=True)
class QuantityTier:
    """A contiguous inclusive quantity interval and its per-unit price."""

    minimum_quantity: int
    maximum_quantity: int | None
    unit_price_minor: int


@dataclass(frozen=True, slots=True)
class BundleCondition:
    """A minimum quantity of one listing or product required by a bundle."""

    selector_kind: SelectorKind
    selector_id: str
    minimum_quantity: int


@dataclass(frozen=True, slots=True)
class BundleDiscount:
    """A one-shot fixed-minor or cart-subtotal-basis-point discount."""

    bundle_id: str
    conditions: tuple[BundleCondition, ...]
    discount_minor: int | None
    discount_bps: int | None


@dataclass(frozen=True, slots=True)
class PricingComponent:
    """One cumulative shipping, tax, or fee formula.

    When its inclusive lower and exclusive upper subtotal bounds match, the
    amount is ``fixed_minor + per_unit_minor * quantity +
    floor(subtotal_minor * subtotal_rate_bps / 10_000)``.
    """

    component_id: str
    kind: ComponentKind
    fixed_minor: int
    per_unit_minor: int
    subtotal_rate_bps: int
    minimum_subtotal_minor: int
    maximum_subtotal_minor: int | None


@dataclass(frozen=True, slots=True)
class PricingPolicyRevision:
    """One merchant-owned append-only pricing policy revision."""

    market_id: str
    merchant_id: str
    owner_id: str
    policy_id: str
    revision: int
    actor_id: str
    idempotency_key: str
    currency: str
    listing_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    quantity_tiers: tuple[QuantityTier, ...]
    bundle_discounts: tuple[BundleDiscount, ...]
    bundle_stacking: BundleStacking
    components: tuple[PricingComponent, ...]
    published_at_tick: int
    effective_from_tick: int
    expires_at_tick: int | None
    predecessor_digest: str
    policy_digest: str = ""
    schema_id: str = PRICING_POLICY_SCHEMA


def build_quantity_tier(
    *,
    minimum_quantity: int,
    maximum_quantity: int | None,
    unit_price_minor: int,
) -> QuantityTier:
    tier = QuantityTier(
        minimum_quantity=minimum_quantity,
        maximum_quantity=maximum_quantity,
        unit_price_minor=unit_price_minor,
    )
    validate_quantity_tier(tier)
    return tier


def validate_quantity_tier(tier: QuantityTier) -> None:
    if not isinstance(tier, QuantityTier):
        raise PricingPolicySchemaError("quantity tier must be QuantityTier")
    _require_positive_int("minimum_quantity", tier.minimum_quantity)
    if tier.maximum_quantity is not None:
        _require_positive_int("maximum_quantity", tier.maximum_quantity)
        if tier.maximum_quantity < tier.minimum_quantity:
            raise PricingPolicySchemaError("quantity tier maximum precedes minimum")
    _require_positive_int("unit_price_minor", tier.unit_price_minor)


def build_bundle_condition(
    *,
    selector_kind: SelectorKind,
    selector_id: str,
    minimum_quantity: int,
) -> BundleCondition:
    condition = BundleCondition(
        selector_kind=selector_kind,
        selector_id=selector_id,
        minimum_quantity=minimum_quantity,
    )
    validate_bundle_condition(condition)
    return condition


def validate_bundle_condition(condition: BundleCondition) -> None:
    if not isinstance(condition, BundleCondition):
        raise PricingPolicySchemaError("bundle condition must be BundleCondition")
    if condition.selector_kind not in _SELECTOR_KINDS:
        raise PricingPolicySchemaError(
            f"unsupported bundle selector_kind: {condition.selector_kind!r}"
        )
    _require_text("selector_id", condition.selector_id)
    _require_positive_int("minimum_quantity", condition.minimum_quantity)


def build_bundle_discount(
    *,
    bundle_id: str,
    conditions: Iterable[BundleCondition],
    discount_minor: int | None = None,
    discount_bps: int | None = None,
) -> BundleDiscount:
    canonical_conditions = tuple(
        sorted(tuple(conditions), key=lambda item: (item.selector_kind, item.selector_id))
    )
    discount = BundleDiscount(
        bundle_id=bundle_id,
        conditions=canonical_conditions,
        discount_minor=discount_minor,
        discount_bps=discount_bps,
    )
    validate_bundle_discount(discount)
    return discount


def validate_bundle_discount(discount: BundleDiscount) -> None:
    if not isinstance(discount, BundleDiscount):
        raise PricingPolicySchemaError("bundle discount must be BundleDiscount")
    _require_text("bundle_id", discount.bundle_id)
    if not isinstance(discount.conditions, tuple) or len(discount.conditions) < 2:
        raise PricingPolicySchemaError("bundle discount requires at least two conditions")
    for condition in discount.conditions:
        validate_bundle_condition(condition)
    expected = tuple(
        sorted(discount.conditions, key=lambda item: (item.selector_kind, item.selector_id))
    )
    if discount.conditions != expected:
        raise PricingPolicySchemaError("bundle conditions must be in canonical selector order")
    selectors = [(item.selector_kind, item.selector_id) for item in discount.conditions]
    if len(set(selectors)) != len(selectors):
        raise PricingPolicySchemaError("bundle conditions contain a duplicate selector")
    if (discount.discount_minor is None) == (discount.discount_bps is None):
        raise PricingPolicySchemaError(
            "bundle discount must define exactly one of discount_minor or discount_bps"
        )
    if discount.discount_minor is not None:
        _require_positive_int("discount_minor", discount.discount_minor)
    if discount.discount_bps is not None:
        _require_bps("discount_bps", discount.discount_bps, positive=True)


def build_pricing_component(
    *,
    component_id: str,
    kind: ComponentKind,
    fixed_minor: int = 0,
    per_unit_minor: int = 0,
    subtotal_rate_bps: int = 0,
    minimum_subtotal_minor: int = 0,
    maximum_subtotal_minor: int | None = None,
) -> PricingComponent:
    component = PricingComponent(
        component_id=component_id,
        kind=kind,
        fixed_minor=fixed_minor,
        per_unit_minor=per_unit_minor,
        subtotal_rate_bps=subtotal_rate_bps,
        minimum_subtotal_minor=minimum_subtotal_minor,
        maximum_subtotal_minor=maximum_subtotal_minor,
    )
    validate_pricing_component(component)
    return component


def validate_pricing_component(component: PricingComponent) -> None:
    if not isinstance(component, PricingComponent):
        raise PricingPolicySchemaError("pricing component must be PricingComponent")
    _require_text("component_id", component.component_id)
    if component.kind not in _COMPONENT_KINDS:
        raise PricingPolicySchemaError(f"unsupported component kind: {component.kind!r}")
    _require_nonnegative_int("fixed_minor", component.fixed_minor)
    _require_nonnegative_int("per_unit_minor", component.per_unit_minor)
    _require_bps("subtotal_rate_bps", component.subtotal_rate_bps, positive=False)
    if not (component.fixed_minor or component.per_unit_minor or component.subtotal_rate_bps):
        raise PricingPolicySchemaError("pricing component formula cannot be all zero")
    _require_nonnegative_int("minimum_subtotal_minor", component.minimum_subtotal_minor)
    if component.maximum_subtotal_minor is not None:
        _require_nonnegative_int("maximum_subtotal_minor", component.maximum_subtotal_minor)
        if component.maximum_subtotal_minor <= component.minimum_subtotal_minor:
            raise PricingPolicySchemaError(
                "component maximum subtotal must exceed minimum subtotal"
            )


def build_pricing_policy_revision(
    *,
    market_id: str,
    merchant_id: str,
    owner_id: str,
    policy_id: str,
    revision: int,
    actor_id: str,
    idempotency_key: str,
    currency: str,
    listing_ids: Iterable[str],
    product_ids: Iterable[str],
    quantity_tiers: Iterable[QuantityTier],
    bundle_discounts: Iterable[BundleDiscount] = (),
    bundle_stacking: BundleStacking = "best_only",
    components: Iterable[PricingComponent] = (),
    published_at_tick: int,
    effective_from_tick: int,
    expires_at_tick: int | None,
    predecessor_digest: str = GENESIS_PRICING_POLICY_DIGEST,
) -> PricingPolicyRevision:
    """Build a canonical policy revision and seal its content digest."""

    unsigned = PricingPolicyRevision(
        market_id=market_id,
        merchant_id=merchant_id,
        owner_id=owner_id,
        policy_id=policy_id,
        revision=revision,
        actor_id=actor_id,
        idempotency_key=idempotency_key,
        currency=currency,
        listing_ids=tuple(sorted(tuple(listing_ids))),
        product_ids=tuple(sorted(tuple(product_ids))),
        quantity_tiers=tuple(
            sorted(tuple(quantity_tiers), key=lambda tier: tier.minimum_quantity)
        ),
        bundle_discounts=tuple(
            sorted(tuple(bundle_discounts), key=lambda discount: discount.bundle_id)
        ),
        bundle_stacking=bundle_stacking,
        components=tuple(
            sorted(tuple(components), key=lambda component: (component.kind, component.component_id))
        ),
        published_at_tick=published_at_tick,
        effective_from_tick=effective_from_tick,
        expires_at_tick=expires_at_tick,
        predecessor_digest=predecessor_digest,
    )
    return seal_pricing_policy_revision(unsigned)


def seal_pricing_policy_revision(revision: PricingPolicyRevision) -> PricingPolicyRevision:
    """Seal an exact canonical unsigned revision.

    Sealing never sorts or repairs a caller-provided object.  Builders perform
    canonical sorting; direct constructors must already satisfy the contract.
    """

    if not isinstance(revision, PricingPolicyRevision):
        raise PricingPolicySchemaError("pricing policy must be PricingPolicyRevision")
    if revision.policy_digest:
        raise PricingPolicySchemaError("cannot seal a revision that already has a digest")
    _validate_policy_fields(revision, require_digest=False)
    sealed = replace(revision, policy_digest=_digest(_policy_contract(revision)))
    validate_pricing_policy_revision(sealed)
    return sealed


def validate_pricing_policy_revision(revision: PricingPolicyRevision) -> None:
    if not isinstance(revision, PricingPolicyRevision):
        raise PricingPolicySchemaError("pricing policy must be PricingPolicyRevision")
    _validate_policy_fields(revision, require_digest=True)
    if revision.policy_digest != _digest(_policy_contract(revision)):
        raise PricingPolicyDigestMismatch("pricing policy digest mismatch")


def pricing_policy_revision_to_wire(revision: PricingPolicyRevision) -> dict[str, Any]:
    validate_pricing_policy_revision(revision)
    return {**_policy_contract(revision), "policy_digest": revision.policy_digest}


def pricing_policy_revision_to_json(revision: PricingPolicyRevision) -> str:
    return _canonical_json(pricing_policy_revision_to_wire(revision))


def pricing_policy_revision_from_json(payload: str) -> PricingPolicyRevision:
    value = _strict_canonical_json(payload, "pricing policy revision")
    return coerce_pricing_policy_revision(value)


def coerce_pricing_policy_revision(value: Any) -> PricingPolicyRevision:
    row = _exact_mapping(value, _POLICY_FIELDS, "pricing policy revision")
    listing_values = _array(row, "listing_ids")
    product_values = _array(row, "product_ids")
    tier_values = _array(row, "quantity_tiers")
    bundle_values = _array(row, "bundle_discounts")
    component_values = _array(row, "components")
    revision = PricingPolicyRevision(
        schema_id=_text(row, "schema_id"),
        market_id=_text(row, "market_id"),
        merchant_id=_text(row, "merchant_id"),
        owner_id=_text(row, "owner_id"),
        policy_id=_text(row, "policy_id"),
        revision=_integer(row, "revision"),
        actor_id=_text(row, "actor_id"),
        idempotency_key=_text(row, "idempotency_key"),
        currency=_text(row, "currency"),
        listing_ids=tuple(_strict_text_value(item, "listing_ids item") for item in listing_values),
        product_ids=tuple(_strict_text_value(item, "product_ids item") for item in product_values),
        quantity_tiers=tuple(_coerce_quantity_tier(item) for item in tier_values),
        bundle_discounts=tuple(_coerce_bundle_discount(item) for item in bundle_values),
        bundle_stacking=cast(BundleStacking, _text(row, "bundle_stacking")),
        components=tuple(_coerce_pricing_component(item) for item in component_values),
        published_at_tick=_integer(row, "published_at_tick"),
        effective_from_tick=_integer(row, "effective_from_tick"),
        expires_at_tick=_optional_integer(row, "expires_at_tick"),
        predecessor_digest=_text(row, "predecessor_digest"),
        policy_digest=_text(row, "policy_digest"),
    )
    validate_pricing_policy_revision(revision)
    return revision


def classify_pricing_policy_retry(
    existing: PricingPolicyRevision,
    candidate: PricingPolicyRevision,
) -> PricingPolicyRetryDisposition:
    """Classify a stable revision identity or market-scoped actor retry key."""

    validate_pricing_policy_revision(existing)
    validate_pricing_policy_revision(candidate)
    shared_scope = (
        _revision_identity(existing) == _revision_identity(candidate)
        or (
            existing.market_id,
            existing.actor_id,
            existing.idempotency_key,
        )
        == (
            candidate.market_id,
            candidate.actor_id,
            candidate.idempotency_key,
        )
    )
    if not shared_scope:
        raise PricingPolicyIdentityError(
            "policy revisions do not share revision identity or actor idempotency scope"
        )
    if existing.policy_digest == candidate.policy_digest:
        return "idempotent"
    raise PricingPolicyIdempotencyConflict(
        "policy revision identity or actor idempotency key has conflicting content"
    )


def apply_pricing_policy_revision(
    existing: Iterable[PricingPolicyRevision],
    candidate: PricingPolicyRevision,
    *,
    trusted_owner_id: str,
    server_tick: int,
) -> tuple[PricingPolicyRetryDisposition, tuple[PricingPolicyRevision, ...]]:
    """Pure World-table transition with exact-retry and append-only semantics."""

    records = validate_pricing_policy_store(existing)
    validate_pricing_policy_revision(candidate)
    _require_text("trusted_owner_id", trusted_owner_id)
    _require_nonnegative_int("server_tick", server_tick)
    if candidate.owner_id != trusted_owner_id or candidate.actor_id != trusted_owner_id:
        raise PricingPolicyAuthorityError(
            "policy owner and author must equal the trusted merchant owner"
        )

    for record in records:
        try:
            disposition = classify_pricing_policy_retry(record, candidate)
        except PricingPolicyIdentityError:
            continue
        return disposition, records

    if candidate.published_at_tick != server_tick:
        raise PricingPolicyAuthorityError(
            "policy published_at_tick is not the trusted World tick"
        )
    stream = tuple(record for record in records if _stream_identity(record) == _stream_identity(candidate))
    if stream:
        previous = stream[-1]
        if candidate.owner_id != previous.owner_id:
            raise PricingPolicyAuthorityError("pricing policy owner cannot change within a stream")
        if candidate.currency != previous.currency:
            raise PricingPolicySchemaError("pricing policy currency cannot change within a stream")
        expected_revision = previous.revision + 1
        if candidate.revision != expected_revision:
            raise PricingPolicyOrderingError(
                f"pricing policy revision must be contiguous at {expected_revision}"
            )
        if candidate.predecessor_digest != previous.policy_digest:
            raise PricingPolicyOrderingError("pricing policy predecessor digest mismatch")
        if candidate.published_at_tick < previous.published_at_tick:
            raise PricingPolicyOrderingError("pricing policy publication tick regressed")
        if candidate.effective_from_tick < previous.effective_from_tick:
            raise PricingPolicyOrderingError("pricing policy effective tick regressed")
    else:
        if candidate.revision != 1:
            raise PricingPolicyOrderingError("first pricing policy revision must be 1")
        if candidate.predecessor_digest != GENESIS_PRICING_POLICY_DIGEST:
            raise PricingPolicyOrderingError("first policy revision must bind genesis")
    return "append", (*records, candidate)


def validate_pricing_policy_stream(
    revisions: Iterable[PricingPolicyRevision],
) -> tuple[PricingPolicyRevision, ...]:
    """Validate one append-ordered policy stream without external authority."""

    records = tuple(revisions)
    if not records:
        return records
    first = records[0]
    validate_pricing_policy_revision(first)
    expected_stream = _stream_identity(first)
    expected_owner = first.owner_id
    expected_currency = first.currency
    actor_keys: set[tuple[str, str, str]] = set()
    previous: PricingPolicyRevision | None = None
    for expected_revision, revision in enumerate(records, start=1):
        validate_pricing_policy_revision(revision)
        if _stream_identity(revision) != expected_stream:
            raise PricingPolicyIdentityError("pricing policy stream identity changed")
        if revision.owner_id != expected_owner:
            raise PricingPolicyAuthorityError("pricing policy owner changed within stream")
        if revision.currency != expected_currency:
            raise PricingPolicySchemaError("pricing policy currency changed within stream")
        if revision.revision != expected_revision:
            raise PricingPolicyOrderingError("persisted policy revisions are not contiguous")
        expected_predecessor = (
            GENESIS_PRICING_POLICY_DIGEST if previous is None else previous.policy_digest
        )
        if revision.predecessor_digest != expected_predecessor:
            raise PricingPolicyOrderingError("persisted policy predecessor chain is broken")
        if previous is not None:
            if revision.published_at_tick < previous.published_at_tick:
                raise PricingPolicyOrderingError("persisted policy publication tick regressed")
            if revision.effective_from_tick < previous.effective_from_tick:
                raise PricingPolicyOrderingError("persisted policy effective tick regressed")
        actor_key = (revision.market_id, revision.actor_id, revision.idempotency_key)
        if actor_key in actor_keys:
            raise PricingPolicyIdempotencyConflict(
                "persisted policy stream repeats an actor idempotency key"
            )
        actor_keys.add(actor_key)
        previous = revision
    return records


def validate_pricing_policy_store(
    revisions: Iterable[PricingPolicyRevision],
) -> tuple[PricingPolicyRevision, ...]:
    """Validate multiple interleaved streams and global actor retry scopes."""

    records = tuple(revisions)
    streams: dict[tuple[str, str, str], list[PricingPolicyRevision]] = {}
    identities: set[tuple[str, str, str, int]] = set()
    actor_keys: set[tuple[str, str, str]] = set()
    for revision in records:
        validate_pricing_policy_revision(revision)
        identity = _revision_identity(revision)
        if identity in identities:
            raise PricingPolicyIdempotencyConflict("duplicate persisted policy revision identity")
        actor_key = (revision.market_id, revision.actor_id, revision.idempotency_key)
        if actor_key in actor_keys:
            raise PricingPolicyIdempotencyConflict(
                "duplicate persisted market-scoped actor idempotency key"
            )
        identities.add(identity)
        actor_keys.add(actor_key)
        streams.setdefault(_stream_identity(revision), []).append(revision)
    for stream in streams.values():
        validate_pricing_policy_stream(stream)
    return records


def replay_pricing_policy_revisions(
    revisions: Iterable[PricingPolicyRevision],
    *,
    trusted_owner_by_merchant: Mapping[tuple[str, str], str],
) -> tuple[PricingPolicyRevision, ...]:
    """Replay using an external ``(market, merchant)`` owner authority map."""

    replayed: tuple[PricingPolicyRevision, ...] = ()
    for revision in revisions:
        owner_id = trusted_owner_by_merchant.get(
            (revision.market_id, revision.merchant_id)
        )
        if owner_id is None:
            raise PricingPolicyAuthorityError(
                f"no trusted owner registered for merchant {revision.merchant_id!r}"
            )
        _, replayed = apply_pricing_policy_revision(
            replayed,
            revision,
            trusted_owner_id=owner_id,
            server_tick=revision.published_at_tick,
        )
    return replayed


def resolve_active_pricing_policy(
    revisions: Iterable[PricingPolicyRevision],
    *,
    logical_tick: int,
) -> PricingPolicyRevision:
    """Resolve the highest effective revision, without reviving an expired one."""

    records = validate_pricing_policy_stream(revisions)
    _require_nonnegative_int("logical_tick", logical_tick)
    effective = [record for record in records if record.effective_from_tick <= logical_tick]
    if not effective:
        raise PricingPolicyStaleError("pricing policy is not yet effective")
    revision = effective[-1]
    if revision.expires_at_tick is not None and logical_tick > revision.expires_at_tick:
        raise PricingPolicyStaleError("latest effective pricing policy revision has expired")
    return revision


def _validate_policy_fields(
    revision: PricingPolicyRevision,
    *,
    require_digest: bool,
) -> None:
    if revision.schema_id != PRICING_POLICY_SCHEMA:
        raise PricingPolicySchemaError(
            f"unsupported pricing policy schema: {revision.schema_id!r}"
        )
    for name in (
        "market_id",
        "merchant_id",
        "owner_id",
        "policy_id",
        "actor_id",
        "idempotency_key",
    ):
        _require_text(name, getattr(revision, name))
    if revision.actor_id != revision.owner_id:
        raise PricingPolicyAuthorityError("policy author must equal its bound owner")
    _require_positive_int("revision", revision.revision)
    if not _CURRENCY_RE.fullmatch(revision.currency):
        raise PricingPolicySchemaError("currency must be an uppercase ISO-style code")
    _validate_canonical_text_tuple("listing_ids", revision.listing_ids)
    _validate_canonical_text_tuple("product_ids", revision.product_ids)
    if not revision.listing_ids and not revision.product_ids:
        raise PricingPolicySchemaError("policy scope requires a listing_id or product_id")
    _validate_quantity_tiers(revision.quantity_tiers)
    if not isinstance(revision.bundle_discounts, tuple):
        raise PricingPolicySchemaError("bundle_discounts must be a canonical tuple")
    for discount in revision.bundle_discounts:
        validate_bundle_discount(discount)
    if revision.bundle_discounts != tuple(
        sorted(revision.bundle_discounts, key=lambda discount: discount.bundle_id)
    ):
        raise PricingPolicySchemaError("bundle discounts must be sorted by bundle_id")
    if len({discount.bundle_id for discount in revision.bundle_discounts}) != len(
        revision.bundle_discounts
    ):
        raise PricingPolicySchemaError("bundle discounts contain duplicate bundle_id")
    if revision.bundle_stacking not in _BUNDLE_STACKING:
        raise PricingPolicySchemaError(
            f"unsupported bundle_stacking: {revision.bundle_stacking!r}"
        )
    if not isinstance(revision.components, tuple):
        raise PricingPolicySchemaError("components must be a canonical tuple")
    for component in revision.components:
        validate_pricing_component(component)
    if revision.components != tuple(
        sorted(revision.components, key=lambda component: (component.kind, component.component_id))
    ):
        raise PricingPolicySchemaError("components must be in canonical kind and id order")
    if len({component.component_id for component in revision.components}) != len(
        revision.components
    ):
        raise PricingPolicySchemaError("pricing components contain duplicate component_id")
    _require_nonnegative_int("published_at_tick", revision.published_at_tick)
    _require_nonnegative_int("effective_from_tick", revision.effective_from_tick)
    if revision.effective_from_tick < revision.published_at_tick:
        raise PricingPolicySchemaError("policy cannot become effective before publication")
    if revision.expires_at_tick is not None:
        _require_nonnegative_int("expires_at_tick", revision.expires_at_tick)
        if revision.expires_at_tick <= revision.effective_from_tick:
            raise PricingPolicySchemaError("policy expiry must follow its effective tick")
    _require_digest("predecessor_digest", revision.predecessor_digest)
    if revision.revision == 1:
        if revision.predecessor_digest != GENESIS_PRICING_POLICY_DIGEST:
            raise PricingPolicySchemaError("first policy revision must bind genesis")
    elif revision.predecessor_digest == GENESIS_PRICING_POLICY_DIGEST:
        raise PricingPolicySchemaError("non-first policy revision cannot bind genesis")
    if require_digest:
        _require_digest("policy_digest", revision.policy_digest)


def _validate_quantity_tiers(tiers: tuple[QuantityTier, ...]) -> None:
    if not isinstance(tiers, tuple) or not tiers:
        raise PricingPolicySchemaError("quantity_tiers must be a non-empty canonical tuple")
    for tier in tiers:
        validate_quantity_tier(tier)
    if tiers != tuple(sorted(tiers, key=lambda tier: tier.minimum_quantity)):
        raise PricingPolicySchemaError("quantity tiers must be sorted by minimum_quantity")
    if tiers[0].minimum_quantity != 1:
        raise PricingPolicySchemaError("quantity tiers must begin at quantity 1")
    for previous, current in zip(tiers, tiers[1:]):
        if previous.maximum_quantity is None:
            raise PricingPolicySchemaError("only the last quantity tier may be open ended")
        if current.minimum_quantity != previous.maximum_quantity + 1:
            raise PricingPolicySchemaError("quantity tiers must be contiguous and non-overlapping")
    if tiers[-1].maximum_quantity is not None:
        raise PricingPolicySchemaError("last quantity tier must be open ended")


def _policy_contract(revision: PricingPolicyRevision) -> dict[str, Any]:
    return {
        "schema_id": revision.schema_id,
        "market_id": revision.market_id,
        "merchant_id": revision.merchant_id,
        "owner_id": revision.owner_id,
        "policy_id": revision.policy_id,
        "revision": revision.revision,
        "actor_id": revision.actor_id,
        "idempotency_key": revision.idempotency_key,
        "currency": revision.currency,
        "listing_ids": list(revision.listing_ids),
        "product_ids": list(revision.product_ids),
        "quantity_tiers": [_quantity_tier_contract(tier) for tier in revision.quantity_tiers],
        "bundle_discounts": [
            _bundle_discount_contract(discount) for discount in revision.bundle_discounts
        ],
        "bundle_stacking": revision.bundle_stacking,
        "components": [_pricing_component_contract(component) for component in revision.components],
        "published_at_tick": revision.published_at_tick,
        "effective_from_tick": revision.effective_from_tick,
        "expires_at_tick": revision.expires_at_tick,
        "predecessor_digest": revision.predecessor_digest,
    }


def _quantity_tier_contract(tier: QuantityTier) -> dict[str, Any]:
    return {
        "minimum_quantity": tier.minimum_quantity,
        "maximum_quantity": tier.maximum_quantity,
        "unit_price_minor": tier.unit_price_minor,
    }


def _bundle_condition_contract(condition: BundleCondition) -> dict[str, Any]:
    return {
        "selector_kind": condition.selector_kind,
        "selector_id": condition.selector_id,
        "minimum_quantity": condition.minimum_quantity,
    }


def _bundle_discount_contract(discount: BundleDiscount) -> dict[str, Any]:
    return {
        "bundle_id": discount.bundle_id,
        "conditions": [_bundle_condition_contract(item) for item in discount.conditions],
        "discount_minor": discount.discount_minor,
        "discount_bps": discount.discount_bps,
    }


def _pricing_component_contract(component: PricingComponent) -> dict[str, Any]:
    return {
        "component_id": component.component_id,
        "kind": component.kind,
        "fixed_minor": component.fixed_minor,
        "per_unit_minor": component.per_unit_minor,
        "subtotal_rate_bps": component.subtotal_rate_bps,
        "minimum_subtotal_minor": component.minimum_subtotal_minor,
        "maximum_subtotal_minor": component.maximum_subtotal_minor,
    }


_QUANTITY_TIER_FIELDS = frozenset(
    {"minimum_quantity", "maximum_quantity", "unit_price_minor"}
)
_BUNDLE_CONDITION_FIELDS = frozenset(
    {"selector_kind", "selector_id", "minimum_quantity"}
)
_BUNDLE_DISCOUNT_FIELDS = frozenset(
    {"bundle_id", "conditions", "discount_minor", "discount_bps"}
)
_PRICING_COMPONENT_FIELDS = frozenset(
    {
        "component_id",
        "kind",
        "fixed_minor",
        "per_unit_minor",
        "subtotal_rate_bps",
        "minimum_subtotal_minor",
        "maximum_subtotal_minor",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema_id",
        "market_id",
        "merchant_id",
        "owner_id",
        "policy_id",
        "revision",
        "actor_id",
        "idempotency_key",
        "currency",
        "listing_ids",
        "product_ids",
        "quantity_tiers",
        "bundle_discounts",
        "bundle_stacking",
        "components",
        "published_at_tick",
        "effective_from_tick",
        "expires_at_tick",
        "predecessor_digest",
        "policy_digest",
    }
)


def _coerce_quantity_tier(value: Any) -> QuantityTier:
    row = _exact_mapping(value, _QUANTITY_TIER_FIELDS, "quantity tier")
    tier = QuantityTier(
        minimum_quantity=_integer(row, "minimum_quantity"),
        maximum_quantity=_optional_integer(row, "maximum_quantity"),
        unit_price_minor=_integer(row, "unit_price_minor"),
    )
    validate_quantity_tier(tier)
    return tier


def _coerce_bundle_condition(value: Any) -> BundleCondition:
    row = _exact_mapping(value, _BUNDLE_CONDITION_FIELDS, "bundle condition")
    condition = BundleCondition(
        selector_kind=cast(SelectorKind, _text(row, "selector_kind")),
        selector_id=_text(row, "selector_id"),
        minimum_quantity=_integer(row, "minimum_quantity"),
    )
    validate_bundle_condition(condition)
    return condition


def _coerce_bundle_discount(value: Any) -> BundleDiscount:
    row = _exact_mapping(value, _BUNDLE_DISCOUNT_FIELDS, "bundle discount")
    conditions = _array(row, "conditions")
    discount = BundleDiscount(
        bundle_id=_text(row, "bundle_id"),
        conditions=tuple(_coerce_bundle_condition(item) for item in conditions),
        discount_minor=_optional_integer(row, "discount_minor"),
        discount_bps=_optional_integer(row, "discount_bps"),
    )
    validate_bundle_discount(discount)
    return discount


def _coerce_pricing_component(value: Any) -> PricingComponent:
    row = _exact_mapping(value, _PRICING_COMPONENT_FIELDS, "pricing component")
    component = PricingComponent(
        component_id=_text(row, "component_id"),
        kind=cast(ComponentKind, _text(row, "kind")),
        fixed_minor=_integer(row, "fixed_minor"),
        per_unit_minor=_integer(row, "per_unit_minor"),
        subtotal_rate_bps=_integer(row, "subtotal_rate_bps"),
        minimum_subtotal_minor=_integer(row, "minimum_subtotal_minor"),
        maximum_subtotal_minor=_optional_integer(row, "maximum_subtotal_minor"),
    )
    validate_pricing_component(component)
    return component


def _stream_identity(revision: PricingPolicyRevision) -> tuple[str, str, str]:
    return (revision.market_id, revision.merchant_id, revision.policy_id)


def _revision_identity(revision: PricingPolicyRevision) -> tuple[str, str, str, int]:
    return (*_stream_identity(revision), revision.revision)


def _validate_canonical_text_tuple(label: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise PricingPolicySchemaError(f"{label} must be a canonical tuple")
    for value in values:
        _require_text(f"{label} item", value)
    if values != tuple(sorted(values)):
        raise PricingPolicySchemaError(f"{label} must be sorted")
    if len(set(values)) != len(values):
        raise PricingPolicySchemaError(f"{label} contains duplicates")


def _strict_canonical_json(payload: str, label: str) -> Any:
    if not isinstance(payload, str):
        raise PricingPolicySchemaError(f"{label} wire payload must be text")

    def reject_constant(value: str) -> None:
        raise PricingPolicySchemaError(f"{label} contains non-finite number {value}")

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PricingPolicySchemaError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except PricingPolicySchemaError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PricingPolicySchemaError(f"invalid {label} JSON: {exc}") from exc
    if payload != _canonical_json(value):
        raise PricingPolicySchemaError(f"{label} wire JSON is not canonical")
    return value


def _exact_mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PricingPolicySchemaError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise PricingPolicySchemaError(
            f"{label} fields mismatch; missing={missing!r}, extra={extra!r}"
        )
    return value


def _array(row: Mapping[str, Any], name: str) -> list[Any]:
    value = row[name]
    if not isinstance(value, list):
        raise PricingPolicySchemaError(f"{name} must be an array")
    return value


def _text(row: Mapping[str, Any], name: str) -> str:
    return _strict_text_value(row[name], name)


def _strict_text_value(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise PricingPolicySchemaError(f"{name} must be text")
    return value


def _integer(row: Mapping[str, Any], name: str) -> int:
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PricingPolicySchemaError(f"{name} must be an integer")
    return cast(int, value)


def _optional_integer(row: Mapping[str, Any], name: str) -> int | None:
    value = row[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PricingPolicySchemaError(f"{name} must be an integer or null")
    return cast(int, value)


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PricingPolicySchemaError(f"{name} must be non-empty text")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PricingPolicySchemaError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PricingPolicySchemaError(f"{name} must be a non-negative integer")


def _require_bps(name: str, value: Any, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PricingPolicySchemaError(f"{name} must be integer basis points")
    minimum = 1 if positive else 0
    if not minimum <= value <= 10_000:
        qualifier = "positive " if positive else ""
        raise PricingPolicySchemaError(
            f"{name} must be {qualifier}integer basis points no greater than 10000"
        )


def _require_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PricingPolicySchemaError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PricingPolicySchemaError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "GENESIS_PRICING_POLICY_DIGEST",
    "PRICING_POLICY_SCHEMA",
    "BundleCondition",
    "BundleDiscount",
    "PricingComponent",
    "PricingPolicyAuthorityError",
    "PricingPolicyContractError",
    "PricingPolicyDigestMismatch",
    "PricingPolicyIdempotencyConflict",
    "PricingPolicyIdentityError",
    "PricingPolicyOrderingError",
    "PricingPolicyRevision",
    "PricingPolicySchemaError",
    "PricingPolicyStaleError",
    "QuantityTier",
    "apply_pricing_policy_revision",
    "build_bundle_condition",
    "build_bundle_discount",
    "build_pricing_component",
    "build_pricing_policy_revision",
    "build_quantity_tier",
    "classify_pricing_policy_retry",
    "coerce_pricing_policy_revision",
    "pricing_policy_revision_from_json",
    "pricing_policy_revision_to_json",
    "pricing_policy_revision_to_wire",
    "replay_pricing_policy_revisions",
    "resolve_active_pricing_policy",
    "seal_pricing_policy_revision",
    "validate_bundle_condition",
    "validate_bundle_discount",
    "validate_pricing_component",
    "validate_pricing_policy_revision",
    "validate_pricing_policy_store",
    "validate_pricing_policy_stream",
    "validate_quantity_tier",
]
