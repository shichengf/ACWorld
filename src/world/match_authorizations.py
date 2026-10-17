"""Core-backed authorization fixtures for existing CommerceWorld orders.

The scenario surface declares only an authorization identity, an existing
order, a mandate identity, and a lifetime.  Every commercial binding is read
from authoritative World state.  The resulting search session, offer,
acceptance, and certificate are built with the normal matching contracts and
persisted through the normal World methods.  No scenario may seed a detached
certificate row or supply a digest, revision, party, quantity, or price.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping

from protocol.matching import (
    MatchAcceptance,
    MatchCertificate,
    MatchValidationError,
    OfferSnapshot,
    SearchSession,
    canonical_digest,
    issue_match_certificate as build_match_certificate,
    seal_match_acceptance,
    seal_search_session,
)
from world.types import InventoryRow, Listing, Order


MATCH_AUTHORIZATION_FIXTURE_SCHEMA = "cwe.match-authorization-fixture.v1"


@dataclass(frozen=True, slots=True)
class MatchAuthorizationFixture:
    """Compact declaration for one core-derived matching authorization."""

    authorization_id: str
    order_id: str
    mandate_id: str
    ttl_ticks: int


@dataclass(frozen=True, slots=True)
class MatchAuthorizationChain:
    """The exact authority chain persisted for one fixture."""

    fixture: MatchAuthorizationFixture
    session: SearchSession
    acceptance: MatchAcceptance
    certificate: MatchCertificate


def coerce_match_authorization_fixture(value: Any) -> MatchAuthorizationFixture:
    """Parse the strict public fixture contract."""

    if isinstance(value, MatchAuthorizationFixture):
        fixture = value
    else:
        if not isinstance(value, Mapping):
            raise MatchValidationError("match authorization fixture must be a mapping")
        allowed = {
            "schema_version",
            "authorization_id",
            "order_id",
            "mandate_id",
            "ttl_ticks",
        }
        if set(value) - allowed or {
            "authorization_id",
            "order_id",
            "mandate_id",
            "ttl_ticks",
        } - set(value):
            raise MatchValidationError(
                "match authorization fixture has invalid fields"
            )
        schema = value.get("schema_version", MATCH_AUTHORIZATION_FIXTURE_SCHEMA)
        if schema != MATCH_AUTHORIZATION_FIXTURE_SCHEMA:
            raise MatchValidationError(
                f"unsupported match authorization fixture schema {schema!r}"
            )
        ttl = value["ttl_ticks"]
        if isinstance(ttl, bool) or not isinstance(ttl, int):
            raise MatchValidationError(
                "match authorization fixture ttl_ticks must be an integer"
            )
        text_fields: dict[str, str] = {}
        for name in ("authorization_id", "order_id", "mandate_id"):
            field = value[name]
            if not isinstance(field, str):
                raise MatchValidationError(
                    f"match authorization fixture {name} must be a string"
                )
            text_fields[name] = field
        fixture = MatchAuthorizationFixture(
            authorization_id=text_fields["authorization_id"],
            order_id=text_fields["order_id"],
            mandate_id=text_fields["mandate_id"],
            ttl_ticks=ttl,
        )
    for name in ("authorization_id", "order_id", "mandate_id"):
        if not getattr(fixture, name).strip():
            raise MatchValidationError(
                f"match authorization fixture {name} must be non-empty"
            )
    if fixture.ttl_ticks <= 0:
        raise MatchValidationError(
            "match authorization fixture ttl_ticks must be positive"
        )
    return fixture


def match_authorization_acceptance_key(authorization_id: str) -> str:
    """Return the buyer-scoped acceptance idempotency key for a fixture id."""

    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise MatchValidationError("authorization_id must be non-empty")
    digest = canonical_digest(
        {
            "schema_version": MATCH_AUTHORIZATION_FIXTURE_SCHEMA,
            "authorization_id": authorization_id,
            "stage": "acceptance",
        }
    )
    return f"match-authorization:accept:{digest[:32]}"


def issue_order_match_authorization(
    world: Any,
    fixture: MatchAuthorizationFixture | Mapping[str, Any],
) -> MatchAuthorizationChain:
    """Derive and persist a complete matching chain through core World APIs."""

    parsed = coerce_match_authorization_fixture(fixture)
    chain = _derive_chain(world, parsed)
    persisted_session = world.create_search_session(
        session=chain.session,
        by_actor="platform:aggregator",
        idempotency_key=chain.session.search_idempotency_key,
    )
    if persisted_session != chain.session:
        raise MatchValidationError(
            "persisted match authorization search session differs from derivation"
        )
    persisted_certificate = world.issue_match_certificate(
        acceptance=chain.acceptance,
        by_actor="platform:aggregator",
        original_actor=chain.acceptance.buyer_id,
    )
    if persisted_certificate != chain.certificate:
        raise MatchValidationError(
            "persisted match authorization certificate differs from derivation"
        )
    return chain


def _derive_chain(
    world: Any,
    fixture: MatchAuthorizationFixture,
) -> MatchAuthorizationChain:
    order = world.read("orders", fixture.order_id, caller="platform:aggregator")
    if not isinstance(order, Order):
        raise MatchValidationError(
            f"match authorization order {fixture.order_id!r} does not exist"
        )
    listing = world.read("catalog", order.sku_id, caller="platform:aggregator")
    inventory = world.read("inventory", order.sku_id, caller="platform:aggregator")
    if not isinstance(listing, Listing) or not isinstance(inventory, InventoryRow):
        raise MatchValidationError(
            "match authorization order lacks authoritative catalog or inventory state"
        )
    if (
        str(listing.sku_id) != str(order.sku_id)
        or str(listing.merchant_id) != str(order.merchant_id)
        or str(inventory.sku_id) != str(order.sku_id)
        or str(inventory.merchant_id) != str(order.merchant_id)
    ):
        raise MatchValidationError(
            "match authorization order, listing, and inventory bindings disagree"
        )
    if listing.list_price != order.agreed_price:
        raise MatchValidationError(
            "match authorization listing price differs from authoritative order price"
        )
    current_tick = int(world.logical_time)
    expires_at_tick = current_tick + fixture.ttl_ticks
    stem = canonical_digest(
        {
            "schema_version": MATCH_AUTHORIZATION_FIXTURE_SCHEMA,
            "authorization_id": fixture.authorization_id,
            "order_id": str(order.order_id),
            "buyer_id": str(order.buyer_id),
        }
    )
    search_key = f"match-authorization:search:{stem[:32]}"
    session_id = f"match-authorization:session:{stem[:32]}"
    offer_id = f"match-authorization:offer:{stem[:32]}"
    acceptance_key = match_authorization_acceptance_key(fixture.authorization_id)
    catalog_revision = _catalog_revision(listing)
    inventory_revision = inventory.version
    unit_price_cents = _money_cents(listing.list_price.amount)
    session = seal_search_session(
        SearchSession(
            session_id=session_id,
            buyer_id=str(order.buyer_id),
            mandate_id=fixture.mandate_id,
            search_request_id=search_key,
            search_idempotency_key=search_key,
            query_digest=canonical_digest(
                {
                    "authorization_id": fixture.authorization_id,
                    "order_id": str(order.order_id),
                    "source": "authoritative-existing-order",
                }
            ),
            issued_at_tick=current_tick,
            expires_at_tick=expires_at_tick,
            offers=(
                OfferSnapshot(
                    offer_id=offer_id,
                    session_id=session_id,
                    buyer_id=str(order.buyer_id),
                    mandate_id=fixture.mandate_id,
                    merchant_id=str(order.merchant_id),
                    sku_id=str(order.sku_id),
                    unit_price_cents=unit_price_cents,
                    currency=order.agreed_price.currency,
                    qty=order.qty,
                    catalog_revision=catalog_revision,
                    inventory_revision=inventory_revision,
                    issued_at_tick=current_tick,
                    expires_at_tick=expires_at_tick,
                ),
            ),
        )
    )
    offer = session.offers[0]
    acceptance = seal_match_acceptance(
        MatchAcceptance(
            request_msg_id=f"match-authorization:request:{stem[:32]}",
            idempotency_key=acceptance_key,
            session_id=session.session_id,
            session_digest=session.session_digest,
            offer_id=offer.offer_id,
            offer_digest=offer.offer_digest,
            buyer_id=session.buyer_id,
            mandate_id=session.mandate_id,
            order_id=str(order.order_id),
            merchant_id=offer.merchant_id,
            sku_id=offer.sku_id,
            unit_price_cents=offer.unit_price_cents,
            currency=offer.currency,
            qty=offer.qty,
            catalog_revision=offer.catalog_revision,
            inventory_revision=offer.inventory_revision,
        )
    )
    certificate = build_match_certificate(
        session,
        acceptance,
        current_tick=current_tick,
        current_catalog_revision=catalog_revision,
        current_inventory_revision=inventory_revision,
    )
    return MatchAuthorizationChain(
        fixture=fixture,
        session=session,
        acceptance=acceptance,
        certificate=certificate,
    )


def _catalog_revision(listing: Listing) -> int:
    value = (listing.attributes or {}).get("catalog_revision", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MatchValidationError("catalog revision must be a positive integer")
    return value


def _money_cents(amount: Decimal) -> int:
    return int((amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


__all__ = [
    "MATCH_AUTHORIZATION_FIXTURE_SCHEMA",
    "MatchAuthorizationChain",
    "MatchAuthorizationFixture",
    "coerce_match_authorization_fixture",
    "issue_order_match_authorization",
    "match_authorization_acceptance_key",
]
