"""Replay-stable contracts for discovery sessions and match certification.

The objects in this module are passive wire contracts.  They deliberately do
not keep an in-memory registry and do not read World state.  A Platform store
can persist their canonical wire forms, reconstruct them after a restart, and
use the pure validators here before issuing or consuming a certificate.

Only mechanical facts are certified: actor and mandate identity, membership
in one search session, the exact listing/merchant/price/quantity snapshot,
catalog and inventory freshness, expiry, and the target order.  This module
does not claim that budget, delivery, or product claims were checked.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from protocol.errors import IdempotencyConflict, SchemaError


OFFER_SNAPSHOT_SCHEMA = "cwe.offer-snapshot.v1"
SEARCH_SESSION_SCHEMA = "cwe.search-session.v1"
MATCH_ACCEPTANCE_SCHEMA = "cwe.match-acceptance.v1"
MATCH_CERTIFICATE_SCHEMA = "cwe.match-certificate.v1"

# These labels describe only bindings that this module actually verifies.
# In particular, there is no blanket ``checks_passed`` claim for budget,
# delivery, product claims, or any other policy owned by another subsystem.
MATCH_BINDING_VALIDATIONS = (
    "actor_binding",
    "mandate_binding",
    "session_membership",
    "offer_digest",
    "merchant_binding",
    "sku_binding",
    "price_binding",
    "quantity_binding",
    "catalog_freshness",
    "inventory_freshness",
    "expiry",
    "order_binding",
    "idempotency_binding",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_OFFER_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "offer_id",
        "session_id",
        "buyer_id",
        "mandate_id",
        "merchant_id",
        "sku_id",
        "unit_price_cents",
        "currency",
        "qty",
        "catalog_revision",
        "inventory_revision",
        "issued_at_tick",
        "expires_at_tick",
        "offer_digest",
    }
)
_SEARCH_SESSION_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "session_id",
        "buyer_id",
        "mandate_id",
        "search_request_id",
        "search_idempotency_key",
        "query_digest",
        "issued_at_tick",
        "expires_at_tick",
        "offers",
        "session_digest",
    }
)
_MATCH_ACCEPTANCE_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "request_msg_id",
        "idempotency_key",
        "session_id",
        "session_digest",
        "offer_id",
        "offer_digest",
        "buyer_id",
        "mandate_id",
        "order_id",
        "merchant_id",
        "sku_id",
        "unit_price_cents",
        "currency",
        "qty",
        "catalog_revision",
        "inventory_revision",
        "acceptance_digest",
    }
)
_MATCH_CERTIFICATE_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "cert_id",
        "session_id",
        "session_digest",
        "search_request_id",
        "offer_id",
        "offer_digest",
        "acceptance_request_id",
        "acceptance_digest",
        "idempotency_key",
        "buyer_id",
        "mandate_id",
        "order_id",
        "merchant_id",
        "sku_id",
        "unit_price_cents",
        "currency",
        "qty",
        "catalog_revision",
        "inventory_revision",
        "issued_at_tick",
        "expires_at_tick",
        "validations",
        "certificate_digest",
    }
)


class MatchValidationError(SchemaError):
    """A well-shaped match object violated a binding or freshness rule."""


class MatchAcceptanceRejected(MatchValidationError):
    """An actor acceptance conflicts with authoritative matching state.

    This narrow subtype is reserved for a buyer-triggered acceptance that is
    unknown, stale, no longer a member of its persisted search session, or
    conflicts with an earlier acceptance.  Generic ``MatchValidationError``
    remains an integrity error for malformed persisted matching state.
    """


@dataclass(frozen=True, slots=True)
class OfferSnapshot:
    """One immutable Platform-authored offer inside a search session."""

    offer_id: str
    session_id: str
    buyer_id: str
    mandate_id: str
    merchant_id: str
    sku_id: str
    unit_price_cents: int
    currency: str
    qty: int
    catalog_revision: int
    inventory_revision: int
    issued_at_tick: int
    expires_at_tick: int
    offer_digest: str = ""


@dataclass(frozen=True, slots=True)
class SearchSession:
    """A ranked result set bound to one buyer, mandate, and search request."""

    session_id: str
    buyer_id: str
    mandate_id: str
    search_request_id: str
    search_idempotency_key: str
    query_digest: str
    issued_at_tick: int
    expires_at_tick: int
    offers: tuple[OfferSnapshot, ...]
    session_digest: str = ""


@dataclass(frozen=True, slots=True)
class MatchAcceptance:
    """Formal buyer acceptance of one exact, persisted offer snapshot.

    The duplicated offer fields are intentional.  They make an altered price,
    quantity, merchant, SKU, or revision visible at the Platform boundary
    instead of letting downstream order construction silently drift from the
    offer the buyer saw.
    """

    request_msg_id: str
    idempotency_key: str
    session_id: str
    session_digest: str
    offer_id: str
    offer_digest: str
    buyer_id: str
    mandate_id: str
    order_id: str
    merchant_id: str
    sku_id: str
    unit_price_cents: int
    currency: str
    qty: int
    catalog_revision: int
    inventory_revision: int
    acceptance_digest: str = ""


@dataclass(frozen=True, slots=True)
class MatchCertificate:
    """Platform certificate for one validated acceptance and target order."""

    cert_id: str
    session_id: str
    session_digest: str
    search_request_id: str
    offer_id: str
    offer_digest: str
    acceptance_request_id: str
    acceptance_digest: str
    idempotency_key: str
    buyer_id: str
    mandate_id: str
    order_id: str
    merchant_id: str
    sku_id: str
    unit_price_cents: int
    currency: str
    qty: int
    catalog_revision: int
    inventory_revision: int
    issued_at_tick: int
    expires_at_tick: int
    validations: tuple[str, ...]
    certificate_digest: str = ""


def canonical_digest(value: Any) -> str:
    """Return a SHA-256 digest over strict, compact, sorted canonical JSON."""

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


def offer_snapshot_digest(offer: OfferSnapshot) -> str:
    return canonical_digest(_offer_contract(offer))


def seal_offer_snapshot(offer: OfferSnapshot) -> OfferSnapshot:
    _validate_offer_fields(offer)
    sealed = replace(offer, offer_digest=offer_snapshot_digest(offer))
    validate_offer_snapshot(sealed)
    return sealed


def validate_offer_snapshot(offer: OfferSnapshot) -> None:
    _validate_offer_fields(offer)
    _require_digest(offer.offer_digest, "offer_digest")
    if offer.offer_digest != offer_snapshot_digest(offer):
        raise MatchValidationError("offer digest mismatch")


def offer_snapshot_to_wire(offer: OfferSnapshot) -> dict[str, Any]:
    validate_offer_snapshot(offer)
    return {**_offer_contract(offer), "offer_digest": offer.offer_digest}


def coerce_offer_snapshot(value: Any) -> OfferSnapshot:
    if isinstance(value, OfferSnapshot):
        validate_offer_snapshot(value)
        return value
    row = _mapping(value, "offer snapshot")
    _require_exact_keys(row, _OFFER_WIRE_KEYS, "offer snapshot")
    _require_schema(row, OFFER_SNAPSHOT_SCHEMA)
    try:
        offer = OfferSnapshot(
            offer_id=_wire_text(row, "offer_id"),
            session_id=_wire_text(row, "session_id"),
            buyer_id=_wire_text(row, "buyer_id"),
            mandate_id=_wire_text(row, "mandate_id"),
            merchant_id=_wire_text(row, "merchant_id"),
            sku_id=_wire_text(row, "sku_id"),
            unit_price_cents=_wire_int(row, "unit_price_cents"),
            currency=_wire_text(row, "currency"),
            qty=_wire_int(row, "qty"),
            catalog_revision=_wire_int(row, "catalog_revision"),
            inventory_revision=_wire_int(row, "inventory_revision"),
            issued_at_tick=_wire_int(row, "issued_at_tick"),
            expires_at_tick=_wire_int(row, "expires_at_tick"),
            offer_digest=_wire_text(row, "offer_digest"),
        )
    except KeyError as exc:
        raise SchemaError(f"offer snapshot missing field {exc.args[0]!r}") from exc
    validate_offer_snapshot(offer)
    return offer


def search_session_digest(session: SearchSession) -> str:
    return canonical_digest(_search_session_contract(session))


def seal_search_session(session: SearchSession) -> SearchSession:
    sealed_offers = tuple(seal_offer_snapshot(offer) for offer in session.offers)
    unsigned = replace(session, offers=sealed_offers, session_digest="")
    _validate_search_session_fields(unsigned)
    sealed = replace(unsigned, session_digest=search_session_digest(unsigned))
    validate_search_session(sealed)
    return sealed


def validate_search_session(session: SearchSession) -> None:
    _validate_search_session_fields(session)
    _require_digest(session.session_digest, "session_digest")
    if session.session_digest != search_session_digest(session):
        raise MatchValidationError("search session digest mismatch")


def search_session_to_wire(session: SearchSession) -> dict[str, Any]:
    validate_search_session(session)
    return {
        "schema_version": SEARCH_SESSION_SCHEMA,
        "session_id": session.session_id,
        "buyer_id": session.buyer_id,
        "mandate_id": session.mandate_id,
        "search_request_id": session.search_request_id,
        "search_idempotency_key": session.search_idempotency_key,
        "query_digest": session.query_digest,
        "issued_at_tick": session.issued_at_tick,
        "expires_at_tick": session.expires_at_tick,
        "offers": [offer_snapshot_to_wire(offer) for offer in session.offers],
        "session_digest": session.session_digest,
    }


def coerce_search_session(value: Any) -> SearchSession:
    if isinstance(value, SearchSession):
        validate_search_session(value)
        return value
    row = _mapping(value, "search session")
    _require_exact_keys(row, _SEARCH_SESSION_WIRE_KEYS, "search session")
    _require_schema(row, SEARCH_SESSION_SCHEMA)
    try:
        raw_offers = row["offers"]
        if not isinstance(raw_offers, (list, tuple)):
            raise SchemaError("search session offers must be an array")
        session = SearchSession(
            session_id=_wire_text(row, "session_id"),
            buyer_id=_wire_text(row, "buyer_id"),
            mandate_id=_wire_text(row, "mandate_id"),
            search_request_id=_wire_text(row, "search_request_id"),
            search_idempotency_key=_wire_text(row, "search_idempotency_key"),
            query_digest=_wire_text(row, "query_digest"),
            issued_at_tick=_wire_int(row, "issued_at_tick"),
            expires_at_tick=_wire_int(row, "expires_at_tick"),
            offers=tuple(coerce_offer_snapshot(item) for item in raw_offers),
            session_digest=_wire_text(row, "session_digest"),
        )
    except KeyError as exc:
        raise SchemaError(f"search session missing field {exc.args[0]!r}") from exc
    validate_search_session(session)
    return session


def match_acceptance_digest(acceptance: MatchAcceptance) -> str:
    return canonical_digest(_acceptance_contract(acceptance))


def seal_match_acceptance(acceptance: MatchAcceptance) -> MatchAcceptance:
    _validate_acceptance_fields(acceptance)
    sealed = replace(
        acceptance,
        acceptance_digest=match_acceptance_digest(acceptance),
    )
    validate_match_acceptance(sealed)
    return sealed


def validate_match_acceptance(acceptance: MatchAcceptance) -> None:
    _validate_acceptance_fields(acceptance)
    _require_digest(acceptance.acceptance_digest, "acceptance_digest")
    if acceptance.acceptance_digest != match_acceptance_digest(acceptance):
        raise MatchValidationError("match acceptance digest mismatch")


def match_acceptance_to_wire(acceptance: MatchAcceptance) -> dict[str, Any]:
    validate_match_acceptance(acceptance)
    return {
        **_acceptance_contract(acceptance),
        "acceptance_digest": acceptance.acceptance_digest,
    }


def coerce_match_acceptance(value: Any) -> MatchAcceptance:
    if isinstance(value, MatchAcceptance):
        validate_match_acceptance(value)
        return value
    row = _mapping(value, "match acceptance")
    _require_exact_keys(row, _MATCH_ACCEPTANCE_WIRE_KEYS, "match acceptance")
    _require_schema(row, MATCH_ACCEPTANCE_SCHEMA)
    try:
        acceptance = MatchAcceptance(
            request_msg_id=_wire_text(row, "request_msg_id"),
            idempotency_key=_wire_text(row, "idempotency_key"),
            session_id=_wire_text(row, "session_id"),
            session_digest=_wire_text(row, "session_digest"),
            offer_id=_wire_text(row, "offer_id"),
            offer_digest=_wire_text(row, "offer_digest"),
            buyer_id=_wire_text(row, "buyer_id"),
            mandate_id=_wire_text(row, "mandate_id"),
            order_id=_wire_text(row, "order_id"),
            merchant_id=_wire_text(row, "merchant_id"),
            sku_id=_wire_text(row, "sku_id"),
            unit_price_cents=_wire_int(row, "unit_price_cents"),
            currency=_wire_text(row, "currency"),
            qty=_wire_int(row, "qty"),
            catalog_revision=_wire_int(row, "catalog_revision"),
            inventory_revision=_wire_int(row, "inventory_revision"),
            acceptance_digest=_wire_text(row, "acceptance_digest"),
        )
    except KeyError as exc:
        raise SchemaError(f"match acceptance missing field {exc.args[0]!r}") from exc
    validate_match_acceptance(acceptance)
    return acceptance


def validate_acceptance_against_session(
    session: SearchSession,
    acceptance: MatchAcceptance,
    *,
    current_tick: int,
    current_catalog_revision: int,
    current_inventory_revision: int,
) -> OfferSnapshot:
    """Validate an acceptance against a persisted session and current World revisions."""

    validate_search_session(session)
    validate_match_acceptance(acceptance)
    _require_nonnegative_int(current_tick, "current_tick")
    _require_nonnegative_int(current_catalog_revision, "current_catalog_revision")
    _require_nonnegative_int(current_inventory_revision, "current_inventory_revision")

    _same("session_id", acceptance.session_id, session.session_id)
    _same("session_digest", acceptance.session_digest, session.session_digest)
    _same("buyer_id", acceptance.buyer_id, session.buyer_id)
    _same("mandate_id", acceptance.mandate_id, session.mandate_id)
    if current_tick >= session.expires_at_tick:
        raise MatchValidationError("search session expired")

    matches = [offer for offer in session.offers if offer.offer_id == acceptance.offer_id]
    if len(matches) != 1:
        raise MatchValidationError("accepted offer is not a unique member of the search session")
    offer = matches[0]
    _same("offer_digest", acceptance.offer_digest, offer.offer_digest)
    _same("buyer_id", acceptance.buyer_id, offer.buyer_id)
    _same("mandate_id", acceptance.mandate_id, offer.mandate_id)
    _same("merchant_id", acceptance.merchant_id, offer.merchant_id)
    _same("sku_id", acceptance.sku_id, offer.sku_id)
    _same("unit_price_cents", acceptance.unit_price_cents, offer.unit_price_cents)
    _same("currency", acceptance.currency, offer.currency)
    _same("qty", acceptance.qty, offer.qty)
    _same("catalog_revision", acceptance.catalog_revision, offer.catalog_revision)
    _same("inventory_revision", acceptance.inventory_revision, offer.inventory_revision)
    _same("current catalog revision", current_catalog_revision, offer.catalog_revision)
    _same("current inventory revision", current_inventory_revision, offer.inventory_revision)
    if current_tick >= offer.expires_at_tick:
        raise MatchValidationError("offer expired")
    return offer


def issue_match_certificate(
    session: SearchSession,
    acceptance: MatchAcceptance,
    *,
    current_tick: int,
    current_catalog_revision: int,
    current_inventory_revision: int,
    existing_certificate: MatchCertificate | None = None,
) -> MatchCertificate:
    """Issue a deterministic certificate, or replay an exact prior result.

    ``existing_certificate`` is the durable idempotency-store result for the
    acceptance key.  An exact retry returns those same bytes even if time or a
    World revision has advanced.  Reusing the key with any different bound
    acceptance raises :class:`IdempotencyConflict`; the returned certificate
    must still pass freshness validation when it is later consumed.
    """

    validate_search_session(session)
    validate_match_acceptance(acceptance)
    if existing_certificate is not None:
        validate_match_certificate(
            existing_certificate,
            session=session,
        )
        if existing_certificate.idempotency_key != acceptance.idempotency_key:
            raise IdempotencyConflict("match certificate idempotency key mismatch")
        if existing_certificate.acceptance_digest != acceptance.acceptance_digest:
            raise IdempotencyConflict(
                "match acceptance changed under an existing idempotency key"
            )
        validate_match_certificate(existing_certificate, acceptance=acceptance)
        return existing_certificate

    offer = validate_acceptance_against_session(
        session,
        acceptance,
        current_tick=current_tick,
        current_catalog_revision=current_catalog_revision,
        current_inventory_revision=current_inventory_revision,
    )
    unsigned = MatchCertificate(
        cert_id=match_certificate_id(acceptance.acceptance_digest),
        session_id=session.session_id,
        session_digest=session.session_digest,
        search_request_id=session.search_request_id,
        offer_id=offer.offer_id,
        offer_digest=offer.offer_digest,
        acceptance_request_id=acceptance.request_msg_id,
        acceptance_digest=acceptance.acceptance_digest,
        idempotency_key=acceptance.idempotency_key,
        buyer_id=acceptance.buyer_id,
        mandate_id=acceptance.mandate_id,
        order_id=acceptance.order_id,
        merchant_id=offer.merchant_id,
        sku_id=offer.sku_id,
        unit_price_cents=offer.unit_price_cents,
        currency=offer.currency,
        qty=offer.qty,
        catalog_revision=offer.catalog_revision,
        inventory_revision=offer.inventory_revision,
        issued_at_tick=current_tick,
        expires_at_tick=min(session.expires_at_tick, offer.expires_at_tick),
        validations=MATCH_BINDING_VALIDATIONS,
    )
    sealed = replace(unsigned, certificate_digest=match_certificate_digest(unsigned))
    validate_match_certificate(
        sealed,
        session=session,
        acceptance=acceptance,
        current_tick=current_tick,
        current_catalog_revision=current_catalog_revision,
        current_inventory_revision=current_inventory_revision,
        expected_buyer_id=acceptance.buyer_id,
        expected_order_id=acceptance.order_id,
    )
    return sealed


def match_certificate_digest(certificate: MatchCertificate) -> str:
    return canonical_digest(_certificate_contract(certificate))


def validate_match_certificate(
    certificate: MatchCertificate,
    *,
    session: SearchSession | None = None,
    acceptance: MatchAcceptance | None = None,
    current_tick: int | None = None,
    current_catalog_revision: int | None = None,
    current_inventory_revision: int | None = None,
    expected_buyer_id: str | None = None,
    expected_order_id: str | None = None,
) -> None:
    """Validate certificate integrity and any supplied authoritative context.

    Integrity alone does not prove Platform provenance.  A consumer must load
    the certificate from the trusted Platform store or canonical audit stream,
    then provide the persisted session and current World revisions here.
    """

    _validate_certificate_fields(certificate)
    _require_digest(certificate.certificate_digest, "certificate_digest")
    if certificate.certificate_digest != match_certificate_digest(certificate):
        raise MatchValidationError("match certificate digest mismatch")
    if certificate.cert_id != match_certificate_id(certificate.acceptance_digest):
        raise MatchValidationError("match certificate id is not deterministic")
    if certificate.validations != MATCH_BINDING_VALIDATIONS:
        raise MatchValidationError("match certificate validation set mismatch")

    offer: OfferSnapshot | None = None
    if session is not None:
        validate_search_session(session)
        _same("session_id", certificate.session_id, session.session_id)
        _same("session_digest", certificate.session_digest, session.session_digest)
        _same("search_request_id", certificate.search_request_id, session.search_request_id)
        _same("buyer_id", certificate.buyer_id, session.buyer_id)
        _same("mandate_id", certificate.mandate_id, session.mandate_id)
        matches = [item for item in session.offers if item.offer_id == certificate.offer_id]
        if len(matches) != 1:
            raise MatchValidationError("certified offer is not a unique session member")
        offer = matches[0]
        _same("offer_digest", certificate.offer_digest, offer.offer_digest)
        _same("merchant_id", certificate.merchant_id, offer.merchant_id)
        _same("sku_id", certificate.sku_id, offer.sku_id)
        _same("unit_price_cents", certificate.unit_price_cents, offer.unit_price_cents)
        _same("currency", certificate.currency, offer.currency)
        _same("qty", certificate.qty, offer.qty)
        _same("catalog_revision", certificate.catalog_revision, offer.catalog_revision)
        _same("inventory_revision", certificate.inventory_revision, offer.inventory_revision)
        expected_expiry = min(session.expires_at_tick, offer.expires_at_tick)
        _same("expires_at_tick", certificate.expires_at_tick, expected_expiry)

    if acceptance is not None:
        validate_match_acceptance(acceptance)
        _same("acceptance_digest", certificate.acceptance_digest, acceptance.acceptance_digest)
        _same("acceptance_request_id", certificate.acceptance_request_id, acceptance.request_msg_id)
        _same("idempotency_key", certificate.idempotency_key, acceptance.idempotency_key)
        for field in (
            "session_id",
            "session_digest",
            "offer_id",
            "offer_digest",
            "buyer_id",
            "mandate_id",
            "order_id",
            "merchant_id",
            "sku_id",
            "unit_price_cents",
            "currency",
            "qty",
            "catalog_revision",
            "inventory_revision",
        ):
            _same(field, getattr(certificate, field), getattr(acceptance, field))

    if current_tick is not None:
        _require_nonnegative_int(current_tick, "current_tick")
        if current_tick >= certificate.expires_at_tick:
            raise MatchValidationError("match certificate expired")
    if current_catalog_revision is not None:
        _require_nonnegative_int(current_catalog_revision, "current_catalog_revision")
        _same(
            "current catalog revision",
            current_catalog_revision,
            certificate.catalog_revision,
        )
    if current_inventory_revision is not None:
        _require_nonnegative_int(current_inventory_revision, "current_inventory_revision")
        _same(
            "current inventory revision",
            current_inventory_revision,
            certificate.inventory_revision,
        )
    if expected_buyer_id is not None:
        _same("expected buyer", expected_buyer_id, certificate.buyer_id)
    if expected_order_id is not None:
        _same("expected order", expected_order_id, certificate.order_id)


def match_certificate_to_wire(certificate: MatchCertificate) -> dict[str, Any]:
    validate_match_certificate(certificate)
    return {
        **_certificate_contract(certificate),
        "certificate_digest": certificate.certificate_digest,
    }


def coerce_match_certificate(value: Any) -> MatchCertificate:
    if isinstance(value, MatchCertificate):
        validate_match_certificate(value)
        return value
    row = _mapping(value, "match certificate")
    _require_exact_keys(row, _MATCH_CERTIFICATE_WIRE_KEYS, "match certificate")
    _require_schema(row, MATCH_CERTIFICATE_SCHEMA)
    try:
        raw_validations = row["validations"]
        if not isinstance(raw_validations, (list, tuple)) or not all(
            isinstance(item, str) for item in raw_validations
        ):
            raise SchemaError("match certificate validations must be a string array")
        certificate = MatchCertificate(
            cert_id=_wire_text(row, "cert_id"),
            session_id=_wire_text(row, "session_id"),
            session_digest=_wire_text(row, "session_digest"),
            search_request_id=_wire_text(row, "search_request_id"),
            offer_id=_wire_text(row, "offer_id"),
            offer_digest=_wire_text(row, "offer_digest"),
            acceptance_request_id=_wire_text(row, "acceptance_request_id"),
            acceptance_digest=_wire_text(row, "acceptance_digest"),
            idempotency_key=_wire_text(row, "idempotency_key"),
            buyer_id=_wire_text(row, "buyer_id"),
            mandate_id=_wire_text(row, "mandate_id"),
            order_id=_wire_text(row, "order_id"),
            merchant_id=_wire_text(row, "merchant_id"),
            sku_id=_wire_text(row, "sku_id"),
            unit_price_cents=_wire_int(row, "unit_price_cents"),
            currency=_wire_text(row, "currency"),
            qty=_wire_int(row, "qty"),
            catalog_revision=_wire_int(row, "catalog_revision"),
            inventory_revision=_wire_int(row, "inventory_revision"),
            issued_at_tick=_wire_int(row, "issued_at_tick"),
            expires_at_tick=_wire_int(row, "expires_at_tick"),
            validations=tuple(raw_validations),
            certificate_digest=_wire_text(row, "certificate_digest"),
        )
    except KeyError as exc:
        raise SchemaError(f"match certificate missing field {exc.args[0]!r}") from exc
    validate_match_certificate(certificate)
    return certificate


def _offer_contract(offer: OfferSnapshot) -> dict[str, Any]:
    return {
        "schema_version": OFFER_SNAPSHOT_SCHEMA,
        "offer_id": offer.offer_id,
        "session_id": offer.session_id,
        "buyer_id": offer.buyer_id,
        "mandate_id": offer.mandate_id,
        "merchant_id": offer.merchant_id,
        "sku_id": offer.sku_id,
        "unit_price_cents": offer.unit_price_cents,
        "currency": offer.currency,
        "qty": offer.qty,
        "catalog_revision": offer.catalog_revision,
        "inventory_revision": offer.inventory_revision,
        "issued_at_tick": offer.issued_at_tick,
        "expires_at_tick": offer.expires_at_tick,
    }


def _search_session_contract(session: SearchSession) -> dict[str, Any]:
    return {
        "schema_version": SEARCH_SESSION_SCHEMA,
        "session_id": session.session_id,
        "buyer_id": session.buyer_id,
        "mandate_id": session.mandate_id,
        "search_request_id": session.search_request_id,
        "search_idempotency_key": session.search_idempotency_key,
        "query_digest": session.query_digest,
        "issued_at_tick": session.issued_at_tick,
        "expires_at_tick": session.expires_at_tick,
        "offer_digests": [offer.offer_digest for offer in session.offers],
    }


def _acceptance_contract(acceptance: MatchAcceptance) -> dict[str, Any]:
    return {
        "schema_version": MATCH_ACCEPTANCE_SCHEMA,
        "request_msg_id": acceptance.request_msg_id,
        "idempotency_key": acceptance.idempotency_key,
        "session_id": acceptance.session_id,
        "session_digest": acceptance.session_digest,
        "offer_id": acceptance.offer_id,
        "offer_digest": acceptance.offer_digest,
        "buyer_id": acceptance.buyer_id,
        "mandate_id": acceptance.mandate_id,
        "order_id": acceptance.order_id,
        "merchant_id": acceptance.merchant_id,
        "sku_id": acceptance.sku_id,
        "unit_price_cents": acceptance.unit_price_cents,
        "currency": acceptance.currency,
        "qty": acceptance.qty,
        "catalog_revision": acceptance.catalog_revision,
        "inventory_revision": acceptance.inventory_revision,
    }


def _certificate_contract(certificate: MatchCertificate) -> dict[str, Any]:
    return {
        "schema_version": MATCH_CERTIFICATE_SCHEMA,
        "cert_id": certificate.cert_id,
        "session_id": certificate.session_id,
        "session_digest": certificate.session_digest,
        "search_request_id": certificate.search_request_id,
        "offer_id": certificate.offer_id,
        "offer_digest": certificate.offer_digest,
        "acceptance_request_id": certificate.acceptance_request_id,
        "acceptance_digest": certificate.acceptance_digest,
        "idempotency_key": certificate.idempotency_key,
        "buyer_id": certificate.buyer_id,
        "mandate_id": certificate.mandate_id,
        "order_id": certificate.order_id,
        "merchant_id": certificate.merchant_id,
        "sku_id": certificate.sku_id,
        "unit_price_cents": certificate.unit_price_cents,
        "currency": certificate.currency,
        "qty": certificate.qty,
        "catalog_revision": certificate.catalog_revision,
        "inventory_revision": certificate.inventory_revision,
        "issued_at_tick": certificate.issued_at_tick,
        "expires_at_tick": certificate.expires_at_tick,
        "validations": list(certificate.validations),
    }


def _validate_offer_fields(offer: OfferSnapshot) -> None:
    for field in (
        "offer_id",
        "session_id",
        "buyer_id",
        "mandate_id",
        "merchant_id",
        "sku_id",
        "currency",
    ):
        _require_text(getattr(offer, field), field)
    _require_positive_int(offer.unit_price_cents, "unit_price_cents")
    _require_positive_int(offer.qty, "qty")
    _require_nonnegative_int(offer.catalog_revision, "catalog_revision")
    _require_nonnegative_int(offer.inventory_revision, "inventory_revision")
    _validate_tick_range(offer.issued_at_tick, offer.expires_at_tick, "offer")


def _validate_search_session_fields(session: SearchSession) -> None:
    for field in (
        "session_id",
        "buyer_id",
        "mandate_id",
        "search_request_id",
        "search_idempotency_key",
    ):
        _require_text(getattr(session, field), field)
    _require_digest(session.query_digest, "query_digest")
    _validate_tick_range(session.issued_at_tick, session.expires_at_tick, "search session")
    if not isinstance(session.offers, tuple):
        raise SchemaError("search session offers must be a tuple")
    ids: set[str] = set()
    digests: set[str] = set()
    for offer in session.offers:
        validate_offer_snapshot(offer)
        _same("offer session_id", offer.session_id, session.session_id)
        _same("offer buyer_id", offer.buyer_id, session.buyer_id)
        _same("offer mandate_id", offer.mandate_id, session.mandate_id)
        if offer.issued_at_tick < session.issued_at_tick:
            raise MatchValidationError("offer predates its search session")
        if offer.expires_at_tick > session.expires_at_tick:
            raise MatchValidationError("offer outlives its search session")
        if offer.offer_id in ids:
            raise MatchValidationError("duplicate offer_id in search session")
        if offer.offer_digest in digests:
            raise MatchValidationError("duplicate offer digest in search session")
        ids.add(offer.offer_id)
        digests.add(offer.offer_digest)


def _validate_acceptance_fields(acceptance: MatchAcceptance) -> None:
    for field in (
        "request_msg_id",
        "idempotency_key",
        "session_id",
        "offer_id",
        "buyer_id",
        "mandate_id",
        "order_id",
        "merchant_id",
        "sku_id",
        "currency",
    ):
        _require_text(getattr(acceptance, field), field)
    _require_digest(acceptance.session_digest, "session_digest")
    _require_digest(acceptance.offer_digest, "offer_digest")
    _require_positive_int(acceptance.unit_price_cents, "unit_price_cents")
    _require_positive_int(acceptance.qty, "qty")
    _require_nonnegative_int(acceptance.catalog_revision, "catalog_revision")
    _require_nonnegative_int(acceptance.inventory_revision, "inventory_revision")


def _validate_certificate_fields(certificate: MatchCertificate) -> None:
    for field in (
        "cert_id",
        "session_id",
        "search_request_id",
        "offer_id",
        "acceptance_request_id",
        "idempotency_key",
        "buyer_id",
        "mandate_id",
        "order_id",
        "merchant_id",
        "sku_id",
        "currency",
    ):
        _require_text(getattr(certificate, field), field)
    for field in ("session_digest", "offer_digest", "acceptance_digest"):
        _require_digest(getattr(certificate, field), field)
    _require_positive_int(certificate.unit_price_cents, "unit_price_cents")
    _require_positive_int(certificate.qty, "qty")
    _require_nonnegative_int(certificate.catalog_revision, "catalog_revision")
    _require_nonnegative_int(certificate.inventory_revision, "inventory_revision")
    _validate_tick_range(
        certificate.issued_at_tick,
        certificate.expires_at_tick,
        "match certificate",
    )
    if not isinstance(certificate.validations, tuple) or not all(
        isinstance(item, str) and item for item in certificate.validations
    ):
        raise SchemaError("match certificate validations must be a string tuple")


def match_certificate_id(acceptance_digest: str) -> str:
    """Return the canonical certificate id for one sealed acceptance.

    This identity helper is public because trusted Platform services sometimes
    need to resolve a certificate from a persisted acceptance without accepting
    an untrusted certificate digest from a scenario or agent.
    """

    _require_digest(acceptance_digest, "acceptance_digest")
    return f"cert:{acceptance_digest[:32]}"


def _validate_tick_range(issued: int, expires: int, label: str) -> None:
    _require_nonnegative_int(issued, f"{label} issued_at_tick")
    _require_nonnegative_int(expires, f"{label} expires_at_tick")
    if expires <= issued:
        raise SchemaError(f"{label} expiry must be after issuance")


def _same(label: str, actual: Any, expected: Any) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise MatchValidationError(f"{label} mismatch")


def _require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{label} must be a non-empty string")


def _require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SchemaError(f"{label} must be a lowercase SHA-256 hex digest")


def _require_positive_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _require_schema(row: Mapping[str, Any], expected: str) -> None:
    if row.get("schema_version") != expected:
        raise SchemaError(f"unsupported schema_version for {expected}")


def _require_exact_keys(
    row: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = frozenset(row.keys())
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        raise SchemaError(f"{label} has invalid fields: {', '.join(details)}")


def _wire_text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{key} must be a non-empty string")
    return value


def _wire_int(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaError(f"{key} must be an integer")
    return value


__all__ = [
    "MATCH_ACCEPTANCE_SCHEMA",
    "MATCH_BINDING_VALIDATIONS",
    "MATCH_CERTIFICATE_SCHEMA",
    "OFFER_SNAPSHOT_SCHEMA",
    "SEARCH_SESSION_SCHEMA",
    "MatchAcceptance",
    "MatchAcceptanceRejected",
    "MatchCertificate",
    "MatchValidationError",
    "OfferSnapshot",
    "SearchSession",
    "canonical_digest",
    "coerce_match_acceptance",
    "coerce_match_certificate",
    "coerce_offer_snapshot",
    "coerce_search_session",
    "issue_match_certificate",
    "match_acceptance_digest",
    "match_acceptance_to_wire",
    "match_certificate_digest",
    "match_certificate_id",
    "match_certificate_to_wire",
    "offer_snapshot_digest",
    "offer_snapshot_to_wire",
    "search_session_digest",
    "search_session_to_wire",
    "seal_match_acceptance",
    "seal_offer_snapshot",
    "seal_search_session",
    "validate_acceptance_against_session",
    "validate_match_acceptance",
    "validate_match_certificate",
    "validate_offer_snapshot",
    "validate_search_session",
]
