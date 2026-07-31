"""Persistent, replay-stable multi-line cart quote contracts.

This module is a domain protocol only.  It does not read catalog or inventory
tables, persist quotes, reserve stock, settle orders, or register Platform
actions.  A later World and Platform integration must construct these objects
from authoritative catalog, inventory, mandate, policy, and logical-time state.

All monetary values are integer minor units in the quote currency.  A line's
priced quantity is ``fulfill_now_qty + backorder_qty``.  Unfilled units are
recorded for evidence but are never included in the line total.  The sealed
quote commits every line owner, quantity split, price, source revision, charge,
buyer mandate, policy revision, and expiry through one canonical SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeAlias, cast

from protocol.errors import SchemaError


PERSISTENT_CART_QUOTE_SCHEMA = "cwe.persistent-cart-quote.v1"

FillPolicy: TypeAlias = Literal["all_or_none", "allow_partial"]
BackorderPolicy: TypeAlias = Literal["reject", "allow"]
ChargeKind: TypeAlias = Literal["shipping", "tax", "fee"]
ChargeScope: TypeAlias = Literal["cart", "merchant", "line"]
QuoteRetryDisposition: TypeAlias = Literal["append", "idempotent"]
Availability: TypeAlias = Literal[
    "available", "partially_available", "backordered", "mixed", "unavailable"
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_FILL_POLICIES = frozenset({"all_or_none", "allow_partial"})
_BACKORDER_POLICIES = frozenset({"reject", "allow"})
_CHARGE_KINDS = frozenset({"shipping", "tax", "fee"})
_CHARGE_SCOPES = frozenset({"cart", "merchant", "line"})


class CartQuoteContractError(SchemaError):
    """Base error for the persistent cart quote protocol."""


class CartQuoteSchemaError(CartQuoteContractError):
    """A quote or nested value violates its exact schema."""


class CartQuoteDigestMismatch(CartQuoteSchemaError):
    """A quote digest does not match its canonical semantic content."""


class CartQuoteAuthorityError(CartQuoteContractError):
    """Trusted issuer, buyer, mandate, or logical-time authority disagrees."""


class CartQuoteStaleError(CartQuoteAuthorityError):
    """A catalog, inventory, ownership, or expiry snapshot is no longer valid."""


class CartQuoteIdempotencyConflict(CartQuoteContractError):
    """A stable quote identity or idempotency scope has conflicting content."""


class CartQuoteIdentityError(CartQuoteContractError):
    """Two quotes do not share an identity or idempotency scope."""


@dataclass(frozen=True, slots=True)
class CartQuoteLine:
    """One SKU, its owner, accepted quantity split, and source revisions."""

    line_id: str
    sku_id: str
    product_id: str | None
    merchant_id: str
    requested_qty: int
    fulfill_now_qty: int
    backorder_qty: int
    unfilled_qty: int
    unit_price_minor: int
    line_total_minor: int
    catalog_revision: int
    catalog_digest: str
    inventory_revision: int
    inventory_digest: str
    expected_restock_tick: int | None = None

    @property
    def priced_qty(self) -> int:
        return self.fulfill_now_qty + self.backorder_qty


@dataclass(frozen=True, slots=True)
class CartQuoteCharge:
    """One non-negative shipping, tax, or other fee component."""

    charge_id: str
    kind: ChargeKind
    scope: ChargeScope
    scope_id: str | None
    amount_minor: int
    basis: str
    policy_revision: int
    policy_digest: str


@dataclass(frozen=True, slots=True)
class PersistentCartQuote:
    """An immutable, mandate-bound quote suitable for durable World storage."""

    quote_id: str
    request_id: str
    market_id: str
    buyer_id: str
    principal_id: str
    requested_by: str
    issuer_id: str
    mandate_id: str
    mandate_revision: int
    mandate_digest: str
    idempotency_key: str
    fill_policy: FillPolicy
    backorder_policy: BackorderPolicy
    currency: str
    lines: tuple[CartQuoteLine, ...]
    subtotal_minor: int
    charges: tuple[CartQuoteCharge, ...]
    shipping_minor: int
    tax_minor: int
    fee_minor: int
    grand_total_minor: int
    pricing_policy_revision: int
    pricing_policy_digest: str
    issued_at_tick: int
    expires_at_tick: int
    quote_digest: str = ""
    schema_id: str = PERSISTENT_CART_QUOTE_SCHEMA


def build_cart_quote_line(
    *,
    line_id: str,
    sku_id: str,
    product_id: str | None,
    merchant_id: str,
    requested_qty: int,
    fulfill_now_qty: int,
    backorder_qty: int,
    unfilled_qty: int,
    unit_price_minor: int,
    catalog_revision: int,
    catalog_digest: str,
    inventory_revision: int,
    inventory_digest: str,
    expected_restock_tick: int | None = None,
) -> CartQuoteLine:
    """Build one line and derive its arithmetic total."""

    line = CartQuoteLine(
        line_id=line_id,
        sku_id=sku_id,
        product_id=product_id,
        merchant_id=merchant_id,
        requested_qty=requested_qty,
        fulfill_now_qty=fulfill_now_qty,
        backorder_qty=backorder_qty,
        unfilled_qty=unfilled_qty,
        unit_price_minor=unit_price_minor,
        line_total_minor=unit_price_minor * (fulfill_now_qty + backorder_qty),
        catalog_revision=catalog_revision,
        catalog_digest=catalog_digest,
        inventory_revision=inventory_revision,
        inventory_digest=inventory_digest,
        expected_restock_tick=expected_restock_tick,
    )
    validate_cart_quote_line(line)
    return line


def validate_cart_quote_line(line: CartQuoteLine) -> None:
    if not isinstance(line, CartQuoteLine):
        raise CartQuoteSchemaError("line must be CartQuoteLine")
    for name in ("line_id", "sku_id", "merchant_id"):
        _require_text(name, getattr(line, name))
    if line.product_id is not None:
        _require_text("product_id", line.product_id)
    _require_positive_int("requested_qty", line.requested_qty)
    for name in ("fulfill_now_qty", "backorder_qty", "unfilled_qty"):
        _require_nonnegative_int(name, getattr(line, name))
    if line.fulfill_now_qty + line.backorder_qty + line.unfilled_qty != line.requested_qty:
        raise CartQuoteSchemaError(
            "requested quantity must equal fulfill-now, backorder, and unfilled quantities"
        )
    _require_positive_int("unit_price_minor", line.unit_price_minor)
    _require_nonnegative_int("line_total_minor", line.line_total_minor)
    if line.line_total_minor != line.unit_price_minor * line.priced_qty:
        raise CartQuoteSchemaError("line total disagrees with unit price and priced quantity")
    _require_nonnegative_int("catalog_revision", line.catalog_revision)
    _require_digest("catalog_digest", line.catalog_digest)
    _require_nonnegative_int("inventory_revision", line.inventory_revision)
    _require_digest("inventory_digest", line.inventory_digest)
    if line.backorder_qty:
        _require_nonnegative_int("expected_restock_tick", line.expected_restock_tick)
    elif line.expected_restock_tick is not None:
        raise CartQuoteSchemaError("only a backordered line may have expected_restock_tick")


def cart_quote_line_availability(line: CartQuoteLine) -> Availability:
    validate_cart_quote_line(line)
    if line.fulfill_now_qty == line.requested_qty:
        return "available"
    if line.fulfill_now_qty and line.backorder_qty:
        return "mixed"
    if line.backorder_qty and not line.fulfill_now_qty:
        return "backordered"
    if line.fulfill_now_qty:
        return "partially_available"
    return "unavailable"


def build_cart_quote_charge(
    *,
    charge_id: str,
    kind: ChargeKind,
    scope: ChargeScope,
    scope_id: str | None,
    amount_minor: int,
    basis: str,
    policy_revision: int,
    policy_digest: str,
) -> CartQuoteCharge:
    charge = CartQuoteCharge(
        charge_id=charge_id,
        kind=kind,
        scope=scope,
        scope_id=scope_id,
        amount_minor=amount_minor,
        basis=basis,
        policy_revision=policy_revision,
        policy_digest=policy_digest,
    )
    validate_cart_quote_charge(charge)
    return charge


def validate_cart_quote_charge(charge: CartQuoteCharge) -> None:
    if not isinstance(charge, CartQuoteCharge):
        raise CartQuoteSchemaError("charge must be CartQuoteCharge")
    _require_text("charge_id", charge.charge_id)
    if charge.kind not in _CHARGE_KINDS:
        raise CartQuoteSchemaError(f"unsupported charge kind: {charge.kind!r}")
    if charge.scope not in _CHARGE_SCOPES:
        raise CartQuoteSchemaError(f"unsupported charge scope: {charge.scope!r}")
    if charge.scope == "cart":
        if charge.scope_id is not None:
            raise CartQuoteSchemaError("cart-scoped charge scope_id must be null")
    else:
        _require_text("scope_id", charge.scope_id)
    _require_nonnegative_int("amount_minor", charge.amount_minor)
    _require_text("basis", charge.basis)
    _require_nonnegative_int("policy_revision", charge.policy_revision)
    _require_digest("policy_digest", charge.policy_digest)


def build_persistent_cart_quote(
    *,
    quote_id: str,
    request_id: str,
    market_id: str,
    buyer_id: str,
    principal_id: str,
    requested_by: str,
    issuer_id: str,
    mandate_id: str,
    mandate_revision: int,
    mandate_digest: str,
    idempotency_key: str,
    fill_policy: FillPolicy,
    backorder_policy: BackorderPolicy,
    currency: str,
    lines: Iterable[CartQuoteLine],
    charges: Iterable[CartQuoteCharge],
    pricing_policy_revision: int,
    pricing_policy_digest: str,
    issued_at_tick: int,
    expires_at_tick: int,
) -> PersistentCartQuote:
    """Build, canonicalize, validate, and seal a persistent quote."""

    provided_lines = tuple(lines)
    provided_charges = tuple(charges)
    for line in provided_lines:
        validate_cart_quote_line(line)
    for charge in provided_charges:
        validate_cart_quote_charge(charge)
    canonical_lines = tuple(sorted(provided_lines, key=lambda line: line.line_id))
    canonical_charges = tuple(sorted(provided_charges, key=lambda charge: charge.charge_id))
    subtotal = sum(line.line_total_minor for line in canonical_lines)
    shipping = sum(charge.amount_minor for charge in canonical_charges if charge.kind == "shipping")
    tax = sum(charge.amount_minor for charge in canonical_charges if charge.kind == "tax")
    fees = sum(charge.amount_minor for charge in canonical_charges if charge.kind == "fee")
    unsigned = PersistentCartQuote(
        quote_id=quote_id,
        request_id=request_id,
        market_id=market_id,
        buyer_id=buyer_id,
        principal_id=principal_id,
        requested_by=requested_by,
        issuer_id=issuer_id,
        mandate_id=mandate_id,
        mandate_revision=mandate_revision,
        mandate_digest=mandate_digest,
        idempotency_key=idempotency_key,
        fill_policy=fill_policy,
        backorder_policy=backorder_policy,
        currency=currency,
        lines=canonical_lines,
        subtotal_minor=subtotal,
        charges=canonical_charges,
        shipping_minor=shipping,
        tax_minor=tax,
        fee_minor=fees,
        grand_total_minor=subtotal + shipping + tax + fees,
        pricing_policy_revision=pricing_policy_revision,
        pricing_policy_digest=pricing_policy_digest,
        issued_at_tick=issued_at_tick,
        expires_at_tick=expires_at_tick,
    )
    _validate_quote_fields(unsigned, require_digest=False)
    sealed = _replace_quote_digest(unsigned, _digest(_quote_contract(unsigned)))
    validate_persistent_cart_quote(sealed)
    return sealed


def validate_persistent_cart_quote(quote: PersistentCartQuote) -> None:
    if not isinstance(quote, PersistentCartQuote):
        raise CartQuoteSchemaError("quote must be PersistentCartQuote")
    _validate_quote_fields(quote, require_digest=True)
    if quote.quote_digest != _digest(_quote_contract(quote)):
        raise CartQuoteDigestMismatch("persistent cart quote digest mismatch")


def persistent_cart_quote_to_dict(quote: PersistentCartQuote) -> dict[str, Any]:
    validate_persistent_cart_quote(quote)
    return {**_quote_contract(quote), "quote_digest": quote.quote_digest}


def persistent_cart_quote_to_json(quote: PersistentCartQuote) -> str:
    return _canonical_json(persistent_cart_quote_to_dict(quote))


def persistent_cart_quote_from_json(payload: str) -> PersistentCartQuote:
    """Parse only the exact canonical JSON representation."""

    value = _strict_json_loads(payload)
    if payload != _canonical_json(value):
        raise CartQuoteSchemaError("persistent cart quote wire JSON is not canonical")
    row = _exact_mapping(value, _QUOTE_FIELDS, "persistent cart quote")
    lines_value = row["lines"]
    charges_value = row["charges"]
    if not isinstance(lines_value, list):
        raise CartQuoteSchemaError("lines must be an array")
    if not isinstance(charges_value, list):
        raise CartQuoteSchemaError("charges must be an array")
    quote = PersistentCartQuote(
        schema_id=_text(row, "schema_id"),
        quote_id=_text(row, "quote_id"),
        request_id=_text(row, "request_id"),
        market_id=_text(row, "market_id"),
        buyer_id=_text(row, "buyer_id"),
        principal_id=_text(row, "principal_id"),
        requested_by=_text(row, "requested_by"),
        issuer_id=_text(row, "issuer_id"),
        mandate_id=_text(row, "mandate_id"),
        mandate_revision=_integer(row, "mandate_revision"),
        mandate_digest=_text(row, "mandate_digest"),
        idempotency_key=_text(row, "idempotency_key"),
        fill_policy=cast(FillPolicy, _text(row, "fill_policy")),
        backorder_policy=cast(BackorderPolicy, _text(row, "backorder_policy")),
        currency=_text(row, "currency"),
        lines=tuple(_coerce_line(item) for item in lines_value),
        subtotal_minor=_integer(row, "subtotal_minor"),
        charges=tuple(_coerce_charge(item) for item in charges_value),
        shipping_minor=_integer(row, "shipping_minor"),
        tax_minor=_integer(row, "tax_minor"),
        fee_minor=_integer(row, "fee_minor"),
        grand_total_minor=_integer(row, "grand_total_minor"),
        pricing_policy_revision=_integer(row, "pricing_policy_revision"),
        pricing_policy_digest=_text(row, "pricing_policy_digest"),
        issued_at_tick=_integer(row, "issued_at_tick"),
        expires_at_tick=_integer(row, "expires_at_tick"),
        quote_digest=_text(row, "quote_digest"),
    )
    validate_persistent_cart_quote(quote)
    return quote


def classify_cart_quote_retry(
    existing: PersistentCartQuote,
    candidate: PersistentCartQuote,
) -> QuoteRetryDisposition:
    """Classify a quote with a shared stable or actor-idempotency identity."""

    validate_persistent_cart_quote(existing)
    validate_persistent_cart_quote(candidate)
    shared_scope = (
        existing.quote_id == candidate.quote_id
        or (existing.market_id, existing.request_id) == (candidate.market_id, candidate.request_id)
        or (
            existing.market_id,
            existing.requested_by,
            existing.idempotency_key,
        )
        == (
            candidate.market_id,
            candidate.requested_by,
            candidate.idempotency_key,
        )
    )
    if not shared_scope:
        raise CartQuoteIdentityError("quotes do not share an identity or idempotency scope")
    if existing.quote_digest == candidate.quote_digest:
        return "idempotent"
    raise CartQuoteIdempotencyConflict(
        "quote identity or actor-scoped idempotency key has conflicting content"
    )


def apply_cart_quote_record(
    existing: Iterable[PersistentCartQuote],
    candidate: PersistentCartQuote,
    *,
    trusted_issuer_id: str,
    server_tick: int,
) -> tuple[QuoteRetryDisposition, tuple[PersistentCartQuote, ...]]:
    """Pure append/idempotency semantics for a future durable quote table."""

    records = tuple(existing)
    validate_persistent_cart_quote(candidate)
    _require_text("trusted_issuer_id", trusted_issuer_id)
    _require_nonnegative_int("server_tick", server_tick)
    if candidate.issuer_id != trusted_issuer_id:
        raise CartQuoteAuthorityError("quote issuer does not match trusted issuer")
    for record in records:
        validate_persistent_cart_quote(record)
        try:
            disposition = classify_cart_quote_retry(record, candidate)
        except CartQuoteIdentityError:
            continue
        return disposition, records
    if candidate.issued_at_tick != server_tick:
        raise CartQuoteAuthorityError("quote issued_at_tick is not the trusted server tick")
    return "append", (*records, candidate)


def replay_cart_quote_records(
    records: Iterable[PersistentCartQuote],
    *,
    trusted_issuer_id: str,
) -> tuple[PersistentCartQuote, ...]:
    replayed: tuple[PersistentCartQuote, ...] = ()
    for record in records:
        _, replayed = apply_cart_quote_record(
            replayed,
            record,
            trusted_issuer_id=trusted_issuer_id,
            server_tick=record.issued_at_tick,
        )
    return replayed


def assert_cart_quote_usable(
    quote: PersistentCartQuote,
    *,
    buyer_id: str,
    principal_id: str,
    mandate_id: str,
    mandate_revision: int,
    mandate_digest: str,
    logical_tick: int,
) -> None:
    """Check owner, exact mandate snapshot, and inclusive expiry at use time."""

    validate_persistent_cart_quote(quote)
    _require_text("buyer_id", buyer_id)
    _require_text("principal_id", principal_id)
    _require_text("mandate_id", mandate_id)
    _require_nonnegative_int("mandate_revision", mandate_revision)
    _require_digest("mandate_digest", mandate_digest)
    _require_nonnegative_int("logical_tick", logical_tick)
    if buyer_id != quote.buyer_id:
        raise CartQuoteAuthorityError("quote buyer binding mismatch")
    if principal_id != quote.principal_id:
        raise CartQuoteAuthorityError("quote principal binding mismatch")
    if (
        mandate_id != quote.mandate_id
        or mandate_revision != quote.mandate_revision
        or mandate_digest != quote.mandate_digest
    ):
        raise CartQuoteAuthorityError("quote mandate binding mismatch")
    if logical_tick < quote.issued_at_tick:
        raise CartQuoteAuthorityError("quote cannot be used before issuance")
    if logical_tick > quote.expires_at_tick:
        raise CartQuoteStaleError("quote has expired")


def assert_cart_quote_snapshots_current(
    quote: PersistentCartQuote,
    *,
    catalog_revisions: Mapping[str, int],
    catalog_digests: Mapping[str, str],
    inventory_revisions: Mapping[str, int],
    inventory_digests: Mapping[str, str],
    merchant_by_sku: Mapping[str, str],
) -> None:
    """Compare every sealed source revision and owner with authoritative state.

    Snapshot digests remain committed in the quote and must be recomputed by the
    future World integration.  This helper checks the revision counters and
    merchant ownership that World can query without trusting the quote itself.
    """

    validate_persistent_cart_quote(quote)
    for line in quote.lines:
        catalog_revision = catalog_revisions.get(line.sku_id)
        if (
            isinstance(catalog_revision, bool)
            or not isinstance(catalog_revision, int)
            or catalog_revision != line.catalog_revision
        ):
            raise CartQuoteStaleError(f"catalog revision changed for {line.sku_id!r}")
        if catalog_digests.get(line.sku_id) != line.catalog_digest:
            raise CartQuoteStaleError(f"catalog digest changed for {line.sku_id!r}")
        inventory_revision = inventory_revisions.get(line.sku_id)
        if (
            isinstance(inventory_revision, bool)
            or not isinstance(inventory_revision, int)
            or inventory_revision != line.inventory_revision
        ):
            raise CartQuoteStaleError(f"inventory revision changed for {line.sku_id!r}")
        if inventory_digests.get(line.sku_id) != line.inventory_digest:
            raise CartQuoteStaleError(f"inventory digest changed for {line.sku_id!r}")
        if merchant_by_sku.get(line.sku_id) != line.merchant_id:
            raise CartQuoteStaleError(f"merchant ownership changed for {line.sku_id!r}")


def _validate_quote_fields(
    quote: PersistentCartQuote,
    *,
    require_digest: bool,
) -> None:
    if quote.schema_id != PERSISTENT_CART_QUOTE_SCHEMA:
        raise CartQuoteSchemaError(f"unsupported cart quote schema: {quote.schema_id!r}")
    for name in (
        "quote_id",
        "request_id",
        "market_id",
        "buyer_id",
        "principal_id",
        "requested_by",
        "issuer_id",
        "mandate_id",
        "idempotency_key",
    ):
        _require_text(name, getattr(quote, name))
    _require_nonnegative_int("mandate_revision", quote.mandate_revision)
    _require_digest("mandate_digest", quote.mandate_digest)
    if quote.fill_policy not in _FILL_POLICIES:
        raise CartQuoteSchemaError(f"unsupported fill_policy: {quote.fill_policy!r}")
    if quote.backorder_policy not in _BACKORDER_POLICIES:
        raise CartQuoteSchemaError(f"unsupported backorder_policy: {quote.backorder_policy!r}")
    _require_currency(quote.currency)
    if not isinstance(quote.lines, tuple) or not quote.lines:
        raise CartQuoteSchemaError("quote lines must be a non-empty canonical tuple")
    for line in quote.lines:
        validate_cart_quote_line(line)
    if quote.lines != tuple(sorted(quote.lines, key=lambda line: line.line_id)):
        raise CartQuoteSchemaError("quote lines must be sorted by line_id")
    if len({line.line_id for line in quote.lines}) != len(quote.lines):
        raise CartQuoteSchemaError("quote contains duplicate line_id")
    if len({line.sku_id for line in quote.lines}) != len(quote.lines):
        raise CartQuoteSchemaError("quote contains duplicate sku_id")
    _validate_availability_policy(quote)
    _require_nonnegative_int("subtotal_minor", quote.subtotal_minor)
    if quote.subtotal_minor != sum(line.line_total_minor for line in quote.lines):
        raise CartQuoteSchemaError("quote subtotal disagrees with line totals")
    if not isinstance(quote.charges, tuple):
        raise CartQuoteSchemaError("quote charges must be a canonical tuple")
    for charge in quote.charges:
        validate_cart_quote_charge(charge)
    if quote.charges != tuple(sorted(quote.charges, key=lambda charge: charge.charge_id)):
        raise CartQuoteSchemaError("quote charges must be sorted by charge_id")
    if len({charge.charge_id for charge in quote.charges}) != len(quote.charges):
        raise CartQuoteSchemaError("quote contains duplicate charge_id")
    _validate_charge_scopes(quote)
    expected_shipping = sum(
        charge.amount_minor for charge in quote.charges if charge.kind == "shipping"
    )
    expected_tax = sum(charge.amount_minor for charge in quote.charges if charge.kind == "tax")
    expected_fees = sum(charge.amount_minor for charge in quote.charges if charge.kind == "fee")
    for name, actual, expected in (
        ("shipping_minor", quote.shipping_minor, expected_shipping),
        ("tax_minor", quote.tax_minor, expected_tax),
        ("fee_minor", quote.fee_minor, expected_fees),
    ):
        _require_nonnegative_int(name, actual)
        if actual != expected:
            raise CartQuoteSchemaError(f"{name} disagrees with charge components")
    _require_nonnegative_int("grand_total_minor", quote.grand_total_minor)
    if quote.grand_total_minor != (
        quote.subtotal_minor + quote.shipping_minor + quote.tax_minor + quote.fee_minor
    ):
        raise CartQuoteSchemaError("quote grand total disagrees with subtotal and charges")
    _require_nonnegative_int("pricing_policy_revision", quote.pricing_policy_revision)
    _require_digest("pricing_policy_digest", quote.pricing_policy_digest)
    _require_nonnegative_int("issued_at_tick", quote.issued_at_tick)
    _require_nonnegative_int("expires_at_tick", quote.expires_at_tick)
    if quote.expires_at_tick <= quote.issued_at_tick:
        raise CartQuoteSchemaError("quote expiry must be after issuance")
    for line in quote.lines:
        if (
            line.expected_restock_tick is not None
            and line.expected_restock_tick < quote.issued_at_tick
        ):
            raise CartQuoteSchemaError("expected restock tick predates quote issuance")
    if require_digest:
        _require_digest("quote_digest", quote.quote_digest)


def _validate_availability_policy(quote: PersistentCartQuote) -> None:
    if quote.fill_policy == "all_or_none":
        if any(line.unfilled_qty for line in quote.lines):
            raise CartQuoteSchemaError("all_or_none quote cannot leave units unfilled")
    if quote.backorder_policy == "reject":
        if any(line.backorder_qty for line in quote.lines):
            raise CartQuoteSchemaError("backorder-reject quote cannot contain backordered units")
    if quote.fill_policy == "all_or_none" and quote.backorder_policy == "reject":
        if any(line.fulfill_now_qty != line.requested_qty for line in quote.lines):
            raise CartQuoteSchemaError(
                "strict all-or-none quote requires immediate full availability"
            )
    if not any(line.priced_qty for line in quote.lines):
        raise CartQuoteSchemaError("quote must price at least one requested unit")


def _validate_charge_scopes(quote: PersistentCartQuote) -> None:
    merchant_ids = {line.merchant_id for line in quote.lines}
    line_ids = {line.line_id for line in quote.lines}
    for charge in quote.charges:
        if charge.scope == "merchant" and charge.scope_id not in merchant_ids:
            raise CartQuoteSchemaError("merchant-scoped charge references no quoted merchant")
        if charge.scope == "line" and charge.scope_id not in line_ids:
            raise CartQuoteSchemaError("line-scoped charge references no quoted line")


def _quote_contract(quote: PersistentCartQuote) -> dict[str, Any]:
    return {
        "schema_id": quote.schema_id,
        "quote_id": quote.quote_id,
        "request_id": quote.request_id,
        "market_id": quote.market_id,
        "buyer_id": quote.buyer_id,
        "principal_id": quote.principal_id,
        "requested_by": quote.requested_by,
        "issuer_id": quote.issuer_id,
        "mandate_id": quote.mandate_id,
        "mandate_revision": quote.mandate_revision,
        "mandate_digest": quote.mandate_digest,
        "idempotency_key": quote.idempotency_key,
        "fill_policy": quote.fill_policy,
        "backorder_policy": quote.backorder_policy,
        "currency": quote.currency,
        "lines": [_line_contract(line) for line in quote.lines],
        "subtotal_minor": quote.subtotal_minor,
        "charges": [_charge_contract(charge) for charge in quote.charges],
        "shipping_minor": quote.shipping_minor,
        "tax_minor": quote.tax_minor,
        "fee_minor": quote.fee_minor,
        "grand_total_minor": quote.grand_total_minor,
        "pricing_policy_revision": quote.pricing_policy_revision,
        "pricing_policy_digest": quote.pricing_policy_digest,
        "issued_at_tick": quote.issued_at_tick,
        "expires_at_tick": quote.expires_at_tick,
    }


def _line_contract(line: CartQuoteLine) -> dict[str, Any]:
    return {
        "line_id": line.line_id,
        "sku_id": line.sku_id,
        "product_id": line.product_id,
        "merchant_id": line.merchant_id,
        "requested_qty": line.requested_qty,
        "fulfill_now_qty": line.fulfill_now_qty,
        "backorder_qty": line.backorder_qty,
        "unfilled_qty": line.unfilled_qty,
        "unit_price_minor": line.unit_price_minor,
        "line_total_minor": line.line_total_minor,
        "catalog_revision": line.catalog_revision,
        "catalog_digest": line.catalog_digest,
        "inventory_revision": line.inventory_revision,
        "inventory_digest": line.inventory_digest,
        "expected_restock_tick": line.expected_restock_tick,
    }


def _charge_contract(charge: CartQuoteCharge) -> dict[str, Any]:
    return {
        "charge_id": charge.charge_id,
        "kind": charge.kind,
        "scope": charge.scope,
        "scope_id": charge.scope_id,
        "amount_minor": charge.amount_minor,
        "basis": charge.basis,
        "policy_revision": charge.policy_revision,
        "policy_digest": charge.policy_digest,
    }


def _replace_quote_digest(
    quote: PersistentCartQuote,
    digest: str,
) -> PersistentCartQuote:
    return PersistentCartQuote(
        quote_id=quote.quote_id,
        request_id=quote.request_id,
        market_id=quote.market_id,
        buyer_id=quote.buyer_id,
        principal_id=quote.principal_id,
        requested_by=quote.requested_by,
        issuer_id=quote.issuer_id,
        mandate_id=quote.mandate_id,
        mandate_revision=quote.mandate_revision,
        mandate_digest=quote.mandate_digest,
        idempotency_key=quote.idempotency_key,
        fill_policy=quote.fill_policy,
        backorder_policy=quote.backorder_policy,
        currency=quote.currency,
        lines=quote.lines,
        subtotal_minor=quote.subtotal_minor,
        charges=quote.charges,
        shipping_minor=quote.shipping_minor,
        tax_minor=quote.tax_minor,
        fee_minor=quote.fee_minor,
        grand_total_minor=quote.grand_total_minor,
        pricing_policy_revision=quote.pricing_policy_revision,
        pricing_policy_digest=quote.pricing_policy_digest,
        issued_at_tick=quote.issued_at_tick,
        expires_at_tick=quote.expires_at_tick,
        quote_digest=digest,
        schema_id=quote.schema_id,
    )


_LINE_FIELDS = frozenset(
    {
        "line_id",
        "sku_id",
        "product_id",
        "merchant_id",
        "requested_qty",
        "fulfill_now_qty",
        "backorder_qty",
        "unfilled_qty",
        "unit_price_minor",
        "line_total_minor",
        "catalog_revision",
        "catalog_digest",
        "inventory_revision",
        "inventory_digest",
        "expected_restock_tick",
    }
)
_CHARGE_FIELDS = frozenset(
    {
        "charge_id",
        "kind",
        "scope",
        "scope_id",
        "amount_minor",
        "basis",
        "policy_revision",
        "policy_digest",
    }
)
_QUOTE_FIELDS = frozenset(
    {
        "schema_id",
        "quote_id",
        "request_id",
        "market_id",
        "buyer_id",
        "principal_id",
        "requested_by",
        "issuer_id",
        "mandate_id",
        "mandate_revision",
        "mandate_digest",
        "idempotency_key",
        "fill_policy",
        "backorder_policy",
        "currency",
        "lines",
        "subtotal_minor",
        "charges",
        "shipping_minor",
        "tax_minor",
        "fee_minor",
        "grand_total_minor",
        "pricing_policy_revision",
        "pricing_policy_digest",
        "issued_at_tick",
        "expires_at_tick",
        "quote_digest",
    }
)


def _coerce_line(value: Any) -> CartQuoteLine:
    row = _exact_mapping(value, _LINE_FIELDS, "cart quote line")
    line = CartQuoteLine(
        line_id=_text(row, "line_id"),
        sku_id=_text(row, "sku_id"),
        product_id=_optional_text(row, "product_id"),
        merchant_id=_text(row, "merchant_id"),
        requested_qty=_integer(row, "requested_qty"),
        fulfill_now_qty=_integer(row, "fulfill_now_qty"),
        backorder_qty=_integer(row, "backorder_qty"),
        unfilled_qty=_integer(row, "unfilled_qty"),
        unit_price_minor=_integer(row, "unit_price_minor"),
        line_total_minor=_integer(row, "line_total_minor"),
        catalog_revision=_integer(row, "catalog_revision"),
        catalog_digest=_text(row, "catalog_digest"),
        inventory_revision=_integer(row, "inventory_revision"),
        inventory_digest=_text(row, "inventory_digest"),
        expected_restock_tick=_optional_integer(row, "expected_restock_tick"),
    )
    validate_cart_quote_line(line)
    return line


def _coerce_charge(value: Any) -> CartQuoteCharge:
    row = _exact_mapping(value, _CHARGE_FIELDS, "cart quote charge")
    charge = CartQuoteCharge(
        charge_id=_text(row, "charge_id"),
        kind=cast(ChargeKind, _text(row, "kind")),
        scope=cast(ChargeScope, _text(row, "scope")),
        scope_id=_optional_text(row, "scope_id"),
        amount_minor=_integer(row, "amount_minor"),
        basis=_text(row, "basis"),
        policy_revision=_integer(row, "policy_revision"),
        policy_digest=_text(row, "policy_digest"),
    )
    validate_cart_quote_charge(charge)
    return charge


def _exact_mapping(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CartQuoteSchemaError(f"{label} must be an object")
    actual = frozenset(value.keys())
    if any(not isinstance(key, str) for key in actual):
        raise CartQuoteSchemaError(f"{label} has non-string fields")
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise CartQuoteSchemaError(
            f"{label} invalid fields: missing={missing!r}, unknown={unknown!r}"
        )
    return cast(Mapping[str, Any], value)


def _strict_json_loads(payload: str) -> Any:
    if not isinstance(payload, str):
        raise CartQuoteSchemaError("persistent cart quote JSON must be a string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CartQuoteSchemaError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise CartQuoteSchemaError(f"non-finite JSON number: {value!r}")

    try:
        return json.loads(payload, object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except json.JSONDecodeError as exc:
        raise CartQuoteSchemaError(f"invalid persistent cart quote JSON: {exc}") from exc


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CartQuoteSchemaError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    _require_text(key, value)
    return cast(str, value)


def _optional_text(row: Mapping[str, Any], key: str) -> str | None:
    value = row[key]
    if value is None:
        return None
    _require_text(key, value)
    return cast(str, value)


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise CartQuoteSchemaError(f"{key} must be an integer")
    return cast(int, value)


def _optional_integer(row: Mapping[str, Any], key: str) -> int | None:
    if row[key] is None:
        return None
    return _integer(row, key)


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CartQuoteSchemaError(f"{name} must be a non-empty string")


def _require_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CartQuoteSchemaError(f"{name} must be a lowercase SHA-256 digest")


def _require_currency(value: Any) -> None:
    if not isinstance(value, str) or _CURRENCY_RE.fullmatch(value) is None:
        raise CartQuoteSchemaError("currency must be a three-letter uppercase code")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CartQuoteSchemaError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CartQuoteSchemaError(f"{name} must be a non-negative integer")


__all__ = [
    "PERSISTENT_CART_QUOTE_SCHEMA",
    "Availability",
    "BackorderPolicy",
    "CartQuoteAuthorityError",
    "CartQuoteCharge",
    "CartQuoteContractError",
    "CartQuoteDigestMismatch",
    "CartQuoteIdempotencyConflict",
    "CartQuoteIdentityError",
    "CartQuoteLine",
    "CartQuoteSchemaError",
    "CartQuoteStaleError",
    "ChargeKind",
    "ChargeScope",
    "FillPolicy",
    "PersistentCartQuote",
    "QuoteRetryDisposition",
    "apply_cart_quote_record",
    "assert_cart_quote_snapshots_current",
    "assert_cart_quote_usable",
    "build_cart_quote_charge",
    "build_cart_quote_line",
    "build_persistent_cart_quote",
    "cart_quote_line_availability",
    "classify_cart_quote_retry",
    "persistent_cart_quote_from_json",
    "persistent_cart_quote_to_dict",
    "persistent_cart_quote_to_json",
    "replay_cart_quote_records",
    "validate_cart_quote_charge",
    "validate_cart_quote_line",
    "validate_persistent_cart_quote",
]
