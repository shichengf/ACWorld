"""World-persisted authorization for merchant-authored cart quotes.

The requester submits only a mandate identity and requested SKU quantities.
World derives every commercial binding from authoritative state and seals the
result.  A merchant may later issue a quote only by presenting the opaque
``request_id``.  The record intentionally contains no mandate values such as a
budget, so quote issuance cannot be used as a budget-probing oracle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from protocol.cart_quote_state import BackorderPolicy, FillPolicy
from protocol.errors import SchemaError
from protocol.matching import canonical_digest


CART_QUOTE_REQUEST_SCHEMA = "cwe.cart-quote-request.v1"


class CartQuoteRequestError(SchemaError):
    """Base error for cart quote request authorization records."""


class CartQuoteRequestAuthorityError(CartQuoteRequestError):
    """The authenticated actor or current authority does not match."""


class CartQuoteRequestStaleError(CartQuoteRequestAuthorityError):
    """The request expired or an authoritative binding changed."""


class CartQuoteRequestConflict(CartQuoteRequestError):
    """An identity or idempotency key was reused with different content."""


@dataclass(frozen=True, slots=True)
class CartQuoteRequestLine:
    """One requested SKU with World-derived listing ownership and revision."""

    sku_id: str
    product_id: str | None
    merchant_id: str
    qty: int
    catalog_revision: int
    catalog_digest: str


@dataclass(frozen=True, slots=True)
class PersistentCartQuoteRequest:
    """Immutable authorization for a merchant to price an exact cart."""

    request_id: str
    market_id: str
    buyer_id: str
    principal_id: str
    created_by: str
    issuer_id: str
    mandate_id: str
    mandate_revision: int
    mandate_digest: str
    idempotency_key: str
    fill_policy: FillPolicy
    backorder_policy: BackorderPolicy
    currency: str
    lines: tuple[CartQuoteRequestLine, ...]
    allowed_merchant_ids: tuple[str, ...]
    issued_at_tick: int
    expires_at_tick: int
    request_digest: str = ""
    schema_id: str = CART_QUOTE_REQUEST_SCHEMA


def build_persistent_cart_quote_request(
    *,
    request_id: str,
    market_id: str,
    buyer_id: str,
    principal_id: str,
    created_by: str,
    issuer_id: str,
    mandate_id: str,
    mandate_revision: int,
    mandate_digest: str,
    idempotency_key: str,
    fill_policy: FillPolicy,
    backorder_policy: BackorderPolicy,
    currency: str,
    lines: Iterable[CartQuoteRequestLine],
    issued_at_tick: int,
    expires_at_tick: int,
) -> PersistentCartQuoteRequest:
    """Build and seal a request after World has supplied all authority facts."""

    canonical_lines = tuple(lines)
    merchants = tuple(sorted({line.merchant_id for line in canonical_lines}))
    unsigned = PersistentCartQuoteRequest(
        request_id=request_id,
        market_id=market_id,
        buyer_id=buyer_id,
        principal_id=principal_id,
        created_by=created_by,
        issuer_id=issuer_id,
        mandate_id=mandate_id,
        mandate_revision=mandate_revision,
        mandate_digest=mandate_digest,
        idempotency_key=idempotency_key,
        fill_policy=fill_policy,
        backorder_policy=backorder_policy,
        currency=currency,
        lines=canonical_lines,
        allowed_merchant_ids=merchants,
        issued_at_tick=issued_at_tick,
        expires_at_tick=expires_at_tick,
    )
    sealed = _replace_digest(unsigned, canonical_digest(_semantic_wire(unsigned)))
    validate_persistent_cart_quote_request(sealed)
    return sealed


def validate_persistent_cart_quote_request(
    request: PersistentCartQuoteRequest,
) -> None:
    if not isinstance(request, PersistentCartQuoteRequest):
        raise CartQuoteRequestError("request must be PersistentCartQuoteRequest")
    if request.schema_id != CART_QUOTE_REQUEST_SCHEMA:
        raise CartQuoteRequestError("unsupported cart quote request schema")
    for name in (
        "request_id",
        "market_id",
        "buyer_id",
        "principal_id",
        "created_by",
        "issuer_id",
        "mandate_id",
        "mandate_digest",
        "idempotency_key",
        "currency",
        "request_digest",
    ):
        _text(getattr(request, name), name)
    if not request.buyer_id.startswith("buyer:"):
        raise CartQuoteRequestError("buyer_id must be fully qualified")
    if request.created_by not in {request.buyer_id, request.principal_id}:
        raise CartQuoteRequestAuthorityError(
            "request creator must be the bound buyer or principal"
        )
    if request.fill_policy not in {"all_or_none", "allow_partial"}:
        raise CartQuoteRequestError("invalid fill policy")
    if request.backorder_policy not in {"reject", "allow"}:
        raise CartQuoteRequestError("invalid backorder policy")
    if request.mandate_revision < 1:
        raise CartQuoteRequestError("mandate revision must be positive")
    if request.issued_at_tick < 0 or request.expires_at_tick <= request.issued_at_tick:
        raise CartQuoteRequestError("request expiry must follow issue time")
    if not request.lines or len(request.lines) > 64:
        raise CartQuoteRequestError("request requires between 1 and 64 lines")
    if tuple(sorted(request.allowed_merchant_ids)) != request.allowed_merchant_ids:
        raise CartQuoteRequestError("allowed merchant ids must be canonical")
    if request.allowed_merchant_ids != tuple(
        sorted({line.merchant_id for line in request.lines})
    ):
        raise CartQuoteRequestError("allowed merchants disagree with request lines")
    seen: set[str] = set()
    for line in request.lines:
        if not isinstance(line, CartQuoteRequestLine):
            raise CartQuoteRequestError("request line has wrong type")
        for name in ("sku_id", "merchant_id", "catalog_digest"):
            _text(getattr(line, name), f"line.{name}")
        if line.product_id is not None:
            _text(line.product_id, "line.product_id")
        if line.qty <= 0 or isinstance(line.qty, bool):
            raise CartQuoteRequestError("request line quantity must be positive")
        if line.catalog_revision < 0 or isinstance(line.catalog_revision, bool):
            raise CartQuoteRequestError("catalog revision must be non-negative")
        if line.sku_id in seen:
            raise CartQuoteRequestError("request cannot repeat a SKU")
        seen.add(line.sku_id)
    expected = canonical_digest(_semantic_wire(request))
    if request.request_digest != expected:
        raise CartQuoteRequestError("cart quote request digest mismatch")


def persistent_cart_quote_request_to_dict(
    request: PersistentCartQuoteRequest,
) -> dict[str, Any]:
    validate_persistent_cart_quote_request(request)
    return asdict(request)


def coerce_persistent_cart_quote_request(value: Any) -> PersistentCartQuoteRequest:
    if isinstance(value, PersistentCartQuoteRequest):
        validate_persistent_cart_quote_request(value)
        return value
    if not isinstance(value, Mapping):
        raise CartQuoteRequestError("cart quote request must be an object")
    try:
        raw_lines = value["lines"]
        if not isinstance(raw_lines, (list, tuple)):
            raise TypeError
        lines = tuple(
            CartQuoteRequestLine(
                sku_id=str(line["sku_id"]),
                product_id=(
                    None if line.get("product_id") is None else str(line["product_id"])
                ),
                merchant_id=str(line["merchant_id"]),
                qty=int(line["qty"]),
                catalog_revision=int(line["catalog_revision"]),
                catalog_digest=str(line["catalog_digest"]),
            )
            for line in raw_lines
            if isinstance(line, Mapping)
        )
        request = PersistentCartQuoteRequest(
            request_id=str(value["request_id"]),
            market_id=str(value["market_id"]),
            buyer_id=str(value["buyer_id"]),
            principal_id=str(value["principal_id"]),
            created_by=str(value["created_by"]),
            issuer_id=str(value["issuer_id"]),
            mandate_id=str(value["mandate_id"]),
            mandate_revision=int(value["mandate_revision"]),
            mandate_digest=str(value["mandate_digest"]),
            idempotency_key=str(value["idempotency_key"]),
            fill_policy=str(value["fill_policy"]),  # type: ignore[arg-type]
            backorder_policy=str(value["backorder_policy"]),  # type: ignore[arg-type]
            currency=str(value["currency"]),
            lines=lines,
            allowed_merchant_ids=tuple(str(item) for item in value["allowed_merchant_ids"]),
            issued_at_tick=int(value["issued_at_tick"]),
            expires_at_tick=int(value["expires_at_tick"]),
            request_digest=str(value["request_digest"]),
            schema_id=str(value.get("schema_id", CART_QUOTE_REQUEST_SCHEMA)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CartQuoteRequestError("invalid cart quote request object") from exc
    validate_persistent_cart_quote_request(request)
    return request


def assert_cart_quote_request_usable(
    request: PersistentCartQuoteRequest,
    *,
    merchant_id: str,
    mandate_revision: int,
    mandate_digest: str,
    logical_tick: int,
) -> None:
    """Validate the opaque authorization without consulting mandate values."""

    validate_persistent_cart_quote_request(request)
    if merchant_id not in request.allowed_merchant_ids:
        raise CartQuoteRequestAuthorityError("merchant is not authorized by request")
    if any(line.merchant_id != merchant_id for line in request.lines):
        raise CartQuoteRequestAuthorityError(
            "merchant quote requests must contain only that merchant's listings"
        )
    if (
        request.mandate_revision != mandate_revision
        or request.mandate_digest != mandate_digest
    ):
        raise CartQuoteRequestStaleError("request mandate revision is stale")
    if logical_tick >= request.expires_at_tick:
        raise CartQuoteRequestStaleError("cart quote request expired")


def _semantic_wire(request: PersistentCartQuoteRequest) -> dict[str, Any]:
    return {
        "schema_id": request.schema_id,
        "request_id": request.request_id,
        "market_id": request.market_id,
        "buyer_id": request.buyer_id,
        "principal_id": request.principal_id,
        "created_by": request.created_by,
        "issuer_id": request.issuer_id,
        "mandate_id": request.mandate_id,
        "mandate_revision": request.mandate_revision,
        "mandate_digest": request.mandate_digest,
        "idempotency_key": request.idempotency_key,
        "fill_policy": request.fill_policy,
        "backorder_policy": request.backorder_policy,
        "currency": request.currency,
        "lines": [asdict(line) for line in request.lines],
        "allowed_merchant_ids": list(request.allowed_merchant_ids),
        "issued_at_tick": request.issued_at_tick,
        "expires_at_tick": request.expires_at_tick,
    }


def _replace_digest(
    request: PersistentCartQuoteRequest, digest: str
) -> PersistentCartQuoteRequest:
    values = asdict(request)
    values["lines"] = request.lines
    values["allowed_merchant_ids"] = request.allowed_merchant_ids
    values["request_digest"] = digest
    return PersistentCartQuoteRequest(**values)


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CartQuoteRequestError(f"{field} must be non-empty text")


__all__ = [
    "CART_QUOTE_REQUEST_SCHEMA",
    "CartQuoteRequestAuthorityError",
    "CartQuoteRequestConflict",
    "CartQuoteRequestError",
    "CartQuoteRequestLine",
    "CartQuoteRequestStaleError",
    "PersistentCartQuoteRequest",
    "assert_cart_quote_request_usable",
    "build_persistent_cart_quote_request",
    "coerce_persistent_cart_quote_request",
    "persistent_cart_quote_request_to_dict",
    "validate_persistent_cart_quote_request",
]
