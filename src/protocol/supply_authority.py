"""World-issued authority for a direct purchase from an exact supply read.

The record is deliberately independent from benchmark tasks.  A Platform
supply service asks World to mint it from current catalog and inventory state;
the payment service later consumes the persisted record as transaction
authority.  Agent text is never an authority source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from protocol.errors import SchemaError


SUPPLY_PURCHASE_AUTHORITY_SCHEMA_VERSION = "cwe.supply-purchase-authority.v1"
DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS = 10_000
MAX_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS = 1_000_000


def validate_supply_purchase_authority_ttl_ticks(value: Any) -> int:
    """Return a bounded, positive authority lifetime.

    The lifetime is measured in explicit CommerceWorld market-clock ticks.
    Issuing another actor's read authority does not advance that clock.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS
    ):
        raise SchemaError(
            "supply authority ttl_ticks must be between 1 and "
            f"{MAX_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS}"
        )
    return value


@dataclass(frozen=True, slots=True)
class SupplyPurchaseAuthority:
    """One immutable, buyer-scoped right to request a supply purchase."""

    authority_id: str
    authority_digest: str
    buyer_id: str
    merchant_id: str
    sku_id: str
    order_id: str
    unit_price_cents: int
    currency: str
    available_qty: int
    supply_version: int
    issued_at_tick: int
    expires_at_tick: int
    request_idempotency_key: str
    issuer_id: str = "world"
    schema_version: str = SUPPLY_PURCHASE_AUTHORITY_SCHEMA_VERSION


def supply_purchase_authority_id(
    *,
    buyer_id: str,
    request_idempotency_key: str,
    sku_id: str,
) -> str:
    """Derive a stable authority id for an exact supply-read request."""

    canonical = _canonical_json({
        "buyer_id": _text(buyer_id, "buyer_id"),
        "request_idempotency_key": _text(
            request_idempotency_key, "request_idempotency_key"
        ),
        "sku_id": _text(sku_id, "sku_id"),
    })
    return "supply-authority:" + hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:32]


def supply_purchase_order_id(
    *,
    buyer_id: str,
    request_idempotency_key: str,
    sku_id: str,
) -> str:
    """Derive the stable order identity attached to one supply authority."""

    authority_id = supply_purchase_authority_id(
        buyer_id=buyer_id,
        request_idempotency_key=request_idempotency_key,
        sku_id=sku_id,
    )
    return "order:supply:" + hashlib.sha256(
        authority_id.encode("utf-8")
    ).hexdigest()[:32]


def build_supply_purchase_authority(
    *,
    buyer_id: str,
    merchant_id: str,
    sku_id: str,
    unit_price_cents: int,
    currency: str,
    available_qty: int,
    supply_version: int,
    issued_at_tick: int,
    expires_at_tick: int,
    request_idempotency_key: str,
) -> SupplyPurchaseAuthority:
    """Build and seal an authority entirely from trusted World values."""

    authority_id = supply_purchase_authority_id(
        buyer_id=buyer_id,
        request_idempotency_key=request_idempotency_key,
        sku_id=sku_id,
    )
    authority = SupplyPurchaseAuthority(
        authority_id=authority_id,
        authority_digest="",
        buyer_id=buyer_id,
        merchant_id=merchant_id,
        sku_id=sku_id,
        order_id=supply_purchase_order_id(
            buyer_id=buyer_id,
            request_idempotency_key=request_idempotency_key,
            sku_id=sku_id,
        ),
        unit_price_cents=unit_price_cents,
        currency=currency,
        available_qty=available_qty,
        supply_version=supply_version,
        issued_at_tick=issued_at_tick,
        expires_at_tick=expires_at_tick,
        request_idempotency_key=request_idempotency_key,
    )
    authority = replace(
        authority,
        authority_digest=_authority_digest(authority),
    )
    validate_supply_purchase_authority(authority)
    return authority


def validate_supply_purchase_authority(
    authority: SupplyPurchaseAuthority,
) -> None:
    """Validate shape, identity derivation, time interval, and seal."""

    if not isinstance(authority, SupplyPurchaseAuthority):
        raise SchemaError("supply purchase authority has the wrong type")
    for name in (
        "authority_id",
        "authority_digest",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "order_id",
        "currency",
        "request_idempotency_key",
        "issuer_id",
        "schema_version",
    ):
        _text(getattr(authority, name), name)
    if not authority.buyer_id.startswith("buyer:"):
        raise SchemaError("supply authority buyer_id must name a buyer")
    if not authority.merchant_id.startswith("merchant:"):
        raise SchemaError("supply authority merchant_id must name a merchant")
    if authority.issuer_id != "world":
        raise SchemaError("supply authority issuer must be World")
    if authority.schema_version != SUPPLY_PURCHASE_AUTHORITY_SCHEMA_VERSION:
        raise SchemaError("unsupported supply authority schema version")
    for name in (
        "unit_price_cents",
        "available_qty",
        "supply_version",
        "issued_at_tick",
        "expires_at_tick",
    ):
        value = getattr(authority, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError(f"supply authority {name} must be non-negative")
    if authority.supply_version < 1:
        raise SchemaError("supply authority version must be positive")
    if authority.expires_at_tick <= authority.issued_at_tick:
        raise SchemaError("supply authority expiry must follow issuance")
    expected_id = supply_purchase_authority_id(
        buyer_id=authority.buyer_id,
        request_idempotency_key=authority.request_idempotency_key,
        sku_id=authority.sku_id,
    )
    if authority.authority_id != expected_id:
        raise SchemaError("supply authority id is not canonical")
    expected_order = supply_purchase_order_id(
        buyer_id=authority.buyer_id,
        request_idempotency_key=authority.request_idempotency_key,
        sku_id=authority.sku_id,
    )
    if authority.order_id != expected_order:
        raise SchemaError("supply authority order id is not canonical")
    if authority.authority_digest != _authority_digest(authority):
        raise SchemaError("supply authority digest is invalid")


def supply_purchase_authority_to_dict(
    authority: SupplyPurchaseAuthority,
) -> dict[str, Any]:
    validate_supply_purchase_authority(authority)
    return _authority_payload(authority, include_digest=True)


def coerce_supply_purchase_authority(value: Any) -> SupplyPurchaseAuthority:
    if isinstance(value, SupplyPurchaseAuthority):
        validate_supply_purchase_authority(value)
        return value
    if not isinstance(value, Mapping):
        raise SchemaError("supply purchase authority must be an object")
    required = {
        "authority_id",
        "authority_digest",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "order_id",
        "unit_price_cents",
        "currency",
        "available_qty",
        "supply_version",
        "issued_at_tick",
        "expires_at_tick",
        "request_idempotency_key",
        "issuer_id",
        "schema_version",
    }
    if set(value) != required:
        raise SchemaError("supply purchase authority fields are not exact")
    try:
        authority = SupplyPurchaseAuthority(
            authority_id=value["authority_id"],
            authority_digest=value["authority_digest"],
            buyer_id=value["buyer_id"],
            merchant_id=value["merchant_id"],
            sku_id=value["sku_id"],
            order_id=value["order_id"],
            unit_price_cents=value["unit_price_cents"],
            currency=value["currency"],
            available_qty=value["available_qty"],
            supply_version=value["supply_version"],
            issued_at_tick=value["issued_at_tick"],
            expires_at_tick=value["expires_at_tick"],
            request_idempotency_key=value["request_idempotency_key"],
            issuer_id=value["issuer_id"],
            schema_version=value["schema_version"],
        )
    except TypeError as exc:
        raise SchemaError("supply purchase authority is malformed") from exc
    validate_supply_purchase_authority(authority)
    return authority


def supply_purchase_authority_to_json(
    authority: SupplyPurchaseAuthority,
) -> str:
    return _canonical_json(supply_purchase_authority_to_dict(authority))


def supply_purchase_authority_from_json(value: str) -> SupplyPurchaseAuthority:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError("supply authority JSON is invalid") from exc
    return coerce_supply_purchase_authority(payload)


def _authority_digest(authority: SupplyPurchaseAuthority) -> str:
    return hashlib.sha256(
        _canonical_json(
            _authority_payload(authority, include_digest=False)
        ).encode("utf-8")
    ).hexdigest()


def _authority_payload(
    authority: SupplyPurchaseAuthority,
    *,
    include_digest: bool,
) -> dict[str, Any]:
    payload = {
        "authority_id": authority.authority_id,
        "buyer_id": authority.buyer_id,
        "merchant_id": authority.merchant_id,
        "sku_id": authority.sku_id,
        "order_id": authority.order_id,
        "unit_price_cents": authority.unit_price_cents,
        "currency": authority.currency,
        "available_qty": authority.available_qty,
        "supply_version": authority.supply_version,
        "issued_at_tick": authority.issued_at_tick,
        "expires_at_tick": authority.expires_at_tick,
        "request_idempotency_key": authority.request_idempotency_key,
        "issuer_id": authority.issuer_id,
        "schema_version": authority.schema_version,
    }
    if include_digest:
        payload["authority_digest"] = authority.authority_digest
    return payload


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"supply authority {name} must be non-empty text")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "SUPPLY_PURCHASE_AUTHORITY_SCHEMA_VERSION",
    "SupplyPurchaseAuthority",
    "build_supply_purchase_authority",
    "coerce_supply_purchase_authority",
    "supply_purchase_authority_from_json",
    "supply_purchase_authority_id",
    "supply_purchase_authority_to_dict",
    "supply_purchase_authority_to_json",
    "supply_purchase_order_id",
    "validate_supply_purchase_authority",
]
