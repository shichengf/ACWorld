"""Privacy-safe Platform projection of current World negotiation state.

The projection carries only fields an Agent compiler needs to bind the next
high-level response.  A terminal rejection or withdrawal deliberately omits
the current price so the Platform cannot reintroduce a private budget or floor
that the public relay removed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from protocol.errors import SchemaError
from protocol.negotiation_state import NegotiationThread, validate_negotiation_thread


NEGOTIATION_TURN_PROJECTION_SCHEMA = "cwe.negotiation-turn-projection.v2"


@dataclass(frozen=True, slots=True)
class NegotiationTurnProjection:
    schema_id: str
    negotiation_id: str
    offer_id: str
    buyer_id: str
    merchant_id: str
    sku_id: str
    listing_digest: str
    currency: str
    qty: int
    round_no: int
    max_rounds: int
    status: str
    last_actor_id: str
    last_counterparty_id: str
    current_unit_price: int | None
    projection_digest: str = ""


def build_negotiation_turn_projection(
    thread: NegotiationThread,
    *,
    disclose_price: bool,
) -> NegotiationTurnProjection:
    validate_negotiation_thread(thread)
    value = NegotiationTurnProjection(
        schema_id=NEGOTIATION_TURN_PROJECTION_SCHEMA,
        negotiation_id=thread.negotiation_id,
        offer_id=thread.offer_id,
        buyer_id=thread.buyer_id,
        merchant_id=thread.merchant_id,
        sku_id=thread.sku_id,
        listing_digest=thread.listing_digest,
        currency=thread.currency,
        qty=thread.qty,
        round_no=thread.round_no,
        max_rounds=thread.max_rounds,
        status=thread.status,
        last_actor_id=thread.last_actor_id,
        last_counterparty_id=thread.last_counterparty_id,
        current_unit_price=(thread.current_unit_price if disclose_price else None),
    )
    sealed = replace(value, projection_digest=_digest(_unsigned(value)))
    validate_negotiation_turn_projection(sealed)
    return sealed


def validate_negotiation_turn_projection(value: NegotiationTurnProjection) -> None:
    if not isinstance(value, NegotiationTurnProjection):
        raise SchemaError("negotiation turn projection has the wrong type")
    if value.schema_id != NEGOTIATION_TURN_PROJECTION_SCHEMA:
        raise SchemaError("unsupported negotiation turn projection schema")
    for name in (
        "negotiation_id",
        "offer_id",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "listing_digest",
        "currency",
        "status",
        "last_actor_id",
        "last_counterparty_id",
    ):
        field = getattr(value, name)
        if not isinstance(field, str) or not field.strip():
            raise SchemaError(f"negotiation turn projection {name} is invalid")
    if value.status not in {"active", "accepted", "rejected", "withdrawn"}:
        raise SchemaError("negotiation turn projection status is invalid")
    for name in ("qty", "round_no", "max_rounds"):
        field = getattr(value, name)
        if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
            raise SchemaError(f"negotiation turn projection {name} is invalid")
    if value.round_no > value.max_rounds:
        raise SchemaError("negotiation turn projection round exceeds its limit")
    if value.current_unit_price is not None and (
        isinstance(value.current_unit_price, bool)
        or not isinstance(value.current_unit_price, int)
        or value.current_unit_price <= 0
    ):
        raise SchemaError("negotiation turn projection price is invalid")
    if value.status in {"rejected", "withdrawn"} and value.current_unit_price is not None:
        raise SchemaError("terminal negotiation projection must hide price")
    if value.projection_digest != _digest(_unsigned(value)):
        raise SchemaError("negotiation turn projection digest mismatch")


def negotiation_turn_projection_to_dict(
    value: NegotiationTurnProjection,
) -> dict[str, Any]:
    validate_negotiation_turn_projection(value)
    return asdict(value)


def coerce_negotiation_turn_projection(value: Any) -> NegotiationTurnProjection:
    if isinstance(value, NegotiationTurnProjection):
        validate_negotiation_turn_projection(value)
        return value
    if not isinstance(value, Mapping):
        raise SchemaError("negotiation turn projection must be an object")
    expected = {field.name for field in NegotiationTurnProjection.__dataclass_fields__.values()}
    if set(value) != expected:
        raise SchemaError("negotiation turn projection fields are not exact")
    try:
        row = NegotiationTurnProjection(**dict(value))
    except TypeError as exc:
        raise SchemaError("negotiation turn projection is malformed") from exc
    validate_negotiation_turn_projection(row)
    return row


def _unsigned(value: NegotiationTurnProjection) -> dict[str, Any]:
    row = asdict(value)
    row.pop("projection_digest", None)
    return row


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "NEGOTIATION_TURN_PROJECTION_SCHEMA",
    "NegotiationTurnProjection",
    "build_negotiation_turn_projection",
    "coerce_negotiation_turn_projection",
    "negotiation_turn_projection_to_dict",
    "validate_negotiation_turn_projection",
]
