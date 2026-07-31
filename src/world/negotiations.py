"""World-owned negotiation intent normalization and authoritative derivation.

Actors submit only commercial intent.  Listing ownership, participants, World
clock, event sequence, lineage digests, and agreements are derived inside the
World transaction boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping, NotRequired, TypedDict, cast

from protocol.negotiation_state import (
    ACCEPT_OFFER,
    COUNTER_OFFER,
    NEGOTIATION_ACTIONS,
    PROPOSE_OFFER,
    REJECT_OFFER,
    WITHDRAW_OFFER,
    NegotiationAction,
    NegotiationSchemaError,
)
from world.types import Listing


class NegotiationIntent(TypedDict):
    """Strict compact intent accepted by the World authority method."""

    action_kind: NegotiationAction
    negotiation_id: str
    offer_id: str
    sku_id: str
    counterparty_id: str
    unit_price: NotRequired[int]
    round_no: NotRequired[int]
    qty: NotRequired[int]


_BASE_FIELDS = frozenset(
    {"action_kind", "negotiation_id", "offer_id", "sku_id", "counterparty_id"}
)
_OPTIONAL_FIELDS = frozenset({"unit_price", "round_no", "qty"})


def normalize_negotiation_intent(value: Mapping[str, Any]) -> NegotiationIntent:
    """Return one exact compact intent and reject authority-bearing fields."""

    if not isinstance(value, Mapping):
        raise NegotiationSchemaError("negotiation intent must be an object")
    if any(not isinstance(key, str) for key in value):
        raise NegotiationSchemaError("negotiation intent fields must be strings")
    actual = frozenset(value)
    missing = _BASE_FIELDS - actual
    unknown = actual - _BASE_FIELDS - _OPTIONAL_FIELDS
    if missing:
        raise NegotiationSchemaError(
            "negotiation intent missing fields: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise NegotiationSchemaError(
            "negotiation intent has authority or unknown fields: "
            + ", ".join(sorted(unknown))
        )

    action_kind = value["action_kind"]
    if not isinstance(action_kind, str) or action_kind not in NEGOTIATION_ACTIONS:
        raise NegotiationSchemaError("unsupported negotiation action")
    normalized: dict[str, Any] = {"action_kind": action_kind}
    for key in ("negotiation_id", "offer_id", "sku_id", "counterparty_id"):
        item = value[key]
        if not isinstance(item, str) or not item.strip():
            raise NegotiationSchemaError(f"{key} must be a non-empty string")
        normalized[key] = item.strip()

    if "unit_price" in value:
        normalized["unit_price"] = _positive_int(value["unit_price"], "unit_price")
    if "round_no" in value:
        normalized["round_no"] = _positive_int(value["round_no"], "round_no")
    if "qty" in value:
        normalized["qty"] = _positive_int(value["qty"], "qty")

    if action_kind in {PROPOSE_OFFER, COUNTER_OFFER, ACCEPT_OFFER}:
        if "unit_price" not in normalized:
            raise NegotiationSchemaError("unit_price is required for this action")
    if action_kind != PROPOSE_OFFER and "qty" in normalized:
        raise NegotiationSchemaError("qty may be supplied only on the proposal")
    return cast(NegotiationIntent, normalized)


def negotiation_intent_fingerprint(
    intent: NegotiationIntent,
    *,
    max_rounds: int,
    deadline_ticks: int,
) -> str:
    """Bind compact actor intent and trusted Platform policy configuration."""

    return _digest(
        {
            "intent": dict(intent),
            "max_rounds": _positive_int(max_rounds, "max_rounds"),
            "deadline_ticks": _positive_int(deadline_ticks, "deadline_ticks"),
        }
    )


def negotiation_event_id(
    negotiation_id: str,
    actor_id: str,
    idempotency_key: str,
) -> str:
    """Derive an immutable event id from the actor-scoped authority identity."""

    digest = _digest(
        {
            "negotiation_id": negotiation_id,
            "actor_id": actor_id,
            "idempotency_key": idempotency_key,
        }
    )
    return f"negotiation-event:{digest[:32]}"


def negotiation_listing_digest(listing: Listing) -> str:
    """Hash only the complete public authoritative listing representation."""

    amount = listing.list_price.amount
    if not isinstance(amount, Decimal):
        amount = Decimal(str(amount))
    cents = int((amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return _digest(
        {
            "sku_id": str(listing.sku_id),
            "product_id": listing.product_id,
            "merchant_id": str(listing.merchant_id),
            "category": listing.category,
            "name": listing.name,
            "attributes": _jsonable(listing.attributes),
            "list_price": {
                "amount_cents": cents,
                "currency": listing.list_price.currency,
            },
        }
    )


def negotiation_status_for_action(action_kind: str) -> str:
    if action_kind == ACCEPT_OFFER:
        return "accepted"
    if action_kind == REJECT_OFFER:
        return "rejected"
    if action_kind == WITHDRAW_OFFER:
        return "withdrawn"
    return "active"


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NegotiationSchemaError(f"{label} must be a positive integer")
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
    except (TypeError, ValueError) as exc:
        raise NegotiationSchemaError(f"listing is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NegotiationSchemaError(f"negotiation value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "NegotiationIntent",
    "negotiation_event_id",
    "negotiation_intent_fingerprint",
    "negotiation_listing_digest",
    "negotiation_status_for_action",
    "normalize_negotiation_intent",
]
