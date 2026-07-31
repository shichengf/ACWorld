"""Cross-gate helpers for the merchant-lane skill selector.

Anything reused by 2+ gate files lands here. The convention is small,
pure functions over ``_SelectorContext``-like inputs (or just the
envelope when that's all the gate needs).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from protocol.envelope import Envelope

#: Wire kinds that constitute "an offer turn" — the merchant has a buyer
#: offer that needs a price-decision response. ``accept_offer`` and
#: ``reject_offer`` are inbound for the merchant when the BUYER is the
#: one accepting / rejecting (Branch N-B / N-D); ``pricing-negotiate``
#: still owns the turn (logs + emits ``no_reply``).
OFFER_KINDS: frozenset[str] = frozenset({
    "commerce.propose_offer",
    "commerce.counter_offer",
    "commerce.accept_offer",
    "commerce.reject_offer",
})

#: Decision-required offer kinds — those where the merchant must compute
#: and emit a price response (not just record).
DECISION_OFFER_KINDS: frozenset[str] = frozenset({
    "commerce.propose_offer",
    "commerce.counter_offer",
})


def is_offer_turn(env: "Envelope") -> bool:
    """True when the inbound is in the offer lane (any of OFFER_KINDS)."""
    return str(env.action.get("kind", "")) in OFFER_KINDS


def is_decision_offer_turn(env: "Envelope") -> bool:
    """True when the merchant must emit a price decision (propose / counter)."""
    return str(env.action.get("kind", "")) in DECISION_OFFER_KINDS


#: Inbound envelope kinds that demand an active response from the
#: merchant lane. An "idle turn" is anything NOT in this set —
#: proactive skills (aging-markdown / demand-driven-markup /
#: peer-pricing / restock-signal idle branch) fire only on idle turns.
_REACTIVE_KINDS: frozenset[str] = OFFER_KINDS | frozenset({
    "commerce.create_order",
    "commerce.cancel_order",
    "commerce.request_return",
    "commerce.send_message",       # buyer inquiry
    "commerce.search",             # discovery routed direct to merchant (legacy)
    "commerce.get_sku",            # listing request
    "platform.settlement_receipt",
    "platform.open_dispute",
    "platform.reputation_updated",
    "platform.catalog_ack",
    "platform.catalog_listing",  # World-backed continuation of a listing read
    "platform.listing_claim_updated",
    "platform.cart_quote_request",
    "platform.cart_quote",
    "platform.cart_settlement",
    "platform.supply_state",
    "platform.supply_event_applied",
    "platform.fulfillment_allocation",
    "platform.allocation_batch",
    "platform.shipment_state",
    "platform.shipment_resolved",
    "platform.after_sales_updated",
    "platform.after_sales_snapshot",
    "platform.governance_updated",
    "platform.governance_snapshot",
    "platform.governance_case_notice",
    "platform.remediation_plan_notice",
    "platform.remediation_audit_request",
    "platform.evidence_record_persisted",
    "platform.deliver_protocol_event",
    "platform.protocol_event_receipt",
    "delegate.update_listing",
    "delegate.receive_shipment",
})


def is_idle_turn(env: "Envelope") -> bool:
    """True when the inbound is NOT in the merchant's reactive surface."""
    kind = str(env.action.get("kind", ""))
    return kind not in _REACTIVE_KINDS


def sku_from_payload(env: "Envelope") -> str | None:
    """Extract sku_id from the inbound payload, or None if absent."""
    payload = env.action.get("payload") or {}
    sku = payload.get("sku_id")
    return sku if isinstance(sku, str) and sku else None


def payload_text(env: "Envelope") -> str:
    """Return normalized public instruction text from a message payload.

    Operational kickoffs are regular ``commerce.send_message`` envelopes.  A
    selector therefore cannot route them by wire kind alone.  This helper
    extracts only human-authored instruction-like fields, recursively, and
    deliberately ignores task identifiers and private agent state.
    """

    payload = env.action.get("payload") or {}
    fragments: list[str] = []

    def visit(value: Any, *, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, key=str(child_key).casefold())
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key=key)
            return
        if isinstance(value, str) and key in {
            "category",
            "goal",
            "instruction",
            "operation",
            "question",
            "reason",
            "summary",
        }:
            fragments.append(value.casefold())

    visit(payload)
    return " ".join(fragments)


def message_mentions(env: "Envelope", terms: frozenset[str]) -> bool:
    """Whether a generic message contains one of the public workflow terms."""

    if str(env.action.get("kind", "")) != "commerce.send_message":
        return False
    text = payload_text(env)
    return any(term in text for term in terms)


__all__ = [
    "DECISION_OFFER_KINDS",
    "OFFER_KINDS",
    "is_decision_offer_turn",
    "is_idle_turn",
    "is_offer_turn",
    "message_mentions",
    "payload_text",
    "sku_from_payload",
]
