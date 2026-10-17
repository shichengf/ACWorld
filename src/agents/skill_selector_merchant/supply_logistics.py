"""Activation predicate for skills/merchant/supply-logistics/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.skill_selector_merchant._shared import message_mentions

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


_KINDS = frozenset({
    "platform.supply_state",
    "platform.supply_event_applied",
    "platform.fulfillment_allocation",
    "platform.allocation_batch",
    "platform.shipment_state",
    "platform.shipment_resolved",
})
_MESSAGE_TERMS = frozenset({
    "allocation",
    "backorder",
    "fulfillment",
    "inventory",
    "logistics",
    "shipment",
    "stock",
    "supply",
})


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = str(ctx.env.action.get("kind", ""))
    if kind in _KINDS:
        return True, f"authoritative supply or logistics continuation ({kind})"
    if message_mentions(ctx.env, _MESSAGE_TERMS):
        return True, "message requests supply or logistics work"
    return False, f"not a supply or logistics workflow ({kind!r})"
