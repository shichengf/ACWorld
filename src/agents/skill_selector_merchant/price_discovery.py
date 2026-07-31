"""Activation predicate for skills/merchant/price-discovery/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.types import MemoryType


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on the first turn of a sell mandate or on pricing_intent change.

    Pragmatic activation: fire whenever the merchant doesn't yet have
    a ``target_band`` written in memory for the inbound SKU. The skill
    body otherwise short-circuits.
    """
    from agents.skill_selector_merchant._shared import sku_from_payload, is_offer_turn
    if not is_offer_turn(ctx.env):
        return False, "no offer to discover a price for"
    sku = sku_from_payload(ctx.env)
    if sku is None:
        return False, "no sku_id in payload"
    target_band = ctx.memory.read(MemoryType.PREFERENCE, f"target_band:{sku}")
    if target_band is not None:
        return False, f"target_band already established for {sku!r}"
    return True, f"first offer turn for {sku!r}; price-discovery needed"
