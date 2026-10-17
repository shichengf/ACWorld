"""Activation predicate for skills/merchant/restock-signal/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.skill_selector_merchant._shared import is_idle_turn


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Two triggers per SKILL.md:

    * Immediately after fulfillment.dispatch on a settled order
      (post-settle path); inbound here is ``platform.settlement_receipt``.
    * On an idle turn when any SKU's count <= stockout_threshold.
    """
    kind = ctx.env.action.get("kind", "")
    if kind == "platform.settlement_receipt":
        return True, "post-settle restock check"
    if not is_idle_turn(ctx.env):
        return False, "not an idle turn and not a settlement receipt"
    if ctx.merchant_data is None or ctx.inventory_view is None:
        return False, "no inventory_view to check stock against"
    threshold = int(ctx.merchant_data.policy.stockout_threshold)
    for sku in ctx.merchant_data.private_states:
        qty = ctx.inventory_view.qty_available(sku)
        if qty <= threshold:
            return True, f"sku {sku!r} qty={qty} <= threshold={threshold}"
    return False, "no SKU at or below stockout threshold"
