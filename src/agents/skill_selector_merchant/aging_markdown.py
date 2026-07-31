"""Activation predicate for skills/merchant/aging-markdown/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.skill_selector_merchant._shared import is_idle_turn


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on idle turns when stock is aged AND not scarce.

    The skill body's "When NOT to fire" also forbids running during
    propose/counter turns (pricing-negotiate owns those) — covered by
    the idle-turn check.
    """
    if not is_idle_turn(ctx.env):
        return False, "not an idle turn"
    if ctx.merchant_data is None:
        return False, "no merchant_data attached"
    policy = ctx.merchant_data.policy
    after_days = int(policy.markdown_rules.get("after_days", 30))
    window = int(policy.scarcity_window_days)
    # Aging-markdown applies to a SKU; iterate the merchant's known
    # SKUs and fire if any of them satisfy the aged + not-scarce gate.
    # The skill body then picks the specific SKU. We don't peek at qty
    # here — InventoryView lookup happens in the skill itself.
    for sku, state in ctx.merchant_data.private_states.items():
        if state.inventory_age_days > after_days:
            # Velocity-aware: "not scarce" means days_to_stockout > window.
            v = state.velocity_per_day or 0.0
            if ctx.inventory_view is not None and v > 0:
                qty = ctx.inventory_view.qty_available(sku)
                dts = qty / v
                if dts <= window:
                    continue
            return True, f"sku {sku!r} aged ({state.inventory_age_days}d) + not scarce"
    return False, "no SKU is aged + not-scarce"
