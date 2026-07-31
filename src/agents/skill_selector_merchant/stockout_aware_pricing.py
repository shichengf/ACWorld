"""Activation predicate for skills/merchant/stockout-aware-pricing/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.skill_selector_merchant._shared import is_decision_offer_turn, sku_from_payload


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on offer turns when stock is scarce AND demand is high."""
    if not is_decision_offer_turn(ctx.env):
        return False, "not a decision-offer turn"
    if ctx.merchant_data is None or ctx.inventory_view is None:
        return False, "no merchant_data / inventory_view attached"
    sku = sku_from_payload(ctx.env)
    if sku is None:
        return False, "inbound has no sku_id"
    state = ctx.merchant_data.private_states.get(sku)
    if state is None or not state.velocity_per_day:
        return False, "velocity unknown for sku"
    qty = ctx.inventory_view.qty_available(sku)
    dts = qty / state.velocity_per_day
    policy = ctx.merchant_data.policy
    if dts > policy.scarcity_window_days:
        return False, f"days_to_stockout={dts:.1f} > window={policy.scarcity_window_days}"
    demand = ctx.merchant_data.demand.get(sku)
    opp = demand.opp_score if demand else 0
    if opp < policy.demand_floor_for_scarcity:
        return False, f"opp_score={opp} < floor={policy.demand_floor_for_scarcity}"
    return True, f"scarce ({dts:.1f}d) + in-demand ({opp})"
