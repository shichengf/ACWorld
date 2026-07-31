"""Activation predicate for skills/merchant/demand-driven-markup/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.skill_selector_merchant._shared import is_idle_turn


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on idle turns when demand + velocity high AND stock not aged."""
    if not is_idle_turn(ctx.env):
        return False, "not an idle turn"
    if ctx.merchant_data is None:
        return False, "no merchant_data attached"
    policy = ctx.merchant_data.policy
    demand_floor = int(policy.markup_demand_floor)
    velocity_floor = float(policy.markup_velocity_floor)
    aged_after = int(policy.markdown_rules.get("after_days", 30))

    for sku, state in ctx.merchant_data.private_states.items():
        if state.inventory_age_days > aged_after:
            continue  # aged — aging-markdown's lane
        if (state.velocity_per_day or 0.0) < velocity_floor:
            continue
        demand = ctx.merchant_data.demand.get(sku)
        if demand is None or demand.opp_score < demand_floor:
            continue
        return True, f"sku {sku!r} fresh + fast + in-demand"
    return False, "no SKU meets markup criteria"
