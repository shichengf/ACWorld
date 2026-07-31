"""Activation predicate for skills/merchant/pricing-negotiate/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.skill_selector_merchant._shared import is_offer_turn


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on any offer-lane inbound (propose / counter / accept / reject)."""
    if is_offer_turn(ctx.env):
        return True, f"offer-lane inbound ({ctx.env.action.get('kind')})"
    return False, "not an offer turn"
