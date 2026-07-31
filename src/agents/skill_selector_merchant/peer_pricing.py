"""Activation predicate for skills/merchant/peer-pricing/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.skill_selector_merchant._shared import is_idle_turn


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on idle turns when the merchant has an active listing and
    peers exist. The skill body checks cooldown + min_peers + similarity
    against world; we don't replicate those costly checks here.
    """
    if not is_idle_turn(ctx.env):
        return False, "not an idle turn"
    if ctx.merchant_data is None:
        return False, "no merchant_data attached"
    if not ctx.merchant_data.private_states:
        return False, "merchant has no active listings"
    # The "no recent alignment" / "peers above min_peers" gates require
    # cross-turn memory + world reads; the selector activates broadly
    # and the skill body short-circuits if those preconditions fail.
    return True, "merchant has active listings; idle turn permits peer-alignment check"
