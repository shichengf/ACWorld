"""Activation predicate for skills/merchant/reputation-aware-pricing/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext

from agents.skill_selector_merchant._shared import is_offer_turn


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on offer turns. The "no scarcity bias already written" defer
    is handled at the selector level (scarcity dominates reputation) — see
    ``_apply_defer_rules`` in this package's __init__.
    """
    if is_offer_turn(ctx.env):
        return True, "offer turn — reputation bias candidate"
    return False, "not an offer turn"
