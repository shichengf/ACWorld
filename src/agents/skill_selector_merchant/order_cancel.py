"""Activation predicate for skills/merchant/order-cancel/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = ctx.env.action.get("kind", "")
    if kind == "commerce.cancel_order":
        return True, "cancel request inbound"
    return False, f"not a cancel ({kind!r})"
