"""Activation predicate for skills/merchant/fulfillment/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = ctx.env.action.get("kind", "")
    if kind == "platform.settlement_receipt":
        return True, "settled order needs dispatch"
    if kind == "commerce.request_return":
        return True, "return path enters fulfillment then return-adjudicate"
    return False, f"no fulfillment trigger ({kind!r})"
