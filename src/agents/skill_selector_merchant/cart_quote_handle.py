"""Activation predicate for skills/merchant/cart-quote-handle/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


_KINDS = frozenset({
    "platform.cart_quote_request",
    "platform.cart_quote",
    "platform.cart_settlement",
})


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = str(ctx.env.action.get("kind", ""))
    if kind in _KINDS:
        return True, f"cart quote lifecycle continuation ({kind})"
    return False, f"not a cart quote continuation ({kind!r})"
