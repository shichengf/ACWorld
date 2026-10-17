"""Activation predicate for skills/merchant/dispute-defense/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = ctx.env.action.get("kind", "")
    if kind == "platform.open_dispute":
        return True, "platform opened a dispute"
    return False, f"not a dispute trigger ({kind!r})"
