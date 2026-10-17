"""Activation predicate for skills/merchant/listing-publish/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on delegate.update_listing for the merchant:catalog role."""
    kind = ctx.env.action.get("kind", "")
    if kind != "delegate.update_listing":
        return False, f"not a listing directive ({kind!r})"
    payload = ctx.env.action.get("payload") or {}
    op = payload.get("op")
    if op not in ("publish", "update", "delist"):
        return False, f"unknown op ({op!r})"
    return True, f"owner asked to {op}"
