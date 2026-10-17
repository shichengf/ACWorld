"""Activation predicate for skills/merchant/catalog-serve/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on direct buyer requests (legacy ``commerce.search`` /
    ``commerce.get_sku`` routed straight to the merchant)."""
    kind = ctx.env.action.get("kind", "")
    if kind in ("commerce.search", "commerce.get_sku"):
        return True, f"buyer requested catalog ({kind})"
    return False, f"not a catalog request (got {kind!r})"
