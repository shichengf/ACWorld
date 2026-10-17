"""Activation predicate for skills/merchant/inquiry-handle/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires on commerce.send_message from a buyer (not owner)
    with no unit_price (it's a question, not an offer)."""
    kind = ctx.env.action.get("kind", "")
    if kind == "platform.catalog_listing":
        return True, "catalog grounding returned for the pending inquiry"
    if kind != "commerce.send_message":
        return False, f"not a free-text message ({kind!r})"
    sender = ctx.env.from_ or ""
    if not sender.startswith("buyer"):
        return False, f"sender is not buyer ({sender!r})"
    payload = ctx.env.action.get("payload") or {}
    if "unit_price" in payload:
        return False, "payload has unit_price — it is an offer, not a question"
    return True, "buyer asked a question"
