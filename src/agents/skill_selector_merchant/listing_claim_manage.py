"""Activation predicate for skills/merchant/listing-claim-manage/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.skill_selector_merchant._shared import message_mentions

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


_MESSAGE_TERMS = frozenset({
    "claim",
    "attribute",
    "correction",
    "evidence",
    "listing",
    "retract",
    "truthful",
})


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = str(ctx.env.action.get("kind", ""))
    if kind == "platform.listing_claim_updated":
        return True, "Platform returned an authoritative listing-claim update"
    if message_mentions(ctx.env, _MESSAGE_TERMS):
        return True, "message requests claim-grounded catalog work"
    return False, f"not a listing-claim workflow ({kind!r})"
