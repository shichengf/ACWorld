"""Activation predicate for skills/merchant/claim-truthfulness/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    """Fires as a pre-step on inbounds whose payload may carry claims
    (commerce.respond_inquiry, propose/counter offers with GroundedOffer).

    Conservative activation: any offer-lane turn OR any inquiry turn.
    The skill body's filter_truthful_claims is a no-op if the outbound
    has no claims field.
    """
    kind = ctx.env.action.get("kind", "")
    if kind in ("commerce.propose_offer", "commerce.counter_offer",
                "commerce.send_message", "commerce.get_sku",
                "commerce.search"):
        return True, f"claim-bearing outbound possible ({kind})"
    return False, f"no claim-bearing outbound expected ({kind!r})"
