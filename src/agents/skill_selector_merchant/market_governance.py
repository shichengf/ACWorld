"""Activation predicate for skills/merchant/market-governance/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.skill_selector_merchant._shared import message_mentions

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


_KINDS = frozenset({
    "platform.governance_updated",
    "platform.governance_snapshot",
    "platform.governance_case_notice",
    "platform.remediation_plan_notice",
    "platform.remediation_audit_request",
    "platform.evidence_record_persisted",
})
_MESSAGE_TERMS = frozenset({
    "campaign",
    "collusion",
    "coordination",
    "disclosure",
    "governance",
    "placement",
    "remediation",
    "review",
    "sponsor",
})


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = str(ctx.env.action.get("kind", ""))
    if kind in _KINDS:
        return True, f"authoritative governance continuation ({kind})"
    if message_mentions(ctx.env, _MESSAGE_TERMS):
        return True, "message requests marketplace governance work"
    return False, f"not a governance workflow ({kind!r})"
