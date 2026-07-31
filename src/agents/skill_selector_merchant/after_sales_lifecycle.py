"""Activation predicate for skills/merchant/after-sales-lifecycle/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.skill_selector_merchant._shared import message_mentions

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


_KINDS = frozenset({
    "platform.after_sales_updated",
    "platform.after_sales_snapshot",
})
_MESSAGE_TERMS = frozenset({
    "after-sales",
    "after sales",
    "cancel",
    "dispute",
    "exchange",
    "ledger",
    "reconcile",
    "refund",
    "return",
})
_AFTER_SALES_REFERENCE_KEYS = frozenset({
    "order_id",
    "order_ids",
    "request_id",
    "authorization_id",
    "case_id",
    "dispute_id",
})


def _has_after_sales_reference(ctx: "_SelectorContext") -> bool:
    """Recognize structured follow-ups that omit redundant instruction text."""

    if str(ctx.env.action.get("kind", "")) != "commerce.send_message":
        return False
    payload = ctx.env.action.get("payload")
    return isinstance(payload, dict) and bool(
        _AFTER_SALES_REFERENCE_KEYS.intersection(payload)
    )


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = str(ctx.env.action.get("kind", ""))
    if kind in _KINDS:
        return True, f"authoritative after-sales continuation ({kind})"
    if message_mentions(ctx.env, _MESSAGE_TERMS):
        return True, "message requests an after-sales operation"
    if _has_after_sales_reference(ctx):
        return True, "message continues a referenced after-sales case"
    return False, f"not an after-sales workflow ({kind!r})"
