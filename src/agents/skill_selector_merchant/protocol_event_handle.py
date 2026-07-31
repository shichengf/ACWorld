"""Activation predicate for skills/merchant/protocol-event-handle/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


_KINDS = frozenset({
    "platform.deliver_protocol_event",
    "platform.protocol_event_receipt",
})


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = str(ctx.env.action.get("kind", ""))
    if kind in _KINDS:
        return True, f"persistent protocol-event continuation ({kind})"
    return False, f"not a protocol-event workflow ({kind!r})"
