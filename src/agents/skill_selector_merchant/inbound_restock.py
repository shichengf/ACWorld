"""Activation predicate for skills/merchant/inbound-restock/SKILL.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    kind = ctx.env.action.get("kind", "")
    if kind != "delegate.receive_shipment":
        return False, f"not a receive_shipment directive ({kind!r})"
    payload = ctx.env.action.get("payload") or {}
    sku = payload.get("sku_id")
    qty = payload.get("qty")
    if not sku or not isinstance(qty, int) or qty <= 0:
        return False, "missing sku_id or non-positive qty"
    return True, f"owner shipped qty={qty} of {sku!r}"
