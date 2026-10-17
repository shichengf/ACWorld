"""Example: a market event and matching deterministic oracle primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from episode.extensions import ORACLE_PRIMITIVES, WORLD_EVENT_HANDLERS
from world.types import SkuId


@WORLD_EVENT_HANDLERS.decorator(
    "example.flash-restock",
    description="Add a deterministic quantity to one existing inventory row",
)
def apply_event(world: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    sku = SkuId(str(payload["sku_id"]))
    quantity = int(payload["quantity"])
    if quantity <= 0:
        raise ValueError("restock quantity must be positive")
    before = world.read("inventory", sku, caller="platform:event")
    if before is None:
        raise ValueError(f"unknown inventory sku: {sku}")
    after = replace(before, qty_available=before.qty_available + quantity)
    world.write("inventory", sku, after, by_action="world.update_inventory")
    return {
        "sku_id": str(sku),
        "before": before.qty_available,
        "after": after.qty_available,
        "quantity": quantity,
    }


@ORACLE_PRIMITIVES.decorator(
    "example.flash-restock-delta",
    description="Verify that a restock changed only the declared quantity",
)
def verify(
    *,
    before: int | None = None,
    after: int | None = None,
    quantity: int | None = None,
    context: Any | None = None,
    event_id: str | None = None,
) -> bool:
    """Verify direct operands or consume one runtime event result by id."""
    if event_id is not None:
        if context is None:
            raise ValueError("context is required when event_id is supplied")
        event = context.world_events[event_id]
        if not isinstance(event, Mapping):
            raise ValueError(f"world event {event_id!r} did not return a mapping")
        before = int(event["before"])
        after = int(event["after"])
        quantity = int(event["quantity"])
    if before is None or after is None or quantity is None:
        raise ValueError("before, after, and quantity are required")
    return quantity > 0 and after - before == quantity
