"""Shared deterministic invariants for atomic settlement and refund paths."""

from __future__ import annotations

from typing import Any, Literal

from world.errors import OrderIdentityMismatch
from world.types import InventoryRow, Order, Receipt


_ORDER_IDENTITY_FIELDS = (
    "order_id",
    "buyer_id",
    "merchant_id",
    "sku_id",
    "qty",
    "agreed_price",
)


def validate_persisted_order_identity(
    persisted: Order,
    requested: Order,
) -> None:
    """Require a replay/update request to preserve immutable order identity.

    ``state`` is deliberately excluded: callers normally replay the original
    pre-settlement order after World has advanced the persisted state. A fresh
    idempotency key may still replay a completed transaction, but it must not
    use the completed ``order_id`` as an alias for different parties, SKU,
    quantity, or money.
    """

    mismatches = [
        field
        for field in _ORDER_IDENTITY_FIELDS
        if getattr(persisted, field) != getattr(requested, field)
    ]
    if mismatches:
        raise OrderIdentityMismatch(
            "persisted order identity mismatch: " + ", ".join(mismatches)
        )


def scope_idempotency_key(actor_id: str, raw_key: str) -> str:
    """Injectively bind a caller key to its owning commerce actor.

    Service actors such as ``platform:psp`` are shared by every buyer. Without
    this binding, two independent buyers choosing the same otherwise-valid key
    collide in the World's service-wide idempotency namespace. The actor length
    makes the representation unambiguous even when IDs or keys contain colons.
    """

    actor = str(actor_id)
    key = str(raw_key)
    return f"actor-scope-v1:{len(actor)}:{actor}:{key}"


def validate_transaction_identity(
    order: Order,
    receipt: Receipt,
    inventory: InventoryRow | int | None,
    *,
    expected_qty: int | None = None,
    expected_effect: Literal["charge", "refund"] | None = None,
) -> None:
    """Require one coherent buyer/merchant/SKU/quantity/price identity."""
    mismatches: list[str] = []
    fields = ("order_id", "buyer_id", "merchant_id", "sku_id")
    for field in fields:
        if getattr(receipt, field) != getattr(order, field):
            mismatches.append(f"receipt.{field}")
    paid_qty = order.qty if expected_qty is None else expected_qty
    if receipt.qty != paid_qty:
        mismatches.append("receipt.qty")
    if expected_effect == "refund":
        if (
            receipt.price.currency != order.agreed_price.currency
            or receipt.price.amount <= 0
        ):
            mismatches.append("receipt.price")
    elif receipt.price != order.agreed_price:
        mismatches.append("receipt.price")
    if receipt.effect not in {"charge", "refund"}:
        mismatches.append("receipt.effect")
    elif expected_effect is not None and receipt.effect != expected_effect:
        mismatches.append("receipt.effect")
    if isinstance(inventory, InventoryRow) and inventory.merchant_id != order.merchant_id:
        mismatches.append("inventory.merchant_id")
    if (
        isinstance(order.qty, bool)
        or order.qty <= 0
        or isinstance(paid_qty, bool)
        or paid_qty <= 0
    ):
        mismatches.append("order.qty")
    if mismatches:
        raise OrderIdentityMismatch(
            "transaction identity mismatch: " + ", ".join(sorted(mismatches))
        )


def validate_listing_owner(order: Order, listing: Any | None) -> None:
    """When a listing is available, require it to be owned by the order merchant."""
    if listing is None:
        return
    if getattr(listing, "merchant_id", None) != order.merchant_id:
        raise OrderIdentityMismatch("transaction identity mismatch: listing.merchant_id")


__all__ = [
    "validate_listing_owner",
    "validate_persisted_order_identity",
    "validate_transaction_identity",
    "scope_idempotency_key",
]
