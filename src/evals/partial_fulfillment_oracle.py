"""Pure deterministic oracle for S22 full/partial/backorder allocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from world.types import OrderState

FULL = "full"
PARTIAL = "partial"
BACKORDERED = "backordered"
BRANCHES = frozenset({FULL, PARTIAL, BACKORDERED})


@dataclass(frozen=True)
class PartialFulfillmentScore:
    """One S22 verdict reconstructed only from wire and World truth."""

    applicable: bool
    passed: bool | None
    expected_branch: str | None
    observed_branch: str | None
    order_id: str | None
    request_msg_id: str | None
    requested_qty: int | None
    fulfilled_qty: int | None
    backordered_qty: int | None
    reasons: tuple[str, ...]


def score_partial_fulfillment(
    *,
    expected: dict[str, Any],
    audit: Iterable[Any],
    final_snapshot: Any,
    initial_snapshot: Any | None = None,
) -> PartialFulfillmentScore:
    """Evaluate S22 without trusting model-authored allocation quantities.

    The authoritative allocation, order, ledger, and inventory rows all come
    from World.  The request merely proves that the owning buyer opted into the
    partial-settlement path.  If an initial snapshot is supplied, the oracle
    also verifies the exact inventory reservation delta.
    """

    branch = expected.get("fulfillment_branch")
    if branch is None:
        return PartialFulfillmentScore(
            False, None, None, None, None, None, None, None, None,
            ("partial-fulfillment oracle not requested",),
        )
    branch = str(branch)
    if branch not in BRANCHES:
        return PartialFulfillmentScore(
            True, False, branch, None, None, None, None, None, None,
            ("unknown partial-fulfillment branch",),
        )

    order_id = str(expected.get("expected_order_id") or "")
    allocations = tuple(getattr(final_snapshot, "fulfillments", ()) or ())
    if not order_id and len(allocations) == 1:
        order_id = str(allocations[0].order_id)
    allocation = next(
        (row for row in allocations if str(row.order_id) == order_id),
        None,
    )
    reasons: list[str] = []
    passed = True
    if allocation is None:
        return PartialFulfillmentScore(
            True, False, branch, None, order_id or None, None, None, None, None,
            ("authoritative fulfillment allocation is missing",),
        )

    order = _order(final_snapshot, order_id)
    request = _partial_request(
        audit,
        order_id=order_id,
        buyer_id=str(allocation.buyer_id),
    )
    request_id = _message_id(request) if request is not None else None
    requested = int(allocation.requested_qty)
    fulfilled = int(allocation.fulfilled_qty)
    backordered = int(allocation.backordered_qty)

    if fulfilled == requested and backordered == 0:
        observed = FULL
        expected_state = OrderState.SETTLED
    elif fulfilled == 0 and backordered == requested:
        observed = BACKORDERED
        expected_state = OrderState.BACKORDERED
    elif 0 < fulfilled < requested and backordered == requested - fulfilled:
        observed = PARTIAL
        expected_state = OrderState.PARTIALLY_SETTLED
    else:
        observed = None
        expected_state = None
        passed = False
        reasons.append("allocation violates requested=fulfilled+backordered bounds")

    if request is None:
        passed = False
        reasons.append("owning buyer emitted no allow_partial settlement request")
    if observed != branch:
        passed = False
        reasons.append(
            f"observed fulfillment branch {observed!r} differs from expected {branch!r}"
        )
    for key, actual in (
        ("requested_qty", requested),
        ("fulfilled_qty", fulfilled),
        ("backordered_qty", backordered),
    ):
        declared = expected.get(key)
        if declared is not None and (
            isinstance(declared, bool) or not isinstance(declared, int) or declared != actual
        ):
            passed = False
            reasons.append(f"{key} does not match the deterministic answer key")

    if order is None:
        passed = False
        reasons.append("allocation has no authoritative order")
    else:
        identity_ok = (
            str(order.buyer_id) == str(allocation.buyer_id)
            and str(order.merchant_id) == str(allocation.merchant_id)
            and str(order.sku_id) == str(allocation.sku_id)
            and int(order.qty) == requested
        )
        if not identity_ok:
            passed = False
            reasons.append("order identity does not match its fulfillment allocation")
        if expected_state is not None and order.state != expected_state:
            passed = False
            reasons.append("order state does not match the observed allocation branch")

    receipts = [
        row for row in (getattr(final_snapshot, "ledger", ()) or ())
        if str(row.order_id) == order_id and not str(row.txn_id).startswith("refund")
    ]
    if fulfilled == 0:
        if receipts or allocation.receipt_txn_id is not None:
            passed = False
            reasons.append("zero-fill backorder created payment evidence")
    else:
        matching = [
            row for row in receipts
            if allocation.receipt_txn_id is not None
            and str(row.txn_id) == str(allocation.receipt_txn_id)
            and int(row.qty) == fulfilled
            and str(row.buyer_id) == str(allocation.buyer_id)
            and str(row.merchant_id) == str(allocation.merchant_id)
            and str(row.sku_id) == str(allocation.sku_id)
        ]
        if len(matching) != 1 or len(receipts) != 1:
            passed = False
            reasons.append("positive fill lacks one exact fulfilled-quantity receipt")

    if initial_snapshot is not None:
        initial = _inventory(initial_snapshot, str(allocation.sku_id))
        final = _inventory(final_snapshot, str(allocation.sku_id))
        if initial is None or final is None:
            passed = False
            reasons.append("inventory row is missing from an S22 snapshot")
        else:
            initial_available, initial_reserved = initial
            final_available, final_reserved = final
            if (
                initial_available - final_available != fulfilled
                or final_reserved - initial_reserved != fulfilled
            ):
                passed = False
                reasons.append("inventory reservation delta does not equal fulfilled quantity")

    if passed:
        reasons.append(
            "World allocation, order, ledger, and inventory agree with the expected branch"
        )
    return PartialFulfillmentScore(
        applicable=True,
        passed=passed,
        expected_branch=branch,
        observed_branch=observed,
        order_id=order_id,
        request_msg_id=request_id,
        requested_qty=requested,
        fulfilled_qty=fulfilled,
        backordered_qty=backordered,
        reasons=tuple(reasons),
    )


def _partial_request(
    audit: Iterable[Any],
    *,
    order_id: str,
    buyer_id: str,
) -> Any | None:
    for item in audit:
        action = _field(item, "action", {})
        payload = action.get("payload") if isinstance(action, dict) else None
        if (
            isinstance(payload, dict)
            and action.get("kind") == "platform.settle_payment"
            and _field(item, "from", _field(item, "from_", "")) == buyer_id
            and str(payload.get("order_id")) == order_id
            and payload.get("allow_partial") is True
        ):
            return item
    return None


def _order(snapshot: Any, order_id: str) -> Any | None:
    return next(
        (
            row for row in (getattr(snapshot, "orders", ()) or ())
            if str(row.order_id) == order_id
        ),
        None,
    )


def _inventory(snapshot: Any, sku_id: str) -> tuple[int, int] | None:
    for key, row in (getattr(snapshot, "inventory", {}) or {}).items():
        if str(key) != sku_id:
            continue
        available = getattr(row, "qty_available", None)
        if available is None:
            return int(row), 0
        reserved = int(getattr(row, "qty_reserved", 0) or 0)
        return int(available) - reserved, reserved
    return None


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        if name == "from_":
            return item.get("from_", item.get("from", default))
        return item.get(name, default)
    return getattr(item, name, default)


def _message_id(item: Any) -> str | None:
    value = _field(item, "msg_id")
    return str(value) if value not in (None, "") else None


__all__ = [
    "BACKORDERED",
    "BRANCHES",
    "FULL",
    "PARTIAL",
    "PartialFulfillmentScore",
    "score_partial_fulfillment",
]
