"""Pure deterministic oracle for S14 authoritative return windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from world.types import OrderState

AUTHORIZED = "authorized"
REJECTED_LATE = "rejected_late"
BRANCHES = frozenset({AUTHORIZED, REJECTED_LATE})

_FORGED_TIME_FIELDS = frozenset({
    "ts",
    "timestamp",
    "today",
    "current_time",
    "current_tick",
    "logical_time",
    "delivered_at",
    "settled_at_tick",
    "dispatched_at_tick",
    "return_authorized_at_tick",
    "return_window_ticks",
    "deadline_tick",
})


@dataclass(frozen=True)
class ReturnWindowScore:
    applicable: bool
    passed: bool | None
    expected_branch: str | None
    observed_branch: str | None
    order_id: str | None
    request_msg_id: str | None
    deadline_tick: int | None
    authorization_tick: int | None
    reasons: tuple[str, ...]


def score_return_window(
    *,
    expected: dict[str, Any],
    audit: Iterable[Any],
    final_snapshot: Any,
    initial_snapshot: Any | None = None,
) -> ReturnWindowScore:
    """Evaluate S14 exclusively from audited messages and World snapshots.

    No envelope timestamp or model-authored time field is a truth source.  The
    late branch additionally proves zero state effect when ``initial_snapshot``
    is supplied by the step scorer.
    """
    branch = expected.get("return_window_branch")
    if branch is None:
        return ReturnWindowScore(
            False, None, None, None, None, None, None, None,
            ("return-window oracle not requested",),
        )
    branch = str(branch)
    if branch not in BRANCHES:
        return ReturnWindowScore(
            True, False, branch, None, None, None, None, None,
            ("unknown return-window oracle branch",),
        )

    order_id = str(expected.get("expected_order_id") or "")
    timelines = tuple(getattr(final_snapshot, "order_timelines", ()) or ())
    if not order_id and len(timelines) == 1:
        order_id = str(timelines[0].order_id)
    timeline = next(
        (row for row in timelines if str(row.order_id) == order_id),
        None,
    )
    if timeline is None:
        return ReturnWindowScore(
            True, False, branch, None, order_id or None, None, None, None,
            ("authoritative order timeline is missing",),
        )

    dispatch_tick = timeline.dispatched_at_tick
    window = timeline.return_window_ticks
    deadline = (
        dispatch_tick + window
        if dispatch_tick is not None and window is not None
        else None
    )
    request, forged = _return_request(
        audit,
        order_id=order_id,
        buyer_id=str(timeline.buyer_id),
    )
    request_id = _message_id(request) if request is not None else None
    authorization = timeline.return_authorized_at_tick
    order = _order(final_snapshot, order_id)
    reasons: list[str] = []
    passed = True

    if request is None:
        passed = False
        reasons.append("exact owning buyer emitted no return request for the target order")
    if forged:
        passed = False
        reasons.append("return request attempted to supply a forged time field")
    if deadline is None:
        passed = False
        reasons.append("dispatch tick or explicit return window is missing")

    observed: str | None = None
    if authorization is not None:
        observed = AUTHORIZED
    elif deadline is not None and int(getattr(final_snapshot, "logical_time", 0)) + 1 > deadline:
        observed = REJECTED_LATE

    if branch == AUTHORIZED:
        if authorization is None or deadline is None or authorization > deadline:
            passed = False
            reasons.append("return was not authorized on or before the World deadline")
        if order is None or order.state != OrderState.REFUNDED:
            passed = False
            reasons.append("authorized branch did not atomically reach REFUNDED")
    else:
        current = int(getattr(final_snapshot, "logical_time", 0))
        if deadline is None or current + 1 <= deadline:
            passed = False
            reasons.append("rejection was not proven late at the next World event tick")
        if authorization is not None or timeline.refunded_at_tick is not None:
            passed = False
            reasons.append("late branch contains return authorization/refund timing evidence")
        if order is None or order.state == OrderState.REFUNDED:
            passed = False
            reasons.append("late branch incorrectly refunded the order")
        if initial_snapshot is not None and not _target_state_unchanged(
            initial_snapshot, final_snapshot, order_id=order_id, sku_id=str(order.sku_id)
        ):
            passed = False
            reasons.append("late rejection changed target order, inventory, ledger, timeline, or clock")

    if passed:
        reasons.append(
            "World timing and the expected return-window branch agree"
        )
    return ReturnWindowScore(
        applicable=True,
        passed=passed,
        expected_branch=branch,
        observed_branch=observed,
        order_id=order_id,
        request_msg_id=request_id,
        deadline_tick=deadline,
        authorization_tick=authorization,
        reasons=tuple(reasons),
    )


def _return_request(
    audit: Iterable[Any],
    *,
    order_id: str,
    buyer_id: str,
) -> tuple[Any | None, bool]:
    matching: list[Any] = []
    forged = False
    for record in audit:
        env = _envelope(record)
        if _kind(env) != "commerce.request_return":
            continue
        payload = _payload(env)
        if str(payload.get("order_id", "")) != order_id:
            continue
        if _from(env) != buyer_id:
            continue
        matching.append(env)
        forged = forged or bool(_FORGED_TIME_FIELDS.intersection(payload))
    return (matching[-1] if matching else None), forged


def _envelope(record: Any) -> Any:
    if isinstance(record, dict) and "envelope" in record:
        value = record["envelope"]
        return json.loads(value) if isinstance(value, str) else value
    return record


def _kind(env: Any) -> str:
    action = env.get("action", {}) if isinstance(env, dict) else getattr(env, "action", {})
    return str(action.get("kind", "")) if isinstance(action, dict) else ""


def _payload(env: Any) -> dict[str, Any]:
    action = env.get("action", {}) if isinstance(env, dict) else getattr(env, "action", {})
    payload = action.get("payload", {}) if isinstance(action, dict) else {}
    return payload if isinstance(payload, dict) else {}


def _from(env: Any) -> str:
    if isinstance(env, dict):
        return str(env.get("from", env.get("from_", "")))
    return str(getattr(env, "from_", ""))


def _message_id(env: Any) -> str | None:
    value = env.get("msg_id") if isinstance(env, dict) else getattr(env, "msg_id", None)
    return None if value is None else str(value)


def _order(snapshot: Any, order_id: str) -> Any | None:
    return next(
        (row for row in getattr(snapshot, "orders", ()) or () if str(row.order_id) == order_id),
        None,
    )


def _target_state_unchanged(
    initial: Any,
    final: Any,
    *,
    order_id: str,
    sku_id: str,
) -> bool:
    def receipts(snapshot: Any) -> tuple[Any, ...]:
        return tuple(
            row for row in getattr(snapshot, "ledger", ()) or ()
            if str(row.order_id) == order_id
        )

    def timeline(snapshot: Any) -> Any | None:
        return next(
            (
                row for row in getattr(snapshot, "order_timelines", ()) or ()
                if str(row.order_id) == order_id
            ),
            None,
        )

    initial_inventory = (getattr(initial, "inventory", {}) or {}).get(sku_id)
    final_inventory = (getattr(final, "inventory", {}) or {}).get(sku_id)
    return (
        _order(initial, order_id) == _order(final, order_id)
        and receipts(initial) == receipts(final)
        and timeline(initial) == timeline(final)
        and initial_inventory == final_inventory
        and int(getattr(initial, "logical_time", 0))
        == int(getattr(final, "logical_time", 0))
    )


__all__ = [
    "AUTHORIZED",
    "BRANCHES",
    "REJECTED_LATE",
    "ReturnWindowScore",
    "score_return_window",
]
