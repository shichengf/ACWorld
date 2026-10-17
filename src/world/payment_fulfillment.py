"""World-authoritative payment and pre-dispatch fulfillment records.

CommerceWorld historically represented a captured payment only through a
ledger ``Receipt``.  That is insufficient for an authorized but uncaptured
order, and ``Order`` plus ``Shipment`` leave no durable PACKED state.  These
records fill those general environment gaps.  They are not benchmark fixtures.

Only trusted World services build transitions.  Agent-facing after-sales
intents never carry payment state, amount, parties, packing state, logical
time, versions, or causal digests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, Mapping, TypeAlias, cast

from world.errors import IdempotencyConflict, WorldError, WriteNotAuthorized
from world.types import Order, OrderState, Receipt


PAYMENT_STATE_SCHEMA = "cwe.world-payment-state.v2"
PACKING_RECORD_SCHEMA = "cwe.world-packing-record.v1"

PaymentState: TypeAlias = Literal[
    "authorized",
    "captured",
    "partially_refunded",
    "voided",
    "refunded",
]
PackingState: TypeAlias = Literal["created", "packed", "cancelled", "handed_off"]
PaymentDisposition: TypeAlias = Literal["append", "idempotent"]


class PaymentFulfillmentError(WorldError):
    """Payment or packing state violates the World authority contract."""


@dataclass(frozen=True, slots=True)
class PaymentStateRecord:
    """Immutable version in one order's World-owned payment lifecycle."""

    payment_id: str
    order_id: str
    owner_id: str
    merchant_id: str
    sku_id: str
    qty: int
    amount: int
    captured_amount: int
    refunded_amount: int
    currency: str
    state: PaymentState
    version: int
    previous_digest: str | None
    capture_receipt_digest: str | None
    ledger_receipt_digest: str | None
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = PAYMENT_STATE_SCHEMA


@dataclass(frozen=True, slots=True)
class PackingRecord:
    """Immutable version in the pre-dispatch fulfillment lifecycle."""

    packing_id: str
    order_id: str
    owner_id: str
    merchant_id: str
    sku_id: str
    packed_qty: int
    payment_digest: str
    state: PackingState
    version: int
    previous_digest: str | None
    actor_id: str
    logical_tick: int
    idempotency_key: str
    record_digest: str = ""
    schema_id: str = PACKING_RECORD_SCHEMA


def derive_payment_authorization(
    order: Order,
    *,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
    psp_actor_ids: tuple[str, ...] = ("platform:psp",),
) -> PaymentStateRecord:
    """Authorize the exact order amount without creating a ledger receipt."""

    _validate_order(order)
    _require_psp(original_actor, psp_actor_ids)
    if order.state not in {OrderState.PROPOSED, OrderState.ACCEPTED}:
        raise PaymentFulfillmentError(
            "payment authorization requires proposed or accepted order"
        )
    return _build_payment(
        order,
        state="authorized",
        version=1,
        previous_digest=None,
        capture_receipt_digest=None,
        ledger_receipt_digest=None,
        captured_amount=0,
        refunded_amount=0,
        actor_id=original_actor,
        logical_tick=server_tick,
        idempotency_key=idempotency_key,
    )


def derive_payment_capture(
    order: Order,
    receipt: Receipt,
    *,
    previous: PaymentStateRecord | None,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
    psp_actor_ids: tuple[str, ...] = ("platform:psp",),
) -> PaymentStateRecord:
    """Bind a captured state to the exact persisted settlement receipt."""

    _validate_order_receipt(order, receipt, expected_effect="charge")
    _require_psp(original_actor, psp_actor_ids)
    receipt_digest = authoritative_payment_receipt_digest(receipt)
    if previous is None:
        version = 1
        previous_digest = None
    else:
        validate_payment_state_record(previous)
        if previous.order_id != str(order.order_id):
            raise PaymentFulfillmentError("payment order changed")
        if previous.state != "authorized":
            raise PaymentFulfillmentError("only authorized payment may be captured")
        version = previous.version + 1
        previous_digest = previous.record_digest
    candidate = _build_payment(
        order,
        state="captured",
        version=version,
        previous_digest=previous_digest,
        capture_receipt_digest=receipt_digest,
        ledger_receipt_digest=receipt_digest,
        captured_amount=_minor_units(receipt.price.amount, receipt.qty),
        refunded_amount=0,
        actor_id=original_actor,
        logical_tick=server_tick,
        idempotency_key=idempotency_key,
    )
    validate_payment_transition(previous, candidate, server_tick=server_tick)
    return candidate


def derive_payment_resolution(
    order: Order,
    previous: PaymentStateRecord,
    *,
    outcome: Literal["voided", "refunded"],
    refund_receipt: Receipt | None,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
    psp_actor_ids: tuple[str, ...] = ("platform:psp",),
) -> PaymentStateRecord:
    """Derive void or refund from current payment state and trusted ledger."""

    _validate_order(order)
    validate_payment_state_record(previous)
    _require_psp(original_actor, psp_actor_ids)
    if previous.order_id != str(order.order_id):
        raise PaymentFulfillmentError("payment order changed")
    if outcome == "voided":
        if previous.state != "authorized" or refund_receipt is not None:
            raise PaymentFulfillmentError(
                "void requires authorized payment and no refund receipt"
            )
        capture_receipt_digest = None
        ledger_receipt_digest = None
        captured_amount = 0
        refunded_amount = 0
    else:
        if previous.state not in {"captured", "partially_refunded"} or refund_receipt is None:
            raise PaymentFulfillmentError(
                "refund requires captured or partially refunded payment and refund receipt"
            )
        _validate_order_receipt(
            order,
            refund_receipt,
            expected_effect="refund",
            partial_refund=True,
        )
        refund_delta = _minor_units(refund_receipt.price.amount, refund_receipt.qty)
        if refund_delta <= 0:
            raise PaymentFulfillmentError("refund amount must be positive")
        refunded_amount = previous.refunded_amount + refund_delta
        if refunded_amount > previous.captured_amount:
            raise PaymentFulfillmentError("refund exceeds remaining captured amount")
        capture_receipt_digest = previous.capture_receipt_digest
        ledger_receipt_digest = authoritative_payment_receipt_digest(refund_receipt)
        captured_amount = previous.captured_amount
        outcome = (
            "refunded"
            if refunded_amount == captured_amount
            else "partially_refunded"
        )
    candidate = _build_payment(
        order,
        state=outcome,
        version=previous.version + 1,
        previous_digest=previous.record_digest,
        capture_receipt_digest=capture_receipt_digest,
        ledger_receipt_digest=ledger_receipt_digest,
        captured_amount=captured_amount,
        refunded_amount=refunded_amount,
        actor_id=original_actor,
        logical_tick=server_tick,
        idempotency_key=idempotency_key,
    )
    validate_payment_transition(previous, candidate, server_tick=server_tick)
    return candidate


def normalize_payment_intent(value: Mapping[str, Any]) -> dict[str, str]:
    """Accept only compact trusted PSP commands, never state or money."""

    return _normalize_intent(
        value,
        allowed_ops=frozenset({"authorize", "capture", "void", "refund"}),
    )


def normalize_packing_intent(value: Mapping[str, Any]) -> dict[str, str]:
    """Accept only compact fulfillment commands, never stage or quantity."""

    return _normalize_intent(
        value,
        allowed_ops=frozenset({"create", "pack", "cancel", "hand_off"}),
    )


def derive_packing_transition(
    order: Order,
    payment: PaymentStateRecord,
    *,
    previous: PackingRecord | None,
    target_state: PackingState,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> PackingRecord:
    """Derive a pre-dispatch fulfillment version from trusted World rows."""

    _validate_order(order)
    validate_payment_state_record(payment)
    if payment.order_id != str(order.order_id):
        raise PaymentFulfillmentError("packing payment order mismatch")
    if original_actor not in {str(order.merchant_id), "platform:fulfillment"}:
        raise WriteNotAuthorized("only the order merchant may record packing")
    if previous is None:
        if target_state != "created":
            raise PaymentFulfillmentError("packing history must start at created")
        if payment.state not in {"captured", "partially_refunded"}:
            raise PaymentFulfillmentError("packing creation requires captured payment")
        version = 1
        previous_digest = None
    else:
        validate_packing_record(previous)
        if previous.order_id != str(order.order_id):
            raise PaymentFulfillmentError("packing order changed")
        version = previous.version + 1
        previous_digest = previous.record_digest
        if target_state in {"packed", "handed_off"} and payment.state not in {
            "captured",
            "partially_refunded",
        }:
            raise PaymentFulfillmentError(
                "packing progression requires captured payment"
            )
        if target_state == "cancelled" and payment.state not in {
            "voided",
            "refunded",
        }:
            raise PaymentFulfillmentError(
                "packing cancellation requires resolved payment"
            )
    unsigned = PackingRecord(
        packing_id=f"packing:{order.order_id}",
        order_id=str(order.order_id),
        owner_id=str(order.buyer_id),
        merchant_id=str(order.merchant_id),
        sku_id=str(order.sku_id),
        packed_qty=order.qty,
        payment_digest=payment.record_digest,
        state=target_state,
        version=version,
        previous_digest=previous_digest,
        actor_id=original_actor,
        logical_tick=_nonnegative_int(server_tick, "server_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    result = PackingRecord(
        **{
            **{
                name: getattr(unsigned, name)
                for name in unsigned.__dataclass_fields__
                if name != "record_digest"
            },
            "record_digest": _digest(_packing_contract(unsigned)),
        }
    )
    validate_packing_record(result)
    validate_packing_transition(previous, result, server_tick=server_tick)
    return result


def derive_dispatch_packing_sequence(
    order: Order,
    payment: PaymentStateRecord,
    *,
    previous: PackingRecord | None,
    original_actor: str,
    server_tick: int,
    idempotency_key: str,
) -> tuple[PackingRecord, ...]:
    """Derive every missing packing version needed for dispatch.

    Dispatch is one authoritative World transition, so an untouched order
    receives ``created``, ``packed``, and ``handed_off`` at one logical tick.
    Orders that were already advanced through the public packing interface
    receive only the missing suffix.  A cancelled packing history cannot be
    dispatched.

    Each version receives a stage-scoped idempotency key.  This preserves the
    append-only transition rule while keeping the whole sequence causally tied
    to the single dispatch request.
    """

    if previous is None:
        targets: tuple[PackingState, ...] = (
            "created",
            "packed",
            "handed_off",
        )
    elif previous.state == "created":
        targets = ("packed", "handed_off")
    elif previous.state == "packed":
        targets = ("handed_off",)
    elif previous.state == "handed_off":
        targets = ()
    else:
        raise PaymentFulfillmentError(
            "cancelled packing history cannot be dispatched"
        )

    result: list[PackingRecord] = []
    cursor = previous
    for target in targets:
        row = derive_packing_transition(
            order,
            payment,
            previous=cursor,
            target_state=target,
            original_actor=original_actor,
            server_tick=server_tick,
            idempotency_key=f"{idempotency_key}:{target}",
        )
        result.append(row)
        cursor = row
    return tuple(result)


def validate_payment_transition(
    previous: PaymentStateRecord | None,
    candidate: PaymentStateRecord,
    *,
    server_tick: int,
) -> PaymentDisposition:
    validate_payment_state_record(candidate)
    if candidate.logical_tick != server_tick:
        raise PaymentFulfillmentError("payment tick is not World time")
    if previous is None:
        if (
            candidate.version != 1
            or candidate.previous_digest is not None
            or candidate.state not in {"authorized", "captured"}
        ):
            raise PaymentFulfillmentError("payment history has invalid genesis")
        return "append"
    validate_payment_state_record(previous)
    if candidate.record_digest == previous.record_digest:
        return "idempotent"
    if (
        candidate.actor_id == previous.actor_id
        and candidate.idempotency_key == previous.idempotency_key
    ):
        raise IdempotencyConflict("conflicting payment idempotency retry")
    identity = (
        "payment_id",
        "order_id",
        "owner_id",
        "merchant_id",
        "sku_id",
        "qty",
        "amount",
        "currency",
    )
    if any(getattr(previous, name) != getattr(candidate, name) for name in identity):
        raise PaymentFulfillmentError("payment immutable identity changed")
    if (
        candidate.version != previous.version + 1
        or candidate.previous_digest != previous.record_digest
        or candidate.logical_tick < previous.logical_tick
    ):
        raise PaymentFulfillmentError("payment version lineage is invalid")
    allowed = {
        "authorized": frozenset({"captured", "voided"}),
        "captured": frozenset({"partially_refunded", "refunded"}),
        "partially_refunded": frozenset({"partially_refunded", "refunded"}),
        "voided": frozenset(),
        "refunded": frozenset(),
    }
    if candidate.state not in allowed[previous.state]:
        raise PaymentFulfillmentError("payment state transition is invalid")
    if previous.state == "authorized":
        if candidate.state == "captured" and (
            candidate.captured_amount != candidate.amount
            or candidate.refunded_amount != 0
        ):
            raise PaymentFulfillmentError("capture amount is not the authorized amount")
    elif previous.state in {"captured", "partially_refunded"}:
        if (
            candidate.captured_amount != previous.captured_amount
            or candidate.capture_receipt_digest
            != previous.capture_receipt_digest
            or candidate.refunded_amount <= previous.refunded_amount
            or candidate.ledger_receipt_digest == previous.ledger_receipt_digest
        ):
            raise PaymentFulfillmentError("refund progress is not monotonic")
    return "append"


def validate_payment_state_record(record: PaymentStateRecord) -> None:
    if not isinstance(record, PaymentStateRecord):
        raise PaymentFulfillmentError("payment row has wrong type")
    if record.schema_id != PAYMENT_STATE_SCHEMA:
        raise PaymentFulfillmentError("unsupported payment schema")
    for label in (
        "payment_id",
        "order_id",
        "owner_id",
        "merchant_id",
        "sku_id",
        "actor_id",
        "idempotency_key",
    ):
        _text(getattr(record, label), label)
    if record.owner_id == record.merchant_id:
        raise PaymentFulfillmentError("payment parties must be distinct")
    _positive_int(record.qty, "qty")
    _nonnegative_int(record.amount, "amount")
    _nonnegative_int(record.captured_amount, "captured_amount")
    _nonnegative_int(record.refunded_amount, "refunded_amount")
    if len(record.currency) != 3 or not record.currency.isupper():
        raise PaymentFulfillmentError("payment currency is invalid")
    if record.state not in {
        "authorized",
        "captured",
        "partially_refunded",
        "voided",
        "refunded",
    }:
        raise PaymentFulfillmentError("payment state is invalid")
    _positive_int(record.version, "version")
    _nonnegative_int(record.logical_tick, "logical_tick")
    if record.version == 1:
        if record.previous_digest is not None:
            raise PaymentFulfillmentError("payment genesis has predecessor")
    else:
        _digest_text(record.previous_digest, "previous_digest")
    if record.state in {"captured", "partially_refunded", "refunded"}:
        _digest_text(record.capture_receipt_digest, "capture_receipt_digest")
        _digest_text(record.ledger_receipt_digest, "ledger_receipt_digest")
    elif (
        record.capture_receipt_digest is not None
        or record.ledger_receipt_digest is not None
    ):
        raise PaymentFulfillmentError("uncaptured payment cannot bind ledger receipt")
    if (
        record.state == "captured"
        and record.capture_receipt_digest != record.ledger_receipt_digest
    ):
        raise PaymentFulfillmentError(
            "captured payment must bind the capture receipt as current"
        )
    _digest_text(record.record_digest, "record_digest")
    if record.record_digest != _digest(_payment_contract(record)):
        raise PaymentFulfillmentError("payment digest mismatch")
    if record.captured_amount > record.amount:
        raise PaymentFulfillmentError("captured amount exceeds authorized amount")
    if record.refunded_amount > record.captured_amount:
        raise PaymentFulfillmentError("refunded amount exceeds captured amount")
    if record.state in {"authorized", "voided"} and (
        record.captured_amount != 0 or record.refunded_amount != 0
    ):
        raise PaymentFulfillmentError("uncaptured payment has monetary progress")
    if record.state == "captured" and (
        record.captured_amount != record.amount or record.refunded_amount != 0
    ):
        raise PaymentFulfillmentError("captured payment amounts are inconsistent")
    if record.state == "partially_refunded" and not (
        0 < record.refunded_amount < record.captured_amount == record.amount
    ):
        raise PaymentFulfillmentError("partial refund amounts are inconsistent")
    if record.state == "refunded" and not (
        0 < record.refunded_amount == record.captured_amount == record.amount
    ):
        raise PaymentFulfillmentError("full refund amounts are inconsistent")


def validate_packing_record(record: PackingRecord) -> None:
    if not isinstance(record, PackingRecord):
        raise PaymentFulfillmentError("packing row has wrong type")
    if record.schema_id != PACKING_RECORD_SCHEMA:
        raise PaymentFulfillmentError("unsupported packing schema")
    for label in (
        "packing_id",
        "order_id",
        "owner_id",
        "merchant_id",
        "sku_id",
        "actor_id",
        "idempotency_key",
    ):
        _text(getattr(record, label), label)
    if record.owner_id == record.merchant_id:
        raise PaymentFulfillmentError("packing parties must be distinct")
    _positive_int(record.packed_qty, "packed_qty")
    _digest_text(record.payment_digest, "payment_digest")
    if record.state not in {"created", "packed", "cancelled", "handed_off"}:
        raise PaymentFulfillmentError("packing state is invalid")
    _positive_int(record.version, "version")
    if record.version == 1:
        if record.previous_digest is not None:
            raise PaymentFulfillmentError("packing genesis has predecessor")
    else:
        _digest_text(record.previous_digest, "previous_digest")
    _nonnegative_int(record.logical_tick, "logical_tick")
    _digest_text(record.record_digest, "record_digest")
    if record.record_digest != _digest(_packing_contract(record)):
        raise PaymentFulfillmentError("packing digest mismatch")


def validate_packing_transition(
    previous: PackingRecord | None,
    candidate: PackingRecord,
    *,
    server_tick: int,
) -> PaymentDisposition:
    validate_packing_record(candidate)
    if candidate.logical_tick != server_tick:
        raise PaymentFulfillmentError("packing tick is not World time")
    if previous is None:
        if (
            candidate.state != "created"
            or candidate.version != 1
            or candidate.previous_digest is not None
        ):
            raise PaymentFulfillmentError("packing history has invalid genesis")
        return "append"
    validate_packing_record(previous)
    if previous.record_digest == candidate.record_digest:
        return "idempotent"
    if (
        previous.actor_id == candidate.actor_id
        and previous.idempotency_key == candidate.idempotency_key
    ):
        raise IdempotencyConflict("conflicting packing retry")
    identity = (
        "packing_id",
        "order_id",
        "owner_id",
        "merchant_id",
        "sku_id",
        "packed_qty",
    )
    if any(getattr(previous, name) != getattr(candidate, name) for name in identity):
        raise PaymentFulfillmentError("packing immutable identity changed")
    if (
        candidate.version != previous.version + 1
        or candidate.previous_digest != previous.record_digest
        or candidate.logical_tick < previous.logical_tick
    ):
        raise PaymentFulfillmentError("packing version lineage is invalid")
    allowed = {
        "created": frozenset({"packed", "cancelled"}),
        "packed": frozenset({"cancelled", "handed_off"}),
        "cancelled": frozenset(),
        "handed_off": frozenset(),
    }
    if candidate.state not in allowed[previous.state]:
        raise PaymentFulfillmentError("packing state transition is invalid")
    return "append"


def authoritative_payment_receipt_digest(receipt: Receipt) -> str:
    if not isinstance(receipt, Receipt):
        raise PaymentFulfillmentError("ledger receipt has wrong type")
    return _digest(
        {
            "txn_id": str(receipt.txn_id),
            "ts": receipt.ts,
            "order_id": str(receipt.order_id),
            "buyer_id": str(receipt.buyer_id),
            "merchant_id": str(receipt.merchant_id),
            "sku_id": str(receipt.sku_id),
            "qty": receipt.qty,
            "price": {
                "amount": str(receipt.price.amount),
                "currency": receipt.price.currency,
            },
            "idempotency_key": receipt.idempotency_key,
            "effect": receipt.effect,
        }
    )


def payment_state_to_dict(record: PaymentStateRecord) -> dict[str, Any]:
    validate_payment_state_record(record)
    return {**_payment_contract(record), "record_digest": record.record_digest}


def payment_state_key(record: PaymentStateRecord) -> str:
    """Return the canonical first-class World key for one payment version."""

    validate_payment_state_record(record)
    return f"{record.payment_id}:v{record.version}"


def payment_state_from_dict(value: Mapping[str, Any]) -> PaymentStateRecord:
    expected = {
        "schema_id",
        "payment_id",
        "order_id",
        "owner_id",
        "merchant_id",
        "sku_id",
        "qty",
        "amount",
        "captured_amount",
        "refunded_amount",
        "currency",
        "state",
        "version",
        "previous_digest",
        "capture_receipt_digest",
        "ledger_receipt_digest",
        "actor_id",
        "logical_tick",
        "idempotency_key",
        "record_digest",
    }
    if set(value) != expected:
        raise PaymentFulfillmentError("payment payload fields are not exact")
    record = PaymentStateRecord(**cast(Any, dict(value)))
    validate_payment_state_record(record)
    return record


def packing_record_to_dict(record: PackingRecord) -> dict[str, Any]:
    validate_packing_record(record)
    return {**_packing_contract(record), "record_digest": record.record_digest}


def packing_record_key(record: PackingRecord) -> str:
    """Return the canonical first-class World key for one packing version."""

    validate_packing_record(record)
    return f"{record.packing_id}:v{record.version}"


def packing_record_from_dict(value: Mapping[str, Any]) -> PackingRecord:
    expected = {
        "schema_id",
        "packing_id",
        "order_id",
        "owner_id",
        "merchant_id",
        "sku_id",
        "packed_qty",
        "payment_digest",
        "state",
        "version",
        "previous_digest",
        "actor_id",
        "logical_tick",
        "idempotency_key",
        "record_digest",
    }
    if set(value) != expected:
        raise PaymentFulfillmentError("packing payload fields are not exact")
    record = PackingRecord(**cast(Any, dict(value)))
    validate_packing_record(record)
    return record


def _build_payment(
    order: Order,
    *,
    state: PaymentState,
    version: int,
    previous_digest: str | None,
    capture_receipt_digest: str | None,
    ledger_receipt_digest: str | None,
    captured_amount: int,
    refunded_amount: int,
    actor_id: str,
    logical_tick: int,
    idempotency_key: str,
) -> PaymentStateRecord:
    unsigned = PaymentStateRecord(
        payment_id=f"payment:{order.order_id}",
        order_id=str(order.order_id),
        owner_id=str(order.buyer_id),
        merchant_id=str(order.merchant_id),
        sku_id=str(order.sku_id),
        qty=order.qty,
        amount=_minor_units(order.agreed_price.amount, order.qty),
        captured_amount=captured_amount,
        refunded_amount=refunded_amount,
        currency=order.agreed_price.currency,
        state=state,
        version=version,
        previous_digest=previous_digest,
        capture_receipt_digest=capture_receipt_digest,
        ledger_receipt_digest=ledger_receipt_digest,
        actor_id=actor_id,
        logical_tick=_nonnegative_int(logical_tick, "logical_tick"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
    )
    result = PaymentStateRecord(
        **{
            **{
                name: getattr(unsigned, name)
                for name in unsigned.__dataclass_fields__
                if name != "record_digest"
            },
            "record_digest": _digest(_payment_contract(unsigned)),
        }
    )
    validate_payment_state_record(result)
    return result


def _payment_contract(record: PaymentStateRecord) -> dict[str, Any]:
    return {
        "schema_id": record.schema_id,
        "payment_id": record.payment_id,
        "order_id": record.order_id,
        "owner_id": record.owner_id,
        "merchant_id": record.merchant_id,
        "sku_id": record.sku_id,
        "qty": record.qty,
        "amount": record.amount,
        "captured_amount": record.captured_amount,
        "refunded_amount": record.refunded_amount,
        "currency": record.currency,
        "state": record.state,
        "version": record.version,
        "previous_digest": record.previous_digest,
        "capture_receipt_digest": record.capture_receipt_digest,
        "ledger_receipt_digest": record.ledger_receipt_digest,
        "actor_id": record.actor_id,
        "logical_tick": record.logical_tick,
        "idempotency_key": record.idempotency_key,
    }


def _packing_contract(record: PackingRecord) -> dict[str, Any]:
    return {
        "schema_id": record.schema_id,
        "packing_id": record.packing_id,
        "order_id": record.order_id,
        "owner_id": record.owner_id,
        "merchant_id": record.merchant_id,
        "sku_id": record.sku_id,
        "packed_qty": record.packed_qty,
        "payment_digest": record.payment_digest,
        "state": record.state,
        "version": record.version,
        "previous_digest": record.previous_digest,
        "actor_id": record.actor_id,
        "logical_tick": record.logical_tick,
        "idempotency_key": record.idempotency_key,
    }


def _validate_order(order: Order) -> None:
    if not isinstance(order, Order):
        raise PaymentFulfillmentError("order has wrong type")
    _text(str(order.order_id), "order_id")
    _text(str(order.buyer_id), "buyer_id")
    _text(str(order.merchant_id), "merchant_id")
    if str(order.buyer_id) == str(order.merchant_id):
        raise PaymentFulfillmentError("order parties must be distinct")
    _positive_int(order.qty, "order.qty")
    if order.agreed_price.amount < 0:
        raise PaymentFulfillmentError("order price cannot be negative")


def _validate_order_receipt(
    order: Order,
    receipt: Receipt,
    *,
    expected_effect: Literal["charge", "refund"],
    partial_refund: bool = False,
) -> None:
    _validate_order(order)
    if not isinstance(receipt, Receipt):
        raise PaymentFulfillmentError("receipt has wrong type")
    if (
        str(receipt.order_id) != str(order.order_id)
        or str(receipt.buyer_id) != str(order.buyer_id)
        or str(receipt.merchant_id) != str(order.merchant_id)
        or str(receipt.sku_id) != str(order.sku_id)
        or receipt.qty != order.qty
        or receipt.effect != expected_effect
        or receipt.price.currency != order.agreed_price.currency
        or (
            not partial_refund
            and receipt.price != order.agreed_price
        )
    ):
        raise PaymentFulfillmentError("receipt identity does not match order")


def _require_psp(actor: str, allowed: tuple[str, ...]) -> None:
    _text(actor, "original_actor")
    if actor not in allowed:
        raise WriteNotAuthorized("only trusted PSP may mutate payment state")


def _normalize_intent(
    value: Mapping[str, Any], *, allowed_ops: frozenset[str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"op", "order_id"}:
        raise PaymentFulfillmentError(
            "payment and fulfillment intents require exactly op and order_id"
        )
    op = value.get("op")
    order_id = value.get("order_id")
    if not isinstance(op, str) or op not in allowed_ops:
        raise PaymentFulfillmentError("unsupported compact lifecycle operation")
    return {"op": op, "order_id": _text(order_id, "order_id")}


def _minor_units(unit_price: Decimal, qty: int) -> int:
    value = (unit_price * qty * Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    if value < 0:
        raise PaymentFulfillmentError("payment amount cannot be negative")
    return int(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaymentFulfillmentError(f"{label} must be non-empty text")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PaymentFulfillmentError(f"{label} must be positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PaymentFulfillmentError(f"{label} must be non-negative integer")
    return value


def _digest_text(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise PaymentFulfillmentError(f"{label} must be lowercase sha256")
    return text


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "PACKING_RECORD_SCHEMA",
    "PAYMENT_STATE_SCHEMA",
    "PackingRecord",
    "PaymentFulfillmentError",
    "PaymentState",
    "PaymentStateRecord",
    "authoritative_payment_receipt_digest",
    "derive_packing_transition",
    "derive_dispatch_packing_sequence",
    "derive_payment_authorization",
    "derive_payment_capture",
    "derive_payment_resolution",
    "normalize_packing_intent",
    "normalize_payment_intent",
    "packing_record_key",
    "packing_record_from_dict",
    "packing_record_to_dict",
    "payment_state_from_dict",
    "payment_state_key",
    "payment_state_to_dict",
    "validate_packing_record",
    "validate_packing_transition",
    "validate_payment_state_record",
    "validate_payment_transition",
]
