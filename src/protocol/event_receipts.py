"""Strict persistent commerce protocol events and actor decision receipts.

This module defines only immutable protocol contracts and pure append/replay
validation.  It does not register actions, deliver messages, read order state,
persist rows, or execute an operation in World.  A later World and Platform
integration must construct bindings from authoritative order parties, issue
events at World logical time, persist both streams, and join processed receipts
to the exact World commit or previously persisted effect they reference.

The contract is task agnostic.  Event chains can represent payment,
fulfillment, refund, certificate, lifecycle, or other commerce callbacks
without embedding benchmark, scenario, or model-specific fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeAlias, cast

from protocol.errors import SchemaError


PROTOCOL_EVENT_BINDING_SCHEMA = "cwe.protocol-event-binding.v1"
PROTOCOL_EVENT_SCHEMA = "cwe.protocol-event.v1"
PROTOCOL_EVENT_RECEIPT_SCHEMA = "cwe.protocol-event-receipt.v1"
GENESIS_EVENT_DIGEST = "0" * 64

EventReferenceKind: TypeAlias = Literal["operation", "certificate"]
ReceiptDecision: TypeAlias = Literal["acknowledge", "reject", "process"]
RetryDisposition: TypeAlias = Literal["append", "idempotent"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_KINDS = frozenset({"operation", "certificate"})
_RECEIPT_DECISIONS = frozenset({"acknowledge", "reject", "process"})


class ProtocolEventContractError(SchemaError):
    """Base error for protocol event and receipt contracts."""


class ProtocolEventSchemaError(ProtocolEventContractError):
    """A binding, event, or receipt violates its exact schema."""


class ProtocolEventDigestMismatch(ProtocolEventSchemaError):
    """A persisted digest disagrees with canonical semantic content."""


class ProtocolEventBindingError(ProtocolEventContractError):
    """A record was applied to the wrong stream, order, or parties."""


class ProtocolEventAuthorityError(ProtocolEventContractError):
    """An untrusted actor or logical tick attempted to author a record."""


class ProtocolEventOrderingError(ProtocolEventContractError):
    """A sequence, predecessor, revision, or append tick regressed."""


class ProtocolEventStaleError(ProtocolEventContractError):
    """An actor tried to acknowledge or process a stale event."""


class ProtocolEventIdentityError(ProtocolEventContractError):
    """Two records do not share a stable or actor-idempotency identity."""


class ProtocolEventIdempotencyConflict(ProtocolEventContractError):
    """A stable identity or actor idempotency key has conflicting content."""


@dataclass(frozen=True, slots=True)
class ProtocolEventBinding:
    """Authoritative stream, order, party, recipient, and issuer identity."""

    market_id: str
    stream_id: str
    order_id: str
    buyer_id: str
    merchant_id: str
    recipient_id: str
    authority_id: str
    binding_digest: str = ""
    schema_id: str = PROTOCOL_EVENT_BINDING_SCHEMA


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    """One authority-issued, state-guarded event in a contiguous stream."""

    event_id: str
    binding: ProtocolEventBinding
    event_kind: str
    sequence: int
    required_order_state: str
    required_state_revision: int
    reference_kind: EventReferenceKind
    reference_digest: str
    issued_at_tick: int
    expires_at_tick: int
    actor_id: str
    idempotency_key: str
    predecessor_digest: str
    event_digest: str = ""
    schema_id: str = PROTOCOL_EVENT_SCHEMA


@dataclass(frozen=True, slots=True)
class ProtocolEventReceipt:
    """One recipient-authored decision bound to one exact event digest."""

    receipt_id: str
    binding: ProtocolEventBinding
    event_id: str
    event_digest: str
    decision: ReceiptDecision
    actor_id: str
    observed_order_state: str
    observed_state_revision: int
    reason: str
    effect_reference_digests: tuple[str, ...]
    logical_tick: int
    idempotency_key: str
    receipt_digest: str = ""
    schema_id: str = PROTOCOL_EVENT_RECEIPT_SCHEMA


def build_protocol_event_binding(
    *,
    market_id: str,
    stream_id: str,
    order_id: str,
    buyer_id: str,
    merchant_id: str,
    recipient_id: str,
    authority_id: str,
) -> ProtocolEventBinding:
    unsigned = ProtocolEventBinding(
        market_id=market_id,
        stream_id=stream_id,
        order_id=order_id,
        buyer_id=buyer_id,
        merchant_id=merchant_id,
        recipient_id=recipient_id,
        authority_id=authority_id,
    )
    _validate_binding_fields(unsigned, require_digest=False)
    sealed = _replace_binding_digest(unsigned, _digest(_binding_contract(unsigned)))
    validate_protocol_event_binding(sealed)
    return sealed


def validate_protocol_event_binding(binding: ProtocolEventBinding) -> None:
    if not isinstance(binding, ProtocolEventBinding):
        raise ProtocolEventSchemaError("binding must be ProtocolEventBinding")
    _validate_binding_fields(binding, require_digest=True)
    if binding.binding_digest != _digest(_binding_contract(binding)):
        raise ProtocolEventDigestMismatch("protocol event binding digest mismatch")


def protocol_event_binding_to_dict(binding: ProtocolEventBinding) -> dict[str, Any]:
    validate_protocol_event_binding(binding)
    return {**_binding_contract(binding), "binding_digest": binding.binding_digest}


def protocol_event_binding_to_json(binding: ProtocolEventBinding) -> str:
    return _canonical_json(protocol_event_binding_to_dict(binding))


def protocol_event_binding_from_json(payload: str) -> ProtocolEventBinding:
    value = _strict_canonical_json(payload, "protocol event binding")
    return _coerce_binding(value)


def build_protocol_event(
    binding: ProtocolEventBinding,
    *,
    event_id: str,
    event_kind: str,
    sequence: int,
    required_order_state: str,
    required_state_revision: int,
    reference_kind: EventReferenceKind,
    reference_digest: str,
    issued_at_tick: int,
    expires_at_tick: int,
    actor_id: str,
    idempotency_key: str,
    predecessor_digest: str,
) -> ProtocolEvent:
    validate_protocol_event_binding(binding)
    unsigned = ProtocolEvent(
        event_id=event_id,
        binding=binding,
        event_kind=event_kind,
        sequence=sequence,
        required_order_state=required_order_state,
        required_state_revision=required_state_revision,
        reference_kind=reference_kind,
        reference_digest=reference_digest,
        issued_at_tick=issued_at_tick,
        expires_at_tick=expires_at_tick,
        actor_id=actor_id,
        idempotency_key=idempotency_key,
        predecessor_digest=predecessor_digest,
    )
    _validate_event_fields(unsigned, require_digest=False)
    sealed = _replace_event_digest(unsigned, _digest(_event_contract(unsigned)))
    validate_protocol_event(sealed)
    return sealed


def build_next_protocol_event(
    events: Iterable[ProtocolEvent],
    binding: ProtocolEventBinding,
    *,
    event_id: str,
    event_kind: str,
    required_order_state: str,
    required_state_revision: int,
    reference_kind: EventReferenceKind,
    reference_digest: str,
    issued_at_tick: int,
    expires_at_tick: int,
    actor_id: str,
    idempotency_key: str,
) -> ProtocolEvent:
    records = tuple(events)
    validate_protocol_event_stream(records, binding=binding)
    return build_protocol_event(
        binding,
        event_id=event_id,
        event_kind=event_kind,
        sequence=len(records) + 1,
        required_order_state=required_order_state,
        required_state_revision=required_state_revision,
        reference_kind=reference_kind,
        reference_digest=reference_digest,
        issued_at_tick=issued_at_tick,
        expires_at_tick=expires_at_tick,
        actor_id=actor_id,
        idempotency_key=idempotency_key,
        predecessor_digest=(GENESIS_EVENT_DIGEST if not records else records[-1].event_digest),
    )


def validate_protocol_event(event: ProtocolEvent) -> None:
    if not isinstance(event, ProtocolEvent):
        raise ProtocolEventSchemaError("event must be ProtocolEvent")
    _validate_event_fields(event, require_digest=True)
    if event.event_digest != _digest(_event_contract(event)):
        raise ProtocolEventDigestMismatch("protocol event digest mismatch")


def protocol_event_to_dict(event: ProtocolEvent) -> dict[str, Any]:
    validate_protocol_event(event)
    return {**_event_contract(event), "event_digest": event.event_digest}


def protocol_event_to_json(event: ProtocolEvent) -> str:
    return _canonical_json(protocol_event_to_dict(event))


def protocol_event_from_json(payload: str) -> ProtocolEvent:
    value = _strict_canonical_json(payload, "protocol event")
    return _coerce_event(value)


def build_protocol_event_receipt(
    event: ProtocolEvent,
    *,
    receipt_id: str,
    decision: ReceiptDecision,
    actor_id: str,
    observed_order_state: str,
    observed_state_revision: int,
    reason: str,
    effect_reference_digests: Iterable[str],
    logical_tick: int,
    idempotency_key: str,
) -> ProtocolEventReceipt:
    validate_protocol_event(event)
    unsigned = ProtocolEventReceipt(
        receipt_id=receipt_id,
        binding=event.binding,
        event_id=event.event_id,
        event_digest=event.event_digest,
        decision=decision,
        actor_id=actor_id,
        observed_order_state=observed_order_state,
        observed_state_revision=observed_state_revision,
        reason=reason,
        effect_reference_digests=tuple(sorted(tuple(effect_reference_digests))),
        logical_tick=logical_tick,
        idempotency_key=idempotency_key,
    )
    _validate_receipt_fields(unsigned, require_digest=False)
    sealed = _replace_receipt_digest(unsigned, _digest(_receipt_contract(unsigned)))
    validate_protocol_event_receipt(sealed)
    return sealed


def validate_protocol_event_receipt(receipt: ProtocolEventReceipt) -> None:
    if not isinstance(receipt, ProtocolEventReceipt):
        raise ProtocolEventSchemaError("receipt must be ProtocolEventReceipt")
    _validate_receipt_fields(receipt, require_digest=True)
    if receipt.receipt_digest != _digest(_receipt_contract(receipt)):
        raise ProtocolEventDigestMismatch("protocol event receipt digest mismatch")


def protocol_event_receipt_to_dict(receipt: ProtocolEventReceipt) -> dict[str, Any]:
    validate_protocol_event_receipt(receipt)
    return {**_receipt_contract(receipt), "receipt_digest": receipt.receipt_digest}


def protocol_event_receipt_to_json(receipt: ProtocolEventReceipt) -> str:
    return _canonical_json(protocol_event_receipt_to_dict(receipt))


def protocol_event_receipt_from_json(payload: str) -> ProtocolEventReceipt:
    value = _strict_canonical_json(payload, "protocol event receipt")
    return _coerce_receipt(value)


def classify_protocol_event_retry(
    existing: ProtocolEvent,
    candidate: ProtocolEvent,
) -> RetryDisposition:
    validate_protocol_event(existing)
    validate_protocol_event(candidate)
    shared_scope = (
        existing.event_id == candidate.event_id
        or (
            existing.binding.binding_digest,
            existing.sequence,
        )
        == (
            candidate.binding.binding_digest,
            candidate.sequence,
        )
        or (
            existing.binding.binding_digest,
            existing.actor_id,
            existing.idempotency_key,
        )
        == (
            candidate.binding.binding_digest,
            candidate.actor_id,
            candidate.idempotency_key,
        )
    )
    if not shared_scope:
        raise ProtocolEventIdentityError(
            "events do not share id, stream sequence, or actor idempotency scope"
        )
    if existing.event_digest == candidate.event_digest:
        return "idempotent"
    raise ProtocolEventIdempotencyConflict(
        "event identity, sequence, or actor idempotency scope has conflicting content"
    )


def classify_protocol_receipt_retry(
    existing: ProtocolEventReceipt,
    candidate: ProtocolEventReceipt,
) -> RetryDisposition:
    validate_protocol_event_receipt(existing)
    validate_protocol_event_receipt(candidate)
    shared_scope = (
        existing.receipt_id == candidate.receipt_id
        or (
            existing.binding.binding_digest,
            existing.event_digest,
        )
        == (
            candidate.binding.binding_digest,
            candidate.event_digest,
        )
        or (
            existing.binding.binding_digest,
            existing.actor_id,
            existing.idempotency_key,
        )
        == (
            candidate.binding.binding_digest,
            candidate.actor_id,
            candidate.idempotency_key,
        )
    )
    if not shared_scope:
        raise ProtocolEventIdentityError(
            "receipts do not share id, event, or actor idempotency scope"
        )
    if existing.receipt_digest == candidate.receipt_digest:
        return "idempotent"
    raise ProtocolEventIdempotencyConflict(
        "receipt identity, event, or actor idempotency scope has conflicting content"
    )


def apply_protocol_event(
    events: Iterable[ProtocolEvent],
    candidate: ProtocolEvent,
    *,
    binding: ProtocolEventBinding,
    server_tick: int,
    current_order_state: str,
    current_state_revision: int,
) -> tuple[RetryDisposition, tuple[ProtocolEvent, ...]]:
    """Pure append validation for a future World protocol-event table."""

    records = tuple(events)
    validate_protocol_event_binding(binding)
    validate_protocol_event(candidate)
    _verify_record_binding(candidate.binding, binding)
    validate_protocol_event_stream(records, binding=binding)
    if candidate.actor_id != binding.authority_id:
        raise ProtocolEventAuthorityError("only the bound authority may issue events")
    for existing in records:
        try:
            disposition = classify_protocol_event_retry(existing, candidate)
        except ProtocolEventIdentityError:
            continue
        return disposition, records

    _require_nonnegative_int("server_tick", server_tick)
    _require_text("current_order_state", current_order_state)
    _require_nonnegative_int("current_state_revision", current_state_revision)
    if candidate.issued_at_tick != server_tick:
        raise ProtocolEventAuthorityError("event issued_at_tick is not trusted server tick")
    if (
        candidate.required_order_state != current_order_state
        or candidate.required_state_revision != current_state_revision
    ):
        raise ProtocolEventStaleError(
            "event required order state or revision disagrees with authoritative state"
        )
    expected_sequence = len(records) + 1
    expected_predecessor = GENESIS_EVENT_DIGEST if not records else records[-1].event_digest
    if candidate.sequence != expected_sequence:
        raise ProtocolEventOrderingError(
            f"event sequence must be contiguous at {expected_sequence}"
        )
    if candidate.predecessor_digest != expected_predecessor:
        raise ProtocolEventOrderingError("event predecessor digest mismatch")
    if records:
        previous = records[-1]
        if candidate.issued_at_tick < previous.issued_at_tick:
            raise ProtocolEventOrderingError("event logical issuance time regressed")
        if candidate.required_state_revision < previous.required_state_revision:
            raise ProtocolEventOrderingError("event required state revision regressed")
    return "append", (*records, candidate)


def validate_protocol_event_stream(
    events: Iterable[ProtocolEvent],
    *,
    binding: ProtocolEventBinding,
) -> tuple[ProtocolEvent, ...]:
    records = tuple(events)
    validate_protocol_event_binding(binding)
    event_ids: set[str] = set()
    actor_keys: set[tuple[str, str]] = set()
    previous: ProtocolEvent | None = None
    for expected_sequence, event in enumerate(records, start=1):
        validate_protocol_event(event)
        _verify_record_binding(event.binding, binding)
        if event.actor_id != binding.authority_id:
            raise ProtocolEventAuthorityError("persisted event has wrong authority actor")
        if event.sequence != expected_sequence:
            raise ProtocolEventOrderingError("persisted event sequence is not contiguous")
        expected_predecessor = GENESIS_EVENT_DIGEST if previous is None else previous.event_digest
        if event.predecessor_digest != expected_predecessor:
            raise ProtocolEventOrderingError("persisted event predecessor chain is broken")
        if previous is not None:
            if event.issued_at_tick < previous.issued_at_tick:
                raise ProtocolEventOrderingError("persisted event logical time regressed")
            if event.required_state_revision < previous.required_state_revision:
                raise ProtocolEventOrderingError("persisted required state revision regressed")
        if event.event_id in event_ids:
            raise ProtocolEventIdempotencyConflict("duplicate persisted event_id")
        actor_key = (event.actor_id, event.idempotency_key)
        if actor_key in actor_keys:
            raise ProtocolEventIdempotencyConflict("duplicate persisted actor idempotency key")
        event_ids.add(event.event_id)
        actor_keys.add(actor_key)
        previous = event
    return records


def replay_protocol_events(
    binding: ProtocolEventBinding,
    events: Iterable[ProtocolEvent],
) -> tuple[ProtocolEvent, ...]:
    replayed: tuple[ProtocolEvent, ...] = ()
    for event in events:
        _, replayed = apply_protocol_event(
            replayed,
            event,
            binding=binding,
            server_tick=event.issued_at_tick,
            current_order_state=event.required_order_state,
            current_state_revision=event.required_state_revision,
        )
    return replayed


def apply_protocol_receipt(
    receipts: Iterable[ProtocolEventReceipt],
    events: Iterable[ProtocolEvent],
    candidate: ProtocolEventReceipt,
    *,
    binding: ProtocolEventBinding,
    server_tick: int,
    current_order_state: str,
    current_state_revision: int,
) -> tuple[RetryDisposition, tuple[ProtocolEventReceipt, ...]]:
    """Pure append validation for one typed recipient decision.

    ``acknowledge`` and ``process`` are fail-closed when an event has expired or
    its required order state no longer matches.  ``reject`` remains recordable
    in those conditions so a recipient can durably report a stale or
    out-of-order delivery, but its observed state must still equal the trusted
    current state supplied by World.
    """

    event_records = validate_protocol_event_stream(events, binding=binding)
    receipt_records = validate_protocol_receipt_stream(
        receipts,
        events=event_records,
        binding=binding,
    )
    validate_protocol_event_receipt(candidate)
    _verify_record_binding(candidate.binding, binding)
    event = _event_for_receipt(event_records, candidate)
    if candidate.actor_id != binding.recipient_id:
        raise ProtocolEventAuthorityError("receipt actor must equal bound recipient")
    for existing in receipt_records:
        try:
            disposition = classify_protocol_receipt_retry(existing, candidate)
        except ProtocolEventIdentityError:
            continue
        return disposition, receipt_records

    _require_nonnegative_int("server_tick", server_tick)
    _require_text("current_order_state", current_order_state)
    _require_nonnegative_int("current_state_revision", current_state_revision)
    if candidate.logical_tick != server_tick:
        raise ProtocolEventAuthorityError("receipt logical_tick is not trusted server tick")
    if candidate.logical_tick < event.issued_at_tick:
        raise ProtocolEventOrderingError("receipt predates event issuance")
    if receipt_records and candidate.logical_tick < receipt_records[-1].logical_tick:
        raise ProtocolEventOrderingError("receipt append logical time regressed")
    if (
        candidate.observed_order_state != current_order_state
        or candidate.observed_state_revision != current_state_revision
    ):
        raise ProtocolEventAuthorityError(
            "receipt observed state is not the trusted current order state"
        )
    if candidate.decision != "reject":
        if candidate.logical_tick > event.expires_at_tick:
            raise ProtocolEventStaleError("cannot acknowledge or process an expired event")
        if (
            candidate.observed_order_state != event.required_order_state
            or candidate.observed_state_revision != event.required_state_revision
        ):
            raise ProtocolEventStaleError(
                "cannot acknowledge or process stale required state or revision"
            )
    return "append", (*receipt_records, candidate)


def validate_protocol_receipt_stream(
    receipts: Iterable[ProtocolEventReceipt],
    *,
    events: Iterable[ProtocolEvent],
    binding: ProtocolEventBinding,
) -> tuple[ProtocolEventReceipt, ...]:
    event_records = validate_protocol_event_stream(events, binding=binding)
    records = tuple(receipts)
    receipt_ids: set[str] = set()
    event_digests: set[str] = set()
    actor_keys: set[tuple[str, str]] = set()
    previous_tick: int | None = None
    for receipt in records:
        validate_protocol_event_receipt(receipt)
        _verify_record_binding(receipt.binding, binding)
        event = _event_for_receipt(event_records, receipt)
        if receipt.actor_id != binding.recipient_id:
            raise ProtocolEventAuthorityError("persisted receipt actor is not recipient")
        if receipt.logical_tick < event.issued_at_tick:
            raise ProtocolEventOrderingError("persisted receipt predates event")
        if previous_tick is not None and receipt.logical_tick < previous_tick:
            raise ProtocolEventOrderingError("persisted receipt logical time regressed")
        if receipt.decision != "reject":
            if receipt.logical_tick > event.expires_at_tick:
                raise ProtocolEventStaleError("persisted active receipt is expired")
            if (
                receipt.observed_order_state != event.required_order_state
                or receipt.observed_state_revision != event.required_state_revision
            ):
                raise ProtocolEventStaleError("persisted active receipt has stale state")
        actor_key = (receipt.actor_id, receipt.idempotency_key)
        if receipt.receipt_id in receipt_ids:
            raise ProtocolEventIdempotencyConflict("duplicate persisted receipt_id")
        if receipt.event_digest in event_digests:
            raise ProtocolEventIdempotencyConflict("event already has a receipt decision")
        if actor_key in actor_keys:
            raise ProtocolEventIdempotencyConflict(
                "duplicate persisted receipt actor idempotency key"
            )
        receipt_ids.add(receipt.receipt_id)
        event_digests.add(receipt.event_digest)
        actor_keys.add(actor_key)
        previous_tick = receipt.logical_tick
    return records


def replay_protocol_receipts(
    binding: ProtocolEventBinding,
    events: Iterable[ProtocolEvent],
    receipts: Iterable[ProtocolEventReceipt],
) -> tuple[ProtocolEventReceipt, ...]:
    """Structurally replay receipts using their persisted observed snapshots.

    A full World replay must additionally compare those snapshots with the
    authoritative order revision at each receipt tick.  This pure helper cannot
    manufacture that external state history.
    """

    event_records = validate_protocol_event_stream(events, binding=binding)
    replayed: tuple[ProtocolEventReceipt, ...] = ()
    for receipt in receipts:
        _, replayed = apply_protocol_receipt(
            replayed,
            event_records,
            receipt,
            binding=binding,
            server_tick=receipt.logical_tick,
            current_order_state=receipt.observed_order_state,
            current_state_revision=receipt.observed_state_revision,
        )
    return replayed


def _event_for_receipt(
    events: tuple[ProtocolEvent, ...],
    receipt: ProtocolEventReceipt,
) -> ProtocolEvent:
    by_id = [event for event in events if event.event_id == receipt.event_id]
    if len(by_id) != 1:
        raise ProtocolEventBindingError(
            "receipt event_id does not identify one persisted stream event"
        )
    event = by_id[0]
    if receipt.event_digest != event.event_digest:
        raise ProtocolEventBindingError("receipt event digest does not match persisted event")
    return event


def _verify_record_binding(
    actual: ProtocolEventBinding,
    expected: ProtocolEventBinding,
) -> None:
    validate_protocol_event_binding(actual)
    validate_protocol_event_binding(expected)
    if actual.binding_digest != expected.binding_digest or actual != expected:
        raise ProtocolEventBindingError(
            "record stream, order, party, recipient, or authority binding mismatch"
        )


def _validate_binding_fields(
    binding: ProtocolEventBinding,
    *,
    require_digest: bool,
) -> None:
    if binding.schema_id != PROTOCOL_EVENT_BINDING_SCHEMA:
        raise ProtocolEventSchemaError(
            f"unsupported protocol event binding schema: {binding.schema_id!r}"
        )
    for name in (
        "market_id",
        "stream_id",
        "order_id",
        "buyer_id",
        "merchant_id",
        "recipient_id",
        "authority_id",
    ):
        _require_text(name, getattr(binding, name))
    if binding.buyer_id == binding.merchant_id:
        raise ProtocolEventSchemaError("buyer and merchant must be distinct")
    if binding.recipient_id not in {binding.buyer_id, binding.merchant_id}:
        raise ProtocolEventSchemaError("recipient must be the bound buyer or merchant")
    if require_digest:
        _require_digest("binding_digest", binding.binding_digest)


def _validate_event_fields(event: ProtocolEvent, *, require_digest: bool) -> None:
    if event.schema_id != PROTOCOL_EVENT_SCHEMA:
        raise ProtocolEventSchemaError(f"unsupported protocol event schema: {event.schema_id!r}")
    validate_protocol_event_binding(event.binding)
    for name in (
        "event_id",
        "event_kind",
        "required_order_state",
        "actor_id",
        "idempotency_key",
    ):
        _require_text(name, getattr(event, name))
    _require_positive_int("sequence", event.sequence)
    _require_nonnegative_int("required_state_revision", event.required_state_revision)
    if event.reference_kind not in _REFERENCE_KINDS:
        raise ProtocolEventSchemaError(
            f"unsupported event reference_kind: {event.reference_kind!r}"
        )
    _require_digest("reference_digest", event.reference_digest)
    _require_nonnegative_int("issued_at_tick", event.issued_at_tick)
    _require_nonnegative_int("expires_at_tick", event.expires_at_tick)
    if event.expires_at_tick <= event.issued_at_tick:
        raise ProtocolEventSchemaError("event expiry must be after issuance")
    _require_digest("predecessor_digest", event.predecessor_digest)
    if event.sequence == 1:
        if event.predecessor_digest != GENESIS_EVENT_DIGEST:
            raise ProtocolEventSchemaError("first event must bind genesis predecessor")
    elif event.predecessor_digest == GENESIS_EVENT_DIGEST:
        raise ProtocolEventSchemaError("non-first event cannot bind genesis predecessor")
    if require_digest:
        _require_digest("event_digest", event.event_digest)


def _validate_receipt_fields(
    receipt: ProtocolEventReceipt,
    *,
    require_digest: bool,
) -> None:
    if receipt.schema_id != PROTOCOL_EVENT_RECEIPT_SCHEMA:
        raise ProtocolEventSchemaError(
            f"unsupported protocol event receipt schema: {receipt.schema_id!r}"
        )
    validate_protocol_event_binding(receipt.binding)
    for name in (
        "receipt_id",
        "event_id",
        "actor_id",
        "observed_order_state",
        "reason",
        "idempotency_key",
    ):
        _require_text(name, getattr(receipt, name))
    _require_digest("event_digest", receipt.event_digest)
    if receipt.decision not in _RECEIPT_DECISIONS:
        raise ProtocolEventSchemaError(f"unsupported receipt decision: {receipt.decision!r}")
    _require_nonnegative_int("observed_state_revision", receipt.observed_state_revision)
    if not isinstance(receipt.effect_reference_digests, tuple):
        raise ProtocolEventSchemaError("effect_reference_digests must be a tuple")
    for digest in receipt.effect_reference_digests:
        _require_digest("effect_reference_digest", digest)
    if receipt.effect_reference_digests != tuple(sorted(set(receipt.effect_reference_digests))):
        raise ProtocolEventSchemaError("effect_reference_digests must be unique and sorted")
    if receipt.decision == "process":
        if not receipt.effect_reference_digests:
            raise ProtocolEventSchemaError("processed receipt needs an effect reference")
    elif receipt.effect_reference_digests:
        raise ProtocolEventSchemaError(
            "acknowledge or reject receipt cannot claim effect references"
        )
    _require_nonnegative_int("logical_tick", receipt.logical_tick)
    if require_digest:
        _require_digest("receipt_digest", receipt.receipt_digest)


def _binding_contract(binding: ProtocolEventBinding) -> dict[str, Any]:
    return {
        "schema_id": binding.schema_id,
        "market_id": binding.market_id,
        "stream_id": binding.stream_id,
        "order_id": binding.order_id,
        "buyer_id": binding.buyer_id,
        "merchant_id": binding.merchant_id,
        "recipient_id": binding.recipient_id,
        "authority_id": binding.authority_id,
    }


def _event_contract(event: ProtocolEvent) -> dict[str, Any]:
    return {
        "schema_id": event.schema_id,
        "event_id": event.event_id,
        "binding": protocol_event_binding_to_dict(event.binding),
        "event_kind": event.event_kind,
        "sequence": event.sequence,
        "required_order_state": event.required_order_state,
        "required_state_revision": event.required_state_revision,
        "reference_kind": event.reference_kind,
        "reference_digest": event.reference_digest,
        "issued_at_tick": event.issued_at_tick,
        "expires_at_tick": event.expires_at_tick,
        "actor_id": event.actor_id,
        "idempotency_key": event.idempotency_key,
        "predecessor_digest": event.predecessor_digest,
    }


def _receipt_contract(receipt: ProtocolEventReceipt) -> dict[str, Any]:
    return {
        "schema_id": receipt.schema_id,
        "receipt_id": receipt.receipt_id,
        "binding": protocol_event_binding_to_dict(receipt.binding),
        "event_id": receipt.event_id,
        "event_digest": receipt.event_digest,
        "decision": receipt.decision,
        "actor_id": receipt.actor_id,
        "observed_order_state": receipt.observed_order_state,
        "observed_state_revision": receipt.observed_state_revision,
        "reason": receipt.reason,
        "effect_reference_digests": list(receipt.effect_reference_digests),
        "logical_tick": receipt.logical_tick,
        "idempotency_key": receipt.idempotency_key,
    }


def _replace_binding_digest(
    binding: ProtocolEventBinding,
    digest: str,
) -> ProtocolEventBinding:
    return ProtocolEventBinding(
        market_id=binding.market_id,
        stream_id=binding.stream_id,
        order_id=binding.order_id,
        buyer_id=binding.buyer_id,
        merchant_id=binding.merchant_id,
        recipient_id=binding.recipient_id,
        authority_id=binding.authority_id,
        binding_digest=digest,
        schema_id=binding.schema_id,
    )


def _replace_event_digest(event: ProtocolEvent, digest: str) -> ProtocolEvent:
    return ProtocolEvent(
        event_id=event.event_id,
        binding=event.binding,
        event_kind=event.event_kind,
        sequence=event.sequence,
        required_order_state=event.required_order_state,
        required_state_revision=event.required_state_revision,
        reference_kind=event.reference_kind,
        reference_digest=event.reference_digest,
        issued_at_tick=event.issued_at_tick,
        expires_at_tick=event.expires_at_tick,
        actor_id=event.actor_id,
        idempotency_key=event.idempotency_key,
        predecessor_digest=event.predecessor_digest,
        event_digest=digest,
        schema_id=event.schema_id,
    )


def _replace_receipt_digest(
    receipt: ProtocolEventReceipt,
    digest: str,
) -> ProtocolEventReceipt:
    return ProtocolEventReceipt(
        receipt_id=receipt.receipt_id,
        binding=receipt.binding,
        event_id=receipt.event_id,
        event_digest=receipt.event_digest,
        decision=receipt.decision,
        actor_id=receipt.actor_id,
        observed_order_state=receipt.observed_order_state,
        observed_state_revision=receipt.observed_state_revision,
        reason=receipt.reason,
        effect_reference_digests=receipt.effect_reference_digests,
        logical_tick=receipt.logical_tick,
        idempotency_key=receipt.idempotency_key,
        receipt_digest=digest,
        schema_id=receipt.schema_id,
    )


_BINDING_FIELDS = frozenset(
    {
        "schema_id",
        "market_id",
        "stream_id",
        "order_id",
        "buyer_id",
        "merchant_id",
        "recipient_id",
        "authority_id",
        "binding_digest",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "schema_id",
        "event_id",
        "binding",
        "event_kind",
        "sequence",
        "required_order_state",
        "required_state_revision",
        "reference_kind",
        "reference_digest",
        "issued_at_tick",
        "expires_at_tick",
        "actor_id",
        "idempotency_key",
        "predecessor_digest",
        "event_digest",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_id",
        "receipt_id",
        "binding",
        "event_id",
        "event_digest",
        "decision",
        "actor_id",
        "observed_order_state",
        "observed_state_revision",
        "reason",
        "effect_reference_digests",
        "logical_tick",
        "idempotency_key",
        "receipt_digest",
    }
)


def _coerce_binding(value: Any) -> ProtocolEventBinding:
    row = _exact_mapping(value, _BINDING_FIELDS, "protocol event binding")
    binding = ProtocolEventBinding(
        schema_id=_text(row, "schema_id"),
        market_id=_text(row, "market_id"),
        stream_id=_text(row, "stream_id"),
        order_id=_text(row, "order_id"),
        buyer_id=_text(row, "buyer_id"),
        merchant_id=_text(row, "merchant_id"),
        recipient_id=_text(row, "recipient_id"),
        authority_id=_text(row, "authority_id"),
        binding_digest=_text(row, "binding_digest"),
    )
    validate_protocol_event_binding(binding)
    return binding


def _coerce_event(value: Any) -> ProtocolEvent:
    row = _exact_mapping(value, _EVENT_FIELDS, "protocol event")
    event = ProtocolEvent(
        schema_id=_text(row, "schema_id"),
        event_id=_text(row, "event_id"),
        binding=_coerce_binding(row["binding"]),
        event_kind=_text(row, "event_kind"),
        sequence=_integer(row, "sequence"),
        required_order_state=_text(row, "required_order_state"),
        required_state_revision=_integer(row, "required_state_revision"),
        reference_kind=cast(EventReferenceKind, _text(row, "reference_kind")),
        reference_digest=_text(row, "reference_digest"),
        issued_at_tick=_integer(row, "issued_at_tick"),
        expires_at_tick=_integer(row, "expires_at_tick"),
        actor_id=_text(row, "actor_id"),
        idempotency_key=_text(row, "idempotency_key"),
        predecessor_digest=_text(row, "predecessor_digest"),
        event_digest=_text(row, "event_digest"),
    )
    validate_protocol_event(event)
    return event


def _coerce_receipt(value: Any) -> ProtocolEventReceipt:
    row = _exact_mapping(value, _RECEIPT_FIELDS, "protocol event receipt")
    effects = row["effect_reference_digests"]
    if not isinstance(effects, list):
        raise ProtocolEventSchemaError("effect_reference_digests must be an array")
    receipt = ProtocolEventReceipt(
        schema_id=_text(row, "schema_id"),
        receipt_id=_text(row, "receipt_id"),
        binding=_coerce_binding(row["binding"]),
        event_id=_text(row, "event_id"),
        event_digest=_text(row, "event_digest"),
        decision=cast(ReceiptDecision, _text(row, "decision")),
        actor_id=_text(row, "actor_id"),
        observed_order_state=_text(row, "observed_order_state"),
        observed_state_revision=_integer(row, "observed_state_revision"),
        reason=_text(row, "reason"),
        effect_reference_digests=tuple(_array_text(effects, "effect_reference_digests")),
        logical_tick=_integer(row, "logical_tick"),
        idempotency_key=_text(row, "idempotency_key"),
        receipt_digest=_text(row, "receipt_digest"),
    )
    validate_protocol_event_receipt(receipt)
    return receipt


def _strict_canonical_json(payload: str, label: str) -> Any:
    if not isinstance(payload, str):
        raise ProtocolEventSchemaError(f"{label} JSON must be a string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolEventSchemaError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise ProtocolEventSchemaError(f"non-finite JSON number: {value!r}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )
    except json.JSONDecodeError as exc:
        raise ProtocolEventSchemaError(f"invalid {label} JSON: {exc}") from exc
    if payload != _canonical_json(value):
        raise ProtocolEventSchemaError(f"{label} wire JSON is not canonical")
    return value


def _exact_mapping(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolEventSchemaError(f"{label} must be an object")
    actual = frozenset(value.keys())
    if any(not isinstance(key, str) for key in actual):
        raise ProtocolEventSchemaError(f"{label} has non-string fields")
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ProtocolEventSchemaError(
            f"{label} invalid fields: missing={missing!r}, unknown={unknown!r}"
        )
    return cast(Mapping[str, Any], value)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolEventSchemaError(f"value is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    _require_text(key, value)
    return cast(str, value)


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolEventSchemaError(f"{key} must be an integer")
    return cast(int, value)


def _array_text(values: list[Any], label: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        _require_text(label, value)
        result.append(cast(str, value))
    return tuple(result)


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolEventSchemaError(f"{name} must be a non-empty string")


def _require_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolEventSchemaError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolEventSchemaError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolEventSchemaError(f"{name} must be a non-negative integer")


__all__ = [
    "GENESIS_EVENT_DIGEST",
    "PROTOCOL_EVENT_BINDING_SCHEMA",
    "PROTOCOL_EVENT_RECEIPT_SCHEMA",
    "PROTOCOL_EVENT_SCHEMA",
    "EventReferenceKind",
    "ProtocolEvent",
    "ProtocolEventAuthorityError",
    "ProtocolEventBinding",
    "ProtocolEventBindingError",
    "ProtocolEventContractError",
    "ProtocolEventDigestMismatch",
    "ProtocolEventIdempotencyConflict",
    "ProtocolEventIdentityError",
    "ProtocolEventOrderingError",
    "ProtocolEventReceipt",
    "ProtocolEventSchemaError",
    "ProtocolEventStaleError",
    "ReceiptDecision",
    "RetryDisposition",
    "apply_protocol_event",
    "apply_protocol_receipt",
    "build_next_protocol_event",
    "build_protocol_event",
    "build_protocol_event_binding",
    "build_protocol_event_receipt",
    "classify_protocol_event_retry",
    "classify_protocol_receipt_retry",
    "protocol_event_binding_from_json",
    "protocol_event_binding_to_dict",
    "protocol_event_binding_to_json",
    "protocol_event_from_json",
    "protocol_event_receipt_from_json",
    "protocol_event_receipt_to_dict",
    "protocol_event_receipt_to_json",
    "protocol_event_to_dict",
    "protocol_event_to_json",
    "replay_protocol_events",
    "replay_protocol_receipts",
    "validate_protocol_event",
    "validate_protocol_event_binding",
    "validate_protocol_event_receipt",
    "validate_protocol_event_stream",
    "validate_protocol_receipt_stream",
]
