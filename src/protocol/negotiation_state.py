"""Replay-stable core contracts for bilateral commerce negotiation.

The current Platform mediation code can use these passive contracts as the
domain boundary for durable negotiation state.  This module itself does not
own a database, a router, or a clock.  Instead, callers supply a trusted
:class:`NegotiationBinding` and the current server logical tick.  Every event
duplicates and hashes the immutable commercial identity, and every successor
links to the previous event digest.

The state machine is task agnostic.  It models proposal, counteroffer,
acceptance, rejection, and withdrawal.  Private utility, benchmark lanes, prompts, and
agent reasoning are deliberately absent.  Persisted events and threads are
strict canonical JSON suitable for identical in-process, HTTP, and replay
paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeAlias, TypedDict, cast

from protocol.errors import SchemaError


NEGOTIATION_EVENT_SCHEMA_ID = "cwe.negotiation-event.v1"
NEGOTIATION_AGREEMENT_SCHEMA_ID = "cwe.negotiation-agreement.v1"
NEGOTIATION_THREAD_SCHEMA_ID = "cwe.negotiation-thread.v1"

PROPOSE_OFFER = "commerce.propose_offer"
COUNTER_OFFER = "commerce.counter_offer"
ACCEPT_OFFER = "commerce.accept_offer"
REJECT_OFFER = "commerce.reject_offer"
WITHDRAW_OFFER = "commerce.withdraw_offer"

NegotiationAction: TypeAlias = Literal[
    "commerce.propose_offer",
    "commerce.counter_offer",
    "commerce.accept_offer",
    "commerce.reject_offer",
    "commerce.withdraw_offer",
]
NegotiationStatus: TypeAlias = Literal["active", "accepted", "rejected", "withdrawn"]
NegotiationDisposition: TypeAlias = Literal["append", "idempotent"]

NEGOTIATION_ACTIONS = frozenset(
    {PROPOSE_OFFER, COUNTER_OFFER, ACCEPT_OFFER, REJECT_OFFER, WITHDRAW_OFFER}
)
_STATUSES = frozenset({"active", "accepted", "rejected", "withdrawn"})


def negotiated_order_id(negotiation_id: str, offer_id: str) -> str:
    """Return the single order identity authorized by a negotiation thread.

    A negotiated agreement does not carry a caller-selected order id.  The
    protocol therefore binds the required ``neg:<mandate>:<offer>`` namespace
    to the existing ``ord-<mandate>-<offer>`` namespace.  The authoritative
    agreement supplies ``offer_id`` so colons inside either component are not
    parsed heuristically.  PSP checks this identity before accepting an
    agreement as settlement authority, which prevents one accepted thread
    from authorizing several differently named orders.
    """

    if not isinstance(negotiation_id, str) or not negotiation_id.startswith("neg:"):
        raise NegotiationSchemaError(
            "negotiation_id must use the non-empty 'neg:' namespace"
        )
    if not isinstance(offer_id, str) or not offer_id:
        raise NegotiationSchemaError(
            "negotiated order identity requires a non-empty offer_id"
        )
    suffix = negotiation_id.removeprefix("neg:")
    offer_suffix = f":{offer_id}"
    if not suffix.endswith(offer_suffix):
        raise NegotiationSchemaError(
            "negotiation_id must end with its authoritative offer_id"
        )
    mandate_id = suffix[: -len(offer_suffix)]
    if not mandate_id:
        raise NegotiationSchemaError(
            "negotiation_id must bind a non-empty mandate identity"
        )
    return f"ord-{mandate_id}-{offer_id}"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")

_EVENT_FIELDS = frozenset(
    {
        "schema_id",
        "event_id",
        "negotiation_id",
        "action_kind",
        "buyer_id",
        "merchant_id",
        "offer_id",
        "sku_id",
        "listing_digest",
        "listing_revision",
        "currency",
        "qty",
        "unit_price",
        "round_no",
        "sequence_no",
        "max_rounds",
        "opened_at_tick",
        "expires_at_tick",
        "previous_digest",
        "actor_id",
        "counterparty_id",
        "idempotency_key",
        "server_tick",
        "event_digest",
    }
)
_AGREEMENT_FIELDS = frozenset(
    {
        "schema_id",
        "agreement_id",
        "negotiation_id",
        "buyer_id",
        "merchant_id",
        "offer_id",
        "sku_id",
        "listing_digest",
        "listing_revision",
        "currency",
        "qty",
        "unit_price",
        "round_no",
        "expires_at_tick",
        "accepted_by_id",
        "offered_by_id",
        "accepted_at_tick",
        "acceptance_event_id",
        "acceptance_idempotency_key",
        "acceptance_sequence_no",
        "previous_event_digest",
        "acceptance_event_digest",
        "agreement_digest",
    }
)
_THREAD_FIELDS = frozenset(
    {
        "schema_id",
        "negotiation_id",
        "buyer_id",
        "merchant_id",
        "offer_id",
        "sku_id",
        "listing_digest",
        "listing_revision",
        "currency",
        "qty",
        "current_unit_price",
        "round_no",
        "max_rounds",
        "opened_at_tick",
        "expires_at_tick",
        "status",
        "event_count",
        "last_actor_id",
        "last_counterparty_id",
        "last_event_id",
        "last_idempotency_key",
        "last_event_tick",
        "head_event_digest",
        "agreement",
        "thread_digest",
    }
)


class NegotiationStateError(SchemaError):
    """Base error for strict negotiation state contracts."""


class NegotiationSchemaError(NegotiationStateError):
    """A negotiation object is not an exact valid v1 wire object."""


class NegotiationDigestMismatch(NegotiationSchemaError):
    """A canonical negotiation digest does not match its content."""


class NegotiationBindingError(NegotiationStateError):
    """An event or agreement was replayed against another commercial binding."""


class NegotiationTransitionError(NegotiationStateError):
    """An otherwise valid event is illegal in the current state."""


class NegotiationExpired(NegotiationTransitionError):
    """A server-authored event tick is after the negotiation deadline."""


class NegotiationIdempotencyConflict(NegotiationTransitionError):
    """An event identity or actor-scoped idempotency key has conflicting bytes."""


class NegotiationEventWire(TypedDict):
    schema_id: str
    event_id: str
    negotiation_id: str
    action_kind: NegotiationAction
    buyer_id: str
    merchant_id: str
    offer_id: str
    sku_id: str
    listing_digest: str
    listing_revision: int
    currency: str
    qty: int
    unit_price: int
    round_no: int
    sequence_no: int
    max_rounds: int
    opened_at_tick: int
    expires_at_tick: int
    previous_digest: str | None
    actor_id: str
    counterparty_id: str
    idempotency_key: str
    server_tick: int
    event_digest: str


class NegotiationAgreementWire(TypedDict):
    schema_id: str
    agreement_id: str
    negotiation_id: str
    buyer_id: str
    merchant_id: str
    offer_id: str
    sku_id: str
    listing_digest: str
    listing_revision: int
    currency: str
    qty: int
    unit_price: int
    round_no: int
    expires_at_tick: int
    accepted_by_id: str
    offered_by_id: str
    accepted_at_tick: int
    acceptance_event_id: str
    acceptance_idempotency_key: str
    acceptance_sequence_no: int
    previous_event_digest: str
    acceptance_event_digest: str
    agreement_digest: str


class NegotiationThreadWire(TypedDict):
    schema_id: str
    negotiation_id: str
    buyer_id: str
    merchant_id: str
    offer_id: str
    sku_id: str
    listing_digest: str
    listing_revision: int
    currency: str
    qty: int
    current_unit_price: int
    round_no: int
    max_rounds: int
    opened_at_tick: int
    expires_at_tick: int
    status: NegotiationStatus
    event_count: int
    last_actor_id: str
    last_counterparty_id: str
    last_event_id: str
    last_idempotency_key: str
    last_event_tick: int
    head_event_digest: str
    agreement: NegotiationAgreementWire | None
    thread_digest: str


@dataclass(frozen=True, slots=True)
class NegotiationBinding:
    """Trusted Platform and World facts that define one negotiation thread."""

    negotiation_id: str
    buyer_id: str
    merchant_id: str
    offer_id: str
    sku_id: str
    listing_digest: str
    listing_revision: int
    currency: str
    qty: int
    max_rounds: int
    opened_at_tick: int
    expires_at_tick: int

    def __post_init__(self) -> None:
        _validate_binding(self)


@dataclass(frozen=True, slots=True)
class NegotiationEvent:
    """One append-only, server-timestamped negotiation event."""

    event_id: str
    negotiation_id: str
    action_kind: NegotiationAction
    buyer_id: str
    merchant_id: str
    offer_id: str
    sku_id: str
    listing_digest: str
    listing_revision: int
    currency: str
    qty: int
    unit_price: int
    round_no: int
    sequence_no: int
    max_rounds: int
    opened_at_tick: int
    expires_at_tick: int
    previous_digest: str | None
    actor_id: str
    counterparty_id: str
    idempotency_key: str
    server_tick: int
    event_digest: str = ""
    schema_id: str = NEGOTIATION_EVENT_SCHEMA_ID


@dataclass(frozen=True, slots=True)
class NegotiationAgreement:
    """Terminal agreement derived only from a validated acceptance event."""

    agreement_id: str
    negotiation_id: str
    buyer_id: str
    merchant_id: str
    offer_id: str
    sku_id: str
    listing_digest: str
    listing_revision: int
    currency: str
    qty: int
    unit_price: int
    round_no: int
    expires_at_tick: int
    accepted_by_id: str
    offered_by_id: str
    accepted_at_tick: int
    acceptance_event_id: str
    acceptance_idempotency_key: str
    acceptance_sequence_no: int
    previous_event_digest: str
    acceptance_event_digest: str
    agreement_digest: str = ""
    schema_id: str = NEGOTIATION_AGREEMENT_SCHEMA_ID


@dataclass(frozen=True, slots=True)
class NegotiationThread:
    """Immutable materialized state derived from an append-only event chain."""

    negotiation_id: str
    buyer_id: str
    merchant_id: str
    offer_id: str
    sku_id: str
    listing_digest: str
    listing_revision: int
    currency: str
    qty: int
    current_unit_price: int
    round_no: int
    max_rounds: int
    opened_at_tick: int
    expires_at_tick: int
    status: NegotiationStatus
    event_count: int
    last_actor_id: str
    last_counterparty_id: str
    last_event_id: str
    last_idempotency_key: str
    last_event_tick: int
    head_event_digest: str
    agreement: NegotiationAgreement | None
    thread_digest: str = ""
    schema_id: str = NEGOTIATION_THREAD_SCHEMA_ID


@dataclass(frozen=True, slots=True)
class NegotiationTransition:
    """Result of a pure state-machine application."""

    disposition: NegotiationDisposition
    thread: NegotiationThread
    event: NegotiationEvent


@dataclass(frozen=True, slots=True)
class NegotiationReplay:
    """A verified materialization and the unique append events that formed it."""

    thread: NegotiationThread | None
    events: tuple[NegotiationEvent, ...]


def build_negotiation_event(
    binding: NegotiationBinding,
    *,
    event_id: str,
    action_kind: NegotiationAction,
    actor_id: str,
    idempotency_key: str,
    unit_price: int,
    round_no: int,
    sequence_no: int,
    previous_digest: str | None,
    server_tick: int,
) -> NegotiationEvent:
    """Build and seal one event from trusted binding and server clock facts."""

    _validate_binding(binding)
    counterparty_id = _counterparty(binding, actor_id)
    return seal_negotiation_event(
        NegotiationEvent(
            event_id=event_id,
            negotiation_id=binding.negotiation_id,
            action_kind=action_kind,
            buyer_id=binding.buyer_id,
            merchant_id=binding.merchant_id,
            offer_id=binding.offer_id,
            sku_id=binding.sku_id,
            listing_digest=binding.listing_digest,
            listing_revision=binding.listing_revision,
            currency=binding.currency,
            qty=binding.qty,
            unit_price=unit_price,
            round_no=round_no,
            sequence_no=sequence_no,
            max_rounds=binding.max_rounds,
            opened_at_tick=binding.opened_at_tick,
            expires_at_tick=binding.expires_at_tick,
            previous_digest=previous_digest,
            actor_id=actor_id,
            counterparty_id=counterparty_id,
            idempotency_key=idempotency_key,
            server_tick=server_tick,
        )
    )


def build_next_negotiation_event(
    thread: NegotiationThread,
    *,
    event_id: str,
    action_kind: NegotiationAction,
    actor_id: str,
    idempotency_key: str,
    server_tick: int,
    unit_price: int | None = None,
    round_no: int | None = None,
) -> NegotiationEvent:
    """Build a successor using canonical sequence, lineage, and round defaults.

    This helper does not apply the event.  The caller must still pass it to
    :func:`apply_negotiation_event` with a trusted binding and server tick.
    Explicit ``unit_price`` and ``round_no`` values are retained so that a bad
    client claim reaches the state machine and is rejected rather than silently
    corrected.  Rejection and withdrawal may omit price; their canonical event
    binds the current offered price.
    """

    validate_negotiation_thread(thread)
    if round_no is None:
        round_no = thread.round_no + 1 if action_kind == COUNTER_OFFER else thread.round_no
    if unit_price is None:
        if action_kind not in {REJECT_OFFER, WITHDRAW_OFFER}:
            raise NegotiationSchemaError(
                "unit_price is required except for canonical rejection or withdrawal"
            )
        unit_price = thread.current_unit_price
    return build_negotiation_event(
        _binding_from_thread(thread),
        event_id=event_id,
        action_kind=action_kind,
        actor_id=actor_id,
        idempotency_key=idempotency_key,
        unit_price=unit_price,
        round_no=round_no,
        sequence_no=thread.event_count + 1,
        previous_digest=thread.head_event_digest,
        server_tick=server_tick,
    )


def seal_negotiation_event(event: NegotiationEvent) -> NegotiationEvent:
    """Return an event carrying its canonical SHA-256 digest."""

    if not isinstance(event, NegotiationEvent):
        raise NegotiationSchemaError("negotiation event must be NegotiationEvent")
    unsigned = _replace_event_digest(event, "")
    _validate_event_fields(unsigned)
    sealed = _replace_event_digest(unsigned, negotiation_event_digest(unsigned))
    validate_negotiation_event(sealed)
    return sealed


def negotiation_event_digest(event: NegotiationEvent) -> str:
    return _sha256_json(_event_contract(event))


def validate_negotiation_event(event: NegotiationEvent) -> None:
    if not isinstance(event, NegotiationEvent):
        raise NegotiationSchemaError("negotiation event must be NegotiationEvent")
    _validate_event_fields(event)
    _require_digest("event_digest", event.event_digest)
    if event.event_digest != negotiation_event_digest(event):
        raise NegotiationDigestMismatch("negotiation event digest mismatch")


def negotiation_event_to_dict(event: NegotiationEvent) -> NegotiationEventWire:
    validate_negotiation_event(event)
    return cast(
        NegotiationEventWire,
        {**_event_contract(event), "event_digest": event.event_digest},
    )


def negotiation_event_to_json(event: NegotiationEvent) -> str:
    return _canonical_json(negotiation_event_to_dict(event))


def coerce_negotiation_event(value: Any) -> NegotiationEvent:
    if isinstance(value, NegotiationEvent):
        validate_negotiation_event(value)
        return value
    row = _exact_mapping(value, _EVENT_FIELDS, "negotiation event")
    event = NegotiationEvent(
        schema_id=_text(row, "schema_id"),
        event_id=_text(row, "event_id"),
        negotiation_id=_text(row, "negotiation_id"),
        action_kind=cast(NegotiationAction, _text(row, "action_kind")),
        buyer_id=_text(row, "buyer_id"),
        merchant_id=_text(row, "merchant_id"),
        offer_id=_text(row, "offer_id"),
        sku_id=_text(row, "sku_id"),
        listing_digest=_text(row, "listing_digest"),
        listing_revision=_integer(row, "listing_revision"),
        currency=_text(row, "currency"),
        qty=_integer(row, "qty"),
        unit_price=_integer(row, "unit_price"),
        round_no=_integer(row, "round_no"),
        sequence_no=_integer(row, "sequence_no"),
        max_rounds=_integer(row, "max_rounds"),
        opened_at_tick=_integer(row, "opened_at_tick"),
        expires_at_tick=_integer(row, "expires_at_tick"),
        previous_digest=_optional_digest_value(row, "previous_digest"),
        actor_id=_text(row, "actor_id"),
        counterparty_id=_text(row, "counterparty_id"),
        idempotency_key=_text(row, "idempotency_key"),
        server_tick=_integer(row, "server_tick"),
        event_digest=_text(row, "event_digest"),
    )
    validate_negotiation_event(event)
    return event


def negotiation_event_from_json(payload: str) -> NegotiationEvent:
    return coerce_negotiation_event(_strict_json_loads(payload, "negotiation event"))


def negotiation_agreement_digest(agreement: NegotiationAgreement) -> str:
    return _sha256_json(_agreement_contract(agreement))


def validate_negotiation_agreement(agreement: NegotiationAgreement) -> None:
    if not isinstance(agreement, NegotiationAgreement):
        raise NegotiationSchemaError(
            "negotiation agreement must be NegotiationAgreement"
        )
    _validate_agreement_fields(agreement)
    _require_digest("agreement_digest", agreement.agreement_digest)
    if agreement.agreement_digest != negotiation_agreement_digest(agreement):
        raise NegotiationDigestMismatch("negotiation agreement digest mismatch")


def negotiation_agreement_to_dict(
    agreement: NegotiationAgreement,
) -> NegotiationAgreementWire:
    validate_negotiation_agreement(agreement)
    return cast(
        NegotiationAgreementWire,
        {**_agreement_contract(agreement), "agreement_digest": agreement.agreement_digest},
    )


def negotiation_agreement_to_json(agreement: NegotiationAgreement) -> str:
    return _canonical_json(negotiation_agreement_to_dict(agreement))


def coerce_negotiation_agreement(value: Any) -> NegotiationAgreement:
    if isinstance(value, NegotiationAgreement):
        validate_negotiation_agreement(value)
        return value
    row = _exact_mapping(value, _AGREEMENT_FIELDS, "negotiation agreement")
    agreement = NegotiationAgreement(
        schema_id=_text(row, "schema_id"),
        agreement_id=_text(row, "agreement_id"),
        negotiation_id=_text(row, "negotiation_id"),
        buyer_id=_text(row, "buyer_id"),
        merchant_id=_text(row, "merchant_id"),
        offer_id=_text(row, "offer_id"),
        sku_id=_text(row, "sku_id"),
        listing_digest=_text(row, "listing_digest"),
        listing_revision=_integer(row, "listing_revision"),
        currency=_text(row, "currency"),
        qty=_integer(row, "qty"),
        unit_price=_integer(row, "unit_price"),
        round_no=_integer(row, "round_no"),
        expires_at_tick=_integer(row, "expires_at_tick"),
        accepted_by_id=_text(row, "accepted_by_id"),
        offered_by_id=_text(row, "offered_by_id"),
        accepted_at_tick=_integer(row, "accepted_at_tick"),
        acceptance_event_id=_text(row, "acceptance_event_id"),
        acceptance_idempotency_key=_text(row, "acceptance_idempotency_key"),
        acceptance_sequence_no=_integer(row, "acceptance_sequence_no"),
        previous_event_digest=_text(row, "previous_event_digest"),
        acceptance_event_digest=_text(row, "acceptance_event_digest"),
        agreement_digest=_text(row, "agreement_digest"),
    )
    validate_negotiation_agreement(agreement)
    return agreement


def negotiation_agreement_from_json(payload: str) -> NegotiationAgreement:
    return coerce_negotiation_agreement(
        _strict_json_loads(payload, "negotiation agreement")
    )


def negotiation_thread_digest(thread: NegotiationThread) -> str:
    return _sha256_json(_thread_contract(thread))


def validate_negotiation_thread(thread: NegotiationThread) -> None:
    if not isinstance(thread, NegotiationThread):
        raise NegotiationSchemaError("negotiation thread must be NegotiationThread")
    _validate_thread_fields(thread)
    _require_digest("thread_digest", thread.thread_digest)
    if thread.thread_digest != negotiation_thread_digest(thread):
        raise NegotiationDigestMismatch("negotiation thread digest mismatch")


def negotiation_thread_to_dict(thread: NegotiationThread) -> NegotiationThreadWire:
    validate_negotiation_thread(thread)
    return cast(
        NegotiationThreadWire,
        {**_thread_contract(thread), "thread_digest": thread.thread_digest},
    )


def negotiation_thread_to_json(thread: NegotiationThread) -> str:
    return _canonical_json(negotiation_thread_to_dict(thread))


def coerce_negotiation_thread(value: Any) -> NegotiationThread:
    if isinstance(value, NegotiationThread):
        validate_negotiation_thread(value)
        return value
    row = _exact_mapping(value, _THREAD_FIELDS, "negotiation thread")
    raw_agreement = row["agreement"]
    if raw_agreement is not None and not isinstance(raw_agreement, Mapping):
        raise NegotiationSchemaError("agreement must be an object or null")
    thread = NegotiationThread(
        schema_id=_text(row, "schema_id"),
        negotiation_id=_text(row, "negotiation_id"),
        buyer_id=_text(row, "buyer_id"),
        merchant_id=_text(row, "merchant_id"),
        offer_id=_text(row, "offer_id"),
        sku_id=_text(row, "sku_id"),
        listing_digest=_text(row, "listing_digest"),
        listing_revision=_integer(row, "listing_revision"),
        currency=_text(row, "currency"),
        qty=_integer(row, "qty"),
        current_unit_price=_integer(row, "current_unit_price"),
        round_no=_integer(row, "round_no"),
        max_rounds=_integer(row, "max_rounds"),
        opened_at_tick=_integer(row, "opened_at_tick"),
        expires_at_tick=_integer(row, "expires_at_tick"),
        status=cast(NegotiationStatus, _text(row, "status")),
        event_count=_integer(row, "event_count"),
        last_actor_id=_text(row, "last_actor_id"),
        last_counterparty_id=_text(row, "last_counterparty_id"),
        last_event_id=_text(row, "last_event_id"),
        last_idempotency_key=_text(row, "last_idempotency_key"),
        last_event_tick=_integer(row, "last_event_tick"),
        head_event_digest=_text(row, "head_event_digest"),
        agreement=(
            coerce_negotiation_agreement(raw_agreement)
            if raw_agreement is not None
            else None
        ),
        thread_digest=_text(row, "thread_digest"),
    )
    validate_negotiation_thread(thread)
    return thread


def negotiation_thread_from_json(payload: str) -> NegotiationThread:
    return coerce_negotiation_thread(_strict_json_loads(payload, "negotiation thread"))


def verify_negotiation_event_binding(
    event: NegotiationEvent,
    binding: NegotiationBinding,
) -> None:
    """Reject events replayed across actor, offer, SKU, listing, or policy state."""

    validate_negotiation_event(event)
    _validate_binding(binding)
    checks = {
        "negotiation_id": (event.negotiation_id, binding.negotiation_id),
        "buyer_id": (event.buyer_id, binding.buyer_id),
        "merchant_id": (event.merchant_id, binding.merchant_id),
        "offer_id": (event.offer_id, binding.offer_id),
        "sku_id": (event.sku_id, binding.sku_id),
        "listing_digest": (event.listing_digest, binding.listing_digest),
        "listing_revision": (event.listing_revision, binding.listing_revision),
        "currency": (event.currency, binding.currency),
        "qty": (event.qty, binding.qty),
        "max_rounds": (event.max_rounds, binding.max_rounds),
        "opened_at_tick": (event.opened_at_tick, binding.opened_at_tick),
        "expires_at_tick": (event.expires_at_tick, binding.expires_at_tick),
    }
    mismatches = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if mismatches:
        raise NegotiationBindingError(
            "negotiation event binding mismatch: " + ", ".join(mismatches)
        )
    expected_counterparty = _counterparty(binding, event.actor_id)
    if event.counterparty_id != expected_counterparty:
        raise NegotiationBindingError("negotiation counterparty_id mismatch")


def classify_negotiation_retry(
    existing: NegotiationEvent,
    candidate: NegotiationEvent,
) -> NegotiationDisposition:
    """Classify byte-exact retry or fail on a reused identity/key.

    Persistence should call this after looking up either ``event_id`` or the
    actor-scoped key ``(negotiation_id, actor_id, idempotency_key)``.
    """

    validate_negotiation_event(existing)
    validate_negotiation_event(candidate)
    same_event_id = existing.event_id == candidate.event_id
    same_idempotency_scope = (
        existing.negotiation_id,
        existing.actor_id,
        existing.idempotency_key,
    ) == (
        candidate.negotiation_id,
        candidate.actor_id,
        candidate.idempotency_key,
    )
    if not same_event_id and not same_idempotency_scope:
        raise NegotiationBindingError(
            "existing event does not share event_id or actor idempotency scope"
        )
    if existing.event_digest == candidate.event_digest:
        return "idempotent"
    collision = "event_id" if same_event_id else "actor idempotency key"
    raise NegotiationIdempotencyConflict(
        f"conflicting negotiation event for reused {collision}"
    )


def apply_negotiation_event(
    thread: NegotiationThread | None,
    event: NegotiationEvent,
    *,
    binding: NegotiationBinding,
    server_tick: int,
    existing_event: NegotiationEvent | None = None,
) -> NegotiationTransition:
    """Purely apply one event to a thread using a trusted logical clock value."""

    verify_negotiation_event_binding(event, binding)
    _require_nonnegative_int("server_tick", server_tick)

    if existing_event is not None:
        disposition = classify_negotiation_retry(existing_event, event)
        if thread is None:
            raise NegotiationTransitionError(
                "idempotent retry requires its materialized negotiation thread"
            )
        validate_negotiation_thread(thread)
        _verify_thread_binding(thread, binding)
        return NegotiationTransition(disposition, thread, existing_event)

    if thread is not None:
        validate_negotiation_thread(thread)
        _verify_thread_binding(thread, binding)
        if event.event_digest == thread.head_event_digest:
            return NegotiationTransition("idempotent", thread, event)

    if event.server_tick != server_tick:
        raise NegotiationBindingError("event server_tick is not the trusted server tick")

    if thread is None:
        return _start_thread(binding, event, server_tick)

    if event.event_id == thread.last_event_id:
        raise NegotiationIdempotencyConflict(
            "conflicting negotiation event for reused event_id"
        )
    if (
        event.actor_id == thread.last_actor_id
        and event.idempotency_key == thread.last_idempotency_key
    ):
        raise NegotiationIdempotencyConflict(
            "conflicting negotiation event for reused actor idempotency key"
        )
    if thread.status != "active":
        raise NegotiationTransitionError("negotiation is terminal")
    if server_tick > thread.expires_at_tick:
        raise NegotiationExpired("negotiation deadline has expired")
    if event.server_tick < thread.last_event_tick:
        raise NegotiationTransitionError("server logical tick moved backwards")
    if event.sequence_no != thread.event_count + 1:
        raise NegotiationTransitionError(
            f"event sequence mismatch: expected {thread.event_count + 1}"
        )
    if event.previous_digest != thread.head_event_digest:
        raise NegotiationTransitionError("negotiation previous_digest mismatch")
    if event.action_kind == WITHDRAW_OFFER:
        if event.actor_id != thread.last_actor_id:
            raise NegotiationTransitionError("only the current offer actor may withdraw")
        if event.counterparty_id != thread.last_counterparty_id:
            raise NegotiationTransitionError("withdrawal counterparty mismatch")
    else:
        if event.actor_id == thread.last_actor_id:
            raise NegotiationTransitionError("same side cannot act consecutively")
        if event.counterparty_id != thread.last_actor_id:
            raise NegotiationTransitionError("event counterparty is not the last offer actor")
    if event.action_kind == PROPOSE_OFFER:
        raise NegotiationTransitionError("proposal is valid only as the first event")

    agreement: NegotiationAgreement | None = None
    if event.action_kind == COUNTER_OFFER:
        if event.round_no != thread.round_no + 1:
            raise NegotiationTransitionError("counteroffer round mismatch")
        if event.round_no > thread.max_rounds:
            raise NegotiationTransitionError("negotiation round limit exceeded")
        status: NegotiationStatus = "active"
        price = event.unit_price
    elif event.action_kind == ACCEPT_OFFER:
        if event.round_no != thread.round_no:
            raise NegotiationTransitionError("acceptance round mismatch")
        if event.unit_price != thread.current_unit_price:
            raise NegotiationTransitionError("acceptance price mismatch")
        status = "accepted"
        price = thread.current_unit_price
        agreement = _agreement_from_acceptance(thread, event)
    elif event.action_kind == REJECT_OFFER:
        if event.round_no != thread.round_no:
            raise NegotiationTransitionError("rejection round mismatch")
        if event.unit_price != thread.current_unit_price:
            raise NegotiationTransitionError("rejection price must bind current offer")
        status = "rejected"
        price = thread.current_unit_price
    elif event.action_kind == WITHDRAW_OFFER:
        if event.round_no != thread.round_no:
            raise NegotiationTransitionError("withdrawal round mismatch")
        if event.unit_price != thread.current_unit_price:
            raise NegotiationTransitionError("withdrawal price must bind current offer")
        status = "withdrawn"
        price = thread.current_unit_price
    else:  # pragma: no cover - strict event validation guards this branch
        raise NegotiationTransitionError("unsupported negotiation action")

    updated = _seal_thread(
        NegotiationThread(
            negotiation_id=thread.negotiation_id,
            buyer_id=thread.buyer_id,
            merchant_id=thread.merchant_id,
            offer_id=thread.offer_id,
            sku_id=thread.sku_id,
            listing_digest=thread.listing_digest,
            listing_revision=thread.listing_revision,
            currency=thread.currency,
            qty=thread.qty,
            current_unit_price=price,
            round_no=event.round_no,
            max_rounds=thread.max_rounds,
            opened_at_tick=thread.opened_at_tick,
            expires_at_tick=thread.expires_at_tick,
            status=status,
            event_count=event.sequence_no,
            last_actor_id=event.actor_id,
            last_counterparty_id=event.counterparty_id,
            last_event_id=event.event_id,
            last_idempotency_key=event.idempotency_key,
            last_event_tick=event.server_tick,
            head_event_digest=event.event_digest,
            agreement=agreement,
        )
    )
    return NegotiationTransition("append", updated, event)


def replay_negotiation_events(
    binding: NegotiationBinding,
    events: Iterable[NegotiationEvent],
) -> NegotiationReplay:
    """Replay an event stream, collapsing exact retries and rejecting conflicts."""

    _validate_binding(binding)
    thread: NegotiationThread | None = None
    appended: list[NegotiationEvent] = []
    by_event_id: dict[str, NegotiationEvent] = {}
    by_idempotency: dict[tuple[str, str, str], NegotiationEvent] = {}

    for event in events:
        validate_negotiation_event(event)
        identity = (event.negotiation_id, event.actor_id, event.idempotency_key)
        collisions = {
            row.event_digest: row
            for row in (
                by_event_id.get(event.event_id),
                by_idempotency.get(identity),
            )
            if row is not None
        }
        if collisions:
            if len(collisions) != 1:
                raise NegotiationIdempotencyConflict(
                    "event_id and actor idempotency key resolve to different events"
                )
            existing = next(iter(collisions.values()))
            transition = apply_negotiation_event(
                thread,
                event,
                binding=binding,
                server_tick=event.server_tick,
                existing_event=existing,
            )
            thread = transition.thread
            continue

        transition = apply_negotiation_event(
            thread,
            event,
            binding=binding,
            server_tick=event.server_tick,
        )
        thread = transition.thread
        appended.append(event)
        by_event_id[event.event_id] = event
        by_idempotency[identity] = event
    return NegotiationReplay(thread=thread, events=tuple(appended))


def _start_thread(
    binding: NegotiationBinding,
    event: NegotiationEvent,
    server_tick: int,
) -> NegotiationTransition:
    if event.action_kind != PROPOSE_OFFER:
        raise NegotiationTransitionError("negotiation must start with a proposal")
    if event.sequence_no != 1:
        raise NegotiationTransitionError("first negotiation event sequence must be 1")
    if event.round_no != 1:
        raise NegotiationTransitionError("proposal round must be 1")
    if event.previous_digest is not None:
        raise NegotiationTransitionError("proposal previous_digest must be null")
    if server_tick < binding.opened_at_tick:
        raise NegotiationTransitionError("proposal predates negotiation opening")
    if server_tick > binding.expires_at_tick:
        raise NegotiationExpired("negotiation deadline has expired")
    thread = _seal_thread(
        NegotiationThread(
            negotiation_id=binding.negotiation_id,
            buyer_id=binding.buyer_id,
            merchant_id=binding.merchant_id,
            offer_id=binding.offer_id,
            sku_id=binding.sku_id,
            listing_digest=binding.listing_digest,
            listing_revision=binding.listing_revision,
            currency=binding.currency,
            qty=binding.qty,
            current_unit_price=event.unit_price,
            round_no=1,
            max_rounds=binding.max_rounds,
            opened_at_tick=binding.opened_at_tick,
            expires_at_tick=binding.expires_at_tick,
            status="active",
            event_count=1,
            last_actor_id=event.actor_id,
            last_counterparty_id=event.counterparty_id,
            last_event_id=event.event_id,
            last_idempotency_key=event.idempotency_key,
            last_event_tick=event.server_tick,
            head_event_digest=event.event_digest,
            agreement=None,
        )
    )
    return NegotiationTransition("append", thread, event)


def _agreement_from_acceptance(
    thread: NegotiationThread,
    event: NegotiationEvent,
) -> NegotiationAgreement:
    unsigned = NegotiationAgreement(
        agreement_id=f"agreement:{event.event_digest[:32]}",
        negotiation_id=thread.negotiation_id,
        buyer_id=thread.buyer_id,
        merchant_id=thread.merchant_id,
        offer_id=thread.offer_id,
        sku_id=thread.sku_id,
        listing_digest=thread.listing_digest,
        listing_revision=thread.listing_revision,
        currency=thread.currency,
        qty=thread.qty,
        unit_price=thread.current_unit_price,
        round_no=thread.round_no,
        expires_at_tick=thread.expires_at_tick,
        accepted_by_id=event.actor_id,
        offered_by_id=thread.last_actor_id,
        accepted_at_tick=event.server_tick,
        acceptance_event_id=event.event_id,
        acceptance_idempotency_key=event.idempotency_key,
        acceptance_sequence_no=event.sequence_no,
        previous_event_digest=cast(str, event.previous_digest),
        acceptance_event_digest=event.event_digest,
    )
    _validate_agreement_fields(unsigned)
    sealed = _replace_agreement_digest(
        unsigned, negotiation_agreement_digest(unsigned)
    )
    validate_negotiation_agreement(sealed)
    return sealed


def _seal_thread(thread: NegotiationThread) -> NegotiationThread:
    unsigned = _replace_thread_digest(thread, "")
    _validate_thread_fields(unsigned)
    sealed = _replace_thread_digest(unsigned, negotiation_thread_digest(unsigned))
    validate_negotiation_thread(sealed)
    return sealed


def _validate_binding(binding: NegotiationBinding) -> None:
    if not isinstance(binding, NegotiationBinding):
        raise NegotiationBindingError("binding must be NegotiationBinding")
    for field in (
        "negotiation_id",
        "buyer_id",
        "merchant_id",
        "offer_id",
        "sku_id",
    ):
        _require_text(field, getattr(binding, field))
    if binding.buyer_id == binding.merchant_id:
        raise NegotiationBindingError("buyer and merchant must be distinct actors")
    if _role(binding.buyer_id) != "buyer" or _role(binding.merchant_id) != "merchant":
        raise NegotiationBindingError("binding requires buyer and merchant actor roles")
    _require_digest("listing_digest", binding.listing_digest)
    _require_nonnegative_int("listing_revision", binding.listing_revision)
    _require_currency(binding.currency)
    _require_positive_int("qty", binding.qty)
    _require_positive_int("max_rounds", binding.max_rounds)
    _require_nonnegative_int("opened_at_tick", binding.opened_at_tick)
    _require_nonnegative_int("expires_at_tick", binding.expires_at_tick)
    if binding.expires_at_tick <= binding.opened_at_tick:
        raise NegotiationBindingError("deadline must be after opening tick")


def _validate_event_fields(event: NegotiationEvent) -> None:
    if event.schema_id != NEGOTIATION_EVENT_SCHEMA_ID:
        raise NegotiationSchemaError(
            f"unsupported negotiation event schema_id: {event.schema_id!r}"
        )
    for field in (
        "event_id",
        "negotiation_id",
        "buyer_id",
        "merchant_id",
        "offer_id",
        "sku_id",
        "actor_id",
        "counterparty_id",
        "idempotency_key",
    ):
        _require_text(field, getattr(event, field))
    if event.action_kind not in NEGOTIATION_ACTIONS:
        raise NegotiationSchemaError(
            f"unsupported negotiation action: {event.action_kind!r}"
        )
    if event.buyer_id == event.merchant_id:
        raise NegotiationSchemaError("buyer and merchant must be distinct actors")
    if _role(event.buyer_id) != "buyer" or _role(event.merchant_id) != "merchant":
        raise NegotiationSchemaError("event requires buyer and merchant actor roles")
    if {event.actor_id, event.counterparty_id} != {
        event.buyer_id,
        event.merchant_id,
    } or event.actor_id == event.counterparty_id:
        raise NegotiationSchemaError("event actor/counterparty binding is invalid")
    _require_digest("listing_digest", event.listing_digest)
    _require_nonnegative_int("listing_revision", event.listing_revision)
    _require_currency(event.currency)
    _require_positive_int("qty", event.qty)
    _require_positive_int("unit_price", event.unit_price)
    _require_positive_int("round_no", event.round_no)
    _require_positive_int("sequence_no", event.sequence_no)
    _require_positive_int("max_rounds", event.max_rounds)
    if event.round_no > event.max_rounds:
        raise NegotiationSchemaError("round_no exceeds max_rounds")
    _require_nonnegative_int("opened_at_tick", event.opened_at_tick)
    _require_nonnegative_int("expires_at_tick", event.expires_at_tick)
    if event.expires_at_tick <= event.opened_at_tick:
        raise NegotiationSchemaError("event deadline must be after opening tick")
    _require_nonnegative_int("server_tick", event.server_tick)
    if event.sequence_no == 1:
        if event.previous_digest is not None:
            raise NegotiationSchemaError("first event previous_digest must be null")
    else:
        _require_digest("previous_digest", event.previous_digest)


def _validate_agreement_fields(agreement: NegotiationAgreement) -> None:
    if agreement.schema_id != NEGOTIATION_AGREEMENT_SCHEMA_ID:
        raise NegotiationSchemaError(
            f"unsupported negotiation agreement schema_id: {agreement.schema_id!r}"
        )
    for field in (
        "agreement_id",
        "negotiation_id",
        "buyer_id",
        "merchant_id",
        "offer_id",
        "sku_id",
        "accepted_by_id",
        "offered_by_id",
        "acceptance_event_id",
        "acceptance_idempotency_key",
    ):
        _require_text(field, getattr(agreement, field))
    if {agreement.accepted_by_id, agreement.offered_by_id} != {
        agreement.buyer_id,
        agreement.merchant_id,
    } or agreement.accepted_by_id == agreement.offered_by_id:
        raise NegotiationSchemaError("agreement actor binding is invalid")
    _require_digest("listing_digest", agreement.listing_digest)
    _require_nonnegative_int("listing_revision", agreement.listing_revision)
    _require_currency(agreement.currency)
    _require_positive_int("qty", agreement.qty)
    _require_positive_int("unit_price", agreement.unit_price)
    _require_positive_int("round_no", agreement.round_no)
    _require_nonnegative_int("expires_at_tick", agreement.expires_at_tick)
    _require_nonnegative_int("accepted_at_tick", agreement.accepted_at_tick)
    if agreement.accepted_at_tick > agreement.expires_at_tick:
        raise NegotiationSchemaError("agreement was accepted after deadline")
    _require_positive_int(
        "acceptance_sequence_no", agreement.acceptance_sequence_no
    )
    if agreement.acceptance_sequence_no < 2:
        raise NegotiationSchemaError(
            "agreement acceptance sequence must follow an offered event"
        )
    _require_digest("previous_event_digest", agreement.previous_event_digest)
    _require_digest("acceptance_event_digest", agreement.acceptance_event_digest)
    if agreement.agreement_id != f"agreement:{agreement.acceptance_event_digest[:32]}":
        raise NegotiationSchemaError("agreement_id does not bind acceptance event")


def _validate_thread_fields(thread: NegotiationThread) -> None:
    if thread.schema_id != NEGOTIATION_THREAD_SCHEMA_ID:
        raise NegotiationSchemaError(
            f"unsupported negotiation thread schema_id: {thread.schema_id!r}"
        )
    binding = _binding_from_thread(thread)
    _validate_binding(binding)
    _require_positive_int("current_unit_price", thread.current_unit_price)
    _require_positive_int("round_no", thread.round_no)
    if thread.round_no > thread.max_rounds:
        raise NegotiationSchemaError("thread round_no exceeds max_rounds")
    if thread.status not in _STATUSES:
        raise NegotiationSchemaError(f"unsupported negotiation status: {thread.status!r}")
    _require_positive_int("event_count", thread.event_count)
    expected_event_count = (
        thread.round_no if thread.status == "active" else thread.round_no + 1
    )
    if thread.event_count != expected_event_count:
        raise NegotiationSchemaError(
            "thread event_count is inconsistent with round and terminal status"
        )
    for field in (
        "last_actor_id",
        "last_counterparty_id",
        "last_event_id",
        "last_idempotency_key",
    ):
        _require_text(field, getattr(thread, field))
    if {thread.last_actor_id, thread.last_counterparty_id} != {
        thread.buyer_id,
        thread.merchant_id,
    } or thread.last_actor_id == thread.last_counterparty_id:
        raise NegotiationSchemaError("thread last actor binding is invalid")
    _require_nonnegative_int("last_event_tick", thread.last_event_tick)
    if not thread.opened_at_tick <= thread.last_event_tick <= thread.expires_at_tick:
        raise NegotiationSchemaError("thread last event tick is outside its deadline")
    _require_digest("head_event_digest", thread.head_event_digest)
    if thread.status == "accepted":
        if thread.agreement is None:
            raise NegotiationSchemaError("accepted thread must carry an agreement")
        validate_negotiation_agreement(thread.agreement)
        checks = {
            "negotiation_id": (
                thread.agreement.negotiation_id,
                thread.negotiation_id,
            ),
            "buyer_id": (thread.agreement.buyer_id, thread.buyer_id),
            "merchant_id": (thread.agreement.merchant_id, thread.merchant_id),
            "offer_id": (thread.agreement.offer_id, thread.offer_id),
            "sku_id": (thread.agreement.sku_id, thread.sku_id),
            "listing_digest": (
                thread.agreement.listing_digest,
                thread.listing_digest,
            ),
            "listing_revision": (
                thread.agreement.listing_revision,
                thread.listing_revision,
            ),
            "currency": (thread.agreement.currency, thread.currency),
            "qty": (thread.agreement.qty, thread.qty),
            "unit_price": (
                thread.agreement.unit_price,
                thread.current_unit_price,
            ),
            "round_no": (thread.agreement.round_no, thread.round_no),
            "expires_at_tick": (
                thread.agreement.expires_at_tick,
                thread.expires_at_tick,
            ),
            "accepted_by_id": (
                thread.agreement.accepted_by_id,
                thread.last_actor_id,
            ),
            "offered_by_id": (
                thread.agreement.offered_by_id,
                thread.last_counterparty_id,
            ),
            "accepted_at_tick": (
                thread.agreement.accepted_at_tick,
                thread.last_event_tick,
            ),
            "acceptance_event_id": (
                thread.agreement.acceptance_event_id,
                thread.last_event_id,
            ),
            "acceptance_idempotency_key": (
                thread.agreement.acceptance_idempotency_key,
                thread.last_idempotency_key,
            ),
            "acceptance_sequence_no": (
                thread.agreement.acceptance_sequence_no,
                thread.event_count,
            ),
            "acceptance_event_digest": (
                thread.agreement.acceptance_event_digest,
                thread.head_event_digest,
            ),
        }
        mismatches = [name for name, pair in checks.items() if pair[0] != pair[1]]
        if mismatches:
            raise NegotiationSchemaError(
                "thread agreement mismatch: " + ", ".join(mismatches)
            )
    elif thread.agreement is not None:
        raise NegotiationSchemaError("only accepted thread may carry an agreement")


def _verify_thread_binding(
    thread: NegotiationThread,
    binding: NegotiationBinding,
) -> None:
    checks = {
        "negotiation_id": (thread.negotiation_id, binding.negotiation_id),
        "buyer_id": (thread.buyer_id, binding.buyer_id),
        "merchant_id": (thread.merchant_id, binding.merchant_id),
        "offer_id": (thread.offer_id, binding.offer_id),
        "sku_id": (thread.sku_id, binding.sku_id),
        "listing_digest": (thread.listing_digest, binding.listing_digest),
        "listing_revision": (thread.listing_revision, binding.listing_revision),
        "currency": (thread.currency, binding.currency),
        "qty": (thread.qty, binding.qty),
        "max_rounds": (thread.max_rounds, binding.max_rounds),
        "opened_at_tick": (thread.opened_at_tick, binding.opened_at_tick),
        "expires_at_tick": (thread.expires_at_tick, binding.expires_at_tick),
    }
    mismatches = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if mismatches:
        raise NegotiationBindingError(
            "negotiation thread binding mismatch: " + ", ".join(mismatches)
        )


def _event_contract(event: NegotiationEvent) -> dict[str, Any]:
    return {
        "schema_id": event.schema_id,
        "event_id": event.event_id,
        "negotiation_id": event.negotiation_id,
        "action_kind": event.action_kind,
        "buyer_id": event.buyer_id,
        "merchant_id": event.merchant_id,
        "offer_id": event.offer_id,
        "sku_id": event.sku_id,
        "listing_digest": event.listing_digest,
        "listing_revision": event.listing_revision,
        "currency": event.currency,
        "qty": event.qty,
        "unit_price": event.unit_price,
        "round_no": event.round_no,
        "sequence_no": event.sequence_no,
        "max_rounds": event.max_rounds,
        "opened_at_tick": event.opened_at_tick,
        "expires_at_tick": event.expires_at_tick,
        "previous_digest": event.previous_digest,
        "actor_id": event.actor_id,
        "counterparty_id": event.counterparty_id,
        "idempotency_key": event.idempotency_key,
        "server_tick": event.server_tick,
    }


def _agreement_contract(agreement: NegotiationAgreement) -> dict[str, Any]:
    return {
        "schema_id": agreement.schema_id,
        "agreement_id": agreement.agreement_id,
        "negotiation_id": agreement.negotiation_id,
        "buyer_id": agreement.buyer_id,
        "merchant_id": agreement.merchant_id,
        "offer_id": agreement.offer_id,
        "sku_id": agreement.sku_id,
        "listing_digest": agreement.listing_digest,
        "listing_revision": agreement.listing_revision,
        "currency": agreement.currency,
        "qty": agreement.qty,
        "unit_price": agreement.unit_price,
        "round_no": agreement.round_no,
        "expires_at_tick": agreement.expires_at_tick,
        "accepted_by_id": agreement.accepted_by_id,
        "offered_by_id": agreement.offered_by_id,
        "accepted_at_tick": agreement.accepted_at_tick,
        "acceptance_event_id": agreement.acceptance_event_id,
        "acceptance_idempotency_key": agreement.acceptance_idempotency_key,
        "acceptance_sequence_no": agreement.acceptance_sequence_no,
        "previous_event_digest": agreement.previous_event_digest,
        "acceptance_event_digest": agreement.acceptance_event_digest,
    }


def _thread_contract(thread: NegotiationThread) -> dict[str, Any]:
    return {
        "schema_id": thread.schema_id,
        "negotiation_id": thread.negotiation_id,
        "buyer_id": thread.buyer_id,
        "merchant_id": thread.merchant_id,
        "offer_id": thread.offer_id,
        "sku_id": thread.sku_id,
        "listing_digest": thread.listing_digest,
        "listing_revision": thread.listing_revision,
        "currency": thread.currency,
        "qty": thread.qty,
        "current_unit_price": thread.current_unit_price,
        "round_no": thread.round_no,
        "max_rounds": thread.max_rounds,
        "opened_at_tick": thread.opened_at_tick,
        "expires_at_tick": thread.expires_at_tick,
        "status": thread.status,
        "event_count": thread.event_count,
        "last_actor_id": thread.last_actor_id,
        "last_counterparty_id": thread.last_counterparty_id,
        "last_event_id": thread.last_event_id,
        "last_idempotency_key": thread.last_idempotency_key,
        "last_event_tick": thread.last_event_tick,
        "head_event_digest": thread.head_event_digest,
        "agreement": (
            negotiation_agreement_to_dict(thread.agreement)
            if thread.agreement is not None
            else None
        ),
    }


def _binding_from_thread(thread: NegotiationThread) -> NegotiationBinding:
    return NegotiationBinding(
        negotiation_id=thread.negotiation_id,
        buyer_id=thread.buyer_id,
        merchant_id=thread.merchant_id,
        offer_id=thread.offer_id,
        sku_id=thread.sku_id,
        listing_digest=thread.listing_digest,
        listing_revision=thread.listing_revision,
        currency=thread.currency,
        qty=thread.qty,
        max_rounds=thread.max_rounds,
        opened_at_tick=thread.opened_at_tick,
        expires_at_tick=thread.expires_at_tick,
    )


def _counterparty(binding: NegotiationBinding, actor_id: str) -> str:
    if actor_id == binding.buyer_id:
        return binding.merchant_id
    if actor_id == binding.merchant_id:
        return binding.buyer_id
    raise NegotiationBindingError("actor is not a negotiation participant")


def _replace_event_digest(event: NegotiationEvent, digest: str) -> NegotiationEvent:
    return NegotiationEvent(
        schema_id=event.schema_id,
        event_id=event.event_id,
        negotiation_id=event.negotiation_id,
        action_kind=event.action_kind,
        buyer_id=event.buyer_id,
        merchant_id=event.merchant_id,
        offer_id=event.offer_id,
        sku_id=event.sku_id,
        listing_digest=event.listing_digest,
        listing_revision=event.listing_revision,
        currency=event.currency,
        qty=event.qty,
        unit_price=event.unit_price,
        round_no=event.round_no,
        sequence_no=event.sequence_no,
        max_rounds=event.max_rounds,
        opened_at_tick=event.opened_at_tick,
        expires_at_tick=event.expires_at_tick,
        previous_digest=event.previous_digest,
        actor_id=event.actor_id,
        counterparty_id=event.counterparty_id,
        idempotency_key=event.idempotency_key,
        server_tick=event.server_tick,
        event_digest=digest,
    )


def _replace_agreement_digest(
    agreement: NegotiationAgreement,
    digest: str,
) -> NegotiationAgreement:
    return NegotiationAgreement(
        schema_id=agreement.schema_id,
        agreement_id=agreement.agreement_id,
        negotiation_id=agreement.negotiation_id,
        buyer_id=agreement.buyer_id,
        merchant_id=agreement.merchant_id,
        offer_id=agreement.offer_id,
        sku_id=agreement.sku_id,
        listing_digest=agreement.listing_digest,
        listing_revision=agreement.listing_revision,
        currency=agreement.currency,
        qty=agreement.qty,
        unit_price=agreement.unit_price,
        round_no=agreement.round_no,
        expires_at_tick=agreement.expires_at_tick,
        accepted_by_id=agreement.accepted_by_id,
        offered_by_id=agreement.offered_by_id,
        accepted_at_tick=agreement.accepted_at_tick,
        acceptance_event_id=agreement.acceptance_event_id,
        acceptance_idempotency_key=agreement.acceptance_idempotency_key,
        acceptance_sequence_no=agreement.acceptance_sequence_no,
        previous_event_digest=agreement.previous_event_digest,
        acceptance_event_digest=agreement.acceptance_event_digest,
        agreement_digest=digest,
    )


def _replace_thread_digest(thread: NegotiationThread, digest: str) -> NegotiationThread:
    return NegotiationThread(
        schema_id=thread.schema_id,
        negotiation_id=thread.negotiation_id,
        buyer_id=thread.buyer_id,
        merchant_id=thread.merchant_id,
        offer_id=thread.offer_id,
        sku_id=thread.sku_id,
        listing_digest=thread.listing_digest,
        listing_revision=thread.listing_revision,
        currency=thread.currency,
        qty=thread.qty,
        current_unit_price=thread.current_unit_price,
        round_no=thread.round_no,
        max_rounds=thread.max_rounds,
        opened_at_tick=thread.opened_at_tick,
        expires_at_tick=thread.expires_at_tick,
        status=thread.status,
        event_count=thread.event_count,
        last_actor_id=thread.last_actor_id,
        last_counterparty_id=thread.last_counterparty_id,
        last_event_id=thread.last_event_id,
        last_idempotency_key=thread.last_idempotency_key,
        last_event_tick=thread.last_event_tick,
        head_event_digest=thread.head_event_digest,
        agreement=thread.agreement,
        thread_digest=digest,
    )


def _exact_mapping(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NegotiationSchemaError(f"{label} must be an object")
    actual = frozenset(value.keys())
    if any(not isinstance(key, str) for key in actual):
        raise NegotiationSchemaError(f"{label} has non-string fields")
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        raise NegotiationSchemaError(
            f"{label} has invalid fields: {', '.join(details)}"
        )
    return cast(Mapping[str, Any], value)


def _strict_json_loads(payload: str, label: str) -> Any:
    if not isinstance(payload, str):
        raise NegotiationSchemaError(f"{label} JSON must be a string")

    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NegotiationSchemaError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise NegotiationSchemaError(
            f"{label} JSON contains non-finite number {value!r}"
        )

    try:
        return json.loads(
            payload,
            object_pairs_hook=without_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise NegotiationSchemaError(f"invalid {label} JSON: {exc}") from exc


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
        raise NegotiationSchemaError(f"value is not canonical JSON: {exc}") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    _require_text(key, value)
    return cast(str, value)


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise NegotiationSchemaError(f"{key} must be an integer")
    return cast(int, value)


def _optional_digest_value(row: Mapping[str, Any], key: str) -> str | None:
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise NegotiationSchemaError(f"{key} must be a digest or null")
    return value


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise NegotiationSchemaError(f"{name} must be a non-empty string")


def _require_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NegotiationSchemaError(f"{name} must be a lowercase SHA-256 digest")


def _require_currency(value: Any) -> None:
    if not isinstance(value, str) or _CURRENCY_RE.fullmatch(value) is None:
        raise NegotiationSchemaError("currency must be a three-letter uppercase code")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NegotiationSchemaError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NegotiationSchemaError(f"{name} must be a non-negative integer")


def _role(actor_id: str) -> str:
    return actor_id.split(":", 1)[0]


__all__ = [
    "ACCEPT_OFFER",
    "COUNTER_OFFER",
    "NEGOTIATION_ACTIONS",
    "NEGOTIATION_AGREEMENT_SCHEMA_ID",
    "NEGOTIATION_EVENT_SCHEMA_ID",
    "NEGOTIATION_THREAD_SCHEMA_ID",
    "PROPOSE_OFFER",
    "REJECT_OFFER",
    "WITHDRAW_OFFER",
    "NegotiationAction",
    "NegotiationAgreement",
    "NegotiationAgreementWire",
    "NegotiationBinding",
    "NegotiationBindingError",
    "NegotiationDigestMismatch",
    "NegotiationDisposition",
    "NegotiationEvent",
    "NegotiationEventWire",
    "NegotiationExpired",
    "NegotiationIdempotencyConflict",
    "NegotiationReplay",
    "NegotiationSchemaError",
    "NegotiationStateError",
    "NegotiationStatus",
    "NegotiationThread",
    "NegotiationThreadWire",
    "NegotiationTransition",
    "NegotiationTransitionError",
    "apply_negotiation_event",
    "build_negotiation_event",
    "build_next_negotiation_event",
    "classify_negotiation_retry",
    "coerce_negotiation_agreement",
    "coerce_negotiation_event",
    "coerce_negotiation_thread",
    "negotiation_agreement_digest",
    "negotiation_agreement_from_json",
    "negotiation_agreement_to_dict",
    "negotiation_agreement_to_json",
    "negotiated_order_id",
    "negotiation_event_digest",
    "negotiation_event_from_json",
    "negotiation_event_to_dict",
    "negotiation_event_to_json",
    "negotiation_thread_digest",
    "negotiation_thread_from_json",
    "negotiation_thread_to_dict",
    "negotiation_thread_to_json",
    "replay_negotiation_events",
    "seal_negotiation_event",
    "validate_negotiation_agreement",
    "validate_negotiation_event",
    "validate_negotiation_thread",
    "verify_negotiation_event_binding",
]
