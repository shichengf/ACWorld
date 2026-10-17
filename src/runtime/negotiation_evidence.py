"""Exact Runtime, Platform, and World evidence for commerce negotiation.

This contract is part of CommerceWorld's reusable evidence layer.  It is not a
benchmark fixture.  Any scenario may submit compact negotiation intents to the
Platform and ask this module to prove that the audited requests caused the
canonical, atomic World state that appears in the episode snapshot.

The verifier deliberately joins all four authority surfaces.  It binds the
authenticated actor request to the Platform decision and transparent relay,
then to the actor-scoped authority operation, the append-only event, the
materialized thread, the World logical clock, and the immutable listing.  A
valid digest in only one of those surfaces is therefore insufficient.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from protocol.negotiation_state import (
    ACCEPT_OFFER,
    COUNTER_OFFER,
    NEGOTIATION_ACTIONS,
    PROPOSE_OFFER,
    REJECT_OFFER,
    WITHDRAW_OFFER,
    NegotiationBinding,
    NegotiationEvent,
    NegotiationThread,
    coerce_negotiation_event,
    coerce_negotiation_thread,
    negotiation_event_to_dict,
    negotiation_thread_to_dict,
    replay_negotiation_events,
)
from protocol.negotiation_turn_projection import (
    coerce_negotiation_turn_projection,
)
from protocol.errors import SchemaError
from runtime.exact_join import (
    DEFAULT_OPERATION_EVIDENCE_REGISTRY,
    ExactJoinContext,
    ExactJoinError,
    LinkedPlatformExchange,
    OperationEvidenceContract,
)
from world.evidence_contracts import authority_operation_key
from world.negotiations import (
    negotiation_event_id,
    negotiation_intent_fingerprint,
    negotiation_listing_digest,
    negotiation_status_for_action,
    normalize_negotiation_intent,
)
from world.types import AgentId, Listing, Money, SkuId


NEGOTIATION_EVIDENCE_CONTRACT = "commerceworld.negotiation.v1"
_NEGOTIATION_ENDPOINT = "platform:negotiation"
_PRICED_ACTIONS = frozenset({PROPOSE_OFFER, COUNTER_OFFER, ACCEPT_OFFER})
_PRICE_OPTIONAL_ACTIONS = frozenset({REJECT_OFFER, WITHDRAW_OFFER})
_REQUIRED_COMMIT_INVARIANTS = (
    "platform-mediated",
    "participant-and-listing-owner-bound",
    "actor-scoped-idempotency",
    "world-clock-and-sequence-derived",
    "event-thread-agreement-atomic",
    "private-utility-excluded",
)


@dataclass(frozen=True, slots=True)
class VerifiedNegotiationRequest:
    """One accepted actor request joined to its exact relay and World event."""

    exchange: LinkedPlatformExchange
    relay: dict[str, Any]
    intent: dict[str, Any]
    request_fingerprint: str
    event_id: str


@dataclass(frozen=True, slots=True)
class VerifiedRejectedNegotiationRequest:
    """One exact Platform rejection proven to have no negotiation side effect."""

    exchange: LinkedPlatformExchange
    negotiation_id: str | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class VerifiedNegotiationOperation:
    """One unique World event and every exact retry that resolved to it."""

    event: NegotiationEvent
    thread_before: NegotiationThread | None
    thread_after: NegotiationThread
    authority_operation: dict[str, Any]
    commit: dict[str, Any] | None
    requests: tuple[VerifiedNegotiationRequest, ...]

    @property
    def exact_retry_count(self) -> int:
        """Number of accepted requests beyond the single authority effect."""

        return max(0, len(self.requests) - 1)


@dataclass(frozen=True, slots=True)
class VerifiedNegotiationEvidence:
    """Complete negotiation authority graph selected by one contract call."""

    operations: tuple[VerifiedNegotiationOperation, ...]
    rejected_requests: tuple[VerifiedRejectedNegotiationRequest, ...] = ()

    @property
    def requests(self) -> tuple[VerifiedNegotiationRequest, ...]:
        return tuple(
            request
            for operation in self.operations
            for request in operation.requests
        )

    @property
    def relays(self) -> tuple[dict[str, Any], ...]:
        return tuple(request.relay for request in self.requests)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_requests)


def verify_negotiation_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedNegotiationEvidence:
    """Prove an exact Platform-mediated negotiation authority graph.

    ``expected_negotiation_ids`` bounds the selected market episode.  The
    policy values are trusted scenario configuration and are included in every
    actor-intent fingerprint.  Existing rows are supported for persistent
    worlds, but they must be an unchanged canonical prefix of final state.
    """

    expected_ids, max_rounds, deadline_ticks = _verification_options(options)
    expected_set = set(expected_ids)

    initial_event_rows = _unique_rows(
        context.initial_tables, "negotiation_events", "event_id"
    )
    final_event_rows = _unique_rows(
        context.final_tables, "negotiation_events", "event_id"
    )
    initial_thread_rows = _unique_rows(
        context.initial_tables, "negotiation_threads", "negotiation_id"
    )
    final_thread_rows = _unique_rows(
        context.final_tables, "negotiation_threads", "negotiation_id"
    )

    if not set(initial_event_rows).issubset(final_event_rows) or any(
        final_event_rows[event_id] != row
        for event_id, row in initial_event_rows.items()
    ):
        raise ExactJoinError("pre-existing negotiation event changed or disappeared")
    if not set(initial_thread_rows).issubset(final_thread_rows) or any(
        final_thread_rows[negotiation_id] != row
        for negotiation_id, row in initial_thread_rows.items()
        if negotiation_id not in expected_set
    ):
        raise ExactJoinError("pre-existing negotiation thread disappeared")
    unexpected_new_events = {
        event_id
        for event_id, row in final_event_rows.items()
        if event_id not in initial_event_rows
        and row.get("negotiation_id") not in expected_set
    }
    if unexpected_new_events:
        raise ExactJoinError("an unrelated negotiation gained new event rows")
    unexpected_new_threads = {
        negotiation_id
        for negotiation_id in final_thread_rows
        if negotiation_id not in initial_thread_rows
        and negotiation_id not in expected_set
    }
    if unexpected_new_threads:
        raise ExactJoinError("an unrelated negotiation gained a new thread")
    all_typed_events = _canonical_events(final_event_rows)
    all_typed_threads = _canonical_threads(final_thread_rows)
    typed_events = {
        event_id: event
        for event_id, event in all_typed_events.items()
        if event.negotiation_id in expected_set
    }
    typed_threads = {
        negotiation_id: thread
        for negotiation_id, thread in all_typed_threads.items()
        if negotiation_id in expected_set
    }
    initial_typed_events = {
        event_id: event
        for event_id, event in all_typed_events.items()
        if event_id in initial_event_rows and event.negotiation_id in expected_set
    }
    initial_typed_threads = {
        negotiation_id: thread
        for negotiation_id, thread in _canonical_threads(initial_thread_rows).items()
        if negotiation_id in expected_set
    }

    requests_by_event, rejected_requests = _verify_platform_exchanges(
        context,
        expected_set=expected_set,
        max_rounds=max_rounds,
        deadline_ticks=deadline_ticks,
        events=typed_events,
    )
    relevant_rejections = [
        row
        for row in rejected_requests
        if row.negotiation_id in expected_set
        or (row.negotiation_id is None and len(expected_set) == 1)
    ]
    missing_threads = expected_set - set(typed_threads)
    rejected_ids = {
        row.negotiation_id
        for row in relevant_rejections
        if row.negotiation_id is not None
    }
    if missing_threads - rejected_ids and not (
        len(missing_threads) == 1
        and any(row.negotiation_id is None for row in relevant_rejections)
    ):
        raise ExactJoinError(
            "expected negotiation has neither persisted state nor a zero-effect rejection"
        )

    catalog_initial = _unique_rows(context.initial_tables, "catalog", "sku_id")
    catalog_final = _unique_rows(context.final_tables, "catalog", "sku_id")
    _verify_replay_and_listing_bindings(
        expected_ids=tuple(
            negotiation_id
            for negotiation_id in expected_ids
            if negotiation_id in typed_threads
        ),
        max_rounds=max_rounds,
        deadline_ticks=deadline_ticks,
        events=typed_events,
        threads=typed_threads,
        initial_events=initial_typed_events,
        initial_threads=initial_typed_threads,
        initial_catalog=catalog_initial,
        final_catalog=catalog_final,
        context=context,
    )
    selected_initial_event_ids = set(initial_typed_events)
    new_event_ids = set(typed_events) - selected_initial_event_ids
    if set(requests_by_event) - set(typed_events):
        raise ExactJoinError("accepted negotiation request has no final World event")
    if not new_event_ids.issubset(requests_by_event):
        raise ExactJoinError("new negotiation event has no accepted Platform request")

    initial_authority = _unique_rows(
        context.initial_tables, "authority_operations", "operation_key"
    )
    final_authority = _unique_rows(
        context.final_tables, "authority_operations", "operation_key"
    )
    authority_by_event = _verify_authority_rows(
        expected_set=expected_set,
        initial_rows=initial_authority,
        final_rows=final_authority,
        events=typed_events,
        requests_by_event=requests_by_event,
    )

    commits_by_event = _verify_negotiation_commits(
        context,
        new_event_ids=new_event_ids,
        events=typed_events,
        threads=typed_threads,
        event_rows=final_event_rows,
        authority_by_event=authority_by_event,
        requests_by_event=requests_by_event,
    )
    _verify_world_clock_chain(context)

    operations: list[VerifiedNegotiationOperation] = []
    for event in sorted(typed_events.values(), key=_event_sort_key):
        requests = tuple(requests_by_event.get(event.event_id, ()))
        if not requests:
            continue
        before, after = _thread_transition_for_event(typed_events, event)
        operations.append(
            VerifiedNegotiationOperation(
                event=event,
                thread_before=before,
                thread_after=after,
                authority_operation=authority_by_event[event.event_id],
                commit=commits_by_event.get(event.event_id),
                requests=requests,
            )
        )
    if not operations and not relevant_rejections:
        raise ExactJoinError("negotiation evidence contains no selected attempts")
    return VerifiedNegotiationEvidence(
        tuple(operations),
        tuple(relevant_rejections),
    )


def _verification_options(
    options: Mapping[str, Any],
) -> tuple[tuple[str, ...], int, int]:
    required = {"expected_negotiation_ids", "max_rounds", "deadline_ticks"}
    if set(options) != required:
        raise ExactJoinError(
            "negotiation evidence options must be exactly "
            "expected_negotiation_ids, max_rounds, and deadline_ticks"
        )
    raw_ids = options.get("expected_negotiation_ids")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ExactJoinError("expected_negotiation_ids must be a non-empty sequence")
    expected = tuple(raw_ids)
    if any(not isinstance(value, str) or not value.strip() for value in expected):
        raise ExactJoinError("expected negotiation ids must be non-empty strings")
    if len(set(expected)) != len(expected):
        raise ExactJoinError("expected negotiation ids must be unique")
    max_rounds = _positive_int(options.get("max_rounds"), "max_rounds")
    deadline_ticks = _positive_int(options.get("deadline_ticks"), "deadline_ticks")
    return expected, max_rounds, deadline_ticks


def _canonical_events(
    rows: Mapping[str, dict[str, Any]],
) -> dict[str, NegotiationEvent]:
    typed: dict[str, NegotiationEvent] = {}
    for event_id, row in rows.items():
        try:
            event = coerce_negotiation_event(row)
        except Exception as exc:
            raise ExactJoinError(
                f"negotiation event {event_id!r} fails the canonical schema"
            ) from exc
        if dict(negotiation_event_to_dict(event)) != row or event.event_id != event_id:
            raise ExactJoinError(f"negotiation event {event_id!r} is not canonical")
        typed[event_id] = event
    return typed


def _canonical_threads(
    rows: Mapping[str, dict[str, Any]],
) -> dict[str, NegotiationThread]:
    typed: dict[str, NegotiationThread] = {}
    for negotiation_id, row in rows.items():
        try:
            thread = coerce_negotiation_thread(row)
        except Exception as exc:
            raise ExactJoinError(
                f"negotiation thread {negotiation_id!r} fails the canonical schema"
            ) from exc
        if (
            dict(negotiation_thread_to_dict(thread)) != row
            or thread.negotiation_id != negotiation_id
        ):
            raise ExactJoinError(
                f"negotiation thread {negotiation_id!r} is not canonical"
            )
        typed[negotiation_id] = thread
    return typed


def _verify_replay_and_listing_bindings(
    *,
    expected_ids: Sequence[str],
    max_rounds: int,
    deadline_ticks: int,
    events: Mapping[str, NegotiationEvent],
    threads: Mapping[str, NegotiationThread],
    initial_events: Mapping[str, NegotiationEvent],
    initial_threads: Mapping[str, NegotiationThread],
    initial_catalog: Mapping[str, dict[str, Any]],
    final_catalog: Mapping[str, dict[str, Any]],
    context: ExactJoinContext,
) -> None:
    for negotiation_id in expected_ids:
        stream = sorted(
            (event for event in events.values() if event.negotiation_id == negotiation_id),
            key=lambda event: (event.sequence_no, event.event_id),
        )
        if not stream:
            raise ExactJoinError(
                f"expected negotiation {negotiation_id!r} has no final events"
            )
        first = stream[0]
        binding = _binding_from_event(first)
        if (
            binding.max_rounds != max_rounds
            or binding.expires_at_tick - binding.opened_at_tick != deadline_ticks
        ):
            raise ExactJoinError("negotiation event policy binding differs from options")
        try:
            replay = replay_negotiation_events(binding, stream)
        except Exception as exc:
            raise ExactJoinError(
                f"negotiation {negotiation_id!r} fails strict replay"
            ) from exc
        if replay.thread != threads[negotiation_id]:
            raise ExactJoinError(
                f"negotiation {negotiation_id!r} final thread differs from replay"
            )
        _verify_agreement(threads[negotiation_id], stream[-1])

        prefix = [event for event in stream if event.event_id in initial_events]
        if set(event.event_id for event in prefix) != {
            event_id
            for event_id, event in initial_events.items()
            if event.negotiation_id == negotiation_id
        }:
            raise ExactJoinError("initial negotiation event set is not a replay prefix")
        if prefix:
            expected_prefix_ids = [event.event_id for event in stream[: len(prefix)]]
            if [event.event_id for event in prefix] != expected_prefix_ids:
                raise ExactJoinError("initial negotiation events are not a strict prefix")
            try:
                prefix_replay = replay_negotiation_events(binding, prefix)
            except Exception as exc:
                raise ExactJoinError("initial negotiation prefix fails strict replay") from exc
            if initial_threads.get(negotiation_id) != prefix_replay.thread:
                raise ExactJoinError("initial negotiation thread differs from replay")
        elif negotiation_id in initial_threads:
            raise ExactJoinError("initial negotiation thread has no event prefix")

        if not prefix:
            # A newly opened thread must bind to the listing revision visible
            # at its commit.  The catalog may legitimately change before or
            # after that point, so current initial/final equality is neither
            # required nor sufficient.
            opening_row = _catalog_row_at_opening_commit(
                context,
                event_id=first.event_id,
                sku_id=first.sku_id,
                initial_catalog=initial_catalog,
                final_catalog=final_catalog,
            )
            listing, revision = _listing_from_row(opening_row)
            if (
                str(listing.merchant_id) != first.merchant_id
                or listing.list_price.currency != first.currency
                or revision != first.listing_revision
                or negotiation_listing_digest(listing) != first.listing_digest
            ):
                raise ExactJoinError(
                    "opening event is not bound to its contemporaneous listing"
                )
        # For an E1 prefix the opening listing may have been revised or removed
        # before this evidence window.  The immutable canonical prefix and
        # replayed thread digest are the authority for that historic binding.


def _verify_platform_exchanges(
    context: ExactJoinContext,
    *,
    expected_set: set[str],
    max_rounds: int,
    deadline_ticks: int,
    events: Mapping[str, NegotiationEvent],
) -> tuple[
    dict[str, list[VerifiedNegotiationRequest]],
    list[VerifiedRejectedNegotiationRequest],
]:
    selected: list[LinkedPlatformExchange] = []
    rejected: list[VerifiedRejectedNegotiationRequest] = []
    for exchange in context.exchanges:
        endpoint = exchange.decision.get("platform_endpoint")
        kind = exchange.decision.get("action_kind")
        if endpoint == _NEGOTIATION_ENDPOINT or kind in NEGOTIATION_ACTIONS:
            if endpoint != _NEGOTIATION_ENDPOINT or kind not in NEGOTIATION_ACTIONS:
                raise ExactJoinError(
                    "negotiation action and Platform endpoint do not match"
                )
            if exchange.decision.get("decision") == "accepted":
                selected.append(exchange)
            else:
                rejected.append(_verify_rejected_exchange(context, exchange))
    if not selected and not rejected:
        raise ExactJoinError("no Platform negotiation request was audited")

    claimed_positions = {
        position
        for exchange in (*selected, *(row.exchange for row in rejected))
        for position in (exchange.request_position, *exchange.response_positions)
    }
    for position, envelope in enumerate(context.envelopes):
        action = envelope.get("action")
        kind = action.get("kind") if isinstance(action, Mapping) else None
        if kind in NEGOTIATION_ACTIONS and position not in claimed_positions:
            raise ExactJoinError(
                "negotiation envelope bypassed the accepted Platform exchange"
            )

    requests_by_event: dict[str, list[VerifiedNegotiationRequest]] = defaultdict(list)
    for exchange in selected:
        request = exchange.request
        action = _action(request)
        payload = _payload(action)
        kind = str(action["kind"])
        intent = _compact_intent(kind, payload)
        negotiation_id = intent["negotiation_id"]
        if negotiation_id not in expected_set:
            raise ExactJoinError("accepted negotiation request names an unexpected thread")
        actor_id = _required_text(request.get("from"), "request actor")
        idempotency_key = _required_text(
            request.get("idempotency_key"), "request idempotency key"
        )
        event_id = negotiation_event_id(
            negotiation_id, actor_id, idempotency_key
        )
        event = events.get(event_id)
        if event is None:
            raise ExactJoinError("accepted negotiation request has no derived event id")
        fingerprint = negotiation_intent_fingerprint(
            intent,
            max_rounds=max_rounds,
            deadline_ticks=deadline_ticks,
        )
        relay = _verify_exchange_against_event(
            exchange,
            event=event,
            intent=intent,
        )
        requests_by_event[event_id].append(
            VerifiedNegotiationRequest(
                exchange=exchange,
                relay=relay,
                intent=dict(intent),
                request_fingerprint=fingerprint,
                event_id=event_id,
            )
        )

    for event_id, requests in requests_by_event.items():
        identities = {
            (
                request.exchange.request.get("from"),
                request.exchange.request.get("idempotency_key"),
                request.request_fingerprint,
                request.event_id,
            )
            for request in requests
        }
        if len(identities) != 1:
            raise ExactJoinError("exact retry changed actor, key, intent, or event")
        requests.sort(key=lambda row: row.exchange.request_position)
    return dict(requests_by_event), rejected


def _verify_rejected_exchange(
    context: ExactJoinContext,
    exchange: LinkedPlatformExchange,
) -> VerifiedRejectedNegotiationRequest:
    """Verify a rejected attempt and prove it emitted no authority effect."""

    decision = exchange.decision
    request = exchange.request
    if exchange.responses or exchange.response_positions:
        raise ExactJoinError("rejected negotiation request emitted a relay")
    if (
        decision.get("response_msg_ids") != []
        or decision.get("response_sha256s") != []
        or decision.get("response_kinds") != []
        or decision.get("decision_metadata") != {}
    ):
        raise ExactJoinError("rejected negotiation decision claims a response or metadata")
    reason_code = decision.get("reason_code")
    if not isinstance(reason_code, str) or not reason_code:
        raise ExactJoinError("rejected negotiation decision has no reason code")
    action = _action(request)
    payload_value = action.get("payload")
    payload = dict(payload_value) if isinstance(payload_value, Mapping) else {}
    negotiation_id = payload.get("negotiation_id")
    if not isinstance(negotiation_id, str) or not negotiation_id:
        negotiation_id = None
    actor_id = request.get("from")
    key = request.get("idempotency_key")
    for commit in context.world_commits:
        if (
            commit.get("authority_action") == "world.apply_negotiation_intent"
            and commit.get("actor_id") == actor_id
            and commit.get("idempotency_key") == key
        ):
            raise ExactJoinError("rejected negotiation request has a World commit")
        writes = commit.get("table_writes")
        if not isinstance(writes, list):
            raise ExactJoinError("World commit table_writes must be an array")
        for write in writes:
            if not isinstance(write, Mapping):
                continue
            after = write.get("after")
            if (
                write.get("table") in {"negotiation_events", "authority_operations"}
                and isinstance(after, Mapping)
                and after.get("actor_id") == actor_id
                and after.get("idempotency_key") == key
            ):
                raise ExactJoinError("rejected negotiation request has an authority row")
    return VerifiedRejectedNegotiationRequest(
        exchange=exchange,
        negotiation_id=negotiation_id,
        reason_code=reason_code,
    )


def _verify_exchange_against_event(
    exchange: LinkedPlatformExchange,
    *,
    event: NegotiationEvent,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    request = exchange.request
    decision = exchange.decision
    action = _action(request)
    payload = _payload(action)
    actor_id = _required_text(request.get("from"), "request actor")
    key = _required_text(request.get("idempotency_key"), "request idempotency key")
    if request.get("to") != _NEGOTIATION_ENDPOINT:
        raise ExactJoinError("negotiation request used the wrong Platform endpoint")
    expected_event_fields = {
        "action_kind": action.get("kind"),
        "negotiation_id": intent.get("negotiation_id"),
        "offer_id": intent.get("offer_id"),
        "sku_id": intent.get("sku_id"),
        "counterparty_id": intent.get("counterparty_id"),
        "actor_id": actor_id,
        "idempotency_key": key,
    }
    for field, expected in expected_event_fields.items():
        if getattr(event, field) != expected:
            raise ExactJoinError(
                f"Platform request and World event differ on {field!r}"
            )
    for field in ("unit_price", "round_no", "qty"):
        if field in intent and getattr(event, field) != intent[field]:
            raise ExactJoinError(
                f"Platform request and World event differ on optional {field!r}"
            )
    if event.action_kind in _PRICED_ACTIONS and "unit_price" not in intent:
        raise ExactJoinError("priced negotiation event has a price-free request")
    if event.action_kind not in _PRICED_ACTIONS | _PRICE_OPTIONAL_ACTIONS:
        raise ExactJoinError("unsupported negotiation event action")
    if (
        decision.get("actor_id") != actor_id
        or decision.get("idempotency_key") != key
        or decision.get("normalized_action") != action
    ):
        raise ExactJoinError("Platform decision changed authenticated request identity")
    if len(exchange.responses) != 1:
        raise ExactJoinError("accepted negotiation request must have one exact relay")
    relay = exchange.responses[0]
    metadata = {
        "mediated_by": _NEGOTIATION_ENDPOINT,
        "submission_msg_id": request.get("msg_id"),
        "submitted_by": actor_id,
        "recipient_id": event.counterparty_id,
        "negotiation_id": event.negotiation_id,
        "canonical_round": event.round_no,
        "status": negotiation_status_for_action(event.action_kind),
        "listing_digest": event.listing_digest,
    }
    if decision.get("decision_metadata") != metadata:
        raise ExactJoinError("Platform decision metadata differs from World event")
    expected_payload = dict(payload)
    expected_payload.update(
        {
            "negotiation_id": event.negotiation_id,
            "offer_id": event.offer_id,
            "sku_id": event.sku_id,
            "currency": event.currency,
            # Platform restores the immutable public quantity from the
            # authoritative thread on every relay. Follow-up actor intents may
            # omit it, but exact evidence must still bind the canonical relay
            # to the World event quantity.
            "qty": event.qty,
            "counterparty_id": event.counterparty_id,
            "round_no": event.round_no,
            "platform_mediation": metadata,
        }
    )
    if event.action_kind in _PRICED_ACTIONS:
        expected_payload["unit_price"] = event.unit_price
    else:
        expected_payload.pop("unit_price", None)
    relay_action = _action(relay)
    relay_payload = relay_action.get("payload")
    if not isinstance(relay_payload, Mapping):
        raise ExactJoinError("Platform negotiation relay payload is not an object")
    projection_value = relay_payload.get("world_thread_projection")
    try:
        projection = coerce_negotiation_turn_projection(projection_value)
    except SchemaError as exc:
        raise ExactJoinError(
            "Platform negotiation relay has an invalid World projection"
        ) from exc
    projection_expected = {
        "negotiation_id": event.negotiation_id,
        "offer_id": event.offer_id,
        "buyer_id": event.buyer_id,
        "merchant_id": event.merchant_id,
        "sku_id": event.sku_id,
        "listing_digest": event.listing_digest,
        "currency": event.currency,
        "qty": event.qty,
        "round_no": event.round_no,
        "max_rounds": event.max_rounds,
        "status": negotiation_status_for_action(event.action_kind),
        "last_actor_id": event.actor_id,
        "last_counterparty_id": event.counterparty_id,
        # The price remains at the established top-level relay path.  The
        # compiler projection intentionally never duplicates it.
        "current_unit_price": None,
    }
    if any(
        getattr(projection, name) != expected
        for name, expected in projection_expected.items()
    ):
        raise ExactJoinError(
            "Platform negotiation projection differs from the World event"
        )
    projected_action = {
        "kind": relay_action.get("kind"),
        "payload": {
            key: value
            for key, value in relay_payload.items()
            if key != "world_thread_projection"
        },
    }
    if projected_action != {"kind": event.action_kind, "payload": expected_payload}:
        raise ExactJoinError("Platform relay payload or metadata differs from event")
    expected_relay_fields = {
        "msg_id": f"platform-negotiation:{request.get('msg_id')}",
        "ts": request.get("ts"),
        "from": _NEGOTIATION_ENDPOINT,
        "to": event.counterparty_id,
        "in_reply_to": request.get("msg_id"),
        "idempotency_key": f"platform-negotiation:{key}",
    }
    for field, expected in expected_relay_fields.items():
        if relay.get(field) != expected:
            raise ExactJoinError(f"Platform relay differs on {field!r}")
    return relay


def _verify_authority_rows(
    *,
    expected_set: set[str],
    initial_rows: Mapping[str, dict[str, Any]],
    final_rows: Mapping[str, dict[str, Any]],
    events: Mapping[str, NegotiationEvent],
    requests_by_event: Mapping[str, Sequence[VerifiedNegotiationRequest]],
) -> dict[str, dict[str, Any]]:
    if not set(initial_rows).issubset(final_rows) or any(
        final_rows[key] != row for key, row in initial_rows.items()
    ):
        raise ExactJoinError("pre-existing authority operation changed or disappeared")
    all_negotiation_initial = _negotiation_authority_rows(initial_rows)
    all_negotiation_final = _negotiation_authority_rows(final_rows)
    negotiation_initial = {
        key: row
        for key, row in all_negotiation_initial.items()
        if _authority_negotiation_id(row) in expected_set
    }
    negotiation_final = {
        key: row
        for key, row in all_negotiation_final.items()
        if _authority_negotiation_id(row) in expected_set
    }
    unrelated_initial = {
        key: row
        for key, row in all_negotiation_initial.items()
        if _authority_negotiation_id(row) not in expected_set
    }
    unrelated_final = {
        key: row
        for key, row in all_negotiation_final.items()
        if _authority_negotiation_id(row) not in expected_set
    }
    if unrelated_final != unrelated_initial:
        raise ExactJoinError("unrelated negotiation authority rows changed")
    if len(negotiation_final) != len(events):
        raise ExactJoinError(
            "final negotiation events and authority operation rows differ"
        )
    by_event: dict[str, dict[str, Any]] = {}
    for operation_key, row in negotiation_final.items():
        event_id = row.get("outcome_key")
        if not isinstance(event_id, str) or event_id not in events:
            raise ExactJoinError("negotiation authority row names an unknown event")
        if event_id in by_event:
            raise ExactJoinError("negotiation event has duplicate authority rows")
        event = events[event_id]
        requests = requests_by_event.get(event_id, ())
        fingerprint = (
            requests[0].request_fingerprint
            if requests
            else row.get("request_fingerprint")
        )
        if not _is_sha256(fingerprint):
            raise ExactJoinError("negotiation authority fingerprint is not SHA-256")
        expected_key = authority_operation_key(
            f"negotiation-event:{event.negotiation_id}",
            event.actor_id,
            event.idempotency_key,
        )
        expected = {
            "operation_key": expected_key,
            "scope": f"negotiation-event:{event.negotiation_id}",
            "actor_id": event.actor_id,
            "idempotency_key": event.idempotency_key,
            "request_fingerprint": fingerprint,
            "outcome_table": "negotiation_events",
            "outcome_key": event.event_id,
            "outcome_listing": None,
        }
        if operation_key != expected_key or row != expected:
            raise ExactJoinError(
                "negotiation authority operation does not exactly bind actor and outcome"
            )
        by_event[event_id] = row
    if not set(negotiation_initial).issubset(negotiation_final):
        # The global unchanged-row check above also proves byte equality.  This
        # branch keeps the negotiation-specific completeness invariant clear.
        raise ExactJoinError("initial negotiation authority rows are not preserved")
    if any(row.get("outcome_key") not in events for row in negotiation_initial.values()):
        raise ExactJoinError("initial negotiation authority row has no final event")
    return by_event


def _verify_negotiation_commits(
    context: ExactJoinContext,
    *,
    new_event_ids: set[str],
    events: Mapping[str, NegotiationEvent],
    threads: Mapping[str, NegotiationThread],
    event_rows: Mapping[str, dict[str, Any]],
    authority_by_event: Mapping[str, dict[str, Any]],
    requests_by_event: Mapping[str, Sequence[VerifiedNegotiationRequest]],
) -> dict[str, dict[str, Any]]:
    relevant = [commit for commit in context.world_commits if _is_negotiation_commit(commit)]
    if len(relevant) != len(new_event_ids):
        raise ExactJoinError(
            "new negotiation events and atomic World commits differ"
        )
    by_event: dict[str, dict[str, Any]] = {}
    for commit in relevant:
        writes = commit.get("table_writes")
        if not isinstance(writes, list):
            raise ExactJoinError("negotiation commit table_writes must be an array")
        tables = [write.get("table") for write in writes if isinstance(write, Mapping)]
        required_tables = {
            "negotiation_events",
            "negotiation_threads",
            "authority_operations",
            "logical_time",
        }
        if len(writes) != 4 or len(set(tables)) != 4 or set(tables) != required_tables:
            raise ExactJoinError(
                "negotiation commit must atomically write exactly four authority tables"
            )
        by_table = {str(write["table"]): dict(write) for write in writes}
        event_write = by_table["negotiation_events"]
        event_id = event_write.get("key")
        if not isinstance(event_id, str) or event_id not in new_event_ids:
            raise ExactJoinError("negotiation commit writes an unexpected event")
        if event_id in by_event:
            raise ExactJoinError("negotiation event has duplicate World commits")
        event = events[event_id]
        requests = requests_by_event.get(event_id, ())
        if not requests:
            raise ExactJoinError("negotiation commit has no accepted actor request")
        fingerprint = requests[0].request_fingerprint
        if any(request.request_fingerprint != fingerprint for request in requests):
            raise ExactJoinError("exact retry changed its compact intent fingerprint")
        expected_commit_fields = {
            "commit_kind": "transaction",
            "operation": "negotiation_event",
            "subject_id": event.negotiation_id,
            "authority_action": "world.apply_negotiation_intent",
            "actor_id": event.actor_id,
            "idempotency_key": event.idempotency_key,
            "request_fingerprint": fingerprint,
            "invariants_held": list(_REQUIRED_COMMIT_INVARIANTS),
        }
        for field, expected in expected_commit_fields.items():
            if commit.get(field) != expected:
                raise ExactJoinError(
                    f"negotiation commit differs on authority field {field!r}"
                )
        if event_write != {
            "table": "negotiation_events",
            "key": event_id,
            "op": "create",
            "before": None,
            "after": event_rows[event_id],
        }:
            raise ExactJoinError("negotiation event write is not an exact append")
        before_thread, after_thread = _thread_transition_for_event(events, event)
        thread_write = by_table["negotiation_threads"]
        expected_thread_write = {
            "table": "negotiation_threads",
            "key": event.negotiation_id,
            "op": "create" if before_thread is None else "update",
            "before": (
                None
                if before_thread is None
                else dict(negotiation_thread_to_dict(before_thread))
            ),
            "after": dict(negotiation_thread_to_dict(after_thread)),
        }
        if thread_write != expected_thread_write:
            raise ExactJoinError(
                "negotiation thread before and after rows are not replay-contiguous"
            )
        authority = authority_by_event[event_id]
        authority_write = by_table["authority_operations"]
        if authority_write != {
            "table": "authority_operations",
            "key": authority["operation_key"],
            "op": "create",
            "before": None,
            "after": authority,
        }:
            raise ExactJoinError("negotiation authority operation was not atomically created")
        clock_write = by_table["logical_time"]
        if clock_write != {
            "table": "logical_time",
            "key": "world",
            "op": "update",
            "before": event.server_tick - 1,
            "after": event.server_tick,
        }:
            raise ExactJoinError("negotiation event is not bound to one World clock tick")
        by_event[event_id] = commit
    if set(by_event) != new_event_ids:
        raise ExactJoinError("negotiation commits leave an event unclaimed")
    return by_event


def _verify_world_clock_chain(context: ExactJoinContext) -> None:
    initial = context.initial_tables.get("logical_time")
    final = context.final_tables.get("logical_time")
    if not _is_nonnegative_int(initial) or not _is_nonnegative_int(final):
        raise ExactJoinError("World logical_time snapshots must be non-negative integers")
    current = int(initial)
    previous_sequence: int | None = None
    for commit in context.world_commits:
        sequence = commit.get("sequence")
        if not _is_nonnegative_int(sequence):
            raise ExactJoinError("World commit sequence must be a non-negative integer")
        if previous_sequence is not None and int(sequence) != previous_sequence + 1:
            raise ExactJoinError("World commit sequence is not contiguous")
        previous_sequence = int(sequence)
        writes = commit.get("table_writes")
        if not isinstance(writes, list):
            raise ExactJoinError("World commit table_writes must be an array")
        clocks = [
            write
            for write in writes
            if isinstance(write, Mapping) and write.get("table") == "logical_time"
        ]
        if len(clocks) > 1:
            raise ExactJoinError("World commit contains duplicate logical_time writes")
        if clocks:
            clock = clocks[0]
            after = clock.get("after")
            if (
                clock.get("key") != "world"
                or clock.get("op") != "update"
                or clock.get("before") != current
                or not _is_nonnegative_int(after)
                or int(after) <= current
            ):
                raise ExactJoinError("World logical_time commit chain is discontinuous")
            current = int(after)
    if current != final:
        raise ExactJoinError("final World logical_time differs from the commit chain")


def _thread_transition_for_event(
    events: Mapping[str, NegotiationEvent],
    event: NegotiationEvent,
) -> tuple[NegotiationThread | None, NegotiationThread]:
    stream = sorted(
        (
            candidate
            for candidate in events.values()
            if candidate.negotiation_id == event.negotiation_id
        ),
        key=lambda candidate: (candidate.sequence_no, candidate.event_id),
    )
    index = next(
        index for index, candidate in enumerate(stream) if candidate.event_id == event.event_id
    )
    binding = _binding_from_event(stream[0])
    before = (
        replay_negotiation_events(binding, stream[:index]).thread
        if index > 0
        else None
    )
    after = replay_negotiation_events(binding, stream[: index + 1]).thread
    if after is None:  # pragma: no cover - one event always materializes a thread
        raise ExactJoinError("negotiation replay did not materialize a thread")
    return before, after


def _binding_from_event(event: NegotiationEvent) -> NegotiationBinding:
    return NegotiationBinding(
        negotiation_id=event.negotiation_id,
        buyer_id=event.buyer_id,
        merchant_id=event.merchant_id,
        offer_id=event.offer_id,
        sku_id=event.sku_id,
        listing_digest=event.listing_digest,
        listing_revision=event.listing_revision,
        currency=event.currency,
        qty=event.qty,
        max_rounds=event.max_rounds,
        opened_at_tick=event.opened_at_tick,
        expires_at_tick=event.expires_at_tick,
    )


def _verify_agreement(thread: NegotiationThread, last: NegotiationEvent) -> None:
    if thread.status == "accepted":
        agreement = thread.agreement
        if agreement is None or (
            last.action_kind != ACCEPT_OFFER
            or agreement.acceptance_event_id != last.event_id
            or agreement.acceptance_event_digest != last.event_digest
            or agreement.acceptance_idempotency_key != last.idempotency_key
            or agreement.accepted_by_id != last.actor_id
            or agreement.offered_by_id != last.counterparty_id
            or agreement.unit_price != last.unit_price
            or agreement.round_no != last.round_no
        ):
            raise ExactJoinError("accepted negotiation agreement differs from its event")
    elif thread.agreement is not None:
        raise ExactJoinError("non-accepted negotiation carries an agreement")


def _catalog_row_at_opening_commit(
    context: ExactJoinContext,
    *,
    event_id: str,
    sku_id: str,
    initial_catalog: Mapping[str, dict[str, Any]],
    final_catalog: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct the listing revision visible when a new thread opened."""

    current = initial_catalog.get(sku_id)
    opening: dict[str, Any] | None = None
    for commit in context.world_commits:
        writes = commit.get("table_writes")
        if not isinstance(writes, list):
            raise ExactJoinError("World commit table_writes must be an array")
        if any(
            isinstance(write, Mapping)
            and write.get("table") == "negotiation_events"
            and write.get("key") == event_id
            for write in writes
        ):
            if opening is not None:
                raise ExactJoinError("opening negotiation event has duplicate commits")
            if current is None:
                raise ExactJoinError("opening negotiation has no contemporaneous listing")
            opening = dict(current)
        for write in writes:
            if (
                not isinstance(write, Mapping)
                or write.get("table") != "catalog"
                or write.get("key") != sku_id
            ):
                continue
            before = write.get("before")
            after = write.get("after")
            if before != current:
                raise ExactJoinError("catalog writes are not contiguous for negotiation SKU")
            if after is not None and not isinstance(after, Mapping):
                raise ExactJoinError("catalog write after state must be an object or null")
            current = None if after is None else dict(after)
    if current != final_catalog.get(sku_id):
        raise ExactJoinError("final catalog row differs from the World commit chain")
    if opening is None:
        raise ExactJoinError("new negotiation opening has no World event commit")
    return opening


def _listing_from_row(row: Mapping[str, Any]) -> tuple[Listing, int]:
    required = {
        "sku_id",
        "category",
        "name",
        "attributes",
        "list_price",
        "merchant_id",
    }
    if not required.issubset(row) or set(row) - required - {"product_id"}:
        raise ExactJoinError("catalog listing has an unsupported serialized shape")
    attributes = row.get("attributes")
    price = row.get("list_price")
    if not isinstance(attributes, Mapping) or not isinstance(price, Mapping):
        raise ExactJoinError("catalog listing attributes and price must be objects")
    if set(price) != {"amount", "currency"}:
        raise ExactJoinError("catalog listing price is not canonical")
    try:
        listing = Listing(
            sku_id=SkuId(_required_text(row.get("sku_id"), "listing sku_id")),
            category=_required_text(row.get("category"), "listing category"),
            name=_required_text(row.get("name"), "listing name"),
            attributes=dict(attributes),
            list_price=Money(
                Decimal(str(price["amount"])),
                _required_text(price.get("currency"), "listing currency"),
            ),
            merchant_id=AgentId(
                _required_text(row.get("merchant_id"), "listing merchant_id")
            ),
            product_id=(
                None
                if row.get("product_id") is None
                else _required_text(row.get("product_id"), "listing product_id")
            ),
        )
    except Exception as exc:
        raise ExactJoinError("catalog listing cannot be reconstructed") from exc
    revision = attributes.get("catalog_revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ExactJoinError("catalog revision must be a positive integer")
    return listing, revision


def _compact_intent(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {"action_kind": kind}
    for field in ("negotiation_id", "offer_id", "sku_id", "counterparty_id"):
        if field not in payload:
            raise ExactJoinError(f"negotiation request is missing {field!r}")
        raw[field] = payload[field]
    for field in ("unit_price", "round_no", "qty"):
        if field in payload:
            raw[field] = payload[field]
    try:
        return dict(normalize_negotiation_intent(raw))
    except Exception as exc:
        raise ExactJoinError("negotiation request has no valid compact intent") from exc


def _negotiation_authority_rows(
    rows: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for key, row in rows.items():
        scope = row.get("scope")
        if not isinstance(scope, str) or not scope.startswith("negotiation-event:"):
            continue
        selected[key] = row
    return selected


def _authority_negotiation_id(row: Mapping[str, Any]) -> str:
    scope = row.get("scope")
    if not isinstance(scope, str) or not scope.startswith("negotiation-event:"):
        raise ExactJoinError("negotiation authority row has an invalid scope")
    return scope.removeprefix("negotiation-event:")


def _is_negotiation_commit(commit: Mapping[str, Any]) -> bool:
    if (
        commit.get("operation") == "negotiation_event"
        or commit.get("authority_action") == "world.apply_negotiation_intent"
    ):
        return True
    writes = commit.get("table_writes")
    if not isinstance(writes, list):
        return False
    for write in writes:
        if not isinstance(write, Mapping):
            continue
        if write.get("table") in {"negotiation_events", "negotiation_threads"}:
            return True
        after = write.get("after")
        if (
            write.get("table") == "authority_operations"
            and isinstance(after, Mapping)
            and str(after.get("scope", "")).startswith("negotiation-event:")
        ):
            return True
    return False


def _unique_rows(
    tables: Mapping[str, Any],
    table: str,
    key_field: str,
) -> dict[str, dict[str, Any]]:
    raw = tables.get(table, [])
    if not isinstance(raw, list):
        raise ExactJoinError(f"World table {table!r} must be a row array")
    rows: dict[str, dict[str, Any]] = {}
    for ordinal, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"World table {table!r} row {ordinal} is not an object")
        row = dict(value)
        key = row.get(key_field)
        if not isinstance(key, str) or not key:
            raise ExactJoinError(
                f"World table {table!r} row {ordinal} has no {key_field!r}"
            )
        if key in rows:
            raise ExactJoinError(f"World table {table!r} has duplicate key {key!r}")
        rows[key] = row
    return rows


def _action(envelope: Mapping[str, Any]) -> dict[str, Any]:
    action = envelope.get("action")
    if not isinstance(action, Mapping):
        raise ExactJoinError("negotiation envelope action must be an object")
    return dict(action)


def _payload(action: Mapping[str, Any]) -> dict[str, Any]:
    payload = action.get("payload")
    if not isinstance(payload, Mapping):
        raise ExactJoinError("negotiation action payload must be an object")
    return dict(payload)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExactJoinError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExactJoinError(f"{label} must be a positive integer")
    return value


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _event_sort_key(event: NegotiationEvent) -> tuple[str, int, str]:
    return event.negotiation_id, event.sequence_no, event.event_id


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        NEGOTIATION_EVIDENCE_CONTRACT,
        verify_negotiation_evidence_contract,
    )
)


__all__ = [
    "NEGOTIATION_EVIDENCE_CONTRACT",
    "VerifiedNegotiationEvidence",
    "VerifiedNegotiationOperation",
    "VerifiedNegotiationRequest",
    "VerifiedRejectedNegotiationRequest",
    "verify_negotiation_evidence_contract",
]
