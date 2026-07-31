"""Reusable exact joins across Runtime, Platform, and World evidence.

Hash manifests and deterministic replay prove that stored bytes are complete
and replayable.  They do not, by themselves, prove that one actor request is
the request that caused a particular World commit.  This module supplies that
semantic authority boundary.

Operation families register task-agnostic contracts.  A contract receives
byte-linked Platform exchanges, the World commit journal, and the initial and
final serialized World tables.  It must either return exact verified joins or
raise :class:`ExactJoinError`.  Benchmark scorers consume only those verified
joins; scenarios never manufacture operation state or receipts.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from threading import RLock
from typing import Any

from protocol.envelope import from_json, to_json
from protocol.event_receipts import (
    ProtocolEvent,
    ProtocolEventReceipt,
    protocol_event_from_json,
    protocol_event_receipt_from_json,
    protocol_event_receipt_to_dict,
    protocol_event_to_dict,
    validate_protocol_event_stream,
    validate_protocol_receipt_stream,
)
from protocol.matching import MATCH_CERTIFICATE_SCHEMA, coerce_match_certificate
from runtime.evidence import (
    PLATFORM_DECISION_SCHEMA,
    PLATFORM_RESPONSE_DISPOSITION_SCHEMA,
)
from world.catalog_mutations import (
    catalog_mutation_fingerprint,
    catalog_owner_for_actor,
    normalize_catalog_mutation_intent,
)
from world.match_authorizations import match_authorization_acceptance_key
from world.payment_fulfillment import (
    authoritative_payment_receipt_digest,
    packing_record_from_dict,
    packing_record_key,
    packing_record_to_dict,
    payment_state_from_dict,
    payment_state_key,
    payment_state_to_dict,
)
from world.state import (
    order_operation_reference_digest_from_row,
    protocol_operation_effect_idempotency_key,
    protocol_operation_effect_reference_digest,
    protocol_operation_outcome_identity,
)
from world.types import AgentId, Money, OrderId, Receipt, SkuId, TxnId


PROTOCOL_EVENT_EVIDENCE_CONTRACT = "commerceworld.protocol-event.v1"
CATALOG_MUTATION_EVIDENCE_CONTRACT = "commerceworld.catalog-mutation.v1"


class ExactJoinError(ValueError):
    """Evidence is individually well formed but fails a cross-layer join."""


@dataclass(frozen=True, slots=True)
class LinkedPlatformExchange:
    """One Platform request joined to its exact decision and responses."""

    request: dict[str, Any]
    decision: dict[str, Any]
    responses: tuple[dict[str, Any], ...]
    request_position: int
    response_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExactJoinContext:
    """Transport-neutral serialized evidence supplied to operation contracts."""

    envelopes: tuple[dict[str, Any], ...]
    exchanges: tuple[LinkedPlatformExchange, ...]
    world_commits: tuple[dict[str, Any], ...]
    initial_tables: dict[str, Any]
    final_tables: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OperationEvidenceContract:
    """A named verifier that can be added without changing the core linker."""

    contract_id: str
    verifier: Callable[[ExactJoinContext, Mapping[str, Any]], Any]

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("operation evidence contract id must not be blank")


class OperationEvidenceRegistry:
    """Thread-safe registry for task-agnostic operation evidence contracts."""

    def __init__(self) -> None:
        self._contracts: dict[str, OperationEvidenceContract] = {}
        self._lock = RLock()

    def register(self, contract: OperationEvidenceContract) -> None:
        with self._lock:
            if contract.contract_id in self._contracts:
                raise ValueError(
                    f"operation evidence contract already registered: "
                    f"{contract.contract_id!r}"
                )
            self._contracts[contract.contract_id] = contract

    def verify(
        self,
        contract_id: str,
        context: ExactJoinContext,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        with self._lock:
            contract = self._contracts.get(contract_id)
        if contract is None:
            raise ExactJoinError(
                f"unregistered operation evidence contract {contract_id!r}"
            )
        return contract.verifier(context, dict(options or {}))

    @property
    def contract_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._contracts))


@dataclass(frozen=True, slots=True)
class VerifiedProtocolEventJoin:
    """One authority event and its optional actor decision/effect chain."""

    event_id: str
    event: ProtocolEvent
    issue_exchange: LinkedPlatformExchange
    delivery: dict[str, Any] | None
    publish_commit: dict[str, Any]
    decision_exchange: LinkedPlatformExchange | None
    receipt_response: dict[str, Any] | None
    receipt: ProtocolEventReceipt | None
    receipt_commit: dict[str, Any] | None
    operation: str | None
    outcome_table: str | None
    outcome_key: str | None
    outcome_row: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class VerifiedProtocolEventEvidence:
    """Complete verified protocol-event authority graph for one evidence set."""

    events: tuple[VerifiedProtocolEventJoin, ...]

    def by_event_id(self) -> dict[str, VerifiedProtocolEventJoin]:
        return {row.event_id: row for row in self.events}


@dataclass(frozen=True, slots=True)
class VerifiedCatalogMutationJoin:
    """One catalog actor intent joined to its authoritative World outcome."""

    request: dict[str, Any]
    exchange: LinkedPlatformExchange
    response: dict[str, Any]
    intent: dict[str, Any]
    request_fingerprint: str
    commit: dict[str, Any]
    authority_operation: dict[str, Any]
    outcome_listing: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedCatalogMutationEvidence:
    mutations: tuple[VerifiedCatalogMutationJoin, ...]


def wire_envelope_sha256(envelope: Mapping[str, Any]) -> str:
    """Hash a decoded envelope through the canonical wire codec."""

    try:
        parsed = from_json(_canonical_text(dict(envelope)))
    except Exception as exc:
        raise ExactJoinError("cannot canonicalize audited envelope") from exc
    return hashlib.sha256(to_json(parsed).encode("utf-8")).hexdigest()


def build_exact_join_context(
    *,
    envelopes: Iterable[Mapping[str, Any]],
    platform_decisions: Iterable[Mapping[str, Any]],
    platform_response_dispositions: Iterable[Mapping[str, Any]] = (),
    world_commits: Iterable[Mapping[str, Any]],
    initial_tables: Mapping[str, Any],
    final_tables: Mapping[str, Any],
    allow_not_audited_platform_responses: bool = False,
) -> ExactJoinContext:
    """Normalize evidence and establish all request/decision/response links."""

    normalized_envelopes = tuple(
        _json_object(row, f"envelope[{index}]")
        for index, row in enumerate(envelopes)
    )
    normalized_decisions = tuple(
        _json_object(row, f"platform decision[{index}]")
        for index, row in enumerate(platform_decisions)
    )
    normalized_commits = tuple(
        _json_object(row, f"World commit[{index}]")
        for index, row in enumerate(world_commits)
    )
    normalized_dispositions = tuple(
        _json_object(row, f"Platform response disposition[{index}]")
        for index, row in enumerate(platform_response_dispositions)
    )
    return ExactJoinContext(
        envelopes=normalized_envelopes,
        exchanges=link_platform_exchanges(
            normalized_envelopes,
            normalized_decisions,
            normalized_dispositions,
            allow_not_audited=allow_not_audited_platform_responses,
        ),
        world_commits=normalized_commits,
        initial_tables=_json_object(initial_tables, "initial World tables"),
        final_tables=_json_object(final_tables, "final World tables"),
    )


def link_platform_exchanges(
    envelopes: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    response_dispositions: Iterable[Mapping[str, Any]] = (),
    *,
    allow_not_audited: bool = False,
) -> tuple[LinkedPlatformExchange, ...]:
    """Byte-link the complete Platform request, decision, and response streams.

    The linker consumes every Platform request and decision before callers may
    filter by action kind.  Response occurrences are claimed exactly once.
    This makes an exact idempotent retry valid only when both the request and
    response were actually audited again, while rejecting a duplicated or
    cross-decision response splice.
    """

    normalized_envelopes = tuple(
        _json_object(row, f"envelope[{index}]")
        for index, row in enumerate(envelopes)
    )
    normalized_decisions = tuple(
        _json_object(row, f"platform decision[{index}]")
        for index, row in enumerate(decisions)
    )
    normalized_dispositions = tuple(
        _json_object(row, f"Platform response disposition[{index}]")
        for index, row in enumerate(response_dispositions)
    )
    return _link_platform_exchanges(
        normalized_envelopes,
        normalized_decisions,
        normalized_dispositions,
        allow_not_audited=allow_not_audited,
    )


DEFAULT_OPERATION_EVIDENCE_REGISTRY = OperationEvidenceRegistry()


def verify_registered_operation_evidence(
    contract_id: str,
    *,
    envelopes: Iterable[Mapping[str, Any]],
    platform_decisions: Iterable[Mapping[str, Any]],
    platform_response_dispositions: Iterable[Mapping[str, Any]] = (),
    world_commits: Iterable[Mapping[str, Any]],
    initial_tables: Mapping[str, Any],
    final_tables: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
    allow_not_audited_platform_responses: bool = False,
    registry: OperationEvidenceRegistry = DEFAULT_OPERATION_EVIDENCE_REGISTRY,
) -> Any:
    """Run one registered exact-join contract over serialized evidence."""

    context = build_exact_join_context(
        envelopes=envelopes,
        platform_decisions=platform_decisions,
        platform_response_dispositions=platform_response_dispositions,
        world_commits=world_commits,
        initial_tables=initial_tables,
        final_tables=final_tables,
        allow_not_audited_platform_responses=(
            allow_not_audited_platform_responses
        ),
    )
    return registry.verify(contract_id, context, options=options)


def verify_protocol_event_evidence(
    *,
    expected_event_ids: Iterable[str],
    envelopes: Iterable[Mapping[str, Any]],
    platform_decisions: Iterable[Mapping[str, Any]],
    world_commits: Iterable[Mapping[str, Any]],
    initial_tables: Mapping[str, Any],
    final_tables: Mapping[str, Any],
) -> VerifiedProtocolEventEvidence:
    """Verify protocol issuance, actor decisions, receipts, and effects."""

    result = verify_registered_operation_evidence(
        PROTOCOL_EVENT_EVIDENCE_CONTRACT,
        envelopes=envelopes,
        platform_decisions=platform_decisions,
        world_commits=world_commits,
        initial_tables=initial_tables,
        final_tables=final_tables,
        options={"expected_event_ids": tuple(expected_event_ids)},
    )
    if not isinstance(result, VerifiedProtocolEventEvidence):
        raise ExactJoinError("protocol event contract returned the wrong result type")
    return result


def verify_catalog_mutation_evidence(
    *,
    envelopes: Iterable[Mapping[str, Any]],
    platform_decisions: Iterable[Mapping[str, Any]],
    world_commits: Iterable[Mapping[str, Any]],
    initial_tables: Mapping[str, Any],
    final_tables: Mapping[str, Any],
) -> VerifiedCatalogMutationEvidence:
    """Verify every accepted compact catalog mutation in an evidence set."""

    result = verify_registered_operation_evidence(
        CATALOG_MUTATION_EVIDENCE_CONTRACT,
        envelopes=envelopes,
        platform_decisions=platform_decisions,
        world_commits=world_commits,
        initial_tables=initial_tables,
        final_tables=final_tables,
    )
    if not isinstance(result, VerifiedCatalogMutationEvidence):
        raise ExactJoinError("catalog mutation contract returned the wrong result type")
    return result


def _validated_platform_response_lifecycle(
    decisions: tuple[dict[str, Any], ...],
    dispositions: tuple[dict[str, Any], ...],
    *,
    allow_not_audited: bool,
) -> dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] | None:
    """Validate Runtime's append-only produced-response transition stream."""

    if not dispositions:
        return None
    enqueue_by_occurrence: dict[str, dict[str, Any]] = {}
    terminal_by_occurrence: dict[str, dict[str, Any]] = {}
    key_by_occurrence: dict[str, tuple[int, int]] = {}
    identity_fields = (
        "run_id",
        "occurrence_id",
        "decision_sequence",
        "request_msg_id",
        "response_index",
        "response_msg_id",
        "response_envelope_sha256",
        "response_kind",
    )
    for event_index, event in enumerate(dispositions):
        if event.get("schema_version") != PLATFORM_RESPONSE_DISPOSITION_SCHEMA:
            raise ExactJoinError(
                f"Platform response disposition[{event_index}] has unsupported schema"
            )
        if event.get("sequence") != event_index:
            raise ExactJoinError(
                f"Platform response disposition[{event_index}] is out of sequence"
            )
        occurrence_id = event.get("occurrence_id")
        decision_sequence = event.get("decision_sequence")
        response_index = event.get("response_index")
        if (
            not isinstance(occurrence_id, str)
            or not occurrence_id
            or isinstance(decision_sequence, bool)
            or not isinstance(decision_sequence, int)
            or decision_sequence < 0
            or isinstance(response_index, bool)
            or not isinstance(response_index, int)
            or response_index < 0
        ):
            raise ExactJoinError(
                f"Platform response disposition[{event_index}] has invalid identity"
            )
        state = event.get("state")
        if state == "enqueued":
            if occurrence_id in enqueue_by_occurrence:
                raise ExactJoinError("Platform response occurrence was enqueued twice")
            enqueue_by_occurrence[occurrence_id] = event
            key_by_occurrence[occurrence_id] = (decision_sequence, response_index)
            continue
        if state not in {"audited", "not_audited_at_shutdown"}:
            raise ExactJoinError(
                f"Platform response disposition[{event_index}] has invalid state"
            )
        enqueue = enqueue_by_occurrence.get(occurrence_id)
        if enqueue is None or occurrence_id in terminal_by_occurrence:
            raise ExactJoinError("Platform response terminal transition is invalid")
        if any(event.get(field) != enqueue.get(field) for field in identity_fields):
            raise ExactJoinError("Platform response identity changed across transition")
        if state == "not_audited_at_shutdown" and not allow_not_audited:
            raise ExactJoinError(
                "non-audited Platform response requires a verified scoreable abort"
            )
        terminal_by_occurrence[occurrence_id] = event
    if set(enqueue_by_occurrence) != set(terminal_by_occurrence):
        raise ExactJoinError("Platform response lifecycle has a pending occurrence")

    lifecycle: dict[
        tuple[int, int], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    for occurrence_id, enqueue in enqueue_by_occurrence.items():
        key = key_by_occurrence[occurrence_id]
        if key in lifecycle:
            raise ExactJoinError("Platform response decision/index key is ambiguous")
        lifecycle[key] = (enqueue, terminal_by_occurrence[occurrence_id])

    expected_keys: set[tuple[int, int]] = set()
    for decision_index, decision in enumerate(decisions):
        response_ids = decision.get("response_msg_ids")
        response_hashes = decision.get("response_sha256s")
        response_kinds = decision.get("response_kinds")
        if (
            not isinstance(response_ids, list)
            or not isinstance(response_hashes, list)
            or len(response_ids) != len(response_hashes)
            or not isinstance(response_kinds, list)
        ):
            raise ExactJoinError(
                f"Platform decision[{decision_index}] has invalid response linkage"
            )
        produced_kinds: list[str] = []
        for response_index, (response_id, response_hash) in enumerate(
            zip(response_ids, response_hashes, strict=True)
        ):
            key = (decision_index, response_index)
            expected_keys.add(key)
            pair = lifecycle.get(key)
            if pair is None:
                raise ExactJoinError(
                    f"Platform decision[{decision_index}] response was not enqueued"
                )
            enqueue, _terminal = pair
            if (
                enqueue.get("decision_sequence") != decision_index
                or enqueue.get("request_msg_id") != decision.get("request_msg_id")
                or enqueue.get("run_id") != decision.get("run_id")
                or enqueue.get("response_msg_id") != response_id
                or enqueue.get("response_envelope_sha256") != response_hash
            ):
                raise ExactJoinError(
                    f"Platform decision[{decision_index}] response lifecycle contradicts production"
                )
            produced_kinds.append(str(enqueue.get("response_kind", "")))
        if sorted(produced_kinds) != response_kinds:
            raise ExactJoinError(
                f"Platform decision[{decision_index}] response kinds contradict lifecycle"
            )
    if set(lifecycle) != expected_keys:
        raise ExactJoinError("Platform response lifecycle has an unclaimed occurrence")
    return lifecycle


def _link_platform_exchanges(
    envelopes: tuple[dict[str, Any], ...],
    decisions: tuple[dict[str, Any], ...],
    response_dispositions: tuple[dict[str, Any], ...] = (),
    *,
    allow_not_audited: bool = False,
) -> tuple[LinkedPlatformExchange, ...]:
    platform_request_occurrences = tuple(
        (position, row)
        for position, row in enumerate(envelopes)
        if row.get("to") == "platform"
        or str(row.get("to", "")).startswith("platform:")
    )
    if len(platform_request_occurrences) != len(decisions):
        raise ExactJoinError(
            "Platform request and decision streams have different lengths"
        )
    lifecycle = _validated_platform_response_lifecycle(
        decisions,
        response_dispositions,
        allow_not_audited=allow_not_audited,
    )
    by_msg_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for position, envelope in enumerate(envelopes):
        msg_id = envelope.get("msg_id")
        if not isinstance(msg_id, str) or not msg_id:
            raise ExactJoinError(f"audited envelope[{position}] has no msg_id")
        by_msg_id.setdefault(msg_id, []).append((position, envelope))

    claimed_response_occurrences: Counter[tuple[str, str]] = Counter()
    produced_response_ids: set[str] = set()
    for ordinal, decision in enumerate(decisions):
        response_ids = decision.get("response_msg_ids")
        response_hashes = decision.get("response_sha256s")
        if not isinstance(response_ids, list) or not isinstance(
            response_hashes, list
        ) or len(response_ids) != len(response_hashes):
            raise ExactJoinError(
                f"Platform decision[{ordinal}] has invalid response linkage"
            )
        for response_index, (response_id, response_hash) in enumerate(zip(
            response_ids, response_hashes, strict=True
        )):
            if isinstance(response_id, str):
                produced_response_ids.add(response_id)
            if (
                lifecycle is not None
                and lifecycle[(ordinal, response_index)][1].get("state")
                != "audited"
            ):
                continue
            if not isinstance(response_id, str) or not isinstance(
                response_hash, str
            ):
                raise ExactJoinError(
                    f"Platform decision[{ordinal}] has invalid response identity"
                )
            claimed_response_occurrences[(response_id, response_hash)] += 1
    audited_platform_responses: Counter[tuple[str, str]] = Counter()
    platform_request_ids = {
        str(request.get("msg_id", ""))
        for _position, request in platform_request_occurrences
    }
    for envelope in envelopes:
        sender = str(envelope.get("from", ""))
        if sender == "platform" or sender.startswith("platform:"):
            if lifecycle is not None and (
                str(envelope.get("msg_id", "")) not in produced_response_ids
                and envelope.get("in_reply_to") not in platform_request_ids
            ):
                # Platform-authored scenario inputs are not responses.  With a
                # Runtime lifecycle present, response occurrences are exactly
                # those linked by decision id or request correlation.
                continue
            audited_platform_responses[
                (str(envelope.get("msg_id", "")), wire_envelope_sha256(envelope))
            ] += 1
    # Exact retries may audit the same response identity more than once.  They
    # are valid only when there is one audited occurrence per decision claim.
    # Every Platform-authored envelope is covered.  A new unclaimed message id,
    # a gratuitous duplicate, or one response reused by two decisions is thus
    # fail-closed instead of becoming ambient scorer evidence.
    if audited_platform_responses != claimed_response_occurrences:
        raise ExactJoinError(
            "audited Platform response occurrences do not match Platform "
            "decision claims exactly"
        )

    claimed_responses: set[int] = set()
    linked: list[LinkedPlatformExchange] = []
    for ordinal, ((request_position, request), decision) in enumerate(
        zip(platform_request_occurrences, decisions, strict=True)
    ):
        action = request.get("action")
        if not isinstance(action, Mapping):
            raise ExactJoinError(f"Platform request[{ordinal}] has no action")
        if decision.get("schema_version") != PLATFORM_DECISION_SCHEMA:
            raise ExactJoinError(
                f"Platform decision[{ordinal}] has unsupported schema"
            )
        if decision.get("sequence") != ordinal:
            raise ExactJoinError(
                f"Platform decision[{ordinal}] has non-contiguous sequence"
            )
        normalized_action = _json_object(action, "normalized Platform action")
        expected_fields = {
            "request_msg_id": request.get("msg_id"),
            "request_envelope_sha256": wire_envelope_sha256(request),
            "actor_id": request.get("from"),
            "platform_endpoint": request.get("to"),
            "action_kind": normalized_action.get("kind"),
            "idempotency_key": request.get("idempotency_key"),
            "normalized_action": normalized_action,
            "normalized_action_sha256": _sha256_json(normalized_action),
        }
        for field, expected in expected_fields.items():
            if decision.get(field) != expected:
                raise ExactJoinError(
                    f"Platform decision[{ordinal}] does not join request field "
                    f"{field!r}"
                )
        if decision.get("decision") not in {"accepted", "rejected"}:
            raise ExactJoinError(
                f"Platform decision[{ordinal}] has invalid outcome"
            )
        response_ids = decision.get("response_msg_ids")
        response_hashes = decision.get("response_sha256s")
        response_kinds = decision.get("response_kinds")
        if (
            not isinstance(response_ids, list)
            or not isinstance(response_hashes, list)
            or not isinstance(response_kinds, list)
            or len(response_ids) != len(response_hashes)
            or len(set(response_ids)) != len(response_ids)
        ):
            raise ExactJoinError(
                f"Platform decision[{ordinal}] has invalid response linkage"
            )
        responses: list[dict[str, Any]] = []
        positions: list[int] = []
        produced_kinds: list[str] = []
        for response_index, (response_id, response_hash) in enumerate(zip(
            response_ids, response_hashes, strict=True
        )):
            lifecycle_pair = (
                lifecycle.get((ordinal, response_index))
                if lifecycle is not None
                else None
            )
            if lifecycle_pair is not None:
                produced_kinds.append(str(lifecycle_pair[0].get("response_kind", "")))
                if lifecycle_pair[1].get("state") == "not_audited_at_shutdown":
                    continue
            candidates = [
                (position, candidate)
                for position, candidate in by_msg_id.get(str(response_id), ())
                if wire_envelope_sha256(candidate) == response_hash
                and position not in claimed_responses
                and position > request_position
            ]
            if not candidates:
                raise ExactJoinError(
                    f"Platform decision[{ordinal}] response {response_id!r} "
                    "does not resolve to one unclaimed audited occurrence"
                )
            position, response = candidates[0]
            positions.append(position)
            responses.append(response)
        observed_kinds = sorted(
            str((response.get("action") or {}).get("kind", ""))
            for response in responses
        )
        if lifecycle is None:
            expected_kind_evidence = observed_kinds
        else:
            expected_kind_evidence = sorted(produced_kinds)
        if expected_kind_evidence != response_kinds:
            raise ExactJoinError(
                f"Platform decision[{ordinal}] response kinds do not match"
            )
        claimed_responses.update(positions)
        linked.append(
            LinkedPlatformExchange(
                request=request,
                decision=decision,
                responses=tuple(responses),
                request_position=request_position,
                response_positions=tuple(positions),
            )
        )
    return tuple(linked)


def _verify_protocol_event_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedProtocolEventEvidence:
    raw_expected = options.get("expected_event_ids")
    if not isinstance(raw_expected, (list, tuple)) or not raw_expected:
        raise ExactJoinError("protocol event verification needs expected_event_ids")
    expected = tuple(str(value) for value in raw_expected)
    if any(not value for value in expected) or len(set(expected)) != len(expected):
        raise ExactJoinError("expected protocol event ids must be unique and non-empty")
    expected_set = set(expected)

    initial_events = _unique_rows(
        context.initial_tables, "protocol_events", "event_id"
    )
    if expected_set.intersection(initial_events):
        raise ExactJoinError("expected protocol event was already present initially")
    event_rows = _unique_rows(
        context.final_tables, "protocol_events", "event_id"
    )
    if set(event_rows) != set(initial_events) | expected_set:
        raise ExactJoinError(
            "final protocol event set differs from the initial and expected sets"
        )
    if any(event_rows[event_id] != row for event_id, row in initial_events.items()):
        raise ExactJoinError("pre-existing protocol event changed during the episode")

    typed_events: dict[str, ProtocolEvent] = {}
    for event_id, row in event_rows.items():
        try:
            event = protocol_event_from_json(_canonical_text(row))
        except Exception as exc:
            raise ExactJoinError(
                f"final protocol event {event_id!r} fails its core schema"
            ) from exc
        if protocol_event_to_dict(event) != row:
            raise ExactJoinError(
                f"final protocol event {event_id!r} is not canonical"
            )
        typed_events[event_id] = event
    _validate_event_streams(tuple(typed_events.values()))

    initial_receipts = _unique_rows(
        context.initial_tables, "protocol_receipts", "receipt_id"
    )
    receipt_by_id = _unique_rows(
        context.final_tables, "protocol_receipts", "receipt_id"
    )
    if not set(initial_receipts).issubset(receipt_by_id) or any(
        receipt_by_id[receipt_id] != row
        for receipt_id, row in initial_receipts.items()
    ):
        raise ExactJoinError("pre-existing protocol receipt changed during the episode")
    receipt_rows_by_event: dict[str, dict[str, Any]] = {}
    typed_receipts_by_event: dict[str, ProtocolEventReceipt] = {}
    all_typed_receipts: list[ProtocolEventReceipt] = []
    seen_receipt_events: set[str] = set()
    for receipt_id, row in receipt_by_id.items():
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or event_id not in event_rows:
            raise ExactJoinError(
                f"protocol receipt {receipt_id!r} names an unexpected event"
            )
        if event_id in seen_receipt_events:
            raise ExactJoinError(
                f"protocol event {event_id!r} has duplicate final receipts"
            )
        seen_receipt_events.add(event_id)
        try:
            receipt = protocol_event_receipt_from_json(_canonical_text(row))
        except Exception as exc:
            raise ExactJoinError(
                f"final protocol receipt {receipt_id!r} fails its core schema"
            ) from exc
        if protocol_event_receipt_to_dict(receipt) != row:
            raise ExactJoinError(
                f"final protocol receipt {receipt_id!r} is not canonical"
            )
        all_typed_receipts.append(receipt)
        if receipt_id not in initial_receipts:
            if event_id not in expected_set:
                raise ExactJoinError(
                    "new protocol receipt belongs to a pre-existing event"
                )
            if event_id in receipt_rows_by_event:
                raise ExactJoinError(
                    f"protocol event {event_id!r} has duplicate new receipts"
                )
            receipt_rows_by_event[event_id] = row
            typed_receipts_by_event[event_id] = receipt
    _validate_receipt_streams(
        tuple(typed_events.values()),
        tuple(all_typed_receipts),
    )

    issue_by_event: dict[str, list[LinkedPlatformExchange]] = {}
    decision_by_event: dict[str, list[LinkedPlatformExchange]] = {}
    for exchange in context.exchanges:
        if exchange.decision.get("decision") != "accepted":
            continue
        action_kind = exchange.decision.get("action_kind")
        payload = _action_payload(exchange.request)
        event_id = payload.get("event_id")
        if action_kind == "platform.issue_protocol_event":
            if not isinstance(event_id, str) or event_id not in expected_set:
                raise ExactJoinError("accepted issuance names an unexpected event")
            issue_by_event.setdefault(event_id, []).append(exchange)
        elif action_kind in {
            "commerce.acknowledge_protocol_event",
            "commerce.reject_protocol_event",
            "commerce.process_protocol_event",
        }:
            if not isinstance(event_id, str) or event_id not in expected_set:
                raise ExactJoinError("accepted decision names an unexpected event")
            decision_by_event.setdefault(event_id, []).append(exchange)
    if set(issue_by_event) != expected_set:
        raise ExactJoinError("each final protocol event needs one accepted issuance")

    event_commit_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    receipt_commit_by_event: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    _index_protocol_commits(
        context.world_commits,
        expected_event_ids=expected_set,
        event_commits=event_commit_by_id,
        receipt_commits=receipt_commit_by_event,
    )
    if set(event_commit_by_id) != expected_set:
        raise ExactJoinError("each final protocol event needs one publish commit")
    if set(receipt_commit_by_event) != set(receipt_rows_by_event):
        raise ExactJoinError(
            "final protocol receipts and World receipt commits differ"
        )
    if set(decision_by_event) != set(receipt_rows_by_event):
        raise ExactJoinError(
            "accepted actor decisions and final protocol receipts differ"
        )

    final_ledger = _unique_rows(context.final_tables, "ledger", "txn_id")
    initial_ledger = _unique_rows(context.initial_tables, "ledger", "txn_id")
    final_shipments = _unique_rows(
        context.final_tables, "shipments", "shipment_id"
    )
    initial_shipments = _unique_rows(
        context.initial_tables, "shipments", "shipment_id"
    )
    final_certificates = _unique_rows(
        context.final_tables, "match_certificates", "cert_id"
    )
    final_orders = _unique_rows(context.final_tables, "orders", "order_id")

    verified: list[VerifiedProtocolEventJoin] = []
    for event_id in expected:
        event = typed_events[event_id]
        order_row = final_orders.get(event.binding.order_id)
        if order_row is None:
            raise ExactJoinError("protocol event order is absent from final World")
        issue_retries = issue_by_event[event_id]
        issue = issue_retries[0]
        delivery = _one_response_or_undelivered(
            issue,
            "platform.deliver_protocol_event",
            expected_undelivered=_protocol_delivery_wire(issue, event),
        )
        for retry in issue_retries:
            retry_delivery = _one_response_or_undelivered(
                retry,
                "platform.deliver_protocol_event",
                expected_undelivered=_protocol_delivery_wire(retry, event),
            )
            _verify_issue_exchange(
                retry,
                retry_delivery,
                event,
                certificate_rows=final_certificates,
            )
            if not _same_request_intent(issue.request, retry.request):
                raise ExactJoinError(
                    "accepted protocol issuance retry changed request intent"
                )
        publish_commit, publish_write = event_commit_by_id[event_id]
        _verify_publish_commit(
            issue.request,
            event,
            event_rows[event_id],
            publish_commit,
            publish_write,
        )

        decision_retries = decision_by_event.get(event_id, [])
        decision_exchange = decision_retries[0] if decision_retries else None
        receipt = typed_receipts_by_event.get(event_id)
        if decision_exchange is None:
            verified.append(
                VerifiedProtocolEventJoin(
                    event_id=event_id,
                    event=event,
                    issue_exchange=issue,
                    delivery=delivery,
                    publish_commit=publish_commit,
                    decision_exchange=None,
                    receipt_response=None,
                    receipt=None,
                    receipt_commit=None,
                    operation=None,
                    outcome_table=None,
                    outcome_key=None,
                    outcome_row=None,
                )
            )
            continue
        if receipt is None:
            raise ExactJoinError("accepted protocol decision has no final receipt")
        receipt_response = _one_response_or_undelivered(
            decision_exchange,
            "platform.protocol_event_receipt",
            expected_undelivered=_protocol_receipt_wire(
                decision_exchange,
                receipt,
            ),
        )
        for retry in decision_retries:
            retry_response = _one_response_or_undelivered(
                retry,
                "platform.protocol_event_receipt",
                expected_undelivered=_protocol_receipt_wire(retry, receipt),
            )
            _verify_decision_exchange(
                retry,
                retry_response,
                event,
                receipt,
                receipt_rows_by_event[event_id],
            )
            if not _same_request_intent(
                decision_exchange.request, retry.request
            ):
                raise ExactJoinError(
                    "accepted protocol decision retry changed request intent"
                )
        receipt_commit, receipt_write = receipt_commit_by_event[event_id]
        if (
            not issue.response_positions
            or decision_exchange.request_position <= issue.response_positions[0]
            or int(receipt_commit.get("sequence", -1))
            <= int(publish_commit.get("sequence", -1))
        ):
            raise ExactJoinError(
                "protocol decision or World effect precedes event delivery"
            )
        _verify_receipt_commit(
            decision_exchange.request,
            event,
            receipt,
            receipt_rows_by_event[event_id],
            receipt_commit,
            receipt_write,
        )

        operation: str | None = None
        outcome_table: str | None = None
        outcome_key: str | None = None
        outcome_row: dict[str, Any] | None = None
        if receipt.decision == "process":
            operation, outcome_table, outcome_key = (
                protocol_operation_outcome_identity(event)
            )
            final_index = (
                final_ledger if outcome_table == "ledger" else final_shipments
            )
            initial_index = (
                initial_ledger if outcome_table == "ledger" else initial_shipments
            )
            if outcome_key in initial_index or outcome_key not in final_index:
                raise ExactJoinError(
                    f"process event {event_id!r} lacks one new durable outcome"
                )
            outcome_row = final_index[outcome_key]
            _verify_process_effect(
                event,
                receipt,
                receipt_commit,
                operation=operation,
                outcome_table=outcome_table,
                outcome_key=outcome_key,
                outcome_row=outcome_row,
                certificate_rows=final_certificates,
                final_tables=context.final_tables,
            )
        else:
            if receipt.effect_reference_digests:
                raise ExactJoinError("non-process receipt carries business effects")
        verified.append(
            VerifiedProtocolEventJoin(
                event_id=event_id,
                event=event,
                issue_exchange=issue,
                delivery=delivery,
                publish_commit=publish_commit,
                decision_exchange=decision_exchange,
                receipt_response=receipt_response,
                receipt=receipt,
                receipt_commit=receipt_commit,
                operation=operation,
                outcome_table=outcome_table,
                outcome_key=outcome_key,
                outcome_row=outcome_row,
            )
        )
    verified_by_stream: dict[str, list[VerifiedProtocolEventJoin]] = {}
    for row in verified:
        verified_by_stream.setdefault(
            row.event.binding.binding_digest, []
        ).append(row)
    for stream in verified_by_stream.values():
        publish_sequences = [
            int(row.publish_commit.get("sequence", -1))
            for row in sorted(stream, key=lambda item: item.event.sequence)
        ]
        if publish_sequences != sorted(publish_sequences) or len(
            set(publish_sequences)
        ) != len(publish_sequences):
            raise ExactJoinError(
                "protocol publish commit order differs from event stream order"
            )
    return VerifiedProtocolEventEvidence(tuple(verified))


def _verify_catalog_mutation_contract(
    context: ExactJoinContext,
    _options: Mapping[str, Any],
) -> VerifiedCatalogMutationEvidence:
    accepted = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("decision") == "accepted"
        and exchange.decision.get("action_kind")
        in {"commerce.update_listing", "commerce.adjust_price"}
    )
    commits = tuple(
        commit
        for commit in context.world_commits
        if commit.get("operation") == "catalog_mutation"
        or commit.get("authority_action") == "world.apply_catalog_mutation"
    )
    initial_catalog = _unique_rows(context.initial_tables, "catalog", "sku_id")
    final_catalog = _unique_rows(context.final_tables, "catalog", "sku_id")
    initial_authority = _unique_rows(
        context.initial_tables, "authority_operations", "operation_key"
    )
    authority_rows = _unique_rows(
        context.final_tables, "authority_operations", "operation_key"
    )

    prepared: dict[
        tuple[str, str, str, str],
        list[tuple[LinkedPlatformExchange, dict[str, Any], str]],
    ] = {}
    for exchange in accepted:
        request = exchange.request
        if exchange.decision.get("platform_endpoint") != "platform:catalog":
            raise ExactJoinError("catalog mutation used the wrong Platform endpoint")
        intent = _catalog_intent_from_request(exchange)
        fingerprint = catalog_mutation_fingerprint(
            normalize_catalog_mutation_intent(intent)
        )
        actor_id = request.get("from")
        idempotency_key = request.get("idempotency_key")
        sku_id = intent["sku_id"]
        if not isinstance(actor_id, str) or not isinstance(
            idempotency_key, str
        ) or not idempotency_key:
            raise ExactJoinError("catalog mutation lacks an authenticated merchant")
        try:
            owner_id = catalog_owner_for_actor(actor_id)
        except Exception as exc:
            raise ExactJoinError(
                "catalog mutation lacks an authenticated merchant"
            ) from exc
        identity = (actor_id, idempotency_key, fingerprint, sku_id)
        prepared.setdefault(identity, []).append((exchange, intent, owner_id))

    commit_by_identity: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    commit_ids: set[str] = set()
    for commit in commits:
        identity = (
            str(commit.get("actor_id", "")),
            str(commit.get("idempotency_key", "")),
            str(commit.get("request_fingerprint", "")),
            str(commit.get("subject_id", "")),
        )
        commit_id = str(commit.get("commit_id", ""))
        if (
            not all(identity)
            or not commit_id
            or commit_id in commit_ids
            or identity in commit_by_identity
        ):
            raise ExactJoinError("catalog World commit identity is missing or duplicate")
        commit_ids.add(commit_id)
        commit_by_identity[identity] = commit
    if set(commit_by_identity) != set(prepared):
        raise ExactJoinError(
            "accepted catalog mutation groups and catalog World commits differ"
        )

    current_catalog = {
        sku_id: _json_object(row, f"initial catalog {sku_id!r}")
        for sku_id, row in initial_catalog.items()
    }
    seen_operation_keys: set[str] = set()
    verified_by_identity: dict[
        tuple[str, str, str, str],
        tuple[dict[str, Any], dict[str, Any]],
    ] = {}
    ordered_commits = sorted(
        commit_by_identity.items(),
        key=lambda item: int(item[1].get("sequence", -1)),
    )
    for identity, commit in ordered_commits:
        actor_id, idempotency_key, fingerprint, sku_id = identity
        group = prepared[identity]
        first_exchange, intent, owner_id = group[0]
        for exchange, retry_intent, retry_owner in group:
            if retry_intent != intent or retry_owner != owner_id:
                raise ExactJoinError("catalog exact retry changed normalized intent")
            response = _one_response(exchange, "platform.catalog_ack")
            _verify_catalog_response(exchange, response, intent)
        if (
            commit.get("operation") != "catalog_mutation"
            or commit.get("authority_action") != "world.apply_catalog_mutation"
            or commit.get("commit_kind") != "transaction"
        ):
            raise ExactJoinError("catalog commit has the wrong authority contract")
        writes = _commit_writes(commit)
        authority_writes = [
            write for write in writes if write.get("table") == "authority_operations"
        ]
        catalog_writes = [
            write for write in writes if write.get("table") == "catalog"
        ]
        if (
            len(authority_writes) != 1
            or len(catalog_writes) > 1
            or len(writes) != len(authority_writes) + len(catalog_writes)
        ):
            raise ExactJoinError("catalog commit has invalid atomic table writes")
        authority_write = authority_writes[0]
        operation = authority_write.get("after")
        if not isinstance(operation, dict):
            raise ExactJoinError("catalog commit has no authority operation snapshot")
        operation_key = operation.get("operation_key")
        expected_operation = {
            "scope": "catalog-mutation",
            "actor_id": actor_id,
            "idempotency_key": idempotency_key,
            "request_fingerprint": fingerprint,
            "outcome_table": "catalog",
            "outcome_key": sku_id,
        }
        for field, value in expected_operation.items():
            if operation.get(field) != value:
                raise ExactJoinError(
                    f"catalog authority operation mismatches {field!r}"
                )
        if (
            not isinstance(operation_key, str)
            or operation_key in initial_authority
            or operation_key in seen_operation_keys
            or authority_write.get("key") != operation_key
            or authority_write.get("op") != "create"
            or authority_write.get("before") is not None
            or authority_rows.get(operation_key) != operation
        ):
            raise ExactJoinError(
                "catalog authority operation is not persisted exactly once"
            )
        seen_operation_keys.add(operation_key)
        outcome = operation.get("outcome_listing")
        if not isinstance(outcome, dict):
            raise ExactJoinError("catalog authority operation has no Listing outcome")
        if outcome.get("sku_id") != sku_id or outcome.get("merchant_id") != owner_id:
            raise ExactJoinError("catalog outcome is not bound to actor and SKU")
        before = current_catalog.get(sku_id)
        if catalog_writes:
            write = catalog_writes[0]
            expected_op = "create" if before is None else "update"
            if (
                write.get("key") != sku_id
                or write.get("op") != expected_op
                or write.get("before") != before
                or write.get("after") != outcome
            ):
                raise ExactJoinError("catalog write does not equal authority outcome")
            current_catalog[sku_id] = outcome
        elif before is None or before != outcome:
            raise ExactJoinError(
                "catalog semantic no-op does not equal its current pre-state"
            )
        verified_by_identity[identity] = (operation, outcome)

    if current_catalog != final_catalog:
        raise ExactJoinError(
            "replayed catalog mutation outcomes differ from final catalog"
        )
    new_catalog_authorities = {
        key
        for key, row in authority_rows.items()
        if row.get("scope") == "catalog-mutation" and key not in initial_authority
    }
    if new_catalog_authorities != seen_operation_keys:
        raise ExactJoinError(
            "final catalog authority operations differ from verified commits"
        )

    verified: list[VerifiedCatalogMutationJoin] = []
    for identity, group in prepared.items():
        commit = commit_by_identity[identity]
        operation, outcome = verified_by_identity[identity]
        fingerprint = identity[2]
        for exchange, intent, _owner in group:
            response = _one_response(exchange, "platform.catalog_ack")
            verified.append(
                VerifiedCatalogMutationJoin(
                    request=exchange.request,
                    exchange=exchange,
                    response=response,
                    intent=intent,
                    request_fingerprint=fingerprint,
                    commit=commit,
                    authority_operation=operation,
                    outcome_listing=outcome,
                )
            )
    return VerifiedCatalogMutationEvidence(tuple(verified))


def _validate_event_streams(events: tuple[ProtocolEvent, ...]) -> None:
    by_binding: dict[str, list[ProtocolEvent]] = {}
    for event in events:
        by_binding.setdefault(event.binding.binding_digest, []).append(event)
    for records in by_binding.values():
        ordered = tuple(sorted(records, key=lambda row: row.sequence))
        try:
            validate_protocol_event_stream(ordered, binding=ordered[0].binding)
        except Exception as exc:
            raise ExactJoinError("persisted protocol event stream is invalid") from exc


def _validate_receipt_streams(
    events: tuple[ProtocolEvent, ...],
    receipts: tuple[ProtocolEventReceipt, ...],
) -> None:
    events_by_binding: dict[str, list[ProtocolEvent]] = {}
    receipts_by_binding: dict[str, list[ProtocolEventReceipt]] = {}
    for event in events:
        events_by_binding.setdefault(event.binding.binding_digest, []).append(event)
    for receipt in receipts:
        receipts_by_binding.setdefault(
            receipt.binding.binding_digest, []
        ).append(receipt)
    if not set(receipts_by_binding).issubset(events_by_binding):
        raise ExactJoinError("protocol receipt stream has no event stream")
    for binding_digest, records in receipts_by_binding.items():
        event_records = tuple(
            sorted(events_by_binding[binding_digest], key=lambda row: row.sequence)
        )
        event_sequence = {row.event_id: row.sequence for row in event_records}
        receipt_records = tuple(
            sorted(
                records,
                key=lambda row: (
                    row.logical_tick,
                    event_sequence.get(row.event_id, 0),
                    row.receipt_id,
                ),
            )
        )
        try:
            validate_protocol_receipt_stream(
                receipt_records,
                events=event_records,
                binding=event_records[0].binding,
            )
        except Exception as exc:
            raise ExactJoinError("persisted protocol receipt stream is invalid") from exc


def _index_protocol_commits(
    commits: tuple[dict[str, Any], ...],
    *,
    expected_event_ids: set[str],
    event_commits: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    receipt_commits: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    for commit in commits:
        writes = _commit_writes(commit)
        event_writes = [write for write in writes if write.get("table") == "protocol_events"]
        receipt_writes = [write for write in writes if write.get("table") == "protocol_receipts"]
        operation = commit.get("operation")
        authority = commit.get("authority_action")
        if event_writes or operation == "publish_protocol_event" or authority == "world.publish_protocol_event":
            if (
                operation != "publish_protocol_event"
                or authority != "world.publish_protocol_event"
                or len(event_writes) != 1
                or len(writes) != 1
            ):
                raise ExactJoinError("malformed protocol event publish commit")
            after = event_writes[0].get("after")
            event_id = after.get("event_id") if isinstance(after, Mapping) else None
            if not isinstance(event_id, str) or event_id not in expected_event_ids:
                raise ExactJoinError("publish commit writes an unexpected event")
            _insert_unique(event_commits, event_id, (commit, event_writes[0]), "publish commit")
        if receipt_writes or operation in {"process_protocol_event", "append_protocol_receipt"} or authority in {"world.process_protocol_event", "world.append_protocol_receipt"}:
            if len(receipt_writes) != 1:
                raise ExactJoinError("protocol receipt commit must write one receipt")
            if operation == "process_protocol_event":
                if authority != "world.process_protocol_event":
                    raise ExactJoinError("process receipt commit has wrong authority")
            elif operation == "append_protocol_receipt":
                if authority != "world.append_protocol_receipt" or len(writes) != 1:
                    raise ExactJoinError("evidence-only receipt commit is malformed")
            else:
                raise ExactJoinError("protocol receipt was written by an unknown operation")
            after = receipt_writes[0].get("after")
            event_id = after.get("event_id") if isinstance(after, Mapping) else None
            if not isinstance(event_id, str) or event_id not in expected_event_ids:
                raise ExactJoinError("receipt commit writes an unexpected event")
            _insert_unique(receipt_commits, event_id, (commit, receipt_writes[0]), "receipt commit")


def _verify_issue_exchange(
    exchange: LinkedPlatformExchange,
    delivery: dict[str, Any] | None,
    event: ProtocolEvent,
    *,
    certificate_rows: Mapping[str, dict[str, Any]],
) -> None:
    request = exchange.request
    payload = _action_payload(request)
    required = {
        "market_id", "stream_id", "order_id", "recipient_id",
        "event_id", "event_kind", "ttl_ticks",
    }
    optional = {
        "reference_kind",
        "reference_digest",
        "reference_authorization_id",
    }
    if set(payload) - required - optional or required - set(payload):
        raise ExactJoinError("protocol issuance payload has invalid fields")
    binding = event.binding
    expected = {
        "event_id": event.event_id,
        "event_kind": event.event_kind,
        "market_id": binding.market_id,
        "stream_id": binding.stream_id,
        "order_id": binding.order_id,
        "recipient_id": binding.recipient_id,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ExactJoinError(f"protocol issuance mismatches {field!r}")
    ttl = payload.get("ttl_ticks")
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, int)
        or event.expires_at_tick - event.issued_at_tick != ttl
        or event.idempotency_key != request.get("idempotency_key")
        or event.actor_id != "platform:events"
    ):
        raise ExactJoinError("protocol issuance TTL or authority identity mismatches")
    if payload.get("reference_kind", "operation") != event.reference_kind:
        raise ExactJoinError("protocol issuance reference kind mismatches")
    if payload.get("reference_digest") is not None and payload.get("reference_digest") != event.reference_digest:
        raise ExactJoinError("protocol issuance reference digest mismatches")
    authorization_id = payload.get("reference_authorization_id")
    if authorization_id is not None:
        if (
            not isinstance(authorization_id, str)
            or not authorization_id.strip()
            or event.reference_kind != "certificate"
        ):
            raise ExactJoinError(
                "protocol issuance match authorization is invalid"
            )
        matches = [
            row
            for row in certificate_rows.values()
            if row.get("certificate_digest") == event.reference_digest
        ]
        if len(matches) != 1:
            raise ExactJoinError(
                "protocol issuance match authorization does not resolve uniquely"
            )
        try:
            certificate = coerce_match_certificate(
                {"schema_version": MATCH_CERTIFICATE_SCHEMA, **matches[0]}
            )
        except Exception as exc:
            raise ExactJoinError(
                "protocol issuance match authorization certificate is invalid"
            ) from exc
        if (
            certificate.idempotency_key
            != match_authorization_acceptance_key(authorization_id)
            or certificate.order_id != binding.order_id
            or certificate.buyer_id != binding.buyer_id
            or certificate.merchant_id != binding.merchant_id
        ):
            raise ExactJoinError(
                "protocol issuance match authorization is not event-bound"
            )
    if delivery is None:
        return
    response_payload = _action_payload(delivery)
    if (
        set(response_payload) != {"event"}
        or response_payload.get("event") != protocol_event_to_dict(event)
    ):
        raise ExactJoinError("Platform delivery does not carry the persisted event")
    if (
        delivery.get("from") != "platform:events"
        or delivery.get("to") != binding.recipient_id
        or delivery.get("in_reply_to") != request.get("msg_id")
        or delivery.get("idempotency_key") != request.get("idempotency_key")
    ):
        raise ExactJoinError("protocol delivery envelope identity mismatches")


def _protocol_delivery_wire(
    exchange: LinkedPlatformExchange,
    event: ProtocolEvent,
) -> dict[str, Any]:
    request = exchange.request
    return {
        "msg_id": f"{event.event_id}:delivery",
        "ts": request.get("ts"),
        "from": "platform:events",
        "to": event.binding.recipient_id,
        "in_reply_to": request.get("msg_id"),
        "idempotency_key": event.idempotency_key,
        "signature": None,
        "action": {
            "kind": "platform.deliver_protocol_event",
            "payload": {"event": protocol_event_to_dict(event)},
        },
    }


def _protocol_receipt_wire(
    exchange: LinkedPlatformExchange,
    receipt: ProtocolEventReceipt,
) -> dict[str, Any]:
    """Reconstruct the one Platform receipt already bound by World state.

    This is used only after the Runtime response-disposition journal proves a
    scoreable abort left the produced response unaudited.  The decision journal
    must still contain this exact envelope id, kind, and wire hash.
    """

    request = exchange.request
    return {
        "msg_id": f"{request.get('msg_id')}:protocol-receipt",
        "ts": request.get("ts"),
        "from": "platform:events",
        "to": request.get("from"),
        "in_reply_to": request.get("msg_id"),
        "idempotency_key": request.get("idempotency_key"),
        "signature": None,
        "action": {
            "kind": "platform.protocol_event_receipt",
            "payload": {"receipt": protocol_event_receipt_to_dict(receipt)},
        },
    }


def _verify_publish_commit(
    request: dict[str, Any],
    event: ProtocolEvent,
    event_row: dict[str, Any],
    commit: dict[str, Any],
    write: dict[str, Any],
) -> None:
    if (
        commit.get("commit_kind") != "transaction"
        or commit.get("actor_id") != "platform:events"
        or commit.get("idempotency_key") != request.get("idempotency_key")
        or commit.get("subject_id") != event.binding.order_id
        or write.get("key") != event.event_id
        or write.get("op") != "create"
        or write.get("before") is not None
        or write.get("after") != event_row
    ):
        raise ExactJoinError("protocol publish commit does not exactly join issuance")


def _verify_decision_exchange(
    exchange: LinkedPlatformExchange,
    response: dict[str, Any] | None,
    event: ProtocolEvent,
    receipt: ProtocolEventReceipt,
    receipt_row: dict[str, Any],
) -> None:
    request = exchange.request
    payload = _action_payload(request)
    if set(payload) != {"event_id", "reason"}:
        raise ExactJoinError("protocol decision payload has invalid fields")
    decision_by_kind = {
        "commerce.acknowledge_protocol_event": "acknowledge",
        "commerce.reject_protocol_event": "reject",
        "commerce.process_protocol_event": "process",
    }
    expected_decision = decision_by_kind.get(exchange.decision.get("action_kind"))
    if (
        expected_decision is None
        or payload.get("event_id") != event.event_id
        or request.get("from") != event.binding.recipient_id
        or receipt.decision != expected_decision
        or receipt.actor_id != request.get("from")
        or receipt.idempotency_key != request.get("idempotency_key")
        or receipt.reason != payload.get("reason")
        or receipt.event_id != event.event_id
        or receipt.event_digest != event.event_digest
        or receipt.binding != event.binding
    ):
        raise ExactJoinError("actor protocol decision does not join its receipt")
    if response is not None:
        response_payload = _action_payload(response)
        if (
            set(response_payload) != {"receipt"}
            or response_payload.get("receipt") != receipt_row
        ):
            raise ExactJoinError("Platform receipt response differs from final receipt")
        if (
            response.get("from") != "platform:events"
            or response.get("to") != request.get("from")
            or response.get("in_reply_to") != request.get("msg_id")
            or response.get("idempotency_key") != request.get("idempotency_key")
        ):
            raise ExactJoinError("Platform receipt response identity mismatches")


def _verify_receipt_commit(
    request: dict[str, Any],
    event: ProtocolEvent,
    receipt: ProtocolEventReceipt,
    receipt_row: dict[str, Any],
    commit: dict[str, Any],
    write: dict[str, Any],
) -> None:
    expected_operation = (
        "process_protocol_event"
        if receipt.decision == "process"
        else "append_protocol_receipt"
    )
    expected_authority = f"world.{expected_operation}"
    if (
        commit.get("commit_kind") != "transaction"
        or commit.get("operation") != expected_operation
        or commit.get("authority_action") != expected_authority
        or commit.get("actor_id") != request.get("from")
        or commit.get("idempotency_key") != request.get("idempotency_key")
        or commit.get("subject_id") != event.binding.order_id
        or write.get("key") != receipt.receipt_id
        or write.get("op") != "create"
        or write.get("before") is not None
        or write.get("after") != receipt_row
    ):
        raise ExactJoinError("protocol receipt commit does not exactly join request")


def _verify_process_effect(
    event: ProtocolEvent,
    receipt: ProtocolEventReceipt,
    commit: dict[str, Any],
    *,
    operation: str,
    outcome_table: str,
    outcome_key: str,
    outcome_row: dict[str, Any],
    certificate_rows: Mapping[str, dict[str, Any]],
    final_tables: Mapping[str, Any],
) -> None:
    commit_writes = _commit_writes(commit)
    writes = [
        write
        for write in commit_writes
        if write.get("table") == outcome_table and write.get("key") == outcome_key
    ]
    if len(writes) != 1:
        raise ExactJoinError("process commit lacks one exact business outcome write")
    write = writes[0]
    if write.get("op") != "create" or write.get("before") is not None or write.get("after") != outcome_row:
        raise ExactJoinError("process business write differs from final outcome")
    expected_digest = protocol_operation_effect_reference_digest(
        event,
        operation=operation,
        outcome_table=outcome_table,
        outcome_key=outcome_key,
    )
    if tuple(receipt.effect_reference_digests) != (expected_digest,):
        raise ExactJoinError("process receipt effect reference is not authoritative")
    invariants = commit.get("invariants_held")
    required = {
        "business-and-receipt-atomic",
        "committed-effect-reference",
        f"registered-operation:{operation}",
    }
    if not isinstance(invariants, list) or not required.issubset(invariants):
        raise ExactJoinError("process commit lacks atomic operation invariants")
    binding = event.binding
    if (
        outcome_row.get("order_id") != binding.order_id
        or outcome_row.get("buyer_id") != binding.buyer_id
        or outcome_row.get("merchant_id") != binding.merchant_id
    ):
        raise ExactJoinError("process outcome is not bound to event parties/order")
    if outcome_table == "ledger" and (
        outcome_row.get("txn_id") != outcome_key
        or outcome_row.get("idempotency_key")
        != protocol_operation_effect_idempotency_key(event)
    ):
        raise ExactJoinError("protocol ledger outcome identity mismatches")
    if outcome_table == "shipments" and outcome_row.get("shipment_id") != outcome_key:
        raise ExactJoinError("protocol shipment outcome identity mismatches")
    _verify_registered_operation_write_set(
        event,
        receipt,
        commit_writes,
        operation=operation,
        outcome_table=outcome_table,
        outcome_key=outcome_key,
        outcome_row=outcome_row,
        certificate_rows=certificate_rows,
        final_tables=final_tables,
    )


def _verify_registered_operation_write_set(
    event: ProtocolEvent,
    receipt: ProtocolEventReceipt,
    writes: list[dict[str, Any]],
    *,
    operation: str,
    outcome_table: str,
    outcome_key: str,
    outcome_row: dict[str, Any],
    certificate_rows: Mapping[str, dict[str, Any]],
    final_tables: Mapping[str, Any],
) -> None:
    """Reject effects outside the registered World operation boundary."""

    required_counts_by_operation = {
        "settle_order": Counter(
            {
                "orders": 1,
                "ledger": 1,
                "order_timelines": 1,
                "payment_states": 1,
                "logical_time": 1,
                "protocol_receipts": 1,
            }
        ),
        "dispatch_order": Counter(
            {
                "orders": 1,
                "shipments": 1,
                "order_timelines": 1,
                "packing_records": 3,
                "logical_time": 1,
                "protocol_receipts": 1,
            }
        ),
        "refund_order": Counter(
            {
                "orders": 1,
                "ledger": 1,
                "order_timelines": 1,
                "payment_states": 1,
                "logical_time": 1,
                "protocol_receipts": 1,
            }
        ),
    }
    optional_counts_by_operation = {
        "settle_order": Counter({"inventory": 1}),
        "dispatch_order": Counter(),
        "refund_order": Counter({"inventory": 1}),
    }
    required = required_counts_by_operation.get(operation)
    optional = optional_counts_by_operation.get(operation)
    if required is None or optional is None:
        raise ExactJoinError("unknown registered protocol operation write set")
    tables = [str(write.get("table", "")) for write in writes]
    actual = Counter(tables)
    if any(actual[table] != count for table, count in required.items()) or any(
        actual[table] > count for table, count in optional.items()
    ) or set(actual) != set(required) | {
        table for table, count in optional.items() if actual[table] == count
    }:
        raise ExactJoinError(
            "process commit contains missing, duplicate, or extra operation effects"
        )
    writes_by_table: dict[str, list[dict[str, Any]]] = {}
    for write in writes:
        writes_by_table.setdefault(str(write.get("table")), []).append(write)
    by_table = {
        table: rows[0]
        for table, rows in writes_by_table.items()
        if len(rows) == 1
    }
    binding = event.binding
    exact_keys = {
        "orders": binding.order_id,
        "order_timelines": binding.order_id,
        "logical_time": "world",
        "protocol_receipts": receipt.receipt_id,
        outcome_table: outcome_key,
    }
    for table, key in exact_keys.items():
        if by_table[table].get("key") != key:
            raise ExactJoinError(
                f"registered operation wrote the wrong {table!r} identity"
            )
    for table in ("orders", "order_timelines"):
        after = by_table[table].get("after")
        if not isinstance(after, Mapping) or (
            after.get("order_id") != binding.order_id
            or after.get("buyer_id") != binding.buyer_id
            or after.get("merchant_id") != binding.merchant_id
        ):
            raise ExactJoinError(
                f"registered operation {table!r} write is not party-bound"
            )
    order_write = by_table["orders"]
    order_before = order_write.get("before")
    order_after = order_write.get("after")
    expected_state = {
        "settle_order": "settled",
        "dispatch_order": "dispatched",
        "refund_order": "refunded",
    }[operation]
    if (
        order_write.get("op") != "update"
        or not isinstance(order_before, Mapping)
        or not isinstance(order_after, Mapping)
        or order_before.get("order_id") != binding.order_id
        or order_before.get("buyer_id") != binding.buyer_id
        or order_before.get("merchant_id") != binding.merchant_id
        or order_before.get("sku_id") != order_after.get("sku_id")
        or order_before.get("state") != event.required_order_state
        or order_after.get("state") != expected_state
    ):
        raise ExactJoinError(
            "registered operation has an invalid order state transition"
        )
    if event.reference_kind == "operation":
        try:
            operation_reference = order_operation_reference_digest_from_row(
                order_before,
                event.required_state_revision,
            )
        except (TypeError, ValueError) as exc:
            raise ExactJoinError(
                "registered operation has an invalid authoritative order pre-state"
            ) from exc
        if operation_reference != event.reference_digest:
            raise ExactJoinError(
                "protocol event does not reference its committed order pre-state"
            )
    else:
        matching_certificates = [
            row
            for row in certificate_rows.values()
            if row.get("certificate_digest") == event.reference_digest
        ]
        if len(matching_certificates) != 1:
            raise ExactJoinError(
                "protocol event certificate reference does not resolve uniquely"
            )
        try:
            certificate = coerce_match_certificate(
                {
                    "schema_version": MATCH_CERTIFICATE_SCHEMA,
                    **matching_certificates[0],
                }
            )
            price = order_before["agreed_price"]
            amount = Decimal(str(price["amount"]))
            unit_price_cents = int(
                (amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            raise ExactJoinError(
                "protocol event certificate or order pre-state is invalid"
            ) from exc
        if (
            certificate.order_id != binding.order_id
            or certificate.buyer_id != binding.buyer_id
            or certificate.merchant_id != binding.merchant_id
            or certificate.sku_id != order_before.get("sku_id")
            or certificate.qty != order_before.get("qty")
            or certificate.currency != price.get("currency")
            or certificate.unit_price_cents != unit_price_cents
            or receipt.logical_tick < certificate.issued_at_tick
            or receipt.logical_tick > certificate.expires_at_tick
        ):
            raise ExactJoinError(
                "protocol event certificate does not bind its order or decision tick"
            )
    if (
        receipt.observed_order_state != event.required_order_state
        or receipt.observed_state_revision != event.required_state_revision
    ):
        raise ExactJoinError(
            "protocol receipt observation differs from the event precondition"
        )
    outcome_sku = (
        outcome_row.get("sku_id")
        if outcome_table == "ledger"
        else outcome_row.get("original_sku_id")
    )
    if (
        not isinstance(outcome_sku, str)
        or not outcome_sku
        or order_after.get("sku_id") != outcome_sku
        or outcome_row.get("order_id") != order_after.get("order_id")
        or outcome_row.get("buyer_id") != order_after.get("buyer_id")
        or outcome_row.get("merchant_id") != order_after.get("merchant_id")
    ):
        raise ExactJoinError(
            "registered operation outcome and order write name different commerce facts"
        )
    if outcome_table == "ledger" and (
        outcome_row.get("qty") != order_after.get("qty")
        or outcome_row.get("price") != order_after.get("agreed_price")
    ):
        raise ExactJoinError(
            "registered protocol ledger outcome differs from the order"
        )
    receipt_after = by_table["protocol_receipts"].get("after")
    if receipt_after != protocol_event_receipt_to_dict(receipt):
        raise ExactJoinError(
            "registered operation receipt write differs from verified receipt"
        )
    inventory_write = by_table.get("inventory")
    if inventory_write is not None:
        inventory_before = inventory_write.get("before")
        inventory_after = inventory_write.get("after")
        if (
            inventory_write.get("op") != "update"
            or not isinstance(inventory_before, Mapping)
            or not isinstance(inventory_after, Mapping)
            or inventory_write.get("key") != outcome_sku
            or inventory_before.get("sku_id") != outcome_sku
            or inventory_before.get("merchant_id") != binding.merchant_id
            or (
            inventory_after.get("merchant_id") != binding.merchant_id
            or inventory_after.get("sku_id") != outcome_sku
            )
        ):
            raise ExactJoinError(
                "registered operation inventory write is not owner and SKU bound"
            )
    timeline_write = by_table["order_timelines"]
    timeline_after = timeline_write.get("after")
    if not isinstance(timeline_after, Mapping):
        raise ExactJoinError("registered operation has no timeline outcome")
    timeline_field = {
        "settle_order": "settled_at_tick",
        "dispatch_order": "dispatched_at_tick",
        "refund_order": "refunded_at_tick",
    }[operation]
    if timeline_after.get(timeline_field) is None:
        raise ExactJoinError(
            "registered operation did not record its lifecycle timeline effect"
        )
    logical_time = by_table["logical_time"]
    if (
        logical_time.get("op") != "update"
        or isinstance(logical_time.get("before"), bool)
        or not isinstance(logical_time.get("before"), int)
        or logical_time.get("after") != logical_time.get("before") + 1
    ):
        raise ExactJoinError("registered operation has an invalid World clock write")
    receipt_write = by_table["protocol_receipts"]
    if (
        receipt_write.get("op") != "create"
        or receipt_write.get("before") is not None
    ):
        raise ExactJoinError("registered operation did not append its receipt")
    _verify_payment_and_packing_effects(
        operation=operation,
        event=event,
        writes_by_table=writes_by_table,
        by_table=by_table,
        final_tables=final_tables,
        order_after=order_after,
        outcome_row=outcome_row,
        logical_tick=logical_time["after"],
    )


def _verify_payment_and_packing_effects(
    *,
    operation: str,
    event: ProtocolEvent,
    writes_by_table: Mapping[str, list[dict[str, Any]]],
    by_table: Mapping[str, dict[str, Any]],
    final_tables: Mapping[str, Any],
    order_after: Mapping[str, Any],
    outcome_row: Mapping[str, Any],
    logical_tick: int,
) -> None:
    """Validate first-class payment and packing rows, not only table names."""

    binding = event.binding
    payment_rows = _exact_join_table_rows(final_tables, "payment_states")
    effect_key = protocol_operation_effect_idempotency_key(event)
    if operation in {"settle_order", "refund_order"}:
        write = by_table["payment_states"]
        after = write.get("after")
        if (
            write.get("op") != "create"
            or write.get("before") is not None
            or not isinstance(after, Mapping)
        ):
            raise ExactJoinError(
                "registered payment operation did not append one payment version"
            )
        try:
            payment = payment_state_from_dict(after)
        except Exception as exc:
            raise ExactJoinError(
                "registered payment operation wrote an invalid payment record"
            ) from exc
        canonical_payment = payment_state_to_dict(payment)
        expected_state = "captured" if operation == "settle_order" else "refunded"
        expected_key_prefix = (
            "settlement-capture:" if operation == "settle_order"
            else "refund-resolution:"
        )
        if (
            canonical_payment != dict(after)
            or write.get("key") != payment_state_key(payment)
            or sum(row == canonical_payment for row in payment_rows) != 1
            or payment.payment_id != f"payment:{binding.order_id}"
            or payment.order_id != binding.order_id
            or payment.owner_id != binding.buyer_id
            or payment.merchant_id != binding.merchant_id
            or payment.sku_id != order_after.get("sku_id")
            or payment.qty != order_after.get("qty")
            or payment.state != expected_state
            or payment.actor_id != "platform:psp"
            or payment.logical_tick != logical_tick
            or payment.idempotency_key != f"{expected_key_prefix}{effect_key}"
        ):
            raise ExactJoinError(
                "registered payment effect is not canonical and order-bound"
            )
        price = outcome_row.get("price")
        if not isinstance(price, Mapping):
            raise ExactJoinError("registered payment outcome has no exact price")
        try:
            amount = Decimal(str(price["amount"]))
            currency = str(price["currency"])
            expected_minor = int(
                (amount * payment.qty * 100).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            receipt = Receipt(
                txn_id=TxnId(str(outcome_row["txn_id"])),
                ts=str(outcome_row["ts"]),
                order_id=OrderId(str(outcome_row["order_id"])),
                buyer_id=AgentId(str(outcome_row["buyer_id"])),
                merchant_id=AgentId(str(outcome_row["merchant_id"])),
                sku_id=SkuId(str(outcome_row["sku_id"])),
                qty=int(outcome_row["qty"]),
                price=Money(amount, currency),
                idempotency_key=str(outcome_row["idempotency_key"]),
                effect=str(outcome_row["effect"]),  # type: ignore[arg-type]
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            raise ExactJoinError(
                "registered payment outcome cannot form an authoritative receipt"
            ) from exc
        if (
            payment.amount != expected_minor
            or payment.currency != currency
            or payment.ledger_receipt_digest
            != authoritative_payment_receipt_digest(receipt)
        ):
            raise ExactJoinError(
                "registered payment version does not bind the ledger effect"
            )
        predecessors = [
            row
            for row in payment_rows
            if row.get("record_digest") == payment.previous_digest
        ]
        if operation == "settle_order":
            inventory_written = "inventory" in by_table
            if payment.version == 1:
                if payment.previous_digest is not None or not inventory_written:
                    raise ExactJoinError(
                        "fresh settlement must reserve inventory and start payment v1"
                    )
            elif payment.version == 2:
                if inventory_written or len(predecessors) != 1:
                    raise ExactJoinError(
                        "authorized settlement must consume one payment predecessor"
                    )
                try:
                    predecessor = payment_state_from_dict(predecessors[0])
                except Exception as exc:
                    raise ExactJoinError(
                        "authorized settlement predecessor is invalid"
                    ) from exc
                if (
                    predecessor.state != "authorized"
                    or predecessor.version != 1
                    or predecessor.payment_id != payment.payment_id
                ):
                    raise ExactJoinError(
                        "settlement predecessor is not its exact authorization"
                    )
            else:
                raise ExactJoinError("settlement produced an invalid payment version")
        else:
            if len(predecessors) != 1:
                raise ExactJoinError(
                    "refund payment does not name one captured predecessor"
                )
            try:
                predecessor = payment_state_from_dict(predecessors[0])
            except Exception as exc:
                raise ExactJoinError("refund payment predecessor is invalid") from exc
            if (
                predecessor.state != "captured"
                or predecessor.version + 1 != payment.version
                or predecessor.payment_id != payment.payment_id
            ):
                raise ExactJoinError(
                    "refund payment does not continue its captured history"
                )

    if operation != "dispatch_order":
        return
    packing_writes = writes_by_table.get("packing_records", [])
    packing_rows = _exact_join_table_rows(final_tables, "packing_records")
    records = []
    for write in packing_writes:
        after = write.get("after")
        if (
            write.get("op") != "create"
            or write.get("before") is not None
            or not isinstance(after, Mapping)
        ):
            raise ExactJoinError("dispatch did not append its packing history")
        try:
            record = packing_record_from_dict(after)
        except Exception as exc:
            raise ExactJoinError("dispatch wrote an invalid packing record") from exc
        canonical = packing_record_to_dict(record)
        if (
            canonical != dict(after)
            or write.get("key") != packing_record_key(record)
            or sum(row == canonical for row in packing_rows) != 1
        ):
            raise ExactJoinError("dispatch packing record is not canonical")
        records.append(record)
    records.sort(key=lambda row: row.version)
    if (
        [row.version for row in records] != [1, 2, 3]
        or [row.state for row in records] != ["created", "packed", "handed_off"]
    ):
        raise ExactJoinError(
            "fresh protocol dispatch needs created, packed, and handed-off versions"
        )
    for index, record in enumerate(records):
        predecessor = None if index == 0 else records[index - 1]
        if (
            record.packing_id != f"packing:{binding.order_id}"
            or record.order_id != binding.order_id
            or record.owner_id != binding.buyer_id
            or record.merchant_id != binding.merchant_id
            or record.sku_id != order_after.get("sku_id")
            or record.packed_qty != order_after.get("qty")
            or record.actor_id != binding.recipient_id
            or record.logical_tick != logical_tick
            or record.idempotency_key
            != f"dispatch:{binding.order_id}:{record.state}"
            or record.previous_digest
            != (None if predecessor is None else predecessor.record_digest)
        ):
            raise ExactJoinError(
                "dispatch packing history is not actor, order, and causality bound"
            )
    payment_matches = [
        row
        for row in payment_rows
        if row.get("record_digest") == records[0].payment_digest
    ]
    if len(payment_matches) != 1:
        raise ExactJoinError("dispatch packing does not resolve one captured payment")
    try:
        payment = payment_state_from_dict(payment_matches[0])
    except Exception as exc:
        raise ExactJoinError("dispatch payment reference is invalid") from exc
    if payment.state != "captured" or any(
        row.payment_digest != payment.record_digest for row in records
    ):
        raise ExactJoinError("dispatch packing is not bound to captured payment")


def _exact_join_table_rows(
    tables: Mapping[str, Any],
    table: str,
) -> tuple[dict[str, Any], ...]:
    raw_rows = tables.get(table, [])
    if raw_rows is None:
        raw_rows = []
    if not isinstance(raw_rows, list):
        raise ExactJoinError(f"World table {table!r} must be a row array")
    return tuple(
        _json_object(row, f"{table}[{index}]")
        for index, row in enumerate(raw_rows)
    )


def _catalog_intent_from_request(
    exchange: LinkedPlatformExchange,
) -> dict[str, Any]:
    payload = _action_payload(exchange.request)
    kind = exchange.decision.get("action_kind")
    if kind == "commerce.update_listing":
        # CatalogPolicy keeps actor-side bookkeeping and correlation evidence
        # on the audited request.  World deliberately receives only the
        # authority-bearing compact intent.  Reproduce that projection here
        # instead of requiring the whole actor payload to be a World intent.
        # The allowlist remains fail closed so an evidence splice cannot be
        # hidden merely because its field is not forwarded to World.
        compact_fields = frozenset({"op", "sku_id", "fields"})
        metadata_fields = frozenset(
            {"product", "op_id", "verification_source_id"}
        )
        unsupported = set(payload) - compact_fields - metadata_fields
        if unsupported:
            raise ExactJoinError(
                "update-listing request has unsupported fields: "
                + ", ".join(sorted(unsupported))
            )
        for field in ("product", "op_id", "verification_source_id"):
            if field not in payload:
                continue
            value = payload[field]
            if not isinstance(value, str) or not value.strip():
                raise ExactJoinError(
                    f"update-listing request has invalid {field!r} metadata"
                )
        op_value = payload.get("op")
        op = op_value.lower() if isinstance(op_value, str) else op_value
        fields = payload.get("fields", {})
        if fields is None and op == "delist":
            fields = {}
        intent = {
            "op": op,
            "sku_id": payload.get("sku_id"),
            "fields": fields,
        }
    elif kind == "commerce.adjust_price":
        if set(payload) != {"sku_id", "list_price"}:
            raise ExactJoinError("adjust-price request has invalid fields")
        intent = {
            "op": "adjust_price",
            "sku_id": payload.get("sku_id"),
            "fields": {"list_price": payload.get("list_price")},
        }
    else:
        raise ExactJoinError("unsupported catalog mutation action")
    try:
        return dict(normalize_catalog_mutation_intent(intent))
    except Exception as exc:
        raise ExactJoinError("catalog request is not a compact valid intent") from exc


def _verify_catalog_response(
    exchange: LinkedPlatformExchange,
    response: dict[str, Any],
    intent: dict[str, Any],
) -> None:
    request = exchange.request
    payload = _action_payload(response)
    required = {
        "kind": "world.apply_catalog_mutation",
        "op": intent["op"],
        "sku_id": intent["sku_id"],
        "status": "ok",
    }
    for field, value in required.items():
        if payload.get(field) != value:
            raise ExactJoinError(f"catalog acknowledgement mismatches {field!r}")
    allowed = set(required)
    if intent["op"] == "adjust_price":
        allowed.add("list_price_minor")
        if payload.get("list_price_minor") != intent["fields"]["list_price"]:
            raise ExactJoinError("catalog acknowledgement price mismatches")
    if set(payload) != allowed:
        raise ExactJoinError("catalog acknowledgement has unexpected fields")
    if (
        response.get("from") != "platform:catalog"
        or response.get("to") != request.get("from")
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
    ):
        raise ExactJoinError("catalog acknowledgement envelope identity mismatches")


def _one_response(
    exchange: LinkedPlatformExchange,
    kind: str,
) -> dict[str, Any]:
    matches = [
        response
        for response in exchange.responses
        if (response.get("action") or {}).get("kind") == kind
    ]
    if len(matches) != 1 or len(exchange.responses) != 1:
        raise ExactJoinError(
            f"accepted {exchange.decision.get('action_kind')!r} needs one {kind!r}"
        )
    return matches[0]


def _one_response_or_undelivered(
    exchange: LinkedPlatformExchange,
    kind: str,
    *,
    expected_undelivered: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an audited response or prove its scoreable-abort disposition.

    ``build_exact_join_context`` only omits produced responses after validating
    Runtime's append-only disposition journal and a verified scoreable
    termination.  This helper additionally pins the missing output to the
    accepted Platform decision.  It is used only for setup operations whose
    World commit must remain claimable even when the evaluated actor never saw
    the queued response.
    """

    if exchange.responses:
        return _one_response(exchange, kind)
    decision = exchange.decision
    expected = _json_object(
        expected_undelivered, f"expected undelivered {kind} response"
    )
    if (
        decision.get("response_kinds") != [kind]
        or decision.get("response_msg_ids") != [expected.get("msg_id")]
        or decision.get("response_sha256s")
        != [wire_envelope_sha256(expected)]
        or exchange.response_positions
    ):
        raise ExactJoinError(
            f"accepted {exchange.decision.get('action_kind')!r} has invalid "
            f"undelivered {kind!r} metadata"
        )
    return None


def _unique_rows(
    tables: Mapping[str, Any],
    table: str,
    key_field: str,
) -> dict[str, dict[str, Any]]:
    values = tables.get(table, [])
    if values is None:
        values = []
    if not isinstance(values, list):
        raise ExactJoinError(f"World table {table!r} must be a row array")
    rows: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise ExactJoinError(f"World table {table!r} row[{position}] is invalid")
        row = _json_object(raw, f"{table}[{position}]")
        key = row.get(key_field)
        if not isinstance(key, str) or not key:
            raise ExactJoinError(
                f"World table {table!r} row[{position}] has no {key_field!r}"
            )
        if key in rows:
            raise ExactJoinError(f"World table {table!r} has duplicate key {key!r}")
        rows[key] = row
    return rows


def _commit_writes(commit: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = commit.get("table_writes")
    if not isinstance(raw, list) or not raw:
        raise ExactJoinError("World commit has no table writes")
    return [
        _json_object(write, f"World commit table_write[{index}]")
        for index, write in enumerate(raw)
    ]


def _action_payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExactJoinError("audited action payload must be an object")
    return _json_object(payload, "action payload")


def _insert_unique(
    target: dict[str, Any],
    key: str,
    value: Any,
    label: str,
) -> None:
    if key in target:
        raise ExactJoinError(f"duplicate {label} for {key!r}")
    target[key] = value


def _same_request_intent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Compare the authority-bearing fields of two exact retry requests."""

    return all(
        left.get(field) == right.get(field)
        for field in ("from", "to", "idempotency_key", "action")
    )


def _json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_text(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExactJoinError(f"{label} is not canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise ExactJoinError(f"{label} must be an object")
    return normalized


def _canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_text(value).encode("utf-8")).hexdigest()


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        PROTOCOL_EVENT_EVIDENCE_CONTRACT,
        _verify_protocol_event_contract,
    )
)
DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        CATALOG_MUTATION_EVIDENCE_CONTRACT,
        _verify_catalog_mutation_contract,
    )
)

# Market governance is a first-class built-in contract.  Importing it here,
# after the registry and core contracts exist, keeps the operation-specific
# verifier out of this generic linker while making the public registry entry
# available even when a scorer has not imported the family module first.
importlib.import_module("runtime.market_governance_evidence")
importlib.import_module("runtime.match_certificate_evidence")
importlib.import_module("runtime.supply_fulfillment_evidence")
importlib.import_module("runtime.authority_operation_evidence")


__all__ = [
    "CATALOG_MUTATION_EVIDENCE_CONTRACT",
    "DEFAULT_OPERATION_EVIDENCE_REGISTRY",
    "ExactJoinContext",
    "ExactJoinError",
    "LinkedPlatformExchange",
    "OperationEvidenceContract",
    "OperationEvidenceRegistry",
    "PROTOCOL_EVENT_EVIDENCE_CONTRACT",
    "VerifiedCatalogMutationEvidence",
    "VerifiedCatalogMutationJoin",
    "VerifiedProtocolEventEvidence",
    "VerifiedProtocolEventJoin",
    "build_exact_join_context",
    "link_platform_exchanges",
    "verify_catalog_mutation_evidence",
    "verify_protocol_event_evidence",
    "verify_registered_operation_evidence",
    "wire_envelope_sha256",
]
