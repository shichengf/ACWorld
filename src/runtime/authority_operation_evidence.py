"""Exact authority evidence for shared CommerceWorld service operations.

These contracts cover durable operations that are used by more than one
benchmark family but are not themselves the family's headline business
operation.  Each verifier consumes the complete linked Platform exchange set
and the complete World commit journal for its operation.  A successful result
therefore proves both directions of the authority relation: every accepted
request owns one exact World commit, and every matching World commit is owned
by one accepted request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from protocol.evidence_records import coerce_evidence_record, evidence_record_to_dict
from protocol.listing_claims import coerce_listing_claim, listing_claim_to_wire
from protocol.matching import coerce_search_session, search_session_to_wire
from runtime.exact_join import (
    DEFAULT_OPERATION_EVIDENCE_REGISTRY,
    ExactJoinContext,
    ExactJoinError,
    LinkedPlatformExchange,
    OperationEvidenceContract,
    wire_envelope_sha256,
)


SEARCH_SESSION_EVIDENCE_CONTRACT = "commerceworld.search-session.v1"
LISTING_CLAIM_EVIDENCE_CONTRACT = "commerceworld.listing-claim.v1"
EVIDENCE_RECORD_EVIDENCE_CONTRACT = "commerceworld.evidence-record.v1"
MARKET_CLOCK_EVIDENCE_CONTRACT = "commerceworld.market-clock.v1"


@dataclass(frozen=True, slots=True)
class VerifiedSearchSessionJoin:
    """One search request, rank response, and immutable World session."""

    exchange: LinkedPlatformExchange
    response: dict[str, Any]
    session: dict[str, Any]
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedSearchSessionEvidence:
    """All fresh search sessions, minus commits claimed by another contract."""

    sessions: tuple[VerifiedSearchSessionJoin, ...]
    all_session_ids: tuple[str, ...]
    excluded_session_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedListingClaimJoin:
    """One compact merchant intent and its exact append-only claim commit."""

    exchange: LinkedPlatformExchange
    response: dict[str, Any]
    intent: dict[str, Any]
    claim: dict[str, Any]
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedListingClaimEvidence:
    claims: tuple[VerifiedListingClaimJoin, ...]


@dataclass(frozen=True, slots=True)
class VerifiedEvidenceRecordJoin:
    """One authenticated evidence publication and atomic World append."""

    exchange: LinkedPlatformExchange
    acknowledgement: dict[str, Any] | None
    owner_notification: dict[str, Any] | None
    record: dict[str, Any]
    authority_operation: dict[str, Any]
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedEvidenceRecordEvidence:
    records: tuple[VerifiedEvidenceRecordJoin, ...]


@dataclass(frozen=True, slots=True)
class VerifiedMarketClockJoin:
    """One Runtime clock request and its authoritative logical-time write."""

    exchange: LinkedPlatformExchange
    response: dict[str, Any] | None
    before_tick: int
    after_tick: int
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedMarketClockEvidence:
    advances: tuple[VerifiedMarketClockJoin, ...]


def verify_search_session_evidence(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedSearchSessionEvidence:
    """Exact-join every accepted search to its persisted search session."""

    unknown = sorted(set(options) - {"exclude_session_ids"})
    if unknown:
        raise ExactJoinError(
            "unknown search-session evidence options: " + ", ".join(unknown)
        )
    excluded = _text_set(options.get("exclude_session_ids", ()), "exclude_session_ids")
    exchanges = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("decision") == "accepted"
        and exchange.decision.get("action_kind") == "commerce.search"
    )
    commits = _operation_commits(
        context,
        operation="create_search_session",
        authority_action="world.create_search_session",
    )
    initial = _row_index(context.initial_tables, "search_sessions", "session_id")
    final = _row_index(context.final_tables, "search_sessions", "session_id")

    prepared: dict[str, tuple[LinkedPlatformExchange, dict[str, Any], dict[str, Any]]] = {}
    for exchange in exchanges:
        if exchange.decision.get("platform_endpoint") != "platform:aggregator":
            raise ExactJoinError("search request used the wrong Platform endpoint")
        response = _one_response(exchange, "platform.rank_offers")
        request = exchange.request
        actor_id = _text(request.get("from"), "search actor")
        idempotency_key = _text(request.get("idempotency_key"), "search idempotency key")
        if request.get("to") != "platform:aggregator":
            raise ExactJoinError("search request did not target the aggregator")
        payload = _payload(response)
        raw_session = payload.get("search_session")
        if not isinstance(raw_session, Mapping):
            raise ExactJoinError("rank response has no typed search session")
        try:
            typed = coerce_search_session(raw_session)
            canonical_wire = search_session_to_wire(typed)
        except Exception as exc:
            raise ExactJoinError("rank response search session is invalid") from exc
        if canonical_wire != dict(raw_session):
            raise ExactJoinError("rank response search session is not canonical")
        session = _strip_schema(canonical_wire)
        session_id = _text(session.get("session_id"), "search session id")
        if session_id in prepared:
            raise ExactJoinError("duplicate accepted search session")
        if (
            session.get("buyer_id") != actor_id
            or session.get("search_request_id") != idempotency_key
            or session.get("search_idempotency_key") != idempotency_key
        ):
            raise ExactJoinError("search session does not bind its actor request")
        _verify_response_identity(
            exchange,
            response,
            service_id="platform:aggregator",
            recipient_id=actor_id,
        )
        _verify_rank_projection(payload, canonical_wire)
        prepared[session_id] = (exchange, response, session)

    commit_by_session: dict[str, dict[str, Any]] = {}
    for commit in commits:
        session_id = _text(commit.get("subject_id"), "search commit subject")
        if session_id in commit_by_session:
            raise ExactJoinError("duplicate search-session World commit")
        commit_by_session[session_id] = commit
    if set(prepared) != set(commit_by_session):
        raise ExactJoinError(
            "accepted searches and search-session World commits differ"
        )
    if not excluded.issubset(prepared):
        raise ExactJoinError("excluded search session is absent from this evidence")

    replayed = dict(initial)
    joins: list[VerifiedSearchSessionJoin] = []
    ordered = sorted(
        prepared.items(),
        key=lambda item: _commit_sequence(commit_by_session[item[0]]),
    )
    for session_id, (exchange, response, session) in ordered:
        commit = commit_by_session[session_id]
        request = exchange.request
        expected_identity = {
            "commit_kind": "transaction",
            "operation": "create_search_session",
            "authority_action": "world.create_search_session",
            "actor_id": "platform:aggregator",
            "idempotency_key": request.get("idempotency_key"),
            "request_fingerprint": None,
            "subject_id": session_id,
        }
        _require_fields(commit, expected_identity, "search-session World commit")
        if commit.get("invariants_held") != [
            "server-authored-session",
            "catalog-revision-bound",
            "inventory-revision-bound",
            "buyer-scoped-idempotency",
        ]:
            raise ExactJoinError("search-session World invariants changed")
        [write] = _exact_writes(commit, 1)
        if (
            session_id in replayed
            or write.get("table") != "search_sessions"
            or write.get("key") != session_id
            or write.get("op") != "create"
            or write.get("before") is not None
            or write.get("after") != session
        ):
            raise ExactJoinError("search-session World write is not an exact append")
        replayed[session_id] = session
        if session_id not in excluded:
            joins.append(
                VerifiedSearchSessionJoin(
                    exchange=exchange,
                    response=response,
                    session=session,
                    commit=commit,
                )
            )
    if replayed != final:
        raise ExactJoinError("search-session commits differ from final World state")
    return VerifiedSearchSessionEvidence(
        sessions=tuple(joins),
        all_session_ids=tuple(session_id for session_id, _ in ordered),
        excluded_session_ids=tuple(sorted(excluded)),
    )


def verify_listing_claim_evidence(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedListingClaimEvidence:
    """Exact-join every accepted compact claim intent to World history."""

    _reject_options(options, "listing-claim")
    exchanges = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("decision") == "accepted"
        and exchange.decision.get("action_kind") == "commerce.apply_listing_claim"
    )
    commits = _operation_commits(
        context,
        operation="apply_listing_claim",
        authority_action="world.apply_listing_claim",
    )
    initial = _row_index(context.initial_tables, "listing_claims", "claim_id")
    final = _row_index(context.final_tables, "listing_claims", "claim_id")

    prepared: dict[tuple[str, str], tuple[LinkedPlatformExchange, dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for exchange in exchanges:
        if exchange.decision.get("platform_endpoint") != "platform:claims":
            raise ExactJoinError("listing claim used the wrong Platform endpoint")
        request = exchange.request
        actor_id = _text(request.get("from"), "claim merchant")
        idempotency_key = _text(request.get("idempotency_key"), "claim idempotency key")
        if request.get("to") != "platform:claims":
            raise ExactJoinError("listing claim did not target platform:claims")
        intent = _payload(request)
        claim_id = _text(intent.get("claim_id"), "claim id")
        response = _one_response(exchange, "platform.listing_claim_updated")
        _verify_response_identity(
            exchange,
            response,
            service_id="platform:claims",
            recipient_id=actor_id,
        )
        raw_claim = _payload(response).get("claim")
        if not isinstance(raw_claim, Mapping):
            raise ExactJoinError("listing-claim response has no claim")
        try:
            typed = coerce_listing_claim(raw_claim)
            canonical_wire = listing_claim_to_wire(typed)
        except Exception as exc:
            raise ExactJoinError("listing-claim response is invalid") from exc
        if canonical_wire != dict(raw_claim):
            raise ExactJoinError("listing-claim response is not canonical")
        claim = _claim_snapshot(canonical_wire)
        if (
            typed.claim_id != claim_id
            or typed.listing_id != intent.get("listing_id")
            or typed.subject != intent.get("subject")
            or typed.merchant_id != actor_id
            or typed.issuer_id != actor_id
            or typed.current.idempotency_key != idempotency_key
            or typed.current.operation != intent.get("operation")
            or dict(typed.current.content) != dict(intent.get("content") or typed.current.content)
            or typed.current.reason != intent.get("reason")
            or tuple(sorted(row.source_id for row in typed.current.evidence))
            != tuple(sorted(intent.get("evidence_record_ids", ())))
        ):
            raise ExactJoinError("listing claim does not bind its compact merchant intent")
        identity = (idempotency_key, claim_id)
        if identity in prepared:
            raise ExactJoinError("duplicate accepted listing-claim intent")
        prepared[identity] = (exchange, response, dict(intent), claim)

    commit_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for commit in commits:
        identity = (
            _text(commit.get("idempotency_key"), "claim commit idempotency key"),
            _text(commit.get("subject_id"), "claim commit subject"),
        )
        if identity in commit_by_identity:
            raise ExactJoinError("duplicate listing-claim World commit")
        commit_by_identity[identity] = commit
    if set(prepared) != set(commit_by_identity):
        raise ExactJoinError(
            "accepted listing claims and listing-claim World commits differ"
        )

    replayed = dict(initial)
    verified: list[VerifiedListingClaimJoin] = []
    ordered = sorted(
        prepared.items(),
        key=lambda item: _commit_sequence(commit_by_identity[item[0]]),
    )
    for identity, (exchange, response, intent, claim) in ordered:
        idempotency_key, claim_id = identity
        commit = commit_by_identity[identity]
        _require_fields(
            commit,
            {
                "commit_kind": "transaction",
                "operation": "apply_listing_claim",
                "authority_action": "world.apply_listing_claim",
                "actor_id": "platform:claims",
                "idempotency_key": idempotency_key,
                "request_fingerprint": None,
                "subject_id": claim_id,
            },
            "listing-claim World commit",
        )
        if commit.get("invariants_held") != [
            "merchant-owner-authenticated",
            "listing-identity-bound",
            "evidence-digest-subject-bound",
            "append-only-version-history",
            "idempotent",
        ]:
            raise ExactJoinError("listing-claim World invariants changed")
        [write] = _exact_writes(commit, 1)
        before = replayed.get(claim_id)
        if (
            write.get("table") != "listing_claims"
            or write.get("key") != claim_id
            or write.get("op") != ("create" if before is None else "update")
            or write.get("before") != before
            or write.get("after") != claim
        ):
            raise ExactJoinError("listing-claim World write broke its history")
        replayed[claim_id] = claim
        verified.append(
            VerifiedListingClaimJoin(
                exchange=exchange,
                response=response,
                intent=intent,
                claim=claim,
                commit=commit,
            )
        )
    if replayed != final:
        raise ExactJoinError("listing-claim commits differ from final World state")
    return VerifiedListingClaimEvidence(tuple(verified))


def verify_evidence_record_evidence(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedEvidenceRecordEvidence:
    """Exact-join every accepted evidence publication and atomic World write."""

    _reject_options(options, "evidence-record")
    action_kinds = {"commerce.publish_evidence_record", "platform.publish_evidence_record"}
    exchanges = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("decision") == "accepted"
        and exchange.decision.get("action_kind") in action_kinds
    )
    commits = tuple(
        commit
        for commit in context.world_commits
        if commit.get("authority_action") == "world.persist_evidence_record"
        or commit.get("operation") in {"persist_evidence_record", "bind_evidence_idempotency"}
    )
    initial_records = _composite_row_index(
        context.initial_tables,
        "evidence_records",
        ("record_id", "version"),
    )
    final_records = _composite_row_index(
        context.final_tables,
        "evidence_records",
        ("record_id", "version"),
    )
    initial_authority = _row_index(
        context.initial_tables, "authority_operations", "operation_key"
    )
    final_authority = _row_index(
        context.final_tables, "authority_operations", "operation_key"
    )

    prepared: dict[
        tuple[str, str, str],
        tuple[
            LinkedPlatformExchange,
            dict[str, Any] | None,
            dict[str, Any] | None,
            dict[str, Any],
        ],
    ] = {}
    for exchange in exchanges:
        if exchange.decision.get("platform_endpoint") != "platform:evidence":
            raise ExactJoinError("evidence publication used the wrong Platform endpoint")
        request = exchange.request
        actor_id = _text(request.get("from"), "evidence issuer")
        idempotency_key = _text(
            request.get("idempotency_key"), "evidence idempotency key"
        )
        if request.get("to") != "platform:evidence":
            raise ExactJoinError("evidence publication did not target platform:evidence")
        raw_record = _payload(request).get("record")
        if not isinstance(raw_record, Mapping):
            raise ExactJoinError("evidence publication has no record")
        try:
            typed = coerce_evidence_record(raw_record)
            record = evidence_record_to_dict(typed)
        except Exception as exc:
            raise ExactJoinError("evidence publication record is invalid") from exc
        if record != dict(raw_record) or record.get("issuer_id") != actor_id:
            raise ExactJoinError("evidence record is not canonical and issuer-bound")
        expected_response_count = (
            2 if record.get("kind") == "independent_governance_audit" else 1
        )
        acknowledgement_wire = _platform_response_wire(
            request,
            msg_id=(
                f"{request.get('msg_id')}:platform.evidence_record_persisted"
            ),
            service_id="platform:evidence",
            recipient_id=actor_id,
            idempotency_key=idempotency_key,
            kind="platform.evidence_record_persisted",
            payload={"record": record},
        )
        expected_responses = [acknowledgement_wire]
        owner_wire: dict[str, Any] | None = None
        if expected_response_count == 2:
            owner_wire = _platform_response_wire(
                request,
                msg_id=f"{request.get('msg_id')}:owner-notice",
                service_id="platform:evidence",
                recipient_id=_text(record.get("owner_id"), "evidence owner"),
                idempotency_key=f"{idempotency_key}:owner-notice",
                kind="platform.evidence_record_persisted",
                payload={"record": record},
            )
            expected_responses.append(owner_wire)
        _verify_expected_response_bundle(exchange, expected_responses)
        observed = {
            _text(row.get("msg_id"), "evidence response msg_id"): row
            for row in exchange.responses
        }
        expected_by_id = {row["msg_id"]: row for row in expected_responses}
        if len(observed) != len(exchange.responses) or any(
            expected_by_id.get(msg_id) != row
            for msg_id, row in observed.items()
        ):
            raise ExactJoinError("evidence publication response bytes changed")
        acknowledgement = observed.get(acknowledgement_wire["msg_id"])
        owner_notification = (
            None if owner_wire is None else observed.get(owner_wire["msg_id"])
        )
        if acknowledgement is not None:
            _verify_response_identity(
                exchange,
                acknowledgement,
                service_id="platform:evidence",
                recipient_id=actor_id,
            )
            if _payload(acknowledgement).get("record") != record:
                raise ExactJoinError("evidence acknowledgement changed the record")
        if owner_notification is not None:
            _verify_response_identity(
                exchange,
                owner_notification,
                service_id="platform:evidence",
                recipient_id=_text(record.get("owner_id"), "evidence owner"),
                require_request_idempotency=False,
            )
            if (
                owner_notification.get("idempotency_key")
                != f"{idempotency_key}:owner-notice"
                or _payload(owner_notification).get("record") != record
            ):
                raise ExactJoinError(
                    "governance owner notification changed the record"
                )
        identity = (
            actor_id,
            idempotency_key,
            _text(record.get("record_digest"), "evidence record digest"),
        )
        if identity in prepared:
            raise ExactJoinError("duplicate accepted evidence publication")
        prepared[identity] = (
            exchange,
            acknowledgement,
            owner_notification,
            record,
        )

    commit_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for commit in commits:
        identity = (
            _text(commit.get("actor_id"), "evidence commit actor"),
            _text(commit.get("idempotency_key"), "evidence commit idempotency key"),
            _text(commit.get("request_fingerprint"), "evidence request fingerprint"),
        )
        if identity in commit_by_identity:
            raise ExactJoinError("duplicate evidence-record World commit")
        commit_by_identity[identity] = commit
    if set(prepared) != set(commit_by_identity):
        raise ExactJoinError(
            "accepted evidence publications and evidence World commits differ"
        )

    replayed_records = dict(initial_records)
    replayed_authority = dict(initial_authority)
    new_evidence_authority_keys: set[str] = set()
    verified: list[VerifiedEvidenceRecordJoin] = []
    ordered = sorted(
        prepared.items(),
        key=lambda item: _commit_sequence(commit_by_identity[item[0]]),
    )
    for identity, (exchange, acknowledgement, owner_notification, record) in ordered:
        actor_id, idempotency_key, digest = identity
        commit = commit_by_identity[identity]
        operation = str(commit.get("operation", ""))
        if operation not in {"persist_evidence_record", "bind_evidence_idempotency"}:
            raise ExactJoinError("evidence World commit has the wrong operation")
        _require_fields(
            commit,
            {
                "commit_kind": "transaction",
                "authority_action": "world.persist_evidence_record",
                "actor_id": actor_id,
                "idempotency_key": idempotency_key,
                "request_fingerprint": digest,
                "subject_id": record.get("record_id"),
            },
            "evidence-record World commit",
        )
        writes = _exact_writes(commit, 2 if operation == "persist_evidence_record" else 1)
        by_table = {str(write.get("table")): write for write in writes}
        if len(by_table) != len(writes) or set(by_table) != (
            {"evidence_records", "authority_operations"}
            if operation == "persist_evidence_record"
            else {"authority_operations"}
        ):
            raise ExactJoinError("evidence commit is not an exact atomic write")
        authority_write = by_table["authority_operations"]
        authority = authority_write.get("after")
        if not isinstance(authority, dict):
            raise ExactJoinError("evidence commit has no authority operation")
        authority_key = _text(authority.get("operation_key"), "evidence authority key")
        record_key = _text(authority.get("outcome_key"), "evidence outcome key")
        expected_authority = {
            "scope": "evidence",
            "actor_id": actor_id,
            "idempotency_key": idempotency_key,
            "request_fingerprint": digest,
            "outcome_table": "evidence_records",
            "outcome_key": record_key,
            "outcome_listing": None,
        }
        _require_fields(authority, expected_authority, "evidence authority operation")
        if (
            authority_key in replayed_authority
            or authority_write.get("key") != authority_key
            or authority_write.get("op") != "create"
            or authority_write.get("before") is not None
        ):
            raise ExactJoinError("evidence authority operation is not a fresh append")
        replayed_authority[authority_key] = authority
        new_evidence_authority_keys.add(authority_key)
        record_identity = (str(record.get("record_id")), int(record.get("version", -1)))
        if operation == "persist_evidence_record":
            record_write = by_table["evidence_records"]
            if (
                record_identity in replayed_records
                or record_write.get("key") != record_key
                or record_write.get("op") != "create"
                or record_write.get("before") is not None
                or record_write.get("after") != record
            ):
                raise ExactJoinError("evidence record is not an exact append")
            replayed_records[record_identity] = record
            if commit.get("invariants_held") != [
                "issuer-authenticated",
                "owner-acl-bound",
                "version-contiguous",
                "digest-verified",
                "append-only",
            ]:
                raise ExactJoinError("evidence-record World invariants changed")
        elif replayed_records.get(record_identity) != record:
            raise ExactJoinError("evidence idempotency binding names another record")
        verified.append(
            VerifiedEvidenceRecordJoin(
                exchange=exchange,
                acknowledgement=acknowledgement,
                owner_notification=owner_notification,
                record=record,
                authority_operation=authority,
                commit=commit,
            )
        )
    if replayed_records != final_records:
        raise ExactJoinError("evidence commits differ from final evidence records")
    if any(final_authority.get(key) != row for key, row in replayed_authority.items()):
        raise ExactJoinError("evidence commits differ from final authority operations")
    observed_new_evidence = {
        key
        for key, row in final_authority.items()
        if row.get("scope") == "evidence" and key not in initial_authority
    }
    if observed_new_evidence != new_evidence_authority_keys:
        raise ExactJoinError("unclaimed evidence authority operation exists")
    return VerifiedEvidenceRecordEvidence(tuple(verified))


def verify_market_clock_evidence(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedMarketClockEvidence:
    """Exact-join Runtime clock requests to monotonic World clock writes."""

    _reject_options(options, "market-clock")
    exchanges = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("decision") == "accepted"
        and exchange.decision.get("action_kind") == "platform.advance_market_clock"
    )
    commits = _operation_commits(
        context,
        operation="advance_clock",
        authority_action="world.advance_logical_time",
    )
    if len(exchanges) != len(commits):
        raise ExactJoinError("accepted clock advances and World commits differ")
    ordered_exchanges = sorted(exchanges, key=lambda row: row.request_position)
    ordered_commits = sorted(commits, key=_commit_sequence)
    initial_tick = _tick(context.initial_tables.get("logical_time"), "initial logical time")
    final_tick = _tick(context.final_tables.get("logical_time"), "final logical time")
    previous_after = initial_tick
    verified: list[VerifiedMarketClockJoin] = []
    seen_targets: set[int] = set()
    for exchange, commit in zip(ordered_exchanges, ordered_commits, strict=True):
        if exchange.decision.get("platform_endpoint") != "platform:events":
            raise ExactJoinError("clock advance used the wrong Platform endpoint")
        request = exchange.request
        if request.get("from") != "runtime:clock" or request.get("to") != "platform:events":
            raise ExactJoinError("clock advance is not Runtime-authenticated")
        request_payload = _payload(request)
        if set(request_payload) != {"to_tick"}:
            raise ExactJoinError("clock request fields changed")
        target = _tick(request_payload.get("to_tick"), "clock target")
        if target in seen_targets:
            raise ExactJoinError("duplicate market-clock target")
        seen_targets.add(target)
        response: dict[str, Any] | None
        if exchange.responses:
            response = _one_response(exchange, "platform.market_clock_advanced")
            _verify_response_identity(
                exchange,
                response,
                service_id="platform:events",
                recipient_id="runtime:clock",
            )
            if _payload(response) != {"logical_time": target}:
                raise ExactJoinError("clock response differs from its request")
        else:
            expected_response = _platform_response_wire(
                request,
                msg_id=f"{request.get('msg_id')}:market-clock-advanced",
                service_id="platform:events",
                recipient_id="runtime:clock",
                idempotency_key=request.get("idempotency_key"),
                kind="platform.market_clock_advanced",
                payload={"logical_time": target},
            )
            _verify_expected_response_bundle(exchange, [expected_response])
            response = None
        _require_fields(
            commit,
            {
                "commit_kind": "transaction",
                "operation": "advance_clock",
                "authority_action": "world.advance_logical_time",
                "actor_id": "runtime:clock",
                "idempotency_key": None,
                "request_fingerprint": None,
                "subject_id": "world",
            },
            "market-clock World commit",
        )
        if commit.get("invariants_held") != [
            "runtime-clock-only",
            "monotonic",
            "deterministic",
        ]:
            raise ExactJoinError("market-clock World invariants changed")
        [write] = _exact_writes(commit, 1)
        before = _tick(write.get("before"), "clock write before")
        after = _tick(write.get("after"), "clock write after")
        if (
            write.get("table") != "logical_time"
            or write.get("key") != "world"
            or write.get("op") != "update"
            or after != target
            or before != previous_after
            or after <= before
        ):
            raise ExactJoinError("market-clock World write is not monotonic and exact")
        previous_after = after
        verified.append(
            VerifiedMarketClockJoin(
                exchange=exchange,
                response=response,
                before_tick=before,
                after_tick=after,
                commit=commit,
            )
        )
    if final_tick < previous_after:
        raise ExactJoinError("final World clock precedes a verified clock commit")
    return VerifiedMarketClockEvidence(tuple(verified))


def _verify_rank_projection(
    payload: Mapping[str, Any],
    session_wire: Mapping[str, Any],
) -> None:
    summary = {
        "session_id": session_wire.get("session_id"),
        "session_digest": session_wire.get("session_digest"),
        "mandate_id": session_wire.get("mandate_id"),
        "issued_at_tick": session_wire.get("issued_at_tick"),
        "expires_at_tick": session_wire.get("expires_at_tick"),
    }
    _require_fields(payload, summary, "rank response summary")
    candidates = payload.get("candidates")
    offers = session_wire.get("offers")
    if not isinstance(candidates, list) or not isinstance(offers, list):
        raise ExactJoinError("rank response candidates are invalid")
    if len(candidates) != len(offers):
        raise ExactJoinError("rank response candidate count changed")
    authoritative_fields = (
        "offer_id",
        "session_id",
        "buyer_id",
        "mandate_id",
        "merchant_id",
        "sku_id",
        "unit_price_cents",
        "currency",
        "qty",
        "catalog_revision",
        "inventory_revision",
        "issued_at_tick",
        "expires_at_tick",
        "offer_digest",
    )
    for candidate, offer in zip(candidates, offers, strict=True):
        if not isinstance(candidate, Mapping) or not isinstance(offer, Mapping):
            raise ExactJoinError("rank response candidate is invalid")
        if any(candidate.get(field) != offer.get(field) for field in authoritative_fields):
            raise ExactJoinError("rank response candidate differs from its session offer")


def _verify_response_identity(
    exchange: LinkedPlatformExchange,
    response: Mapping[str, Any],
    *,
    service_id: str,
    recipient_id: str,
    require_request_idempotency: bool = True,
) -> None:
    request = exchange.request
    if (
        response.get("from") != service_id
        or response.get("to") != recipient_id
        or response.get("in_reply_to") != request.get("msg_id")
        or (
            require_request_idempotency
            and response.get("idempotency_key") != request.get("idempotency_key")
        )
    ):
        raise ExactJoinError("Platform response identity changed")


def _one_response(
    exchange: LinkedPlatformExchange,
    kind: str,
) -> dict[str, Any]:
    matches = tuple(
        response
        for response in exchange.responses
        if (response.get("action") or {}).get("kind") == kind
    )
    if len(matches) != 1 or len(exchange.responses) != 1:
        raise ExactJoinError(f"accepted request has no unique {kind!r} response")
    return matches[0]


def _platform_response_wire(
    request: Mapping[str, Any],
    *,
    msg_id: str,
    service_id: str,
    recipient_id: str,
    idempotency_key: Any,
    kind: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "msg_id": msg_id,
        "ts": request.get("ts"),
        "from": service_id,
        "to": recipient_id,
        "in_reply_to": request.get("msg_id"),
        "idempotency_key": idempotency_key,
        "signature": None,
        "action": {"kind": kind, "payload": dict(payload)},
    }


def _verify_expected_response_bundle(
    exchange: LinkedPlatformExchange,
    expected_responses: list[dict[str, Any]],
) -> None:
    """Pin produced responses to exact World-derived envelope bytes.

    Missing members are permitted only because the generic exact linker has
    already validated their ``not_audited_at_shutdown`` dispositions for a
    verified scoreable termination.  Reconstructing the complete bundle here
    prevents a forged id or arbitrary 64-character hash from receiving
    completed-prefix authority credit.
    """

    decision = exchange.decision
    expected_ids = [row["msg_id"] for row in expected_responses]
    expected_kinds = sorted(
        str((row.get("action") or {}).get("kind")) for row in expected_responses
    )
    expected_hashes = [wire_envelope_sha256(row) for row in expected_responses]
    if (
        decision.get("response_msg_ids") != expected_ids
        or sorted(decision.get("response_kinds") or []) != expected_kinds
        or decision.get("response_sha256s") != expected_hashes
    ):
        raise ExactJoinError("Platform response metadata changed")


def _operation_commits(
    context: ExactJoinContext,
    *,
    operation: str,
    authority_action: str,
) -> tuple[dict[str, Any], ...]:
    commits = tuple(
        commit
        for commit in context.world_commits
        if commit.get("operation") == operation
        or commit.get("authority_action") == authority_action
    )
    for commit in commits:
        if (
            commit.get("operation") != operation
            or commit.get("authority_action") != authority_action
        ):
            raise ExactJoinError(
                f"World commit partially matches {operation!r} authority"
            )
    commit_ids = [_text(commit.get("commit_id"), "World commit id") for commit in commits]
    if len(commit_ids) != len(set(commit_ids)):
        raise ExactJoinError(f"duplicate {operation!r} World commit id")
    return commits


def _exact_writes(commit: Mapping[str, Any], expected_count: int) -> list[dict[str, Any]]:
    writes = commit.get("table_writes")
    if not isinstance(writes, list) or len(writes) != expected_count or not all(
        isinstance(write, dict) for write in writes
    ):
        raise ExactJoinError("World commit table-write count changed")
    return writes


def _row_index(
    tables: Mapping[str, Any],
    table: str,
    key_field: str,
) -> dict[str, dict[str, Any]]:
    rows = tables.get(table, [])
    if not isinstance(rows, list):
        raise ExactJoinError(f"World table {table!r} is not a row array")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ExactJoinError(f"World table {table!r} has an invalid row")
        key = _text(row.get(key_field), f"{table}.{key_field}")
        if key in output:
            raise ExactJoinError(f"World table {table!r} has duplicate key {key!r}")
        output[key] = row
    return output


def _composite_row_index(
    tables: Mapping[str, Any],
    table: str,
    key_fields: tuple[str, str],
) -> dict[tuple[str, int], dict[str, Any]]:
    rows = tables.get(table, [])
    if not isinstance(rows, list):
        raise ExactJoinError(f"World table {table!r} is not a row array")
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ExactJoinError(f"World table {table!r} has an invalid row")
        key = (
            _text(row.get(key_fields[0]), f"{table}.{key_fields[0]}"),
            _tick(row.get(key_fields[1]), f"{table}.{key_fields[1]}"),
        )
        if key in output:
            raise ExactJoinError(f"World table {table!r} has duplicate composite key")
        output[key] = row
    return output


def _claim_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _strip_schema(value.get(key))
        for key in (
            "claim_id",
            "listing_id",
            "merchant_id",
            "subject",
            "issuer_id",
            "versions",
        )
    }


def _strip_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_schema(child)
            for key, child in value.items()
            if key != "schema_version"
        }
    if isinstance(value, (list, tuple)):
        return [_strip_schema(child) for child in value]
    return value


def _payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, dict):
        raise ExactJoinError("audited action payload must be an object")
    return payload


def _require_fields(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ExactJoinError(f"{label} mismatches {field!r}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExactJoinError(f"{label} must be non-empty text")
    return value


def _text_set(value: Any, label: str) -> frozenset[str]:
    if not isinstance(value, (tuple, list, set, frozenset)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ExactJoinError(f"{label} must be a text sequence")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ExactJoinError(f"{label} must not contain duplicates")
    return frozenset(result)


def _tick(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError(f"{label} must be a non-negative integer")
    return value


def _commit_sequence(commit: Mapping[str, Any]) -> int:
    value = commit.get("sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError("World commit sequence is invalid")
    return value


def _reject_options(options: Mapping[str, Any], label: str) -> None:
    if options:
        raise ExactJoinError(
            f"unknown {label} evidence options: " + ", ".join(sorted(options))
        )


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        SEARCH_SESSION_EVIDENCE_CONTRACT,
        verify_search_session_evidence,
    )
)
DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        LISTING_CLAIM_EVIDENCE_CONTRACT,
        verify_listing_claim_evidence,
    )
)
DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        EVIDENCE_RECORD_EVIDENCE_CONTRACT,
        verify_evidence_record_evidence,
    )
)
DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        MARKET_CLOCK_EVIDENCE_CONTRACT,
        verify_market_clock_evidence,
    )
)


__all__ = [
    "EVIDENCE_RECORD_EVIDENCE_CONTRACT",
    "LISTING_CLAIM_EVIDENCE_CONTRACT",
    "MARKET_CLOCK_EVIDENCE_CONTRACT",
    "SEARCH_SESSION_EVIDENCE_CONTRACT",
    "VerifiedEvidenceRecordEvidence",
    "VerifiedEvidenceRecordJoin",
    "VerifiedListingClaimEvidence",
    "VerifiedListingClaimJoin",
    "VerifiedMarketClockEvidence",
    "VerifiedMarketClockJoin",
    "VerifiedSearchSessionEvidence",
    "VerifiedSearchSessionJoin",
    "verify_evidence_record_evidence",
    "verify_listing_claim_evidence",
    "verify_market_clock_evidence",
    "verify_search_session_evidence",
]
