"""Exact Runtime, Platform, and World evidence for match certificates.

The contract in this module proves that a buyer's audited
``commerce.accept_offer`` request was accepted by the real Platform
aggregator, produced the exact audited certificate response, and committed
the same sealed acceptance and certificate atomically in World.  It never
reconstructs a successful outcome from scenario configuration or agent text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from protocol.matching import (
    MATCH_ACCEPTANCE_SCHEMA,
    OFFER_SNAPSHOT_SCHEMA,
    SEARCH_SESSION_SCHEMA,
    MatchAcceptance,
    MatchCertificate,
    SearchSession,
    coerce_match_acceptance,
    coerce_match_certificate,
    coerce_search_session,
    match_acceptance_to_wire,
    match_certificate_to_wire,
    search_session_to_wire,
    seal_match_acceptance,
    validate_match_certificate,
)
from runtime.exact_join import (
    DEFAULT_OPERATION_EVIDENCE_REGISTRY,
    ExactJoinContext,
    ExactJoinError,
    LinkedPlatformExchange,
    OperationEvidenceContract,
)


MATCH_CERTIFICATE_EVIDENCE_CONTRACT = "commerceworld.match-certificate.v1"

_COMMIT_INVARIANTS = (
    "persisted-session-membership",
    "exact-offer-binding",
    "current-revisions",
    "order-binding",
    "atomic-certificate-write",
)


@dataclass(frozen=True, slots=True)
class VerifiedMatchCertificateJoin:
    """One fresh World certificate and all exact accepted retries for it."""

    exchanges: tuple[LinkedPlatformExchange, ...]
    responses: tuple[dict[str, Any], ...]
    search_exchanges: tuple[LinkedPlatformExchange, ...]
    rank_responses: tuple[dict[str, Any], ...]
    session: SearchSession
    search_commit: dict[str, Any]
    acceptance_key: str
    acceptance: MatchAcceptance
    certificate: MatchCertificate
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedRejectedMatchRequest:
    """A rejected aggregator acceptance with no claimed World effect."""

    exchange: LinkedPlatformExchange
    reason_code: str


@dataclass(frozen=True, slots=True)
class VerifiedMatchCertificateEvidence:
    """Complete match-certificate authority graph for an evidence window."""

    certificates: tuple[VerifiedMatchCertificateJoin, ...]
    rejected_requests: tuple[VerifiedRejectedMatchRequest, ...]

    @property
    def certificate_ids(self) -> tuple[str, ...]:
        return tuple(row.certificate.cert_id for row in self.certificates)

    @property
    def order_ids(self) -> tuple[str, ...]:
        return tuple(row.certificate.order_id for row in self.certificates)

    @property
    def sku_ids(self) -> tuple[str, ...]:
        return tuple(row.certificate.sku_id for row in self.certificates)


def verify_match_certificate_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedMatchCertificateEvidence:
    """Verify every aggregator acceptance and certificate commit exactly."""

    allowed_options = {
        "expected_buyer_id",
        "expected_certificate_ids",
        "expected_order_ids",
        "expected_sku_id",
        "expected_sku_ids",
        "allow_rejected",
    }
    unknown = sorted(set(options) - allowed_options)
    if unknown:
        raise ExactJoinError(
            "unknown match-certificate evidence options: " + ", ".join(unknown)
        )
    expected_buyer = _optional_text(
        options.get("expected_buyer_id"), "expected_buyer_id"
    )
    expected_certificate_ids = _text_sequence(
        options.get("expected_certificate_ids", ()),
        "expected_certificate_ids",
    )
    expected_order_ids = _text_sequence(
        options.get("expected_order_ids", ()), "expected_order_ids"
    )
    expected_sku_id = _optional_text(
        options.get("expected_sku_id"), "expected_sku_id"
    )
    expected_sku_ids = _text_sequence(
        options.get("expected_sku_ids", ()), "expected_sku_ids"
    )
    allow_rejected = options.get("allow_rejected", True)
    if not isinstance(allow_rejected, bool):
        raise ExactJoinError("allow_rejected must be boolean")

    exchanges = tuple(
        row
        for row in context.exchanges
        if row.decision.get("action_kind") == "commerce.accept_offer"
        and row.decision.get("platform_endpoint") == "platform:aggregator"
    )
    initial_sessions = _row_array(
        context.initial_tables, "search_sessions", "session_id"
    )
    final_sessions = _row_array(
        context.final_tables, "search_sessions", "session_id"
    )
    initial_acceptances = _row_mapping(
        context.initial_tables, "match_acceptances"
    )
    final_acceptances = _row_mapping(context.final_tables, "match_acceptances")
    initial_certificates = _row_array(
        context.initial_tables, "match_certificates", "cert_id"
    )
    final_certificates = _row_array(
        context.final_tables, "match_certificates", "cert_id"
    )
    _require_immutable_prefix(initial_sessions, final_sessions, "search session")
    _require_immutable_prefix(
        initial_acceptances, final_acceptances, "match acceptance"
    )
    _require_immutable_prefix(
        initial_certificates, final_certificates, "match certificate"
    )

    typed_sessions = {
        key: _coerce_session(row) for key, row in final_sessions.items()
    }
    match_commits = tuple(
        row
        for row in context.world_commits
        if row.get("operation") == "issue_match_certificate"
        or row.get("authority_action") == "world.issue_match_certificate"
    )

    accepted_groups: dict[str, list[tuple[LinkedPlatformExchange, dict[str, Any]]]] = {}
    rejected: list[VerifiedRejectedMatchRequest] = []
    for exchange in exchanges:
        decision = exchange.decision
        buyer_id = _text(decision.get("actor_id"), "match buyer id")
        if buyer_id.split(":", 1)[0] != "buyer":
            raise ExactJoinError("match certificate request did not come from a buyer")
        key = _text(
            exchange.request.get("idempotency_key"),
            "match certificate idempotency key",
        )
        if decision.get("decision") == "rejected":
            if exchange.responses:
                raise ExactJoinError(
                    "rejected match certificate request emitted a response"
                )
            if not allow_rejected:
                raise ExactJoinError(
                    "selected match-certificate flow contains a rejection"
                )
            rejected.append(
                VerifiedRejectedMatchRequest(
                    exchange=exchange,
                    reason_code=_text(decision.get("reason_code"), "reason_code"),
                )
            )
            continue
        if decision.get("decision") != "accepted":
            raise ExactJoinError("match certificate Platform decision is invalid")
        response = _one_certificate_response(exchange)
        acceptance_key = _acceptance_key(buyer_id, key)
        accepted_groups.setdefault(acceptance_key, []).append((exchange, response))

    claimed_commits: set[str] = set()
    verified: list[VerifiedMatchCertificateJoin] = []
    for acceptance_key, retries in accepted_groups.items():
        acceptance_row = final_acceptances.get(acceptance_key)
        if acceptance_row is None:
            raise ExactJoinError(
                "accepted match certificate request has no final World acceptance"
            )
        acceptance = _coerce_acceptance(acceptance_row)
        session = typed_sessions.get(acceptance.session_id)
        if session is None:
            raise ExactJoinError("match acceptance has no final World search session")

        exchanges_for_join: list[LinkedPlatformExchange] = []
        responses_for_join: list[dict[str, Any]] = []
        search_exchanges_for_join: list[LinkedPlatformExchange] = []
        rank_responses_for_join: list[dict[str, Any]] = []
        certificates: list[MatchCertificate] = []
        for exchange, response in retries:
            search_exchange, rank_response = _verify_search_predecessor(
                context,
                accept_exchange=exchange,
                session=session,
            )
            derived = _derive_acceptance(exchange, session=session)
            if derived != acceptance:
                raise ExactJoinError(
                    "match acceptance differs from the audited Platform request"
                )
            certificate = _coerce_response_certificate(response)
            _verify_response_envelope(exchange, response, certificate)
            try:
                validate_match_certificate(
                    certificate,
                    session=session,
                    acceptance=acceptance,
                    expected_buyer_id=acceptance.buyer_id,
                    expected_order_id=acceptance.order_id,
                )
            except Exception as exc:
                raise ExactJoinError(
                    "Platform match certificate does not bind World state"
                ) from exc
            final_certificate = final_certificates.get(certificate.cert_id)
            if final_certificate != _without_schema(match_certificate_to_wire(certificate)):
                raise ExactJoinError(
                    "Platform match certificate differs from final World"
                )
            exchanges_for_join.append(exchange)
            responses_for_join.append(response)
            search_exchanges_for_join.append(search_exchange)
            rank_responses_for_join.append(rank_response)
            certificates.append(certificate)
        if any(row != certificates[0] for row in certificates[1:]):
            raise ExactJoinError("accepted match retry returned another certificate")
        certificate = certificates[0]

        search_matches = [
            row
            for row in context.world_commits
            if row.get("operation") == "create_search_session"
            and row.get("authority_action") == "world.create_search_session"
            and row.get("subject_id") == session.session_id
        ]
        if len(search_matches) != 1:
            raise ExactJoinError(
                "matched search session has no unique fresh World commit"
            )
        search_commit = search_matches[0]
        _verify_search_commit(
            search_commit,
            session=session,
            final_sessions=final_sessions,
        )

        matches = [
            row
            for row in match_commits
            if _commit_acceptance_key(row) == acceptance_key
        ]
        if len(matches) != 1:
            raise ExactJoinError(
                "accepted match certificate has no unique fresh World commit"
            )
        commit = matches[0]
        commit_id = _text(commit.get("commit_id"), "match commit id")
        if commit_id in claimed_commits:
            raise ExactJoinError("two match certificates claimed one World commit")
        claimed_commits.add(commit_id)
        _verify_commit(
            commit,
            acceptance_key=acceptance_key,
            acceptance=acceptance,
            certificate=certificate,
            final_acceptances=final_acceptances,
            final_certificates=final_certificates,
        )
        verified.append(
            VerifiedMatchCertificateJoin(
                exchanges=tuple(exchanges_for_join),
                responses=tuple(responses_for_join),
                search_exchanges=tuple(search_exchanges_for_join),
                rank_responses=tuple(rank_responses_for_join),
                session=session,
                search_commit=search_commit,
                acceptance_key=acceptance_key,
                acceptance=acceptance,
                certificate=certificate,
                commit=commit,
            )
        )

    unclaimed = [
        row
        for row in match_commits
        if row.get("commit_id") not in claimed_commits
    ]
    if unclaimed:
        raise ExactJoinError(
            "match-certificate evidence contains an unclaimed World operation"
        )
    _verify_final_delta(
        initial_acceptances=initial_acceptances,
        final_acceptances=final_acceptances,
        initial_certificates=initial_certificates,
        final_certificates=final_certificates,
        commits=match_commits,
    )

    verified.sort(key=lambda row: _commit_sequence(row.commit))
    evidence = VerifiedMatchCertificateEvidence(
        certificates=tuple(verified), rejected_requests=tuple(rejected)
    )
    if (
        expected_buyer is not None
        and not any(
            row.certificate.buyer_id == expected_buyer for row in evidence.certificates
        )
    ):
        raise ExactJoinError("selected buyer has no verified match certificate")
    if (
        expected_certificate_ids
        and evidence.certificate_ids != expected_certificate_ids
    ):
        raise ExactJoinError("match certificate id sequence changed")
    if expected_order_ids and evidence.order_ids != expected_order_ids:
        raise ExactJoinError("match certificate order id sequence changed")
    if expected_sku_id is not None and not any(
        row.certificate.sku_id == expected_sku_id for row in evidence.certificates
    ):
        raise ExactJoinError("selected sku has no verified match certificate")
    if expected_sku_ids and evidence.sku_ids != expected_sku_ids:
        raise ExactJoinError("match certificate sku sequence changed")
    return evidence


def _derive_acceptance(
    exchange: LinkedPlatformExchange,
    *,
    session: SearchSession,
) -> MatchAcceptance:
    request = exchange.request
    actor = _text(request.get("from"), "match request actor")
    key = _text(request.get("idempotency_key"), "match request idempotency key")
    payload = _payload(request)
    offer_id = _text(payload.get("offer_id"), "accepted offer id")
    try:
        if "session_id" in payload:
            required = {
                "session_id",
                "session_digest",
                "offer_id",
                "offer_digest",
                "mandate_id",
                "order_id",
                "merchant_id",
                "sku_id",
                "unit_price_cents",
                "currency",
                "qty",
                "catalog_revision",
                "inventory_revision",
            }
            if set(payload) != required:
                raise ExactJoinError(
                    "strict match acceptance fields changed"
                )
            if str(payload["session_id"]) != session.session_id:
                raise ExactJoinError("match request names another search session")
            acceptance = MatchAcceptance(
                request_msg_id=key,
                idempotency_key=key,
                session_id=str(payload["session_id"]),
                session_digest=str(payload["session_digest"]),
                offer_id=offer_id,
                offer_digest=str(payload["offer_digest"]),
                buyer_id=actor,
                mandate_id=str(payload["mandate_id"]),
                order_id=str(payload["order_id"]),
                merchant_id=str(payload["merchant_id"]),
                sku_id=str(payload["sku_id"]),
                unit_price_cents=int(payload["unit_price_cents"]),
                currency=str(payload["currency"]),
                qty=int(payload["qty"]),
                catalog_revision=int(payload["catalog_revision"]),
                inventory_revision=int(payload["inventory_revision"]),
            )
        else:
            if set(payload) not in (
                {"offer_id"},
                {"offer_id", "mandate_id"},
                {"offer_id", "sku_id"},
                {"offer_id", "mandate_id", "sku_id"},
            ):
                raise ExactJoinError("legacy match acceptance fields changed")
            offers = [row for row in session.offers if row.offer_id == offer_id]
            if len(offers) != 1:
                raise ExactJoinError(
                    "legacy match request does not identify one session offer"
                )
            offer = offers[0]
            supplied_sku = payload.get("sku_id")
            if supplied_sku is not None and str(supplied_sku) != offer.sku_id:
                raise ExactJoinError(
                    "legacy match request sku differs from its session offer"
                )
            supplied_mandate = payload.get("mandate_id")
            order_mandate = str(supplied_mandate or session.mandate_id)
            acceptance = MatchAcceptance(
                request_msg_id=key,
                idempotency_key=key,
                session_id=session.session_id,
                session_digest=session.session_digest,
                offer_id=offer.offer_id,
                offer_digest=offer.offer_digest,
                buyer_id=actor,
                mandate_id=session.mandate_id,
                order_id=f"ord-{order_mandate}-{offer.offer_id}",
                merchant_id=offer.merchant_id,
                sku_id=offer.sku_id,
                unit_price_cents=offer.unit_price_cents,
                currency=offer.currency,
                qty=offer.qty,
                catalog_revision=offer.catalog_revision,
                inventory_revision=offer.inventory_revision,
            )
        return seal_match_acceptance(acceptance)
    except ExactJoinError:
        raise
    except Exception as exc:
        raise ExactJoinError("audited match acceptance is not exact") from exc


def _verify_search_predecessor(
    context: ExactJoinContext,
    *,
    accept_exchange: LinkedPlatformExchange,
    session: SearchSession,
) -> tuple[LinkedPlatformExchange, dict[str, Any]]:
    response_id = _text(
        accept_exchange.request.get("in_reply_to"),
        "match acceptance rank predecessor",
    )
    buyer_id = _text(accept_exchange.request.get("from"), "match buyer id")
    matches: list[tuple[LinkedPlatformExchange, dict[str, Any]]] = []
    for exchange in context.exchanges:
        decision = exchange.decision
        if (
            decision.get("action_kind") != "commerce.search"
            or decision.get("platform_endpoint") != "platform:aggregator"
            or decision.get("actor_id") != buyer_id
            or decision.get("decision") != "accepted"
        ):
            continue
        for response in exchange.responses:
            if (
                response.get("msg_id") == response_id
                and (response.get("action") or {}).get("kind")
                == "platform.rank_offers"
            ):
                matches.append((exchange, response))
    if len(matches) != 1:
        raise ExactJoinError(
            "match acceptance has no unique accepted search predecessor"
        )
    exchange, response = matches[0]
    request = exchange.request
    if (
        response.get("from") != "platform:aggregator"
        or response.get("to") != buyer_id
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
    ):
        raise ExactJoinError("rank predecessor envelope identity changed")
    payload = _payload(response)
    if payload.get("search_session") != search_session_to_wire(session):
        raise ExactJoinError(
            "rank predecessor search session differs from final World"
        )
    required = {
        "session_id": session.session_id,
        "session_digest": session.session_digest,
        "mandate_id": session.mandate_id,
        "issued_at_tick": session.issued_at_tick,
        "expires_at_tick": session.expires_at_tick,
    }
    if any(payload.get(name) != value for name, value in required.items()):
        raise ExactJoinError("rank predecessor session summary changed")
    if (
        session.buyer_id != buyer_id
        or session.search_request_id != request.get("idempotency_key")
        or session.search_idempotency_key != request.get("idempotency_key")
    ):
        raise ExactJoinError("search session does not bind its Platform request")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(session.offers):
        raise ExactJoinError("rank predecessor candidate count changed")
    for candidate, offer in zip(candidates, session.offers, strict=True):
        if not isinstance(candidate, Mapping):
            raise ExactJoinError("rank predecessor candidate is invalid")
        authoritative = {
            "offer_id": offer.offer_id,
            "session_id": offer.session_id,
            "buyer_id": offer.buyer_id,
            "mandate_id": offer.mandate_id,
            "merchant_id": offer.merchant_id,
            "sku_id": offer.sku_id,
            "unit_price_cents": offer.unit_price_cents,
            "currency": offer.currency,
            "qty": offer.qty,
            "catalog_revision": offer.catalog_revision,
            "inventory_revision": offer.inventory_revision,
            "issued_at_tick": offer.issued_at_tick,
            "expires_at_tick": offer.expires_at_tick,
            "offer_digest": offer.offer_digest,
        }
        if any(candidate.get(name) != value for name, value in authoritative.items()):
            raise ExactJoinError("rank predecessor candidate changed")
    return exchange, response


def _verify_search_commit(
    commit: Mapping[str, Any],
    *,
    session: SearchSession,
    final_sessions: Mapping[str, dict[str, Any]],
) -> None:
    required = {
        "commit_kind": "transaction",
        "operation": "create_search_session",
        "authority_action": "world.create_search_session",
        "actor_id": "platform:aggregator",
        "idempotency_key": session.search_idempotency_key,
        "subject_id": session.session_id,
        "request_fingerprint": None,
    }
    if any(commit.get(name) != value for name, value in required.items()):
        raise ExactJoinError("search session World commit identity changed")
    invariants = commit.get("invariants_held")
    expected_invariants = [
        "server-authored-session",
        "catalog-revision-bound",
        "inventory-revision-bound",
        "buyer-scoped-idempotency",
    ]
    if invariants != expected_invariants:
        raise ExactJoinError("search session World invariants changed")
    writes = _writes(commit)
    session_wire = _search_session_snapshot_wire(session)
    if len(writes) != 1:
        raise ExactJoinError("search session commit write count changed")
    write = writes[0]
    if (
        write.get("table") != "search_sessions"
        or write.get("key") != session.session_id
        or write.get("op") != "create"
        or write.get("before") is not None
        or write.get("after") != session_wire
        or final_sessions.get(session.session_id) != session_wire
    ):
        raise ExactJoinError("search session commit differs from final World")


def _verify_commit(
    commit: Mapping[str, Any],
    *,
    acceptance_key: str,
    acceptance: MatchAcceptance,
    certificate: MatchCertificate,
    final_acceptances: Mapping[str, dict[str, Any]],
    final_certificates: Mapping[str, dict[str, Any]],
) -> None:
    required = {
        "commit_kind": "transaction",
        "operation": "issue_match_certificate",
        "authority_action": "world.issue_match_certificate",
        "actor_id": "platform:aggregator",
        "idempotency_key": acceptance.idempotency_key,
        "subject_id": certificate.cert_id,
        "request_fingerprint": None,
    }
    if any(commit.get(name) != value for name, value in required.items()):
        raise ExactJoinError("match certificate World commit identity changed")
    invariants = commit.get("invariants_held")
    if not isinstance(invariants, list) or tuple(invariants) != _COMMIT_INVARIANTS:
        raise ExactJoinError("match certificate World invariants changed")
    writes = _writes(commit)
    if len(writes) != 2:
        raise ExactJoinError("match certificate commit is not atomic")
    acceptance_wire = _without_schema(match_acceptance_to_wire(acceptance))
    certificate_wire = _without_schema(match_certificate_to_wire(certificate))
    expected = (
        ("match_acceptances", acceptance_key, acceptance_wire),
        ("match_certificates", certificate.cert_id, certificate_wire),
    )
    for write, (table, key, after) in zip(writes, expected, strict=True):
        if (
            write.get("table") != table
            or write.get("key") != key
            or write.get("op") != "create"
            or write.get("before") is not None
            or write.get("after") != after
        ):
            raise ExactJoinError("match certificate commit write changed")
    if final_acceptances.get(acceptance_key) != acceptance_wire:
        raise ExactJoinError("match acceptance commit is absent from final World")
    if final_certificates.get(certificate.cert_id) != certificate_wire:
        raise ExactJoinError("match certificate commit is absent from final World")


def _verify_response_envelope(
    exchange: LinkedPlatformExchange,
    response: Mapping[str, Any],
    certificate: MatchCertificate,
) -> None:
    request = exchange.request
    if (
        response.get("from") != "platform:aggregator"
        or response.get("to") != request.get("from")
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
        or _payload(response) != match_certificate_to_wire(certificate)
    ):
        raise ExactJoinError("match certificate response envelope is forged")


def _one_certificate_response(
    exchange: LinkedPlatformExchange,
) -> dict[str, Any]:
    matches = [
        row
        for row in exchange.responses
        if (row.get("action") or {}).get("kind")
        == "platform.create_match_certificate"
    ]
    if len(matches) != 1 or len(exchange.responses) != 1:
        raise ExactJoinError(
            "accepted match request needs one match certificate response"
        )
    return matches[0]


def _coerce_response_certificate(response: Mapping[str, Any]) -> MatchCertificate:
    try:
        return coerce_match_certificate(_payload(response))
    except Exception as exc:
        raise ExactJoinError("Platform match certificate failed strict codec") from exc


def _coerce_session(row: Mapping[str, Any]) -> SearchSession:
    try:
        raw_offers = row.get("offers")
        if not isinstance(raw_offers, list):
            raise ExactJoinError("World search session offers are not an array")
        offers = []
        for offer in raw_offers:
            if not isinstance(offer, Mapping):
                raise ExactJoinError("World search session offer is not an object")
            offers.append(
                {"schema_version": OFFER_SNAPSHOT_SCHEMA, **dict(offer)}
            )
        return coerce_search_session(
            {
                "schema_version": SEARCH_SESSION_SCHEMA,
                **dict(row),
                "offers": offers,
            }
        )
    except Exception as exc:
        raise ExactJoinError("World search session failed strict codec") from exc


def _coerce_acceptance(row: Mapping[str, Any]) -> MatchAcceptance:
    try:
        return coerce_match_acceptance(
            {"schema_version": MATCH_ACCEPTANCE_SCHEMA, **dict(row)}
        )
    except Exception as exc:
        raise ExactJoinError("World match acceptance failed strict codec") from exc


def _commit_acceptance_key(commit: Mapping[str, Any]) -> str | None:
    matches = [
        row
        for row in _writes(commit)
        if row.get("table") == "match_acceptances"
    ]
    if len(matches) != 1:
        return None
    key = matches[0].get("key")
    return key if isinstance(key, str) and key else None


def _verify_final_delta(
    *,
    initial_acceptances: Mapping[str, dict[str, Any]],
    final_acceptances: Mapping[str, dict[str, Any]],
    initial_certificates: Mapping[str, dict[str, Any]],
    final_certificates: Mapping[str, dict[str, Any]],
    commits: tuple[dict[str, Any], ...],
) -> None:
    committed_acceptances: dict[str, dict[str, Any]] = {}
    committed_certificates: dict[str, dict[str, Any]] = {}
    for commit in commits:
        for write in _writes(commit):
            table = write.get("table")
            key = write.get("key")
            after = write.get("after")
            if not isinstance(key, str) or not isinstance(after, Mapping):
                raise ExactJoinError("match commit has an invalid final row")
            target = (
                committed_acceptances
                if table == "match_acceptances"
                else committed_certificates
                if table == "match_certificates"
                else None
            )
            if target is None:
                raise ExactJoinError("match commit wrote an unrelated World table")
            if key in target:
                raise ExactJoinError("duplicate match key in World commits")
            target[key] = _object(after, "match commit row")
    new_acceptances = {
        key: row
        for key, row in final_acceptances.items()
        if key not in initial_acceptances
    }
    new_certificates = {
        key: row
        for key, row in final_certificates.items()
        if key not in initial_certificates
    }
    if new_acceptances != committed_acceptances:
        raise ExactJoinError("final match acceptance delta differs from World commits")
    if new_certificates != committed_certificates:
        raise ExactJoinError("final match certificate delta differs from World commits")


def _row_array(
    tables: Mapping[str, Any], table: str, key_field: str
) -> dict[str, dict[str, Any]]:
    raw = tables.get(table, [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ExactJoinError(f"World table {table!r} must be a row array")
    result: dict[str, dict[str, Any]] = {}
    for position, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"World table {table!r} row[{position}] is invalid")
        row = _object(value, f"{table}[{position}]")
        key = row.get(key_field)
        if not isinstance(key, str) or not key:
            raise ExactJoinError(f"World table {table!r} row has no {key_field!r}")
        if key in result:
            raise ExactJoinError(f"World table {table!r} has duplicate key {key!r}")
        result[key] = row
    return result


def _row_mapping(
    tables: Mapping[str, Any], table: str
) -> dict[str, dict[str, Any]]:
    raw = tables.get(table, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ExactJoinError(f"World table {table!r} must be a keyed object")
    result: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key or not isinstance(value, Mapping):
            raise ExactJoinError(f"World table {table!r} has an invalid row")
        result[key] = _object(value, f"{table}[{key!r}]")
    return result


def _require_immutable_prefix(
    initial: Mapping[str, dict[str, Any]],
    final: Mapping[str, dict[str, Any]],
    label: str,
) -> None:
    for key, row in initial.items():
        if final.get(key) != row:
            raise ExactJoinError(f"pre-existing {label} changed")


def _writes(commit: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = commit.get("table_writes")
    if not isinstance(raw, list) or not raw:
        raise ExactJoinError("match World commit has no table writes")
    writes: list[dict[str, Any]] = []
    for position, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ExactJoinError(
                f"match table write[{position}] must be an object"
            )
        writes.append(_object(row, f"match table write[{position}]"))
    return writes


def _payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExactJoinError("match action payload must be an object")
    return _object(payload, "match action payload")


def _without_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    row.pop("schema_version", None)
    return row


def _search_session_snapshot_wire(session: SearchSession) -> dict[str, Any]:
    """Return the generic dataclass form used by snapshot and commit export."""

    row = _without_schema(search_session_to_wire(session))
    offers = row.get("offers")
    if not isinstance(offers, list):  # pragma: no cover - strict codec guarantees it
        raise ExactJoinError("search session wire offers are invalid")
    row["offers"] = [
        _without_schema(offer)
        for offer in offers
        if isinstance(offer, Mapping)
    ]
    if len(row["offers"]) != len(offers):
        raise ExactJoinError("search session wire contains an invalid offer")
    return row


def _acceptance_key(buyer_id: str, idempotency_key: str) -> str:
    return f"{buyer_id}\x1f{idempotency_key}"


def _commit_sequence(commit: Mapping[str, Any]) -> int:
    value = commit.get("sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError("match World commit sequence is invalid")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExactJoinError(f"{label} must be non-empty")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _text_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(row, str) and row for row in value
    ):
        raise ExactJoinError(f"{label} must be a string sequence")
    return tuple(value)


def _object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        result = json.loads(
            json.dumps(
                dict(value), allow_nan=False, separators=(",", ":"), sort_keys=True
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExactJoinError(f"{label} is not canonical JSON") from exc
    if not isinstance(result, dict):
        raise ExactJoinError(f"{label} must be an object")
    return result


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
        verify_match_certificate_evidence_contract,
    )
)


__all__ = [
    "MATCH_CERTIFICATE_EVIDENCE_CONTRACT",
    "VerifiedMatchCertificateEvidence",
    "VerifiedMatchCertificateJoin",
    "VerifiedRejectedMatchRequest",
    "verify_match_certificate_evidence_contract",
]
