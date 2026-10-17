"""Exact Runtime, Platform, and World evidence for market governance.

This module is a reusable CommerceWorld authority contract.  It does not know
about benchmark task identifiers or scoring lanes.  It joins authenticated
governance requests to Platform decisions and responses, then to the unique
World authority operation, atomic commit, and canonical governance envelope.

The episode evidence manifest has already proved byte integrity and replay
before this verifier is called.  The checks here add the semantic guarantees
needed by a deterministic scorer: no response is accepted as truth, no World
commit may be claimed twice, no unclaimed governance commit may remain, and
every stable id, version, semantic digest, envelope digest, projection, and
final snapshot row must agree exactly.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from protocol.market_governance import (
    Campaign,
    GovernanceCase,
    MarketSignal,
    RankingContext,
    RemediationPlan,
    ReputationEvent,
    ReviewAggregate,
    ReviewEvidence,
)
from protocol.evidence_records import coerce_evidence_record, evidence_record_to_dict
from protocol.matching import (
    coerce_search_session,
    offer_snapshot_to_wire,
    search_session_to_wire,
)
from protocol.remediation_audit import (
    REMEDIATION_AUDITOR_SERVICE_ID,
    build_remediation_audit_request,
)
from runtime.exact_join import (
    DEFAULT_OPERATION_EVIDENCE_REGISTRY,
    ExactJoinContext,
    ExactJoinError,
    LinkedPlatformExchange,
    OperationEvidenceContract,
    wire_envelope_sha256,
)
from world.evidence_contracts import authority_operation_key
from world.market_governance_core import (
    AdsCampaignTerms,
    GovernanceResponseAttestation,
    GovernanceResolutionDecision,
    RemediationBlueprint,
    ReputationPolicyRevision,
    ReviewAccountBinding,
    governance_intent_fingerprint,
    normalize_governance_intent,
)
from world.market_governance_persistence import (
    GovernanceEnvelope,
    envelope_key,
    envelope_version,
    policy_envelope_from_wire,
    policy_envelope_to_wire,
    policy_payload_to_wire,
    record_envelope_from_wire,
    record_envelope_to_wire,
)
from world.market_governance_service import command_fingerprint
from world.market_governance_world import market_governance_request_fingerprint


MARKET_GOVERNANCE_EVIDENCE_CONTRACT = "commerceworld.market-governance.v1"
MARKET_GOVERNANCE_RESPONSE_KIND = "platform.governance_updated"
MARKET_GOVERNANCE_READ_RESPONSE_KIND = "platform.governance_snapshot"

ACTOR_ACTIONS: Mapping[str, tuple[str, str, str]] = {
    "commerce.publish_campaign": (
        "publish_campaign",
        "platform:ads",
        "merchant",
    ),
    "commerce.disclose_placement": (
        "disclose_placement",
        "platform:ads",
        "merchant",
    ),
    "commerce.activate_campaign": (
        "activate_campaign",
        "platform:ads",
        "merchant",
    ),
    "commerce.submit_review": (
        "submit_review",
        "platform:reviews",
        "buyer",
    ),
    "commerce.reject_review_manipulation": (
        "reject_review_manipulation",
        "platform:governance",
        "merchant",
    ),
    "commerce.reject_coordination": (
        "reject_coordination",
        "platform:governance",
        "merchant",
    ),
    "commerce.accept_remediation_plan": (
        "accept_remediation_plan",
        "platform:remediation",
        "merchant",
    ),
    "commerce.complete_remediation_step": (
        "complete_remediation_step",
        "platform:remediation",
        "merchant",
    ),
}

TRUSTED_ACTIONS: Mapping[str, tuple[str, str, frozenset[str]]] = {
    "platform.aggregate_reviews": (
        "aggregate_reviews",
        "platform:reviews",
        frozenset({"runtime:reviews", "platform:orchestrator"}),
    ),
    "platform.ingest_review_observation": (
        "ingest_review_observation",
        "platform:reviews",
        frozenset({"runtime:reviews"}),
    ),
    "platform.ingest_market_observation": (
        "ingest_market_observation",
        "platform:governance",
        frozenset({"runtime:governance"}),
    ),
    "platform.resolve_governance_case": (
        "resolve_governance_case",
        "platform:governance",
        frozenset({"runtime:governance", "platform:orchestrator"}),
    ),
    "platform.apply_governance_reputation": (
        "apply_governance_reputation",
        "platform:reputation",
        frozenset({"runtime:reputation", "platform:orchestrator"}),
    ),
    "platform.create_remediation_plan": (
        "create_remediation_plan",
        "platform:remediation",
        frozenset({"runtime:remediation", "platform:orchestrator"}),
    ),
    "platform.verify_remediation_step": (
        "verify_remediation_step",
        "platform:remediation",
        frozenset({"runtime:remediation", "platform:orchestrator"}),
    ),
    "platform.persist_ranking_context": (
        "persist_ranking_context",
        "platform:ranking",
        frozenset({"platform:ranking"}),
    ),
}

_POLICY_SERVICE_BY_KIND: Mapping[str, str] = {
    "ads_campaign_terms": "platform:ads",
    "review_account_binding": "platform:reviews",
    "reputation_policy_revision": "platform:reputation",
    "remediation_blueprint": "platform:remediation",
}

_REQUIRED_INVARIANTS = frozenset(
    {
        "governance-context-cas",
        "typed-governance-records",
        "actor-scoped-authority",
        "atomic-projection",
        "single-clock-advance",
    }
)

_GOVERNANCE_ROWS_BY_OPERATION: Mapping[str, int] = {
    "publish_governance_policy": 1,
    "publish_campaign": 1,
    "disclose_placement": 1,
    "activate_campaign": 1,
    "submit_review": 1,
    "reject_review_manipulation": 1,
    "reject_coordination": 1,
    "accept_remediation_plan": 1,
    "complete_remediation_step": 1,
    "aggregate_reviews": 1,
    "ingest_review_observation": 1,
    "ingest_market_observation": 2,
    "resolve_governance_case": 2,
    "apply_governance_reputation": 1,
    "create_remediation_plan": 1,
    "verify_remediation_step": 1,
    "persist_ranking_context": 1,
}

_PROJECTION_BY_OPERATION: Mapping[str, str] = {
    "submit_review": "reviews",
    "ingest_review_observation": "reviews",
    "apply_governance_reputation": "reputation",
}


@dataclass(frozen=True, slots=True)
class VerifiedGovernanceEnvelope:
    """One strict governance policy or record from the final World snapshot."""

    collection: str
    key: str
    kind: str
    stable_id: str
    version: int
    semantic_digest: str
    envelope_digest: str
    wire: dict[str, Any]
    typed: GovernanceEnvelope


@dataclass(frozen=True, slots=True)
class VerifiedMarketGovernanceOperation:
    """One exact World governance effect caused by a Platform request."""

    operation: str
    service_actor: str
    original_actor: str
    request: dict[str, Any]
    request_fingerprint: str
    authority_operation: dict[str, Any]
    primary_result: VerifiedGovernanceEnvelope
    result_rows: tuple[VerifiedGovernanceEnvelope, ...]
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedMarketGovernanceRequest:
    """One accepted request and all World operations it uniquely owns."""

    exchange: LinkedPlatformExchange
    response: dict[str, Any] | None
    actor_request: bool
    operations: tuple[VerifiedMarketGovernanceOperation, ...]


@dataclass(frozen=True, slots=True)
class VerifiedRejectedMarketGovernanceRequest:
    """One rejected request proven to have no matching World effect."""

    exchange: LinkedPlatformExchange
    operation: str | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class VerifiedMarketGovernanceRead:
    """One caller-scoped history response matched to World envelopes."""

    exchange: LinkedPlatformExchange
    actor_id: str
    record_kind: str
    stable_id: str
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedMarketGovernanceEvidence:
    """Complete selected market-governance authority graph."""

    requests: tuple[VerifiedMarketGovernanceRequest, ...]
    rejected_requests: tuple[VerifiedRejectedMarketGovernanceRequest, ...] = ()
    reads: tuple[VerifiedMarketGovernanceRead, ...] = ()

    @property
    def operation_evidence(self) -> tuple[VerifiedMarketGovernanceOperation, ...]:
        values = [
            operation
            for request in self.requests
            for operation in request.operations
        ]
        return tuple(sorted(values, key=lambda row: _commit_sequence(row.commit)))

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(row.operation for row in self.operation_evidence)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_requests)

    def requests_for_actor(
        self, actor_id: str
    ) -> tuple[VerifiedMarketGovernanceRequest, ...]:
        return tuple(
            row
            for row in self.requests
            if row.exchange.decision.get("actor_id") == actor_id
        )


@dataclass(frozen=True, slots=True)
class _RequestContract:
    action_kind: str
    operation: str
    service_actor: str
    original_actor: str
    request: dict[str, Any]
    idempotency_key: str
    scope: str
    outer_fingerprint: str
    inner_fingerprint: str
    actor_request: bool


def _has_ranking_followup(exchange: LinkedPlatformExchange) -> bool:
    if len(exchange.responses) != 1:
        return False
    action = exchange.responses[0].get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    return (
        isinstance(action, Mapping)
        and action.get("kind") == "platform.rank_offers"
        and isinstance(payload, Mapping)
        and payload.get("ranking_context_reference") is not None
    )


def _search_followup_contract(
    context: ExactJoinContext,
    *,
    exchange: LinkedPlatformExchange,
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
) -> _RequestContract:
    decision = exchange.decision
    actor_id = _text(decision.get("actor_id"), "search actor_id")
    if (
        actor_id.split(":", 1)[0] != "buyer"
        or decision.get("platform_endpoint") != "platform:aggregator"
    ):
        raise ExactJoinError("governed search has the wrong actor or endpoint")
    key = "ranking-context:" + _text(
        exchange.request.get("idempotency_key"), "search idempotency_key"
    )
    candidates = [
        row
        for row in context.world_commits
        if row.get("commit_kind") == "transaction"
        and row.get("operation") == "persist_ranking_context"
        and row.get("authority_action") == "world.persist_ranking_context"
        and row.get("actor_id") == actor_id
        and row.get("idempotency_key") == key
    ]
    if len(candidates) != 1:
        raise ExactJoinError("governed search has no unique ranking follow-up")
    primary = _unverified_primary(candidates[0], final_rows=final_rows)
    ranking = primary.typed.payload
    if not isinstance(ranking, RankingContext):
        raise ExactJoinError("governed search outcome is not a RankingContext")

    response = exchange.responses[0]
    action = response.get("action")
    body = action.get("payload") if isinstance(action, Mapping) else None
    session_wire = body.get("search_session") if isinstance(body, Mapping) else None
    if not isinstance(session_wire, Mapping):
        raise ExactJoinError("governed search response has no SearchSession")
    try:
        session = coerce_search_session(session_wire)
    except Exception as exc:
        raise ExactJoinError("governed search SearchSession is invalid") from exc
    if search_session_to_wire(session) != session_wire:
        raise ExactJoinError("governed search SearchSession is not canonical")
    candidate_skus = tuple(offer.sku_id for offer in session.offers)
    request = {
        "request_id": session.session_id,
        "mandate_id": session.mandate_id,
        "candidate_sku_ids": list(candidate_skus),
        "ranked_sku_ids": list(candidate_skus),
        "policy_id": ranking.policy_id,
        "policy_version": ranking.policy_version,
    }
    if (
        ranking.request_id != session.session_id
        or ranking.buyer_id != actor_id
        or ranking.mandate_id != session.mandate_id
        or ranking.candidate_sku_ids != candidate_skus
        or ranking.ranked_sku_ids != candidate_skus
    ):
        raise ExactJoinError("RankingContext differs from its search session")
    service_actor = "platform:ranking"
    operation = "persist_ranking_context"
    return _RequestContract(
        action_kind="commerce.search",
        operation=operation,
        service_actor=service_actor,
        original_actor=actor_id,
        request=request,
        idempotency_key=key,
        scope=operation,
        outer_fingerprint=market_governance_request_fingerprint(
            operation, service_actor, actor_id, request
        ),
        inner_fingerprint=command_fingerprint(
            operation=operation,
            service_actor=service_actor,
            original_actor=actor_id,
            request=request,
        ),
        actor_request=True,
    )


def _unverified_primary(
    commit: Mapping[str, Any],
    *,
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
) -> VerifiedGovernanceEnvelope:
    authority_writes = [
        row
        for row in _commit_writes(commit)
        if row.get("table") == "authority_operations"
        and isinstance(row.get("after"), Mapping)
    ]
    if len(authority_writes) != 1:
        raise ExactJoinError("governance commit has no unique authority outcome")
    authority = authority_writes[0]["after"]
    table = _text(authority.get("outcome_table"), "outcome_table")
    key = _text(authority.get("outcome_key"), "outcome_key")
    row = final_rows.get((table, key))
    if row is None:
        raise ExactJoinError("governance authority outcome is absent from final World")
    return row


def verify_market_governance_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedMarketGovernanceEvidence:
    """Verify all selected governance exchanges and every governance commit."""

    allowed_options = {
        "expected_actor_id",
        "expected_operations",
        "expected_actor_operations",
        "expected_service_operations",
        "expected_primary_stable_ids",
        "expected_read_streams",
        "allow_rejected",
    }
    unknown = sorted(set(options) - allowed_options)
    if unknown:
        raise ExactJoinError(
            "unknown market-governance evidence options: " + ", ".join(unknown)
        )
    expected_actor = _optional_text(options.get("expected_actor_id"), "expected_actor_id")
    expected_operations = _text_sequence(
        options.get("expected_operations", ()), "expected_operations"
    )
    expected_actor_operations = _text_sequence(
        options.get("expected_actor_operations", ()),
        "expected_actor_operations",
    )
    expected_service_operations = _text_sequence(
        options.get("expected_service_operations", ()),
        "expected_service_operations",
    )
    expected_primary_stable_ids = _text_sequence(
        options.get("expected_primary_stable_ids", ()),
        "expected_primary_stable_ids",
    )
    expected_read_streams = _read_stream_sequence(
        options.get("expected_read_streams", ())
    )
    allow_rejected = options.get("allow_rejected", True)
    if not isinstance(allow_rejected, bool):
        raise ExactJoinError("allow_rejected must be boolean")

    action_kinds = {*ACTOR_ACTIONS, *TRUSTED_ACTIONS, "platform.publish_governance_policy"}
    exchanges = tuple(
        row
        for row in context.exchanges
        if row.decision.get("action_kind") in action_kinds
        or row.decision.get("action_kind") == "commerce.read_governance_history"
        or (
            row.decision.get("action_kind") == "commerce.search"
            and _has_ranking_followup(row)
        )
    )
    if not exchanges:
        raise ExactJoinError("market-governance evidence has no Platform exchange")

    initial_rows = _governance_rows(context.initial_tables)
    final_rows = _governance_rows(context.final_tables)
    for identity, row in initial_rows.items():
        if final_rows.get(identity) != row:
            raise ExactJoinError("pre-existing governance envelope changed")
    final_authority = _authority_operations(context.final_tables)

    accepted: list[VerifiedMarketGovernanceRequest] = []
    rejected: list[VerifiedRejectedMarketGovernanceRequest] = []
    read_exchanges: list[LinkedPlatformExchange] = []
    claimed_commits: set[str] = set()

    for exchange in exchanges:
        decision = exchange.decision
        action_kind = _text(decision.get("action_kind"), "Platform action_kind")
        if decision.get("actor_id") == "platform:orchestrator":
            _verify_internal_causal_request(context, exchange)
        if action_kind == "commerce.read_governance_history":
            read_exchanges.append(exchange)
            continue
        if action_kind == "commerce.search":
            if decision.get("decision") != "accepted":
                raise ExactJoinError("ranked governance search was not accepted")
            contract = _search_followup_contract(
                context,
                exchange=exchange,
                final_rows=final_rows,
            )
            matches = _matching_commits(context, contract)
            if len(matches) != 1:
                raise ExactJoinError(
                    "governed search has no unique ranking-context World commit"
                )
            commit = matches[0]
            commit_id = _text(commit.get("commit_id"), "commit_id")
            if commit_id in claimed_commits:
                raise ExactJoinError("two governance requests claimed one World commit")
            claimed_commits.add(commit_id)
            operation = _verify_commit(
                context,
                commit=commit,
                contract=contract,
                final_rows=final_rows,
                final_authority=final_authority,
            )
            response = _verify_search_response(
                context,
                exchange=exchange,
                operation=operation,
                final_rows=final_rows,
            )
            accepted.append(
                VerifiedMarketGovernanceRequest(
                    exchange=exchange,
                    response=response,
                    actor_request=True,
                    operations=(operation,),
                )
            )
            continue
        contract = _request_contract(exchange, context=context)
        if decision.get("decision") == "rejected":
            if exchange.responses:
                raise ExactJoinError("rejected governance request emitted a response")
            if not allow_rejected:
                raise ExactJoinError("selected governance flow contains a rejection")
            if _matching_commits(context, contract):
                raise ExactJoinError("rejected governance request has a World effect")
            rejected.append(
                VerifiedRejectedMarketGovernanceRequest(
                    exchange=exchange,
                    operation=contract.operation,
                    reason_code=_text(decision.get("reason_code"), "reason_code"),
                )
            )
            continue
        if decision.get("decision") != "accepted":
            raise ExactJoinError("governance Platform decision is invalid")

        matches = _matching_commits(context, contract)
        if len(matches) != 1:
            raise ExactJoinError("governance request has no unique World commit")
        commit = matches[0]
        commit_id = _text(commit.get("commit_id"), "commit_id")
        if commit_id in claimed_commits:
            raise ExactJoinError("two governance requests claimed one World commit")
        claimed_commits.add(commit_id)
        operation = _verify_commit(
            context,
            commit=commit,
            contract=contract,
            final_rows=final_rows,
            final_authority=final_authority,
        )
        response = _verify_response(
            context,
            exchange=exchange,
            operation=operation,
            final_rows=final_rows,
        )
        accepted.append(
            VerifiedMarketGovernanceRequest(
                exchange=exchange,
                response=response,
                actor_request=contract.actor_request,
                operations=(operation,),
            )
        )

    governance_commits = _governance_commits(context)
    unclaimed = [
        row
        for row in governance_commits
        if row.get("commit_id") not in claimed_commits
    ]
    if unclaimed:
        raise ExactJoinError(
            "market-governance evidence contains an unclaimed World operation"
        )
    _verify_final_governance_delta(
        initial_rows=initial_rows,
        final_rows=final_rows,
        commits=governance_commits,
    )
    _verify_projection_chains(context, governance_commits)

    reads = [
        _verify_read_response(
            exchange,
            initial_rows=initial_rows,
            accepted=accepted,
        )
        for exchange in read_exchanges
    ]

    accepted.sort(
        key=lambda row: min(
            _commit_sequence(operation.commit) for operation in row.operations
        )
    )
    observed_operations = tuple(
        row.operation
        for request in accepted
        for row in sorted(
            request.operations, key=lambda value: _commit_sequence(value.commit)
        )
    )
    if expected_operations and observed_operations != expected_operations:
        raise ExactJoinError(
            "market-governance operation sequence changed: "
            f"expected {expected_operations!r}, observed {observed_operations!r}"
        )
    primary_ids = tuple(
        row.primary_result.stable_id
        for request in accepted
        for row in sorted(
            request.operations, key=lambda value: _commit_sequence(value.commit)
        )
    )
    if (
        expected_primary_stable_ids
        and primary_ids != expected_primary_stable_ids
    ):
        raise ExactJoinError("governance primary stable-id sequence changed")

    if expected_actor is not None:
        actor_requests = [
            row
            for row in accepted
            if row.exchange.decision.get("actor_id") == expected_actor
        ]
        actor_operations = tuple(
            operation.operation
            for request in actor_requests
            for operation in request.operations
        )
        if not actor_requests and not any(row.actor_id == expected_actor for row in reads):
            raise ExactJoinError("selected actor has no governance request")
        if expected_actor_operations and actor_operations != expected_actor_operations:
            raise ExactJoinError("selected actor governance operation sequence changed")

    service_operations = tuple(
        operation.operation
        for request in accepted
        if not request.actor_request
        for operation in request.operations
    )
    if expected_service_operations and service_operations != expected_service_operations:
        raise ExactJoinError("trusted governance operation sequence changed")
    observed_read_streams = tuple(
        (row.record_kind, row.stable_id) for row in reads
    )
    if expected_read_streams and observed_read_streams != expected_read_streams:
        raise ExactJoinError("governance history read sequence changed")

    return VerifiedMarketGovernanceEvidence(
        requests=tuple(accepted),
        rejected_requests=tuple(rejected),
        reads=tuple(reads),
    )


def _verify_internal_causal_request(
    context: ExactJoinContext,
    exchange: LinkedPlatformExchange,
) -> None:
    """Bind one orchestrator request to the exact prior Platform response."""

    predecessors: list[LinkedPlatformExchange] = []
    for candidate in context.exchanges:
        for response, position in zip(
            candidate.responses, candidate.response_positions, strict=True
        ):
            if (
                position == exchange.request_position
                and response == exchange.request
            ):
                predecessors.append(candidate)
    if len(predecessors) != 1:
        raise ExactJoinError(
            "Platform orchestrator request has no unique causal response predecessor"
        )
    [predecessor] = predecessors
    if (
        exchange.request.get("in_reply_to")
        != predecessor.request.get("msg_id")
        or exchange.request_position <= predecessor.request_position
    ):
        raise ExactJoinError("Platform orchestrator causal reply link changed")


def _request_contract(
    exchange: LinkedPlatformExchange,
    *,
    context: ExactJoinContext,
) -> _RequestContract:
    decision = exchange.decision
    action_kind = _text(decision.get("action_kind"), "Platform action_kind")
    actor_id = _text(decision.get("actor_id"), "Platform actor_id")
    endpoint = _text(decision.get("platform_endpoint"), "Platform endpoint")
    idempotency_key = _text(
        exchange.request.get("idempotency_key"), "governance idempotency_key"
    )
    payload = _request_payload(exchange)

    actor = ACTOR_ACTIONS.get(action_kind)
    if actor is not None:
        operation, service_actor, required_role = actor
        if endpoint != service_actor:
            raise ExactJoinError("governance actor request targeted another service")
        if actor_id.split(":", 1)[0] != required_role:
            raise ExactJoinError("governance actor request has the wrong role")
        if "op" in payload:
            raise ExactJoinError("actor governance payload injected a World operation")
        try:
            request = dict(
                normalize_governance_intent({"op": operation, **payload})
            )
            inner = governance_intent_fingerprint(
                request,
                original_actor=actor_id,
                trusted_context_digests=(),
            )
        except Exception as exc:
            raise ExactJoinError("actor governance intent is not exact") from exc
        scope = f"apply_governance_intent:{operation}"
        outer = market_governance_request_fingerprint(
            scope, service_actor, actor_id, request
        )
        return _RequestContract(
            action_kind,
            operation,
            service_actor,
            actor_id,
            request,
            idempotency_key,
            scope,
            outer,
            inner,
            True,
        )

    if action_kind == "platform.publish_governance_policy":
        _require_exact_fields(payload, {"policy_intent"}, action_kind)
        request = _mapping(payload.get("policy_intent"), "policy_intent")
        kind = _text(request.get("kind"), "governance policy kind")
        try:
            service_actor = _POLICY_SERVICE_BY_KIND[kind]
        except KeyError as exc:
            raise ExactJoinError(f"unsupported governance policy kind {kind!r}") from exc
        if actor_id != "runtime:governance" or endpoint != service_actor:
            raise ExactJoinError("governance policy request authority changed")
        operation = "publish_governance_policy"
        scope = operation
        inner = command_fingerprint(
            operation=operation,
            service_actor=service_actor,
            original_actor=service_actor,
            request=request,
        )
        return _RequestContract(
            action_kind,
            operation,
            service_actor,
            service_actor,
            request,
            idempotency_key,
            scope,
            market_governance_request_fingerprint(
                operation, service_actor, service_actor, request
            ),
            inner,
            False,
        )

    trusted = TRUSTED_ACTIONS.get(action_kind)
    if trusted is None:
        raise ExactJoinError(f"unsupported governance action {action_kind!r}")
    operation, service_actor, requesters = trusted
    if actor_id not in requesters or endpoint != service_actor:
        raise ExactJoinError("trusted governance request authority changed")
    request, original_actor = _trusted_request(
        action_kind,
        payload,
        service_actor=service_actor,
        context=context,
        requester=actor_id,
    )
    scope = operation
    inner = command_fingerprint(
        operation=operation,
        service_actor=service_actor,
        original_actor=original_actor,
        request=request,
    )
    return _RequestContract(
        action_kind,
        operation,
        service_actor,
        original_actor,
        request,
        idempotency_key,
        scope,
        market_governance_request_fingerprint(
            operation, service_actor, original_actor, request
        ),
        inner,
        False,
    )


def _trusted_request(
    action_kind: str,
    payload: Mapping[str, Any],
    *,
    service_actor: str,
    context: ExactJoinContext,
    requester: str,
) -> tuple[dict[str, Any], str]:
    if action_kind == "platform.aggregate_reviews":
        _require_exact_fields(payload, {"sku_id"}, action_kind)
        request = {"sku_id": _text(payload.get("sku_id"), "sku_id")}
        return request, service_actor
    if action_kind == "platform.ingest_review_observation":
        _require_exact_fields(payload, {"record_id"}, action_kind)
        record_id = _text(payload.get("record_id"), "record_id")
        # The original actor is derived by Platform from the trusted World
        # EvidenceRecord.  It is never accepted from the request payload.
        return {"record_id": record_id}, _evidence_issuer(context, record_id)
    if action_kind == "platform.ingest_market_observation":
        allowed_shapes = (
            {"record_id", "detector_id"},
            {"record_id", "detector_id", "resolution_template"},
            {
                "record_id",
                "detector_id",
                "resolution_template",
                "remediation_template",
            },
        )
        if set(payload) not in allowed_shapes:
            raise ExactJoinError(
                "market observation causal template fields are not exact"
            )
        if "resolution_template" in payload:
            _resolution_template(payload.get("resolution_template"))
        if "remediation_template" in payload:
            _remediation_template(payload.get("remediation_template"))
        return (
            {"record_id": _text(payload.get("record_id"), "record_id")},
            _text(payload.get("detector_id"), "detector_id"),
        )
    if action_kind == "platform.resolve_governance_case":
        if set(payload) == {"decision_intent"}:
            pass
        elif (
            set(payload) == {"decision_intent", "remediation_template"}
            and requester == "platform:orchestrator"
        ):
            _remediation_template(payload.get("remediation_template"))
        else:
            raise ExactJoinError(
                "governance resolution causal template fields are not exact"
            )
        return _mapping(payload.get("decision_intent"), "decision_intent"), service_actor
    if action_kind == "platform.apply_governance_reputation":
        _require_exact_fields(payload, {"source_intent"}, action_kind)
        return _mapping(payload.get("source_intent"), "source_intent"), service_actor
    if action_kind == "platform.create_remediation_plan":
        _require_exact_fields(payload, {"plan_intent"}, action_kind)
        return _mapping(payload.get("plan_intent"), "plan_intent"), service_actor
    if action_kind == "platform.verify_remediation_step":
        _require_exact_fields(payload, {"plan_id", "step_id"}, action_kind)
        return (
            {
                "plan_id": _text(payload.get("plan_id"), "plan_id"),
                "step_id": _text(payload.get("step_id"), "step_id"),
            },
            service_actor,
        )
    if action_kind == "platform.persist_ranking_context":
        _require_exact_fields(payload, {"ranking_result", "buyer_id"}, action_kind)
        return (
            _mapping(payload.get("ranking_result"), "ranking_result"),
            _text(payload.get("buyer_id"), "buyer_id"),
        )
    raise ExactJoinError(f"unsupported trusted governance action {action_kind!r}")


def _matching_commits(
    context: ExactJoinContext,
    contract: _RequestContract,
) -> list[dict[str, Any]]:
    return [
        row
        for row in context.world_commits
        if _commit_matches(row, contract)
    ]


def _commit_matches(row: Mapping[str, Any], contract: _RequestContract) -> bool:
    return (
        row.get("commit_kind") == "transaction"
        and row.get("authority_action") == f"world.{contract.operation}"
        and row.get("operation") == contract.operation
        and row.get("actor_id") == contract.original_actor
        and row.get("idempotency_key") == contract.idempotency_key
        and row.get("request_fingerprint") == contract.outer_fingerprint
    )


def _verify_commit(
    context: ExactJoinContext,
    *,
    commit: Mapping[str, Any],
    contract: _RequestContract,
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
    final_authority: Mapping[str, Mapping[str, Any]],
) -> VerifiedMarketGovernanceOperation:
    key = _text(commit.get("idempotency_key"), "governance idempotency_key")
    if key != contract.idempotency_key:
        raise ExactJoinError("governance commit idempotency key changed")
    if commit.get("actor_id") != contract.original_actor:
        raise ExactJoinError("governance commit actor provenance changed")
    if commit.get("request_fingerprint") != contract.outer_fingerprint:
        raise ExactJoinError("governance commit request fingerprint changed")
    if commit.get("authority_action") != f"world.{contract.operation}":
        raise ExactJoinError("governance commit authority action changed")

    invariants = commit.get("invariants_held")
    if (
        not isinstance(invariants, list)
        or len(invariants) != len(_REQUIRED_INVARIANTS)
        or set(invariants) != _REQUIRED_INVARIANTS
    ):
        raise ExactJoinError("governance commit authority invariants changed")
    writes = _commit_writes(commit)
    expected_tables = Counter(
        {
            "governance_policies"
            if contract.operation == "publish_governance_policy"
            else "governance_records": _GOVERNANCE_ROWS_BY_OPERATION[
                contract.operation
            ],
            "authority_operations": 1,
            "logical_time": 1,
        }
    )
    projection = _PROJECTION_BY_OPERATION.get(contract.operation)
    if projection is not None:
        expected_tables[projection] = 1
    if Counter(_text(row.get("table"), "commit table") for row in writes) != expected_tables:
        raise ExactJoinError("governance commit write multiset changed")

    clocks = [row for row in writes if row.get("table") == "logical_time"]
    if len(clocks) != 1 or (
        clocks[0].get("key") != "world"
        or clocks[0].get("op") != "update"
        or isinstance(clocks[0].get("before"), bool)
        or not isinstance(clocks[0].get("before"), int)
        or clocks[0].get("after") != clocks[0].get("before") + 1
    ):
        raise ExactJoinError("governance commit has an invalid clock advance")
    tick = clocks[0]["after"]

    authority_writes = [
        row
        for row in writes
        if row.get("table") == "authority_operations"
    ]
    [authority_write] = authority_writes
    if (
        authority_write.get("op") != "create"
        or authority_write.get("before") is not None
        or not isinstance(authority_write.get("after"), Mapping)
    ):
        raise ExactJoinError("governance authority write changed")
    authority = _json_object(authority_write["after"], "authority operation")
    operation_key = authority_operation_key(
        contract.scope, contract.original_actor, key
    )
    expected_authority = {
        "operation_key": operation_key,
        "scope": contract.scope,
        "actor_id": contract.original_actor,
        "idempotency_key": key,
        "request_fingerprint": contract.outer_fingerprint,
        "outcome_listing": None,
    }
    if authority_write.get("key") != operation_key or any(
        authority.get(name) != value for name, value in expected_authority.items()
    ):
        raise ExactJoinError("governance authority operation identity changed")
    if final_authority.get(operation_key) != authority:
        raise ExactJoinError("governance authority operation is absent from final World")

    outcome_table = _text(authority.get("outcome_table"), "outcome_table")
    outcome_key = _text(authority.get("outcome_key"), "outcome_key")
    primary = final_rows.get((outcome_table, outcome_key))
    if primary is None:
        raise ExactJoinError("governance primary outcome is absent from final World")
    expected_inner_fingerprint = contract.inner_fingerprint
    if contract.operation == "publish_governance_policy":
        expected_inner_fingerprint = command_fingerprint(
            operation=contract.operation,
            service_actor=contract.service_actor,
            original_actor=contract.original_actor,
            request={"payload": policy_payload_to_wire(primary.typed.payload)},
        )

    governance_results: list[VerifiedGovernanceEnvelope] = []
    for row in writes:
        if row.get("table") not in {"governance_policies", "governance_records"}:
            continue
        table = _text(row.get("table"), "governance collection")
        row_key = _text(row.get("key"), "governance row key")
        result = final_rows.get((table, row_key))
        if result is None:
            raise ExactJoinError("governance commit row is absent from final World")
        if (
            row.get("op") != "create"
            or row.get("before") is not None
            or row.get("after") != result.wire
        ):
            raise ExactJoinError("governance commit row differs from final World")
        typed = result.typed
        if (
            typed.logical_tick != tick
            or typed.service_actor != contract.service_actor
            or typed.original_actor != contract.original_actor
            or typed.idempotency_key != key
            or typed.request_fingerprint != expected_inner_fingerprint
        ):
            raise ExactJoinError("governance envelope authority metadata changed")
        governance_results.append(result)
    if primary not in governance_results:
        raise ExactJoinError("authority outcome is not in the atomic governance commit")

    expected_subject = _primary_subject(primary)
    if commit.get("subject_id") != expected_subject:
        raise ExactJoinError("governance commit subject changed")
    _verify_projection_write(
        operation=contract.operation,
        writes=writes,
        primary=primary,
    )
    return VerifiedMarketGovernanceOperation(
        operation=contract.operation,
        service_actor=contract.service_actor,
        original_actor=contract.original_actor,
        request=dict(contract.request),
        request_fingerprint=contract.outer_fingerprint,
        authority_operation=authority,
        primary_result=primary,
        result_rows=tuple(
            sorted(governance_results, key=lambda row: (row.collection, row.key))
        ),
        commit=_json_object(commit, "World governance commit"),
    )


def _verify_projection_write(
    *,
    operation: str,
    writes: tuple[dict[str, Any], ...],
    primary: VerifiedGovernanceEnvelope,
) -> None:
    table = _PROJECTION_BY_OPERATION.get(operation)
    if table is None:
        return
    [write] = [row for row in writes if row.get("table") == table]
    if write.get("op") not in {"create", "update"}:
        raise ExactJoinError("governance projection write has an invalid operation")
    expected_key: str
    payload = primary.typed.payload
    if isinstance(payload, ReviewEvidence):
        expected_key = payload.review_id
    elif isinstance(payload, ReputationEvent):
        expected_key = payload.merchant_id
    else:
        raise ExactJoinError("governance projection has the wrong primary result")
    if write.get("key") != expected_key:
        raise ExactJoinError("governance projection key changed")
    if not isinstance(write.get("after"), Mapping):
        raise ExactJoinError("governance projection has no canonical after row")


def _verify_projection_chains(
    context: ExactJoinContext,
    commits: tuple[dict[str, Any], ...],
) -> None:
    """Replay governance-owned materialized projections in commit order."""

    governance_ids = {
        _text(row.get("commit_id"), "governance commit_id") for row in commits
    }
    external_projection_commits = tuple(
        row
        for row in context.world_commits
        if row.get("commit_kind") == "transaction"
        and row.get("commit_id") not in governance_ids
        and any(
            isinstance(write, Mapping)
            and write.get("table") in {"reviews", "reputation"}
            for write in row.get("table_writes", ())
        )
    )
    if external_projection_commits:
        from runtime.supply_fulfillment_evidence import (
            verify_settlement_reputation_followup_commits,
        )

        verified_external = verify_settlement_reputation_followup_commits(context)
        if {
            _text(row.get("commit_id"), "external projection commit_id")
            for row in external_projection_commits
        } != {
            _text(row.get("commit_id"), "settlement reputation commit_id")
            for row in verified_external
        }:
            raise ExactJoinError(
                "non-governance projection write lacks exact settlement evidence"
            )
        commits = (*commits, *verified_external)

    for table in ("reviews", "reputation"):
        state = dict(_projection_rows(context.initial_tables, table))
        for commit in sorted(commits, key=_commit_sequence):
            for write in _commit_writes(commit):
                if write.get("table") != table:
                    continue
                key = _text(write.get("key"), f"{table} projection key")
                prior = state.get(key)
                if write.get("before") != prior:
                    raise ExactJoinError(
                        f"{table} projection commit predecessor changed"
                    )
                expected_op = "create" if prior is None else "update"
                if write.get("op") != expected_op:
                    raise ExactJoinError(
                        f"{table} projection commit operation changed"
                    )
                after = write.get("after")
                if not isinstance(after, Mapping):
                    raise ExactJoinError(
                        f"{table} projection commit has no after row"
                    )
                state[key] = _json_value(after)
        if state != _projection_rows(context.final_tables, table):
            raise ExactJoinError(
                f"final {table} projection differs from the World commit chain"
            )


def _verify_response(
    context: ExactJoinContext,
    *,
    exchange: LinkedPlatformExchange,
    operation: VerifiedMarketGovernanceOperation,
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
) -> dict[str, Any] | None:
    request = exchange.request
    action_kind = _text(
        exchange.decision.get("action_kind"), "governance action kind"
    )
    request_actor = _text(request.get("from"), "governance request actor")
    ack_target = request_actor
    if request_actor == "platform:orchestrator":
        ack_target = {
            "platform.aggregate_reviews": "runtime:reviews",
            "platform.resolve_governance_case": "runtime:governance",
            "platform.create_remediation_plan": "runtime:remediation",
            "platform.verify_remediation_step": "runtime:remediation",
            "platform.apply_governance_reputation": "runtime:reputation",
        }.get(action_kind, "runtime:governance")
    expected: list[dict[str, Any]] = [
        _response_envelope(
            request,
            suffix="governance-updated",
            from_actor=operation.service_actor,
            to_actor=ack_target,
            idempotency_key=_text(
                request.get("idempotency_key"), "governance idempotency key"
            ),
            kind=MARKET_GOVERNANCE_RESPONSE_KIND,
            payload={
                "operation": operation.operation,
                **_platform_safe_result(operation.primary_result),
            },
        )
    ]
    payload = _request_payload(exchange)
    primary = operation.primary_result.typed.payload
    tick = _operation_tick(operation)

    if operation.operation in {"submit_review", "ingest_review_observation"}:
        if not isinstance(primary, ReviewEvidence):
            raise ExactJoinError("review aggregation follow-up requires a review")
        expected.append(
            _internal_envelope(
                request,
                suffix="aggregate",
                to_actor="platform:reviews",
                kind="platform.aggregate_reviews",
                payload={"sku_id": primary.sku_id},
            )
        )

    if operation.operation in {
        "ingest_market_observation",
        "resolve_governance_case",
    }:
        if not isinstance(primary, GovernanceCase):
            raise ExactJoinError("governance causal response requires a case")
        expected.extend(
            _case_notice_envelopes(
                request,
                governance_case=primary,
                final_rows=final_rows,
                world_tick=tick,
            )
        )
        if (
            operation.operation == "ingest_market_observation"
            and "resolution_template" in payload
        ):
            internal_payload: dict[str, Any] = {
                "decision_intent": {
                    "case_id": primary.case_id,
                    **_resolution_template(payload["resolution_template"]),
                }
            }
            if "remediation_template" in payload:
                internal_payload["remediation_template"] = _remediation_template(
                    payload["remediation_template"]
                )
            expected.append(
                _internal_envelope(
                    request,
                    suffix="auto-resolve",
                    to_actor="platform:governance",
                    kind="platform.resolve_governance_case",
                    payload=internal_payload,
                )
            )
        elif (
            operation.operation == "resolve_governance_case"
            and "remediation_template" in payload
        ):
            expected.append(
                _internal_envelope(
                    request,
                    suffix="create-plan",
                    to_actor="platform:remediation",
                    kind="platform.create_remediation_plan",
                    payload={
                        "plan_intent": {
                            "case_id": primary.case_id,
                            **_remediation_template(payload["remediation_template"]),
                        }
                    },
                )
            )

    if operation.operation in {
        "accept_remediation_plan",
        "create_remediation_plan",
        "verify_remediation_step",
    }:
        if not isinstance(primary, RemediationPlan):
            raise ExactJoinError("governance causal response requires a plan")
        expected.append(
            _plan_notice_envelope(request, plan=primary, world_tick=tick)
        )
        audit_request = _remediation_audit_request_envelope(
            request, plan=primary, world_tick=tick
        )
        if audit_request is not None:
            expected.append(audit_request)

    if operation.operation == "complete_remediation_step":
        if not isinstance(primary, RemediationPlan):
            raise ExactJoinError("remediation completion did not return a plan")
        expected.append(
            _internal_envelope(
                request,
                suffix="verify",
                to_actor="platform:remediation",
                kind="platform.verify_remediation_step",
                payload={
                    "plan_id": primary.plan_id,
                    "step_id": _text(payload.get("step_id"), "step_id"),
                },
            )
        )

    if operation.operation == "verify_remediation_step":
        assert isinstance(primary, RemediationPlan)
        if primary.status == "completed":
            expected.append(
                _internal_envelope(
                    request,
                    suffix="reputation",
                    to_actor="platform:reputation",
                    kind="platform.apply_governance_reputation",
                    payload={
                        "source_intent": {
                            "source_kind": "remediation_plan",
                            "source_id": primary.plan_id,
                        }
                    },
                )
            )

    observed = tuple(
        _json_object(row, "governance response") for row in exchange.responses
    )
    decision = exchange.decision
    expected_ids = [row["msg_id"] for row in expected]
    # Platform validation decisions store the response kind inventory in
    # canonical sorted order, while ids and hashes preserve production order.
    expected_kinds = sorted(row["action"]["kind"] for row in expected)
    expected_hashes = [wire_envelope_sha256(row) for row in expected]
    if (
        decision.get("response_msg_ids") != expected_ids
        or decision.get("response_kinds") != expected_kinds
        or decision.get("response_sha256s") != expected_hashes
    ):
        raise ExactJoinError(
            "governance response metadata differs from its World-derived "
            "causal chain"
        )
    expected_by_id = {row["msg_id"]: row for row in expected}
    if len(expected_by_id) != len(expected) or any(
        expected_by_id.get(row.get("msg_id")) != row for row in observed
    ):
        raise ExactJoinError(
            "governance response bundle is forged or differs from its "
            "World-derived causal chain"
        )
    # The first response is always the safe acknowledgement.  Causal notices
    # and internal requests were verified above and remain claimed by the
    # generic Platform linker and, for internal requests, their own exchange.
    acknowledgement_id = expected[0]["msg_id"]
    return next(
        (row for row in observed if row.get("msg_id") == acknowledgement_id),
        None,
    )


def _response_envelope(
    request: Mapping[str, Any],
    *,
    suffix: str,
    from_actor: str,
    to_actor: str,
    idempotency_key: str,
    kind: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    msg_id = _text(request.get("msg_id"), "governance request msg_id")
    return {
        "msg_id": f"{msg_id}:{suffix}",
        "ts": _text(request.get("ts"), "governance request timestamp"),
        "from": from_actor,
        "to": to_actor,
        "in_reply_to": msg_id,
        "idempotency_key": idempotency_key,
        "signature": None,
        "action": {"kind": kind, "payload": _json_value(payload)},
    }


def _internal_envelope(
    request: Mapping[str, Any],
    *,
    suffix: str,
    to_actor: str,
    kind: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    key = _text(request.get("idempotency_key"), "governance idempotency key")
    return _response_envelope(
        request,
        suffix=suffix,
        from_actor="platform:orchestrator",
        to_actor=to_actor,
        idempotency_key=f"{key}:{suffix}",
        kind=kind,
        payload=payload,
    )


def _case_notice_envelopes(
    request: Mapping[str, Any],
    *,
    governance_case: GovernanceCase,
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
    world_tick: int,
) -> list[dict[str, Any]]:
    opened = [
        row
        for row in final_rows.values()
        if row.kind == "governance_case"
        and row.stable_id == governance_case.case_id
        and row.version == 1
    ]
    if len(opened) != 1:
        raise ExactJoinError("governance case has no unique opened version")
    opened_reference = _platform_safe_result(opened[0])["references"]
    current_rows = [
        row
        for row in final_rows.values()
        if row.kind == "governance_case"
        and row.stable_id == governance_case.case_id
        and row.semantic_digest == governance_case.case_digest
    ]
    if len(current_rows) != 1:
        raise ExactJoinError("governance case current reference is ambiguous")
    current_reference = _platform_safe_result(current_rows[0])["references"]
    response_actions: list[str] = []
    if governance_case.status == "open":
        action = {
            "review_integrity": "commerce.reject_review_manipulation",
            "competition": "commerce.reject_coordination",
        }.get(governance_case.case_kind)
        if action is not None:
            response_actions.append(action)
    key = _text(request.get("idempotency_key"), "governance idempotency key")
    notices: list[dict[str, Any]] = []
    for index, merchant_id in enumerate(
        sorted(governance_case.subject_merchant_ids), start=1
    ):
        notices.append(
            _response_envelope(
                request,
                suffix=f"case-notice:{index}",
                from_actor="platform:governance",
                to_actor=merchant_id,
                idempotency_key=f"{key}:case-notice:{index}",
                kind="platform.governance_case_notice",
                payload={
                    "opened_case_reference": opened_reference,
                    "current_case_reference": current_reference,
                    "case_kind": governance_case.case_kind,
                    "status": governance_case.status,
                    "response_actions": response_actions,
                    "world_tick": world_tick,
                },
            )
        )
    return notices


def _plan_notice_envelope(
    request: Mapping[str, Any],
    *,
    plan: RemediationPlan,
    world_tick: int,
) -> dict[str, Any]:
    if plan.status == "draft":
        response_actions = ["commerce.accept_remediation_plan"]
    elif plan.status == "active" and any(
        step.status == "pending" for step in plan.steps
    ):
        response_actions = ["commerce.complete_remediation_step"]
    else:
        response_actions = []
    key = _text(request.get("idempotency_key"), "governance idempotency key")
    reference = {
        "record_kind": "remediation_plan",
        "stable_id": plan.plan_id,
        "record_digest": plan.plan_digest,
    }
    return _response_envelope(
        request,
        suffix="plan-notice",
        from_actor="platform:remediation",
        to_actor=plan.owner_merchant_id,
        idempotency_key=f"{key}:plan-notice",
        kind="platform.remediation_plan_notice",
        payload={
            "plan_reference": reference,
            "governance_case_id": plan.governance_case_id,
            "status": plan.status,
            "steps": [
                {
                    "step_id": step.step_id,
                    "sequence_no": step.sequence_no,
                    "action_kind": step.action_kind,
                    "status": step.status,
                }
                for step in plan.steps
            ],
            "response_actions": response_actions,
            "world_tick": world_tick,
        },
    )


def _remediation_audit_request_envelope(
    request: Mapping[str, Any],
    *,
    plan: RemediationPlan,
    world_tick: int,
) -> dict[str, Any] | None:
    if plan.status != "active":
        return None
    pending = next((step for step in plan.steps if step.status == "pending"), None)
    if pending is None:
        return None
    key = _text(request.get("idempotency_key"), "governance idempotency key")
    return _response_envelope(
        request,
        suffix="plan-notice:audit-request",
        from_actor="platform:remediation",
        to_actor=REMEDIATION_AUDITOR_SERVICE_ID,
        idempotency_key=f"{key}:plan-notice:audit-request",
        kind="platform.remediation_audit_request",
        payload=build_remediation_audit_request(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            plan_digest=plan.plan_digest,
            owner_merchant_id=plan.owner_merchant_id,
            step_id=pending.step_id,
            sequence_no=pending.sequence_no,
            action_kind=pending.action_kind,
            world_tick=world_tick,
        ),
    )


def _operation_tick(operation: VerifiedMarketGovernanceOperation) -> int:
    clocks = [
        row
        for row in _commit_writes(operation.commit)
        if row.get("table") == "logical_time"
    ]
    if len(clocks) != 1:
        raise ExactJoinError("governance operation has no unique logical clock")
    value = clocks[0].get("after")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError("governance operation clock is invalid")
    return value


def _verify_search_response(
    context: ExactJoinContext,
    *,
    exchange: LinkedPlatformExchange,
    operation: VerifiedMarketGovernanceOperation,
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
) -> dict[str, Any]:
    if len(exchange.responses) != 1:
        raise ExactJoinError("governed search needs one rank response")
    response = exchange.responses[0]
    request = exchange.request
    if (
        response.get("from") != "platform:aggregator"
        or response.get("to") != request.get("from")
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
    ):
        raise ExactJoinError("governed search response identity changed")
    action = response.get("action")
    body = action.get("payload") if isinstance(action, Mapping) else None
    expected_fields = {
        "schema_version",
        "session_id",
        "session_digest",
        "mandate_id",
        "issued_at_tick",
        "expires_at_tick",
        "search_session",
        "candidates",
        "ranking_context_reference",
        "ranking_context_projection",
    }
    if (
        not isinstance(action, Mapping)
        or action.get("kind") != "platform.rank_offers"
        or not isinstance(body, Mapping)
        or set(body) != expected_fields
        or body.get("schema_version") != "cwe.platform-rank-offers.v2"
    ):
        raise ExactJoinError("governed search response schema changed")
    try:
        session = coerce_search_session(body["search_session"])
    except Exception as exc:
        raise ExactJoinError("governed search response has an invalid session") from exc
    session_wire = search_session_to_wire(session)
    if session_wire != body["search_session"]:
        raise ExactJoinError("governed search session wire changed")
    expected_session_fields = {
        "session_id": session.session_id,
        "session_digest": session.session_digest,
        "mandate_id": session.mandate_id,
        "issued_at_tick": session.issued_at_tick,
        "expires_at_tick": session.expires_at_tick,
    }
    if any(body.get(name) != value for name, value in expected_session_fields.items()):
        raise ExactJoinError("governed search response differs from its session")
    _verify_search_session_final(context, session_wire=session_wire)
    _verify_search_candidates(body.get("candidates"), session=session)

    expected_reference = _platform_safe_result(operation.primary_result)[
        "references"
    ]
    if body.get("ranking_context_reference") != expected_reference:
        raise ExactJoinError("search response forged its RankingContext reference")
    expected_projection = _ranking_projection(
        context,
        ranking=operation.primary_result.typed.payload,
        final_rows=final_rows,
    )
    if body.get("ranking_context_projection") != expected_projection:
        raise ExactJoinError("search response changed its World ranking projection")
    return _json_object(response, "governed search response")


def _verify_search_session_final(
    context: ExactJoinContext,
    *,
    session_wire: Mapping[str, Any],
) -> None:
    raw = context.final_tables.get("search_sessions", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World search_sessions must be an array")
    expected = _strip_schema_versions(session_wire)
    matches = [
        row
        for row in raw
        if isinstance(row, Mapping)
        and row.get("session_id") == session_wire.get("session_id")
    ]
    if len(matches) != 1 or _json_object(matches[0], "SearchSession") != expected:
        raise ExactJoinError("governed SearchSession is absent from final World")


def _strip_schema_versions(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_schema_versions(item)
            for key, item in value.items()
            if key != "schema_version"
        }
    if isinstance(value, list):
        return [_strip_schema_versions(item) for item in value]
    return value


def _verify_search_candidates(value: Any, *, session: Any) -> None:
    if not isinstance(value, list) or len(value) != len(session.offers):
        raise ExactJoinError("governed search candidates changed")
    for candidate, offer in zip(value, session.offers, strict=True):
        if not isinstance(candidate, Mapping):
            raise ExactJoinError("governed search candidate is invalid")
        authoritative = offer_snapshot_to_wire(offer)
        if any(candidate.get(name) != item for name, item in authoritative.items()):
            raise ExactJoinError("governed search candidate differs from World offer")


def _ranking_projection(
    context: ExactJoinContext,
    *,
    ranking: Any,
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
) -> dict[str, Any]:
    if not isinstance(ranking, RankingContext):
        raise ExactJoinError("ranking projection requires a RankingContext")

    placements: dict[str, Any] = {}
    aggregates: dict[str, ReviewAggregate] = {}
    cases: dict[str, GovernanceCase] = {}
    reputation: dict[str, ReputationEvent] = {}
    for row in final_rows.values():
        payload = row.typed.payload
        if isinstance(payload, Campaign):
            for placement in payload.placements:
                prior = placements.get(placement.placement_digest)
                if prior is not None and prior != placement:
                    raise ExactJoinError("ranking placement digest is ambiguous")
                placements[placement.placement_digest] = placement
        elif isinstance(payload, ReviewAggregate):
            _insert_digest(aggregates, payload.aggregate_digest, payload, "aggregate")
        elif isinstance(payload, GovernanceCase):
            _insert_digest(cases, payload.case_digest, payload, "case")
        elif isinstance(payload, ReputationEvent):
            _insert_digest(reputation, payload.event_digest, payload, "reputation")

    selected_placements = _resolve_digests(
        ranking.placement_digests, placements, "ranking placement"
    )
    selected_aggregates = _resolve_digests(
        ranking.review_aggregate_digests,
        aggregates,
        "ranking review aggregate",
    )
    selected_cases = _resolve_digests(
        ranking.governance_case_digests, cases, "ranking governance case"
    )
    selected_reputation = _resolve_digests(
        ranking.reputation_event_digests,
        reputation,
        "ranking reputation event",
    )
    catalog = _catalog_merchants(context.final_tables)
    annotations: list[dict[str, Any]] = []
    for sku_id in ranking.candidate_sku_ids:
        merchant_id = catalog.get(sku_id)
        if merchant_id is None:
            raise ExactJoinError("RankingContext candidate is absent from catalog")
        sku_placements = sorted(
            (
                row
                for row in selected_placements
                if row.sku_id == sku_id and row.owner_merchant_id == merchant_id
            ),
            key=lambda row: row.placement_id,
        )
        sku_aggregates = [
            row for row in selected_aggregates if row.sku_id == sku_id
        ]
        merchant_reputation = [
            row for row in selected_reputation if row.merchant_id == merchant_id
        ]
        if len(sku_aggregates) > 1 or len(merchant_reputation) > 1:
            raise ExactJoinError("RankingContext public evidence is ambiguous")
        resolved_cases = sorted(
            (
                row
                for row in selected_cases
                if merchant_id in row.subject_merchant_ids
                and row.status in {"resolved", "dismissed"}
                and row.resolution_code is not None
            ),
            key=lambda row: row.case_id,
        )
        aggregate = None if not sku_aggregates else sku_aggregates[0]
        reputation_event = (
            None if not merchant_reputation else merchant_reputation[0]
        )
        annotations.append(
            {
                "sku_id": sku_id,
                "merchant_id": merchant_id,
                "sponsored_placements": [
                    {
                        "placement_id": row.placement_id,
                        "disclosure_status": row.disclosure_status,
                        "disclosure_text": row.disclosure_text,
                    }
                    for row in sku_placements
                ],
                "review_summary": (
                    None
                    if aggregate is None
                    else {
                        "review_count": aggregate.review_count,
                        "verified_review_count": aggregate.verified_review_count,
                        "rating_sum": aggregate.rating_sum,
                        "verified_rating_sum": aggregate.verified_rating_sum,
                    }
                ),
                "resolved_cases": [
                    {
                        "case_id": row.case_id,
                        "case_kind": row.case_kind,
                        "resolution_code": row.resolution_code,
                    }
                    for row in resolved_cases
                ],
                "reputation": (
                    None
                    if reputation_event is None
                    else {
                        "event_kind": reputation_event.event_kind,
                        "outcome_bps": reputation_event.outcome_bps,
                        "version": reputation_event.version,
                    }
                ),
            }
        )
    return {
        "schema_version": "cwe.governance-ranking-projection.v1",
        "context_id": ranking.context_id,
        "context_digest": ranking.context_digest,
        "candidate_annotations": annotations,
    }


def _insert_digest(
    rows: dict[str, Any], digest: str, value: Any, label: str
) -> None:
    prior = rows.get(digest)
    if prior is not None and prior != value:
        raise ExactJoinError(f"ranking {label} digest is ambiguous")
    rows[digest] = value


def _resolve_digests(
    digests: tuple[str, ...], rows: Mapping[str, Any], label: str
) -> tuple[Any, ...]:
    resolved: list[Any] = []
    for digest in digests:
        value = rows.get(digest)
        if value is None:
            raise ExactJoinError(f"{label} digest is absent from final World")
        resolved.append(value)
    return tuple(resolved)


def _catalog_merchants(tables: Mapping[str, Any]) -> dict[str, str]:
    raw = tables.get("catalog", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World catalog must be an array")
    result: dict[str, str] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ExactJoinError(f"catalog[{index}] is invalid")
        sku_id = _text(row.get("sku_id"), "catalog sku_id")
        merchant_id = _text(row.get("merchant_id"), "catalog merchant_id")
        if sku_id in result:
            raise ExactJoinError("World catalog SKU is duplicated")
        result[sku_id] = merchant_id
    return result


def _verify_read_response(
    exchange: LinkedPlatformExchange,
    *,
    initial_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
    accepted: list[VerifiedMarketGovernanceRequest],
) -> VerifiedMarketGovernanceRead:
    decision = exchange.decision
    actor_id = _text(decision.get("actor_id"), "governance read actor_id")
    if actor_id.split(":", 1)[0] not in {"buyer", "merchant"}:
        raise ExactJoinError("governance history reader is not a commerce actor")
    if decision.get("platform_endpoint") != "platform:governance":
        raise ExactJoinError("governance history targeted another service")
    payload = _request_payload(exchange)
    _require_exact_fields(
        payload,
        {"record_kind", "stable_id"},
        "commerce.read_governance_history",
    )
    record_kind = _text(payload.get("record_kind"), "record_kind")
    stable_id = _text(payload.get("stable_id"), "stable_id")
    if decision.get("decision") != "accepted":
        raise ExactJoinError("governance history read was not accepted")
    if len(exchange.responses) != 1:
        raise ExactJoinError("accepted governance history read needs one response")
    response = exchange.responses[0]
    request = exchange.request
    if (
        response.get("from") != "platform:governance"
        or response.get("to") != actor_id
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
    ):
        raise ExactJoinError("governance history response identity changed")
    action = response.get("action")
    body = action.get("payload") if isinstance(action, Mapping) else None
    if (
        not isinstance(action, Mapping)
        or action.get("kind") != MARKET_GOVERNANCE_READ_RESPONSE_KIND
        or not isinstance(body, Mapping)
        or set(body) != {"resource", "record_kind", "stable_id", "records"}
        or body.get("resource") != "governance_history"
        or body.get("record_kind") != record_kind
        or body.get("stable_id") != stable_id
        or not isinstance(body.get("records"), list)
    ):
        raise ExactJoinError("governance history response schema changed")

    visible: list[VerifiedGovernanceEnvelope] = [
        row
        for row in initial_rows.values()
        if row.kind == record_kind
        and row.stable_id == stable_id
        and _can_read(row.typed, actor_id)
    ]
    for accepted_request in accepted:
        if accepted_request.exchange.request_position >= exchange.request_position:
            continue
        for operation in accepted_request.operations:
            visible.extend(
                row
                for row in operation.result_rows
                if row.kind == record_kind
                and row.stable_id == stable_id
                and _can_read(row.typed, actor_id)
            )
    visible.sort(key=lambda row: row.version)
    expected = [_platform_safe_result(row) for row in visible]
    if body.get("records") != expected:
        raise ExactJoinError(
            "governance history differs from the caller-scoped World projection"
        )
    return VerifiedMarketGovernanceRead(
        exchange=exchange,
        actor_id=actor_id,
        record_kind=record_kind,
        stable_id=stable_id,
        records=tuple(_json_object(row, "governance history record") for row in expected),
    )


def _platform_safe_result(row: VerifiedGovernanceEnvelope) -> dict[str, Any]:
    """Return the exact public result projection derived from a typed World row.

    The VCP boundary owns this schema.  Keeping the derivation here based on
    the canonical payload, rather than copying fields from the Platform ack,
    makes a forged stable id, version, digest, status, or nested reference fail.
    """

    value = row.typed.payload
    base = {
        "result_kind": row.kind,
        "result_id": row.stable_id,
        "version": row.version,
        "status": _result_status(value),
        "references": {
            "record_kind": row.kind,
            "stable_id": row.stable_id,
            "record_digest": row.semantic_digest,
        },
    }
    if isinstance(value, Campaign):
        base["placements"] = [
            {
                "placement_id": item.placement_id,
                "sku_id": item.sku_id,
                "disclosure_status": item.disclosure_status,
                "disclosure_text": item.disclosure_text,
            }
            for item in value.placements
        ]
    elif isinstance(value, ReviewEvidence):
        base.update(
            {
                "sku_id": value.sku_id,
                "rating": value.rating,
                "verified_purchase": value.verified_purchase,
            }
        )
    elif isinstance(value, ReviewAggregate):
        base.update(
            {
                "sku_id": value.sku_id,
                "review_count": value.review_count,
                "verified_review_count": value.verified_review_count,
                "rating_sum": value.rating_sum,
                "verified_rating_sum": value.verified_rating_sum,
            }
        )
    elif isinstance(value, MarketSignal):
        base["signal_kind"] = value.signal_kind
    elif isinstance(value, GovernanceCase):
        base.update(
            {
                "case_kind": value.case_kind,
                "resolution_code": value.resolution_code,
                "subject_merchant_ids": list(value.subject_merchant_ids),
            }
        )
    elif isinstance(value, GovernanceResponseAttestation):
        base.update(
            {
                "case_id": value.case_id,
                "subject_merchant_id": value.subject_merchant_id,
                "response_kind": value.response_kind,
            }
        )
    elif isinstance(value, GovernanceResolutionDecision):
        base.update(
            {
                "case_id": value.case_id,
                "resolution_code": value.resolution_code,
            }
        )
    elif isinstance(value, ReputationEvent):
        base.update(
            {
                "event_id": value.event_id,
                "merchant_id": value.merchant_id,
                "event_kind": value.event_kind,
                "outcome_bps": value.outcome_bps,
            }
        )
    elif isinstance(value, RemediationPlan):
        base.update(
            {
                "governance_case_id": value.governance_case_id,
                "owner_merchant_id": value.owner_merchant_id,
                "steps": [
                    {
                        "step_id": item.step_id,
                        "sequence_no": item.sequence_no,
                        "action_kind": item.action_kind,
                        "status": item.status,
                    }
                    for item in value.steps
                ],
            }
        )
    elif isinstance(value, RankingContext):
        base.update(
            {
                "request_id": value.request_id,
                "ranked_sku_ids": list(value.ranked_sku_ids),
            }
        )
    return base


def _result_status(value: Any) -> str:
    if isinstance(
        value,
        (
            AdsCampaignTerms,
            ReviewAccountBinding,
            ReputationPolicyRevision,
            RemediationBlueprint,
        ),
    ):
        return "published"
    if isinstance(value, Campaign):
        return value.status
    if isinstance(value, ReviewEvidence):
        return "recorded"
    if isinstance(value, ReviewAggregate):
        return "computed"
    if isinstance(value, MarketSignal):
        return "recorded"
    if isinstance(value, GovernanceCase):
        return value.status
    if isinstance(
        value,
        (
            GovernanceResponseAttestation,
            GovernanceResolutionDecision,
            ReputationEvent,
            RankingContext,
        ),
    ):
        return "recorded"
    if isinstance(value, RemediationPlan):
        return value.status
    raise ExactJoinError(f"unsupported governance result {type(value).__name__}")


def _governance_rows(
    tables: Mapping[str, Any],
) -> dict[tuple[str, str], VerifiedGovernanceEnvelope]:
    rows: dict[tuple[str, str], VerifiedGovernanceEnvelope] = {}
    streams: dict[tuple[str, str, str], list[VerifiedGovernanceEnvelope]] = {}
    for collection in ("governance_policies", "governance_records"):
        raw = tables.get(collection, [])
        if not isinstance(raw, list):
            raise ExactJoinError(f"World {collection} must be an array")
        for index, value in enumerate(raw):
            if not isinstance(value, Mapping):
                raise ExactJoinError(f"{collection}[{index}] is not an object")
            wire = _json_object(value, f"{collection}[{index}]")
            try:
                if collection == "governance_policies":
                    typed = policy_envelope_from_wire(wire)
                    canonical = policy_envelope_to_wire(typed)
                else:
                    typed = record_envelope_from_wire(wire)
                    canonical = record_envelope_to_wire(typed)
            except Exception as exc:
                raise ExactJoinError(
                    f"{collection}[{index}] failed its strict World codec"
                ) from exc
            if canonical != wire:
                raise ExactJoinError("governance envelope is not canonical")
            key = envelope_key(typed)
            identity = (collection, key)
            if identity in rows:
                raise ExactJoinError("governance final snapshot has a duplicate key")
            result = VerifiedGovernanceEnvelope(
                collection=collection,
                key=key,
                kind=typed.kind,
                stable_id=typed.stable_id,
                version=envelope_version(typed),
                semantic_digest=typed.semantic_digest,
                envelope_digest=typed.envelope_digest,
                wire=wire,
                typed=typed,
            )
            rows[identity] = result
            streams.setdefault(
                (collection, typed.kind, typed.stable_id), []
            ).append(result)
    for stream in streams.values():
        stream.sort(key=lambda row: row.version)
        for ordinal, row in enumerate(stream, start=1):
            if row.version != ordinal:
                raise ExactJoinError("governance envelope versions are not contiguous")
            expected_previous = None if ordinal == 1 else stream[ordinal - 2].envelope_digest
            if row.typed.previous_envelope_digest != expected_previous:
                raise ExactJoinError("governance envelope predecessor digest changed")
    return rows


def _authority_operations(
    tables: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = tables.get("authority_operations", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World authority_operations must be an array")
    rows: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"authority_operations[{index}] is invalid")
        row = _json_object(value, f"authority_operations[{index}]")
        key = _text(row.get("operation_key"), "operation_key")
        if key in rows:
            raise ExactJoinError("World authority operation is duplicated")
        rows[key] = row
    return rows


def _evidence_issuer(context: ExactJoinContext, record_id: str) -> str:
    """Resolve review-import provenance from the replayed World evidence table."""

    raw = context.final_tables.get("evidence_records", [])
    if not isinstance(raw, list):
        raise ExactJoinError("World evidence_records must be an array")
    matches: list[Any] = []
    seen: set[tuple[str, int]] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"evidence_records[{index}] is invalid")
        wire = _json_object(value, f"evidence_records[{index}]")
        try:
            record = coerce_evidence_record(wire)
        except Exception as exc:
            raise ExactJoinError("World EvidenceRecord failed its strict codec") from exc
        canonical = _json_value(evidence_record_to_dict(record))
        if canonical != wire:
            raise ExactJoinError("World EvidenceRecord is not canonical")
        identity = (record.record_id, record.version)
        if identity in seen:
            raise ExactJoinError("World EvidenceRecord version is duplicated")
        seen.add(identity)
        if record.record_id == record_id:
            matches.append(record)
    if not matches:
        raise ExactJoinError("review observation EvidenceRecord is absent from World")
    matches.sort(key=lambda row: row.version)
    if [row.version for row in matches] != list(range(1, len(matches) + 1)):
        raise ExactJoinError("review observation EvidenceRecord history is not contiguous")
    return _text(matches[-1].issuer_id, "review observation issuer_id")


def _governance_commits(context: ExactJoinContext) -> tuple[dict[str, Any], ...]:
    commits: list[dict[str, Any]] = []
    known = set(_GOVERNANCE_ROWS_BY_OPERATION)
    for commit in context.world_commits:
        writes = commit.get("table_writes")
        has_governance_write = isinstance(writes, list) and any(
            isinstance(row, Mapping)
            and row.get("table") in {"governance_policies", "governance_records"}
            for row in writes
        )
        named = (
            commit.get("operation") in known
            or commit.get("authority_action")
            in {f"world.{operation}" for operation in known}
        )
        if has_governance_write != named:
            raise ExactJoinError("World commit governance identity is inconsistent")
        if has_governance_write:
            commits.append(_json_object(commit, "governance commit"))
    ids = [_text(row.get("commit_id"), "commit_id") for row in commits]
    if len(ids) != len(set(ids)):
        raise ExactJoinError("governance commit id is duplicated")
    return tuple(sorted(commits, key=_commit_sequence))


def _verify_final_governance_delta(
    *,
    initial_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
    final_rows: Mapping[tuple[str, str], VerifiedGovernanceEnvelope],
    commits: tuple[dict[str, Any], ...],
) -> None:
    created: dict[tuple[str, str], dict[str, Any]] = {}
    for commit in commits:
        for write in _commit_writes(commit):
            collection = write.get("table")
            if collection not in {"governance_policies", "governance_records"}:
                continue
            key = _text(write.get("key"), "governance write key")
            identity = (collection, key)
            if identity in initial_rows or identity in created:
                raise ExactJoinError("governance physical row was written more than once")
            if (
                write.get("op") != "create"
                or write.get("before") is not None
                or not isinstance(write.get("after"), Mapping)
            ):
                raise ExactJoinError("governance append-only write changed")
            created[identity] = _json_object(
                write["after"], "governance commit row"
            )
    if set(final_rows) != set(initial_rows) | set(created):
        raise ExactJoinError("final governance snapshot has an unexplained row delta")
    for identity, wire in created.items():
        if final_rows[identity].wire != wire:
            raise ExactJoinError("final governance row differs from its World commit")


def _projection_rows(tables: Mapping[str, Any], table: str) -> dict[str, Any]:
    raw = tables.get(table, {} if table == "reputation" else [])
    if table == "reputation":
        if not isinstance(raw, Mapping):
            raise ExactJoinError("World reputation projection must be an object")
        return {str(key): _json_value(value) for key, value in raw.items()}
    if not isinstance(raw, list):
        raise ExactJoinError("World review projection must be an array")
    rows: dict[str, Any] = {}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"reviews[{index}] is invalid")
        row = _json_object(value, f"reviews[{index}]")
        key = _text(row.get("review_id"), "review_id")
        if key in rows:
            raise ExactJoinError("World review projection is duplicated")
        rows[key] = row
    return rows


def _primary_subject(row: VerifiedGovernanceEnvelope) -> str:
    payload = row.typed.payload
    case_id = getattr(payload, "case_id", None)
    if isinstance(case_id, str) and case_id:
        return case_id
    campaign_id = getattr(payload, "campaign_id", None)
    if isinstance(campaign_id, str) and campaign_id:
        return campaign_id
    return row.key


def _request_payload(exchange: LinkedPlatformExchange) -> dict[str, Any]:
    action = exchange.request.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExactJoinError("governance request payload is missing")
    return _json_object(payload, "governance request payload")


def _commit_writes(commit: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = commit.get("table_writes")
    if not isinstance(raw, list) or not raw:
        raise ExactJoinError("governance commit has no table writes")
    writes = tuple(
        _json_object(row, "governance table write")
        for row in raw
        if isinstance(row, Mapping)
    )
    if len(writes) != len(raw) or any(
        set(row) != {"table", "key", "op", "before", "after"}
        for row in writes
    ):
        raise ExactJoinError("governance commit contains an invalid table write")
    return writes


def _can_read(row: GovernanceEnvelope, caller: str) -> bool:
    return caller in {*row.owner_ids, *row.subject_ids, row.original_actor}


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], action: str
) -> None:
    if set(value) != expected:
        raise ExactJoinError(f"{action} payload fields are not exact")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise ExactJoinError(f"{label} must be an object with text keys")
    return _json_object(value, label)


def _resolution_template(value: Any) -> dict[str, Any]:
    row = _mapping(value, "resolution_template")
    _require_exact_fields(
        row,
        {"resolution_kind", "policy_id", "policy_version"},
        "resolution_template",
    )
    resolution_kind = _text(row.get("resolution_kind"), "resolution_kind")
    if resolution_kind not in {
        "violation_confirmed",
        "no_violation",
        "compliant_rejection_recorded",
    }:
        raise ExactJoinError("resolution_template kind is unsupported")
    policy_version = row.get("policy_version")
    if (
        isinstance(policy_version, bool)
        or not isinstance(policy_version, int)
        or policy_version < 1
    ):
        raise ExactJoinError("resolution_template policy_version is invalid")
    return {
        "resolution_kind": resolution_kind,
        "policy_id": _text(row.get("policy_id"), "policy_id"),
        "policy_version": policy_version,
    }


def _remediation_template(value: Any) -> dict[str, str]:
    row = _mapping(value, "remediation_template")
    _require_exact_fields(
        row,
        {"blueprint_id", "sku_id"},
        "remediation_template",
    )
    return {
        "blueprint_id": _text(row.get("blueprint_id"), "blueprint_id"),
        "sku_id": _text(row.get("sku_id"), "sku_id"),
    }


def _read_stream_sequence(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ExactJoinError("expected_read_streams must be an array")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ExactJoinError("expected_read_streams entries must be pairs")
        result.append(
            (_text(item[0], "record_kind"), _text(item[1], "stable_id"))
        )
    return tuple(result)


def _text_sequence(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ExactJoinError(f"{label} must be an array")
    return tuple(_text(item, label) for item in value)


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExactJoinError(f"{label} must be non-empty text")
    return value


def _commit_sequence(commit: Mapping[str, Any]) -> int:
    value = commit.get("sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError("governance World commit sequence is invalid")
    return value


def _json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise ExactJoinError(f"{label} must be an object")
    return normalized


def _json_value(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExactJoinError("evidence contains a non-canonical JSON value") from exc


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        MARKET_GOVERNANCE_EVIDENCE_CONTRACT,
        verify_market_governance_evidence_contract,
    )
)


__all__ = [
    "ACTOR_ACTIONS",
    "MARKET_GOVERNANCE_EVIDENCE_CONTRACT",
    "MARKET_GOVERNANCE_READ_RESPONSE_KIND",
    "MARKET_GOVERNANCE_RESPONSE_KIND",
    "TRUSTED_ACTIONS",
    "VerifiedGovernanceEnvelope",
    "VerifiedMarketGovernanceEvidence",
    "VerifiedMarketGovernanceOperation",
    "VerifiedMarketGovernanceRead",
    "VerifiedMarketGovernanceRequest",
    "VerifiedRejectedMarketGovernanceRequest",
    "verify_market_governance_evidence_contract",
]
