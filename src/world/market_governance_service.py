"""Pure staging service for atomic CommerceWorld governance transitions.

The domain functions in :mod:`world.market_governance_core` derive trusted
payloads from catalog, order, ledger, evidence, mandate, and predecessor rows.
This module performs the next boundary only.  It wraps those already-derived
payloads in strict persistence envelopes and returns one compare-and-swap
commit for the enclosing World transaction.

It intentionally has no scenario, benchmark, transport, SQLite, or mutable
sidecar.  A caller must not use this planner as a substitute for World.  The
caller loads a :class:`TrustedGovernanceContext` from World, invokes core
derivations, stages the result here, then atomically applies the returned
``GovernanceCommit`` together with existing review or reputation projections,
the authority-operation row, logical time, and the global World journal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

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
from world.errors import IdempotencyConflict
from world.market_governance_core import (
    AdsCampaignTerms,
    GovernanceResolutionDecision,
    GovernanceResponseAttestation,
    RemediationBlueprint,
    ReputationPolicyRevision,
    ReviewAccountBinding,
    governance_intent_fingerprint,
    normalize_governance_intent,
)
from world.market_governance_persistence import (
    GovernanceCommit,
    GovernanceOperationRecord,
    GovernancePolicyEnvelope,
    GovernancePolicyPayload,
    GovernanceProjectionEffect,
    GovernanceRecordEnvelope,
    GovernanceRecordPayload,
    GovernanceTables,
    GovernanceWrite,
    build_governance_commit,
    build_governance_operation,
    build_policy_envelope,
    build_projection_effect,
    build_record_envelope,
    canonical_digest,
    envelope_key,
    policy_payload_to_wire,
    result_ref,
)


CommandDisposition: TypeAlias = Literal["committed", "idempotent"]


class MarketGovernanceServiceError(ValueError):
    """Trusted inputs cannot be staged as one governance transition."""


class MarketGovernanceContextConflict(MarketGovernanceServiceError):
    """The enclosing World changed after the trusted context was loaded."""


@dataclass(frozen=True, slots=True)
class TrustedGovernanceContext:
    """World-owned rows and digests consumed by one core derivation.

    ``trusted_context_digests`` names the exact catalog, order, ledger,
    EvidenceRecord, mandate, policy, or predecessor rows loaded by World.
    ``projection_digests`` binds the existing ``reviews`` or ``reputation``
    values that a projection effect compares before writing.
    """

    logical_tick: int
    tables: GovernanceTables
    trusted_context_digests: tuple[str, ...] = ()
    projection_digests: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_tick, bool)
            or not isinstance(self.logical_tick, int)
            or self.logical_tick < 0
        ):
            raise MarketGovernanceServiceError("logical_tick is invalid")
        if not isinstance(self.tables, GovernanceTables):
            raise MarketGovernanceServiceError("tables must be GovernanceTables")
        if self.trusted_context_digests != tuple(
            sorted(set(self.trusted_context_digests))
        ):
            raise MarketGovernanceServiceError(
                "trusted context digests are not canonical"
            )
        for digest in self.trusted_context_digests:
            _digest(digest, "trusted context digest")
        if self.projection_digests != tuple(sorted(set(self.projection_digests))):
            raise MarketGovernanceServiceError(
                "projection digests are not canonical"
            )
        for key, digest in self.projection_digests:
            _text(key, "projection key")
            _digest(digest, "projection digest")

    @property
    def context_digest(self) -> str:
        return canonical_digest(
            {
                "logical_tick": self.logical_tick,
                "governance_state_digest": self.tables.state_digest(),
                "trusted_context_digests": list(self.trusted_context_digests),
                "projection_digests": [
                    {"key": key, "digest": digest}
                    for key, digest in self.projection_digests
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class GovernanceCommandResult:
    disposition: CommandDisposition
    operation: GovernanceOperationRecord
    commit: GovernanceCommit | None


class MarketGovernancePlanner:
    """Stage core-derived governance payloads as one replayable World commit."""

    def plan_policy_publication(
        self,
        context: TrustedGovernanceContext,
        payload: GovernancePolicyPayload,
        *,
        service_actor: str,
        original_actor: str,
        idempotency_key: str,
        owner_ids: Iterable[str] = (),
        subject_ids: Iterable[str] = (),
    ) -> GovernanceCommandResult:
        """Stage one trusted policy publication.

        Policies are service publications, not agent-authored final state.  The
        authenticated service and original actor must therefore be the same
        trusted publisher.  Agent-facing policy intents must be rejected before
        this method is called.
        """

        if service_actor != original_actor:
            raise MarketGovernanceServiceError(
                "governance policy publication requires its trusted publisher"
            )
        operation = "publish_governance_policy"
        fingerprint = command_fingerprint(
            operation=operation,
            service_actor=service_actor,
            original_actor=original_actor,
            request={"payload": policy_payload_to_wire(payload)},
        )
        replay = self._retry(
            context,
            operation=operation,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
        )
        if replay is not None:
            return GovernanceCommandResult("idempotent", replay, None)
        tick = _next_tick(context.logical_tick)
        previous = _latest_policy_for_payload(context.tables, payload)
        envelope = build_policy_envelope(
            payload,
            logical_tick=tick,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            owner_ids=owner_ids,
            subject_ids=subject_ids,
            previous=previous,
        )
        write = GovernanceWrite(
            "governance_policies", envelope_key(envelope), envelope
        )
        return self._build_result(
            context,
            operation=operation,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            writes=(write,),
            projection_effects=(),
        )

    def plan_actor_transition(
        self,
        context: TrustedGovernanceContext,
        intent: Mapping[str, Any],
        *,
        derived_records: Iterable[GovernanceRecordPayload],
        service_actor: str,
        original_actor: str,
        idempotency_key: str,
        projection_effects: Iterable[GovernanceProjectionEffect] = (),
        extra_subject_ids: Mapping[str, Iterable[str]] | None = None,
    ) -> GovernanceCommandResult:
        """Stage records derived from one exact compact actor intent.

        ``derived_records`` must be outputs of ``market_governance_core`` using
        the same authenticated identities and the supplied trusted context.
        This method never accepts campaign budgets, owners, outcomes, versions,
        ticks, or digests from the actor intent.
        """

        normalized = normalize_governance_intent(intent)
        operation = cast(str, normalized["op"])
        fingerprint = governance_intent_fingerprint(
            normalized,
            original_actor=original_actor,
            trusted_context_digests=(),
        )
        return self._plan_records(
            context,
            operation=operation,
            request_fingerprint=fingerprint,
            derived_records=derived_records,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            projection_effects=projection_effects,
            extra_subject_ids=extra_subject_ids,
            actor_intent=True,
        )

    def plan_service_transition(
        self,
        context: TrustedGovernanceContext,
        operation: str,
        request: Mapping[str, Any],
        *,
        derived_records: Iterable[GovernanceRecordPayload],
        service_actor: str,
        original_actor: str,
        idempotency_key: str,
        projection_effects: Iterable[GovernanceProjectionEffect] = (),
        extra_subject_ids: Mapping[str, Iterable[str]] | None = None,
    ) -> GovernanceCommandResult:
        """Stage a trusted internal service command such as case resolution.

        The request is identity only.  World-owned versions, subjects, evidence
        digests, policy digests, catalog revision, and logical time live in the
        core-derived payloads and context rather than this request object.
        """

        fingerprint = command_fingerprint(
            operation=operation,
            service_actor=service_actor,
            original_actor=original_actor,
            request=request,
        )
        return self._plan_records(
            context,
            operation=operation,
            request_fingerprint=fingerprint,
            derived_records=derived_records,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            projection_effects=projection_effects,
            extra_subject_ids=extra_subject_ids,
            actor_intent=False,
        )

    def _plan_records(
        self,
        context: TrustedGovernanceContext,
        *,
        operation: str,
        request_fingerprint: str,
        derived_records: Iterable[GovernanceRecordPayload],
        service_actor: str,
        original_actor: str,
        idempotency_key: str,
        projection_effects: Iterable[GovernanceProjectionEffect],
        extra_subject_ids: Mapping[str, Iterable[str]] | None,
        actor_intent: bool,
    ) -> GovernanceCommandResult:
        replay = self._retry(
            context,
            operation=operation,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
        )
        if replay is not None:
            return GovernanceCommandResult("idempotent", replay, None)
        payloads = tuple(derived_records)
        if not payloads:
            raise MarketGovernanceServiceError(
                "governance transition requires core-derived records"
            )
        if actor_intent:
            _validate_actor_result_kinds(operation, payloads)
        tick = _next_tick(context.logical_tick)
        subjects = {} if extra_subject_ids is None else dict(extra_subject_ids)
        writes: list[GovernanceWrite] = []
        seen_streams: set[tuple[str, str, int]] = set()
        for payload in payloads:
            stable_id, version = _record_coordinates(payload)
            stream = (type(payload).__name__, stable_id, version)
            if stream in seen_streams:
                raise MarketGovernanceServiceError(
                    "one governance command produced a duplicate stream version"
                )
            seen_streams.add(stream)
            previous = _latest_record_for_payload(context.tables, payload)
            envelope = build_record_envelope(
                payload,
                logical_tick=tick,
                service_actor=service_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                subject_ids=subjects.get(stable_id, ()),
                previous=previous,
            )
            writes.append(
                GovernanceWrite(
                    "governance_records", envelope_key(envelope), envelope
                )
            )
        ordered_writes = tuple(sorted(writes, key=lambda row: row.key))
        effects = tuple(
            sorted(projection_effects, key=lambda row: (row.table, row.key))
        )
        _validate_projection_bindings(payloads, effects)
        return self._build_result(
            context,
            operation=operation,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            writes=ordered_writes,
            projection_effects=effects,
        )

    def _build_result(
        self,
        context: TrustedGovernanceContext,
        *,
        operation: str,
        service_actor: str,
        original_actor: str,
        idempotency_key: str,
        request_fingerprint: str,
        writes: tuple[GovernanceWrite, ...],
        projection_effects: tuple[GovernanceProjectionEffect, ...],
    ) -> GovernanceCommandResult:
        tick = _next_tick(context.logical_tick)
        operation_row = build_governance_operation(
            operation=operation,
            service_actor=service_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            logical_tick=tick,
            result_refs=(result_ref(write) for write in writes),
            projection_effects=projection_effects,
        )
        commit = build_governance_commit(
            expected_logical_tick=context.logical_tick,
            logical_tick=tick,
            context_digest=context.context_digest,
            writes=writes,
            projection_effects=projection_effects,
            operation=operation_row,
        )
        return GovernanceCommandResult("committed", operation_row, commit)

    @staticmethod
    def _retry(
        context: TrustedGovernanceContext,
        *,
        operation: str,
        service_actor: str,
        original_actor: str,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> GovernanceOperationRecord | None:
        existing = context.tables.operation_for_retry(
            service_actor=service_actor,
            original_actor=original_actor,
            operation=operation,
            idempotency_key=idempotency_key,
        )
        if existing is None:
            return None
        if existing.request_fingerprint != request_fingerprint:
            raise IdempotencyConflict(
                "governance idempotency key was reused with a changed request"
            )
        return existing


def command_fingerprint(
    *,
    operation: str,
    service_actor: str,
    original_actor: str,
    request: Mapping[str, Any],
) -> str:
    """Stable authenticated request identity, independent of mutable state.

    Causal World rows remain bound by the payload provenance, envelope
    predecessor, and commit context digest.  Keeping mutable state out of this
    fingerprint lets an exact retry match after the original commit.
    """

    return canonical_digest(
        {
            "schema_id": "cwe.world-governance-command.v1",
            "operation": _text(operation, "operation"),
            "service_actor": _text(service_actor, "service_actor"),
            "original_actor": _text(original_actor, "original_actor"),
            "request": dict(request),
        }
    )


def review_projection_effect(
    review: ReviewEvidence,
    *,
    text: str = "",
    before_digest: str | None = None,
) -> GovernanceProjectionEffect:
    """Build the narrow legacy ``Review`` projection for one accepted review."""

    if not isinstance(review, ReviewEvidence):
        raise MarketGovernanceServiceError("review projection requires ReviewEvidence")
    return build_projection_effect(
        table="reviews",
        key=review.review_id,
        before_digest=before_digest,
        payload={
            "review_id": review.review_id,
            "reviewer_id": review.reviewer_id,
            "sku_id": review.sku_id,
            "merchant_id": review.merchant_id,
            "rating": review.rating,
            "text": text,
        },
    )


def reputation_projection_effect(
    event: ReputationEvent,
    *,
    before_digest: str | None,
) -> GovernanceProjectionEffect:
    """Build one deterministic input to the existing reputation projection."""

    if not isinstance(event, ReputationEvent):
        raise MarketGovernanceServiceError(
            "reputation projection requires ReputationEvent"
        )
    return build_projection_effect(
        table="reputation",
        key=event.merchant_id,
        before_digest=before_digest,
        payload={
            "merchant_id": event.merchant_id,
            "event_id": event.event_id,
            "event_digest": event.event_digest,
            "event_kind": event.event_kind,
            "outcome_bps": event.outcome_bps,
        },
    )


def assert_context_unchanged(
    expected: TrustedGovernanceContext,
    current: TrustedGovernanceContext,
) -> None:
    """Compare the exact CAS boundary before an enclosing World writes."""

    if (
        current.logical_tick != expected.logical_tick
        or current.context_digest != expected.context_digest
    ):
        raise MarketGovernanceContextConflict(
            "World governance context changed before atomic commit"
        )


def _validate_actor_result_kinds(
    operation: str, payloads: tuple[GovernanceRecordPayload, ...]
) -> None:
    allowed: dict[str, tuple[type[Any], ...]] = {
        "publish_campaign": (Campaign,),
        "activate_campaign": (Campaign,),
        "disclose_placement": (Campaign,),
        "submit_review": (ReviewEvidence,),
        "reject_review_manipulation": (GovernanceResponseAttestation,),
        "reject_coordination": (GovernanceResponseAttestation,),
        "accept_remediation_plan": (RemediationPlan,),
        "complete_remediation_step": (RemediationPlan,),
    }
    expected = allowed.get(operation)
    if expected is None:
        raise MarketGovernanceServiceError("unsupported actor governance operation")
    if len(payloads) != 1 or not isinstance(payloads[0], expected):
        raise MarketGovernanceServiceError(
            "core-derived record kind does not match compact actor operation"
        )


def _validate_projection_bindings(
    payloads: tuple[GovernanceRecordPayload, ...],
    effects: tuple[GovernanceProjectionEffect, ...],
) -> None:
    reviews = [row for row in payloads if isinstance(row, ReviewEvidence)]
    reputation = [row for row in payloads if isinstance(row, ReputationEvent)]
    review_effects = [row for row in effects if row.table == "reviews"]
    reputation_effects = [row for row in effects if row.table == "reputation"]
    if len(review_effects) != len(reviews):
        raise MarketGovernanceServiceError(
            "accepted review and legacy Review projection must be atomic"
        )
    if len(reputation_effects) != len(reputation):
        raise MarketGovernanceServiceError(
            "reputation event and current reputation projection must be atomic"
        )
    expected_reviews = {
        (
            row.review_id,
            row.reviewer_id,
            row.sku_id,
            row.merchant_id,
            row.rating,
        )
        for row in reviews
    }
    actual_reviews = {
        (
            cast(str, row.payload["review_id"]),
            cast(str, row.payload["reviewer_id"]),
            cast(str, row.payload["sku_id"]),
            cast(str, row.payload["merchant_id"]),
            cast(int, row.payload["rating"]),
        )
        for row in review_effects
    }
    if actual_reviews != expected_reviews:
        raise MarketGovernanceServiceError("review projection identity mismatch")
    expected_reputation = {
        (
            row.merchant_id,
            row.event_id,
            row.event_digest,
            row.event_kind,
            row.outcome_bps,
        )
        for row in reputation
    }
    actual_reputation = {
        (
            cast(str, row.payload["merchant_id"]),
            cast(str, row.payload["event_id"]),
            cast(str, row.payload["event_digest"]),
            cast(str, row.payload["event_kind"]),
            cast(int, row.payload["outcome_bps"]),
        )
        for row in reputation_effects
    }
    if actual_reputation != expected_reputation:
        raise MarketGovernanceServiceError("reputation projection identity mismatch")
    unexpected = [
        row
        for row in effects
        if row.table == "reviews" and not reviews
        or row.table == "reputation" and not reputation
    ]
    if unexpected:
        raise MarketGovernanceServiceError("projection effect has no typed source")


def _latest_policy_for_payload(
    tables: GovernanceTables, payload: GovernancePolicyPayload
) -> GovernancePolicyEnvelope | None:
    stable_id, revision = _policy_coordinates(payload)
    if revision == 1:
        return None
    candidates = [
        cast(GovernancePolicyEnvelope, row)
        for _, row in tables.internal_all("governance_policies")
        if isinstance(row, GovernancePolicyEnvelope)
        and row.stable_id == stable_id
        and row.revision == revision - 1
    ]
    if len(candidates) != 1:
        raise MarketGovernanceServiceError(
            "policy revision requires exactly one persisted predecessor"
        )
    return candidates[0]


def _latest_record_for_payload(
    tables: GovernanceTables, payload: GovernanceRecordPayload
) -> GovernanceRecordEnvelope | None:
    stable_id, version = _record_coordinates(payload)
    if version == 1:
        return None
    candidates = [
        cast(GovernanceRecordEnvelope, row)
        for _, row in tables.internal_all("governance_records")
        if isinstance(row, GovernanceRecordEnvelope)
        and row.stable_id == stable_id
        and row.version == version - 1
        and type(row.payload) is type(payload)
    ]
    if len(candidates) != 1:
        raise MarketGovernanceServiceError(
            "record version requires exactly one persisted predecessor"
        )
    return candidates[0]


def _policy_coordinates(payload: GovernancePolicyPayload) -> tuple[str, int]:
    if isinstance(payload, AdsCampaignTerms):
        return payload.campaign_id, 1
    if isinstance(payload, ReviewAccountBinding):
        return payload.reviewer_id, 1
    if isinstance(payload, ReputationPolicyRevision):
        return payload.policy_id, payload.revision
    if isinstance(payload, RemediationBlueprint):
        return payload.blueprint_id, 1
    raise MarketGovernanceServiceError("unknown policy payload")


def _record_coordinates(payload: GovernanceRecordPayload) -> tuple[str, int]:
    if isinstance(payload, Campaign):
        return payload.campaign_id, payload.version
    if isinstance(payload, ReviewEvidence):
        return payload.review_id, payload.version
    if isinstance(payload, ReviewAggregate):
        return payload.aggregate_id, payload.version
    if isinstance(payload, MarketSignal):
        return payload.signal_id, payload.version
    if isinstance(payload, GovernanceCase):
        return payload.case_id, payload.version
    if isinstance(payload, GovernanceResponseAttestation):
        return payload.response_id, 1
    if isinstance(payload, GovernanceResolutionDecision):
        return payload.decision_id, 1
    if isinstance(payload, ReputationEvent):
        return payload.merchant_id, payload.version
    if isinstance(payload, RemediationPlan):
        return payload.plan_id, payload.version
    if isinstance(payload, RankingContext):
        return payload.context_id, payload.version
    raise MarketGovernanceServiceError("unknown record payload")


def _next_tick(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MarketGovernanceServiceError("logical time is invalid")
    return value + 1


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketGovernanceServiceError(f"{name} must be non-empty text")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MarketGovernanceServiceError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


__all__ = [
    "CommandDisposition",
    "GovernanceCommandResult",
    "MarketGovernanceContextConflict",
    "MarketGovernancePlanner",
    "MarketGovernanceServiceError",
    "TrustedGovernanceContext",
    "assert_context_unchanged",
    "command_fingerprint",
    "reputation_projection_effect",
    "review_projection_effect",
]
