"""First-class marketplace-governance operations for CommerceWorld.

This mixin owns the authority boundary shared by the memory World and the
SQLite backend's transactional staging path.  It derives every outcome from
catalog, order, ledger, evidence, mandate, policy, and predecessor rows already
held by World, then commits the typed governance envelopes, legacy projection,
authority binding, logical clock, and global journal as one transition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from protocol.evidence_records import EvidenceRecord, MandateRevision
from protocol.market_governance import (
    Campaign,
    GovernanceCase,
    MarketSignal,
    Placement,
    RankingContext,
    RemediationPlan,
    ReputationEvent,
    ReviewAggregate,
    ReviewEvidence,
)
from world.evidence_contracts import authority_operation_key
from world.errors import IdempotencyConflict, WorldError, WriteNotAuthorized
from world.market_governance_core import (
    AdsCampaignTerms,
    AdsPlacementTerms,
    GovernanceResponseAttestation,
    RemediationBlueprint,
    RemediationBlueprintStep,
    ReputationPolicyRevision,
    ReviewAccountBinding,
    build_ads_campaign_terms,
    build_market_observation,
    build_remediation_blueprint,
    build_reputation_policy_revision,
    build_review_account_binding,
    derive_campaign_activation,
    derive_campaign_publish,
    derive_governance_case,
    derive_governance_case_resolution,
    derive_governance_reputation_event,
    derive_governance_resolution_decision,
    derive_governance_response_attestation,
    derive_listing_governance_binding,
    derive_market_signal,
    derive_placement_disclosure,
    derive_purchase_governance_binding,
    derive_ranking_context,
    derive_record_governance_evidence,
    derive_remediation_plan,
    derive_remediation_plan_activation,
    derive_remediation_reputation_event,
    derive_remediation_step_completion,
    derive_remediation_step_verification,
    derive_review_aggregate,
    derive_review_evidence,
    derive_settlement_reputation_event,
    normalize_governance_intent,
)
from world.market_governance_persistence import (
    GovernancePolicyEnvelope,
    GovernanceRecordEnvelope,
    GovernanceTables,
    GovernanceWrite,
    canonical_digest,
)
from world.market_governance_service import (
    GovernanceCommandResult,
    MarketGovernancePlanner,
    TrustedGovernanceContext,
    reputation_projection_effect,
    review_projection_effect,
)
from world.types import (
    AgentId,
    AuthorityOperationRecord,
    Listing,
    Order,
    OrderState,
    Receipt,
    ReputationScore,
    Review,
    ReviewId,
    SkuId,
)


_POLICY_AUTHORITIES = {
    "ads_campaign_terms": "platform:ads",
    "review_account_binding": "platform:reviews",
    "reputation_policy_revision": "platform:reputation",
    "remediation_blueprint": "platform:remediation",
}

GOVERNANCE_RANKING_PROJECTION_SCHEMA = (
    "cwe.governance-ranking-projection.v1"
)

_REVIEW_OBSERVATION_FORBIDDEN_FIELDS = frozenset(
    {
        "reviewer_id",
        "merchant_id",
        "verified_purchase",
        "order_id",
        "txn_id",
        "receipt_id",
        "outcome",
        "account_created_at_tick",
        "burst_group_id",
        "submitted_at_tick",
        "source_ref",
        "source_refs",
        "provenance",
        "provenance_digests",
        "record_digest",
        "evidence_digest",
    }
)


def governance_policy_authority(policy_kind: str) -> str:
    """Return the trusted service that may publish one policy kind.

    Scenario hydration and live Platform routing share this lookup so neither
    layer can silently invent a service identity from scenario-owned data.
    """

    try:
        return _POLICY_AUTHORITIES[policy_kind]
    except KeyError as exc:
        raise WorldError(
            f"unsupported governance policy kind {policy_kind!r}"
        ) from exc


def market_governance_request_fingerprint(
    operation: str,
    service_actor: str,
    original_actor: str,
    request: Mapping[str, Any],
) -> str:
    """Bind one governance request to its operation and trusted identities."""

    return canonical_digest(
        {
            "schema_id": "cwe.world-governance-request.v1",
            "operation": operation,
            "service_actor": service_actor,
            "original_actor": original_actor,
            "request": dict(request),
        }
    )


class MarketGovernanceWorldMixin:
    """World methods for authoritative marketplace governance.

    Concrete hosts provide ``_tables``, ``_lock``, ``_logical_time``, and
    ``_commit_simple_authority_transition``.  No benchmark package is imported.
    """

    _tables: dict[str, Any]
    _lock: Any
    _logical_time: int

    def publish_governance_policy(
        self,
        policy_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        request = _plain_mapping(policy_intent)
        kind = _required_text(request, "kind")
        expected = governance_policy_authority(kind)
        if by_actor != expected or original_actor != expected:
            raise WriteNotAuthorized(
                f"{kind} must be published by its trusted service {expected}"
            )
        fingerprint = market_governance_request_fingerprint(
            "publish_governance_policy", by_actor, original_actor, request
        )
        scope = "publish_governance_policy"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return retry
            payload: Any
            owner_ids: tuple[str, ...] = ()
            if kind == "ads_campaign_terms":
                _require_exact_fields(
                    request,
                    required={
                        "kind",
                        "campaign_id",
                        "budget_cents",
                        "currency",
                        "starts_at_tick",
                        "ends_at_tick",
                        "placements",
                    },
                )
                raw_placements = request.get("placements")
                if not isinstance(raw_placements, list):
                    raise WorldError("campaign placements must be an array")
                placement_rows = tuple(_plain_mapping(row) for row in raw_placements)
                for row in placement_rows:
                    _require_exact_fields(
                        row, required={"sku_id", "bid_cents", "fee_cents"}
                    )
                placements = tuple(
                    AdsPlacementTerms(
                        _required_text(row, "sku_id"),
                        _required_int(row, "bid_cents", minimum=0),
                        _required_int(row, "fee_cents", minimum=0),
                    )
                    for row in placement_rows
                )
                listings = tuple(self._listing(row.sku_id) for row in placements)
                owners = {str(row.merchant_id) for row in listings}
                if len(owners) != 1:
                    raise WorldError("campaign terms must bind one listing owner")
                owner_ids = tuple(owners)
                payload = build_ads_campaign_terms(
                    campaign_id=_required_text(request, "campaign_id"),
                    budget_cents=_required_int(request, "budget_cents", minimum=1),
                    currency=_required_text(request, "currency"),
                    starts_at_tick=_required_int(
                        request, "starts_at_tick", minimum=0
                    ),
                    ends_at_tick=_required_int(request, "ends_at_tick", minimum=0),
                    placements=placements,
                    issued_by_id=by_actor,
                )
            elif kind == "review_account_binding":
                _require_exact_fields(
                    request,
                    required={
                        "kind",
                        "reviewer_id",
                        "account_created_at_tick",
                    },
                    optional={"burst_group_id"},
                )
                burst = request.get("burst_group_id")
                if burst is not None and not isinstance(burst, str):
                    raise WorldError("burst_group_id must be text or null")
                payload = build_review_account_binding(
                    reviewer_id=_required_text(request, "reviewer_id"),
                    account_created_at_tick=_required_int(
                        request, "account_created_at_tick", minimum=0
                    ),
                    burst_group_id=burst,
                    authority_id=by_actor,
                )
            elif kind == "reputation_policy_revision":
                _require_exact_fields(
                    request,
                    required={
                        "kind",
                        "policy_id",
                        "effective_tick",
                        "fulfilled_order_bps",
                        "disputed_order_bps",
                        "refund_bps",
                        "remediation_verified_bps",
                        "compliance_violation_bps",
                    },
                )
                policy_id = _required_text(request, "policy_id")
                previous = self._latest_policy("reputation_policy_revision", policy_id)
                revision = 1 if previous is None else previous.revision + 1
                payload = build_reputation_policy_revision(
                    policy_id=policy_id,
                    revision=revision,
                    previous_digest=None if previous is None else previous.policy_digest,
                    effective_tick=_required_int(request, "effective_tick", minimum=0),
                    published_by_id=by_actor,
                    fulfilled_order_bps=_required_int(
                        request, "fulfilled_order_bps", minimum=0, maximum=10_000
                    ),
                    disputed_order_bps=_required_int(
                        request, "disputed_order_bps", minimum=0, maximum=10_000
                    ),
                    refund_bps=_required_int(
                        request, "refund_bps", minimum=0, maximum=10_000
                    ),
                    remediation_verified_bps=_required_int(
                        request,
                        "remediation_verified_bps",
                        minimum=0,
                        maximum=10_000,
                    ),
                    compliance_violation_bps=_required_int(
                        request,
                        "compliance_violation_bps",
                        minimum=0,
                        maximum=10_000,
                    ),
                )
            else:
                _require_exact_fields(
                    request,
                    required={
                        "kind",
                        "blueprint_id",
                        "governance_case_kind",
                        "steps",
                    },
                )
                raw_steps = request.get("steps")
                if not isinstance(raw_steps, list):
                    raise WorldError("remediation blueprint steps must be an array")
                step_rows = tuple(_plain_mapping(row) for row in raw_steps)
                for row in step_rows:
                    _require_exact_fields(
                        row,
                        required={"action_kind"},
                        optional={"prerequisite_sequence_nos"},
                    )
                payload = build_remediation_blueprint(
                    blueprint_id=_required_text(request, "blueprint_id"),
                    governance_case_kind=_required_text(
                        request, "governance_case_kind"
                    ),
                    steps=tuple(
                        RemediationBlueprintStep(
                            action_kind=_required_text(
                                _plain_mapping(row), "action_kind"
                            ),
                            prerequisite_sequence_nos=tuple(
                                int(value)
                                for value in _plain_mapping(row).get(
                                    "prerequisite_sequence_nos", []
                                )
                            ),
                        )
                        for row in step_rows
                    ),
                    issued_by_id=by_actor,
                )
            context = self._governance_context()
            planned = MarketGovernancePlanner().plan_policy_publication(
                context,
                payload,
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                owner_ids=owner_ids,
            )
            self._commit_governance(
                planned,
                primary=payload,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return payload

    def apply_governance_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        normalized = normalize_governance_intent(intent)
        op = cast(str, normalized["op"])
        authority = {
            "publish_campaign": "platform:ads",
            "disclose_placement": "platform:ads",
            "activate_campaign": "platform:ads",
            "submit_review": "platform:reviews",
            "reject_review_manipulation": "platform:governance",
            "reject_coordination": "platform:governance",
            "accept_remediation_plan": "platform:remediation",
            "complete_remediation_step": "platform:remediation",
        }[op]
        if by_actor != authority:
            raise WriteNotAuthorized(f"{op} requires {authority}")
        fingerprint = market_governance_request_fingerprint(
            f"apply_governance_intent:{op}", by_actor, original_actor, normalized
        )
        scope = f"apply_governance_intent:{op}"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return retry
            tick = self._logical_time + 1
            projection_effects: tuple[Any, ...] = ()
            if op in {"publish_campaign", "disclose_placement", "activate_campaign"}:
                campaign_id = cast(str, normalized["campaign_id"])
                terms = cast(
                    AdsCampaignTerms,
                    self._required_policy("ads_campaign_terms", campaign_id),
                )
                current = cast(
                    Campaign | None, self._latest_record("campaign", campaign_id)
                )
                bindings = tuple(
                    self._listing_binding(item.sku_id) for item in terms.placements
                )
                if op == "publish_campaign":
                    result = derive_campaign_publish(
                        normalized,
                        terms=terms,
                        listing_bindings=bindings,
                        original_actor=original_actor,
                        ads_authority_id=by_actor,
                        server_tick=tick,
                        idempotency_key=idempotency_key,
                        current=current,
                    )
                elif op == "disclose_placement":
                    if current is None:
                        raise WorldError("campaign is not published")
                    placement_id = cast(str, normalized["placement_id"])
                    placement = next(
                        (
                            row
                            for row in current.placements
                            if row.placement_id == placement_id
                        ),
                        None,
                    )
                    if placement is None:
                        raise WorldError("placement is not in campaign")
                    result = derive_placement_disclosure(
                        normalized,
                        current=current,
                        listing_binding=self._listing_binding(placement.sku_id),
                        original_actor=original_actor,
                        ads_authority_id=by_actor,
                        server_tick=tick,
                        idempotency_key=idempotency_key,
                    )
                else:
                    if current is None:
                        raise WorldError("campaign is not published")
                    result = derive_campaign_activation(
                        normalized,
                        current=current,
                        original_actor=original_actor,
                        ads_authority_id=by_actor,
                        server_tick=tick,
                        idempotency_key=idempotency_key,
                    )
            elif op == "submit_review":
                sku_id = cast(str, normalized["sku_id"])
                account = cast(
                    ReviewAccountBinding,
                    self._required_policy("review_account_binding", original_actor),
                )
                listing = self._listing(sku_id)
                purchase = self._purchase_binding(
                    buyer_id=original_actor,
                    sku_id=sku_id,
                    merchant_id=str(listing.merchant_id),
                )
                result = derive_review_evidence(
                    normalized,
                    listing_binding=self._listing_binding(sku_id),
                    account_binding=account,
                    purchase_binding=purchase,
                    original_actor=original_actor,
                    review_authority_id=by_actor,
                    server_tick=tick,
                    idempotency_key=idempotency_key,
                )
                projection_effects = (
                    review_projection_effect(
                        result,
                        text=str(normalized.get("review_text", "")),
                    ),
                )
            elif op in {"reject_review_manipulation", "reject_coordination"}:
                case_id = cast(str, normalized["case_id"])
                governance_case = cast(
                    GovernanceCase,
                    self._required_record("governance_case", case_id),
                )
                signals = self._signals_for_case(governance_case)
                result = derive_governance_response_attestation(
                    normalized,
                    current=governance_case,
                    signals=signals,
                    original_actor=original_actor,
                    governance_authority_id=by_actor,
                    server_tick=tick,
                    idempotency_key=idempotency_key,
                )
            elif op == "accept_remediation_plan":
                plan_id = cast(str, normalized["plan_id"])
                plan = cast(
                    RemediationPlan,
                    self._required_record("remediation_plan", plan_id),
                )
                result = derive_remediation_plan_activation(
                    normalized,
                    current=plan,
                    original_actor=original_actor,
                    remediation_authority_id=by_actor,
                    server_tick=tick,
                    idempotency_key=idempotency_key,
                )
            else:
                plan_id = cast(str, normalized["plan_id"])
                step_id = cast(str, normalized["step_id"])
                plan = cast(
                    RemediationPlan,
                    self._required_record("remediation_plan", plan_id),
                )
                step = next((row for row in plan.steps if row.step_id == step_id), None)
                if step is None:
                    raise WorldError("step is not in remediation plan")
                evidence = tuple(
                    derive_record_governance_evidence(
                        evidence_record=row,
                        merchant_id=plan.owner_merchant_id,
                        expected_subject_id=step_id,
                        evidence_kind=step.action_kind,
                        verified_by_id=by_actor,
                    )
                    for row in self._evidence_for_subject(step_id)
                    if row.owner_id == plan.owner_merchant_id
                    and row.facts.get("evidence_kind") == step.action_kind
                )
                result = derive_remediation_step_completion(
                    normalized,
                    current=plan,
                    evidence=evidence,
                    original_actor=original_actor,
                    remediation_authority_id=by_actor,
                    server_tick=tick,
                    idempotency_key=idempotency_key,
                )
            context = self._governance_context()
            planned = MarketGovernancePlanner().plan_actor_transition(
                context,
                normalized,
                derived_records=(result,),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                projection_effects=projection_effects,
            )
            self._commit_governance(
                planned,
                primary=result,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return result

    def aggregate_reviews(
        self,
        sku_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ReviewAggregate:
        if by_actor != "platform:reviews" or original_actor != by_actor:
            raise WriteNotAuthorized("review aggregation requires platform:reviews")
        request = {"sku_id": sku_id}
        fingerprint = market_governance_request_fingerprint(
            "aggregate_reviews", by_actor, original_actor, request
        )
        scope = "aggregate_reviews"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(ReviewAggregate, retry)
            # Governance tables are keyed for deterministic persistence, not in
            # causal submission order.  Aggregation must therefore reconstruct
            # the protocol's authoritative review order explicitly.  Relying on
            # table iteration makes the second review for a SKU depend on its
            # derived review id and is correctly rejected by the core binding
            # validator as non-authoritative input.
            evidence = tuple(
                sorted(
                    (
                        cast(ReviewEvidence, row.payload)
                        for row in self._record_envelopes("review_evidence")
                        if cast(ReviewEvidence, row.payload).sku_id == sku_id
                    ),
                    key=lambda item: (item.submitted_at_tick, item.review_id),
                )
            )
            aggregate_rows = tuple(
                cast(ReviewAggregate, row.payload)
                for row in self._record_envelopes("review_aggregate")
                if cast(ReviewAggregate, row.payload).sku_id == sku_id
            )
            current = (
                None
                if not aggregate_rows
                else max(aggregate_rows, key=lambda item: item.version)
            )
            result = derive_review_aggregate(
                listing_binding=self._listing_binding(sku_id),
                evidence=evidence,
                aggregation_policy_id="verified-aware",
                aggregation_policy_version=1,
                review_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
                current=current,
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "aggregate_reviews",
                request,
                derived_records=(result,),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
            )
            self._commit_governance(
                planned,
                primary=result,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return result

    def ingest_review_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ReviewEvidence:
        """Derive one review only from persisted evidence and commerce facts.

        The observation may name a SKU and rating.  Reviewer identity comes
        from the evidence owner, account metadata from the current policy, and
        verified-purchase status from an authoritative order and charge ledger.
        """

        if by_actor != "platform:reviews":
            raise WriteNotAuthorized(
                "review observation ingestion requires platform:reviews"
            )
        request = {"record_id": record_id}
        fingerprint = market_governance_request_fingerprint(
            "ingest_review_observation", by_actor, original_actor, request
        )
        scope = "ingest_review_observation"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(ReviewEvidence, retry)
            record = self._current_evidence(record_id)
            if record is None or record.issuer_id != original_actor:
                raise WriteNotAuthorized("review observation issuer mismatch")
            if record.kind != "review_observation":
                raise WorldError("evidence is not a review observation")
            if record.subject_id != record.owner_id:
                raise WriteNotAuthorized(
                    "review observation subject must be its reviewer owner"
                )
            if not record.owner_id.startswith("buyer:"):
                raise WriteNotAuthorized("review observation owner must be a buyer")
            if by_actor != record.owner_id and by_actor not in record.read_acl:
                raise WriteNotAuthorized(
                    "review observation is not readable by platform:reviews"
                )
            if set(record.facts) != {"sku_id", "rating"}:
                raise WorldError(
                    "review observation facts must be exactly sku_id and rating"
                )
            forbidden = _forbidden_nested_fields(
                {"facts": record.facts, "trust": record.trust},
                _REVIEW_OBSERVATION_FORBIDDEN_FIELDS,
            )
            if forbidden:
                raise WriteNotAuthorized(
                    "review observation self-reports World authority fields: "
                    + ", ".join(forbidden)
                )
            sku_id = _required_text(record.facts, "sku_id")
            rating = _required_int(record.facts, "rating", minimum=1, maximum=5)
            reviewer_id = record.owner_id
            account = cast(
                ReviewAccountBinding,
                self._required_policy("review_account_binding", reviewer_id),
            )
            listing = self._listing(sku_id)
            purchase = self._purchase_binding(
                buyer_id=reviewer_id,
                sku_id=sku_id,
                merchant_id=str(listing.merchant_id),
            )
            result = derive_review_evidence(
                {"op": "submit_review", "sku_id": sku_id, "rating": rating},
                listing_binding=self._listing_binding(sku_id),
                account_binding=account,
                purchase_binding=purchase,
                original_actor=reviewer_id,
                review_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
                source_record=record,
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "ingest_review_observation",
                request,
                derived_records=(result,),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                projection_effects=(review_projection_effect(result),),
            )
            self._commit_governance(
                planned,
                primary=result,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return result

    def ingest_market_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> GovernanceCase:
        if by_actor != "platform:governance":
            raise WriteNotAuthorized("market observation requires platform:governance")
        request = {"record_id": record_id}
        fingerprint = market_governance_request_fingerprint(
            "ingest_market_observation", by_actor, original_actor, request
        )
        scope = "ingest_market_observation"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(GovernanceCase, retry)
            record = self._current_evidence(record_id)
            if record is None or record.issuer_id != original_actor:
                raise WriteNotAuthorized("detector evidence issuer mismatch")
            if record.kind != "market_detector_observation":
                raise WorldError("evidence is not a market detector observation")
            facts = record.facts
            sku_values = facts.get("subject_sku_ids")
            source_values = facts.get("source_refs")
            if not isinstance(sku_values, tuple) or not isinstance(source_values, tuple):
                raise WorldError("market detector facts have invalid arrays")
            confidence = record.trust.get("confidence_bps")
            if isinstance(confidence, bool) or not isinstance(confidence, int):
                raise WorldError("market detector confidence is invalid")
            observation = build_market_observation(
                signal_kind=str(facts.get("signal_kind", "")),
                subject_listings=tuple(
                    self._listing_binding(str(value)) for value in sku_values
                ),
                source_refs=tuple(str(value) for value in source_values),
                confidence_bps=confidence,
                evidence_record=record,
                governance_authority_id=by_actor,
            )
            signal = derive_market_signal(
                observation=observation,
                governance_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
            )
            case_kind = (
                "review_integrity"
                if "review" in observation.signal_kind
                else "competition"
            )
            governance_case = derive_governance_case(
                case_kind=case_kind,
                signals=(signal,),
                governance_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "ingest_market_observation",
                request,
                derived_records=(signal, governance_case),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
            )
            self._commit_governance(
                planned,
                primary=governance_case,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return governance_case

    def resolve_governance_case(
        self,
        decision_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> GovernanceCase:
        if by_actor != "platform:governance" or original_actor != by_actor:
            raise WriteNotAuthorized("only platform:governance may resolve a case")
        request = _plain_mapping(decision_intent)
        _require_exact_fields(
            request,
            required={"case_id", "resolution_kind", "policy_id", "policy_version"},
        )
        fingerprint = market_governance_request_fingerprint(
            "resolve_governance_case", by_actor, original_actor, request
        )
        scope = "resolve_governance_case"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(GovernanceCase, retry)
            case_id = _required_text(request, "case_id")
            current = cast(
                GovernanceCase,
                self._required_record("governance_case", case_id),
            )
            signals = self._signals_for_case(current)
            responses = tuple(
                cast(GovernanceResponseAttestation, row.payload)
                for row in self._record_envelopes(
                    "governance_response_attestation"
                )
                if cast(GovernanceResponseAttestation, row.payload).case_id == case_id
            )
            decision = derive_governance_resolution_decision(
                governance_case=current,
                signals=signals,
                responses=responses,
                resolution_kind=_required_text(request, "resolution_kind"),
                policy_id=_required_text(request, "policy_id"),
                policy_version=_required_int(request, "policy_version", minimum=1),
                original_actor=original_actor,
                governance_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
            )
            resolved = derive_governance_case_resolution(
                current=current,
                signals=signals,
                decision=decision,
                governance_authority_id=by_actor,
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "resolve_governance_case",
                request,
                derived_records=(decision, resolved),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                extra_subject_ids={decision.decision_id: current.subject_merchant_ids},
            )
            self._commit_governance(
                planned,
                primary=resolved,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return resolved

    def apply_governance_reputation(
        self,
        source_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ReputationEvent:
        if by_actor != "platform:reputation" or original_actor != by_actor:
            raise WriteNotAuthorized("reputation requires platform:reputation")
        request = _plain_mapping(source_intent)
        _require_exact_fields(
            request,
            required={"source_kind", "source_id"},
            optional={"subject_sku_id"},
        )
        fingerprint = market_governance_request_fingerprint(
            "apply_governance_reputation", by_actor, original_actor, request
        )
        scope = "apply_governance_reputation"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(ReputationEvent, retry)
            policy = cast(
                ReputationPolicyRevision,
                self._latest_policy_of_kind("reputation_policy_revision"),
            )
            if policy is None:
                raise WorldError("reputation policy is missing")
            source_kind = _required_text(request, "source_kind")
            source_id = _required_text(request, "source_id")
            sku_id = request.get("subject_sku_id")
            if sku_id is not None and not isinstance(sku_id, str):
                raise WorldError("subject_sku_id must be text")
            if source_kind == "governance_case":
                source = cast(
                    GovernanceCase,
                    self._required_record("governance_case", source_id),
                )
                listing = self._select_subject_listing(
                    source.subject_merchant_ids, sku_id
                )
                current = cast(
                    ReputationEvent | None,
                    self._latest_record("reputation_event", str(listing.merchant_id)),
                )
                result = derive_governance_reputation_event(
                    governance_case=source,
                    listing_binding=self._listing_binding(str(listing.sku_id)),
                    policy=policy,
                    reputation_authority_id=by_actor,
                    server_tick=self._logical_time + 1,
                    idempotency_key=idempotency_key,
                    current=current,
                )
            elif source_kind == "remediation_plan":
                source = cast(
                    RemediationPlan,
                    self._required_record("remediation_plan", source_id),
                )
                listing = self._select_subject_listing(
                    (source.owner_merchant_id,), sku_id
                )
                current = cast(
                    ReputationEvent | None,
                    self._latest_record("reputation_event", source.owner_merchant_id),
                )
                result = derive_remediation_reputation_event(
                    remediation_plan=source,
                    listing_binding=self._listing_binding(str(listing.sku_id)),
                    policy=policy,
                    reputation_authority_id=by_actor,
                    server_tick=self._logical_time + 1,
                    idempotency_key=idempotency_key,
                    current=current,
                )
            elif source_kind == "settlement":
                purchase = self._purchase_binding(order_id=source_id)
                if purchase is None:
                    raise WorldError("settlement source is missing")
                current = cast(
                    ReputationEvent | None,
                    self._latest_record("reputation_event", purchase.merchant_id),
                )
                result = derive_settlement_reputation_event(
                    purchase_binding=purchase,
                    listing_binding=self._listing_binding(purchase.sku_id),
                    policy=policy,
                    reputation_authority_id=by_actor,
                    server_tick=self._logical_time + 1,
                    idempotency_key=idempotency_key,
                    current=current,
                )
            else:
                raise WorldError(f"unsupported reputation source {source_kind!r}")
            before = self._tables["reputation"].read(result.merchant_id, caller=None)
            before_digest = (
                None if before is None else canonical_digest(_reputation_wire(before))
            )
            effect = reputation_projection_effect(
                result, before_digest=before_digest
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "apply_governance_reputation",
                request,
                derived_records=(result,),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
                projection_effects=(effect,),
            )
            self._commit_governance(
                planned,
                primary=result,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return result

    def create_remediation_plan(
        self,
        plan_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> RemediationPlan:
        if by_actor != "platform:remediation" or original_actor != by_actor:
            raise WriteNotAuthorized("plan creation requires platform:remediation")
        request = _plain_mapping(plan_intent)
        _require_exact_fields(
            request, required={"case_id", "sku_id", "blueprint_id"}
        )
        fingerprint = market_governance_request_fingerprint(
            "create_remediation_plan", by_actor, original_actor, request
        )
        scope = "create_remediation_plan"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(RemediationPlan, retry)
            governance_case = cast(
                GovernanceCase,
                self._required_record(
                    "governance_case", _required_text(request, "case_id")
                ),
            )
            sku_id = _required_text(request, "sku_id")
            blueprint = cast(
                RemediationBlueprint,
                self._required_policy(
                    "remediation_blueprint",
                    _required_text(request, "blueprint_id"),
                ),
            )
            result = derive_remediation_plan(
                governance_case=governance_case,
                listing_binding=self._listing_binding(sku_id),
                blueprint=blueprint,
                remediation_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "create_remediation_plan",
                request,
                derived_records=(result,),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
            )
            self._commit_governance(
                planned,
                primary=result,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return result

    def verify_remediation_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> RemediationPlan:
        if by_actor != "platform:remediation" or original_actor != by_actor:
            raise WriteNotAuthorized("step verification requires platform:remediation")
        request = {"plan_id": plan_id, "step_id": step_id}
        fingerprint = market_governance_request_fingerprint(
            "verify_remediation_step", by_actor, original_actor, request
        )
        scope = "verify_remediation_step"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(RemediationPlan, retry)
            current = cast(
                RemediationPlan,
                self._required_record("remediation_plan", plan_id),
            )
            result = derive_remediation_step_verification(
                current=current,
                step_id=step_id,
                original_actor=original_actor,
                remediation_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "verify_remediation_step",
                request,
                derived_records=(result,),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
            )
            self._commit_governance(
                planned,
                primary=result,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return result

    def persist_ranking_context(
        self,
        ranking_result: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> RankingContext:
        if by_actor != "platform:ranking":
            raise WriteNotAuthorized("ranking context requires platform:ranking")
        request = _plain_mapping(ranking_result)
        _require_exact_fields(
            request,
            required={
                "request_id",
                "mandate_id",
                "candidate_sku_ids",
                "ranked_sku_ids",
                "policy_id",
                "policy_version",
            },
        )
        fingerprint = market_governance_request_fingerprint(
            "persist_ranking_context", by_actor, original_actor, request
        )
        scope = "persist_ranking_context"
        with self._lock:
            retry = self._governance_retry(
                scope, original_actor, idempotency_key, fingerprint
            )
            if retry is not None:
                return cast(RankingContext, retry)
            mandate_id = _required_text(request, "mandate_id")
            mandate = self._latest_mandate(mandate_id)
            if mandate is None:
                raise WorldError("ranking mandate revision is missing")
            candidates_raw = request.get("candidate_sku_ids")
            ranked_raw = request.get("ranked_sku_ids")
            if not isinstance(candidates_raw, list) or not isinstance(ranked_raw, list):
                raise WorldError("ranking candidates and result must be arrays")
            candidates = tuple(str(value) for value in candidates_raw)
            merchants = {str(self._listing(value).merchant_id) for value in candidates}
            placements = tuple(
                placement
                for envelope in self._record_envelopes("campaign")
                for placement in cast(Campaign, envelope.payload).placements
                if cast(Campaign, envelope.payload).status == "active"
                and placement.sku_id in candidates
            )
            aggregates = tuple(
                cast(ReviewAggregate, row.payload)
                for row in self._latest_records_by_stream("review_aggregate")
                if cast(ReviewAggregate, row.payload).sku_id in candidates
            )
            cases = tuple(
                cast(GovernanceCase, row.payload)
                for row in self._latest_records_by_stream("governance_case")
                if set(cast(GovernanceCase, row.payload).subject_merchant_ids).issubset(
                    merchants
                )
            )
            reputation = tuple(
                cast(ReputationEvent, row.payload)
                for row in self._latest_records_by_stream("reputation_event")
                if cast(ReputationEvent, row.payload).merchant_id in merchants
            )
            current = next(
                (
                    cast(RankingContext, row.payload)
                    for row in reversed(self._record_envelopes("ranking_context"))
                    if cast(RankingContext, row.payload).request_id
                    == _required_text(request, "request_id")
                    and cast(RankingContext, row.payload).buyer_id == original_actor
                ),
                None,
            )
            result = derive_ranking_context(
                request_id=_required_text(request, "request_id"),
                requester_id=original_actor,
                mandate_revision=mandate,
                candidate_bindings=tuple(
                    self._listing_binding(value) for value in candidates
                ),
                ranked_sku_ids=tuple(str(value) for value in ranked_raw),
                placements=placements,
                review_aggregates=aggregates,
                governance_cases=cases,
                reputation_events=reputation,
                policy_id=_required_text(request, "policy_id"),
                policy_version=_required_int(request, "policy_version", minimum=1),
                ranking_authority_id=by_actor,
                server_tick=self._logical_time + 1,
                idempotency_key=idempotency_key,
                current=current,
            )
            planned = MarketGovernancePlanner().plan_service_transition(
                self._governance_context(),
                "persist_ranking_context",
                request,
                derived_records=(result,),
                service_actor=by_actor,
                original_actor=original_actor,
                idempotency_key=idempotency_key,
            )
            self._commit_governance(
                planned,
                primary=result,
                scope=scope,
                outer_fingerprint=fingerprint,
            )
            return result

    def ranking_context_projection(
        self,
        context_id: str,
        *,
        caller: str,
    ) -> dict[str, Any]:
        """Return the minimal public annotations sealed into one rank result.

        Every annotation is resolved by an exact digest already present in the
        immutable RankingContext.  Newer campaign, review, case, or reputation
        versions are intentionally ignored, so a later World change cannot
        rewrite what the buyer observed for this search.
        """

        with self._lock:
            context = cast(
                RankingContext,
                self._required_record("ranking_context", context_id),
            )
            if caller not in {context.buyer_id, "platform:aggregator"}:
                raise WriteNotAuthorized(
                    "ranking projection is visible only to its buyer or aggregator"
                )

            placements: dict[str, Placement] = {}
            for envelope in self._record_envelopes("campaign"):
                campaign = cast(Campaign, envelope.payload)
                for placement in campaign.placements:
                    previous = placements.get(placement.placement_digest)
                    if previous is not None and previous != placement:
                        raise WorldError(
                            "placement digest identifies conflicting campaign values"
                        )
                    placements[placement.placement_digest] = placement

            aggregates = {
                cast(ReviewAggregate, row.payload).aggregate_digest: cast(
                    ReviewAggregate, row.payload
                )
                for row in self._record_envelopes("review_aggregate")
            }
            cases = {
                cast(GovernanceCase, row.payload).case_digest: cast(
                    GovernanceCase, row.payload
                )
                for row in self._record_envelopes("governance_case")
            }
            reputation = {
                cast(ReputationEvent, row.payload).event_digest: cast(
                    ReputationEvent, row.payload
                )
                for row in self._record_envelopes("reputation_event")
            }

            selected_placements = _exact_digest_rows(
                context.placement_digests,
                placements,
                label="ranking placement",
            )
            selected_aggregates = _exact_digest_rows(
                context.review_aggregate_digests,
                aggregates,
                label="ranking review aggregate",
            )
            selected_cases = _exact_digest_rows(
                context.governance_case_digests,
                cases,
                label="ranking governance case",
            )
            selected_reputation = _exact_digest_rows(
                context.reputation_event_digests,
                reputation,
                label="ranking reputation event",
            )

            annotations: list[dict[str, Any]] = []
            for sku_id in context.candidate_sku_ids:
                listing = self._listing(sku_id)
                merchant_id = str(listing.merchant_id)
                sku_placements = sorted(
                    (
                        row
                        for row in selected_placements
                        if row.sku_id == sku_id
                        and row.owner_merchant_id == merchant_id
                    ),
                    key=lambda row: row.placement_id,
                )
                sku_aggregates = [
                    row for row in selected_aggregates if row.sku_id == sku_id
                ]
                if len(sku_aggregates) > 1:
                    raise WorldError(
                        "ranking context binds multiple review aggregates for one SKU"
                    )
                merchant_reputation = [
                    row
                    for row in selected_reputation
                    if row.merchant_id == merchant_id
                ]
                if len(merchant_reputation) > 1:
                    raise WorldError(
                        "ranking context binds multiple reputation events for one merchant"
                    )
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
                "schema_version": GOVERNANCE_RANKING_PROJECTION_SCHEMA,
                "context_id": context.context_id,
                "context_digest": context.context_digest,
                "candidate_annotations": annotations,
            }

    def governance_history(
        self, record_kind: str, stable_id: str, *, caller: str
    ) -> tuple[Any, ...]:
        with self._lock:
            return tuple(
                row.payload
                for row in self._governance_tables().history(
                    cast(Any, record_kind), stable_id, caller=caller
                )
            )

    # ---- trusted World loaders and one atomic commit -----------------

    def _governance_tables(self) -> GovernanceTables:
        projection = GovernanceTables()
        # Physical tables expose deterministic key order, but the version
        # suffix is textual (``...:10`` sorts before ``...:2``).  Rehydrating
        # the typed governance projection must append every stream in numeric
        # lineage order or the first two-digit version appears to have no
        # predecessor.  Keep streams independent and sort by their typed
        # revision/version metadata rather than persistence-key spelling.
        policy_rows = sorted(
            self._tables["governance_policies"].all(),
            key=lambda item: (
                item[1].kind,
                item[1].stable_id,
                item[1].revision,
            ),
        )
        for key, row in policy_rows:
            projection.append(GovernanceWrite("governance_policies", key, row))
        record_rows = sorted(
            self._tables["governance_records"].all(),
            key=lambda item: (
                item[1].kind,
                item[1].stable_id,
                item[1].version,
            ),
        )
        for key, row in record_rows:
            projection.append(GovernanceWrite("governance_records", key, row))
        return projection

    def _governance_context(self) -> TrustedGovernanceContext:
        return TrustedGovernanceContext(
            logical_tick=self._logical_time,
            tables=self._governance_tables(),
        )

    def _governance_retry(
        self,
        scope: str,
        actor: str,
        idempotency_key: str,
        fingerprint: str,
    ) -> Any | None:
        key = authority_operation_key(scope, actor, idempotency_key)
        operation = cast(
            AuthorityOperationRecord | None,
            self._tables["authority_operations"].read(key, caller="runtime"),
        )
        if operation is None:
            return None
        if operation.request_fingerprint != fingerprint:
            raise IdempotencyConflict(
                "governance idempotency key was reused for another request"
            )
        envelope = self._tables[operation.outcome_table].read(
            operation.outcome_key, caller="world"
        )
        if not isinstance(envelope, (GovernancePolicyEnvelope, GovernanceRecordEnvelope)):
            raise WorldError("governance idempotency outcome is missing")
        return envelope.payload

    def _commit_governance(
        self,
        planned: GovernanceCommandResult,
        *,
        primary: Any,
        scope: str,
        outer_fingerprint: str,
    ) -> None:
        commit = planned.commit
        if commit is None:
            raise WorldError("fresh governance command produced no commit")
        mutations: list[tuple[str, str, Any | None, Any]] = []
        primary_collection = ""
        primary_key = ""
        for write in commit.writes:
            mutations.append((write.collection, write.key, None, write.value))
            if write.value.payload == primary:
                primary_collection, primary_key = write.collection, write.key
        if not primary_key:
            raise WorldError("governance primary result is absent from commit")
        for effect in commit.projection_effects:
            before = self._tables[effect.table].read(effect.key, caller=None)
            if effect.table == "reviews":
                after = Review(
                    review_id=ReviewId(str(effect.payload["review_id"])),
                    reviewer_id=AgentId(str(effect.payload["reviewer_id"])),
                    sku_id=SkuId(str(effect.payload["sku_id"])),
                    merchant_id=AgentId(str(effect.payload["merchant_id"])),
                    rating=int(effect.payload["rating"]),
                    text=str(effect.payload["text"]),
                )
                observed_digest = None if before is None else canonical_digest(before)
            else:
                if before is not None and not isinstance(before, ReputationScore):
                    raise WorldError("reputation projection has wrong type")
                outcome_bps = int(effect.payload["outcome_bps"])
                after = ReputationScore(
                    merchant_id=AgentId(str(effect.payload["merchant_id"])),
                    rolling_avg=outcome_bps / 2000.0,
                    n_settled=(
                        (0 if before is None else before.n_settled)
                        + (1 if effect.payload["event_kind"] == "fulfilled_order" else 0)
                    ),
                    n_disputed=(
                        (0 if before is None else before.n_disputed)
                        + (
                            1
                            if effect.payload["event_kind"]
                            in {"compliance_violation", "disputed_order"}
                            else 0
                        )
                    ),
                )
                observed_digest = (
                    None if before is None else canonical_digest(_reputation_wire(before))
                )
            if observed_digest != effect.before_digest:
                raise WorldError("governance projection changed before commit")
            mutations.append((effect.table, effect.key, before, after))
        self._commit_simple_authority_transition(
            operation=planned.operation.operation,
            authority_action=f"world.{planned.operation.operation}",
            scope=scope,
            original_actor=planned.operation.original_actor,
            idempotency_key=planned.operation.idempotency_key,
            request_fingerprint=outer_fingerprint,
            subject_id=getattr(primary, "case_id", getattr(primary, "campaign_id", primary_key)),
            outcome_table=primary_collection,
            outcome_key=primary_key,
            tick=commit.logical_tick,
            mutations=tuple(mutations),
            invariants=(
                "governance-context-cas",
                "typed-governance-records",
                "actor-scoped-authority",
                "atomic-projection",
                "single-clock-advance",
            ),
        )

    def _record_envelopes(self, kind: str) -> tuple[GovernanceRecordEnvelope, ...]:
        return tuple(
            row
            for _, row in self._tables["governance_records"].all()
            if row.kind == kind
        )

    def _latest_records_by_stream(
        self, kind: str
    ) -> tuple[GovernanceRecordEnvelope, ...]:
        latest: dict[str, GovernanceRecordEnvelope] = {}
        for row in self._record_envelopes(kind):
            if row.stable_id not in latest or row.version > latest[row.stable_id].version:
                latest[row.stable_id] = row
        return tuple(latest[key] for key in sorted(latest))

    def _latest_record(self, kind: str, stable_id: str) -> Any | None:
        rows = [
            row
            for row in self._record_envelopes(kind)
            if row.stable_id == stable_id
        ]
        return None if not rows else max(rows, key=lambda row: row.version).payload

    def _required_record(self, kind: str, stable_id: str) -> Any:
        value = self._latest_record(kind, stable_id)
        if value is None:
            raise WorldError(f"governance record {kind}:{stable_id} is missing")
        return value

    def _latest_policy(self, kind: str, stable_id: str) -> Any | None:
        rows = [
            row
            for _, row in self._tables["governance_policies"].all()
            if row.kind == kind and row.stable_id == stable_id
        ]
        return None if not rows else max(rows, key=lambda row: row.revision).payload

    def _latest_policy_of_kind(self, kind: str) -> Any | None:
        rows = [
            row
            for _, row in self._tables["governance_policies"].all()
            if row.kind == kind
        ]
        return None if not rows else max(rows, key=lambda row: row.revision).payload

    def _required_policy(self, kind: str, stable_id: str) -> Any:
        value = self._latest_policy(kind, stable_id)
        if value is None:
            raise WorldError(f"governance policy {kind}:{stable_id} is missing")
        return value

    def _listing(self, sku_id: str) -> Listing:
        row = self._tables["catalog"].read(sku_id, caller=None)
        if not isinstance(row, Listing):
            raise WorldError(f"catalog listing {sku_id!r} is missing")
        return row

    def _listing_binding(self, sku_id: str) -> Any:
        listing = self._listing(sku_id)
        return derive_listing_governance_binding(
            listing,
            catalog_revision=_listing_revision(listing),
        )

    def _purchase_binding(
        self,
        *,
        order_id: str | None = None,
        buyer_id: str | None = None,
        sku_id: str | None = None,
        merchant_id: str | None = None,
    ) -> Any | None:
        orders = [
            row
            for _, row in self._tables["orders"].all()
            if isinstance(row, Order)
            and (order_id is None or str(row.order_id) == order_id)
            and (buyer_id is None or str(row.buyer_id) == buyer_id)
            and (sku_id is None or str(row.sku_id) == sku_id)
            and (merchant_id is None or str(row.merchant_id) == merchant_id)
            and row.state
            in {
                OrderState.PARTIALLY_SETTLED,
                OrderState.SETTLED,
                OrderState.DISPATCHED,
                OrderState.RETURNED,
                OrderState.REFUNDED,
            }
        ]
        for order in sorted(orders, key=lambda row: str(row.order_id)):
            receipts = [
                receipt
                for _, receipt in self._tables["ledger"].all()
                if isinstance(receipt, Receipt)
                and str(receipt.order_id) == str(order.order_id)
                and receipt.effect == "charge"
            ]
            if not receipts:
                continue
            receipt = sorted(receipts, key=lambda row: str(row.txn_id))[0]
            listing = self._listing(str(order.sku_id))
            return derive_purchase_governance_binding(
                order=order,
                receipt=receipt,
                listing=listing,
                settled_at_tick=_receipt_tick(receipt),
                catalog_revision=_listing_revision(listing),
            )
        return None

    def _current_evidence(self, record_id: str) -> EvidenceRecord | None:
        rows = [
            row
            for _, row in self._tables["evidence_records"].all()
            if row.record_id == record_id
        ]
        return None if not rows else max(rows, key=lambda row: row.version)

    def _evidence_for_subject(self, subject_id: str) -> tuple[EvidenceRecord, ...]:
        return tuple(
            row
            for _, row in self._tables["evidence_records"].all()
            if row.subject_id == subject_id
        )

    def _signals_for_case(self, governance_case: GovernanceCase) -> tuple[MarketSignal, ...]:
        by_digest = {
            cast(MarketSignal, row.payload).signal_digest: cast(MarketSignal, row.payload)
            for row in self._record_envelopes("market_signal")
        }
        try:
            return tuple(by_digest[digest] for digest in governance_case.signal_digests)
        except KeyError as exc:
            raise WorldError("governance case signal is missing") from exc

    def _latest_mandate(self, mandate_id: str) -> MandateRevision | None:
        rows = [
            row
            for _, row in self._tables["mandate_revisions"].all()
            if row.mandate_id == mandate_id
        ]
        return None if not rows else max(rows, key=lambda row: row.revision)

    def _select_subject_listing(
        self, merchant_ids: Sequence[str], sku_id: str | None
    ) -> Listing:
        if sku_id is not None:
            listing = self._listing(sku_id)
            if str(listing.merchant_id) not in merchant_ids:
                raise WriteNotAuthorized("selector listing is not a case subject")
            return listing
        candidates = [
            row
            for _, row in self._tables["catalog"].all()
            if str(row.merchant_id) in merchant_ids
        ]
        if len(candidates) != 1:
            raise WorldError("governance source requires an unambiguous SKU selector")
        return candidates[0]


def _plain_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorldError("governance request must be an object")
    if any(not isinstance(key, str) for key in value):
        raise WorldError("governance request keys must be text")
    return dict(value)


def _forbidden_nested_fields(
    value: Any,
    forbidden: frozenset[str],
    *,
    path: tuple[str, ...] = (),
) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            next_path = (*path, key)
            if key in forbidden:
                found.append(".".join(next_path))
            found.extend(
                _forbidden_nested_fields(nested, forbidden, path=next_path)
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            found.extend(
                _forbidden_nested_fields(
                    nested,
                    forbidden,
                    path=(*path, f"[{index}]"),
                )
            )
    return tuple(sorted(set(found)))


def _exact_digest_rows(
    digests: Sequence[str],
    rows: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Any, ...]:
    if len(set(digests)) != len(digests):
        raise WorldError(f"{label} digests are not unique")
    try:
        return tuple(rows[digest] for digest in digests)
    except KeyError as exc:
        raise WorldError(f"{label} digest has no exact persisted record") from exc


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (set() if optional is None else optional)
    actual = set(value)
    if actual != required and not (
        optional is not None and required.issubset(actual) and actual.issubset(allowed)
    ):
        missing = sorted(required - actual)
        extra = sorted(actual - allowed)
        raise WorldError(
            f"governance request fields are not exact; missing={missing}, extra={extra}"
        )


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise WorldError(f"{key} must be non-empty text")
    return result


def _required_int(
    value: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise WorldError(f"{key} is outside its allowed range")
    if maximum is not None and result > maximum:
        raise WorldError(f"{key} is outside its allowed range")
    return result


def _listing_revision(listing: Listing) -> int:
    value = listing.attributes.get("catalog_revision", 1)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1


def _receipt_tick(receipt: Receipt) -> int:
    prefix = "world-tick:"
    if receipt.ts.startswith(prefix):
        try:
            return int(receipt.ts[len(prefix) :])
        except ValueError:
            pass
    return 0


def _reputation_wire(value: ReputationScore) -> dict[str, Any]:
    return {
        "merchant_id": str(value.merchant_id),
        "rolling_avg": value.rolling_avg,
        "n_settled": value.n_settled,
        "n_disputed": value.n_disputed,
    }


__all__ = [
    "MarketGovernanceWorldMixin",
    "governance_policy_authority",
    "market_governance_request_fingerprint",
]
