"""Framework authority for one high-level marketplace-governance turn.

The model chooses a business decision.  It never authors campaign,
placement, case, plan, step, order, receipt, actor, version, digest, or
destination fields.  This module validates an authenticated current Platform
delivery against exact World-visible records and compiles the model's small
choice into the existing ``commerce.*`` wire contract.

The boundary is intentionally independent from benchmark tasks and scenario
oracles.  It also does not implement a second governance state machine.  All
record shape, digest, owner, and successor validation is delegated to the
canonical market-governance protocol validators.  World and Platform remain
responsible for the final atomic transition and idempotency check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from agents.business_decision import BusinessIntentSpec
from agents.decision_errors import (
    ModelBusinessDecisionError,
    PlatformContractError,
    SemanticBoundaryError,
)
from protocol.envelope import Envelope
from protocol.errors import SchemaError
from protocol.evidence_records import (
    EvidenceRecord,
    coerce_evidence_record,
    validate_evidence_record,
)
from protocol.market_governance import (
    Campaign,
    GovernanceCase,
    GovernanceRecord,
    RemediationPlan,
    ReviewEvidence,
    validate_campaign,
    validate_governance_case,
    validate_remediation_plan,
    validate_review_evidence,
    validate_version_successor,
)
from protocol.remediation_audit import REMEDIATION_AUDITOR_SERVICE_ID
from world.market_governance_core import (
    AdsCampaignTerms,
    GovernanceBindingError,
    validate_ads_campaign_terms,
)


_ADS = "platform:ads"
_GOVERNANCE = "platform:governance"
_REMEDIATION = "platform:remediation"
_REVIEWS = "platform:reviews"
_EVIDENCE = "platform:evidence"

_CAMPAIGN_TERMS_NOTICE = "platform.campaign_terms_notice"
_REVIEW_REQUEST = "platform.review_submission_requested"
_GOVERNANCE_UPDATED = "platform.governance_updated"
_CASE_NOTICE = "platform.governance_case_notice"
_PLAN_NOTICE = "platform.remediation_plan_notice"
_EVIDENCE_PERSISTED = "platform.evidence_record_persisted"

_PUBLISH = "publish_campaign"
_DISCLOSE = "disclose_placement"
_ACTIVATE = "activate_campaign"
_REJECT_COORDINATION = "reject_coordination"
_REJECT_REVIEW = "reject_review_manipulation"
_ACCEPT_PLAN = "accept_remediation_plan"
_COMPLETE_STEP = "complete_remediation_step"
_READ_HISTORY = "read_governance_history"
_SUBMIT_REVIEW = "submit_review"

_REJECT_COORDINATION_NOTICE_ACTION = "commerce.reject_coordination"
_REJECT_REVIEW_NOTICE_ACTION = "commerce.reject_review_manipulation"
_ACCEPT_PLAN_NOTICE_ACTION = "commerce.accept_remediation_plan"
_COMPLETE_STEP_NOTICE_ACTION = "commerce.complete_remediation_step"

_MAX_DISCLOSURE_LENGTH = 1_000
_MAX_REVIEW_TEXT_LENGTH = 4_000
_DIGEST_LENGTH = 64


class GovernanceTurnError(SemanticBoundaryError):
    """Base error at the high-level governance Agent boundary."""


class GovernanceFrameworkAuthorityError(
    PlatformContractError,
    GovernanceTurnError,
):
    """Authenticated Platform or World authority is absent or inconsistent."""


class _NoGovernanceBusinessTurn(GovernanceFrameworkAuthorityError):
    """A valid projection is terminal and grants no model-owned transition."""


class GovernanceModelChoiceError(ModelBusinessDecisionError, GovernanceTurnError):
    """The model selected a business intent or argument outside its contract."""


# Short aliases keep the boundary vocabulary consistent with the other
# high-level Agent authority modules while retaining grep-friendly domain names.
FrameworkAuthorityError = GovernanceFrameworkAuthorityError
ModelChoiceError = GovernanceModelChoiceError


@dataclass(frozen=True, slots=True)
class VerifiedPurchaseRecord:
    """A caller-visible World projection that authorizes one verified review.

    World is expected to derive this projection from one settled order and its
    charge receipt.  ``record_digest`` seals the exact projection so a Platform
    notice cannot silently swap the buyer, merchant, SKU, or receipt.
    """

    order_id: str
    txn_id: str
    buyer_id: str
    merchant_id: str
    sku_id: str
    settled_at_tick: int
    record_digest: str


@dataclass(frozen=True, slots=True)
class GovernanceHistory:
    """One exact World-visible governance stream in version order."""

    record_kind: Literal["campaign", "governance_case", "remediation_plan"]
    stable_id: str
    records: tuple[GovernanceRecord, ...]


@dataclass(frozen=True, slots=True)
class CompiledGovernanceAction:
    """Business operation and authority-bound payload ready for Agent routing."""

    operation: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Choice:
    operation: str
    wire_payload: tuple[tuple[str, Any], ...]
    selector: str | None = None
    selector_description: str | None = None

    def payload(self) -> dict[str, Any]:
        return dict(self.wire_payload)


@dataclass(frozen=True, slots=True)
class GovernanceTurnAuthority:
    """Validated authority and business-decision surface for one current inbound."""

    actor_id: str
    inbound_msg_id: str
    inbound_kind: str
    current_tick: int
    choices: tuple[_Choice, ...]

    @classmethod
    def from_inbound(
        cls,
        inbound: Envelope,
        *,
        actor_id: str,
        current_tick: int,
        campaign_terms: Sequence[AdsCampaignTerms] = (),
        campaign_listing_owners: Mapping[str, str] | None = None,
        campaigns: Sequence[Campaign] = (),
        governance_cases: Sequence[GovernanceCase] = (),
        remediation_plans: Sequence[RemediationPlan] = (),
        histories: Sequence[GovernanceHistory] = (),
        evidence_records: Sequence[EvidenceRecord | Mapping[str, Any]] = (),
        verified_purchases: Sequence[VerifiedPurchaseRecord] = (),
        existing_reviews: Sequence[ReviewEvidence] = (),
    ) -> "GovernanceTurnAuthority":
        """Validate one authenticated Platform delivery and current World view.

        Every supplied record must be the result of an actor-authorized World
        read for this turn.  The method validates all supplied records, not
        only the one eventually selected, so a cross-owner or corrupt record
        cannot hide behind an unused model choice.
        """

        _require_actor(actor_id)
        tick = _framework_nonnegative_int(current_tick, "current World tick")
        kind, payload = _authenticated_platform_payload(inbound, actor_id=actor_id)

        terms = _validate_terms(campaign_terms)
        owners = _validate_listing_owners(campaign_listing_owners or {})
        campaign_rows = _validate_campaigns(campaigns)
        case_rows = _validate_cases(governance_cases)
        plan_rows = _validate_plans(remediation_plans)
        history_rows = _validate_histories(histories)
        evidence_rows = _validate_evidence(evidence_records, actor_id=actor_id)
        purchase_rows = _validate_purchases(verified_purchases, actor_id=actor_id)
        review_rows = _validate_reviews(existing_reviews)

        if kind == _CAMPAIGN_TERMS_NOTICE:
            choices = _campaign_publish_choices(
                payload,
                actor_id=actor_id,
                current_tick=tick,
                terms=terms,
                listing_owners=owners,
                campaigns=campaign_rows,
            )
        elif kind == _GOVERNANCE_UPDATED:
            choices = _campaign_update_choices(
                payload,
                actor_id=actor_id,
                current_tick=tick,
                campaigns=campaign_rows,
                histories=history_rows,
            )
        elif kind == _CASE_NOTICE:
            choices = _case_notice_choices(
                payload,
                actor_id=actor_id,
                current_tick=tick,
                cases=case_rows,
                histories=history_rows,
            )
        elif kind == _PLAN_NOTICE:
            choices = _plan_notice_choices(
                payload,
                actor_id=actor_id,
                current_tick=tick,
                plans=plan_rows,
                histories=history_rows,
            )
        elif kind == _EVIDENCE_PERSISTED:
            choices = _remediation_evidence_choices(
                payload,
                actor_id=actor_id,
                current_tick=tick,
                plans=plan_rows,
                histories=history_rows,
                evidence_records=evidence_rows,
            )
        elif kind == _REVIEW_REQUEST:
            choices = _review_choices(
                payload,
                actor_id=actor_id,
                current_tick=tick,
                purchases=purchase_rows,
                existing_reviews=review_rows,
            )
        else:
            raise GovernanceFrameworkAuthorityError(
                "current inbound is not a governance authority surface"
            )
        if not choices:
            raise GovernanceFrameworkAuthorityError(
                "current governance delivery has no legal business action"
            )
        return cls(
            actor_id=actor_id,
            inbound_msg_id=inbound.msg_id,
            inbound_kind=kind,
            current_tick=tick,
            choices=tuple(choices),
        )

    @classmethod
    def from_platform_projection(
        cls,
        inbound: Envelope,
        *,
        actor_id: str,
        cache: "GovernanceProjectionCache",
    ) -> "GovernanceTurnAuthority":
        """Bind a turn from an authenticated, actor-private Platform projection.

        Unlike :meth:`from_inbound`, this entry point does not require the
        commerce Agent to reach into World tables.  The Platform delivery is
        already the actor-authorized World projection; ``cache`` verifies its
        exact shape, digest/version succession, owner, and causal evidence
        closure before returning choices.  Dynamic identifiers stay solely in
        the returned private authority and never enter a model-facing schema.
        """

        if not isinstance(cache, GovernanceProjectionCache):
            raise GovernanceFrameworkAuthorityError(
                "governance projection authority cache has the wrong type"
            )
        if cache.actor_id != actor_id:
            raise GovernanceFrameworkAuthorityError(
                "governance projection authority crosses Agent actors"
            )
        return cache.turn_authority(inbound)

    def decision_schema(self) -> tuple[BusinessIntentSpec, ...]:
        """Return business choices without framework-owned identity fields."""

        by_operation: dict[str, list[_Choice]] = {}
        for choice in self.choices:
            by_operation.setdefault(choice.operation, []).append(choice)
        return tuple(
            _decision_spec(name, tuple(rows)) for name, rows in sorted(by_operation.items())
        )

    def compile(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> CompiledGovernanceAction:
        """Bind one model business choice to exact authority-owned fields."""

        if not isinstance(arguments, Mapping):
            raise GovernanceModelChoiceError("governance tool arguments must be an object")
        candidates = [row for row in self.choices if row.operation == operation]
        if not candidates:
            raise GovernanceModelChoiceError(
                "model selected an intent outside the current governance turn"
            )
        selector_required = len(candidates) > 1
        required: set[str] = set()
        optional: set[str] = set()
        if selector_required:
            required.add("target")
        if operation == _DISCLOSE:
            required.add("disclosure_text")
        elif operation == _SUBMIT_REVIEW:
            required.add("rating")
            optional.add("review_text")
        _require_model_keys(arguments, required=required, optional=optional)

        selected = candidates[0]
        if selector_required:
            target = _model_text(arguments.get("target"), "target")
            matches = [row for row in candidates if row.selector == target]
            if len(matches) != 1:
                raise GovernanceModelChoiceError(
                    "selected target is outside the current World-authorized enum"
                )
            selected = matches[0]

        payload = selected.payload()
        if operation == _DISCLOSE:
            payload["disclosure_text"] = _bounded_model_text(
                arguments.get("disclosure_text"),
                "disclosure_text",
                _MAX_DISCLOSURE_LENGTH,
            )
        elif operation == _SUBMIT_REVIEW:
            payload["rating"] = _model_rating(arguments.get("rating"))
            if "review_text" in arguments:
                review_text = _bounded_model_text(
                    arguments.get("review_text"),
                    "review_text",
                    _MAX_REVIEW_TEXT_LENGTH,
                    allow_empty=True,
                )
                if review_text:
                    payload["review_text"] = review_text
        return CompiledGovernanceAction(
            operation=operation,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class _ProjectionPlacement:
    placement_id: str
    sku_id: str
    disclosure_status: str
    disclosure_text: str | None


@dataclass(frozen=True, slots=True)
class _CampaignProjection:
    campaign_id: str
    version: int
    campaign_digest: str
    status: str
    placements: tuple[_ProjectionPlacement, ...]


@dataclass(frozen=True, slots=True)
class _CaseProjection:
    case_id: str
    case_digest: str
    opened_case_id: str
    opened_case_digest: str
    case_kind: str
    status: str
    response_action: str | None
    world_tick: int
    response_recorded: bool = False


@dataclass(frozen=True, slots=True)
class _ProjectionStep:
    step_id: str
    sequence_no: int
    action_kind: str
    status: str


@dataclass(frozen=True, slots=True)
class _PlanProjection:
    plan_id: str
    plan_digest: str
    version: int
    governance_case_id: str
    owner_merchant_id: str
    status: str
    steps: tuple[_ProjectionStep, ...]
    world_tick: int


class GovernanceProjectionCache:
    """Actor-private closure over authenticated governance projections.

    This cache is framework state, not model memory.  It records only values
    delivered by registered Platform services and advances them through exact
    version successors.  In particular, remediation completion is unavailable
    until an independently persisted evidence record matches the cached live
    plan digest, version, owner, and pending step.
    """

    def __init__(self, *, actor_id: str) -> None:
        # Agent factories also support the role-only identities ``buyer`` and
        # ``merchant`` in isolated unit/SDK use.  Governance authority built
        # for a real delivery still requires the envelope recipient to match
        # this identity exactly; accepting the role-only form here merely
        # keeps this dormant private cache compatible with the Agent surface.
        _require_cache_actor(actor_id)
        self.actor_id = actor_id
        self._campaigns: dict[str, _CampaignProjection] = {}
        self._cases: dict[str, _CaseProjection] = {}
        self._plans: dict[str, _PlanProjection] = {}
        self._observed_messages: dict[str, str] = {}

    def observe(self, inbound: Envelope) -> None:
        """Validate and remember one relevant Platform delivery, if any."""

        if not isinstance(inbound, Envelope):
            raise GovernanceFrameworkAuthorityError(
                "governance projection observation requires an Envelope"
            )
        kind = inbound.action.get("kind")
        if kind not in {
            _GOVERNANCE_UPDATED,
            _CASE_NOTICE,
            _PLAN_NOTICE,
            _EVIDENCE_PERSISTED,
        }:
            return
        authenticated_kind, payload = _authenticated_platform_payload(
            inbound,
            actor_id=self.actor_id,
        )
        fingerprint = _canonical_digest(
            {
                "from": inbound.from_,
                "to": inbound.to,
                "kind": authenticated_kind,
                "payload": dict(payload),
            }
        )
        previous = self._observed_messages.get(inbound.msg_id)
        if previous is not None:
            if previous != fingerprint:
                raise GovernanceFrameworkAuthorityError(
                    "governance Platform message identity was reused with different authority"
                )
            return

        if authenticated_kind == _GOVERNANCE_UPDATED:
            self._observe_governance_update(payload)
        elif authenticated_kind == _CASE_NOTICE:
            self._store_case(_case_projection_from_notice(payload))
        elif authenticated_kind == _PLAN_NOTICE:
            plan_reference = _single_reference(payload.get("plan_reference"))
            self._store_plan(
                _plan_projection_from_notice(
                    payload,
                    actor_id=self.actor_id,
                    current=self._plans.get(plan_reference["stable_id"]),
                )
            )
        else:
            # Evidence is intentionally not consumed here.  The same current
            # delivery may be parsed more than once during one model repair;
            # plan succession after the compiled action removes its authority.
            self._validated_evidence_choice(payload)
        self._observed_messages[inbound.msg_id] = fingerprint

    def has_business_turn(self, inbound: Envelope) -> bool:
        """Whether this exact projection contains a model-owned decision.

        The classification intentionally lives beside projection validation,
        not in the generic Agent loop.  ``turn_authority`` remains the single
        implementation of the governance kind/status/operation frontier; a
        valid terminal acknowledgement is represented by the private sentinel
        exception below, while malformed authority continues to fail closed.
        """

        kind = inbound.action.get("kind") if isinstance(inbound, Envelope) else None
        if kind not in {
            _GOVERNANCE_UPDATED,
            _CASE_NOTICE,
            _PLAN_NOTICE,
            _EVIDENCE_PERSISTED,
        }:
            return False
        try:
            self.turn_authority(inbound)
        except _NoGovernanceBusinessTurn:
            return False
        return True

    def turn_authority(self, inbound: Envelope) -> GovernanceTurnAuthority:
        """Return the legal business choices for the exact current delivery."""

        self.observe(inbound)
        kind, payload = _authenticated_platform_payload(
            inbound,
            actor_id=self.actor_id,
        )
        choices: tuple[_Choice, ...]
        tick: int
        if kind == _GOVERNANCE_UPDATED:
            operation = payload.get("operation")
            if operation not in {_PUBLISH, _DISCLOSE, _ACTIVATE}:
                raise _NoGovernanceBusinessTurn(
                    "governance acknowledgement has no model-owned follow-up"
                )
            projection = _campaign_projection_from_update(payload)
            current = self._campaigns.get(projection.campaign_id)
            if current != projection:
                raise GovernanceFrameworkAuthorityError(
                    "campaign projection is no longer current Agent authority"
                )
            choices = _campaign_projection_choices(projection, operation=str(operation))
            tick = projection.version
        elif kind == _CASE_NOTICE:
            projection = _case_projection_from_notice(payload)
            current = self._cases.get(projection.case_id)
            if current is None or current.response_recorded:
                raise _NoGovernanceBusinessTurn(
                    "governance case response is no longer current Agent authority"
                )
            if (
                current.case_digest != projection.case_digest
                or current.case_kind != projection.case_kind
                or current.status != projection.status
            ):
                raise GovernanceFrameworkAuthorityError(
                    "governance case notice is stale against Agent-private authority"
                )
            if projection.response_action is None or projection.status != "open":
                raise _NoGovernanceBusinessTurn(
                    "governance case notice has no actor rejection transition"
                )
            operation = {
                _REJECT_COORDINATION_NOTICE_ACTION: _REJECT_COORDINATION,
                _REJECT_REVIEW_NOTICE_ACTION: _REJECT_REVIEW,
            }[projection.response_action]
            choices = (_choice(operation, {"case_id": projection.case_id}),)
            tick = projection.world_tick
        elif kind == _PLAN_NOTICE:
            plan_reference = _single_reference(payload.get("plan_reference"))
            projection = _plan_projection_from_notice(
                payload,
                actor_id=self.actor_id,
                current=self._plans.get(plan_reference["stable_id"]),
            )
            current = self._plans.get(projection.plan_id)
            if current != projection:
                raise GovernanceFrameworkAuthorityError(
                    "remediation plan notice is no longer current Agent authority"
                )
            if projection.status != "draft":
                raise _NoGovernanceBusinessTurn(
                    "remediation plan notice has no model-owned transition"
                )
            choices = (_choice(_ACCEPT_PLAN, {"plan_id": projection.plan_id}),)
            tick = projection.world_tick
        elif kind == _EVIDENCE_PERSISTED:
            choices, tick = self._validated_evidence_choice(payload)
        else:
            raise _NoGovernanceBusinessTurn(
                "current Platform projection is not a governance business turn"
            )
        if not choices:
            raise GovernanceFrameworkAuthorityError(
                "current governance projection has no legal business action"
            )
        return GovernanceTurnAuthority(
            actor_id=self.actor_id,
            inbound_msg_id=inbound.msg_id,
            inbound_kind=kind,
            current_tick=tick,
            choices=choices,
        )

    def _observe_governance_update(self, payload: Mapping[str, Any]) -> None:
        operation = payload.get("operation")
        result_kind = payload.get("result_kind")
        if operation in {_PUBLISH, _DISCLOSE, _ACTIVATE}:
            self._store_campaign(_campaign_projection_from_update(payload))
            return
        if result_kind == "remediation_plan":
            result_id = _framework_text(
                payload.get("result_id"),
                "remediation result_id",
            )
            self._store_plan(
                _plan_projection_from_update(
                    payload,
                    actor_id=self.actor_id,
                    current=self._plans.get(result_id),
                )
            )
            return
        if operation in {_REJECT_COORDINATION, _REJECT_REVIEW}:
            _require_sender_payload_fields(
                payload,
                {
                    "operation",
                    "result_kind",
                    "result_id",
                    "version",
                    "status",
                    "references",
                    "case_id",
                    "subject_merchant_id",
                    "response_kind",
                },
                _GOVERNANCE_UPDATED,
            )
            if result_kind != "governance_response_attestation":
                raise GovernanceFrameworkAuthorityError(
                    "governance rejection acknowledgement has the wrong result kind"
                )
            if payload.get("subject_merchant_id") != self.actor_id:
                raise GovernanceFrameworkAuthorityError(
                    "governance rejection acknowledgement crosses actor authority"
                )
            case_id = _framework_text(payload.get("case_id"), "governance case_id")
            reference = _single_reference(payload.get("references"))
            if (
                reference["record_kind"] != "governance_response_attestation"
                or reference["stable_id"] != payload.get("result_id")
                or payload.get("status") != "recorded"
                or payload.get("version") != 1
            ):
                raise GovernanceFrameworkAuthorityError(
                    "governance rejection acknowledgement reference is malformed"
                )
            current = self._cases.get(case_id)
            if current is None:
                raise GovernanceFrameworkAuthorityError(
                    "governance rejection acknowledgement has no cached case authority"
                )
            expected_kind = str(operation)
            if payload.get("response_kind") != expected_kind:
                raise GovernanceFrameworkAuthorityError(
                    "governance rejection acknowledgement has the wrong response kind"
                )
            self._cases[case_id] = _CaseProjection(
                case_id=current.case_id,
                case_digest=current.case_digest,
                opened_case_id=current.opened_case_id,
                opened_case_digest=current.opened_case_digest,
                case_kind=current.case_kind,
                status=current.status,
                response_action=current.response_action,
                world_tick=current.world_tick,
                response_recorded=True,
            )

    def _store_campaign(self, candidate: _CampaignProjection) -> None:
        current = self._campaigns.get(candidate.campaign_id)
        if current is not None and current != candidate:
            if candidate.version != current.version + 1:
                raise GovernanceFrameworkAuthorityError(
                    "campaign projection skipped or replayed a World version"
                )
            if candidate.campaign_digest == current.campaign_digest:
                raise GovernanceFrameworkAuthorityError(
                    "campaign projection successor reused its prior digest"
                )
            _validate_campaign_projection_successor(current, candidate)
        self._campaigns[candidate.campaign_id] = candidate

    def _store_case(self, candidate: _CaseProjection) -> None:
        current = self._cases.get(candidate.case_id)
        if current is not None and current != candidate:
            if current.response_recorded:
                if (
                    candidate.status == "open"
                    or candidate.response_action is not None
                    or candidate.opened_case_id != current.opened_case_id
                    or candidate.opened_case_digest != current.opened_case_digest
                    or candidate.case_kind != current.case_kind
                    or candidate.world_tick < current.world_tick
                ):
                    raise GovernanceFrameworkAuthorityError(
                        "governance case notice replayed after the actor response"
                    )
                # A later resolved/dismissed World notice is a legitimate
                # terminal successor.  Preserve the private fact that this
                # actor already responded so it can never reopen authority.
                candidate = replace(candidate, response_recorded=True)
            if (
                candidate.opened_case_id != current.opened_case_id
                or candidate.opened_case_digest != current.opened_case_digest
                or candidate.world_tick < current.world_tick
            ):
                raise GovernanceFrameworkAuthorityError(
                    "governance case projection broke its cached authority chain"
                )
        self._cases[candidate.case_id] = candidate

    def _store_plan(self, candidate: _PlanProjection) -> None:
        current = self._plans.get(candidate.plan_id)
        if current is not None and current != candidate:
            same_record = (
                candidate.plan_digest == current.plan_digest
                and candidate.version == current.version
                and candidate.governance_case_id == current.governance_case_id
                and candidate.owner_merchant_id == current.owner_merchant_id
                and candidate.status == current.status
                and candidate.steps == current.steps
            )
            if same_record:
                if candidate.world_tick < current.world_tick:
                    raise GovernanceFrameworkAuthorityError(
                        "remediation projection World time moved backwards"
                    )
                self._plans[candidate.plan_id] = candidate
                return
            if candidate.version != current.version + 1:
                raise GovernanceFrameworkAuthorityError(
                    "remediation projection skipped or replayed a World version"
                )
            if (
                candidate.governance_case_id != current.governance_case_id
                or candidate.owner_merchant_id != current.owner_merchant_id
                or candidate.plan_digest == current.plan_digest
                or candidate.world_tick < current.world_tick
            ):
                raise GovernanceFrameworkAuthorityError(
                    "remediation projection broke its cached authority chain"
                )
            _validate_plan_projection_successor(current, candidate)
        self._plans[candidate.plan_id] = candidate

    def _validated_evidence_choice(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[tuple[_Choice, ...], int]:
        _require_sender_payload_fields(payload, {"record"}, _EVIDENCE_PERSISTED)
        record = _coerce_evidence(payload["record"])
        if (
            record.kind != "independent_governance_audit"
            or record.issuer_id != REMEDIATION_AUDITOR_SERVICE_ID
            or record.owner_id != self.actor_id
            or self.actor_id not in record.read_acl
        ):
            raise GovernanceFrameworkAuthorityError(
                "remediation evidence lacks exact auditor, owner, or read authority"
            )
        facts = record.facts
        required_facts = {
            "evidence_kind",
            "result",
            "plan_id",
            "plan_version",
            "plan_digest",
            "sequence_no",
            "request_msg_id",
            "request_fingerprint",
        }
        if set(facts) != required_facts:
            raise GovernanceFrameworkAuthorityError(
                "remediation audit evidence facts are not exact"
            )
        plan_id = _framework_text(facts.get("plan_id"), "remediation evidence plan_id")
        plan = self._plans.get(plan_id)
        if plan is None or plan.status != "active":
            raise GovernanceFrameworkAuthorityError(
                "remediation evidence has no cached active actor-owned plan"
            )
        pending = [step for step in plan.steps if step.status == "pending"]
        if not pending:
            raise GovernanceFrameworkAuthorityError(
                "remediation evidence has no current pending plan step"
            )
        target = pending[0]
        if (
            record.subject_id != target.step_id
            or facts.get("plan_version") != plan.version
            or facts.get("plan_digest") != plan.plan_digest
            or facts.get("sequence_no") != target.sequence_no
            or facts.get("evidence_kind") != target.action_kind
            or facts.get("result") != "passed"
            or record.issued_at_tick < plan.world_tick
        ):
            raise GovernanceFrameworkAuthorityError(
                "remediation evidence is stale or crosses cached plan/step authority"
            )
        _framework_text(
            facts.get("request_msg_id"),
            "remediation audit request_msg_id",
        )
        _framework_digest(
            facts.get("request_fingerprint"),
            "remediation audit request_fingerprint",
        )
        return (
            (
                _choice(
                    _COMPLETE_STEP,
                    {"plan_id": plan.plan_id, "step_id": target.step_id},
                ),
            ),
            record.issued_at_tick,
        )


def _campaign_projection_from_update(
    payload: Mapping[str, Any],
) -> _CampaignProjection:
    _require_sender_payload_fields(
        payload,
        {
            "operation",
            "result_kind",
            "result_id",
            "version",
            "status",
            "references",
            "placements",
        },
        _GOVERNANCE_UPDATED,
    )
    operation = payload.get("operation")
    if operation not in {_PUBLISH, _DISCLOSE, _ACTIVATE}:
        raise GovernanceFrameworkAuthorityError(
            "campaign projection has an unsupported operation"
        )
    if payload.get("result_kind") != "campaign":
        raise GovernanceFrameworkAuthorityError(
            "campaign projection has the wrong result kind"
        )
    reference = _single_reference(payload.get("references"))
    campaign_id = _framework_text(payload.get("result_id"), "campaign result_id")
    if reference["record_kind"] != "campaign" or reference["stable_id"] != campaign_id:
        raise GovernanceFrameworkAuthorityError(
            "campaign projection reference and result identity differ"
        )
    version = _framework_positive_int(payload.get("version"), "campaign version")
    status = _framework_text(payload.get("status"), "campaign status")
    if status not in {"draft", "active", "paused", "closed"}:
        raise GovernanceFrameworkAuthorityError("campaign projection status is invalid")
    placements = _projection_steps_or_placements(
        payload.get("placements"),
        kind="campaign",
    )
    assert all(isinstance(row, _ProjectionPlacement) for row in placements)
    placement_rows = tuple(placements)
    if operation == _PUBLISH and (
        version != 1
        or status != "draft"
        or any(row.disclosure_status != "pending" for row in placement_rows)
    ):
        raise GovernanceFrameworkAuthorityError(
            "campaign publication projection is not a new pending draft"
        )
    if operation == _DISCLOSE and (
        status not in {"draft", "paused"}
        or not any(row.disclosure_status == "disclosed" for row in placement_rows)
    ):
        raise GovernanceFrameworkAuthorityError(
            "campaign disclosure projection has no disclosed placement"
        )
    if operation == _ACTIVATE and (
        status != "active"
        or any(row.disclosure_status == "pending" for row in placement_rows)
    ):
        raise GovernanceFrameworkAuthorityError(
            "campaign activation projection is not fully disclosed and active"
        )
    return _CampaignProjection(
        campaign_id=campaign_id,
        version=version,
        campaign_digest=reference["record_digest"],
        status=status,
        placements=placement_rows,
    )


def _campaign_projection_choices(
    projection: _CampaignProjection,
    *,
    operation: str,
) -> tuple[_Choice, ...]:
    pending = [
        row for row in projection.placements if row.disclosure_status == "pending"
    ]
    if pending:
        return tuple(
            _choice(
                _DISCLOSE,
                {
                    "campaign_id": projection.campaign_id,
                    "placement_id": row.placement_id,
                },
                selector=f"option_{index}",
                description=(
                    f"pending sponsored placement {index} in the authenticated "
                    "observation order"
                ),
            )
            for index, row in enumerate(pending, start=1)
        )
    if projection.status in {"draft", "paused"}:
        return (_choice(_ACTIVATE, {"campaign_id": projection.campaign_id}),)
    if projection.status == "active" and operation == _ACTIVATE:
        raise _NoGovernanceBusinessTurn(
            "campaign activation acknowledgement is terminal, not a new model turn"
        )
    raise GovernanceFrameworkAuthorityError(
        "campaign projection has no legal non-duplicate transition"
    )


def _case_projection_from_notice(payload: Mapping[str, Any]) -> _CaseProjection:
    _require_sender_payload_fields(
        payload,
        {
            "opened_case_reference",
            "current_case_reference",
            "case_kind",
            "status",
            "response_actions",
            "world_tick",
        },
        _CASE_NOTICE,
    )
    opened = _single_reference(payload.get("opened_case_reference"))
    current = _single_reference(payload.get("current_case_reference"))
    if (
        opened["record_kind"] != "governance_case"
        or current["record_kind"] != "governance_case"
        or opened["stable_id"] != current["stable_id"]
    ):
        raise GovernanceFrameworkAuthorityError(
            "governance case notice crosses record kind or stable identity"
        )
    case_kind = _framework_text(payload.get("case_kind"), "governance case kind")
    expected_action = {
        "review_integrity": _REJECT_REVIEW_NOTICE_ACTION,
        "competition": _REJECT_COORDINATION_NOTICE_ACTION,
    }.get(case_kind)
    if expected_action is None:
        raise GovernanceFrameworkAuthorityError(
            "governance case notice has an unsupported case kind"
        )
    status = _framework_text(payload.get("status"), "governance case status")
    response_actions = payload.get("response_actions")
    expected_actions = [expected_action] if status == "open" else []
    if response_actions != expected_actions:
        raise GovernanceFrameworkAuthorityError(
            "governance case notice advertises incorrect response actions"
        )
    return _CaseProjection(
        case_id=current["stable_id"],
        case_digest=current["record_digest"],
        opened_case_id=opened["stable_id"],
        opened_case_digest=opened["record_digest"],
        case_kind=case_kind,
        status=status,
        response_action=expected_action if status == "open" else None,
        world_tick=_framework_nonnegative_int(
            payload.get("world_tick"),
            "governance case world_tick",
        ),
    )


def _plan_projection_from_notice(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current: _PlanProjection | None = None,
) -> _PlanProjection:
    _require_sender_payload_fields(
        payload,
        {
            "plan_reference",
            "governance_case_id",
            "status",
            "steps",
            "response_actions",
            "world_tick",
        },
        _PLAN_NOTICE,
    )
    reference = _single_reference(payload.get("plan_reference"))
    if reference["record_kind"] != "remediation_plan":
        raise GovernanceFrameworkAuthorityError(
            "remediation notice has the wrong record kind"
        )
    plan_id = reference["stable_id"]
    if current is not None and current.plan_id != plan_id:
        raise GovernanceFrameworkAuthorityError(
            "remediation notice current authority has a different plan identity"
        )
    status = _framework_text(payload.get("status"), "remediation plan status")
    if status not in {"draft", "active", "completed", "terminated"}:
        raise GovernanceFrameworkAuthorityError("remediation plan status is invalid")
    raw_steps = _projection_steps_or_placements(payload.get("steps"), kind="plan")
    assert all(isinstance(row, _ProjectionStep) for row in raw_steps)
    steps = tuple(raw_steps)
    expected_actions = (
        [_ACCEPT_PLAN_NOTICE_ACTION]
        if status == "draft"
        else [_COMPLETE_STEP_NOTICE_ACTION]
        if status == "active" and any(row.status == "pending" for row in steps)
        else []
    )
    if payload.get("response_actions") != expected_actions:
        raise GovernanceFrameworkAuthorityError(
            "remediation plan notice advertises incorrect response actions"
        )
    if current is None:
        if status != "draft":
            raise GovernanceFrameworkAuthorityError(
                "first remediation plan projection must be the draft owner notice"
            )
        version = 1
    elif current.plan_digest == reference["record_digest"]:
        version = current.version
    else:
        version = current.version + 1
    return _PlanProjection(
        plan_id=plan_id,
        plan_digest=reference["record_digest"],
        version=version,
        governance_case_id=_framework_text(
            payload.get("governance_case_id"),
            "remediation governance_case_id",
        ),
        owner_merchant_id=actor_id,
        status=status,
        steps=steps,
        world_tick=_framework_nonnegative_int(
            payload.get("world_tick"),
            "remediation plan world_tick",
        ),
    )


def _plan_projection_from_update(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current: _PlanProjection | None,
) -> _PlanProjection:
    _require_sender_payload_fields(
        payload,
        {
            "operation",
            "result_kind",
            "result_id",
            "version",
            "status",
            "references",
            "governance_case_id",
            "owner_merchant_id",
            "steps",
        },
        _GOVERNANCE_UPDATED,
    )
    if payload.get("result_kind") != "remediation_plan":
        raise GovernanceFrameworkAuthorityError(
            "remediation update has the wrong result kind"
        )
    reference = _single_reference(payload.get("references"))
    plan_id = _framework_text(payload.get("result_id"), "remediation result_id")
    if reference["record_kind"] != "remediation_plan" or reference["stable_id"] != plan_id:
        raise GovernanceFrameworkAuthorityError(
            "remediation update reference and result identity differ"
        )
    if payload.get("owner_merchant_id") != actor_id:
        raise GovernanceFrameworkAuthorityError(
            "remediation update crosses Agent owner authority"
        )
    raw_steps = _projection_steps_or_placements(payload.get("steps"), kind="plan")
    assert all(isinstance(row, _ProjectionStep) for row in raw_steps)
    return _PlanProjection(
        plan_id=plan_id,
        plan_digest=reference["record_digest"],
        version=_framework_positive_int(payload.get("version"), "remediation version"),
        governance_case_id=_framework_text(
            payload.get("governance_case_id"),
            "remediation governance_case_id",
        ),
        owner_merchant_id=actor_id,
        status=_framework_plan_status(payload.get("status")),
        steps=tuple(raw_steps),
        # ``platform.governance_updated`` intentionally omits World time.  The
        # preceding actor-visible projection is the monotonic time lower bound;
        # the following exact plan notice refreshes it for evidence checks.
        world_tick=current.world_tick if current is not None else 0,
    )


def _projection_steps_or_placements(
    value: Any,
    *,
    kind: Literal["campaign", "plan"],
) -> tuple[_ProjectionPlacement | _ProjectionStep, ...]:
    if not isinstance(value, list) or not value:
        raise GovernanceFrameworkAuthorityError(
            f"{kind} projection requires a non-empty row array"
        )
    rows: list[_ProjectionPlacement | _ProjectionStep] = []
    if kind == "campaign":
        for item in value:
            if not isinstance(item, Mapping) or set(item) != {
                "placement_id",
                "sku_id",
                "disclosure_status",
                "disclosure_text",
            }:
                raise GovernanceFrameworkAuthorityError(
                    "campaign placement projection fields are not exact"
                )
            disclosure_status = _framework_text(
                item.get("disclosure_status"),
                "placement disclosure status",
            )
            if disclosure_status not in {"pending", "disclosed", "not_required"}:
                raise GovernanceFrameworkAuthorityError(
                    "campaign placement disclosure status is invalid"
                )
            disclosure_text = item.get("disclosure_text")
            if disclosure_text is not None and not isinstance(disclosure_text, str):
                raise GovernanceFrameworkAuthorityError(
                    "campaign placement disclosure text must be text or null"
                )
            if disclosure_status == "disclosed" and not disclosure_text:
                raise GovernanceFrameworkAuthorityError(
                    "disclosed campaign placement has no disclosure text"
                )
            if disclosure_status == "pending" and disclosure_text is not None:
                raise GovernanceFrameworkAuthorityError(
                    "pending campaign placement unexpectedly has disclosure text"
                )
            rows.append(
                _ProjectionPlacement(
                    placement_id=_framework_text(
                        item.get("placement_id"),
                        "campaign placement_id",
                    ),
                    sku_id=_framework_text(item.get("sku_id"), "campaign placement sku_id"),
                    disclosure_status=disclosure_status,
                    disclosure_text=disclosure_text,
                )
            )
        placements = tuple(row for row in rows if isinstance(row, _ProjectionPlacement))
        if (
            len({row.placement_id for row in placements}) != len(placements)
            or len({row.sku_id for row in placements}) != len(placements)
        ):
            raise GovernanceFrameworkAuthorityError(
                "campaign projection contains duplicate placement or SKU identities"
            )
        return tuple(rows)

    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping) or set(item) != {
            "step_id",
            "sequence_no",
            "action_kind",
            "status",
        }:
            raise GovernanceFrameworkAuthorityError(
                "remediation step projection fields are not exact"
            )
        sequence_no = _framework_positive_int(
            item.get("sequence_no"),
            "remediation step sequence_no",
        )
        if sequence_no != index:
            raise GovernanceFrameworkAuthorityError(
                "remediation step projection is not in canonical sequence order"
            )
        status = _framework_text(item.get("status"), "remediation step status")
        if status not in {"pending", "completed", "verified", "rejected"}:
            raise GovernanceFrameworkAuthorityError(
                "remediation step projection status is invalid"
            )
        rows.append(
            _ProjectionStep(
                step_id=_framework_text(item.get("step_id"), "remediation step_id"),
                sequence_no=sequence_no,
                action_kind=_framework_text(
                    item.get("action_kind"),
                    "remediation step action_kind",
                ),
                status=status,
            )
        )
    steps = tuple(row for row in rows if isinstance(row, _ProjectionStep))
    if len({row.step_id for row in steps}) != len(steps):
        raise GovernanceFrameworkAuthorityError(
            "remediation projection contains duplicate step identities"
        )
    _validate_plan_step_frontier(steps)
    return tuple(rows)


def _validate_campaign_projection_successor(
    previous: _CampaignProjection,
    current: _CampaignProjection,
) -> None:
    if tuple((row.placement_id, row.sku_id) for row in previous.placements) != tuple(
        (row.placement_id, row.sku_id) for row in current.placements
    ):
        raise GovernanceFrameworkAuthorityError(
            "campaign projection successor changed placement identity or order"
        )
    allowed_status = {
        "draft": {"draft", "active", "closed"},
        "active": {"active", "paused", "closed"},
        "paused": {"paused", "active", "closed"},
        "closed": {"closed"},
    }
    if current.status not in allowed_status[previous.status]:
        raise GovernanceFrameworkAuthorityError(
            "campaign projection has an illegal status transition"
        )
    changed = 0
    for old, new in zip(previous.placements, current.placements, strict=True):
        if old == new:
            continue
        if not (
            old.disclosure_status == "pending"
            and old.disclosure_text is None
            and new.disclosure_status == "disclosed"
            and bool(new.disclosure_text)
        ):
            raise GovernanceFrameworkAuthorityError(
                "campaign projection has an illegal placement transition"
            )
        changed += 1
    if changed > 1:
        raise GovernanceFrameworkAuthorityError(
            "campaign projection advanced multiple placements in one version"
        )


def _validate_plan_projection_successor(
    previous: _PlanProjection,
    current: _PlanProjection,
) -> None:
    if tuple(
        (row.step_id, row.sequence_no, row.action_kind) for row in previous.steps
    ) != tuple(
        (row.step_id, row.sequence_no, row.action_kind) for row in current.steps
    ):
        raise GovernanceFrameworkAuthorityError(
            "remediation projection successor changed step identity or order"
        )
    allowed_plan = {
        "draft": {"active", "terminated"},
        "active": {"active", "completed", "terminated"},
        "completed": {"completed"},
        "terminated": {"terminated"},
    }
    if current.status not in allowed_plan[previous.status]:
        raise GovernanceFrameworkAuthorityError(
            "remediation projection has an illegal plan status transition"
        )
    allowed_step = {
        "pending": {"pending", "completed", "rejected"},
        "completed": {"completed", "verified", "rejected"},
        "verified": {"verified"},
        "rejected": {"rejected"},
    }
    changes = 0
    for old, new in zip(previous.steps, current.steps, strict=True):
        if new.status not in allowed_step[old.status]:
            raise GovernanceFrameworkAuthorityError(
                "remediation projection has an illegal step transition"
            )
        changes += old.status != new.status
    if changes > 1:
        raise GovernanceFrameworkAuthorityError(
            "remediation projection advanced multiple steps in one version"
        )


def _validate_plan_step_frontier(steps: Sequence[_ProjectionStep]) -> None:
    frontier_seen = False
    for row in steps:
        if row.status == "verified":
            if frontier_seen:
                raise GovernanceFrameworkAuthorityError(
                    "remediation projection verifies a step beyond the current frontier"
                )
            continue
        frontier_seen = True
        if row.status not in {"pending", "completed", "rejected"}:
            raise GovernanceFrameworkAuthorityError(
                "remediation projection has an invalid step frontier"
            )


def _framework_positive_int(value: Any, label: str) -> int:
    result = _framework_nonnegative_int(value, label)
    if result < 1:
        raise GovernanceFrameworkAuthorityError(f"{label} must be positive")
    return result


def _framework_plan_status(value: Any) -> str:
    status = _framework_text(value, "remediation plan status")
    if status not in {"draft", "active", "completed", "terminated"}:
        raise GovernanceFrameworkAuthorityError("remediation plan status is invalid")
    return status


def build_verified_purchase_record(
    *,
    order_id: str,
    txn_id: str,
    buyer_id: str,
    merchant_id: str,
    sku_id: str,
    settled_at_tick: int,
) -> VerifiedPurchaseRecord:
    """Seal an exact projection returned by a World verified-purchase read."""

    row = VerifiedPurchaseRecord(
        order_id=_framework_text(order_id, "order_id"),
        txn_id=_framework_text(txn_id, "txn_id"),
        buyer_id=_framework_text(buyer_id, "buyer_id"),
        merchant_id=_framework_text(merchant_id, "merchant_id"),
        sku_id=_framework_text(sku_id, "sku_id"),
        settled_at_tick=_framework_nonnegative_int(settled_at_tick, "settled_at_tick"),
        record_digest="",
    )
    return VerifiedPurchaseRecord(
        **{
            **_purchase_payload(row, include_digest=False),
            "record_digest": _canonical_digest(_purchase_payload(row, include_digest=False)),
        }
    )


def _campaign_publish_choices(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current_tick: int,
    terms: tuple[AdsCampaignTerms, ...],
    listing_owners: Mapping[str, str],
    campaigns: tuple[Campaign, ...],
) -> tuple[_Choice, ...]:
    _require_sender_payload_fields(
        payload,
        {
            "campaign_references",
            "owner_merchant_id",
            "status",
            "world_tick",
        },
        _CAMPAIGN_TERMS_NOTICE,
    )
    if payload["owner_merchant_id"] != actor_id:
        raise GovernanceFrameworkAuthorityError(
            "campaign terms notice owner differs from the Agent actor"
        )
    if payload["status"] != "publishable":
        raise GovernanceFrameworkAuthorityError("campaign terms notice is not publishable")
    _same_framework("campaign notice world_tick", payload["world_tick"], current_tick)
    references = _reference_list(payload["campaign_references"])
    by_id = {row.campaign_id: row for row in terms}
    existing = {row.campaign_id for row in campaigns}
    if set(by_id) != {row["stable_id"] for row in references}:
        raise GovernanceFrameworkAuthorityError(
            "campaign terms notice and current World terms differ"
        )
    choices: list[_Choice] = []
    for index, reference in enumerate(references, start=1):
        if reference["record_kind"] != "ads_campaign_terms":
            raise GovernanceFrameworkAuthorityError(
                "campaign notice reference has the wrong record kind"
            )
        term = by_id[reference["stable_id"]]
        _same_framework("campaign terms digest", reference["record_digest"], term.terms_digest)
        if term.campaign_id in existing:
            raise GovernanceFrameworkAuthorityError(
                "campaign publication notice duplicates an existing campaign"
            )
        if current_tick > term.ends_at_tick:
            raise GovernanceFrameworkAuthorityError("campaign publication authority is stale")
        for placement in term.placements:
            if listing_owners.get(placement.sku_id) != actor_id:
                raise GovernanceFrameworkAuthorityError(
                    "campaign publication crosses listing owner authority"
                )
        choices.append(
            _choice(
                _PUBLISH,
                {"campaign_id": term.campaign_id},
                selector=f"option_{index}",
                description=f"campaign with {len(term.placements)} sponsored placements",
            )
        )
    return tuple(choices)


def _campaign_update_choices(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current_tick: int,
    campaigns: tuple[Campaign, ...],
    histories: tuple[GovernanceHistory, ...],
) -> tuple[_Choice, ...]:
    operation = payload.get("operation")
    if operation not in {
        "publish_campaign",
        "disclose_placement",
        "activate_campaign",
    }:
        raise GovernanceFrameworkAuthorityError("governance update is not a campaign operation")
    if len(campaigns) != 1:
        raise GovernanceFrameworkAuthorityError(
            "campaign update requires one exact current World campaign"
        )
    campaign = campaigns[0]
    if campaign.owner_merchant_id != actor_id:
        raise GovernanceFrameworkAuthorityError(
            "campaign update owner differs from the Agent actor"
        )
    expected = _campaign_safe_payload(campaign, operation=operation)
    if dict(payload) != expected:
        raise GovernanceFrameworkAuthorityError(
            "campaign update is not the exact current World projection"
        )
    _require_current_history(
        histories,
        record_kind="campaign",
        stable_id=campaign.campaign_id,
        current=campaign,
    )
    if current_tick > campaign.ends_at_tick:
        raise GovernanceFrameworkAuthorityError("campaign action authority is stale")
    pending = [row for row in campaign.placements if row.disclosure_status == "pending"]
    if pending:
        return tuple(
            _choice(
                _DISCLOSE,
                {
                    "campaign_id": campaign.campaign_id,
                    "placement_id": row.placement_id,
                },
                selector=f"option_{index}",
                description=f"sponsored placement for SKU {row.sku_id}",
            )
            for index, row in enumerate(pending, start=1)
        )
    if campaign.status in {"draft", "paused"}:
        if not campaign.starts_at_tick <= current_tick <= campaign.ends_at_tick:
            raise GovernanceFrameworkAuthorityError(
                "campaign activation is outside its World time window"
            )
        return (_choice(_ACTIVATE, {"campaign_id": campaign.campaign_id}),)
    if campaign.status == "active" and operation == "activate_campaign":
        raise GovernanceFrameworkAuthorityError(
            "campaign activation acknowledgement is terminal, not a new Agent turn"
        )
    raise GovernanceFrameworkAuthorityError("campaign update has no legal non-duplicate transition")


def _case_notice_choices(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current_tick: int,
    cases: tuple[GovernanceCase, ...],
    histories: tuple[GovernanceHistory, ...],
) -> tuple[_Choice, ...]:
    _require_sender_payload_fields(
        payload,
        {
            "opened_case_reference",
            "current_case_reference",
            "case_kind",
            "status",
            "response_actions",
            "world_tick",
        },
        _CASE_NOTICE,
    )
    if len(cases) != 1:
        raise GovernanceFrameworkAuthorityError(
            "governance case notice requires one exact current World case"
        )
    case = cases[0]
    history = _require_current_history(
        histories,
        record_kind="governance_case",
        stable_id=case.case_id,
        current=case,
    )
    if actor_id not in case.subject_merchant_ids:
        raise GovernanceFrameworkAuthorityError(
            "governance case notice actor is not a case subject"
        )
    _same_framework("case notice world_tick", payload["world_tick"], current_tick)
    if current_tick < case.updated_at_tick:
        raise GovernanceFrameworkAuthorityError(
            "governance case notice predates current World state"
        )
    if payload["case_kind"] != case.case_kind or payload["status"] != case.status:
        raise GovernanceFrameworkAuthorityError("governance case notice status or kind is stale")
    opened = _single_reference(payload["opened_case_reference"])
    current = _single_reference(payload["current_case_reference"])
    _assert_reference(opened, history.records[0])
    _assert_reference(current, case)
    expected_choice = {
        "review_integrity": (_REJECT_REVIEW, _REJECT_REVIEW_NOTICE_ACTION),
        "competition": (
            _REJECT_COORDINATION,
            _REJECT_COORDINATION_NOTICE_ACTION,
        ),
    }.get(case.case_kind)
    expected_actions = (
        [expected_choice[1]] if case.status == "open" and expected_choice is not None else []
    )
    if payload["response_actions"] != expected_actions:
        raise GovernanceFrameworkAuthorityError(
            "governance case notice advertises incorrect response actions"
        )
    if expected_choice is None or case.status != "open":
        raise GovernanceFrameworkAuthorityError(
            "governance case is terminal or has no actor rejection transition"
        )
    return (
        _choice(expected_choice[0], {"case_id": case.case_id}),
        _choice(
            _READ_HISTORY,
            {"record_kind": "governance_case", "stable_id": case.case_id},
        ),
    )


def _plan_notice_choices(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current_tick: int,
    plans: tuple[RemediationPlan, ...],
    histories: tuple[GovernanceHistory, ...],
) -> tuple[_Choice, ...]:
    _require_sender_payload_fields(
        payload,
        {
            "plan_reference",
            "governance_case_id",
            "status",
            "steps",
            "response_actions",
            "world_tick",
        },
        _PLAN_NOTICE,
    )
    if len(plans) != 1:
        raise GovernanceFrameworkAuthorityError(
            "remediation notice requires one exact current World plan"
        )
    plan = plans[0]
    _require_current_history(
        histories,
        record_kind="remediation_plan",
        stable_id=plan.plan_id,
        current=plan,
    )
    if plan.owner_merchant_id != actor_id:
        raise GovernanceFrameworkAuthorityError(
            "remediation notice owner differs from the Agent actor"
        )
    _same_framework("remediation notice world_tick", payload["world_tick"], current_tick)
    if current_tick < plan.updated_at_tick:
        raise GovernanceFrameworkAuthorityError("remediation notice predates current World state")
    _assert_reference(_single_reference(payload["plan_reference"]), plan)
    expected_steps = [
        {
            "step_id": row.step_id,
            "sequence_no": row.sequence_no,
            "action_kind": row.action_kind,
            "status": row.status,
        }
        for row in plan.steps
    ]
    expected_actions = (
        [_ACCEPT_PLAN_NOTICE_ACTION]
        if plan.status == "draft"
        else [_COMPLETE_STEP_NOTICE_ACTION]
        if plan.status == "active" and any(row.status == "pending" for row in plan.steps)
        else []
    )
    if (
        payload["governance_case_id"] != plan.governance_case_id
        or payload["status"] != plan.status
        or payload["steps"] != expected_steps
        or payload["response_actions"] != expected_actions
    ):
        raise GovernanceFrameworkAuthorityError(
            "remediation notice is not the exact current World projection"
        )
    if plan.status == "draft":
        return (
            _choice(_ACCEPT_PLAN, {"plan_id": plan.plan_id}),
            _choice(
                _READ_HISTORY,
                {"record_kind": "remediation_plan", "stable_id": plan.plan_id},
            ),
        )
    if plan.status == "active":
        # A plan notice alone does not authorize completion.  The independent
        # evidence delivery is the exact current inbound for that mutation.
        return (
            _choice(
                _READ_HISTORY,
                {"record_kind": "remediation_plan", "stable_id": plan.plan_id},
            ),
        )
    raise GovernanceFrameworkAuthorityError(
        "remediation notice is terminal and cannot start a model turn"
    )


def _remediation_evidence_choices(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current_tick: int,
    plans: tuple[RemediationPlan, ...],
    histories: tuple[GovernanceHistory, ...],
    evidence_records: tuple[EvidenceRecord, ...],
) -> tuple[_Choice, ...]:
    _require_sender_payload_fields(payload, {"record"}, _EVIDENCE_PERSISTED)
    if len(evidence_records) != 1:
        raise GovernanceFrameworkAuthorityError(
            "remediation evidence turn requires one exact current World record"
        )
    delivered = _coerce_evidence(payload["record"])
    record = evidence_records[0]
    if delivered != record:
        raise GovernanceFrameworkAuthorityError(
            "Platform evidence delivery is not the exact current World record"
        )
    if record.kind != "independent_governance_audit":
        raise GovernanceFrameworkAuthorityError(
            "remediation completion requires an independent governance audit"
        )
    if record.owner_id != actor_id or actor_id not in record.read_acl:
        raise GovernanceFrameworkAuthorityError(
            "remediation evidence crosses actor owner or read authority"
        )
    if record.issued_at_tick > current_tick:
        raise GovernanceFrameworkAuthorityError("remediation evidence is from the future")
    if len(plans) != 1:
        raise GovernanceFrameworkAuthorityError(
            "remediation evidence requires one exact current World plan"
        )
    plan = plans[0]
    _require_current_history(
        histories,
        record_kind="remediation_plan",
        stable_id=plan.plan_id,
        current=plan,
    )
    if plan.owner_merchant_id != actor_id or plan.status != "active":
        raise GovernanceFrameworkAuthorityError(
            "remediation evidence does not bind an active actor-owned plan"
        )
    facts = record.facts
    plan_id = facts.get("plan_id")
    if plan_id != plan.plan_id:
        raise GovernanceFrameworkAuthorityError(
            "remediation evidence crosses plan or case authority"
        )
    target = next((row for row in plan.steps if row.step_id == record.subject_id), None)
    if target is None:
        raise GovernanceFrameworkAuthorityError(
            "remediation evidence subject is not a step in the current plan"
        )
    if (
        target.status != "pending"
        or facts.get("plan_version") != plan.version
        or facts.get("plan_digest") != plan.plan_digest
        or facts.get("sequence_no") != target.sequence_no
        or facts.get("evidence_kind") != target.action_kind
        or facts.get("result") != "passed"
    ):
        raise GovernanceFrameworkAuthorityError(
            "remediation evidence is stale or differs from the current step"
        )
    _framework_text(facts.get("request_msg_id"), "remediation audit request_msg_id")
    _framework_digest(
        facts.get("request_fingerprint"),
        "remediation audit request_fingerprint",
    )
    completed = {row.step_id for row in plan.steps if row.status == "verified"}
    if not set(target.prerequisite_step_ids).issubset(completed):
        raise GovernanceFrameworkAuthorityError("remediation step prerequisites are not verified")
    return (
        _choice(
            _COMPLETE_STEP,
            {"plan_id": plan.plan_id, "step_id": target.step_id},
        ),
    )


def _review_choices(
    payload: Mapping[str, Any],
    *,
    actor_id: str,
    current_tick: int,
    purchases: tuple[VerifiedPurchaseRecord, ...],
    existing_reviews: tuple[ReviewEvidence, ...],
) -> tuple[_Choice, ...]:
    _require_sender_payload_fields(
        payload,
        {"purchase_records", "world_tick"},
        _REVIEW_REQUEST,
    )
    _same_framework("review request world_tick", payload["world_tick"], current_tick)
    references = payload["purchase_records"]
    if not isinstance(references, list):
        raise GovernanceFrameworkAuthorityError("review request purchase_records must be an array")
    projected = [_coerce_purchase_payload(row) for row in references]
    expected = [_purchase_payload(row, include_digest=True) for row in purchases]
    if projected != expected:
        raise GovernanceFrameworkAuthorityError(
            "review request differs from current World verified purchases"
        )
    reviewed = {
        row.sku_id
        for row in existing_reviews
        if row.reviewer_id == actor_id and row.verified_purchase
    }
    choices: list[_Choice] = []
    seen_skus: set[str] = set()
    for row in purchases:
        if row.settled_at_tick > current_tick:
            raise GovernanceFrameworkAuthorityError("verified purchase is from the future")
        if row.sku_id in reviewed:
            raise GovernanceFrameworkAuthorityError(
                "review request duplicates an existing verified review"
            )
        if row.sku_id in seen_skus:
            continue
        seen_skus.add(row.sku_id)
        choices.append(
            _choice(
                _SUBMIT_REVIEW,
                {"sku_id": row.sku_id},
                selector=f"option_{len(choices) + 1}",
                description=f"verified purchase from merchant {row.merchant_id}",
            )
        )
    return tuple(choices)


def _authenticated_platform_payload(
    inbound: Envelope,
    *,
    actor_id: str,
) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(inbound, Envelope):
        raise GovernanceFrameworkAuthorityError(
            "governance authority requires an authenticated Envelope"
        )
    if inbound.to != actor_id:
        raise GovernanceFrameworkAuthorityError(
            "governance delivery recipient does not match the Agent actor"
        )
    if set(inbound.action) != {"kind", "payload"}:
        raise GovernanceFrameworkAuthorityError(
            "governance delivery action must contain exactly kind and payload"
        )
    kind = inbound.action.get("kind")
    sender = {
        _CAMPAIGN_TERMS_NOTICE: _ADS,
        _GOVERNANCE_UPDATED: inbound.from_,
        _CASE_NOTICE: _GOVERNANCE,
        _PLAN_NOTICE: _REMEDIATION,
        _EVIDENCE_PERSISTED: _EVIDENCE,
        _REVIEW_REQUEST: _REVIEWS,
    }.get(kind)
    if sender is None:
        raise GovernanceFrameworkAuthorityError(
            "current inbound is not a governance authority delivery"
        )
    if kind == _GOVERNANCE_UPDATED and inbound.from_ not in {
        _ADS,
        _GOVERNANCE,
        _REMEDIATION,
        _REVIEWS,
    }:
        raise GovernanceFrameworkAuthorityError(
            "governance update has an unregistered Platform sender"
        )
    if inbound.from_ != sender:
        raise GovernanceFrameworkAuthorityError("governance delivery has the wrong Platform sender")
    payload = inbound.action.get("payload")
    if not isinstance(payload, Mapping):
        raise GovernanceFrameworkAuthorityError("governance delivery payload must be an object")
    return str(kind), payload


def _validate_terms(values: Sequence[AdsCampaignTerms]) -> tuple[AdsCampaignTerms, ...]:
    rows = tuple(values)
    seen: set[str] = set()
    for row in rows:
        try:
            validate_ads_campaign_terms(row, expected_authority=_ADS)
        except (SchemaError, GovernanceBindingError) as exc:
            raise GovernanceFrameworkAuthorityError(
                f"invalid current World campaign terms: {exc}"
            ) from exc
        if row.campaign_id in seen:
            raise GovernanceFrameworkAuthorityError(
                "current World campaign terms contain duplicate identities"
            )
        seen.add(row.campaign_id)
    return tuple(sorted(rows, key=lambda row: row.campaign_id))


def _validate_campaigns(values: Sequence[Campaign]) -> tuple[Campaign, ...]:
    rows = tuple(values)
    seen: set[str] = set()
    for row in rows:
        try:
            validate_campaign(
                row,
                expected_authority=_ADS,
                expected_owner_id=row.owner_merchant_id,
            )
        except SchemaError as exc:
            raise GovernanceFrameworkAuthorityError(
                f"invalid current World campaign: {exc}"
            ) from exc
        if row.campaign_id in seen:
            raise GovernanceFrameworkAuthorityError(
                "current World campaign set contains duplicate identities"
            )
        seen.add(row.campaign_id)
    return rows


def _validate_cases(values: Sequence[GovernanceCase]) -> tuple[GovernanceCase, ...]:
    rows = tuple(values)
    seen: set[str] = set()
    for row in rows:
        try:
            validate_governance_case(row, expected_authority=_GOVERNANCE)
        except SchemaError as exc:
            raise GovernanceFrameworkAuthorityError(
                f"invalid current World governance case: {exc}"
            ) from exc
        if row.case_id in seen:
            raise GovernanceFrameworkAuthorityError(
                "current World governance cases contain duplicate identities"
            )
        seen.add(row.case_id)
    return rows


def _validate_plans(values: Sequence[RemediationPlan]) -> tuple[RemediationPlan, ...]:
    rows = tuple(values)
    seen: set[str] = set()
    for row in rows:
        try:
            validate_remediation_plan(
                row,
                expected_authority=_REMEDIATION,
                expected_owner_id=row.owner_merchant_id,
            )
        except SchemaError as exc:
            raise GovernanceFrameworkAuthorityError(
                f"invalid current World remediation plan: {exc}"
            ) from exc
        if row.plan_id in seen:
            raise GovernanceFrameworkAuthorityError(
                "current World remediation plans contain duplicate identities"
            )
        seen.add(row.plan_id)
    return rows


def _validate_reviews(values: Sequence[ReviewEvidence]) -> tuple[ReviewEvidence, ...]:
    rows = tuple(values)
    for row in rows:
        try:
            validate_review_evidence(row, expected_authority=_REVIEWS)
        except SchemaError as exc:
            raise GovernanceFrameworkAuthorityError(
                f"invalid current World review evidence: {exc}"
            ) from exc
    return rows


def _validate_histories(values: Sequence[GovernanceHistory]) -> tuple[GovernanceHistory, ...]:
    rows = tuple(values)
    seen: set[tuple[str, str]] = set()
    for history in rows:
        if not isinstance(history, GovernanceHistory) or not history.records:
            raise GovernanceFrameworkAuthorityError(
                "current World governance history must be a non-empty typed stream"
            )
        key = (history.record_kind, history.stable_id)
        if key in seen:
            raise GovernanceFrameworkAuthorityError(
                "current World governance history contains duplicate streams"
            )
        seen.add(key)
        first = history.records[0]
        if _record_kind(first) != history.record_kind or _record_id(first) != history.stable_id:
            raise GovernanceFrameworkAuthorityError(
                "governance history stream identity is inconsistent"
            )
        authority = _record_authority(first)
        owner = _record_owner(first)
        _validate_record(first)
        for previous, current in zip(history.records, history.records[1:]):
            if _record_kind(current) != history.record_kind:
                raise GovernanceFrameworkAuthorityError("governance history crosses record kinds")
            try:
                validate_version_successor(
                    previous,
                    current,
                    expected_authority=authority,
                    expected_owner_id=owner,
                )
            except SchemaError as exc:
                raise GovernanceFrameworkAuthorityError(
                    f"invalid current World governance history: {exc}"
                ) from exc
    return rows


def _validate_evidence(
    values: Sequence[EvidenceRecord | Mapping[str, Any]],
    *,
    actor_id: str,
) -> tuple[EvidenceRecord, ...]:
    rows = tuple(_coerce_evidence(value) for value in values)
    for row in rows:
        if row.owner_id != actor_id and actor_id not in row.read_acl:
            raise GovernanceFrameworkAuthorityError(
                "current World evidence is not visible to the Agent actor"
            )
    return rows


def _validate_purchases(
    values: Sequence[VerifiedPurchaseRecord],
    *,
    actor_id: str,
) -> tuple[VerifiedPurchaseRecord, ...]:
    rows = tuple(values)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, VerifiedPurchaseRecord):
            raise GovernanceFrameworkAuthorityError("verified purchase record has the wrong type")
        expected = _canonical_digest(_purchase_payload(row, include_digest=False))
        if row.record_digest != expected:
            raise GovernanceFrameworkAuthorityError("verified purchase record digest mismatch")
        if row.buyer_id != actor_id:
            raise GovernanceFrameworkAuthorityError("verified purchase crosses buyer authority")
        if not row.merchant_id.startswith("merchant:"):
            raise GovernanceFrameworkAuthorityError(
                "verified purchase merchant identity is invalid"
            )
        key = (row.order_id, row.txn_id)
        if key in seen:
            raise GovernanceFrameworkAuthorityError(
                "verified purchase set contains duplicate records"
            )
        seen.add(key)
    return tuple(sorted(rows, key=lambda row: (row.sku_id, row.order_id, row.txn_id)))


def _validate_listing_owners(value: Mapping[str, str]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for sku_id, owner_id in value.items():
        owners[_framework_text(sku_id, "listing SKU")] = _framework_text(owner_id, "listing owner")
    return owners


def _require_current_history(
    histories: tuple[GovernanceHistory, ...],
    *,
    record_kind: str,
    stable_id: str,
    current: GovernanceRecord,
) -> GovernanceHistory:
    matches = [
        row for row in histories if row.record_kind == record_kind and row.stable_id == stable_id
    ]
    if len(matches) != 1:
        raise GovernanceFrameworkAuthorityError("exact current World governance history is missing")
    history = matches[0]
    if history.records[-1] != current:
        raise GovernanceFrameworkAuthorityError("governance history latest version is stale")
    return history


def _validate_record(value: GovernanceRecord) -> None:
    try:
        if isinstance(value, Campaign):
            validate_campaign(value, expected_authority=_ADS)
        elif isinstance(value, GovernanceCase):
            validate_governance_case(value, expected_authority=_GOVERNANCE)
        elif isinstance(value, RemediationPlan):
            validate_remediation_plan(value, expected_authority=_REMEDIATION)
        else:
            raise GovernanceFrameworkAuthorityError("unsupported governance history record kind")
    except SchemaError as exc:
        raise GovernanceFrameworkAuthorityError(
            f"invalid current World governance record: {exc}"
        ) from exc


def _record_kind(value: GovernanceRecord) -> str:
    if isinstance(value, Campaign):
        return "campaign"
    if isinstance(value, GovernanceCase):
        return "governance_case"
    if isinstance(value, RemediationPlan):
        return "remediation_plan"
    raise GovernanceFrameworkAuthorityError("unsupported governance history record kind")


def _record_id(value: GovernanceRecord) -> str:
    if isinstance(value, Campaign):
        return value.campaign_id
    if isinstance(value, GovernanceCase):
        return value.case_id
    if isinstance(value, RemediationPlan):
        return value.plan_id
    raise GovernanceFrameworkAuthorityError("unsupported governance history record kind")


def _record_digest(value: GovernanceRecord) -> str:
    if isinstance(value, Campaign):
        return value.campaign_digest
    if isinstance(value, GovernanceCase):
        return value.case_digest
    if isinstance(value, RemediationPlan):
        return value.plan_digest
    raise GovernanceFrameworkAuthorityError("unsupported governance history record kind")


def _record_authority(value: GovernanceRecord) -> str:
    return value.authored_by


def _record_owner(value: GovernanceRecord) -> str | None:
    if isinstance(value, Campaign):
        return value.owner_merchant_id
    if isinstance(value, RemediationPlan):
        return value.owner_merchant_id
    return None


def _assert_reference(reference: Mapping[str, str], record: GovernanceRecord) -> None:
    expected = {
        "record_kind": _record_kind(record),
        "stable_id": _record_id(record),
        "record_digest": _record_digest(record),
    }
    if dict(reference) != expected:
        raise GovernanceFrameworkAuthorityError(
            "Platform governance reference differs from World record identity or digest"
        )


def _campaign_safe_payload(campaign: Campaign, *, operation: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "result_kind": "campaign",
        "result_id": campaign.campaign_id,
        "version": campaign.version,
        "status": campaign.status,
        "references": {
            "record_kind": "campaign",
            "stable_id": campaign.campaign_id,
            "record_digest": campaign.campaign_digest,
        },
        "placements": [
            {
                "placement_id": row.placement_id,
                "sku_id": row.sku_id,
                "disclosure_status": row.disclosure_status,
                "disclosure_text": row.disclosure_text,
            }
            for row in campaign.placements
        ],
    }


def _single_reference(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "record_kind",
        "stable_id",
        "record_digest",
    }:
        raise GovernanceFrameworkAuthorityError("Platform governance reference has invalid fields")
    row = {
        str(name): _framework_text(item, f"governance reference {name}")
        for name, item in value.items()
    }
    _framework_digest(row["record_digest"], "governance reference digest")
    return row


def _reference_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise GovernanceFrameworkAuthorityError(
            "Platform campaign references must be a non-empty array"
        )
    rows = [_single_reference(item) for item in value]
    if len({(row["record_kind"], row["stable_id"]) for row in rows}) != len(rows):
        raise GovernanceFrameworkAuthorityError("Platform campaign references contain duplicates")
    return rows


def _coerce_evidence(value: Any) -> EvidenceRecord:
    try:
        row = coerce_evidence_record(value)
        validate_evidence_record(row)
        return row
    except SchemaError as exc:
        raise GovernanceFrameworkAuthorityError(
            f"invalid current World evidence record: {exc}"
        ) from exc


def _purchase_payload(
    row: VerifiedPurchaseRecord,
    *,
    include_digest: bool,
) -> dict[str, Any]:
    payload = {
        "order_id": row.order_id,
        "txn_id": row.txn_id,
        "buyer_id": row.buyer_id,
        "merchant_id": row.merchant_id,
        "sku_id": row.sku_id,
        "settled_at_tick": row.settled_at_tick,
    }
    if include_digest:
        payload["record_digest"] = row.record_digest
    return payload


def _coerce_purchase_payload(value: Any) -> dict[str, Any]:
    fields = {
        "order_id",
        "txn_id",
        "buyer_id",
        "merchant_id",
        "sku_id",
        "settled_at_tick",
        "record_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GovernanceFrameworkAuthorityError(
            "Platform verified-purchase projection has invalid fields"
        )
    row = {
        "order_id": _framework_text(value["order_id"], "order_id"),
        "txn_id": _framework_text(value["txn_id"], "txn_id"),
        "buyer_id": _framework_text(value["buyer_id"], "buyer_id"),
        "merchant_id": _framework_text(value["merchant_id"], "merchant_id"),
        "sku_id": _framework_text(value["sku_id"], "sku_id"),
        "settled_at_tick": _framework_nonnegative_int(value["settled_at_tick"], "settled_at_tick"),
        "record_digest": _framework_digest(value["record_digest"], "verified purchase digest"),
    }
    return row


def _decision_spec(
    operation: str,
    choices: tuple[_Choice, ...],
) -> BusinessIntentSpec:
    properties: dict[str, Any] = {}
    required: list[str] = []
    if len(choices) > 1:
        properties["target"] = {
            "type": "string",
            "enum": [row.selector for row in choices],
            "description": "; ".join(
                f"{row.selector}: {row.selector_description}" for row in choices
            ),
        }
        required.append("target")
    if operation == _DISCLOSE:
        properties["disclosure_text"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_DISCLOSURE_LENGTH,
        }
        required.append("disclosure_text")
    elif operation == _SUBMIT_REVIEW:
        properties["rating"] = {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        }
        properties["review_text"] = {
            "type": "string",
            "maxLength": _MAX_REVIEW_TEXT_LENGTH,
        }
        required.append("rating")
    return BusinessIntentSpec(
        intent=operation,
        description=_operation_description(operation),
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        category="act",
        source_name=operation,
    )


def _operation_description(operation: str) -> str:
    return {
        _PUBLISH: "Publish a current World-authorized advertising campaign.",
        _DISCLOSE: "Disclose a current pending sponsored placement.",
        _ACTIVATE: "Activate the current fully disclosed campaign.",
        _REJECT_COORDINATION: "Reject coordination in the current governance case.",
        _REJECT_REVIEW: "Reject review manipulation in the current governance case.",
        _ACCEPT_PLAN: "Accept the current remediation plan.",
        _COMPLETE_STEP: "Complete the current independently evidenced remediation step.",
        _READ_HISTORY: "Read the current governance object's World history.",
        _SUBMIT_REVIEW: "Submit a review for a current verified-purchase option.",
    }[operation]


def _choice(
    operation: str,
    payload: Mapping[str, Any],
    *,
    selector: str | None = None,
    description: str | None = None,
) -> _Choice:
    return _Choice(
        operation=operation,
        wire_payload=tuple(payload.items()),
        selector=selector,
        selector_description=description,
    )


def _require_sender_payload_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    action: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise GovernanceFrameworkAuthorityError(
            f"{action} payload fields are not exact: missing={missing!r}, unknown={unknown!r}"
        )


def _require_actor(value: Any) -> None:
    if not isinstance(value, str):
        raise GovernanceFrameworkAuthorityError(
            "governance Agent actor must be a full buyer or merchant identity"
        )
    parts = value.split(":")
    if len(parts) < 2 or not all(parts) or parts[0] not in {"buyer", "merchant"}:
        raise GovernanceFrameworkAuthorityError(
            "governance Agent actor must be a full buyer or merchant identity"
        )


def _require_cache_actor(value: Any) -> None:
    if not isinstance(value, str):
        raise GovernanceFrameworkAuthorityError(
            "governance projection cache actor identity is invalid"
        )
    parts = value.split(":")
    if parts[0] not in {"buyer", "merchant"} or not all(parts):
        raise GovernanceFrameworkAuthorityError(
            "governance projection cache actor identity is invalid"
        )
    if len(parts) == 1:
        return
    _require_actor(value)


def _framework_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceFrameworkAuthorityError(f"{label} must be non-empty text")
    return value


def _framework_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GovernanceFrameworkAuthorityError(f"{label} must be a nonnegative integer")
    return value


def _framework_digest(value: Any, label: str) -> str:
    digest = _framework_text(value, label)
    if len(digest) != _DIGEST_LENGTH or any(char not in "0123456789abcdef" for char in digest):
        raise GovernanceFrameworkAuthorityError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _same_framework(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise GovernanceFrameworkAuthorityError(f"{label} authority mismatch")


def _require_model_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise GovernanceModelChoiceError("governance tool argument names must be strings")
    actual = set(value)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise GovernanceModelChoiceError("invalid governance tool arguments: " + "; ".join(details))


def _model_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceModelChoiceError(f"{label} must be non-empty text")
    return value.strip()


def _bounded_model_text(
    value: Any,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise GovernanceModelChoiceError(f"{label} must be text")
    result = value.strip()
    if not result and not allow_empty:
        raise GovernanceModelChoiceError(f"{label} must be non-empty text")
    if len(result) > maximum:
        raise GovernanceModelChoiceError(f"{label} exceeds the {maximum}-character limit")
    return result


def _model_rating(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise GovernanceModelChoiceError("rating must be an integer from 1 to 5")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    import hashlib
    import json

    try:
        body = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GovernanceFrameworkAuthorityError(
            f"verified purchase is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(body).hexdigest()


__all__ = [
    "CompiledGovernanceAction",
    "FrameworkAuthorityError",
    "GovernanceFrameworkAuthorityError",
    "GovernanceHistory",
    "GovernanceModelChoiceError",
    "GovernanceProjectionCache",
    "GovernanceTurnAuthority",
    "GovernanceTurnError",
    "ModelChoiceError",
    "VerifiedPurchaseRecord",
    "build_verified_purchase_record",
]
