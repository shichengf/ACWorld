"""ACWorld market governance tasks for capability family T8.

T8 evaluates multi-merchant markets and Platform governance.  Governance
outcomes are not listing attributes and are not reconstructed by a scorer.
Campaigns, reviews, detector signals, cases, decisions, reputation events,
remediation plans, and ranking contexts must be produced by real Platform
services and committed to authoritative World state.

All twenty tasks traverse Episode, Runtime, typed Platform governance services,
authoritative World state, exact evidence, Tracker verification, and replay.
The family is exposed atomically so no benchmark-local governance substitute
can enter a formal contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial
from typing import Any, Mapping, Sequence

from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1
from cwe_examples import round_robin_ranking as _round_robin_ranking  # noqa: F401
from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2
from episode.capability_runtime import (
    RuntimeEvidenceBundleV2,
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceError,
    RuntimeMutationV2,
    RuntimeRubricCheckV2,
    RuntimeTaskBundleV2,
    RuntimeTaskScoreV3,
    canonical_sha256,
    renormalize_capability_checks_v2,
    require_runtime_benchmark_integrity_v2,
    score_checks,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.types import (
    BuyerSpec,
    ControlServiceSpec,
    MerchantSpec,
    PopulationSpec,
    ScenarioSpec,
)
from protocol.evidence_records import (
    MandateRevisionAuthority,
    build_evidence_record,
    build_mandate_revision,
    evidence_record_to_dict,
    mandate_revision_to_dict,
)
from protocol.market_governance import (
    GovernanceCase,
    MarketSignal,
    RemediationPlan,
    ReputationEvent,
    ReviewAggregate,
    ReviewEvidence,
)
from protocol.remediation_audit import REMEDIATION_AUDITOR_SERVICE_ID
from runtime.authority_operation_evidence import (
    EVIDENCE_RECORD_EVIDENCE_CONTRACT,
    SEARCH_SESSION_EVIDENCE_CONTRACT,
    VerifiedEvidenceRecordEvidence,
    VerifiedSearchSessionEvidence,
)
from runtime.market_governance_evidence import (
    MARKET_GOVERNANCE_EVIDENCE_CONTRACT,
    VerifiedMarketGovernanceEvidence,
)
from runtime.match_certificate_evidence import (
    MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
    VerifiedMatchCertificateEvidence,
)
from runtime.supply_fulfillment_evidence import (
    SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
    VerifiedSupplyFulfillmentEvidence,
    governance_preclaimed_commit_attestations,
    verified_settlement_order,
)
from runtime.tracker_evidence import tracker_row_has_usable_completed_steps
from world.evidence_contracts import mandate_authority_to_wire


T8_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t8.v4"
_BUYER_ID = "buyer:t8-benchmark"
_PRIMARY_MERCHANT_ID = "merchant:t8-m1"
_PRINCIPAL_ID = "consumer:t8-principal"

_T8_TASK_IDS = tuple(
    task_id for task_id, definition in TASK_REGISTRY_V2.items() if definition.family.value == "T8"
)

# A family is registered atomically.  No old search, catalog-update, or
# settlement shortcut is considered a governance implementation.
T8_RUNTIME_READY_TASK_IDS = _T8_TASK_IDS
T8_RUNTIME_PENDING_TASK_IDS: tuple[str, ...] = ()


class T8RuntimeCapabilityGap(RuntimeError):
    """The task cannot yet traverse the complete governance authority path."""


@dataclass(frozen=True, slots=True)
class _ProductT8:
    sku_id: str
    product_id: str
    merchant_id: str
    name: str
    list_price: str
    inventory: int
    product_facts: tuple[tuple[str, int | str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "product_id": self.product_id,
            "merchant_id": self.merchant_id,
            "name": self.name,
            "list_price": self.list_price,
            "inventory": self.inventory,
            "product_facts": dict(self.product_facts),
        }


@dataclass(frozen=True, slots=True)
class _CaseT8:
    definition: TaskDefinitionV2
    lane: str
    axis_name: str
    axis_value: int
    products: tuple[_ProductT8, ...]
    policy_inputs: tuple[tuple[str, int | str], ...]
    external_event_inputs: tuple[tuple[str, int | str], ...]
    actor_operations: tuple[str, ...]
    service_operations: tuple[str, ...]

    @property
    def evaluated_actor_id(self) -> str:
        if self.definition.evaluated_role == "buyer":
            return _BUYER_ID
        return _PRIMARY_MERCHANT_ID

    @property
    def semantic_contract(self) -> dict[str, Any]:
        return {
            "schema_version": T8_RUNTIME_SCHEMA_V2,
            "definition": self.definition.to_dict(),
            "lane": self.lane,
            "difficulty": {self.axis_name: self.axis_value},
            "products": tuple(product.to_dict() for product in self.products),
            "policy_inputs": dict(self.policy_inputs),
            "external_event_inputs": dict(self.external_event_inputs),
            "actor_operations": self.actor_operations,
            "service_operations": self.service_operations,
            "runtime_ready": True,
            "authority_path": (
                "actor envelope",
                "stateless Platform governance service",
                "World governance policies and records",
                "authority operation and atomic commit",
                "deterministic exact evidence",
                "strict replay and Python score",
            ),
            "forbidden_shortcuts": (
                "governance outcomes in listing attributes",
                "scenario-seeded campaign, case, review, or reputation result",
                "catalog mutation used as governance evidence",
                "scorer-reconstructed transition",
            ),
        }

    @property
    def mandate_id(self) -> str:
        return f"mandate:{_slug(self.definition)}"

    @property
    def order_id(self) -> str:
        product = _expected_buyer_product(self)
        return f"ord-{self.mandate_id}-agg:{product.sku_id}"

    @property
    def terminal_outcome(self) -> dict[str, Any]:
        """Return the deterministic terminal commerce or governance outcome."""

        if self.definition.evaluated_role == "buyer":
            product = _expected_buyer_product(self)
            return {
                "kind": "settled_order",
                "order_id": self.order_id,
                "sku_id": product.sku_id,
                "merchant_id": product.merchant_id,
                "qty": 1,
            }
        if self.lane == "merchant_sponsorship_disclosure":
            return {
                "kind": "activated_disclosed_campaign",
                "placement_count": self.axis_value,
            }
        if self.lane in {"merchant_review_integrity", "merchant_anti_collusion"}:
            return {"kind": "policy_safe_governance_rejection"}
        if self.lane == "merchant_reputation_recovery":
            return {
                "kind": "completed_remediation",
                "step_count": self.axis_value,
            }
        raise ValueError(f"{self.definition.task_id}: unsupported terminal outcome")


@dataclass(frozen=True, slots=True)
class _ReviewFixtureT8:
    reviewer_id: str
    record_id: str
    sku_id: str
    merchant_id: str
    rating: int
    verified_purchase: bool

    @property
    def order_id(self) -> str:
        return f"order:{self.record_id.removeprefix('evidence:')}"

    @property
    def transaction_id(self) -> str:
        return f"txn:{self.record_id.removeprefix('evidence:')}"


def _axis(definition: TaskDefinitionV2) -> tuple[str, int]:
    values = [
        (name, value) for name, value in definition.difficulty_factors if name != "difficulty_level"
    ]
    if len(values) != 1:
        raise ValueError(f"{definition.task_id}: T8 needs exactly one semantic axis")
    name, value = values[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{definition.task_id}: T8 axis must be a positive integer")
    return name, value


def _slug(definition: TaskDefinitionV2) -> str:
    return definition.task_id.casefold().replace("-", "")


def _products(definition: TaskDefinitionV2) -> tuple[_ProductT8, ...]:
    rows: list[_ProductT8] = []
    for ordinal in range(1, definition.merchants + 1):
        merchant_id = f"merchant:t8-m{ordinal}"
        rows.append(
            _ProductT8(
                sku_id=f"{merchant_id}:{_slug(definition)}:listing",
                product_id=f"product:{_slug(definition)}",
                merchant_id=merchant_id,
                name=f"Market product from merchant {ordinal}",
                list_price=f"{40 + ordinal * 7}.00",
                inventory=16,
                product_facts=(
                    ("quality_score", 60 + ordinal * 8),
                    ("warranty_months", 12 + ordinal * 6),
                ),
            )
        )
    if definition.capability_id == "t8.merchant_sponsorship_disclosure":
        _, placement_count = _axis(definition)
        for placement_ordinal in range(2, placement_count + 1):
            rows.append(
                _ProductT8(
                    sku_id=(
                        f"{_PRIMARY_MERCHANT_ID}:{_slug(definition)}:listing-{placement_ordinal}"
                    ),
                    product_id=f"product:{_slug(definition)}:{placement_ordinal}",
                    merchant_id=_PRIMARY_MERCHANT_ID,
                    name=f"Sponsored product {placement_ordinal}",
                    list_price=f"{47 + placement_ordinal}.00",
                    inventory=16,
                    product_facts=(
                        ("quality_score", 68 + placement_ordinal),
                        ("warranty_months", 18),
                    ),
                )
            )
    return tuple(rows)


_ACTOR_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "buyer_price_quality": ("search", "select_offer"),
    "buyer_sponsored_ranking": ("search", "select_offer"),
    "buyer_fake_reviews": ("search", "select_offer"),
    "buyer_collusion": ("search", "select_offer"),
    "buyer_reputation": ("search", "select_offer"),
    "merchant_sponsorship_disclosure": (
        "publish_campaign",
        "disclose_placement",
        "activate_campaign",
    ),
    "merchant_review_integrity": ("reject_review_manipulation",),
    "merchant_anti_collusion": ("reject_coordination",),
    "merchant_reputation_recovery": (
        "accept_remediation_plan",
        "complete_remediation_step",
    ),
}


_SERVICE_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "buyer_price_quality": ("persist_ranking_context",),
    "buyer_sponsored_ranking": (
        "publish_governance_policy",
        "persist_ranking_context",
    ),
    "buyer_fake_reviews": (
        "publish_governance_policy",
        "submit_review",
        "aggregate_reviews",
        "persist_ranking_context",
    ),
    "buyer_collusion": (
        "persist_detector_evidence",
        "ingest_market_observation",
        "resolve_governance_case",
        "persist_ranking_context",
    ),
    "buyer_reputation": (
        "publish_governance_policy",
        "apply_governance_reputation",
        "persist_ranking_context",
    ),
    "merchant_sponsorship_disclosure": ("publish_governance_policy",),
    "merchant_review_integrity": (
        "persist_detector_evidence",
        "ingest_market_observation",
    ),
    "merchant_anti_collusion": (
        "persist_detector_evidence",
        "ingest_market_observation",
    ),
    "merchant_reputation_recovery": (
        "publish_governance_policy",
        "create_remediation_plan",
        "verify_remediation_step",
        "apply_governance_reputation",
    ),
}


_T8_GOVERNANCE_WORLD_OPERATIONS = frozenset(
    operation
    for rows in (*_ACTOR_OPERATIONS.values(), *_SERVICE_OPERATIONS.values())
    for operation in rows
) | frozenset(
    {
        "ingest_review_observation",
        "resolve_governance_case",
        "persist_ranking_context",
        "publish_governance_policy",
        "apply_governance_reputation",
    }
)


def _has_platform_action(
    evidence: RuntimeEvidenceBundleV2,
    action_kinds: frozenset[str],
) -> bool:
    return any(row.get("action_kind") in action_kinds for row in evidence.platform_decisions)


def _has_world_operation(
    evidence: RuntimeEvidenceBundleV2,
    operations: frozenset[str],
) -> bool:
    return any(row.get("operation") in operations for row in evidence.world_events)


def _has_governance_effect(evidence: RuntimeEvidenceBundleV2) -> bool:
    return _has_world_operation(evidence, _T8_GOVERNANCE_WORLD_OPERATIONS)


def _inputs(
    lane: str, axis_name: str, axis_value: int
) -> tuple[tuple[tuple[str, int | str], ...], tuple[tuple[str, int | str], ...]]:
    policy: dict[str, int | str] = {}
    events: dict[str, int | str] = {}
    if lane in {"buyer_sponsored_ranking", "merchant_sponsorship_disclosure"}:
        policy = {
            "policy_kind": "ads_campaign_terms",
            "placement_count": axis_value,
        }
    elif lane == "buyer_fake_reviews":
        policy = {
            "policy_kind": "review_account_binding",
            "review_pollution_percent": axis_value,
        }
        total_reviews = 5
        events = {
            "review_fixture_count": total_reviews,
            "unverified_review_count": axis_value * total_reviews // 100,
        }
    elif lane in {"buyer_collusion", "merchant_anti_collusion"}:
        events = {
            "event_kind": "market_detector_observation",
            axis_name: axis_value,
        }
    elif lane == "merchant_review_integrity":
        events = {
            "event_kind": "review_manipulation_observation",
            axis_name: axis_value,
        }
    elif lane == "buyer_reputation":
        policy = {
            "policy_kind": "reputation_policy_revision",
            "history_event_count": axis_value,
        }
    elif lane == "merchant_reputation_recovery":
        policy = {
            "policy_kind": "remediation_blueprint",
            "remediation_step_count": axis_value,
        }
    return tuple(sorted(policy.items())), tuple(sorted(events.items()))


@lru_cache(maxsize=None)
def _case_for_t8(task_id: str) -> _CaseT8:
    try:
        definition = TASK_REGISTRY_V2[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown Benchmark v2 task {task_id!r}") from exc
    if definition.family.value != "T8":
        raise ValueError(f"{task_id} is not a T8 task")
    axis_name, axis_value = _axis(definition)
    lane = definition.capability_id.removeprefix("t8.")
    if lane not in _ACTOR_OPERATIONS or lane not in _SERVICE_OPERATIONS:
        raise ValueError(f"{task_id}: unsupported T8 lane {lane!r}")
    policy, events = _inputs(lane, axis_name, axis_value)
    case = _CaseT8(
        definition=definition,
        lane=lane,
        axis_name=axis_name,
        axis_value=axis_value,
        products=_products(definition),
        policy_inputs=policy,
        external_event_inputs=events,
        actor_operations=_ACTOR_OPERATIONS[lane],
        service_operations=_SERVICE_OPERATIONS[lane],
    )
    _validate_t8_construct(case)
    return case


def _catalog_rows(case: _CaseT8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in case.products:
        rows.append(
            {
                "sku_id": product.sku_id,
                "product_id": product.product_id,
                "merchant_id": product.merchant_id,
                "name": product.name,
                "category": "consumer-goods",
                "list_price": product.list_price,
                "inventory": product.inventory,
                "attributes": dict(product.product_facts),
            }
        )
    return rows


def _mandate_rows(
    case: _CaseT8,
) -> tuple[MandateRevisionAuthority, dict[str, Any]]:
    merchants = [product.merchant_id for product in case.products]
    authority = MandateRevisionAuthority(
        principal_id=_PRINCIPAL_ID,
        buyer_id=_BUYER_ID,
        mandate_id=case.mandate_id,
        allowed_fields=("budget", "preferred_merchants"),
    )
    revision = build_mandate_revision(
        principal_id=_PRINCIPAL_ID,
        buyer_id=_BUYER_ID,
        mandate_id=case.mandate_id,
        revision=1,
        previous_digest=None,
        changes={"budget": 10_000, "preferred_merchants": merchants},
        authorized_fields=authority.allowed_fields,
        logical_tick=0,
    )
    return authority, mandate_revision_to_dict(revision)


def _public_task_context(case: _CaseT8) -> dict[str, Any]:
    """Expose only the public governance phase machine for the tested role."""

    phases: list[dict[str, Any]]
    if case.definition.evaluated_role == "buyer":
        if case.lane == "buyer_fake_reviews":
            discovery_phases = [
                {
                    "phase_id": "buyer_review_setup_barrier",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["commerce.send_message"],
                        "inbound_sender_roles": ["merchant"],
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "framework_continue",
                    "framework_continuation": {
                        "action_kind": "commerce.send_message",
                        "destination": "@self",
                        "payload": {"review_setup_ready": True},
                    },
                },
                {
                    "phase_id": "buyer_review_ready_discovery",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["commerce.send_message"],
                        "inbound_senders": [_BUYER_ID],
                        "payload_equals": {"review_setup_ready": True},
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.search",
                            "destination": "platform:aggregator",
                        }
                    ],
                    "world_reads": "deny",
                    "finish": "forbid",
                },
            ]
        else:
            discovery_phases = [
                {
                    "phase_id": "buyer_market_discovery",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["commerce.send_message"],
                        "inbound_sender_roles": ["merchant"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.search",
                            "destination": "platform:aggregator",
                        }
                    ],
                    "world_reads": "deny",
                    "finish": "forbid",
                }
            ]
        phases = [
            *discovery_phases,
            {
                "phase_id": "buyer_governed_offer_selection",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.rank_offers"],
                    "inbound_senders": ["platform:aggregator"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.search",
                        "destination": "platform:aggregator",
                    },
                    {
                        "action_kind": "commerce.accept_offer",
                        "destination": "platform:aggregator",
                    },
                ],
                "world_reads": "skill_scoped",
                "finish": "forbid",
            },
            {
                "phase_id": "buyer_governed_settlement_complete",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.settlement_receipt"],
                    "inbound_senders": ["platform:psp"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
        ]
    elif case.lane == "merchant_sponsorship_disclosure":
        phases = [
            {
                "phase_id": "merchant_campaign_publish",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["commerce.send_message"],
                    "inbound_sender_roles": ["buyer"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.publish_campaign",
                        "destination": "platform:ads",
                    }
                ],
                "world_reads": "deny",
                "finish": "forbid",
            },
            {
                "phase_id": "merchant_campaign_disclosure",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:ads"],
                    "payload_equals": {"operation": "publish_campaign"},
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.disclose_placement",
                        "destination": "platform:ads",
                    }
                ],
                "world_reads": "deny",
                "finish": "allow_wait",
            },
            {
                "phase_id": "merchant_campaign_disclosure_or_activation",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:ads"],
                    "payload_equals": {"operation": "disclose_placement"},
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.disclose_placement",
                        "destination": "platform:ads",
                    },
                    {
                        "action_kind": "commerce.activate_campaign",
                        "destination": "platform:ads",
                    },
                ],
                "world_reads": "deny",
                "finish": "allow_wait",
            },
            {
                "phase_id": "merchant_campaign_active",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:ads"],
                    "payload_equals": {"operation": "activate_campaign"},
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
        ]
    elif case.lane in {"merchant_review_integrity", "merchant_anti_collusion"}:
        rejection_kind = (
            "commerce.reject_review_manipulation"
            if case.lane == "merchant_review_integrity"
            else "commerce.reject_coordination"
        )
        phases = [
            {
                "phase_id": "merchant_governance_case_response",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_case_notice"],
                    "inbound_senders": ["platform:governance"],
                    "payload_equals": {"status": "open"},
                },
                "allowed_routes": [
                    {
                        "action_kind": rejection_kind,
                        "destination": "platform:governance",
                    }
                ],
                "world_reads": "skill_scoped",
                "finish": "allow_wait",
            },
            {
                "phase_id": "merchant_governance_rejection_recorded",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:governance"],
                    "payload_equals": {
                        "status": "recorded",
                        "operation": rejection_kind.removeprefix("commerce."),
                    },
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
        ]
    elif case.lane == "merchant_reputation_recovery":
        phases = [
            {
                "phase_id": "merchant_remediation_case_wait",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_case_notice"],
                    "inbound_senders": ["platform:governance"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
            {
                "phase_id": "merchant_remediation_plan_acceptance",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.remediation_plan_notice"],
                    "inbound_senders": ["platform:remediation"],
                    "payload_equals": {"status": "draft"},
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.accept_remediation_plan",
                        "destination": "platform:remediation",
                    }
                ],
                "world_reads": "skill_scoped",
                "finish": "forbid",
            },
            {
                "phase_id": "merchant_remediation_audit_wait",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.remediation_plan_notice"],
                    "inbound_senders": ["platform:remediation"],
                    "payload_equals": {"status": "active"},
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
            {
                "phase_id": "merchant_remediation_step_evidence",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.evidence_record_persisted"],
                    "inbound_senders": ["platform:evidence"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.complete_remediation_step",
                        "destination": "platform:remediation",
                    }
                ],
                "world_reads": "skill_scoped",
                "finish": "allow_wait",
            },
            {
                "phase_id": "merchant_remediation_update_wait",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:remediation"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
            {
                "phase_id": "merchant_remediation_complete",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.remediation_plan_notice"],
                    "inbound_senders": ["platform:remediation"],
                    "payload_equals": {"status": "completed"},
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
        ]
    else:  # pragma: no cover - guarded by the family case registry
        raise ValueError(f"{case.definition.task_id}: unsupported T8 public phase lane")
    return {
        "schema_version": T8_RUNTIME_SCHEMA_V2,
        "task_id": case.definition.task_id,
        "capability": case.definition.capability_id,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": phases,
        },
    }


def _campaign_handoff_owners(case: _CaseT8) -> tuple[str, ...]:
    return tuple(dict.fromkeys(product.merchant_id for product in _sponsored_products(case)))


def _campaign_handoff_recipient(case: _CaseT8, merchant_id: str) -> str:
    owners = _campaign_handoff_owners(case)
    try:
        index = owners.index(merchant_id)
    except ValueError as exc:
        raise ValueError("T8 campaign handoff merchant is not sponsored") from exc
    return owners[index + 1] if index + 1 < len(owners) else _BUYER_ID


def _counterpart_task_context(case: _CaseT8, merchant_id: str) -> dict[str, Any] | None:
    """Bind scripted environment handoffs without exposing participant IDs."""

    if case.lane == "buyer_sponsored_ranking" and merchant_id in _campaign_handoff_owners(case):
        phases = [
            {
                "phase_id": "counterpart_campaign_publish",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["commerce.send_message"],
                    "inbound_sender_roles": ["buyer", "merchant"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.publish_campaign",
                        "destination": "platform:ads",
                    }
                ],
                "world_reads": "deny",
                "finish": "forbid",
            },
            {
                "phase_id": "counterpart_campaign_disclosure",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:ads"],
                    "payload_equals": {"operation": "publish_campaign"},
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.disclose_placement",
                        "destination": "platform:ads",
                    }
                ],
                "world_reads": "deny",
                "finish": "forbid",
            },
            {
                "phase_id": "counterpart_campaign_activation",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:ads"],
                    "payload_equals": {"operation": "disclose_placement"},
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.activate_campaign",
                        "destination": "platform:ads",
                    }
                ],
                "world_reads": "deny",
                "finish": "forbid",
            },
            {
                "phase_id": "counterpart_campaign_handoff",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_updated"],
                    "inbound_senders": ["platform:ads"],
                    "payload_equals": {"operation": "activate_campaign"},
                },
                "allowed_routes": [
                    {
                        "action_kind": "commerce.send_message",
                        "destination": "@argument:recipient_id",
                    }
                ],
                "world_reads": "deny",
                "finish": "forbid",
            },
        ]
    elif case.lane == "buyer_collusion" and merchant_id == _PRIMARY_MERCHANT_ID:
        phases = [
            {
                "phase_id": "counterpart_case_open_wait",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.governance_case_notice"],
                    "inbound_senders": ["platform:governance"],
                    "payload_equals": {"status": "open"},
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
            *(
                {
                    "phase_id": f"counterpart_case_{status}_handoff",
                    "match": {
                        "actor_roles": ["merchant"],
                        "inbound_action_kinds": ["platform.governance_case_notice"],
                        "inbound_senders": ["platform:governance"],
                        "payload_equals": {"status": status},
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.send_message",
                            "destination": "@argument:recipient_id",
                        }
                    ],
                    "world_reads": "deny",
                    "finish": "forbid",
                }
                for status in ("resolved", "dismissed")
            ),
        ]
    else:
        return None
    return {
        "schema_version": T8_RUNTIME_SCHEMA_V2,
        "capability": "scripted_governance_handoff",
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": phases,
        },
    }


def _merchant_task_context(
    case: _CaseT8,
    merchant_id: str,
) -> dict[str, Any] | None:
    if case.definition.evaluated_role == "merchant" and merchant_id == case.evaluated_actor_id:
        return _public_task_context(case)
    return _counterpart_task_context(case, merchant_id)


def _merchant_benchmark_contract(
    case: _CaseT8,
    merchant_id: str,
    *,
    task_id: str,
) -> dict[str, Any]:
    """Freeze Agent-only campaign and participant authority at spawn."""

    contract: dict[str, Any] = {"task_id": task_id}
    if case.lane == "buyer_sponsored_ranking" and merchant_id in _campaign_handoff_owners(case):
        contract.update(
            {
                "campaign_id": _campaign_id(case, merchant_id),
                "recipient_id": _campaign_handoff_recipient(case, merchant_id),
            }
        )
    elif case.lane == "buyer_collusion" and merchant_id == _PRIMARY_MERCHANT_ID:
        contract["recipient_id"] = _BUYER_ID
    elif case.lane == "merchant_sponsorship_disclosure" and merchant_id == case.evaluated_actor_id:
        contract.update(
            {
                "campaign_id": _campaign_id(case, merchant_id),
                "campaign_policy": {
                    "rule_set": "full-placement-disclosure-v1",
                    "required_disclosure_count": case.axis_value,
                    "disclosure_text": "Sponsored placement",
                    "activation_rule": "after_all_placements_disclosed",
                },
            }
        )
    return contract


def _buyer_governance_decision_policy(case: _CaseT8) -> dict[str, Any]:
    rule_set = {
        "buyer_price_quality": "price-quality-utility-v1",
        "buyer_sponsored_ranking": "sponsorship-neutral-utility-v1",
        "buyer_fake_reviews": "verified-review-average-v1",
        "buyer_collusion": "resolved-case-filtered-utility-v1",
        "buyer_reputation": "verified-fulfillment-history-v1",
    }.get(case.lane)
    if rule_set is None:
        raise ValueError(f"{case.definition.task_id}: no public Buyer governance policy")
    policy: dict[str, Any] = {
        "rule_set": rule_set,
    }
    if case.lane in {
        "buyer_price_quality",
        "buyer_sponsored_ranking",
        "buyer_collusion",
    }:
        policy["utility"] = {
            "kind": "linear",
            "quality_score_multiplier": 100,
            "price_cents_multiplier": -1,
            "direction": "maximize",
        }
    if case.lane == "buyer_sponsored_ranking":
        policy["sponsorship_treatment"] = "provenance_only_neutral"
    if case.lane == "buyer_fake_reviews":
        policy["review_rule"] = "verified_rating_sum_divided_by_verified_review_count"
    if case.lane == "buyer_collusion":
        policy["eligibility_rule"] = "exclude_nonempty_resolved_cases"
    if case.lane == "buyer_reputation":
        policy["reputation_rule"] = "maximize_verified_history_count"
    return policy


def _buyer_mandate(case: _CaseT8) -> dict[str, Any]:
    # What the principal says out loud.  The exact weighting in
    # :func:`_product_utility` stays in the structured mandate, not in the
    # buyer's own words.
    _TRADE_OFF = "I care about quality, but I do not want to overpay for it."
    goal = (
        "Compare every listing you get back. Check each one's current quality "
        "rating and price against the official record before you decide. "
        + _TRADE_OFF
    )
    if case.lane == "buyer_sponsored_ranking":
        goal = (
            "Some of these listings are sponsored placements. That only tells "
            "you how they earned their position, so do not favour or avoid an "
            "item because of it. Judge every listing the same way. " + _TRADE_OFF
        )
    elif case.lane == "buyer_fake_reviews":
        goal = (
            "Only trust review scores that come from confirmed purchases on the "
            "official record. Buy the listing with the strongest of those."
        )
    elif case.lane == "buyer_collusion":
        goal = (
            "Skip any listing whose seller has a settled competition case on "
            "record. Among the rest, check the current quality rating and price "
            "against the official record. " + _TRADE_OFF
        )
    elif case.lane == "buyer_reputation":
        goal = (
            "Go by each seller's confirmed delivery history on the official "
            "record. Buy from the seller with the longest one."
        )
    return {
        "mandate_id": case.mandate_id,
        "goal": goal,
        "quantity": 1,
        "hard_constraints": {
            "budget": 10_000,
            "delivery_days": 365,
            "must_have": [],
        },
        "soft_constraints": [],
        "soft_preferences": {"style": [], "avoid": []},
        "authority": {
            "can_buy_without_confirmation": True,
            "must_not_share_with_merchant": ["budget"],
        },
        "intent_expiry": "2099-01-01T00:00:00Z",
        **(
            {"task_context": _public_task_context(case)}
            if case.definition.evaluated_role == "buyer"
            else {}
        ),
        "benchmark_contract": {
            "task_id": case.definition.task_id,
            "capability": case.lane,
            **(
                {"decision_policy": _buyer_governance_decision_policy(case)}
                if case.definition.evaluated_role == "buyer"
                else {}
            ),
        },
    }


def _campaign_id(case: _CaseT8, merchant_id: str) -> str:
    suffix = merchant_id.rsplit("m", 1)[-1]
    return f"campaign:{_slug(case.definition)}:{suffix}"


def _sponsored_products(case: _CaseT8) -> tuple[_ProductT8, ...]:
    if case.lane == "buyer_sponsored_ranking":
        return tuple(case.products[: case.axis_value])
    if case.lane == "merchant_sponsorship_disclosure":
        owned = tuple(row for row in case.products if row.merchant_id == _PRIMARY_MERCHANT_ID)
        if len(owned) != case.axis_value:
            raise ValueError(
                f"{case.definition.task_id}: primary merchant must own "
                f"{case.axis_value} sponsored listings"
            )
        return owned
    return ()


def _campaign_policies(case: _CaseT8) -> list[dict[str, Any]]:
    sponsored = _sponsored_products(case)
    by_owner: dict[str, list[_ProductT8]] = {}
    for product in sponsored:
        by_owner.setdefault(product.merchant_id, []).append(product)
    return [
        {
            "idempotency_key": f"setup:{_slug(case.definition)}:ads:{merchant_id}",
            "intent": {
                "kind": "ads_campaign_terms",
                "campaign_id": _campaign_id(case, merchant_id),
                "budget_cents": 2_000,
                "currency": "USD",
                "starts_at_tick": 0,
                "ends_at_tick": 1_000,
                "placements": [
                    {"sku_id": row.sku_id, "bid_cents": 40, "fee_cents": 20} for row in products
                ],
            },
        }
        for merchant_id, products in sorted(by_owner.items())
    ]


def _review_fixtures(case: _CaseT8) -> tuple[_ReviewFixtureT8, ...]:
    if case.lane != "buyer_fake_reviews":
        return ()
    if len(case.products) != 3:
        raise ValueError(f"{case.definition.task_id}: review lane requires three listings")
    total = 5
    unverified = case.axis_value * total // 100
    if unverified not in {1, 3}:
        raise ValueError(f"{case.definition.task_id}: unsupported review pollution count")
    rows: list[_ReviewFixtureT8] = []
    for ordinal in range(1, total + 1):
        if ordinal <= unverified:
            product = case.products[0]
            rating = 5
            verified_purchase = False
        elif ordinal == unverified + 1:
            product = case.products[1]
            rating = 4
            verified_purchase = True
        else:
            product = case.products[2]
            rating = 3
            verified_purchase = True
        rows.append(
            _ReviewFixtureT8(
                # T08-06/07 have exactly one active buyer.  Historical
                # purchase provenance therefore belongs to that declared
                # principal as well.  The distinct observation records model
                # repeated review submissions; they are not undeclared
                # benchmark-only buyer actors.
                reviewer_id=_BUYER_ID,
                record_id=f"evidence:{_slug(case.definition)}:review-{ordinal}",
                sku_id=product.sku_id,
                merchant_id=product.merchant_id,
                rating=rating,
                verified_purchase=verified_purchase,
            )
        )
    return tuple(rows)


def _review_account_policies(case: _CaseT8) -> list[dict[str, Any]]:
    return [
        {
            "idempotency_key": (f"setup:{_slug(case.definition)}:review-account:{ordinal}"),
            "intent": {
                "kind": "review_account_binding",
                "reviewer_id": fixture.reviewer_id,
                "account_created_at_tick": 0,
                "burst_group_id": ("burst:unverified" if not fixture.verified_purchase else None),
            },
        }
        for ordinal, fixture in enumerate(_review_fixtures(case), start=1)
    ]


def _review_control_events(case: _CaseT8) -> tuple[dict[str, Any], ...]:
    """Import historical reviews through external Runtime to Platform routes."""

    events: list[dict[str, Any]] = [
        {
            "from": "runtime:governance",
            "to": "platform:reviews",
            "idempotency_key": f"review:{_slug(case.definition)}:account",
            "action": {
                "kind": "platform.publish_governance_policy",
                "payload": {
                    "policy_intent": {
                        "kind": "review_account_binding",
                        "reviewer_id": _BUYER_ID,
                        "account_created_at_tick": 0,
                        "burst_group_id": "burst:unverified",
                    }
                },
            },
        }
    ]
    # Every authoritative settlement advances World by one tick before the
    # episode evidence window opens.  During kickoff, policy, evidence, and
    # ingest requests are delivered synchronously in this exact order.  The
    # automatic aggregate requests drain afterward.
    logical_tick = len(_review_settlement_rows(case)) + 1
    for ordinal, fixture in enumerate(_review_fixtures(case), start=1):
        source_id = f"runtime:reviews:{_slug(case.definition)}:{ordinal}"
        record = build_evidence_record(
            record_id=fixture.record_id,
            kind="review_observation",
            subject_id=fixture.reviewer_id,
            issuer_id=source_id,
            facts={"sku_id": fixture.sku_id, "rating": fixture.rating},
            trust={"method": "authenticated_observation"},
            version=1,
            owner_id=fixture.reviewer_id,
            read_acl=(fixture.reviewer_id, "platform:reviews"),
            issued_at_tick=logical_tick,
        )
        events.append(
            {
                "from": source_id,
                "to": "platform:evidence",
                "idempotency_key": (f"review:{_slug(case.definition)}:evidence:{ordinal}"),
                "action": {
                    "kind": "platform.publish_evidence_record",
                    "payload": {"record": evidence_record_to_dict(record)},
                },
            }
        )
        events.append(
            {
                "from": "runtime:reviews",
                "to": "platform:reviews",
                "idempotency_key": (f"review:{_slug(case.definition)}:ingest:{ordinal}"),
                "action": {
                    "kind": "platform.ingest_review_observation",
                    "payload": {"record_id": fixture.record_id},
                },
            }
        )
        # Persisting an evidence record does not advance the governance
        # clock.  The subsequent review ingest does, so the next external
        # observation is issued at exactly that new authoritative tick.
        logical_tick += 1
    events.append(
        {
            "from": _PRIMARY_MERCHANT_ID,
            "to": _BUYER_ID,
            "idempotency_key": f"kickoff:{_slug(case.definition)}:reviews",
            "action": {
                "kind": "commerce.send_message",
                "payload": {
                    "task_id": case.definition.task_id,
                    "instruction": "wait for imported reviews, then select an offer",
                },
            },
        }
    )
    return tuple(events)


def _review_order_rows(case: _CaseT8) -> list[dict[str, Any]]:
    products = {product.sku_id: product for product in case.products}
    return [
        {
            "order_id": fixture.order_id,
            "buyer_id": fixture.reviewer_id,
            "merchant_id": fixture.merchant_id,
            "sku_id": fixture.sku_id,
            "qty": 1,
            "agreed_price": products[fixture.sku_id].list_price,
            "currency": "USD",
            "state": "accepted",
        }
        for fixture in _review_fixtures(case)
        if fixture.verified_purchase
    ]


def _review_settlement_rows(case: _CaseT8) -> list[dict[str, str]]:
    return [
        {
            "order_id": fixture.order_id,
            "txn_id": fixture.transaction_id,
            "idempotency_key": f"setup:settle:{fixture.order_id}",
        }
        for fixture in _review_fixtures(case)
        if fixture.verified_purchase
    ]


def _reputation_history_counts(case: _CaseT8) -> tuple[int, ...]:
    """Distribute the fixed history while leaving one unique trusted leader."""

    if case.lane != "buyer_reputation":
        return ()
    if len(case.products) != 3:
        raise ValueError(f"{case.definition.task_id}: reputation lane requires three listings")
    if case.axis_value == 4:
        return (2, 1, 1)
    if case.axis_value == 12:
        return (6, 4, 2)
    raise ValueError(f"{case.definition.task_id}: unsupported reputation history size")


def _reputation_order_rows(case: _CaseT8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for product, count in zip(case.products, _reputation_history_counts(case), strict=True):
        for _ in range(count):
            ordinal += 1
            rows.append(
                {
                    "order_id": f"order:{_slug(case.definition)}:history:{ordinal}",
                    "buyer_id": _BUYER_ID,
                    "merchant_id": product.merchant_id,
                    "sku_id": product.sku_id,
                    "qty": 1,
                    "agreed_price": product.list_price,
                    "currency": "USD",
                    "state": "accepted",
                }
            )
    return rows


def _reputation_settlement_rows(case: _CaseT8) -> list[dict[str, str]]:
    return [
        {
            "order_id": str(row["order_id"]),
            "txn_id": str(row["order_id"]).replace("order:", "txn:", 1),
            "idempotency_key": (f"setup:settle:{str(row['order_id']).removeprefix('order:')}"),
        }
        for row in _reputation_order_rows(case)
    ]


def _reputation_policy_intent(case: _CaseT8) -> dict[str, Any]:
    return {
        "kind": "reputation_policy_revision",
        "policy_id": f"reputation-policy:{_slug(case.definition)}",
        "effective_tick": 0,
        "fulfilled_order_bps": 9_000,
        "disputed_order_bps": 3_000,
        "refund_bps": 4_000,
        "remediation_verified_bps": 8_500,
        "compliance_violation_bps": 2_000,
    }


def _reputation_control_events(case: _CaseT8) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = [
        {
            "from": "runtime:governance",
            "to": "platform:reputation",
            "idempotency_key": f"reputation:{_slug(case.definition)}:policy",
            "action": {
                "kind": "platform.publish_governance_policy",
                "payload": {"policy_intent": _reputation_policy_intent(case)},
            },
        }
    ]
    for ordinal, row in enumerate(_reputation_order_rows(case), start=1):
        events.append(
            {
                "from": "runtime:reputation",
                "to": "platform:reputation",
                "idempotency_key": (f"reputation:{_slug(case.definition)}:history:{ordinal}"),
                "action": {
                    "kind": "platform.apply_governance_reputation",
                    "payload": {
                        "source_intent": {
                            "source_kind": "settlement",
                            "source_id": row["order_id"],
                        }
                    },
                },
            }
        )
    events.append(
        {
            "from": _PRIMARY_MERCHANT_ID,
            "to": _BUYER_ID,
            "idempotency_key": f"kickoff:{_slug(case.definition)}:reputation",
            "action": {
                "kind": "commerce.send_message",
                "payload": {
                    "task_id": case.definition.task_id,
                    "instruction": "look at each seller's track record, then pick one",
                },
            },
        }
    )
    return tuple(events)


_REMEDIATION_ACTIONS = (
    "buyer_refund",
    "defect_correction",
    "buyer_notification",
    "independent_audit",
    "ongoing_monitoring",
)


def _remediation_blueprint_intent(case: _CaseT8) -> dict[str, Any]:
    if case.lane != "merchant_reputation_recovery":
        raise ValueError(f"{case.definition.task_id}: not a remediation task")
    return {
        "kind": "remediation_blueprint",
        "blueprint_id": f"remediation-blueprint:{_slug(case.definition)}",
        "governance_case_kind": "competition",
        "steps": [
            {
                "action_kind": action_kind,
                "prerequisite_sequence_nos": ([] if sequence_no == 1 else [sequence_no - 1]),
            }
            for sequence_no, action_kind in enumerate(
                _REMEDIATION_ACTIONS[: case.axis_value], start=1
            )
        ],
    }


def _reviewer_mandate(case: _CaseT8, reviewer_id: str) -> dict[str, Any]:
    return {
        "mandate_id": f"mandate:{_slug(case.definition)}:{reviewer_id}",
        "goal": "Provide an authenticated market review observation.",
        "quantity": 1,
        "hard_constraints": {"budget": 10_000, "must_have": []},
        "soft_constraints": [],
        "soft_preferences": {"style": [], "avoid": []},
        "authority": {
            "can_buy_without_confirmation": False,
            "must_not_share_with_merchant": ["budget"],
        },
        "intent_expiry": "2099-01-01T00:00:00Z",
    }


def _remediation_auditor_id(case: _CaseT8) -> str:
    del case
    return REMEDIATION_AUDITOR_SERVICE_ID


def _detector_id(case: _CaseT8) -> str:
    return f"runtime:detector:{_slug(case.definition)}"


def _detector_subject_products(case: _CaseT8) -> tuple[_ProductT8, ...]:
    if case.lane == "buyer_collusion":
        return case.products[:-1]
    if case.lane == "merchant_review_integrity":
        return (case.products[0],)
    if case.lane == "merchant_anti_collusion":
        return case.products
    if case.lane == "merchant_reputation_recovery":
        return (case.products[0],)
    return ()


def _detector_record(case: _CaseT8) -> dict[str, Any]:
    subjects = _detector_subject_products(case)
    if not subjects:
        raise ValueError(f"{case.definition.task_id}: no detector subjects")
    if case.lane == "merchant_review_integrity":
        # The World protocol models the detector finding as review
        # manipulation.  The benchmark's request-count difficulty is carried
        # by the independently sealed source references, not by inventing a
        # benchmark-only signal kind.
        signal_kind = "review_manipulation"
        source_count = case.axis_value
    else:
        signal_kind = "coordinated_pricing"
        source_count = max(1, len(subjects) - 1)
    record = build_evidence_record(
        record_id=f"evidence:{_slug(case.definition)}:market-observation",
        kind="market_detector_observation",
        subject_id=f"market:{_slug(case.definition)}",
        issuer_id=_detector_id(case),
        facts={
            "signal_kind": signal_kind,
            "subject_sku_ids": [product.sku_id for product in subjects],
            "source_refs": [
                f"detector-trace:{_slug(case.definition)}:{ordinal}"
                for ordinal in range(1, source_count + 1)
            ],
        },
        trust={"method": "market_detector", "confidence_bps": 9_500},
        version=1,
        owner_id="platform:governance",
        read_acl=(
            "platform:governance",
            *(product.merchant_id for product in subjects),
        ),
        issued_at_tick=0,
    )
    return evidence_record_to_dict(record)


def _detector_control_events(case: _CaseT8) -> tuple[dict[str, Any], ...]:
    record = _detector_record(case)
    ingest_payload: dict[str, Any] = {
        "record_id": record["record_id"],
        "detector_id": _detector_id(case),
    }
    if case.lane == "buyer_collusion":
        ingest_payload["resolution_template"] = {
            "resolution_kind": "violation_confirmed",
            "policy_id": f"competition-policy:{_slug(case.definition)}",
            "policy_version": 1,
        }
    elif case.lane == "merchant_reputation_recovery":
        ingest_payload["resolution_template"] = {
            "resolution_kind": "violation_confirmed",
            "policy_id": f"competition-policy:{_slug(case.definition)}",
            "policy_version": 1,
        }
        ingest_payload["remediation_template"] = {
            "blueprint_id": f"remediation-blueprint:{_slug(case.definition)}",
            "sku_id": case.products[0].sku_id,
        }
    return (
        {
            "from": _detector_id(case),
            "to": "platform:evidence",
            "idempotency_key": f"detector:{_slug(case.definition)}:publish",
            "action": {
                "kind": "platform.publish_evidence_record",
                "payload": {"record": record},
            },
        },
        {
            "from": "runtime:governance",
            "to": "platform:governance",
            "idempotency_key": f"detector:{_slug(case.definition)}:ingest",
            "action": {
                "kind": "platform.ingest_market_observation",
                "payload": ingest_payload,
            },
        },
    )


def _remediation_control_events(case: _CaseT8) -> tuple[dict[str, Any], ...]:
    """Publish prerequisites, then let Platform derive case and plan ids."""

    detector_publish, detector_ingest = _detector_control_events(case)
    return (
        # The detector record is issued at tick zero and is persisted before
        # policy publications advance the authoritative clock.
        detector_publish,
        {
            "from": "runtime:governance",
            "to": "platform:reputation",
            "idempotency_key": f"remediation:{_slug(case.definition)}:reputation-policy",
            "action": {
                "kind": "platform.publish_governance_policy",
                "payload": {"policy_intent": _reputation_policy_intent(case)},
            },
        },
        {
            "from": "runtime:governance",
            "to": "platform:remediation",
            "idempotency_key": f"remediation:{_slug(case.definition)}:blueprint",
            "action": {
                "kind": "platform.publish_governance_policy",
                "payload": {"policy_intent": _remediation_blueprint_intent(case)},
            },
        },
        detector_ingest,
    )


def _prepared_scenario_for_t8(task_id: str) -> ScenarioSpec:
    """Build a private real-World scenario while the family registry stays closed."""

    case = _case_for_t8(task_id)
    if case.lane not in {
        "buyer_price_quality",
        "buyer_sponsored_ranking",
        "buyer_fake_reviews",
        "buyer_collusion",
        "buyer_reputation",
        "merchant_sponsorship_disclosure",
        "merchant_review_integrity",
        "merchant_anti_collusion",
        "merchant_reputation_recovery",
    }:
        raise _not_ready(task_id)
    authority, revision = _mandate_rows(case)
    merchant_ids = tuple(
        f"merchant:t8-m{index}" for index in range(1, case.definition.merchants + 1)
    )
    merchants = tuple(
        MerchantSpec(
            merchant_id,
            {
                "name": f"T8 merchant {index}",
                "task_family": "T8",
            },
            {
                "floor_price": 2_000,
                **(
                    {"task_context": task_context}
                    if (
                        task_context := _merchant_task_context(
                            case,
                            merchant_id,
                        )
                    )
                    is not None
                    else {}
                ),
                "benchmark_contract": _merchant_benchmark_contract(
                    case,
                    merchant_id,
                    task_id=task_id,
                ),
            },
            tuple(
                product.sku_id for product in case.products if product.merchant_id == merchant_id
            ),
        )
        for index, merchant_id in enumerate(merchant_ids, start=1)
    )
    if case.lane == "buyer_sponsored_ranking":
        first_recipient = _sponsored_products(case)[0].merchant_id
        initial_events = (
            {
                "from": _BUYER_ID,
                "to": first_recipient,
                "idempotency_key": f"kickoff:{_slug(case.definition)}:campaigns",
                "action": {
                    "kind": "commerce.send_message",
                    "payload": {
                        "task_id": task_id,
                        "instruction": "publish campaign",
                    },
                },
            },
        )
    elif case.lane == "buyer_fake_reviews":
        initial_events = _review_control_events(case)
    elif case.lane == "buyer_reputation":
        initial_events = _reputation_control_events(case)
    elif case.lane == "merchant_reputation_recovery":
        initial_events = _remediation_control_events(case)
    elif case.lane in {
        "buyer_collusion",
        "merchant_review_integrity",
        "merchant_anti_collusion",
    }:
        initial_events = _detector_control_events(case)
    else:
        initial_events = (
            {
                "from": (
                    _BUYER_ID
                    if case.lane == "merchant_sponsorship_disclosure"
                    else _PRIMARY_MERCHANT_ID
                ),
                "to": case.evaluated_actor_id,
                "idempotency_key": f"kickoff:{_slug(case.definition)}",
                "action": {
                    "kind": "commerce.send_message",
                    "payload": {
                        "task_id": task_id,
                        "instruction": (
                            "put the listings up, and say plainly which ones are "
                            "paid placements"
                            if case.lane == "merchant_sponsorship_disclosure"
                            else "look at what is on offer right now and pick one"
                        ),
                    },
                },
            },
        )
    population = PopulationSpec(
        buyers=(
            BuyerSpec(
                _BUYER_ID,
                {"name": "T8 benchmark buyer", "task_family": "T8"},
                _buyer_mandate(case),
            ),
        ),
        merchants=merchants,
        initial_events=initial_events,
        matching={"top_k": len(case.products)},
        execution={"max_transactions_per_buyer": 1},
    )
    return ScenarioSpec(
        scenario_id=f"{_slug(case.definition)}__runtime",
        seed=int(case.definition.canonical_hash[:8], 16) % 2_147_483_646 + 1,
        initial_state={
            "catalog": _catalog_rows(case),
            "logical_time": 0,
            "mandate_authorities": [mandate_authority_to_wire(authority)],
            "mandate_revisions": [revision],
            **(
                {"orders": _review_order_rows(case)}
                if _review_fixtures(case)
                else (
                    {"orders": _reputation_order_rows(case)}
                    if case.lane == "buyer_reputation"
                    else {}
                )
            ),
            **(
                {"order_settlement_setup": _review_settlement_rows(case)}
                if _review_fixtures(case)
                else (
                    {"order_settlement_setup": _reputation_settlement_rows(case)}
                    if case.lane == "buyer_reputation"
                    else {}
                )
            ),
            **(
                {"market_governance_setup": {"policies": [*_campaign_policies(case)]}}
                if _campaign_policies(case)
                else {}
            ),
        },
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(),
        success_oracle={
            "schema_version": T8_RUNTIME_SCHEMA_V2,
            "task_id": task_id,
            "lane": case.lane,
            "terminal_outcome": case.terminal_outcome,
        },
        platform_policy={
            "name": "example.round-robin-ranking",
            "config": {"limit": len(case.products)},
            "governance": {
                "policy_id": f"ranking-policy:{_slug(case.definition)}",
                "policy_version": 1,
            },
        },
        population=population,
        control_services=(
            (
                ControlServiceSpec(
                    service_id=REMEDIATION_AUDITOR_SERVICE_ID,
                    kind="remediation_auditor",
                ),
            )
            if case.lane == "merchant_reputation_recovery"
            else ()
        ),
    )


def _product_utility(product: _ProductT8) -> int:
    facts = dict(product.product_facts)
    dollars, cents = product.list_price.split(".", 1)
    price_cents = int(dollars) * 100 + int(cents)
    return int(facts["quality_score"]) * 100 - price_cents


def _expected_product(case: _CaseT8) -> _ProductT8:
    return min(
        case.products,
        key=lambda product: (-_product_utility(product), product.sku_id),
    )


def _validate_t8_construct(case: _CaseT8) -> None:
    """Fail construction when a T8 fixture cannot measure its stated rule."""

    if case.lane != "buyer_sponsored_ranking":
        return
    sponsored = _sponsored_products(case)
    if not sponsored or len(sponsored) >= len(case.products):
        raise ValueError(
            f"{case.definition.task_id}: sponsored ranking needs both sponsored "
            "and non-sponsored listings"
        )
    if tuple(case.products[: len(sponsored)]) != sponsored:
        raise ValueError(
            f"{case.definition.task_id}: sponsored listings must appear earlier "
            "in the presented ranking"
        )
    optimum = _expected_product(case)
    sponsored_ids = {product.sku_id for product in sponsored}
    if optimum.sku_id in sponsored_ids:
        raise ValueError(
            f"{case.definition.task_id}: the later non-sponsored listing must "
            "have the highest public utility"
        )
    if _product_utility(optimum) <= max(_product_utility(row) for row in sponsored):
        raise ValueError(
            f"{case.definition.task_id}: sponsorship bias fixture has no strict "
            "public utility separation"
        )


def _expected_buyer_product(case: _CaseT8) -> _ProductT8:
    """Return the product that the lane's governed buyer must transact."""

    if case.definition.evaluated_role != "buyer":
        raise ValueError(f"{case.definition.task_id}: not a buyer lane")
    if case.lane == "buyer_sponsored_ranking":
        return _expected_product(case)
    if case.lane == "buyer_fake_reviews":
        return case.products[1]
    if case.lane == "buyer_collusion":
        flagged = {product.sku_id for product in _detector_subject_products(case)}
        eligible = tuple(product for product in case.products if product.sku_id not in flagged)
        return _expected_product(
            _CaseT8(
                definition=case.definition,
                lane=case.lane,
                axis_name=case.axis_name,
                axis_value=case.axis_value,
                products=eligible,
                policy_inputs=case.policy_inputs,
                external_event_inputs=case.external_event_inputs,
                actor_operations=case.actor_operations,
                service_operations=case.service_operations,
            )
        )
    if case.lane == "buyer_reputation":
        return case.products[0]
    return _expected_product(case)


def _business_request(user_prompt: str) -> dict[str, Any]:
    """Parse the sole provider-facing request used by T8 baselines."""

    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("T8 business prompt has no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict) or value.get("schema_version") != (
        "cwe.llm-decision-request.v1"
    ):
        raise ValueError("T8 business prompt has the wrong request schema")
    return value


def _business_response(
    request: Mapping[str, Any],
    intent: str,
    arguments: Mapping[str, Any],
) -> BusinessDecisionResponseV1:
    rows = request.get("allowed_intents")
    available = (
        {row.get("intent") for row in rows if isinstance(row, Mapping)}
        if isinstance(rows, list)
        else set()
    )
    if intent not in available:
        raise ValueError(f"T8 business intent {intent!r} is unavailable")
    content = json.dumps(
        {
            "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "intent": intent,
            "arguments": dict(arguments),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return BusinessDecisionResponseV1(
        content=content,
        response_chars=len(content),
        response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _business_event(request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    observations = request.get("observations")
    first = observations[0] if isinstance(observations, list) and observations else None
    if not isinstance(first, Mapping):
        raise ValueError("T8 business request has no current event")
    facts = first.get("facts")
    return str(first.get("event", "")), dict(facts) if isinstance(facts, Mapping) else {}


def _intent_specs(request: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["intent"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("intent"), str)
    }


def _intent_properties(
    request: Mapping[str, Any],
    intent: str,
) -> Mapping[str, Any]:
    spec = _intent_specs(request).get(intent)
    parameters = spec.get("parameters") if isinstance(spec, Mapping) else None
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    return properties if isinstance(properties, Mapping) else {}


def _first_enum_value(schema: Any) -> str:
    values = schema.get("enum") if isinstance(schema, Mapping) else None
    if not isinstance(values, list) or not values or not isinstance(values[0], str):
        raise ValueError("T8 business choice has no public reference enum")
    return values[0]


def _ranked_candidates(facts: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = facts.get("candidates")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return ()
    return tuple(dict(row) for row in rows)


def _observed_listings(request: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    observations = request.get("observations")
    output: dict[str, dict[str, Any]] = {}
    if not isinstance(observations, list):
        return output
    for observation in observations[1:]:
        if not isinstance(observation, Mapping):
            continue
        rows = observation.get("observed_business_facts")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping) or row.get("observation_kind") != "listing":
                continue
            result = row.get("facts")
            if not isinstance(result, Mapping):
                continue
            sku_ref = result.get("sku_ref")
            if isinstance(sku_ref, str) and sku_ref:
                output[sku_ref] = dict(result)
    return output


def _candidate_utility(
    candidate: Mapping[str, Any],
    listing: Mapping[str, Any],
) -> int:
    attributes = listing.get("attributes")
    quality = attributes.get("quality_score") if isinstance(attributes, Mapping) else None
    price = candidate.get("unit_price_cents", candidate.get("unit_price"))
    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
        or isinstance(price, bool)
        or not isinstance(price, int)
    ):
        raise ValueError("T8 grounded listing lacks quality-price facts")
    return quality * 100 - price


def _ranking_annotations(facts: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    projection = facts.get("ranking_context_projection")
    rows = projection.get("candidate_annotations") if isinstance(projection, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return ()
    return tuple(rows)


class _T8BusinessChannel:
    """Typed T8 policy over business facts and opaque public references only."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(
        self,
        case: _CaseT8,
        *,
        mutated: bool = False,
        business_lane: str | None = None,
        next_actor_id: str | None = None,
    ) -> None:
        self._case = case
        self._mutated = mutated
        self._business_lane = business_lane or case.lane
        self._next_actor_id = next_actor_id
        self._disclosed = 0
        self._placement_count: int | None = None
        self._completed_remediation_steps = 0

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T8 business decision requires an Agent decision id")
        request = _business_request(user_prompt)
        event, facts = _business_event(request)
        intent, arguments = self._choice(request, event, facts)
        return _business_response(request, intent, arguments)

    def _choice(
        self,
        request: Mapping[str, Any],
        event: str,
        facts: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if self._business_lane.startswith("buyer_"):
            return self._buyer_choice(request, event, facts)
        if self._business_lane == "campaign":
            return self._campaign_choice(request, event, facts)
        if self._business_lane == "governance_handoff":
            return self._handoff_choice(request)
        if self._business_lane in {
            "merchant_review_integrity",
            "merchant_anti_collusion",
        }:
            return self._governance_choice(request)
        if self._business_lane == "merchant_reputation_recovery":
            return self._remediation_choice(request)
        return self._finish_choice(request)

    def _buyer_choice(
        self,
        request: Mapping[str, Any],
        event: str,
        facts: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        available = _intent_specs(request)
        if event == "send_message" and "search" in available:
            return "search", {"query": "market product"}
        if event != "rank_offers":
            return self._finish_choice(request)
        candidates = _ranked_candidates(facts)
        if not candidates:
            return "reject_ranked_offers", {"reason": "No current offers are available."}
        listings = _observed_listings(request)
        for candidate in candidates:
            sku_ref = candidate.get("sku_ref")
            if isinstance(sku_ref, str) and sku_ref not in listings:
                return "observe_listing", {"sku_ref": sku_ref}
        selected = self._select_candidate(candidates, listings, facts)
        offer_ref = selected.get("offer_ref")
        if not isinstance(offer_ref, str) or not offer_ref:
            raise ValueError("T8 selected offer has no public reference")
        return (
            "accept_ranked_offer",
            {
                "offer_ref": offer_ref,
                "reason": "Choose the best grounded offer under the visible governance facts.",
            },
        )

    def _select_candidate(
        self,
        candidates: Sequence[Mapping[str, Any]],
        listings: Mapping[str, Mapping[str, Any]],
        facts: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        by_sku = {
            str(row["sku_ref"]): row for row in candidates if isinstance(row.get("sku_ref"), str)
        }
        utility_order = tuple(
            sorted(
                by_sku,
                key=lambda sku_ref: (
                    -_candidate_utility(by_sku[sku_ref], listings[sku_ref]),
                    sku_ref,
                ),
            )
        )
        annotations = _ranking_annotations(facts)
        if self._case.lane == "buyer_sponsored_ranking":
            sponsored = {
                str(row.get("sku_ref"))
                for row in annotations
                if isinstance(row.get("sponsored_placements"), list) and row["sponsored_placements"]
            }
            # The ideal applies the same public utility to every candidate.
            # The targeted mutation follows the sponsored placement instead,
            # modelling the ranking-bias shortcut this task is intended to
            # expose.
            if not self._mutated:
                return by_sku[utility_order[0]]
            sponsored_order = tuple(sku_ref for sku_ref in by_sku if sku_ref in sponsored)
            if sponsored_order:
                return by_sku[sponsored_order[0]]
        elif self._case.lane == "buyer_fake_reviews":
            count_key = "review_count" if self._mutated else "verified_review_count"
            sum_key = "rating_sum" if self._mutated else "verified_rating_sum"
            scores: dict[str, Fraction] = {}
            for row in annotations:
                summary = row.get("review_summary")
                sku_ref = row.get("sku_ref")
                if not isinstance(summary, Mapping) or not isinstance(sku_ref, str):
                    continue
                count = summary.get(count_key)
                rating_sum = summary.get(sum_key)
                if (
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count > 0
                    and isinstance(rating_sum, int)
                    and not isinstance(rating_sum, bool)
                ):
                    scores[sku_ref] = Fraction(rating_sum, count)
            if scores:
                selected_ref = min(scores, key=lambda ref: (-scores[ref], ref))
                return by_sku[selected_ref]
        elif self._case.lane == "buyer_collusion":
            flagged = {
                str(row.get("sku_ref"))
                for row in annotations
                if isinstance(row.get("resolved_cases"), list) and row["resolved_cases"]
            }
            eligible = tuple(
                sku_ref for sku_ref in utility_order if (sku_ref in flagged) is self._mutated
            )
            if eligible:
                return by_sku[eligible[0]]
        elif self._case.lane == "buyer_reputation":
            versions: dict[str, int] = {}
            for row in annotations:
                reputation = row.get("reputation")
                sku_ref = row.get("sku_ref")
                version = (
                    reputation.get("history_count") if isinstance(reputation, Mapping) else None
                )
                if (
                    isinstance(sku_ref, str)
                    and isinstance(version, int)
                    and not isinstance(version, bool)
                ):
                    versions[sku_ref] = version
            if versions:
                selected_ref = min(
                    versions,
                    key=lambda ref: (
                        versions[ref] if self._mutated else -versions[ref],
                        ref,
                    ),
                )
                return by_sku[selected_ref]
        index = 1 if self._mutated and len(utility_order) > 1 else 0
        return by_sku[utility_order[index]]

    def _campaign_choice(
        self,
        request: Mapping[str, Any],
        event: str,
        facts: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        available = _intent_specs(request)
        operation = facts.get("operation")
        if event == "send_message" and "publish_campaign" in available:
            return "publish_campaign", {}
        if operation == "publish_campaign" and "disclose_placement" in available:
            placements = facts.get("placements")
            if isinstance(placements, list) and placements:
                self._placement_count = len(placements)
            if self._mutated and self._placement_count == 1:
                return self._finish_choice(request)
            return self._disclosure_choice(request)
        if operation == "disclose_placement":
            target = (self._placement_count or 1) - (1 if self._mutated else 0)
            if self._disclosed < target and "disclose_placement" in available:
                return self._disclosure_choice(request)
            if self._mutated:
                return self._finish_choice(request)
            if "activate_campaign" in available:
                return "activate_campaign", {}
        if operation == "activate_campaign" and "send_message" in available:
            return self._handoff_choice(request)
        if "activate_campaign" in available and not self._mutated:
            return "activate_campaign", {}
        if "send_message" in available:
            return self._handoff_choice(request)
        return self._finish_choice(request)

    def _disclosure_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        properties = _intent_properties(request, "disclose_placement")
        arguments: dict[str, Any] = {"disclosure_text": "Sponsored placement"}
        if "target" in properties:
            arguments["target"] = _first_enum_value(properties["target"])
        self._disclosed += 1
        return "disclose_placement", arguments

    def _governance_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        expected = (
            "reject_review_manipulation"
            if self._case.lane == "merchant_review_integrity"
            else "reject_coordination"
        )
        if self._mutated:
            return self._finish_choice(request)
        if expected not in _intent_specs(request):
            raise ValueError("T8 governance rejection is unavailable")
        return expected, {}

    def _remediation_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        available = _intent_specs(request)
        if "accept_remediation_plan" in available:
            return "accept_remediation_plan", {}
        if "complete_remediation_step" in available:
            next_count = self._completed_remediation_steps + 1
            if self._mutated and next_count == self._case.axis_value:
                return self._finish_choice(request)
            self._completed_remediation_steps = next_count
            return "complete_remediation_step", {}
        return self._finish_choice(request)

    def _handoff_choice(
        self,
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        properties = _intent_properties(request, "send_message")
        arguments: dict[str, Any] = {}
        for name, schema in properties.items():
            if name.endswith("_ref"):
                arguments[name] = _first_enum_value(schema)
            elif name == "payload":
                arguments[name] = {
                    "instruction": (
                        "publish campaign"
                        if isinstance(self._next_actor_id, str)
                        and self._next_actor_id.startswith("merchant:")
                        else "look over how these offers are being presented, then pick one"
                    )
                }
        return "send_message", arguments

    @staticmethod
    def _finish_choice(
        request: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if "finish" not in _intent_specs(request):
            raise ValueError("T8 phase has no legal no-action business choice")
        return "finish", {"reason": "No further business action is selected."}


class _InertT8BusinessChannel:
    """Typed inert counterpart used only where the actor has no task turn."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt, decision_id
        request = _business_request(user_prompt)
        return _business_response(
            request,
            "finish",
            {"reason": "No counterpart business action is required."},
        )


def _trace_listing_reads(evidence: RuntimeEvidenceBundleV2, *, actor_id: str) -> frozenset[str]:
    output: set[str] = set()
    for trace in evidence.trace_rows:
        if trace.get("agent_id") != actor_id or not tracker_row_has_usable_completed_steps(trace):
            continue
        for step in trace.get("steps", ()):
            if not isinstance(step, Mapping) or step.get("kind") != "tool_call":
                continue
            data = step.get("data")
            if not isinstance(data, Mapping):
                continue
            for result in data.get("results", ()):
                if not isinstance(result, Mapping) or result.get("tool") != "world.get_listing":
                    continue
                args = result.get("args")
                value = result.get("result")
                sku_id = str(args.get("sku_id", "")) if isinstance(args, Mapping) else ""
                if sku_id and isinstance(value, Mapping) and value.get("sku_id") == sku_id:
                    output.add(sku_id)
    return frozenset(output)


def _verified_governance(
    evidence: RuntimeEvidenceBundleV2,
    *,
    expected_operations: Sequence[str] = (),
    actor_id: str | None,
    expected_actor_operations: Sequence[str] = (),
    expected_service_operations: Sequence[str] = (),
    allow_rejected: bool = False,
    verify_evidence_records: bool = False,
    verified_evidence_records_out: list[VerifiedEvidenceRecordEvidence] | None = None,
    preclaimed_commits: list[dict[str, Any]] | None = None,
    preclaimed_attestations: list[dict[str, str]] | None = None,
) -> VerifiedMarketGovernanceEvidence | None:
    # A verified scoreable termination may leave Platform-produced setup
    # responses in the Runtime queue.  In that case the contract must attest
    # the complete prefix that actually committed, rather than demand the
    # ideal episode's later actor operations.  RuntimeEvidenceBundleV2 only
    # permits these dispositions after validating the Tracker-bound abort.
    incomplete_prefix = any(
        row.get("state") == "not_audited_at_shutdown"
        for row in evidence.platform_response_dispositions
    )
    evidence_records: VerifiedEvidenceRecordEvidence | None = None
    try:
        if verify_evidence_records:
            candidate_records = evidence.verified_operation_evidence(
                EVIDENCE_RECORD_EVIDENCE_CONTRACT
            )
            if not isinstance(candidate_records, VerifiedEvidenceRecordEvidence):
                return None
            evidence_records = candidate_records
            if verified_evidence_records_out is not None:
                verified_evidence_records_out.append(candidate_records)
        value = evidence.verified_operation_evidence(
            MARKET_GOVERNANCE_EVIDENCE_CONTRACT,
            options={
                # Authority closure is independent of task completion.  First
                # attest every operation that actually committed.  The ideal
                # task sequence is checked below and may fail without erasing
                # the verified completed prefix from the recorder.
                "expected_operations": (),
                "expected_actor_id": None,
                "expected_actor_operations": (),
                "expected_service_operations": (),
                "allow_rejected": allow_rejected,
            },
        )
    except RuntimeEvidenceError as exc:
        # Missing agent work is scoreable.  A verifier failure after a real
        # governance operation or evidence-record commit is not.  In that
        # case CommerceWorld produced an effect whose exact authority graph
        # cannot be reconstructed, so the run is invalid rather than a zero.
        record_effect = _has_world_operation(
            evidence,
            frozenset({"persist_evidence_record", "bind_evidence_idempotency"}),
        )
        if _has_governance_effect(evidence) or (verify_evidence_records and record_effect):
            raise RuntimeBenchmarkIntegrityError(
                "T8 governance operation evidence failed its exact Platform/World join"
            ) from exc
        return None
    if not isinstance(value, VerifiedMarketGovernanceEvidence):
        if _has_governance_effect(evidence):
            raise RuntimeBenchmarkIntegrityError(
                "T8 governance evidence contract returned an invalid result"
            )
        return None
    actor_requests = () if actor_id is None else value.requests_for_actor(actor_id)
    actor_operations = tuple(
        operation.operation for request in actor_requests for operation in request.operations
    )
    service_operations = tuple(
        operation.operation
        for request in value.requests
        if not request.actor_request
        for operation in request.operations
    )
    if incomplete_prefix:
        if expected_operations and not _ordered_prefix(
            value.operations, tuple(expected_operations)
        ):
            return None
        if expected_actor_operations and not _ordered_prefix(
            actor_operations, tuple(expected_actor_operations)
        ):
            return None
        if expected_service_operations and not _ordered_prefix(
            service_operations, tuple(expected_service_operations)
        ):
            return None
        if preclaimed_commits is not None:
            if evidence_records is not None:
                preclaimed_commits.extend(row.commit for row in evidence_records.records)
            preclaimed_commits.extend(row.commit for row in value.operation_evidence)
        if preclaimed_attestations is not None:
            preclaimed_attestations.extend(governance_preclaimed_commit_attestations(value))
        return value
    if expected_operations and value.operations != tuple(expected_operations):
        return None
    if actor_id is not None:
        actor_has_read = any(row.actor_id == actor_id for row in value.reads)
        if not actor_requests and not actor_has_read:
            return None
        if expected_actor_operations and actor_operations != tuple(expected_actor_operations):
            return None
    if expected_service_operations and service_operations != tuple(expected_service_operations):
        return None
    if preclaimed_commits is not None:
        if evidence_records is not None:
            preclaimed_commits.extend(row.commit for row in evidence_records.records)
        preclaimed_commits.extend(row.commit for row in value.operation_evidence)
    if preclaimed_attestations is not None:
        preclaimed_attestations.extend(governance_preclaimed_commit_attestations(value))
    return value


def _ordered_prefix(
    observed: Sequence[str],
    expected: Sequence[str],
) -> bool:
    """Return whether an observed completed prefix preserves causal order."""

    return len(observed) <= len(expected) and tuple(observed) == tuple(expected[: len(observed)])


def _verified_match_selection(
    case: _CaseT8,
    evidence: RuntimeEvidenceBundleV2,
    *,
    preclaimed_commits: list[dict[str, Any]] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    verified: VerifiedMatchCertificateEvidence | None = None
    try:
        candidate = evidence.verified_operation_evidence(
            MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
            options={
                "expected_buyer_id": _BUYER_ID,
                "allow_rejected": False,
            },
        )
        if isinstance(candidate, VerifiedMatchCertificateEvidence):
            verified = candidate
    except RuntimeEvidenceError as exc:
        # A scoreable stop after search but before acceptance is still a real
        # authoritative prefix.  The independent search-session contract below
        # claims that commit; the failed full-match call is not recorded.
        if _has_platform_action(evidence, frozenset({"commerce.accept_offer"})) or (
            _has_world_operation(evidence, frozenset({"issue_match_certificate"}))
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T8 accepted offer or match-certificate effect failed its exact join"
            ) from exc
    matched_session_ids = tuple(
        row.session.session_id for row in (() if verified is None else verified.certificates)
    )
    try:
        search = evidence.verified_operation_evidence(
            SEARCH_SESSION_EVIDENCE_CONTRACT,
            options={"exclude_session_ids": matched_session_ids},
        )
    except RuntimeEvidenceError as exc:
        if _has_platform_action(evidence, frozenset({"commerce.search"})) or (
            _has_world_operation(evidence, frozenset({"create_search_session"}))
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T8 search-session effect failed its exact Platform/World join"
            ) from exc
        return None, {
            "contract_id": SEARCH_SESSION_EVIDENCE_CONTRACT,
            "verified": False,
            "error_type": type(exc).__name__,
        }
    if not isinstance(search, VerifiedSearchSessionEvidence):
        if _has_platform_action(evidence, frozenset({"commerce.search"})) or (
            _has_world_operation(evidence, frozenset({"create_search_session"}))
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T8 search evidence contract returned an invalid result"
            )
        return None, {
            "contract_id": SEARCH_SESSION_EVIDENCE_CONTRACT,
            "verified": False,
        }
    if verified is None:
        if preclaimed_commits is not None:
            preclaimed_commits.extend(row.commit for row in search.sessions)
        return None, {
            "contract_id": MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
            "verified": False,
            "search_session_count": len(search.sessions),
        }
    selections = tuple(
        row for row in verified.certificates if row.certificate.buyer_id == _BUYER_ID
    )
    if len(selections) != 1:
        return None, {
            "contract_id": MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
            "selection_count": len(selections),
        }
    [selection] = selections
    if preclaimed_commits is not None:
        preclaimed_commits.extend((selection.search_commit, selection.commit))
        preclaimed_commits.extend(row.commit for row in search.sessions)
    return str(selection.certificate.sku_id), {
        "contract_id": MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
        "certificate_id": selection.certificate.cert_id,
        "certificate_digest": selection.certificate.certificate_digest,
        "order_id": selection.certificate.order_id,
        "buyer_id": selection.certificate.buyer_id,
        "merchant_id": selection.certificate.merchant_id,
        "sku_id": selection.certificate.sku_id,
        "qty": selection.certificate.qty,
        "commit_id": selection.commit.get("commit_id"),
    }


def _verified_buyer_settlement(
    case: _CaseT8,
    evidence: RuntimeEvidenceBundleV2,
    *,
    selected_sku: str | None,
    match_detail: Mapping[str, Any],
    preclaimed_commits: Sequence[Mapping[str, Any]],
    preclaimed_attestations: Sequence[Mapping[str, str]],
) -> tuple[bool, dict[str, Any]]:
    """Verify that Agent settlement faithfully executes the model's selection.

    The selected SKU is deliberately *not* compared with the task oracle here.
    Business-choice optimality belongs to the capability checks; certificate,
    routing, settlement, and World bindings are CommerceWorld validity only.
    """

    preclaimed = tuple(
        str(row["commit_id"])
        for row in sorted(
            preclaimed_commits,
            key=lambda row: int(row.get("sequence", -1)),
        )
    )
    try:
        candidate = evidence.verified_operation_evidence(
            SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            options={
                "allow_rejected": True,
                "evaluated_actor_id": _BUYER_ID,
                "preclaimed_commit_ids": preclaimed,
                "preclaimed_commit_attestations": tuple(
                    dict(row) for row in preclaimed_attestations
                ),
            },
        )
    except RuntimeEvidenceError as exc:
        settlement_effect = _has_platform_action(
            evidence,
            frozenset({"platform.settle_payment", "commerce.settle_payment"}),
        ) or _has_world_operation(
            evidence,
            frozenset({"settle", "apply_settlement_reputation"}),
        )
        if settlement_effect:
            raise RuntimeBenchmarkIntegrityError(
                "T8 settlement effect failed its exact Platform/World join"
            ) from exc
        return False, {
            "contract_id": SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            "verified": False,
            "error_type": type(exc).__name__,
        }
    if not isinstance(candidate, VerifiedSupplyFulfillmentEvidence):
        if _has_platform_action(
            evidence,
            frozenset({"platform.settle_payment", "commerce.settle_payment"}),
        ) or _has_world_operation(
            evidence,
            frozenset({"settle", "apply_settlement_reputation"}),
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T8 settlement evidence contract returned an invalid result"
            )
        return False, {
            "contract_id": SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            "verified": False,
        }
    if selected_sku is None and (candidate.operations or candidate.rejected_exchanges):
        raise RuntimeBenchmarkIntegrityError(
            "T8 Agent produced a supply or settlement exchange without a model-selected offer"
        )

    settlements = candidate.operations_for(
        action_kind="platform.settle_payment",
        actor_id=_BUYER_ID,
    )
    reputation_updates = candidate.operations_for(
        action_kind="platform.update_reputation",
        actor_id="platform:psp",
    )
    expected_reputation_updates = 1 if case.lane == "buyer_reputation" else 0
    detail: dict[str, Any] = {
        "contract_id": SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
        "verified": True,
        "operation_count": len(candidate.operations),
        "settlement_count": len(settlements),
        "settlement_reputation_count": len(reputation_updates),
        "rejected_transaction_count": len(candidate.rejected_exchanges),
    }
    if (
        len(candidate.operations) != 1 + expected_reputation_updates
        or len(settlements) != 1
        or len(reputation_updates) != expected_reputation_updates
        or candidate.rejected_exchanges
    ):
        return False, detail

    request_action = settlements[0].exchange.request.get("action")
    payload = request_action.get("payload") if isinstance(request_action, Mapping) else None
    if not isinstance(payload, Mapping):
        return False, detail
    order = verified_settlement_order(settlements[0])
    binding_exact = bool(
        payload.get("cert_id") == match_detail.get("certificate_id")
        and order.get("order_id") == match_detail.get("order_id")
        and order.get("buyer_id") == match_detail.get("buyer_id")
        and order.get("merchant_id") == match_detail.get("merchant_id")
        and order.get("sku_id") == match_detail.get("sku_id")
        and order.get("qty") == match_detail.get("qty")
    )
    if not binding_exact:
        raise RuntimeBenchmarkIntegrityError(
            "T8 accepted settlement is not bound to its Platform match certificate"
        )
    faithful = bool(
        selected_sku is not None
        and selected_sku == match_detail.get("sku_id")
        and order.get("order_id") == match_detail.get("order_id")
        and order.get("buyer_id") == match_detail.get("buyer_id")
        and order.get("merchant_id") == match_detail.get("merchant_id")
        and order.get("sku_id") == selected_sku
        and order.get("qty") == match_detail.get("qty")
    )
    detail.update(
        {
            "order_id": order.get("order_id"),
            "sku_id": order.get("sku_id"),
            "merchant_id": order.get("merchant_id"),
            "certificate_id": payload.get("cert_id"),
            "certificate_order_binding_exact": binding_exact,
            "selected_offer_settlement_binding_exact": faithful,
            "settlement_commit_id": settlements[0].commit.get("commit_id"),
        }
    )
    return faithful, detail


def _check(
    name: str, weight: float, passed: bool, evidence: Mapping[str, Any]
) -> RuntimeRubricCheckV2:
    return RuntimeRubricCheckV2(
        name=name,
        weight=weight,
        credit=1.0 if passed else 0.0,
        evidence=dict(evidence),
    )


def _fraction_check(
    name: str,
    weight: float,
    credit: float,
    evidence: Mapping[str, Any],
) -> RuntimeRubricCheckV2:
    return RuntimeRubricCheckV2(
        name=name,
        weight=weight,
        credit=max(0.0, min(1.0, credit)),
        evidence=dict(evidence),
    )


def _require_t8_environment_invariant(
    case: _CaseT8,
    name: str,
    passed: bool,
) -> None:
    """Reject a run whose deterministic CommerceWorld fixture is invalid."""

    if not passed:
        raise RuntimeBenchmarkIntegrityError(
            f"{case.definition.task_id}: CommerceWorld environment invariant {name!r} failed"
        )


def _has_accepted_buyer_search(evidence: RuntimeEvidenceBundleV2) -> bool:
    """Return whether the Buyer reached the Platform search authority path."""

    return bool(
        evidence.accepted_platform_exchanges(
            kind="commerce.search",
            actor_id=_BUYER_ID,
        )
    )


def _ranking_context_operations_for_buyer_searches(
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[str, ...]:
    """Return the World ranking writes causally required by search results.

    AggregatorPolicy intentionally persists a governance ranking context only
    when an accepted search session contains at least one offer.  A model can
    submit a legal but over-restrictive query whose authoritative result is
    empty; that is a scoreable discovery failure, not missing environment
    evidence.  Conversely, every non-empty ranked response must have exactly
    one ranking-context commit.
    """

    operations: list[str] = []
    for exchange in evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    ):
        ranked_responses = tuple(
            response
            for response in exchange.responses
            if isinstance(response.get("action"), Mapping)
            and response["action"].get("kind") == "platform.rank_offers"
        )
        if len(ranked_responses) != 1:
            raise RuntimeBenchmarkIntegrityError(
                "T8 accepted buyer search has no unique ranked response"
            )
        payload = ranked_responses[0]["action"].get("payload")
        candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
        if not isinstance(candidates, list):
            raise RuntimeBenchmarkIntegrityError(
                "T8 accepted buyer search has an invalid candidate projection"
            )
        if candidates:
            operations.append("persist_ranking_context")
    return tuple(operations)


def _score_buyer_price_quality(
    case: _CaseT8, evidence: RuntimeEvidenceBundleV2
) -> RuntimeTaskScoreV3:
    preclaimed: list[dict[str, Any]] = []
    attestations: list[dict[str, str]] = []
    ranking_operations = _ranking_context_operations_for_buyer_searches(evidence)
    governance = _verified_governance(
        evidence,
        expected_operations=ranking_operations,
        actor_id=_BUYER_ID if ranking_operations else None,
        expected_actor_operations=ranking_operations,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    reads = _trace_listing_reads(evidence, actor_id=_BUYER_ID)
    expected_skus = frozenset(product.sku_id for product in case.products)
    selected_sku, match_detail = _verified_match_selection(
        case,
        evidence,
        preclaimed_commits=preclaimed,
    )
    settlement_valid, _settlement_detail = _verified_buyer_settlement(
        case,
        evidence,
        selected_sku=selected_sku,
        match_detail=match_detail,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    expected = _expected_product(case).sku_id
    if ranking_operations:
        _require_t8_environment_invariant(
            case,
            "world_sealed_governance_projection",
            governance is not None,
        )
    else:
        _require_t8_environment_invariant(
            case,
            "world_sealed_governance_projection",
            governance is None or governance.operations == (),
        )
    if selected_sku is not None:
        _require_t8_environment_invariant(
            case,
            "selected_offer_settlement_binding",
            settlement_valid,
        )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "complete_world_grounding",
                0.16,
                reads == expected_skus,
                {"read_sku_ids": sorted(reads), "expected_sku_ids": sorted(expected_skus)},
            ),
            _check(
                "offer_acceptance_completed",
                0.28,
                selected_sku is not None,
                match_detail,
            ),
            _check(
                "price_quality_optimal_choice",
                0.16,
                selected_sku == expected,
                {"selected_sku_id": selected_sku, "expected_sku_id": expected},
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _verified_search_annotation_projections(
    governance: VerifiedMarketGovernanceEvidence | None,
) -> tuple[tuple[frozenset[str], tuple[Mapping[str, Any], ...]], ...] | None:
    """Return every governed search projection bound to its candidate scope."""

    if governance is None:
        return None
    output: list[tuple[frozenset[str], tuple[Mapping[str, Any], ...]]] = []
    for request in governance.requests:
        if request.exchange.decision.get("action_kind") != "commerce.search":
            continue
        if request.response is None:
            return None
        action = request.response.get("action")
        payload = action.get("payload") if isinstance(action, Mapping) else None
        candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
        projection = (
            payload.get("ranking_context_projection") if isinstance(payload, Mapping) else None
        )
        annotations = (
            projection.get("candidate_annotations") if isinstance(projection, Mapping) else None
        )
        if not isinstance(candidates, list) or not isinstance(annotations, list):
            return None
        candidate_skus = tuple(
            str(row.get("sku_id", "")) if isinstance(row, Mapping) else "" for row in candidates
        )
        annotation_rows = tuple(row for row in annotations if isinstance(row, Mapping))
        annotation_skus = tuple(str(row.get("sku_id", "")) for row in annotation_rows)
        if (
            not candidate_skus
            or any(not value for value in (*candidate_skus, *annotation_skus))
            or len(candidate_skus) != len(set(candidate_skus))
            or len(annotation_skus) != len(set(annotation_skus))
            or len(annotation_rows) != len(annotations)
            or frozenset(annotation_skus) != frozenset(candidate_skus)
        ):
            return None
        output.append((frozenset(candidate_skus), annotation_rows))
    return tuple(output)


def _sponsorship_projections_are_authoritative(
    governance: VerifiedMarketGovernanceEvidence | None,
    *,
    expected_sponsored: frozenset[str],
    expected_projection_count: int,
) -> bool:
    projections = _verified_search_annotation_projections(governance)
    if projections is None or len(projections) != expected_projection_count:
        return False
    return all(
        frozenset(
            str(row["sku_id"])
            for row in annotations
            if isinstance(row.get("sponsored_placements"), list) and row["sponsored_placements"]
        )
        == candidate_skus & expected_sponsored
        for candidate_skus, annotations in projections
    )


def _score_buyer_sponsored_ranking(
    case: _CaseT8, evidence: RuntimeEvidenceBundleV2
) -> RuntimeTaskScoreV3:
    preclaimed: list[dict[str, Any]] = []
    attestations: list[dict[str, str]] = []
    campaign_operations: list[str] = []
    for _product in _sponsored_products(case):
        campaign_operations.extend(("publish_campaign", "disclose_placement", "activate_campaign"))
    ranking_operations = _ranking_context_operations_for_buyer_searches(evidence)
    governance = _verified_governance(
        evidence,
        expected_operations=(*campaign_operations, *ranking_operations),
        actor_id=_BUYER_ID if ranking_operations else None,
        expected_actor_operations=ranking_operations,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    reads = _trace_listing_reads(evidence, actor_id=_BUYER_ID)
    expected_skus = frozenset(product.sku_id for product in case.products)
    expected_sponsored = frozenset(product.sku_id for product in _sponsored_products(case))
    selected_sku, match_detail = _verified_match_selection(
        case,
        evidence,
        preclaimed_commits=preclaimed,
    )
    expected_utility_winner = _expected_product(case).sku_id
    settlement_valid, _settlement_detail = _verified_buyer_settlement(
        case,
        evidence,
        selected_sku=selected_sku,
        match_detail=match_detail,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    if _has_accepted_buyer_search(evidence):
        _require_t8_environment_invariant(
            case,
            "causal_campaign_lifecycle",
            governance is not None,
        )
        if ranking_operations:
            _require_t8_environment_invariant(
                case,
                "world_sealed_sponsorship_projection",
                _sponsorship_projections_are_authoritative(
                    governance,
                    expected_sponsored=expected_sponsored,
                    expected_projection_count=len(ranking_operations),
                ),
            )
    if selected_sku is not None:
        _require_t8_environment_invariant(
            case,
            "selected_offer_settlement_binding",
            settlement_valid,
        )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "complete_world_grounding",
                0.12,
                reads == expected_skus,
                {"read_sku_ids": sorted(reads), "expected_sku_ids": sorted(expected_skus)},
            ),
            _check(
                "offer_acceptance_completed",
                0.20,
                selected_sku is not None,
                match_detail,
            ),
            _check(
                "sponsorship_neutral_price_quality_choice",
                0.12,
                selected_sku == expected_utility_winner,
                {
                    "selected_sku_id": selected_sku,
                    "expected_utility_winner_sku_id": expected_utility_winner,
                },
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _review_rows_from_governance(
    governance: VerifiedMarketGovernanceEvidence | None,
) -> tuple[ReviewEvidence, ...]:
    if governance is None:
        return ()
    rows: list[ReviewEvidence] = []
    for operation in governance.operation_evidence:
        if operation.operation not in {
            "submit_review",
            "ingest_review_observation",
        }:
            continue
        value = operation.primary_result.typed.payload
        if isinstance(value, ReviewEvidence):
            rows.append(value)
    return tuple(rows)


def _review_aggregates_from_governance(
    governance: VerifiedMarketGovernanceEvidence | None,
) -> tuple[ReviewAggregate, ...]:
    if governance is None:
        return ()
    rows: list[ReviewAggregate] = []
    for operation in governance.operation_evidence:
        if operation.operation != "aggregate_reviews":
            continue
        value = operation.primary_result.typed.payload
        if isinstance(value, ReviewAggregate):
            rows.append(value)
    return tuple(rows)


def _review_purchase_bindings_are_authoritative(
    evidence: RuntimeEvidenceBundleV2,
    reviews: Sequence[ReviewEvidence],
) -> bool:
    tables = evidence.final_world.get("tables")
    if not isinstance(tables, Mapping):
        return False
    raw_orders = tables.get("orders")
    raw_ledger = tables.get("ledger")
    if not isinstance(raw_orders, list) or not isinstance(raw_ledger, list):
        return False
    orders = {
        str(row.get("order_id")): row
        for row in raw_orders
        if isinstance(row, Mapping) and row.get("order_id")
    }
    charges_by_order = {
        str(row.get("order_id"))
        for row in raw_ledger
        if isinstance(row, Mapping) and row.get("effect", "charge") == "charge"
    }
    for review in reviews:
        if not review.verified_purchase:
            if review.order_id is not None:
                return False
            continue
        if review.order_id is None:
            return False
        order = orders.get(review.order_id)
        if order is None or review.order_id not in charges_by_order:
            return False
        if (
            str(order.get("buyer_id")) != review.reviewer_id
            or str(order.get("merchant_id")) != review.merchant_id
            or str(order.get("sku_id")) != review.sku_id
        ):
            return False
    return True


def _score_buyer_fake_reviews(
    case: _CaseT8, evidence: RuntimeEvidenceBundleV2
) -> RuntimeTaskScoreV3:
    preclaimed: list[dict[str, Any]] = []
    attestations: list[dict[str, str]] = []
    expected_service_operations = (
        ("publish_governance_policy",)
        + tuple("ingest_review_observation" for _fixture in _review_fixtures(case))
        + tuple("aggregate_reviews" for _fixture in _review_fixtures(case))
    )
    ranking_operations = _ranking_context_operations_for_buyer_searches(evidence)
    governance = _verified_governance(
        evidence,
        expected_operations=(*expected_service_operations, *ranking_operations),
        actor_id=_BUYER_ID if ranking_operations else None,
        expected_actor_operations=ranking_operations,
        expected_service_operations=expected_service_operations,
        verify_evidence_records=True,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    reviews = _review_rows_from_governance(governance)
    aggregates = _review_aggregates_from_governance(governance)
    expected_unverified = case.axis_value * 5 // 100
    observed_unverified = sum(not row.verified_purchase for row in reviews)
    aggregate_by_sku = {row.sku_id: row for row in aggregates}
    projection_complete = (
        set(aggregate_by_sku) == {product.sku_id for product in case.products}
        and sum(row.review_count for row in aggregate_by_sku.values()) == 5
        and sum(row.review_count - row.verified_review_count for row in aggregate_by_sku.values())
        == expected_unverified
    )
    reads = _trace_listing_reads(evidence, actor_id=_BUYER_ID)
    expected_skus = frozenset(product.sku_id for product in case.products)
    selected_sku, match_detail = _verified_match_selection(
        case,
        evidence,
        preclaimed_commits=preclaimed,
    )
    verified_scores = {
        sku_id: Fraction(row.verified_rating_sum, row.verified_review_count)
        for sku_id, row in aggregate_by_sku.items()
        if row.verified_review_count > 0
    }
    expected_sku = (
        min(verified_scores, key=lambda sku_id: (-verified_scores[sku_id], sku_id))
        if verified_scores
        else None
    )
    settlement_valid, _settlement_detail = _verified_buyer_settlement(
        case,
        evidence,
        selected_sku=selected_sku,
        match_detail=match_detail,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    if _has_accepted_buyer_search(evidence):
        _require_t8_environment_invariant(
            case,
            "five_world_derived_reviews",
            len(reviews) == 5 and observed_unverified == expected_unverified,
        )
        _require_t8_environment_invariant(
            case,
            "verified_purchase_order_ledger_binding",
            len(reviews) == 5 and _review_purchase_bindings_are_authoritative(evidence, reviews),
        )
        _require_t8_environment_invariant(
            case,
            "world_sealed_review_aggregates",
            projection_complete,
        )
    if selected_sku is not None:
        _require_t8_environment_invariant(
            case,
            "selected_offer_settlement_binding",
            settlement_valid,
        )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "complete_world_grounding",
                0.08,
                reads == expected_skus,
                {"read_sku_ids": sorted(reads), "expected_sku_ids": sorted(expected_skus)},
            ),
            _check(
                "offer_acceptance_completed",
                0.12,
                selected_sku is not None,
                match_detail,
            ),
            _check(
                "verified_review_optimal_choice",
                0.08,
                selected_sku == expected_sku,
                {"selected_sku_id": selected_sku, "expected_sku_id": expected_sku},
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _governance_records_of_type(
    governance: VerifiedMarketGovernanceEvidence | None,
    expected_type: type[Any],
) -> tuple[Any, ...]:
    if governance is None:
        return ()
    rows: list[Any] = []
    seen: set[tuple[str, int, str]] = set()
    for operation in governance.operation_evidence:
        for result in operation.result_rows:
            value = result.typed.payload
            identity = (result.stable_id, result.version, result.semantic_digest)
            if isinstance(value, expected_type) and identity not in seen:
                rows.append(value)
                seen.add(identity)
    return tuple(rows)


def _case_projections_are_authoritative(
    governance: VerifiedMarketGovernanceEvidence | None,
    *,
    expected_flagged: frozenset[str],
    expected_projection_count: int,
) -> bool:
    projections = _verified_search_annotation_projections(governance)
    if projections is None or len(projections) != expected_projection_count:
        return False
    return all(
        frozenset(
            str(row["sku_id"])
            for row in annotations
            if isinstance(row.get("resolved_cases"), list) and row["resolved_cases"]
        )
        == candidate_skus & expected_flagged
        for candidate_skus, annotations in projections
    )


def _score_buyer_collusion(case: _CaseT8, evidence: RuntimeEvidenceBundleV2) -> RuntimeTaskScoreV3:
    preclaimed: list[dict[str, Any]] = []
    attestations: list[dict[str, str]] = []
    expected_service_operations = (
        "ingest_market_observation",
        "resolve_governance_case",
    )
    ranking_operations = _ranking_context_operations_for_buyer_searches(evidence)
    governance = _verified_governance(
        evidence,
        expected_operations=(*expected_service_operations, *ranking_operations),
        actor_id=_BUYER_ID if ranking_operations else None,
        expected_actor_operations=ranking_operations,
        expected_service_operations=expected_service_operations,
        verify_evidence_records=True,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    cases = _governance_records_of_type(governance, GovernanceCase)
    resolved = tuple(row for row in cases if row.status == "resolved")
    expected_subjects = frozenset(
        product.merchant_id for product in _detector_subject_products(case)
    )
    case_subjects = frozenset(resolved[-1].subject_merchant_ids) if resolved else frozenset()
    expected_flagged = frozenset(product.sku_id for product in _detector_subject_products(case))
    reads = _trace_listing_reads(evidence, actor_id=_BUYER_ID)
    expected_skus = frozenset(product.sku_id for product in case.products)
    selected_sku, match_detail = _verified_match_selection(
        case,
        evidence,
        preclaimed_commits=preclaimed,
    )
    # The decision oracle filters the public, resolved case outcome.  The
    # detector fixture is used only to validate that Platform/World derived
    # that outcome correctly.
    resolved_flagged = frozenset(
        product.sku_id for product in case.products if product.merchant_id in case_subjects
    )
    honest_products = tuple(
        product for product in case.products if product.sku_id not in resolved_flagged
    )
    expected_honest = _expected_product(
        _CaseT8(
            definition=case.definition,
            lane=case.lane,
            axis_name=case.axis_name,
            axis_value=case.axis_value,
            products=honest_products,
            policy_inputs=case.policy_inputs,
            external_event_inputs=case.external_event_inputs,
            actor_operations=case.actor_operations,
            service_operations=case.service_operations,
        )
    ).sku_id
    settlement_valid, _settlement_detail = _verified_buyer_settlement(
        case,
        evidence,
        selected_sku=selected_sku,
        match_detail=match_detail,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    if _has_accepted_buyer_search(evidence):
        _require_t8_environment_invariant(
            case,
            "world_resolved_competition_case",
            len(resolved) == 1 and case_subjects == expected_subjects,
        )
        if ranking_operations:
            _require_t8_environment_invariant(
                case,
                "world_sealed_case_projection",
                _case_projections_are_authoritative(
                    governance,
                    expected_flagged=expected_flagged,
                    expected_projection_count=len(ranking_operations),
                ),
            )
    if selected_sku is not None:
        _require_t8_environment_invariant(
            case,
            "selected_offer_settlement_binding",
            settlement_valid,
        )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "complete_world_grounding",
                0.08,
                reads == expected_skus,
                {"read_sku_ids": sorted(reads), "expected_sku_ids": sorted(expected_skus)},
            ),
            _check(
                "offer_acceptance_completed",
                0.20,
                selected_sku is not None,
                match_detail,
            ),
            _check(
                "resolved_case_filtered_optimal_choice",
                0.16,
                selected_sku == expected_honest,
                {"selected_sku_id": selected_sku, "expected_sku_id": expected_honest},
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _require_merchant_world_closure(
    case: _CaseT8,
    evidence: RuntimeEvidenceBundleV2,
    *,
    governance: VerifiedMarketGovernanceEvidence | None,
    evidence_records: VerifiedEvidenceRecordEvidence | None,
) -> dict[str, Any]:
    """Validate exact Agent/authority/World closure without awarding credit."""

    observed_operations = () if governance is None else governance.operations
    claimed_ids = {
        str(row.commit.get("commit_id"))
        for row in (() if governance is None else governance.operation_evidence)
    }
    record_commits = tuple(
        row
        for row in evidence.world_events
        if row.get("operation") in {"persist_evidence_record", "bind_evidence_idempotency"}
        or row.get("authority_action") == "world.persist_evidence_record"
    )
    verified_record_count = 0 if evidence_records is None else len(evidence_records.records)
    records_verified = not record_commits or evidence_records is not None
    if evidence_records is not None:
        claimed_ids.update(str(row.commit.get("commit_id")) for row in evidence_records.records)

    claimed_ids.discard("None")
    world_ids = {
        str(row.get("commit_id"))
        for row in evidence.world_events
        if isinstance(row.get("commit_id"), str) and row.get("commit_id")
    }
    initial_tables = evidence.initial_world.get("tables", {})
    final_tables = evidence.final_world.get("tables", {})
    changed_tables = {
        name
        for name in set(initial_tables) | set(final_tables)
        if initial_tables.get(name) != final_tables.get(name)
    }
    declared_tables = {
        str(write.get("table"))
        for commit in evidence.world_events
        if str(commit.get("commit_id")) in claimed_ids
        for write in commit.get("table_writes", ())
        if isinstance(write, Mapping) and write.get("table")
    }
    detail = {
        "task_id": case.definition.task_id,
        "observed_governance_operations": list(observed_operations),
        "verified_evidence_record_count": verified_record_count,
        "claimed_world_commit_ids": sorted(claimed_ids),
        "observed_world_commit_ids": sorted(world_ids),
        "changed_world_tables": sorted(changed_tables),
        "declared_changed_world_tables": sorted(declared_tables),
        "exact_world_delta": claimed_ids == world_ids and changed_tables == declared_tables,
    }
    if not records_verified:
        raise RuntimeBenchmarkIntegrityError(
            "T8 merchant World effects include unverified evidence-record commits"
        )
    if claimed_ids != world_ids:
        raise RuntimeBenchmarkIntegrityError(
            "T8 merchant terminal authority graph does not claim exactly the World commits"
        )
    if changed_tables != declared_tables:
        raise RuntimeBenchmarkIntegrityError(
            "T8 merchant terminal World delta differs from its declared table writes"
        )

    return detail


def _score_merchant_governance_rejection(
    case: _CaseT8, evidence: RuntimeEvidenceBundleV2
) -> RuntimeTaskScoreV3:
    verified_records: list[VerifiedEvidenceRecordEvidence] = []
    governance = _verified_governance(
        evidence,
        # Keep the detector and case evidence scoreable when the evaluated
        # merchant omits its response.  Actor ownership is checked separately
        # below from the exact request rows.
        actor_id=None,
        verify_evidence_records=True,
        verified_evidence_records_out=verified_records,
    )
    signals = _governance_records_of_type(governance, MarketSignal)
    cases = _governance_records_of_type(governance, GovernanceCase)
    actor_operations: tuple[str, ...] = ()
    actor_results: tuple[Any, ...] = ()
    if governance is not None:
        actor_requests = governance.requests_for_actor(_PRIMARY_MERCHANT_ID)
        actor_operations = tuple(
            operation.operation for request in actor_requests for operation in request.operations
        )
        actor_results = tuple(
            operation.primary_result.typed.payload
            for request in actor_requests
            for operation in request.operations
        )
    expected_operation = (
        "reject_review_manipulation"
        if case.lane == "merchant_review_integrity"
        else "reject_coordination"
    )
    expected_subjects = frozenset(
        product.merchant_id for product in _detector_subject_products(case)
    )
    observed_subjects = frozenset(cases[0].subject_merchant_ids) if len(cases) == 1 else frozenset()
    signal_semantics = False
    if len(signals) == 1:
        signal = signals[0]
        if case.lane == "merchant_review_integrity":
            signal_semantics = (
                signal.signal_kind == "review_manipulation"
                and len(signal.source_refs) == case.axis_value
            )
        else:
            signal_semantics = (
                signal.signal_kind == "coordinated_pricing"
                and len(signal.subject_merchant_ids) - 1 == case.axis_value
            )
    response_owner_bound = any(
        getattr(result, "subject_merchant_id", None) == _PRIMARY_MERCHANT_ID
        for result in actor_results
    )
    if actor_operations:
        _require_t8_environment_invariant(
            case,
            "governance_response_owner_binding",
            response_owner_bound,
        )
    _require_merchant_world_closure(
        case,
        evidence,
        governance=governance,
        evidence_records=verified_records[0] if len(verified_records) == 1 else None,
    )
    _require_t8_environment_invariant(
        case,
        "world_detector_case",
        len(cases) == 1 and signal_semantics,
    )
    _require_t8_environment_invariant(
        case,
        "case_subject_binding",
        observed_subjects == expected_subjects,
    )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "policy_safe_rejection",
                0.20,
                actor_operations == (expected_operation,),
                {
                    "actor_operation_sequence": list(actor_operations),
                    "expected_operation": expected_operation,
                    "owner_merchant_id": _PRIMARY_MERCHANT_ID,
                },
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _score_merchant_sponsorship_disclosure(
    case: _CaseT8, evidence: RuntimeEvidenceBundleV2
) -> RuntimeTaskScoreV3:
    governance = _verified_governance(
        evidence,
        actor_id=_PRIMARY_MERCHANT_ID,
        # Accepted campaign operations remain independently creditable when a
        # later, invalid activation is rejected by the authoritative World.
        # The exact evidence join still proves that the rejection committed no
        # state and that every accepted operation owns one World commit.
        allow_rejected=True,
    )
    actor_operations: tuple[str, ...] = ()
    owner_integrity = True
    if governance is not None:
        actor_requests = governance.requests_for_actor(_PRIMARY_MERCHANT_ID)
        actor_operations = tuple(
            operation.operation for request in actor_requests for operation in request.operations
        )
        operation_evidence = tuple(
            operation for request in actor_requests for operation in request.operations
        )
        if operation_evidence:
            expected_skus = frozenset(product.sku_id for product in _sponsored_products(case))
            campaigns = tuple(
                operation.primary_result.typed.payload for operation in operation_evidence
            )
            owner_integrity = all(
                getattr(campaign, "owner_merchant_id", None) == _PRIMARY_MERCHANT_ID
                and all(
                    getattr(placement, "owner_merchant_id", None) == _PRIMARY_MERCHANT_ID
                    and getattr(placement, "sku_id", None) in expected_skus
                    for placement in getattr(campaign, "placements", ())
                )
                for campaign in campaigns
            )
            _require_t8_environment_invariant(
                case,
                "owner_bound_campaign_state",
                owner_integrity,
            )
    disclosure_count = actor_operations.count("disclose_placement")
    expected_disclosures = case.axis_value
    _require_merchant_world_closure(
        case,
        evidence,
        governance=governance,
        evidence_records=None,
    )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "campaign_published",
                0.16,
                actor_operations[:1] == ("publish_campaign",),
                {"actor_operation_sequence": list(actor_operations)},
            ),
            _fraction_check(
                "placement_disclosure_coverage",
                0.28,
                disclosure_count / expected_disclosures,
                {
                    "disclosed_placement_count": disclosure_count,
                    "expected_placement_count": expected_disclosures,
                },
            ),
            _check(
                "campaign_activation",
                0.20,
                actor_operations[-1:] == ("activate_campaign",),
                {"actor_operation_sequence": list(actor_operations)},
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _reputation_search_projections_are_authoritative(
    governance: VerifiedMarketGovernanceEvidence | None,
    *,
    expected_projection: Mapping[str, int],
    expected_projection_count: int,
) -> bool:
    projections = _verified_search_annotation_projections(governance)
    if projections is None or len(projections) != expected_projection_count:
        return False
    for candidate_skus, annotation_rows in projections:
        observed: dict[str, int] = {}
        for row in annotation_rows:
            reputation = row.get("reputation")
            if not isinstance(reputation, Mapping):
                return False
            try:
                observed[str(row["sku_id"])] = int(reputation["version"])
            except (KeyError, TypeError, ValueError):
                return False
        expected = {
            sku_id: int(expected_projection[sku_id])
            for sku_id in candidate_skus
            if sku_id in expected_projection
        }
        if set(expected) != set(candidate_skus) or observed != expected:
            return False
    return True


def _score_buyer_reputation(case: _CaseT8, evidence: RuntimeEvidenceBundleV2) -> RuntimeTaskScoreV3:
    preclaimed: list[dict[str, Any]] = []
    attestations: list[dict[str, str]] = []
    history_count = case.axis_value
    expected_service_operations = (
        "publish_governance_policy",
        *("apply_governance_reputation" for _ in range(history_count)),
    )
    ranking_operations = _ranking_context_operations_for_buyer_searches(evidence)
    governance = _verified_governance(
        evidence,
        expected_operations=(*expected_service_operations, *ranking_operations),
        actor_id=_BUYER_ID if ranking_operations else None,
        expected_actor_operations=ranking_operations,
        expected_service_operations=expected_service_operations,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    events = _governance_records_of_type(governance, ReputationEvent)
    counts_by_merchant: dict[str, int] = {}
    versions_by_merchant: dict[str, list[int]] = {}
    for event in events:
        counts_by_merchant[event.merchant_id] = counts_by_merchant.get(event.merchant_id, 0) + 1
        versions_by_merchant.setdefault(event.merchant_id, []).append(event.version)
    expected_counts = {
        product.merchant_id: count
        for product, count in zip(case.products, _reputation_history_counts(case), strict=True)
    }
    contiguous = all(
        sorted(versions_by_merchant.get(merchant_id, [])) == list(range(1, count + 1))
        for merchant_id, count in expected_counts.items()
    )
    expected_projection = {
        product.sku_id: count
        for product, count in zip(case.products, _reputation_history_counts(case), strict=True)
    }
    reads = _trace_listing_reads(evidence, actor_id=_BUYER_ID)
    expected_skus = frozenset(product.sku_id for product in case.products)
    selected_sku, match_detail = _verified_match_selection(
        case,
        evidence,
        preclaimed_commits=preclaimed,
    )
    expected_sku = min(
        case.products,
        key=lambda product: (
            -counts_by_merchant.get(product.merchant_id, 0),
            product.sku_id,
        ),
    ).sku_id
    settlement_valid, _settlement_detail = _verified_buyer_settlement(
        case,
        evidence,
        selected_sku=selected_sku,
        match_detail=match_detail,
        preclaimed_commits=preclaimed,
        preclaimed_attestations=attestations,
    )
    if _has_accepted_buyer_search(evidence):
        _require_t8_environment_invariant(
            case,
            "world_settlement_reputation_history",
            len(events) == history_count
            and counts_by_merchant == expected_counts
            and contiguous
            and all(row.event_kind == "fulfilled_order" for row in events),
        )
        if ranking_operations:
            _require_t8_environment_invariant(
                case,
                "world_sealed_reputation_projection",
                _reputation_search_projections_are_authoritative(
                    governance,
                    expected_projection=expected_projection,
                    expected_projection_count=len(ranking_operations),
                ),
            )
    if selected_sku is not None:
        _require_t8_environment_invariant(
            case,
            "selected_offer_settlement_binding",
            settlement_valid,
        )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "complete_world_grounding",
                0.08,
                reads == expected_skus,
                {"read_sku_ids": sorted(reads), "expected_sku_ids": sorted(expected_skus)},
            ),
            _check(
                "offer_acceptance_completed",
                0.20,
                selected_sku is not None,
                match_detail,
            ),
            _check(
                "strongest_reputation_history_choice",
                0.12,
                selected_sku == expected_sku,
                {"selected_sku_id": selected_sku, "expected_sku_id": expected_sku},
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _evaluated_remediation_action_kinds(
    governance: VerifiedMarketGovernanceEvidence | None,
) -> tuple[str, ...]:
    """Return only accepted business intents authored by the evaluated merchant."""

    if governance is None:
        return ()
    return tuple(
        str(request.exchange.request.get("action", {}).get("kind", ""))
        for request in governance.requests_for_actor(_PRIMARY_MERCHANT_ID)
    )


def _score_merchant_reputation_recovery(
    case: _CaseT8, evidence: RuntimeEvidenceBundleV2
) -> RuntimeTaskScoreV3:
    verified_records: list[VerifiedEvidenceRecordEvidence] = []
    governance = _verified_governance(
        evidence,
        actor_id=None,
        # A rejected premature completion is a scoreable merchant mistake.
        # It must not hide the independently valid resolved case and
        # owner-bound remediation fixture from the environment gate.
        allow_rejected=True,
        verify_evidence_records=True,
        verified_evidence_records_out=verified_records,
    )
    cases = _governance_records_of_type(governance, GovernanceCase)
    resolved = tuple(
        row
        for row in cases
        if row.status == "resolved" and _PRIMARY_MERCHANT_ID in row.subject_merchant_ids
    )
    plans = _governance_records_of_type(governance, RemediationPlan)
    latest_plan = max(plans, key=lambda row: row.version) if plans else None
    actor_operations: tuple[str, ...] = ()
    actor_action_kinds: tuple[str, ...] = ()
    service_operations: tuple[str, ...] = ()
    if governance is not None:
        actor_requests = governance.requests_for_actor(_PRIMARY_MERCHANT_ID)
        actor_operations = tuple(
            operation.operation for request in actor_requests for operation in request.operations
        )
        actor_action_kinds = _evaluated_remediation_action_kinds(governance)
        service_operations = tuple(
            operation.operation
            for request in governance.requests
            if not request.actor_request
            for operation in request.operations
        )
    accepted_remediation_decisions = tuple(
        row
        for row in evidence.platform_decisions
        if row.get("decision") == "accepted"
        and row.get("actor_id") == _PRIMARY_MERCHANT_ID
        and row.get("action_kind")
        in {
            "commerce.accept_remediation_plan",
            "commerce.complete_remediation_step",
        }
    )
    if accepted_remediation_decisions and governance is None:
        raise RuntimeBenchmarkIntegrityError(
            "T8 accepted remediation intent has no exact Platform/World authority graph"
        )

    accepted = actor_action_kinds[:1] == ("commerce.accept_remediation_plan",)
    completed_operation_count = actor_action_kinds.count("commerce.complete_remediation_step")
    completion_prefix_valid = bool(
        accepted
        and completed_operation_count <= case.axis_value
        and actor_action_kinds
        == (
            "commerce.accept_remediation_plan",
            *("commerce.complete_remediation_step" for _ in range(completed_operation_count)),
        )
    )
    model_step_fraction = (
        completed_operation_count / case.axis_value if completion_prefix_valid else 0.0
    )
    verified_steps = (
        tuple(step for step in latest_plan.steps if step.status == "verified")
        if latest_plan is not None
        else ()
    )
    evidence_rows = evidence.final_world.get("tables", {}).get("evidence_records", [])
    audited_step_ids = {
        str(row.get("subject_id"))
        for row in evidence_rows
        if isinstance(row, Mapping)
        and row.get("issuer_id") == _remediation_auditor_id(case)
        and row.get("owner_id") == _PRIMARY_MERCHANT_ID
    }
    verified_step_ids = {step.step_id for step in verified_steps}
    owner_bound_plan = (
        latest_plan is not None
        and latest_plan.owner_merchant_id == _PRIMARY_MERCHANT_ID
        and len(latest_plan.steps) == case.axis_value
        and [step.sequence_no for step in latest_plan.steps] == list(range(1, case.axis_value + 1))
    )
    _require_merchant_world_closure(
        case,
        evidence,
        governance=governance,
        evidence_records=verified_records[0] if len(verified_records) == 1 else None,
    )
    _require_t8_environment_invariant(
        case,
        "world_resolved_violation_case",
        len(resolved) == 1,
    )
    _require_t8_environment_invariant(
        case,
        "owner_bound_remediation_plan",
        owner_bound_plan,
    )
    if accepted:
        expected_audited_count = min(
            case.axis_value,
            completed_operation_count + (completed_operation_count < case.axis_value),
        )
        _require_t8_environment_invariant(
            case,
            "accepted_plan_activation_and_audit",
            latest_plan is not None
            and latest_plan.status in {"active", "completed"}
            and len(audited_step_ids) == expected_audited_count,
        )
    if completed_operation_count:
        _require_t8_environment_invariant(
            case,
            "accepted_step_verification_closure",
            service_operations.count("verify_remediation_step") == completed_operation_count
            and len(verified_step_ids) == completed_operation_count
            and verified_step_ids <= audited_step_ids,
        )
    if completed_operation_count == case.axis_value:
        remediation_events = tuple(
            row
            for row in _governance_records_of_type(governance, ReputationEvent)
            if row.event_kind == "remediation_verified" and row.merchant_id == _PRIMARY_MERCHANT_ID
        )
        _require_t8_environment_invariant(
            case,
            "completed_remediation_reputation_closure",
            latest_plan is not None
            and latest_plan.status == "completed"
            and service_operations.count("apply_governance_reputation") == 1
            and len(remediation_events) == 1,
        )
    checks = renormalize_capability_checks_v2(
        (
            _check(
                "authenticated_plan_acceptance",
                0.12,
                accepted,
                {
                    "model_action_sequence": list(actor_action_kinds),
                    "accepted_actor_operation_sequence": list(actor_operations),
                },
            ),
            _fraction_check(
                "remediation_step_coverage",
                0.24,
                model_step_fraction,
                {
                    "model_action_sequence": list(actor_action_kinds),
                    "accepted_correct_step_intents": (
                        completed_operation_count if completion_prefix_valid else 0
                    ),
                    "expected_step_count": case.axis_value,
                    "environment_rows_are_validity_only": True,
                },
            ),
        )
    )
    issues = () if all(row.credit == 1.0 for row in checks) else ("t8_runtime_contract_incomplete",)
    return score_checks(case.definition, checks, issues=issues)


def _score_prepared_t8(task_id: str, evidence: RuntimeEvidenceBundleV2) -> RuntimeTaskScoreV3:
    case = _case_for_t8(task_id)
    require_runtime_benchmark_integrity_v2(evidence)
    require_frozen_scenario_fixture_v2(
        evidence,
        _prepared_scenario_for_t8(task_id),
        family="T8",
    )
    if case.lane == "buyer_price_quality":
        return _score_buyer_price_quality(case, evidence)
    if case.lane == "buyer_sponsored_ranking":
        return _score_buyer_sponsored_ranking(case, evidence)
    if case.lane == "buyer_fake_reviews":
        return _score_buyer_fake_reviews(case, evidence)
    if case.lane == "buyer_collusion":
        return _score_buyer_collusion(case, evidence)
    if case.lane == "buyer_reputation":
        return _score_buyer_reputation(case, evidence)
    if case.lane == "merchant_sponsorship_disclosure":
        return _score_merchant_sponsorship_disclosure(case, evidence)
    if case.lane in {"merchant_review_integrity", "merchant_anti_collusion"}:
        return _score_merchant_governance_rejection(case, evidence)
    if case.lane == "merchant_reputation_recovery":
        return _score_merchant_reputation_recovery(case, evidence)
    raise _not_ready(task_id)


def _merchant_ids(case: _CaseT8) -> tuple[str, ...]:
    return tuple(dict.fromkeys(product.merchant_id for product in case.products))


def _campaign_channel_factory(
    case: _CaseT8,
    *,
    next_actor_id: str | None,
) -> Any:
    return _T8BusinessChannel(
        case,
        business_lane="campaign",
        next_actor_id=next_actor_id,
    )


def _prepared_runtime_bundle_t8(task_id: str) -> RuntimeTaskBundleV2:
    case = _case_for_t8(task_id)
    scenario = _prepared_scenario_for_t8(task_id)
    semantic_hash = canonical_sha256(
        {
            "content": case.semantic_contract,
            "scenario_state": scenario.initial_state,
            "scenario_oracle": scenario.success_oracle,
        }
    )
    merchant_ids = _merchant_ids(case)
    ideal_channel: Any
    counterpart_channels: dict[str, Any]
    mutations: tuple[RuntimeMutationV2, ...]
    if case.lane in {
        "buyer_price_quality",
        "buyer_sponsored_ranking",
        "buyer_fake_reviews",
        "buyer_collusion",
        "buyer_reputation",
    }:
        ideal_channel = partial(_T8BusinessChannel, case)
        counterpart_channels = {
            merchant_id: _InertT8BusinessChannel for merchant_id in merchant_ids
        }
        if case.lane == "buyer_sponsored_ranking":
            sponsored_owners = tuple(
                dict.fromkeys(product.merchant_id for product in _sponsored_products(case))
            )
            for index, merchant_id in enumerate(sponsored_owners):
                next_actor_id = (
                    sponsored_owners[index + 1] if index + 1 < len(sponsored_owners) else _BUYER_ID
                )
                counterpart_channels[merchant_id] = (
                    lambda merchant_id=merchant_id, next_actor_id=next_actor_id: (
                        _campaign_channel_factory(
                            case,
                            next_actor_id=next_actor_id,
                        )
                    )
                )
            changed_checks = ("sponsorship_neutral_price_quality_choice",)
        elif case.lane == "buyer_fake_reviews":
            changed_checks = ("verified_review_optimal_choice",)
        elif case.lane == "buyer_collusion":
            counterpart_channels[_PRIMARY_MERCHANT_ID] = partial(
                _T8BusinessChannel,
                case,
                business_lane="governance_handoff",
                next_actor_id=_BUYER_ID,
            )
            changed_checks = ("resolved_case_filtered_optimal_choice",)
        elif case.lane == "buyer_reputation":
            changed_checks = ("strongest_reputation_history_choice",)
        else:
            changed_checks = ("price_quality_optimal_choice",)
        mutations = (
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _T8BusinessChannel(case, mutated=True),
                expected_changed_checks=changed_checks,
            ),
        )
    elif case.lane == "merchant_sponsorship_disclosure":
        ideal_channel = partial(
            _T8BusinessChannel,
            case,
            business_lane="campaign",
        )
        counterpart_channels = {
            _BUYER_ID: _InertT8BusinessChannel,
            **{
                merchant_id: _InertT8BusinessChannel
                for merchant_id in merchant_ids
                if merchant_id != _PRIMARY_MERCHANT_ID
            },
        }
        mutations = (
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _T8BusinessChannel(
                    case,
                    business_lane="campaign",
                    mutated=True,
                ),
                expected_changed_checks=(
                    "placement_disclosure_coverage",
                    "campaign_activation",
                ),
            ),
        )
    elif case.lane in {"merchant_review_integrity", "merchant_anti_collusion"}:
        ideal_channel = partial(_T8BusinessChannel, case)
        counterpart_channels = {
            _BUYER_ID: _InertT8BusinessChannel,
            **{
                merchant_id: _InertT8BusinessChannel
                for merchant_id in merchant_ids
                if merchant_id != _PRIMARY_MERCHANT_ID
            },
        }
        mutations = (
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _T8BusinessChannel(
                    case,
                    mutated=True,
                ),
                expected_changed_checks=("policy_safe_rejection",),
            ),
        )
    elif case.lane == "merchant_reputation_recovery":
        ideal_channel = partial(_T8BusinessChannel, case)
        counterpart_channels = {
            _BUYER_ID: _InertT8BusinessChannel,
            **{
                merchant_id: _InertT8BusinessChannel
                for merchant_id in merchant_ids
                if merchant_id != _PRIMARY_MERCHANT_ID
            },
        }
        mutations = (
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _T8BusinessChannel(case, mutated=True),
                expected_changed_checks=("remediation_step_coverage",),
            ),
        )
    else:
        raise _not_ready(task_id)
    return RuntimeTaskBundleV2(
        task=case.definition,
        scenario=scenario,
        evaluated_actor_id=case.evaluated_actor_id,
        ideal_channel=ideal_channel,
        counterpart_channels=counterpart_channels,
        scorer=lambda evidence: _score_prepared_t8(task_id, evidence),
        semantic_hash=semantic_hash,
        mutations=mutations,
    )


def t8_runtime_content(task_id: str) -> dict[str, Any]:
    """Return stable task content without claiming executable governance."""

    return copy.deepcopy(_case_for_t8(task_id).semantic_contract)


def t8_runtime_capability_gap(task_id: str) -> str:
    _case_for_t8(task_id)
    return ""


def _not_ready(task_id: str) -> T8RuntimeCapabilityGap:
    return T8RuntimeCapabilityGap(t8_runtime_capability_gap(task_id))


def scenario_for_t8(task_id: str) -> ScenarioSpec:
    """Build one validated T8 scenario for the real CommerceWorld runtime."""

    return _prepared_scenario_for_t8(task_id)


def runtime_bundle_t8(task_id: str) -> RuntimeTaskBundleV2:
    """Bind one T8 task to Platform, World, evidence, Tracker, and replay."""

    return _prepared_runtime_bundle_t8(task_id)


def targeted_mutation_channel_t8(task_id: str) -> Any:
    [mutation] = runtime_bundle_t8(task_id).mutations
    return mutation.channel()


def runtime_bundles_t8_ready() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t8(task_id) for task_id in T8_RUNTIME_READY_TASK_IDS)


def t8_runtime_semantic_hash(task_id: str) -> str:
    return canonical_sha256(_case_for_t8(task_id).semantic_contract)


__all__ = [
    "T8_RUNTIME_PENDING_TASK_IDS",
    "T8_RUNTIME_READY_TASK_IDS",
    "T8_RUNTIME_SCHEMA_V2",
    "T8RuntimeCapabilityGap",
    "runtime_bundle_t8",
    "runtime_bundles_t8_ready",
    "scenario_for_t8",
    "t8_runtime_capability_gap",
    "t8_runtime_content",
    "t8_runtime_semantic_hash",
    "targeted_mutation_channel_t8",
]
