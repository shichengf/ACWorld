"""Zero-model 2x2/5x5 preflight over the real CommerceWorld Agent path.

This is intentionally separate from the legacy homogeneous/custom-clearing
studies.  Every business write is produced by a Buyer or Merchant Agent from a
typed business decision and crosses Platform before World.  The scenario uses
one shared World, actor-specific channels/memory/authority, a deliberately
contended single-stock listing, and one delivered-order return/refund branch.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from agents.business_decision import LLMBusinessDecisionV1
from episode.capability_materializer import scenario_content_hash_v2
from episode.capability_runtime import RuntimeEvidenceBundleV2, canonical_sha256
from episode.runner import Episode, EpisodeBatch
from episode.scenario import (
    build_secret_registry,
    population_for_scenario,
)
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from evals.serialize import to_canonical
from experiments.environment_study import (
    ARTIFACT_INDEX_NAME,
    DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES,
    ENVIRONMENT_STUDY_ARTIFACT_ORDER,
    ActorCoverageV1,
    EnvironmentStudyContractV1,
    EnvironmentStudyReportV1,
    InvariantResultV1,
    SourceManifestV1,
    build_artifact_index,
    build_environment_study_contract,
    build_source_manifest,
    capture_git_metadata,
    iter_loaded_module_files,
    network_disabled,
    repo_source_paths,
    unlisted_loaded_sources,
    verify_environment_study_contract,
    write_json_artifact,
)
from experiments.environment_smoke import decision_log_artifact
from experiments.scripted_channel import (
    ScriptedChannelFactoryV1,
    ScriptedDecisionContextV1,
)
from protocol.evidence_records import (
    MandateRevisionAuthority,
    build_evidence_record,
    build_mandate_revision,
    evidence_record_to_dict,
    mandate_revision_to_dict,
)
from runtime.after_sales_evidence import (
    AFTER_SALES_EVIDENCE_CONTRACT,
    VerifiedAfterSalesEvidence,
)
from runtime.tracker_evidence import (
    verified_model_business_choices,
    verify_all_active_actor_tracker_evidence,
)
from world.evidence_contracts import mandate_authority_to_wire
from world.store import DatabaseWorld


M2M_PREFLIGHT_SCHEMA = "cwe.multiagent-scripted-preflight.v1"
M2M_MARKET_ID = "market:cwenv-m2m-preflight"
M2M_SEED = 2_026_0721
M2M_STUDY_ID = "CWENV-M2M-PREFLIGHT-01"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_M2M_INVARIANTS = (
    "scenario_content_frozen",
    "two_by_two_smoke",
    "ten_actor_boundary",
    "buyer_business_alternatives",
    "actor_channel_isolation",
    "provider_boundary",
    "zero_provider_calls",
    "tracker_causal_closure",
    "agent_platform_world_lineage",
    "no_direct_settlement",
    "inventory_contention",
    "after_sales_refund_lineage",
    "inprocess_http_semantic_parity",
    "database_reopen_parity",
    "strict_replay",
    "episode_completed_without_termination",
    "execution_sources_declared",
)

_M2M_EVIDENCE_REFS: Mapping[str, tuple[str, ...]] = {
    "scenario_content_frozen": ("contract.json", "world.initial.json"),
    "two_by_two_smoke": ("2x2/world.final.json", "2x2/world.commits.jsonl"),
    "ten_actor_boundary": ("5x5/actor.evidence.jsonl",),
    "buyer_business_alternatives": ("contract.json",),
    "actor_channel_isolation": ("hashes-only-decision-log.json",),
    "provider_boundary": ("hashes-only-decision-log.json", "audit.trace.jsonl"),
    "zero_provider_calls": ("hashes-only-decision-log.json", "network-guard.json"),
    "tracker_causal_closure": ("actor.evidence.jsonl", "audit.trace.jsonl"),
    "agent_platform_world_lineage": (
        "platform.decisions.jsonl",
        "world.commits.jsonl",
    ),
    "no_direct_settlement": ("audit.trace.jsonl", "world.commits.jsonl"),
    "inventory_contention": ("world.initial.json", "world.final.json"),
    "after_sales_refund_lineage": (
        "platform.decisions.jsonl",
        "world.commits.jsonl",
        "world.final.json",
    ),
    "inprocess_http_semantic_parity": ("report.json",),
    "database_reopen_parity": ("report.json",),
    "strict_replay": ("evidence-manifest.json", "world.commits.jsonl"),
    "episode_completed_without_termination": ("termination.json-absent",),
    "execution_sources_declared": ("source-manifest.json", "loaded-module-audit"),
}

_ACTION_TO_WORLD_OPERATION = {
    "commerce.create_cart_quote_request": "create_cart_quote_request",
    "commerce.request_cart_quote": "issue_cart_quote",
    "platform.checkout_cart": "checkout_cart_quote",
    "commerce.search": "create_search_session",
    "commerce.request_return": "request_return",
    "commerce.authorize_return": "authorize_return",
    "commerce.receive_return": "receive_return",
    "commerce.open_refund_case": "open_refund_case",
    "commerce.approve_refund": "approve_refund",
}


def buyer_id(index: int) -> str:
    return f"buyer:cwenv-m2m-b{index}"


def merchant_id(index: int) -> str:
    return f"merchant:cwenv-m2m-m{index}"


def sku_id(index: int) -> str:
    return f"{merchant_id(index)}:sku:preflight-{index}"


def mandate_id(index: int) -> str:
    return f"mandate:cwenv-m2m-b{index}"


def principal_id(index: int) -> str:
    return f"consumer:cwenv-m2m-b{index}"


_AFTER_SALES_ORDER_ID = "order:cwenv-m2m-after-sales"
_AFTER_SALES_EVIDENCE_ID = "evidence:cwenv-m2m-return-inspection"


class MultiAgentPreflightError(RuntimeError):
    """The scripted multi-Agent environment preflight is invalid."""


def _cart_execution_contract() -> dict[str, Any]:
    routes = [
        {
            "action_kind": "commerce.request_cart_quote",
            "destination": "platform:checkout",
        }
    ]
    return {
        "schema_version": "cwe.public-task-execution.v1",
        "phases": [
            {
                "phase_id": "buyer_cart_authorization_ack",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.cart_quote_request"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
            {
                "phase_id": "merchant_cart_request",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.cart_quote_request"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": routes,
                "world_reads": "skill_scoped",
                "finish": "forbid",
            },
            {
                "phase_id": "merchant_cart_quote_complete",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["platform.cart_quote"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
            {
                "phase_id": "buyer_cart_quote",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.cart_quote"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [
                    {
                        "action_kind": "platform.checkout_cart",
                        "destination": "platform:checkout",
                    }
                ],
                "world_reads": "deny",
                "finish": "allow_wait",
            },
            {
                "phase_id": "buyer_cart_settlement",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.cart_settlement"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_terminal",
            },
        ],
    }


def _after_sales_execution_contract() -> dict[str, Any]:
    buyer_routes = [
        {
            "action_kind": "commerce.request_return",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.open_refund_case",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.read_after_sales_policy",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.read_after_sales_history",
            "destination": "platform:after-sales",
        },
    ]
    merchant_routes = [
        {
            "action_kind": "commerce.authorize_return",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.deny_return",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.receive_return",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.approve_refund",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.deny_refund",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.read_after_sales_policy",
            "destination": "platform:after-sales",
        },
        {
            "action_kind": "commerce.read_after_sales_history",
            "destination": "platform:after-sales",
        },
    ]
    phases: list[dict[str, Any]] = [
        {
            "phase_id": "buyer_after_sales_message",
            "match": {
                "actor_roles": ["buyer"],
                "inbound_action_kinds": ["commerce.send_message"],
                "inbound_sender_roles": ["merchant"],
            },
            "allowed_routes": buyer_routes,
            "world_reads": "skill_scoped",
            "finish": "allow_wait",
        },
        {
            "phase_id": "buyer_after_sales_snapshot",
            "match": {
                "actor_roles": ["buyer"],
                "inbound_action_kinds": ["platform.after_sales_snapshot"],
                "inbound_sender_roles": ["platform"],
            },
            "allowed_routes": buyer_routes,
            "world_reads": "skill_scoped",
            "finish": "allow_wait",
        },
        {
            "phase_id": "merchant_after_sales_message",
            "match": {
                "actor_roles": ["merchant"],
                "inbound_action_kinds": ["commerce.send_message"],
                "inbound_sender_roles": ["buyer"],
            },
            "allowed_routes": merchant_routes,
            "world_reads": "skill_scoped",
            "finish": "allow_wait",
        },
        {
            "phase_id": "merchant_after_sales_snapshot",
            "match": {
                "actor_roles": ["merchant"],
                "inbound_action_kinds": ["platform.after_sales_snapshot"],
                "inbound_sender_roles": ["platform"],
            },
            "allowed_routes": merchant_routes,
            "world_reads": "skill_scoped",
            "finish": "allow_wait",
        },
        {
            "phase_id": "merchant_authorize_return_progress",
            "match": {
                "actor_roles": ["merchant"],
                "inbound_action_kinds": ["platform.after_sales_updated"],
                "inbound_sender_roles": ["platform"],
                "payload_equals": {"operation": "authorize_return"},
            },
            "allowed_routes": merchant_routes,
            "world_reads": "skill_scoped",
            "finish": "allow_wait",
        },
    ]
    for phase_id, role, operation, destination in (
        (
            "buyer_request_return_progress",
            "buyer",
            "request_return",
            "@bound_counterparty",
        ),
        (
            "merchant_receive_return_progress",
            "merchant",
            "receive_return",
            "@bound_counterparty",
        ),
        (
            "buyer_open_refund_progress",
            "buyer",
            "open_refund_case",
            "@bound_counterparty",
        ),
    ):
        phases.append(
            {
                "phase_id": phase_id,
                "match": {
                    "actor_roles": [role],
                    "inbound_action_kinds": ["platform.after_sales_updated"],
                    "inbound_sender_roles": ["platform"],
                    "payload_equals": {"operation": operation},
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "framework_continue",
                "framework_continuation": {
                    "action_kind": "commerce.send_message",
                    "destination": destination,
                    "payload": {"category": "after_sales"},
                },
            }
        )
    phases.append(
        {
            "phase_id": "merchant_refund_terminal",
            "match": {
                "actor_roles": ["merchant"],
                "inbound_action_kinds": ["platform.after_sales_updated"],
                "inbound_sender_roles": ["platform"],
                "payload_equals": {"operation": "approve_refund"},
            },
            "allowed_routes": [],
            "world_reads": "deny",
            "finish": "framework_terminal",
        }
    )
    return {"schema_version": "cwe.public-task-execution.v1", "phases": phases}


def _search_execution_contract() -> dict[str, Any]:
    return {
        "schema_version": "cwe.public-task-execution.v1",
        "phases": [
            {
                "phase_id": "buyer_competing_demand_search",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["delegate.create_purchase_mandate"],
                    "inbound_sender_roles": ["consumer"],
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
            {
                "phase_id": "buyer_competing_demand_observed",
                "match": {
                    "actor_roles": ["buyer"],
                    "inbound_action_kinds": ["platform.rank_offers"],
                    "inbound_sender_roles": ["platform"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "allow_wait",
            },
        ],
    }


def _idle_merchant_execution_contract() -> dict[str, Any]:
    return {
        "schema_version": "cwe.public-task-execution.v1",
        "phases": [
            {
                "phase_id": "merchant_public_market_observation",
                "match": {
                    "actor_roles": ["merchant"],
                    "inbound_action_kinds": ["commerce.send_message"],
                    "inbound_sender_roles": ["buyer"],
                },
                "allowed_routes": [],
                "world_reads": "deny",
                "finish": "allow_wait",
            }
        ],
    }


def _pricing_terms(index: int) -> list[dict[str, Any]]:
    return [
        {
            "sku_ids": [sku_id(index)],
            "quantity_tiers": [
                {
                    "minimum_quantity": 1,
                    "maximum_quantity": None,
                    "unit_price_minor": 10_000 + index * 100,
                }
            ],
            "bundle_discounts": [],
            "bundle_stacking": "best_only",
            "charges": [],
        }
    ]


def _pricing_fixture(index: int) -> dict[str, Any]:
    return {
        "merchant_id": merchant_id(index),
        "idempotency_key": f"seed:m2m:pricing:{index}",
        "intent": {
            "market_id": M2M_MARKET_ID,
            "policy_id": f"policy:m2m:pricing:{index}",
            "listing_ids": [sku_id(index)],
            "product_ids": [],
            "quantity_tiers": [
                {
                    "minimum_quantity": 1,
                    "maximum_quantity": None,
                    "unit_price_minor": 10_000 + index * 100,
                }
            ],
            "bundle_discounts": [],
            "bundle_stacking": "best_only",
            "components": [],
            "effective_after_ticks": 0,
            "expires_after_ticks": None,
        },
    }


def _mandate_rows(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = MandateRevisionAuthority(
        principal_id=principal_id(index),
        buyer_id=buyer_id(index),
        mandate_id=mandate_id(index),
        allowed_fields=("budget_minor",),
    )
    revision = build_mandate_revision(
        principal_id=principal_id(index),
        buyer_id=buyer_id(index),
        mandate_id=mandate_id(index),
        revision=1,
        previous_digest=None,
        changes={"budget_minor": 50_000},
        authorized_fields=("budget_minor",),
        logical_tick=0,
    )
    return mandate_authority_to_wire(authority), dict(
        mandate_revision_to_dict(revision)
    )


def _cart_authorization_event(*, buyer_index: int, merchant_index: int, ordinal: int) -> dict[str, Any]:
    return {
        "msg_id": f"kickoff:m2m:cart:{ordinal}",
        "ts": "2026-07-21T12:00:00Z",
        "from": buyer_id(buyer_index),
        "to": "platform:checkout",
        "idempotency_key": f"authorize:m2m:{ordinal}",
        "action": {
            "kind": "commerce.create_cart_quote_request",
            "payload": {
                "market_id": M2M_MARKET_ID,
                "mandate_id": mandate_id(buyer_index),
                "lines": [{"sku_id": sku_id(merchant_index), "qty": 1}],
                "fill_policy": "all_or_none",
                "backorder_policy": "reject",
            },
        },
    }


def _after_sales_initial_state(size: int) -> dict[str, Any]:
    buyer = buyer_id(size)
    merchant = merchant_id(size)
    evidence = build_evidence_record(
        record_id=_AFTER_SALES_EVIDENCE_ID,
        kind="inspection_report",
        subject_id=_AFTER_SALES_ORDER_ID,
        issuer_id="inspector:trusted",
        facts={"condition": "new", "inspection_reference": "inspection:m2m:1"},
        trust={"verified": True, "verification_method": "authenticated_inspection"},
        version=1,
        owner_id=buyer,
        read_acl=(buyer, merchant, "platform:after-sales"),
        issued_at_tick=2,
    )
    return {
        "orders": [
            {
                "order_id": _AFTER_SALES_ORDER_ID,
                "buyer_id": buyer,
                "merchant_id": merchant,
                "sku_id": sku_id(size),
                "qty": 1,
                "agreed_price": f"{100 + size}.00",
                "currency": "USD",
                "state": "dispatched",
            }
        ],
        "ledger": [
            {
                "txn_id": "txn:cwenv-m2m-after-sales-charge",
                "buyer_id": buyer,
                "merchant_id": merchant,
                "sku_id": sku_id(size),
                "qty": 1,
                "price": f"{100 + size}.00",
                "currency": "USD",
                "order_id": _AFTER_SALES_ORDER_ID,
                "idempotency_key": "seed:m2m:after-sales-charge",
                "effect": "charge",
            }
        ],
        "logical_time": 3,
        "evidence_records": [evidence_record_to_dict(evidence)],
        "order_timelines": [
            {
                "order_id": _AFTER_SALES_ORDER_ID,
                "buyer_id": buyer,
                "merchant_id": merchant,
                "settled_at_tick": 0,
                "dispatched_at_tick": 1,
                "return_window_ticks": None,
                "return_authorized_at_tick": None,
                "returned_at_tick": None,
                "refunded_at_tick": None,
            }
        ],
        "shipments": [
            {
                "shipment_id": "shipment:cwenv-m2m-after-sales",
                "order_id": _AFTER_SALES_ORDER_ID,
                "buyer_id": buyer,
                "merchant_id": merchant,
                "original_sku_id": sku_id(size),
                "status": "delivered",
                "status_history": [
                    {
                        "event_id": "shipment-event:cwenv-m2m-delivered",
                        "status": "delivered",
                        "logical_time": 2,
                    }
                ],
                "version": 1,
            }
        ],
        "after_sales_setup": {
            "policies": [
                {
                    "merchant_id": merchant,
                    "idempotency_key": "seed:m2m:after-sales-policy",
                    "intent": {
                        "policy_id": "policy:cwenv-m2m-after-sales",
                        "return_window_ticks": 30,
                        "allowed_return_conditions": ["new", "opened", "damaged"],
                        "max_refund_bps": 10_000,
                        "split_refund_bps": 5_000,
                        "owner_paid_cancel_allowed": True,
                        "merchant_paid_cancel_allowed": True,
                        "return_authorizer_ids": [merchant],
                        "return_receiver_ids": [merchant],
                        "exchange_authorizer_ids": [merchant],
                        "refund_decider_ids": [merchant],
                        "adjudicator_ids": ["platform:adjudicator"],
                        "evidence_service_ids": ["platform:evidence"],
                        "ledger_requester_ids": [buyer, merchant],
                        "ledger_reconciler_ids": ["platform:accounting"],
                    },
                }
            ],
            "payment_transitions": [
                {
                    "idempotency_key": "seed:m2m:payment-capture",
                    "intent": {"op": "capture", "order_id": _AFTER_SALES_ORDER_ID},
                }
            ],
            "packing_transitions": [],
        },
    }


def build_multiagent_scenario(size: int = 5) -> ScenarioSpec:
    """Build the 2x2 smoke or complete 5x5 shared-market scenario."""

    if size not in {2, 5}:
        raise ValueError("multi-Agent preflight supports exactly 2x2 or 5x5")
    cart_context = {
        "schema_version": "cwe.environment-study-m2m-cart.v1",
        "execution_contract": _cart_execution_contract(),
    }
    after_context = {
        "schema_version": "cwe.environment-study-m2m-after-sales.v1",
        "execution_contract": _after_sales_execution_contract(),
    }
    search_context = {
        "schema_version": "cwe.environment-study-m2m-search.v1",
        "execution_contract": _search_execution_contract(),
    }
    idle_merchant_context = {
        "schema_version": "cwe.environment-study-m2m-observer.v1",
        "execution_contract": _idle_merchant_execution_contract(),
    }
    after_authority = {
        "schema_version": "cwe.environment-study-after-sales-authority.v1",
        "order_id": _AFTER_SALES_ORDER_ID,
        "order_ids": [_AFTER_SALES_ORDER_ID],
        "evidence_record_ids": [_AFTER_SALES_EVIDENCE_ID],
        "allowed_return_conditions": ["new", "opened", "damaged"],
    }
    buyers = tuple(
        BuyerSpec(
            buyer_id=buyer_id(index),
            persona={"name": f"Preflight buyer {index}", "role": "buyer"},
            mandate={
                "mandate_id": mandate_id(index),
                "goal": (
                    "Complete the delivered-order return and refund."
                    if index == size
                    else (
                        "Search the shared market for eligible stock, then abstain."
                        if size == 5 and index == 2
                        else "Evaluate the authorized quote and checkout when instructed."
                    )
                ),
                "quantity": 1,
                "return_after_purchase": False,
                "hard_constraints": {
                    "budget": 50_000,
                    "delivery_days": 14,
                    "must_have": [],
                },
                "soft_constraints": [],
                "soft_preferences": {"style": [], "avoid": []},
                "authority": {
                    "can_buy_without_confirmation": True,
                    "must_not_share_with_merchant": ["budget"],
                },
                "intent_expiry": "2099-12-31T00:00:00Z",
                "task_context": copy.deepcopy(
                    after_context
                    if index == size
                    else (search_context if size == 5 and index == 2 else cart_context)
                ),
                **(
                    {"benchmark_contract": copy.deepcopy(after_authority)}
                    if index == size
                    else (
                        {
                            "benchmark_contract": cast(
                                dict[str, Any],
                                {
                                    "schema_version": "cwe.environment-study-search-authority.v1",
                                    "instruction": "Observe eligible stock and abstain.",
                                    "constraints": [
                                        {
                                            "constraint_id": "shipping",
                                            "field": "shipping_days",
                                            "operator": "at_most",
                                            "value": 14,
                                            "description": "delivery within 14 days",
                                        }
                                    ],
                                    "selection_policy": {
                                        "selection_mode": "observation_only",
                                        "hard_constraint_rule": "all",
                                        "candidate_order_semantics": "irrelevant",
                                    },
                                    "optional_filters": [],
                                    "required_search_rounds": 1,
                                },
                            )
                        }
                        if size == 5 and index == 2
                        else {}
                    )
                ),
            },
        )
        for index in range(1, size + 1)
    )
    merchants = tuple(
        MerchantSpec(
            merchant_id=merchant_id(index),
            persona={"name": f"Preflight merchant {index}", "role": "merchant"},
            policy={
                "floor_price": 5_000,
                "margin_target_bps": 1_000,
                "max_negotiation_rounds": 1,
                "refund_policy": "30_day_return",
                "claim_aggressiveness": "neutral",
                "task_context": {
                    **copy.deepcopy(
                        after_context
                        if index == size
                        else (
                            idle_merchant_context
                            if size == 5 and index == 4
                            else cart_context
                        )
                    ),
                    **(
                        {}
                        if index == size or (size == 5 and index == 4)
                        else {"pricing_terms": _pricing_terms(index)}
                    ),
                },
                **(
                    {"benchmark_contract": copy.deepcopy(after_authority)}
                    if index == size
                    else {}
                ),
            },
            catalog_scope=(sku_id(index),),
        )
        for index in range(1, size + 1)
    )
    catalog = [
        {
            "sku_id": sku_id(index),
            "merchant_id": merchant_id(index),
            "product_id": f"product:cwenv-m2m-{index}",
            "name": f"Preflight product {index}",
            "category": "environment-study",
            "list_price": f"{100 + index}.00",
            "inventory": 1 if index == 1 else 3,
            "qty_reserved": 1 if index == size else 0,
            "attributes": {
                "in_stock": True,
                "shipping_days": 3,
                "returnable": True,
                "candidate_group": "preflight-general",
            },
        }
        for index in range(1, size + 1)
    ]
    authorities: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    for index in range(1, size):
        authority, revision = _mandate_rows(index)
        authorities.append(authority)
        revisions.append(revision)
    after = _after_sales_initial_state(size)
    initial_state = {
        "catalog": catalog,
        "pricing_policy_fixtures": [
            _pricing_fixture(index) for index in range(1, size)
        ],
        "mandate_authorities": authorities,
        "mandate_revisions": revisions,
        **after,
    }
    cart_events: list[dict[str, Any]] = []
    if size == 2:
        cart_events.append(_cart_authorization_event(buyer_index=1, merchant_index=1, ordinal=1))
    else:
        # b1 checks out the single-stock listing while b2 independently searches
        # the same shared market and then abstains.  This represents competing
        # demand without issuing a deliberately invalid oversell checkout.
        for ordinal, (b_index, m_index) in enumerate(
            ((1, 1), (3, 2), (4, 3)), start=1
        ):
            cart_events.append(
                _cart_authorization_event(
                    buyer_index=b_index,
                    merchant_index=m_index,
                    ordinal=ordinal,
                )
            )
        cart_events.extend(
            [
                {
                    "msg_id": "kickoff:m2m:competing-search",
                    "ts": "2026-07-21T12:00:00Z",
                    "from": principal_id(2),
                    "to": buyer_id(2),
                    "idempotency_key": "kickoff:m2m:competing-search",
                    "action": {
                        "kind": "delegate.create_purchase_mandate",
                        "payload": copy.deepcopy(buyers[1].mandate),
                    },
                },
                {
                    "msg_id": "kickoff:m2m:merchant-observer",
                    "ts": "2026-07-21T12:00:00Z",
                    "from": "buyer:external-observer",
                    "to": merchant_id(4),
                    "idempotency_key": "kickoff:m2m:merchant-observer",
                    "action": {
                        "kind": "commerce.send_message",
                        "payload": {
                            "category": "general",
                            "instruction": "Acknowledge the shared market observation only.",
                        },
                    },
                },
            ]
        )
    after_event = {
        "msg_id": "kickoff:m2m:after-sales",
        "ts": "2026-07-21T12:00:00Z",
        "from": merchant_id(size),
        "to": buyer_id(size),
        "idempotency_key": "kickoff:m2m:after-sales",
        "action": {
            "kind": "commerce.send_message",
            "payload": {
                "category": "after_sales",
                "order_id": _AFTER_SALES_ORDER_ID,
                "instruction": "Return the delivered item and complete the refund.",
                "evidence_refs": [_AFTER_SALES_EVIDENCE_ID],
            },
        },
    }
    return ScenarioSpec(
        scenario_id=f"cwenv_m2m_preflight_{size}x{size}",
        seed=M2M_SEED + size,
        initial_state=initial_state,
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(),
        success_oracle={"schema_version": "cwe.environment-study-no-score.v1"},
        population=PopulationSpec(
            buyers=buyers,
            merchants=merchants,
            initial_events=tuple([*cart_events, after_event]),
            matching={"top_k": size},
            execution={"max_transactions_per_buyer": 1},
        ),
    )


def _walk(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for child in value.values():
            rows.extend(_walk(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            rows.extend(_walk(child))
    return tuple(rows)


def _intent_schema(context: ScriptedDecisionContextV1, name: str) -> Mapping[str, Any]:
    value = context.allowed_intent(name)["parameters"]
    if not isinstance(value, Mapping):
        raise MultiAgentPreflightError(f"{name} has no public parameter schema")
    return cast(Mapping[str, Any], value)


def _enum_ref(context: ScriptedDecisionContextV1, intent: str, field: str) -> str:
    schema = _intent_schema(context, intent)
    properties = schema.get("properties")
    field_schema = properties.get(field) if isinstance(properties, Mapping) else None
    values = field_schema.get("enum") if isinstance(field_schema, Mapping) else None
    if not isinstance(values, tuple) or len(values) != 1 or not isinstance(values[0], str):
        raise MultiAgentPreflightError(f"{intent}.{field} has no unique public ref")
    return values[0]


class CartMerchantPolicyV1:
    """Quote only from request lines and public pricing terms."""

    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        names = context.allowed_intent_names
        request_lines: list[tuple[str, int]] = []
        terms: list[Mapping[str, Any]] = []
        for row in _walk(context.observations):
            raw_lines = row.get("lines")
            if isinstance(raw_lines, tuple):
                for line in raw_lines:
                    if isinstance(line, Mapping) and isinstance(line.get("sku_ref"), str):
                        qty = line.get("qty")
                        if isinstance(qty, int) and not isinstance(qty, bool) and qty > 0:
                            request_lines.append((str(line["sku_ref"]), qty))
            raw_terms = row.get("pricing_terms")
            if isinstance(raw_terms, tuple):
                terms.extend(item for item in raw_terms if isinstance(item, Mapping))
        request_lines = list(dict.fromkeys(request_lines))
        if "observe_listing" in names and "request_cart_quote" not in names:
            ref = _enum_ref(context, "observe_listing", "sku_ref")
            return LLMBusinessDecisionV1(
                intent="observe_listing", arguments={"sku_ref": ref}
            )
        if "request_cart_quote" not in names or not request_lines:
            return LLMBusinessDecisionV1(
                intent="finish", arguments={"reason": "No current quote is available."}
            )
        line_quotes: list[dict[str, Any]] = []
        for ref, qty in request_lines:
            matching = [
                term
                for term in terms
                if isinstance(term.get("sku_refs"), tuple) and ref in term["sku_refs"]
            ]
            if len(matching) != 1:
                raise MultiAgentPreflightError("public pricing term is not unique")
            tiers = matching[0].get("quantity_tiers")
            if not isinstance(tiers, tuple):
                raise MultiAgentPreflightError("public pricing tiers are missing")
            applicable = [
                tier
                for tier in tiers
                if isinstance(tier, Mapping)
                and isinstance(tier.get("minimum_quantity"), int)
                and qty >= int(tier["minimum_quantity"])
                and (
                    tier.get("maximum_quantity") is None
                    or qty <= int(tier["maximum_quantity"])
                )
            ]
            if len(applicable) != 1:
                raise MultiAgentPreflightError("public quantity tier is not unique")
            unit = int(applicable[0]["unit_price_minor"])
            line_quotes.append(
                {
                    "sku_ref": ref,
                    "qty": qty,
                    "unit_price_minor": unit,
                    "line_total_minor": unit * qty,
                    "applied_rule_kinds": ["quantity_tier"],
                }
            )
        subtotal = sum(int(row["line_total_minor"]) for row in line_quotes)
        return LLMBusinessDecisionV1(
            intent="request_cart_quote",
            arguments={
                "line_quotes": line_quotes,
                "charges": [],
                "subtotal_minor": subtotal,
                "grand_total_minor": subtotal,
            },
        )


class CartBuyerPolicyV1:
    def __init__(self, *, checkout: bool) -> None:
        self._checkout = checkout

    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        if "checkout_cart" in context.allowed_intent_names and self._checkout:
            return LLMBusinessDecisionV1(intent="checkout_cart", arguments={})
        return LLMBusinessDecisionV1(
            intent="finish",
            arguments={"reason": "Decline this quote to avoid overselling contested stock."},
        )


class SearchObserverBuyerPolicyV1:
    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        if "search" in context.allowed_intent_names:
            return LLMBusinessDecisionV1(
                intent="search", arguments={"query": "", "filters": {}}
            )
        return LLMBusinessDecisionV1(
            intent="finish",
            arguments={"reason": "Observed competing stock; no purchase is required."},
        )


class IdleMerchantPolicyV1:
    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        del context
        return LLMBusinessDecisionV1(
            intent="finish",
            arguments={"reason": "No merchant action is required for this observation."},
        )


class AfterSalesBuyerPolicyV1:
    def __init__(self) -> None:
        self._step = 0

    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        if self._step == 0 and "request_return" in context.allowed_intent_names:
            self._step += 1
            evidence_refs: list[str] = []
            schema = _intent_schema(context, "request_return")
            properties = schema.get("properties")
            evidence = properties.get("evidence_refs") if isinstance(properties, Mapping) else None
            items = evidence.get("items") if isinstance(evidence, Mapping) else None
            values = items.get("enum") if isinstance(items, Mapping) else None
            if isinstance(values, tuple):
                evidence_refs = [str(value) for value in values]
            arguments: dict[str, Any] = {
                "requested_qty": 1,
                "reason": "Request a policy-covered return.",
            }
            if evidence_refs:
                arguments["evidence_refs"] = evidence_refs
            return LLMBusinessDecisionV1(intent="request_return", arguments=arguments)
        if self._step == 1 and "open_refund_case" in context.allowed_intent_names:
            self._step += 1
            return LLMBusinessDecisionV1(
                intent="open_refund_case",
                arguments={"reason": "The authorized return was received."},
            )
        return LLMBusinessDecisionV1(intent="finish", arguments={"reason": "Done."})


class AfterSalesMerchantPolicyV1:
    def __init__(self) -> None:
        self._step = 0

    def __call__(self, context: ScriptedDecisionContextV1) -> LLMBusinessDecisionV1:
        choices = (
            (
                "decide_return",
                {"decision": "approve", "reason": "The return is policy-covered."},
            ),
            ("receive_return", {"received_qty": 1, "condition": "new"}),
            (
                "decide_refund",
                {"decision": "approve", "reason": "The return was received."},
            ),
        )
        if self._step < len(choices):
            intent, arguments = choices[self._step]
            if intent in context.allowed_intent_names:
                self._step += 1
                return LLMBusinessDecisionV1(intent=intent, arguments=arguments)
        return LLMBusinessDecisionV1(intent="finish", arguments={"reason": "Done."})


def make_multiagent_factory(size: int = 5) -> ScriptedChannelFactoryV1:
    actor_policies: dict[str, Any] = {}
    for index in range(1, size + 1):
        if index == size:
            actor_policies[buyer_id(index)] = AfterSalesBuyerPolicyV1()
            actor_policies[merchant_id(index)] = AfterSalesMerchantPolicyV1()
        else:
            actor_policies[buyer_id(index)] = (
                SearchObserverBuyerPolicyV1()
                if size == 5 and index == 2
                else CartBuyerPolicyV1(checkout=True)
            )
            actor_policies[merchant_id(index)] = (
                IdleMerchantPolicyV1()
                if size == 5 and index == 4
                else CartMerchantPolicyV1()
            )
    return ScriptedChannelFactoryV1(policies={}, actor_policies=actor_policies)


@dataclass(frozen=True, slots=True)
class MultiAgentRunV1:
    scenario: ScenarioSpec
    evidence: RuntimeEvidenceBundleV2
    factory: ScriptedChannelFactoryV1
    transport: str
    database_path: str | None = None
    reopened_digest: str | None = None


def _snapshot_digest(value: Any) -> str:
    return str(canonical_sha256(to_canonical(value)))


def run_multiagent_episode(
    *,
    scenario: ScenarioSpec,
    factory: ScriptedChannelFactoryV1,
    out_root: str | Path,
    transport: str = "in_process",
    database_path: str | Path | None = None,
    world_mode: Literal["E0", "E1"] = "E0",
    reset_before_run: bool = False,
) -> MultiAgentRunV1:
    """Run one scenario through Agent, optionally on a persistent SQLite World.

    ``world_mode`` and ``reset_before_run`` are used by bounded persistence
    studies that reopen one SQLite database between Episodes.  Defaults retain
    the original isolated-E0 behavior.  A reset is deliberately unavailable
    for the in-memory/HTTP branches so callers cannot claim cross-process E1
    persistence without a durable backend.
    """

    if reset_before_run and database_path is None:
        raise ValueError("reset_before_run requires a persistent database_path")
    if reset_before_run and world_mode != "E1":
        raise ValueError("reset_before_run is only valid for E1 persistence")

    if transport == "http_vcp":
        from episode.http_launcher import run_http_episode

        episode_dir = Path(out_root) / scenario.scenario_id
        with factory.claim_for_episode(scenario.scenario_id):
            run_http_episode(
                scenario=scenario,
                channels=factory,
                out_dir=episode_dir,
                strict_skill_selection=True,
            )
        return MultiAgentRunV1(
            scenario=scenario,
            evidence=RuntimeEvidenceBundleV2.load(episode_dir),
            factory=factory,
            transport=transport,
        )
    if database_path is None:
        with factory.claim_for_episode(scenario.scenario_id):
            EpisodeBatch(
                scenarios=[scenario],
                channels=factory,
                out_root=out_root,
                strict_skill_selection=True,
                strict_tracker_capture=True,
            ).run()
        episode_dir = Path(out_root) / scenario.scenario_id
        return MultiAgentRunV1(
            scenario=scenario,
            evidence=RuntimeEvidenceBundleV2.load(episode_dir),
            factory=factory,
            transport=transport,
        )

    from agents.platform import PlatformService
    from episode.actor_evidence import build_actor_evidence_services
    from episode.artifacts import prepare_out_dir
    from runtime import AuditLog, Router, Runtime

    db_path = Path(database_path)
    episode_dir = prepare_out_dir(Path(out_root) / scenario.scenario_id)
    world = DatabaseWorld(db_path, mode=world_mode)
    if reset_before_run:
        world.reset("E1")
    audit = AuditLog(episode_dir / "audit.jsonl", run_id=scenario.scenario_id)
    actor_context_resolver, actor_evidence_journal = build_actor_evidence_services(
        scenario, out_dir=episode_dir
    )
    population = population_for_scenario(scenario)
    platform = PlatformService(
        world=cast(Any, world),
        audit=audit,
        platform_policy=scenario.platform_policy,
        max_search_results=int(population.matching["top_k"]),
        max_transactions_per_buyer=int(
            population.execution["max_transactions_per_buyer"]
        ),
        secrets=build_secret_registry(scenario),
        negotiation_participants=frozenset(
            [buyer.buyer_id for buyer in population.buyers]
            + [merchant.merchant_id for merchant in population.merchants]
        ),
    )
    runtime = Runtime(
        world=cast(Any, world),
        router=Router(build_secret_registry(scenario)),
        audit=audit,
        platform=platform,
        actor_context_resolver=actor_context_resolver,
        actor_evidence_journal=actor_evidence_journal,
        strict_tracker_capture=True,
    )
    try:
        with factory.claim_for_episode(scenario.scenario_id):
            Episode(
                scenario=scenario,
                world=cast(Any, world),
                runtime=runtime,
                audit=audit,
                channels=factory,
                out_dir=episode_dir,
                mode=world_mode,
                strict_skill_selection=True,
            ).run()
    finally:
        world.close()
    reopened = DatabaseWorld(db_path)
    try:
        reopened_digest = _snapshot_digest(reopened.snapshot())
    finally:
        reopened.close()
    return MultiAgentRunV1(
        scenario=scenario,
        evidence=RuntimeEvidenceBundleV2.load(episode_dir),
        factory=factory,
        transport="database_world",
        database_path=str(db_path),
        reopened_digest=reopened_digest,
    )


def _multiagent_scenario_digest() -> str:
    return str(
        canonical_sha256(
            {
                f"{size}x{size}": scenario_content_hash_v2(
                    build_multiagent_scenario(size)
                )
                for size in (2, 5)
            }
        )
    )


def _policy_source_for_actor(actor_id: str, *, size: int) -> str:
    policy_sources = {
        "buyer_cart": inspect.getsource(CartBuyerPolicyV1),
        "merchant_cart": inspect.getsource(CartMerchantPolicyV1),
        "buyer_after_sales": inspect.getsource(AfterSalesBuyerPolicyV1),
        "merchant_after_sales": inspect.getsource(AfterSalesMerchantPolicyV1),
        "buyer_search": inspect.getsource(SearchObserverBuyerPolicyV1),
        "merchant_idle": inspect.getsource(IdleMerchantPolicyV1),
    }
    if actor_id == buyer_id(size):
        return policy_sources["buyer_after_sales"]
    if actor_id == merchant_id(size):
        return policy_sources["merchant_after_sales"]
    if size == 5 and actor_id == buyer_id(2):
        return policy_sources["buyer_search"]
    if size == 5 and actor_id == merchant_id(4):
        return policy_sources["merchant_idle"]
    if actor_id.startswith("buyer:"):
        return policy_sources["buyer_cart"]
    return policy_sources["merchant_cart"]


def build_multiagent_contract(
    *, repo_root: str | Path = _REPO_ROOT
) -> tuple[EnvironmentStudyContractV1, SourceManifestV1]:
    """Freeze the complete 2x2/5x5, three-transport scripted preflight."""

    manifest = build_source_manifest(
        repo_root, DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES
    )
    actor_ids = tuple(
        [buyer_id(index) for index in range(1, 6)]
        + [merchant_id(index) for index in range(1, 6)]
    )
    digests = tuple(
        (
            actor_id,
            hashlib.sha256(
                _policy_source_for_actor(actor_id, size=5).encode()
            ).hexdigest(),
        )
        for actor_id in sorted(actor_ids)
    )
    contract = build_environment_study_contract(
        git=capture_git_metadata(repo_root),
        scenario_digest=_multiagent_scenario_digest(),
        actor_policy_digests=digests,
        component_digests=manifest.component_digests(),
        data_provenance_digest=str(
            canonical_sha256(
                {
                    "schema_version": M2M_PREFLIGHT_SCHEMA,
                    "catalog_kind": "synthetic_shared_market",
                    "cross_merchant_product_linkage": False,
                    "sizes": [2, 5],
                }
            )
        ),
        backend="shared_commerceworld",
        transport="in_process+http_vcp+database_world",
        seed=M2M_SEED,
        invariants=_M2M_INVARIANTS,
    )
    verify_environment_study_contract(contract, manifest, repo_root)
    return contract, manifest


@dataclass(frozen=True, slots=True)
class MultiAgentPreflightRunsV1:
    smoke_2x2: MultiAgentRunV1
    inprocess_5x5: MultiAgentRunV1
    http_5x5: MultiAgentRunV1
    database_5x5: MultiAgentRunV1


@dataclass(frozen=True, slots=True)
class PersistedMultiAgentPreflightV1:
    report: EnvironmentStudyReportV1
    artifacts_dir: str
    artifact_index: Mapping[str, Any]
    network_guard: Mapping[str, Any]


def _tables(evidence: RuntimeEvidenceBundleV2, *, final: bool = True) -> Mapping[str, Any]:
    snapshot = evidence.final_world if final else evidence.initial_world
    tables = snapshot.get("tables")
    if not isinstance(tables, Mapping):
        raise MultiAgentPreflightError("World snapshot has no table object")
    return tables


def _normalized_search_semantics(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise MultiAgentPreflightError("search_sessions must be an array")
    sessions: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise MultiAgentPreflightError("search session row is invalid")
        issued = value.get("issued_at_tick")
        expires = value.get("expires_at_tick")
        if (
            isinstance(issued, bool)
            or not isinstance(issued, int)
            or isinstance(expires, bool)
            or not isinstance(expires, int)
            or expires <= issued
        ):
            raise MultiAgentPreflightError("search session lifetime is invalid")
        offers: list[dict[str, Any]] = []
        raw_offers = value.get("offers")
        if not isinstance(raw_offers, list):
            raise MultiAgentPreflightError("search session offers are invalid")
        for offer in raw_offers:
            if not isinstance(offer, Mapping):
                raise MultiAgentPreflightError("search offer is invalid")
            offers.append(
                {
                    str(key): copy.deepcopy(item)
                    for key, item in offer.items()
                    if key
                    not in {
                        "issued_at_tick",
                        "expires_at_tick",
                        "offer_digest",
                    }
                }
            )
        sessions.append(
            {
                str(key): copy.deepcopy(item)
                for key, item in value.items()
                if key
                not in {
                    "issued_at_tick",
                    "expires_at_tick",
                    "offers",
                    "session_digest",
                }
            }
            | {"lifetime_ticks": expires - issued, "offers": offers}
        )
    return sorted(sessions, key=lambda row: str(row.get("session_id")))


def _transport_invariant_tables(evidence: RuntimeEvidenceBundleV2) -> dict[str, Any]:
    tables = copy.deepcopy(dict(_tables(evidence)))
    tables["search_sessions"] = _normalized_search_semantics(
        tables.get("search_sessions", [])
    )
    return tables


def _decision_hashes(factory: ScriptedChannelFactoryV1) -> dict[str, tuple[str, ...]]:
    return {
        actor_id: tuple(
            row.decision_sha256
            for row in factory.channel_for(actor_id).decision_log
        )
        for actor_id in factory.actor_ids()
    }


def _world_operation_semantics(
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[tuple[str, str | None, str | None, str], ...]:
    return tuple(
        sorted(
            (
                str(row.get("operation")),
                (
                    None
                    if row.get("idempotency_key") is None
                    else str(row.get("idempotency_key"))
                ),
                None if row.get("actor_id") is None else str(row.get("actor_id")),
                str(row.get("subject_id")),
            )
            for row in evidence.world_events
        )
    )


def verify_multiagent_transport_parity(
    in_process: MultiAgentRunV1, http: MultiAgentRunV1
) -> tuple[bool, tuple[str, ...]]:
    """Compare transport-invariant semantics, not transport-specific clock ticks."""

    mismatches: list[str] = []
    if in_process.transport == http.transport:
        mismatches.append("transport_identity")
    if _transport_invariant_tables(in_process.evidence) != _transport_invariant_tables(
        http.evidence
    ):
        mismatches.append("world_tables")
    if _decision_hashes(in_process.factory) != _decision_hashes(http.factory):
        mismatches.append("business_decisions")
    if _world_operation_semantics(in_process.evidence) != _world_operation_semantics(
        http.evidence
    ):
        mismatches.append("world_operations")
    return not mismatches, tuple(mismatches)


def _tracker_verdict(
    run: MultiAgentRunV1,
) -> Any:
    actor_ids = run.factory.actor_ids()
    try:
        verdict = verify_all_active_actor_tracker_evidence(
            run.evidence,
            declared_actor_ids=actor_ids,
            evaluated_actor_id=actor_ids[0],
            evaluated_actor_strict=True,
        )
    except Exception:  # noqa: BLE001 - verifier failure invalidates the study
        return None
    return verdict if verdict.verified else None


def _causal_commit_links(
    evidence: RuntimeEvidenceBundleV2,
) -> dict[str, tuple[str, ...]]:
    accepted = evidence.accepted_platform_exchanges()
    by_actor: dict[str, list[str]] = {}
    claimed: set[str] = set()
    for commit in evidence.world_events:
        operation = commit.get("operation")
        idempotency_key = commit.get("idempotency_key")
        commit_id = commit.get("commit_id")
        matches = [
            row
            for row in accepted
            if row.decision.get("idempotency_key") == idempotency_key
            and _ACTION_TO_WORLD_OPERATION.get(
                str(row.decision.get("action_kind"))
            )
            == operation
        ]
        if (
            len(matches) != 1
            or not isinstance(commit_id, str)
            or not commit_id
            or commit_id in claimed
        ):
            raise MultiAgentPreflightError(
                "World commit does not exact-join to one accepted Platform operation"
            )
        claimed.add(commit_id)
        actor_id = matches[0].decision.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id.startswith(
            ("buyer:", "merchant:")
        ):
            raise MultiAgentPreflightError("causal Platform actor id is invalid")
        by_actor.setdefault(actor_id, []).append(commit_id)
    if len(claimed) != len(evidence.world_events):
        raise MultiAgentPreflightError("World commit lineage is incomplete")
    return {actor: tuple(values) for actor, values in sorted(by_actor.items())}


def _actor_coverage(
    run: MultiAgentRunV1,
    tracker: Any,
    commit_links: Mapping[str, tuple[str, ...]],
) -> tuple[ActorCoverageV1, ...]:
    verdicts = (
        {row.evaluated_actor_id: row for row in tracker.actor_verdicts}
        if tracker is not None
        else {}
    )
    rows: list[ActorCoverageV1] = []
    for actor_id in run.factory.actor_ids():
        channel = run.factory.channel_for(actor_id)
        verdict = verdicts.get(actor_id)
        choices = (
            verified_model_business_choices(
                run.evidence,
                evaluated_actor_id=actor_id,
                strict_ideal=False,
            )
            if verdict is not None
            else ()
        )
        rows.append(
            ActorCoverageV1(
                actor_id=actor_id,
                role=channel.role,
                scripted_business_decisions=len(channel.decision_log),
                verified_business_decisions=len(choices),
                complete_tracker_records=(
                    int(verdict.complete_record_count) if verdict is not None else 0
                ),
                accepted_platform_operations=len(
                    run.evidence.accepted_platform_exchanges(actor_id=actor_id)
                ),
                causally_linked_world_commits=len(commit_links.get(actor_id, ())),
            )
        )
    return tuple(rows)


def _buyer_business_alternatives(scenario: ScenarioSpec) -> bool:
    population = population_for_scenario(scenario)
    for buyer in population.buyers:
        context = buyer.mandate.get("task_context")
        execution = context.get("execution_contract") if isinstance(context, Mapping) else None
        phases = execution.get("phases") if isinstance(execution, Mapping) else None
        if not isinstance(phases, list):
            return False
        choices: set[str] = set()
        for phase in phases:
            if not isinstance(phase, Mapping):
                return False
            match = phase.get("match")
            roles = match.get("actor_roles") if isinstance(match, Mapping) else None
            if not isinstance(roles, list) or "buyer" not in roles:
                continue
            routes = phase.get("allowed_routes")
            if not isinstance(routes, list):
                return False
            choices.update(
                str(route.get("action_kind"))
                for route in routes
                if isinstance(route, Mapping)
                and isinstance(route.get("action_kind"), str)
            )
            if phase.get("finish") == "allow_wait":
                choices.add("agent.finish")
        if len(choices) < 2:
            return False
    return True


def _after_sales_evidence(evidence: RuntimeEvidenceBundleV2) -> VerifiedAfterSalesEvidence | None:
    try:
        candidate = evidence.verified_operation_evidence(
            AFTER_SALES_EVIDENCE_CONTRACT,
            options={
                "expected_order_id": _AFTER_SALES_ORDER_ID,
                "expected_operations": [
                    "request_return",
                    "authorize_return",
                    "receive_return",
                    "open_refund_case",
                    "approve_refund",
                ],
                "allow_rejected": False,
            },
        )
    except Exception:  # noqa: BLE001 - exact evidence is a validity gate
        return None
    return candidate if isinstance(candidate, VerifiedAfterSalesEvidence) else None


def _replay_result(run: MultiAgentRunV1) -> Mapping[str, Any]:
    from episode.replay import verify_episode_replay

    payload = verify_episode_replay(
        run.evidence.episode_dir, strict=True
    ).to_dict()
    # A persisted report is repository evidence, not a host-local run log.
    # Keep the verified replay facts while excluding the absolute temp path.
    payload.pop("target", None)
    payload["transport"] = run.transport
    return payload


def _combined_decision_log(runs: MultiAgentPreflightRunsV1) -> dict[str, Any]:
    payload: list[dict[str, Any]] = []
    for name, run in (
        ("smoke_2x2", runs.smoke_2x2),
        ("inprocess_5x5", runs.inprocess_5x5),
        ("http_5x5", runs.http_5x5),
        ("database_5x5", runs.database_5x5),
    ):
        row = decision_log_artifact(run.factory)
        payload.append({"run_name": name, "records": row["records"]})
    return {
        "schema_version": "cwe.multiagent-scripted-decision-log.v1",
        "provider_calls": 0,
        "runs": payload,
    }


def build_multiagent_report(
    *,
    contract: EnvironmentStudyContractV1,
    manifest: SourceManifestV1,
    runs: MultiAgentPreflightRunsV1,
    loaded_sources: Sequence[str],
) -> EnvironmentStudyReportV1:
    replay = {
        "smoke_2x2": _replay_result(runs.smoke_2x2),
        "inprocess_5x5": _replay_result(runs.inprocess_5x5),
        "http_5x5": _replay_result(runs.http_5x5),
        "database_5x5": _replay_result(runs.database_5x5),
    }
    main = runs.inprocess_5x5
    smoke = runs.smoke_2x2
    tracker = _tracker_verdict(main)
    try:
        commit_links = _causal_commit_links(main.evidence)
    except MultiAgentPreflightError:
        commit_links = {}
    coverage = _actor_coverage(main, tracker, commit_links)
    parity_ok, parity_mismatches = verify_multiagent_transport_parity(
        main, runs.http_5x5
    )
    after_sales = _after_sales_evidence(main.evidence)
    main_initial = _tables(main.evidence, final=False)
    main_final = _tables(main.evidence)
    inventory_before = main_initial.get("inventory")
    inventory_after = main_final.get("inventory")
    contested_sku = sku_id(1)
    before_contested = (
        inventory_before.get(contested_sku)
        if isinstance(inventory_before, Mapping)
        else None
    )
    after_contested = (
        inventory_after.get(contested_sku)
        if isinstance(inventory_after, Mapping)
        else None
    )
    raw_sessions = main_final.get("search_sessions")
    sessions = raw_sessions if isinstance(raw_sessions, list) else []
    searched_skus = {
        str(offer.get("sku_id"))
        for session in sessions
        if isinstance(session, Mapping)
        for offer in session.get("offers", [])
        if isinstance(offer, Mapping)
    }
    orders = [
        row for row in main_final.get("orders", []) if isinstance(row, Mapping)
    ]
    ledger = [
        row for row in main_final.get("ledger", []) if isinstance(row, Mapping)
    ]
    order_groups = [
        row
        for row in main_final.get("order_groups", [])
        if isinstance(row, Mapping)
    ]
    final_after_order = next(
        (row for row in orders if row.get("order_id") == _AFTER_SALES_ORDER_ID),
        None,
    )
    refund_rows = [row for row in ledger if row.get("effect") == "refund"]
    no_oversell = bool(
        isinstance(inventory_after, Mapping)
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("qty_available"), int)
            and isinstance(row.get("qty_reserved"), int)
            and 0 <= int(row["qty_reserved"]) <= int(row["qty_available"])
            for row in inventory_after.values()
        )
    )
    checkout_choices = sum(
        choice.intent == "checkout_cart"
        for actor in main.factory.actor_ids()
        for choice in verified_model_business_choices(
            main.evidence,
            evaluated_actor_id=actor,
            strict_ideal=False,
        )
    )
    checkout_commits = sum(
        row.get("operation") == "checkout_cart_quote"
        for row in main.evidence.world_events
    )
    unlisted = unlisted_loaded_sources(manifest, loaded_sources)
    all_runs = (
        smoke,
        main,
        runs.http_5x5,
        runs.database_5x5,
    )
    checks = {
        "scenario_content_frozen": (
            contract.scenario_digest == _multiagent_scenario_digest()
            and contract.seed == M2M_SEED
        ),
        "two_by_two_smoke": (
            len(smoke.factory.actor_ids()) == 4
            and smoke.factory.provider_calls == 0
            and bool(smoke.evidence.world_events)
        ),
        "ten_actor_boundary": (
            len(main.factory.actor_ids()) == 10
            and len({id(main.factory.channel_for(actor)) for actor in main.factory.actor_ids()})
            == 10
        ),
        "buyer_business_alternatives": _buyer_business_alternatives(main.scenario),
        "actor_channel_isolation": all(
            run.factory.lifecycle_state == "sealed"
            and len(run.factory.actor_ids()) == (4 if run is smoke else 10)
            for run in all_runs
        ),
        "provider_boundary": all(
            run.factory.actor_ids()
            and all(
                run.factory.channel_for(actor).decision_log
                for actor in run.factory.actor_ids()
            )
            for run in all_runs
        ),
        "zero_provider_calls": all(run.factory.provider_calls == 0 for run in all_runs),
        "tracker_causal_closure": tracker is not None
        and all(row.complete_tracker_records > 0 for row in coverage),
        "agent_platform_world_lineage": len(commit_links) > 0
        and sum(len(values) for values in commit_links.values())
        == len(main.evidence.world_events),
        "no_direct_settlement": checkout_choices == checkout_commits == 3,
        "inventory_contention": bool(
            isinstance(before_contested, Mapping)
            and isinstance(after_contested, Mapping)
            and before_contested.get("qty_available") == 1
            and before_contested.get("qty_reserved") == 0
            and after_contested.get("qty_reserved") == 1
            and contested_sku in searched_skus
            and no_oversell
        ),
        "after_sales_refund_lineage": bool(
            after_sales is not None
            and len(after_sales.requests) == 5
            and isinstance(final_after_order, Mapping)
            and final_after_order.get("state") == "refunded"
            and len(refund_rows) == 1
        ),
        "inprocess_http_semantic_parity": parity_ok,
        "database_reopen_parity": (
            runs.database_5x5.reopened_digest
            == runs.database_5x5.evidence.final_digest
        ),
        "strict_replay": all(
            bool(result.get("replay_ok")) for result in replay.values()
        ),
        "episode_completed_without_termination": all(
            not (run.evidence.episode_dir / "termination.json").exists()
            for run in all_runs
        ),
        "execution_sources_declared": not unlisted,
    }
    invariants = tuple(
        InvariantResultV1(
            name=name,
            passed=bool(checks[name]),
            evidence_refs=_M2M_EVIDENCE_REFS[name],
            failure_reason=(
                None
                if checks[name]
                else (
                    "undeclared loaded source: " + ", ".join(unlisted)
                    if name == "execution_sources_declared" and unlisted
                    else (
                        "transport semantic mismatches: "
                        + ", ".join(parity_mismatches)
                        if name == "inprocess_http_semantic_parity"
                        and parity_mismatches
                        else f"{name} not proven by scripted preflight evidence"
                    )
                )
            ),
        )
        for name in _M2M_INVARIANTS
    )
    return EnvironmentStudyReportV1(
        study_id=M2M_STUDY_ID,
        contract_id=contract.contract_id,
        valid=all(checks.values()),
        invariants=invariants,
        actor_coverage=coverage,
        transaction_summary={
            "order_groups": len(order_groups),
            "orders": len(orders),
            "settled_orders": sum(row.get("state") == "settled" for row in orders),
            "refunded_orders": sum(row.get("state") == "refunded" for row in orders),
            "world_commits": len(main.evidence.world_events),
        },
        inventory_summary={
            "contested_sku_seen_by_two_buyers": contested_sku in searched_skus,
            "contested_initial_available": (
                before_contested.get("qty_available")
                if isinstance(before_contested, Mapping)
                else None
            ),
            "contested_final_reserved": (
                after_contested.get("qty_reserved")
                if isinstance(after_contested, Mapping)
                else None
            ),
            "no_oversell": no_oversell,
        },
        ledger_summary={
            "entries": len(ledger),
            "charges": sum(row.get("effect") == "charge" for row in ledger),
            "refunds": len(refund_rows),
        },
        fulfillment_summary={
            "delivered_order_prerequisites": sum(
                row.get("status") == "delivered"
                for row in main_initial.get("shipments", [])
                if isinstance(row, Mapping)
            ),
            "returned_orders": sum(
                row.get("returned_at_tick") is not None
                for row in main_final.get("order_timelines", [])
                if isinstance(row, Mapping)
            ),
        },
        after_sales_summary={
            "order_id": _AFTER_SALES_ORDER_ID,
            "verified_operations": (
                [row.operation for row in after_sales.requests]
                if after_sales is not None
                else []
            ),
            "final_state": (
                final_after_order.get("state")
                if isinstance(final_after_order, Mapping)
                else None
            ),
        },
        replay=replay,
        diagnostics={
            "transport_parity": parity_ok,
            "transport_mismatches": list(parity_mismatches),
            "loaded_source_count": len(set(loaded_sources)),
            "unlisted_source_count": len(unlisted),
            "scripted_request_count": sum(
                len(main.factory.channel_for(actor).decision_log)
                for actor in main.factory.actor_ids()
            ),
            "estimated_paid_attempts": 17,
            "maximum_paid_attempts_with_repairs": 34,
            "maximum_input_tokens": 680_000,
            "maximum_output_tokens": 34_000,
            "maximum_cost_usd": 3.0,
        },
        data_scope={
            "scenario_kind": "synthetic_shared_market",
            "buyers": 5,
            "merchants": 5,
            "shared_catalog_listings": 5,
            "cross_merchant_product_linkage": False,
            "paid_provider_used": False,
            "delivered_order_fixture": True,
        },
    )


def run_persisted_multiagent_preflight(
    *,
    out_root: str | Path,
    artifacts_dir: str | Path,
    repo_root: str | Path = _REPO_ROOT,
) -> PersistedMultiAgentPreflightV1:
    """Freeze first, run all scripted lanes egress-free, and persist evidence."""

    root = Path(repo_root)
    output = Path(out_root)
    database_path = output / "database-world.sqlite3"
    if database_path.exists():
        raise MultiAgentPreflightError(
            "multi-Agent preflight database already exists; use a fresh out root"
        )
    contract, manifest = build_multiagent_contract(repo_root=root)
    adir = Path(artifacts_dir)
    write_json_artifact(manifest.to_dict(), adir / "source-manifest.json")
    write_json_artifact(contract.to_dict(), adir / "contract.json")

    with network_disabled() as guard:
        smoke = run_multiagent_episode(
            scenario=build_multiagent_scenario(2),
            factory=make_multiagent_factory(2),
            out_root=output / "smoke-2x2",
        )
        inprocess = run_multiagent_episode(
            scenario=build_multiagent_scenario(5),
            factory=make_multiagent_factory(5),
            out_root=output / "inprocess-5x5",
        )
        http = run_multiagent_episode(
            scenario=build_multiagent_scenario(5),
            factory=make_multiagent_factory(5),
            out_root=output / "http-5x5",
            transport="http_vcp",
        )
        database = run_multiagent_episode(
            scenario=build_multiagent_scenario(5),
            factory=make_multiagent_factory(5),
            out_root=output / "database-5x5",
            database_path=database_path,
        )
    runs = MultiAgentPreflightRunsV1(
        smoke_2x2=smoke,
        inprocess_5x5=inprocess,
        http_5x5=http,
        database_5x5=database,
    )
    loaded = repo_source_paths(iter_loaded_module_files(), root)
    report = build_multiagent_report(
        contract=contract,
        manifest=manifest,
        runs=runs,
        loaded_sources=tuple(loaded),
    )
    write_json_artifact(
        _combined_decision_log(runs), adir / "hashes-only-decision-log.json"
    )
    write_json_artifact(guard.to_dict(), adir / "network-guard.json")
    write_json_artifact(report.to_dict(), adir / "report.json")
    index = build_artifact_index(adir, ENVIRONMENT_STUDY_ARTIFACT_ORDER)
    write_json_artifact(index, adir / ARTIFACT_INDEX_NAME)
    return PersistedMultiAgentPreflightV1(
        report=report,
        artifacts_dir=str(adir),
        artifact_index=index,
        network_guard=guard.to_dict(),
    )


__all__ = [
    "M2M_PREFLIGHT_SCHEMA",
    "M2M_STUDY_ID",
    "MultiAgentPreflightError",
    "MultiAgentPreflightRunsV1",
    "MultiAgentRunV1",
    "PersistedMultiAgentPreflightV1",
    "build_multiagent_contract",
    "build_multiagent_report",
    "build_multiagent_scenario",
    "make_multiagent_factory",
    "run_multiagent_episode",
    "run_persisted_multiagent_preflight",
    "verify_multiagent_transport_parity",
]
