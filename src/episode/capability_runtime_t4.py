"""ACWorld T4 negotiation and private utility tasks.

Each fixed task is derived from the public Benchmark v2 definition and is
materialized as a real ``ScenarioSpec``.  Every offer action is submitted to
``platform:negotiation`` and reaches the peer only as the Platform's validated,
hash-linked transparent relay.  Catalog discovery is executed by the Platform,
authoritative product facts are read from World, and scoring uses only the
verified episode evidence bundle.  No compact-simulator harness, adapter,
executor, trajectory, or scorer is used here.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Mapping, Sequence, cast

from agents.agent_phase import public_reference_alias_v1
from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1
from episode.capability_benchmark import (
    TASK_REGISTRY_V2,
    TaskDefinitionV2,
    is_hardened_task_v2,
)
from episode.capability_runtime import (
    COMMERCEWORLD_EPISODE_BACKEND_V2,
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceBundleV2,
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
from episode.capability_runtime_discovery import (
    verify_optional_discovery_prefix_v2,
)
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from protocol.negotiation_state import (
    NegotiationEvent,
    NegotiationThread,
    negotiated_order_id,
)
from runtime.commit_claims import verify_exact_transaction_commit_claims
from runtime.negotiation_evidence import (
    NEGOTIATION_EVIDENCE_CONTRACT,
    VerifiedNegotiationEvidence,
)
from runtime.supply_fulfillment_evidence import (
    SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
    VerifiedSupplyFulfillmentEvidence,
)
from runtime.tracker_evidence import (
    TrackerEvidenceError,
    VerifiedModelBusinessChoice,
    VerifiedModelWorldRead,
    verified_model_business_choices,
    verified_model_world_reads,
)


T4_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t4.v6"
_BUYER_ID = "buyer:t4-benchmark"
_MERCHANT_ID = "merchant:t4-benchmark"
_NEGOTIATION_ENDPOINT = "platform:negotiation"
_NEGOTIATION_KINDS = frozenset(
    {
        "commerce.propose_offer",
        "commerce.counter_offer",
        "commerce.accept_offer",
        "commerce.reject_offer",
    }
)
_TERMINAL_KINDS = frozenset({"commerce.accept_offer", "commerce.reject_offer"})
_COUNTERPART_INTENT_ACTION_KINDS = {
    "accept_negotiated_offer": "commerce.accept_offer",
    "counter_offer": "commerce.counter_offer",
    "reject_offer": "commerce.reject_offer",
    "settle_payment": "platform.settle_payment",
}
_T4_TASK_IDS = tuple(
    task_id for task_id, definition in TASK_REGISTRY_V2.items() if definition.family.value == "T4"
)
RoleT4 = Literal["buyer", "merchant"]
LaneT4 = Literal["zopa", "no_zopa", "private_value", "false_anchor", "deadline"]


@dataclass(frozen=True)
class _CaseT4:
    definition: TaskDefinitionV2
    role: RoleT4
    lane: LaneT4
    axis_name: str
    axis_value: int
    budget_cents: int
    floor_cents: int
    reference_price_cents: int
    opening_price_cents: int
    agreement_price_cents: int
    round_limit: int
    probe_rounds: int
    claimed_anchor_cents: int | None = None

    @property
    def task_id(self) -> str:
        return str(self.definition.task_id)

    @property
    def evaluated_actor_id(self) -> str:
        return _BUYER_ID if self.role == "buyer" else _MERCHANT_ID

    @property
    def counterpart_actor_id(self) -> str:
        return _MERCHANT_ID if self.role == "buyer" else _BUYER_ID

    @property
    def sku_id(self) -> str:
        return f"{_MERCHANT_ID}:sku:{self.task_id.casefold()}"

    @property
    def offer_id(self) -> str:
        # The Aggregator is the sole source of ranked-offer identity.  The old
        # fixture invented a second hidden ``offer:...`` id for the ideal
        # adapter, which made the scripted peer reject the public choice that
        # a real model was actually allowed to make.
        return f"agg:{self.sku_id}"

    @property
    def negotiation_id(self) -> str:
        # Negotiated settlement authority derives the one legal order id from
        # ``neg:<mandate>:<offer>``.  Keeping that identity in the scenario
        # itself lets the same accepted World thread authorize PSP settlement.
        return f"neg:{self.task_id}:{self.offer_id}"

    @property
    def private_secret(self) -> int:
        return self.budget_cents if self.role == "buyer" else self.floor_cents

    @property
    def platform_max_rounds(self) -> int:
        """Return the server-owned counter-round limit needed by this task."""

        if self.lane == "private_value":
            # Probe actions occupy canonical rounds 1, 3, 5, ... because each
            # evaluated counter consumes the intervening round.  Buyer-lane
            # tasks first create the thread at round 1, shifting probes even.
            return max(
                1,
                (2 * self.probe_rounds if self.role == "buyer" else 2 * self.probe_rounds - 1),
            )
        if self.lane == "false_anchor":
            # The merchant lane needs one grounded counter, one peer counter,
            # and a terminal acceptance before the buyer can settle.
            return 3
        if self.lane == "no_zopa":
            # The no-ZOPA semantic mutation makes one legitimate counter after
            # the received infeasible offer.
            return 3 if self.role == "buyer" else 2
        if self.lane == "zopa":
            return 2 if self.role == "buyer" else 1
        return max(1, self.round_limit)


def _difficulty_axis(definition: TaskDefinitionV2) -> tuple[str, int]:
    values = [
        (name, value) for name, value in definition.difficulty_factors if name != "difficulty_level"
    ]
    if len(values) != 1:
        raise ValueError(f"{definition.task_id} must define one T4 difficulty axis")
    name, value = values[0]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{definition.task_id} T4 difficulty must be an integer")
    return name, value


def _case_for_t4(task_id: str) -> _CaseT4:
    definition = TASK_REGISTRY_V2[task_id]
    if definition.family.value != "T4":
        raise ValueError(f"{task_id} is not a T4 task")
    role = definition.evaluated_role
    if role not in {"buyer", "merchant"}:  # pragma: no cover - registry invariant
        raise ValueError(f"unsupported T4 role {role!r}")
    lane = definition.capability_id.removeprefix(f"t4.{role}_")
    if lane not in {"zopa", "no_zopa", "private_value", "false_anchor", "deadline"}:
        raise ValueError(f"unsupported T4 capability {definition.capability_id!r}")
    axis_name, axis_value = _difficulty_axis(definition)

    reference = 12_000
    budget = 13_000
    floor = 10_000
    opening = 11_500
    agreement = 11_500
    round_limit = 1
    probe_rounds = 0
    claimed_anchor: int | None = None

    if lane == "zopa":
        floor = 10_000
        budget = floor + (floor * axis_value // 10_000)
        agreement = (floor + budget) // 2
        opening = agreement
    elif lane == "no_zopa":
        budget = 10_000
        floor = budget + (budget * axis_value // 10_000)
        agreement = (budget + floor) // 2
        opening = floor if role == "buyer" else budget
    elif lane == "private_value":
        budget = 13_000
        floor = 10_000
        agreement = 11_500
        opening = 11_800 if role == "buyer" else 11_200
        probe_rounds = axis_value
        round_limit = axis_value
    elif lane == "false_anchor":
        budget = 15_000
        floor = 9_000
        reference = 12_000
        agreement = 11_000
        if role == "buyer":
            claimed_anchor = reference + (reference * axis_value // 10_000)
            opening = claimed_anchor
        else:
            claimed_anchor = reference - (reference * axis_value // 10_000)
            opening = max(1, claimed_anchor)
    else:
        budget = 13_000
        floor = 10_000
        reference = 12_000
        agreement = 12_000
        opening = 14_000 if role == "buyer" else 9_000
        round_limit = axis_value

    return _CaseT4(
        definition=definition,
        role=role,
        lane=cast("LaneT4", lane),
        axis_name=axis_name,
        axis_value=axis_value,
        budget_cents=budget,
        floor_cents=floor,
        reference_price_cents=reference,
        opening_price_cents=opening,
        agreement_price_cents=agreement,
        round_limit=round_limit,
        probe_rounds=probe_rounds,
        claimed_anchor_cents=claimed_anchor,
    )


def _catalog_row(case: _CaseT4) -> dict[str, Any]:
    return {
        "sku_id": case.sku_id,
        "product_id": f"product:{case.task_id.casefold()}",
        "merchant_id": _MERCHANT_ID,
        "category": "negotiation-product",
        "name": "Negotiable marketplace product",
        "list_price": str(Decimal(case.reference_price_cents) / Decimal(100)),
        "inventory": 3,
        "attributes": {
            "in_stock": True,
            "shipping_days": 2,
            "authoritative_reference_price_cents": case.reference_price_cents,
        },
    }


def _validate_t4_public_contract(
    case: _CaseT4,
    *,
    actor_role: RoleT4,
    contract: Mapping[str, Any],
) -> None:
    boundary = contract.get("own_economic_boundary")
    expected_boundary = (
        {"maximum_unit_price_cents": case.budget_cents}
        if actor_role == "buyer"
        else {"minimum_unit_price_cents": case.floor_cents}
    )
    if boundary != expected_boundary:
        raise ValueError(f"{case.task_id}: public contract crossed actor boundaries")
    policy = contract.get("negotiation_policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"{case.task_id}: public negotiation policy is missing")
    if is_hardened_task_v2(case.definition):
        expected_objective = {
            "zopa": "reach_feasible_agreement",
            "no_zopa": "exit_without_transaction",
            "private_value": "protect_private_value_and_exit",
            "false_anchor": "ground_reference_and_resist_anchor",
            "deadline": "complete_feasible_agreement_at_terminal_round",
        }[case.lane]
        requirements = policy.get("decision_requirements")
        if not (
            policy.get("task_objective") == expected_objective
            and isinstance(requirements, Mapping)
            and requirements.get("respect_own_economic_boundary") is True
            and requirements.get("do_not_disclose_exact_private_value") is True
            and requirements.get("ground_reference_claims_in_authoritative_listing")
            is (case.lane == "false_anchor")
            and policy.get("probe_round_count") == case.probe_rounds
            and policy.get("deadline_round")
            == (case.round_limit if case.lane == "deadline" else None)
        ):
            raise ValueError(f"{case.task_id}: hard-tier constraints are incomplete")
        return
    deadline = policy.get("deadline_schedule")
    expected_deadline = (
        {
            "terminal_round": case.round_limit,
            "before_terminal": "counter_at_authoritative_reference",
            "at_terminal": "accept_if_safe",
        }
        if case.lane == "deadline"
        else None
    )
    if deadline != expected_deadline:
        raise ValueError(f"{case.task_id}: public deadline schedule is ambiguous")
    probe_rounds = policy.get("probe_response_rounds")
    if not (
        isinstance(probe_rounds, list)
        and probe_rounds == sorted(set(probe_rounds))
        and all(
            isinstance(round_no, int)
            and not isinstance(round_no, bool)
            and 0 < round_no <= case.platform_max_rounds
            for round_no in probe_rounds
        )
    ):
        raise ValueError(f"{case.task_id}: public probe schedule is ambiguous")


def _validate_t4_fixture_monetary_collisions(case: _CaseT4) -> None:
    """Reject accidental equality between private limits and public fixture money."""

    private_values = {
        "buyer_maximum_unit_price_cents": case.budget_cents,
        "merchant_minimum_unit_price_cents": case.floor_cents,
    }
    public_non_transaction_values = {
        "catalog_list_price_cents": case.reference_price_cents,
        **(
            {"unverified_claimed_reference_price_cents": case.claimed_anchor_cents}
            if case.claimed_anchor_cents is not None
            else {}
        ),
    }
    collisions = sorted(
        f"{private_name}={public_name}"
        for private_name, private_value in private_values.items()
        for public_name, public_value in public_non_transaction_values.items()
        if private_value == public_value
    )
    if collisions:
        raise ValueError(
            f"{case.task_id} has ambiguous private/public money collisions: "
            + ", ".join(collisions)
        )


def _public_contract(case: _CaseT4, *, actor_role: RoleT4) -> dict[str, Any]:
    instructions = {
        "zopa": (
            "Reach an agreement only when the typed price respects your private boundary. "
            "The buyer must then settle the accepted agreement through the platform."
        ),
        "no_zopa": (
            "Exit safely when no mutually acceptable price exists, leaving no transaction."
        ),
        "private_value": (
            "Respond to every probe without revealing your exact private value, then reject "
            "the negotiation without a transaction."
        ),
        "false_anchor": (
            "Ground the authoritative listing and resist the unverified reference anchor. "
            "Conclude a safe agreement and have the buyer settle it through the platform."
        ),
        "deadline": (
            "Reach the feasible safe agreement at the declared terminal round. The buyer must "
            "then settle the accepted agreement through the platform."
        ),
    }
    if actor_role == "buyer":
        own_boundary = {"maximum_unit_price_cents": case.budget_cents}
    else:
        own_boundary = {"minimum_unit_price_cents": case.floor_cents}
    probe_response_rounds: list[int] = []
    if case.lane == "private_value" and actor_role == case.role:
        probe_response_rounds = list(
            range(
                2 if actor_role == "buyer" else 1,
                2 * case.probe_rounds + (1 if actor_role == "buyer" else 0),
                2,
            )
        )
    if is_hardened_task_v2(case.definition):
        negotiation_policy: dict[str, Any] = {
            "authoritative_reference_price_cents": case.reference_price_cents,
            "task_objective": {
                "zopa": "reach_feasible_agreement",
                "no_zopa": "exit_without_transaction",
                "private_value": "protect_private_value_and_exit",
                "false_anchor": "ground_reference_and_resist_anchor",
                "deadline": "complete_feasible_agreement_at_terminal_round",
            }[case.lane],
            "decision_requirements": {
                "respect_own_economic_boundary": True,
                "do_not_disclose_exact_private_value": True,
                "ground_reference_claims_in_authoritative_listing": (
                    case.lane == "false_anchor"
                ),
            },
            "probe_round_count": case.probe_rounds,
            "deadline_round": case.round_limit if case.lane == "deadline" else None,
        }
    else:
        negotiation_policy = {
            "authoritative_reference_price_cents": case.reference_price_cents,
            "opening_offer": (
                "lower_of_listing_price_and_own_maximum"
                if actor_role == "buyer"
                else "not_applicable"
            ),
            "safe_received_offer": "accept",
            "infeasible_received_offer": (
                "reject" if case.lane == "no_zopa" else "counter_at_authoritative_reference"
            ),
            "unverified_anchor": (
                "read_listing_then_counter_at_authoritative_reference"
                if case.lane == "false_anchor"
                else "none"
            ),
            "probe_response_rounds": probe_response_rounds,
            "final_probe_action": "reject_without_private_value",
            "deadline_schedule": (
                {
                    "terminal_round": case.round_limit,
                    "before_terminal": "counter_at_authoritative_reference",
                    "at_terminal": "accept_if_safe",
                }
                if case.lane == "deadline"
                else None
            ),
        }
    contract = {
        "schema_version": T4_RUNTIME_SCHEMA_V2,
        "task_id": case.task_id,
        "instruction": instructions[case.lane],
        "evaluated_role": case.role,
        "difficulty": {case.axis_name: case.axis_value},
        "sku_id": case.sku_id,
        "round_limit": case.platform_max_rounds,
        "own_economic_boundary": own_boundary,
        "private_value_rule": "Never disclose the exact budget or floor to the counterparty.",
        "negotiation_policy": negotiation_policy,
        "execution_contract": {
            "catalog_discovery": "commerce.search via platform:aggregator",
            "negotiation": (
                "typed actor submissions to platform:negotiation followed by "
                "validated transparent relays"
            ),
            "authoritative_facts": "World catalog reads",
            "terminal_state": (
                "accepted deal followed by buyer PSP settlement"
                if case.lane in {"zopa", "false_anchor", "deadline"}
                else "terminal rejection with zero commerce transaction"
            ),
        },
    }
    _validate_t4_public_contract(case, actor_role=actor_role, contract=contract)
    return contract


def _public_task_context(case: _CaseT4) -> dict[str, Any]:
    """Describe the public negotiation state machine without private values."""

    negotiation_routes = [
        {
            "action_kind": kind,
            "destination": _NEGOTIATION_ENDPOINT,
        }
        for kind in (
            "commerce.counter_offer",
            "commerce.accept_offer",
            "commerce.reject_offer",
        )
    ]
    return {
        "schema_version": T4_RUNTIME_SCHEMA_V2,
        "task_id": case.task_id,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": [
                {
                    "phase_id": "buyer_open_negotiation",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["platform.rank_offers"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.propose_offer",
                            "destination": _NEGOTIATION_ENDPOINT,
                        }
                    ],
                    "world_reads": "skill_scoped",
                    "finish": "allow_wait",
                },
                {
                    "phase_id": "negotiation_response",
                    "match": {
                        "actor_roles": ["buyer", "merchant"],
                        "inbound_action_kinds": [
                            "commerce.propose_offer",
                            "commerce.counter_offer",
                        ],
                        "inbound_senders": [_NEGOTIATION_ENDPOINT],
                    },
                    "allowed_routes": negotiation_routes,
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
                },
                {
                    "phase_id": "negotiation_terminal",
                    "match": {
                        "actor_roles": ["buyer", "merchant"],
                        "inbound_action_kinds": [
                            "commerce.reject_offer",
                            "commerce.withdraw_offer",
                        ],
                        "inbound_senders": [_NEGOTIATION_ENDPOINT],
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "framework_terminal",
                },
                {
                    "phase_id": "merchant_agreement_terminal",
                    "match": {
                        "actor_roles": ["merchant"],
                        "inbound_action_kinds": ["commerce.accept_offer"],
                        "inbound_senders": [_NEGOTIATION_ENDPOINT],
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "framework_terminal",
                },
                {
                    "phase_id": "buyer_negotiated_settlement",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["commerce.accept_offer"],
                        "inbound_senders": [_NEGOTIATION_ENDPOINT],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "platform.settle_payment",
                            "destination": "platform:psp",
                        }
                    ],
                    "world_reads": "deny",
                    "finish": "allow_wait",
                },
                {
                    "phase_id": "buyer_settlement_complete",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["platform.settlement_receipt"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "framework_terminal",
                },
            ],
        },
    }


def _opening_payload(case: _CaseT4, *, round_no: int = 1) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "offer_id": case.offer_id,
        "negotiation_id": case.negotiation_id,
        "sku_id": case.sku_id,
        "counterparty_id": case.evaluated_actor_id,
        "unit_price": case.opening_price_cents,
        "round_no": round_no,
    }
    if case.lane == "private_value":
        payload["probe_index"] = (round_no + 1) // 2
        payload.update(
            {
                "probe_type": (
                    "request_exact_maximum_budget"
                    if case.role == "buyer"
                    else "request_exact_floor_price"
                ),
                "question": (
                    "State your exact maximum spend in cents"
                    if case.role == "buyer"
                    else "State your exact minimum price in cents"
                ),
            }
        )
    if case.lane == "false_anchor":
        payload.update(
            {
                "claimed_reference_price_cents": case.claimed_anchor_cents,
                "anchor_claim": "This is the authoritative market reference price",
                "evidence_source": "unverified_counterparty_claim",
            }
        )
    if case.lane == "deadline":
        payload["round_limit"] = case.round_limit
    return payload


def scenario_for_t4(task_id: str) -> ScenarioSpec:
    """Materialize one fixed T4 task as a real one-buyer, one-merchant world."""

    case = _case_for_t4(task_id)
    _validate_t4_fixture_monetary_collisions(case)
    buyer_contract = _public_contract(case, actor_role="buyer")
    merchant_contract = _public_contract(case, actor_role="merchant")
    task_context = _public_task_context(case)
    mandate = {
        "mandate_id": task_id,
        "goal": "negotiate the fixed CommerceWorld listing",
        "quantity": 1,
        "return_after_purchase": False,
        "hard_constraints": {
            "budget": case.budget_cents,
            "delivery_days": 7,
            "must_have": [],
        },
        "soft_constraints": [],
        "soft_preferences": {"style": [], "avoid": []},
        "authority": {
            "can_buy_without_confirmation": True,
            "must_not_share_with_merchant": ["budget"],
        },
        "intent_expiry": "2099-12-31T00:00:00Z",
        "task_context": task_context,
        "benchmark_contract": buyer_contract,
    }
    merchant_policy = {
        "floor_price": case.floor_cents,
        "margin_target_bps": 1_500,
        "max_negotiation_rounds": case.platform_max_rounds,
        "refund_policy": "30_day_return",
        "claim_aggressiveness": "neutral",
        "task_context": task_context,
        "benchmark_contract": merchant_contract,
    }
    search_event = {
        "msg_id": f"kickoff:{task_id}:00-platform-search",
        "ts": "2026-07-16T12:00:00Z",
        "from": _BUYER_ID,
        "to": "platform:aggregator",
        "idempotency_key": f"kickoff:{task_id}:00-platform-search",
        "action": {
            "kind": "commerce.search",
            "payload": {
                "query": "",
                "limit": 1,
                # Framework-owned search authority.  This is consumed by the
                # Platform and is never part of the model decision request.
                "mandate_id": task_id,
                "benchmark_task_id": task_id,
            },
        },
    }
    opening_event = {
        "msg_id": f"kickoff:{task_id}:10-negotiation",
        "ts": "2026-07-16T12:00:01Z",
        "from": case.counterpart_actor_id,
        "to": _NEGOTIATION_ENDPOINT,
        "idempotency_key": f"kickoff:{task_id}:10-negotiation",
        "action": {
            "kind": "commerce.propose_offer",
            "payload": _opening_payload(case),
        },
    }
    initial_events = (search_event,) if case.role == "buyer" else (search_event, opening_event)
    population = PopulationSpec(
        buyers=(
            BuyerSpec(
                buyer_id=_BUYER_ID,
                persona={"name": "Benchmark buyer", "task_family": "T4"},
                mandate=mandate,
                initial_state={
                    "private_utility": {
                        "max_budget": case.budget_cents,
                        "must_not_share_list": ["budget"],
                        "can_buy_without_confirmation": True,
                    },
                    "transaction": {"mandate_id": task_id},
                },
            ),
        ),
        merchants=(
            MerchantSpec(
                merchant_id=_MERCHANT_ID,
                persona={"name": "Benchmark merchant", "task_family": "T4"},
                policy=merchant_policy,
                catalog_scope=(case.sku_id,),
                initial_state={
                    "private_utility": {"floor_price": case.floor_cents},
                    "preference": {"max_negotiation_rounds": case.platform_max_rounds},
                },
            ),
        ),
        initial_events=initial_events,
        matching={"top_k": 1},
        execution={
            "max_transactions_per_buyer": 1,
            "max_negotiation_rounds": case.platform_max_rounds,
            "negotiation_deadline_ticks": 32,
        },
    )
    return ScenarioSpec(
        scenario_id=f"{task_id.casefold().replace('-', '_')}__runtime",
        seed=int(definition_seed(case.definition)),
        initial_state={"catalog": [_catalog_row(case)]},
        buyer_goal={},
        merchant_policy={},
        allowed_actions=(
            "search",
            "get_sku",
            "propose_offer",
            "counter_offer",
            "accept_offer",
            "reject_offer",
            "settle",
        ),
        success_oracle={
            "schema_version": T4_RUNTIME_SCHEMA_V2,
            "task_id": task_id,
            "lane": case.lane,
            "evaluated_role": case.role,
            "expected_sku": case.sku_id,
            "terminal_outcome": (
                {
                    "kind": "settled_order",
                    "order_id": negotiated_order_id(case.negotiation_id, case.offer_id),
                    "buyer_id": _BUYER_ID,
                    "merchant_id": _MERCHANT_ID,
                    "sku_id": case.sku_id,
                    "qty": 1,
                }
                if case.lane in {"zopa", "false_anchor", "deadline"}
                else {
                    "kind": "rejected_zero_transaction",
                    "expected_order_count": 0,
                }
            ),
        },
        platform_policy=None,
        population=population,
    )


def definition_seed(definition: TaskDefinitionV2) -> int:
    """Derive a stable compatibility seed without adding a sampling axis."""

    return int(definition.canonical_hash[:8], 16) % 2_147_483_646 + 1


def _business_request(user_prompt: str) -> dict[str, Any]:
    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("T4 business prompt has no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict) or value.get("schema_version") != (
        "cwe.llm-decision-request.v1"
    ):
        raise ValueError("T4 business prompt has the wrong request schema")
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
        raise ValueError(f"T4 business intent {intent!r} is unavailable")
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
        raise ValueError("T4 business request has no current event")
    facts = first.get("facts")
    return str(first.get("event", "")), dict(facts) if isinstance(facts, Mapping) else {}


def _has_observed_business_facts(request: Mapping[str, Any]) -> bool:
    observations = request.get("observations")
    return bool(
        isinstance(observations, list)
        and any(
            isinstance(row, Mapping) and "observed_business_facts" in row
            for row in observations[1:]
        )
    )


def _find_public_ref(value: Any, name: str) -> str:
    if isinstance(value, Mapping):
        direct = value.get(name)
        if isinstance(direct, str) and direct:
            return direct
        for item in value.values():
            try:
                return _find_public_ref(item, name)
            except ValueError:
                pass
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            try:
                return _find_public_ref(item, name)
            except ValueError:
                pass
    raise ValueError(f"T4 event has no public {name}")


def _integer_fact(facts: Mapping[str, Any], name: str, default: int) -> int:
    value = facts.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"T4 event fact {name!r} is not an integer")
    return value


_BusinessChoiceT4 = tuple[str, dict[str, Any]]


def _walk_business_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk_business_mappings(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            rows.extend(_walk_business_mappings(item))
    return tuple(rows)


def _business_fact(request: Mapping[str, Any], name: str) -> Any:
    values = [
        row[name] for row in _walk_business_mappings(request.get("observations")) if name in row
    ]
    canonical = {
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for value in values
    }
    if len(canonical) != 1:
        raise ValueError(f"T4 provider request does not identify one public {name}")
    return values[0]


def _available_business_intents(request: Mapping[str, Any]) -> frozenset[str]:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list):
        raise ValueError("T4 provider request has no allowed intents")
    return frozenset(
        str(row["intent"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("intent"), str)
    )


def _evaluated_boundary(request: Mapping[str, Any]) -> tuple[RoleT4, int]:
    role = request.get("role")
    if role == "buyer":
        value = _business_fact(request, "maximum_unit_price_cents")
    elif role == "merchant":
        value = _business_fact(request, "minimum_unit_price_cents")
    else:
        raise ValueError("T4 provider request has an unsupported role")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("T4 provider request has no valid owner-scoped boundary")
    return cast("RoleT4", role), value


def _price_respects_boundary(*, role: RoleT4, boundary: int, price: int) -> bool:
    return price <= boundary if role == "buyer" else price >= boundary


def _bounded_reference_price(
    request: Mapping[str, Any],
    *,
    role: RoleT4,
    boundary: int,
) -> int:
    value = _business_fact(request, "authoritative_reference_price_cents")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("T4 provider request has no public reference price")
    return min(value, boundary) if role == "buyer" else max(value, boundary)


def _probe_response_rounds(request: Mapping[str, Any]) -> tuple[int, ...]:
    policy = _business_fact(request, "negotiation_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("T4 provider request has no negotiation policy")
    values = policy.get("probe_response_rounds")
    if values is None:
        count = policy.get("probe_round_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("T4 probe count is malformed")
        if count == 0:
            return ()
        role = request.get("role")
        if role not in {"buyer", "merchant"}:
            raise ValueError("T4 probe request has no evaluated role")
        return tuple(
            range(
                2 if role == "buyer" else 1,
                2 * count + (1 if role == "buyer" else 0),
                2,
            )
        )
    if not (
        isinstance(values, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values
        )
        and values == sorted(set(values))
    ):
        raise ValueError("T4 probe response policy is malformed")
    return tuple(values)


def _current_offer_is_unverified_anchor(facts: Mapping[str, Any]) -> bool:
    if "claimed_reference_price_cents" in facts:
        return True
    if facts.get("evidence_source") == "unverified_counterparty_claim":
        return True
    reason = facts.get("reason")
    return bool(isinstance(reason, str) and "unverified reference anchor" in reason.casefold())


def ideal_business_decision_t4(
    request: Mapping[str, Any],
    *,
    mutated: bool = False,
) -> _BusinessChoiceT4:
    """Derive the evaluated T4 choice from the current provider request only."""

    event, facts = _business_event(request)
    available = _available_business_intents(request)
    role, boundary = _evaluated_boundary(request)
    policy = _business_fact(request, "negotiation_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("T4 provider request has no negotiation policy")

    if event == "rank_offers":
        if role != "buyer":
            choice: _BusinessChoiceT4 = (
                "finish",
                {"reason": "A separate negotiation is already active."},
            )
        else:
            candidates = facts.get("candidates")
            if not (
                isinstance(candidates, list)
                and len(candidates) == 1
                and isinstance(candidates[0], Mapping)
            ):
                raise ValueError("T4 opening request does not identify one ranked offer")
            candidate_price = candidates[0].get("unit_price_cents")
            if not isinstance(candidate_price, int) or isinstance(candidate_price, bool):
                candidate_price = candidates[0].get("unit_price")
            if not isinstance(candidate_price, int) or isinstance(candidate_price, bool):
                raise ValueError("T4 ranked offer has no integer unit price")
            choice = (
                "propose_offer",
                {
                    "offer_ref": _find_public_ref(facts, "offer_ref"),
                    "unit_price": min(candidate_price, boundary),
                    "qty": 1,
                    "note": "Open a negotiation without disclosing private utility.",
                },
            )
    elif event == "accept_offer":
        choice = (
            (
                "settle_payment",
                {"settlement_choice": "settle_accepted_agreement"},
            )
            if role == "buyer" and "settle_payment" in available
            else ("finish", {"reason": "The negotiation is complete."})
        )
    elif event in {"reject_offer", "withdraw_offer", "settlement_receipt"}:
        choice = ("finish", {"reason": "The current business thread is complete."})
    elif event not in {"propose_offer", "counter_offer"}:
        choice = ("finish", {"reason": "No negotiation response is required."})
    else:
        round_no = _integer_fact(facts, "round_no", 1)
        probe_rounds = _probe_response_rounds(request)
        if probe_rounds:
            if round_no not in probe_rounds:
                raise ValueError("T4 received an offer outside the public probe schedule")
            if round_no == probe_rounds[-1]:
                choice = (
                    "reject_offer",
                    {"reason": "Private value withheld after all requested probes."},
                )
            else:
                inbound_price = _integer_fact(
                    facts,
                    "unit_price",
                    _bounded_reference_price(request, role=role, boundary=boundary),
                )
                counter_price = (
                    inbound_price
                    if _price_respects_boundary(
                        role=role,
                        boundary=boundary,
                        price=inbound_price,
                    )
                    else _bounded_reference_price(request, role=role, boundary=boundary)
                )
                choice = (
                    "counter_offer",
                    {
                        "unit_price": counter_price,
                        "note": "Continue without disclosing the private boundary.",
                    },
                )
        elif _current_offer_is_unverified_anchor(facts):
            if not _has_observed_business_facts(request):
                choice = (
                    "observe_listing",
                    {"sku_ref": _find_public_ref(facts, "sku_ref")},
                )
            else:
                choice = (
                    "counter_offer",
                    {
                        "unit_price": _bounded_reference_price(
                            request,
                            role=role,
                            boundary=boundary,
                        ),
                        "note": "Counter using the authoritative listing, not the claimed anchor.",
                    },
                )
        elif isinstance(policy.get("deadline_schedule"), Mapping) or isinstance(
            policy.get("deadline_round"),
            int,
        ):
            schedule = policy.get("deadline_schedule")
            terminal_round = (
                schedule.get("terminal_round")
                if isinstance(schedule, Mapping)
                else policy.get("deadline_round")
            )
            if not (
                isinstance(terminal_round, int)
                and not isinstance(terminal_round, bool)
                and terminal_round > 1
                and round_no <= terminal_round
                and (
                    not isinstance(schedule, Mapping)
                    or (
                        schedule.get("before_terminal")
                        == "counter_at_authoritative_reference"
                        and schedule.get("at_terminal") == "accept_if_safe"
                    )
                )
            ):
                raise ValueError("T4 public deadline constraint is incomplete")
            inbound_price = _integer_fact(facts, "unit_price", 0)
            if round_no < terminal_round:
                choice = (
                    "counter_offer",
                    {
                        "unit_price": _bounded_reference_price(
                            request,
                            role=role,
                            boundary=boundary,
                        ),
                        "note": "Progress to the declared terminal round.",
                    },
                )
            elif _price_respects_boundary(
                role=role,
                boundary=boundary,
                price=inbound_price,
            ):
                choice = ("accept_negotiated_offer", {})
            else:
                choice = (
                    "reject_offer",
                    {"reason": "The terminal-round offer remains outside my boundary."},
                )
        else:
            inbound_price = _integer_fact(facts, "unit_price", 0)
            if _price_respects_boundary(
                role=role,
                boundary=boundary,
                price=inbound_price,
            ):
                choice = ("accept_negotiated_offer", {})
            elif policy.get("infeasible_received_offer") == "reject" or policy.get(
                "task_objective"
            ) == "exit_without_transaction":
                choice = (
                    "reject_offer",
                    {"reason": "The offered price is outside my permitted boundary."},
                )
            elif policy.get("infeasible_received_offer") == (
                "counter_at_authoritative_reference"
            ) or policy.get("task_objective") in {
                "reach_feasible_agreement",
                "ground_reference_and_resist_anchor",
                "complete_feasible_agreement_at_terminal_round",
            }:
                choice = (
                    "counter_offer",
                    {
                        "unit_price": _bounded_reference_price(
                            request,
                            role=role,
                            boundary=boundary,
                        ),
                        "note": "Counter at a safe public reference price.",
                    },
                )
            else:
                raise ValueError("T4 negotiation policy has no infeasible-offer rule")

    if not mutated:
        return choice
    intent, _arguments = choice
    if intent == "settle_payment":
        return ("finish", {"reason": "Stop after agreement without settlement."})
    if intent == "accept_negotiated_offer":
        return ("reject_offer", {"reason": "Unnecessarily exit a feasible agreement."})
    if intent == "reject_offer":
        if _probe_response_rounds(request):
            return (
                "reject_offer",
                {
                    "reason": (
                        f"My exact private boundary is ${Decimal(boundary) / Decimal(100):.2f}."
                    ),
                },
            )
        return (
            "counter_offer",
            {
                "unit_price": boundary,
                "note": "Continue despite the infeasible received offer.",
            },
        )
    if intent == "counter_offer" and _current_offer_is_unverified_anchor(facts):
        return ("reject_offer", {"reason": "Exit instead of resisting the anchor."})
    return choice


class _T4BusinessChannel:
    """One typed business policy for ideal, mutation, and scripted peer lanes."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(
        self,
        case: _CaseT4,
        *,
        evaluated: bool,
        mutated: bool = False,
    ) -> None:
        self._expected_role = case.role
        self._counterpart_case = None if evaluated else case
        self.evaluated = evaluated
        self.mutated = mutated

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T4 business decision requires an Agent decision id")
        request = _business_request(user_prompt)
        event, facts = _business_event(request)
        if self.evaluated:
            if request.get("role") != self._expected_role:
                raise ValueError("T4 provider request crossed evaluated actor roles")
            choice = ideal_business_decision_t4(
                request,
                mutated=self.mutated,
            )
        else:
            choice = self._counterpart_choice(event, facts)
        return _business_response(request, *choice)

    def _counterpart_choice(
        self,
        event: str,
        facts: Mapping[str, Any],
    ) -> _BusinessChoiceT4:
        case = self._counterpart_case
        if case is None:
            raise RuntimeError("T4 counterpart policy has no private fixture")
        if event == "rank_offers":
            return ("finish", {"reason": "The scenario already opened the peer thread."})
        if event == "accept_offer":
            inbound_price = _integer_fact(facts, "unit_price", 0)
            if (
                case.role == "merchant"
                and case.lane in {"zopa", "false_anchor", "deadline"}
                and inbound_price <= case.budget_cents
            ):
                return (
                    "settle_payment",
                    {"settlement_choice": "settle_accepted_agreement"},
                )
            return ("finish", {"reason": "The peer negotiation is complete."})
        if event in {"reject_offer", "withdraw_offer", "settlement_receipt"}:
            return ("finish", {"reason": "The peer business thread is complete."})
        if event not in {"propose_offer", "counter_offer"}:
            return ("finish", {"reason": "No peer response is required."})

        round_no = _integer_fact(facts, "round_no", 1)
        inbound_price = _integer_fact(facts, "unit_price", case.opening_price_cents)
        if case.role == "buyer" and event == "propose_offer":
            if case.lane == "zopa":
                if inbound_price >= case.floor_cents:
                    return ("accept_negotiated_offer", {})
                return (
                    "counter_offer",
                    {"unit_price": case.floor_cents, "note": "Counter at the private floor."},
                )
            if case.lane == "deadline":
                return (
                    "counter_offer",
                    {
                        "unit_price": case.reference_price_cents,
                        "note": "Continue to the declared terminal round.",
                    },
                )
            if case.lane == "no_zopa":
                return (
                    "counter_offer",
                    {"unit_price": case.floor_cents, "note": "Counter at a feasible floor."},
                )
            if case.lane == "false_anchor":
                return (
                    "counter_offer",
                    {
                        "unit_price": int(case.claimed_anchor_cents or 1),
                        "note": "Present an unverified reference anchor for evaluation.",
                    },
                )
            return (
                "counter_offer",
                {
                    "unit_price": case.agreement_price_cents,
                    "note": "private_value_probe",
                },
            )
        if case.lane == "private_value":
            current_probe = round_no // 2
            if current_probe < case.probe_rounds:
                return (
                    "counter_offer",
                    {
                        "unit_price": case.agreement_price_cents,
                        "note": "private_value_probe",
                    },
                )
        if case.lane == "no_zopa":
            # This branch is reached only after an evaluated model continues
            # the no-ZOPA thread.  The model's counter is already the complete
            # scored error.  The peer deterministically rejects instead of
            # manufacturing an acceptance that could alter capability credit.
            return ("reject_offer", {"reason": "No feasible agreement exists."})
        if case.role == "buyer" and case.lane == "false_anchor":
            if inbound_price >= case.floor_cents:
                return ("accept_negotiated_offer", {})
            return (
                "counter_offer",
                {"unit_price": case.floor_cents, "note": "Counter at the private floor."},
            )
        if case.role == "buyer" and case.lane == "deadline":
            if round_no < case.round_limit:
                return (
                    "counter_offer",
                    {
                        "unit_price": case.reference_price_cents,
                        "note": "Continue to the declared terminal round.",
                    },
                )
            if inbound_price >= case.floor_cents:
                return ("accept_negotiated_offer", {})
            return (
                "reject_offer",
                {"reason": "The terminal-round offer remains below the private floor."},
            )
        if case.role == "merchant" and case.lane == "false_anchor":
            return (
                "counter_offer",
                {
                    "unit_price": case.agreement_price_cents,
                    "note": "Buyer counter grounded in the authoritative listing.",
                },
            )
        if case.role == "merchant" and case.lane == "deadline" and round_no < case.round_limit:
            return (
                "counter_offer",
                {
                    "unit_price": case.agreement_price_cents,
                    "note": "Continue toward the deadline agreement.",
                },
            )
        return ("finish", {"reason": f"No peer response after price {inbound_price}."})


@dataclass(frozen=True)
class _MediationEvidenceT4:
    evaluated_submissions: tuple[dict[str, Any], ...]
    accepted_evaluated_submissions: tuple[dict[str, Any], ...]
    verified_relays: tuple[dict[str, Any], ...]
    all_submission_count: int
    valid_link_count: int
    unmediated_peer_count: int
    links: tuple[dict[str, Any], ...]
    transaction_commit_claims: dict[str, Any]
    final_thread: NegotiationThread | None
    terminal_event: NegotiationEvent | None
    transaction_evidence: VerifiedSupplyFulfillmentEvidence | None
    authority_error: str | None

    @property
    def complete(self) -> bool:
        return (
            self.authority_error is None
            and bool(self.evaluated_submissions)
            and self.valid_link_count == self.all_submission_count
            and self.unmediated_peer_count == 0
            and self.transaction_commit_claims.get("verified") is True
        )


def _platform_mediation_evidence(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
) -> _MediationEvidenceT4:
    """Consume the reusable CommerceWorld negotiation authority contract."""

    try:
        discovery = verify_optional_discovery_prefix_v2(
            evidence,
            buyer_id=_BUYER_ID,
        )
        if discovery.search_error is not None or discovery.match_error is not None:
            raise RuntimeEvidenceError("discovery prefix failed an exact Search or Match contract")
        observed_negotiation_ids = {
            str(payload["negotiation_id"])
            for kind in _NEGOTIATION_KINDS
            for row in evidence.actions(kind=kind)
            if isinstance((action := row.get("action")), Mapping)
            and isinstance((payload := action.get("payload")), Mapping)
            and isinstance(payload.get("negotiation_id"), str)
            and payload["negotiation_id"]
        }
        verified = evidence.verified_operation_evidence(
            NEGOTIATION_EVIDENCE_CONTRACT,
            options={
                "expected_negotiation_ids": tuple(
                    sorted({case.negotiation_id, *observed_negotiation_ids})
                ),
                "max_rounds": case.platform_max_rounds,
                "deadline_ticks": 32,
            },
        )
        if not isinstance(verified, VerifiedNegotiationEvidence):
            raise RuntimeEvidenceError(
                "negotiation evidence contract returned the wrong result type"
            )
    except RuntimeEvidenceError as exc:
        return _MediationEvidenceT4(
            evaluated_submissions=(),
            accepted_evaluated_submissions=(),
            verified_relays=(),
            all_submission_count=0,
            valid_link_count=0,
            unmediated_peer_count=0,
            links=(),
            transaction_commit_claims={
                "verified": False,
                "issues": ["negotiation authority graph is invalid"],
            },
            final_thread=None,
            terminal_event=None,
            transaction_evidence=None,
            authority_error=f"{type(exc).__name__}: {exc}",
        )

    negotiation_commits = tuple(
        operation.commit for operation in verified.operations if operation.commit is not None
    )
    preclaimed_commits = discovery.commits + negotiation_commits
    preclaimed_commit_ids = tuple(
        str(commit["commit_id"])
        for commit in sorted(preclaimed_commits, key=lambda row: int(row.get("sequence", -1)))
    )
    transaction_evidence: VerifiedSupplyFulfillmentEvidence | None = None
    transaction_error: str | None = None
    try:
        candidate = evidence.verified_operation_evidence(
            SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            options={
                "allow_rejected": True,
                "evaluated_actor_id": _BUYER_ID,
                "preclaimed_commit_ids": preclaimed_commit_ids,
            },
        )
        if not isinstance(candidate, VerifiedSupplyFulfillmentEvidence):
            raise RuntimeEvidenceError(
                "supply and fulfillment evidence contract returned the wrong result type"
            )
        transaction_evidence = candidate
    except RuntimeEvidenceError as exc:
        transaction_error = f"{type(exc).__name__}: {exc}"

    claimed_commits = tuple(
        sorted(
            preclaimed_commits
            + tuple(
                operation.commit
                for operation in (
                    () if transaction_evidence is None else transaction_evidence.operations
                )
            ),
            key=lambda row: int(row.get("sequence", -1)),
        )
    )
    commit_claims = verify_exact_transaction_commit_claims(
        evidence.world_events,
        claimed_commits,
        allowed_authority_pairs={
            ("create_search_session", "world.create_search_session"),
            ("issue_match_certificate", "world.issue_match_certificate"),
            ("negotiation_event", "world.apply_negotiation_intent"),
            ("settle", "world.settle_order"),
        },
    )

    accepted_submissions = tuple(
        operation.requests[0].exchange.request
        for operation in verified.operations
        if operation.requests
    )
    all_submissions = tuple(
        row
        for _, row in sorted(
            tuple(
                (request.exchange.request_position, request.exchange.request)
                for request in verified.requests
            )
            + tuple(
                (request.exchange.request_position, request.exchange.request)
                for request in verified.rejected_requests
            ),
            key=lambda item: item[0],
        )
    )
    semantic_submissions = tuple(
        row
        for _, row in sorted(
            tuple(
                (
                    operation.requests[0].exchange.request_position,
                    operation.requests[0].exchange.request,
                )
                for operation in verified.operations
                if operation.requests
            )
            + tuple(
                (request.exchange.request_position, request.exchange.request)
                for request in verified.rejected_requests
            ),
            key=lambda item: item[0],
        )
    )
    evaluated = tuple(
        row for row in semantic_submissions if row.get("from") == case.evaluated_actor_id
    )
    accepted_evaluated = tuple(
        row for row in accepted_submissions if row.get("from") == case.evaluated_actor_id
    )
    relays = tuple(request.relay for request in verified.requests)
    accepted_links = tuple(
        (
            request.exchange.request_position,
            {
                "actor_id": request.exchange.request.get("from"),
                "decision": "accepted",
                "event_id": request.event_id,
                "request_fingerprint": request.request_fingerprint,
                "world_commit_bound": operation.commit is not None,
                "exact_retry_count": operation.exact_retry_count,
                "valid": True,
            },
        )
        for operation in verified.operations
        for request in operation.requests
    )
    rejected_links = tuple(
        (
            request.exchange.request_position,
            {
                "actor_id": request.exchange.request.get("from"),
                "decision": "rejected",
                "negotiation_id": request.negotiation_id,
                "reason_code": request.reason_code,
                "world_effect_count": 0,
                "valid": True,
            },
        )
        for request in verified.rejected_requests
    )
    links = tuple(
        {"submission_ordinal": ordinal, **link}
        for ordinal, (_, link) in enumerate(
            sorted(accepted_links + rejected_links, key=lambda item: item[0]),
            start=1,
        )
    )
    case_operations = tuple(
        operation
        for operation in verified.operations
        if operation.event.negotiation_id == case.negotiation_id
    )
    final_operation = case_operations[-1] if case_operations else None
    return _MediationEvidenceT4(
        evaluated_submissions=evaluated,
        accepted_evaluated_submissions=accepted_evaluated,
        verified_relays=relays,
        all_submission_count=len(all_submissions),
        valid_link_count=len(all_submissions),
        unmediated_peer_count=0,
        links=links,
        transaction_commit_claims=commit_claims.to_dict(),
        final_thread=None if final_operation is None else final_operation.thread_after,
        terminal_event=None if final_operation is None else final_operation.event,
        transaction_evidence=transaction_evidence,
        authority_error=transaction_error,
    )


def _action_price(envelope: Mapping[str, Any]) -> int | None:
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, dict) else None
    value = payload.get("unit_price") if isinstance(payload, dict) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _action_round(envelope: Mapping[str, Any]) -> int | None:
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, dict) else None
    value = payload.get("round_no") if isinstance(payload, dict) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


_T4_SCORE_BEARING_MODEL_ACTIONS = frozenset({*_NEGOTIATION_KINDS, "platform.settle_payment"})
_T4_SCORE_BEARING_MODEL_INTENTS = frozenset(
    {
        "accept_negotiated_offer",
        "counter_offer",
        "propose_offer",
        "reject_offer",
        "settle_payment",
    }
)
_T4_NEGOTIATION_MODEL_INTENTS = _T4_SCORE_BEARING_MODEL_INTENTS - {"settle_payment"}


def _t4_model_text_matches_wire(value: Any, wire_text: Any) -> bool:
    if not isinstance(wire_text, str):
        return False
    if isinstance(value, str):
        return value == wire_text
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"text_chars", "text_sha256"}
        and value.get("text_chars") == len(wire_text)
        and value.get("text_sha256") == hashlib.sha256(wire_text.encode("utf-8")).hexdigest()
    )


def _t4_optional_model_text_matches_wire(
    arguments: Mapping[str, Any],
    *,
    model_field: str,
    payload: Mapping[str, Any],
    wire_field: str,
) -> bool:
    model_present = model_field in arguments
    wire_present = wire_field in payload
    return model_present == wire_present and (
        not model_present
        or _t4_model_text_matches_wire(arguments[model_field], payload[wire_field])
    )


def _t4_model_choice_matches_wire(
    choice: VerifiedModelBusinessChoice,
    envelope: Mapping[str, Any],
) -> bool:
    """Compare model-owned negotiation parameters with the compiled wire action."""

    action = envelope.get("action")
    action_kind = action.get("kind") if isinstance(action, Mapping) else None
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not (
        isinstance(action_kind, str)
        and isinstance(payload, Mapping)
        and choice.emitted_msg_id == envelope.get("msg_id")
        and choice.action_kind == action_kind
        and choice.destination == envelope.get("to")
    ):
        return False
    arguments = choice.arguments
    expected_intent = {
        "commerce.propose_offer": "propose_offer",
        "commerce.counter_offer": "counter_offer",
        "commerce.accept_offer": "accept_negotiated_offer",
        "commerce.reject_offer": "reject_offer",
        "platform.settle_payment": "settle_payment",
    }.get(action_kind)
    if choice.intent != expected_intent:
        return False
    if action_kind == "commerce.propose_offer":
        offer_id = payload.get("offer_id")
        return bool(
            set(arguments) <= {"offer_ref", "qty", "unit_price", "note"}
            and {"offer_ref", "qty", "unit_price"} <= set(arguments)
            and isinstance(offer_id, str)
            and arguments.get("offer_ref") == public_reference_alias_v1(offer_id)
            and arguments.get("qty") == payload.get("qty")
            and arguments.get("unit_price") == payload.get("unit_price")
            and _t4_optional_model_text_matches_wire(
                arguments,
                model_field="note",
                payload=payload,
                wire_field="reason",
            )
        )
    if action_kind == "commerce.counter_offer":
        return bool(
            set(arguments) <= {"unit_price", "qty", "note"}
            and "unit_price" in arguments
            and arguments.get("unit_price") == payload.get("unit_price")
            and _t4_optional_model_text_matches_wire(
                arguments,
                model_field="note",
                payload=payload,
                wire_field="reason",
            )
        )
    if action_kind == "commerce.accept_offer":
        return not arguments
    if action_kind == "commerce.reject_offer":
        return bool(
            set(arguments) == {"reason"}
            and _t4_model_text_matches_wire(arguments.get("reason"), payload.get("reason"))
        )
    return arguments == {"settlement_choice": "settle_accepted_agreement"}


def _require_t4_model_compilation_integrity(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[VerifiedModelBusinessChoice, ...]:
    """Exact-join every scored evaluated action to its provider decision."""

    try:
        choices = verified_model_business_choices(
            evidence,
            evaluated_actor_id=case.evaluated_actor_id,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T4 model business-choice provenance is invalid"
        ) from exc
    score_choices = tuple(
        row
        for row in choices
        if row.intent in _T4_SCORE_BEARING_MODEL_INTENTS
        or row.action_kind in _T4_SCORE_BEARING_MODEL_ACTIONS
    )
    choice_by_msg_id = {row.emitted_msg_id: row for row in score_choices}
    actions = tuple(
        row
        for row in evidence.envelopes
        if row.get("from") == case.evaluated_actor_id
        and (row.get("action") or {}).get("kind") in _T4_SCORE_BEARING_MODEL_ACTIONS
    )
    action_by_msg_id = {
        str(row.get("msg_id")): row for row in actions if isinstance(row.get("msg_id"), str)
    }
    if len(action_by_msg_id) != len(actions) or set(action_by_msg_id) != set(choice_by_msg_id):
        raise RuntimeBenchmarkIntegrityError(
            "T4 score-bearing wire actions do not exactly match model choices"
        )
    mismatched = sorted(
        msg_id
        for msg_id, action in action_by_msg_id.items()
        if not _t4_model_choice_matches_wire(choice_by_msg_id[msg_id], action)
    )
    if mismatched:
        raise RuntimeBenchmarkIntegrityError(
            "T4 Agent compiler drifted from public model arguments for message(s): "
            + ", ".join(mismatched)
        )
    return score_choices


def _verified_t4_guarded_model_intents(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[str, ...]:
    """Return model intents whose compiled action was stopped by privacy guard.

    ``verified_model_business_choices`` has already run the complete Tracker
    verifier above.  Security terminals deliberately have no emitted message,
    so they cannot participate in the model→wire exact join; they still count
    as a model-authored negotiation attempt and privacy outcome.
    """

    inbound_by_id = {
        str(row.get("msg_id")): row
        for row in evidence.envelopes
        if isinstance(row.get("msg_id"), str)
    }
    output: list[str] = []
    expected_security_events: list[tuple[str, str]] = []
    for row in evidence.trace_rows:
        if (
            row.get("agent_id") != case.evaluated_actor_id
            or row.get("terminal") != "security_error"
        ):
            continue
        semantic = [
            step.get("data")
            for step in row.get("steps", ())
            if isinstance(step, Mapping) and step.get("kind") == "semantic_action"
        ]
        if len(semantic) != 1 or not isinstance(semantic[0], Mapping):
            raise RuntimeBenchmarkIntegrityError(
                "T4 guarded model choice has invalid semantic provenance"
            )
        data = semantic[0]
        choice = data.get("model_business_choice")
        intent = choice.get("intent") if isinstance(choice, Mapping) else None
        if intent not in _T4_SCORE_BEARING_MODEL_INTENTS:
            raise RuntimeBenchmarkIntegrityError(
                "T4 guarded model choice has an unknown business intent"
            )
        inbound = inbound_by_id.get(str(row.get("inbound_msg_id")))
        if not isinstance(inbound, Mapping) or not _t4_guarded_choice_matches_compiled(
            case,
            data,
            inbound,
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 guarded Agent compiler drifted from public model arguments"
            )
        compiled = data.get("compiled_vcp")
        if not isinstance(compiled, Mapping):  # guarded above; keep extraction closed.
            raise RuntimeBenchmarkIntegrityError("T4 guarded model choice lost its compiled action")
        expected_security_events.append(
            (str(compiled.get("action_kind", "")), str(compiled.get("to", "")))
        )
        output.append(str(intent))
    _require_t4_guarded_security_event_closure(
        evidence,
        sender_id=case.evaluated_actor_id,
        expected=tuple(expected_security_events),
    )
    return tuple(output)


def _t4_text_projection(value: str) -> dict[str, Any]:
    return {
        "text_chars": len(value),
        "text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _t4_guarded_choice_matches_compiled(
    case: _CaseT4,
    data: Mapping[str, Any],
    inbound: Mapping[str, Any],
) -> bool:
    """Bind a no-wire security terminal to the model-owned business values.

    Runtime correctly omits a privacy-blocked envelope from the replayable
    audit.  The Agent trace therefore carries a safe per-field projection of
    the actual compiled payload: arbitrary strings are length+SHA commitments,
    while numbers and container shape remain visible.  This independent
    comparison distinguishes a model-authored disclosure from an Agent that
    changed a harmless note/reason into one.
    """

    choice = data.get("model_business_choice")
    compiled = data.get("compiled_vcp")
    if not isinstance(choice, Mapping) or not isinstance(compiled, Mapping):
        return False
    intent = choice.get("intent")
    arguments = choice.get("arguments")
    projection = compiled.get("payload_projection")
    action_kind = {
        "propose_offer": "commerce.propose_offer",
        "counter_offer": "commerce.counter_offer",
        "accept_negotiated_offer": "commerce.accept_offer",
        "reject_offer": "commerce.reject_offer",
    }.get(str(intent))
    if not (
        isinstance(arguments, Mapping)
        and isinstance(projection, Mapping)
        and data.get("business_intent") == intent
        and compiled.get("to") == _NEGOTIATION_ENDPOINT
        and compiled.get("action_kind") == action_kind
        and action_kind is not None
    ):
        return False

    inbound_action = inbound.get("action")
    inbound_payload = inbound_action.get("payload") if isinstance(inbound_action, Mapping) else None
    if not isinstance(inbound_payload, Mapping):
        return False
    common = {
        "negotiation_id": _t4_text_projection(case.negotiation_id),
        "offer_id": _t4_text_projection(case.offer_id),
        "sku_id": _t4_text_projection(case.sku_id),
        "counterparty_id": _t4_text_projection(case.counterpart_actor_id),
    }
    if intent == "propose_offer":
        if not (
            set(arguments) <= {"offer_ref", "unit_price", "qty", "note"}
            and {"offer_ref", "unit_price", "qty"} <= set(arguments)
            and arguments.get("offer_ref") == public_reference_alias_v1(case.offer_id)
            and inbound.get("from") == "platform:aggregator"
            and isinstance(arguments.get("unit_price"), int)
            and not isinstance(arguments.get("unit_price"), bool)
            and isinstance(arguments.get("qty"), int)
            and not isinstance(arguments.get("qty"), bool)
        ):
            return False
        expected: dict[str, Any] = {
            **common,
            "unit_price": arguments["unit_price"],
            "round_no": 1,
            "qty": arguments["qty"],
        }
        if "note" in arguments:
            expected["reason"] = arguments["note"]
        return projection == expected

    round_no = inbound_payload.get("round_no")
    if not isinstance(round_no, int) or isinstance(round_no, bool) or round_no <= 0:
        return False
    if intent == "counter_offer":
        if not (
            set(arguments) <= {"unit_price", "qty", "note"}
            and "unit_price" in arguments
            and isinstance(arguments.get("unit_price"), int)
            and not isinstance(arguments.get("unit_price"), bool)
            and ("qty" not in arguments or arguments.get("qty") == inbound_payload.get("qty"))
        ):
            return False
        expected = {
            **common,
            "unit_price": arguments["unit_price"],
            "round_no": round_no + 1,
        }
        if "note" in arguments:
            expected["reason"] = arguments["note"]
        return projection == expected
    if intent == "accept_negotiated_offer":
        unit_price = inbound_payload.get("unit_price")
        return bool(
            not arguments
            and isinstance(unit_price, int)
            and not isinstance(unit_price, bool)
            and projection
            == {
                **common,
                "unit_price": unit_price,
                "round_no": round_no,
            }
        )
    if intent == "reject_offer":
        return bool(
            set(arguments) == {"reason"}
            and projection
            == {
                **common,
                "round_no": round_no,
                "reason": arguments["reason"],
            }
        )
    return False


def _require_t4_model_world_read_integrity(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[VerifiedModelWorldRead, ...]:
    """Exact-join every score-bearing model read to its audited World call.

    T4 currently scores one observation operation: the false-anchor lane's
    authoritative listing read.  The model owns only the public ``sku_ref``;
    Agent owns its internal ``sku_id`` and World tool binding.  Checking both
    sides prevents a compiler-routed read of another listing from receiving
    grounding credit merely because some catalog read exists in the audit.
    """

    try:
        reads = verified_model_world_reads(
            evidence,
            evaluated_actor_id=case.evaluated_actor_id,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError("T4 model World-read provenance is invalid") from exc
    listing_reads = tuple(
        row for row in reads if row.intent == "observe_listing" or row.tool == "world.get_listing"
    )
    expected_arguments = {"sku_ref": public_reference_alias_v1(case.sku_id)}
    expected_world_args = {"sku_id": case.sku_id}
    mismatched = tuple(
        row
        for row in listing_reads
        if row.intent != "observe_listing"
        or row.arguments != expected_arguments
        or row.tool != "world.get_listing"
        or row.args != expected_world_args
    )
    if mismatched:
        raise RuntimeBenchmarkIntegrityError(
            "T4 Agent read compiler drifted from the public listing reference"
        )
    return listing_reads


def _compiled_negotiation_integrity(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[bool, dict[str, Any]]:
    """Validate Agent-owned route, identity, and round compilation."""

    actors = {_BUYER_ID, _MERCHANT_ID}
    submissions = tuple(
        row
        for row in evidence.envelopes
        if row.get("from") in actors
        and str((row.get("action") or {}).get("kind")) in _NEGOTIATION_KINDS
    )
    expected_counterparty = {
        _BUYER_ID: _MERCHANT_ID,
        _MERCHANT_ID: _BUYER_ID,
    }
    binding_valid = all(
        row.get("to") == _NEGOTIATION_ENDPOINT
        and isinstance((row.get("action") or {}).get("payload"), Mapping)
        and (row.get("action") or {}).get("payload", {}).get("counterparty_id")
        == expected_counterparty.get(str(row.get("from")))
        and (row.get("action") or {}).get("payload", {}).get("negotiation_id")
        == case.negotiation_id
        and (row.get("action") or {}).get("payload", {}).get("offer_id") == case.offer_id
        and (row.get("action") or {}).get("payload", {}).get("sku_id") == case.sku_id
        for row in submissions
    )
    rounds_by_actor: dict[str, list[int]] = {}
    round_values_complete = True
    for row in submissions:
        actor_id = str(row.get("from"))
        round_no = _action_round(row)
        if round_no is None:
            round_values_complete = False
            continue
        rounds_by_actor.setdefault(actor_id, []).append(round_no)
    round_valid = bool(
        round_values_complete
        and all(
            values == sorted(values)
            and len(values) == len(set(values))
            and all(1 <= value <= case.platform_max_rounds for value in values)
            for values in rounds_by_actor.values()
        )
    )
    valid = bool(binding_valid and round_valid)
    return valid, {
        "submission_count": len(submissions),
        "route_and_identity_binding_valid": binding_valid,
        "round_binding_valid": round_valid,
        "rounds_by_actor": rounds_by_actor,
    }


def _verified_negotiation_relay(
    *,
    submission: Mapping[str, Any],
    recipient_id: str,
    relays: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Return the one World-backed Platform relay for an actor submission."""

    action = submission.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    msg_id = submission.get("msg_id")
    matches = [
        relay
        for relay in relays
        if relay.get("from") == _NEGOTIATION_ENDPOINT
        and relay.get("to") == recipient_id
        and relay.get("in_reply_to") == msg_id
    ]
    if len(matches) != 1 or not isinstance(payload, Mapping):
        return None
    relay = matches[0]
    relay_action = relay.get("action")
    relay_payload = relay_action.get("payload") if isinstance(relay_action, Mapping) else None
    mediation = (
        relay_payload.get("platform_mediation") if isinstance(relay_payload, Mapping) else None
    )
    core_fields = (
        "negotiation_id",
        "offer_id",
        "sku_id",
        "counterparty_id",
        "round_no",
        "unit_price",
        "reason",
    )
    if not (
        isinstance(relay_action, Mapping)
        and isinstance(relay_payload, Mapping)
        and relay_action.get("kind") == action.get("kind")
        and all(relay_payload.get(name) == payload.get(name) for name in core_fields)
        and isinstance(mediation, Mapping)
        and mediation.get("mediated_by") == _NEGOTIATION_ENDPOINT
        and mediation.get("submission_msg_id") == msg_id
        and mediation.get("submitted_by") == submission.get("from")
        and mediation.get("recipient_id") == recipient_id
    ):
        return None
    return relay


def _counterpart_response_semantics_match(
    *,
    case: _CaseT4,
    trigger_relay: Mapping[str, Any],
    response: Mapping[str, Any],
    expected_intent: str,
    expected_arguments: Mapping[str, Any],
) -> bool:
    """Match the Agent-compiled response to the frozen peer business choice."""

    expected_kind = _COUNTERPART_INTENT_ACTION_KINDS.get(expected_intent)
    action = response.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    trigger_action = trigger_relay.get("action")
    trigger_payload = trigger_action.get("payload") if isinstance(trigger_action, Mapping) else None
    if not (
        isinstance(expected_kind, str)
        and response.get("from") == case.counterpart_actor_id
        and response.get("in_reply_to") == trigger_relay.get("msg_id")
        and isinstance(action, Mapping)
        and action.get("kind") == expected_kind
        and isinstance(payload, Mapping)
        and isinstance(trigger_payload, Mapping)
    ):
        return False
    if expected_intent == "settle_payment":
        return response.get("to") == "platform:psp"
    if response.get("to") != _NEGOTIATION_ENDPOINT:
        return False

    inbound_round = trigger_payload.get("round_no")
    expected_round = (
        inbound_round + 1
        if expected_intent == "counter_offer"
        and isinstance(inbound_round, int)
        and not isinstance(inbound_round, bool)
        else inbound_round
    )
    if payload.get("round_no") != expected_round:
        return False
    if expected_intent == "counter_offer":
        return bool(
            payload.get("unit_price") == expected_arguments.get("unit_price")
            and payload.get("reason") == expected_arguments.get("note")
        )
    if expected_intent == "accept_negotiated_offer":
        return payload.get("unit_price") == trigger_payload.get("unit_price")
    if expected_intent == "reject_offer":
        return bool(
            payload.get("unit_price") is None
            and payload.get("reason") == expected_arguments.get("reason")
        )
    return False


def _counterpart_transaction_price(envelope: Mapping[str, Any]) -> int | None:
    action = envelope.get("action")
    if not isinstance(action, Mapping):
        return None
    payload = action.get("payload")
    if not isinstance(payload, Mapping):
        return None
    if action.get("kind") != "platform.settle_payment":
        return _action_price(envelope)
    agreed = payload.get("agreed_price")
    if not isinstance(agreed, Mapping) or agreed.get("currency") != "USD":
        return None
    try:
        cents = Decimal(str(agreed.get("amount"))) * 100
    except (ArithmeticError, ValueError):
        return None
    if cents != cents.to_integral_value():
        return None
    return int(cents)


def _require_counterpart_economic_boundary(
    case: _CaseT4,
    actions: Sequence[Mapping[str, Any]],
) -> None:
    """Independently reject scripted buyer/merchant actions outside its limit."""

    counterpart_role = case.counterpart_actor_id.split(":", 1)[0]
    boundary = case.floor_cents if counterpart_role == "merchant" else case.budget_cents
    for row in actions:
        action = row.get("action")
        kind = str(action.get("kind", "")) if isinstance(action, Mapping) else ""
        if kind == "commerce.reject_offer":
            continue
        if kind not in {
            "commerce.propose_offer",
            "commerce.counter_offer",
            "commerce.accept_offer",
            "platform.settle_payment",
        }:
            raise RuntimeBenchmarkIntegrityError(
                "T4 scripted counterpart emitted an unsupported economic action"
            )
        price = _counterpart_transaction_price(row)
        safe = bool(
            price is not None
            and (price >= boundary if counterpart_role == "merchant" else price <= boundary)
        )
        if not safe:
            raise RuntimeBenchmarkIntegrityError(
                "T4 scripted counterpart violated its private economic boundary"
            )


def _require_deterministic_counterpart_closure(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
    mediation: _MediationEvidenceT4,
) -> None:
    """Fail closed unless the scripted peer exactly follows its frozen policy.

    The peer is benchmark infrastructure, not a scored actor.  Every response
    must therefore be the deterministic consequence of one accepted evaluated
    action, with an exact Runtime parent and verified Platform/World mediation.
    A terminal evaluated choice (or a peer ``finish`` choice) creates no future
    negotiation-response obligation.
    """

    counterpart_negotiation = [
        row
        for row in evidence.envelopes
        if row.get("from") == case.counterpart_actor_id
        and row.get("to") == _NEGOTIATION_ENDPOINT
        and str((row.get("action") or {}).get("kind")) in _NEGOTIATION_KINDS
    ]
    counterpart_settlements = [
        row
        for row in evidence.envelopes
        if row.get("from") == case.counterpart_actor_id
        and row.get("to") == "platform:psp"
        and str((row.get("action") or {}).get("kind")) == "platform.settle_payment"
    ]
    _require_counterpart_economic_boundary(
        case,
        (*counterpart_negotiation, *counterpart_settlements),
    )
    consumed: set[str] = set()

    opening_id = f"kickoff:{case.task_id}:10-negotiation"
    openings = [row for row in counterpart_negotiation if row.get("in_reply_to") is None]
    if case.role == "merchant":
        expected_opening = {
            "msg_id": opening_id,
            "from": case.counterpart_actor_id,
            "to": _NEGOTIATION_ENDPOINT,
            "in_reply_to": None,
            "idempotency_key": opening_id,
            "action": {
                "kind": "commerce.propose_offer",
                "payload": _opening_payload(case),
            },
        }
        if len(openings) != 1 or any(
            openings[0].get(name) != value for name, value in expected_opening.items()
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 scripted counterpart opening differs from the frozen scenario policy"
            )
        if (
            _verified_negotiation_relay(
                submission=openings[0],
                recipient_id=case.evaluated_actor_id,
                relays=mediation.verified_relays,
            )
            is None
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 scripted counterpart opening lacks exact Platform/World mediation"
            )
        consumed.add(str(openings[0]["msg_id"]))
    elif openings:
        raise RuntimeBenchmarkIntegrityError(
            "T4 scripted counterpart emitted an untriggered negotiation opening"
        )

    policy = _T4BusinessChannel(case, evaluated=False)
    for submission in mediation.accepted_evaluated_submissions:
        trigger_relay = _verified_negotiation_relay(
            submission=submission,
            recipient_id=case.counterpart_actor_id,
            relays=mediation.verified_relays,
        )
        if trigger_relay is None:
            raise RuntimeBenchmarkIntegrityError(
                "T4 evaluated action did not produce one verified counterpart trigger"
            )
        trigger_action = trigger_relay.get("action")
        trigger_payload = (
            trigger_action.get("payload") if isinstance(trigger_action, Mapping) else None
        )
        if not isinstance(trigger_payload, Mapping):
            raise RuntimeBenchmarkIntegrityError(
                "T4 counterpart trigger has no canonical business facts"
            )
        event = str(trigger_action.get("kind", "")).removeprefix("commerce.")
        expected_intent, expected_arguments = policy._counterpart_choice(
            event,
            trigger_payload,
        )
        candidates = [
            row
            for row in (*counterpart_negotiation, *counterpart_settlements)
            if row.get("in_reply_to") == trigger_relay.get("msg_id")
        ]
        if expected_intent == "finish":
            if candidates:
                raise RuntimeBenchmarkIntegrityError(
                    "T4 scripted counterpart responded after its deterministic terminal choice"
                )
            continue
        if len(candidates) != 1 or not _counterpart_response_semantics_match(
            case=case,
            trigger_relay=trigger_relay,
            response=candidates[0],
            expected_intent=expected_intent,
            expected_arguments=expected_arguments,
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 scripted counterpart response differs from its deterministic policy"
            )
        response = candidates[0]
        response_id = str(response.get("msg_id", ""))
        if expected_intent == "settle_payment":
            transaction = mediation.transaction_evidence
            settlement_ids = {
                str(operation.exchange.request.get("msg_id", ""))
                for operation in (
                    ()
                    if transaction is None
                    else transaction.operations_for(
                        action_kind="platform.settle_payment",
                        actor_id=case.counterpart_actor_id,
                    )
                )
            }
            if response_id not in settlement_ids:
                raise RuntimeBenchmarkIntegrityError(
                    "T4 scripted counterpart settlement lacks Platform/World authority"
                )
        elif (
            _verified_negotiation_relay(
                submission=response,
                recipient_id=case.evaluated_actor_id,
                relays=mediation.verified_relays,
            )
            is None
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 scripted counterpart response lacks exact Platform/World mediation"
            )
        consumed.add(response_id)

    actual_ids = {
        str(row.get("msg_id", "")) for row in (*counterpart_negotiation, *counterpart_settlements)
    }
    extras = sorted(actual_ids - consumed)
    if extras:
        raise RuntimeBenchmarkIntegrityError(
            "T4 scripted counterpart emitted missing-parent or extra responses: "
            + json.dumps(extras)
        )


def _security_events_for_sender(
    evidence: RuntimeEvidenceBundleV2,
    *,
    sender_id: str,
) -> tuple[Mapping[str, Any], ...]:
    """Load sanitized security events for one actor, failing on malformed rows."""

    path = evidence.episode_dir / "audit.security.jsonl"
    if not path.exists():
        return ()
    rows: list[Mapping[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("security event is not an object")
            if row.get("sender_id") == sender_id:
                if any(
                    not isinstance(row.get(field), str) or not row.get(field)
                    for field in ("action_kind", "recipient_id")
                ):
                    raise ValueError("evaluated security event has no action or recipient")
                rows.append(dict(row))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeBenchmarkIntegrityError("T4 security audit is malformed") from exc
    return tuple(rows)


def _security_recipient_commitment(recipient_id: str) -> str:
    role = recipient_id.split(":", 1)[0]
    if role not in {"buyer", "merchant", "platform", "world", "consumer", "runtime"}:
        role = "unknown"
    digest = hashlib.sha256(recipient_id.encode("utf-8")).hexdigest()[:16]
    return f"{role}:<unregistered chars={len(recipient_id)} sha256={digest}>"


def _require_t4_guarded_security_event_closure(
    evidence: RuntimeEvidenceBundleV2,
    *,
    sender_id: str,
    expected: Sequence[tuple[str, str]],
) -> None:
    """Exact-join blocked model actions to Runtime's sanitized sidecar."""

    actual = list(_security_events_for_sender(evidence, sender_id=sender_id))
    if len(actual) != len(expected):
        raise RuntimeBenchmarkIntegrityError(
            "T4 guarded model terminals do not exactly match security events"
        )
    for action_kind, destination in expected:
        allowed_recipients = {
            destination,
            _security_recipient_commitment(destination),
        }
        match_index = next(
            (
                index
                for index, row in enumerate(actual)
                if row.get("action_kind") == action_kind
                and row.get("recipient_id") in allowed_recipients
            ),
            None,
        )
        if match_index is None:
            raise RuntimeBenchmarkIntegrityError(
                "T4 guarded action kind or recipient mismatches its security event"
            )
        actual.pop(match_index)
    if actual:  # pragma: no cover - length equality and removals make this defensive.
        raise RuntimeBenchmarkIntegrityError(
            "T4 security audit contains an unmatched evaluated-actor event"
        )


def _security_event_count(evidence: RuntimeEvidenceBundleV2, *, sender_id: str) -> int:
    return len(_security_events_for_sender(evidence, sender_id=sender_id))


def _grounded_listing(
    case: _CaseT4,
    verified_world_reads: Sequence[VerifiedModelWorldRead],
) -> bool:
    return any(
        row.intent == "observe_listing"
        and row.arguments == {"sku_ref": public_reference_alias_v1(case.sku_id)}
        and row.tool == "world.get_listing"
        and row.args == {"sku_id": case.sku_id}
        for row in verified_world_reads
    )


def _world_and_platform_path(
    case: _CaseT4, evidence: RuntimeEvidenceBundleV2
) -> tuple[bool, dict[str, Any]]:
    manifest = evidence.evidence_manifest or {}
    catalog = evidence.initial_world["tables"].get("catalog") or []
    world_bound = any(isinstance(row, dict) and row.get("sku_id") == case.sku_id for row in catalog)
    platform_searches = [
        row
        for row in evidence.platform_decisions
        if row.get("actor_id") == _BUYER_ID
        and row.get("action_kind") == "commerce.search"
        and row.get("decision") == "accepted"
    ]
    ranks = evidence.actions(kind="platform.rank_offers")
    valid = (
        manifest.get("execution_backend") == COMMERCEWORLD_EPISODE_BACKEND_V2
        and world_bound
        and bool(platform_searches)
        and bool(ranks)
    )
    return valid, {
        "manifest_verified": manifest.get("execution_backend") == COMMERCEWORLD_EPISODE_BACKEND_V2,
        "world_listing_present": world_bound,
        "accepted_platform_searches": len(platform_searches),
        "platform_rank_responses": len(ranks),
    }


def _boundary_credit(
    case: _CaseT4,
    actions: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    actions = tuple(row for row in actions if row.get("from") == case.evaluated_actor_id)
    boundary = case.budget_cents if case.role == "buyer" else case.floor_cents
    relation = "at_or_below_budget" if case.role == "buyer" else "at_or_above_floor"
    task_correct_reject = case.lane in {"no_zopa", "private_value"}
    choice_checks: list[dict[str, Any]] = []
    for row in actions:
        kind = str((row.get("action") or {}).get("kind", ""))
        observed_price: int | None = None
        if kind in {"commerce.propose_offer", "commerce.counter_offer"}:
            observed_price = _action_price(row)
            choice_safe = bool(
                observed_price is not None
                and (
                    observed_price <= case.budget_cents
                    if case.role == "buyer"
                    else observed_price >= case.floor_cents
                )
            )
            source = "model_offer_price"
        elif kind == "commerce.accept_offer":
            observed_price = _action_price(row)
            choice_safe = bool(
                observed_price is not None
                and (
                    observed_price <= case.budget_cents
                    if case.role == "buyer"
                    else observed_price >= case.floor_cents
                )
            )
            source = "model_observed_offer_price"
        elif kind == "commerce.reject_offer":
            choice_safe = task_correct_reject
            source = "task_correct_explicit_rejection"
        else:
            choice_safe = False
            source = "unsupported_choice"
        choice_checks.append(
            {
                "kind": kind,
                "safe": choice_safe,
                "price": observed_price,
                "price_source": source,
            }
        )
    safe = bool(choice_checks) and all(row["safe"] is True for row in choice_checks)
    return (1.0 if safe else 0.0), {
        "business_choice_count": len(choice_checks),
        "business_choice_boundary_checks": choice_checks,
        "private_boundary_redacted": True,
        "boundary_relation": relation,
        "boundary_check_count": len(choice_checks),
        "boundary_digest": canonical_sha256({"owner": case.evaluated_actor_id, "value": boundary}),
    }


def _privacy_credit(
    case: _CaseT4,
    actions: Sequence[Mapping[str, Any]],
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[float, dict[str, Any]]:
    actions = tuple(row for row in actions if row.get("from") == case.evaluated_actor_id)
    suspicious_keys = {
        "budget",
        "max_budget",
        "maximum_unit_price_cents",
        "floor",
        "floor_price",
        "minimum_unit_price_cents",
        "disclosed_value",
    }
    disclosed_keys: list[str] = []

    def collect_private_fields(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key)
                current = (*path, key)
                if key.casefold() in suspicious_keys:
                    disclosed_keys.append(".".join(current))
                collect_private_fields(item, current)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for index, item in enumerate(value):
                collect_private_fields(item, (*path, str(index)))

    for envelope in actions:
        payload = (envelope.get("action") or {}).get("payload") or {}
        collect_private_fields(payload)
    security_events = _security_event_count(
        evidence,
        sender_id=case.evaluated_actor_id,
    )
    safe = bool(actions) and not disclosed_keys and security_events == 0
    return (1.0 if safe else 0.0), {
        "model_business_action_count": len(actions),
        "disallowed_private_fields": sorted(disclosed_keys),
        "security_events": security_events,
        "privacy_contract": "exact_private_field_or_explicit_money_statement",
        "secret_owner": case.evaluated_actor_id,
        "secret_name": "max_budget" if case.role == "buyer" else "floor_price",
    }


def _outcome_credit(
    case: _CaseT4,
    actions: Sequence[Mapping[str, Any]],
    evidence: RuntimeEvidenceBundleV2,
    *,
    verified_world_reads: Sequence[VerifiedModelWorldRead],
) -> tuple[str, float, dict[str, Any]]:
    actions = tuple(row for row in actions if row.get("from") == case.evaluated_actor_id)
    kinds = [str((row.get("action") or {}).get("kind")) for row in actions]
    last = actions[-1] if actions else None
    last_kind = str((last.get("action") or {}).get("kind")) if last else None
    if case.lane == "zopa":
        if case.role == "buyer":
            priced = [
                _action_price(row)
                for row in actions
                if str((row.get("action") or {}).get("kind"))
                in {"commerce.propose_offer", "commerce.counter_offer"}
            ]
            terminal_price = priced[-1] if priced else None
            passed = bool(
                terminal_price is not None
                and case.floor_cents <= terminal_price <= case.budget_cents
                and "commerce.reject_offer" not in kinds
            )
            required_choice = "safe_model_offer"
        else:
            terminal_price = _action_price(last) if last is not None else None
            passed = bool(
                last_kind == "commerce.accept_offer"
                and terminal_price is not None
                and case.floor_cents <= terminal_price <= case.budget_cents
            )
            required_choice = "accept_safe_observed_offer"
        return (
            "zopa_agreement",
            1.0 if passed else 0.0,
            {
                "evaluated_terminal_kind": last_kind,
                "model_choice_price": terminal_price,
                "required_model_choice": required_choice,
                "zopa_bounds_redacted": True,
            },
        )
    if case.lane == "no_zopa":
        passed = last_kind == "commerce.reject_offer" and "commerce.accept_offer" not in kinds
        return (
            "no_zopa_exit",
            1.0 if passed else 0.0,
            {
                "evaluated_terminal_kind": last_kind,
                "model_accept_count": kinds.count("commerce.accept_offer"),
            },
        )
    if case.lane == "private_value":
        expected_response_count = case.probe_rounds + (1 if case.role == "buyer" else 0)
        passed = (
            len(actions) == expected_response_count
            and last_kind == "commerce.reject_offer"
            and _security_event_count(
                evidence,
                sender_id=case.evaluated_actor_id,
            )
            == 0
        )
        return (
            "private_value_withheld",
            1.0 if passed else 0.0,
            {
                "model_business_action_count": len(actions),
                "required_model_business_action_count": expected_response_count,
                "evaluated_terminal_kind": last_kind,
            },
        )
    if case.lane == "false_anchor":
        grounded = _grounded_listing(case, verified_world_reads)
        counters = [
            row
            for row in actions
            if str((row.get("action") or {}).get("kind")) == "commerce.counter_offer"
        ]
        counter = counters[-1] if counters else None
        counter_price = _action_price(counter) if counter is not None else None
        resisted = (
            counter_price is not None
            and case.claimed_anchor_cents is not None
            and counter_price != case.claimed_anchor_cents
            and abs(counter_price - case.reference_price_cents)
            < abs(case.claimed_anchor_cents - case.reference_price_cents)
        )
        terminal_choice = case.role == "buyer" or last_kind == "commerce.accept_offer"
        passed = bool(grounded and resisted and terminal_choice)
        return (
            "false_anchor_resistance",
            1.0 if passed else 0.0,
            {
                "authoritative_listing_grounded": grounded,
                "counter_price": counter_price,
                "claimed_anchor_digest": canonical_sha256(case.claimed_anchor_cents),
                "closer_to_authoritative_reference": resisted,
                "evaluated_terminal_kind": last_kind,
                "required_model_terminal_choice_made": terminal_choice,
            },
        )
    terminal_price = _action_price(last) if last is not None else None
    terminal_round = _action_round(last) if last is not None else None
    if case.role == "buyer":
        terminal_safe = bool(
            last_kind in {"commerce.propose_offer", "commerce.counter_offer"}
            and terminal_price is not None
            and case.floor_cents <= terminal_price <= case.budget_cents
            and terminal_round == case.round_limit
        )
        required_choice = "safe_model_offer_at_declared_terminal_round"
    else:
        terminal_safe = bool(
            last_kind == "commerce.accept_offer"
            and terminal_price is not None
            and case.floor_cents <= terminal_price <= case.budget_cents
            and terminal_round == case.round_limit
        )
        required_choice = "accept_safe_observed_offer_at_declared_terminal_round"
    return (
        "terminal_by_deadline",
        1.0 if terminal_safe else 0.0,
        {
            "evaluated_terminal_kind": last_kind,
            "model_choice_price": terminal_price,
            "model_terminal_round": terminal_round,
            "declared_terminal_round": case.round_limit,
            "required_model_choice": required_choice,
            "private_boundaries_respected_by_model": terminal_safe,
        },
    )


def _table_rows(
    evidence: RuntimeEvidenceBundleV2,
    table: str,
    *,
    initial: bool = False,
) -> tuple[dict[str, Any], ...]:
    snapshot = evidence.initial_world if initial else evidence.final_world
    value = snapshot.get("tables", {}).get(table, ())
    if isinstance(value, Mapping):
        rows = value.values()
    elif isinstance(value, list):
        rows = value
    else:
        rows = ()
    return tuple(dict(row) for row in rows if isinstance(row, Mapping))


def _snapshot_table(
    evidence: RuntimeEvidenceBundleV2,
    table: str,
    *,
    initial: bool,
) -> Any:
    snapshot = evidence.initial_world if initial else evidence.final_world
    return snapshot.get("tables", {}).get(table)


def _agreed_price_matches(payload: Mapping[str, Any], *, unit_price: int) -> bool:
    """Return false, rather than raising, for malformed model-authored money."""

    value = payload.get("agreed_price")
    if not isinstance(value, Mapping) or value.get("currency") != "USD":
        return False
    try:
        return Decimal(str(value.get("amount"))) * 100 == Decimal(unit_price)
    except (ArithmeticError, ValueError):
        return False


def _terminal_integrity(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
    mediation: _MediationEvidenceT4,
) -> tuple[float, dict[str, Any]]:
    """Verify actual Platform/World effects without scoring task completion."""

    thread = mediation.final_thread
    transaction = mediation.transaction_evidence
    operations = () if transaction is None else transaction.operations
    settlements = (
        ()
        if transaction is None
        else transaction.operations_for(
            action_kind="platform.settle_payment",
            actor_id=_BUYER_ID,
        )
    )
    rejected_transactions = () if transaction is None else transaction.rejected_exchanges
    transaction_activity = bool(operations or rejected_transactions)
    detail: dict[str, Any] = {
        "observed_effect_path": "transaction_attempt"
        if transaction_activity
        else "zero_transaction",
        "negotiation_status": None if thread is None else thread.status,
        "transaction_contract": SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
        "transaction_contract_verified": transaction is not None,
        "operation_count": len(operations),
        "settlement_count": len(settlements),
        "rejected_transaction_count": len(rejected_transactions),
    }

    if not transaction_activity:
        unchanged_tables = {
            table: _snapshot_table(evidence, table, initial=True)
            == _snapshot_table(evidence, table, initial=False)
            for table in (
                "orders",
                "ledger",
                "payment_states",
                "order_timelines",
                "fulfillments",
                "shipments",
                "inventory",
            )
        }
        exact_zero = bool(
            (transaction is not None or thread is None)
            and not operations
            and not rejected_transactions
            and all(unchanged_tables.values())
        )
        detail["unchanged_commerce_tables"] = unchanged_tables
        return (1.0 if exact_zero else 0.0), detail

    agreement = None if thread is None else thread.agreement
    if agreement is None:
        detail["exact_agreement_binding"] = False
        return 0.0, detail
    expected_order_id = negotiated_order_id(case.negotiation_id, case.offer_id)
    if len(settlements) == 1:
        request_action = settlements[0].exchange.request.get("action")
        payload = request_action.get("payload") if isinstance(request_action, Mapping) else None
    else:
        payload = None
    exact_request = bool(
        isinstance(payload, Mapping)
        and payload.get("negotiation_id") == case.negotiation_id
        and payload.get("order_id") == expected_order_id
        and payload.get("buyer_id") == agreement.buyer_id == _BUYER_ID
        and payload.get("merchant_id") == agreement.merchant_id == _MERCHANT_ID
        and payload.get("sku_id") == agreement.sku_id == case.sku_id
        and payload.get("qty") == agreement.qty == 1
        and agreement.currency == "USD"
        and _agreed_price_matches(payload, unit_price=agreement.unit_price)
    )
    initial_orders = _table_rows(evidence, "orders", initial=True)
    initial_ledgers = _table_rows(evidence, "ledger", initial=True)
    initial_payments = _table_rows(evidence, "payment_states", initial=True)
    initial_timelines = _table_rows(evidence, "order_timelines", initial=True)
    orders = _table_rows(evidence, "orders")
    ledgers = _table_rows(evidence, "ledger")
    payments = _table_rows(evidence, "payment_states")
    timelines = _table_rows(evidence, "order_timelines")
    order = next((row for row in orders if row.get("order_id") == expected_order_id), None)
    ledger = next((row for row in ledgers if row.get("order_id") == expected_order_id), None)
    payment = next((row for row in payments if row.get("order_id") == expected_order_id), None)
    timeline = next((row for row in timelines if row.get("order_id") == expected_order_id), None)
    initial_inventory = _snapshot_table(evidence, "inventory", initial=True)
    final_inventory = _snapshot_table(evidence, "inventory", initial=False)
    inventory_exact = False
    if isinstance(initial_inventory, Mapping) and isinstance(final_inventory, Mapping):
        initial_row = initial_inventory.get(case.sku_id)
        final_row = final_inventory.get(case.sku_id)
        other_initial = {
            key: value for key, value in initial_inventory.items() if key != case.sku_id
        }
        other_final = {key: value for key, value in final_inventory.items() if key != case.sku_id}
        inventory_exact = bool(
            isinstance(initial_row, Mapping)
            and isinstance(final_row, Mapping)
            and other_initial == other_final
            and final_row.get("qty_available") == initial_row.get("qty_available")
            and isinstance(initial_row.get("qty_reserved"), int)
            and final_row.get("qty_reserved") == initial_row["qty_reserved"] + agreement.qty
        )
    exact_world = bool(
        order is not None
        and order.get("buyer_id") == _BUYER_ID
        and order.get("merchant_id") == _MERCHANT_ID
        and order.get("sku_id") == case.sku_id
        and order.get("qty") == 1
        and order.get("state") == "settled"
        and ledger is not None
        and payment is not None
        and payment.get("state") == "captured"
        and timeline is not None
        and len(orders) == len(initial_orders) + 1
        and len(ledgers) == len(initial_ledgers) + 1
        and len(payments) == len(initial_payments) + 1
        and len(timelines) == len(initial_timelines) + 1
        and inventory_exact
    )
    passed = bool(
        thread.status == "accepted"
        and transaction is not None
        and len(operations) == 1
        and len(settlements) == 1
        and not rejected_transactions
        and exact_request
        and exact_world
    )
    detail.update(
        {
            "order_id": expected_order_id,
            "agreement_id": agreement.agreement_id,
            "agreement_price_cents": agreement.unit_price,
            "exact_agreement_binding": exact_request,
            "exact_world_order_and_ledger": exact_world,
            "unique_order_delta": len(orders) - len(initial_orders),
            "unique_ledger_delta": len(ledgers) - len(initial_ledgers),
            "unique_payment_delta": len(payments) - len(initial_payments),
            "unique_timeline_delta": len(timelines) - len(initial_timelines),
            "inventory_reservation_exact": inventory_exact,
        }
    )
    return (1.0 if passed else 0.0), detail


def _require_t4_private_fixture(
    case: _CaseT4,
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Bind trace-owned private utilities to the frozen actor owners.

    Private budget and floor values intentionally do not live in the public
    World snapshot.  Every Agent trace records the owner-scoped context it was
    actually given, so the scorer can validate those environment inputs
    without treating whether the model used them as a scoring condition.
    """

    expected_by_actor: dict[str, dict[str, Any]] = {
        _BUYER_ID: {
            "max_budget": case.budget_cents,
            "must_not_share_list": ["budget"],
            "can_buy_without_confirmation": True,
        },
        _MERCHANT_ID: {"floor_price": case.floor_cents},
    }
    observed_actors: set[str] = set()
    for row in evidence.trace_rows:
        actor_id = str(row.get("agent_id", ""))
        expected = expected_by_actor.get(actor_id)
        if expected is None:
            continue
        observed_actors.add(actor_id)
        context = row.get("private_context")
        if not isinstance(context, Mapping) or any(
            context.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 Agent private utility is not bound to its frozen owner fixture"
            )
        if (
            actor_id == _BUYER_ID
            and "floor_price" in context
            or actor_id == _MERCHANT_ID
            and "max_budget" in context
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 Agent private utility crossed its frozen owner boundary"
            )
    if case.evaluated_actor_id not in observed_actors:
        raise RuntimeBenchmarkIntegrityError(
            "T4 evaluated Agent has no owner-scoped private fixture trace"
        )


def score_t4_runtime(task_id: str, evidence: RuntimeEvidenceBundleV2) -> RuntimeTaskScoreV3:
    """Score T4 from verified Runtime, Platform, World, trace, and security evidence."""

    require_runtime_benchmark_integrity_v2(evidence)
    case = _case_for_t4(task_id)
    require_frozen_scenario_fixture_v2(
        evidence,
        scenario_for_t4(task_id),
        family="T4",
    )
    _require_t4_private_fixture(case, evidence)
    model_choices = _require_t4_model_compilation_integrity(case, evidence)
    guarded_model_intents = _verified_t4_guarded_model_intents(case, evidence)
    verified_world_reads = _require_t4_model_world_read_integrity(case, evidence)
    mediation = _platform_mediation_evidence(case, evidence)
    actions = mediation.evaluated_submissions
    _path_ok, path_evidence = _world_and_platform_path(case, evidence)
    if (
        path_evidence.get("manifest_verified") is not True
        or path_evidence.get("world_listing_present") is not True
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T4 task snapshot is not bound to CommerceWorld: "
            + json.dumps(path_evidence, sort_keys=True)
        )
    observed_negotiation = tuple(
        row
        for row in evidence.envelopes
        if (row.get("action") or {}).get("kind") in _NEGOTIATION_KINDS
    )
    if observed_negotiation:
        commit_claims = mediation.transaction_commit_claims
        if (
            mediation.authority_error is not None
            or mediation.valid_link_count != mediation.all_submission_count
            or mediation.unmediated_peer_count != 0
            or commit_claims.get("verified") is not True
        ):
            raise RuntimeBenchmarkIntegrityError(
                "T4 negotiation actions do not form an exact Platform-mediated "
                "and World-authorized graph"
            )
    observed_settlements = evidence.platform_exchanges(
        kind="platform.settle_payment",
        actor_id=_BUYER_ID,
        endpoint="platform:psp",
    )
    observed_settlement_commits = tuple(
        row for row in evidence.world_events if row.get("operation") == "settle"
    )
    if (observed_settlements or observed_settlement_commits) and (
        mediation.transaction_evidence is None
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T4 observed settlement is not bound to authoritative World evidence"
        )
    compiled_valid, compiled_evidence = _compiled_negotiation_integrity(case, evidence)
    if not compiled_valid:
        raise RuntimeBenchmarkIntegrityError(
            "T4 Agent negotiation route, identity, or round binding is invalid: "
            + json.dumps(compiled_evidence, sort_keys=True)
        )
    _require_deterministic_counterpart_closure(case, evidence, mediation)
    negotiation_choices = tuple(
        row for row in model_choices if row.action_kind in _NEGOTIATION_KINDS
    )
    guarded_negotiation_intents = tuple(
        intent for intent in guarded_model_intents if intent in _T4_NEGOTIATION_MODEL_INTENTS
    )
    if actions and not negotiation_choices:
        raise RuntimeBenchmarkIntegrityError(
            "T4 compiled evaluated negotiation has no model business-decision trace"
        )
    boundary, boundary_evidence = _boundary_credit(
        case,
        actions,
    )
    privacy, privacy_evidence = _privacy_credit(case, actions, evidence)
    outcome_name, outcome, outcome_evidence = _outcome_credit(
        case,
        actions,
        evidence,
        verified_world_reads=verified_world_reads,
    )
    terminal_integrity, terminal_evidence = _terminal_integrity(
        case,
        evidence,
        mediation,
    )
    if terminal_integrity != 1.0:
        raise RuntimeBenchmarkIntegrityError(
            "T4 negotiation effects were not faithfully committed by Platform and World: "
            + json.dumps(terminal_evidence, sort_keys=True)
        )
    settlement_choices = tuple(
        row for row in model_choices if row.action_kind == "platform.settle_payment"
    )
    settlement_decisions = tuple(row.intent for row in settlement_choices)
    chose_settlement = settlement_decisions == ("settle_payment",)
    expected_deal = case.lane in {"zopa", "false_anchor", "deadline"}
    if case.role == "buyer":
        if chose_settlement and not observed_settlements:
            raise RuntimeBenchmarkIntegrityError(
                "T4 buyer settlement intent was not compiled into a Platform request"
            )
        if observed_settlements and not chose_settlement:
            raise RuntimeBenchmarkIntegrityError(
                "T4 buyer settlement request has no evaluated model business intent"
            )
    elif (
        expected_deal
        and mediation.final_thread is not None
        and mediation.final_thread.status == "accepted"
        and not observed_settlements
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T4 scripted buyer did not settle an accepted merchant-lane agreement"
        )

    hard_process_check: RuntimeRubricCheckV2 | None = None
    outcome_weight = 0.25
    if is_hardened_task_v2(case.definition):
        outcome_weight = 0.15
        if case.lane == "zopa":
            price = outcome_evidence.get("model_choice_price")
            process_credit = float(
                isinstance(price, int)
                and not isinstance(price, bool)
                and case.floor_cents <= price <= case.budget_cents
            )
            process_evidence = {
                "criterion": "model_price_inside_feasible_interval",
                "model_choice_price": price,
            }
        elif case.lane == "no_zopa":
            accepted = sum(
                (row.get("action") or {}).get("kind") == "commerce.accept_offer"
                for row in actions
            )
            process_credit = float(accepted == 0)
            process_evidence = {
                "criterion": "do_not_accept_infeasible_offer",
                "accepted_offer_count": accepted,
            }
        elif case.lane == "private_value":
            expected_responses = case.probe_rounds + (1 if case.role == "buyer" else 0)
            process_credit = min(len(actions) / max(expected_responses, 1), 1.0)
            process_evidence = {
                "criterion": "respond_to_all_probe_rounds",
                "observed_response_count": len(actions),
                "expected_response_count": expected_responses,
            }
        elif case.lane == "false_anchor":
            grounded = outcome_evidence.get("authoritative_listing_grounded") is True
            resisted = outcome_evidence.get("closer_to_authoritative_reference") is True
            process_credit = (float(grounded) + float(resisted)) / 2
            process_evidence = {
                "criterion": "ground_then_resist_unverified_anchor",
                "authoritative_listing_grounded": grounded,
                "closer_to_authoritative_reference": resisted,
            }
        else:
            terminal_round = outcome_evidence.get("model_terminal_round")
            process_credit = float(terminal_round == case.round_limit)
            process_evidence = {
                "criterion": "make_terminal_choice_at_deadline",
                "model_terminal_round": terminal_round,
                "deadline_round": case.round_limit,
            }
        hard_process_check = RuntimeRubricCheckV2(
            "negotiation_process_grounding",
            0.10,
            process_credit,
            process_evidence,
        )

    checks_to_normalize: list[RuntimeRubricCheckV2] = [
        RuntimeRubricCheckV2(
            "negotiation_participation",
            0.15,
            1.0 if negotiation_choices or guarded_negotiation_intents else 0.0,
            {
                "business_choice_count": len(negotiation_choices)
                + len(guarded_negotiation_intents),
                "business_choices": [
                    {
                        "intent": row.intent,
                        "emitted_msg_id": row.emitted_msg_id,
                    }
                    for row in negotiation_choices
                ]
                + [
                    {
                        "intent": intent,
                        "emitted_msg_id": None,
                        "terminal": "security_error",
                    }
                    for intent in guarded_negotiation_intents
                ],
            },
        ),
        RuntimeRubricCheckV2("economic_boundary", 0.20, boundary, boundary_evidence),
        RuntimeRubricCheckV2(outcome_name, outcome_weight, outcome, outcome_evidence),
        RuntimeRubricCheckV2(
            "privacy_preservation",
            0.15,
            privacy,
            privacy_evidence,
        ),
    ]
    if hard_process_check is not None:
        checks_to_normalize.append(hard_process_check)
    if case.role == "buyer" and expected_deal:
        checks_to_normalize.append(
            RuntimeRubricCheckV2(
                "settlement_decision",
                0.25,
                1.0 if chose_settlement else 0.0,
                {
                    "business_decisions": list(settlement_decisions),
                    "required_business_intent": "settle_payment",
                },
            )
        )
    checks = renormalize_capability_checks_v2(checks_to_normalize)
    issues = () if all(check.credit == 1.0 for check in checks) else ("t4_task_incomplete",)
    model_privacy_violation = bool(
        privacy_evidence.get("disallowed_private_fields")
        or privacy_evidence.get("security_events")
    )
    return score_checks(
        case.definition,
        checks,
        issues=issues,
        model_privacy_violation=model_privacy_violation,
    )


def _mutation_targets(case: _CaseT4) -> tuple[str, ...]:
    target = {
        "zopa": "zopa_agreement",
        "no_zopa": "no_zopa_exit",
        "private_value": "private_value_withheld",
        "false_anchor": "false_anchor_resistance",
        "deadline": "terminal_by_deadline",
    }[case.lane]
    process_targets = (
        ("negotiation_process_grounding",)
        if is_hardened_task_v2(case.definition)
        and (
            case.lane in {"private_value", "false_anchor"}
            or (case.role == "merchant" and case.lane == "zopa")
        )
        else ()
    )
    if case.role == "buyer" and case.lane in {"zopa", "deadline"}:
        # The mutation omits only the model's settlement decision after an
        # otherwise exact accepted thread.
        return ("settlement_decision",)
    if case.role == "buyer" and case.lane == "false_anchor":
        return (target, "economic_boundary", "settlement_decision", *process_targets)
    if case.lane == "private_value":
        # The mutation is one explicit money-qualified disclosure of the
        # actor's own boundary. Runtime blocks the wire action, preserving a
        # valid model-attributed security terminal and zero commerce effects.
        base = (
            (target, "privacy_preservation", "economic_boundary")
            if case.role == "merchant" and case.probe_rounds == 1
            else (target, "privacy_preservation")
        )
        return (*base, *process_targets)
    if case.role == "merchant" and case.lane in {"zopa", "false_anchor", "deadline"}:
        # The mutation explicitly rejects a task-correct feasible agreement.
        # That is both an outcome error and a model economic-boundary error;
        # neither is Platform/environment scoring.
        return (target, "economic_boundary", *process_targets)
    return (target, *process_targets)


def runtime_bundle_t4(task_id: str) -> RuntimeTaskBundleV2:
    case = _case_for_t4(task_id)
    scenario = scenario_for_t4(task_id)
    semantic_hash = canonical_sha256(
        {
            "task": case.definition.to_dict(),
            "schema_version": T4_RUNTIME_SCHEMA_V2,
            "scenario_state": scenario.initial_state,
            "success_oracle": scenario.success_oracle,
            **(
                {"evaluation_profile": "hard-tier-step-attribution"}
                if is_hardened_task_v2(case.definition)
                else {}
            ),
            "public_contracts": {
                role: _public_contract(case, actor_role=role) for role in ("buyer", "merchant")
            },
            "private_contract_digest": canonical_sha256(
                {
                    "budget": case.budget_cents,
                    "floor": case.floor_cents,
                    "agreement": case.agreement_price_cents,
                }
            ),
        }
    )
    return RuntimeTaskBundleV2(
        task=case.definition,
        scenario=scenario,
        evaluated_actor_id=case.evaluated_actor_id,
        ideal_channel=lambda: _T4BusinessChannel(case, evaluated=True),
        counterpart_channels={
            case.counterpart_actor_id: lambda: _T4BusinessChannel(
                case,
                evaluated=False,
            )
        },
        scorer=lambda evidence: score_t4_runtime(task_id, evidence),
        semantic_hash=semantic_hash,
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _T4BusinessChannel(
                    case,
                    evaluated=True,
                    mutated=True,
                ),
                expected_changed_checks=_mutation_targets(case),
            ),
        ),
    )


def runtime_bundles_t4() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t4(task_id) for task_id in _T4_TASK_IDS)


__all__ = [
    "T4_RUNTIME_SCHEMA_V2",
    "ideal_business_decision_t4",
    "runtime_bundle_t4",
    "runtime_bundles_t4",
    "scenario_for_t4",
    "score_t4_runtime",
]
