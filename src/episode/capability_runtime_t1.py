"""ACWorld T1 product discovery tasks.

The fixed T1 problem data originated in the direct-simulation pilot, but this
module uses it only as immutable task content.  It never calls the direct
harness, adapter, executor, or scorer.  Every observation and action used for
formal scoring comes from a real ScenarioSpec executed by Runtime, Platform,
and World.
"""

from __future__ import annotations

import json
import hashlib
from decimal import Decimal
from typing import Any, Mapping, Sequence

from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1
from agents.agent_phase import public_reference_alias_v1
from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2
from episode.capability_runtime_t1_content import (
    HardConstraintT1,
    ProductCandidateT1,
    RuntimeTaskContentT1,
    T1_PUBLIC_OBJECTIVE_WEIGHTS,
    T1_PUBLIC_TIE_BREAK_FIELD,
    T1_SELECTION_RULE_SET_V1,
    T1_RUNTIME_TASKS,
)
from episode.capability_runtime import (
    RuntimeBenchmarkIntegrityError,
    RuntimeEvidenceBundleV2,
    RuntimeEvidenceError,
    RuntimeMutationV2,
    RuntimeRubricCheckV2,
    RuntimeTaskBundleV2,
    RuntimeTaskScoreV3,
    attested_world_catalog_reads_v2,
    canonical_sha256,
    renormalize_capability_checks_v2,
    require_runtime_benchmark_integrity_v2,
    score_checks,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from runtime.authority_operation_evidence import (
    SEARCH_SESSION_EVIDENCE_CONTRACT,
    VerifiedSearchSessionEvidence,
)
from runtime.match_certificate_evidence import (
    MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
    VerifiedMatchCertificateEvidence,
)
from runtime.supply_fulfillment_evidence import (
    SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
    VerifiedSupplyFulfillmentEvidence,
    verified_settlement_order,
)


T1_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t1.v2"
_BUYER_ID = "buyer:benchmark"
_MERCHANT_ID = "merchant:benchmark"
_T1_BATCH_CATALOG_LIMIT = 64


def _compatibility_seed(task: RuntimeTaskContentT1) -> int:
    return int(task.content_sha256[:8], 16) % 2_147_483_646 + 1


def _constraint_payload(rule: HardConstraintT1) -> dict[str, Any]:
    return {
        "constraint_id": rule.constraint_id,
        "field": rule.field,
        "operator": rule.operator,
        "value": rule.value,
        "description": rule.description,
    }


def _visible_constraint_payload(rule: HardConstraintT1) -> dict[str, Any]:
    """Render the actor-visible mandate constraint used at every decision point."""

    if rule.field == "price_cents":
        return {
            "constraint_id": rule.constraint_id,
            "field": rule.field,
            "operator": rule.operator,
            "value": rule.value,
            "description": rule.description,
            "visibility": "actor_private",
        }
    return _constraint_payload(rule)


def _public_selection_policy(task: RuntimeTaskContentT1) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "rule_set": T1_SELECTION_RULE_SET_V1,
        "selection_mode": task.selection_mode,
    }
    if task.selection_mode == "best_feasible":
        policy["objective"] = {
            "kind": "weighted_sum",
            "direction": "maximize",
            "weights": dict(T1_PUBLIC_OBJECTIVE_WEIGHTS),
        }
        policy["tie_break"] = {
            "field": T1_PUBLIC_TIE_BREAK_FIELD,
            "direction": "ascending",
        }
    return policy


def _preference_model_for_model_view(task: RuntimeTaskContentT1) -> dict[str, Any]:
    """Project the Agent-held preference model without evaluation metadata.

    Every T1 optimum is unique, so the ascending-reference tie-break never
    decides an answer.  It stays in the oracle and is withheld from the model
    view, where it would only read as grading detail.
    """

    policy = _public_selection_policy(task)
    policy.pop("tie_break", None)
    return policy


def _candidate_row(candidate: ProductCandidateT1) -> dict[str, Any]:
    return {
        "sku_id": candidate.sku_id,
        "product_id": f"product:{candidate.sku_id.rsplit(':', 1)[-1]}",
        "merchant_id": _MERCHANT_ID,
        "category": "general-merchandise",
        "name": candidate.name,
        "list_price": str(Decimal(candidate.price_cents) / Decimal(100)),
        "inventory": 3 if candidate.in_stock else 0,
        "attributes": {
            "in_stock": candidate.in_stock,
            "shipping_days": candidate.shipping_days,
            "warranty_months": candidate.warranty_months,
            "return_days": candidate.return_days,
            "energy_score": candidate.energy_score,
            "quality_score": candidate.quality_score,
            "features": list(candidate.features),
            **{feature: True for feature in candidate.features},
            "benchmark_task_id": candidate.sku_id.split(":")[-2],
        },
    }


def _budget_for(task: RuntimeTaskContentT1) -> int:
    price_caps = [
        int(rule.value)
        for rule in task.hard_constraints
        if rule.field == "price_cents" and rule.operator == "at_most"
    ]
    if price_caps:
        return min(price_caps)
    return max(candidate.price_cents for candidate in task.candidates) + 10_000


def _public_task_context(task: RuntimeTaskContentT1) -> dict[str, Any]:
    """Expose T1's legal commerce phases without serializing its answer."""

    return {
        "schema_version": T1_RUNTIME_SCHEMA_V2,
        "task_id": task.task_id,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": [
                {
                    "phase_id": "buyer_discovery",
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
                    "phase_id": "buyer_offer_selection",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["platform.rank_offers"],
                        "inbound_sender_roles": ["platform"],
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
                        {
                            "action_kind": "delegate.reject_purchase",
                            "destination": "@inbound_sender",
                        },
                    ],
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
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


def scenario_for_t1(task_id: str) -> ScenarioSpec:
    """Build one fixed T1 task as a real, one-buyer CommerceWorld scenario."""

    task = T1_RUNTIME_TASKS[task_id]
    catalog = [_candidate_row(candidate) for candidate in task.candidates]
    budget = _budget_for(task)
    mandate = {
        "mandate_id": task.task_id,
        "goal": task.prompt,
        "quantity": 1,
        "return_after_purchase": False,
        "hard_constraints": {
            "budget": budget,
            # T1's full typed predicates live in benchmark_constraints.  These
            # native fields additionally let the standard buyer skill enforce
            # the common price and delivery subset.
            "delivery_days": min(
                [
                    int(rule.value)
                    for rule in task.hard_constraints
                    if rule.field == "shipping_days" and rule.operator == "at_most"
                ]
                or [365]
            ),
            "must_have": [],
        },
        "soft_constraints": [
            {"feature": feature, "importance": len(task.soft_preferences) - index}
            for index, feature in enumerate(task.soft_preferences)
        ],
        "soft_preferences": {"style": [], "avoid": []},
        "authority": {
            "can_buy_without_confirmation": True,
            "must_not_share_with_merchant": ["budget"],
        },
        "intent_expiry": "2099-12-31T00:00:00Z",
        "task_context": _public_task_context(task),
        # The Agent-held view of the mandate.  ``goal`` already carries the
        # buyer's own words, so the instruction is not repeated here.
        "benchmark_contract": {
            "schema_version": T1_RUNTIME_SCHEMA_V2,
            "task_id": task.task_id,
            "constraints": [_visible_constraint_payload(rule) for rule in task.hard_constraints],
            "selection_policy": _preference_model_for_model_view(task),
            "optional_filters": list(task.soft_preferences),
        },
    }
    floor = max(1, min(candidate.price_cents for candidate in task.candidates) // 2)
    population = PopulationSpec(
        buyers=(
            BuyerSpec(
                buyer_id=_BUYER_ID,
                persona={"name": "Benchmark buyer", "task_family": "T1"},
                mandate=mandate,
            ),
        ),
        merchants=(
            MerchantSpec(
                merchant_id=_MERCHANT_ID,
                persona={"name": "Deterministic catalog merchant"},
                policy={
                    "floor_price": floor,
                    "margin_target_bps": 1_500,
                    "max_negotiation_rounds": 3,
                    "refund_policy": "30_day_return",
                    "claim_aggressiveness": "neutral",
                },
                catalog_scope=tuple(candidate.sku_id for candidate in task.candidates),
            ),
        ),
        matching={"top_k": max(1, len(task.candidates))},
        execution={"max_transactions_per_buyer": 1},
    )
    return ScenarioSpec(
        scenario_id=task.task_id.casefold().replace("-", "_") + "__runtime",
        seed=_compatibility_seed(task),
        initial_state={"catalog": catalog},
        buyer_goal={},
        merchant_policy={},
        allowed_actions=("search", "accept_offer", "reject_offer", "settle"),
        success_oracle={
            "schema_version": T1_RUNTIME_SCHEMA_V2,
            "task_id": task.task_id,
            "accepted_skus": list(task.acceptable_skus),
            "constraints": [_constraint_payload(rule) for rule in task.hard_constraints],
            "selection_policy": _public_selection_policy(task),
            "terminal_outcome": (
                {
                    "kind": "settled_order",
                    "acceptable_sku_ids": list(task.acceptable_skus),
                    "qty": 1,
                }
                if task.acceptable_skus
                else {"kind": "no_transaction"}
            ),
        },
        platform_policy=None,
        population=population,
    )


def _constraint_filters(
    task: RuntimeTaskContentT1,
    *,
    required_features: Sequence[str] = (),
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    key_by_rule = {
        ("shipping_days", "at_most"): "shipping_days_max",
        ("warranty_months", "at_least"): "warranty_months_min",
        ("return_days", "at_least"): "return_days_min",
        ("energy_score", "at_least"): "energy_score_min",
        ("in_stock", "equals"): "in_stock",
    }
    for rule in task.hard_constraints:
        key = key_by_rule.get((rule.field, rule.operator))
        if key is not None:
            filters[key] = rule.value
    if required_features:
        filters["required_features"] = list(required_features)
    return filters


def _business_request(user_prompt: str) -> dict[str, Any]:
    """Parse the sole provider-facing request used by scripted baselines."""

    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("T1 business prompt has no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict) or value.get("schema_version") != (
        "cwe.llm-decision-request.v1"
    ):
        raise ValueError("T1 business prompt has the wrong request schema")
    return value


def _business_response(
    request: Mapping[str, Any],
    intent: str,
    arguments: Mapping[str, Any],
) -> BusinessDecisionResponseV1:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list) or intent not in {
        row.get("intent") for row in rows if isinstance(row, Mapping)
    }:
        raise ValueError(f"T1 business intent {intent!r} is unavailable")
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
        raise ValueError("T1 business request has no current event")
    facts = first.get("facts")
    return str(first.get("event", "")), dict(facts) if isinstance(facts, Mapping) else {}


def _ranked_offer_rows(facts: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = facts.get("candidates")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return ()
    return tuple(dict(row) for row in rows)


def _offer_ref(row: Mapping[str, Any]) -> str:
    value = row.get("offer_ref")
    if not isinstance(value, str) or not value:
        raise ValueError("T1 ranked offer has no public offer reference")
    return value


def _sku_ref(row: Mapping[str, Any]) -> str:
    value = row.get("sku_ref")
    if not isinstance(value, str) or not value:
        raise ValueError("T1 ranked offer has no public SKU reference")
    return value


def _persistent_benchmark_facts(request: Mapping[str, Any]) -> dict[str, Any]:
    observations = request.get("observations")
    if not isinstance(observations, list):
        raise ValueError("T1 business request has no observations")
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        persistent = observation.get("persistent_task_business_facts")
        benchmark = persistent.get("brief") if isinstance(persistent, Mapping) else None
        if isinstance(benchmark, Mapping):
            return dict(benchmark)
    raise ValueError("T1 business request has no persistent benchmark facts")


def _provider_observed_business_facts(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    observations = request.get("observations")
    if not isinstance(observations, list):
        return ()
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        rows = observation.get("observed_business_facts")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                output.append(dict(row))
    return tuple(output)


def _provider_constraints(benchmark: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = benchmark.get("constraints")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("T1 provider facts have no complete hard constraints")
    return tuple(dict(row) for row in rows)


def _provider_search_choice(
    request: Mapping[str, Any],
    *,
    remaining_features: int,
) -> tuple[str, dict[str, Any]]:
    benchmark = _persistent_benchmark_facts(request)
    filters: dict[str, Any] = {}
    key_by_rule = {
        ("shipping_days", "at_most"): "shipping_days_max",
        ("warranty_months", "at_least"): "warranty_months_min",
        ("return_days", "at_least"): "return_days_min",
        ("energy_score", "at_least"): "energy_score_min",
        ("in_stock", "equals"): "in_stock",
    }
    for rule in _provider_constraints(benchmark):
        key = key_by_rule.get((rule.get("field"), rule.get("operator")))
        if key is not None:
            filters[key] = rule.get("value")
    optional = benchmark.get("optional_filters", [])
    if not isinstance(optional, list) or not all(isinstance(row, str) for row in optional):
        raise ValueError("T1 provider optional filters are malformed")
    features = optional[:remaining_features]
    if features:
        filters["required_features"] = features
    return "search", {"query": "", "filters": filters}


def _provider_constraint_accepts(
    constraint: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> bool:
    field = constraint.get("field")
    operator = constraint.get("operator")
    if not isinstance(field, str) or field not in attributes or "value" not in constraint:
        raise ValueError("T1 provider constraint lacks a visible comparison fact")
    observed = attributes[field]
    expected = constraint["value"]
    if operator == "at_most":
        return int(observed) <= int(expected)
    if operator == "at_least":
        return int(observed) >= int(expected)
    if operator == "equals":
        return observed == expected
    raise ValueError(f"T1 provider constraint has unsupported operator {operator!r}")


def _provider_listing_price_minor(listing: Mapping[str, Any]) -> int:
    money = listing.get("list_price")
    amount = money.get("amount") if isinstance(money, Mapping) else None
    if not isinstance(amount, str):
        raise ValueError("T1 provider listing has no public decimal price")
    whole, separator, fraction = amount.partition(".")
    if (
        not whole.isdigit()
        or (separator and (not fraction.isdigit() or len(fraction) > 2))
    ):
        raise ValueError("T1 provider listing price is malformed")
    return int(whole) * 100 + int((fraction + "00")[:2])


def _provider_objective_value(
    attributes: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> int:
    objective = policy.get("objective")
    if not isinstance(objective, Mapping) or objective.get("kind") != "weighted_sum":
        raise ValueError("T1 provider objective is missing or unsupported")
    if objective.get("direction") != "maximize":
        raise ValueError("T1 provider objective must be maximized")
    weights = objective.get("weights")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("T1 provider objective has no public weights")
    value = 0
    for field, weight in weights.items():
        if not isinstance(field, str) or isinstance(weight, bool) or not isinstance(weight, int):
            raise ValueError("T1 provider objective weight is malformed")
        observed = attributes.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise ValueError(f"T1 provider objective fact {field!r} is unavailable")
        value += observed * weight
    return value


def _provider_offer_choice(
    request: Mapping[str, Any],
    *,
    alternative: bool,
) -> tuple[str, dict[str, Any]]:
    """Solve the score-bearing T1 choice from this provider request alone."""

    _event, facts = _business_event(request)
    offers = _ranked_offer_rows(facts)
    offers_by_sku = {_sku_ref(row): row for row in offers}
    if len(offers_by_sku) != len(offers):
        raise ValueError("T1 provider ranked offers contain duplicate SKU references")

    listings: dict[str, Mapping[str, Any]] = {}
    directly_observed: set[str] = set()

    def bind_listing(sku_ref: str, result: Mapping[str, Any]) -> None:
        projected_ref = result.get("sku_ref")
        if projected_ref != sku_ref:
            raise ValueError("T1 provider listing contradicts its public SKU reference")
        prior = listings.get(sku_ref)
        if prior is not None and prior != result:
            raise ValueError("T1 provider listing facts changed within one decision")
        listings[sku_ref] = result

    for row in _provider_observed_business_facts(request):
        if row.get("observation_kind") == "catalog_search":
            result = row.get("facts")
            if not isinstance(result, list) or not all(
                isinstance(item, Mapping) for item in result
            ):
                raise ValueError("T1 provider catalog batch is malformed")
            seen_batch_refs: set[str] = set()
            for item in result:
                sku_ref = item.get("sku_ref")
                if not isinstance(sku_ref, str) or not sku_ref:
                    raise ValueError("T1 provider catalog row has no public SKU reference")
                if sku_ref in seen_batch_refs:
                    raise ValueError("T1 provider catalog batch repeats a SKU reference")
                seen_batch_refs.add(sku_ref)
                if sku_ref in offers_by_sku:
                    bind_listing(sku_ref, item)
            continue
        args = row.get("criteria")
        sku_ref = args.get("sku_ref") if isinstance(args, Mapping) else None
        if not isinstance(sku_ref, str) or sku_ref not in offers_by_sku:
            continue
        if row.get("observation_kind") == "listing":
            result = row.get("facts")
            if not isinstance(result, Mapping):
                raise ValueError("T1 provider listing read returned no object")
            bind_listing(sku_ref, result)
            directly_observed.add(sku_ref)

    missing = sorted(set(offers_by_sku) - set(listings))
    if missing:
        return "observe_listing", {"sku_ref": missing[0]}

    benchmark = _persistent_benchmark_facts(request)
    constraints = _provider_constraints(benchmark)
    policy = benchmark.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("T1 provider request has no public selection policy")
    mode = policy.get("selection_mode")
    attributes_by_sku: dict[str, dict[str, Any]] = {}
    feasible: list[str] = []
    for sku_ref, offer in offers_by_sku.items():
        listing = listings[sku_ref]
        attributes = listing.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("T1 provider listing has no public attributes")
        combined = dict(attributes)
        price = offer.get("unit_price_cents", offer.get("unit_price"))
        if isinstance(price, bool) or not isinstance(price, int):
            raise ValueError("T1 provider offer has no integer price")
        listing_price_cents = _provider_listing_price_minor(listing)
        if listing_price_cents != price:
            raise ValueError("T1 provider listing price contradicts its ranked offer")
        in_stock = combined.get("in_stock")
        if not isinstance(in_stock, bool):
            raise ValueError("T1 provider listing has no public availability fact")
        combined["price_cents"] = price
        attributes_by_sku[sku_ref] = combined
        if in_stock and all(_provider_constraint_accepts(rule, combined) for rule in constraints):
            feasible.append(sku_ref)

    selected: str | None = None
    if mode == "any_feasible":
        selected = min(feasible, default=None)
    elif mode == "best_feasible":
        # The ascending reference order below settles an exact tie on its own,
        # so no separate tie-break declaration is required from the model view.
        selected = min(
            feasible,
            key=lambda ref: (-_provider_objective_value(attributes_by_sku[ref], policy), ref),
            default=None,
        )
    elif mode != "abstain_if_none":
        raise ValueError(f"T1 provider selection mode {mode!r} is unsupported")

    if alternative:
        if mode == "best_feasible" and selected is not None:
            alternatives = sorted(ref for ref in feasible if ref != selected)
        else:
            alternatives = sorted(ref for ref in offers_by_sku if ref not in feasible)
        if not alternatives:
            raise ValueError("T1 fixture has no targeted wrong business choice")
        selected = alternatives[0]

    grounding_refs = set(offers_by_sku) if selected is None else {selected}
    missing_grounding = sorted(grounding_refs - directly_observed)
    if missing_grounding:
        return "observe_listing", {"sku_ref": missing_grounding[0]}

    if selected is None:
        return (
            "reject_purchase",
            {"reason": "No observed listing satisfies every hard requirement."},
        )
    return (
        "accept_ranked_offer",
        {
            "offer_ref": _offer_ref(offers_by_sku[selected]),
            "reason": (
                "Choose an intentionally non-optimal or infeasible observed listing."
                if alternative
                else "Choose the observed listing required by the public selection policy."
            ),
        },
    )


class _T1BusinessChannel:
    """Typed ideal/mutation policy over public business facts and refs only."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(self, task: RuntimeTaskContentT1, *, alternative: bool = False) -> None:
        del task
        self.alternative = alternative
        self._active_decision_id: str | None = None
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._terminal: tuple[str, dict[str, Any]] | None = None

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T1 business decision requires an Agent decision id")
        request = _business_request(user_prompt)
        event, facts = _business_event(request)
        if decision_id != self._active_decision_id:
            self._active_decision_id = decision_id
            self._pending = []
            self._terminal = None
            self._prepare_turn(request, event, facts)
        if self._pending:
            intent, arguments = self._pending.pop(0)
            return _business_response(request, intent, arguments)
        if self._terminal is not None:
            intent, arguments = self._terminal
        elif event == "rank_offers":
            intent, arguments = _provider_offer_choice(
                request,
                alternative=self.alternative,
            )
        else:
            raise ValueError(f"T1 business event {event!r} has no scripted choice")
        return _business_response(request, intent, arguments)

    def _prepare_turn(
        self,
        request: Mapping[str, Any],
        event: str,
        facts: Mapping[str, Any],
    ) -> None:
        if event == "create_purchase_mandate":
            benchmark = _persistent_benchmark_facts(request)
            optional = benchmark.get("optional_filters", [])
            if not isinstance(optional, list):
                raise ValueError("T1 provider optional filters are malformed")
            self._terminal = _provider_search_choice(
                request,
                remaining_features=len(optional),
            )
            return
        if event != "rank_offers":
            self._terminal = ("finish", {"reason": "No further business action is needed."})
            return

        offers = _ranked_offer_rows(facts)
        benchmark = _persistent_benchmark_facts(request)
        optional = benchmark.get("optional_filters", [])
        # An empty ranking is the buyer's own trigger to drop one wish and look
        # again.  No declared round count is needed to decide that.
        if not offers and isinstance(optional, list) and optional:
            prior_choices = next(
                (
                    observation.get("prior_validated_business_choices")
                    for observation in request.get("observations", ())
                    if isinstance(observation, Mapping)
                    and isinstance(
                        observation.get("prior_validated_business_choices"),
                        list,
                    )
                ),
                [],
            )
            completed_searches = sum(
                isinstance(choice, Mapping) and choice.get("intent") == "search"
                for choice in prior_choices
            )
            remaining = max(len(optional) - completed_searches, 0)
            self._terminal = _provider_search_choice(
                request,
                remaining_features=remaining,
            )
            return

        if offers:
            self._pending.append(
                (
                    "observe_search_catalog",
                    {"query": "", "filters": {}, "limit": _T1_BATCH_CATALOG_LIMIT},
                )
            )


class _NoReplyT1Channel:
    """Typed inert counterpart; framework-terminal turns bypass this channel."""

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


def _catalog_index(evidence: RuntimeEvidenceBundleV2) -> dict[str, dict[str, Any]]:
    rows = evidence.initial_world["tables"].get("catalog")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("sku_id")): row for row in rows if isinstance(row, dict) and row.get("sku_id")
    }


def _inventory_index(evidence: RuntimeEvidenceBundleV2) -> dict[str, dict[str, Any]]:
    rows = evidence.initial_world["tables"].get("inventory")
    if not isinstance(rows, dict):
        return {}
    return {str(key): value for key, value in rows.items() if isinstance(value, dict)}


def _observed_value(row: Mapping[str, Any], field: str) -> Any:
    if field == "price_cents":
        price = row.get("list_price")
        amount = price.get("amount") if isinstance(price, dict) else None
        return int(Decimal(str(amount)) * 100)
    attrs = row.get("attributes")
    if not isinstance(attrs, dict):
        return None
    return attrs.get(field)


def _satisfies(rule: HardConstraintT1, row: Mapping[str, Any]) -> bool:
    value = _observed_value(row, rule.field)
    if value is None:
        return False
    if rule.operator == "at_most":
        return int(value) <= int(rule.value)
    if rule.operator == "at_least":
        return int(value) >= int(rule.value)
    return value == rule.value


def _world_objective_value(row: Mapping[str, Any]) -> int:
    value = 0
    for field, weight in T1_PUBLIC_OBJECTIVE_WEIGHTS.items():
        observed = _observed_value(row, field)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise RuntimeBenchmarkIntegrityError(
                f"T1 authoritative catalog lacks objective field {field!r}"
            )
        value += observed * weight
    return value


def _acceptable_world_skus(
    task: RuntimeTaskContentT1,
    catalog: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Derive scored choices from authoritative public business facts."""

    feasible = tuple(
        sorted(
            sku
            for sku, row in catalog.items()
            if all(_satisfies(rule, row) for rule in task.hard_constraints)
            and int(inventory.get(sku, {}).get("qty_available", 0)) > 0
        )
    )
    if task.selection_mode == "any_feasible":
        return feasible
    if task.selection_mode == "abstain_if_none" or not feasible:
        return ()
    selected = min(
        feasible,
        key=lambda sku: (
            -_world_objective_value(catalog[sku]),
            public_reference_alias_v1(sku),
        ),
    )
    return (selected,)


def _verified_match_evidence(
    evidence: RuntimeEvidenceBundleV2,
) -> VerifiedMatchCertificateEvidence | None:
    """Return the exact Runtime, Platform, and World match authority graph."""

    try:
        value = evidence.verified_operation_evidence(
            MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
            options={
                "allow_rejected": False,
            },
        )
    except RuntimeEvidenceError:
        return None
    if not isinstance(value, VerifiedMatchCertificateEvidence) or any(
        row.certificate.buyer_id != _BUYER_ID for row in value.certificates
    ):
        return None
    return value


def _verified_search_evidence(
    evidence: RuntimeEvidenceBundleV2,
    *,
    exclude_session_ids: Sequence[str] = (),
) -> VerifiedSearchSessionEvidence | None:
    try:
        value = evidence.verified_operation_evidence(
            SEARCH_SESSION_EVIDENCE_CONTRACT,
            options={"exclude_session_ids": tuple(exclude_session_ids)},
        )
    except RuntimeEvidenceError:
        return None
    return value if isinstance(value, VerifiedSearchSessionEvidence) else None


def _discovery_commit_ids(
    verified_match: VerifiedMatchCertificateEvidence | None,
    verified_search: VerifiedSearchSessionEvidence | None,
) -> tuple[str, ...]:
    """Return discovery commits already owned by exact evidence contracts."""

    rows: list[dict[str, Any]] = []
    if verified_match is not None:
        for joined in verified_match.certificates:
            rows.extend((joined.search_commit, joined.commit))
    if verified_search is not None:
        rows.extend(joined.commit for joined in verified_search.sessions)
    rows.sort(key=lambda row: int(row.get("sequence", -1)))
    return tuple(str(row["commit_id"]) for row in rows)


def _verified_transaction_evidence(
    evidence: RuntimeEvidenceBundleV2,
    *,
    preclaimed_commit_ids: Sequence[str],
) -> VerifiedSupplyFulfillmentEvidence | None:
    """Verify settlement and zero-transaction outcomes through World evidence."""

    try:
        value = evidence.verified_operation_evidence(
            SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
            options={
                "allow_rejected": True,
                "evaluated_actor_id": _BUYER_ID,
                "preclaimed_commit_ids": tuple(preclaimed_commit_ids),
            },
        )
    except RuntimeEvidenceError:
        return None
    return value if isinstance(value, VerifiedSupplyFulfillmentEvidence) else None


def _transaction_integrity(
    *,
    verified_match: VerifiedMatchCertificateEvidence | None,
    verified_transaction: VerifiedSupplyFulfillmentEvidence | None,
    accepted: bool,
    abstained: bool,
) -> tuple[float, dict[str, Any]]:
    """Validate Agent-owned settlement effects without granting model credit."""

    operations = verified_transaction.operations if verified_transaction is not None else ()
    settlements = (
        verified_transaction.operations_for(
            action_kind="platform.settle_payment",
            actor_id=_BUYER_ID,
        )
        if verified_transaction is not None
        else ()
    )
    rejected_count = (
        len(verified_transaction.rejected_exchanges) if verified_transaction is not None else 0
    )
    evidence: dict[str, Any] = {
        "observed_business_outcome": (
            "accepted_offer" if accepted else "rejected_purchase" if abstained else "incomplete"
        ),
        "verified_transaction_contract": verified_transaction is not None,
        "operation_count": len(operations),
        "settlement_count": len(settlements),
        "rejected_transaction_count": rejected_count,
        "abstained": abstained,
    }

    if not accepted:
        exact_zero = bool(
            verified_transaction is not None
            and not operations
            and not verified_transaction.rejected_exchanges
            and (verified_match is None or not verified_match.certificates)
        )
        evidence["exact_zero_transaction"] = exact_zero
        return (1.0 if exact_zero else 0.0), evidence

    if (
        verified_transaction is None
        or len(operations) != 1
        or len(settlements) != 1
        or verified_transaction.rejected_exchanges
        or verified_match is None
        or len(verified_match.certificates) != 1
    ):
        return 0.0, evidence

    settlement = settlements[0]
    certificate = verified_match.certificates[0].certificate
    request_action = settlement.exchange.request.get("action")
    payload = request_action.get("payload") if isinstance(request_action, Mapping) else None
    if not isinstance(payload, Mapping):
        return 0.0, evidence
    order = verified_settlement_order(settlement)
    exact_binding = bool(
        payload.get("cert_id") == certificate.cert_id
        and order.get("order_id") == certificate.order_id
        and order.get("buyer_id") == certificate.buyer_id
        and order.get("merchant_id") == certificate.merchant_id
        and order.get("sku_id") == certificate.sku_id
        and order.get("qty") == certificate.qty
    )
    evidence.update(
        {
            "order_id": order.get("order_id"),
            "sku_id": order.get("sku_id"),
            "certificate_id": payload.get("cert_id"),
            "exact_certificate_binding": exact_binding,
        }
    )
    return (1.0 if exact_binding else 0.0), evidence


def _selected_sku(
    verified: VerifiedMatchCertificateEvidence | None,
) -> str | None:
    if verified is None or len(verified.certificates) != 1:
        return None
    return verified.certificates[0].certificate.sku_id


def _ranked_skus(evidence: RuntimeEvidenceBundleV2) -> tuple[str, ...]:
    output: list[str] = []
    for exchange in evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    ):
        for envelope in exchange.responses:
            if envelope["action"].get("kind") != "platform.rank_offers":
                continue
            payload = envelope["action"].get("payload") or {}
            for row in payload.get("candidates", ()):
                if isinstance(row, dict) and row.get("sku_id"):
                    output.append(str(row["sku_id"]))
    return tuple(output)


def _grounded_skus(evidence: RuntimeEvidenceBundleV2) -> frozenset[str]:
    return frozenset(attested_world_catalog_reads_v2(evidence, actor_id=_BUYER_ID))


def _search_binding_observation(
    evidence: RuntimeEvidenceBundleV2,
    *,
    verified_match: VerifiedMatchCertificateEvidence | None,
    verified_search: VerifiedSearchSessionEvidence | None,
) -> dict[str, Any]:
    """Describe the verified search prefix without awarding protocol credit."""

    searches = evidence.accepted_platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.rank_offers",
    )
    rank_count = sum(
        (response.get("action") or {}).get("kind") == "platform.rank_offers"
        for exchange in searches
        for response in exchange.responses
    )
    matched_session_ids = {
        row.session.session_id
        for row in (() if verified_match is None else verified_match.certificates)
    }
    verified_session_ids = (
        set(verified_search.all_session_ids) if verified_search is not None else matched_session_ids
    )
    observed_session_ids = {
        str((response.get("action") or {}).get("payload", {}).get("session_id"))
        for exchange in searches
        for response in exchange.responses
        if (response.get("action") or {}).get("kind") == "platform.rank_offers"
    }
    search_authority_bound = bool(
        verified_match is not None
        and verified_session_ids == observed_session_ids
        and len(verified_session_ids) == len(searches)
    )
    verified_search_prefix = bool(
        searches and rank_count == len(searches) and search_authority_bound
    )
    return {
        "buyer_search_count": len(searches),
        "platform_rank_count": rank_count,
        "world_linked_search_session_ids": sorted(verified_session_ids),
        "search_authority_bound": search_authority_bound,
        "verified_search_prefix": verified_search_prefix,
    }


def _score_query(
    definition: TaskDefinitionV2,
    task: RuntimeTaskContentT1,
    evidence: RuntimeEvidenceBundleV2,
    *,
    verified_match: VerifiedMatchCertificateEvidence | None,
    verified_search: VerifiedSearchSessionEvidence | None,
) -> RuntimeTaskScoreV3:
    searches = tuple(
        exchange.request
        for exchange in evidence.accepted_platform_exchanges(
            kind="commerce.search",
            actor_id=_BUYER_ID,
            endpoint="platform:aggregator",
            response_kind="platform.rank_offers",
        )
    )
    required = task.required_search_rounds
    hard_filters = _constraint_filters(task)
    hard_ids = [rule.constraint_id for rule in task.hard_constraints]
    stages = [
        list(task.soft_preferences[:remaining])
        for remaining in range(len(task.soft_preferences), -1, -1)
    ]
    preserved = 0
    staged = 0
    for index, envelope in enumerate(searches):
        payload = envelope["action"].get("payload") or {}
        if payload.get("hard_constraint_ids") != hard_ids:
            raise RuntimeBenchmarkIntegrityError(
                "T1 Agent search authority does not bind the exact frozen constraint IDs"
            )
        filters = payload.get("filters") or {}
        if all(filters.get(key) == value for key, value in hard_filters.items()):
            preserved += 1
        if index < len(stages) and filters.get("required_features", []) == stages[index]:
            staged += 1
    selected = _selected_sku(verified_match)
    ranked = _ranked_skus(evidence)
    acceptable = _acceptable_world_skus(
        task,
        _catalog_index(evidence),
        _inventory_index(evidence),
    )
    checks = renormalize_capability_checks_v2(
        (
            RuntimeRubricCheckV2(
                "hard_constraint_preservation",
                0.15,
                min(preserved / required, 1.0),
                {"preserved_rounds": preserved, "required_rounds": required},
            ),
            RuntimeRubricCheckV2(
                "staged_reformulation",
                0.20,
                min(staged / required, 1.0),
                {"matching_stages": staged, "required_rounds": required},
            ),
            RuntimeRubricCheckV2(
                "visible_candidate_evidence",
                0.15,
                1.0 if selected is not None and selected in ranked else 0.0,
                {"selected_sku": selected, "ranked_skus": list(ranked)},
            ),
            RuntimeRubricCheckV2(
                "best_feasible",
                0.20,
                1.0 if selected in acceptable else 0.0,
                {"selected_sku": selected, "acceptable_skus": list(acceptable)},
            ),
        )
    )
    issues = () if all(row.credit == 1 for row in checks) else ("query_reformulation_incomplete",)
    return score_checks(definition, checks, issues=issues)


def score_t1_runtime(
    task_id: str,
    evidence: RuntimeEvidenceBundleV2,
) -> RuntimeTaskScoreV3:
    """Score T1 exclusively from stored Runtime, Platform, and World evidence."""

    require_runtime_benchmark_integrity_v2(evidence)
    require_frozen_scenario_fixture_v2(
        evidence,
        scenario_for_t1(task_id),
        family="T1",
    )
    definition = TASK_REGISTRY_V2[task_id]
    task = T1_RUNTIME_TASKS[task_id]
    verified_match = _verified_match_evidence(evidence)
    accepted_matches = evidence.accepted_platform_exchanges(
        kind="commerce.accept_offer",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
        response_kind="platform.create_match_certificate",
    )
    if accepted_matches and verified_match is None:
        raise RuntimeBenchmarkIntegrityError(
            "T1 accepted offer is not bound to an authoritative World match certificate"
        )
    matched_session_ids = tuple(
        row.session.session_id
        for row in (() if verified_match is None else verified_match.certificates)
    )
    # Every accepted search creates authoritative World state, including a
    # partial run that stops before offer acceptance.  Always invoke the
    # search-session contract so authority-commit closure can claim those
    # search-only commits.  A completed match already claims its predecessor
    # search commit, so exclude that session here to keep each World commit
    # owned by exactly one contract call.
    verified_search = _verified_search_evidence(
        evidence,
        exclude_session_ids=matched_session_ids,
    )
    observed_searches = evidence.platform_exchanges(
        kind="commerce.search",
        actor_id=_BUYER_ID,
        endpoint="platform:aggregator",
    )
    observed_search_commits = tuple(
        row for row in evidence.world_events if row.get("operation") == "create_search_session"
    )
    if (observed_searches or observed_search_commits) and (
        verified_match is None and verified_search is None
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T1 observed search is not bound to an authoritative World search session"
        )
    verified_transaction = _verified_transaction_evidence(
        evidence,
        preclaimed_commit_ids=_discovery_commit_ids(
            verified_match,
            verified_search,
        ),
    )
    observed_transactions = evidence.platform_exchanges(
        kind="platform.settle_payment",
        actor_id=_BUYER_ID,
        endpoint="platform:psp",
    )
    observed_transaction_commits = tuple(
        row for row in evidence.world_events if row.get("operation") == "settle"
    )
    if (observed_transactions or observed_transaction_commits) and (verified_transaction is None):
        raise RuntimeBenchmarkIntegrityError(
            "T1 observed settlement is not bound to an authoritative World transaction"
        )
    abstentions = evidence.actions(kind="delegate.reject_purchase", actor_id=_BUYER_ID)
    transaction_integrity, transaction_evidence = _transaction_integrity(
        verified_match=verified_match,
        verified_transaction=verified_transaction,
        accepted=bool(accepted_matches),
        abstained=bool(abstentions),
    )
    if transaction_integrity != 1.0:
        raise RuntimeBenchmarkIntegrityError(
            "T1 Agent settlement effects are not an exact authoritative transaction: "
            + json.dumps(transaction_evidence, sort_keys=True)
        )
    if task.required_search_rounds:
        return _score_query(
            definition,
            task,
            evidence,
            verified_match=verified_match,
            verified_search=verified_search,
        )

    catalog = _catalog_index(evidence)
    inventory = _inventory_index(evidence)
    selected = _selected_sku(verified_match)
    ranked = _ranked_skus(evidence)
    grounded = _grounded_skus(evidence)
    selected_row = catalog.get(selected or "")
    constraint_fraction = 0.0
    if selected_row is not None:
        passed = sum(_satisfies(rule, selected_row) for rule in task.hard_constraints)
        constraint_fraction = passed / len(task.hard_constraints)
    no_feasible = not any(
        all(_satisfies(rule, row) for rule in task.hard_constraints)
        and int(inventory.get(sku, {}).get("qty_available", 0)) > 0
        for sku, row in catalog.items()
    )
    acceptable = _acceptable_world_skus(task, catalog, inventory)

    if task.selection_mode == "abstain_if_none":
        decision_credit = 1.0 if no_feasible and bool(abstentions) and selected is None else 0.0
        constraint_fraction = 1.0 if no_feasible else 0.0
    else:
        decision_credit = 1.0 if selected in acceptable else 0.0

    search_binding = _search_binding_observation(
        evidence,
        verified_match=verified_match,
        verified_search=verified_search,
    )
    verified_search_prefix = search_binding["verified_search_prefix"] is True
    grounding_credit = 0.0
    if selected is not None and verified_search_prefix:
        grounding_credit = (
            1.0
            if selected in ranked and selected in grounded
            else 0.5
            if selected in ranked
            else 0.0
        )
    elif verified_search_prefix:
        visible = set(ranked)
        grounding_credit = (
            1.0
            if visible and visible <= set(grounded)
            else 0.5
            if visible and visible & set(grounded)
            else 0.0
        )

    checks = renormalize_capability_checks_v2(
        (
            RuntimeRubricCheckV2(
                "candidate_evidence_coverage",
                0.15,
                grounding_credit,
                {"ranked_skus": list(ranked), "grounded_skus": sorted(grounded)},
            ),
            RuntimeRubricCheckV2(
                "hard_constraints",
                0.25,
                constraint_fraction,
                {"selected_sku": selected, "no_feasible_listing": no_feasible},
            ),
            RuntimeRubricCheckV2(
                "task_decision",
                0.25,
                decision_credit,
                {
                    "selected_sku": selected,
                    "acceptable_skus": list(acceptable),
                    "abstained": bool(abstentions),
                },
            ),
        )
    )
    issues = () if all(row.credit == 1 for row in checks) else ("t1_task_incomplete",)
    return score_checks(definition, checks, issues=issues)


def runtime_bundle_t1(task_id: str) -> RuntimeTaskBundleV2:
    task = T1_RUNTIME_TASKS[task_id]
    definition = TASK_REGISTRY_V2[task_id]
    scenario = scenario_for_t1(task_id)
    semantic_hash = canonical_sha256(
        {
            "task_content_sha256": task.content_sha256,
            "scenario_oracle": scenario.success_oracle,
            "scenario_state": scenario.initial_state,
        }
    )
    mutation_targets = {
        "t1.basic_feasible_discovery": (
            "hard_constraints",
            "task_decision",
        ),
        "t1.hard_constraint_filtering": (
            "hard_constraints",
            "task_decision",
        ),
        "t1.best_feasible_selection": ("task_decision",),
        "t1.correct_abstention": ("task_decision",),
        "t1.query_reformulation": ("best_feasible",),
    }[definition.capability_id]
    return RuntimeTaskBundleV2(
        task=definition,
        scenario=scenario,
        evaluated_actor_id=_BUYER_ID,
        ideal_channel=lambda: _T1BusinessChannel(task),
        counterpart_channels={_MERCHANT_ID: _NoReplyT1Channel},
        scorer=lambda evidence: score_t1_runtime(task_id, evidence),
        semantic_hash=semantic_hash,
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: _T1BusinessChannel(task, alternative=True),
                expected_changed_checks=mutation_targets,
            ),
        ),
    )


def runtime_bundles_t1() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t1(task_id) for task_id in sorted(T1_RUNTIME_TASKS))


__all__ = [
    "T1_RUNTIME_SCHEMA_V2",
    "runtime_bundle_t1",
    "runtime_bundles_t1",
    "scenario_for_t1",
    "score_t1_runtime",
]
