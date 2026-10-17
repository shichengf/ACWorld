"""ACWorld T5 multiline pricing and cart tasks.

T5 is intentionally built on the production cart authority path.  Scenarios
seed only catalog rows, a principal mandate, and compact pricing-policy
intents.  World derives and persists policy revisions, cart authorizations,
quotes, orders, receipts, and the group payment record.  Agents can submit
only compact intents and a buyer can checkout only by ``quote_id``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any, Mapping, Sequence

from agents.agent_phase import public_reference_alias_v1
from agents.business_decision import LLM_BUSINESS_DECISION_SCHEMA_V1
from agents.inference import BusinessDecisionResponseV1, InferenceChannel
from episode.capability_benchmark import TASK_REGISTRY_V2, TaskDefinitionV2
from episode.capability_runtime import (
    COMMERCEWORLD_EPISODE_BACKEND_V2,
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
from episode.capability_runtime_discovery import (
    verify_optional_discovery_prefix_v2,
)
from episode.capability_runtime_fixture import require_frozen_scenario_fixture_v2
from episode.capability_t5_planning import (
    CartLines,
    T5_CART_PLANNING_RULE_SET_V1,
    T5_CART_PLANNING_SCHEMA_V1,
    T5CartOracleV1,
    build_t5_cart_oracle_v1,
    evaluate_t5_cart_lines_v1,
    public_t5_problem_copy_v1,
)
from episode.types import BuyerSpec, MerchantSpec, PopulationSpec, ScenarioSpec
from protocol.cart_quote_state import (
    CartQuoteContractError,
    persistent_cart_quote_from_json,
)
from protocol.evidence_records import (
    MandateRevisionAuthority,
    build_mandate_revision,
    mandate_revision_to_dict,
)
from runtime.cart_evidence import (
    CART_AUTHORIZATION_PREFIX_EVIDENCE_CONTRACT,
    CART_EVIDENCE_CONTRACT,
    CART_QUOTE_PREFIX_EVIDENCE_CONTRACT,
    VerifiedCartAuthorizationPrefixEvidence,
    VerifiedCartEvidence,
    VerifiedCartQuotePrefixEvidence,
)
from runtime.commit_claims import verify_exact_transaction_commit_claims
from runtime.tracker_evidence import (
    TrackerEvidenceError,
    VerifiedModelBusinessChoice,
    verified_model_business_choices,
)
from world.evidence_contracts import mandate_authority_to_wire


T5_RUNTIME_SCHEMA_V2 = "cwe.runtime-task.t5.v4"
_BUYER_ID = "buyer:t5-benchmark"
_PRINCIPAL_ID = "consumer:t5-benchmark"
_PRIMARY_MERCHANT_ID = "merchant:t5-primary"
_T5_TASK_IDS = tuple(
    task_id for task_id, definition in TASK_REGISTRY_V2.items() if definition.family.value == "T5"
)


@dataclass(frozen=True)
class _CaseT5:
    definition: TaskDefinitionV2
    lane: str
    axis_name: str
    axis_value: int
    market_id: str
    catalog: tuple[dict[str, Any], ...]
    pricing_fixtures: tuple[dict[str, Any], ...]
    merchant_requested_lines: CartLines
    buyer_problem: Mapping[str, Any] | None
    buyer_oracle: T5CartOracleV1 | None
    merchant_ids: tuple[str, ...]
    budget_minor: int

    @property
    def evaluated_actor_id(self) -> str:
        if self.definition.evaluated_role == "buyer":
            return _BUYER_ID
        return _PRIMARY_MERCHANT_ID

    @property
    def required_skus(self) -> tuple[str, ...]:
        if self.definition.evaluated_role == "buyer":
            return tuple(str(row["sku_id"]) for row in self.catalog)
        return tuple(sku_id for sku_id, _ in self.merchant_requested_lines)

    @property
    def semantic_contract(self) -> dict[str, Any]:
        return {
            "schema_version": T5_RUNTIME_SCHEMA_V2,
            "definition": self.definition.to_dict(),
            "lane": self.lane,
            "difficulty": {self.axis_name: self.axis_value},
            "market_id": self.market_id,
            "catalog": self.catalog,
            "pricing_fixtures": self.pricing_fixtures,
            "merchant_requested_lines": self.merchant_requested_lines,
            "buyer_problem": self.buyer_problem,
            "buyer_acceptable_plans": (
                self.buyer_oracle.acceptable_plans if self.buyer_oracle is not None else None
            ),
            "merchant_ids": self.merchant_ids,
            "budget_minor": self.budget_minor,
            "execution_path": (
                "optional search and match discovery through Platform",
                "commerce.create_cart_quote_request or commerce.request_cart_quote",
                "World persistent cart authorization and quote",
                "platform.checkout_cart with quote_id only",
                "World atomic cart checkout",
            ),
        }


def _axis(definition: TaskDefinitionV2) -> tuple[str, int]:
    values = [
        (name, value) for name, value in definition.difficulty_factors if name != "difficulty_level"
    ]
    if len(values) != 1:
        raise ValueError(f"{definition.task_id}: T5 needs one semantic axis")
    name, value = values[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{definition.task_id}: T5 axis must be a positive integer")
    return name, value


def _merchant_id(ordinal: int) -> str:
    if ordinal == 0:
        return _PRIMARY_MERCHANT_ID
    return f"merchant:t5-{ordinal + 1}"


def _sku_id(definition: TaskDefinitionV2, ordinal: int) -> str:
    suffix = definition.task_id.casefold().replace("-", "")
    return f"sku:{suffix}:{ordinal + 1:02d}"


def _catalog_row(
    definition: TaskDefinitionV2,
    *,
    ordinal: int,
    merchant_id: str,
    price_minor: int,
    inventory: int = 64,
) -> dict[str, Any]:
    sku_id = _sku_id(definition, ordinal)
    return {
        "sku_id": sku_id,
        "product_id": f"product:{sku_id}",
        "merchant_id": merchant_id,
        "category": "cart-goods",
        "name": f"Cart item {ordinal + 1}",
        "list_price": str(Decimal(price_minor) / Decimal(100)),
        "inventory": inventory,
        # Executable pricing and benchmark semantics live outside catalog attributes.
        "attributes": {},
    }


def _fixture(
    *,
    task_id: str,
    ordinal: int,
    market_id: str,
    merchant_id: str,
    listing_ids: Sequence[str],
    tiers: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]] = (),
    components: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "merchant_id": merchant_id,
        "idempotency_key": f"seed:{task_id.casefold()}:policy:{ordinal:02d}",
        "intent": {
            "market_id": market_id,
            "policy_id": f"policy:{task_id.casefold()}:{ordinal:02d}",
            "listing_ids": list(listing_ids),
            "product_ids": [],
            "quantity_tiers": [dict(row) for row in tiers],
            "bundle_discounts": [copy.deepcopy(dict(row)) for row in bundles],
            "bundle_stacking": "best_only",
            "components": [dict(row) for row in components],
            "effective_after_ticks": 0,
            "expires_after_ticks": None,
        },
    }


def _one_tier(unit_price_minor: int) -> list[dict[str, Any]]:
    return [
        {
            "minimum_quantity": 1,
            "maximum_quantity": None,
            "unit_price_minor": unit_price_minor,
        }
    ]


def _tier_ladder(count: int) -> tuple[list[dict[str, Any]], int]:
    minima = (1, 2, 4, 8, 16)[:count]
    tiers = []
    for index, minimum in enumerate(minima):
        maximum = minima[index + 1] - 1 if index + 1 < len(minima) else None
        tiers.append(
            {
                "minimum_quantity": minimum,
                "maximum_quantity": maximum,
                "unit_price_minor": 1_500 - index * 125,
            }
        )
    return tiers, minima[-1]


_T5_CALCULATION_RULES: dict[str, str] = {
    "tier_scope": "sum_selected_quantity_within_pricing_term",
    "bundle_discount_scope": "selected_cart",
    "charge_basis": "base_subtotal_before_bundle_discount",
    "bps_rounding": "floor_minor_units",
    "grand_total": "discounted_line_subtotal_plus_active_charges",
}
_T5_OBJECTIVE: dict[str, Any] = {
    "kind": "lexicographic_min",
    "criteria": ["grand_total_minor", "max_delivery_days", "merchant_count"],
    "tie_break": "canonical_selected_sku_refs_then_qty_ascending",
}


def _buyer_problem(
    *,
    catalog: Sequence[Mapping[str, Any]],
    pricing_fixtures: Sequence[Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    budget_minor: int,
    max_delivery_days: int = 14,
) -> dict[str, Any]:
    offers: list[dict[str, Any]] = []
    for row in catalog:
        amount = Decimal(str(row["list_price"])) * Decimal(100)
        if amount != amount.to_integral_value():
            raise ValueError("T5 catalog prices must use integral minor units")
        offers.append(
            {
                "sku_id": str(row["sku_id"]),
                "merchant_id": str(row["merchant_id"]),
                "product_family": str(row["product_family"]),
                "list_price_minor": int(amount),
                "available_qty": int(row["inventory"]),
                "delivery_days": int(row["delivery_days"]),
            }
        )
    return {
        "schema_version": T5_CART_PLANNING_SCHEMA_V1,
        "rule_set": T5_CART_PLANNING_RULE_SET_V1,
        "currency": "USD",
        "listing_offers": offers,
        "requirements": [copy.deepcopy(dict(row)) for row in requirements],
        "relations": [copy.deepcopy(dict(row)) for row in relations],
        "pricing_terms": _pricing_terms_from_fixtures(pricing_fixtures),
        "hard_constraints": {
            "budget_minor": budget_minor,
            "max_delivery_days": max_delivery_days,
            "inventory_rule": "selected_qty_lte_available_qty",
            "requirement_rule": "exact_declared_demand",
            "relation_rule": "enforce_all_declared_relations",
        },
        "calculation_rules": copy.deepcopy(_T5_CALCULATION_RULES),
        "objective": copy.deepcopy(_T5_OBJECTIVE),
    }


def _planning_catalog_row(
    definition: TaskDefinitionV2,
    *,
    ordinal: int,
    merchant_id: str,
    product_family: str,
    price_minor: int,
    inventory: int = 64,
    delivery_days: int = 5,
) -> dict[str, Any]:
    row = _catalog_row(
        definition,
        ordinal=ordinal,
        merchant_id=merchant_id,
        price_minor=price_minor,
        inventory=inventory,
    )
    row["product_family"] = product_family
    row["delivery_days"] = delivery_days
    return row


def _quantity_offer_tiers(count: int, *, first: int, last: int) -> tuple[list[dict[str, Any]], int]:
    minima = (1, 2, 4, 8, 16)[:count]
    values = [first - ((first - last) * index // (count - 1)) for index in range(count)]
    values[-1] = last
    return (
        [
            {
                "minimum_quantity": minimum,
                "maximum_quantity": (minima[index + 1] - 1 if index + 1 < len(minima) else None),
                "unit_price_minor": values[index],
            }
            for index, minimum in enumerate(minima)
        ],
        minima[-1],
    )


@lru_cache(maxsize=None)
def _case_for_t5(task_id: str) -> _CaseT5:
    definition = TASK_REGISTRY_V2[task_id]
    if definition.family.value != "T5":
        raise ValueError(f"{task_id} is not a T5 task")
    axis_name, count = _axis(definition)
    lane = definition.capability_id.removeprefix("t5.")
    market_id = f"market:{task_id.casefold()}"
    rows: list[dict[str, Any]] = []
    lines: list[tuple[str, int]] = []
    fixtures: list[dict[str, Any]] = []
    merchants: list[str] = [_PRIMARY_MERCHANT_ID]
    requirements: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    budget_minor = 1_000_000
    buyer_problem: dict[str, Any] | None = None
    buyer_oracle: T5CartOracleV1 | None = None

    if lane == "multi_item_cart":
        for requirement_index in range(count):
            eligible: list[str] = []
            winning_position = (int(task_id[-2:]) + requirement_index) % 2
            for position in range(2):
                ordinal = requirement_index * 2 + position
                price = 800 + requirement_index * 80 + (0 if position == winning_position else 180)
                row = _planning_catalog_row(
                    definition,
                    ordinal=ordinal,
                    merchant_id=_PRIMARY_MERCHANT_ID,
                    product_family=f"need-{requirement_index + 1}",
                    price_minor=price,
                    delivery_days=3 + (position % 2),
                )
                rows.append(row)
                eligible.append(str(row["sku_id"]))
                fixtures.append(
                    _fixture(
                        task_id=task_id,
                        ordinal=ordinal + 1,
                        market_id=market_id,
                        merchant_id=_PRIMARY_MERCHANT_ID,
                        listing_ids=(str(row["sku_id"]),),
                        tiers=_one_tier(price),
                    )
                )
            requirements.append(
                {
                    "requirement_key": f"need-{requirement_index + 1}",
                    "product_family": f"need-{requirement_index + 1}",
                    "required_qty": 1,
                    "eligible_sku_ids": eligible,
                    "selection_rule": "choose_exactly_one_substitute",
                }
            )
        budget_minor = 50_000

    elif lane == "quantity_tiers" and definition.evaluated_role == "buyer":
        eligible = []
        winning_position = int(task_id[-2:]) % 3
        final_prices = [1_000, 1_080, 1_160]
        final_prices[0], final_prices[winning_position] = (
            final_prices[winning_position],
            final_prices[0],
        )
        quantity = 0
        for position in range(3):
            first_price = 1_220 + position * 70
            tiers, quantity = _quantity_offer_tiers(
                count,
                first=first_price,
                last=final_prices[position],
            )
            row = _planning_catalog_row(
                definition,
                ordinal=position,
                merchant_id=_PRIMARY_MERCHANT_ID,
                product_family="bulk-need",
                price_minor=first_price,
                inventory=quantity + 8,
                delivery_days=4 + position,
            )
            rows.append(row)
            eligible.append(str(row["sku_id"]))
            fixtures.append(
                _fixture(
                    task_id=task_id,
                    ordinal=position + 1,
                    market_id=market_id,
                    merchant_id=_PRIMARY_MERCHANT_ID,
                    listing_ids=(str(row["sku_id"]),),
                    tiers=tiers,
                )
            )
        requirements.append(
            {
                "requirement_key": "bulk-need",
                "product_family": "bulk-need",
                "required_qty": quantity,
                "eligible_sku_ids": eligible,
                "selection_rule": "choose_exactly_one_substitute",
            }
        )
        budget_minor = quantity * 2_000

    elif lane == "merchant_tier_quote":
        tiers, quantity = _tier_ladder(count)
        row = _catalog_row(
            definition,
            ordinal=0,
            merchant_id=_PRIMARY_MERCHANT_ID,
            price_minor=1_500,
            inventory=quantity + 8,
        )
        rows.append(row)
        lines.append((str(row["sku_id"]), quantity))
        fixtures.append(
            _fixture(
                task_id=task_id,
                ordinal=1,
                market_id=market_id,
                merchant_id=_PRIMARY_MERCHANT_ID,
                listing_ids=(str(row["sku_id"]),),
                tiers=tiers,
            )
        )

    elif lane == "bundle_relations":
        for relation in range(count):
            bundled_pair: list[str] = []
            standalone_pair: list[str] = []
            for requirement_offset in range(2):
                eligible = []
                family = f"module-{relation + 1}-part-{requirement_offset + 1}"
                bundled_ordinal = relation * 4 + requirement_offset
                bundled = _planning_catalog_row(
                    definition,
                    ordinal=bundled_ordinal,
                    merchant_id=_PRIMARY_MERCHANT_ID,
                    product_family=family,
                    price_minor=1_100 + relation * 20,
                )
                standalone = _planning_catalog_row(
                    definition,
                    ordinal=relation * 4 + 2 + requirement_offset,
                    merchant_id=_PRIMARY_MERCHANT_ID,
                    product_family=family,
                    price_minor=950 + relation * 20,
                    delivery_days=4,
                )
                rows.extend((bundled, standalone))
                bundled_pair.append(str(bundled["sku_id"]))
                standalone_pair.append(str(standalone["sku_id"]))
                eligible.extend((str(bundled["sku_id"]), str(standalone["sku_id"])))
                requirements.append(
                    {
                        "requirement_key": family,
                        "product_family": family,
                        "required_qty": 1,
                        "eligible_sku_ids": eligible,
                        "selection_rule": "choose_exactly_one_substitute",
                    }
                )
            fixtures.append(
                _fixture(
                    task_id=task_id,
                    ordinal=relation * 2 + 1,
                    market_id=market_id,
                    merchant_id=_PRIMARY_MERCHANT_ID,
                    listing_ids=bundled_pair,
                    tiers=_one_tier(1_100 + relation * 20),
                    bundles=(
                        {
                            "bundle_id": f"bundle:{task_id.casefold()}:{relation + 1}",
                            "conditions": [
                                {
                                    "selector_kind": "listing",
                                    "selector_id": sku_id,
                                    "minimum_quantity": 1,
                                }
                                for sku_id in bundled_pair
                            ],
                            "discount_minor": 400,
                            "discount_bps": None,
                        },
                    ),
                )
            )
            fixtures.append(
                _fixture(
                    task_id=task_id,
                    ordinal=relation * 2 + 2,
                    market_id=market_id,
                    merchant_id=_PRIMARY_MERCHANT_ID,
                    listing_ids=standalone_pair,
                    tiers=_one_tier(950 + relation * 20),
                )
            )
            if relation % 2 == 0:
                relations.append(
                    {
                        "kind": "complement_all_or_none",
                        "sku_ids": bundled_pair,
                    }
                )
            else:
                relations.append(
                    {
                        "kind": "required_with",
                        "trigger_sku_id": bundled_pair[0],
                        "required_sku_id": bundled_pair[1],
                        "minimum_qty": 1,
                    }
                )
        budget_minor = 50_000

    elif lane == "total_budget" and definition.evaluated_role == "buyer":
        merchants = [_merchant_id(index) for index in range(3)]
        offers_by_merchant: dict[str, list[str]] = {merchant: [] for merchant in merchants}
        for requirement_index in range(2):
            eligible = []
            family = f"fee-need-{requirement_index + 1}"
            for merchant_index, merchant_id in enumerate(merchants):
                ordinal = requirement_index * 3 + merchant_index
                price = (700, 780, 900)[merchant_index]
                row = _planning_catalog_row(
                    definition,
                    ordinal=ordinal,
                    merchant_id=merchant_id,
                    product_family=family,
                    price_minor=price,
                    delivery_days=3 + merchant_index,
                )
                rows.append(row)
                eligible.append(str(row["sku_id"]))
                offers_by_merchant[merchant_id].append(str(row["sku_id"]))
            requirements.append(
                {
                    "requirement_key": family,
                    "product_family": family,
                    "required_qty": 1,
                    "eligible_sku_ids": eligible,
                    "selection_rule": "choose_exactly_one_substitute",
                }
            )
        for merchant_index, merchant_id in enumerate(merchants):
            components = []
            if merchant_index == 0:
                components = [
                    {
                        "component_id": f"component:{index + 1}",
                        "kind": ("shipping", "tax", "fee")[index % 3],
                        "fixed_minor": 100,
                        "per_unit_minor": 0,
                        "subtotal_rate_bps": 0,
                        "minimum_subtotal_minor": 0,
                        "maximum_subtotal_minor": None,
                    }
                    for index in range(count)
                ]
            fixtures.append(
                _fixture(
                    task_id=task_id,
                    ordinal=merchant_index + 1,
                    market_id=market_id,
                    merchant_id=merchant_id,
                    listing_ids=offers_by_merchant[merchant_id],
                    tiers=_one_tier((700, 780, 900)[merchant_index]),
                    components=components,
                )
            )
        budget_minor = 2_500

    elif lane == "merchant_total_quote":
        sku_ids: list[str] = []
        for index in range(2):
            row = _catalog_row(
                definition,
                ordinal=index,
                merchant_id=_PRIMARY_MERCHANT_ID,
                price_minor=1_250,
            )
            rows.append(row)
            lines.append((str(row["sku_id"]), 1))
            sku_ids.append(str(row["sku_id"]))
        components = [
            {
                "component_id": f"component:{index + 1}",
                "kind": ("shipping", "tax", "fee")[index % 3],
                "fixed_minor": 100 + index * 25,
                "per_unit_minor": 0,
                "subtotal_rate_bps": 0,
                "minimum_subtotal_minor": 0,
                "maximum_subtotal_minor": None,
            }
            for index in range(count)
        ]
        fixtures.append(
            _fixture(
                task_id=task_id,
                ordinal=1,
                market_id=market_id,
                merchant_id=_PRIMARY_MERCHANT_ID,
                listing_ids=sku_ids,
                tiers=_one_tier(1_250),
                components=components,
            )
        )

    elif lane == "cross_merchant_cart":
        merchants = [_merchant_id(index) for index in range(count)]
        for requirement_index in range(2):
            eligible = []
            family = f"merchant-need-{requirement_index + 1}"
            for merchant_index, merchant_id in enumerate(merchants):
                ordinal = requirement_index * count + merchant_index
                preferred = requirement_index % count
                price = 700 if merchant_index == preferred else 1_100 + merchant_index * 50
                row = _planning_catalog_row(
                    definition,
                    ordinal=ordinal,
                    merchant_id=merchant_id,
                    product_family=family,
                    price_minor=price,
                    delivery_days=3 + merchant_index,
                )
                rows.append(row)
                eligible.append(str(row["sku_id"]))
                fixtures.append(
                    _fixture(
                        task_id=task_id,
                        ordinal=ordinal + 1,
                        market_id=market_id,
                        merchant_id=merchant_id,
                        listing_ids=(str(row["sku_id"]),),
                        tiers=_one_tier(price),
                        components=(
                            {
                                "component_id": "component:shipping",
                                "kind": "shipping",
                                "fixed_minor": 50,
                                "per_unit_minor": 0,
                                "subtotal_rate_bps": 0,
                                "minimum_subtotal_minor": 0,
                                "maximum_subtotal_minor": None,
                            },
                        ),
                    )
                )
            requirements.append(
                {
                    "requirement_key": family,
                    "product_family": family,
                    "required_qty": 1,
                    "eligible_sku_ids": eligible,
                    "selection_rule": "choose_exactly_one_substitute",
                }
            )
        budget_minor = 4_000

    elif lane == "merchant_bundle_quote":
        sku_ids = []
        for index in range(count):
            row = _catalog_row(
                definition,
                ordinal=index,
                merchant_id=_PRIMARY_MERCHANT_ID,
                price_minor=1_200,
            )
            rows.append(row)
            lines.append((str(row["sku_id"]), 1))
            sku_ids.append(str(row["sku_id"]))
        fixtures.append(
            _fixture(
                task_id=task_id,
                ordinal=1,
                market_id=market_id,
                merchant_id=_PRIMARY_MERCHANT_ID,
                listing_ids=sku_ids,
                tiers=_one_tier(1_200),
                bundles=(
                    {
                        "bundle_id": f"bundle:{task_id.casefold()}:all",
                        "conditions": [
                            {
                                "selector_kind": "listing",
                                "selector_id": sku_id,
                                "minimum_quantity": 1,
                            }
                            for sku_id in sku_ids
                        ],
                        "discount_minor": 100,
                        "discount_bps": None,
                    },
                ),
            )
        )
    else:  # pragma: no cover - task registry dispatch is exhaustive.
        raise ValueError(f"unsupported T5 capability {definition.capability_id!r}")

    if definition.evaluated_role == "buyer":
        buyer_problem = _buyer_problem(
            catalog=rows,
            pricing_fixtures=fixtures,
            requirements=requirements,
            relations=relations,
            budget_minor=budget_minor,
        )
        buyer_oracle = build_t5_cart_oracle_v1(
            buyer_problem,
            reference_kind="internal",
            public_reference=public_reference_alias_v1,
        )

    return _CaseT5(
        definition=definition,
        lane=lane,
        axis_name=axis_name,
        axis_value=count,
        market_id=market_id,
        catalog=tuple(rows),
        pricing_fixtures=tuple(fixtures),
        merchant_requested_lines=tuple(lines),
        buyer_problem=buyer_problem,
        buyer_oracle=buyer_oracle,
        merchant_ids=tuple(merchants),
        budget_minor=budget_minor,
    )


def _public_contract(case: _CaseT5) -> dict[str, Any]:
    if case.definition.evaluated_role == "buyer":
        instruction = (
            "Select the lexicographically best feasible cart from the complete public "
            "offer set. Read every selected listing to ground its current terms; the "
            "cart-quote action becomes available after those reads. Then request its "
            "World-authoritative quote and settle by quote reference."
        )
        lines: list[dict[str, Any]] | None = None
    else:
        instruction = (
            "Apply the public pricing rules to the inbound requested lines and issue one "
            "World-authoritative quote."
        )
        lines = [{"sku_id": sku_id, "qty": qty} for sku_id, qty in case.merchant_requested_lines]
    return {
        "schema_version": T5_RUNTIME_SCHEMA_V2,
        "task_id": case.definition.task_id,
        "instruction": instruction,
        "evaluated_role": case.definition.evaluated_role,
        "capability": case.lane,
        "difficulty": {case.axis_name: case.axis_value},
        "market_id": case.market_id,
        "mandate_id": case.definition.task_id,
        **({"requested_lines": lines} if lines is not None else {}),
        "authority": {
            "pricing_and_quote": "World persistent cart and pricing tables",
            "checkout_payload": ["quote_id"],
            "settlement": "World atomic order group commit",
        },
    }


def _pricing_terms_from_fixtures(
    pricing_fixtures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project merchant-owned commercial rules without World authority metadata."""

    terms: list[dict[str, Any]] = []
    for fixture in pricing_fixtures:
        intent = fixture["intent"]
        bundles: list[dict[str, Any]] = []
        for bundle in intent["bundle_discounts"]:
            bundles.append(
                {
                    "conditions": [
                        {
                            "sku_id": condition["selector_id"],
                            "minimum_quantity": condition["minimum_quantity"],
                        }
                        for condition in bundle["conditions"]
                    ],
                    "discount_minor": bundle["discount_minor"],
                    "discount_bps": bundle["discount_bps"],
                }
            )
        terms.append(
            {
                "sku_ids": copy.deepcopy(intent["listing_ids"]),
                "quantity_tiers": copy.deepcopy(intent["quantity_tiers"]),
                "bundle_discounts": bundles,
                "bundle_stacking": intent["bundle_stacking"],
                "charges": [
                    {
                        key: copy.deepcopy(component[key])
                        for key in (
                            "kind",
                            "fixed_minor",
                            "per_unit_minor",
                            "subtotal_rate_bps",
                            "minimum_subtotal_minor",
                            "maximum_subtotal_minor",
                        )
                    }
                    for component in intent["components"]
                ],
            }
        )
    return terms


def _public_pricing_terms(case: _CaseT5) -> list[dict[str, Any]]:
    return _pricing_terms_from_fixtures(case.pricing_fixtures)


def _public_task_context(case: _CaseT5) -> dict[str, Any]:
    """Expose the cart state machine and pricing rules, never the quote result.

    The phase contract is shared by every T5 difficulty instance.  It names
    only public inbound kinds and registered actor routes. Merchant-owned tier,
    bundle, and charge rules are actor-visible business facts; quote values and
    the scorer's expected outcome remain outside the execution contract.
    """

    facts: dict[str, Any] = {
        "schema_version": T5_RUNTIME_SCHEMA_V2,
        "task_id": case.definition.task_id,
        "capability": case.lane,
        "evaluated_role": case.definition.evaluated_role,
        "execution_contract": {
            "schema_version": "cwe.public-task-execution.v1",
            "phases": [
                {
                    "phase_id": "buyer_cart_request",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["delegate.create_purchase_mandate"],
                        "inbound_sender_roles": ["consumer"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.request_cart_quote",
                            "destination": "platform:checkout",
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
                    "phase_id": "buyer_cart_quote",
                    "match": {
                        "actor_roles": ["buyer"],
                        "inbound_action_kinds": ["platform.cart_quote"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.request_cart_quote",
                            "destination": "platform:checkout",
                        },
                        {
                            "action_kind": "platform.checkout_cart",
                            "destination": "platform:checkout",
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
                    "phase_id": "buyer_cart_authorization_notice",
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
                {
                    "phase_id": "merchant_cart_request",
                    "match": {
                        "actor_roles": ["merchant"],
                        "inbound_action_kinds": ["platform.cart_quote_request"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [
                        {
                            "action_kind": "commerce.request_cart_quote",
                            "destination": "platform:checkout",
                        }
                    ],
                    "world_reads": "skill_scoped",
                    "finish": "forbid",
                },
                {
                    "phase_id": "merchant_cart_quote",
                    "match": {
                        "actor_roles": ["merchant"],
                        "inbound_action_kinds": ["platform.cart_quote"],
                        "inbound_sender_roles": ["platform"],
                    },
                    "allowed_routes": [],
                    "world_reads": "deny",
                    "finish": "framework_terminal",
                },
            ],
        },
    }
    if case.definition.evaluated_role == "buyer":
        if case.buyer_problem is None:
            raise RuntimeBenchmarkIntegrityError("T5 Buyer planning problem is missing")
        facts["cart_planning_problem"] = public_t5_problem_copy_v1(case.buyer_problem)
    else:
        facts["pricing_terms"] = _public_pricing_terms(case)
    return facts


def _merchant_kickoff(case: _CaseT5) -> dict[str, Any]:
    """A real buyer intent routed through Platform into World authorization."""

    return {
        "msg_id": f"kickoff:{case.definition.task_id}:cart-authorization",
        "ts": "2026-07-16T12:00:00Z",
        "from": _BUYER_ID,
        "to": "platform:checkout",
        "idempotency_key": f"authorize:{case.definition.task_id.casefold()}",
        "action": {
            "kind": "commerce.create_cart_quote_request",
            "payload": {
                "market_id": case.market_id,
                "mandate_id": case.definition.task_id,
                "lines": [
                    {"sku_id": sku_id, "qty": qty} for sku_id, qty in case.merchant_requested_lines
                ],
                "fill_policy": "all_or_none",
                "backorder_policy": "reject",
            },
        },
    }


def scenario_for_t5(task_id: str) -> ScenarioSpec:
    """Materialize one T5 task without benchmark-local commerce state."""

    case = _case_for_t5(task_id)
    contract = _public_contract(case)
    task_context = _public_task_context(case)
    by_merchant: dict[str, list[str]] = {merchant_id: [] for merchant_id in case.merchant_ids}
    for row in case.catalog:
        by_merchant[str(row["merchant_id"])].append(str(row["sku_id"]))
    merchants = tuple(
        MerchantSpec(
            merchant_id=merchant_id,
            persona={"name": f"T5 merchant {merchant_id}", "task_family": "T5"},
            policy={
                "floor_price": 731,
                "margin_target_bps": 1_000,
                "max_negotiation_rounds": 1,
                "refund_policy": "30_day_return",
                "claim_aggressiveness": "neutral",
                "task_context": task_context,
                "benchmark_contract": contract,
            },
            catalog_scope=tuple(by_merchant[merchant_id]),
        )
        for merchant_id in case.merchant_ids
    )
    if case.definition.evaluated_role == "buyer":
        if case.buyer_problem is None or case.buyer_oracle is None:
            raise RuntimeBenchmarkIntegrityError("T5 Buyer oracle is missing")
        mandate_quantity = sum(
            int(row["required_qty"]) for row in case.buyer_problem["requirements"]
        )
        delivery_days = int(case.buyer_problem["hard_constraints"]["max_delivery_days"])
        buyer_goal = (
            "Choose the unique lexicographic optimum among the declared feasible carts "
            "using all public prices, tiers, bundles, charges, inventory, ETA, and relations."
        )
        expected_line_count = len(case.buyer_oracle.optimum)
    else:
        mandate_quantity = sum(qty for _, qty in case.merchant_requested_lines)
        delivery_days = 30
        buyer_goal = "complete the authoritative multi-line cart contract"
        expected_line_count = len(case.merchant_requested_lines)
    buyer = BuyerSpec(
        buyer_id=_BUYER_ID,
        persona={"name": "T5 benchmark buyer", "task_family": "T5"},
        mandate={
            "mandate_id": case.definition.task_id,
            "goal": buyer_goal,
            "quantity": mandate_quantity,
            "return_after_purchase": False,
            "hard_constraints": {
                "budget": case.budget_minor,
                "delivery_days": delivery_days,
                "must_have": [],
            },
            "soft_constraints": [],
            "soft_preferences": {"style": [], "avoid": []},
            "authority": {
                "can_buy_without_confirmation": True,
                "must_not_share_with_merchant": ["budget"],
            },
            "intent_expiry": "2099-12-31T00:00:00Z",
            "cart_quote_authority": {
                "market_id": case.market_id,
                "mandate_id": case.definition.task_id,
                "fill_policy": "all_or_none",
                "backorder_policy": "reject",
            },
            "task_context": task_context,
            "benchmark_contract": contract,
        },
    )
    authority = MandateRevisionAuthority(
        principal_id=_PRINCIPAL_ID,
        buyer_id=_BUYER_ID,
        mandate_id=case.definition.task_id,
        allowed_fields=("budget_minor",),
    )
    revision = build_mandate_revision(
        principal_id=_PRINCIPAL_ID,
        buyer_id=_BUYER_ID,
        mandate_id=case.definition.task_id,
        revision=1,
        previous_digest=None,
        changes={"budget_minor": case.budget_minor},
        authorized_fields=("budget_minor",),
        logical_tick=0,
    )
    initial_events: tuple[dict[str, Any], ...] = ()
    if case.definition.evaluated_role == "merchant":
        initial_events = (_merchant_kickoff(case),)
    population = PopulationSpec(
        buyers=(buyer,),
        merchants=merchants,
        initial_events=initial_events,
        matching={"top_k": max(1, len(case.catalog))},
        execution={"max_transactions_per_buyer": 1},
    )
    return ScenarioSpec(
        scenario_id=f"{task_id.casefold().replace('-', '_')}__runtime",
        seed=int(case.definition.canonical_hash[:8], 16) % 2_147_483_646 + 1,
        initial_state={
            "catalog": [copy.deepcopy(row) for row in case.catalog],
            "pricing_policy_fixtures": [copy.deepcopy(row) for row in case.pricing_fixtures],
            "mandate_authorities": [mandate_authority_to_wire(authority)],
            "mandate_revisions": [mandate_revision_to_dict(revision)],
        },
        buyer_goal={},
        merchant_policy={},
        allowed_actions=("search", "send_message"),
        success_oracle={
            "schema_version": T5_RUNTIME_SCHEMA_V2,
            "task_id": task_id,
            "lane": case.lane,
            "expected_line_count": expected_line_count,
            "expected_atomic_order_groups": 1,
        },
        platform_policy=None,
        population=population,
    )


def _business_request(user_prompt: str) -> dict[str, Any]:
    start = user_prompt.find("{")
    if start < 0:
        raise ValueError("Agent business prompt contains no request object")
    value = json.loads(user_prompt[start:])
    if not isinstance(value, dict):
        raise ValueError("Agent business request must be an object")
    return value


def _allowed_intent(
    request: Mapping[str, Any],
    intent: str,
) -> Mapping[str, Any] | None:
    rows = request.get("allowed_intents")
    if not isinstance(rows, list):
        raise ValueError("Agent business request has no allowed intents")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("intent") == intent]
    if len(matches) > 1:
        raise ValueError(f"business intent {intent!r} is ambiguous")
    return matches[0] if matches else None


def _intent_refs(
    request: Mapping[str, Any],
    intent: str,
    field: str,
) -> tuple[str, ...]:
    row = _allowed_intent(request, intent)
    parameters = row.get("parameters") if isinstance(row, Mapping) else None
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    schema = properties.get(field) if isinstance(properties, Mapping) else None
    values = schema.get("enum") if isinstance(schema, Mapping) else None
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{intent}.{field} has no finite public reference set")
    return tuple(values)


def _walk_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        rows.append(value)
        for item in value.values():
            rows.extend(_walk_mappings(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_walk_mappings(item))
    return tuple(rows)


def _visible_merchant_cart_lines(request: Mapping[str, Any]) -> CartLines:
    """Read an inbound merchant quote request solely from provider-visible facts."""

    candidates: list[tuple[tuple[str, int], ...]] = []
    for row in (
        *_walk_mappings(request.get("goal")),
        *_walk_mappings(request.get("observations")),
    ):
        raw = row.get("lines")
        if not isinstance(raw, list) or not raw:
            continue
        parsed: list[tuple[str, int]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                parsed = []
                break
            sku_ref = item.get("sku_ref")
            qty = item.get("qty")
            if (
                not isinstance(sku_ref, str)
                or not sku_ref
                or isinstance(qty, bool)
                or not isinstance(qty, int)
                or qty <= 0
            ):
                parsed = []
                break
            parsed.append((sku_ref, qty))
        if parsed:
            candidate = tuple(parsed)
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        raise ValueError("business request omits public cart lines")
    longest = max(len(row) for row in candidates)
    matches = [row for row in candidates if len(row) == longest]
    if len(matches) != 1:
        raise ValueError("business request contains ambiguous public cart lines")
    return matches[0]


def _visible_cart_planning_problem(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Locate the one self-contained public planning problem in this request."""

    candidates: dict[str, Mapping[str, Any]] = {}
    for row in (
        *_walk_mappings(request.get("goal")),
        *_walk_mappings(request.get("observations")),
    ):
        if row.get("rule_set") != T5_CART_PLANNING_RULE_SET_V1:
            continue
        candidate = dict(row)
        candidates[canonical_sha256(candidate)] = candidate
    if len(candidates) != 1:
        raise ValueError("business request omits or ambiguously repeats cart planning facts")
    return next(iter(candidates.values()))


def _visible_pricing_terms(request: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Read merchant pricing rules only from the provider-visible request."""

    candidates: list[tuple[Mapping[str, Any], ...]] = []
    for row in (
        *_walk_mappings(request.get("goal")),
        *_walk_mappings(request.get("observations")),
    ):
        raw = row.get("pricing_terms")
        if (
            not isinstance(raw, list)
            or not raw
            or any(not isinstance(item, Mapping) for item in raw)
        ):
            continue
        candidate = tuple(dict(item) for item in raw)
        if canonical_sha256(candidate) not in {
            canonical_sha256(existing) for existing in candidates
        }:
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError("business request omits or ambiguously repeats pricing terms")
    return candidates[0]


def _merchant_quote_conclusion(
    request: Mapping[str, Any],
    lines: tuple[tuple[str, int], ...],
    *,
    mutated: bool,
) -> dict[str, Any]:
    """Apply visible tier, bundle, and charge rules without using an oracle."""

    quantities = dict(lines)
    unit_prices: dict[str, int] = {}
    rule_kinds: dict[str, list[str]] = {}
    charges: list[dict[str, Any]] = []
    for term in _visible_pricing_terms(request):
        sku_refs = term.get("sku_refs")
        tiers = term.get("quantity_tiers")
        if (
            not isinstance(sku_refs, list)
            or not sku_refs
            or any(not isinstance(ref, str) or ref not in quantities for ref in sku_refs)
            or not isinstance(tiers, list)
            or not tiers
        ):
            raise ValueError("visible pricing term has invalid SKU or tier scope")
        total_qty = sum(quantities[ref] for ref in sku_refs)
        active = next(
            (
                tier
                for tier in tiers
                if isinstance(tier, Mapping)
                and total_qty >= int(tier.get("minimum_quantity", 0))
                and (
                    tier.get("maximum_quantity") is None
                    or total_qty <= int(tier["maximum_quantity"])
                )
            ),
            None,
        )
        if active is None:
            raise ValueError("visible pricing terms have no active quantity tier")
        base_price = int(active["unit_price_minor"])
        for ref in sku_refs:
            unit_prices[ref] = base_price
            rule_kinds[ref] = ["quantity_tier" if len(tiers) > 1 else "catalog_base"]

        base_subtotal = base_price * total_qty
        discounts: list[int] = []
        raw_bundles = term.get("bundle_discounts", [])
        if not isinstance(raw_bundles, list):
            raise ValueError("visible bundle rules must be an array")
        for bundle in raw_bundles:
            conditions = bundle.get("conditions") if isinstance(bundle, Mapping) else None
            if not isinstance(conditions, list):
                raise ValueError("visible bundle rule has no conditions")
            satisfied = all(
                isinstance(condition, Mapping)
                and isinstance(condition.get("sku_ref"), str)
                and quantities.get(str(condition["sku_ref"]), 0)
                >= int(condition.get("minimum_quantity", 0))
                for condition in conditions
            )
            if not satisfied:
                continue
            discount_minor = bundle.get("discount_minor")
            discount_bps = bundle.get("discount_bps")
            discounts.append(
                int(discount_minor)
                if discount_minor is not None
                else base_subtotal * int(discount_bps) // 10_000
            )
        discount = min(base_subtotal, max(discounts, default=0))
        remaining = discount
        for ref in sorted(sku_refs):
            qty = quantities[ref]
            reduction = min(unit_prices[ref] - 1, remaining // qty)
            unit_prices[ref] -= reduction
            remaining -= reduction * qty
            if discount:
                rule_kinds[ref].append("bundle_discount")
        if remaining:
            raise ValueError("visible bundle discount cannot be represented per unit")

        raw_charges = term.get("charges", [])
        if not isinstance(raw_charges, list):
            raise ValueError("visible charge rules must be an array")
        for component in raw_charges:
            if not isinstance(component, Mapping):
                raise ValueError("visible charge rule must be an object")
            lower = int(component.get("minimum_subtotal_minor", 0))
            upper = component.get("maximum_subtotal_minor")
            if base_subtotal < lower or (upper is not None and base_subtotal >= int(upper)):
                continue
            amount = (
                int(component.get("fixed_minor", 0))
                + int(component.get("per_unit_minor", 0)) * total_qty
                + base_subtotal * int(component.get("subtotal_rate_bps", 0)) // 10_000
            )
            charges.append({"kind": str(component["kind"]), "amount_minor": amount})

    if set(unit_prices) != set(quantities):
        raise ValueError("visible pricing terms do not cover the requested cart")
    line_quotes = [
        {
            "sku_ref": ref,
            "qty": qty,
            "unit_price_minor": unit_prices[ref],
            "line_total_minor": unit_prices[ref] * qty,
            "applied_rule_kinds": rule_kinds[ref],
        }
        for ref, qty in lines
    ]
    subtotal = sum(int(row["line_total_minor"]) for row in line_quotes)
    grand_total = subtotal + sum(int(row["amount_minor"]) for row in charges)
    if mutated:
        first = line_quotes[0]
        first["unit_price_minor"] = int(first["unit_price_minor"]) + 1
        delta = int(first["qty"])
        first["line_total_minor"] = int(first["line_total_minor"]) + delta
        subtotal += delta
        grand_total += delta
    return {
        "line_quotes": line_quotes,
        "charges": charges,
        "subtotal_minor": subtotal,
        "grand_total_minor": grand_total,
    }


def _business_response(intent: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "intent": intent,
            "arguments": copy.deepcopy(dict(arguments)),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class _TypedT5Channel:
    """Reviewed cart policy over the sole typed business-decision seam."""

    supports_business_decisions = True
    supports_decision_evidence_context = True

    def __init__(self, *, role: str, mutated: bool = False, checkout_only: bool = False) -> None:
        self._role = role
        self._mutated = mutated
        self._checkout_only = checkout_only
        self._read_refs: set[str] = set()

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        del system_prompt
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("T5 business channel requires Agent decision evidence")
        request = _business_request(user_prompt)
        if request.get("role") != self._role:
            raise ValueError("T5 business request crossed actor roles")
        intent, arguments = self._choose(request)
        content = _business_response(intent, arguments)
        return BusinessDecisionResponseV1(
            content=content,
            response_chars=len(content),
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def _choose(self, request: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        if _allowed_intent(request, "checkout_cart") is not None:
            return "checkout_cart", {}
        if self._checkout_only:
            if _allowed_intent(request, "finish") is not None:
                return "finish", {"reason": "No cart checkout is pending."}
            raise ValueError("buyer counterpart has no legal cart checkout choice")

        if self._role == "merchant":
            visible_lines = _visible_merchant_cart_lines(request)
            quote_lines = visible_lines
        else:
            public_oracle = build_t5_cart_oracle_v1(
                _visible_cart_planning_problem(request),
                reference_kind="public",
            )
            quote_lines = public_oracle.runner_up if self._mutated else public_oracle.optimum

        read_intent = _allowed_intent(request, "observe_listing")
        if read_intent is not None:
            allowed_refs = frozenset(_intent_refs(request, "observe_listing", "sku_ref"))
            for sku_ref, _qty in quote_lines:
                if sku_ref not in allowed_refs:
                    raise ValueError("public cart line is outside listing-read authority")
                if sku_ref not in self._read_refs:
                    self._read_refs.add(sku_ref)
                    return "observe_listing", {"sku_ref": sku_ref}

        if _allowed_intent(request, "request_cart_quote") is not None:
            if self._role == "merchant":
                return "request_cart_quote", _merchant_quote_conclusion(
                    request,
                    quote_lines,
                    mutated=self._mutated,
                )
            return "request_cart_quote", {
                "lines": [{"sku_ref": sku_ref, "qty": qty} for sku_ref, qty in quote_lines]
            }
        raise ValueError("T5 business phase has no legal quote decision")


class _NoReplyT5(_TypedT5Channel):
    def __init__(self) -> None:
        super().__init__(role="merchant", checkout_only=True)


def _channel_for(case: _CaseT5, *, mutated: bool = False) -> InferenceChannel:
    return _TypedT5Channel(
        role=case.definition.evaluated_role,
        mutated=mutated,
    )


def targeted_mutation_channel_t5(task_id: str) -> InferenceChannel:
    case = _case_for_t5(task_id)
    return _channel_for(case, mutated=True)


def _action_payload(envelope: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        return {}
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    return dict(payload) if isinstance(payload, Mapping) else {}


def _table(snapshot: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    tables = snapshot.get("tables")
    value = tables.get(name) if isinstance(tables, Mapping) else None
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _money_minor(value: Any) -> int:
    if not isinstance(value, Mapping) or set(value) != {"amount", "currency"}:
        raise ValueError("typed Money object required")
    amount = Decimal(str(value["amount"])) * Decimal(100)
    if amount != amount.to_integral_value():
        raise ValueError("Money must use integral minor units")
    return int(amount)


def _request_lines(payload: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    raw = payload.get("lines")
    if not isinstance(raw, list):
        return ()
    output: list[tuple[str, int]] = []
    for row in raw:
        if not isinstance(row, Mapping):
            return ()
        try:
            output.append((str(row["sku_id"]), int(row["qty"])))
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(output)


def _grounded_skus(evidence: RuntimeEvidenceBundleV2, actor_id: str) -> frozenset[str]:
    return frozenset(attested_world_catalog_reads_v2(evidence, actor_id=actor_id))


def _canonical_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _quote_math_valid(
    quote: Mapping[str, Any],
    group: Mapping[str, Any],
    orders: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> bool:
    try:
        typed = persistent_cart_quote_from_json(_canonical_text(quote))
        if sum(int(row["line_total_minor"]) for row in quote["lines"]) != int(
            quote["subtotal_minor"]
        ):
            return False
        charge_total = sum(int(row["amount_minor"]) for row in quote["charges"])
        if int(quote["grand_total_minor"]) != int(quote["subtotal_minor"]) + charge_total:
            return False
        if group.get("quote_id") != typed.quote_id or group.get("quote_hash") != typed.quote_digest:
            return False
        if _money_minor(group.get("subtotal")) != typed.subtotal_minor:
            return False
        if _money_minor(group.get("grand_total")) != typed.grand_total_minor:
            return False
        group_fees = group.get("fee_breakdown")
        if not isinstance(group_fees, list):
            return False
        if sum(_money_minor(row.get("amount")) for row in group_fees) != charge_total:
            return False
        lines_by_sku = {str(row["sku_id"]): row for row in quote["lines"]}
        orders_by_sku = {str(row.get("sku_id")): row for row in orders}
        if set(lines_by_sku) != set(orders_by_sku):
            return False
        for sku_id, line in lines_by_sku.items():
            order = orders_by_sku[sku_id]
            if int(order.get("qty", -1)) != int(line["fulfill_now_qty"]):
                return False
            if str(order.get("merchant_id")) != str(line["merchant_id"]):
                return False
            if _money_minor(order.get("agreed_price")) != int(line["unit_price_minor"]):
                return False
        receipt_total = sum(
            _money_minor(row.get("price")) * int(row.get("qty", 0)) for row in ledger
        )
        return receipt_total == typed.subtotal_minor
    except (CartQuoteContractError, KeyError, TypeError, ValueError, ArithmeticError):
        return False


def _merchant_rule_kinds_by_sku(case: _CaseT5) -> dict[str, tuple[str, ...]]:
    """Return the commercial rules a correct merchant decision must identify."""

    quantities = dict(case.merchant_requested_lines)
    result: dict[str, tuple[str, ...]] = {}
    for fixture in case.pricing_fixtures:
        intent = fixture.get("intent")
        if not isinstance(intent, Mapping):
            raise RuntimeBenchmarkIntegrityError("T5 pricing intent is malformed")
        sku_ids = intent.get("listing_ids")
        tiers = intent.get("quantity_tiers")
        bundles = intent.get("bundle_discounts")
        if (
            not isinstance(sku_ids, list)
            or not sku_ids
            or any(not isinstance(sku_id, str) for sku_id in sku_ids)
            or not isinstance(tiers, list)
            or not tiers
            or not isinstance(bundles, list)
        ):
            raise RuntimeBenchmarkIntegrityError("T5 pricing intent scope is malformed")
        total_qty = sum(quantities.get(sku_id, 0) for sku_id in sku_ids)
        active_tier = _active_tier(intent, total_qty)
        if active_tier is None:
            raise RuntimeBenchmarkIntegrityError(
                "T5 authoritative pricing policy has no active quantity tier"
            )
        base_subtotal = int(active_tier["unit_price_minor"]) * total_qty
        discounts: list[int] = []
        for bundle in bundles:
            conditions = bundle.get("conditions") if isinstance(bundle, Mapping) else None
            if not isinstance(conditions, list):
                raise RuntimeBenchmarkIntegrityError("T5 bundle rule is malformed")
            if not all(
                isinstance(condition, Mapping)
                and isinstance(condition.get("selector_id"), str)
                and quantities.get(str(condition["selector_id"]), 0)
                >= int(condition.get("minimum_quantity", 0))
                for condition in conditions
            ):
                continue
            discount_minor = bundle.get("discount_minor")
            discount_bps = bundle.get("discount_bps")
            discounts.append(
                int(discount_minor)
                if discount_minor is not None
                else base_subtotal * int(discount_bps) // 10_000
            )
        has_bundle_discount = min(base_subtotal, max(discounts, default=0)) > 0
        for sku_id in sku_ids:
            kinds = ["quantity_tier" if len(tiers) > 1 else "catalog_base"]
            if has_bundle_discount:
                kinds.append("bundle_discount")
            if sku_id in result:
                raise RuntimeBenchmarkIntegrityError(
                    "T5 pricing policies overlap on one requested SKU"
                )
            result[sku_id] = tuple(kinds)
    if set(result) != set(case.required_skus):
        raise RuntimeBenchmarkIntegrityError(
            "T5 pricing policies do not exactly cover the requested cart"
        )
    return result


def _verified_merchant_quote_choice(
    evidence: RuntimeEvidenceBundleV2,
    *,
    actor_id: str,
    quote_request_msg_id: str | None,
) -> tuple[VerifiedModelBusinessChoice | None, int]:
    """Extract one model quote exact-joined to the authoritative quote request."""

    try:
        choices = verified_model_business_choices(
            evidence,
            evaluated_actor_id=actor_id,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T5 model business-choice provenance is invalid"
        ) from exc
    quote_choices = tuple(
        row
        for row in choices
        if row.intent == "request_cart_quote"
        and row.action_kind == "commerce.request_cart_quote"
        and row.expects_platform_exchange
    )
    if (
        len(quote_choices) != 1
        or not isinstance(quote_request_msg_id, str)
        or quote_choices[0].emitted_msg_id != quote_request_msg_id
    ):
        return None, len(quote_choices)
    return quote_choices[0], 1


def _verified_buyer_cart_choice(
    case: _CaseT5,
    evidence: RuntimeEvidenceBundleV2,
    *,
    quote_request_msg_id: str | None,
) -> tuple[VerifiedModelBusinessChoice | None, CartLines, int, bool]:
    """Recover the model's cart choice and verify Agent reference compilation."""

    try:
        choices = verified_model_business_choices(
            evidence,
            evaluated_actor_id=case.evaluated_actor_id,
        )
    except TrackerEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "T5 Buyer business-choice provenance is invalid"
        ) from exc
    quote_choices = tuple(
        row
        for row in choices
        if row.intent == "request_cart_quote"
        and row.action_kind == "commerce.request_cart_quote"
        and row.expects_platform_exchange
    )
    joined = tuple(
        row
        for row in quote_choices
        if isinstance(quote_request_msg_id, str) and row.emitted_msg_id == quote_request_msg_id
    )
    if quote_request_msg_id is not None and len(joined) != 1:
        raise RuntimeBenchmarkIntegrityError(
            "T5 authoritative quote is not joined to one model cart decision"
        )
    choice = joined[0] if joined else quote_choices[-1] if quote_choices else None
    if choice is None:
        return None, (), 0, False

    raw_lines = choice.arguments.get("lines")
    if not isinstance(raw_lines, list):
        return choice, (), len(quote_choices), bool(joined)
    by_ref = {
        public_reference_alias_v1(str(row["sku_id"])): str(row["sku_id"]) for row in case.catalog
    }
    parsed: list[tuple[str, int]] = []
    for row in raw_lines:
        if not isinstance(row, Mapping):
            return choice, (), len(quote_choices), bool(joined)
        ref = row.get("sku_ref")
        qty = row.get("qty")
        if (
            not isinstance(ref, str)
            or ref not in by_ref
            or isinstance(qty, bool)
            or not isinstance(qty, int)
            or qty <= 0
        ):
            return choice, (), len(quote_choices), bool(joined)
        parsed.append((by_ref[ref], qty))
    internal_lines = tuple(
        sorted(parsed, key=lambda row: (public_reference_alias_v1(row[0]), row[1]))
    )
    if joined:
        if quote_request_msg_id is None:
            raise RuntimeBenchmarkIntegrityError("T5 joined quote request id disappeared")
        exchange_rows = evidence.accepted_platform_exchanges(
            kind="commerce.request_cart_quote",
            actor_id=case.evaluated_actor_id,
            endpoint="platform:checkout",
            response_kind="platform.cart_quote",
        )
        matching = [
            row for row in exchange_rows if row.request.get("msg_id") == quote_request_msg_id
        ]
        if len(matching) != 1:
            raise RuntimeBenchmarkIntegrityError(
                "T5 joined model cart has no unique accepted Platform exchange"
            )
        emitted = tuple(
            sorted(
                _request_lines(_action_payload(matching[0].request)),
                key=lambda row: (public_reference_alias_v1(row[0]), row[1]),
            )
        )
        if emitted != internal_lines:
            raise RuntimeBenchmarkIntegrityError(
                "T5 Agent changed the model-selected public cart while compiling references"
            )
    return choice, internal_lines, len(quote_choices), bool(joined)


def _merchant_quote_checks(
    case: _CaseT5,
    *,
    choice: VerifiedModelBusinessChoice | None,
    choice_count: int,
    quote: Mapping[str, Any],
    verified_authority: bool,
) -> tuple[RuntimeRubricCheckV2, ...]:
    """Score only model-authored quote semantics, never World execution quality."""

    expected_lines: dict[str, Mapping[str, Any]] = {}
    expected_rules = _merchant_rule_kinds_by_sku(case)
    raw_expected_lines = quote.get("lines") if isinstance(quote, Mapping) else None
    if isinstance(raw_expected_lines, list):
        for line in raw_expected_lines:
            if not isinstance(line, Mapping) or not isinstance(line.get("sku_id"), str):
                expected_lines = {}
                break
            sku_id = str(line["sku_id"])
            expected_lines[public_reference_alias_v1(sku_id)] = line

    raw_arguments = choice.arguments if choice is not None else {}
    raw_observed_lines = raw_arguments.get("line_quotes")
    observed_lines: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_observed_lines, list):
        for line in raw_observed_lines:
            ref = line.get("sku_ref") if isinstance(line, Mapping) else None
            if not isinstance(ref, str) or ref in observed_lines:
                observed_lines = {}
                break
            observed_lines[ref] = line

    expected_scope = tuple(
        sorted((ref, int(line.get("requested_qty", -1))) for ref, line in expected_lines.items())
    )
    observed_scope = tuple(
        sorted((ref, int(line.get("qty", -1))) for ref, line in observed_lines.items())
    )
    decision_bound = bool(choice is not None and verified_authority and expected_lines)
    scope_ok = decision_bound and observed_scope == expected_scope

    expected_rule_rows = tuple(
        sorted(
            (
                public_reference_alias_v1(sku_id),
                tuple(sorted(kinds)),
            )
            for sku_id, kinds in expected_rules.items()
        )
    )
    observed_rule_rows = tuple(
        sorted(
            (
                ref,
                tuple(sorted(map(str, line.get("applied_rule_kinds", [])))),
            )
            for ref, line in observed_lines.items()
        )
    )
    rules_ok = scope_ok and observed_rule_rows == expected_rule_rows

    expected_price_rows = tuple(
        sorted(
            (
                ref,
                int(line.get("unit_price_minor", -1)),
                int(line.get("line_total_minor", -1)),
            )
            for ref, line in expected_lines.items()
        )
    )
    observed_price_rows = tuple(
        sorted(
            (
                ref,
                int(line.get("unit_price_minor", -1)),
                int(line.get("line_total_minor", -1)),
            )
            for ref, line in observed_lines.items()
        )
    )
    # A fixed bundle discount may leave an indivisible remainder. World
    # allocates that remainder using Agent-private SKU identities, while the
    # model sees only opaque public references. Compare the price multiset
    # within each public pricing-policy group so hidden ID ordering cannot
    # affect model credit; quantity, arithmetic, subtotal, and policy scope
    # remain exact.
    price_group_by_ref: dict[str, int] = {}
    for group_index, fixture in enumerate(case.pricing_fixtures):
        intent = fixture.get("intent")
        sku_ids = intent.get("listing_ids") if isinstance(intent, Mapping) else None
        if not isinstance(sku_ids, list):
            raise RuntimeBenchmarkIntegrityError("T5 pricing group is malformed")
        for sku_id in sku_ids:
            price_group_by_ref[public_reference_alias_v1(str(sku_id))] = group_index
    expected_grouped_prices = tuple(
        sorted(
            (
                price_group_by_ref.get(ref, -1),
                int(expected_lines[ref].get("requested_qty", -1)),
                unit_price,
                line_total,
            )
            for ref, unit_price, line_total in expected_price_rows
        )
    )
    observed_grouped_prices = tuple(
        sorted(
            (
                price_group_by_ref.get(ref, -1),
                int(observed_lines[ref].get("qty", -1)),
                unit_price,
                line_total,
            )
            for ref, unit_price, line_total in observed_price_rows
        )
    )
    observed_line_math_ok = all(
        int(line.get("line_total_minor", -1))
        == int(line.get("unit_price_minor", -1)) * int(line.get("qty", -1))
        for line in observed_lines.values()
    )
    expected_subtotal = int(quote.get("subtotal_minor", -1)) if quote else -1
    observed_subtotal = raw_arguments.get("subtotal_minor")
    prices_ok = bool(
        scope_ok
        and observed_line_math_ok
        and observed_grouped_prices == expected_grouped_prices
        and observed_subtotal == expected_subtotal
    )

    raw_expected_charges = quote.get("charges") if isinstance(quote, Mapping) else None
    expected_charges = tuple(
        sorted(
            (str(row.get("kind")), int(row.get("amount_minor", -1)))
            for row in raw_expected_charges or ()
            if isinstance(row, Mapping)
        )
    )
    raw_observed_charges = raw_arguments.get("charges")
    observed_charges = tuple(
        sorted(
            (str(row.get("kind")), int(row.get("amount_minor", -1)))
            for row in raw_observed_charges or ()
            if isinstance(row, Mapping)
        )
    )
    expected_grand_total = int(quote.get("grand_total_minor", -1)) if quote else -1
    observed_grand_total = raw_arguments.get("grand_total_minor")
    totals_ok = bool(
        decision_bound
        and isinstance(raw_observed_charges, list)
        and observed_charges == expected_charges
        and observed_grand_total == expected_grand_total
    )

    common = {
        "model_quote_choice_count": choice_count,
        "model_quote_exact_joined": choice is not None,
        "verified_quote_authority": verified_authority,
    }
    return (
        RuntimeRubricCheckV2(
            "quote_scope_and_quantities",
            0.20,
            float(scope_ok),
            {
                **common,
                "expected": [list(row) for row in expected_scope],
                "observed": [list(row) for row in observed_scope],
            },
        ),
        RuntimeRubricCheckV2(
            "tier_or_bundle_rule_application",
            0.20,
            float(rules_ok),
            {
                **common,
                "expected": [[ref, list(kinds)] for ref, kinds in expected_rule_rows],
                "observed": [[ref, list(kinds)] for ref, kinds in observed_rule_rows],
            },
        ),
        RuntimeRubricCheckV2(
            "line_price_and_subtotal_accuracy",
            0.20,
            float(prices_ok),
            {
                **common,
                "expected_lines": [list(row) for row in expected_price_rows],
                "observed_lines": [list(row) for row in observed_price_rows],
                "expected_subtotal_minor": expected_subtotal,
                "observed_subtotal_minor": observed_subtotal,
            },
        ),
        RuntimeRubricCheckV2(
            "charge_and_grand_total_accuracy",
            0.20,
            float(totals_ok),
            {
                **common,
                "expected_charges": [list(row) for row in expected_charges],
                "observed_charges": [list(row) for row in observed_charges],
                "expected_grand_total_minor": expected_grand_total,
                "observed_grand_total_minor": observed_grand_total,
            },
        ),
    )


def _active_tier(policy: Mapping[str, Any], quantity: int) -> Mapping[str, Any] | None:
    tiers = policy.get("quantity_tiers")
    if not isinstance(tiers, list):
        return None
    for tier in tiers:
        if not isinstance(tier, Mapping):
            return None
        minimum = int(tier.get("minimum_quantity", 0))
        maximum = tier.get("maximum_quantity")
        if quantity >= minimum and (maximum is None or quantity <= int(maximum)):
            return tier
    return None


def _accepted_exchange(
    evidence: RuntimeEvidenceBundleV2,
    *,
    kind: str,
    actor_id: str,
    response_kind: str,
) -> Any | None:
    rows = evidence.accepted_platform_exchanges(
        kind=kind,
        actor_id=actor_id,
        endpoint="platform:checkout",
        response_kind=response_kind,
    )
    return rows[0] if len(rows) == 1 else None


def _pricing_fixture_matches(case: _CaseT5, evidence: RuntimeEvidenceBundleV2) -> bool:
    """Verify World-derived pricing revisions against the frozen task fixture."""

    def unordered_rows(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(sorted(_canonical_text(row) for row in value if isinstance(row, Mapping)))

    rows = [
        row
        for row in _table(evidence.initial_world, "pricing_policy_revisions")
        if row.get("market_id") == case.market_id
    ]
    expected = {str(fixture["intent"]["policy_id"]): fixture for fixture in case.pricing_fixtures}
    observed = {str(row.get("policy_id")): row for row in rows if row.get("policy_id")}
    if set(observed) != set(expected) or len(rows) != len(expected):
        return False
    for policy_id, fixture in expected.items():
        intent = fixture["intent"]
        row = observed[policy_id]
        if not all(
            (
                row.get("market_id") == intent.get("market_id"),
                row.get("merchant_id") == fixture.get("merchant_id"),
                row.get("owner_id") == fixture.get("merchant_id"),
                row.get("actor_id") == fixture.get("merchant_id"),
                row.get("idempotency_key") == fixture.get("idempotency_key"),
                row.get("listing_ids") == intent.get("listing_ids"),
                row.get("product_ids") == intent.get("product_ids"),
                row.get("quantity_tiers") == intent.get("quantity_tiers"),
                unordered_rows(row.get("bundle_discounts"))
                == unordered_rows(intent.get("bundle_discounts")),
                row.get("bundle_stacking") == intent.get("bundle_stacking"),
                unordered_rows(row.get("components")) == unordered_rows(intent.get("components")),
                row.get("revision") == 1,
            )
        ):
            return False
    return True


def _response_for(exchange: Any | None, kind: str, recipient: str) -> dict[str, Any] | None:
    if exchange is None:
        return None
    rows = [
        row
        for row in exchange.responses
        if row.get("to") == recipient and (row.get("action") or {}).get("kind") == kind
    ]
    return rows[0] if len(rows) == 1 else None


def _verified_discovery_commits(
    evidence: RuntimeEvidenceBundleV2,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Claim optional search and match work completed before cart actions."""
    verified = verify_optional_discovery_prefix_v2(
        evidence,
        buyer_id=_BUYER_ID,
    )
    return verified.commits, verified.evidence()


def _score_t5(case: _CaseT5, evidence: RuntimeEvidenceBundleV2) -> RuntimeTaskScoreV3:
    require_runtime_benchmark_integrity_v2(evidence)
    require_frozen_scenario_fixture_v2(
        evidence,
        scenario_for_t5(case.definition.task_id),
        family="T5",
    )
    if not _pricing_fixture_matches(case, evidence):
        raise RuntimeBenchmarkIntegrityError(
            "T5 pricing policy fixture is not faithfully represented in World"
        )
    discovery_commits, _discovery_evidence = _verified_discovery_commits(evidence)
    cart_options = {
        "market_id": case.market_id,
        "buyer_id": _BUYER_ID,
        "evaluated_actor_id": case.evaluated_actor_id,
        "preclaimed_commit_ids": tuple(str(row.get("commit_id")) for row in discovery_commits),
        # Re-quoting is a legal Agent business action. Every attempt must join
        # exactly to Platform and World, while only the checkout-bound quote is
        # eligible for task credit.
        "allow_repeated_quotes": True,
    }
    authority_error: str | None = None
    verified_prefix: VerifiedCartAuthorizationPrefixEvidence | None = None
    verified_quote_prefix: VerifiedCartQuotePrefixEvidence | None = None
    try:
        verified_cart = evidence.verified_operation_evidence(
            CART_EVIDENCE_CONTRACT,
            options=cart_options,
        )
        if not isinstance(verified_cart, VerifiedCartEvidence):
            raise RuntimeEvidenceError("cart evidence contract returned wrong type")
        authorization_exchange = verified_cart.authorization_exchange
        quote_exchange = verified_cart.quote_exchange
        checkout_exchange = verified_cart.checkout_exchange
    except RuntimeEvidenceError as exc:
        verified_cart = None
        authorization_exchange = quote_exchange = checkout_exchange = None
        authority_error = f"{type(exc).__name__}: {exc}"
        try:
            quote_prefix = evidence.verified_operation_evidence(
                CART_QUOTE_PREFIX_EVIDENCE_CONTRACT,
                options=cart_options,
            )
            if not isinstance(quote_prefix, VerifiedCartQuotePrefixEvidence):
                raise RuntimeEvidenceError("cart quote prefix contract returned wrong type")
            verified_quote_prefix = quote_prefix
            authorization_exchange = quote_prefix.authorization_exchange
            quote_exchange = quote_prefix.quote_exchange
        except RuntimeEvidenceError as quote_prefix_exc:
            authority_error += (
                f"; quote prefix: {type(quote_prefix_exc).__name__}: {quote_prefix_exc}"
            )
        if verified_quote_prefix is None and case.definition.evaluated_role == "merchant":
            try:
                prefix = evidence.verified_operation_evidence(
                    CART_AUTHORIZATION_PREFIX_EVIDENCE_CONTRACT,
                    options=cart_options,
                )
                if not isinstance(prefix, VerifiedCartAuthorizationPrefixEvidence):
                    raise RuntimeEvidenceError(
                        "cart authorization prefix contract returned wrong type"
                    )
                verified_prefix = prefix
                authorization_exchange = prefix.authorization_exchange
            except RuntimeEvidenceError as prefix_exc:
                authority_error += (
                    f"; authorization prefix: {type(prefix_exc).__name__}: {prefix_exc}"
                )

    claimed_cart_commits = (
        tuple(
            row
            for row in (
                verified_cart.request_commit,
                *verified_cart.quote_commits,
                verified_cart.checkout_commit,
            )
            if row is not None
        )
        if verified_cart is not None
        else (
            tuple(
                row
                for row in (
                    verified_quote_prefix.request_commit,
                    *verified_quote_prefix.quote_commits,
                )
                if row is not None
            )
            if verified_quote_prefix is not None
            else ((verified_prefix.request_commit,) if verified_prefix is not None else ())
        )
    )
    claimed_commits = tuple(
        sorted(
            (*discovery_commits, *claimed_cart_commits),
            key=lambda row: int(row.get("sequence", -1)),
        )
    )
    commit_claims = verify_exact_transaction_commit_claims(
        evidence.world_events,
        claimed_commits,
        allowed_authority_pairs={
            ("create_search_session", "world.create_search_session"),
            ("issue_match_certificate", "world.issue_match_certificate"),
            (
                "create_cart_quote_request",
                "world.create_cart_quote_request",
            ),
            ("issue_cart_quote", "world.issue_cart_quote"),
            ("checkout_cart_quote", "world.checkout_cart_quote"),
        },
    )

    accepted_authorizations = evidence.accepted_platform_exchanges(
        kind="commerce.create_cart_quote_request",
        endpoint="platform:checkout",
    )
    accepted_quotes = evidence.accepted_platform_exchanges(
        kind="commerce.request_cart_quote",
        endpoint="platform:checkout",
    )
    accepted_checkouts = evidence.accepted_platform_exchanges(
        kind="platform.checkout_cart",
        endpoint="platform:checkout",
    )

    quote_response = _response_for(quote_exchange, "platform.cart_quote", case.evaluated_actor_id)
    buyer_quote_response = _response_for(quote_exchange, "platform.cart_quote", _BUYER_ID)
    settlement_response = _response_for(checkout_exchange, "platform.cart_settlement", _BUYER_ID)
    quote_rows = _table(evidence.final_world, "persistent_cart_quotes")
    request_rows = _table(evidence.final_world, "persistent_cart_quote_requests")
    groups = _table(evidence.final_world, "order_groups")
    orders = _table(evidence.final_world, "orders")
    ledger = _table(evidence.final_world, "ledger")
    quote = (
        verified_cart.quote
        if verified_cart is not None
        else verified_quote_prefix.quote
        if verified_quote_prefix is not None
        else {}
    )
    group = groups[0] if len(groups) == 1 else {}

    quote_wire = _action_payload(quote_response).get("quote")
    buyer_quote_wire = _action_payload(buyer_quote_response).get("quote")
    settlement_wire = _action_payload(settlement_response).get("order_group")
    response_bound = (
        isinstance(quote_wire, Mapping)
        and quote_wire.get("quote_id") == quote.get("quote_id")
        and isinstance(buyer_quote_wire, Mapping)
        and buyer_quote_wire.get("quote_id") == quote.get("quote_id")
        and isinstance(settlement_wire, Mapping)
        and settlement_wire.get("order_group_id") == group.get("order_group_id")
    )
    if case.definition.evaluated_role == "buyer":
        # Direct buyer quotes have one response, used in both roles above.
        response_bound = (
            isinstance(quote_wire, Mapping)
            and quote_wire.get("quote_id") == quote.get("quote_id")
            and isinstance(settlement_wire, Mapping)
            and settlement_wire.get("order_group_id") == group.get("order_group_id")
        )

    commits = evidence.world_events
    quote_commits = [
        row
        for row in commits
        if row.get("operation") == "issue_cart_quote"
        and row.get("authority_action") == "world.issue_cart_quote"
        and row.get("actor_id") == case.evaluated_actor_id
    ]
    checkout_commits = [
        row
        for row in commits
        if row.get("operation") == "checkout_cart_quote"
        and row.get("authority_action") == "world.checkout_cart_quote"
        and row.get("actor_id") == _BUYER_ID
    ]
    request_commits = [
        row
        for row in commits
        if row.get("operation") == "create_cart_quote_request"
        and row.get("authority_action") == "world.create_cart_quote_request"
        and row.get("actor_id") == _BUYER_ID
    ]
    protocol_ok = (
        authority_error is None
        and verified_cart is not None
        and commit_claims.verified
        and quote_exchange is not None
        and checkout_exchange is not None
        and response_bound
        and len(quote_commits) == len(verified_cart.quote_commits)
        and len(quote_commits) >= 1
        and len(checkout_commits) == 1
        and (
            case.definition.evaluated_role == "buyer"
            or (authorization_exchange is not None and len(request_commits) == 1)
        )
        and evidence.evidence_manifest is not None
        and evidence.evidence_manifest.get("execution_backend") == COMMERCEWORLD_EPISODE_BACKEND_V2
    )

    grounded = _grounded_skus(evidence, case.evaluated_actor_id)
    grounding_credit = len(grounded.intersection(case.required_skus)) / len(case.required_skus)

    quote_math_ok = bool(quote and group) and _quote_math_valid(quote, group, orders, ledger)

    checkout_writes = (
        checkout_commits[0].get("table_writes", []) if len(checkout_commits) == 1 else []
    )
    write_tables = [str(row.get("table")) for row in checkout_writes if isinstance(row, Mapping)]
    atomic_ok = (
        len(groups) == 1
        and len(checkout_commits) == 1
        and len(orders) == len(ledger) == len(quote.get("lines", []))
        and write_tables.count("order_groups") == 1
        and write_tables.count("orders") == len(orders)
        and write_tables.count("ledger") == len(ledger)
        and write_tables.count("inventory") == len(orders)
        and write_tables.count("order_timelines") == len(orders)
        and write_tables.count("authority_operations") == 1
        and write_tables.count("logical_time") == 1
        and len(write_tables) == 3 + (4 * len(orders))
        and group.get("order_ids") is not None
        and set(map(str, group.get("order_ids", [])))
        == {str(row.get("order_id")) for row in orders}
        and set(map(str, group.get("txn_ids", []))) == {str(row.get("txn_id")) for row in ledger}
    )

    if not commit_claims.verified:
        raise RuntimeBenchmarkIntegrityError(
            "T5 cart operations do not form an exact World commit closure"
        )
    authorization_observed = bool(accepted_authorizations or request_commits or request_rows)
    quote_observed = bool(accepted_quotes or quote_commits or quote_rows)
    checkout_observed = bool(
        accepted_checkouts or checkout_exchange is not None or checkout_commits or groups
    )
    if checkout_observed and verified_cart is None:
        raise RuntimeBenchmarkIntegrityError(
            "T5 checkout is not bound to the complete cart authority contract"
        )
    if quote_observed and verified_cart is None and verified_quote_prefix is None:
        raise RuntimeBenchmarkIntegrityError(
            "T5 quote is not bound to the Platform and World quote contract"
        )
    if authorization_observed and all(
        row is None for row in (verified_cart, verified_quote_prefix, verified_prefix)
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T5 cart authorization is not bound to authoritative World evidence"
        )
    if checkout_observed and not (protocol_ok and quote_math_ok and atomic_ok):
        raise RuntimeBenchmarkIntegrityError(
            "T5 completed checkout contradicts Platform quote math or atomic World settlement"
        )
    evaluated_checkout_actions = evidence.actions(
        kind="platform.checkout_cart",
        actor_id=case.evaluated_actor_id,
    )
    evaluated_quote_authority = bool(
        quote_exchange is not None
        and quote_exchange.request.get("from") == case.evaluated_actor_id
        and verified_cart is not None
    )
    evaluated_checkout_authority = bool(
        checkout_exchange is not None
        and checkout_exchange.request.get("from") == case.evaluated_actor_id
        and verified_cart is not None
    )
    if (
        case.definition.evaluated_role == "merchant"
        and verified_quote_prefix is not None
        and quote_exchange is not None
        and quote_exchange.request.get("from") == case.evaluated_actor_id
        and verified_cart is None
    ):
        raise RuntimeBenchmarkIntegrityError(
            "T5 scripted buyer did not complete checkout after the merchant's verified quote"
        )
    grounding_check = RuntimeRubricCheckV2(
        "cart_evidence_coverage",
        0.20,
        grounding_credit,
        {
            "required_skus": list(case.required_skus),
            "grounded_skus": sorted(grounded),
            "model_observation_intent_required": True,
        },
    )
    if case.definition.evaluated_role == "buyer":
        if case.buyer_problem is None or case.buyer_oracle is None:
            raise RuntimeBenchmarkIntegrityError("T5 Buyer planning oracle is missing")
        quote_request_msg_id = (
            quote_exchange.request.get("msg_id") if quote_exchange is not None else None
        )
        model_choice, selected_lines, model_quote_count, exact_joined = _verified_buyer_cart_choice(
            case,
            evidence,
            quote_request_msg_id=(
                str(quote_request_msg_id) if isinstance(quote_request_msg_id, str) else None
            ),
        )
        evaluation = evaluate_t5_cart_lines_v1(
            case.buyer_problem,
            selected_lines,
            reference_kind="internal",
            public_reference=public_reference_alias_v1,
        )
        hard_credit = float(evaluation.hard_feasible)
        optimal = selected_lines in case.buyer_oracle.acceptable_plans
        execution_ok = bool(
            model_choice is not None
            and exact_joined
            and evaluation.hard_feasible
            and verified_cart is not None
            and protocol_ok
            and quote_math_ok
            and atomic_ok
            and len(evaluated_checkout_actions) == 1
            and evaluated_checkout_authority
        )
        if model_choice is not None and evaluation.hard_feasible and quote_exchange is None:
            raise RuntimeBenchmarkIntegrityError(
                "T5 feasible compiled cart did not receive an authoritative Platform quote"
            )
        if evaluated_checkout_actions and verified_cart is None:
            raise RuntimeBenchmarkIntegrityError(
                "T5 Buyer checkout decision did not produce complete authority closure"
            )
        checks = (
            RuntimeRubricCheckV2(
                "public_hard_constraints",
                0.30,
                hard_credit,
                {
                    "constraint_results": dict(evaluation.constraint_results),
                    "selected_lines": [
                        [public_reference_alias_v1(sku_id), qty] for sku_id, qty in selected_lines
                    ],
                    "model_quote_choice_count": model_quote_count,
                },
            ),
            RuntimeRubricCheckV2(
                "public_oracle_optimality",
                0.50,
                float(optimal),
                {
                    "acceptable_plan_count": len(case.buyer_oracle.acceptable_plans),
                    "selected_grand_total_minor": evaluation.grand_total_minor,
                    "selected_max_delivery_days": evaluation.max_delivery_days,
                    "selected_merchant_count": evaluation.merchant_count,
                },
            ),
            RuntimeRubricCheckV2(
                "quote_bound_execution",
                0.20,
                float(execution_ok),
                {
                    "model_cart_exact_joined": exact_joined,
                    "verified_cart_authority": verified_cart is not None,
                    "verified_checkout_authority": evaluated_checkout_authority,
                    "quote_math_valid": quote_math_ok,
                    "atomic_settlement_valid": atomic_ok,
                },
            ),
        )
    else:
        quote_request_msg_id = (
            quote_exchange.request.get("msg_id") if quote_exchange is not None else None
        )
        model_quote, model_quote_count = _verified_merchant_quote_choice(
            evidence,
            actor_id=case.evaluated_actor_id,
            quote_request_msg_id=(
                str(quote_request_msg_id) if isinstance(quote_request_msg_id, str) else None
            ),
        )
        # Buyer authorization, World quote generation, and scripted checkout
        # are validity prerequisites only. Merchant credit comes exclusively
        # from its exact-joined typed business choice and public evidence read.
        checks = (
            grounding_check,
            *_merchant_quote_checks(
                case,
                choice=model_quote,
                choice_count=model_quote_count,
                quote=quote,
                verified_authority=evaluated_quote_authority,
            ),
        )
    checks = renormalize_capability_checks_v2(checks)
    return score_checks(case.definition, checks)


def _mutation_targets(case: _CaseT5) -> tuple[str, ...]:
    if case.definition.evaluated_role == "merchant":
        return (
            "line_price_and_subtotal_accuracy",
            "charge_and_grand_total_accuracy",
        )
    return ("public_oracle_optimality",)


def runtime_bundle_t5(task_id: str) -> RuntimeTaskBundleV2:
    """Build one T5 task bundle bound to the real Episode runtime."""

    case = _case_for_t5(task_id)
    counterparts: dict[str, Any] = {}
    if case.definition.evaluated_role == "merchant":
        counterparts[_BUYER_ID] = lambda: _TypedT5Channel(
            role="buyer",
            checkout_only=True,
        )
    else:
        counterparts.update(
            {merchant_id: (lambda: _NoReplyT5()) for merchant_id in case.merchant_ids}
        )
    return RuntimeTaskBundleV2(
        task=case.definition,
        scenario=scenario_for_t5(task_id),
        evaluated_actor_id=case.evaluated_actor_id,
        ideal_channel=lambda: _channel_for(case),
        counterpart_channels=counterparts,
        scorer=lambda evidence: _score_t5(case, evidence),
        semantic_hash=canonical_sha256(case.semantic_contract),
        mutations=(
            RuntimeMutationV2(
                mutation_id=f"{task_id}:partial:01",
                channel=lambda: targeted_mutation_channel_t5(task_id),
                expected_changed_checks=_mutation_targets(case),
            ),
        ),
    )


def runtime_bundles_t5() -> tuple[RuntimeTaskBundleV2, ...]:
    return tuple(runtime_bundle_t5(task_id) for task_id in _T5_TASK_IDS)


__all__ = [
    "T5_RUNTIME_SCHEMA_V2",
    "runtime_bundle_t5",
    "runtime_bundles_t5",
    "scenario_for_t5",
    "targeted_mutation_channel_t5",
]
