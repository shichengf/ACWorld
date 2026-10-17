"""Deterministic assets for the compact homogeneous-model 5x5 market study.

The study is deliberately independent from the frozen benchmark-v2 corpus.  It
uses one fixed market to demonstrate bilateral multi-agent composition: five
merchant sessions quote first, five buyer sessions receive ordered top-2
candidates, and a deterministic clearing step settles at most one unit per
buyer and listing.  The model runner may consume the hidden ``market_study``
answer key, while ordinary agent prompts receive only their role-local state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from episode.scenario import seed_world
from evals.market_metrics import (
    AllocationOracle,
    BuyerValuation,
    Exposure,
    MarketTransaction,
    MerchantFloor,
)
from evals.serialize import to_canonical
from experiments.market_artifacts import build_market_artifact, verify_market_artifact
from world import AgentId, Money, Order, OrderId, OrderState, Receipt, SkuId, TxnId, World

if TYPE_CHECKING:
    from episode.types import ScenarioSpec


HOMOGENEOUS_MARKET_STUDY_SCHEMA = "cwe.homogeneous-market-study.v1"
SCENARIO_ID = "s18_homogeneous_matching_42_5x5"
VARIANT_ID = "S18"
TASK_FAMILY = "T8"
SEED = 42
PRODUCT_ID = "product:shared-electric-standing-desk"
TOP_K = 2
BUYER_IDS = tuple(f"buyer:b{index}" for index in range(1, 6))
MERCHANT_IDS = tuple(f"merchant:m{index}" for index in range(1, 6))


def listing_id(merchant_id: str) -> str:
    """Return the globally unique listing key owned by ``merchant_id``."""

    suffix = merchant_id.split(":", 1)[-1]
    return f"{merchant_id}:sku:homogeneous:{SEED}:{suffix}"


_MERCHANT_ROWS: tuple[dict[str, Any], ...] = (
    {
        "merchant_id": "merchant:m1",
        "list_price": 110,
        "floor_minor": 7600,
        "quality_score": 85,
        "shipping_days": 2,
        "inventory": 1,
        "preference_profiles": "studio executive",
        "warranty_years": 5,
    },
    {
        "merchant_id": "merchant:m2",
        "list_price": 104,
        "floor_minor": 7200,
        "quality_score": 78,
        "shipping_days": 1,
        "inventory": 1,
        "preference_profiles": "commuter planner",
        "warranty_years": 3,
    },
    {
        "merchant_id": "merchant:m3",
        "list_price": 122,
        "floor_minor": 9000,
        "quality_score": 94,
        "shipping_days": 3,
        "inventory": 1,
        "preference_profiles": "studio creator",
        "warranty_years": 7,
    },
    {
        "merchant_id": "merchant:m4",
        "list_price": 96,
        "floor_minor": 6700,
        "quality_score": 72,
        "shipping_days": 2,
        "inventory": 1,
        "preference_profiles": "planner executive",
        "warranty_years": 2,
    },
    {
        "merchant_id": "merchant:m5",
        "list_price": 114,
        "floor_minor": 8200,
        "quality_score": 88,
        "shipping_days": 1,
        "inventory": 1,
        "preference_profiles": "commuter creator",
        "warranty_years": 4,
    },
)

# Full hidden 5x5 value matrix.  Rows follow BUYER_IDS and columns follow
# MERCHANT_IDS.  Values are cents and are never included in participant prompts.
_VALUATIONS: tuple[tuple[int, ...], ...] = (
    (13600, 13200, 14200, 11200, 13000),
    (12000, 13500, 12400, 11800, 14000),
    (13200, 12800, 15000, 12000, 13800),
    (11900, 13300, 12700, 13900, 13200),
    (14500, 13100, 13600, 12600, 13400),
)

_BUYER_ROWS: tuple[dict[str, Any], ...] = (
    {
        "buyer_id": "buyer:b1",
        "profile": "studio",
        "budget_minor": 14500,
        "delivery_days": 4,
        "priority": "balanced price, quality, and delivery",
        "ranking_rule": "balanced_value",
    },
    {
        "buyer_id": "buyer:b2",
        "profile": "commuter",
        "budget_minor": 14200,
        "delivery_days": 2,
        "priority": "fast delivery, then price",
        "ranking_rule": "shipping_then_price",
    },
    {
        "buyer_id": "buyer:b3",
        "profile": "creator",
        "budget_minor": 15200,
        "delivery_days": 4,
        "priority": "quality, then delivery",
        "ranking_rule": "quality_then_shipping",
    },
    {
        "buyer_id": "buyer:b4",
        "profile": "planner",
        "budget_minor": 14000,
        "delivery_days": 3,
        "priority": "price, then delivery",
        "ranking_rule": "price_then_shipping",
    },
    {
        "buyer_id": "buyer:b5",
        "profile": "executive",
        "budget_minor": 14600,
        "delivery_days": 3,
        "priority": "quality, then warranty",
        "ranking_rule": "quality_then_warranty",
    },
)

# Each ordered pair is computed from the fixed public profile/quote fixture.
# The graph intentionally has top-choice contention at m1.  The reviewed
# trade-count-first clearing result requires b1 and b3 to use their fallbacks.
_CANDIDATE_MERCHANTS: dict[str, tuple[str, str]] = {
    "buyer:b1": ("merchant:m1", "merchant:m3"),
    "buyer:b2": ("merchant:m2", "merchant:m5"),
    "buyer:b3": ("merchant:m3", "merchant:m5"),
    "buyer:b4": ("merchant:m4", "merchant:m2"),
    "buyer:b5": ("merchant:m1", "merchant:m4"),
}


def build_scenario_spec() -> dict[str, Any]:
    """Return the fixed, Episode-loadable study scenario mapping."""

    catalog = [
        {
            "sku_id": listing_id(str(row["merchant_id"])),
            "merchant_id": row["merchant_id"],
            "product_id": PRODUCT_ID,
            "list_price": row["list_price"],
            "floor_price": int(row["floor_minor"]) // 100,
            "inventory": row["inventory"],
            "attributes": {
                "product_type": "electric standing desk",
                "in_stock": True,
                "quality_score": row["quality_score"],
                "shipping_days": row["shipping_days"],
                "preference_profiles": row["preference_profiles"],
                "warranty_years": row["warranty_years"],
            },
        }
        for row in _MERCHANT_ROWS
    ]
    buyers = [
        {
            "buyer_id": row["buyer_id"],
            "persona": {
                "name": f"market study buyer {str(row['buyer_id']).split(':')[-1]}",
                "preference_profile": row["profile"],
            },
            "mandate": {
                "mandate_id": f"study:matching:{row['buyer_id']}",
                "goal": f"electric standing desk {row['profile']}",
                "quantity": 1,
                "hard_constraints": {
                    "budget": row["budget_minor"],
                    "must_have": ["in_stock"],
                    "delivery_days": row["delivery_days"],
                },
                "soft_constraints": [],
                "soft_preferences": {
                    "profile": row["profile"],
                    "priority": row["priority"],
                    "ranking_rule": row["ranking_rule"],
                    "style": [],
                    "avoid": [],
                },
                "authority": {
                    "can_buy_without_confirmation": True,
                    "must_not_share_with_merchant": ["budget"],
                },
                "intent_expiry": "2099-12-31T00:00:00Z",
            },
        }
        for row in _BUYER_ROWS
    ]
    merchants = [
        {
            "merchant_id": row["merchant_id"],
            "persona": {
                "name": f"market study merchant {str(row['merchant_id']).split(':')[-1]}"
            },
            "policy": {
                "floor_price": row["floor_minor"],
                "quote_price": int(row["list_price"]) * 100,
                "refund_policy": "30_day_return",
                "max_negotiation_rounds": 0,
                "claim_aggressiveness": "neutral",
            },
            "catalog_scope": [listing_id(str(row["merchant_id"]))],
        }
        for row in _MERCHANT_ROWS
    ]
    valuations = [
        {
            "buyer_id": buyer_id,
            "merchant_id": merchant_id,
            "listing_id": listing_id(merchant_id),
            "unit_value_minor": _VALUATIONS[buyer_index][merchant_index],
        }
        for buyer_index, buyer_id in enumerate(BUYER_IDS)
        for merchant_index, merchant_id in enumerate(MERCHANT_IDS)
    ]
    floors = [
        {
            "merchant_id": row["merchant_id"],
            "listing_id": listing_id(str(row["merchant_id"])),
            "unit_floor_minor": row["floor_minor"],
            "capacity": row["inventory"],
        }
        for row in _MERCHANT_ROWS
    ]
    reference_allocation = _reference_allocation_rows()
    market_study = {
        "schema_version": HOMOGENEOUS_MARKET_STUDY_SCHEMA,
        "result_scope": "environment_composition_case_study_not_benchmark",
        "protocol": "merchant_quotes_then_top2_then_buyer_rank_then_clear_settle",
        "buyer_id_prefix": "buyer:",
        "candidate_policy": "public_profile_filter_then_role_local_ranking_rule_then_listing_id",
        "candidate_sets": {
            buyer_id.split(":", 1)[-1]: [listing_id(merchant_id) for merchant_id in merchants]
            for buyer_id, merchants in _CANDIDATE_MERCHANTS.items()
        },
        "reference_quotes": [
            {
                "merchant_id": row["merchant_id"],
                "listing_id": listing_id(str(row["merchant_id"])),
                "unit_price_minor": int(row["list_price"]) * 100,
                "quantity": 1,
            }
            for row in _MERCHANT_ROWS
        ],
        "reference_buyer_ranked_choices": {
            buyer_id.split(":", 1)[-1]: [listing_id(merchant_id) for merchant_id in merchants]
            for buyer_id, merchants in _CANDIDATE_MERCHANTS.items()
        },
        "reference_allocation": reference_allocation,
        "clearing_policy": "maximize_trade_count_then_first_choice_satisfaction_then_social_welfare_then_lexicographic",
        "top_choice_contention": {
            "listing_id": listing_id("merchant:m1"),
            "buyer_ids": ["buyer:b1", "buyer:b5"],
        },
        "global_optimal_welfare_minor": 31900,
        "top_k_reachable_optimal_welfare_minor": 31200,
        "unique_global_optimum": True,
        "unique_top_k_optimum": True,
    }
    primary = _MERCHANT_ROWS[2]
    return {
        "scenario_id": SCENARIO_ID,
        "seed": SEED,
        "initial_state": {"catalog": catalog},
        "buyer_goal": {
            "product_type": "electric standing desk studio",
            "max_budget": 145,
            "quantity": 1,
            "constraints": ["in_stock", "shipping_within_4_days"],
        },
        "merchant_policy": {
            "list_price": primary["list_price"],
            "floor_price": int(primary["floor_minor"]) // 100,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "population": {
            "buyers": buyers,
            "merchants": merchants,
            "initial_events": [],
            "matching": {"top_k": TOP_K},
            "execution": {"max_transactions_per_buyer": 1},
        },
        "allowed_actions": [
            "search",
            "get_sku",
            "propose_offer",
            "counter_offer",
            "accept_offer",
            "reject_offer",
            "create_order",
            "settle",
            "dispatch",
            "request_return",
            "issue_refund",
            "send_message",
        ],
        "success_oracle": {
            "expected_sku": listing_id("merchant:m3"),
            "product_match": True,
            "final_price_lte": 145,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
            "market_oracle": {
                "schema_version": "cwe.market-oracle.v1",
                "buyer_valuations": valuations,
                "merchant_floors": floors,
                "allocation_oracle": {
                    "optimal_social_welfare_minor": 31900,
                    "oracle_id": "homogeneous-study-exact-5x5-global-v1",
                },
            },
            "market_study": market_study,
        },
        "benchmark": {
            "task_family": TASK_FAMILY,
            "variant_id": VARIANT_ID,
            "track": "agent",
            "difficulty": "hard",
            "catalog_source": "inline",
            "catalog_scale": "smoke",
            "scenario_version": "environment-study-1.0",
        },
    }


def build_reference_market_artifact(scenario: "ScenarioSpec") -> dict[str, Any]:
    """Settle, replay, and score the fixed top-k reference allocation."""

    if scenario.scenario_id != SCENARIO_ID:
        raise ValueError(f"unexpected homogeneous-study scenario: {scenario.scenario_id!r}")
    public_candidates = compute_public_candidate_sets(scenario)
    if public_candidates != _CANDIDATE_MERCHANTS:
        raise RuntimeError("public top-k candidate computation drifted from the study oracle")
    valuations, floors = _metric_inputs(scenario)
    global_assignment = optimal_assignments(valuations, floors)
    reachable_edges = {
        (buyer_id, listing_id(merchant_id))
        for buyer_id, merchants in _CANDIDATE_MERCHANTS.items()
        for merchant_id in merchants
    }
    reachable_assignment = optimal_assignments(
        valuations,
        floors,
        allowed_edges=reachable_edges,
    )
    cleared_assignment = clear_ranked_candidates(valuations, floors)
    if _assignment_welfare(global_assignment) != 31900:
        raise RuntimeError("global reference welfare drifted")
    if _assignment_welfare(reachable_assignment) != 31200:
        raise RuntimeError("top-k reference welfare drifted")

    expected_pairs = {
        (str(row["buyer_id"]), str(row["listing_id"]))
        for row in _reference_allocation_rows()
    }
    if cleared_assignment != reachable_assignment:
        raise RuntimeError("reviewed clearing objective and top-k welfare optimum diverged")
    observed_pairs = {(row[0], row[2]) for row in cleared_assignment}
    if observed_pairs != expected_pairs:
        raise RuntimeError("top-k reference allocation drifted")

    world, transactions = _settle(scenario, cleared_assignment)
    replay_world, replay_transactions = _settle(scenario, cleared_assignment)
    state_digest = _world_digest(world)
    replay_digest = _world_digest(replay_world)
    if transactions != replay_transactions or state_digest != replay_digest:
        raise RuntimeError("homogeneous-study deterministic replay failed")

    exposures = tuple(
        Exposure(
            exposure_id=f"exposure:{buyer_id}:{rank}",
            buyer_id=buyer_id,
            merchant_id=merchant_id,
            listing_id=listing_id(merchant_id),
            weight=Decimal(TOP_K - rank + 1),
        )
        for buyer_id, merchants in sorted(_CANDIDATE_MERCHANTS.items())
        for rank, merchant_id in enumerate(merchants, 1)
    )
    artifact: dict[str, Any] = build_market_artifact(
        market_id=f"{SCENARIO_ID}:deterministic-reference",
        variant_id=VARIANT_ID,
        task_family=TASK_FAMILY,
        seed=SEED,
        buyer_ids=BUYER_IDS,
        merchant_ids=MERCHANT_IDS,
        valuations=valuations,
        merchant_floors=floors,
        transactions=transactions,
        exposures=exposures,
        # The model-free reference has no persisted participant observation
        # stream.  Keep both rates explicitly N/A instead of manufacturing
        # all-false events; owner-isolation tests are separate evidence.
        privacy_events=(),
        protocol_events=(),
        allocation_oracle=AllocationOracle(
            optimal_social_welfare_minor=31900,
            oracle_id="homogeneous-study-exact-5x5-global-v1",
        ),
        execution={
            "scenario_id": SCENARIO_ID,
            "result_scope": "environment_reference_baseline",
            "agent_kind": "deterministic_study_reference",
            "model_inference": False,
            "protocol": "merchant_quotes_then_top2_then_buyer_rank_then_clear_settle",
            "clearing_policy": "maximize_trade_count_then_first_choice_satisfaction_then_social_welfare_then_lexicographic",
            "top_k": TOP_K,
            "materialized_candidate_edges": len(reachable_edges),
            "full_valuation_edges": len(valuations),
            "global_optimal_welfare_minor": 31900,
            "top_k_reachable_optimal_welfare_minor": 31200,
            "top_k_reachability_ratio": "0.9780564263322884012539184953",
            "state_digest": state_digest,
            "replay_state_digest": replay_digest,
            "replay_ok": True,
        },
    )
    verify_market_artifact(artifact)
    return artifact


def compute_public_candidate_sets(
    scenario: "ScenarioSpec",
) -> dict[str, tuple[str, str]]:
    """Compute top-2 merchants using only public quotes and buyer-declared rules."""

    if scenario.population is None:
        raise ValueError("homogeneous market study requires an explicit population")
    catalog = tuple(scenario.initial_state.get("catalog", ()))
    results: dict[str, tuple[str, str]] = {}
    for buyer in scenario.population.buyers:
        preferences = buyer.mandate.get("soft_preferences", {})
        if not isinstance(preferences, Mapping):
            raise ValueError(f"{buyer.buyer_id} soft_preferences must be a mapping")
        profile = str(preferences.get("profile", ""))
        ranking_rule = str(preferences.get("ranking_rule", ""))
        matches = [
            row
            for row in catalog
            if isinstance(row, Mapping)
            and profile in str(row.get("attributes", {}).get("preference_profiles", "")).split()
        ]
        if len(matches) < TOP_K:
            raise ValueError(f"{buyer.buyer_id} has fewer than {TOP_K} public candidates")
        ranked = sorted(matches, key=lambda row: _public_rank_key(ranking_rule, row))
        results[buyer.buyer_id] = (
            str(ranked[0]["merchant_id"]),
            str(ranked[1]["merchant_id"]),
        )
    return results


def optimal_assignments(
    valuations: Sequence[BuyerValuation],
    floors: Sequence[MerchantFloor],
    *,
    allowed_edges: set[tuple[str, str]] | None = None,
) -> tuple[tuple[str, str, str, int], ...]:
    """Return the exact deterministic unit-demand allocation for this small study."""

    floor_index = {(row.merchant_id, row.listing_id): row for row in floors}
    by_buyer: dict[str, list[BuyerValuation]] = {buyer_id: [] for buyer_id in BUYER_IDS}
    for valuation in valuations:
        edge = (valuation.buyer_id, valuation.listing_id)
        if allowed_edges is None or edge in allowed_edges:
            by_buyer[valuation.buyer_id].append(valuation)
    best_score = -1
    best_key: tuple[str, ...] | None = None
    best: tuple[tuple[str, str, str, int], ...] = ()

    def visit(
        index: int,
        capacities: dict[tuple[str, str], int],
        selected: list[tuple[str, str, str, int]],
        score: int,
    ) -> None:
        nonlocal best_score, best_key, best
        if index == len(BUYER_IDS):
            key = tuple(f"{row[0]}->{row[1]}/{row[2]}" for row in selected)
            if score > best_score or (score == best_score and (best_key is None or key < best_key)):
                best_score, best_key, best = score, key, tuple(selected)
            return
        buyer_id = BUYER_IDS[index]
        visit(index + 1, capacities, selected, score)
        for valuation in sorted(
            by_buyer[buyer_id], key=lambda row: (row.merchant_id, row.listing_id)
        ):
            offer = (valuation.merchant_id, valuation.listing_id)
            if capacities.get(offer, 0) <= 0:
                continue
            floor = floor_index[offer]
            welfare = valuation.unit_value_minor - floor.unit_floor_minor
            if welfare <= 0:
                continue
            capacities[offer] -= 1
            selected.append((buyer_id, valuation.merchant_id, valuation.listing_id, welfare))
            visit(index + 1, capacities, selected, score + welfare)
            selected.pop()
            capacities[offer] += 1

    visit(
        0,
        {(row.merchant_id, row.listing_id): row.capacity for row in floors},
        [],
        0,
    )
    return best


def clear_ranked_candidates(
    valuations: Sequence[BuyerValuation],
    floors: Sequence[MerchantFloor],
) -> tuple[tuple[str, str, str, int], ...]:
    """Apply the study's reviewed trade-count-first clearing objective."""

    floor_index = {(row.merchant_id, row.listing_id): row for row in floors}
    valuation_index = {
        (row.buyer_id, row.merchant_id, row.listing_id): row for row in valuations
    }
    candidate_rank = {
        (buyer_id, listing_id(merchant_id)): rank
        for buyer_id, merchants in _CANDIDATE_MERCHANTS.items()
        for rank, merchant_id in enumerate(merchants, 1)
    }
    best_objective = (-1, -1, -1)
    best_key: tuple[str, ...] | None = None
    best: tuple[tuple[str, str, str, int], ...] = ()

    def visit(
        index: int,
        capacities: dict[tuple[str, str], int],
        selected: list[tuple[str, str, str, int]],
    ) -> None:
        nonlocal best_objective, best_key, best
        if index == len(BUYER_IDS):
            objective = (
                len(selected),
                sum(candidate_rank[(row[0], row[2])] == 1 for row in selected),
                _assignment_welfare(selected),
            )
            key = tuple(f"{row[0]}->{row[1]}/{row[2]}" for row in selected)
            if objective > best_objective or (
                objective == best_objective and (best_key is None or key < best_key)
            ):
                best_objective, best_key, best = objective, key, tuple(selected)
            return
        buyer_id = BUYER_IDS[index]
        visit(index + 1, capacities, selected)
        for merchant_id in _CANDIDATE_MERCHANTS[buyer_id]:
            sku_id = listing_id(merchant_id)
            offer = (merchant_id, sku_id)
            if capacities.get(offer, 0) <= 0:
                continue
            value = valuation_index[(buyer_id, merchant_id, sku_id)]
            welfare = value.unit_value_minor - floor_index[offer].unit_floor_minor
            if welfare <= 0:
                continue
            capacities[offer] -= 1
            selected.append((buyer_id, merchant_id, sku_id, welfare))
            visit(index + 1, capacities, selected)
            selected.pop()
            capacities[offer] += 1

    visit(
        0,
        {(row.merchant_id, row.listing_id): row.capacity for row in floors},
        [],
    )
    return best


def _metric_inputs(
    scenario: "ScenarioSpec",
) -> tuple[tuple[BuyerValuation, ...], tuple[MerchantFloor, ...]]:
    oracle = scenario.success_oracle.get("market_oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError("scenario is missing market_oracle")
    valuation_rows = oracle.get("buyer_valuations")
    floor_rows = oracle.get("merchant_floors")
    if not isinstance(valuation_rows, list) or not isinstance(floor_rows, list):
        raise ValueError("market_oracle valuations/floors must be lists")
    valuations = tuple(
        BuyerValuation(
            str(row["buyer_id"]),
            str(row["merchant_id"]),
            str(row["listing_id"]),
            int(row["unit_value_minor"]),
        )
        for row in valuation_rows
        if isinstance(row, Mapping)
    )
    floors = tuple(
        MerchantFloor(
            str(row["merchant_id"]),
            str(row["listing_id"]),
            int(row["unit_floor_minor"]),
            int(row["capacity"]),
        )
        for row in floor_rows
        if isinstance(row, Mapping)
    )
    return valuations, floors


def _public_rank_key(rule: str, listing: Mapping[str, Any]) -> tuple[Any, ...]:
    attributes = listing.get("attributes", {})
    if not isinstance(attributes, Mapping):
        raise ValueError("listing attributes must be a mapping")
    quality = int(attributes["quality_score"])
    shipping = int(attributes["shipping_days"])
    warranty = int(attributes["warranty_years"])
    price_minor = int(Decimal(str(listing["list_price"])) * 100)
    sku_id = str(listing["sku_id"])
    if rule == "balanced_value":
        return (price_minor - quality * 100, shipping, sku_id)
    if rule == "shipping_then_price":
        return (shipping, price_minor, -quality, sku_id)
    if rule == "quality_then_shipping":
        return (-quality, shipping, price_minor, sku_id)
    if rule == "price_then_shipping":
        return (price_minor, shipping, -quality, sku_id)
    if rule == "quality_then_warranty":
        return (-quality, -warranty, price_minor, sku_id)
    raise ValueError(f"unsupported public ranking rule: {rule!r}")


def _settle(
    scenario: "ScenarioSpec",
    assignments: Sequence[tuple[str, str, str, int]],
) -> tuple[World, tuple[MarketTransaction, ...]]:
    world = World()
    seed_world(world, scenario)
    catalog = {str(row["sku_id"]): row for row in scenario.initial_state["catalog"]}
    transactions: list[MarketTransaction] = []
    for ordinal, (buyer_id, merchant_id, sku_id, _welfare) in enumerate(
        sorted(assignments), 1
    ):
        before_row = world.read("inventory", SkuId(sku_id), caller="platform")
        if before_row is None:
            raise RuntimeError(f"missing inventory for {sku_id}")
        before = before_row.qty_available - before_row.qty_reserved
        price_minor = int(Decimal(str(catalog[sku_id]["list_price"])) * 100)
        order_id = OrderId(f"order:homogeneous:{ordinal}")
        order = Order(
            order_id=order_id,
            buyer_id=AgentId(buyer_id),
            merchant_id=AgentId(merchant_id),
            sku_id=SkuId(sku_id),
            qty=1,
            agreed_price=Money(Decimal(price_minor) / Decimal(100)),
            state=OrderState.ACCEPTED,
        )
        receipt = Receipt(
            txn_id=TxnId(f"txn:homogeneous:{ordinal}"),
            ts=f"logical:{ordinal}",
            order_id=order_id,
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            sku_id=order.sku_id,
            qty=1,
            price=order.agreed_price,
            idempotency_key=f"settle:homogeneous:{ordinal}",
        )
        world.settle_order(
            order=order,
            receipt=receipt,
            by_role="platform",
            idempotency_key=receipt.idempotency_key,
        )
        after_row = world.read("inventory", SkuId(sku_id), caller="platform")
        if after_row is None:
            raise RuntimeError(f"missing post-settlement inventory for {sku_id}")
        after = after_row.qty_available - after_row.qty_reserved
        transactions.append(MarketTransaction(
            transaction_id=str(receipt.txn_id),
            buyer_id=buyer_id,
            merchant_id=merchant_id,
            listing_id=sku_id,
            unit_price_minor=price_minor,
            quantity=1,
            inventory_before=before,
            inventory_after=after,
        ))
    return world, tuple(transactions)


def _reference_allocation_rows() -> list[dict[str, Any]]:
    assignment = (
        ("buyer:b1", "merchant:m3"),
        ("buyer:b2", "merchant:m2"),
        ("buyer:b3", "merchant:m5"),
        ("buyer:b4", "merchant:m4"),
        ("buyer:b5", "merchant:m1"),
    )
    quote_by_merchant = {
        str(row["merchant_id"]): int(row["list_price"]) * 100 for row in _MERCHANT_ROWS
    }
    return [
        {
            "buyer_id": buyer_id,
            "merchant_id": merchant_id,
            "listing_id": listing_id(merchant_id),
            "unit_price_minor": quote_by_merchant[merchant_id],
            "quantity": 1,
        }
        for buyer_id, merchant_id in assignment
    ]


def _assignment_welfare(rows: Iterable[tuple[str, str, str, int]]) -> int:
    return sum(row[3] for row in rows)


def _world_digest(world: World) -> str:
    payload = json.dumps(
        to_canonical(asdict(world.snapshot())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "BUYER_IDS",
    "HOMOGENEOUS_MARKET_STUDY_SCHEMA",
    "MERCHANT_IDS",
    "PRODUCT_ID",
    "SCENARIO_ID",
    "SEED",
    "TASK_FAMILY",
    "TOP_K",
    "VARIANT_ID",
    "build_reference_market_artifact",
    "build_scenario_spec",
    "clear_ranked_candidates",
    "compute_public_candidate_sets",
    "listing_id",
    "optimal_assignments",
]
