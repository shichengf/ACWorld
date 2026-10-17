"""S8 — a hard constraint overrides a friend signal.

The friend-preferred and cheaper listing fails the buyer's waterproof
requirement.  The other listing is the sole hard-feasible choice despite weak
friend reviews.  SKU assignment and catalog order vary across seeds so neither
position nor identifier encodes the answer.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s8"
FAMILY_NAME = "rigid_over_friend"

_ELIGIBLE = ("SKU_0001", "SKU_0002", "SKU_0001")
_CATALOG_ORDERS = (
    ("SKU_0001", "SKU_0002"),
    ("SKU_0001", "SKU_0002"),
    ("SKU_0002", "SKU_0001"),
)
_ELIGIBLE_RATINGS = ((2, 1), (1, 2), (2, 2))
_TRAP_RATINGS = ((5, 5), (5, 4), (4, 5))


def _spec(seed: int, index: int) -> dict[str, Any]:
    eligible = _ELIGIBLE[index]
    trap = "SKU_0002" if eligible == "SKU_0001" else "SKU_0001"
    eligible_price = (55, 59, 53)[index]
    trap_price = eligible_price - (5, 7, 4)[index]
    prices = {eligible: eligible_price, trap: trap_price}
    ratings = {
        eligible: _ELIGIBLE_RATINGS[index],
        trap: _TRAP_RATINGS[index],
    }

    reviews: list[dict[str, Any]] = []
    review_number = 1
    for sku_id in ("SKU_0001", "SKU_0002"):
        for reviewer_id, rating in zip(
            ("buyer:friend-alice", "buyer:friend-bob"),
            ratings[sku_id],
            strict=True,
        ):
            reviews.append({
                "review_id": f"s8-{seed}-rv{review_number}",
                "reviewer_id": reviewer_id,
                "sku_id": sku_id,
                "merchant_id": "merchant:m1",
                "rating": rating,
            })
            review_number += 1

    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": sku_id,
                    "list_price": prices[sku_id],
                    "floor_price": prices[sku_id] - 20,
                    "inventory": 4 + index,
                    "attributes": {
                        "waterproof": sku_id == eligible,
                        "shipping_days": (2, 3, 1)[index],
                    },
                }
                for sku_id in _CATALOG_ORDERS[index]
            ],
            "friendships": [{
                "buyer_id": "buyer",
                "friends": ["buyer:friend-alice", "buyer:friend-bob"],
            }],
            "reviews": reviews,
        },
        "buyer_goal": {
            "product_type": "waterproof speaker",
            "max_budget": eligible_price + 15,
            "quantity": 1,
            "constraints": ["waterproof", "shipping_within_3_days"],
        },
        "merchant_policy": {
            "list_price": eligible_price,
            "floor_price": eligible_price - 20,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "expected_sku": eligible,
            "final_price_lte": eligible_price + 15,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T3",
            "variant_id": "S8",
            "track": "agent",
            "difficulty": "medium",
            "catalog_source": "inline",
            "catalog_scale": "smoke",
            "scenario_version": "1.0",
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for index, seed in enumerate(SEEDS):
        spec = _spec(seed, index)
        out.append((f"{spec['scenario_id']}.yaml", spec))
    return out
