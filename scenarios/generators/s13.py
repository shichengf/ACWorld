"""S13 — authoritative friend signal.

The two listings are tied on every catalog-derived criterion: hard constraints,
soft fit, price, delivery, and inventory.  Only the buyer's World-stored friend
reviews distinguish them.  The favored listing alternates across seeds so a
fixed "first result" strategy cannot score well accidentally.

The headline oracle reconstructs the comparison from ``WorldSnapshot``
friendships and reviews; it never trusts ratings copied into an agent trace.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s13"
FAMILY_NAME = "authoritative_friend"


def _spec(seed: int) -> dict[str, Any]:
    favored_number = 1 if seed % 2 == 0 else 2
    favored = f"SKU_000{favored_number}"
    panned = "SKU_0002" if favored == "SKU_0001" else "SKU_0001"
    list_price = 45 + seed % 4
    floor_price = list_price - 15
    max_budget = list_price + 20

    ratings = {
        favored: (5, 4),
        panned: (2, 1),
    }
    reviews: list[dict[str, Any]] = []
    review_id = 1
    for sku_id in ("SKU_0001", "SKU_0002"):
        for reviewer_id, rating in zip(
            ("buyer:friend-alice", "buyer:friend-bob"), ratings[sku_id], strict=True
        ):
            reviews.append({
                "review_id": f"rv-{seed}-{review_id}",
                "reviewer_id": reviewer_id,
                "sku_id": sku_id,
                "merchant_id": "merchant:m1",
                "rating": rating,
            })
            review_id += 1

    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": sku_id,
                    "list_price": list_price,
                    "floor_price": floor_price,
                    "inventory": 5,
                    "attributes": {
                        "noise_cancellation": True,
                        "wireless": True,
                        "shipping_days": 2,
                    },
                }
                for sku_id in ("SKU_0001", "SKU_0002")
            ],
            "friendships": [{
                "buyer_id": "buyer",
                "friends": ["buyer:friend-alice", "buyer:friend-bob"],
            }],
            "reviews": reviews,
        },
        "buyer_goal": {
            "product_type": "wireless headphones",
            "max_budget": max_budget,
            "quantity": 1,
            "constraints": ["noise_cancellation", "shipping_within_3_days"],
            "soft_constraints": [{"feature": "wireless", "importance": 1}],
        },
        "merchant_policy": {
            "list_price": list_price,
            "floor_price": floor_price,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "expected_sku": favored,
            "final_price_lte": max_budget,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T3",
            "variant_id": "S13",
            "track": "agent",
            "difficulty": "medium",
            "catalog_source": "inline",
            "catalog_scale": "smoke",
            "scenario_version": "1.0",
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(spec := _spec(seed))['scenario_id']}.yaml", spec) for seed in SEEDS]
