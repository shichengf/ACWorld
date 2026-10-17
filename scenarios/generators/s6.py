"""S6 — friend conflict.

Every instance ties the two listings on hard feasibility, soft utility, price,
delivery, and inventory.  Authoritative World-stored friend reviews are the
only signal that separates them.  The friend-preferred listing and catalog
order vary independently across seeds, so selecting the first result is not a
valid reference policy.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s6"
FAMILY_NAME = "friend_conflict"

_PREFERRED = ("SKU_0001", "SKU_0002", "SKU_0001")
_CATALOG_ORDERS = (
    ("SKU_0001", "SKU_0002"),
    ("SKU_0001", "SKU_0002"),
    ("SKU_0002", "SKU_0001"),
)
_RATINGS = (
    ((5, 4), (1, 2)),
    ((5, 5), (2, 1)),
    ((4, 5), (1, 1)),
)


def _spec(seed: int, index: int) -> dict[str, Any]:
    preferred = _PREFERRED[index]
    panned = "SKU_0002" if preferred == "SKU_0001" else "SKU_0001"
    list_price = (50, 54, 47)[index]
    floor_price = list_price - (20, 18, 17)[index]
    shipping_days = (2, 3, 1)[index]
    preferred_ratings, panned_ratings = _RATINGS[index]
    ratings = {preferred: preferred_ratings, panned: panned_ratings}

    reviews: list[dict[str, Any]] = []
    review_number = 1
    for sku_id in ("SKU_0001", "SKU_0002"):
        for reviewer_id, rating in zip(
            ("buyer:friend-alice", "buyer:friend-bob"),
            ratings[sku_id],
            strict=True,
        ):
            reviews.append({
                "review_id": f"s6-{seed}-rv{review_number}",
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
                    "list_price": list_price,
                    "floor_price": floor_price,
                    "inventory": 4 + index,
                    "attributes": {
                        "noise_cancellation": True,
                        "wireless": True,
                        "shipping_days": shipping_days,
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
            "product_type": "wireless headphones",
            "max_budget": list_price + 20,
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
            "expected_sku": preferred,
            "final_price_lte": list_price + 20,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
        "benchmark": {
            "task_family": "T3",
            "variant_id": "S6",
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
