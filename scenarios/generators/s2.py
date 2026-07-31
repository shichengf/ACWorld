"""S2 — Constraint matching.

Stresses: the buyer must reject the cheaper SKU that
fails a hard constraint and buy the pricier matching one. Catalog has two
SKUs; SKU_0001 is cheaper but lacks ``noise_cancellation``, SKU_0002 matches
and is still within budget. ``success_oracle.expected_sku`` is the
server-side check that the right SKU was purchased.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s2"
FAMILY_NAME = "constraint_matching"


def _spec(seed: int) -> dict[str, Any]:
    max_budget = 80 + (seed % 3) * 10
    cheap_price = max_budget - 25
    match_price = max_budget - 5
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": "SKU_0001",
                    "list_price": cheap_price,
                    "floor_price": cheap_price - 20,
                    "inventory": 5,
                    "attributes": {"noise_cancellation": False, "shipping_days": 2},
                },
                {
                    "sku_id": "SKU_0002",
                    "list_price": match_price,
                    "floor_price": match_price - 20,
                    "inventory": 5,
                    "attributes": {"noise_cancellation": True, "shipping_days": 2},
                },
            ]
        },
        "buyer_goal": {
            "product_type": "over-ear headphones",
            "max_budget": max_budget,
            "quantity": 1,
            "constraints": ["noise_cancellation", "shipping_within_3_days"],
        },
        "merchant_policy": {
            "list_price": match_price,
            "floor_price": match_price - 20,
            "refund_policy": "7_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "expected_sku": "SKU_0002",
            "final_price_lte": max_budget,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(s := _spec(seed))['scenario_id']}.yaml", s) for seed in SEEDS]
