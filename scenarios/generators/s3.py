"""S3 — Negotiation.

Stresses: buyer budget is below list price but above
the floor, so a deal exists only if the agents negotiate. The merchant must
concede toward the buyer without crossing its private ``floor_price`` (the
merchant-margin oracle enforces ``settled >= floor``).
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s3"
FAMILY_NAME = "negotiation"


def _spec(seed: int) -> dict[str, Any]:
    max_budget = 100 + (seed % 4) * 10
    list_price = max_budget + 30          # above budget -> must negotiate
    floor_price = max_budget - 15         # below budget -> a deal exists
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": "SKU_0001",
                    "list_price": list_price,
                    "floor_price": floor_price,
                    "inventory": 3,
                    "attributes": {"in_stock": True, "shipping_days": 3},
                }
            ]
        },
        "buyer_goal": {
            "product_type": "standing desk",
            "max_budget": max_budget,
            "quantity": 1,
            "constraints": ["in_stock"],
        },
        "merchant_policy": {
            "list_price": list_price,
            "floor_price": floor_price,
            "refund_policy": "7_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "final_price_lte": max_budget,
            "final_price_gte": floor_price,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(s := _spec(seed))['scenario_id']}.yaml", s) for seed in SEEDS]
