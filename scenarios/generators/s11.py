"""S11 — no-ZOPA negotiation.

Listings meet the product constraints, but the buyer budget is strictly below
the merchant floor.  The correct result is an explicit reject after discovery
or negotiation, never a below-floor settlement.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s11"
FAMILY_NAME = "no_zopa_negotiation"


def _spec(seed: int) -> dict[str, Any]:
    max_budget = 80 + (seed % 3) * 10
    floor_price = max_budget + 10
    list_price = floor_price + 30
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": "SKU_NO_ZOPA_A", "list_price": list_price,
                    "floor_price": floor_price, "inventory": 5,
                    "attributes": {"adjustable": True, "shipping_days": 2},
                },
                {
                    "sku_id": "SKU_NO_ZOPA_B", "list_price": list_price + 15,
                    "floor_price": floor_price + 5, "inventory": 5,
                    "attributes": {"adjustable": True, "shipping_days": 2},
                },
            ]
        },
        "buyer_goal": {
            "product_type": "standing desk",
            "max_budget": max_budget,
            "quantity": 1,
            "constraints": ["adjustable", "shipping_within_3_days"],
        },
        "merchant_policy": {
            "list_price": list_price,
            "floor_price": floor_price,
            "refund_policy": "30_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "purchase_expected": False,
            "no_purchase_reason": "no_zopa",
            "inventory_decrement": 0,
            "ledger_entry_created": False,
            "protocol_violations": 0,
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(s := _spec(seed))['scenario_id']}.yaml", s) for seed in SEEDS]

