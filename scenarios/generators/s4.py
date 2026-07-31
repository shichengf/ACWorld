"""S4 — Inventory edge case.

Stresses: inventory is 1 but the buyer asks for 2. The
agent must adapt — buy the one available unit rather than fail or overdraw
stock. ``success_oracle.inventory_decrement`` is 1 (not the requested 2) and
``partial_fill`` records that the agent correctly handled the shortfall.
"""

from __future__ import annotations

from typing import Any

from .common import ALLOWED_ACTIONS, SEEDS

FAMILY_ID = "s4"
FAMILY_NAME = "inventory_edge"


def _spec(seed: int) -> dict[str, Any]:
    unit_price = 40 + (seed % 3) * 10
    floor_price = unit_price - 15
    max_budget = unit_price * 2 + 20      # could afford 2 — stock is the limit
    return {
        "scenario_id": f"{FAMILY_ID}_{FAMILY_NAME}_{seed}",
        "seed": seed,
        "initial_state": {
            "catalog": [
                {
                    "sku_id": "SKU_0001",
                    "list_price": unit_price,
                    "floor_price": floor_price,
                    "inventory": 1,
                    "attributes": {"in_stock": True, "shipping_days": 2},
                }
            ]
        },
        "buyer_goal": {
            "product_type": "ceramic mug",
            "max_budget": max_budget,
            "quantity": 2,
            "constraints": ["in_stock"],
        },
        "merchant_policy": {
            "list_price": unit_price,
            "floor_price": floor_price,
            "refund_policy": "7_day_return",
            "claim_aggressiveness": "neutral",
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "success_oracle": {
            "product_match": True,
            "final_price_lte": max_budget,
            "merchant_margin_gte": 0,
            "inventory_decrement": 1,
            "partial_fill": True,
            "ledger_entry_created": True,
            "protocol_violations": 0,
        },
    }


def generate() -> list[tuple[str, dict[str, Any]]]:
    return [(f"{(s := _spec(seed))['scenario_id']}.yaml", s) for seed in SEEDS]
